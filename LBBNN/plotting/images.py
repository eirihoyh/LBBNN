from __future__ import annotations
import numpy as np
from ._common import ensure_parent, get_matplotlib
from .. import explain as expl
from .. import inspection as insp


def plot_model_vision_image(
        net, 
        train_data, 
        train_target, c: int = 0, 
        net_nr: int = 0, 
        threshold: float = 0.5, 
        thresh_w: float = 0.0, 
        save_path=None, 
        show: bool = False):
    
    plt, mcolors, _ = get_matplotlib()
    cmap = mcolors.LinearSegmentedColormap.from_list("", ["white", "red"])
    clean_a = insp.clean_alpha_class(net, threshold, class_in_focus=c)
    p = int(clean_a[0].shape[1] ** 0.5)
    img_avg = np.zeros(p * p)
    w = insp.weight_matrices(net)[-1][c, -p * p:].detach().cpu().numpy()
    w = np.where(clean_a[-1][c, -p * p:].detach().cpu().numpy() == 1, w, 0)
    avg_c_img = train_data[train_target == c].mean(axis=0).reshape((p, p))
    fig, axs = plt.subplots(len(clean_a) + 1, figsize=(10, 10))
    for ind, ca in enumerate(clean_a):
        out = ca.shape[0]
        img_layer = np.zeros(p * p)
        for j in range(out): img_layer += np.where(np.abs(w) >= thresh_w, ca[j, -p * p:].detach().cpu().numpy(), 0)
        img_avg += img_layer
        axs[ind].imshow(avg_c_img, cmap="Greys", vmin=np.min(avg_c_img), vmax=np.max(avg_c_img))
        im = axs[ind].imshow(img_layer.reshape((p, p)), cmap=cmap, alpha=0.5, vmin=0, vmax=max(np.max(img_layer), 1))
        fig.colorbar(im, ax=axs[ind])
        axs[ind].set_title(f"Class {c}, Layer {ind}")
        axs[ind].set_xticks([])
        axs[ind].set_yticks([])
    min_max = max(np.concatenate((img_avg, img_avg * -1)))
    axs[ind + 1].imshow(avg_c_img, cmap="Greys", vmin=np.min(avg_c_img), vmax=np.max(avg_c_img))
    im = axs[ind + 1].imshow(img_avg.reshape((p, p)), cmap=cmap, alpha=0.5, vmin=0, vmax=max(min_max, 1e-9))
    axs[ind + 1].set_title(f"Net: {net_nr} all layers")
    axs[ind + 1].set_xticks([])
    axs[ind + 1].set_yticks([])
    fig.colorbar(im, ax=axs[ind + 1])
    plt.tight_layout()
    if save_path is not None: ensure_parent(save_path)
    fig.savefig(save_path, bbox_inches='tight')
    if show: plt.show()
    plt.close(fig)
    return save_path


def _sample_multiclass_probs(net, explain_this, n_draws: int = 1000):
    all_preds = []
    for _ in range(n_draws):
        net.eval()
        preds = net.forward(explain_this, sample=True, ensemble=False).detach().cpu().numpy()[0]
        exp_preds = np.exp(preds)
        all_preds.append(exp_preds / np.sum(exp_preds))
    return np.quantile(all_preds, [0.025, 0.975], axis=0)


def _plot_img_cred(cred_contribution, net, explain_this, n_classes, class_names, save_path, show):
    plt, mcolors, TwoSlopeNorm = get_matplotlib()
    p = int(explain_this.shape[-1] ** 0.5)
    lower, upper = _sample_multiclass_probs(net, explain_this, n_draws=1000)
    class_names = np.arange(n_classes) if class_names is None else class_names
    cmap = mcolors.LinearSegmentedColormap.from_list("", ["blue", "white", "red"])
    used_img = explain_this.reshape((p, p))
    saved = []
    for i in range(n_classes):
        explained_c = np.array(list(cred_contribution[i].values())[:-1])
        explained_025 = np.where(np.abs(explained_c[:, 0].reshape((p, p))) > 0, explained_c[:, 0].reshape((p, p)), np.nan)
        explained_975 = np.where(np.abs(explained_c[:, 1].reshape((p, p))) > 0, explained_c[:, 1].reshape((p, p)), np.nan)
        maxima = np.nanmax([np.nanmax(explained_025), np.nanmax(explained_975), 0])
        minima = np.nanmin([np.nanmin(explained_025), np.nanmin(explained_975), 0])
        fig, axs = plt.subplots(1, 2, figsize=(8, 8))
        axs[0].imshow(used_img, cmap="Greys", vmin=np.min(used_img), vmax=np.max(used_img) + 0.5)
        axs[1].imshow(used_img, cmap="Greys", vmin=np.min(used_img), vmax=np.max(used_img) + 0.5)
        norm = TwoSlopeNorm(vmin=minima - 0.001, vcenter=0, vmax=maxima + 0.001)
        im0 = axs[0].imshow(explained_025, cmap=cmap, norm=norm)
        im1 = axs[1].imshow(explained_975, cmap=cmap, norm=norm)
        fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)
        fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)
        axs[0].set_title("0.025 quantile")
        axs[1].set_title("0.975 quantile")
        axs[0].set_xticks([])
        axs[0].set_yticks([])
        axs[1].set_xticks([])
        axs[1].set_yticks([])
        fig.suptitle(f"Local explain class: {class_names[i]}. Credibility interval: [{lower[i]:.4f}, {upper[i]:.4f}]")
        plt.tight_layout(rect=[0, 0.03, 1, 1.0])
        if save_path is not None:
            out = f"{save_path}/{class_names[i]}.png"
            ensure_parent(out)
            fig.savefig(out, bbox_inches='tight')
            saved.append(out)
        if show: plt.show()
        plt.close(fig)
    return saved


def plot_local_contribution_images_contribution_empirical(
        net, 
        explain_this, 
        n_classes: int = 1, 
        class_names=None, 
        sample: bool = True, 
        median: bool = True, 
        n_samples: int = 100, 
        quantiles=(0.025, 0.975), 
        save_path=None, 
        show: bool = False):
    
    _, cred_contribution, _ = expl.local_explain_relu(
        net, explain_this, sample=sample, median=median, n_samples=n_samples, quantiles=quantiles)
    
    return _plot_img_cred(cred_contribution, net, explain_this, n_classes, class_names, save_path, show)


def plot_local_contribution_images_contribution_empirical_magnitude(
        net, 
        explain_this, 
        n_classes: int = 1, 
        class_names=None, 
        sample: bool = True, 
        median: bool = True, 
        n_samples: int = 100, 
        quantiles=(0.025, 0.975), 
        save_path=None, 
        include_potential_contribution: bool = False, 
        show: bool = False):
    
    _, cred_contribution, _ = expl.local_explain_relu_magnitude(
        net, explain_this, sample=sample, median=median, n_samples=n_samples, quantiles=quantiles, include_potential_contribution=include_potential_contribution)
    
    return _plot_img_cred(cred_contribution, net, explain_this, n_classes, class_names, save_path, show)
