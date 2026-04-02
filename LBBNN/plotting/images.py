from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch
from numpy.typing import NDArray

from ._common import ensure_parent, get_matplotlib
from .. import explain as expl
from .. import inspection as insp


def plot_model_vision_image(
    net: Any,
    train_data: NDArray[np.floating],
    train_target: NDArray[np.integer] | NDArray[np.floating],
    c: int = 0,
    net_nr: int = 0,
    threshold: float = 0.5,
    thresh_w: float = 0.0,
    save_path: str | None = None,
    show: bool = False,
) -> str | None:
    """Plot layer-wise input usage for a given class as an image overlay.

    Args:
        net: Trained network object.
        train_data: Training inputs as a NumPy array.
        train_target: Training labels as a NumPy array.
        c: Class index to visualize.
        net_nr: Network identifier used in the plot title.
        threshold: Threshold used to clean alpha values.
        thresh_w: Minimum absolute output weight used to include a connection.
        save_path: Optional path to save the figure.
        show: Whether to display the figure.

    Returns:
        The save path if provided, otherwise None.
    """
    plt, mcolors, _ = get_matplotlib()
    cmap = mcolors.LinearSegmentedColormap.from_list("", ["white", "red"])

    clean_alpha_list = insp.clean_alpha_class(net, threshold, class_in_focus=c)
    p = int(clean_alpha_list[0].shape[1] ** 0.5)

    avg_class_image = train_data[train_target == c].mean(axis=0).reshape((p, p))
    aggregated_image = np.zeros(p * p)

    weights = insp.weight_matrices(net)[-1][c, -p * p:].detach().cpu().numpy()
    mask = clean_alpha_list[-1][c, -p * p:].detach().cpu().numpy() == 1
    weights = np.where(mask, weights, 0)

    fig, axes = plt.subplots(len(clean_alpha_list) + 1, figsize=(10, 10))
    axes = np.atleast_1d(axes)

    for idx, clean_alpha in enumerate(clean_alpha_list):
        n_outputs = clean_alpha.shape[0]
        layer_image = np.zeros(p * p)

        for j in range(n_outputs):
            layer_image += np.where(
                np.abs(weights) >= thresh_w,
                clean_alpha[j, -p * p:].detach().cpu().numpy(),
                0,
            )

        aggregated_image += layer_image

        axes[idx].imshow(
            avg_class_image,
            cmap="Greys",
            vmin=np.min(avg_class_image),
            vmax=np.max(avg_class_image),
        )
        image = axes[idx].imshow(
            layer_image.reshape((p, p)),
            cmap=cmap,
            alpha=0.5,
            vmin=0,
            vmax=max(np.max(layer_image), 1),
        )
        fig.colorbar(image, ax=axes[idx])
        axes[idx].set_title(f"Class {c}, Layer {idx}")
        axes[idx].set_xticks([])
        axes[idx].set_yticks([])

    vmax = max(np.max(aggregated_image), np.max(-aggregated_image), 1e-9)
    last_idx = len(clean_alpha_list)

    axes[last_idx].imshow(
        avg_class_image,
        cmap="Greys",
        vmin=np.min(avg_class_image),
        vmax=np.max(avg_class_image),
    )
    image = axes[last_idx].imshow(
        aggregated_image.reshape((p, p)),
        cmap=cmap,
        alpha=0.5,
        vmin=0,
        vmax=vmax,
    )
    axes[last_idx].set_title(f"Net: {net_nr} all layers")
    axes[last_idx].set_xticks([])
    axes[last_idx].set_yticks([])
    fig.colorbar(image, ax=axes[last_idx])

    plt.tight_layout()

    if save_path is not None:
        ensure_parent(save_path)
        fig.savefig(save_path, bbox_inches="tight")

    if show:
        plt.show()

    plt.close(fig)
    return save_path


def _sample_multiclass_probs(
    net: Any,
    explain_this: torch.Tensor,
    n_draws: int = 1000,
) -> NDArray[np.floating]:
    """Sample predictive class probabilities and return quantile bounds.

    Args:
        net: Trained network object.
        explain_this: Input tensor to explain.
        n_draws: Number of stochastic forward passes.

    Returns:
        A NumPy array containing lower and upper quantiles for each class.
    """
    all_preds = []

    net.eval()
    for _ in range(n_draws):
        preds = net.forward(
            explain_this,
            sample=True,
            ensemble=False,
        ).detach().cpu().numpy()[0]
        exp_preds = np.exp(preds)
        all_preds.append(exp_preds / np.sum(exp_preds))

    return np.quantile(all_preds, [0.025, 0.975], axis=0)


def _plot_img_cred(
    cred_contribution: dict[int, dict[int, NDArray[np.floating]]],
    net: Any,
    explain_this: torch.Tensor,
    n_classes: int,
    class_names: Sequence[Any] | None,
    save_path: str | None,
    show: bool,
) -> list[str]:
    """Plot lower and upper contribution quantiles as image overlays.

    Args:
        cred_contribution: Credible intervals for local contributions.
        net: Trained network object.
        explain_this: Input tensor to explain.
        n_classes: Number of output classes.
        class_names: Optional names for the classes.
        save_path: Optional directory path for saving the figures.
        show: Whether to display the figures.

    Returns:
        A list of saved file paths.
    """
    plt, mcolors, two_slope_norm = get_matplotlib()

    p = int(explain_this.shape[-1] ** 0.5)
    lower, upper = _sample_multiclass_probs(net, explain_this, n_draws=1000)

    if class_names is None:
        class_names = np.arange(n_classes)

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "",
        ["blue", "white", "red"],
    )

    used_img = explain_this.detach().cpu().numpy().reshape((p, p))
    saved_paths: list[str] = []

    for i in range(n_classes):
        explained_c = np.array(list(cred_contribution[i].values())[:-1])

        explained_025 = np.where(
            np.abs(explained_c[:, 0].reshape((p, p))) > 0,
            explained_c[:, 0].reshape((p, p)),
            np.nan,
        )
        explained_975 = np.where(
            np.abs(explained_c[:, 1].reshape((p, p))) > 0,
            explained_c[:, 1].reshape((p, p)),
            np.nan,
        )

        maxima = np.nanmax(
            [np.nanmax(explained_025), np.nanmax(explained_975), 0]
        )
        minima = np.nanmin(
            [np.nanmin(explained_025), np.nanmin(explained_975), 0]
        )

        fig, axes = plt.subplots(1, 2, figsize=(8, 8))

        for ax in axes:
            ax.imshow(
                used_img,
                cmap="Greys",
                vmin=np.min(used_img),
                vmax=np.max(used_img) + 0.5,
            )
            ax.set_xticks([])
            ax.set_yticks([])

        norm = two_slope_norm(
            vmin=minima - 0.001,
            vcenter=0,
            vmax=maxima + 0.001,
        )

        image_0 = axes[0].imshow(explained_025, cmap=cmap, norm=norm)
        image_1 = axes[1].imshow(explained_975, cmap=cmap, norm=norm)

        fig.colorbar(image_0, ax=axes[0], fraction=0.046, pad=0.04)
        fig.colorbar(image_1, ax=axes[1], fraction=0.046, pad=0.04)

        axes[0].set_title("0.025 quantile")
        axes[1].set_title("0.975 quantile")
        fig.suptitle(
            f"Local explain class: {class_names[i]}. "
            f"Credibility interval: [{lower[i]:.4f}, {upper[i]:.4f}]"
        )

        plt.tight_layout(rect=[0, 0.03, 1, 1.0])

        if save_path is not None:
            out_path = f"{save_path}/{class_names[i]}.png"
            ensure_parent(out_path)
            fig.savefig(out_path, bbox_inches="tight")
            saved_paths.append(out_path)

        if show:
            plt.show()

        plt.close(fig)

    return saved_paths


def plot_local_contribution_images_contribution_empirical(
    net: Any,
    explain_this: torch.Tensor,
    n_classes: int = 1,
    class_names: Sequence[Any] | None = None,
    sample: bool = True,
    median: bool = True,
    n_samples: int = 100,
    quantiles: tuple[float, float] = (0.025, 0.975),
    save_path: str | None = None,
    show: bool = False,
) -> list[str]:
    """Plot empirical local contribution intervals for image inputs.

    Args:
        net: Trained network object.
        explain_this: Input tensor to explain.
        n_classes: Number of output classes.
        class_names: Optional names for the classes.
        sample: Whether to sample weights during explanation.
        median: Whether to use median inclusion during explanation.
        n_samples: Number of explanation samples.
        quantiles: Lower and upper quantiles for credible intervals.
        save_path: Optional directory path for saving the figures.
        show: Whether to display the figures.

    Returns:
        A list of saved file paths.
    """
    _, cred_contribution, _ = expl.local_explain_relu(
        net,
        explain_this,
        sample=sample,
        median=median,
        n_samples=n_samples,
        quantiles=quantiles,
    )

    return _plot_img_cred(
        cred_contribution=cred_contribution,
        net=net,
        explain_this=explain_this,
        n_classes=n_classes,
        class_names=class_names,
        save_path=save_path,
        show=show,
    )


def plot_local_contribution_images_contribution_empirical_magnitude(
    net: Any,
    explain_this: torch.Tensor,
    n_classes: int = 1,
    class_names: Sequence[Any] | None = None,
    sample: bool = True,
    median: bool = True,
    n_samples: int = 100,
    quantiles: tuple[float, float] = (0.025, 0.975),
    save_path: str | None = None,
    include_potential_contribution: bool = False,
    show: bool = False,
) -> list[str]:
    """Plot empirical local contribution magnitudes for image inputs.

    Args:
        net: Trained network object.
        explain_this: Input tensor to explain.
        n_classes: Number of output classes.
        class_names: Optional names for the classes.
        sample: Whether to sample weights during explanation.
        median: Whether to use median inclusion during explanation.
        n_samples: Number of explanation samples.
        quantiles: Lower and upper quantiles for credible intervals.
        save_path: Optional directory path for saving the figures.
        include_potential_contribution: Whether to include potential
            contributions for inactive inputs.
        show: Whether to display the figures.

    Returns:
        A list of saved file paths.
    """
    _, cred_contribution, _ = expl.local_explain_relu_magnitude(
        net,
        explain_this,
        sample=sample,
        median=median,
        n_samples=n_samples,
        quantiles=quantiles,
        include_potential_contribution=include_potential_contribution,
    )

    return _plot_img_cred(
        cred_contribution=cred_contribution,
        net=net,
        explain_this=explain_this,
        n_classes=n_classes,
        class_names=class_names,
        save_path=save_path,
        show=show,
    )