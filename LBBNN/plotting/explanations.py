from __future__ import annotations
import copy
import numpy as np
from ._common import ensure_parent, get_matplotlib
from .. import explain as expl


def plot_local_contribution_empirical(
        net, 
        data, 
        sample: bool = True, 
        median: bool = True, 
        n_samples: int = 1, 
        include_bias: bool = True, 
        save_path=None, 
        n_classes: int = 1, 
        class_names=None, 
        variable_names=None, 
        quantiles=(0.025, 0.975), 
        include_zero_means: bool = True, 
        magnitude: bool = False, 
        include_potential_contribution: bool = False, 
        show: bool = False):
    
    plt, _, _ = get_matplotlib()
    variable_names = copy.deepcopy(variable_names)
    if magnitude:
        mean_contribution, cred_contribution, preds = expl.local_explain_relu_magnitude(
            net, data, sample=sample, median=median, n_samples=n_samples, quantiles=quantiles, include_potential_contribution=include_potential_contribution)
        
    else:
        mean_contribution, cred_contribution, preds = expl.local_explain_relu(
            net, data, sample=sample, median=median, n_samples=n_samples, quantiles=quantiles)
        
    if class_names is None: class_names = np.arange(n_classes)
    if variable_names is None: variable_names = np.arange(data.shape[-1])

    variable_names = list(np.array(variable_names).astype(str))
    variable_names.append("bias")
    variable_names = np.array(variable_names)
    preds_means = np.mean(preds, 0)[0]
    saved = []
    for c in mean_contribution.keys():
        preds_errors = np.quantile(preds[:, :, c], quantiles)
        variable_names_class = copy.deepcopy(variable_names)
        means = np.array(list(mean_contribution[c].values()))
        errors = np.array(list(cred_contribution[c].values()))
        if not include_bias:
            means = means[:-1]
            errors = errors[:-1]
            variable_names_class = variable_names_class[:-1]
        if not include_zero_means:
            include = means != 0
            variable_names_class = variable_names_class[include]
            means = means[include]
            errors = errors[include]
        means = np.append(means, preds_means[c])
        errors = np.vstack([errors, preds_errors])
        for indx, err in enumerate(errors):
            if err[0] == 0 and err[1] == 0: err[0] = means[indx]
            err[1] = means[indx]
        top = errors[:, 1] - means
        bottom = means - errors[:, 0]
        variable_names_class = np.append(variable_names_class, "Prediction")
        fig, ax = plt.subplots()
        ax.bar(variable_names_class, means, yerr=(bottom, top), align='center', alpha=0.5, ecolor='black', capsize=10)
        ax.set_ylabel('Contribution')
        ax.tick_params(axis='x', rotation=90)
        ax.set_title(f'Empirical explanation of {class_names[c]}')
        ax.grid()
        if save_path is not None:
            out = f"{save_path}_class_{class_names[c]}.png"
            ensure_parent(out)
            fig.savefig(out, bbox_inches='tight')
            saved.append(out)
        if show: plt.show()
        plt.close(fig)
    return saved


def plot_local_explain_piecewise_linear_act(
        net, 
        input_data, 
        median: bool = True, 
        sample: bool = True, 
        n_samples: int = 1, 
        n_classes: int = 1, 
        magnitude: bool = True, 
        include_potential_contribution: bool = True, 
        variable_names=None, 
        class_names=None, 
        include_prediction: bool = True, 
        include_bias: bool = True, 
        no_zero_contributions: bool = False, 
        fig_size=(10, 6), 
        cred_int=(0.025, 0.975), 
        ann: bool = False, 
        thresh: float = 0.005, 
        save_path=None, 
        show: bool = False):
    
    plt, _, _ = get_matplotlib()
    expl_values, preds, p = expl.local_explain_piecewise_linear_act(
        net, 
        input_data, 
        median=median, 
        sample=sample, 
        n_samples=n_samples, 
        magnitude=magnitude, 
        include_potential_contribution=include_potential_contribution, 
        n_classes=n_classes)
    
    if class_names is None: class_names = ["" for _ in range(n_classes)]
    else: class_names = [f"Class: {name}" for name in class_names]
    
    if variable_names is None: variable_names = [f"x{i}" for i in range(p)]
    
    variable_names = [f"{v}={float(input_data[i].cpu().detach().numpy()):.2f}" for i, v in enumerate(variable_names)]
    variable_names = np.array(variable_names)
    if not include_bias:
        variable_names = variable_names[1:]
        expl_values = expl_values[:, 1:]
        p -= 1
    
    saved = []
    for c in range(n_classes):
        expl_class = copy.deepcopy(expl_values[:, :, c])
        p_class = copy.deepcopy(p)
        variable_names_class = copy.deepcopy(variable_names)
        if no_zero_contributions:
            keep = ~np.isclose(expl_class, 0, thresh, thresh).all(axis=0) if ann else ~(expl_class == 0).all(axis=0)
            expl_class = expl_class[:, keep]
            variable_names_class = variable_names_class[keep]
            p_class = expl_class.shape[1]
        if include_prediction:
            expl_class = np.concatenate((expl_class, preds[:, c:c+1].cpu().detach().numpy()), 1)
            variable_names_class = np.append(variable_names_class, ["Prediction"])
            p_class += 1
        means = expl_class.mean(0)
        cred = np.quantile(expl_class, cred_int, axis=0).T
        for indx, err in enumerate(cred):
            if err[0] == 0 and err[1] == 0: err[0] = means[indx]
            err[1] = means[indx]
        top = cred[:, 1] - means
        bottom = means - cred[:, 0]
        fig = plt.figure(figsize=fig_size)
        plt.bar(range(p_class), means, yerr=(bottom, top), align='center', alpha=0.5, edgecolor='k', capsize=10)
        plt.xlabel('Input Variable')
        plt.ylabel('Gradient')
        plt.title(f'Covariate contribution to model prediction. {class_names[c]}')
        plt.xticks(range(p_class), [f'{variable_names_class[i]}' for i in range(p_class)], rotation=90)
        plt.grid()
        plt.tight_layout()
        if save_path is not None:
            out = f"{save_path}{c}.png" if n_classes > 1 else f"{save_path}.png"
            ensure_parent(out)
            fig.savefig(out, bbox_inches='tight')
            saved.append(out)
        if show: plt.show()
        plt.close(fig)
    return saved
