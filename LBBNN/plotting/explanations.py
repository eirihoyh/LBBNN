from __future__ import annotations

import copy
from typing import Any, Sequence

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from ._common import ensure_parent, get_matplotlib
from .. import explain as expl


def plot_local_contribution_empirical(
    net: Any,
    data: Tensor,
    sample: bool = True,
    median: bool = True,
    n_samples: int = 1,
    include_bias: bool = True,
    save_path: str | None = None,
    n_classes: int = 1,
    class_names: Sequence[Any] | None = None,
    variable_names: Sequence[Any] | None = None,
    quantiles: tuple[float, float] = (0.025, 0.975),
    include_zero_means: bool = True,
    magnitude: bool = False,
    include_potential_contribution: bool = False,
    show: bool = False,
) -> list[str]:
    """Plot empirical local contributions with credible intervals.

    Args:
        net: Trained network object.
        data: Input tensor to explain.
        sample: Whether to sample weights during explanation.
        median: Whether to use median inclusion during explanation.
        n_samples: Number of explanation samples.
        include_bias: Whether to include the bias contribution in the plot.
        save_path: Optional file prefix for saving plots.
        n_classes: Number of output classes.
        class_names: Optional names for the output classes.
        variable_names: Optional names for the input variables.
        quantiles: Lower and upper quantiles for credible intervals.
        include_zero_means: Whether to keep variables with zero mean contribution.
        magnitude: Whether to use magnitude-based contributions.
        include_potential_contribution: Whether to include potential
            contributions from inactive inputs.
        show: Whether to display the plots.

    Returns:
        A list of saved plot paths.
    """
    plt, _, _ = get_matplotlib()

    variable_names_copy = copy.deepcopy(variable_names)

    if magnitude:
        mean_contribution, cred_contribution, preds = (
            expl.local_explain_relu_magnitude(
                net,
                data,
                sample=sample,
                median=median,
                n_samples=n_samples,
                quantiles=quantiles,
                include_potential_contribution=include_potential_contribution,
            )
        )
    else:
        mean_contribution, cred_contribution, preds = expl.local_explain_relu(
            net,
            data,
            sample=sample,
            median=median,
            n_samples=n_samples,
            quantiles=quantiles,
        )

    if class_names is None:
        class_names = np.arange(n_classes)

    if variable_names_copy is None:
        variable_names_copy = np.arange(data.shape[-1])

    variable_names_array = np.array(list(np.asarray(variable_names_copy).astype(str)))
    variable_names_array = np.append(variable_names_array, "bias")

    preds_means = np.mean(preds, axis=0)[0]
    saved_paths: list[str] = []

    for class_idx in mean_contribution.keys():
        preds_errors = np.quantile(preds[:, :, class_idx], quantiles)
        variable_names_class = copy.deepcopy(variable_names_array)

        means = np.array(list(mean_contribution[class_idx].values()), dtype=float)
        errors = np.array(list(cred_contribution[class_idx].values()), dtype=float)

        if not include_bias:
            means = means[:-1]
            errors = errors[:-1]
            variable_names_class = variable_names_class[:-1]

        if not include_zero_means:
            include_mask = means != 0
            variable_names_class = variable_names_class[include_mask]
            means = means[include_mask]
            errors = errors[include_mask]

        means = np.append(means, preds_means[class_idx])
        errors = np.vstack([errors, preds_errors])
        variable_names_class = np.append(variable_names_class, "Prediction")

        for idx, err in enumerate(errors):
            if err[0] == 0 and err[1] == 0:
                err[0] = means[idx]
            err[1] = means[idx]

        upper = errors[:, 1] - means
        lower = means - errors[:, 0]

        fig, ax = plt.subplots()
        ax.bar(
            variable_names_class,
            means,
            yerr=(lower, upper),
            align="center",
            alpha=0.5,
            ecolor="black",
            capsize=10,
        )
        ax.set_ylabel("Contribution")
        ax.tick_params(axis="x", rotation=90)
        ax.set_title(f"Empirical explanation of {class_names[class_idx]}")
        ax.grid()

        if save_path is not None:
            out_path = f"{save_path}_class_{class_names[class_idx]}.png"
            ensure_parent(out_path)
            fig.savefig(out_path, bbox_inches="tight")
            saved_paths.append(out_path)

        if show:
            plt.show()

        plt.close(fig)

    return saved_paths


def plot_local_explain_piecewise_linear_act(
    net: Any,
    input_data: Tensor,
    median: bool = True,
    sample: bool = True,
    n_samples: int = 1,
    n_classes: int = 1,
    magnitude: bool = True,
    include_potential_contribution: bool = True,
    variable_names: Sequence[Any] | None = None,
    class_names: Sequence[Any] | None = None,
    include_prediction: bool = True,
    include_bias: bool = True,
    no_zero_contributions: bool = False,
    fig_size: tuple[float, float] = (10, 6),
    cred_int: tuple[float, float] = (0.025, 0.975),
    ann: bool = False,
    thresh: float = 0.005,
    save_path: str | None = None,
    show: bool = False,
) -> list[str]:
    """Plot gradient-based local explanations for piecewise linear activations.

    Args:
        net: Trained network object.
        input_data: One-dimensional input tensor to explain.
        median: Whether to use deterministic inference.
        sample: Whether to sample from the model.
        n_samples: Number of explanation samples.
        n_classes: Number of output classes.
        magnitude: Whether to use raw gradients instead of input-weighted
            contributions.
        include_potential_contribution: Whether to include potential
            contributions for zero-valued inputs.
        variable_names: Optional names for the input variables.
        class_names: Optional names for the output classes.
        include_prediction: Whether to append the prediction to the plot.
        include_bias: Whether to include the bias term.
        no_zero_contributions: Whether to remove variables with zero
            contribution across all samples.
        fig_size: Figure size passed to matplotlib.
        cred_int: Lower and upper quantiles for credible intervals.
        ann: Whether to use approximate zero filtering.
        thresh: Tolerance used when `ann` is True.
        save_path: Optional file prefix for saving plots.
        show: Whether to display the plots.

    Returns:
        A list of saved plot paths.
    """
    plt, _, _ = get_matplotlib()

    expl_values, preds, p = expl.local_explain_piecewise_linear_act(
        net,
        input_data,
        median=median,
        sample=sample,
        n_samples=n_samples,
        magnitude=magnitude,
        include_potential_contribution=include_potential_contribution,
        n_classes=n_classes,
    )

    if class_names is None:
        class_names_list = ["" for _ in range(n_classes)]
    else:
        class_names_list = [f"Class: {name}" for name in class_names]

    if variable_names is None:
        variable_names_list = [f"x{i}" for i in range(p)]
    else:
        variable_names_list = list(variable_names)

    variable_names_array = np.array(
        [
            f"{name}={float(input_data[i].detach().cpu().numpy()):.2f}"
            for i, name in enumerate(variable_names_list)
        ]
    )

    if not include_bias:
        variable_names_array = variable_names_array[1:]
        expl_values = expl_values[:, 1:]
        p -= 1

    saved_paths: list[str] = []

    for class_idx in range(n_classes):
        expl_class = copy.deepcopy(expl_values[:, :, class_idx])
        p_class = copy.deepcopy(p)
        variable_names_class = copy.deepcopy(variable_names_array)

        if no_zero_contributions:
            if ann:
                keep = ~np.isclose(
                    expl_class,
                    0.0,
                    rtol=thresh,
                    atol=thresh,
                ).all(axis=0)
            else:
                keep = ~(expl_class == 0).all(axis=0)

            expl_class = expl_class[:, keep]
            variable_names_class = variable_names_class[keep]
            p_class = expl_class.shape[1]

        if include_prediction:
            expl_class = np.concatenate(
                (expl_class, preds[:, class_idx : class_idx + 1].cpu().numpy()),
                axis=1,
            )
            variable_names_class = np.append(variable_names_class, "Prediction")
            p_class += 1

        means = expl_class.mean(axis=0)
        cred = np.quantile(expl_class, cred_int, axis=0).T

        for idx, err in enumerate(cred):
            if err[0] == 0 and err[1] == 0:
                err[0] = means[idx]
                err[1] = means[idx]

        upper = cred[:, 1] - means
        lower = means - cred[:, 0]

        fig = plt.figure(figsize=fig_size)
        plt.bar(
            range(p_class),
            means,
            yerr=(lower, upper),
            align="center",
            alpha=0.5,
            edgecolor="k",
            capsize=10,
        )
        plt.xlabel("Input Variable")
        plt.ylabel("Gradient")
        plt.title(
            f"Covariate contribution to model prediction. "
            f"{class_names_list[class_idx]}"
        )
        plt.xticks(
            range(p_class),
            [str(variable_names_class[i]) for i in range(p_class)],
            rotation=90,
        )
        plt.grid()
        plt.tight_layout()

        if save_path is not None:
            out_path = (
                f"{save_path}{class_idx}.png"
                if n_classes > 1
                else f"{save_path}.png"
            )
            ensure_parent(out_path)
            fig.savefig(out_path, bbox_inches="tight")
            saved_paths.append(out_path)

        if show:
            plt.show()

        plt.close(fig)

    return saved_paths