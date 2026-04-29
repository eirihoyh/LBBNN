from __future__ import annotations

import copy
from typing import Any, Sequence
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor
import itertools

from ._common import ensure_parent, get_matplotlib
from .._types import BayesianNet
from .. import explain as expl


def plot_local_explain_piecewise_linear_act(
    net: BayesianNet,
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


def plot_what_if_explanations(
    observed_space: NDArray[np.float64],
    contributions: NDArray[np.float64],
    predictions: NDArray[np.float64],
    data: Tensor,
    feature_names: Sequence[str] | None = None,
    class_names: Sequence[str] | None = None,
    feature_in_focus: int | None = None,
    save_path: str | None = None,
) -> None:
    """Plot feature contributions over a what-if intervention range.

    Args:
        observed_space: Evaluated values for the varied feature.
        contributions: Contribution values with shape
            ``(n_samples, n_features, n_expl_per_sample)``.
        predictions: Predicted class indicators with shape
            ``(n_samples, n_prediction_samples)``.
        data: Original one-dimensional input tensor.
        feature_names: Optional names of the input features.
        class_names: Optional names of the output classes.
        feature_in_focus: Index of the adjusted feature.
        save_path: Optional path prefix for saving the plot.

    Returns:
        None.
    """
    plt, _, _ = get_matplotlib()

    n_features = contributions.shape[1]
    n_classes = predictions.shape[1]

    if feature_names is None:
        feature_names = [f"x_{i}" for i in range(n_features)]

    if class_names is None:
        class_names = [f"Class {i}" for i in range(n_classes)]

    if feature_in_focus is None:
        feature_in_focus = 0

    original_value = "Original input: " + ", ".join(
        f"{feature_names[i]}={data[i].item():.2f}"
        for i in range(len(feature_names))
    )

    plt.style.use("seaborn-v0_8-colorblind")
    plt.rc("font", size=14)
    plt.rc("axes", labelsize=14)
    plt.rc("xtick", labelsize=14)

    linestyles = ["-", "--", ":", "-."]
    style_cycler = itertools.cycle(linestyles)

    plt.figure(figsize=(13.5, 5))

    for feature_idx in range(n_features):
        lower, median, upper = np.quantile(
            contributions[:, feature_idx, :],
            [0.025, 0.5, 0.975],
            axis=1,
        )

        plt.plot(
            observed_space,
            median,
            linestyle=next(style_cycler),
            label=feature_names[feature_idx],
            linewidth=2.5,
        )
        plt.fill_between(observed_space, lower, upper, alpha=0.2)

    pred_changes = np.where(predictions.mean(axis=1) > 0.5)[0]
    if pred_changes.size > 0:
        for pred_change in pred_changes[:-1]:
            plt.axvline(
                x=observed_space[pred_change],
                color="red",
                linestyle="-",
                alpha=0.125,
                linewidth=2.5,
            )
        plt.axvline(
            x=observed_space[pred_changes[-1]],
            color="red",
            linestyle="-",
            alpha=0.125,
            linewidth=2.5,
            label="Prediction=1",
        )

    plt.xlabel(f"New {feature_names[feature_in_focus]} Value")
    plt.ylabel("Contribution (∇f(x))")
    plt.title(
        "Covariate contributions. "
        f"{feature_names[feature_in_focus]} is adjusted. "
        f"{original_value}"
    )
    plt.legend()

    if save_path is not None:
        output_path = Path(f"{save_path}.png")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, bbox_inches="tight")
    plt.show()