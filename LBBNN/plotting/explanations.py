from __future__ import annotations

import copy
from typing import Any, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from typing import Literal
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
    feature_names: Sequence[Any] | None = None,
    class_names: Sequence[Any] | None = None,
    include_prediction: bool = True,
    include_bias: bool = True,
    no_zero_contributions: bool = True,
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
        feature_names: Optional names for the input variables.
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

    if feature_names is None:
        feature_names_list = [f"x{i}" for i in range(p)]
    else:
        feature_names_list = list(feature_names)

    feature_names_array = np.array(
        [
            f"{name}={float(input_data[i].detach().cpu().numpy()):.2f}"
            for i, name in enumerate(feature_names_list)
        ]
    )

    if not include_bias:
        feature_names_array = feature_names_array[1:]
        expl_values = expl_values[:, 1:]
        p -= 1

    saved_paths: list[str] = []

    for class_idx in range(n_classes):
        expl_class = copy.deepcopy(expl_values[:, :, class_idx])
        p_class = copy.deepcopy(p)
        feature_names_class = copy.deepcopy(feature_names_array)

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
            feature_names_class = feature_names_class[keep]
            p_class = expl_class.shape[1]

        if include_prediction:
            expl_class = np.concatenate(
                (expl_class, preds[:, class_idx : class_idx + 1].cpu().numpy()),
                axis=1,
            )
            feature_names_class = np.append(feature_names_class, "Prediction")
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
            [str(feature_names_class[i]) for i in range(p_class)],
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
    task: Literal["binary", "multiclass", "regression"] = "binary",
    n_classes: int = 1,
    feature_names: Sequence[str] | None = None,
    class_names: Sequence[str] | None = None,
    feature_in_focus: int | None = None,
    no_zero_contributions: bool = True,
    pred_threshold: float = 0.5,
    save_path: str | None = None,
) -> None:
    """Plot feature contributions over a what-if intervention range.

    Args:
        observed_space: Evaluated values for the varied feature.
        contributions: Contribution values with shape
            ``(n_samples, n_features, n_expl_per_sample)``.
        predictions: For ``"binary"`` and ``"multiclass"``, predicted
            class indicators with shape
            ``(n_samples, n_prediction_samples)``. For ``"regression"``,
            continuous predicted values of the same shape.
        data: Original one-dimensional input tensor.
        task: Type of prediction task. One of ``"binary"``,
            ``"multiclass"``, or ``"regression"``. Controls how
            predictions are visualised alongside contributions:
            - ``"binary"``: red vertical lines where the mean prediction
              crosses ``pred_threshold``.
            - ``"multiclass"``: red vertical lines where the predicted
              class (argmax) changes.
            - ``"regression"``: mean predicted value plotted as a
              continuous line on a secondary y-axis.
        n_classes: Number of output classes. Used to correctly label
            class names when ``task="multiclass"``. Ignored otherwise.
        feature_names: Optional names of the input features.
        class_names: Optional names of the output classes.
        feature_in_focus: Index of the adjusted feature.
        no_zero_contributions: Whether to exclude features whose
            contributions are zero across all samples.
        pred_threshold: Decision threshold used to draw vertical lines
            in the ``"binary"`` case. Ignored otherwise.
        save_path: Optional path prefix for saving the plot.

    Returns:
        None.
    """
    plt, _, _ = get_matplotlib()

    n_features = contributions.shape[1]

    if feature_names is None:
        feature_names = [f"x_{i}" for i in range(n_features)]

    if class_names is None:
        class_names = [f"Class {i}" for i in range(n_classes)]

    if feature_in_focus is None:
        feature_in_focus = 0

    if no_zero_contributions:
        keep = ~(contributions == 0).all(axis=(0, 2))
        contributions = contributions[:, keep, :]
        feature_names = [name for name, k in zip(feature_names, keep) if k]
        n_features = contributions.shape[1]

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

    fig, ax1 = plt.subplots(figsize=(13.5, 5))

    for feature_idx in range(n_features):
        lower, median, upper = np.quantile(
            contributions[:, feature_idx, :],
            [0.025, 0.5, 0.975],
            axis=1,
        )
        ax1.plot(
            observed_space,
            median,
            linestyle=next(style_cycler),
            label=feature_names[feature_idx],
            linewidth=2.5,
        )
        ax1.fill_between(observed_space, lower, upper, alpha=0.2)

    ax1.set_xlabel(f"New {feature_names[feature_in_focus]} Value")
    ax1.set_ylabel("Contribution (∇f(x))")

    if task == "binary":
        pred_changes = np.where(predictions.mean(axis=1) > pred_threshold)[0]
        if pred_changes.size > 0:
            for pred_change in pred_changes[:-1]:
                ax1.axvline(
                    x=observed_space[pred_change],
                    color="red",
                    linestyle="-",
                    alpha=0.125,
                    linewidth=2.5,
                )
            ax1.axvline(
                x=observed_space[pred_changes[-1]],
                color="red",
                linestyle="-",
                alpha=0.125,
                linewidth=2.5,
                label=f"Prediction > {pred_threshold}",
            )

    elif task == "multiclass":
        predicted_classes = predictions.mean(axis=1).argmax(axis=1)
        class_changes = np.where(np.diff(predicted_classes) != 0)[0]
        if class_changes.size > 0:
            for change in class_changes[:-1]:
                ax1.axvline(
                    x=observed_space[change],
                    color="red",
                    linestyle="-",
                    alpha=0.125,
                    linewidth=2.5,
                )
            ax1.axvline(
                x=observed_space[class_changes[-1]],
                color="red",
                linestyle="-",
                alpha=0.125,
                linewidth=2.5,
                label="Predicted class change",
            )

    elif task == "regression":
        ax2 = ax1.twinx()
        lower_pred, mean_preds, upper_pred = np.quantile(
            predictions,
            [0.025, 0.5, 0.975],
            axis=1,
        )
        ax2.plot(
            observed_space,
            mean_preds,
            color="red",
            linestyle="-",
            linewidth=2.5,
            label="Mean prediction",
            alpha=0.7,
        )
        ax2.fill_between(
            observed_space,
            lower_pred,
            upper_pred,
            color="red",
            alpha=0.2,
        )
        ax2.set_ylabel("Predicted value")
        ax2.legend(loc="upper right")

    ax1.set_title(
        "Covariate contributions. "
        f"{feature_names[feature_in_focus]} is adjusted. "
        f"{original_value}"
    )
    ax1.legend(loc="upper left")
    plt.tight_layout()

    if save_path is not None:
        output_path = Path(f"{save_path}.png")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, bbox_inches="tight")
    plt.show()


def plot_global_explain_piecewise_linear_act(
    contributions: NDArray[np.float64],
    predictions: NDArray,
    n_classes: int = 1,
    task: Literal["binary", "multiclass", "regression"] = "binary",
    feature_names: Sequence[Any] | None = None,
    class_names: Sequence[Any] | None = None,
    covariate_indices: Sequence[int] | None = None,
    no_zero_contributions: bool = True,
    split: bool = True,
    fig_size: tuple[float, float] = (10, 6),
    violin_width: float = 1.0,
    save_path: str | None = None,
    show: bool = False,
) -> list[str]:
    """Plot violin-based global explanations for piecewise linear activations.

    For binary classification, a single violin plot is produced with
    predicted class as the hue. For multi-class problems, one violin plot
    is produced per class, showing contributions of samples predicted to
    belong to that class against all others, with zero-contribution
    filtering applied per class. For regression, a single violin plot
    without class-based hue splitting is produced, with the mean predicted
    value annotated in the title.

    Args:
        contributions: Contributions array of shape
            ``(n_samples * n_expl_per_sample, n_features, n_classes)``,
            as returned by ``compute_global_explain_piecewise_linear_act``.
        predictions: For classification tasks, predicted class
            indices of shape ``(n_samples * n_expl_per_sample,)``. For
            regression, mean predicted values of the same shape. Both
            are returned by ``compute_global_explain_piecewise_linear_act``.
        n_classes: Number of output classes. Ignored for regression.
        task: Type of prediction task. One of ``"binary"``,
            ``"multiclass"``, or ``"regression"``. Must match the value
            used in ``compute_global_explain_piecewise_linear_act``.
        feature_names: Optional names for the input features. Defaults
            to ``["x0", "x1", ...]``.
        class_names: Optional names for the output classes. Used in plot
            titles and legend labels. Ignored for regression.
        covariate_indices: Indices of the feature columns to include in
            the violin plot. Defaults to all features.
        no_zero_contributions: Whether to exclude features whose
            contributions are zero across all samples. For multiclass,
            filtering is applied per class so that only features relevant
            to each specific class are shown.
        split: Whether to show a split violin plot with both predicted
            and non-predicted class contributions side by side. When
            ``False``, only samples predicted as the current class are
            shown in a single violin. Applies to both ``"binary"`` and
            ``"multiclass"`` tasks.
        fig_size: Figure size passed to matplotlib.
        violin_width: Width of the violin plots. Increase beyond the
            default of ``1.0`` for wider violins when fewer covariates
            are displayed, or reduce it when many covariates are shown.
        save_path: Optional file prefix for saving plots. When
            ``task="multiclass"`` a class index suffix is appended before
            the extension.
        show: Whether to display the plots interactively.

    Returns:
        A list of saved plot file paths.
    """
    plt, _, _ = get_matplotlib()
    import seaborn as sns

    n_features = contributions.shape[1]

    if feature_names is None:
        feature_names_list = [f"x{i}" for i in range(n_features)]
    else:
        feature_names_list = list(feature_names)

    if class_names is None:
        class_names_list = [f"Class {i}" for i in range(max(n_classes, 2))]
    else:
        class_names_list = list(class_names)

    # For binary and regression, filter globally before the loop
    if no_zero_contributions and task != "multiclass":
        keep = ~(contributions == 0).all(axis=(0, 2))
        contributions = contributions[:, keep, :]
        feature_names_list = [name for name, k in zip(feature_names_list, keep) if k]

    if covariate_indices is None:
        covariate_indices = list(range(contributions.shape[1]))

    saved_paths: list[str] = []
    n_plots = 1 if task in ("binary", "regression") else n_classes

    for class_idx in range(n_plots):

        # For multiclass, filter zero contributions per class
        if no_zero_contributions and task == "multiclass":
            keep = ~(contributions[:, :, class_idx] == 0).all(axis=0)
            class_contributions = contributions[:, keep, class_idx]
            class_feature_names = [
                name for name, k in zip(feature_names_list, keep) if k
            ]
        else:
            class_contributions = contributions[:, :, class_idx]
            class_feature_names = feature_names_list

        class_selected_cols = [
            class_feature_names[i]
            for i in covariate_indices
            if i < len(class_feature_names)
        ]

        df = pd.DataFrame(class_contributions, columns=class_feature_names)

        fig = plt.figure(figsize=fig_size)
        ax = fig.add_subplot(111)
        sns.set_theme(style="whitegrid")

        if task == "regression":
            selected_cols = [feature_names_list[i] for i in covariate_indices]
            dfm = df[selected_cols].melt(var_name="covariates", value_name="β-value")

            data_per_covariate = [
                dfm[dfm["covariates"] == col]["β-value"].values for col in selected_cols
            ]
            parts = ax.violinplot(
                data_per_covariate,
                positions=range(len(selected_cols)),
                widths=violin_width * 0.4,
                showmedians=False,
                showextrema=False,
            )
            for pc in parts["bodies"]:
                pc.set_facecolor("steelblue")
                pc.set_alpha(0.7)
                pc.set_edgecolor("k")
                pc.set_linewidth(0.5)

            for i, covariate in enumerate(selected_cols):
                subset = dfm[dfm["covariates"] == covariate]["β-value"]
                q05, q50, q95 = np.percentile(subset, [5, 50, 95])
                ax.vlines(i, q05, q95, linewidth=3, colors="k")
                ax.scatter(
                    i, q50, color="white", s=10, edgecolors="k", linewidths=1, zorder=3
                )

            ax.set_xticks(range(len(selected_cols)))
            ax.set_xticklabels(selected_cols, rotation=45, ha="right")
            mean_pred = predictions.mean()
            ax.set_title(
                f"Global covariate contributions to model prediction "
                f"(mean predicted value: {mean_pred:.3f})"
            )

        else:
            if task == "binary":
                hue_label = "Predicted class"
                legend_labels = class_names_list[:2]
                if split:
                    df["predictions"] = predictions
                else:
                    df = df[predictions == class_idx]
            else:
                hue_label = f"Predicted as {class_names_list[class_idx]}"
                legend_labels = [class_names_list[class_idx]]
                if split:
                    df["predictions"] = (predictions == class_idx).astype(int)
                else:
                    df = df[predictions == class_idx]

            if split:
                dfm = df[class_selected_cols + ["predictions"]].melt(
                    "predictions",
                    var_name="covariates",
                    value_name="β-value",
                )
                sns.violinplot(
                    data=dfm,
                    x="covariates",
                    y="β-value",
                    hue="predictions",
                    split=True,
                    gap=0.1,
                    cut=2,
                    width=violin_width,
                    ax=ax,
                )

                handles, _ = ax.get_legend_handles_labels()
                ax.legend(handles, legend_labels, title=hue_label)
            else:
                dfm = df[class_selected_cols].melt(
                    var_name="covariates",
                    value_name="β-value",
                )
                sns.violinplot(
                    data=dfm,
                    x="covariates",
                    y="β-value",
                    cut=2,
                    width=violin_width,
                    ax=ax,
                )

            ax.set_title(
                f"Global covariate contributions — {class_names_list[class_idx]}"
                if task == "multiclass"
                else "Global covariate contributions to model prediction"
            )

            handles, _ = ax.get_legend_handles_labels()
            ax.legend(handles, legend_labels, title=hue_label)
            ax.set_title(
                f"Global covariate contributions — {class_names_list[class_idx]}"
                if task == "multiclass"
                else "Global covariate contributions to model prediction"
            )

        ax.set_xlabel("covariates")
        ax.set_ylabel("β-value")
        plt.xticks(rotation=90)
        plt.tight_layout()

        if save_path is not None:
            out_path = (
                f"{save_path}{class_idx}.png"
                if task == "multiclass"
                else f"{save_path}.png"
            )
            ensure_parent(out_path)
            fig.savefig(out_path, bbox_inches="tight")
            saved_paths.append(out_path)

        if show:
            plt.show()

        plt.close(fig)

    return saved_paths