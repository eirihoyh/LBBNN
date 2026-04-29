from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ._common import ensure_parent, get_matplotlib
from .._types import BayesianNet
from .. import inspection as insp


def plot_model_vision_image(
    net: BayesianNet,
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


