from __future__ import annotations

from typing import Any

import numpy as np

from ._common import ensure_parent
from .._types import BayesianNet
from .. import inspection as insp


def get_metrics(net: BayesianNet, threshold: float = 0.5) -> dict[str, Any]:
    """Compute structural summary metrics for a network.

    Args:
        net: Trained network object.
        threshold: Threshold used to clean inclusion probabilities.

    Returns:
        A dictionary with layer names, density measures, path statistics,
        and input inclusion summaries.
    """
    net.eval()

    clean_alpha_list = insp.clean_alpha(net, threshold)
    p = clean_alpha_list[0].shape[1]

    layer_names = insp.create_layer_name_list(
        n_layers=len(clean_alpha_list) + 1,
    )
    density, used_weights, total_weights = insp.network_density_reduction(
        clean_alpha_list
    )
    expected_nr_weights = insp.expected_number_of_weights(net)
    mean_path_length, _ = insp.average_path_length(clean_alpha_list)
    include_inputs = insp.include_input_from_layer(clean_alpha_list)
    input_inclusion_prob = insp.input_inclusion_prob(net)
    width_prob = insp.prob_width(net, p)

    return {
        "layer_names": layer_names,
        "tot_weights": total_weights,
        "used_weights_median": used_weights,
        "density_median": density,
        "expected_nr_weights_full": expected_nr_weights,
        "density_full": expected_nr_weights / total_weights,
        "avg_path_length": mean_path_length,
        "include_inputs": include_inputs,
        "input_inclusion_prob": input_inclusion_prob,
        "width_prob": width_prob,
    }


def save_metrics(
    net: BayesianNet,
    threshold: float = 0.5,
    path: str = "results/all_metrics",
) -> tuple[str, str]:
    """Compute and save network metrics to disk.

    Args:
        net: Trained network object.
        threshold: Threshold used to clean inclusion probabilities.
        path: Base path used when saving the metric files.

    Returns:
        A tuple containing the saved median-metric path and full-metric path.
    """
    m = get_metrics(net, threshold)

    ensure_parent(path)

    metrics_median = {
        "layer_names": m["layer_names"],
        "tot_weights": m["tot_weights"],
        "used_weights": m["used_weights_median"],
        "density": m["density_median"],
        "avg_path_length": m["avg_path_length"],
        "include_inputs": m["include_inputs"],
    }
    median_path = f"{path}_median.npy"
    np.save(f"{path}_median", metrics_median)

    metrics_full = {
        "layer_names": m["layer_names"],
        "tot_weights": m["tot_weights"],
        "expected_nr_of_weights": m["expected_nr_weights_full"],
        "density": m["density_full"],
        "expected_nr": m["input_inclusion_prob"],
        "width_prob": m["width_prob"],
    }
    full_path = f"{path}_full.npy"
    np.save(f"{path}_full", metrics_full)

    return median_path, full_path