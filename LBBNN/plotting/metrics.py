from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ._common import ensure_parent
from .._types import BayesianNet
from .. import inspection as insp


def build_path_graph_table(
    alpha_list: list[Any],
    weight_list: list[Any],
    all_connections: list[Any],
    save_path: str | None = None,
    to_markdown: bool = True,
) -> pd.DataFrame:
    """Build a summary table of edge weights and alpha values for all
    active paths in the network.

    Each row corresponds to one active connection, identified by its
    source and target node names, and includes both the alpha and weight
    values for that edge.

    Args:
        alpha_list: List of alpha tensors or arrays per layer.
        weight_list: List of weight tensors or arrays per layer.
        all_connections: List of active connections per layer.
        save_path: Optional path prefix for saving the table as a
            ``.csv`` file. If ``None`` the table is only returned.
        to_markdown: Optional markdown table of csv file if 
            save_path is given. 

    Returns:
        A DataFrame with columns ``["From", "To", "α", "w"]``.
    """
    n_layers = len(alpha_list) + 1
    dim = alpha_list[0].shape[0]
    layer_names = insp.create_layer_name_list(n_layers=n_layers)

    rows = []

    for layer_ind, connections in enumerate(all_connections):
        for to_idx, from_idx in connections:
            to_idx = int(to_idx)
            from_idx = int(from_idx)

            # Format from_node label
            if from_idx >= dim:
                skip_idx = from_idx - dim
                from_label = f"I_{skip_idx}"
            else:
                from_label = insp.format_node_name(
                    index=from_idx,
                    layer_ind=layer_ind,
                    dim=dim,
                    n_layers=n_layers,
                    layer_names=layer_names,
                )

            # Format to_node label
            to_label = insp.format_node_name(
                index=to_idx,
                layer_ind=layer_ind + 1,
                dim=dim,
                n_layers=n_layers,
                layer_names=layer_names,
            )

            alpha_value = float(alpha_list[layer_ind][to_idx][from_idx])
            weight_value = float(weight_list[layer_ind][to_idx][from_idx])

            rows.append({
                "From": from_label,
                "To": to_label,
                "α": round(alpha_value, 4),
                "w": round(weight_value, 4),
            })

    df = pd.DataFrame(rows, columns=["From", "To", "α", "w"])

    if save_path is not None:
        ensure_parent(save_path)
        df.to_csv(f"{save_path}.csv", index=False)
        if to_markdown:
            with open(f"{save_path}.md", "w") as f:
                f.write(df.to_markdown(index=False))

    return df

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