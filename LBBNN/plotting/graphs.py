from __future__ import annotations

from typing import Any

from ._common import ensure_parent, get_graphviz_digraph
from .._types import BayesianNet
from .. import inspection as insp


def _build_path_graph(
    value_list: list[Any],
    all_connections: list[Any],
    save_path: str | None,
    label_prefix: str,
    show: bool = False,
) -> Any:
    """Build a graph of active paths and optionally render it.

    Args:
        value_list: List of tensors or arrays containing edge values.
        all_connections: List of active connections per layer.
        save_path: Output path for the rendered graph. If ``None`` no
            file is written and the unrendered Digraph is returned.
        label_prefix: Prefix used in edge labels, e.g. ``'α'`` or ``'w'``.
        show: Whether to open the rendered graph after saving (only
            applies when ``save_path`` is not None).

    Returns:
        The Graphviz digraph object (rendered if ``save_path`` was given).
    """
    Digraph = get_graphviz_digraph()
    dot = Digraph("All paths")

    n_layers = len(value_list) + 1
    dim = value_list[0].shape[0]
    layer_names = insp.create_layer_name_list(n_layers=n_layers)
    seen_nodes: set[str] = set()

    for layer_ind, connections in enumerate(all_connections):
        for to_idx, from_idx in connections:
            to_idx = int(to_idx)
            from_idx = int(from_idx)

            from_node = _format_node_name(
                index=from_idx,
                layer_ind=layer_ind,
                dim=dim,
                n_layers=n_layers,
                layer_names=layer_names,
            )
            to_node = _format_node_name(
                index=to_idx,
                layer_ind=layer_ind + 1,
                dim=dim,
                n_layers=n_layers,
                layer_names=layer_names,
            )

            if from_node not in seen_nodes:
                dot.node(from_node)
                seen_nodes.add(from_node)

            if to_node not in seen_nodes:
                dot.node(to_node)
                seen_nodes.add(to_node)

            value = float(value_list[layer_ind][to_idx][from_idx])
            dot.edge(from_node, to_node, label=f"{label_prefix}={value:.2f}")

    dot.node("All paths", shape="Msquare")
    dot.format = "png"
    dot.strict = True

    if save_path is not None:
        ensure_parent(save_path)
        dot.render(str(save_path), view=show)

    return dot


def plot_whole_path_graph(
    alpha_list: list[Any],
    all_connections: list[Any],
    save_path: str | None = None,
    show: bool = False,
) -> Any:
    """Plot all active paths using alpha values as edge labels.

    Args:
        alpha_list: List of alpha tensors or arrays per layer.
        all_connections: List of active connections per layer.
        save_path: Output path for the rendered graph. If ``None`` the
            graph is built but not written to disk.
        show: Whether to open the rendered graph after saving.

    Returns:
        The Graphviz digraph object.
    """
    return _build_path_graph(
        value_list=alpha_list,
        all_connections=all_connections,
        save_path=save_path,
        label_prefix="α",
        show=show,
    )


def plot_whole_path_graph_weight(
    weight_list: list[Any],
    all_connections: list[Any],
    save_path: str | None = None,
    show: bool = False,
) -> Any:
    """Plot all active paths using weight values as edge labels.

    Args:
        weight_list: List of weight matrices per layer.
        all_connections: List of active connections per layer.
        save_path: Output path for the rendered graph. If ``None`` the
            graph is built but not written to disk.
        show: Whether to open the rendered graph after saving.

    Returns:
        The Graphviz digraph object.
    """
    return _build_path_graph(
        value_list=weight_list,
        all_connections=all_connections,
        save_path=save_path,
        label_prefix="w",
        show=show,
    )


def run_path_graph(
    net: BayesianNet,
    threshold: float = 0.5,
    save_path: str | None = None,
    show: bool = False,
) -> Any:
    """Build a path graph from the network's alpha values.

    Args:
        net: Trained network object.
        threshold: Threshold used to clean alpha values.
        save_path: Output path for the rendered graph. If ``None`` the
            graph is built but not written to disk.
        show: Whether to open the rendered graph after saving.

    Returns:
        The Graphviz digraph object.
    """
    alpha_list = insp.get_alphas(net)
    clean_alpha_list = insp.clean_alpha(net, threshold)
    all_connections = insp.get_active_weights(clean_alpha_list)

    return plot_whole_path_graph(
        alpha_list=alpha_list,
        all_connections=all_connections,
        save_path=save_path,
        show=show,
    )


def run_path_graph_weight(
    net: BayesianNet,
    threshold: float = 0.5,
    save_path: str | None = None,
    show: bool = False,
    flow: bool = False,
) -> Any:
    """Build a path graph from the network's weight values.

    Args:
        net: Trained network object.
        threshold: Threshold used to clean alpha values.
        save_path: Output path for the rendered graph. If ``None`` the
            graph is built but not written to disk.
        show: Whether to open the rendered graph after saving.
        flow: Whether to apply flow-adjusted weights.

    Returns:
        The Graphviz digraph object.
    """
    weight_list = insp.weight_matrices_numpy(net, flow=flow)
    clean_alpha_list = insp.clean_alpha(net, threshold)
    all_connections = insp.get_active_weights(clean_alpha_list)

    return plot_whole_path_graph_weight(
        weight_list=weight_list,
        all_connections=all_connections,
        save_path=save_path,
        show=show,
    )


def plot_path_individual_classes(
    net: BayesianNet,
    classes: int,
    path: str | None = None,
    show: bool = False,
) -> list[str]:
    """Build per-class path graphs and optionally write them to disk.

    Args:
        net: Trained network object.
        classes: Number of output classes.
        path: Base directory for saved graphs. If ``None`` the graphs
            are built but not written to disk and an empty list is
            returned.
        show: Whether to open each rendered graph after saving.

    Returns:
        A list of saved image paths (empty when ``path`` is None).
    """
    saved_paths: list[str] = []

    for class_idx in range(classes):
        include_mask = [True] * classes
        include_mask[class_idx] = False

        alphas = insp.get_alphas(net)
        alphas[-1][include_mask, :] = 0

        clean_alphas = insp.clean_alpha(net, 0.5, alpha_list=alphas)
        all_connections = insp.get_active_weights(clean_alphas)

        target_path = f"{path}/class{class_idx}" if path is not None else None
        plot_whole_path_graph(
            alpha_list=alphas,
            all_connections=all_connections,
            save_path=target_path,
            show=show,
        )
        if target_path is not None:
            saved_paths.append(f"{target_path}.png")

    return saved_paths