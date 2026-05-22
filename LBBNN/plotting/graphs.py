from __future__ import annotations

from typing import Any

from ._common import ensure_parent, get_graphviz_digraph
from .._types import BayesianNet
from .. import inspection as insp


def _build_path_graph(
    value_list: list[Any],
    all_connections: list[Any],
    save_path: str | None,
    label_prefix: Literal['α', 'w'] = 'α',
    show: bool = False,
    splines: str = "line",
    rankdir: Literal["TB", "LR"] = "LR",
    node_shape: str = "box",
    node_color: str = "lightblue",
    skip_color: str = "lightyellow",
    show_edge_labels: bool = True,
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
        splines: Edge routing style passed to Graphviz. Controls how
            edges are drawn between nodes. Common values are
            ``"ortho"`` for right-angle edges, ``"polyline"`` for
            straight lines with bends, ``"curved"`` for smooth curves,
            and ``"line"`` for direct straight lines. Defaults to
            ``"ortho"``.
        rankdir: Direction of the graph layout. ``"LR"`` renders the
            network left to right, which is recommended when skip
            connections are present as it makes the network structure
            more readable. ``"TB"`` renders top to bottom. Defaults
            to ``"LR"``.
        node_shape: Shape of regular layer nodes. Defaults to ``"box"``.
        node_color: Fill colour of regular layer nodes. Defaults to
            ``"lightblue"``.
        skip_color: Fill colour of skip connection nodes, which are
            rendered as diamonds to visually distinguish them from
            regular nodes. Defaults to ``"lightyellow"``. Skip
            connection nodes are duplicated per layer occurrence so
            they appear at the rank they originate from, making the
            skip connection structure of the network explicit.

    Returns:
        The Graphviz digraph object (rendered if ``save_path`` was given).
    """
    Digraph = get_graphviz_digraph()
    dot = Digraph("All paths")

    dot.graph_attr.update(
        rankdir=rankdir,
        splines=splines,
        nodesep="0.5",
        ranksep="0.8",
        bgcolor="white",
    )
    dot.node_attr.update(
        shape=node_shape,
        style="rounded,filled",
        fillcolor=node_color,
        fontname="Helvetica",
        fontsize="11",
    )
    dot.edge_attr.update(
        fontname="Helvetica",
        fontsize="9",
        color="gray40",
    )

    n_layers = len(value_list) + 1
    dim = value_list[0].shape[0]
    layer_names = insp.create_layer_name_list(n_layers=n_layers)
    seen_nodes: set[str] = set()
    layer_nodes: dict[int, list[str]] = {i: [] for i in range(n_layers)}

    for layer_ind, connections in enumerate(all_connections):
        for to_idx, from_idx in connections:
            to_idx = int(to_idx)
            from_idx = int(from_idx)

            # For skip connections, create a unique node per layer occurrence
            # so it appears at the rank it originates from
            if from_idx >= dim:
                skip_idx = from_idx - dim
                from_node_id = f"I_{skip_idx}_at_{layer_ind}"
                from_node_label = f"I_{skip_idx}"
                if from_node_id not in seen_nodes:
                    dot.node(
                        from_node_id,
                        label=from_node_label,
                        shape="diamond",
                        fillcolor=skip_color,
                    )
                    seen_nodes.add(from_node_id)
                    layer_nodes[layer_ind].append(from_node_id)
            else:
                from_node_id = insp.format_node_name(
                    index=from_idx,
                    layer_ind=layer_ind,
                    dim=dim,
                    n_layers=n_layers,
                    layer_names=layer_names,
                )
                if from_node_id not in seen_nodes:
                    if layer_ind == 0:
                        dot.node(
                            from_node_id,
                            shape="diamond",
                            fillcolor=skip_color,
                        )
                    else:
                        dot.node(from_node_id)
                    seen_nodes.add(from_node_id)
                    layer_nodes[layer_ind].append(from_node_id)

            to_node = insp.format_node_name(
                index=to_idx,
                layer_ind=layer_ind + 1,
                dim=dim,
                n_layers=n_layers,
                layer_names=layer_names,
            )
            
            if to_node not in seen_nodes:
                dot.node(to_node)
                seen_nodes.add(to_node)
                layer_nodes[layer_ind + 1].append(to_node)
            
            value = float(value_list[layer_ind][to_idx][from_idx])
            if show_edge_labels:
                dot.edge(from_node_id, to_node, xlabel=f"{label_prefix}={value:.2f}")
            else:
                dot.edge(from_node_id, to_node)

    # Enforce same rank per layer so nodes align vertically
    for layer_ind, nodes in layer_nodes.items():
        if nodes:
            with dot.subgraph() as sub:
                sub.attr(rank="same")
                for node in nodes:
                    sub.node(node)

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
    show_edge_labels: bool = False,
    splines: str = "line",
    rankdir: Literal["TB", "LR"] = "LR",
) -> Any:
    """Plot all active paths using alpha values as edge labels.

    Args:
        alpha_list: List of alpha tensors or arrays per layer.
        all_connections: List of active connections per layer.
        save_path: Output path for the rendered graph. If ``None`` the
            graph is built but not written to disk.
        show: Whether to open the rendered graph after saving.
        show_edge_labels: Whether to display alpha values as edge labels.
        splines: Edge routing style passed to Graphviz.
        rankdir: Direction of the graph layout. ``"LR"`` or ``"TB"``.

    Returns:
        The Graphviz digraph object.
    """
    return _build_path_graph(
        value_list=alpha_list,
        all_connections=all_connections,
        save_path=save_path,
        label_prefix="α",
        show=show,
        show_edge_labels=show_edge_labels,
        splines=splines,
        rankdir=rankdir,
    )


def plot_whole_path_graph_weight(
    weight_list: list[Any],
    all_connections: list[Any],
    save_path: str | None = None,
    show: bool = False,
    show_edge_labels: bool = False,
    splines: str = "line",
    rankdir: Literal["TB", "LR"] = "LR",
) -> Any:
    """Plot all active paths using weight values as edge labels.

    Args:
        weight_list: List of weight matrices per layer.
        all_connections: List of active connections per layer.
        save_path: Output path for the rendered graph. If ``None`` the
            graph is built but not written to disk.
        show: Whether to open the rendered graph after saving.
        show_edge_labels: Whether to display weight values as edge labels.
        splines: Edge routing style passed to Graphviz.
        rankdir: Direction of the graph layout. ``"LR"`` or ``"TB"``.

    Returns:
        The Graphviz digraph object.
    """
    return _build_path_graph(
        value_list=weight_list,
        all_connections=all_connections,
        save_path=save_path,
        label_prefix="w",
        show=show,
        show_edge_labels=show_edge_labels,
        splines=splines,
        rankdir=rankdir,
    )


def run_path_graph(
    net: BayesianNet,
    threshold: float = 0.5,
    save_path: str | None = None,
    show: bool = False,
    show_edge_labels: bool = False,
    splines: str = "line",
    rankdir: Literal["TB", "LR"] = "LR",
) -> Any:
    """Build a path graph from the network's alpha values.

    Args:
        net: Trained network object.
        threshold: Threshold used to clean alpha values.
        save_path: Output path for the rendered graph. If ``None`` the
            graph is built but not written to disk.
        show: Whether to open the rendered graph after saving.
        show_edge_labels: Whether to display alpha values as edge labels.
        splines: Edge routing style passed to Graphviz.
        rankdir: Direction of the graph layout. ``"LR"`` or ``"TB"``.

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
        show_edge_labels=show_edge_labels,
        splines=splines,
        rankdir=rankdir,
    )


def run_path_graph_weight(
    net: BayesianNet,
    threshold: float = 0.5,
    save_path: str | None = None,
    show: bool = False,
    flow: bool = False,
    show_edge_labels: bool = False,
    splines: str = "line",
    rankdir: Literal["TB", "LR"] = "LR",
) -> Any:
    """Build a path graph from the network's weight values.

    Args:
        net: Trained network object.
        threshold: Threshold used to clean alpha values.
        save_path: Output path for the rendered graph. If ``None`` the
            graph is built but not written to disk.
        show: Whether to open the rendered graph after saving.
        flow: Whether to apply flow-adjusted weights.
        show_edge_labels: Whether to display weight values as edge labels.
        splines: Edge routing style passed to Graphviz.
        rankdir: Direction of the graph layout. ``"LR"`` or ``"TB"``.

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
        show_edge_labels=show_edge_labels,
        splines=splines,
        rankdir=rankdir,
    )


def plot_path_individual_classes(
    net: BayesianNet,
    classes: int,
    path: str | None = None,
    show: bool = False,
    show_edge_labels: bool = False,
    splines: str = "line",
    rankdir: Literal["TB", "LR"] = "LR",
) -> list[str]:
    """Build per-class path graphs and optionally write them to disk.

    Args:
        net: Trained network object.
        classes: Number of output classes.
        path: Base directory for saved graphs. If ``None`` the graphs
            are built but not written to disk and an empty list is
            returned.
        show: Whether to open each rendered graph after saving.
        show_edge_labels: Whether to display alpha values as edge labels.
        splines: Edge routing style passed to Graphviz.
        rankdir: Direction of the graph layout. ``"LR"`` or ``"TB"``.

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
            show_edge_labels=show_edge_labels,
            splines=splines,
            rankdir=rankdir,
        )
        if target_path is not None:
            saved_paths.append(f"{target_path}.png")

    return saved_paths