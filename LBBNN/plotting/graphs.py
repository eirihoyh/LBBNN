from __future__ import annotations
from ._common import ensure_parent, get_graphviz_digraph
from .. import inspection as insp


def plot_whole_path_graph(
        alpha_list, 
        all_connections, 
        save_path, 
        show: bool = False):

    Digraph = get_graphviz_digraph()
    dot = Digraph("All paths")
    n_layers = len(alpha_list) + 1
    dim = alpha_list[0].shape[0]
    layer_list = insp.create_layer_name_list(n_layers=n_layers)
    seen = []
    for layer_ind, connection in enumerate(all_connections):
        for t, f in connection:
            t, f = int(t), int(f)
            from_node = f"I_{f-dim}" if f >= dim else f"{layer_list[layer_ind]}_{f}"
            to_node = f"I_{t-dim}" if (t >= dim and layer_ind + 1 < n_layers) else f"{layer_list[layer_ind+1]}_{t}"
            if from_node not in seen: dot.node(from_node); seen.append(from_node)
            if to_node not in seen: dot.node(to_node); seen.append(to_node)
            dot.edge(from_node, to_node, label=f"α={float(alpha_list[layer_ind][t][f]):.2f}")
    dot.node("All paths", shape="Msquare")
    dot.format = "png"
    dot.strict = True
    ensure_parent(save_path)
    dot.render(str(save_path), view=show)
    return dot


def plot_whole_path_graph_weight(
        weight_list, 
        all_connections, 
        save_path, 
        show: bool = False):
    
    Digraph = get_graphviz_digraph()
    dot = Digraph("All paths")
    n_layers = len(weight_list) + 1
    dim = weight_list[0].shape[0]
    layer_list = insp.create_layer_name_list(n_layers=n_layers)
    seen = []
    for layer_ind, connection in enumerate(all_connections):
        for t, f in connection:
            t, f = int(t), int(f)
            from_node = f"I_{f-dim}" if f >= dim else f"{layer_list[layer_ind]}_{f}"
            to_node = f"I_{t-dim}" if (t >= dim and layer_ind + 1 < n_layers) else f"{layer_list[layer_ind+1]}_{t}"
            if from_node not in seen: dot.node(from_node); seen.append(from_node)
            if to_node not in seen: dot.node(to_node); seen.append(to_node)
            dot.edge(from_node, to_node, label=f"w={float(weight_list[layer_ind][t][f]):.2f}")
    dot.node("All paths", shape="Msquare")
    dot.format = "png"
    dot.strict = True
    ensure_parent(save_path)
    dot.render(str(save_path), view=show)
    return dot


def run_path_graph(
        net, 
        threshold: float = 0.5, 
        save_path: str = "path_graphs/all_paths_input_skip", 
        show: bool = False):
    
    alpha_list = insp.get_alphas(net)
    clean_alpha_list = insp.clean_alpha(net, threshold)
    all_connections = insp.get_active_weights(clean_alpha_list)
    return plot_whole_path_graph(alpha_list, all_connections, save_path=save_path, show=show)


def run_path_graph_weight(
        net, 
        threshold: float = 0.5, 
        save_path: str = "path_graphs/all_paths_input_skip", 
        show: bool = False, 
        flow: bool = False):
    
    weight_list = insp.weight_matrices_numpy(net, flow=flow)
    clean_alpha_list = insp.clean_alpha(net, threshold)
    all_connections = insp.get_active_weights(clean_alpha_list)
    return plot_whole_path_graph_weight(weight_list, all_connections, save_path=save_path, show=show)


def plot_path_individual_classes(
        net, 
        classes: int, 
        path: str = "individual_classes", 
        show: bool = False):
    saved = []
    for c in range(classes):
        include_list = [True] * classes
        include_list[c] = False
        a = insp.get_alphas(net)
        a[-1][include_list, :] = 0
        clean_a = insp.clean_alpha(net, 0.5, alpha_list=a)
        all_connections = insp.get_active_weights(clean_a)
        target = f"{path}/class{c}"
        plot_whole_path_graph(a, all_connections, save_path=target, show=show)
        saved.append(target + ".png")
    return saved
