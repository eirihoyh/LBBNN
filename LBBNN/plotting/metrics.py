from __future__ import annotations
import copy
import numpy as np
from ._common import ensure_parent
from .. import inspection as insp


def get_metrics(net, threshold: float = 0.5):
    # net = copy.deepcopy(net)
    net.eval()
    clean_alpha_list = insp.clean_alpha(net, threshold)
    p = clean_alpha_list[0].shape[1]
    layer_names = insp.create_layer_name_list(n_layers=len(clean_alpha_list) + 1)
    density, used_weights, tot_weights = insp.network_density_reduction(clean_alpha_list)
    expected_density = insp.expected_number_of_weights(net)
    mean_path_length, _ = insp.average_path_length(clean_alpha_list)
    include_inputs = insp.include_input_from_layer(clean_alpha_list)
    prob_include_input = insp.input_inclusion_prob(net)
    width_prob = insp.prob_width(net, p)
    return {
        "layer_names": layer_names, 
        "tot_weights": tot_weights, 
        "used_weights_median": used_weights, 
        "density_median": density, 
        "expected_nr_weights_full": expected_density, 
        "density_full": expected_density / tot_weights, 
        "avg_path_length": mean_path_length, 
        "include_inputs": include_inputs, 
        "input_inclusion_prob": prob_include_input, 
        "width_prob": width_prob}


def save_metrics(net, threshold: float = 0.5, path: str = "results/all_metrics"):
    clean_alpha_list = insp.clean_alpha(net, threshold)
    p = clean_alpha_list[0].shape[1]
    layer_names = insp.create_layer_name_list(n_layers=len(clean_alpha_list) + 1)
    density, used_weights, tot_weights = insp.network_density_reduction(clean_alpha_list)
    mean_path_length, _ = insp.average_path_length(clean_alpha_list)
    include_inputs = insp.include_input_from_layer(clean_alpha_list)
    
    metrics_median = {
        "layer_names": layer_names, 
        "tot_weights": tot_weights, 
        "used_weights": used_weights, 
        "density": density, 
        "avg_path_length": mean_path_length, 
        "include_inputs": include_inputs}
    
    ensure_parent(path)
    np.save(str(path) + "_median", metrics_median)
    
    expected_density = insp.expected_number_of_weights(net)
    prob_include_input = insp.input_inclusion_prob(net)
    
    metrics_full = {
        "layer_names": layer_names, 
        "tot_weights": tot_weights, 
        "expected_nr_of_weights": expected_density, 
        "density": expected_density / tot_weights, 
        "expected_nr": prob_include_input, 
        "width_prob": insp.prob_width(net, p)}
    
    np.save(str(path) + "_full", metrics_full)
    
    return str(path) + "_median.npy", str(path) + "_full.npy"
