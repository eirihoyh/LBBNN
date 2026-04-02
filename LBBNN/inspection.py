from __future__ import annotations
import copy
import numpy as np
import torch

def nr_hidden_layers(net) -> int:
    return len(net.linears) - 1

def weight_matrices(net):
    return [copy.deepcopy(layer.weight_mu.data) for layer in net.linears]

def weight_matrices_numpy(net, flow: bool = False):
    w = [tensor.detach().cpu().numpy() for tensor in weight_matrices(net)]
    if flow:
        z = z_matrices_numpy(net)
        for j in range(min(len(w), len(z))):
            w[j] *= z[j]
    return w

def z_matrices(net):
    vals = []
    for layer in net.linears:
        if hasattr(layer, 'q0_mean'):
            vals.append(copy.deepcopy(layer.q0_mean.data))
    return vals

def z_matrices_numpy(net):
    return [tensor.detach().cpu().numpy() for tensor in z_matrices(net)]

def weight_matrices_std(net):
    return [copy.deepcopy(torch.log1p(torch.exp(layer.weight_rho)).cpu().data) for layer in net.linears]

def weight_matrices_std_numpy(net):
    return [tensor.detach().numpy() for tensor in weight_matrices_std(net)]

def get_alphas(net):
    return [copy.deepcopy(torch.sigmoid(layer.lambdal.detach().cpu())) for layer in net.linears]

def get_alphas_numpy(net):
    return [alpha.detach().numpy() for alpha in get_alphas(net)]

def clean_alpha(net, threshold: float, alpha_list=None):
    if alpha_list is None:
        alpha_list = get_alphas(net)
    dim = alpha_list[0].shape[0]
    clean_dict = {ind: (alpha > threshold).float() for ind, alpha in enumerate(alpha_list)}
    for ind in np.arange(1, len(alpha_list))[::-1]:
        clean_dict[ind - 1] = (clean_dict[ind - 1].T * (torch.sum(clean_dict[ind][:, :dim], dim=0) > 0)).T.float()
    for ind in np.arange(1, len(alpha_list)):
        clean_dict[ind] = torch.cat(
            ((clean_dict[ind][:, :dim] * (torch.sum(clean_dict[ind - 1].T, dim=0) > 0)).float(), clean_dict[ind][:, dim:]), 
            dim=1)
    return list(clean_dict.values())

def clean_alpha_class(net, threshold: float, class_in_focus: int = 0, alpha_list=None):
    if alpha_list is None:
        alpha_list = get_alphas(net)
    alpha_list = [a.clone() for a in alpha_list]
    num_classes = alpha_list[-1].shape[0]
    remove_mask = torch.ones(num_classes, dtype=torch.bool)
    remove_mask[class_in_focus] = False
    alpha_list[-1][remove_mask, :] = 0
    return clean_alpha(net, threshold, alpha_list=alpha_list)

def get_active_weights(clean_alpha_list):
    return [alpha.nonzero() for alpha in clean_alpha_list]

def network_density_reduction(clean_alpha_list):
    used_weights = 0.0
    total_weights = 0
    for a in clean_alpha_list:
        used_weights += float(a.sum().item())
        total_weights += int(a.numel())
    return used_weights / total_weights, used_weights, total_weights

def create_layer_name_list(n_layers: int | None = None, net=None):
    if net is not None:
        n_layers = nr_hidden_layers(net) + 2
    if n_layers is None:
        raise ValueError('Provide either n_layers or net.')
    layers = ['I']
    for layer in range(n_layers - 2):
        layers.append(f'H{layer+1}')
    layers.append('Output')
    return layers

def input_inclusion_prob(net, a=None):
    if a is None:
        a = get_alphas_numpy(net)
    length = len(a)
    p = a[0].shape[1]
    prob_paths = {}
    layer_names = create_layer_name_list(n_layers=length + 1)
    for name in layer_names[:-1]:
        for i in range(p):
            prob_paths[f'Prob I{i} from {name}'] = 0
    lims = np.arange(1, length, 1)[::-1]
    if len(lims) == 0:
        name = layer_names[0]
        for xi in range(p):
            prob_paths[f'Prob I{xi} from {name}'] = float(a[0][0][xi])
    else:
        for i, name in enumerate(layer_names[:-1]):
            probs = a[i][:, -p:].T
            count = 0
            while i < len(lims) and count < lims[i]:
                count += 1
                probs = probs @ a[i + count][:, :-p].T
            for xi in range(p):
                prob_paths[f'Prob I{xi} from {name}'] = float(probs[xi][0])
    return prob_paths

def expected_number_of_weights(net):
    return float(sum(np.sum(a) for a in get_alphas_numpy(net)))

def include_input_from_layer(clean_alpha_list):
    p = clean_alpha_list[0].shape[1]
    return [np.sum(alpha[:, -p:].detach().numpy(), axis=0) > 0 for alpha in clean_alpha_list]

def average_path_length(clean_alpha_list):
    length_list = len(clean_alpha_list)
    p = clean_alpha_list[0].shape[1]
    sum_dists = np.array([])
    for i in range(length_list):
        for xi in range(p):
            path_length = clean_alpha_list[i][:, -(xi + 1)].detach().numpy() * (length_list - i)
            path_length = path_length[path_length != 0]
            if path_length.size:
                sum_dists = np.concatenate((sum_dists, path_length))
    return (float(np.mean(sum_dists)) if sum_dists.size else 0.0), sum_dists

def prob_width(net, p):
    probs = input_inclusion_prob(net)
    vals = list(probs.values())
    return {i: float(min(np.sum(vals[i::p]), 1.0)) for i in range(p)}

def get_weight_and_bias_std(net, alphas_numpy, threshold: float = 0.5):
    std_weight = weight_matrices_std_numpy(net)
    for i in range(len(std_weight)):
        std_weight[i] *= (alphas_numpy[i] > threshold) * 1.0
    return std_weight

def get_weight_and_bias(net, alphas_numpy, median: bool = True, sample: bool = False, threshold: float = 0.5):
    weights = weight_matrices_numpy(net)
    std_weight = weight_matrices_std_numpy(net)
    if sample:
        for i in range(len(weights)):
            weights[i] = weights[i] + np.random.normal(0, std_weight[i])
    if median:
        for i in range(len(weights)):
            weights[i] *= (alphas_numpy[i] > threshold) * 1.0
    else:
        for i in range(len(weights)):
            include = np.random.binomial(1, alphas_numpy[i]) * 1.0
            weights[i] *= include
            alphas_numpy[i] = copy.deepcopy(include)
    return weights, alphas_numpy
