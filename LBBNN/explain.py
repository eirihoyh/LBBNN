from __future__ import annotations
import copy
import numpy as np
import torch
from .inspection import clean_alpha_class, get_alphas_numpy, get_weight_and_bias

def relu_activation(input_data: torch.Tensor, weights):
    output_list = []
    out = np.empty((input_data.shape[0], 0))
    x_in = input_data.detach().cpu().numpy()
    for w in weights[:-1]:
        out = np.concatenate((out, x_in), axis=1)
        out = out @ w.T
        out = out * (out > 0)
        output_list.append(out)
    out = np.concatenate((out, x_in), axis=1)
    out = out @ weights[-1].T
    output_list.append(out)
    return out, output_list

def get_active_nodes(clean_alpha_list, output_list_c):
    active_nodes_alpha_list = [(np.sum(a.detach().numpy(), axis=1) > 0) * 1 for a in clean_alpha_list]
    return np.array([(active_nodes_alpha_list[i] * output_list_c[i][0] > 0) * 1 for i in range(len(clean_alpha_list) - 1)])

def find_active_weights(weights, active_nodes_list, clean_alpha_list, dim):

    active_weights = copy.deepcopy(weights)
    for i in range(len(active_weights) - 1):
        active_weights[i] = active_weights[i] * clean_alpha_list[i].detach().numpy()
        active_weights[i] = np.array([active_weights[i][j, :] * active_nodes_list[i, j] for j in range(len(active_nodes_list[i]))])
        active_weights[i + 1][:, :dim] = np.array(
            [active_weights[i + 1][:, j] * active_nodes_list[i, j] for j in range(len(active_nodes_list[i]))]).T
        
    active_weights[-1] = active_weights[-1] * clean_alpha_list[-1].detach().numpy()
    return active_weights

def local_explain_relu(net, input_data, threshold=0.5, median=True, sample=False, n_samples=1, verbose=False, quantiles=(0.025, 0.975)):
    contributions, preds = {}, []
    for n in range(n_samples):
        alphas_numpy = get_alphas_numpy(net)
        nr_classes = alphas_numpy[-1].shape[0]
        weights, alphas_numpy = get_weight_and_bias(net, alphas_numpy, median, sample, threshold)
        alphas = [torch.tensor(a) for a in copy.deepcopy(alphas_numpy)]
        out, output_list = relu_activation(input_data, weights)
        preds.append(out)
        contribution_classes = {}
        for c in range(nr_classes):
            weights_class = copy.deepcopy(weights)
            weights_class[-1] = weights_class[-1][c:c+1, :]
            clean_alpha_list = clean_alpha_class(net, threshold=0.5, class_in_focus=c, alpha_list=copy.deepcopy(alphas))
            clean_alpha_list[-1] = clean_alpha_list[-1][c:c+1, :]
            dim, p = clean_alpha_list[0].shape
            active_nodes_list = get_active_nodes(clean_alpha_list, copy.deepcopy(output_list))
            active_weights = find_active_weights(weights_class, active_nodes_list, clean_alpha_list, dim)
            pred_impact = {}
            for pi in range(p):
                explain_this_numpy = copy.deepcopy(input_data.detach().cpu().numpy())
                remove_list = [True] * p
                remove_list[pi] = False
                explain_this_numpy[0, remove_list] = 0
                x = np.array([[]])
                for aw in active_weights:
                    x = np.concatenate((x, explain_this_numpy), 1)
                    x = x @ aw.T
                pred_impact[pi] = x[0, 0]
            contribution_classes[c] = pred_impact
        contributions[n] = contribution_classes
    mean_contribution, cred_contribution = {}, {}
    for c in range(nr_classes):
        mean_contribution[c], cred_contribution[c] = {}, {}
        for pi in range(p):
            values = np.array([contributions[s][c][pi] for s in range(n_samples)])
            mean_contribution[c][pi] = float(np.mean(values))
            cred_contribution[c][pi] = np.quantile(values, quantiles)
    return mean_contribution, cred_contribution, np.array(preds)

def local_explain_relu_magnitude(net, input_data, threshold=0.5, median=True, sample=False, n_samples=1, verbose=False, quantiles=(0.025, 0.975), include_potential_contribution=True):
    contributions, preds = {}, []
    for n in range(n_samples):
        alphas_numpy = get_alphas_numpy(net)
        nr_classes = alphas_numpy[-1].shape[0]
        weights, alphas_numpy = get_weight_and_bias(net, alphas_numpy, median, sample, threshold)
        alphas = [torch.tensor(a) for a in copy.deepcopy(alphas_numpy)]
        out, output_list = relu_activation(input_data, weights)
        preds.append(out)
        contribution_classes = {}
        for c in range(nr_classes):
            weights_class = copy.deepcopy(weights)
            weights_class[-1] = weights_class[-1][c:c+1, :]
            clean_alpha_list = clean_alpha_class(net, threshold=0.5, class_in_focus=c, alpha_list=copy.deepcopy(alphas))
            clean_alpha_list[-1] = clean_alpha_list[-1][c:c+1, :]
            dim, p = clean_alpha_list[0].shape
            active_nodes_list = get_active_nodes(clean_alpha_list, copy.deepcopy(output_list))
            active_weights = find_active_weights(weights_class, active_nodes_list, clean_alpha_list, dim)
            pred_impact = {}
            for pi in range(p):
                explain_this_numpy = np.ones((1, p))
                remove_list = [True] * p
                remove_list[pi] = False
                explain_this_numpy[0, remove_list] = 0
                x = np.array([[]])
                for aw in active_weights:
                    x = np.concatenate((x, explain_this_numpy), 1)
                    x = x @ aw.T
                inp = input_data.detach().cpu().numpy()[0, pi]
                pred_impact[pi] = (-1.0 * x[0, 0] if inp == 0 else x[0, 0]) if include_potential_contribution else (0.0 if inp == 0 else x[0, 0])
            contribution_classes[c] = pred_impact
        contributions[n] = contribution_classes
    mean_contribution, cred_contribution = {}, {}
    for c in range(nr_classes):
        mean_contribution[c], cred_contribution[c] = {}, {}
        for pi in range(p):
            values = np.array([contributions[s][c][pi] for s in range(n_samples)])
            mean_contribution[c][pi] = float(np.mean(values))
            cred_contribution[c][pi] = np.quantile(values, quantiles)
    return mean_contribution, cred_contribution, np.array(preds)

def local_explain_piecewise_linear_act(
        net, 
        input_data, 
        median=True, 
        sample=True, 
        n_samples=1, 
        magnitude=True, 
        include_potential_contribution=False, 
        n_classes=1):
    p = input_data.shape[0]
    explanation = torch.zeros((n_samples, p, n_classes))
    preds = torch.zeros((n_samples, n_classes))
    for j in range(n_samples):
        explain_this = input_data.reshape(-1, p)
        explain_this.requires_grad = True
        net.zero_grad()
        output = net.forward_preact(explain_this, sample=sample, ensemble=not median)
        for c in range(n_classes):
            output_value = output[0, c]
            gradients = torch.autograd.grad(output_value, explain_this, grad_outputs=torch.ones_like(output_value), retain_graph=True)
            explanation[j, :, c] = gradients[0]
            preds[j, c] = output[0, c]
    expl = explanation.cpu().detach().numpy()
    inds = np.where(input_data == 0.0)[0]
    expl[:, inds] = -expl[:, inds] if include_potential_contribution else 0
    if not magnitude:
        expl = input_data.cpu().detach().numpy()[:, None] * expl
    return expl, preds, p
