from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from .inspection import clean_alpha_class, get_alphas_numpy, get_weight_and_bias


def relu_activation(
    input_data: Tensor,
    weights: list[NDArray[np.float64] | NDArray[np.float32]],
) -> tuple[NDArray[np.floating], list[NDArray[np.floating]]]:
    """Run a forward pass with ReLU activations and input skip connections.

    Args:
        input_data: Input tensor of shape ``(n_samples, n_features)``.
        weights: List of weight matrices ordered by layer.

    Returns:
        A tuple containing:
            - The final network output.
            - The layer-wise outputs after each linear/ReLU step.
    """
    x_in = input_data.detach().cpu().numpy()
    out = np.empty((input_data.shape[0], 0))
    output_list: list[NDArray[np.floating]] = []

    for weight in weights[:-1]:
        out = np.concatenate((out, x_in), axis=1)
        out = out @ weight.T
        out = out * (out > 0)
        output_list.append(out)

    out = np.concatenate((out, x_in), axis=1)
    out = out @ weights[-1].T
    output_list.append(out)

    return out, output_list


def get_active_nodes(
    clean_alpha_list: list[Tensor],
    output_list_c: list[NDArray[np.floating]],
) -> NDArray[np.int_]:
    """Identify active hidden nodes for the current input.

    Args:
        clean_alpha_list: Cleaned binary alpha masks per layer.
        output_list_c: Layer-wise outputs from a forward pass.

    Returns:
        Array indicating which nodes are both structurally active and
        activated by the current input.
    """
    active_nodes_alpha_list = [
        (np.sum(alpha.detach().cpu().numpy(), axis=1) > 0).astype(int)
        for alpha in clean_alpha_list
    ]

    return np.array(
        [
            ((active_nodes_alpha_list[i] * output_list_c[i][0]) > 0).astype(int)
            for i in range(len(clean_alpha_list) - 1)
        ]
    )


def find_active_weights(
    weights: list[NDArray[np.floating]],
    active_nodes_list: NDArray[np.int_],
    clean_alpha_list: list[Tensor],
    dim: int,
) -> list[NDArray[np.floating]]:
    """Mask weights using active nodes and cleaned alpha values.

    Args:
        weights: List of weight matrices.
        active_nodes_list: Binary activity indicators for hidden nodes.
        clean_alpha_list: Cleaned binary alpha masks per layer.
        dim: Number of non-skip inputs passed between layers.

    Returns:
        List of masked weight matrices.
    """
    active_weights = copy.deepcopy(weights)

    for i in range(len(active_weights) - 1):
        alpha_mask = clean_alpha_list[i].detach().cpu().numpy()
        active_weights[i] = active_weights[i] * alpha_mask
        active_weights[i] = np.array(
            [
                active_weights[i][j, :] * active_nodes_list[i, j]
                for j in range(len(active_nodes_list[i]))
            ]
        )

        active_weights[i + 1][:, :dim] = np.array(
            [
                active_weights[i + 1][:, j] * active_nodes_list[i, j]
                for j in range(len(active_nodes_list[i]))
            ]
        ).T

    active_weights[-1] = (
        active_weights[-1] * clean_alpha_list[-1].detach().cpu().numpy()
    )

    return active_weights


def local_explain_relu(
    net: Any,
    input_data: Tensor,
    threshold: float = 0.5,
    median: bool = True,
    sample: bool = False,
    n_samples: int = 1,
    verbose: bool = False,
    quantiles: tuple[float, float] = (0.025, 0.975),
) -> tuple[
    dict[int, dict[int, float]],
    dict[int, dict[int, NDArray[np.floating]]],
    NDArray[np.floating],
]:
    """Compute local input contributions for a ReLU network.

    Args:
        net: Network object to explain.
        input_data: Input tensor of shape ``(1, n_features)``.
        threshold: Threshold used for alpha cleaning.
        median: Whether to use deterministic median inclusion.
        sample: Whether to sample weights.
        n_samples: Number of explanation samples.
        verbose: Unused flag kept for API compatibility.
        quantiles: Lower and upper quantiles for credible intervals.

    Returns:
        A tuple containing:
            - Mean contribution per class and input.
            - Quantile interval per class and input.
            - Array of sampled predictions.
    """
    del verbose

    contributions: dict[int, dict[int, dict[int, float]]] = {}
    preds: list[NDArray[np.floating]] = []

    for n in range(n_samples):
        alphas_numpy = get_alphas_numpy(net)
        nr_classes = alphas_numpy[-1].shape[0]

        weights, alphas_numpy = get_weight_and_bias(
            net=net,
            alphas_numpy=alphas_numpy,
            median=median,
            sample=sample,
            threshold=threshold,
        )
        alphas = [torch.tensor(alpha) for alpha in copy.deepcopy(alphas_numpy)]

        out, output_list = relu_activation(input_data, weights)
        preds.append(out)

        contribution_classes: dict[int, dict[int, float]] = {}

        for c in range(nr_classes):
            weights_class = copy.deepcopy(weights)
            weights_class[-1] = weights_class[-1][c : c + 1, :]

            clean_alpha_list = clean_alpha_class(
                net=net,
                threshold=threshold,
                class_in_focus=c,
                alpha_list=copy.deepcopy(alphas),
            )
            clean_alpha_list[-1] = clean_alpha_list[-1][c : c + 1, :]

            dim, p = clean_alpha_list[0].shape
            active_nodes_list = get_active_nodes(
                clean_alpha_list,
                copy.deepcopy(output_list),
            )
            active_weights = find_active_weights(
                weights_class,
                active_nodes_list,
                clean_alpha_list,
                dim,
            )

            pred_impact: dict[int, float] = {}
            for pi in range(p):
                explain_this_numpy = input_data.detach().cpu().numpy().copy()
                remove_list = [True] * p
                remove_list[pi] = False
                explain_this_numpy[0, remove_list] = 0

                x = np.array([[]])
                for active_weight in active_weights:
                    x = np.concatenate((x, explain_this_numpy), axis=1)
                    x = x @ active_weight.T

                pred_impact[pi] = float(x[0, 0])

            contribution_classes[c] = pred_impact

        contributions[n] = contribution_classes

    mean_contribution: dict[int, dict[int, float]] = {}
    cred_contribution: dict[int, dict[int, NDArray[np.floating]]] = {}

    for c in range(nr_classes):
        mean_contribution[c] = {}
        cred_contribution[c] = {}

        for pi in range(p):
            values = np.array([contributions[s][c][pi] for s in range(n_samples)])
            mean_contribution[c][pi] = float(np.mean(values))
            cred_contribution[c][pi] = np.quantile(values, quantiles)

    return mean_contribution, cred_contribution, np.array(preds)


def local_explain_relu_magnitude(
    net: Any,
    input_data: Tensor,
    threshold: float = 0.5,
    median: bool = True,
    sample: bool = False,
    n_samples: int = 1,
    verbose: bool = False,
    quantiles: tuple[float, float] = (0.025, 0.975),
    include_potential_contribution: bool = True,
) -> tuple[
    dict[int, dict[int, float]],
    dict[int, dict[int, NDArray[np.floating]]],
    NDArray[np.floating],
]:
    """Compute local magnitude-based input contributions for a ReLU network.

    Args:
        net: Network object to explain.
        input_data: Input tensor of shape ``(1, n_features)``.
        threshold: Threshold used for alpha cleaning.
        median: Whether to use deterministic median inclusion.
        sample: Whether to sample weights.
        n_samples: Number of explanation samples.
        verbose: Unused flag kept for API compatibility.
        quantiles: Lower and upper quantiles for credible intervals.
        include_potential_contribution: Whether to include contributions from
            inactive inputs as potential effects.

    Returns:
        A tuple containing:
            - Mean contribution per class and input.
            - Quantile interval per class and input.
            - Array of sampled predictions.
    """
    del verbose

    contributions: dict[int, dict[int, dict[int, float]]] = {}
    preds: list[NDArray[np.floating]] = []

    for n in range(n_samples):
        alphas_numpy = get_alphas_numpy(net)
        nr_classes = alphas_numpy[-1].shape[0]

        weights, alphas_numpy = get_weight_and_bias(
            net=net,
            alphas_numpy=alphas_numpy,
            median=median,
            sample=sample,
            threshold=threshold,
        )
        alphas = [torch.tensor(alpha) for alpha in copy.deepcopy(alphas_numpy)]

        out, output_list = relu_activation(input_data, weights)
        preds.append(out)

        contribution_classes: dict[int, dict[int, float]] = {}

        for c in range(nr_classes):
            weights_class = copy.deepcopy(weights)
            weights_class[-1] = weights_class[-1][c : c + 1, :]

            clean_alpha_list = clean_alpha_class(
                net=net,
                threshold=threshold,
                class_in_focus=c,
                alpha_list=copy.deepcopy(alphas),
            )
            clean_alpha_list[-1] = clean_alpha_list[-1][c : c + 1, :]

            dim, p = clean_alpha_list[0].shape
            active_nodes_list = get_active_nodes(
                clean_alpha_list,
                copy.deepcopy(output_list),
            )
            active_weights = find_active_weights(
                weights_class,
                active_nodes_list,
                clean_alpha_list,
                dim,
            )

            pred_impact: dict[int, float] = {}
            for pi in range(p):
                explain_this_numpy = np.ones((1, p))
                remove_list = [True] * p
                remove_list[pi] = False
                explain_this_numpy[0, remove_list] = 0

                x = np.array([[]])
                for active_weight in active_weights:
                    x = np.concatenate((x, explain_this_numpy), axis=1)
                    x = x @ active_weight.T

                input_value = float(input_data.detach().cpu().numpy()[0, pi])

                if include_potential_contribution:
                    pred_impact[pi] = float(-x[0, 0] if input_value == 0 else x[0, 0])
                else:
                    pred_impact[pi] = float(0.0 if input_value == 0 else x[0, 0])

            contribution_classes[c] = pred_impact

        contributions[n] = contribution_classes

    mean_contribution: dict[int, dict[int, float]] = {}
    cred_contribution: dict[int, dict[int, NDArray[np.floating]]] = {}

    for c in range(nr_classes):
        mean_contribution[c] = {}
        cred_contribution[c] = {}

        for pi in range(p):
            values = np.array([contributions[s][c][pi] for s in range(n_samples)])
            mean_contribution[c][pi] = float(np.mean(values))
            cred_contribution[c][pi] = np.quantile(values, quantiles)

    return mean_contribution, cred_contribution, np.array(preds)


def local_explain_piecewise_linear_act(
    net: Any,
    input_data: Tensor,
    median: bool = True,
    sample: bool = True,
    n_samples: int = 1,
    magnitude: bool = True,
    include_potential_contribution: bool = False,
    n_classes: int = 1,
) -> tuple[NDArray[np.floating], Tensor, int]:
    """Compute gradient-based local explanations for piecewise linear activations.

    Args:
        net: Network object with a `forward_preact` method.
        input_data: One-dimensional input tensor of shape ``(n_features,)``.
        median: Whether to use deterministic inference.
        sample: Whether to sample from the model.
        n_samples: Number of explanation samples.
        magnitude: Whether to return raw gradients instead of input-weighted
            contributions.
        include_potential_contribution: Whether to include potential
            contributions for zero-valued inputs.
        n_classes: Number of output classes.

    Returns:
        A tuple containing:
            - Explanation array of shape ``(n_samples, n_features, n_classes)``.
            - Prediction tensor of shape ``(n_samples, n_classes)``.
            - Number of input features.
    """
    p = input_data.shape[0]
    explanation = torch.zeros((n_samples, p, n_classes), device=input_data.device)
    preds = torch.zeros((n_samples, n_classes), device=input_data.device)

    for j in range(n_samples):
        explain_this = input_data.reshape(-1, p).clone().detach()
        explain_this.requires_grad_(True)

        net.zero_grad()
        output = net.forward_preact(
            explain_this,
            sample=sample,
            ensemble=not median,
        )

        for c in range(n_classes):
            output_value = output[0, c]
            gradients = torch.autograd.grad(
                output_value,
                explain_this,
                grad_outputs=torch.ones_like(output_value),
                retain_graph=True,
            )[0]

            explanation[j, :, c] = gradients[0]
            preds[j, c] = output[0, c]

    expl = explanation.detach().cpu().numpy()
    input_array = input_data.detach().cpu().numpy()
    inds = np.where(input_array == 0.0)[0]

    if include_potential_contribution:
        expl[:, inds] = -expl[:, inds]
    else:
        expl[:, inds] = 0

    if not magnitude:
        expl = input_array[None, :, None] * expl

    return expl, preds.detach().cpu(), p