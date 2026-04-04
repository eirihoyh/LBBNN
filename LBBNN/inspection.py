from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor


def nr_hidden_layers(net: Any) -> int:
    """Return the number of hidden layers in the network.

    Args:
        net: Network object with a `linears` attribute.

    Returns:
        Number of hidden layers.
    """
    return len(net.linears) - 1


def weight_matrices(net: Any) -> list[Tensor]:
    """Return detached copies of the network weight means.

    Args:
        net: Network object with linear layers containing `weight_mu`.

    Returns:
        List of weight mean tensors.
    """
    return [layer.weight_mu.detach().clone() for layer in net.linears]


def weight_matrices_numpy(net: Any, flow: bool = False) -> list[np.ndarray]:
    """Return weight means as NumPy arrays.

    Args:
        net: Network object with linear layers containing `weight_mu`.
        flow: Whether to multiply each weight matrix by the corresponding
            `z` matrix when available.

    Returns:
        List of weight matrices as NumPy arrays.
    """
    weights = [tensor.detach().cpu().numpy() for tensor in weight_matrices(net)]

    if flow:
        z_values = z_matrices_numpy(net)
        for idx in range(len(z_values)):
            weights[idx] *= z_values[idx]

    return weights


def z_matrices(net: Any) -> list[Tensor]:
    """Return detached copies of available latent inclusion matrices.

    Args:
        net: Network object with linear layers that may contain `q0_mean`.

    Returns:
        List of `q0_mean` tensors for layers where it exists.
    """
    values: list[Tensor] = []

    for layer in net.linears:
        if hasattr(layer, "q0_mean"):
            values.append(layer.q0_mean.detach().clone())

    return values


def z_matrices_numpy(net: Any) -> list[np.ndarray]:
    """Return latent inclusion matrices as NumPy arrays.

    Args:
        net: Network object with linear layers that may contain `q0_mean`.

    Returns:
        List of `q0_mean` matrices as NumPy arrays.
    """
    return [tensor.detach().cpu().numpy() for tensor in z_matrices(net)]


def weight_matrices_std(net: Any) -> list[Tensor]:
    """Return detached copies of the weight standard deviations.

    Args:
        net: Network object with linear layers containing `weight_rho`.

    Returns:
        List of weight standard deviation tensors.
    """
    return [
        torch.log1p(torch.exp(layer.weight_rho)).detach().cpu().clone()
        for layer in net.linears
    ]


def weight_matrices_std_numpy(net: Any) -> list[np.ndarray]:
    """Return weight standard deviations as NumPy arrays.

    Args:
        net: Network object with linear layers containing `weight_rho`.

    Returns:
        List of weight standard deviation matrices as NumPy arrays.
    """
    return [tensor.detach().cpu().numpy() for tensor in weight_matrices_std(net)]


def get_alphas(net: Any) -> list[Tensor]:
    """Return detached copies of the inclusion probabilities.

    Args:
        net: Network object with linear layers containing `lambdal`.

    Returns:
        List of alpha tensors.
    """
    return [torch.sigmoid(layer.lambdal).detach().cpu().clone() for layer in net.linears]


def get_alphas_numpy(net: Any) -> list[np.ndarray]:
    """Return inclusion probabilities as NumPy arrays.

    Args:
        net: Network object with linear layers containing `lambdal`.

    Returns:
        List of alpha matrices as NumPy arrays.
    """
    return [alpha.detach().cpu().numpy() for alpha in get_alphas(net)]


def clean_alpha(
    net: Any,
    threshold: float,
    alpha_list: list[Tensor] | None = None,
) -> list[Tensor]:
    """Threshold and prune alpha matrices to retain valid active paths.

    Args:
        net: Network object used when `alpha_list` is not provided.
        threshold: Threshold used to binarize alpha values.
        alpha_list: Optional list of alpha tensors to clean.

    Returns:
        List of cleaned binary alpha tensors.
    """
    if alpha_list is None:
        alpha_list = get_alphas(net)

    dim = alpha_list[0].shape[0]
    clean_dict = {
        idx: (alpha > threshold).float()
        for idx, alpha in enumerate(alpha_list)
    }

    for idx in np.arange(1, len(alpha_list))[::-1]:
        active_next = torch.sum(clean_dict[idx][:, :dim], dim=0) > 0
        clean_dict[idx - 1] = (
            clean_dict[idx - 1].T * active_next
        ).T.float()

    for idx in np.arange(1, len(alpha_list)):
        active_prev = (torch.sum(clean_dict[idx - 1].T, dim=0) > 0).float()
        clean_dict[idx] = torch.cat(
            (
                clean_dict[idx][:, :dim] * active_prev,
                clean_dict[idx][:, dim:],
            ),
            dim=1,
        )

    return list(clean_dict.values())


def clean_alpha_class(
    net: Any,
    threshold: float,
    class_in_focus: int = 0,
    alpha_list: list[Tensor] | None = None,
) -> list[Tensor]:
    """Clean alpha matrices while keeping paths for one output class only.

    Args:
        net: Network object used when `alpha_list` is not provided.
        threshold: Threshold used to binarize alpha values.
        class_in_focus: Output class index to retain.
        alpha_list: Optional list of alpha tensors to clean.

    Returns:
        List of cleaned binary alpha tensors for the selected class.
    """
    if alpha_list is None:
        alpha_list = get_alphas(net)

    alpha_list = [alpha.clone() for alpha in alpha_list]
    num_classes = alpha_list[-1].shape[0]

    remove_mask = torch.ones(num_classes, dtype=torch.bool)
    remove_mask[class_in_focus] = False
    alpha_list[-1][remove_mask, :] = 0

    return clean_alpha(net, threshold, alpha_list=alpha_list)


def get_active_weights(clean_alpha_list: list[Tensor]) -> list[Tensor]:
    """Return indices of active weights in each cleaned alpha matrix.

    Args:
        clean_alpha_list: List of cleaned binary alpha tensors.

    Returns:
        List of index tensors from `nonzero()`.
    """
    return [alpha.nonzero() for alpha in clean_alpha_list]


def network_density_reduction(
    clean_alpha_list: list[Tensor],
) -> tuple[float, float, int]:
    """Compute the fraction and number of active weights.

    Args:
        clean_alpha_list: List of cleaned binary alpha tensors.

    Returns:
        Tuple containing density, used weights, and total weights.
    """
    used_weights = 0.0
    total_weights = 0

    for alpha in clean_alpha_list:
        used_weights += float(alpha.sum().item())
        total_weights += int(alpha.numel())

    return used_weights / total_weights, used_weights, total_weights


def create_layer_name_list(
    n_layers: int | None = None,
    net: Any | None = None,
) -> list[str]:
    """Create readable names for the input, hidden, and output layers.

    Args:
        n_layers: Total number of layers including input and output.
        net: Optional network object used to infer the number of layers.

    Returns:
        List of layer names.
    """
    if net is not None:
        n_layers = nr_hidden_layers(net) + 2

    if n_layers is None:
        raise ValueError("Provide either n_layers or net.")

    layers = ["I"]
    for layer_idx in range(n_layers - 2):
        layers.append(f"H{layer_idx + 1}")
    layers.append("Output")

    return layers


def input_inclusion_prob(
    net: Any,
    a: list[np.ndarray] | None = None,
) -> dict[str, float]:
    """Compute input inclusion probabilities from each layer.

    Args:
        net: Network object used when `a` is not provided.
        a: Optional list of alpha matrices as NumPy arrays.

    Returns:
        Dictionary mapping descriptive names to inclusion probabilities.
    """
    if a is None:
        a = get_alphas_numpy(net)

    length = len(a)
    p = a[0].shape[1]
    prob_paths: dict[str, float] = {}
    layer_names = create_layer_name_list(n_layers=length + 1)

    for name in layer_names[:-1]:
        for i in range(p):
            prob_paths[f"Prob I{i} from {name}"] = 0.0

    limits = np.arange(1, length, 1)[::-1]

    if len(limits) == 0:
        name = layer_names[0]
        for xi in range(p):
            prob_paths[f"Prob I{xi} from {name}"] = float(a[0][0][xi])
    else:
        for i, name in enumerate(layer_names[:-1]):
            probs = a[i][:, -p:].T
            count = 0

            while i < len(limits) and count < limits[i]:
                count += 1
                probs = probs @ a[i + count][:, :-p].T

            for xi in range(p):
                prob_paths[f"Prob I{xi} from {name}"] = float(probs[xi][0])

    return prob_paths


def expected_number_of_weights(net: Any) -> float:
    """Return the expected number of active weights in the network.

    Args:
        net: Network object with alpha values available.

    Returns:
        Expected number of active weights.
    """
    return float(sum(np.sum(alpha) for alpha in get_alphas_numpy(net)))


def include_input_from_layer(clean_alpha_list: list[Tensor]) -> list[np.ndarray]:
    """Check whether each input is included from each layer.

    Args:
        clean_alpha_list: List of cleaned binary alpha tensors.

    Returns:
        List of boolean NumPy arrays indicating input inclusion per layer.
    """
    p = clean_alpha_list[0].shape[1]
    return [
        np.sum(alpha[:, -p:].detach().cpu().numpy(), axis=0) > 0
        for alpha in clean_alpha_list
    ]


def average_path_length(
    clean_alpha_list: list[Tensor],
) -> tuple[float, np.ndarray]:
    """Compute the average path length over active input connections.

    Args:
        clean_alpha_list: List of cleaned binary alpha tensors.

    Returns:
        Tuple containing the average path length and all collected path lengths.
    """
    length_list = len(clean_alpha_list)
    p = clean_alpha_list[0].shape[1]
    sum_dists = np.array([])

    for i in range(length_list):
        for xi in range(p):
            path_length = (
                clean_alpha_list[i][:, -(xi + 1)].detach().cpu().numpy()
                * (length_list - i)
            )
            path_length = path_length[path_length != 0]

            if path_length.size:
                sum_dists = np.concatenate((sum_dists, path_length))

    return float(np.mean(sum_dists)) if sum_dists.size else 0.0, sum_dists


def prob_width(net: Any, p: int) -> dict[int, float]:
    """Compute capped marginal inclusion probabilities for each input.

    Args:
        net: Network object used to compute input inclusion probabilities.
        p: Number of input features.

    Returns:
        Dictionary mapping input index to probability width.
    """
    probs = input_inclusion_prob(net)
    values = list(probs.values())

    return {i: float(min(np.sum(values[i::p]), 1.0)) for i in range(p)}


def get_weight_and_bias_std(
    net: Any,
    alphas_numpy: list[np.ndarray],
    threshold: float = 0.5,
) -> list[np.ndarray]:
    """Return standard deviations masked by thresholded inclusion probabilities.

    Args:
        net: Network object with weight standard deviations.
        alphas_numpy: List of alpha matrices as NumPy arrays.
        threshold: Threshold used to mask inactive weights.

    Returns:
        List of masked standard deviation matrices.
    """
    std_weight = weight_matrices_std_numpy(net)

    for i in range(len(std_weight)):
        std_weight[i] *= (alphas_numpy[i] > threshold).astype(float)

    return std_weight


def get_weight_and_bias(
    net: Any,
    alphas_numpy: list[np.ndarray],
    median: bool = True,
    sample: bool = False,
    threshold: float = 0.5,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Return weights with optional sampling and inclusion masking.

    Args:
        net: Network object with weight means and standard deviations.
        alphas_numpy: List of alpha matrices as NumPy arrays.
        median: Whether to use deterministic thresholding of alphas.
        sample: Whether to sample Gaussian noise around weight means.
        threshold: Threshold used when `median` is True.

    Returns:
        Tuple containing the processed weight matrices and alpha masks.
    """
    weights = weight_matrices_numpy(net)
    std_weight = weight_matrices_std_numpy(net)

    if sample:
        for i in range(len(weights)):
            weights[i] = weights[i] + np.random.normal(0, std_weight[i])

    if median:
        for i in range(len(weights)):
            weights[i] *= (alphas_numpy[i] > threshold).astype(float)
    else:
        for i in range(len(weights)):
            include = np.random.binomial(1, alphas_numpy[i]).astype(float)
            weights[i] *= include
            alphas_numpy[i] = include.copy()

    return weights, alphas_numpy