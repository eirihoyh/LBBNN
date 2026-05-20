from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray
from typing import Literal
from torch import Tensor

from ._types import BayesianNet


# NOTE: The empirical-explain helpers (`relu_activation`, `get_active_nodes`,
# `find_active_weights`, `local_explain_relu`, `local_explain_relu_magnitude`)
# were removed because they hardcode the input-skip concatenation and so do not
# generalise to networks built with `input_skip=False`. The gradient-based
# `local_explain_piecewise_linear_act` below works for either architecture and
# is the recommended entry point. The earlier code is preserved in git history
# (commit before the input_skip refactor) if it ever needs to be revived.


def local_explain_piecewise_linear_act(
    net: BayesianNet,
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


def what_if_explanations(
    net: BayesianNet,
    data: Tensor,
    feature_index: int,
    minimum: float,
    maximum: float,
    n_samples: int = 1000,
    n_expl_per_sample: int = 10,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Compute local explanations over a range of values for one feature.

    Args:
        net: Trained network object.
        data: One-dimensional input tensor to explain.
        feature_index: Index of the feature to vary.
        minimum: Lower bound for the feature value.
        maximum: Upper bound for the feature value.
        n_samples: Number of feature values to evaluate.
        n_expl_per_sample: Number of explanation samples per feature value.

    Returns:
        A tuple containing:
            - The evaluated feature values.
            - The local contributions for each feature value.
            - The predicted class indicator for each feature value.
    """
    observed_space = np.linspace(minimum, maximum, num=n_samples)
    contributions = np.zeros(
        (n_samples, data.shape[0], n_expl_per_sample),
        dtype=float,
    )
    predictions = np.zeros((n_samples, n_expl_per_sample), dtype=float)

    for i, adjusted_value in enumerate(observed_space):
        data_adjusted = data.clone()
        data_adjusted[feature_index] = torch.tensor(adjusted_value, device=data.device)

        explanation, preds, _ = local_explain_piecewise_linear_act(
            net,
            data_adjusted,
            n_samples=n_expl_per_sample,
        )

        contributions[i] = explanation[:, :, 0].T
        prediction_value = float(
            preds[:, 0].detach().cpu().numpy().mean() > 0.5
        )
        predictions[i, :] = prediction_value

    return observed_space, contributions, predictions


def compute_global_explain_piecewise_linear_act(
    net: BayesianNet,
    X: np.ndarray,
    n_expl_per_sample: int = 10,
    n_classes: int = 1,
    task: Literal["binary", "multiclass", "regression"] = "binary",
    pred_threshold: float = 0.5,
    include_potential_contribution: bool = False,
) -> tuple[NDArray[np.float64], NDArray]:
    """Compute gradient-based local explanations across an entire dataset.

    For each sample in ``X``, draws ``n_expl_per_sample`` gradient-based
    local explanations and aggregates them, giving a global view of
    covariate contributions across the dataset.

    Args:
        net: Trained Bayesian network object.
        X: Input data array of shape ``(n_samples, n_features)``.
        n_expl_per_sample: Number of explanation samples drawn per
            data point.
        n_classes: Number of output classes. Ignored when
            ``task="regression"``.
        task: Type of prediction task. One of ``"binary"``,
            ``"multiclass"``, or ``"regression"``. Determines how
            predictions are summarised:
            - ``"binary"``: applies ``pred_threshold`` to the mean
              output to produce a 0/1 label.
            - ``"multiclass"``: takes the argmax over mean class
              probabilities.
            - ``"regression"``: returns the raw mean predicted value.
        pred_threshold: Decision threshold for binary classification.
            Ignored when ``task`` is not ``"binary"``.
        include_potential_contribution: Whether to include potential
            contributions for zero-valued inputs.

    Returns:
        A tuple containing:
            - Contributions array of shape
              ``(n_samples * n_expl_per_sample, n_features, n_classes)``.
            - For ``"binary"`` and ``"multiclass"``: predicted class
              indices of shape ``(n_samples * n_expl_per_sample,)``
              as ``np.int64``.
            - For ``"regression"``: mean predicted values of shape
              ``(n_samples * n_expl_per_sample,)`` as ``np.float64``.
    """
    n_samples = len(X)
    n_features = X.shape[1]
    if isinstance(X, torch.Tensor):
        data = X.clone().detach().to(dtype=torch.float32)
    else:
        data = torch.tensor(X, dtype=torch.float32)

    contributions = np.zeros((n_samples, n_expl_per_sample, n_features, n_classes))
    predictions = np.zeros((n_samples, n_classes))

    for ind, d in enumerate(data):
        expl_values, preds, _ = local_explain_piecewise_linear_act(
            net,
            d,
            n_samples=n_expl_per_sample,
            n_classes=n_classes,
            include_potential_contribution=include_potential_contribution,
        )
        for c in range(n_classes):
            contributions[ind, :, :, c] = expl_values[:, :, c]

        predictions[ind] = preds.detach().cpu().numpy().mean(axis=0)

    if task == "binary":
        summarised_predictions = np.repeat(
            (predictions[:, 0] > pred_threshold).astype(int), n_expl_per_sample
        )
    elif task == "multiclass":
        summarised_predictions = np.repeat(
            predictions.argmax(axis=1), n_expl_per_sample
        )
    else:  # regression
        summarised_predictions = np.repeat(predictions[:, 0], n_expl_per_sample)

    return (
        contributions.reshape(n_samples * n_expl_per_sample, n_features, n_classes),
        summarised_predictions,
    )