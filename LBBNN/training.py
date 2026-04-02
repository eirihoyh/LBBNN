from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.optim import Optimizer

from .inspection import (
    clean_alpha,
    expected_number_of_weights,
    network_density_reduction,
)


def _r2_score(y_pred: Tensor, y_true: Tensor) -> float:
    """Compute the coefficient of determination (R²).

    Args:
        y_pred: Predicted values.
        y_true: Ground-truth values.

    Returns:
        The R² score as a float.
    """
    y_true = y_true.view(-1).float()
    y_pred = y_pred.view(-1).float()

    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)

    if float(ss_tot) == 0.0:
        return 0.0

    return float((1 - ss_res / ss_tot).detach().cpu())


def _rmse(y_pred: Tensor, y_true: Tensor) -> float:
    """Compute the root mean squared error.

    Args:
        y_pred: Predicted values.
        y_true: Ground-truth values.

    Returns:
        The RMSE as a float.
    """
    y_true = y_true.view(-1).float()
    y_pred = y_pred.view(-1).float()

    return float(torch.sqrt(torch.mean((y_true - y_pred) ** 2)).detach().cpu())


def train_epoch(
    net: Any,
    train_data: Tensor,
    optimizer: Optimizer,
    batch_size: int,
    num_batches: int,
    p: int,
    device: torch.device,
    nr_weights: int,
    multiclass: bool = False,
    verbose: bool = False,
    post_train: bool = False,
) -> tuple[float | None, float | None]:
    """Train the network for one epoch.

    Args:
        net: Neural network model with `loss` and `kl` methods.
        train_data: Training data tensor where the last column is the target.
        optimizer: Optimizer used for parameter updates.
        batch_size: Number of samples per batch.
        num_batches: Total number of batches used to scale the KL term.
        p: Number of input features.
        device: Torch device used for computation.
        nr_weights: Total number of weights in the network.
        multiclass: Whether the task is multiclass classification.
        verbose: Whether to print batch-level metrics.
        post_train: Whether to use post-training inference behavior.

    Returns:
        A tuple containing the last batch NLL and total loss.
    """
    net.train()

    indices = np.random.permutation(len(train_data))
    train_data = train_data[indices]

    last_nll: float | None = None
    last_loss: float | None = None

    for start in range(0, train_data.shape[0], batch_size):
        end = start + batch_size
        batch_data = train_data[start:end]

        x_batch = batch_data[:, :p].to(device)
        y_batch = batch_data[:, -1].to(device)

        if multiclass:
            target = y_batch.long()
        else:
            target = y_batch.float()

        optimizer.zero_grad()

        outputs = net(x_batch, sample=True, post_train=post_train)
        loss_target = target.view(-1) if multiclass else target.unsqueeze(1)
        nll = net.loss(outputs, loss_target.float() if not multiclass else loss_target)

        kl_part = net.kl() / max(num_batches, 1)
        loss = nll + kl_part

        loss.backward()
        optimizer.step()

        last_nll = nll.item()
        last_loss = loss.item()

        if verbose:
            print("loss", last_loss)
            print("nll", last_nll)
            print("density", expected_number_of_weights(net) / nr_weights)

    return last_nll, last_loss


def validate(
    net: Any,
    val_data: Tensor,
    device: torch.device,
    multiclass: bool = False,
    reg: bool = False,
    verbose: bool = False,
) -> tuple[float, float, float]:
    """Evaluate the network on validation data.

    Args:
        net: Neural network model with `loss` and `kl` methods.
        val_data: Validation data tensor where the last column is the target.
        device: Torch device used for computation.
        multiclass: Whether the task is multiclass classification.
        reg: Whether the task is regression.
        verbose: Whether to print validation metrics.

    Returns:
        A tuple containing NLL, total loss, and the validation metric.
    """
    net.eval()

    with torch.no_grad():
        x_val = val_data[:, :-1].to(device)
        y_val = val_data[:, -1].to(device)

        outputs = net(
            x_val,
            ensemble=False,
            calculate_log_probs=True,
        )

        if multiclass:
            target = y_val.long().view(-1)
            nll = net.loss(outputs, target)
            metric_value = float(
                (outputs.argmax(dim=1) == target).float().mean().cpu()
            )
        else:
            target = y_val.unsqueeze(1).float()
            nll = net.loss(outputs, target)

            if reg:
                metric_value = _r2_score(outputs[:, 0], target[:, 0])
            else:
                metric_value = float(
                    (outputs.round().squeeze().cpu() == target[:, 0].cpu())
                    .float()
                    .mean()
                )

        loss = nll + net.kl()

        alpha_clean = clean_alpha(net, threshold=0.5)
        _, used_weights_median, _ = network_density_reduction(alpha_clean)

        if verbose:
            print(
                f"val_loss: {loss.item():.4f}, "
                f"val_nll: {nll.item():.4f}, "
                f"val_metric: {metric_value:.4f}, "
                f"used_weights_median: {used_weights_median}"
            )

        return nll.item(), loss.item(), metric_value


def test_ensemble(
    net: Any,
    test_data: Tensor,
    device: torch.device,
    samples: int,
    classes: int = 1,
    reg: bool = True,
    verbose: bool = False,
    multiclass: bool = False,
) -> tuple[list[float], list[float]]:
    """Evaluate ensemble and median-style predictions on test data.

    Args:
        net: Neural network model used for inference.
        test_data: Test data tensor where the last column is the target.
        device: Torch device used for computation.
        samples: Number of stochastic forward passes.
        classes: Number of output classes or output dimension.
        reg: Whether the task is regression.
        multiclass: Whether the task is multiclass classification.

    Returns:
        A tuple of two metric lists:
            - Ensemble mean metrics.
            - Median-style metrics based on deterministic forward passes.
    """

    net.eval()

    density: list[float] = []
    used_weights: list[float] = []
    ensemble_metrics: list[float] = []
    ensemble_metrics_median: list[float] = []
    ensemble_r2: list[float] = []
    ensemble_r2_median: list[float] = []

    with torch.no_grad():
        x_test = test_data[:, :-1].to(device)
        y_test = test_data[:, -1].to(device)

        outputs = torch.zeros(samples, x_test.shape[0], classes, device=device)
        outputs_median = torch.zeros(samples, x_test.shape[0], classes, device=device)

        for i in range(samples):
            outputs[i] = net.forward(x_test, sample=True, ensemble=True)
            outputs_median[i] = net.forward(x_test, ensemble=False)

        outputs_mean = outputs.mean(dim=0)
        outputs_median_mean = outputs_median.mean(dim=0)

        alpha_clean = clean_alpha(net, threshold=0.5)
        density_median, used_weights_median, _ = network_density_reduction(
            alpha_clean
        )

        density.append(density_median)
        used_weights.append(used_weights_median)

        if reg:
            ensemble_metrics.append(_rmse(outputs_mean[:, 0], y_test))
            ensemble_metrics_median.append(_rmse(outputs_median_mean[:, 0], y_test))
            ensemble_r2.append(_r2_score(outputs_mean[:, 0], y_test))
            ensemble_r2_median.append(_r2_score(outputs_median_mean[:, 0], y_test))
        elif multiclass:
            target = y_test.long().view(-1)
            ensemble_metrics.append(
                float((outputs_mean.argmax(dim=1) == target).float().mean().cpu())
            )
            ensemble_metrics_median.append(
                float(
                    (outputs_median_mean.argmax(dim=1) == target)
                    .float()
                    .mean()
                    .cpu()
                )
            )
        else:
            ensemble_metrics.append(
                float((outputs_mean[:, 0].round().cpu() == y_test.cpu()).float().mean())
            )
            ensemble_metrics_median.append(
                float(
                    (outputs_median_mean[:, 0].round().cpu() == y_test.cpu())
                    .float()
                    .mean()
                )
            )

    metrics = [float(np.mean(ensemble_metrics)), float(np.mean(density))]
    metrics_median = [
        float(np.mean(ensemble_metrics_median)),
        float(np.mean(used_weights)),
    ]

    if reg:
        metrics.append(float(np.mean(ensemble_r2)))
        metrics_median.append(float(np.mean(ensemble_r2_median)))

    return metrics, metrics_median