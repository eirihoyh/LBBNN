"""Shared helpers for LBBNN experiment scripts."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_json_serializable(obj):
    if isinstance(obj, dict):
        return {str(k): _to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, torch.Tensor):
        return obj.item() if obj.ndim == 0 else obj.detach().cpu().tolist()
    return obj


def save_json(obj, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_to_json_serializable(obj), f, indent=2)


def save_history_csv(history: list[dict], path: Path) -> None:
    if not history:
        return
    fieldnames = list(history[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def save_predictions_csv(y_true, y_prob, y_pred, path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["y_true", "y_prob", "y_pred"])
        for yt, yp, yh in zip(y_true, y_prob, y_pred):
            writer.writerow([float(yt), float(yp), float(yh)])


def binary_accuracy(
    y_prob: Tensor, y_true: Tensor, threshold: float = 0.5
) -> float:
    y_hat = (y_prob >= threshold).float()
    return float((y_hat.view(-1) == y_true.view(-1)).float().mean().cpu())

def regression_metrics(
    y_pred: Tensor, y_true: Tensor
) -> tuple[float, float, float]:
    """Compute R², Pearson correlation, and MSE for regression predictions.

    Args:
        y_pred: Predicted values tensor.
        y_true: Ground truth values tensor.

    Returns:
        A tuple containing:
            - R² (coefficient of determination).
            - Pearson correlation coefficient.
            - Mean squared error.
    """
    y_pred = y_pred.view(-1).float().cpu()
    y_true = y_true.view(-1).float().cpu()

    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    r2 = float(1 - ss_res / ss_tot)

    correlation = float(torch.corrcoef(torch.stack([y_pred, y_true]))[0, 1])

    mse = float(((y_pred - y_true) ** 2).mean())

    return r2, correlation, mse

def split_dataset(
    X: Tensor,
    y: Tensor,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
):
    """Randomly split ``(X, y)`` into train / val / test tensors.

    Returns:
        Three ``(X_split, y_split, data_split)`` tuples where ``data_split``
        is ``X`` and ``y`` concatenated column-wise (ready for the training
        utilities).
    """
    n = X.shape[0]
    idx = torch.randperm(n)

    n_train = int(train_frac * n)
    n_val = int(val_frac * n)

    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]

    def _pack(i):
        Xi, yi = X[i], y[i]
        return Xi, yi, torch.cat([Xi, yi.unsqueeze(1)], dim=1)

    return _pack(train_idx), _pack(val_idx), _pack(test_idx)
