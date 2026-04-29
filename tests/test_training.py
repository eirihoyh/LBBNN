"""Tests for ``LBBNN.training`` (train_epoch, validate, test_ensemble)."""
import math

import numpy as np
import torch

from LBBNN import (
    BayesianNetworkFlow,
    BayesianNetworkLRT,
    create_bsr_data,
    train_epoch,
    validate,
)
# Imported under an alias so pytest doesn't try to collect it as a test
# (the LBBNN function is named ``test_ensemble`` for historical API reasons).
from LBBNN import test_ensemble as run_test_ensemble


CPU = torch.device("cpu")


def _binary_data(n: int = 16, p: int = 5, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, p, generator=g)
    y = (torch.rand(n, generator=g) > 0.5).float()
    return torch.cat([x, y.unsqueeze(1)], dim=1)


def _multiclass_data(n: int = 16, p: int = 5, n_classes: int = 3, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, p, generator=g)
    y = torch.randint(0, n_classes, (n,), generator=g).float()
    return torch.cat([x, y.unsqueeze(1)], dim=1)


# ---------------- train_epoch / validate: basic binary ----------------

def test_train_epoch_and_validate_binary_run():
    torch.manual_seed(0)
    model = BayesianNetworkLRT(dim=4, p=5, hidden_layers=2, classification=True, n_classes=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_data = _binary_data()
    nr_weights = sum(p.numel() for p in model.parameters())

    nll, loss = train_epoch(
        net=model, train_data=train_data, optimizer=optimizer,
        batch_size=8, num_batches=2, p=5, device=CPU,
        nr_weights=nr_weights, task="binary",
    )
    assert isinstance(nll, float) and isinstance(loss, float)

    val_nll, val_loss, metric = validate(
        net=model, val_data=train_data, device=CPU, task="binary",
    )
    assert math.isfinite(val_nll) and math.isfinite(val_loss)
    assert 0.0 <= metric <= 1.0


# ---------------- multiclass branches ----------------

def test_train_epoch_multiclass_runs():
    torch.manual_seed(0)
    model = BayesianNetworkLRT(dim=4, p=5, hidden_layers=2, classification=True, n_classes=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_data = _multiclass_data(n_classes=3)
    nr_weights = sum(p.numel() for p in model.parameters())

    nll, loss = train_epoch(
        net=model, train_data=train_data, optimizer=optimizer,
        batch_size=8, num_batches=2, p=5, device=CPU,
        nr_weights=nr_weights, task="multiclass",
    )
    assert math.isfinite(nll) and math.isfinite(loss)


def test_validate_multiclass_returns_accuracy_in_unit_range():
    torch.manual_seed(0)
    model = BayesianNetworkLRT(dim=4, p=5, hidden_layers=2, classification=True, n_classes=3)
    nll, loss, acc = validate(
        net=model, val_data=_multiclass_data(n_classes=3),
        device=CPU, task="multiclass",
    )
    assert math.isfinite(nll) and math.isfinite(loss)
    assert 0.0 <= acc <= 1.0


# ---------------- flow KL stays finite during training ----------------

def test_flow_kl_is_scalar_and_finite_at_init():
    torch.manual_seed(0)
    model = BayesianNetworkFlow(
        dim=4, p=5, hidden_layers=2, num_transforms=2,
        classification=True, n_classes=1,
    )
    kl = model.kl()
    assert kl.ndim == 0
    assert torch.isfinite(kl)


def test_flow_kl_finite_after_training_steps():
    torch.manual_seed(0)
    model = BayesianNetworkFlow(
        dim=4, p=5, hidden_layers=2, num_transforms=2,
        classification=True, n_classes=1,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_data = _binary_data()
    nr_weights = sum(p.numel() for p in model.parameters())

    for _ in range(3):
        train_epoch(
            net=model, train_data=train_data, optimizer=optimizer,
            batch_size=8, num_batches=2, p=5, device=CPU,
            nr_weights=nr_weights, task="binary",
        )
        assert torch.isfinite(model.kl())


# ---------------- loss decreases on a real toy problem ----------------

def test_lrt_loss_decreases_on_bsr_regression():
    torch.manual_seed(0)
    np.random.seed(0)

    y_np, x_np = create_bsr_data(n=200, seed=0, func=1)
    x_t = torch.tensor(x_np, dtype=torch.float32)
    y_t = torch.tensor(y_np, dtype=torch.float32)
    train_data = torch.cat([x_t, y_t.unsqueeze(1)], dim=1)

    model = BayesianNetworkLRT(
        dim=8, p=2, hidden_layers=2, classification=False, n_classes=1,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    nr_weights = sum(p.numel() for p in model.parameters())
    num_batches = math.ceil(len(train_data) / 64)

    losses: list[float] = []
    for _ in range(100):
        _, loss = train_epoch(
            net=model, train_data=train_data, optimizer=optimizer,
            batch_size=64, num_batches=num_batches, p=2, device=CPU,
            nr_weights=nr_weights, task="regression",
        )
        losses.append(loss)

    early = sum(losses[:10]) / 10
    late = sum(losses[-10:]) / 10
    assert late < early, f"Expected loss to drop: early={early:.3f}, late={late:.3f}"


# ---------------- test_ensemble (binary, multiclass, regression) ----------------

def test_ensemble_binary_returns_metric_and_density():
    torch.manual_seed(0)
    model = BayesianNetworkLRT(dim=4, p=5, hidden_layers=2, classification=True, n_classes=1)
    metrics, metrics_median = run_test_ensemble(
        net=model, test_data=_binary_data(), device=CPU,
        samples=2, classes=1, task="binary",
    )
    assert len(metrics) == 2 and len(metrics_median) == 2
    assert 0.0 <= metrics[0] <= 1.0
    assert 0.0 <= metrics_median[0] <= 1.0


def test_ensemble_multiclass():
    torch.manual_seed(0)
    model = BayesianNetworkLRT(dim=4, p=5, hidden_layers=2, classification=True, n_classes=3)
    metrics, metrics_median = run_test_ensemble(
        net=model, test_data=_multiclass_data(n_classes=3), device=CPU,
        samples=2, classes=3, task="multiclass",
    )
    assert 0.0 <= metrics[0] <= 1.0
    assert 0.0 <= metrics_median[0] <= 1.0


def test_ensemble_regression_returns_rmse_and_r2():
    torch.manual_seed(0)
    model = BayesianNetworkLRT(dim=4, p=5, hidden_layers=2, classification=False, n_classes=1)
    g = torch.Generator().manual_seed(0)
    x = torch.randn(16, 5, generator=g)
    y = torch.randn(16, generator=g)
    test_data = torch.cat([x, y.unsqueeze(1)], dim=1)

    metrics, metrics_median = run_test_ensemble(
        net=model, test_data=test_data, device=CPU,
        samples=2, classes=1, task="regression",
    )
    assert len(metrics) == 3 and len(metrics_median) == 3
    assert all(np.isfinite(v) for v in metrics)


# ---------------- explicit CPU device pinning ----------------

def test_lrt_runs_on_cpu_device():
    model = BayesianNetworkLRT(
        dim=4, p=5, hidden_layers=2, classification=True, n_classes=1,
    ).to(CPU)
    x = torch.randn(4, 5, device=CPU)
    y = model(x, ensemble=False)
    assert y.device.type == "cpu"


def test_flow_runs_on_cpu_device():
    model = BayesianNetworkFlow(
        dim=4, p=5, hidden_layers=2, num_transforms=2,
        classification=True, n_classes=1,
    ).to(CPU)
    x = torch.randn(4, 5, device=CPU)
    y = model(x, ensemble=False)
    assert y.device.type == "cpu"
