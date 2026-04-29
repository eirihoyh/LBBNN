"""Tests for ``LBBNN.flow`` (BayesianLinearFlow and BayesianNetworkFlow).

The PropagateFlow / IAF / RNVP transform tests live in ``test_flow_transforms.py``.
"""
import torch

from LBBNN import BayesianLinearFlow, BayesianNetworkFlow


# ====================================================================
# Layer-level: BayesianLinearFlow
# ====================================================================

def test_flow_layer_forward_shape_and_kl_div():
    torch.manual_seed(0)
    layer = BayesianLinearFlow(in_features=5, out_features=3, num_transforms=2)
    out = layer(torch.randn(8, 5))
    assert out.shape == (8, 3)
    kl = layer.kl_div()
    assert kl.ndim == 0
    assert torch.isfinite(kl)


def test_flow_layer_kl_reuses_cached_z():
    """After forward, kl_div must reuse the cached zk rather than resampling."""
    torch.manual_seed(0)
    layer = BayesianLinearFlow(in_features=5, out_features=3, num_transforms=2)
    _ = layer(torch.randn(8, 5))
    cached_zk = layer._cached_zk.clone()
    _ = layer.kl_div()
    assert torch.equal(layer._cached_zk, cached_zk)


# ====================================================================
# Network-level forward shapes
# ====================================================================

def test_flow_binary_forward_shape_and_range():
    model = BayesianNetworkFlow(
        dim=6, p=5, hidden_layers=2, num_transforms=2,
        classification=True, n_classes=1,
    )
    y = model(torch.randn(8, 5), ensemble=False)
    assert y.shape == (8, 1)
    assert torch.all(y >= 0.0) and torch.all(y <= 1.0)


def test_flow_regression_forward_shape():
    model = BayesianNetworkFlow(
        dim=6, p=5, hidden_layers=2, num_transforms=2,
        classification=False, n_classes=1,
    )
    y = model(torch.randn(4, 5), ensemble=False)
    assert y.shape == (4, 1)


def test_flow_multiclass_forward_shape():
    model = BayesianNetworkFlow(
        dim=6, p=5, hidden_layers=2, num_transforms=2,
        classification=True, n_classes=3,
    )
    y = model(torch.randn(7, 5), ensemble=False)
    assert y.shape == (7, 3)


# ====================================================================
# input_skip=False architecture
# ====================================================================

def test_flow_input_skip_false_hidden_layer_input_dim():
    model = BayesianNetworkFlow(
        dim=4, p=5, hidden_layers=3, num_transforms=2,
        classification=True, n_classes=1, input_skip=False,
    )
    assert model.linears[0].weight_mu.shape[1] == 5
    for layer in model.linears[1:]:
        assert layer.weight_mu.shape[1] == 4


def test_flow_input_skip_false_forward_and_kl_finite():
    torch.manual_seed(0)
    model = BayesianNetworkFlow(
        dim=4, p=5, hidden_layers=2, num_transforms=2,
        classification=True, n_classes=1, input_skip=False,
    )
    y = model(torch.randn(8, 5), ensemble=True)
    assert y.shape == (8, 1)
    assert torch.isfinite(model.kl())


# ====================================================================
# custom_loss
# ====================================================================

def test_flow_custom_loss_is_used():
    def my_loss(out: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (out - target).abs().sum()

    model = BayesianNetworkFlow(
        dim=4, p=5, hidden_layers=2, num_transforms=2,
        classification=True, n_classes=1, custom_loss=my_loss,
    )
    assert model.loss is my_loss


# ====================================================================
# predict / predict_proba (inherited from BayesianNetworkBase)
# ====================================================================

def test_flow_predict_binary_returns_zero_one():
    torch.manual_seed(0)
    model = BayesianNetworkFlow(
        dim=4, p=5, hidden_layers=2, num_transforms=2,
        classification=True, n_classes=1,
    )
    pred = model.predict(torch.randn(8, 5))
    assert pred.shape == (8,)
    assert torch.all((pred == 0) | (pred == 1))


def test_flow_predict_multiclass_returns_class_indices():
    torch.manual_seed(0)
    model = BayesianNetworkFlow(
        dim=4, p=5, hidden_layers=2, num_transforms=2,
        classification=True, n_classes=3,
    )
    pred = model.predict(torch.randn(8, 5))
    assert pred.shape == (8,)
    assert torch.all((pred >= 0) & (pred < 3))


def test_flow_predict_proba_multiclass_sums_to_one():
    torch.manual_seed(0)
    model = BayesianNetworkFlow(
        dim=4, p=5, hidden_layers=2, num_transforms=2,
        classification=True, n_classes=4,
    )
    proba = model.predict_proba(torch.randn(8, 5))
    assert proba.shape == (8, 4)
    sums = proba.sum(dim=1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
