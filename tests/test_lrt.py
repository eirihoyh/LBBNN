"""Tests for ``LBBNN.lrt`` (BayesianLinearLRT and BayesianNetworkLRT)."""
import torch
import torch.nn as nn

from LBBNN import BayesianLinearLRT, BayesianNetworkLRT


# ====================================================================
# Layer-level: BayesianLinearLRT
# ====================================================================

def test_lrt_layer_forward_shape_in_training_mode():
    torch.manual_seed(0)
    layer = BayesianLinearLRT(in_features=5, out_features=3)
    out = layer(torch.randn(8, 5))
    assert out.shape == (8, 3)


def test_lrt_layer_kl_set_after_forward_in_training_mode():
    torch.manual_seed(0)
    layer = BayesianLinearLRT(in_features=5, out_features=3)
    _ = layer(torch.randn(8, 5))
    assert torch.is_tensor(layer.kl)
    assert torch.isfinite(layer.kl)


def test_lrt_layer_post_train_uses_thresholded_alpha():
    torch.manual_seed(0)
    layer = BayesianLinearLRT(in_features=5, out_features=3)
    layer.eval()
    x = torch.randn(4, 5)
    with torch.no_grad():
        y1 = layer(x, ensemble=False, sample=False, post_train=True)
        y2 = layer(x, ensemble=False, sample=False, post_train=True)
    assert torch.equal(y1, y2)


# ====================================================================
# Network-level forward shapes
# ====================================================================

def test_binary_forward_shape_and_range():
    model = BayesianNetworkLRT(dim=6, p=5, hidden_layers=2, classification=True, n_classes=1)
    y = model(torch.randn(8, 5), ensemble=False)
    assert y.shape == (8, 1)
    assert torch.all(y >= 0.0) and torch.all(y <= 1.0)


def test_regression_forward_shape():
    model = BayesianNetworkLRT(dim=6, p=5, hidden_layers=2, classification=False, n_classes=1)
    y = model(torch.randn(4, 5), ensemble=False)
    assert y.shape == (4, 1)


def test_multiclass_forward_shape():
    model = BayesianNetworkLRT(dim=6, p=5, hidden_layers=2, classification=True, n_classes=3)
    y = model(torch.randn(7, 5), ensemble=False)
    assert y.shape == (7, 3)


# ====================================================================
# input_skip=False architecture
# ====================================================================

def test_lrt_input_skip_false_hidden_layer_input_dim():
    """Without skip, hidden layers see only ``dim`` features (not ``dim + p``)."""
    model = BayesianNetworkLRT(
        dim=4, p=5, hidden_layers=3,
        classification=True, n_classes=1, input_skip=False,
    )
    # Layer 0 still sees raw input (5 features).
    assert model.linears[0].weight_mu.shape[1] == 5
    # Subsequent layers see only ``dim`` features when skip is off.
    for layer in model.linears[1:]:
        assert layer.weight_mu.shape[1] == 4


def test_lrt_input_skip_false_forward_and_kl_finite():
    torch.manual_seed(0)
    model = BayesianNetworkLRT(
        dim=4, p=5, hidden_layers=2,
        classification=True, n_classes=1, input_skip=False,
    )
    y = model(torch.randn(8, 5), ensemble=True, calculate_log_probs=True)
    assert y.shape == (8, 1)
    assert torch.isfinite(model.kl())


# ====================================================================
# post_train determinism + custom_loss + default loss selection
# ====================================================================

def test_lrt_post_train_is_deterministic_in_eval():
    torch.manual_seed(0)
    model = BayesianNetworkLRT(dim=4, p=5, hidden_layers=2, classification=True, n_classes=1)
    model.eval()
    x = torch.randn(6, 5)
    with torch.no_grad():
        y1 = model(x, ensemble=False, sample=False, post_train=True)
        y2 = model(x, ensemble=False, sample=False, post_train=True)
    assert torch.equal(y1, y2)


def test_lrt_custom_loss_is_used():
    sentinel = []

    def my_loss(out: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        sentinel.append(True)
        return ((out - target) ** 2).sum()

    model = BayesianNetworkLRT(
        dim=4, p=5, hidden_layers=2,
        classification=True, n_classes=1, custom_loss=my_loss,
    )
    assert model.loss is my_loss
    loss = model.loss(torch.zeros(3, 1), torch.ones(3, 1))
    assert sentinel == [True]
    assert torch.allclose(loss, torch.tensor(3.0))


def test_lrt_default_loss_per_task():
    bin_ = BayesianNetworkLRT(dim=4, p=5, hidden_layers=2, classification=True, n_classes=1)
    mc = BayesianNetworkLRT(dim=4, p=5, hidden_layers=2, classification=True, n_classes=3)
    reg = BayesianNetworkLRT(dim=4, p=5, hidden_layers=2, classification=False, n_classes=1)
    assert isinstance(bin_.loss, nn.BCELoss)
    assert isinstance(mc.loss, nn.NLLLoss)
    assert isinstance(reg.loss, nn.MSELoss)


# ====================================================================
# predict / predict_proba (inherited from BayesianNetworkBase)
# ====================================================================

def test_predict_binary_returns_zero_one():
    torch.manual_seed(0)
    model = BayesianNetworkLRT(dim=4, p=5, hidden_layers=2, classification=True, n_classes=1)
    pred = model.predict(torch.randn(8, 5))
    assert pred.shape == (8,)
    assert torch.all((pred == 0) | (pred == 1))


def test_predict_proba_binary_in_unit_range():
    torch.manual_seed(0)
    model = BayesianNetworkLRT(dim=4, p=5, hidden_layers=2, classification=True, n_classes=1)
    proba = model.predict_proba(torch.randn(8, 5))
    assert proba.shape == (8, 1)
    assert torch.all((proba >= 0.0) & (proba <= 1.0))


def test_predict_multiclass_returns_class_indices():
    torch.manual_seed(0)
    model = BayesianNetworkLRT(dim=4, p=5, hidden_layers=2, classification=True, n_classes=3)
    pred = model.predict(torch.randn(8, 5))
    assert pred.shape == (8,)
    assert torch.all((pred >= 0) & (pred < 3))


def test_predict_proba_multiclass_sums_to_one():
    torch.manual_seed(0)
    model = BayesianNetworkLRT(dim=4, p=5, hidden_layers=2, classification=True, n_classes=4)
    proba = model.predict_proba(torch.randn(8, 5))
    assert proba.shape == (8, 4)
    sums = proba.sum(dim=1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_predict_regression_shape():
    torch.manual_seed(0)
    model = BayesianNetworkLRT(dim=4, p=5, hidden_layers=2, classification=False, n_classes=1)
    pred = model.predict(torch.randn(8, 5))
    assert pred.shape == (8,)
    assert torch.all(torch.isfinite(pred))
