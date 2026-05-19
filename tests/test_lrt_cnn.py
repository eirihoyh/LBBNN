"""Tests for ``LBBNN.lrt_cnn`` (BayesianConv2dLRT and BayesianNetworkCNNLRT)."""
import torch
import torch.nn as nn

from LBBNN import BayesianConv2dLRT, BayesianNetworkCNNLRT

# Small image fixture: 1-channel 8x8 images (p = 64)
_C, _H, _W = 1, 8, 8
_P = _C * _H * _W  # 64


def _make_lrt_cnn(**kwargs) -> BayesianNetworkCNNLRT:
    defaults = dict(
        init_in_channels=_C,
        out_channel_list=[4],
        kernel_size=3,
        stride=1,
        padding=1,
        p1=_H,
        p2=_W,
        dim=8,
        hidden_layers=1,
        classification=True,
        n_classes=1,
    )
    defaults.update(kwargs)
    return BayesianNetworkCNNLRT(**defaults)


# ====================================================================
# Layer-level: BayesianConv2dLRT
# ====================================================================

def test_lrt_conv_layer_forward_shape_in_training_mode():
    torch.manual_seed(0)
    layer = BayesianConv2dLRT(in_channels=1, out_channels=4, kernel_size=3, padding=1)
    x = torch.randn(2, 1, 8, 8)
    out = layer(x)
    assert out.shape == (2, 4, 8, 8)


def test_lrt_conv_layer_kl_set_after_forward():
    torch.manual_seed(0)
    layer = BayesianConv2dLRT(in_channels=1, out_channels=4, kernel_size=3, padding=1)
    _ = layer(torch.randn(2, 1, 8, 8))
    assert torch.is_tensor(layer.kl)
    assert torch.isfinite(layer.kl)


def test_lrt_conv_layer_kl_zero_in_eval_without_calculate_log_probs():
    layer = BayesianConv2dLRT(in_channels=1, out_channels=4, kernel_size=3, padding=1)
    layer.eval()
    with torch.no_grad():
        _ = layer(torch.randn(2, 1, 8, 8), ensemble=False, calculate_log_probs=False)
    assert layer.kl == torch.tensor(0.0)


# ====================================================================
# Network-level forward shapes
# ====================================================================

def test_lrt_cnn_binary_forward_shape_and_range():
    torch.manual_seed(0)
    model = _make_lrt_cnn(classification=True, n_classes=1)
    y = model(torch.randn(4, _P), ensemble=False)
    assert y.shape == (4, 1)
    assert torch.all(y >= 0.0) and torch.all(y <= 1.0)


def test_lrt_cnn_multiclass_forward_shape():
    torch.manual_seed(0)
    model = _make_lrt_cnn(classification=True, n_classes=3)
    y = model(torch.randn(4, _P), ensemble=False)
    assert y.shape == (4, 3)


def test_lrt_cnn_regression_forward_shape():
    torch.manual_seed(0)
    model = _make_lrt_cnn(classification=False, n_classes=1)
    y = model(torch.randn(4, _P), ensemble=False)
    assert y.shape == (4, 1)


def test_lrt_cnn_ensemble_forward_shape():
    torch.manual_seed(0)
    model = _make_lrt_cnn()
    y = model(torch.randn(4, _P), ensemble=True)
    assert y.shape == (4, 1)


# ====================================================================
# input_skip=False architecture
# ====================================================================

def test_lrt_cnn_input_skip_false_fc_layer_widths():
    """Without skip, all FC layers after the first should have width == dim."""
    model = _make_lrt_cnn(input_skip=False, dim=8)
    # Second FC layer (linears[1]) should have in_features == dim (not dim + p).
    assert model.linears[1].weight_mu.shape[1] == 8


def test_lrt_cnn_input_skip_false_forward_and_kl():
    torch.manual_seed(0)
    model = _make_lrt_cnn(input_skip=False)
    y = model(torch.randn(4, _P), ensemble=True)
    assert y.shape == (4, 1)
    assert torch.isfinite(model.kl())


# ====================================================================
# KL divergence
# ====================================================================

def test_lrt_cnn_kl_finite_after_forward():
    torch.manual_seed(0)
    model = _make_lrt_cnn()
    _ = model(torch.randn(4, _P), ensemble=True)
    assert torch.isfinite(model.kl())


# ====================================================================
# Post-train determinism
# ====================================================================

def test_lrt_cnn_post_train_is_deterministic():
    torch.manual_seed(0)
    model = _make_lrt_cnn()
    model.eval()
    x = torch.randn(4, _P)
    with torch.no_grad():
        y1 = model(x, ensemble=False, sample=False, post_train=True)
        y2 = model(x, ensemble=False, sample=False, post_train=True)
    assert torch.equal(y1, y2)


# ====================================================================
# Loss selection and custom_loss
# ====================================================================

def test_lrt_cnn_default_loss_per_task():
    binary = _make_lrt_cnn(classification=True, n_classes=1)
    mc = _make_lrt_cnn(classification=True, n_classes=3)
    reg = _make_lrt_cnn(classification=False, n_classes=1)
    assert isinstance(binary.loss, nn.BCELoss)
    assert isinstance(mc.loss, nn.NLLLoss)
    assert isinstance(reg.loss, nn.MSELoss)


def test_lrt_cnn_custom_loss_is_used():
    sentinel = []

    def my_loss(out: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        sentinel.append(True)
        return ((out - target) ** 2).sum()

    model = _make_lrt_cnn(custom_loss=my_loss)
    assert model.loss is my_loss
    model.loss(torch.zeros(3, 1), torch.ones(3, 1))
    assert sentinel == [True]


# ====================================================================
# predict / predict_proba (inherited from BayesianNetworkBase)
# ====================================================================

def test_lrt_cnn_predict_binary_returns_zero_one():
    torch.manual_seed(0)
    model = _make_lrt_cnn(classification=True, n_classes=1)
    pred = model.predict(torch.randn(6, _P))
    assert pred.shape == (6,)
    assert torch.all((pred == 0) | (pred == 1))


def test_lrt_cnn_predict_proba_binary_in_unit_range():
    torch.manual_seed(0)
    model = _make_lrt_cnn(classification=True, n_classes=1)
    proba = model.predict_proba(torch.randn(6, _P))
    assert proba.shape == (6, 1)
    assert torch.all((proba >= 0.0) & (proba <= 1.0))


def test_lrt_cnn_predict_multiclass_returns_class_indices():
    torch.manual_seed(0)
    model = _make_lrt_cnn(classification=True, n_classes=3)
    pred = model.predict(torch.randn(6, _P))
    assert pred.shape == (6,)
    assert torch.all((pred >= 0) & (pred < 3))


def test_lrt_cnn_predict_proba_multiclass_sums_to_one():
    torch.manual_seed(0)
    model = _make_lrt_cnn(classification=True, n_classes=3)
    proba = model.predict_proba(torch.randn(6, _P))
    assert proba.shape == (6, 3)
    sums = proba.sum(dim=1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
