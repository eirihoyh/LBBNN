"""Tests for ``LBBNN.flow_cnn`` (BayesianConv2dFlow and BayesianNetworkCNNFlow)."""
import torch
import torch.nn as nn

from LBBNN import BayesianConv2dFlow, BayesianNetworkCNNFlow

# Small image fixture: 1-channel 8x8 images (p = 64)
_C, _H, _W = 1, 8, 8
_P = _C * _H * _W  # 64

# Use tiny flow sizes for fast tests.
_IAF = (16, 16)


def _make_flow_cnn(**kwargs) -> BayesianNetworkCNNFlow:
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
        num_transforms=2,
        iaf_h_sizes=_IAF,
        classification=True,
        n_classes=1,
    )
    defaults.update(kwargs)
    return BayesianNetworkCNNFlow(**defaults)


# ====================================================================
# Layer-level: BayesianConv2dFlow
# ====================================================================

def test_flow_conv_layer_forward_shape():
    torch.manual_seed(0)
    layer = BayesianConv2dFlow(
        in_channels=1, out_channels=4, kernel_size=3, num_transforms=2,
        padding=1, iaf_h_sizes=_IAF,
    )
    x = torch.randn(2, 1, 8, 8)
    out = layer(x)
    assert out.shape == (2, 4, 8, 8)


def test_flow_conv_layer_kl_div_is_finite_scalar():
    torch.manual_seed(0)
    layer = BayesianConv2dFlow(
        in_channels=1, out_channels=4, kernel_size=3, num_transforms=2,
        padding=1, iaf_h_sizes=_IAF,
    )
    _ = layer(torch.randn(2, 1, 8, 8))
    kl = layer.kl_div()
    assert kl.ndim == 0
    assert torch.isfinite(kl)


# ====================================================================
# Network-level forward shapes
# ====================================================================

def test_flow_cnn_binary_forward_shape_and_range():
    torch.manual_seed(0)
    model = _make_flow_cnn(classification=True, n_classes=1)
    y = model(torch.randn(4, _P), ensemble=False)
    assert y.shape == (4, 1)
    assert torch.all(y >= 0.0) and torch.all(y <= 1.0)


def test_flow_cnn_multiclass_forward_shape():
    torch.manual_seed(0)
    model = _make_flow_cnn(classification=True, n_classes=3)
    y = model(torch.randn(4, _P), ensemble=False)
    assert y.shape == (4, 3)


def test_flow_cnn_regression_forward_shape():
    torch.manual_seed(0)
    model = _make_flow_cnn(classification=False, n_classes=1)
    y = model(torch.randn(4, _P), ensemble=False)
    assert y.shape == (4, 1)


def test_flow_cnn_ensemble_forward_shape():
    torch.manual_seed(0)
    model = _make_flow_cnn()
    y = model(torch.randn(4, _P), ensemble=True)
    assert y.shape == (4, 1)


# ====================================================================
# input_skip=False architecture
# ====================================================================

def test_flow_cnn_input_skip_false_fc_layer_widths():
    """Without skip, all FC layers after the first should have width == dim."""
    model = _make_flow_cnn(input_skip=False, dim=8)
    assert model.linears[1].weight_mu.shape[1] == 8


def test_flow_cnn_input_skip_false_forward_and_kl():
    torch.manual_seed(0)
    model = _make_flow_cnn(input_skip=False)
    y = model(torch.randn(4, _P), ensemble=True)
    assert y.shape == (4, 1)
    assert torch.isfinite(model.kl())


# ====================================================================
# KL divergence
# ====================================================================

def test_flow_cnn_kl_finite_after_forward():
    torch.manual_seed(0)
    model = _make_flow_cnn()
    _ = model(torch.randn(4, _P), ensemble=True)
    assert torch.isfinite(model.kl())


# ====================================================================
# Post-train determinism
# ====================================================================

# def test_flow_cnn_post_train_is_deterministic():
#     torch.manual_seed(0)
#     model = _make_flow_cnn()
#     model.eval()
#     x = torch.randn(4, _P)
#     with torch.no_grad():
#         y1 = model(x, ensemble=False, sample=False, post_train=True)
#         y2 = model(x, ensemble=False, sample=False, post_train=True)
#     assert torch.equal(y1, y2)
# TODO: make a deterministic option for flow networks

# ====================================================================
# Loss selection and custom_loss
# ====================================================================

def test_flow_cnn_default_loss_per_task():
    binary = _make_flow_cnn(classification=True, n_classes=1)
    mc = _make_flow_cnn(classification=True, n_classes=3)
    reg = _make_flow_cnn(classification=False, n_classes=1)
    assert isinstance(binary.loss, nn.BCELoss)
    assert isinstance(mc.loss, nn.NLLLoss)
    assert isinstance(reg.loss, nn.MSELoss)


def test_flow_cnn_custom_loss_is_used():
    sentinel = []

    def my_loss(out: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        sentinel.append(True)
        return ((out - target) ** 2).sum()

    model = _make_flow_cnn(custom_loss=my_loss)
    assert model.loss is my_loss
    model.loss(torch.zeros(3, 1), torch.ones(3, 1))
    assert sentinel == [True]


# ====================================================================
# predict / predict_proba (inherited from BayesianNetworkBase)
# ====================================================================

def test_flow_cnn_predict_binary_returns_zero_one():
    torch.manual_seed(0)
    model = _make_flow_cnn(classification=True, n_classes=1)
    pred = model.predict(torch.randn(6, _P))
    assert pred.shape == (6,)
    assert torch.all((pred == 0) | (pred == 1))


def test_flow_cnn_predict_proba_binary_in_unit_range():
    torch.manual_seed(0)
    model = _make_flow_cnn(classification=True, n_classes=1)
    proba = model.predict_proba(torch.randn(6, _P))
    assert proba.shape == (6, 1)
    assert torch.all((proba >= 0.0) & (proba <= 1.0))


def test_flow_cnn_predict_multiclass_returns_class_indices():
    torch.manual_seed(0)
    model = _make_flow_cnn(classification=True, n_classes=3)
    pred = model.predict(torch.randn(6, _P))
    assert pred.shape == (6,)
    assert torch.all((pred >= 0) & (pred < 3))


def test_flow_cnn_predict_proba_multiclass_sums_to_one():
    torch.manual_seed(0)
    model = _make_flow_cnn(classification=True, n_classes=3)
    proba = model.predict_proba(torch.randn(6, _P))
    assert proba.shape == (6, 3)
    sums = proba.sum(dim=1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
