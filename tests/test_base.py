"""Tests for ``LBBNN._base.BayesianNetworkBase``.

Most of the base class behavior is already covered by the LRT and Flow
network tests; this file exercises the base class directly so a regression
in the shared ``predict`` / ``predict_proba`` logic is detected even if
both subclasses fail in the same way.
"""
import torch
import torch.nn as nn

from LBBNN import BayesianNetworkFlow, BayesianNetworkLRT
from LBBNN._base import BayesianNetworkBase


def test_lrt_and_flow_inherit_from_base():
    lrt = BayesianNetworkLRT(dim=4, p=5, hidden_layers=2, classification=True, n_classes=1)
    flow = BayesianNetworkFlow(
        dim=4, p=5, hidden_layers=2, num_transforms=2,
        classification=True, n_classes=1,
    )
    assert isinstance(lrt, BayesianNetworkBase)
    assert isinstance(flow, BayesianNetworkBase)


class _FakeNet(BayesianNetworkBase):
    """Minimal subclass: a deterministic linear classifier with a fixed forward.

    Used to verify that ``predict`` / ``predict_proba`` work on any class that
    sets ``classification`` and ``multiclass`` and provides a ``forward``.
    """

    def __init__(self, in_features: int, n_classes: int, classification: bool):
        super().__init__()
        self.classification = classification
        self.multiclass = n_classes > 1
        self.linear = nn.Linear(in_features, n_classes)

    def forward(self, x: torch.Tensor, ensemble: bool = False, **_: object) -> torch.Tensor:
        out = self.linear(x)
        if self.classification:
            if self.multiclass:
                return torch.log_softmax(out, dim=1)
            return torch.sigmoid(out)
        return out


def test_base_predict_binary_on_minimal_subclass():
    torch.manual_seed(0)
    net = _FakeNet(in_features=5, n_classes=1, classification=True)
    pred = net.predict(torch.randn(6, 5))
    assert pred.shape == (6,)
    assert torch.all((pred == 0) | (pred == 1))


def test_base_predict_proba_multiclass_sums_to_one():
    torch.manual_seed(0)
    net = _FakeNet(in_features=5, n_classes=3, classification=True)
    proba = net.predict_proba(torch.randn(6, 5))
    assert proba.shape == (6, 3)
    sums = proba.sum(dim=1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_base_predict_regression_passes_output_through():
    torch.manual_seed(0)
    net = _FakeNet(in_features=5, n_classes=1, classification=False)
    pred = net.predict(torch.randn(6, 5))
    assert pred.shape == (6,)
    assert torch.all(torch.isfinite(pred))
