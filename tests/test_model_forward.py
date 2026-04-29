import torch
from LBBNN import BayesianNetworkLRT

def test_binary_forward_shape_and_range():
    model = BayesianNetworkLRT(dim=6, p=5, hidden_layers=2, classification=True, n_classes=1)
    x = torch.randn(8, 5)
    y = model(x, ensemble=False)
    assert y.shape == (8, 1)
    assert torch.all(y >= 0.0)
    assert torch.all(y <= 1.0)

def test_regression_forward_shape():
    model = BayesianNetworkLRT(dim=6, p=5, hidden_layers=2, classification=False, n_classes=1)
    x = torch.randn(4, 5)
    y = model(x, ensemble=False)
    assert y.shape == (4, 1)

def test_multiclass_forward_shape():
    model = BayesianNetworkLRT(dim=6, p=5, hidden_layers=2, classification=True, n_classes=3)
    x = torch.randn(7, 5)
    y = model(x, ensemble=False)
    assert y.shape == (7, 3)
