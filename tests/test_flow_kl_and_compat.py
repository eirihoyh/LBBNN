import torch
from LBBNN import InputSkipFlowNetwork

def test_flow_kl_is_scalar_and_finite():
    torch.manual_seed(0)
    model = InputSkipFlowNetwork(dim=4, p=5, hidden_layers=2, num_transforms=2, classification=True, n_classes=1)
    kl = model.kl()
    assert kl.ndim == 0
    assert torch.isfinite(kl)
