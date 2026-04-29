import pytest
import torch

from LBBNN import IAF, RNVP, PropagateFlow


def test_propagate_flow_iaf_shape_and_finite_log_det():
    torch.manual_seed(0)
    flow = PropagateFlow("IAF", dim=8, num_transforms=2, iaf_h_sizes=(16, 16))
    z = torch.randn(4, 8)
    zk, log_det = flow(z)
    assert zk.shape == z.shape
    assert torch.all(torch.isfinite(zk))
    assert torch.is_tensor(log_det)
    assert torch.all(torch.isfinite(log_det))


def test_propagate_flow_rnvp_shape_and_finite_log_det():
    torch.manual_seed(0)
    flow = PropagateFlow("RNVP", dim=8, num_transforms=2, rnvp_h_sizes=(8, 8))
    z = torch.randn(4, 8)
    zk, log_det = flow(z)
    assert zk.shape == z.shape
    assert torch.all(torch.isfinite(zk))
    assert torch.is_tensor(log_det)
    assert torch.all(torch.isfinite(log_det))


def test_propagate_flow_invalid_transform_raises():
    with pytest.raises(ValueError):
        PropagateFlow("FOO", dim=8, num_transforms=2)


def test_iaf_single_transform_output_shape():
    torch.manual_seed(0)
    layer = IAF(dim=6, h_sizes=(12, 12))
    z = torch.randn(3, 6)
    out = layer(z)
    assert out.shape == z.shape
    log_det = layer.log_det()
    assert torch.all(torch.isfinite(log_det))


def test_rnvp_single_transform_output_shape():
    torch.manual_seed(0)
    layer = RNVP(dim=6, h_sizes=(8, 8))
    z = torch.randn(3, 6)
    out = layer(z)
    assert out.shape == z.shape
    log_det = layer.log_det()
    assert torch.all(torch.isfinite(log_det))
