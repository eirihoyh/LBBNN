"""Minimal example of BayesianNetworkCNNLRT and BayesianNetworkCNNFlow.

Demonstrates a forward pass and KL divergence computation on synthetic
"images" (random 8x8 grayscale tensors). No real dataset required.

Run:
    python examples/cnn_usage.py
"""
import torch

from LBBNN import BayesianNetworkCNNLRT, BayesianNetworkCNNFlow

torch.manual_seed(0)

# Synthetic 1-channel 8x8 images, flattened to (N, p).
N = 32
C, H, W = 1, 8, 8
p = C * H * W   # 64

X = torch.randn(N, p)

# ------------------------------------------------------------------ #
# LRT-based CNN
# ------------------------------------------------------------------ #
lrt_model = BayesianNetworkCNNLRT(
    init_in_channels=C,
    out_channel_list=[8, 16],   # two conv layers
    kernel_size=3,
    stride=1,
    padding=1,
    p1=H,
    p2=W,
    dim=32,
    hidden_layers=1,
    classification=True,
    n_classes=1,
    act_func=torch.relu,
)

with torch.no_grad():
    lrt_preds = lrt_model(X, ensemble=False)

print("=== BayesianNetworkCNNLRT ===")
print(f"Output shape : {lrt_preds.shape}")   # (32, 1)
print(f"Output range : [{lrt_preds.min():.4f}, {lrt_preds.max():.4f}]")

# KL requires a training-mode forward pass first.
lrt_model.train()
_ = lrt_model(X, ensemble=True)
print(f"KL divergence: {lrt_model.kl():.4f}")
print()

# ------------------------------------------------------------------ #
# Flow-based CNN
# ------------------------------------------------------------------ #
flow_model = BayesianNetworkCNNFlow(
    init_in_channels=C,
    out_channel_list=[8, 16],
    kernel_size=3,
    stride=1,
    padding=1,
    p1=H,
    p2=W,
    dim=32,
    hidden_layers=1,
    num_transforms=2,
    iaf_h_sizes=(64, 64),       # smaller IAF for a quick demo
    classification=True,
    n_classes=1,
    act_func=torch.relu,
)

with torch.no_grad():
    flow_preds = flow_model(X, ensemble=False)

print("=== BayesianNetworkCNNFlow ===")
print(f"Output shape : {flow_preds.shape}")  # (32, 1)
print(f"Output range : [{flow_preds.min():.4f}, {flow_preds.max():.4f}]")
print(f"KL divergence: {flow_model.kl():.4f}")
