"""CIFAR-10 classification with BayesianNetworkCNNFlow (3-channel colour).

Images are kept as full RGB 32×32 tensors and flattened to (N, 3072)
before being passed to the model, which internally reshapes them to
(N, 3, 32, 32) for the convolutional stack.  Using all three channels
is the natural choice for a CNN — unlike the feed-forward examples which
convert to grayscale to keep the input tractable.

Run from the image_cnn_examples/ directory:
    python run_flow_cnn_cifar10.py

Requires torchvision:
    pip install torchvision
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

try:
    from torchvision import datasets, transforms
except ImportError as exc:
    raise SystemExit("torchvision is required: pip install torchvision") from exc

from LBBNN import (BayesianNetworkCNNFlow, train_epoch, validate,
                   clean_alpha, network_density_reduction, expected_number_of_weights)
from _common import set_seed, save_json, save_history_csv


# ============================================================
# Configuration
# ============================================================
SEED = 42
DATA_DIR = Path("data")
RESULTS_DIR = Path("results/flow_cnn_cifar10")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

N_CLASSES = 10
IN_CHANNELS = 3
IMG_H, IMG_W = 32, 32
P = IN_CHANNELS * IMG_H * IMG_W   # 3072 (no bias column for CNN)

OUT_CHANNELS = [32, 64, 128]
KERNEL_SIZE = 3
STRIDE = 2
PADDING = 1
NUM_TRANSFORMS = 2
IAF_H_SIZES = (64, 64)

DIM = 256
HIDDEN_LAYERS = 1
LR = 1e-3
EPOCHS = 100
BATCH_SIZE = 256
VAL_FRAC = 0.15
THRESHOLD = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


# ============================================================
# Data helpers
# ============================================================
def _load_split(train: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (X_flat, y_long) with X in CHW order."""
    ds = datasets.CIFAR10(root=str(DATA_DIR), train=train, download=True)
    # ds.data: (N, 32, 32, 3) uint8 numpy array
    X_raw = torch.tensor(ds.data, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0  # (N, 3, 32, 32)
    X_flat = X_raw.reshape(X_raw.shape[0], -1)  # (N, 3072)
    y = torch.tensor(ds.targets).long()
    return X_flat, y


def _pack(X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.cat([X, y.float().unsqueeze(1)], dim=1)


# ============================================================
# Main
# ============================================================
def main() -> None:
    set_seed(SEED)

    # ---- Data -----------------------------------------------
    X_tr_full, y_tr_full = _load_split(train=True)
    X_test, y_test = _load_split(train=False)

    n_val = int(VAL_FRAC * len(X_tr_full))
    idx = torch.randperm(len(X_tr_full))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    X_train, y_train = X_tr_full[train_idx], y_tr_full[train_idx]
    X_val, y_val = X_tr_full[val_idx], y_tr_full[val_idx]

    train_data = _pack(X_train, y_train).to(DEVICE)
    val_data = _pack(X_val, y_val).to(DEVICE)
    X_test_d, y_test_d = X_test.to(DEVICE), y_test.to(DEVICE)
    num_batches = math.ceil(len(train_data) / BATCH_SIZE)

    # ---- Model ----------------------------------------------
    model = BayesianNetworkCNNFlow(
        init_in_channels=IN_CHANNELS,
        out_channel_list=OUT_CHANNELS,
        kernel_size=KERNEL_SIZE,
        stride=STRIDE,
        padding=PADDING,
        p1=IMG_H,
        p2=IMG_W,
        dim=DIM,
        hidden_layers=HIDDEN_LAYERS,
        num_transforms=NUM_TRANSFORMS,
        iaf_h_sizes=IAF_H_SIZES,
        classification=True,
        n_classes=N_CLASSES,
        act_func=torch.relu,
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    nr_weights = (
        sum(c.weight_mu.numel() for c in model.convs)
        + sum(l.weight_mu.numel() for l in model.linears)
    )

    # ---- Training loop --------------------------------------
    history = []
    for epoch in range(1, EPOCHS + 1):
        train_nll, train_loss = train_epoch(
            net=model, train_data=train_data, optimizer=optimizer,
            batch_size=BATCH_SIZE, num_batches=num_batches, p=P,
            device=DEVICE, nr_weights=nr_weights, task="multiclass",
        )
        val_nll, val_loss, val_acc = validate(
            net=model, val_data=val_data, device=DEVICE, task="multiclass",
        )
        history.append({
            "epoch": epoch, "train_nll": float(train_nll),
            "train_loss": float(train_loss), "val_nll": float(val_nll),
            "val_loss": float(val_loss), "val_accuracy": float(val_acc),
        })
        print(
            f"[Flow-CNN CIFAR-10] Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | val_nll={val_nll:.4f} | val_acc={val_acc:.4f}"
        )

    # ---- Test evaluation ------------------------------------
    model.eval()
    with torch.no_grad():
        test_pred = model(X_test_d, ensemble=False).argmax(dim=1)
        test_acc = float((test_pred == y_test_d).float().mean().cpu())
    print(f"[Flow-CNN CIFAR-10] Test accuracy: {test_acc:.4f}")

    # ---- Save -----------------------------------------------
    torch.save(model.state_dict(), RESULTS_DIR / "model.pt")
    save_history_csv(history, RESULTS_DIR / "history.csv")
    np.savetxt(
        RESULTS_DIR / "test_predictions.csv",
        np.stack([y_test.numpy(), test_pred.cpu().numpy()], axis=1),
        header="y_true,y_pred", delimiter=",", comments="", fmt="%d",
    )
    save_json(
        {"model_type": "BayesianNetworkCNNFlow", "dataset": "CIFAR-10 (RGB)",
         "seed": SEED, "device": str(DEVICE),
         "in_channels": IN_CHANNELS, "img_h": IMG_H, "img_w": IMG_W,
         "out_channel_list": OUT_CHANNELS, "kernel_size": KERNEL_SIZE,
         "stride": STRIDE, "padding": PADDING,
         "num_transforms": NUM_TRANSFORMS, "iaf_h_sizes": list(IAF_H_SIZES),
         "dim": DIM, "hidden_layers": HIDDEN_LAYERS,
         "epochs": EPOCHS, "batch_size": BATCH_SIZE,
         "learning_rate": LR, "test_accuracy": test_acc},
        RESULTS_DIR / "summary.json",
    )

    # ---- Structure metrics (FC layers only) -----------------
    model.eval()
    alpha_clean = clean_alpha(model, THRESHOLD)
    density, used_weights, total_weights = network_density_reduction(alpha_clean)
    save_json(
        {"fc_total_weights": total_weights,
         "fc_used_weights_median": used_weights,
         "fc_density_median": density,
         "fc_expected_nr_weights": expected_number_of_weights(model)},
        RESULTS_DIR / "structure_metrics.json",
    )

    print(f"[Flow-CNN CIFAR-10] Results saved to: {RESULTS_DIR.resolve()}")


if __name__ == "__main__":
    main()
