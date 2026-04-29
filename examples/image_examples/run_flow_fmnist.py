"""Fashion-MNIST classification with BayesianNetworkFlow.

Run from the image_examples/ directory:
    python run_flow_fmnist.py

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

from LBBNN import BayesianNetworkFlow, train_epoch, validate, plotting, weight_matrices
from _common import set_seed, save_json, save_history_csv


# ============================================================
# Configuration
# ============================================================
SEED = 42
DATA_DIR = Path("data")
RESULTS_DIR = Path("results/flow_fmnist")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

N_CLASSES = 10
IMG_PIXELS = 28 * 28   # 784
P = IMG_PIXELS + 1     # +1 for leading bias column

DIM = 50
HIDDEN_LAYERS = 2
NUM_TRANSFORMS = 2
LR = 1e-2
EPOCHS = 50
BATCH_SIZE = 256
VAL_FRAC = 0.15
THRESHOLD = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = [
    "T-shirt", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "AnkleBoot",
]


# ============================================================
# Data helpers
# ============================================================
def _load_split(train: bool) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """Return (X_with_bias, y_long, X_pixels_np)."""
    ds = datasets.FashionMNIST(
        root=str(DATA_DIR), train=train, download=True,
        transform=transforms.ToTensor(),
    )
    X_raw = ds.data.float() / 255.0
    X_flat = X_raw.view(X_raw.shape[0], -1)  # (N, 784)
    y = ds.targets.long()
    X_bias = torch.cat([torch.ones(len(X_flat), 1), X_flat], dim=1)  # (N, 785)
    return X_bias, y, X_flat.numpy()


def _pack(X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.cat([X, y.float().unsqueeze(1)], dim=1)


# ============================================================
# Main
# ============================================================
def main() -> None:
    set_seed(SEED)

    # ---- Data -----------------------------------------------
    X_tr_full, y_tr_full, X_raw_full = _load_split(train=True)
    X_test, y_test, _ = _load_split(train=False)

    n_val = int(VAL_FRAC * len(X_tr_full))
    idx = torch.randperm(len(X_tr_full))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    X_train, y_train = X_tr_full[train_idx], y_tr_full[train_idx]
    X_val, y_val = X_tr_full[val_idx], y_tr_full[val_idx]
    X_train_raw = X_raw_full[train_idx.numpy()]

    train_data = _pack(X_train, y_train).to(DEVICE)
    val_data = _pack(X_val, y_val).to(DEVICE)
    X_test_d, y_test_d = X_test.to(DEVICE), y_test.to(DEVICE)
    num_batches = math.ceil(len(train_data) / BATCH_SIZE)

    # ---- Model ----------------------------------------------
    model = BayesianNetworkFlow(
        dim=DIM, p=P, hidden_layers=HIDDEN_LAYERS,
        num_transforms=NUM_TRANSFORMS,
        classification=True, n_classes=N_CLASSES, act_func=torch.relu,
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    nr_weights = sum(param.numel() for param in weight_matrices(net=model))

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
            f"[FLOW FMNIST] Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | val_nll={val_nll:.4f} | val_acc={val_acc:.4f}"
        )

    # ---- Test evaluation ------------------------------------
    model.eval()
    with torch.no_grad():
        test_pred = model(X_test_d, ensemble=False).argmax(dim=1)
        test_acc = float((test_pred == y_test_d).float().mean().cpu())
    print(f"[FLOW FMNIST] Test accuracy: {test_acc:.4f}")

    # ---- Save -----------------------------------------------
    torch.save(model.state_dict(), RESULTS_DIR / "model.pt")
    save_history_csv(history, RESULTS_DIR / "history.csv")
    np.savetxt(
        RESULTS_DIR / "test_predictions.csv",
        np.stack([y_test.numpy(), test_pred.cpu().numpy()], axis=1),
        header="y_true,y_pred", delimiter=",", comments="", fmt="%d",
    )
    save_json(
        {"model_type": "BayesianNetworkFlow", "dataset": "FashionMNIST",
         "seed": SEED, "device": str(DEVICE), "p": P, "dim": DIM,
         "hidden_layers": HIDDEN_LAYERS, "num_transforms": NUM_TRANSFORMS,
         "epochs": EPOCHS, "batch_size": BATCH_SIZE,
         "learning_rate": LR, "test_accuracy": test_acc},
        RESULTS_DIR / "summary.json",
    )

    # ---- Structure metrics ----------------------------------
    save_json(plotting.get_metrics(model, THRESHOLD), RESULTS_DIR / "structure_metrics.json")
    plotting.save_metrics(model, THRESHOLD, str(RESULTS_DIR / "network_metrics"))

    # ---- Image overlay plots --------------------------------
    for c, name in enumerate(CLASS_NAMES):
        plotting.plot_model_vision_image(
            net=model, train_data=X_train_raw, train_target=y_train.numpy(),
            c=c, threshold=THRESHOLD,
            save_path=str(RESULTS_DIR / f"image_overlay_class_{name}"),
        )

    print(f"[FLOW FMNIST] Results saved to: {RESULTS_DIR.resolve()}")


if __name__ == "__main__":
    main()
