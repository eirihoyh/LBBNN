import json
import math
import csv
from pathlib import Path

import numpy as np
import torch

from LBBNN import (
    InputSkipFlowNetwork,
    create_data_unif,
    get_data,
    clean_alpha,
    get_active_weights,
    plotting,
    local_explain_piecewise_linear_act,
)


# ============================================================
# Configuration
# ============================================================
SEED = 42
RESULTS_DIR = Path("results/flow_run")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

N_SAMPLES = 20000
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15

DIM = 16
HIDDEN_LAYERS = 2
NUM_TRANSFORMS = 2
LR = 5e-3
EPOCHS = 30
BATCH_SIZE = 32
THRESHOLD = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Utilities
# ============================================================
def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_json_serializable(obj):
    """
    Recursively convert NumPy / PyTorch objects into JSON-serializable Python types.
    """
    if isinstance(obj, dict):
        return {str(k): _to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_to_json_serializable(v) for v in obj]
    elif isinstance(obj, tuple):
        return [_to_json_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, torch.Tensor):
        if obj.ndim == 0:
            return obj.item()
        return obj.detach().cpu().tolist()
    else:
        return obj


def save_json(obj, path: Path) -> None:
    obj = _to_json_serializable(obj)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def save_history_csv(history, path: Path) -> None:
    if not history:
        return
    fieldnames = list(history[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def save_predictions_csv(y_true, y_prob, y_pred, path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["y_true", "y_prob", "y_pred"])
        for yt, yp, yh in zip(y_true, y_prob, y_pred):
            writer.writerow([float(yt), float(yp), float(yh)])


def binary_accuracy(y_prob: torch.Tensor, y_true: torch.Tensor, threshold: float = 0.5) -> float:
    y_hat = (y_prob >= threshold).float()
    return float((y_hat.view(-1) == y_true.view(-1)).float().mean().cpu())


def split_dataset(X: torch.Tensor, y: torch.Tensor):
    n = X.shape[0]
    idx = torch.randperm(n)

    n_train = int(TRAIN_FRAC * n)
    n_val = int(VAL_FRAC * n)
    n_test = n - n_train - n_val

    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    train_data = torch.cat([X_train, y_train.unsqueeze(1)], dim=1)
    val_data = torch.cat([X_val, y_val.unsqueeze(1)], dim=1)
    test_data = torch.cat([X_test, y_test.unsqueeze(1)], dim=1)

    return (X_train, y_train, train_data), (X_val, y_val, val_data), (X_test, y_test, test_data)


# ============================================================
# FLOW-specific train/validate helpers
# ============================================================
def train_epoch_flow(model, train_data, optimizer, batch_size, num_batches, p, device):
    model.train()

    idx = np.random.permutation(len(train_data))
    train_data = train_data[idx]

    last_nll = None
    last_loss = None
    old_batch = 0

    for batch in range(int(np.ceil(train_data.shape[0] / batch_size))):
        batch = batch + 1
        x_batch = train_data[old_batch: batch_size * batch, 0:p]
        y_batch = train_data[old_batch: batch_size * batch, -1]
        old_batch = batch_size * batch

        data = x_batch.to(device)
        target = y_batch.to(device).float().unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(data, ensemble=True)
        nll = model.loss(outputs, target)
        kl_part = model.kl() / max(num_batches, 1)
        loss = nll + kl_part
        loss.backward()
        optimizer.step()

        last_nll = float(nll.detach().cpu())
        last_loss = float(loss.detach().cpu())

    return last_nll, last_loss


def validate_flow(model, val_data, device):
    model.eval()
    with torch.no_grad():
        x_val = val_data[:, :-1].to(device)
        y_val = val_data[:, -1].to(device).float().unsqueeze(1)

        outputs = model(x_val, ensemble=False)
        nll = model.loss(outputs, y_val)
        loss = nll + model.kl()
        acc = float(((outputs >= THRESHOLD).float() == y_val).float().mean().cpu())

    return float(nll.detach().cpu()), float(loss.detach().cpu()), acc


# ============================================================
# Main
# ============================================================
def main():
    set_seed(SEED)

    # --------------------------------------------------------
    # 1. Data
    # --------------------------------------------------------
    # y_np, X_np = create_data_unif(
    #     n=N_SAMPLES,
    #     classification=True,
    #     seed=SEED,
    # )
    _, y_np, X_np = get_data(n=N_SAMPLES, classification=True)

    X = torch.tensor(X_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.float32)

    (X_train, y_train, train_data), (X_val, y_val, val_data), (X_test, y_test, test_data) = split_dataset(X, y)

    train_data = train_data.to(DEVICE)
    val_data = val_data.to(DEVICE)
    test_data = test_data.to(DEVICE)
    X_test = X_test.to(DEVICE)
    y_test = y_test.to(DEVICE)

    p = X.shape[1]
    num_batches = math.ceil(len(train_data) / BATCH_SIZE)

    # --------------------------------------------------------
    # 2. Model
    # --------------------------------------------------------
    model = InputSkipFlowNetwork(
        dim=DIM,
        p=p,
        hidden_layers=HIDDEN_LAYERS,
        num_transforms=NUM_TRANSFORMS,
        classification=True,
        n_classes=1,
        act_func=torch.relu,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # --------------------------------------------------------
    # 3. Training loop
    # --------------------------------------------------------
    history = []

    for epoch in range(1, EPOCHS + 1):
        train_nll, train_loss = train_epoch_flow(
            model=model,
            train_data=train_data,
            optimizer=optimizer,
            batch_size=BATCH_SIZE,
            num_batches=num_batches,
            p=p,
            device=DEVICE,
        )

        val_nll, val_loss, val_metric = validate_flow(
            model=model,
            val_data=val_data,
            device=DEVICE,
        )

        history.append(
            {
                "epoch": epoch,
                "train_nll": float(train_nll),
                "train_loss": float(train_loss),
                "val_nll": float(val_nll),
                "val_loss": float(val_loss),
                "val_accuracy": float(val_metric),
            }
        )

        print(
            f"[FLOW] Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_metric:.4f}"
        )

    # --------------------------------------------------------
    # 4. Final test evaluation
    # --------------------------------------------------------
    model.eval()
    with torch.no_grad():
        test_prob = model(X_test, ensemble=False).view(-1)
        test_pred = (test_prob >= THRESHOLD).float()
        test_acc = binary_accuracy(test_prob, y_test, threshold=THRESHOLD)

    print(f"[FLOW] Test accuracy: {test_acc:.4f}")

    # --------------------------------------------------------
    # 5. Save model + history + predictions
    # --------------------------------------------------------
    torch.save(model.state_dict(), RESULTS_DIR / "flow_model.pt")
    save_history_csv(history, RESULTS_DIR / "history.csv")

    save_predictions_csv(
        y_true=y_test.detach().cpu().numpy(),
        y_prob=test_prob.detach().cpu().numpy(),
        y_pred=test_pred.detach().cpu().numpy(),
        path=RESULTS_DIR / "test_predictions.csv",
    )

    summary = {
        "model_type": "InputSkipFlowNetwork",
        "seed": SEED,
        "device": str(DEVICE),
        "n_samples": N_SAMPLES,
        "input_dim": int(p),
        "dim": DIM,
        "hidden_layers": HIDDEN_LAYERS,
        "num_transforms": NUM_TRANSFORMS,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LR,
        "test_accuracy": float(test_acc),
        "final_kl": float(model.kl().detach().cpu()),
    }
    save_json(summary, RESULTS_DIR / "summary.json")

    # --------------------------------------------------------
    # 6. Global network structure
    # --------------------------------------------------------
    structure_metrics = plotting.get_metrics(model, threshold=THRESHOLD)
    save_json(structure_metrics, RESULTS_DIR / "global_structure_metrics.json")

    plotting.save_metrics(model, threshold=THRESHOLD, path=str(RESULTS_DIR / "network_metrics"))

    alpha_clean = clean_alpha(model, threshold=THRESHOLD)
    active_connections = []
    for layer_idx, coords in enumerate(get_active_weights(alpha_clean)):
        layer_connections = coords.detach().cpu().tolist()
        active_connections.append(
            {
                "layer_index": layer_idx,
                "connections": layer_connections,
            }
        )
    save_json(active_connections, RESULTS_DIR / "active_connections.json")

    # Optional path graph if graphviz is installed
    try:
        plotting.run_path_graph(
            model,
            threshold=THRESHOLD,
            save_path=str(RESULTS_DIR / "path_graph"),
            show=False,
        )
        plotting.run_path_graph_weight(
            model,
            threshold=THRESHOLD,
            save_path=str(RESULTS_DIR / "path_graph_weight"),
            show=False,
            flow=True
        )
        print("[FLOW] Saved path graph.")
    except Exception as exc:
        print(f"[FLOW] Path graph was skipped ({exc}).")

    # --------------------------------------------------------
    # 7. Local explanation for one test sample
    # --------------------------------------------------------
    device = torch.device("cpu")
    model.to(device)
    x_explain = X_test[0].detach().cpu()

    expl_values, preds, p_expl = local_explain_piecewise_linear_act(
        net=model,
        input_data=x_explain,
        median=True,
        sample=True,
        n_samples=64,
        magnitude=True,
        include_potential_contribution=False,
        n_classes=1,
    )

    np.savez(
        RESULTS_DIR / "local_explanation_raw.npz",
        explanation=expl_values,
        preds=preds.detach().cpu().numpy(),
        p=np.array([p_expl]),
        x=x_explain.numpy(),
    )

    plotting.plot_local_explain_piecewise_linear_act(
        net=model,
        input_data=x_explain,
        median=True,
        sample=True,
        n_samples=64,
        n_classes=1,
        magnitude=True,
        include_potential_contribution=False,
        variable_names=[f"x{i}" for i in range(len(x_explain))],
        class_names=["positive_class"],
        include_prediction=True,
        include_bias=True,
        no_zero_contributions=False,
        save_path=str(RESULTS_DIR / "local_explanation_plot"),
        show=False,
    )

    print(f"[FLOW] Saved all results to: {RESULTS_DIR.resolve()}")


if __name__ == "__main__":
    main()