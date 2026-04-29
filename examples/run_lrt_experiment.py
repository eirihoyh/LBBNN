import json
import math
import csv
from pathlib import Path

import numpy as np
import torch

from LBBNN import (
    BayesianNetworkLRT,
    get_data,
    train_epoch,
    validate,
    clean_alpha,
    get_active_weights,
    plotting,
    local_explain_piecewise_linear_act,
    weight_matrices,
    what_if_explanations,
)


# ============================================================
# Configuration
# ============================================================
SEED = 42
RESULTS_DIR = Path("results/lrt_run")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

N_SAMPLES = 20000
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15

DIM = 20
HIDDEN_LAYERS = 4
LR = 1e-1
EPOCHS = 100
BATCH_SIZE = 1024
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
# Main
# ============================================================
def main():
    set_seed(SEED)

    # --------------------------------------------------------
    # 1. Data
    # --------------------------------------------------------

    _, y_np, X_np = get_data(
        n=N_SAMPLES,
        classification=True,
        non_lin=True,
        seed=SEED,
    )

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
    # NOTE:
    # We use ReLU so that gradient/piecewise-linear 
    # gives exact linear explanations
    # --------------------------------------------------------
    model = BayesianNetworkLRT(
        dim=DIM,
        p=p,
        hidden_layers=HIDDEN_LAYERS,
        classification=True,
        n_classes=1,
        act_func=torch.relu,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # --------------------------------------------------------
    # 3. Training loop
    # --------------------------------------------------------
    history = []
    nr_weights = sum(param.numel() for param in weight_matrices(net=model))

    for epoch in range(1, EPOCHS + 1):
        train_nll, train_loss = train_epoch(
            net=model,
            train_data=train_data,
            optimizer=optimizer,
            batch_size=BATCH_SIZE,
            num_batches=num_batches,
            p=p,
            device=DEVICE,
            nr_weights=nr_weights,
            task="binary",
            verbose=False,
        )

        val_nll, val_loss, val_metric = validate(
            net=model,
            val_data=val_data,
            device=DEVICE,
            task="binary",
            verbose=False,
        )

        history.append(
            {
                "epoch": epoch,
                "kl_model": float(model.kl()),
                "train_nll": float(train_nll),
                "train_loss": float(train_loss),
                "val_nll": float(val_nll),
                "val_loss": float(val_loss),
                "val_accuracy": float(val_metric),
            }
        )

        print(
            f"[LRT] Epoch {epoch:03d} | "
            f"kl_model={model.kl():.4f} | "
            f"train_nll={train_loss:.4f} | "
            f"val_nll={val_nll:.4f} | "
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

    print(f"[LRT] Test accuracy: {test_acc:.4f}")

    # --------------------------------------------------------
    # 5. Save model + history + predictions
    # --------------------------------------------------------
    torch.save(model.state_dict(), RESULTS_DIR / "lrt_model.pt")
    save_history_csv(history, RESULTS_DIR / "history.csv")

    save_predictions_csv(
        y_true=y_test.detach().cpu().numpy(),
        y_prob=test_prob.detach().cpu().numpy(),
        y_pred=test_pred.detach().cpu().numpy(),
        path=RESULTS_DIR / "test_predictions.csv",
    )

    summary = {
        "model_type": "BayesianNetworkLRT",
        "seed": SEED,
        "device": str(DEVICE),
        "n_samples": N_SAMPLES,
        "input_dim": int(p),
        "dim": DIM,
        "hidden_layers": HIDDEN_LAYERS,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LR,
        "test_accuracy": float(test_acc),
    }
    save_json(summary, RESULTS_DIR / "summary.json")

    # --------------------------------------------------------
    # 6. Global network structure
    # --------------------------------------------------------
    structure_metrics = plotting.get_metrics(model, threshold=THRESHOLD)
    save_json(structure_metrics, RESULTS_DIR / "global_structure_metrics.json")

    # Also save numpy metric files through the package helper
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
        )
        print("[LRT] Saved path graph.")
    except Exception as exc:
        print(f"[LRT] Path graph was skipped ({exc}).")

    # --------------------------------------------------------
    # 7. Local explanation for one test sample and 
    # what-if local explanation when adjusting one covariate
    # --------------------------------------------------------

    x_explain = X_test[0].detach()

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
        x=x_explain.detach().cpu().numpy(),
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

    feature_names = ['Bias', 'x_1', 'x_2']
    
    minimum, maximum = X_test.detach().cpu().numpy().min(), X_test.detach().cpu().numpy().max()

    observed_space, contributions_feature_1, predictions_feature_1 = what_if_explanations(
        model, 
        x_explain.detach(), 
        minimum=minimum,
        maximum=maximum,
        feature_index=1,
        n_samples=50,
        n_expl_per_sample=100)
    
    plotting.plot_what_if_explanations(
        observed_space,
        contributions_feature_1,
        predictions_feature_1,
        x_explain.detach().cpu().numpy(),
        feature_names=feature_names,
        feature_in_focus=1,
        save_path=str(RESULTS_DIR / "what-if_explanation_feature_1")
    )

    observed_space, contributions_feature_2, predictions_feature_2 = what_if_explanations(
        model, 
        x_explain.detach(), 
        minimum=minimum,
        maximum=maximum,
        feature_index=2,
        n_samples=50,
        n_expl_per_sample=100)
    
    plotting.plot_what_if_explanations(
        observed_space,
        contributions_feature_2,
        predictions_feature_2,
        x_explain.detach().cpu().numpy(),
        feature_names=feature_names,
        feature_in_focus=2,
        save_path=str(RESULTS_DIR / "what-if_explanation_feature_2")
    )

    print(f"[LRT] Saved all results to: {RESULTS_DIR.resolve()}")


if __name__ == "__main__":
    main()