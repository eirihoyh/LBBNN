import math
import sys
from pathlib import Path

# included such that _common.py from examples/ is available regardless of where the script is ran from
sys.path.insert(0, str(Path(__file__).parent.parent))

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
    compute_global_explain_piecewise_linear_act,
)
from _common import (
    set_seed,
    save_json,
    save_history_csv,
    save_predictions_csv,
    binary_accuracy,
    split_dataset,
)


# ============================================================
# Configuration
# ============================================================
SEED = 42
RESULTS_DIR = Path("results/lrt_run")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DIM = 50
HIDDEN_LAYERS = 2
NUM_TRANSFORMS = 2
LR = 1e-1
EPOCHS = 1000
BATCH_SIZE = 1024
THRESHOLD = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Main
# ============================================================
def main():
    set_seed(SEED)

    # --------------------------------------------------------
    # 1. Data
    # --------------------------------------------------------
    
    X_train_original = np.loadtxt("dataset/X_train.txt", delimiter=",")
    X_test_original = np.loadtxt("dataset/X_test.txt", delimiter=",")
    y_train_original = np.loadtxt("dataset/Y_train.txt", delimiter=",")
    y_test_original = np.loadtxt("dataset/Y_test.txt", delimiter=",")

    # Include bias/intercept column into data
    X_train_original = np.column_stack((np.ones(len(X_train_original)),X_train_original))
    X_test_original = np.column_stack((np.ones(len(X_test_original)),X_test_original))
    
    X = torch.tensor(X_train_original, dtype=torch.float32)
    y = torch.tensor(y_train_original, dtype=torch.float32)
    X_test = torch.tensor(X_test_original, dtype=torch.float32)
    y_test = torch.tensor(y_test_original, dtype=torch.float32)


    train_data = torch.cat([X, y.unsqueeze(1)], dim=1)
    test_data = torch.cat([X_test, y_test.unsqueeze(1)], dim=1)


    train_data = train_data.to(DEVICE)
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

        history.append(
            {
                "epoch": epoch,
                "kl_model": float(model.kl()),
                "train_nll": float(train_nll),
                "train_loss": float(train_loss)
            }
        )

        print(
            f"[LRT] Epoch {epoch:03d} | "
            f"kl_model={model.kl():.4f} | "
            f"train_nll={train_loss:.4f} "
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

    from sklearn.datasets import load_breast_cancer
    # WBC variable names
    t = load_breast_cancer(as_frame=True)
    names = t["data"].columns

    feature_names = np.concatenate((["bias"], names))

    plotting.plot_local_explain_piecewise_linear_act(
        net=model,
        input_data=x_explain,
        median=True,
        sample=True,
        n_samples=64,
        n_classes=1,
        magnitude=True,
        include_potential_contribution=False,
        variable_names=feature_names,
        class_names=["Malignant"],
        include_prediction=True,
        include_bias=True,
        no_zero_contributions=True,
        save_path=str(RESULTS_DIR / "local_explanation_plot"),
        show=False,
    )
    
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


    contributions, predicted_classes = compute_global_explain_piecewise_linear_act(
        net=model,
        X=X_train_original,
        n_expl_per_sample=10,
        n_classes=1,
        pred_threshold=0.5,
    )

    plotting.plot_global_explain_piecewise_linear_act(
        contributions=contributions,
        predictions=predicted_classes,
        n_classes=1,
        variable_names=feature_names,
        class_names=["Benign", "Malignant"],
        save_path=str(RESULTS_DIR / "global_explain"),
        show=True,
    )

    print(f"[LRT] Saved all results to: {RESULTS_DIR.resolve()}")


if __name__ == "__main__":
    main()