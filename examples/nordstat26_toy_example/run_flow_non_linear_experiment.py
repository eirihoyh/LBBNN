import math
import sys
from pathlib import Path

# included such that _common.py from examples/ is available regardless of where the script is ran from
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

from LBBNN import (
    BayesianNetworkFlow,
    get_data,
    train_epoch,
    validate,
    get_alphas,
    clean_alpha,
    get_active_weights,
    weight_matrices_numpy,
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
    regression_metrics,
    split_dataset,
)


# ============================================================
# Configuration
# ============================================================
SEED = 42
RESULTS_DIR = Path("results/flow_non_linear_run")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

N_SAMPLES = 20000
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15

CLASSIFICATION = True

DIM = 20
HIDDEN_LAYERS = 4
NUM_TRANSFORMS = 2
LR = 5e-2
EPOCHS = 1000
BATCH_SIZE = 1024
THRESHOLD = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


task = "binary" if CLASSIFICATION else "regression"

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
        beta=(-5.0,1.5,-1.5,1.0,1.0,1.0,1.0),
        classification=CLASSIFICATION,
        non_lin=True,
        squared_terms=True,
        x_spread=2,
        seed=SEED,
    )
    X = torch.tensor(X_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.float32)

    (X_train, y_train, train_data), (X_val, y_val, val_data), (X_test, y_test, test_data) = split_dataset(
        X, y, train_frac=TRAIN_FRAC, val_frac=VAL_FRAC
    )

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
    model = BayesianNetworkFlow(
        dim=DIM,
        p=p,
        hidden_layers=HIDDEN_LAYERS,
        num_transforms=NUM_TRANSFORMS,
        classification=CLASSIFICATION,
        n_classes=1,
        act_func=torch.relu,
        lower_init_alpha=0.15,
        upper_init_alpha=0.25,
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
            task=task,
            verbose=False,
        )

        val_nll, val_loss, val_metric = validate(
            net=model,
            val_data=val_data,
            device=DEVICE,
            task=task,
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
            f"[FLOW] Epoch {epoch:04d} | "
            f"kl_model={model.kl():.4f} | "
            f"train_nll={train_loss:.4f} | "
            f"val_nll={val_nll:.4f} | "
            f"val_acc={val_metric:.4f}"
        )

    # --------------------------------------------------------
    # 4. Final test evaluation
    # --------------------------------------------------------
    model.eval()
    if CLASSIFICATION:
        with torch.no_grad():
            test_prob = model(X_test, ensemble=False).view(-1)
            test_pred = (test_prob >= THRESHOLD).float()
            test_acc = binary_accuracy(test_prob, y_test, threshold=THRESHOLD)

        print(f"[FLOW] Test accuracy: {test_acc:.4f}")
    else:
        model.eval()
        with torch.no_grad():
            test_pred = model(X_test, ensemble=False).view(-1)
            test_r2, test_correlation, test_mse = regression_metrics(test_pred, y_test)

        print(f"[FLOW] Test r2: {test_r2:.4f} | Test corr: {test_correlation:.4f} | Test mse: {test_mse:.4f} | ")

    # --------------------------------------------------------
    # 5. Save model + history + predictions
    # --------------------------------------------------------
    torch.save(model.state_dict(), RESULTS_DIR / "flow_model.pt")
    save_history_csv(history, RESULTS_DIR / "history.csv")

    if CLASSIFICATION:
        save_predictions_csv(
            y_true=y_test.detach().cpu().numpy(),
            y_prob=test_prob.detach().cpu().numpy(),
            y_pred=test_pred.detach().cpu().numpy(),
            path=RESULTS_DIR / "test_predictions.csv",
        )
        summary = {
            "model_type": "BayesianNetworkFlow",
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
    else:
        save_predictions_csv(
            y_true=y_test.detach().cpu().numpy(),
            y_prob=test_pred.detach().cpu().numpy(),
            y_pred=test_pred.detach().cpu().numpy(),
            path=RESULTS_DIR / "test_predictions.csv",
        )
        summary = {
            "model_type": "BayesianNetworkFlow",
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
            "test_r2": float(test_r2),
            "test_correlation": float(test_correlation),
            "test_mse": float(test_mse),
            "final_kl": float(model.kl().detach().cpu()),
        }

    
    save_json(summary, RESULTS_DIR / "summary.json")

    # --------------------------------------------------------
    # 6. Global network structure
    # --------------------------------------------------------
    structure_metrics = plotting.get_metrics(model, threshold=THRESHOLD)
    save_json(structure_metrics, RESULTS_DIR / "global_structure_metrics.json")

    plotting.save_metrics(model, threshold=THRESHOLD, path=str(RESULTS_DIR / "network_metrics"))

    alpha_list = get_alphas(model)
    alpha_clean = clean_alpha(model, threshold=THRESHOLD)
    weight_list = weight_matrices_numpy(model, flow=True)
    all_connections = get_active_weights(alpha_clean)
    active_connections = []
    for layer_idx, coords in enumerate(all_connections):
        layer_connections = coords.detach().cpu().tolist()
        active_connections.append(
            {
                "layer_index": layer_idx,
                "connections": layer_connections,
            }
        )
    save_json(active_connections, RESULTS_DIR / "active_connections.json")
    _ = plotting.build_path_graph_table(alpha_list, weight_list, all_connections, save_path=str(RESULTS_DIR / "connections_in_active_paths"))
    # Optional path graph if graphviz is installed
    try:
        plotting.run_path_graph(
            model,
            threshold=THRESHOLD,
            save_path=str(RESULTS_DIR / "path_graph"),
            show=False,
            show_edge_labels=False
        )
        print("[FLOW] Saved path graph.")
    except Exception as exc:
        print(f"[FLOW] Path graph was skipped ({exc}).")

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
        feature_names=[f"x{i}" for i in range(len(x_explain))],
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
        task=task,
        feature_index=1,
        n_samples=50,
        n_expl_per_sample=100)
    
    plotting.plot_what_if_explanations(
        observed_space,
        contributions_feature_1,
        predictions_feature_1,
        x_explain.detach().cpu().numpy(),
        task=task,
        feature_names=feature_names,
        feature_in_focus=1,
        save_path=str(RESULTS_DIR / "what-if_explanation_feature_1")
    )

    observed_space, contributions_feature_2, predictions_feature_2 = what_if_explanations(
        model, 
        x_explain.detach(), 
        minimum=minimum,
        maximum=maximum,
        task=task,
        feature_index=2,
        n_samples=50,
        n_expl_per_sample=100)
    
    plotting.plot_what_if_explanations(
        observed_space,
        contributions_feature_2,
        predictions_feature_2,
        x_explain.detach().cpu().numpy(),
        task=task,
        feature_names=feature_names,
        feature_in_focus=2,
        save_path=str(RESULTS_DIR / "what-if_explanation_feature_2")
    )
    
    
    contributions, predicted_classes = compute_global_explain_piecewise_linear_act(
        net=model,
        X=X_train.detach().cpu().numpy(),
        task=task,
        n_expl_per_sample=10,
        n_classes=1,
        pred_threshold=0.5,
    )

    plotting.plot_global_explain_piecewise_linear_act(
        contributions=contributions,
        predictions=predicted_classes,
        n_classes=1,
        task=task,
        feature_names=feature_names,
        covariate_indices=[1,2],
        class_names=["Class 0", "Class 1"],
        save_path=str(RESULTS_DIR / "global_explain"),
        show=True,
    )

    print(f"[FLOW] Saved all results to: {RESULTS_DIR.resolve()}")


if __name__ == "__main__":
    main()