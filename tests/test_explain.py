import numpy as np
import torch

from LBBNN import (
    BayesianNetworkLRT,
    local_explain_piecewise_linear_act,
    what_if_explanations,
)


def _model(n_classes: int = 1) -> BayesianNetworkLRT:
    torch.manual_seed(0)
    return BayesianNetworkLRT(
        dim=4, p=5, hidden_layers=2,
        classification=True, n_classes=n_classes,
        act_func=torch.relu,
    )


def test_local_explain_returns_expected_shapes():
    model = _model(n_classes=1)
    x = torch.tensor([1.0, 0.5, -0.3, 0.7, 1.2])
    expl, preds, p = local_explain_piecewise_linear_act(
        model, x, n_samples=3, n_classes=1, magnitude=True,
    )
    assert expl.shape == (3, 5, 1)
    assert preds.shape == (3, 1)
    assert p == 5
    assert np.all(np.isfinite(expl))


def test_local_explain_zero_inputs_get_zero_contribution_when_no_potential():
    model = _model(n_classes=1)
    # Two features set to exactly 0.0 → those columns should be zeroed.
    x = torch.tensor([1.0, 0.0, -0.3, 0.0, 1.2])
    expl, _, _ = local_explain_piecewise_linear_act(
        model, x, n_samples=2, n_classes=1,
        magnitude=True, include_potential_contribution=False,
    )
    # Columns 1 and 3 correspond to the zero inputs.
    assert np.allclose(expl[:, 1, :], 0.0)
    assert np.allclose(expl[:, 3, :], 0.0)
    # Other columns should generally be non-zero (with high probability).
    assert not np.allclose(expl[:, 0, :], 0.0) or not np.allclose(expl[:, 2, :], 0.0)


def test_local_explain_magnitude_vs_input_weighted_differ():
    model = _model(n_classes=1)
    # Force all gates open and put the model in eval+deterministic mode, so two
    # consecutive calls produce identical gradients and we can analytically
    # verify the input-weighting relationship.
    with torch.no_grad():
        for layer in model.linears:
            layer.lambdal.fill_(2.0)  # sigmoid(2) ≈ 0.88 → all gates active
    model.eval()

    x = torch.tensor([1.0, 0.5, -0.3, 0.7, 1.2])
    expl_mag, _, _ = local_explain_piecewise_linear_act(
        model, x, n_samples=1, n_classes=1, magnitude=True, sample=False,
    )
    expl_weighted, _, _ = local_explain_piecewise_linear_act(
        model, x, n_samples=1, n_classes=1, magnitude=False, sample=False,
    )
    assert not np.allclose(expl_mag, expl_weighted)
    x_np = x.numpy()[None, :, None]
    assert np.allclose(expl_weighted, expl_mag * x_np)


def test_what_if_explanations_output_shapes_and_predictions():
    model = _model(n_classes=1)
    data = torch.tensor([1.0, 0.5, -0.3, 0.7, 1.2])
    observed_space, contributions, predictions = what_if_explanations(
        model, data, feature_index=2, minimum=-1.0, maximum=1.0,
        n_samples=8, n_expl_per_sample=3,
    )
    assert observed_space.shape == (8,)
    assert contributions.shape == (8, 5, 3)
    assert predictions.shape == (8, 3)
    # # `predictions` rows are 0/1 floats (mean(prob) > 0.5).
    # assert set(np.unique(predictions).tolist()).issubset({0.0, 1.0}) # Bad test as it would only be valid for binary, not regression or multiclass
    # TODO: add multiclass tests
