"""Tests for ``LBBNN.inspection`` helpers."""
import numpy as np
import torch

from LBBNN import (
    BayesianNetworkLRT,
    average_path_length,
    clean_alpha,
    clean_alpha_class,
    expected_number_of_weights,
    get_alphas,
    include_input_from_layer,
    input_inclusion_prob,
    network_density_reduction,
    prob_width,
    weight_matrices,
)


def _model(n_classes: int = 1, hidden_layers: int = 2) -> BayesianNetworkLRT:
    torch.manual_seed(0)
    return BayesianNetworkLRT(
        dim=4, p=5, hidden_layers=hidden_layers,
        classification=True, n_classes=n_classes,
    )


# ---------------- weight / alpha lists ----------------

def test_weight_and_alpha_lists_have_expected_length():
    model = _model(hidden_layers=3)
    assert len(weight_matrices(model)) == 4
    assert len(get_alphas(model)) == 4


def test_clean_alpha_and_density_outputs():
    model = _model()
    cleaned = clean_alpha(model, threshold=0.5)
    density, used, total = network_density_reduction(cleaned)
    assert len(cleaned) == 3
    assert 0.0 <= density <= 1.0
    assert used <= total


# ---------------- input_inclusion_prob / prob_width ----------------

def test_input_inclusion_prob_returns_finite_floats():
    model = _model()
    probs = input_inclusion_prob(model)
    assert isinstance(probs, dict)
    assert all(isinstance(v, float) and np.isfinite(v) for v in probs.values())


def test_prob_width_capped_at_one_per_input():
    model = _model()
    widths = prob_width(model, p=5)
    assert set(widths.keys()) == set(range(5))
    for v in widths.values():
        assert 0.0 <= v <= 1.0


# ---------------- path / weight summaries ----------------

def test_average_path_length_non_negative():
    model = _model()
    cleaned = clean_alpha(model, threshold=0.5)
    avg, all_lengths = average_path_length(cleaned)
    assert avg >= 0.0
    assert isinstance(all_lengths, np.ndarray)


def test_expected_number_of_weights_in_range():
    model = _model()
    expected = expected_number_of_weights(model)
    total_weights = sum(layer.lambdal.numel() for layer in model.linears)
    assert 0.0 <= expected <= total_weights


# ---------------- class-conditional cleaning ----------------

def test_clean_alpha_class_zeros_other_classes():
    model = _model(n_classes=3)
    cleaned = clean_alpha_class(model, threshold=0.5, class_in_focus=1)
    out = cleaned[-1]
    # Rows 0 and 2 are zeroed before propagation; clean_alpha only multiplies
    # them, so they must remain entirely zero in the output layer.
    assert torch.all(out[0] == 0)
    assert torch.all(out[2] == 0)


def test_include_input_from_layer_returns_per_layer_bool():
    model = _model()
    cleaned = clean_alpha(model, threshold=0.5)
    flags = include_input_from_layer(cleaned)
    assert len(flags) == len(cleaned)
    for arr in flags:
        assert arr.dtype == bool
        assert arr.shape == (5,)
