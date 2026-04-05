from LBBNN import InputSkipLRTNetwork, clean_alpha, get_alphas, network_density_reduction, weight_matrices

def test_weight_and_alpha_lists_have_expected_length():
    model = InputSkipLRTNetwork(dim=4, p=5, hidden_layers=3, classification=True)
    weights = weight_matrices(model)
    alphas = get_alphas(model)
    assert len(weights) == 4
    assert len(alphas) == 4

def test_clean_alpha_and_density_outputs():
    model = InputSkipLRTNetwork(dim=4, p=5, hidden_layers=2, classification=True)
    cleaned = clean_alpha(model, threshold=0.5)
    density, used, total = network_density_reduction(cleaned)
    assert len(cleaned) == 3
    assert 0.0 <= density <= 1.0
    assert used <= total
