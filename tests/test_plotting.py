from pathlib import Path
import torch
from LBBNN import BayesianNetworkLRT
from LBBNN import plotting
from LBBNN.plotting import metrics as plotting_metrics

def test_get_metrics_returns_expected_keys():
    model = BayesianNetworkLRT(dim=4, p=5, hidden_layers=2, classification=True, n_classes=1)
    metrics = plotting.get_metrics(model)
    expected = {"layer_names", "tot_weights", "used_weights_median", "density_median", "expected_nr_weights_full", "density_full", "avg_path_length", "include_inputs", "input_inclusion_prob", "width_prob"}
    assert expected.issubset(metrics.keys())

def test_metrics_submodule_alias_matches_top_level():
    model = BayesianNetworkLRT(dim=4, p=5, hidden_layers=2, classification=True, n_classes=1)
    assert plotting.get_metrics(model).keys() == plotting_metrics.get_metrics(model).keys()

def test_piecewise_plot_saves_png(tmp_path: Path):
    model = BayesianNetworkLRT(dim=4, p=5, hidden_layers=2, classification=True, n_classes=1)
    x = torch.randn(5)
    out_prefix = tmp_path / "piecewise"
    saved = plotting.plot_local_explain_piecewise_linear_act(model, x, n_samples=2, n_classes=1, save_path=str(out_prefix), show=False)
    assert saved
    assert Path(saved[0]).exists()
