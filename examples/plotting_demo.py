import torch
from LBBNN import BayesianNetworkLRT, plotting

model = BayesianNetworkLRT(dim=4, p=5, hidden_layers=2, classification=True, n_classes=1)
x = torch.randn(5)
saved = plotting.plot_local_explain_piecewise_linear_act(
    model, 
    input_data=x, 
    n_samples=10, 
    n_classes=1, 
    save_path="figures/piecewise_demo", show=False)
print(saved)
