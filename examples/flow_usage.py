import torch
from LBBNN import InputSkipFlowNetwork, create_data_unif, get_data

# y, X = create_data_unif(n=64, classification=True, seed=1)
_, y, X = get_data(n=64, classification=True)
X = torch.tensor(X, dtype=torch.float32)
model = InputSkipFlowNetwork(dim=8, p=X.shape[1], hidden_layers=2, num_transforms=2, classification=True, n_classes=1)
with torch.no_grad():
    preds = model(X, ensemble=False)
print(preds.shape)
print(model.kl())
