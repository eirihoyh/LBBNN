import torch
from LBBNN import InputSkipLRTNetwork, create_data_unif

y, X = create_data_unif(n=64, classification=True, seed=1)
X = torch.tensor(X, dtype=torch.float32)
model = InputSkipLRTNetwork(dim=8, p=X.shape[1], hidden_layers=2, classification=True, n_classes=1)
with torch.no_grad():
    preds = model(X, ensemble=False)
print(preds.shape)
