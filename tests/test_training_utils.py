import torch
from LBBNN import InputSkipLRTNetwork, train_epoch, validate

def test_train_epoch_and_validate_run():
    torch.manual_seed(0)
    model = InputSkipLRTNetwork(dim=4, p=5, hidden_layers=2, classification=True, n_classes=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.randn(16, 5)
    y = (torch.rand(16) > 0.5).float()
    train_data = torch.cat([x, y.unsqueeze(1)], dim=1)
    nll, loss = train_epoch(net=model, train_data=train_data, optimizer=optimizer, batch_size=8, num_batches=2, p=5, device=torch.device('cpu'), nr_weights=sum(p.numel() for p in model.parameters()), task="binary", verbose=False)
    assert isinstance(nll, float)
    assert isinstance(loss, float)
    val_nll, val_loss, metric = validate(net=model, val_data=train_data, device=torch.device('cpu'), task="binary", verbose=False)
    assert isinstance(val_nll, float)
    assert isinstance(val_loss, float)
    assert 0.0 <= metric <= 1.0
