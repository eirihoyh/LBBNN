from __future__ import annotations
import numpy as np
import torch
from .inspection import clean_alpha, expected_number_of_weights, network_density_reduction

def _r2_score(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    y_true = y_true.view(-1).float(); y_pred = y_pred.view(-1).float()
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    return 0.0 if float(ss_tot) == 0.0 else float((1 - ss_res / ss_tot).detach().cpu())

def _rmse(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean((y_true.view(-1).float() - y_pred.view(-1).float()) ** 2)).detach().cpu())

def train_epoch(
        net, 
        train_data, 
        optimizer, 
        batch_size, 
        num_batches, 
        p, 
        device, 
        nr_weights, 
        multiclass=False, 
        verbose=False, 
        post_train=False):
    
    net.train()
    inds = np.arange(0, len(train_data), 1)
    train_data = train_data[np.random.choice(inds, size=len(train_data), replace=False)]
    old_batch, last_nll, last_loss = 0, None, None
    for batch in range(int(np.ceil(train_data.shape[0] / batch_size))):
        batch += 1
        _x = train_data[old_batch: batch_size * batch, 0:p]
        _y = train_data[old_batch: batch_size * batch, -1]
        old_batch = batch_size * batch
        data = _x.to(device)
        target = _y.to(device)
        target = target.long() if multiclass else target.float()
        optimizer.zero_grad()
        outputs = net(data, sample=True, post_train=post_train)
        
        nll = net.loss(outputs, target.view(-1) if multiclass else target.unsqueeze(1).float())
        kl_part = net.kl() / max(num_batches, 1)
        loss = nll + kl_part
        loss.backward()
        optimizer.step()
        last_nll, last_loss = nll.item(), loss.item()
        
        if verbose: print('loss', last_loss); print('nll', last_nll); print('density', expected_number_of_weights(net) / nr_weights)
    
    return last_nll, last_loss

def validate(
        net, 
        val_data, 
        device, 
        multiclass=False, 
        reg=False, 
        verbose=False, 
        post_train=False):
    net.eval()
    with torch.no_grad():
        _x = val_data[:, :-1]
        _y = val_data[:, -1]
        data = _x.to(device)
        target = _y.to(device)
        outputs = net(data, ensemble=False, calculate_log_probs=True, post_train=post_train)
        if multiclass:
            nll = net.loss(outputs, target.long().view(-1))
            metric_value = float((outputs.argmax(dim=1) == target.long().view(-1)).float().mean().cpu())
        else:
            target = target.unsqueeze(1).float()
            nll = net.loss(outputs, target)
            metric_value = _r2_score(
                outputs[:, 0], target[:, 0]) if reg else float((outputs.round().squeeze().cpu() == target[:, 0].cpu()).float().mean())
        loss = nll + net.kl()
        alpha_clean = clean_alpha(net, threshold=0.5)
        density_median, used_weights_median, _ = network_density_reduction(alpha_clean)
        if verbose: print(f'val_loss: {loss.item():.4f}, val_nll: {nll.item():.4f}, val_metric: {metric_value:.4f}, used_weights_median: {used_weights_median}')
        return nll.item(), loss.item(), metric_value

def test_ensemble(
        net, 
        test_data, 
        device, 
        samples, 
        classes=1, 
        reg=True, 
        verbose=False, 
        post_train=False, 
        multiclass=False):
    net.eval()
    density, used_weights = [], []
    ensemble_metrics, ensemble_metrics_median = [], []
    ensemble_r2, ensemble_r2_median = [], []
    with torch.no_grad():
        _x = test_data[:, :-1]
        _y = test_data[:, -1]
        data = _x.to(device)
        target = _y.to(device)
        outputs = torch.zeros(samples, _x.shape[0], classes, device=device)
        outputs_median = torch.zeros(samples, _x.shape[0], classes, device=device)
        for i in range(samples):
            outputs[i] = net.forward(data, sample=True, ensemble=True)
            outputs_median[i] = net.forward(data, ensemble=False)
        outputs_mean = outputs.mean(0)
        outputs_median_mean = outputs_median.mean(0)
        alpha_clean = clean_alpha(net, threshold=0.5)
        density_median, used_weights_median, _ = network_density_reduction(alpha_clean)
        density.append(density_median)
        used_weights.append(used_weights_median)
        if reg:
            ensemble_metrics.append(_rmse(outputs_mean[:, 0], target))
            ensemble_metrics_median.append(_rmse(outputs_median_mean[:, 0], target))
            ensemble_r2.append(_r2_score(outputs_mean[:, 0], target))
            ensemble_r2_median.append(_r2_score(outputs_median_mean[:, 0], target))
        else:
            if multiclass:
                ensemble_metrics.append(float((outputs_mean.argmax(dim=1) == target.long().view(-1)).float().mean().cpu()))
                ensemble_metrics_median.append(float((outputs_median_mean.argmax(dim=1) == target.long().view(-1)).float().mean().cpu()))
            else:
                ensemble_metrics.append(float((outputs_mean[:, 0].round().cpu() == target.cpu()).float().mean()))
                ensemble_metrics_median.append(float((outputs_median_mean[:, 0].round().cpu() == target.cpu()).float().mean()))
    metr = [float(np.mean(ensemble_metrics)), float(np.mean(density))]
    metr_median = [float(np.mean(ensemble_metrics_median)), float(np.mean(used_weights))]
    if reg: metr.append(float(np.mean(ensemble_r2))); metr_median.append(float(np.mean(ensemble_r2_median)))
    return metr, metr_median
