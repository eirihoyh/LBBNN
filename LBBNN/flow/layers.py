from __future__ import annotations
import math
import torch
import torch.nn as nn
from .transforms import PropagateFlow

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
Z_FLOW_TYPE = 'IAF'
R_FLOW_TYPE = 'IAF'

def _log_pi(device, dtype):
    return torch.log(torch.tensor(math.pi, device=device, dtype=dtype))

class BayesianLinearFlow(nn.Module):
    def __init__(
            self, 
            in_features: int, 
            out_features: int, 
            num_transforms: int, 
            lower_init_lambda: float = 2.0, 
            upper_init_lambda: float = 10.0, 
            a_prior: float = 0.1) -> None:
        super().__init__()
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features).uniform_(-0.01, 0.01))
        self.weight_rho = nn.Parameter(-9 + 0.1 * torch.randn(out_features, in_features))
        self.weight_sigma = torch.empty_like(self.weight_rho)
        self.mu_prior = torch.zeros(out_features, in_features, device=DEVICE)
        self.sigma_prior = torch.full((out_features, in_features), 20.0, device=DEVICE)
        self.lambdal = nn.Parameter(torch.empty(out_features, in_features).uniform_(lower_init_lambda, upper_init_lambda))
        self.alpha = torch.empty_like(self.lambdal)
        self.alpha_prior = torch.zeros(out_features, in_features, device=DEVICE) + a_prior
        self.q0_mean = nn.Parameter(torch.randn(in_features))
        self.q0_log_var = nn.Parameter(-9 + torch.randn(in_features))
        self.c1 = nn.Parameter(torch.randn(in_features))
        self.r0_b1 = nn.Parameter(torch.randn(in_features))
        self.r0_b2 = nn.Parameter(torch.randn(in_features))
        self.z_flow = PropagateFlow(Z_FLOW_TYPE, in_features, num_transforms)
        self.r_flow = PropagateFlow(R_FLOW_TYPE, in_features, num_transforms)
        self.z = None

    def sample_z(self):
        q0_std = self.q0_log_var.exp().sqrt()
        epsilon_z = torch.randn_like(q0_std)
        self.z = self.q0_mean + q0_std * epsilon_z
        zk, log_det_q = self.z_flow(self.z)
        return zk, log_det_q.squeeze()

    def kl_div(self):
        self.alpha = torch.sigmoid(self.lambdal)
        self.weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        zk, log_det_q = self.sample_z()
        eps = torch.tensor(1e-45, device=zk.device, dtype=zk.dtype)
        W_mean = zk * self.weight_mu * self.alpha
        W_var = self.alpha * (self.weight_sigma**2 + (1 - self.alpha) * self.weight_mu**2 * zk**2) + eps
        
        log_q0 = (
            -0.5 * _log_pi(
                self.q0_log_var.device, self.q0_log_var.dtype) 
            - 0.5 * self.q0_log_var 
            - 0.5 * ((self.z - self.q0_mean) ** 2 / (self.q0_log_var.exp() + eps))).sum()
        log_q = -log_det_q + log_q0
        
        act_mu = self.c1 @ W_mean.T
        act_var = self.c1**2 @ W_var.T
        act_inner = act_mu + act_var.sqrt() * torch.randn_like(act_var)
        
        act = nn.Hardtanh()(act_inner)
        mean_r = self.r0_b1.outer(act).mean(-1)
        log_var_r = self.r0_b2.outer(act).mean(-1)
        
        zb, log_det_r = self.r_flow(zk)
        log_rb = (
            -0.5 * _log_pi(
                log_var_r.device, log_var_r.dtype) 
            - 0.5 * log_var_r 
            - 0.5 * ((zb - mean_r) ** 2 / (log_var_r.exp() + eps))).sum()
        log_r = log_det_r + log_rb
        
        kl_weight = (
            self.alpha * (
                torch.log(
                    (self.sigma_prior / (self.weight_sigma + eps)) + eps) 
                - 0.5 
                + torch.log(
                    (self.alpha / (self.alpha_prior + eps)) + eps) 
                + (
                    self.weight_sigma**2 + (self.weight_mu * zk - self.mu_prior) ** 2
                    ) / (2 * self.sigma_prior**2 + eps)
                ) 
                + (1 - self.alpha) * torch.log(
                    ((1 - self.alpha) / (1 - self.alpha_prior + eps)) + eps)
                ).sum()
        
        return kl_weight + log_q - log_r

    def forward(self, input: torch.Tensor, ensemble: bool = False, post_train: bool = False):
        self.alpha = torch.sigmoid(self.lambdal)
        if post_train:
            self.alpha = (self.alpha.detach() > 0.5).float()
            self.alpha_prior[self.alpha.detach() < 0.5] = 0.0
        self.weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        zk, _ = self.sample_z()
        if self.training or ensemble:
            e_w = self.weight_mu * self.alpha * zk
            var_w = self.alpha * (self.weight_sigma**2 + (1 - self.alpha) * self.weight_mu**2 * zk**2)
            e_b = input @ e_w.T
            var_b = input.pow(2) @ var_w.T
            eps = torch.randn(size=var_b.size(), device=input.device)
            activations = e_b + torch.sqrt(torch.clamp(var_b, min=0.0)) * eps
        else:
            w = torch.normal(self.weight_mu * zk, self.weight_sigma)
            g = (self.alpha.detach() > 0.5).float()
            activations = input @ (w * g).T
        return activations
