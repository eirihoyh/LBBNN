from __future__ import annotations
import torch
import torch.nn as nn


def get_default_device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class BayesianLinear(nn.Module):
    def __init__(
            self, 
            in_features: int, 
            out_features: int, 
            lower_init_lambda: float = -10, 
            upper_init_lambda: float = -7, 
            a_prior: float = 0.1, 
            std_prior: float = 2.5, p: int | None = None) -> None:
        super().__init__()
        
        # VI parameters
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features).uniform_(-1.2, 1.2))
        self.weight_rho = nn.Parameter(-9 + 0.1 * torch.randn(out_features, in_features))
        self.weight_sigma = torch.empty_like(self.weight_rho)
        
        # Priors
        self.mu_prior = torch.zeros(out_features, in_features, device=get_default_device())
        self.sigma_prior = torch.full((out_features, in_features), std_prior, device=get_default_device())
        
        # Inclusion probability
        init_lambda = torch.empty(out_features, in_features).uniform_(lower_init_lambda, upper_init_lambda)
        if p is not None:
            init_lambda[:, -p:] = 5.0
        self.lambdal = nn.Parameter(init_lambda)
        self.alpha = torch.empty_like(self.lambdal)
        self.alpha_prior = torch.zeros(out_features, in_features, device=get_default_device()) + a_prior

        # KL
        self.kl = torch.tensor(0.0, device=get_default_device())

    def forward(
            self, 
            input: torch.Tensor, 
            ensemble: bool = True, 
            sample: bool = False, 
            calculate_log_probs: bool = False, 
            post_train: bool = False) -> torch.Tensor:
        
        eps_num = torch.tensor(1e-45, device=input.device, dtype=input.dtype)
        self.alpha = torch.sigmoid(self.lambdal)
        alpha_prior = self.alpha_prior.clone()
        if post_train:
            self.alpha = (self.alpha.detach() > 0.5).float()
            alpha_prior[self.alpha.detach() < 0.5] = 0.0
        self.weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        if ensemble or self.training:
            e_w = self.weight_mu * self.alpha
            var_w = self.alpha * (self.weight_sigma**2 + (1 - self.alpha) * self.weight_mu**2)
            e_b = input @ e_w.T
            var_b = input.pow(2) @ var_w.T
            eps = torch.randn_like(var_b)
            activations = e_b + torch.sqrt(torch.clamp(var_b, min=0.0) + eps_num) * eps
        else:
            w = torch.normal(self.weight_mu, self.weight_sigma) if sample else self.weight_mu
            g = (self.alpha.detach() > 0.5).float()
            activations = input @ (w * g).T
            if calculate_log_probs:
                self.alpha = g
        
        if self.training or calculate_log_probs:
            kl_weight = (
                self.alpha * (
                    torch.log((self.sigma_prior / (self.weight_sigma + eps_num)) + eps_num)
                    - 0.5
                    + torch.log((self.alpha / (alpha_prior + eps_num)) + eps_num)
                    + (self.weight_sigma**2 + (self.weight_mu - self.mu_prior) ** 2) / (2 * self.sigma_prior**2 + eps_num)
                )
                + (1 - self.alpha) * torch.log(((1 - self.alpha) / (1 - alpha_prior + eps_num)) + eps_num)
            ).sum()
            self.kl = kl_weight
        else:
            self.kl = torch.tensor(0.0, device=input.device)
        return activations
