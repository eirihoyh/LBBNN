from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


def get_default_device() -> torch.device:
    """Return the default device used for model tensors.

    Returns:
        The CUDA device if available, otherwise the CPU device.
    """
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class BayesianLinearLRT(nn.Module):
    """Bayesian linear layer with local reparameterization and binary gates."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        lower_init_lambda: float = -10.0,
        upper_init_lambda: float = -7.0,
        a_prior: float = 0.1,
        std_prior: float = 2.5,
        p: int | None = None,
    ) -> None:
        """Initialize the Bayesian linear layer.

        Args:
            in_features: Number of input features.
            out_features: Number of output features.
            lower_init_lambda: Lower bound for uniform initialization of
                inclusion logits.
            upper_init_lambda: Upper bound for uniform initialization of
                inclusion logits.
            a_prior: Prior inclusion probability.
            std_prior: Prior standard deviation for the weights.
            p: Number of skip-input features forced to start with high
                inclusion probability.

        Returns:
            None.
        """
        super().__init__()

        device = get_default_device()

        # Variational parameters for the weights.
        self.weight_mu = nn.Parameter(
            torch.empty(out_features, in_features).uniform_(-1.2, 1.2)
        )
        self.weight_rho = nn.Parameter(
            -9.0 + 0.1 * torch.randn(out_features, in_features)
        )
        self.weight_sigma = torch.empty(out_features, in_features, device=device)

        # Prior parameters.
        self.mu_prior = torch.zeros(out_features, in_features, device=device)
        self.sigma_prior = torch.full(
            (out_features, in_features),
            std_prior,
            device=device,
        )

        # Inclusion probabilities.
        init_lambda = torch.empty(out_features, in_features).uniform_(
            lower_init_lambda,
            upper_init_lambda,
        )
        if p is not None:
            init_lambda[:, -p:] = 5.0

        self.lambdal = nn.Parameter(init_lambda)
        self.alpha = torch.empty(out_features, in_features, device=device)
        self.alpha_prior = torch.full(
            (out_features, in_features),
            a_prior,
            device=device,
        )

        # KL divergence term updated during forward passes.
        self.kl = torch.tensor(0.0, device=device)

    def forward(
        self,
        input: Tensor,
        ensemble: bool = True,
        sample: bool = False,
        calculate_log_probs: bool = False,
        post_train: bool = False,
    ) -> Tensor:
        """Compute the forward pass through the Bayesian linear layer.
        
        Args:
            input: Input tensor of shape ``(batch_size, in_features)``.
            ensemble: Whether to use the local reparameterization form.
            sample: Whether to sample weights in deterministic mode.
            calculate_log_probs: Whether to update the KL divergence term.
            post_train: Whether to use thresholded inclusion probabilities.

        Returns:
            Output activations of shape ``(batch_size, out_features)``.
        """
        eps_num = torch.tensor(1e-45, device=input.device, dtype=input.dtype)

        self.alpha = torch.sigmoid(self.lambdal)
        alpha_prior = self.alpha_prior.clone()

        if post_train:
            self.alpha = (self.alpha.detach() > 0.5).float()
            alpha_prior[self.alpha.detach() < 0.5] = 0.0

        self.weight_sigma = torch.log1p(torch.exp(self.weight_rho))

        if ensemble or self.training:
            expected_weight = self.weight_mu * self.alpha
            var_weight = self.alpha * (
                self.weight_sigma**2
                + (1.0 - self.alpha) * self.weight_mu**2
            )

            expected_bias = input @ expected_weight.T
            var_bias = input.pow(2) @ var_weight.T
            noise = torch.randn_like(var_bias)

            activations = expected_bias + torch.sqrt(
                torch.clamp(var_bias, min=0.0) + eps_num
            ) * noise
        else:
            weights = (
                torch.normal(self.weight_mu, self.weight_sigma)
                if sample
                else self.weight_mu
            )
            gates = (self.alpha.detach() > 0.5).float()
            activations = input @ (weights * gates).T

            if calculate_log_probs:
                self.alpha = gates

        if self.training or calculate_log_probs:
            kl_weight = (
                self.alpha
                * (
                    torch.log(
                        (self.sigma_prior / (self.weight_sigma + eps_num))
                        + eps_num
                    )
                    - 0.5
                    + torch.log((self.alpha / (alpha_prior + eps_num)) + eps_num)
                    + (
                        self.weight_sigma**2
                        + (self.weight_mu - self.mu_prior) ** 2
                    )
                    / (2.0 * self.sigma_prior**2 + eps_num)
                )
                + (1.0 - self.alpha)
                * torch.log(
                    ((1.0 - self.alpha) / (1.0 - alpha_prior + eps_num))
                    + eps_num
                )
            ).sum()
            self.kl = kl_weight
        else:
            self.kl = torch.tensor(0.0, device=input.device)

        return activations
