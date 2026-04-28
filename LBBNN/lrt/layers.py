from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

import math


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
        lower_init_alpha: float = 0.30,
        upper_init_alpha: float = 0.49,
        a_prior: float = 0.1,
        sigma_prior: float = 2.5,
        mu_prior: float = 0.0,
        weight_mu_init_range: tuple[float, float] = (-1.2, 1.2),
        weight_rho_init_mean: float = -9.0,
        n_skip_features: int | None = None,
    ) -> None:
        """Initialize the Bayesian linear layer.

        Args:
            in_features: Number of input features.
            out_features: Number of output features.
            lower_init_alpha: Lower bound for uniform initialization of
                inclusion probability.
            upper_init_alpha: Upper bound for uniform initialization of
                inclusion probability.
            a_prior: Prior inclusion probability.
            sigma_prior: Prior standard deviation for the weights.
            mu_prior: Prior mean for the weights.
            weight_mu_init_range: Lower and upper bounds for uniform
                initialization of the weight means.
            weight_rho_init_mean: Mean of the Gaussian used to initialize
                the rho-parameter (softplus-mapped to weight std).
            n_skip_features: Number of skip-input features (the trailing
                columns of the weight matrix) forced to start with high
                inclusion probability.

        Returns:
            None.
        """
        super().__init__()

        lower_init_lambda = math.log(lower_init_alpha / (1-lower_init_alpha))
        upper_init_lambda = math.log(upper_init_alpha / (1-upper_init_alpha))

        # Variational parameters for the weights.
        self.weight_mu = nn.Parameter(
            torch.empty(out_features, in_features).uniform_(*weight_mu_init_range)
        )
        self.weight_rho = nn.Parameter(
            weight_rho_init_mean + 0.1 * torch.randn(out_features, in_features)
        )

        # Inclusion probabilities.
        init_lambda = torch.empty(out_features, in_features).uniform_(
            lower_init_lambda,
            upper_init_lambda,
        )
        if n_skip_features is not None:
            init_lambda[:, -n_skip_features:] = 5.0

        self.lambdal = nn.Parameter(init_lambda)

        # Prior parameters
        self.register_buffer("mu_prior", torch.full((out_features, in_features), mu_prior))
        self.register_buffer("sigma_prior", torch.full((out_features, in_features), sigma_prior))
        self.register_buffer("alpha_prior", torch.full((out_features, in_features), a_prior))

        # KL divergence term updated during forward passes
        self.register_buffer("kl", torch.tensor(0.0))

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

        alpha = torch.sigmoid(self.lambdal)
        alpha_prior = self.alpha_prior.clone()

        if post_train:
            alpha = (alpha.detach() > 0.5).float()
            alpha_prior[alpha.detach() < 0.5] = 0.0

        weight_sigma = torch.log1p(torch.exp(self.weight_rho))

        if ensemble or self.training:
            expected_weight = self.weight_mu * alpha
            var_weight = alpha * (
                weight_sigma**2
                + (1.0 - alpha) * self.weight_mu**2
            )

            expected_bias = input @ expected_weight.T
            var_bias = input.pow(2) @ var_weight.T
            noise = torch.randn_like(var_bias)

            activations = expected_bias + torch.sqrt(
                torch.clamp(var_bias, min=0.0) + eps_num
            ) * noise
        else:
            weights = (
                torch.normal(self.weight_mu, weight_sigma)
                if sample
                else self.weight_mu
            )
            gates = (alpha.detach() > 0.5).float()
            activations = input @ (weights * gates).T

            if calculate_log_probs:
                alpha = gates

        if self.training or calculate_log_probs:
            kl_weight = (
                alpha
                * (
                    torch.log(
                        (self.sigma_prior / (weight_sigma + eps_num))
                        + eps_num
                    )
                    - 0.5
                    + torch.log((alpha / (alpha_prior + eps_num)) + eps_num)
                    + (
                        weight_sigma**2
                        + (self.weight_mu - self.mu_prior) ** 2
                    )
                    / (2.0 * self.sigma_prior**2 + eps_num)
                )
                + (1.0 - alpha)
                * torch.log(
                    ((1.0 - alpha) / (1.0 - alpha_prior + eps_num))
                    + eps_num
                )
            ).sum()
            self.kl = kl_weight
        else:
            self.kl = torch.tensor(0.0, device=input.device)

        return activations
