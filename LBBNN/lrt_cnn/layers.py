from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class BayesianConv2dLRT(nn.Module):
    """Bayesian 2-D convolutional layer with local reparameterization and binary gates."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        padding: int = 0,
        stride: int = 1,
        lower_init_alpha: float = 0.30,
        upper_init_alpha: float = 0.49,
        a_prior: float = 0.1,
        sigma_prior: float = 2.5,
        mu_prior: float = 0.0,
        weight_mu_init_range: tuple[float, float] = (-1.2, 1.2),
        weight_rho_init_mean: float = -9.0,
    ) -> None:
        """Initialize the Bayesian convolutional layer.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Kernel size. An integer is treated as a square kernel.
            padding: Zero-padding added on both spatial sides.
            stride: Convolution stride.
            lower_init_alpha: Lower bound for inclusion probability initialization.
            upper_init_alpha: Upper bound for inclusion probability initialization.
            a_prior: Prior inclusion probability.
            sigma_prior: Prior standard deviation for the weights.
            mu_prior: Prior mean for the weights.
            weight_mu_init_range: Uniform init range for weight means.
            weight_rho_init_mean: Mean of the Gaussian used to initialize
                the rho-parameter (softplus-mapped to weight std).

        Returns:
            None.
        """
        super().__init__()

        self.stride = stride
        self.padding = padding

        kernel: tuple[int, int] = (
            (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        )

        lower_init_lambda = math.log(lower_init_alpha / (1.0 - lower_init_alpha))
        upper_init_lambda = math.log(upper_init_alpha / (1.0 - upper_init_alpha))

        shape = (out_channels, in_channels, kernel[0], kernel[1])

        # Variational parameters for the weights.
        self.weight_mu = nn.Parameter(
            torch.empty(shape).uniform_(*weight_mu_init_range)
        )
        self.weight_rho = nn.Parameter(
            weight_rho_init_mean + 0.1 * torch.randn(shape)
        )

        # Inclusion probabilities.
        self.lambdal = nn.Parameter(
            torch.empty(shape).uniform_(lower_init_lambda, upper_init_lambda)
        )

        # Bias variational parameters.
        self.bias_mu = nn.Parameter(torch.empty(out_channels).uniform_(-0.2, 0.2))
        self.bias_rho = nn.Parameter(
            weight_rho_init_mean + 1.0 * torch.randn(out_channels)
        )

        # Prior buffers (automatically follow the module's device).
        self.register_buffer("mu_prior", torch.full(shape, mu_prior))
        self.register_buffer("sigma_prior", torch.full(shape, sigma_prior))
        self.register_buffer("alpha_prior", torch.full(shape, a_prior))
        self.register_buffer("bias_mu_prior", torch.zeros(out_channels))
        self.register_buffer("bias_sigma_prior", torch.full((out_channels,), sigma_prior))

        # KL divergence term updated during forward passes.
        self.register_buffer("kl", torch.tensor(0.0))

    def forward(
        self,
        input: Tensor,
        ensemble: bool = True,
        sample: bool = False,
        calculate_log_probs: bool = False,
        post_train: bool = False,
    ) -> Tensor:
        """Compute the forward pass through the Bayesian conv layer.

        Args:
            input: Input tensor of shape ``(B, C_in, H, W)``.
            ensemble: Whether to use the local reparameterization form.
            sample: Whether to sample weights in deterministic mode.
            calculate_log_probs: Whether to update the KL divergence term.
            post_train: Whether to use thresholded inclusion probabilities.

        Returns:
            Output activations of shape ``(B, C_out, H', W')``.
        """
        eps_num = torch.tensor(1e-45, device=input.device, dtype=input.dtype)

        alpha = torch.sigmoid(self.lambdal)
        alpha_prior = self.alpha_prior.clone()

        if post_train:
            alpha = (alpha.detach() > 0.5).float()
            alpha_prior[alpha.detach() < 0.5] = 0.0

        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        bias_sigma = torch.log1p(torch.exp(self.bias_rho))

        if self.training or ensemble:
            w_mean = self.weight_mu * alpha
            w_var = alpha * (
                weight_sigma ** 2 + (1.0 - alpha) * self.weight_mu ** 2
            )
            psi = F.conv2d(input, w_mean, self.bias_mu, self.stride, self.padding)
            delta = F.conv2d(input ** 2, w_var, bias_sigma ** 2, self.stride, self.padding)
            activations = psi + torch.sqrt(torch.clamp(delta, min=0.0) + eps_num) * torch.randn_like(delta)
        else:
            weights = (
                torch.normal(self.weight_mu, weight_sigma) if sample else self.weight_mu
            )
            bias = (
                torch.normal(self.bias_mu, bias_sigma) if sample else self.bias_mu
            )
            gates = (alpha.detach() > 0.5).float()
            activations = F.conv2d(input, weights * gates, bias, self.stride, self.padding)

            if calculate_log_probs:
                alpha = gates

        if self.training or calculate_log_probs:
            kl_bias = (
                torch.log(
                    (self.bias_sigma_prior / (bias_sigma + eps_num)) + eps_num
                )
                - 0.5
                + (bias_sigma ** 2 + (self.bias_mu - self.bias_mu_prior) ** 2)
                / (2.0 * self.bias_sigma_prior ** 2 + eps_num)
            ).sum()

            kl_weight = (
                alpha
                * (
                    torch.log(
                        (self.sigma_prior / (weight_sigma + eps_num)) + eps_num
                    )
                    - 0.5
                    + torch.log((alpha / (alpha_prior + eps_num)) + eps_num)
                    + (weight_sigma ** 2 + (self.weight_mu - self.mu_prior) ** 2)
                    / (2.0 * self.sigma_prior ** 2 + eps_num)
                )
                + (1.0 - alpha)
                * torch.log(
                    ((1.0 - alpha) / (1.0 - alpha_prior + eps_num)) + eps_num
                )
            ).sum()

            self.kl = kl_bias + kl_weight
        else:
            self.kl = torch.tensor(0.0, device=input.device)

        return activations
