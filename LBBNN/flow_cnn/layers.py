from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..transforms import PropagateFlow


class BayesianConv2dFlow(nn.Module):
    """Bayesian 2-D convolutional layer with flow-based latent variables and binary gates."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        num_transforms: int,
        padding: int = 0,
        stride: int = 1,
        lower_init_alpha: float = 0.30,
        upper_init_alpha: float = 0.49,
        a_prior: float = 0.1,
        sigma_prior: float = 2.5,
        mu_prior: float = 0.0,
        weight_mu_init_range: tuple[float, float] = (-1.2, 1.2),
        weight_rho_init_mean: float = -9.0,
        z_flow_type: str = "IAF",
        r_flow_type: str = "IAF",
        iaf_h_sizes: Sequence[int] = (250, 250),
        rnvp_h_sizes: Sequence[int] = (10, 10),
    ) -> None:
        """Initialize the flow-based Bayesian convolutional layer.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Kernel size. An integer is treated as a square kernel.
            num_transforms: Number of transforms in each flow.
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
            z_flow_type: Transform type for the variational ``z`` flow
                (``"IAF"`` or ``"RNVP"``).
            r_flow_type: Transform type for the auxiliary ``r`` flow
                (``"IAF"`` or ``"RNVP"``).
            iaf_h_sizes: Hidden layer sizes for IAF MADE networks.
            rnvp_h_sizes: Hidden layer sizes for RNVP coupling networks.

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

        # Prior buffers.
        self.register_buffer("mu_prior", torch.full(shape, mu_prior))
        self.register_buffer("sigma_prior", torch.full(shape, sigma_prior))
        self.register_buffer("alpha_prior", torch.full(shape, a_prior))
        self.register_buffer("bias_mu_prior", torch.zeros(out_channels))
        self.register_buffer("bias_sigma_prior", torch.full((out_channels,), sigma_prior))

        # Flow latent variable parameters (per output channel).
        self.q0_mean = nn.Parameter(torch.randn(out_channels))
        self.q0_log_var = nn.Parameter(weight_rho_init_mean + torch.randn(out_channels))
        self.r0_c = nn.Parameter(torch.randn(out_channels))
        self.r0_b1 = nn.Parameter(torch.randn(out_channels))
        self.r0_b2 = nn.Parameter(torch.randn(out_channels))

        self.z_flow = PropagateFlow(
            z_flow_type, out_channels, num_transforms,
            iaf_h_sizes=iaf_h_sizes, rnvp_h_sizes=rnvp_h_sizes,
        )
        self.r_flow = PropagateFlow(
            r_flow_type, out_channels, num_transforms,
            iaf_h_sizes=iaf_h_sizes, rnvp_h_sizes=rnvp_h_sizes,
        )

        # Cached samples reused by kl_div().
        self._cached_z: Tensor = torch.zeros(out_channels)
        self._cached_zk: Tensor = torch.zeros(out_channels)

    def _sample_z(self) -> tuple[Tensor, Tensor]:
        """Sample the flow-transformed latent variable.

        Returns:
            A ``(zk, log_det_q)`` tuple.
        """
        q0_std = self.q0_log_var.exp().sqrt()
        epsilon = torch.randn_like(q0_std)
        self._cached_z = self.q0_mean + q0_std * epsilon
        zk, log_det_q = self.z_flow(self._cached_z)
        self._cached_zk = zk
        return zk, log_det_q.squeeze()

    def kl_div(self) -> Tensor:
        """Compute the KL divergence term for this layer.

        Must be called after :meth:`forward` so that ``alpha_q`` and
        ``weight_sigma`` are up to date.

        Returns:
            Scalar KL divergence tensor.
        """
        eps_num = torch.tensor(1e-45, device=self.weight_mu.device, dtype=self.weight_mu.dtype)

        z2, log_det_q = self._sample_z()
        z2_view = z2.view(-1, 1, 1, 1)

        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        bias_sigma = torch.log1p(torch.exp(self.bias_rho))

        alpha = torch.sigmoid(self.lambdal)
        alpha_prior = self.alpha_prior

        W_mean = self.weight_mu * z2_view * alpha
        W_var = alpha * (weight_sigma ** 2 + (1.0 - alpha) * self.weight_mu ** 2 * z2_view ** 2)

        log_q0 = (
            -0.5 * torch.log(torch.tensor(math.pi, device=z2.device, dtype=z2.dtype))
            - 0.5 * self.q0_log_var
            - 0.5 * ((self._cached_z - self.q0_mean) ** 2 / self.q0_log_var.exp())
        ).sum()
        log_q = -log_det_q + log_q0

        act_mu = W_mean.view(-1, len(self.r0_c)) @ self.r0_c
        act_var = W_var.view(-1, len(self.r0_c)) @ self.r0_c ** 2
        act = act_mu + act_var.sqrt() * torch.randn_like(act_var)

        mean_r = self.r0_b1.outer(act).mean(-1)
        log_var_r = self.r0_b2.outer(act).mean(-1)

        z_b, log_det_r = self.r_flow(z2)
        log_rb = (
            -0.5 * torch.log(torch.tensor(math.pi, device=z2.device, dtype=z2.dtype))
            - 0.5 * log_var_r
            - 0.5 * ((z_b[-1] - mean_r) ** 2 / log_var_r.exp())
        ).sum()
        log_r = log_det_r + log_rb

        kl_bias = (
            torch.log((self.bias_sigma_prior / (bias_sigma + eps_num)) + eps_num)
            - 0.5
            + (bias_sigma ** 2 + (self.bias_mu - self.bias_mu_prior) ** 2)
            / (2.0 * self.bias_sigma_prior ** 2 + eps_num)
        ).sum()

        kl_weight = (
            alpha
            * (
                torch.log((self.sigma_prior / (weight_sigma + eps_num)) + eps_num)
                - 0.5
                + torch.log((alpha / (alpha_prior + eps_num)) + eps_num)
                + (weight_sigma ** 2 + (self.weight_mu * z2_view - self.mu_prior) ** 2)
                / (2.0 * self.sigma_prior ** 2 + eps_num)
            )
            + (1.0 - alpha)
            * torch.log(((1.0 - alpha) / (1.0 - alpha_prior + eps_num)) + eps_num)
        ).sum()

        return kl_bias + kl_weight + log_q - log_r

    def forward(
        self,
        input: Tensor,
        ensemble: bool = True,
        post_train: bool = False,
    ) -> Tensor:
        """Compute the forward pass through the flow-based Bayesian conv layer.

        Args:
            input: Input tensor of shape ``(B, C_in, H, W)``.
            ensemble: Whether to use the median probability model.
            post_train: Whether to use thresholded inclusion probabilities.

        Returns:
            Output activations of shape ``(B, C_out, H', W')``.
        """
        alpha = torch.sigmoid(self.lambdal)
        alpha_prior = self.alpha_prior.clone()

        if post_train:
            alpha = (alpha.detach() > 0.5).float()
            alpha_prior[alpha.detach() < 0.5] = 0.0

        weight_sigma = torch.log1p(torch.exp(self.weight_rho))

        z, _ = self._sample_z()
        z_view = z.view(-1, 1, 1, 1)

        if self.training or ensemble:
            w_mean = self.weight_mu * z_view * alpha
            w_var = alpha * (
                weight_sigma ** 2 + (1.0 - alpha) * self.weight_mu ** 2 * z_view ** 2
            )
            psi = F.conv2d(input, w_mean, self.bias_mu, self.stride, self.padding)
            delta = F.conv2d(input ** 2, w_var, torch.log1p(torch.exp(self.bias_rho)) ** 2, self.stride, self.padding)
            activations = psi + torch.sqrt(torch.clamp(delta, min=0.0) + 1e-45) * torch.randn_like(delta)
        else:
            w = torch.normal(self.weight_mu * z_view, weight_sigma)
            bias = torch.normal(self.bias_mu, torch.log1p(torch.exp(self.bias_rho)))
            gates = (alpha.detach() > 0.5).float()
            activations = F.conv2d(input, w * gates, bias, self.stride, self.padding)

        return activations
