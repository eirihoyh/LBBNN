from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

from .transforms import PropagateFlow

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
Z_FLOW_TYPE = "IAF"
R_FLOW_TYPE = "IAF"


def _log_pi(device: torch.device, dtype: torch.dtype) -> Tensor:
    """Return the logarithm of pi as a tensor.

    Args:
        device: Device used for the returned tensor.
        dtype: Data type used for the returned tensor.

    Returns:
        Tensor containing ``log(pi)``.
    """
    return torch.log(torch.tensor(math.pi, device=device, dtype=dtype))


class BayesianLinearFlow(nn.Module):
    """Bayesian linear layer with flow-based latent variables and binary gates."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_transforms: int,
        lower_init_lambda: float = -3.0,
        upper_init_lambda: float = -0.0,
        a_prior: float = 0.1,
    ) -> None:
        """Initialize the flow-based Bayesian linear layer.

        Args:
            in_features: Number of input features.
            out_features: Number of output features.
            num_transforms: Number of transforms in each flow.
            lower_init_lambda: Lower bound for inclusion logit initialization.
            upper_init_lambda: Upper bound for inclusion logit initialization.
            a_prior: Prior inclusion probability.

        Returns:
            None.
        """
        super().__init__()

        self.weight_mu = nn.Parameter(
            torch.empty(out_features, in_features).uniform_(-0.01, 0.01)
        )
        self.weight_rho = nn.Parameter(
            -9.0 + 0.1 * torch.randn(out_features, in_features)
        )
        self.weight_sigma = torch.empty_like(self.weight_rho)

        self.mu_prior = torch.zeros(out_features, in_features, device=DEVICE)
        self.sigma_prior = torch.full(
            (out_features, in_features),
            2.0,
            device=DEVICE,
        )

        self.lambdal = nn.Parameter(
            torch.empty(out_features, in_features).uniform_(
                lower_init_lambda,
                upper_init_lambda,
            )
        )
        self.alpha = torch.empty_like(self.lambdal)
        self.alpha_prior = torch.full(
            (out_features, in_features),
            a_prior,
            device=DEVICE,
        )

        self.q0_mean = nn.Parameter(torch.randn(in_features))
        self.q0_log_var = nn.Parameter(-9.0 + torch.randn(in_features))
        self.c1 = nn.Parameter(torch.randn(in_features))
        self.r0_b1 = nn.Parameter(torch.randn(in_features))
        self.r0_b2 = nn.Parameter(torch.randn(in_features))

        self.z_flow = PropagateFlow(Z_FLOW_TYPE, in_features, num_transforms)
        self.r_flow = PropagateFlow(R_FLOW_TYPE, in_features, num_transforms)

        self.z: Tensor | None = None
        self.hardtanh = nn.Hardtanh()

    def sample_z(self) -> tuple[Tensor, Tensor]:
        """Sample the latent flow variable.

        Args:
            None.

        Returns:
            A tuple containing the transformed latent sample and the
            corresponding log-determinant from the variational flow.
        """
        q0_std = torch.exp(0.5 * self.q0_log_var)
        epsilon_z = torch.randn_like(q0_std)

        self.z = self.q0_mean + q0_std * epsilon_z
        zk, log_det_q = self.z_flow(self.z)

        return zk, log_det_q.squeeze()

    def kl_div(self) -> Tensor:
        """Compute the KL divergence term for the layer.

        Args:
            None.

        Returns:
            The KL divergence for the layer.
        """
        if self.z is None:
            raise RuntimeError("Latent variable z must be sampled before KL.")

        self.alpha = torch.sigmoid(self.lambdal)
        self.weight_sigma = torch.log1p(torch.exp(self.weight_rho))

        zk, log_det_q = self.sample_z()
        eps = torch.tensor(1e-45, device=zk.device, dtype=zk.dtype)

        weight_mean = zk * self.weight_mu * self.alpha
        weight_var = self.alpha * (
            self.weight_sigma**2
            + (1.0 - self.alpha) * self.weight_mu**2 * zk**2
        ) + eps

        log_q0 = (
            -0.5 * _log_pi(self.q0_log_var.device, self.q0_log_var.dtype)
            - 0.5 * self.q0_log_var
            - 0.5 * (self.z - self.q0_mean) ** 2 / (torch.exp(self.q0_log_var) + eps)
        ).sum()
        log_q = -log_det_q + log_q0

        act_mu = self.c1 @ weight_mean.T
        act_var = self.c1**2 @ weight_var.T
        act_inner = act_mu + torch.sqrt(torch.clamp(act_var, min=0.0)) * torch.randn_like(
            act_var
        )

        act = self.hardtanh(act_inner)
        mean_r = torch.outer(self.r0_b1, act).mean(dim=-1)
        log_var_r = torch.outer(self.r0_b2, act).mean(dim=-1)

        zb, log_det_r = self.r_flow(zk)
        log_rb = (
            -0.5 * _log_pi(log_var_r.device, log_var_r.dtype)
            - 0.5 * log_var_r
            - 0.5 * (zb - mean_r) ** 2 / (torch.exp(log_var_r) + eps)
        ).sum()
        log_r = log_det_r + log_rb

        kl_weight = (
            self.alpha
            * (
                torch.log((self.sigma_prior / (self.weight_sigma + eps)) + eps)
                - 0.5
                + torch.log((self.alpha / (self.alpha_prior + eps)) + eps)
                + (
                    self.weight_sigma**2
                    + (self.weight_mu * zk - self.mu_prior) ** 2
                )
                / (2.0 * self.sigma_prior**2 + eps)
            )
            + (1.0 - self.alpha)
            * torch.log(
                ((1.0 - self.alpha) / (1.0 - self.alpha_prior + eps)) + eps
            )
        ).sum()

        return kl_weight + log_q - log_r

    def forward(
        self,
        input: Tensor,
        ensemble: bool = False,
        post_train: bool = False,
    ) -> Tensor:
        """Compute the forward pass through the layer.

        Args:
            input: Input tensor of shape ``(batch_size, in_features)``.
            ensemble: Whether to use the local reparameterization form.
            post_train: Whether to threshold the inclusion probabilities.

        Returns:
            Output activations of shape ``(batch_size, out_features)``.
        """
        self.alpha = torch.sigmoid(self.lambdal)

        if post_train:
            self.alpha = (self.alpha.detach() > 0.5).float()

        self.weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        zk, _ = self.sample_z()

        if self.training or ensemble:
            expected_weight = self.weight_mu * self.alpha * zk
            var_weight = self.alpha * (
                self.weight_sigma**2
                + (1.0 - self.alpha) * self.weight_mu**2 * zk**2
            )

            expected_bias = input @ expected_weight.T
            var_bias = input.pow(2) @ var_weight.T
            noise = torch.randn_like(var_bias)

            activations = expected_bias + torch.sqrt(
                torch.clamp(var_bias, min=0.0)
            ) * noise
        else:
            weights = torch.normal(self.weight_mu * zk, self.weight_sigma)
            gates = (self.alpha.detach() > 0.5).float()
            activations = input @ (weights * gates).T

        return activations