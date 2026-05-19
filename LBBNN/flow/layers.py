from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
from torch import Tensor

from ..transforms import PropagateFlow


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
        lower_init_alpha: float = 0.30,
        upper_init_alpha: float = 0.49,
        a_prior: float = 0.1,
        sigma_prior: float = 2.0,
        mu_prior: float = 0.0,
        weight_rho_init_mean: float = -9.0,
        z_flow_type: str = "IAF",
        r_flow_type: str = "IAF",
        iaf_h_sizes: Sequence[int] = (250, 250),
        rnvp_h_sizes: Sequence[int] = (10, 10),
    ) -> None:
        """Initialize the flow-based Bayesian linear layer.

        Args:
            in_features: Number of input features.
            out_features: Number of output features.
            num_transforms: Number of transforms in each flow.
            lower_init_alpha: Lower bound for inclusion probability initialization.
            upper_init_alpha: Upper bound for inclusion probability initialization.
            a_prior: Prior inclusion probability.
            sigma_prior: Prior standard deviation for the weights.
            mu_prior: Prior mean for the weights.
            weight_rho_init_mean: Mean of the Gaussian used to initialize
                the rho-parameter (softplus-mapped to weight std).
            z_flow_type: Transform type used for the variational `z` flow
                (``"IAF"`` or ``"RNVP"``).
            r_flow_type: Transform type used for the auxiliary `r` flow
                (``"IAF"`` or ``"RNVP"``).
            iaf_h_sizes: Hidden layer sizes for IAF MADE networks (used
                whenever ``z_flow_type`` or ``r_flow_type`` is ``"IAF"``).
            rnvp_h_sizes: Hidden layer sizes for RNVP coupling networks
                (used whenever ``z_flow_type`` or ``r_flow_type`` is
                ``"RNVP"``).

        Returns:
            None.
        """
        super().__init__()

        lower_init_lambda = math.log(lower_init_alpha/ (1-lower_init_alpha))
        upper_init_lambda = math.log(upper_init_alpha/ (1-upper_init_alpha))

        # Variational parameters for the weights.
        self.weight_mu = nn.Parameter(
            torch.empty(out_features, in_features).normal_(
                mean=0, std=(2/in_features)**0.5
            )
        )
        self.weight_rho = nn.Parameter(
            weight_rho_init_mean + 0.1 * torch.randn(out_features, in_features)
        )

        self.lambdal = nn.Parameter(
            torch.empty(out_features, in_features).uniform_(
                lower_init_lambda,
                upper_init_lambda,
            )
        )

        # Prior parameters
        self.register_buffer(
            "mu_prior",
            torch.full((out_features, in_features), mu_prior),
        )
        self.register_buffer(
            "sigma_prior",
            torch.full((out_features, in_features), sigma_prior),
        )
        self.register_buffer(
            "alpha_prior",
            torch.full((out_features, in_features), a_prior),
        )

        # normalizing flow specific paramters
        self.q0_mean = nn.Parameter(torch.randn(in_features))
        self.q0_log_var = nn.Parameter(-9.0 + torch.randn(in_features))
        self.c1 = nn.Parameter(torch.randn(in_features))
        self.r0_b1 = nn.Parameter(torch.randn(in_features))
        self.r0_b2 = nn.Parameter(torch.randn(in_features))

        self.z_flow = PropagateFlow(
            z_flow_type, in_features, num_transforms,
            iaf_h_sizes=iaf_h_sizes, rnvp_h_sizes=rnvp_h_sizes,
        )
        self.r_flow = PropagateFlow(
            r_flow_type, in_features, num_transforms,
            iaf_h_sizes=iaf_h_sizes, rnvp_h_sizes=rnvp_h_sizes,
        )

        self.z: Tensor | None = None
        # Cache populated by `sample_z`/`forward` so that a subsequent
        # `kl_div()` call evaluates the variational bound at the *same*
        # latent sample as the one used to compute the activations.
        self._cached_zk: Tensor | None = None
        self._cached_log_det_q: Tensor | None = None
        self.hardtanh = nn.Hardtanh()

    def sample_z(self) -> tuple[Tensor, Tensor]:
        """Sample the latent flow variable.

        Also caches ``z``, ``zk`` and ``log_det_q`` on the layer so that
        :meth:`kl_div` reuses the same sample (the variational bound is
        defined for one shared draw of ``z``).

        Returns:
            A tuple containing the transformed latent sample and the
            corresponding log-determinant from the variational flow.
        """
        q0_std = torch.exp(0.5 * self.q0_log_var)
        epsilon_z = torch.randn_like(q0_std)

        self.z = self.q0_mean + q0_std * epsilon_z
        zk, log_det_q = self.z_flow(self.z)
        log_det_q = log_det_q.squeeze()

        self._cached_zk = zk
        self._cached_log_det_q = log_det_q

        return zk, log_det_q

    def kl_div(self) -> Tensor:
        """Compute the KL divergence term for the layer.

        Reuses the latent sample produced by the most recent
        :meth:`forward` (or :meth:`sample_z`) when available, so the KL
        is evaluated at the same draw of ``z`` as the activations. If no
        forward has been done yet a fresh sample is taken.

        Returns:
            The KL divergence for the layer.
        """

        if self._cached_zk is not None and self._cached_log_det_q is not None:
            zk = self._cached_zk
            log_det_q = self._cached_log_det_q
        else:
            zk, log_det_q = self.sample_z()

        if self.z is None:
            raise RuntimeError("Latent variable z must be sampled before KL.")

        alpha = torch.sigmoid(self.lambdal)
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))

        eps = torch.tensor(1e-45, device=zk.device, dtype=zk.dtype)

        weight_mean = zk * self.weight_mu * alpha
        weight_var = alpha * (
            weight_sigma**2
            + (1.0 - alpha) * self.weight_mu**2 * zk**2
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
            alpha
            * (
                torch.log((self.sigma_prior / (weight_sigma + eps)) + eps)
                - 0.5
                + torch.log((alpha / (self.alpha_prior + eps)) + eps)
                + (
                    weight_sigma**2
                    + (self.weight_mu * zk - self.mu_prior) ** 2
                )
                / (2.0 * self.sigma_prior**2 + eps)
            )
            + (1.0 - alpha)
            * torch.log(
                ((1.0 - alpha) / (1.0 - self.alpha_prior + eps)) + eps
            )
        ).sum()

        return kl_weight + log_q - log_r

    def forward(
        self,
        input: Tensor,
        ensemble: bool = False,
        sample: bool = False,
        post_train: bool = False,
    ) -> Tensor:
        """Compute the forward pass through the layer.

        Args:
            input: Input tensor of shape ``(batch_size, in_features)``.
            ensemble: Whether to use the median probability model.
            sample: When in deterministic mode (``ensemble=False`` and
                not training), whether to draw weights from
                ``N(weight_mu * zk, weight_sigma)`` or use the mean.
            post_train: Whether to threshold the inclusion probabilities.

        Returns:
            Output activations of shape ``(batch_size, out_features)``.
        """
        alpha = torch.sigmoid(self.lambdal)

        if post_train:
            alpha = (alpha.detach() > 0.5).float()

        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        zk, _ = self.sample_z()

        if self.training or ensemble:
            expected_weight = self.weight_mu * alpha * zk
            var_weight = alpha * (
                weight_sigma**2
                + (1.0 - alpha) * self.weight_mu**2 * zk**2
            )

            expected_bias = input @ expected_weight.T
            var_bias = input.pow(2) @ var_weight.T
            noise = torch.randn_like(var_bias)

            activations = expected_bias + torch.sqrt(
                torch.clamp(var_bias, min=0.0)
            ) * noise
        else:
            weights = (
                torch.normal(self.weight_mu * zk, weight_sigma)
                if sample
                else self.weight_mu * zk
            )
            gates = (alpha.detach() > 0.5).float()
            activations = input @ (weights * gates).T

        return activations