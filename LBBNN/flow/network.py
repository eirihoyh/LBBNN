from __future__ import annotations

from typing import Callable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .layers import BayesianLinearFlow
from .._base import BayesianNetworkBase


class BayesianNetworkFlow(BayesianNetworkBase):
    """Bayesian neural network with flow-based layers and input skip connections."""

    def __init__(
        self,
        dim: int,
        p: int,
        hidden_layers: int,
        a_prior: float = 0.05,
        sigma_prior: float = 2.0,
        mu_prior: float = 0.0,
        weight_mu_init_range: tuple[float, float] = (-0.01, 0.01),
        weight_rho_init_mean: float = -9.0,
        num_transforms: int = 2,
        z_flow_type: str = "IAF",
        r_flow_type: str = "IAF",
        iaf_h_sizes: Sequence[int] = (250, 250),
        rnvp_h_sizes: Sequence[int] = (10, 10),
        classification: bool = True,
        n_classes: int = 1,
        act_func: Callable[[Tensor], Tensor] = torch.relu,
        lower_init_alpha: float = 0.30,
        upper_init_alpha: float = 0.49,
        input_skip: bool = True,
        custom_loss: bool | Callable[[Tensor, Tensor], Tensor] = False,
    ) -> None:
        """Initialize the flow-based Bayesian network.

        Args:
            dim: Number of hidden units in each hidden layer.
            p: Number of input features.
            hidden_layers: Number of hidden layers.
            a_prior: Prior inclusion probability.
            sigma_prior: Prior standard deviation for the weights.
            mu_prior: Prior mean for the weights.
            weight_mu_init_range: Lower and upper bounds for uniform
                initialization of the weight means.
            weight_rho_init_mean: Mean of the Gaussian used to initialize
                the rho-parameter (softplus-mapped to weight std).
            num_transforms: Number of flow transforms per layer.
            z_flow_type: Transform type used for the variational `z` flow
                (``"IAF"`` or ``"RNVP"``).
            r_flow_type: Transform type used for the auxiliary `r` flow
                (``"IAF"`` or ``"RNVP"``).
            iaf_h_sizes: Hidden layer sizes for IAF MADE networks. The
                default ``(250, 250)`` matches the historical hardcoded
                value; smaller sizes (e.g. ``(64, 64)`` or ``(32, 32)``)
                make training and inference much faster at the cost of
                some flow expressivity.
            rnvp_h_sizes: Hidden layer sizes for RNVP coupling networks.
            classification: Whether the task is classification.
            n_classes: Number of output classes.
            act_func: Activation function used in hidden layers.
            lower_init_alpha: Lower bound for inclusion probability initialization.
            upper_init_alpha: Upper bound for inclusion probability initialization.
            input_skip: If True (default) every hidden and the output layer
                receives the original input concatenated to its activations.
                If False the network behaves as a standard feed-forward MLP
                with binary gates.
            custom_loss: Either a callable used as the loss, or False to fall
                back to the task-appropriate default (NLLLoss / BCELoss / MSELoss).

        Returns:
            None.
        """
        super().__init__()

        self.p = p
        self.classification = classification
        self.multiclass = n_classes > 1
        self.act = act_func
        self.input_skip = input_skip

        layer_kwargs = dict(
            a_prior=a_prior,
            sigma_prior=sigma_prior,
            mu_prior=mu_prior,
            weight_mu_init_range=weight_mu_init_range,
            weight_rho_init_mean=weight_rho_init_mean,
            num_transforms=num_transforms,
            z_flow_type=z_flow_type,
            r_flow_type=r_flow_type,
            iaf_h_sizes=iaf_h_sizes,
            rnvp_h_sizes=rnvp_h_sizes,
            lower_init_alpha=lower_init_alpha,
            upper_init_alpha=upper_init_alpha,
        )

        hidden_in = dim + p if input_skip else dim

        self.linears = nn.ModuleList(
            [BayesianLinearFlow(p, dim, **layer_kwargs)]
        )

        self.linears.extend(
            [
                BayesianLinearFlow(hidden_in, dim, **layer_kwargs)
                for _ in range(hidden_layers - 1)
            ]
        )

        self.linears.append(
            BayesianLinearFlow(hidden_in, n_classes, **layer_kwargs)
        )

        if custom_loss:
            self.loss = custom_loss
        elif classification and self.multiclass:
            self.loss = nn.NLLLoss(reduction="sum")
        elif classification:
            self.loss = nn.BCELoss(reduction="sum")
        else:
            self.loss = nn.MSELoss(reduction="sum")

    def _forward_logits(
        self,
        x: Tensor,
        sample: bool = False,
        ensemble: bool = True,
        post_train: bool = False,
    ) -> Tensor:
        """Compute logits before the final output transformation.

        Args:
            x: Input tensor.
            sample: When in deterministic mode, whether to draw weights
                from their flow-modulated Gaussian or use the mean.
            ensemble: Whether to use median probability model.
            post_train: Whether to use post-training thresholded inclusion.

        Returns:
            Output logits.
        """
        x_input = x.view(-1, self.p)

        x_hidden = self.act(
            self.linears[0](
                x_input,
                ensemble=ensemble,
                sample=sample,
                post_train=post_train,
            )
        )

        def _next_input(h: Tensor) -> Tensor:
            return torch.cat((h, x_input), dim=1) if self.input_skip else h

        for layer in self.linears[1:-1]:
            x_hidden = self.act(
                layer(
                    _next_input(x_hidden),
                    ensemble=ensemble,
                    sample=sample,
                    post_train=post_train,
                )
            )

        logits = self.linears[-1](
            _next_input(x_hidden),
            ensemble=ensemble,
            sample=sample,
            post_train=post_train,
        )

        return logits

    def forward(
        self,
        x: Tensor,
        sample: bool = False,
        ensemble: bool = True,
        calculate_log_probs: bool = False,
        post_train: bool = False,
    ) -> Tensor:
        """Run a forward pass through the network.

        Args:
            x: Input tensor.
            sample: Whether to sample weights in deterministic mode.
            ensemble: Whether to use median probability model.
            calculate_log_probs: Accepted for API compatibility with the
                LRT network. The flow KL is always computable on demand
                via ``kl()`` so this flag has no effect here.
            post_train: Whether to use post-training thresholded inclusion.

        Returns:
            Network outputs as probabilities or raw values depending on the task.
        """
        del calculate_log_probs

        logits = self._forward_logits(
            x=x,
            sample=sample,
            ensemble=ensemble,
            post_train=post_train,
        )

        if self.classification:
            if self.multiclass:
                return F.log_softmax(logits, dim=1)
            return torch.sigmoid(logits)

        return logits

    def forward_preact(
        self,
        x: Tensor,
        sample: bool = False,
        ensemble: bool = False,
        calculate_log_probs: bool = False,
        post_train: bool = False,
    ) -> Tensor:
        """Run a forward pass and return pre-activation outputs.

        Args:
            x: Input tensor.
            sample: Whether to sample weights in deterministic mode.
            ensemble: Whether to use median probability model.
            calculate_log_probs: Accepted for API compatibility; see
                :meth:`forward`.
            post_train: Whether to use post-training thresholded inclusion.

        Returns:
            Output logits before the final activation.
        """
        del calculate_log_probs

        return self._forward_logits(
            x=x,
            sample=sample,
            ensemble=ensemble,
            post_train=post_train,
        )

    def kl(self) -> Tensor:
        """Return the total KL divergence across all layers.

        Args:
            None.

        Returns:
            The summed KL divergence.
        """
        return sum(layer.kl_div() for layer in self.linears)

