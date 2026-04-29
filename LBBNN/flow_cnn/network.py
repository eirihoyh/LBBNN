from __future__ import annotations

from typing import Callable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .layers import BayesianConv2dFlow
from ..flow.layers import BayesianLinearFlow
from .._base import BayesianNetworkBase
from .._cnn_utils import compute_output_dimensions


class BayesianNetworkCNNFlow(BayesianNetworkBase):
    """Bayesian CNN with flow-based convolutional layers and input skip-connections.

    The network first extracts features via a stack of ``BayesianConv2dFlow``
    layers, flattens the result, then passes it through a sequence of
    ``BayesianLinearFlow`` fully-connected layers. Every FC layer (optionally)
    receives the original flattened image concatenated to its activations via
    an input skip-connection, mirroring the design of
    :class:`~LBBNN.flow.network.BayesianNetworkFlow`.
    """

    def __init__(
        self,
        init_in_channels: int,
        out_channel_list: list[int],
        kernel_size: int,
        stride: int,
        padding: int,
        p1: int,
        p2: int,
        dim: int,
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
        high_init_covariate_prob: bool = False,
        input_skip: bool = True,
        custom_loss: bool | Callable[[Tensor, Tensor], Tensor] = False,
    ) -> None:
        """Initialize the flow-based Bayesian CNN.

        Args:
            init_in_channels: Number of input image channels.
            out_channel_list: Output channel counts for each conv layer.
            kernel_size: Convolution kernel size (assumed square).
            stride: Convolution stride.
            padding: Convolution zero-padding.
            p1: Image height in pixels.
            p2: Image width in pixels.
            dim: Number of hidden units in each FC hidden layer.
            hidden_layers: Number of FC hidden layers.
            a_prior: Prior inclusion probability.
            sigma_prior: Prior standard deviation for the weights.
            mu_prior: Prior mean for the weights.
            weight_mu_init_range: Uniform init range for weight means.
            weight_rho_init_mean: Mean of the Gaussian used to initialize
                the rho-parameter (softplus-mapped to weight std).
            num_transforms: Number of flow transforms per layer.
            z_flow_type: Transform type for the variational ``z`` flow.
            r_flow_type: Transform type for the auxiliary ``r`` flow.
            iaf_h_sizes: Hidden layer sizes for IAF MADE networks.
            rnvp_h_sizes: Hidden layer sizes for RNVP coupling networks.
            classification: Whether the task is classification.
            n_classes: Number of output classes.
            act_func: Activation function for hidden layers.
            lower_init_alpha: Lower bound for inclusion probability init.
            upper_init_alpha: Upper bound for inclusion probability init.
            high_init_covariate_prob: Whether to initialise skip-input
                weights with high inclusion probability.
            input_skip: If ``True`` every FC hidden and output layer
                receives the original flattened image concatenated to its
                activations.
            custom_loss: Optional custom loss callable.

        Returns:
            None.
        """
        super().__init__()

        self.init_in_channels = init_in_channels
        self.p1 = p1
        self.p2 = p2
        self.p = p1 * p2 * init_in_channels
        self.classification = classification
        self.multiclass = n_classes > 1
        self.act = act_func
        self.input_skip = input_skip

        conv_kwargs: dict = dict(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            num_transforms=num_transforms,
            a_prior=a_prior,
            sigma_prior=sigma_prior,
            mu_prior=mu_prior,
            weight_mu_init_range=weight_mu_init_range,
            weight_rho_init_mean=weight_rho_init_mean,
            z_flow_type=z_flow_type,
            r_flow_type=r_flow_type,
            iaf_h_sizes=iaf_h_sizes,
            rnvp_h_sizes=rnvp_h_sizes,
            lower_init_alpha=lower_init_alpha,
            upper_init_alpha=upper_init_alpha,
        )

        linear_kwargs: dict = dict(
            num_transforms=num_transforms,
            a_prior=a_prior,
            sigma_prior=sigma_prior,
            mu_prior=mu_prior,
            weight_mu_init_range=weight_mu_init_range,
            weight_rho_init_mean=weight_rho_init_mean,
            z_flow_type=z_flow_type,
            r_flow_type=r_flow_type,
            iaf_h_sizes=iaf_h_sizes,
            rnvp_h_sizes=rnvp_h_sizes,
            lower_init_alpha=lower_init_alpha,
            upper_init_alpha=upper_init_alpha,
        )

        n_skip = self.p if high_init_covariate_prob else None

        # Convolutional stack.
        channels = [init_in_channels] + list(out_channel_list)
        self.convs = nn.ModuleList([
            BayesianConv2dFlow(channels[i], channels[i + 1], **conv_kwargs)
            for i in range(len(out_channel_list))
        ])

        width_out, height_out = compute_output_dimensions(
            len(self.convs), p1, p2, kernel_size, stride, padding
        )
        cnn_out_dim = out_channel_list[-1] * width_out * height_out

        # Fully-connected stack.
        first_fc_in = cnn_out_dim + self.p if input_skip else cnn_out_dim
        hidden_fc_in = dim + self.p if input_skip else dim

        self.linears = nn.ModuleList([
            BayesianLinearFlow(first_fc_in, dim, **linear_kwargs)
        ])
        self.linears.extend([
            BayesianLinearFlow(hidden_fc_in, dim, **linear_kwargs)
            for _ in range(hidden_layers - 1)
        ])
        self.linears.append(
            BayesianLinearFlow(hidden_fc_in, n_classes, **linear_kwargs)
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
        ensemble: bool = True,
        post_train: bool = False,
    ) -> Tensor:
        """Run the full forward pass and return logits before output activation.

        Args:
            x: Flattened input images of shape ``(B, p)``.
            ensemble: Whether to use the local reparameterization form.
            post_train: Whether to use thresholded inclusion probabilities.

        Returns:
            Logit tensor of shape ``(B, n_classes)``.
        """
        x_input = x.view(-1, self.p)
        h = x_input.view(-1, self.init_in_channels, self.p1, self.p2)

        conv_kw = dict(ensemble=ensemble, post_train=post_train)

        for conv in self.convs:
            h = self.act(conv(h, **conv_kw))

        h = h.flatten(1)

        def _fc_input(feat: Tensor) -> Tensor:
            return torch.cat((feat, x_input), dim=1) if self.input_skip else feat

        linear_kw = dict(ensemble=ensemble, post_train=post_train)

        h = self.act(
            self.linears[0](
                torch.cat((h, x_input), dim=1) if self.input_skip else h,
                **linear_kw,
            )
        )

        for layer in self.linears[1:-1]:
            h = self.act(layer(_fc_input(h), **linear_kw))

        return self.linears[-1](_fc_input(h), **linear_kw)

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
            x: Flattened input images of shape ``(B, p)``.
            sample: Unused; present for API compatibility.
            ensemble: Whether to use the local reparameterization form.
            calculate_log_probs: Unused; present for API compatibility.
            post_train: Whether to use thresholded inclusion probabilities.

        Returns:
            Network outputs as probabilities or raw values depending on task.
        """
        logits = self._forward_logits(x=x, ensemble=ensemble, post_train=post_train)

        if self.classification:
            if self.multiclass:
                return F.log_softmax(logits, dim=1)
            return torch.sigmoid(logits)

        return logits

    def forward_preact(
        self,
        x: Tensor,
        sample: bool = False,
        ensemble: bool = True,
        calculate_log_probs: bool = False,
        post_train: bool = False,
    ) -> Tensor:
        """Run a forward pass and return pre-activation (logit) outputs.

        Args:
            x: Flattened input images of shape ``(B, p)``.
            sample: Unused; present for API compatibility.
            ensemble: Whether to use the local reparameterization form.
            calculate_log_probs: Unused; present for API compatibility.
            post_train: Whether to use thresholded inclusion probabilities.

        Returns:
            Logit tensor of shape ``(B, n_classes)``.
        """
        return self._forward_logits(x=x, ensemble=ensemble, post_train=post_train)

    def kl(self) -> Tensor:
        """Return the total KL divergence across all convolutional and FC layers.

        Returns:
            Summed KL divergence tensor.
        """
        return sum(c.kl_div() for c in self.convs) + sum(l.kl_div() for l in self.linears)
