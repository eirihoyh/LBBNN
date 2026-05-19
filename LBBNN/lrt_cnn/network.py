from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .layers import BayesianConv2dLRT
from ..lrt.layers import BayesianLinearLRT
from .._base import BayesianNetworkBase
from .._cnn_utils import compute_output_dimensions


class BayesianNetworkCNNLRT(BayesianNetworkBase):
    """Bayesian CNN with LRT-based convolutional layers and input skip-connections.

    The network first extracts features via a stack of ``BayesianConv2dLRT``
    layers, flattens the result, then passes it through a sequence of
    ``BayesianLinearLRT`` fully-connected layers. Every FC layer (optionally)
    receives the original flattened image concatenated to its activations via
    an input skip-connection, mirroring the design of
    :class:`~LBBNN.lrt.network.BayesianNetworkLRT`.
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
        sigma_prior: float = 2.5,
        mu_prior: float = 0.0,
        weight_rho_init_mean: float = -9.0,
        classification: bool = True,
        n_classes: int = 1,
        act_func: Callable[[Tensor], Tensor] = torch.relu,
        lower_init_alpha: float = 0.90,
        upper_init_alpha: float = 0.99,
        high_init_covariate_prob: bool = False,
        input_skip: bool = False,
        custom_loss: bool | Callable[[Tensor, Tensor], Tensor] = False,
    ) -> None:
        """Initialize the Bayesian CNN.

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
            weight_rho_init_mean: Mean of the Gaussian used to initialize
                the rho-parameter (softplus-mapped to weight std).
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
            custom_loss: Optional custom loss callable. When provided it
                replaces the default task-specific loss.

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

        conv_kwargs = dict(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            a_prior=a_prior,
            sigma_prior=sigma_prior,
            mu_prior=mu_prior,
            weight_rho_init_mean=weight_rho_init_mean,
            lower_init_alpha=lower_init_alpha,
            upper_init_alpha=upper_init_alpha,
        )

        linear_kwargs = dict(
            a_prior=a_prior,
            sigma_prior=sigma_prior,
            mu_prior=mu_prior,
            weight_rho_init_mean=weight_rho_init_mean,
            lower_init_alpha=lower_init_alpha,
            upper_init_alpha=upper_init_alpha,
        )

        n_skip = self.p if high_init_covariate_prob else None

        # Convolutional stack.
        channels = [init_in_channels] + list(out_channel_list)
        self.convs = nn.ModuleList([
            BayesianConv2dLRT(channels[i], channels[i + 1], **conv_kwargs)
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
            BayesianLinearLRT(first_fc_in, dim, n_skip_features=n_skip, **linear_kwargs)
        ])
        self.linears.extend([
            BayesianLinearLRT(hidden_fc_in, dim, n_skip_features=n_skip, **linear_kwargs)
            for _ in range(hidden_layers - 1)
        ])
        self.linears.append(
            BayesianLinearLRT(hidden_fc_in, n_classes, n_skip_features=n_skip, **linear_kwargs)
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
        calculate_log_probs: bool = False,
        post_train: bool = False,
    ) -> Tensor:
        """Run the full forward pass and return logits before output activation.

        Args:
            x: Flattened input images of shape ``(B, p)``.
            sample: Whether to sample weights in deterministic mode.
            ensemble: Whether to use the median probability model.
            calculate_log_probs: Whether to update KL-related quantities.
            post_train: Whether to use thresholded inclusion probabilities.

        Returns:
            Logit tensor of shape ``(B, n_classes)``.
        """
        x_input = x.view(-1, self.p)
        h = x_input.view(-1, self.init_in_channels, self.p1, self.p2)

        conv_kw = dict(
            ensemble=ensemble,
            sample=sample,
            calculate_log_probs=calculate_log_probs,
            post_train=post_train,
        )

        for conv in self.convs:
            h = self.act(conv(h, **conv_kw))

        h = h.flatten(1)

        def _fc_input(feat: Tensor) -> Tensor:
            return torch.cat((feat, x_input), dim=1) if self.input_skip else feat

        # First FC layer always receives conv output + (optional) original image.
        h = self.act(
            self.linears[0](
                torch.cat((h, x_input), dim=1) if self.input_skip else h,
                **conv_kw,
            )
        )

        for layer in self.linears[1:-1]:
            h = self.act(layer(_fc_input(h), **conv_kw))

        return self.linears[-1](_fc_input(h), **conv_kw)

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
            sample: Whether to sample weights in deterministic mode.
            ensemble: Whether to use the median probability model.
            calculate_log_probs: Whether to update KL-related quantities.
            post_train: Whether to use thresholded inclusion probabilities.

        Returns:
            Network outputs as probabilities or raw values depending on task.
        """
        logits = self._forward_logits(
            x=x,
            sample=sample,
            ensemble=ensemble,
            calculate_log_probs=calculate_log_probs,
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
        """Run a forward pass and return pre-activation (logit) outputs.

        Args:
            x: Flattened input images of shape ``(B, p)``.
            sample: Whether to sample weights in deterministic mode.
            ensemble: Whether to use the median probability model.
            calculate_log_probs: Whether to update KL-related quantities.
            post_train: Whether to use thresholded inclusion probabilities.

        Returns:
            Logit tensor of shape ``(B, n_classes)``.
        """
        return self._forward_logits(
            x=x,
            sample=sample,
            ensemble=ensemble,
            calculate_log_probs=calculate_log_probs,
            post_train=post_train,
        )

    def kl(self) -> Tensor:
        """Return the total KL divergence across all convolutional and FC layers.

        Returns:
            Summed KL divergence tensor.
        """
        return sum(c.kl for c in self.convs) + sum(l.kl for l in self.linears)
