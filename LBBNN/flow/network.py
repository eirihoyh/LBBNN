from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .layers import BayesianLinearFlow


class BayesianNetworkFlow(nn.Module):
    """Bayesian neural network with flow-based layers and input skip connections."""

    def __init__(
        self,
        dim: int,
        p: int,
        hidden_layers: int,
        a_prior: float = 0.05,
        num_transforms: int = 2,
        classification: bool = True,
        n_classes: int = 1,
        act_func: Callable[[Tensor], Tensor] = torch.sigmoid,
        lower_init_alpha: float = 0.30,
        upper_init_alpha: float = 0.49,
    ) -> None:
        """Initialize the flow-based Bayesian network.

        Args:
            dim: Number of hidden units in each hidden layer.
            p: Number of input features.
            hidden_layers: Number of hidden layers.
            a_prior: Prior inclusion probability.
            num_transforms: Number of flow transforms per layer.
            classification: Whether the task is classification.
            n_classes: Number of output classes.
            act_func: Activation function used in hidden layers.
            lower_init_alpha: Lower bound for inclusion probability initialization.
            upper_init_alpha: Upper bound for inclusion probability initialization.

        Returns:
            None.
        """
        super().__init__()

        self.p = p
        self.classification = classification
        self.multiclass = n_classes > 1
        self.act = act_func

        self.linears = nn.ModuleList(
            [
                BayesianLinearFlow(
                    p,
                    dim,
                    a_prior=a_prior,
                    num_transforms=num_transforms,
                    lower_init_alpha=lower_init_alpha,
                    upper_init_alpha=upper_init_alpha,
                )
            ]
        )

        self.linears.extend(
            [
                BayesianLinearFlow(
                    dim + p,
                    dim,
                    a_prior=a_prior,
                    num_transforms=num_transforms,
                    lower_init_alpha=lower_init_alpha,
                    upper_init_alpha=upper_init_alpha,
                )
                for _ in range(hidden_layers - 1)
            ]
        )

        self.linears.append(
            BayesianLinearFlow(
                dim + p,
                n_classes,
                a_prior=a_prior,
                num_transforms=num_transforms,
                lower_init_alpha=lower_init_alpha,
                upper_init_alpha=upper_init_alpha,
            )
        )

        if classification and self.multiclass:
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
        """Compute logits before the final output transformation.

        Args:
            x: Input tensor.
            ensemble: Whether to use ensemble-style inference.
            post_train: Whether to use post-training thresholded inclusion.

        Returns:
            Output logits.
        """
        x_input = x.view(-1, self.p)

        x_hidden = self.act(
            self.linears[0](
                x_input,
                ensemble=ensemble,
                post_train=post_train,
            )
        )

        for layer in self.linears[1:-1]:
            x_hidden = self.act(
                layer(
                    torch.cat((x_hidden, x_input), dim=1),
                    ensemble=ensemble,
                    post_train=post_train,
                )
            )

        logits = self.linears[-1](
            torch.cat((x_hidden, x_input), dim=1),
            ensemble=ensemble,
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
            sample: Unused flag kept for API compatibility.
            ensemble: Whether to use ensemble-style inference.
            calculate_log_probs: Unused flag kept for API compatibility.
            post_train: Whether to use post-training thresholded inclusion.

        Returns:
            Network outputs as probabilities or raw values depending on the task.
        """
        del sample
        del calculate_log_probs

        logits = self._forward_logits(
            x=x,
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
            sample: Unused flag kept for API compatibility.
            ensemble: Whether to use ensemble-style inference.
            calculate_log_probs: Unused flag kept for API compatibility.
            post_train: Whether to use post-training thresholded inclusion.

        Returns:
            Output logits before the final activation.
        """
        del sample
        del calculate_log_probs

        return self._forward_logits(
            x=x,
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

    def predict(self, x: Tensor, threshold: float = 0.5) -> Tensor:
        """Predict classes or regression values.

        Args:
            x: Input tensor.
            threshold: Threshold used for binary classification.

        Returns:
            Predicted class labels or regression outputs.
        """
        self.eval()

        with torch.no_grad():
            out = self(x, ensemble=False)

            if self.classification:
                if self.multiclass:
                    return out.argmax(dim=1)
                return (out >= threshold).float().view(-1)

            return out.view(-1)

    def predict_proba(self, x: Tensor) -> Tensor:
        """Predict class probabilities.

        Args:
            x: Input tensor.

        Returns:
            Predicted probabilities.
        """
        self.eval()

        with torch.no_grad():
            out = self(x, ensemble=False)
            return torch.exp(out) if self.multiclass else out


class InputSkipFlowNetwork(BayesianNetworkFlow):
    """Alias for the flow-based network with input skip connections."""

    pass