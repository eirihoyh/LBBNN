from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .layers import BayesianLinearLRT


class BayesianNetworkLRT(nn.Module):
    """Bayesian neural network with input skip connections."""

    def __init__(
        self,
        dim: int,
        p: int,
        hidden_layers: int,
        a_prior: float = 0.05,
        sigma_prior: float = 2.5,
        mu_prior: float = 0.0,
        weight_mu_init_range: tuple[float, float] = (-1.2, 1.2),
        weight_rho_init_mean: float = -9.0,
        classification: bool = True,
        n_classes: int = 1,
        act_func: Callable[[Tensor], Tensor] = torch.relu,
        lower_init_alpha: float = 0.30,
        upper_init_alpha: float = 0.49,
        high_init_covariate_prob: bool = False,
        custom_loss: bool | Callable[[Tensor, Tensor], Tensor] = False,
    ) -> None:
        """Initialize the Bayesian network.

        NOTE: First column of input is assmued to only consist of ones
            (works as the intercept/bias).

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
            classification: Whether the task is classification.
            n_classes: Number of output classes.
            act_func: Activation function used in the hidden layers.
            lower_init_alpha: Lower bound for inclusion probability initialization.
            upper_init_alpha: Upper bound for inclusion probability initialization.
            high_init_covariate_prob: Whether to initialize skip covariates with
                high inclusion probability.

        Returns:
            None.
        """
        super().__init__()

        self.p = p
        self.classification = classification
        self.multiclass = n_classes > 1
        self.act = act_func

        n_skip_features = p if high_init_covariate_prob else None

        layer_kwargs = dict(
            a_prior=a_prior,
            sigma_prior=sigma_prior,
            mu_prior=mu_prior,
            weight_mu_init_range=weight_mu_init_range,
            weight_rho_init_mean=weight_rho_init_mean,
            lower_init_alpha=lower_init_alpha,
            upper_init_alpha=upper_init_alpha,
            n_skip_features=n_skip_features,
        )

        self.linears = nn.ModuleList(
            [BayesianLinearLRT(p, dim, **layer_kwargs)]
        )

        self.linears.extend(
            [
                BayesianLinearLRT(dim + p, dim, **layer_kwargs)
                for _ in range(hidden_layers - 1)
            ]
        )

        self.linears.append(
            BayesianLinearLRT(dim + p, n_classes, **layer_kwargs)
        )

        if custom_loss:
            self.loss = custom_loss
        else: 
            if classification and self.multiclass:
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
        """Compute output logits before the final output transform.

        Args:
            x: Input tensor.
            sample: Whether to sample weights during deterministic inference.
            ensemble: Whether to use ensemble-style inference.
            calculate_log_probs: Whether to update KL-related quantities.
            post_train: Whether to use post-training thresholded inclusion.

        Returns:
            The output logits tensor.
        """
        x_input = x.view(-1, self.p)
        x_hidden = self.act(
            self.linears[0](
                x_input,
                ensemble=ensemble,
                sample=sample,
                calculate_log_probs=calculate_log_probs,
                post_train=post_train,
            )
        )

        for layer in self.linears[1:-1]:
            x_hidden = self.act(
                layer(
                    torch.cat((x_hidden, x_input), dim=1),
                    ensemble=ensemble,
                    sample=sample,
                    calculate_log_probs=calculate_log_probs,
                    post_train=post_train,
                )
            )

        return self.linears[-1](
            torch.cat((x_hidden, x_input), dim=1),
            ensemble=ensemble,
            sample=sample,
            calculate_log_probs=calculate_log_probs,
            post_train=post_train,
        )

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
            sample: Whether to sample weights during deterministic inference.
            ensemble: Whether to use ensemble-style inference.
            calculate_log_probs: Whether to update KL-related quantities.
            post_train: Whether to use post-training thresholded inclusion.

        Returns:
            Network outputs as probabilities or raw values depending on the
            task type.
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
        """Run a forward pass and return pre-activation outputs.

        Args:
            x: Input tensor.
            sample: Whether to sample weights during deterministic inference.
            ensemble: Whether to use ensemble-style inference.
            calculate_log_probs: Whether to update KL-related quantities.
            post_train: Whether to use post-training thresholded inclusion.

        Returns:
            The output logits tensor before the final output transform.
        """
        return self._forward_logits(
            x=x,
            sample=sample,
            ensemble=ensemble,
            calculate_log_probs=calculate_log_probs,
            post_train=post_train,
        )

    def kl(self) -> Tensor:
        """Return the total KL divergence across all layers.

        Args:
            None.

        Returns:
            The summed KL divergence tensor.
        """
        return sum(layer.kl for layer in self.linears)

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
            Predicted probabilities for classification tasks.
        """
        self.eval()

        with torch.no_grad():
            out = self(x, ensemble=False)
            return torch.exp(out) if self.multiclass else out


class InputSkipLRTNetwork(BayesianNetworkLRT):
    """Alias for the Bayesian network with input skip connections."""

    pass