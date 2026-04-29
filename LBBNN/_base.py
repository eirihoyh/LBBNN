from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class BayesianNetworkBase(nn.Module):
    """Base class providing shared inference methods for Bayesian networks.

    Subclasses must set ``self.classification`` and ``self.multiclass`` in
    their ``__init__`` before any method here is called.
    """

    classification: bool
    multiclass: bool

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
