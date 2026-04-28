from __future__ import annotations

from typing import Any, Callable, Protocol, runtime_checkable

import torch.nn as nn
from torch import Tensor


@runtime_checkable
class BayesianNet(Protocol):
    """Structural protocol satisfied by LRT- and FLOW-based LBBNN networks.

    Helper functions in `inspection`, `explain`, `training`, and `plotting`
    are typed against this protocol rather than a concrete class so that
    both network families (and any future variants) plug in without
    further changes.
    """

    linears: nn.ModuleList
    loss: Callable[..., Tensor]

    def kl(self) -> Tensor: ...

    def forward(self, x: Tensor, **kwargs: Any) -> Tensor: ...

    def forward_preact(self, x: Tensor, **kwargs: Any) -> Tensor: ...

    def __call__(self, *args: Any, **kwargs: Any) -> Tensor: ...

    def eval(self) -> "BayesianNet": ...

    def train(self, mode: bool = True) -> "BayesianNet": ...

    def zero_grad(self, set_to_none: bool = True) -> None: ...
