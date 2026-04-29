from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.typing import NDArray
from torch import Tensor


class PropagateFlow(nn.Module):
    """Container for a sequence of normalizing flow transforms."""

    def __init__(
        self,
        transform: str,
        dim: int,
        num_transforms: int,
        iaf_h_sizes: Sequence[int] = (250, 250),
        rnvp_h_sizes: Sequence[int] = (10, 10),
    ) -> None:
        """Initialize a stack of flow transforms.

        Args:
            transform: Name of the transform type, either ``"IAF"`` or
                ``"RNVP"``.
            dim: Input and output dimensionality of each transform.
            num_transforms: Number of transforms to apply in sequence.
            iaf_h_sizes: Hidden layer sizes for IAF MADE networks.
                Only used when ``transform == "IAF"``. Smaller values
                make the flow much cheaper at the cost of expressivity.
            rnvp_h_sizes: Hidden layer sizes for RNVP coupling networks.
                Only used when ``transform == "RNVP"``.

        Returns:
            None.
        """
        super().__init__()

        if transform == "IAF":
            self.transforms = nn.ModuleList(
                [IAF(dim, h_sizes=iaf_h_sizes) for _ in range(num_transforms)]
            )
        elif transform == "RNVP":
            self.transforms = nn.ModuleList(
                [RNVP(dim, h_sizes=rnvp_h_sizes) for _ in range(num_transforms)]
            )
        else:
            raise ValueError(f"Transform not implemented: {transform}")

    def forward(self, z: Tensor) -> tuple[Tensor, Tensor]:
        """Apply all transforms in sequence.

        Args:
            z: Input tensor to transform.

        Returns:
            A tuple containing the transformed tensor and the summed
            log-determinant.
        """
        logdet = 0

        for transform in self.transforms:
            z = transform(z)
            logdet += transform.log_det()

        return z, logdet


class MLP(nn.Sequential):
    """Simple multilayer perceptron with LeakyReLU activations."""

    def __init__(
        self,
        *layer_sizes: int,
        leaky_a: float = 0.2,
    ) -> None:
        """Initialize the MLP.

        Args:
            *layer_sizes: Sizes of the layers in order.
            leaky_a: Negative slope used in LeakyReLU activations.

        Returns:
            None.
        """
        layers: list[nn.Module] = []

        for in_size, out_size in zip(layer_sizes, layer_sizes[1:]):
            layers.append(nn.Linear(in_size, out_size))
            layers.append(nn.LeakyReLU(leaky_a))

        # Remove the final activation.
        super().__init__(*layers[:-1])


class RNVP(nn.Module):
    """Real-valued non-volume preserving coupling transform."""

    def __init__(
        self,
        dim: int,
        h_sizes: Sequence[int] = (10, 10),
    ) -> None:
        """Initialize the RNVP transform.

        Args:
            dim: Input and output dimensionality.
            h_sizes: Hidden layer sizes for the internal network.

        Returns:
            None.
        """
        super().__init__()

        self.network = MLP(*([dim] + list(h_sizes)))
        self.t = nn.Linear(h_sizes[-1], dim)
        self.s = nn.Linear(h_sizes[-1], dim)

        self.gate: Tensor | None = None
        self.mask: Tensor | None = None

    def forward(self, z: Tensor) -> Tensor:
        """Apply the RNVP transform.

        Args:
            z: Input tensor.

        Returns:
            Transformed tensor.
        """
        self.mask = torch.bernoulli(0.5 * torch.ones_like(z))
        z1 = (1 - self.mask) * z
        z2 = self.mask * z

        hidden = self.network(z2)
        shift = self.t(hidden)
        scale = self.s(hidden)

        self.gate = torch.sigmoid(scale)
        x = z1 * (self.gate + (1 - self.gate) * shift) + z2

        return x

    def log_det(self) -> Tensor:
        """Compute the log-determinant of the RNVP transform.

        Args:
            None.

        Returns:
            Log-determinant tensor.
        """
        if self.mask is None or self.gate is None:
            raise RuntimeError("Call forward() before log_det().")

        return ((1 - self.mask) * self.gate.log()).sum(dim=-1)


class MaskedLinear(nn.Linear):
    """Linear layer with a fixed binary mask applied to the weights."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ) -> None:
        """Initialize the masked linear layer.

        Args:
            in_features: Number of input features.
            out_features: Number of output features.
            bias: Whether to include a bias term.

        Returns:
            None.
        """
        super().__init__(in_features, out_features, bias)
        self.register_buffer("mask", torch.ones(out_features, in_features))

    def set_mask(self, mask: NDArray[np.bool_] | NDArray[np.integer]) -> None:
        """Set the binary weight mask.

        Args:
            mask: NumPy array containing the mask values.

        Returns:
            None.
        """
        mask_tensor = torch.from_numpy(mask.astype(np.uint8).T)
        self.mask.data.copy_(mask_tensor)

    def forward(self, input: Tensor) -> Tensor:
        """Apply the masked linear transformation.

        Args:
            input: Input tensor.

        Returns:
            Output tensor after the masked linear operation.
        """
        return F.linear(input, self.mask * self.weight, self.bias)


class MADE(nn.Module):
    """Masked autoencoder used for autoregressive flow transforms."""

    def __init__(
        self,
        nin: int,
        hidden_sizes: Sequence[int],
        nout: int,
        num_masks: int = 1,
        natural_ordering: bool = False,
    ) -> None:
        """Initialize the MADE network.

        Args:
            nin: Number of input features.
            hidden_sizes: Sizes of hidden layers.
            nout: Number of output features.
            num_masks: Number of masks to cycle through.
            natural_ordering: Whether to use natural input ordering.

        Returns:
            None.
        """
        super().__init__()

        self.nin = nin
        self.nout = nout
        self.hidden_sizes = list(hidden_sizes)

        if self.nout % self.nin != 0:
            raise ValueError("nout must be a multiple of nin.")

        modules: list[nn.Module] = []
        hidden_structure = [nin] + self.hidden_sizes + [nout]

        for in_size, out_size in zip(hidden_structure, hidden_structure[1:]):
            modules.extend([MaskedLinear(in_size, out_size), nn.Sigmoid()])

        modules.pop()
        self.net = nn.Sequential(*modules)

        self.natural_ordering = natural_ordering
        self.num_masks = num_masks
        self.seed = 0
        self.m: dict[int, NDArray[np.int_]] = {}

        self.update_masks()

    def update_masks(self) -> None:
        """Update the autoregressive masks.

        Args:
            None.

        Returns:
            None.
        """
        if self.m and self.num_masks == 1:
            return

        num_hidden_layers = len(self.hidden_sizes)
        rng = np.random.RandomState(self.seed)
        self.seed = (self.seed + 1) % self.num_masks

        if self.natural_ordering:
            self.m[-1] = np.arange(self.nin)
        else:
            self.m[-1] = rng.permutation(self.nin)

        for layer_idx in range(num_hidden_layers):
            self.m[layer_idx] = rng.randint(
                self.m[layer_idx - 1].min(),
                self.nin - 1,
                size=self.hidden_sizes[layer_idx],
            )

        masks = [
            self.m[layer_idx - 1][:, None] <= self.m[layer_idx][None, :]
            for layer_idx in range(num_hidden_layers)
        ]
        masks.append(self.m[num_hidden_layers - 1][:, None] < self.m[-1][None, :])

        if self.nout > self.nin:
            k = int(self.nout / self.nin)
            masks[-1] = np.concatenate([masks[-1]] * k, axis=1)

        layers = [
            layer for layer in self.net.modules() if isinstance(layer, MaskedLinear)
        ]
        for layer, mask in zip(layers, masks):
            layer.set_mask(mask)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the MADE network.

        Args:
            x: Input tensor.

        Returns:
            Network output tensor.
        """
        return self.net(x)


class IAF(nn.Module):
    """Inverse autoregressive flow transform."""

    def __init__(
        self,
        dim: int,
        h_sizes: Sequence[int] = (250, 250),
    ) -> None:
        """Initialize the IAF transform.

        Args:
            dim: Input and output dimensionality.
            h_sizes: Hidden layer sizes for the MADE network.

        Returns:
            None.
        """
        super().__init__()

        self.net = MADE(nin=dim, hidden_sizes=h_sizes, nout=2 * dim)
        self.gate: Tensor | None = None

    def forward(self, x: Tensor) -> Tensor:
        """Apply the IAF transform.

        Args:
            x: Input tensor.

        Returns:
            Transformed tensor.
        """
        out = self.net(x)
        split_idx = out.shape[-1] // 2

        if out.dim() == 2:
            shift = out[:, :split_idx]
            scale = out[:, split_idx:]
        else:
            shift = out[:split_idx]
            scale = out[split_idx:]

        self.gate = torch.sigmoid(scale)
        return x * self.gate + (1 - self.gate) * shift

    def log_det(self) -> Tensor:
        """Compute the log-determinant of the IAF transform.

        Args:
            None.

        Returns:
            Log-determinant tensor.
        """
        if self.gate is None:
            raise RuntimeError("Call forward() before log_det().")

        return torch.log(self.gate).sum(dim=-1)
