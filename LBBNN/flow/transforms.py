from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class PropagateFlow(nn.Module):
    def __init__(self, transform: str, dim: int, num_transforms: int) -> None:
        super().__init__()
        if transform == 'IAF':
            self.transforms = nn.ModuleList([IAF(dim) for _ in range(num_transforms)])
        elif transform == 'RNVP':
            self.transforms = nn.ModuleList([RNVP(dim) for _ in range(num_transforms)])
        else:
            raise ValueError(f'Transform not implemented: {transform}')

    def forward(self, z: torch.Tensor):
        logdet = 0.0
        for transform in self.transforms:
            z = transform(z)
            logdet = logdet + transform.log_det()
        return z, logdet

class MLP(nn.Sequential):
    def __init__(self, *layer_sizes, leaky_a: float = 0.2):
        layers = []
        for s1, s2 in zip(layer_sizes, layer_sizes[1:]):
            layers.append(nn.Linear(s1, s2))
            layers.append(nn.LeakyReLU(leaky_a))
        super().__init__(*layers[:-1])

class RNVP(nn.Module):
    def __init__(self, dim: int, h_sizes=(10, 10)) -> None:
        super().__init__()
        self.network = MLP(*([dim] + list(h_sizes)))
        self.t = nn.Linear(h_sizes[-1], dim)
        self.s = nn.Linear(h_sizes[-1], dim)
        self.gate = 0
        self.mask = 0

    def forward(self, z: torch.Tensor):
        self.mask = torch.bernoulli(0.5 * torch.ones_like(z))
        z1, z2 = (1 - self.mask) * z, self.mask * z
        y = self.network(z2)
        shift, scale = self.t(y), self.s(y)
        self.gate = torch.sigmoid(scale)
        x = z1 * (self.gate + (1 - self.gate) * shift) + z2
        return x

    def log_det(self):
        return ((1 - self.mask) * self.gate.log()).sum(-1)

class MaskedLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__(in_features, out_features, bias)
        self.register_buffer('mask', torch.ones(out_features, in_features))

    def set_mask(self, mask: np.ndarray):
        self.mask.data.copy_(torch.from_numpy(mask.astype(np.uint8).T))

    def forward(self, input: torch.Tensor):
        return F.linear(input, self.mask * self.weight, self.bias)

class MADE(nn.Module):
    def __init__(self, nin: int, hidden_sizes, nout: int, num_masks: int = 1, natural_ordering: bool = False):
        super().__init__()
        self.nin = nin
        self.nout = nout
        self.hidden_sizes = list(hidden_sizes)
        assert self.nout % self.nin == 0
        modules = []
        hs = [nin] + self.hidden_sizes + [nout]
        for h0, h1 in zip(hs, hs[1:]):
            modules.extend([MaskedLinear(h0, h1), nn.Sigmoid()])
        modules.pop()
        self.net = nn.Sequential(*modules)
        self.natural_ordering = natural_ordering
        self.num_masks = num_masks
        self.seed = 0
        self.m = {}
        self.update_masks()

    def update_masks(self):
        if self.m and self.num_masks == 1:
            return
        L = len(self.hidden_sizes)
        rng = np.random.RandomState(self.seed)
        self.seed = (self.seed + 1) % self.num_masks
        self.m[-1] = np.arange(self.nin) if self.natural_ordering else rng.permutation(self.nin)
        for l in range(L):
            self.m[l] = rng.randint(self.m[l - 1].min(), self.nin - 1, size=self.hidden_sizes[l])
        masks = [self.m[l - 1][:, None] <= self.m[l][None, :] for l in range(L)]
        masks.append(self.m[L - 1][:, None] < self.m[-1][None, :])
        if self.nout > self.nin:
            k = int(self.nout / self.nin)
            masks[-1] = np.concatenate([masks[-1]] * k, axis=1)
        layers = [layer for layer in self.net.modules() if isinstance(layer, MaskedLinear)]
        for layer, mask in zip(layers, masks):
            layer.set_mask(mask)

    def forward(self, x: torch.Tensor):
        return self.net(x)

class IAF(nn.Module):
    def __init__(self, dim: int, h_sizes=(250, 250)) -> None:
        super().__init__()
        self.net = MADE(nin=dim, hidden_sizes=h_sizes, nout=2 * dim)
        self.gate = None

    def forward(self, x: torch.Tensor):
        out = self.net(x)
        first_half = int(out.shape[-1] / 2)
        if out.dim() == 2:
            shift = out[:, :first_half]
            scale = out[:, first_half:]
        else:
            shift = out[:first_half]
            scale = out[first_half:]
        self.gate = torch.sigmoid(scale)
        return x * self.gate + (1 - self.gate) * shift

    def log_det(self):
        return torch.log(self.gate).sum(-1)
