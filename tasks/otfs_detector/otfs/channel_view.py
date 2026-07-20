# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Channel views passed to OTFS equalisers."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .channel import CIR


@dataclass(frozen=True)
class SparseDDChannel:
    """Per-row sparse DD channel representation.

    ``h_vals[..., r, e]`` is the channel coefficient from input column
    ``col_idx[..., r, e]`` to output row ``r``.
    """

    h_vals: torch.Tensor
    col_idx: torch.Tensor
    M: int
    N: int

    @property
    def num_resource_elements(self) -> int:
        return self.M * self.N

    @property
    def top_k(self) -> int:
        return int(self.h_vals.shape[-1])

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        mn = self.num_resource_elements
        if x.shape[-1] != mn:
            raise ValueError(f"x has last dimension {x.shape[-1]}, expected {mn}")
        x_edges = torch.gather(
            x.unsqueeze(-2).expand(*x.shape[:-1], mn, mn),
            -1,
            self.col_idx,
        )
        return (self.h_vals * x_edges).sum(dim=-1)

    def to_dense(self) -> "DenseDDChannel":
        mn = self.num_resource_elements
        h = torch.zeros(
            *self.h_vals.shape[:-2],
            mn,
            mn,
            dtype=self.h_vals.dtype,
            device=self.h_vals.device,
        )
        h.scatter_add_(-1, self.col_idx, self.h_vals)
        return DenseDDChannel(H=h, M=self.M, N=self.N)


@dataclass(frozen=True)
class DenseDDChannel:
    """Dense DD channel matrix derived from a sparse channel view."""

    H: torch.Tensor
    M: int
    N: int

    @property
    def num_resource_elements(self) -> int:
        return self.M * self.N

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("...ij,...j->...i", self.H, x)


@dataclass(frozen=True)
class ChannelView:
    """Receiver-side channel information for one frame batch."""

    cir: CIR
    sparse: SparseDDChannel | None = None
    dense: DenseDDChannel | None = None
