# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Continuous multipath channel impulse-response generation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from .grid import ResourceGrid


_PROFILES: dict[str, dict[str, Any]] = {
    "continuous-p6": {
        "type": "continuous",
        "num_paths": 6,
        "l_max": 14.0,
        "k_sym_max": 0.023,
        "los_first_tap": True,
    },
}


def available_profiles() -> tuple[str, ...]:
    return tuple(sorted(_PROFILES))


@dataclass(frozen=True)
class CIR:
    """Batched continuous-delay/continuous-Doppler impulse response."""

    a: torch.Tensor
    tau: torch.Tensor
    nu: torch.Tensor

    @property
    def batch_size(self) -> int:
        return int(self.a.shape[0])

    @property
    def num_paths(self) -> int:
        return int(self.a.shape[-1])


class MultipathChannel:
    """Generate iid block-fading multipath CIRs for an OTFS frame.

    The default profiles draw continuous delays and Dopplers. Delays
    are uniform on ``[0, l_max]`` delay bins, Dopplers are uniform on
    ``[-k_sym_max, k_sym_max]`` times the subcarrier spacing, and tap
    gains are iid ``CN(0, 1/P)``. If ``los_first_tap`` is enabled, the
    first path is anchored at zero delay and zero Doppler.
    """

    def __init__(
        self,
        grid: ResourceGrid,
        *,
        profile: str = "continuous-p6",
        num_paths: int | None = None,
        l_max: float | None = None,
        k_sym_max: float | None = None,
        los_first_tap: bool | None = None,
    ) -> None:
        self.grid = grid
        key = profile.lower()
        if key not in _PROFILES:
            raise KeyError(
                f"Unknown profile {profile!r}. Available profiles: {available_profiles()}"
            )
        spec = _PROFILES[key]
        if spec["type"] != "continuous":
            raise ValueError(f"Unsupported profile type {spec['type']!r}")
        self.profile = key
        self.num_paths = int(num_paths if num_paths is not None else spec["num_paths"])
        self.l_max = float(l_max if l_max is not None else spec["l_max"])
        self.k_sym_max = float(
            k_sym_max if k_sym_max is not None else spec["k_sym_max"]
        )
        self.los_first_tap = bool(
            los_first_tap if los_first_tap is not None else spec["los_first_tap"]
        )
        if self.num_paths <= 0:
            raise ValueError("num_paths must be positive")
        if self.l_max < 0.0:
            raise ValueError("l_max must be non-negative")
        if self.k_sym_max < 0.0 or not math.isfinite(self.k_sym_max):
            raise ValueError("k_sym_max must be finite and non-negative")

    def __call__(self, batch_size: int) -> CIR:
        b = int(batch_size)
        p = self.num_paths
        dtype = self.grid.dtype
        cdtype = self.grid.cdtype
        device = self.grid.device

        tau_bins = torch.rand((b, p), dtype=dtype, device=device) * self.l_max
        k_sym = (
            torch.rand((b, p), dtype=dtype, device=device) * 2.0 - 1.0
        ) * self.k_sym_max
        if self.los_first_tap:
            tau_bins[:, 0] = 0.0
            k_sym[:, 0] = 0.0

        tau = tau_bins * self.grid.delay_bin_spacing
        nu = k_sym * self.grid.subcarrier_spacing

        sigma = 1.0 / math.sqrt(2.0 * p)
        re = torch.randn((b, p), dtype=dtype, device=device) * sigma
        im = torch.randn((b, p), dtype=dtype, device=device) * sigma
        a = torch.complex(re, im).to(cdtype)
        return CIR(a=a, tau=tau, nu=nu)
