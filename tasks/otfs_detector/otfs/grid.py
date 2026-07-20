# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Resource-grid description for rectangular-CP OTFS simulations."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ResourceGrid:
    """Delay-Doppler grid and physical OFDM/CP timing parameters.

    Parameters
    ----------
    num_delay_bins:
        Number of delay bins, usually denoted by ``M``.
    num_doppler_bins:
        Number of Doppler bins, usually denoted by ``N``.
    subcarrier_spacing:
        OFDM subcarrier spacing in Hz.
    carrier_frequency:
        Carrier frequency in Hz. Stored for scenario bookkeeping.
    cp_samples:
        Cyclic-prefix length in samples. The benchmark assumes this is
        larger than the channel delay spread.
    precision:
        ``"single"`` or ``"double"``.
    device:
        Torch device string.
    """

    num_delay_bins: int
    num_doppler_bins: int
    subcarrier_spacing: float
    carrier_frequency: float = 0.0
    cp_samples: int = 0
    precision: str = "single"
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.num_delay_bins <= 0 or self.num_doppler_bins <= 0:
            raise ValueError("num_delay_bins and num_doppler_bins must be positive")
        if self.subcarrier_spacing <= 0.0:
            raise ValueError("subcarrier_spacing must be positive")
        if self.cp_samples < 0:
            raise ValueError("cp_samples must be non-negative")
        if self.precision not in ("single", "double"):
            raise ValueError("precision must be 'single' or 'double'")

    @property
    def M(self) -> int:
        return int(self.num_delay_bins)

    @property
    def N(self) -> int:
        return int(self.num_doppler_bins)

    @property
    def num_resource_elements(self) -> int:
        return self.M * self.N

    @property
    def delay_period(self) -> float:
        return 1.0 / float(self.subcarrier_spacing)

    @property
    def doppler_period(self) -> float:
        return float(self.subcarrier_spacing)

    @property
    def delay_bin_spacing(self) -> float:
        return self.delay_period / float(self.M)

    @property
    def dtype(self) -> torch.dtype:
        return torch.float32 if self.precision == "single" else torch.float64

    @property
    def cdtype(self) -> torch.dtype:
        return torch.complex64 if self.precision == "single" else torch.complex128
