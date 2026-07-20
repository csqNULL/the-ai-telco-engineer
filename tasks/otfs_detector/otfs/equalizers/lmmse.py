# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Dense DD-domain LMMSE equalizer."""

from __future__ import annotations

import torch
from sionna.phy.mapping import Demapper

from otfs._device import resolve_device
from otfs.channel_view import ChannelView
from otfs.equalizers.base import Equalizer


class LMMSE(Equalizer):
    name = "LMMSE"
    channel_form = "dense"

    def __init__(
        self,
        grid,
        *,
        num_bits_per_symbol: int = 4,
        precision: str = "single",
        device: str = "cuda:0",
        **_unused,
    ) -> None:
        device = resolve_device(device)
        self._mn = grid.num_resource_elements
        self._precision = precision
        self._dtype = torch.float32 if precision == "single" else torch.float64
        self._cdtype = torch.complex64 if precision == "single" else torch.complex128
        self._device = device
        self._eye = torch.eye(self._mn, dtype=self._cdtype, device=device).unsqueeze(0)
        self._demapper = Demapper(
            "maxlog",
            "qam",
            num_bits_per_symbol=num_bits_per_symbol,
            precision=precision,
            device=device,
        )

    @torch.no_grad()
    def llr(self, y_dd: torch.Tensor, channel: ChannelView, no: torch.Tensor) -> torch.Tensor:
        if channel.dense is None:
            raise ValueError("LMMSE requires channel.dense")
        h = channel.dense.H.to(dtype=self._cdtype, device=self._device)
        y = y_dd.to(dtype=self._cdtype, device=self._device)
        no_t = torch.as_tensor(no, dtype=self._dtype, device=self._device)
        if no_t.dim() == 0:
            no_t = no_t.expand(y.shape[0])
        else:
            no_t = no_t.reshape(-1)
            if no_t.numel() == 1:
                no_t = no_t.expand(y.shape[0])

        h_h = h.conj().transpose(-2, -1)
        gram = torch.matmul(h_h, h)
        a = gram + no_t.to(self._cdtype).view(-1, 1, 1) * self._eye
        chol, _ = torch.linalg.cholesky_ex(a, check_errors=False)

        # White-noise LMMSE. For A = H^H H + N0 I,
        # diag(GH) = diag(A^-1 H^H H) = 1 - N0 diag(A^-1).
        # This avoids materializing the full filter G.
        rhs = torch.matmul(h_h, y.unsqueeze(-1))
        z = torch.linalg.solve_triangular(chol, rhs, upper=False)
        gy = torch.linalg.solve_triangular(chol.mH, z, upper=True).squeeze(-1)

        eye = self._eye.expand(y.shape[0], self._mn, self._mn)
        inv_l = torch.linalg.solve_triangular(chol, eye, upper=False)
        inv_diag = (inv_l.real.square() + inv_l.imag.square()).sum(dim=-2)
        d = 1.0 - no_t.view(-1, 1) * inv_diag

        x_hat = gy / d.to(self._cdtype)
        var = (1.0 / d - 1.0).real
        return self._demapper(x_hat, var)
