# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Analytical rectangular-CP OTFS input-output relationship."""

from __future__ import annotations

import math

import torch

from .channel import CIR
from .channel_view import SparseDDChannel
from .grid import ResourceGrid


def dirichlet_kernel(xi: torch.Tensor, period: int) -> torch.Tensor:
    """Evaluate ``1/period * sum_n exp(j 2*pi*n*xi/period)``."""

    eps = torch.finfo(xi.dtype).eps
    pole_tol = max(64.0 * eps, 1e-12)
    period_f = float(period)
    residual = xi - period_f * torch.round(xi / period_f)
    is_pole = residual.abs() < pole_tol

    zero = torch.zeros_like(xi)
    num = torch.exp(torch.complex(zero, 2.0 * math.pi * xi)) - 1.0
    den = period_f * (
        torch.exp(torch.complex(zero, 2.0 * math.pi * xi / period_f)) - 1.0
    )
    den = torch.where(is_pole, torch.ones_like(den), den)
    return torch.where(is_pole, torch.ones_like(num), num / den)


def delay_doppler_windows(
    M: int,
    N: int,
    *,
    delay_window: int,
    doppler_window: int,
    full_period: bool,
    dtype: torch.dtype,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return real and integer delay/Doppler Dirichlet windows."""

    if full_period:
        gamma_tau = torch.arange(-(M // 2), (M + 1) // 2, dtype=dtype, device=device)
        gamma_nu = torch.arange(-(N // 2), (N + 1) // 2, dtype=dtype, device=device)
    else:
        gamma_tau = torch.arange(
            -int(delay_window), int(delay_window) + 1, dtype=dtype, device=device
        )
        gamma_nu = torch.arange(
            -int(doppler_window), int(doppler_window) + 1, dtype=dtype, device=device
        )
    return gamma_tau, gamma_nu, gamma_tau.to(torch.long), gamma_nu.to(torch.long)


class RectCPOtfsIO:
    """CP-aware continuous-delay/continuous-Doppler OTFS DD IO model.

    The physical receive vector is generated from the analytical DD
    relation. Receiver CSI is exposed as a sparse DD channel, and dense
    views are obtained by scattering that same sparse object.
    """

    def __init__(
        self,
        grid: ResourceGrid,
        *,
        delay_window: int = 5,
        doppler_window: int = 5,
        full_period: bool = True,
    ) -> None:
        self.grid = grid
        self.delay_window = int(delay_window)
        self.doppler_window = int(doppler_window)
        self.full_period = bool(full_period)
        if self.delay_window < 0 or self.doppler_window < 0:
            raise ValueError("delay_window and doppler_window must be non-negative")

    def apply(
        self,
        x_dd: torch.Tensor,
        cir: CIR,
        *,
        noise_variance: torch.Tensor | float | None = None,
    ) -> torch.Tensor:
        """Apply the analytical OTFS channel and optionally add AWGN."""

        y = self._apply_fft(cir, x_dd)
        if noise_variance is None:
            return y
        no = torch.as_tensor(
            noise_variance, dtype=self.grid.dtype, device=x_dd.device
        )
        sigma = torch.sqrt(no / 2.0).to(y.dtype)
        noise = sigma * torch.complex(torch.randn_like(y.real), torch.randn_like(y.imag))
        return y + noise

    def sparse(self, cir: CIR, *, top_k: int) -> SparseDDChannel:
        """Build the common receiver sparse DD channel view."""

        if top_k <= 0:
            raise ValueError("top_k must be positive")
        h_vals, col_idx = self._compact_topk_sparse(cir, top_k=top_k)
        return SparseDDChannel(h_vals=h_vals, col_idx=col_idx, M=self.grid.M, N=self.grid.N)

    def _terms(self, cir: CIR) -> dict[str, torch.Tensor | int]:
        a, tau, nu = cir.a, cir.tau, cir.nu
        if a.ndim != 2:
            raise ValueError("CIR tensors must have shape (batch, paths)")

        M, N = self.grid.M, self.grid.N
        mn = M * N
        dtype = self.grid.dtype
        cdtype = self.grid.cdtype
        gamma_g = 1.0 + float(self.grid.cp_samples) / float(M)

        gamma_tau, gamma_nu, gamma_tau_int, gamma_nu_int = delay_doppler_windows(
            M,
            N,
            delay_window=self.delay_window,
            doppler_window=self.doppler_window,
            full_period=self.full_period,
            dtype=dtype,
            device=self.grid.device,
        )
        q_tau = int(gamma_tau.shape[0])
        q_nu = int(gamma_nu.shape[0])

        tau_bin = tau.to(dtype) * float(M) / float(self.grid.delay_period)
        l_int_signed = tau_bin.round().long()
        eps = tau_bin - l_int_signed.to(dtype)
        l_int = l_int_signed % M

        nu_bin = nu.to(dtype) * float(N) / float(self.grid.doppler_period)
        nu_bin_cp = gamma_g * nu_bin
        k_int_cp_signed = nu_bin_cp.round().long()
        kappa_cp = nu_bin_cp - k_int_cp_signed.to(dtype)
        k_int_cp = k_int_cp_signed % N

        d_tau = dirichlet_kernel(
            gamma_tau.view(1, 1, q_tau) - eps.unsqueeze(-1),
            M,
        ).to(cdtype)
        d_nu = dirichlet_kernel(
            gamma_nu.view(1, 1, q_nu) + kappa_cp.unsqueeze(-1),
            N,
        ).to(cdtype)
        path_phase = torch.exp(
            torch.complex(
                torch.zeros_like(tau_bin),
                -2.0 * math.pi * tau_bin * nu_bin / float(mn),
            )
        ).to(cdtype)
        return {
            "M": M,
            "N": N,
            "mn": mn,
            "q_tau": q_tau,
            "q_nu": q_nu,
            "gamma_tau_int": gamma_tau_int,
            "gamma_nu_int": gamma_nu_int,
            "l_int": l_int,
            "k_int_cp": k_int_cp,
            "nu_bin": nu_bin,
            "d_tau": d_tau,
            "d_nu": d_nu,
            "path_phase": path_phase,
            "cdtype": cdtype,
        }

    def _apply_fft(self, cir: CIR, x_dd: torch.Tensor) -> torch.Tensor:
        terms = self._terms(cir)
        M = int(terms["M"])
        N = int(terms["N"])
        mn = int(terms["mn"])
        q_tau = int(terms["q_tau"])
        q_nu = int(terms["q_nu"])
        gamma_tau_int = terms["gamma_tau_int"]
        gamma_nu_int = terms["gamma_nu_int"]
        l_int = terms["l_int"]
        k_int_cp = terms["k_int_cp"]
        nu_bin = terms["nu_bin"]
        d_tau = terms["d_tau"]
        d_nu = terms["d_nu"]
        path_phase = terms["path_phase"]
        cdtype = terms["cdtype"]

        a = cir.a.to(cdtype)
        b, p = a.shape
        kernel = d_tau.unsqueeze(-1) * d_nu.unsqueeze(-2)
        delay_shift = (l_int.unsqueeze(-1) + gamma_tau_int.view(1, 1, q_tau)) % M
        doppler_shift = (k_int_cp.unsqueeze(-1) - gamma_nu_int.view(1, 1, q_nu)) % N
        shift_idx = (
            delay_shift.unsqueeze(-1) * N + doppler_shift.unsqueeze(-2)
        ).reshape(b, p, q_tau * q_nu)

        g_flat = torch.zeros(b, p, mn, dtype=cdtype, device=a.device)
        g_flat.scatter_add_(-1, shift_idx, kernel.reshape(b, p, q_tau * q_nu))
        g = g_flat.reshape(b, p, M, N)

        x = x_dd.to(cdtype).reshape(b, M, N)
        z = torch.fft.ifft2(
            torch.fft.fft2(g, dim=(-2, -1))
            * torch.fft.fft2(x, dim=(-2, -1)).unsqueeze(1),
            dim=(-2, -1),
        )

        l_grid = torch.arange(M, dtype=self.grid.dtype, device=a.device)
        row_phase_arg = 2.0 * math.pi * nu_bin.unsqueeze(-1) * l_grid.view(1, 1, M)
        row_phase_arg = row_phase_arg / float(mn)
        row_phase = torch.exp(
            torch.complex(torch.zeros_like(row_phase_arg), row_phase_arg)
        ).to(cdtype)
        coeff = a * path_phase
        y = (coeff.view(b, p, 1, 1) * row_phase.unsqueeze(-1) * z).sum(dim=1)
        return y.reshape(b, mn)

    def _compact_topk_sparse(self, cir: CIR, *, top_k: int) -> tuple[torch.Tensor, torch.Tensor]:
        terms = self._terms(cir)
        M = int(terms["M"])
        N = int(terms["N"])
        mn = int(terms["mn"])
        q_tau = int(terms["q_tau"])
        q_nu = int(terms["q_nu"])
        gamma_tau_int = terms["gamma_tau_int"]
        gamma_nu_int = terms["gamma_nu_int"]
        l_int = terms["l_int"]
        k_int_cp = terms["k_int_cp"]
        nu_bin = terms["nu_bin"]
        d_tau = terms["d_tau"]
        d_nu = terms["d_nu"]
        path_phase = terms["path_phase"]
        cdtype = terms["cdtype"]

        a = cir.a.to(cdtype)
        b, p = a.shape
        h_kernel = (
            (a * path_phase).unsqueeze(-1).unsqueeze(-1)
            * d_tau.unsqueeze(-1)
            * d_nu.unsqueeze(-2)
        )
        h_flat = h_kernel.reshape(b, p * q_tau * q_nu)

        path_base = torch.arange(p, dtype=torch.long, device=a.device).view(1, p, 1, 1)
        path_idx = path_base.expand(b, p, q_tau, q_nu).reshape(b, -1)
        nu_flat = nu_bin.gather(-1, path_idx)

        delay_shift = (l_int.unsqueeze(-1) + gamma_tau_int.view(1, 1, q_tau)) % M
        delay_shift = delay_shift.unsqueeze(-1).expand(b, p, q_tau, q_nu).reshape(b, -1)
        doppler_delta = (k_int_cp.unsqueeze(-1) - gamma_nu_int.view(1, 1, q_nu)) % N
        doppler_delta = doppler_delta.unsqueeze(-2).expand(b, p, q_tau, q_nu).reshape(b, -1)

        l_grid_long = torch.arange(M, dtype=torch.long, device=a.device)
        l_grid = l_grid_long.to(self.grid.dtype)
        row_phase_arg = 2.0 * math.pi * nu_flat.unsqueeze(1) * l_grid.view(1, M, 1)
        row_phase_arg = row_phase_arg / float(mn)
        row_phase = torch.exp(
            torch.complex(torch.zeros_like(row_phase_arg), row_phase_arg)
        ).to(cdtype)
        src = row_phase * h_flat.unsqueeze(1)

        l_prime = (l_grid_long.view(1, M, 1) - delay_shift.unsqueeze(1)) % M
        scatter_idx = (
            l_grid_long.view(1, M, 1) * mn
            + l_prime * N
            + doppler_delta.unsqueeze(1)
        )
        h_compact_flat = torch.zeros(b, M * mn, dtype=cdtype, device=a.device)
        h_compact_flat.scatter_add_(
            -1,
            scatter_idx.reshape(b, -1),
            src.reshape(b, -1),
        )
        h_compact = h_compact_flat.reshape(b, M, mn)

        k_eff = min(int(top_k), mn)
        _, topk_flat = torch.topk(h_compact.abs(), k=k_eff, dim=-1, sorted=False)
        gamma_lp = topk_flat // N
        gamma_dk = topk_flat % N
        h_vals_per_l = h_compact.gather(-1, topk_flat)

        k_grid = torch.arange(N, dtype=torch.long, device=a.device)
        in_kp = (k_grid.view(1, 1, N, 1) - gamma_dk.view(b, M, 1, k_eff)) % N
        col = gamma_lp.view(b, M, 1, k_eff) * N + in_kp
        col_idx = col.reshape(b, mn, k_eff).contiguous()
        h_vals = (
            h_vals_per_l.view(b, M, 1, k_eff)
            .expand(b, M, N, k_eff)
            .reshape(b, mn, k_eff)
            .contiguous()
        )
        return h_vals, col_idx
