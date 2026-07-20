# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

r"""UAMP (unitary AMP) OTFS detector.

Faithful implementation of the rectangular-waveform UAMP detector from

    Z. Yuan, F. Liu, W. Yuan, Q. Guo, Z. Wang, J. Yuan, "Iterative Detection
    for Orthogonal Time Frequency Space Modulation with Unitary Approximate
    Message Passing," IEEE TWC, 2021 (arXiv:2008.06688), Algorithm 2 + the
    rectangular-waveform modifications of Section IV.

Model (rectangular waveform + CP, as used by this benchmark):

    y = H x + w,     H = (F_N \otimes I_M) H_T (F_N^H \otimes I_M),

with ``H_T = diag(H_1, ..., H_N)`` block-diagonal in the time-block domain.
Transforming by the Doppler-FFT decouples the channel into ``N`` independent
``M x M`` blocks (``y_f = H_T x_f``); these are exactly the per-Doppler blocks
``H_n``.  Each block is whitened by its own SVD ``H_n = U_n diag(d_n) V_n``
(the "unitary transform" that makes AMP robust for a non-i.i.d. channel), so in

    r = U^H y_f = diag(d) (V x_f) + noise

the transfer operator is diagonal and the scalar-variance UAMP recursion of
Algorithm 2 applies.  The noise precision ``epsilon`` is estimated inside the
loop (Lines 3-5 of Algorithm 2); we initialise it from the known ``no`` for a
fair comparison and let it adapt.

Only Lines 2 (``p``) and 9 (``q``) involve the channel; per the paper these are
the block products ``V x_f`` and ``V^H (d . s)`` plus a Doppler FFT/IFFT.
"""

from __future__ import annotations

import torch

from otfs import qam_constellation_with_labels
from otfs._device import resolve_device
from otfs.channel_view import ChannelView
from otfs.equalizers.base import Equalizer


class UAMP(Equalizer):
    name = "UAMP"
    channel_form = "sparse"
    top_k = 2048

    def __init__(
        self,
        grid,
        *,
        num_bits_per_symbol: int = 4,
        num_iters: int = 10,
        estimate_noise: bool = True,
        precision: str = "single",
        device: str = "cuda:0",
        **_unused,
    ):
        device = resolve_device(device)
        self.M = int(grid.M)
        self.N = int(grid.N)
        self.mn = self.M * self.N
        self.K = int(num_bits_per_symbol)
        self.num_iters = int(num_iters)
        self.estimate_noise = bool(estimate_noise)

        points, labels = qam_constellation_with_labels(
            self.K, precision=precision, device=device
        )
        self.points = points.view(1, 1, -1)  # (1,1,|A|)
        self.point_abs2 = (points.real.square() + points.imag.square()).real.view(1, 1, -1)
        self.one_idx = [torch.nonzero(labels[:, k], as_tuple=False).squeeze(-1) for k in range(self.K)]
        self.zero_idx = [torch.nonzero(~labels[:, k], as_tuple=False).squeeze(-1) for k in range(self.K)]

        dev = torch.device(device)
        self.row0 = torch.arange(self.M, dtype=torch.long, device=dev) * self.N
        self.var_floor = 1.0e-7
        self.eps_min = 1.0e-6
        self.eps_max = 1.0e6

    # --- channel: recover the N per-Doppler M x M blocks H_n (= H_T blocks) ---
    def _blocks(self, h_vals: torch.Tensor, col_idx: torch.Tensor) -> torch.Tensor:
        b = h_vals.shape[0]
        h0 = h_vals.index_select(1, self.row0)
        c0 = col_idx.index_select(1, self.row0)
        in_delay = torch.div(c0, self.N, rounding_mode="floor")
        in_doppler = c0 - in_delay * self.N
        h_flat = torch.zeros((b, self.M, self.mn), dtype=h_vals.dtype, device=h_vals.device)
        h_flat.scatter_add_(2, in_delay * self.N + in_doppler, h0)
        h_shift = h_flat.reshape(b, self.M, self.M, self.N)
        return torch.fft.ifft(h_shift, dim=-1).mul(float(self.N)).permute(0, 3, 1, 2).contiguous()

    # --- unitary Doppler transforms (ortho => noise variance preserved) ---
    def _to_freq(self, x_dd: torch.Tensor) -> torch.Tensor:
        b = x_dd.shape[0]
        return torch.fft.fft(x_dd.reshape(b, self.M, self.N), dim=-1, norm="ortho").transpose(1, 2).contiguous()

    def _to_dd(self, x_f: torch.Tensor) -> torch.Tensor:
        b = x_f.shape[0]
        return torch.fft.ifft(x_f.transpose(1, 2).contiguous(), dim=-1, norm="ortho").reshape(b, self.mn)

    def _final_llr(self, q: torch.Tensor, nu_q: torch.Tensor) -> torch.Tensor:
        # q: (b, MN) complex ; nu_q: (b,) scalar pseudo-AWGN variance
        diff = q.unsqueeze(-1) - self.points  # (b, MN, |A|)
        dist = diff.real.square() + diff.imag.square()
        logits = -dist / nu_q.view(-1, 1, 1).clamp_min(self.var_floor)
        out = q.new_empty((q.shape[0], self.mn, self.K), dtype=logits.dtype)
        for k in range(self.K):
            m1 = torch.logsumexp(logits.index_select(-1, self.one_idx[k]), dim=-1)
            m0 = torch.logsumexp(logits.index_select(-1, self.zero_idx[k]), dim=-1)
            out[:, :, k] = m1 - m0
        return out.reshape(q.shape[0], self.mn * self.K)

    @torch.no_grad()
    def llr(self, y_dd: torch.Tensor, channel: ChannelView, no: torch.Tensor) -> torch.Tensor:
        b = y_dd.shape[0]
        rdtype = y_dd.real.dtype

        H = self._blocks(channel.sparse.h_vals, channel.sparse.col_idx)  # (b,N,M,M)
        Hh = H.conj().transpose(-2, -1)
        # Whitening via eigendecomposition of the Hermitian Gram H^H H instead of
        # a complex SVD of H: same right singular vectors / singular values, but
        # ~100x cheaper (a batched eigh vs a batched complex SVD), and we never
        # need to form the left singular matrix U explicitly.
        gram = torch.matmul(Hh, H)  # (b,N,M,M) Hermitian
        w, V = torch.linalg.eigh(gram)  # gram = V diag(w) V^H, w real >= 0
        lam = w.to(rdtype).clamp_min(0.0)  # singular values squared
        d = lam.sqrt()  # (b,N,M)
        Vh = V.conj().transpose(-2, -1)  # paper's block "V_n" (= V^H)

        y_f = self._to_freq(y_dd)  # (b,N,M)
        # r = U^H y_f = diag(1/d) V^H H^H y_f  (no need to materialise U)
        d_safe = d.clamp_min(self.var_floor ** 0.5)
        hhy = torch.matmul(Hh, y_f.unsqueeze(-1)).squeeze(-1)  # (b,N,M)
        r = torch.matmul(Vh, hhy.unsqueeze(-1)).squeeze(-1) / d_safe.to(y_f.dtype)  # (b,N,M)

        noise = torch.as_tensor(no, dtype=rdtype, device=y_dd.device).reshape(-1).real
        if noise.numel() == 1:
            noise = noise.expand(b)
        eps = (1.0 / noise.clamp_min(self.var_floor)).clamp(self.eps_min, self.eps_max)  # (b,)

        s_prev = torch.zeros((b, self.N, self.M), dtype=y_f.dtype, device=y_f.device)
        x_dd = torch.zeros((b, self.mn), dtype=y_f.dtype, device=y_f.device)
        nu_x = torch.ones((b,), dtype=rdtype, device=y_f.device)
        nu_q = torch.ones((b,), dtype=rdtype, device=y_f.device)
        q = x_dd

        for _ in range(self.num_iters):
            nu_p = (nu_x.view(b, 1, 1) * lam).clamp_min(self.var_floor)            # (b,N,M)
            x_f = self._to_freq(x_dd)                                              # (b,N,M)
            Vx = torch.matmul(Vh, x_f.unsqueeze(-1)).squeeze(-1)                   # V x_f
            p = d * Vx - nu_p * s_prev                                            # Line 2

            nu_z = 1.0 / (1.0 / nu_p + eps.view(b, 1, 1))                          # Line 3
            z = nu_z * (p / nu_p + eps.view(b, 1, 1) * r)                          # Line 4

            if self.estimate_noise:
                resid = (r - z)
                num = float(self.mn)
                den = (resid.real.square() + resid.imag.square()).sum(dim=(1, 2)) + nu_z.sum(dim=(1, 2))
                eps = (num / den.clamp_min(self.var_floor)).clamp(self.eps_min, self.eps_max)  # Line 5

            nu_s = 1.0 / (nu_p + (1.0 / eps).view(b, 1, 1))                        # Line 6
            s = nu_s * (r - p)                                                    # Line 7

            # Line 8: scalar variance of the pseudo-observation q.  UAMP defines
            # this via 1/nu_q = (1/MN) lambda^T nu_s, i.e. nu_q = MN/(lambda^T nu_s).
            nu_q = float(self.mn) / (lam * nu_s).sum(dim=(1, 2)).clamp_min(self.var_floor)
            ds = d * s
            Wds = torch.matmul(Vh.conj().transpose(-2, -1), ds.unsqueeze(-1)).squeeze(-1)  # V^H (d.s)
            q = x_dd + nu_q.view(b, 1).to(y_f.dtype) * self._to_dd(Wds)            # Line 9

            # symbol denoiser with scalar variance nu_q (Lines 10-13)
            diff = q.unsqueeze(-1) - self.points                                   # (b,MN,|A|)
            dist = diff.real.square() + diff.imag.square()
            logits = -dist / nu_q.view(b, 1, 1).clamp_min(self.var_floor)
            beta = torch.softmax(logits, dim=-1)
            x_dd = (beta.to(self.points.dtype) * self.points).sum(dim=-1)          # x_hat
            second = (beta * self.point_abs2).sum(dim=-1)
            nu_xj = (second - (x_dd.real.square() + x_dd.imag.square())).clamp_min(self.var_floor)
            nu_x = nu_xj.mean(dim=1)                                              # Line 14 (scalar)

            s_prev = s

        return self._final_llr(q, nu_q)
