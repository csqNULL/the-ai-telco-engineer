# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Sparse expectation-propagation (EP) equalizer (torch.compile-friendly).

Implements the EP detector of

    H. Li, Y. Dong, C. Gong, Z. Zhang, X. Wang, and X. Dai, "Low Complexity
    Receiver via Expectation Propagation for OTFS Modulation," IEEE
    Communications Letters, vol. 25, no. 10, pp. 3128-3132, Oct. 2021.
    (doi:10.1109/LCOMM.2021.3101827)

The exact EP receiver of Section III-A is reproduced edge-for-edge on the
sparse delay-Doppler factor graph:

  * FN -> VN message mean / variance ............ Eq. (15) / (16)
        ``uf_r,uf_i = (y - sum_{a'!=a} h u_{x->f}) / h``,
        ``v_f2x     = (sum_{a'!=a} |h|^2 v_{x->f} + sigma^2) / |h|^2``
        (leave-one-out is done by subtracting the self term from the full
        row sum, so the cost is one pass over the ``top_k`` non-zeros).
  * VN cavity variance / mean ................... Eq. (17) / (18).
  * Posterior mean / variance over the QAM set .. Eq. (19) / (20), with the
        per-symbol posterior of Eq. (21) obtained by scatter-adding the
        per-edge Gaussian log-likelihoods into each variable node.
  * Soft-output a posteriori LLRs ............... Eq. (22) (``_log_post_to_llr``;
        emitted with the Sionna sign convention, ln p(c=1)/p(c=0)).

Beyond the bare Algorithm-1 recursion this adds the three numerical
safeguards that are standard for EP (and assumed implicitly by the paper's
convergence analysis): message damping (``damping``), a variance floor /
ceiling (``v_floor`` / ``v_max``), and a positive-variance mask so a VN
cavity precision is only updated when it stays positive. The channel-
coefficient-bundling AEP approximation of Section III-B is *not* used; this
is the exact EP baseline (graph sparsity is instead controlled by the
benchmark harness via ``top_k``).

All messages are carried as separate real / imaginary tensors so
TorchInductor can codegen and fuse the whole graph -- a complex-valued
implementation forces the dominant sparse gather / scatter / elementwise
kernels to fall back to eager and ``torch.compile`` then buys almost
nothing. The per-edge log-likelihood is built from real broadcasts that
fuse into the scatter instead of materialising a complex ``diff`` of shape
``(B, MN, top_k, Q)``. The whole ``llr`` graph compiles cleanly under
``torch.compile(mode="default")``.
"""

from __future__ import annotations

import torch

from sionna.phy.mapping import Constellation

from otfs._device import resolve_device
from otfs.channel_view import ChannelView
from otfs.equalizers.base import Equalizer


class EP(Equalizer):
    name = "EP"
    channel_form = "sparse"

    def __init__(
        self,
        grid,
        *,
        num_bits_per_symbol: int = 4,
        num_iterations: int = 15,
        damping: float = 0.5,
        v_floor: float = 1e-7,
        v_max: float = 1e7,
        precision: str = "single",
        device: str = "cuda:0",
        **_unused,
    ) -> None:
        device = resolve_device(device)
        self.mn = int(grid.M) * int(grid.N)
        self.K = int(num_bits_per_symbol)
        self.num_iter = int(num_iterations)
        self.damping = float(damping)
        self.v_floor = float(v_floor)
        self.v_max = float(v_max)

        cst = Constellation("qam", num_bits_per_symbol=self.K, precision=precision)
        pts = cst.points.to(device=device)
        self.A_r = pts.real.contiguous().view(1, 1, -1)
        self.A_i = pts.imag.contiguous().view(1, 1, -1)
        self.A_abs2 = (pts.real.square() + pts.imag.square()).contiguous().view(1, 1, -1)
        self.Q = int(pts.numel())

        labels = torch.zeros(self.Q, self.K, dtype=torch.bool, device=device)
        for k in range(self.Q):
            for q in range(self.K):
                labels[k, q] = ((k >> (self.K - 1 - q)) & 1) == 1
        self.bit_labels = labels  # (Q, K)

    def _log_post_to_llr(self, log_post):
        neg_inf = torch.finfo(log_post.dtype).min
        lp = log_post.unsqueeze(-1)                              # (B, MN, Q, 1)
        labels = self.bit_labels                                # (Q, K)
        log_one = torch.where(labels, lp, torch.full_like(lp, neg_inf))
        log_zero = torch.where(~labels, lp, torch.full_like(lp, neg_inf))
        llr = torch.logsumexp(log_one, dim=-2) - torch.logsumexp(log_zero, dim=-2)
        return llr.reshape(log_post.shape[0], self.mn * self.K)

    @torch.no_grad()
    def llr(self, y_dd: torch.Tensor, channel: ChannelView, no: torch.Tensor) -> torch.Tensor:
        sparse = channel.sparse
        if sparse is None:
            raise ValueError("EP requires channel.sparse")
        h_vals = sparse.h_vals
        col_idx = sparse.col_idx.long()
        batch, mn, s = h_vals.shape

        hr = h_vals.real.contiguous()
        hi = h_vals.imag.contiguous()
        rdtype = hr.dtype
        yr = y_dd.real.to(rdtype)
        yi = y_dd.imag.to(rdtype)

        sigma2 = no.to(dtype=rdtype, device=y_dd.device).reshape(batch, 1)

        h_abs2 = hr.square() + hi.square()
        h_abs2_safe = torch.clamp(h_abs2, min=self.v_floor)
        hinv_r = hr / h_abs2_safe
        hinv_i = -hi / h_abs2_safe

        col_flat = col_idx.reshape(batch, mn * s)
        col_flat_q = col_flat.unsqueeze(-1).expand(batch, mn * s, self.Q)

        ur = torch.zeros(batch, mn, s, dtype=rdtype, device=y_dd.device)
        ui = torch.zeros(batch, mn, s, dtype=rdtype, device=y_dd.device)
        v_x2f = torch.ones(batch, mn, s, dtype=rdtype, device=y_dd.device)

        last_log_post = None
        for _ in range(self.num_iter):
            # FN <- VN signal aggregation (complex, expanded)
            htu_r = hr * ur - hi * ui
            htu_i = hr * ui + hi * ur
            ts_r = htu_r.sum(dim=-1, keepdim=True)
            ts_i = htu_i.sum(dim=-1, keepdim=True)
            total_var = (h_abs2 * v_x2f).sum(dim=-1, keepdim=True) + sigma2.unsqueeze(-1)

            es_r = ts_r - htu_r
            es_i = ts_i - htu_i
            excl_var = torch.clamp(total_var - h_abs2 * v_x2f, min=self.v_floor)

            a_r = yr.unsqueeze(-1) - es_r
            a_i = yi.unsqueeze(-1) - es_i
            uf_r = a_r * hinv_r - a_i * hinv_i
            uf_i = a_r * hinv_i + a_i * hinv_r
            v_f2x = torch.clamp(excl_var / h_abs2_safe, min=self.v_floor)

            # Per-edge log-likelihood over the constellation. Use the direct
            # real difference (NOT the |u|^2+|A|^2-2Re(...) expansion): EP's
            # u_f2x can be huge (1/|h|), so the expansion cancels catastrophically.
            # diff_r/diff_i fuse into log_I under Inductor without materialising.
            diff_r = uf_r.unsqueeze(-1) - self.A_r                        # (B,MN,S,Q)
            diff_i = uf_i.unsqueeze(-1) - self.A_i
            log_I = -(diff_r.square() + diff_i.square()) / v_f2x.unsqueeze(-1)

            log_post = torch.zeros(batch, mn, self.Q, dtype=rdtype, device=y_dd.device)
            log_post.scatter_add_(-2, col_flat_q, log_I.reshape(batch, mn * s, self.Q))
            log_post = log_post - log_post.max(dim=-1, keepdim=True).values
            post = torch.softmax(log_post, dim=-1)
            last_log_post = log_post

            ux_r = (post * self.A_r).sum(dim=-1)
            ux_i = (post * self.A_i).sum(dim=-1)
            abs2_mean = (post * self.A_abs2).sum(dim=-1)
            v_x = torch.clamp(abs2_mean - (ux_r.square() + ux_i.square()), min=self.v_floor)

            ux_r_e = torch.gather(ux_r, -1, col_flat).reshape(batch, mn, s)
            ux_i_e = torch.gather(ux_i, -1, col_flat).reshape(batch, mn, s)
            v_x_e = torch.gather(v_x, -1, col_flat).reshape(batch, mn, s)

            inv_v_new = 1.0 / v_x_e - 1.0 / v_f2x
            valid = inv_v_new > (1.0 / self.v_max)
            inv_v_safe = torch.where(valid, inv_v_new, torch.full_like(inv_v_new, 1.0 / self.v_max))
            v_cand = 1.0 / inv_v_safe

            mov_r = ux_r_e / v_x_e - uf_r / v_f2x
            mov_i = ux_i_e / v_x_e - uf_i / v_f2x
            uc_r = v_cand * mov_r
            uc_i = v_cand * mov_i

            if self.damping < 1.0:
                d = self.damping
                un_r = d * uc_r + (1.0 - d) * ur
                un_i = d * uc_i + (1.0 - d) * ui
                vn = d * v_cand + (1.0 - d) * v_x2f
            else:
                un_r, un_i, vn = uc_r, uc_i, v_cand

            ur = torch.where(valid, un_r, ur)
            ui = torch.where(valid, un_i, ui)
            v_x2f = torch.where(valid, vn, v_x2f)

        return self._log_post_to_llr(last_log_post)
