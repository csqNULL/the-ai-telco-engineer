# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

r"""Sparse Gaussian message-passing OTFS equalizer.

Self-contained implementation of the message-passing detector of

    P. Raviteja, K. T. Phan, Y. Hong, and E. Viterbo, "Interference
    Cancellation and Iterative Detection for Orthogonal Time Frequency Space
    Modulation," IEEE Trans. Wireless Commun., vol. 17, no. 10, pp. 6501-6515,
    Oct. 2018 (doi:10.1109/TWC.2018.2860011), Algorithm 1.

Each step maps directly onto the paper:

  * Observation -> variable interference mean / variance .. Eq. (30) / (31)
        (``mu_dk`` / ``sig2_dk``, leave-one-out over the ``S`` non-zeros of a row).
  * Per-edge symbol likelihood ``xi`` ..................... Eq. (33).
  * Damped pmf update with factor ``damping`` ............. Eq. (32).
  * Convergence indicator ``eta`` (threshold ``1-gamma``) . Eq. (34) / (35).
  * Best-so-far log-posterior + backslide stop ........... Eq. (36) and the
        stopping rule with tolerance ``epsilon`` (paper uses ``epsilon=0.2``).

The channel is stored in per-output-row sparse form: for a channel with ``S``
non-zero taps per row, ``h_vals`` / ``col_idx`` have shape ``(B, MN, S)`` and
``y[r] = sum_e h_vals[r, e] * x[col_idx[r, e]] + w[r]``.  Memory therefore
scales as ``O(MN * S * Q)`` rather than ``O(MN * MN * Q)`` for the dense form.

The per-edge update evaluates ``|y_d - mu_{d,k} - h_{d,k} A_j|^2`` by its
algebraic expansion, keeping every per-iteration intermediate at the per-edge
scale ``(B, MN, S, Q)`` (instead of the ``(B, MN, S, Q, Q)`` a naive broadcast
would materialise).  This keeps peak memory low and lets ``torch.compile``
specialise the loop body as a single fused graph.

All messages are carried as separate real / imaginary tensors (as in ``ep.py``)
rather than ``torch.complex``: under ``torch.compile`` complex multiplies fall
back to aten ``unrolled_elementwise`` kernels and the per-edge mean/variance
reductions over the constellation lower to cuBLAS ``gemv`` instead of fusing,
which dominated the runtime.  The real formulation lets TorchInductor codegen
the whole loop body as fused Triton reductions / scatters.

``early_stop`` defaults to ``False`` so the fixed-count loop traces without the
data-dependent ``bool(active.any())`` host sync; best-so-far tracking is
unchanged, so accuracy is identical -- only the (already-converged) trailing
iterations still run.
"""

from __future__ import annotations

import torch
from sionna.phy.mapping import Constellation

from otfs._device import resolve_device
from otfs.channel_view import ChannelView
from otfs.equalizers.base import Equalizer


class MP(Equalizer):
    name = "MP"
    channel_form = "sparse"

    def __init__(
        self,
        grid,
        *,
        num_bits_per_symbol: int = 4,
        num_iterations: int = 15,
        damping: float = 0.6,
        gamma: float = 0.05,
        epsilon: float = 0.2,
        early_stop: bool = False,
        precision: str = "single",
        device: str = "cuda:0",
        **_unused,
    ) -> None:
        if not 0.0 < damping <= 1.0:
            raise ValueError(f"damping must be in (0, 1], got {damping}")
        if num_iterations < 1:
            raise ValueError("num_iterations must be >= 1")

        device = resolve_device(device)
        self.mn = int(grid.M) * int(grid.N)
        self.K = int(num_bits_per_symbol)
        self.num_iter = int(num_iterations)
        self.damping = float(damping)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.early_stop = bool(early_stop)
        self._dtype = torch.float32 if precision == "single" else torch.float64
        self._cdtype = torch.complex64 if precision == "single" else torch.complex128
        self._device = device

        cst = Constellation("qam", num_bits_per_symbol=self.K, precision=precision)
        pts = cst.points.to(device=device)
        # Constellation as flat real tensors; broadcast over the trailing Q axis.
        self.A_r = pts.real.to(dtype=self._dtype).contiguous()      # (Q,)
        self.A_i = pts.imag.to(dtype=self._dtype).contiguous()      # (Q,)
        self.A_abs2 = (pts.real.square() + pts.imag.square()).to(dtype=self._dtype).contiguous()
        self.Q = int(pts.numel())

        labels = torch.zeros(self.Q, self.K, dtype=torch.bool, device=device)
        for q in range(self.Q):
            for k in range(self.K):
                labels[q, k] = ((q >> (self.K - 1 - k)) & 1) == 1
        self.bit_labels = labels  # (Q, K)

    def _log_post_to_llr(self, log_post: torch.Tensor) -> torch.Tensor:
        """Convert ``(B, MN, Q)`` log-posterior into ``(B, MN*K)`` LLRs."""
        batch = log_post.shape[0]
        neg_inf = torch.finfo(log_post.dtype).min
        lp = log_post.unsqueeze(-1)                               # (B, MN, Q, 1)
        labels = self.bit_labels                                  # (Q, K)
        log_one = torch.where(labels, lp, torch.full_like(lp, neg_inf))
        log_zero = torch.where(~labels, lp, torch.full_like(lp, neg_inf))
        llr = torch.logsumexp(log_one, dim=-2) - torch.logsumexp(log_zero, dim=-2)
        return llr.reshape(batch, self.mn * self.K)

    @torch.no_grad()
    def llr(self, y_dd: torch.Tensor, channel: ChannelView, no: torch.Tensor) -> torch.Tensor:
        if channel.sparse is None:
            raise ValueError("MP requires channel.sparse")
        sparse = channel.sparse
        col_idx = sparse.col_idx.to(dtype=torch.long, device=self._device)

        # Split into real tensors straight off the complex inputs (as in ep.py),
        # keeping the device/dtype cast on the *real* parts. A complex .to(device)
        # would put a complex-backed tensor into the compiled graph and trip
        # Inductor's "no codegen for complex operators" fallback.
        rdtype = self._dtype
        dev = self._device
        hr = sparse.h_vals.real.contiguous().to(dtype=rdtype, device=dev)  # (B, MN, S)
        hi = sparse.h_vals.imag.contiguous().to(dtype=rdtype, device=dev)
        yr = y_dd.real.contiguous().to(dtype=rdtype, device=dev)           # (B, MN)
        yi = y_dd.imag.contiguous().to(dtype=rdtype, device=dev)

        batch_shape = hr.shape[:-2]
        s = hr.shape[-1]
        Q = self.Q
        nm = self.mn
        A_r, A_i, A_abs2 = self.A_r, self.A_i, self.A_abs2

        # Loop-invariant per-frame quantities.
        h_abs2 = hr.square() + hi.square()                        # (B, MN, S)
        h2_A2 = h_abs2.unsqueeze(-1) * A_abs2                     # (B, MN, S, Q)

        sigma2 = torch.as_tensor(no, dtype=rdtype, device=self._device)
        while sigma2.ndim < h_abs2.ndim:
            sigma2 = sigma2.unsqueeze(-1)

        p = torch.full(
            (*batch_shape, nm, s, Q), 1.0 / Q, dtype=rdtype, device=self._device,
        )
        best_eta = torch.full(batch_shape, -1.0, dtype=rdtype, device=self._device)
        # Log-posterior for the best-so-far iteration; the returned LLRs come
        # from the iteration with the highest convergence indicator eta.
        best_log_post = torch.zeros(
            *batch_shape, nm, Q, dtype=rdtype, device=self._device,
        )
        active = torch.ones(batch_shape, dtype=torch.bool, device=self._device)

        # Variable-aggregation index broadcast to the Q axis, flat layout.
        col_idx_q = col_idx.unsqueeze(-1).expand(*col_idx.shape, Q).contiguous()
        idx_flat = col_idx_q.reshape(*batch_shape, nm * s, Q)

        for _ in range(self.num_iter):
            # ---- Pass 1: observation update + edge messages -----------
            # Per-edge symbol mean/variance over the constellation (Eq. 30/31),
            # split into real/imag so the reductions fuse under torch.compile.
            m_r = (p * A_r).sum(dim=-1)                           # (B, MN, S)
            m_i = (p * A_i).sum(dim=-1)
            v_dk = (p * A_abs2).sum(dim=-1) - (m_r.square() + m_i.square())

            mh_r = m_r * hr - m_i * hi                            # m_dk * h_dk
            mh_i = m_r * hi + m_i * hr
            vh2 = v_dk * h_abs2
            Md_r = mh_r.sum(dim=-1)                               # (B, MN)
            Md_i = mh_i.sum(dim=-1)
            Vd = vh2.sum(dim=-1) + sigma2.squeeze(-1)

            # Leave-one-out mean/variance for edge (d, k).
            mu_r = Md_r.unsqueeze(-1) - mh_r                      # (B, MN, S)
            mu_i = Md_i.unsqueeze(-1) - mh_i
            sig2_dk = torch.clamp(Vd.unsqueeze(-1) - vh2, min=1e-30)

            # Per-edge log-likelihood without materialising (B, MN, S, Q, Q):
            #   |y_d - mu_dk - h_dk A_j|^2
            #     = |y_d - mu_dk|^2 - 2 Re[(y_d-mu_dk)^* h_dk A_j] + |h_dk|^2 |A_j|^2
            ymm_r = yr.unsqueeze(-1) - mu_r                       # (B, MN, S)
            ymm_i = yi.unsqueeze(-1) - mu_i
            const_d = ymm_r.square() + ymm_i.square()
            ymh_r = ymm_r * hr + ymm_i * hi                       # (y-mu)^* h
            ymh_i = ymm_r * hi - ymm_i * hr
            cross = (
                -2.0 * ymh_r.unsqueeze(-1) * A_r
                + 2.0 * ymh_i.unsqueeze(-1) * A_i
            )                                                     # (B, MN, S, Q)
            log_xi = -(const_d.unsqueeze(-1) + cross + h2_A2) / sig2_dk.unsqueeze(-1)
            log_p_edge = log_xi - torch.logsumexp(log_xi, dim=-1, keepdim=True)

            # Variable-centric aggregation: L_c[c] = sum_d log p_{d->c}.
            log_p_flat = log_p_edge.reshape(*batch_shape, nm * s, Q)
            L_c = torch.zeros(
                *batch_shape, nm, Q, dtype=rdtype, device=self._device,
            )
            L_c.scatter_add_(-2, idx_flat, log_p_flat)

            # ---- Decision + best-so-far tracking ----------------------
            posterior = torch.softmax(L_c, dim=-1)
            max_p = posterior.amax(dim=-1)
            converged_vars = (max_p >= 1.0 - self.gamma).to(rdtype)
            new_eta = converged_vars.mean(dim=-1)

            improved = new_eta > best_eta
            best_eta = torch.where(improved, new_eta, best_eta)
            best_log_post = torch.where(
                improved.reshape(*improved.shape, 1, 1).expand_as(L_c), L_c, best_log_post,
            )

            all_converged = new_eta >= 1.0
            backslide = new_eta < best_eta - self.epsilon
            newly_stopped = active & (all_converged | backslide)
            active = active & ~newly_stopped

            # ---- Pass 2: leave-one-out, damped belief update ----------
            L_c_at_edge = torch.gather(L_c, -2, idx_flat).reshape(*batch_shape, nm, s, Q)
            log_p_tilde = L_c_at_edge - log_p_edge
            log_p_tilde = log_p_tilde - torch.logsumexp(log_p_tilde, dim=-1, keepdim=True)
            p_tilde = torch.exp(log_p_tilde)
            p_new = self.damping * p_tilde + (1.0 - self.damping) * p
            mask = active.reshape(*active.shape, 1, 1, 1)
            p = torch.where(mask, p_new, p)

            if self.early_stop and not bool(active.any()):
                break

        return self._log_post_to_llr(best_log_post)
