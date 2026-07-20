# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import torch

from hp import HP
from otfs.equalizers.base import Equalizer
from otfs.channel_view import ChannelView
from otfs import qam_constellation_with_labels


class EC2RichardsonAPPDetector(Equalizer):
    name = "EC2RichardsonAPPDetector"
    channel_form = "sparse"
    top_k = HP.get("top_k", 2560, choices=[1536, 2048, 2560])

    def __init__(self, grid, *, num_bits_per_symbol=4, precision="single", device="cuda:0", **kwargs):
        self.M = int(grid.M)
        self.N = int(grid.N)
        self.mn = self.M * self.N
        self.K = int(num_bits_per_symbol)

        # Assigned structure: two full Cholesky EC iterations, then one
        # diagonally-preconditioned Richardson linear half-pass seeded from the
        # second nonlinear (tilted QAM) mean. Final demap fuses APP log-metrics
        # from the Richardson A-cavity, the second tilted metric, and the second
        # full linear posterior.
        self.full_iters = int(HP.get("full_iters", 2, low=2, high=2))
        self.richardson_omega = float(HP.get("richardson_omega", 0.92, low=0.35, high=1.45))

        self.damp = float(HP.get("damp", 0.6778741291285645, low=0.25, high=0.85))
        self.var_floor = float(HP.get("var_floor", 0.000277657645398233, low=2.0e-4, high=2.0e-2, log=True))
        self.prec_max = float(HP.get("prec_max", 255.70784975403566, low=5.0e1, high=5.0e3, log=True))
        self.prec_min = 1.0 / self.prec_max
        self.llr_scale = float(HP.get("llr_scale", 0.95, low=0.35, high=1.8))
        self.no_scale = float(HP.get("no_scale", 1.15, low=0.55, high=1.75))

        self.a_cavity_weight = float(HP.get("a_cavity_weight", 1.00, low=0.25, high=1.75))
        self.tilt_fusion_weight = float(HP.get("tilt_fusion_weight", -0.10, low=-0.55, high=0.95))
        self.lin_fusion_weight = float(HP.get("lin_fusion_weight", 0.16, low=-0.45, high=0.95))
        self.final_demap = HP.get("final_demap", "maxlog", choices=["app", "maxlog"])

        pts, labels = qam_constellation_with_labels(self.K, precision=precision, device=device)
        self.points = pts
        self.labels = labels
        self.q = int(pts.shape[0])
        self.one_idx = torch.stack([torch.nonzero(labels[:, kk], as_tuple=False).squeeze(-1) for kk in range(self.K)], dim=0).view(1, 1, 1, self.K, -1)
        self.zero_idx = torch.stack([torch.nonzero(~labels[:, kk], as_tuple=False).squeeze(-1) for kk in range(self.K)], dim=0).view(1, 1, 1, self.K, -1)
        self.point_pow = pts.real * pts.real + pts.imag * pts.imag

        dev = torch.device(device)
        self.row0 = torch.arange(self.M, dtype=torch.long, device=dev) * self.N
        self.mout_base = torch.arange(self.M, dtype=torch.long, device=dev).view(1, self.M, 1)
        self.eyeM = torch.eye(self.M, dtype=pts.dtype, device=dev).unsqueeze(0).unsqueeze(0)
        self.offdiag_mask = (1.0 - torch.eye(self.M, dtype=pts.real.dtype, device=dev)).view(1, 1, self.M, self.M)

    def _blocks_from_sparse(self, h_vals: torch.Tensor, col_idx: torch.Tensor) -> torch.Tensor:
        b = h_vals.shape[0]
        k_eff = h_vals.shape[-1]
        hv = h_vals.index_select(1, self.row0)
        ci = col_idx.index_select(1, self.row0)
        lin = torch.div(ci, self.N, rounding_mode="floor")
        ck = ci - lin * self.N
        delta = torch.remainder(-ck, self.N)
        mout = self.mout_base.expand(b, self.M, k_eff)
        flat_idx = (mout * (self.M * self.N) + lin * self.N + delta).reshape(b, -1)
        g = torch.zeros(b, self.M * self.M * self.N, dtype=h_vals.dtype, device=h_vals.device)
        g.scatter_add_(1, flat_idx, hv.reshape(b, -1))
        g = g.reshape(b, self.M, self.M, self.N)
        hf = torch.fft.fft(g, dim=-1)
        return hf.permute(0, 3, 1, 2).contiguous()

    def _jacobi2_trace(self, A: torch.Tensor) -> torch.Tensor:
        d = A.diagonal(dim1=-2, dim2=-1).real.clamp_min(1.0e-8)
        invd = 1.0 / d
        abs2 = (A.real * A.real + A.imag * A.imag) * self.offdiag_mask
        corr = torch.sum(abs2 * invd.unsqueeze(-2), dim=-1) * invd * invd
        v = (invd + corr).clamp_min(self.var_floor)
        return v.permute(0, 2, 1).mean(dim=-1).clamp_min(self.var_floor)

    def _linear_posterior(self, HhH: torch.Tensor, Hhy: torch.Tensor, pB: torch.Tensor, etaB: torch.Tensor):
        b = etaB.shape[0]
        rB = etaB / pB.unsqueeze(-1).to(etaB.dtype)
        rBf = torch.fft.fft(rB, dim=-1, norm="ortho").permute(0, 2, 1).contiguous()
        A = HhH + self.eyeM * pB.view(b, 1, self.M, 1).to(HhH.dtype)
        rhs = Hhy + pB.view(b, 1, self.M).to(Hhy.dtype) * rBf
        L = torch.linalg.cholesky(A)
        zmean = torch.cholesky_solve(rhs.unsqueeze(-1), L).squeeze(-1)
        xlin = torch.fft.ifft(zmean.permute(0, 2, 1).contiguous(), dim=-1, norm="ortho")
        vlin_d = self._jacobi2_trace(A)
        return xlin, vlin_d

    def _richardson_linear_half(self, HhH: torch.Tensor, Hhy: torch.Tensor, pB: torch.Tensor,
                                etaB: torch.Tensor, seed_x: torch.Tensor):
        b = etaB.shape[0]
        rB = etaB / pB.unsqueeze(-1).to(etaB.dtype)
        rBf = torch.fft.fft(rB, dim=-1, norm="ortho").permute(0, 2, 1).contiguous()
        A = HhH + self.eyeM * pB.view(b, 1, self.M, 1).to(HhH.dtype)
        rhs = Hhy + pB.view(b, 1, self.M).to(Hhy.dtype) * rBf

        # One diagonally preconditioned Richardson update from the nonlinear mean:
        # z1 = z0 + omega * diag(A)^(-1) * (rhs - A z0).
        z = torch.fft.fft(seed_x, dim=-1, norm="ortho").permute(0, 2, 1).contiguous()
        diag_inv = (1.0 / A.diagonal(dim1=-2, dim2=-1).real.clamp_min(1.0e-8)).to(A.dtype)
        res = rhs - torch.matmul(A, z.unsqueeze(-1)).squeeze(-1)
        z = z + (self.richardson_omega * diag_inv) * res

        xlin = torch.fft.ifft(z.permute(0, 2, 1).contiguous(), dim=-1, norm="ortho")
        vlin_d = self._jacobi2_trace(A)
        return xlin, vlin_d

    def _qam_logp(self, r: torch.Tensor, prec_delay: torch.Tensor) -> torch.Tensor:
        p = prec_delay.unsqueeze(-1).unsqueeze(-1)
        diff = r.unsqueeze(-1) - self.points.view(1, 1, 1, self.q)
        return -p * (diff.real * diff.real + diff.imag * diff.imag)

    def _qam_project_from_logp(self, logp: torch.Tensor):
        pts = self.points.view(1, 1, 1, self.q)
        lmax = torch.amax(logp, dim=-1, keepdim=True)
        w = torch.exp(logp - lmax)
        z = torch.sum(w, dim=-1).clamp_min(1.0e-30)
        mean = torch.sum(w * pts, dim=-1) / z
        pow2 = torch.sum(w * self.point_pow.view(1, 1, 1, self.q), dim=-1) / z
        var = (pow2 - (mean.real * mean.real + mean.imag * mean.imag)).clamp_min(self.var_floor)
        return mean, var

    def _maxlog_llr_from_logp(self, logp: torch.Tensor) -> torch.Tensor:
        logp_kq = logp.unsqueeze(-2).expand(-1, -1, -1, self.K, -1)
        idx1 = self.one_idx.expand(logp.shape[0], logp.shape[1], logp.shape[2], -1, -1)
        idx0 = self.zero_idx.expand(logp.shape[0], logp.shape[1], logp.shape[2], -1, -1)
        m1 = torch.amax(torch.gather(logp_kq, -1, idx1), dim=-1)
        m0 = torch.amax(torch.gather(logp_kq, -1, idx0), dim=-1)
        return m1 - m0

    def _app_llr_from_logp(self, logp: torch.Tensor) -> torch.Tensor:
        logp_kq = logp.unsqueeze(-2).expand(-1, -1, -1, self.K, -1)
        idx1 = self.one_idx.expand(logp.shape[0], logp.shape[1], logp.shape[2], -1, -1)
        idx0 = self.zero_idx.expand(logp.shape[0], logp.shape[1], logp.shape[2], -1, -1)
        m1 = torch.logsumexp(torch.gather(logp_kq, -1, idx1), dim=-1)
        m0 = torch.logsumexp(torch.gather(logp_kq, -1, idx0), dim=-1)
        return m1 - m0

    def llr(self, y_dd: torch.Tensor, channel: ChannelView, no: torch.Tensor) -> torch.Tensor:
        b = y_dd.shape[0]
        M = self.M
        N = self.N

        H = self._blocks_from_sparse(channel.sparse.h_vals, channel.sparse.col_idx)
        y = y_dd.reshape(b, M, N)
        yf = torch.fft.fft(y, dim=-1, norm="ortho").permute(0, 2, 1).contiguous()

        noise = (no.to(yf.real.dtype) * self.no_scale).clamp_min(1.0e-6)
        inv_no = (1.0 / noise).view(b, 1, 1, 1)
        Hh = H.conj().transpose(-2, -1)
        Hhy = torch.matmul(Hh, yf.unsqueeze(-1)).squeeze(-1) * inv_no.squeeze(-1)
        HhH = torch.matmul(Hh, H) * inv_no

        pB = torch.ones(b, M, dtype=yf.real.dtype, device=yf.device)
        etaB = torch.zeros(b, M, N, dtype=yf.dtype, device=yf.device)
        last_tilt_logp = torch.zeros(b, M, N, self.q, dtype=yf.real.dtype, device=yf.device)
        second_xlin = torch.zeros(b, M, N, dtype=yf.dtype, device=yf.device)
        second_ppost = torch.ones(b, M, dtype=yf.real.dtype, device=yf.device)
        second_xmean = torch.zeros(b, M, N, dtype=yf.dtype, device=yf.device)

        for _ in range(self.full_iters):
            xlin, vlin_d = self._linear_posterior(HhH, Hhy, pB, etaB)
            ppost = (1.0 / vlin_d.clamp_min(self.var_floor)).clamp_max(self.prec_max)

            pA = (ppost - pB).clamp_min(self.prec_min).clamp_max(self.prec_max)
            etaA = ppost.unsqueeze(-1).to(xlin.dtype) * xlin - etaB
            rA = etaA / pA.unsqueeze(-1).to(xlin.dtype)

            last_tilt_logp = self._qam_logp(rA, pA)
            xmean, xvar = self._qam_project_from_logp(last_tilt_logp)
            second_xlin = xlin
            second_ppost = ppost
            second_xmean = xmean

            vnl_d = xvar.mean(dim=-1).clamp_min(self.var_floor)
            pnl_post = (1.0 / vnl_d).clamp_max(self.prec_max)
            pB_ext = (pnl_post - pA).clamp_min(self.prec_min).clamp_max(self.prec_max)
            etaB_ext = pnl_post.unsqueeze(-1).to(xmean.dtype) * xmean - pA.unsqueeze(-1).to(xmean.dtype) * rA

            pB = ((1.0 - self.damp) * pB + self.damp * pB_ext).clamp_min(self.prec_min).clamp_max(self.prec_max)
            etaB = (1.0 - self.damp) * etaB + self.damp * etaB_ext

        xhalf, vhalf_d = self._richardson_linear_half(HhH, Hhy, pB, etaB, second_xmean)
        ppost_c = (1.0 / vhalf_d.clamp_min(self.var_floor)).clamp_max(self.prec_max)
        pA_c = (ppost_c - pB).clamp_min(self.prec_min).clamp_max(self.prec_max)
        etaA_c = ppost_c.unsqueeze(-1).to(xhalf.dtype) * xhalf - etaB
        rA_c = etaA_c / pA_c.unsqueeze(-1).to(xhalf.dtype)

        cavity_logp = self._qam_logp(rA_c, pA_c)
        second_lin_logp = self._qam_logp(second_xlin, second_ppost)
        fused_logp = (self.a_cavity_weight * cavity_logp
                      + self.tilt_fusion_weight * last_tilt_logp
                      + self.lin_fusion_weight * second_lin_logp)

        if self.final_demap == "app":
            out = self._app_llr_from_logp(fused_logp)
        else:
            out = self._maxlog_llr_from_logp(fused_logp)
        return out.reshape(b, M * N * self.K) * self.llr_scale
