# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Sionna-compatible QAM helpers for OTFS detectors.

The transmit pipeline in the evaluation harness encodes bits with the
5G LDPC encoder and maps them with :class:`sionna.phy.mapping.Mapper`.
A detector must therefore produce LLRs in exactly Sionna's convention,
otherwise the LDPC decoder receives scrambled bits and the BLER
saturates at 1.0 regardless of how clever the detector is.

There are two equally valid ways to write the detector:

* **Symbol-equaliser path**: produce a post-equalised symbol estimate
  ``x_hat`` and a per-symbol noise variance ``var``, then call Sionna's
  :class:`~sionna.phy.mapping.Demapper`. Use :func:`make_demapper` for
  a ready-made instance, or construct one yourself.

* **Per-bit detector path** (EP, MP, BP, ...): compute soft information
  per constellation point and reduce it to per-bit LLRs yourself. Use
  :func:`qam_constellation_with_labels` to get the constellation points
  and the canonical bit-label table; do **not** fabricate either by
  hand.

In both cases the constellation points and bit-label convention come
from Sionna. The label table follows Sionna's "MSB-first index" rule:
the bit-label vector ``[b_0, ..., b_{K-1}]`` maps to the constellation
point at index ``k = (b_0 << (K-1)) | ... | (b_{K-1} << 0)``. Bit ``b_0``
is the MSB.
"""

from __future__ import annotations

from typing import Tuple

import torch

from sionna.phy.mapping import Constellation, Demapper

from ._device import resolve_device


def qam_constellation_with_labels(
    num_bits_per_symbol: int,
    *,
    precision: str = "single",
    device: str | torch.device = "cuda:0",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return Sionna's QAM constellation points and bit-label table.

    Parameters
    ----------
    num_bits_per_symbol : int
        Bits per QAM symbol (4 for 16-QAM).
    precision : {"single", "double"}
        Precision passed to :class:`sionna.phy.mapping.Constellation`.
    device : str or torch.device
        Device on which both tensors are placed.

    Returns
    -------
    points : torch.Tensor
        Complex tensor of shape ``(Q,)`` (``Q = 2**num_bits_per_symbol``)
        with the canonical Sionna QAM points, in index order.
    bit_labels : torch.Tensor
        Boolean tensor of shape ``(Q, num_bits_per_symbol)``. Entry
        ``[q, k]`` is the value of bit ``k`` (MSB-first within a symbol)
        of constellation index ``q``.
    """
    K = int(num_bits_per_symbol)
    if K < 1:
        raise ValueError(f"num_bits_per_symbol must be >= 1, got {K}")

    device = resolve_device(device)
    constellation = Constellation("qam", num_bits_per_symbol=K, precision=precision)
    points = constellation.points.to(device=device)

    Q = 1 << K
    labels = torch.zeros(Q, K, dtype=torch.bool, device=device)
    for q in range(Q):
        for k in range(K):
            labels[q, k] = ((q >> (K - 1 - k)) & 1) == 1
    return points, labels


def make_demapper(
    num_bits_per_symbol: int,
    *,
    mode: str = "maxlog",
    precision: str = "single",
    device: str | torch.device = "cuda:0",
) -> Demapper:
    """Return a Sionna :class:`Demapper` matching the TX-side QAM mapping.

    Convenience wrapper for the "equalise to symbols, then demap" path:
    given a post-equalised symbol estimate ``x_hat`` of shape
    ``(batch, M*N)`` and a per-symbol noise variance ``var`` of the same
    shape, call ``demapper(x_hat, var)`` to obtain LLRs in the shape and
    sign convention the LDPC decoder expects.

    Parameters
    ----------
    num_bits_per_symbol : int
        Must match the TX-side mapper (4 for 16-QAM).
    mode : {"maxlog", "app"}
        ``"maxlog"`` for the standard max-log approximation,
        ``"app"`` for the exact log-APP demapper.
    precision, device : same as for :func:`qam_constellation_with_labels`.
    """
    return Demapper(
        mode,
        "qam",
        num_bits_per_symbol=int(num_bits_per_symbol),
        precision=precision,
        device=resolve_device(device),
    )


__all__ = ["qam_constellation_with_labels", "make_demapper"]
