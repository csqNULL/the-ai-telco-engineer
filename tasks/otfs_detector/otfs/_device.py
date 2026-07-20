# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Device-selection helper for the OTFS toolkit."""

from __future__ import annotations

import torch


def resolve_device(device: str | torch.device) -> str | torch.device:
    """Return ``device`` when usable, otherwise fall back to ``"cpu"``.

    Keeps the requested default (typically ``"cuda:0"``) when a GPU is
    present, but transparently falls back to CPU on machines without CUDA
    so the toolkit stays importable and runnable anywhere.
    """
    if torch.device(device).type == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return device
