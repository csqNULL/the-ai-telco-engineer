# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Common equalizer interface for analytical-IO OTFS benchmarks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, Literal

import torch

from otfs.channel_view import ChannelView


ChannelForm = Literal["sparse", "dense"]


class Equalizer(ABC):
    """Base class for DD-domain OTFS equalisers."""

    name: ClassVar[str] = ""
    channel_form: ClassVar[ChannelForm]

    @abstractmethod
    def llr(
        self,
        y_dd: torch.Tensor,
        channel: ChannelView,
        no: torch.Tensor,
    ) -> torch.Tensor:
        """Return bit LLRs with Sionna convention ``LLR > 0`` means bit 1."""
