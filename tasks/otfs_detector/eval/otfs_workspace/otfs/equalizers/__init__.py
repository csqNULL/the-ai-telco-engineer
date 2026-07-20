# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Equalizers for the analytical-IO OTFS benchmark.

Only the abstract :class:`Equalizer` base is shipped in the agent
workspace; the concrete detectors are provided by the candidate.
"""

from .base import Equalizer

__all__ = ["Equalizer"]
