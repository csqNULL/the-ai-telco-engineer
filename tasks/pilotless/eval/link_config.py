# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
Link configuration for Pilotless Constellation evaluation.

This file defines the system constants used by the evaluation script.
"""

# System constants
NUM_BITS_PER_SYMBOL = 6
NUM_OFDM_SYMBOLS = 14
FFT_SIZE = 72
SUBCARRIER_SPACING = 30e3
CYCLIC_PREFIX_LENGTH = 0
NUM_GUARD_CARRIERS = [0, 0]
DC_NULL = False
CODERATE = 0.7

# Carrier and channel parameters
CARRIER_FREQUENCY = 2.6e9
