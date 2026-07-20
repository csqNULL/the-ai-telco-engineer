# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
Link configuration for OTFS Detector evaluation.

This file defines the system constants used by the evaluation script.

The values here mirror the scenario used to generate the EP baseline BLER
shipped with this task (``baseline_bler.pkl``), so that any candidate
detector is benchmarked under exactly the same channel/link conditions.
"""

# OTFS delay-Doppler grid
M = 64
N = 64

# OFDM/CP physical timing
SUBCARRIER_SPACING = 15e3
CARRIER_FREQUENCY = 4.0e9
CP_SAMPLES = 16

# Multipath channel
PROFILE = "continuous-p6"
NUM_PATHS = 6
DELAY_SPREAD_BINS = 14.0
K_SYM_MAX = 0.5
DOPPLER_HZ_MAX = K_SYM_MAX * SUBCARRIER_SPACING

# Analytical IO settings (full_period_io=True makes the window args unused)
FULL_PERIOD_IO = True
DELAY_WINDOW = 5
DOPPLER_WINDOW = 5

TOP_K_DEFAULT = 256

# Modulation + LDPC
NUM_BITS_PER_SYMBOL = 4
CODERATE = 0.5
LDPC_ITERS = 20

# Evaluation SNR grid [dB] (must match the precomputed EP baseline).
SNR_DB = [13.0, 16.0]

# Monte-Carlo settings (match the EP baseline exactly).
BATCH_SIZE = 8
MAX_MC_ITER = 200
NUM_TARGET_BLOCK_ERRORS = 50
TARGET_BLER = 0.0
EARLY_STOP = True
SEED = 0

PRECISION = "single"
