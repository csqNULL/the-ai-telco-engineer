# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Reproduce the shipped ``baseline_bler.pkl`` reference curve.

The OTFS detector task scores a candidate against a reference coded-BLER
curve stored in ``baseline_bler.pkl`` next to ``eval.py``. That reference
is simply the **EP** detector (``otfs.equalizers.EP``) run through the
*same* coded-OTFS link and Monte-Carlo settings the evaluation harness
uses for every candidate.

This script regenerates that curve from scratch so you can inspect exactly
how it is produced. It imports ``eval.py`` and drives its
:class:`~eval.OtfsEvalModel` + ``sim_ber`` pipeline with the EP detector,
using the scenario in ``link_config.py`` (the single source of truth for
grid size, channel, code rate, SNR grid, seeds, ...) and the same seeding
sequence as :func:`eval.evaluate_detector`. No inputs from the benchmark
tooling are needed.

Usage (run from this ``eval/`` directory)::

    python build_baseline.py                 # writes ./baseline_bler.pkl
    python build_baseline.py --output out.pkl
    python build_baseline.py --verbose        # show sim_ber progress

The output pickle stores a ``(snr_db, bler)`` tuple of ``np.ndarray`` --
the exact layout ``eval.py`` loads via ``_load_baseline_bler``.

Note: the Monte-Carlo simulation runs on GPU when available; tiny run-to-run
differences in the integer block-error counts are possible due to
floating-point non-determinism, so the reproduced BLER may differ from the
shipped values by a small amount.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import pickle
import sys
from contextlib import redirect_stdout

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent          # .../tasks/otfs_detector/eval
_TASK = _HERE.parent                                     # .../tasks/otfs_detector

# Make ``otfs`` (task root) and ``link_config`` / ``eval`` (this dir) importable
# when the script is run directly from the repo, mirroring the staged workspace
# layout the harness runs in.
for _p in (str(_TASK), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch

# Importing eval reproduces the harness's global setup (pins CUDA device 0 when
# unset, fixes the Sionna seed), so the baseline is generated under identical
# conditions to a candidate evaluation.
import eval as eval_mod
from eval import (
    BATCH_SIZE,
    EARLY_STOP,
    MAX_MC_ITER,
    NUM_TARGET_BLOCK_ERRORS,
    SEED,
    TARGET_BLER,
    OtfsEvalModel,
    sim_ber,
)
from otfs.equalizers import EP

import link_config


_DEFAULT_OUTPUT = _HERE / "baseline_bler.pkl"


def build(output_pkl: pathlib.Path, *, verbose: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Simulate the EP reference curve and pickle it to ``output_pkl``."""
    snr_db = np.asarray(link_config.SNR_DB, dtype=np.float64)

    # Same construction + seeding sequence as eval.evaluate_detector so the
    # reproduced curve matches the one candidates are scored against.
    detector = eval_mod._build_detector(EP)
    torch.manual_seed(SEED)
    model = OtfsEvalModel(detector)

    mid_snr_db = float(np.mean(snr_db))
    model.build_llr_fn(mid_snr_db)  # compile (or eager fallback); consumes RNG
    torch.manual_seed(SEED)         # re-seed so sim_ber is reproducible

    ebno_dbs = torch.tensor(snr_db.tolist(), dtype=model.dtype, device=model.device)

    sink = sys.stdout if verbose else open(os.devnull, "w")
    try:
        with redirect_stdout(sink):
            _, bler = sim_ber(
                mc_fun=model.mc_fun,
                ebno_dbs=ebno_dbs,
                batch_size=BATCH_SIZE,
                max_mc_iter=MAX_MC_ITER,
                num_target_block_errors=NUM_TARGET_BLOCK_ERRORS,
                target_bler=TARGET_BLER,
                early_stop=EARLY_STOP,
                soft_estimates=False,
                verbose=verbose,
            )
    finally:
        if sink is not sys.stdout:
            sink.close()

    bler = bler.detach().cpu().numpy().astype(np.float64)

    output_pkl.parent.mkdir(parents=True, exist_ok=True)
    with output_pkl.open("wb") as f:
        pickle.dump((snr_db, bler), f)

    print(f"wrote {output_pkl}")
    print(f"  snr_db = {snr_db.tolist()}")
    print(f"  bler   = {bler.tolist()}")
    return snr_db, bler


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=pathlib.Path, default=_DEFAULT_OUTPUT)
    ap.add_argument("--verbose", action="store_true", help="show sim_ber progress")
    args = ap.parse_args()
    build(args.output, verbose=args.verbose)


if __name__ == "__main__":
    main()
