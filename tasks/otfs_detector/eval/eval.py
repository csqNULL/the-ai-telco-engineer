# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
OTFS Detector Evaluation Script.

This script evaluates an agent-provided DD-domain OTFS detector against a
precomputed EP baseline BLER curve. The candidate detector must be
implemented in a Python file (default ``draft.py``) in the same
directory and must expose **exactly one** ``Equalizer`` subclass
(see ``otfs/equalizers/base.py``).

Each detector class declares:

* ``channel_form``: ``"sparse"`` or ``"dense"`` — the channel
  representation the harness should hand to ``llr``.
* ``top_k`` (optional): per-output-row sparsity used when building the
  channel view. ``None`` falls back to the scenario default
  (``link_config.TOP_K_DEFAULT``).

The metric is the Normalised Validation Error (NVE) — the average ratio
of the candidate's coded BLER to the EP baseline BLER (loaded from
``baseline_bler.pkl``) at the evaluation SNR points. ``baseline_bler.pkl``
stores a ``(snrs, blers)`` tuple; the SNR grid embedded in that file is
the single source of truth for the evaluation SNR points.

The candidate's ``llr`` is compiled **once** with ``torch.compile`` and the
*same* compiled callable is used for both the BLER scoring (``sim_ber`` via
:meth:`OtfsEvalModel.mc_fun`) and the latency measurement, so the search scores
and times the identical artifact. If compilation fails (complex-dtype codegen,
unsupported data-dependent control flow, ...) the harness resets Dynamo and
falls back to eager for *both* paths so the candidate is still scored
(see :meth:`OtfsEvalModel.build_llr_fn`). The compile status (single graph /
graph breaks / eager fallback) and any PyTorch warnings emitted during
compilation are captured and appended to the SUCCESS output
(see :meth:`OtfsEvalModel.compile_report`).

The complexity returned is the average **total** runtime of that compiled
``detector.llr`` alone for one batch of ``BATCH_SIZE`` frames, in
**milliseconds** (channel generation, encoding, decoding etc. are excluded). It
is measured with a **fresh random input drawn for every timed call** so that no
per-frame channel-state caching inside the detector can be exploited. Timing
uses CUDA events, with input generation fenced out of the timed region (see
``_measure_latency``).

Usage::

    python eval.py [source_file]
        source_file: path to the detector module (default: draft.py).
"""

from __future__ import annotations

import importlib.util
import inspect
import os
import pickle
import sys
import time
import traceback
import warnings
from contextlib import redirect_stdout

if os.getenv("CUDA_VISIBLE_DEVICES") is None:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import torch

import sionna.phy
sionna.phy.config.seed = 42

import logging
logging.getLogger("torch._dynamo").setLevel(logging.ERROR)

from sionna.phy.fec.ldpc import LDPC5GDecoder, LDPC5GEncoder
from sionna.phy.mapping import Mapper
from sionna.phy.utils import sim_ber

from otfs import ChannelView, MultipathChannel, RectCPOtfsIO, ResourceGrid
from otfs.equalizers.base import Equalizer

from link_config import (
    BATCH_SIZE,
    CARRIER_FREQUENCY,
    CODERATE,
    CP_SAMPLES,
    DELAY_SPREAD_BINS,
    DELAY_WINDOW,
    DOPPLER_WINDOW,
    EARLY_STOP,
    FULL_PERIOD_IO,
    K_SYM_MAX,
    LDPC_ITERS,
    M,
    MAX_MC_ITER,
    N,
    NUM_BITS_PER_SYMBOL,
    NUM_PATHS,
    NUM_TARGET_BLOCK_ERRORS,
    PRECISION,
    PROFILE,
    SEED,
    SUBCARRIER_SPACING,
    TARGET_BLER,
    TOP_K_DEFAULT,
)


NUM_TIMING_RUNS = 10
"""Number of timed ``detector.llr`` calls, each on a fresh random input."""

NUM_WARMUP_RUNS = 5
"""Warm-up calls before timing (also triggers ``torch.compile``), fresh inputs."""

COMPILE_MODE = "default"
"""``torch.compile`` mode used for latency measurement.

``"default"`` (TorchInductor) is used rather than ``"reduce-overhead"``:
CUDA-graph replay assumes static input buffers, which is incompatible with the
fresh-input-every-call protocol and can report nonsensical (near-zero) times.
"""


def _device_string() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _dtype_real() -> torch.dtype:
    return torch.float32 if PRECISION == "single" else torch.float64


def _dtype_complex() -> torch.dtype:
    return torch.complex64 if PRECISION == "single" else torch.complex128


class OtfsEvalModel:
    """Self-contained end-to-end coded OTFS link used by ``sim_ber``.

    Mirrors :class:`benchmark.model.EndToEndModel` but reads its
    parameters from ``link_config`` and respects an optional
    ``top_k`` class attribute on the agent's detector. The harness
    always builds the channel view at ``equalizer.top_k`` (falling back
    to ``TOP_K_DEFAULT`` if unset / ``None``), so sparse and dense forms
    carry exactly the same information for a given detector.
    """

    def __init__(self, equalizer: Equalizer) -> None:
        device = _device_string()
        self.device = device
        self.dtype = _dtype_real()
        self.cdtype = _dtype_complex()
        self.equalizer = equalizer

        self.grid = ResourceGrid(
            num_delay_bins=M,
            num_doppler_bins=N,
            subcarrier_spacing=SUBCARRIER_SPACING,
            carrier_frequency=CARRIER_FREQUENCY,
            cp_samples=CP_SAMPLES,
            precision=PRECISION,
            device=device,
        )
        self.channel = MultipathChannel(
            self.grid,
            profile=PROFILE,
            num_paths=NUM_PATHS,
            l_max=DELAY_SPREAD_BINS,
            k_sym_max=K_SYM_MAX,
        )
        self.io = RectCPOtfsIO(
            self.grid,
            delay_window=DELAY_WINDOW,
            doppler_window=DOPPLER_WINDOW,
            full_period=FULL_PERIOD_IO,
        )

        self.n_coded = self.grid.num_resource_elements * NUM_BITS_PER_SYMBOL
        self.k_info = int(round(CODERATE * self.n_coded))
        self.encoder = LDPC5GEncoder(
            k=self.k_info,
            n=self.n_coded,
            precision=PRECISION,
            device=device,
        )
        self.decoder = LDPC5GDecoder(
            self.encoder,
            num_iter=LDPC_ITERS,
            hard_out=True,
            return_infobits=True,
            precision=PRECISION,
            device=device,
        )
        self.mapper = Mapper(
            constellation_type="qam",
            num_bits_per_symbol=NUM_BITS_PER_SYMBOL,
            precision=PRECISION,
            device=device,
        )

        self.top_k = self._resolve_top_k(equalizer)
        self._llr_fn = None
        self._llr_compiled: bool | None = None
        self._graph_breaks: int = 0
        self._compile_warnings: list[str] = []

    @staticmethod
    def _resolve_top_k(equalizer: Equalizer) -> int:
        """Read ``top_k`` off the detector class, clamp into ``[1, M*N]``."""
        requested = getattr(equalizer, "top_k", None)
        if requested is None:
            requested = TOP_K_DEFAULT
        try:
            top_k = int(requested)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"Detector.top_k must be an int or None, got {requested!r}"
            ) from exc
        mn = int(M * N)
        if top_k < 1 or top_k > mn:
            raise ValueError(
                f"Detector.top_k={top_k} out of range; expected 1..{mn}."
            )
        return top_k

    def noise_variance(self, snr_db: torch.Tensor) -> torch.Tensor:
        return (10.0 ** (-snr_db / 10.0)).to(dtype=self.dtype, device=self.device)

    def build_channel_view(self, cir) -> ChannelView:
        sparse = self.io.sparse(cir, top_k=self.top_k)
        form = self.equalizer.channel_form
        if form == "sparse":
            return ChannelView(cir=cir, sparse=sparse, dense=None)
        if form == "dense":
            return ChannelView(cir=cir, sparse=None, dense=sparse.to_dense())
        raise ValueError(
            f"Unsupported channel_form {form!r}; expected 'sparse' or 'dense'."
        )

    def _broadcast_no(self, no: torch.Tensor, batch_size: int) -> torch.Tensor:
        """``sim_ber`` may hand us a scalar noise variance; per-frame
        detectors expect a ``(batch,)`` tensor."""
        if no.dim() == 0:
            return no.expand(batch_size)
        return no

    def mc_fun(self, batch_size: int, ebno_db: torch.Tensor):
        no_scalar = self.noise_variance(ebno_db)
        cir = self.channel(batch_size)
        u = (
            torch.rand(batch_size, self.k_info, dtype=self.dtype, device=self.device)
            > 0.5
        ).to(self.dtype)
        c = self.encoder(u)
        x_dd = self.mapper(c)
        y_dd = self.io.apply(x_dd, cir, noise_variance=no_scalar)
        view = self.build_channel_view(cir)
        no_batch = self._broadcast_no(no_scalar, batch_size)
        llr = self.build_llr_fn()(y_dd, view, no_batch)
        u_hat = self.decoder(llr)
        return u, u_hat

    def timing_sample(self, batch_size: int, snr_db: float):
        """Draw a realistic ``(y_dd, channel_view, no)`` sample for timing.

        Uses the same transmit pipeline ``sim_ber`` exercises (random
        bits -> LDPC -> QAM -> analytical DD channel + AWGN) at the
        requested SNR, with the detector-requested channel form/top_k.
        """
        ebno_db = torch.tensor(snr_db, dtype=self.dtype, device=self.device)
        no_scalar = self.noise_variance(ebno_db)
        cir = self.channel(batch_size)
        u = (
            torch.rand(batch_size, self.k_info, dtype=self.dtype, device=self.device)
            > 0.5
        ).to(self.dtype)
        c = self.encoder(u)
        x_dd = self.mapper(c)
        y_dd = self.io.apply(x_dd, cir, noise_variance=no_scalar)
        view = self.build_channel_view(cir)
        no_batch = self._broadcast_no(no_scalar, batch_size)
        return y_dd, view, no_batch

    def build_llr_fn(self, compile_snr_db: float | None = None):
        """Return ``detector.llr`` compiled with ``torch.compile`` (cached).

        Both the BLER scoring (:meth:`mc_fun` via ``sim_ber``) and the latency
        measurement use this single compiled callable, so the search scores and
        times *the same* compiled artifact. Compilation is triggered eagerly on
        one representative sample so that a compile failure is detected *before*
        ``sim_ber`` runs; on failure we reset Dynamo and fall back to eager for
        both paths so the candidate is still scored rather than discarded.

        Triggering consumes from the torch RNG stream; callers that need a
        reproducible ``sim_ber`` should re-seed *after* this returns.
        """
        if self._llr_fn is not None:
            return self._llr_fn
        base = self.equalizer.llr
        snr = compile_snr_db if compile_snr_db is not None else 10.0
        try:
            from torch._dynamo.utils import counters
        except Exception:
            counters = None
        try:
            fn = torch.compile(base, mode=COMPILE_MODE, dynamic=False)
            if counters is not None:
                counters.clear()
            # Capture PyTorch warnings emitted while the graph is lowered on the
            # first (compiling) call -- e.g. complex-dtype codegen fallbacks --
            # so they can be surfaced to the agent (stderr is otherwise dropped).
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                with torch.no_grad():
                    y, view, no = self.timing_sample(BATCH_SIZE, snr)
                    fn(y, view, no)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            self._compile_warnings = _summarize_warnings(caught)
            if counters is not None and "graph_break" in counters:
                self._graph_breaks = int(sum(counters.get("graph_break", {}).values()))
            self._llr_compiled = True
        except Exception:
            try:
                torch._dynamo.reset()
            except Exception:
                pass
            fn = base
            self._llr_compiled = False
        self._llr_fn = fn
        return fn

    def compile_report(self) -> str:
        """Human-readable note on torch.compile status + captured warnings.

        Appended to the SUCCESS line so the agent can tell whether its ``llr``
        actually compiled into a single fused graph, fell back to eager, or hit
        graph breaks / unsupported-op (e.g. complex-dtype) codegen fallbacks
        that make the measured complexity worse than a compile-clean version.
        """
        lines: list[str] = []
        if self._llr_compiled is True:
            if self._graph_breaks > 0:
                lines.append(
                    f"[compile] torch.compile(mode=\"{COMPILE_MODE}\") succeeded but with "
                    f"{self._graph_breaks} graph break(s): part of llr ran eager. "
                    "Removing graph breaks usually lowers the complexity metric."
                )
            else:
                lines.append(
                    f"[compile] torch.compile(mode=\"{COMPILE_MODE}\") captured llr as a "
                    "single graph (0 graph breaks)."
                )
        elif self._llr_compiled is False:
            lines.append(
                f"[compile] torch.compile(mode=\"{COMPILE_MODE}\") FAILED; fell back to "
                "EAGER for both NVE scoring and the complexity measurement."
            )
        if self._compile_warnings:
            lines.append("[compile warnings] PyTorch warnings emitted during compilation:")
            lines.extend(f"  - {w}" for w in self._compile_warnings)
            lines.append(
                "  (note: warnings like \"does not support code generation for complex "
                "operators\" mean Inductor ran those ops eagerly -- the graph compiled but "
                "the complexity metric will be worse than a fully fused implementation.)"
            )
        return "\n".join(lines)


def _summarize_warnings(caught, max_unique: int = 8, max_len: int = 240) -> list[str]:
    """De-duplicate captured warnings, keeping PyTorch-originated ones.

    Returns at most ``max_unique`` ``"<Category>: <message>"`` strings, each
    truncated to ``max_len`` chars. Warnings whose origin file is not under a
    ``torch`` package are dropped to keep the report focused on compile-relevant
    signals rather than unrelated library deprecations.
    """
    seen: set[tuple[str, str]] = set()
    out: list[str] = []
    for w in caught:
        filename = (getattr(w, "filename", "") or "").lower()
        if not any(tag in filename for tag in ("torch", "dynamo", "inductor")):
            continue
        category = getattr(w.category, "__name__", "Warning")
        msg = " ".join(str(w.message).split())
        if len(msg) > max_len:
            msg = msg[: max_len - 1] + "\u2026"
        key = (category, msg)
        if key in seen:
            continue
        seen.add(key)
        out.append(f"{category}: {msg}")
        if len(out) >= max_unique:
            break
    return out


def _short_traceback(limit: int = -5) -> str:
    exc_type, exc_value, exc_tb = sys.exc_info()
    return "".join(traceback.format_exception(exc_type, exc_value, exc_tb, limit=limit))


def _load_module(source_file: str):
    """Load an agent-provided Python module (e.g. draft.py or solution.py)."""
    module_name = os.path.splitext(os.path.basename(source_file))[0]
    spec = importlib.util.spec_from_file_location(module_name, source_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {source_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _find_equalizer_class(module) -> type[Equalizer]:
    """Locate the single ``Equalizer`` subclass defined in ``module``."""
    candidates = []
    for obj in vars(module).values():
        if (
            inspect.isclass(obj)
            and issubclass(obj, Equalizer)
            and obj is not Equalizer
            and obj.__module__ == module.__name__
        ):
            candidates.append(obj)
    if len(candidates) != 1:
        names = [c.__name__ for c in candidates] or ["<none>"]
        raise ValueError(
            "Source file must define exactly one Equalizer subclass; "
            f"found {names}."
        )
    return candidates[0]


def _build_detector(cls: type[Equalizer]) -> Equalizer:
    """Construct the detector with the standard kwargs."""
    device = _device_string()
    grid = ResourceGrid(
        num_delay_bins=M,
        num_doppler_bins=N,
        subcarrier_spacing=SUBCARRIER_SPACING,
        carrier_frequency=CARRIER_FREQUENCY,
        cp_samples=CP_SAMPLES,
        precision=PRECISION,
        device=device,
    )
    detector = cls(
        grid,
        num_bits_per_symbol=NUM_BITS_PER_SYMBOL,
        precision=PRECISION,
        device=device,
    )
    if detector.channel_form not in ("sparse", "dense"):
        raise ValueError(
            f"Detector.channel_form must be 'sparse' or 'dense', "
            f"got {detector.channel_form!r}."
        )
    return detector


def _load_baseline_bler() -> tuple[np.ndarray, np.ndarray]:
    """Load the baseline BLER and the evaluation SNR grid.

    ``baseline_bler.pkl`` stores a ``(snrs, blers)`` tuple. The SNR grid
    embedded in the file is the single source of truth: callers must
    use the returned ``snrs`` rather than a separate constant.
    """
    with open("baseline_bler.pkl", "rb") as f:
        base_snrs, base_blers = pickle.load(f)
    base_snrs = np.asarray(base_snrs, dtype=np.float64)
    base_blers = np.asarray(base_blers, dtype=np.float64)
    if base_snrs.shape != base_blers.shape:
        raise ValueError(
            f"baseline_bler.pkl SNR shape {base_snrs.shape} does not match "
            f"BLER shape {base_blers.shape}"
        )
    return base_snrs, base_blers


def _measure_latency(model: OtfsEvalModel, mid_snr_db: float) -> float:
    """Total runtime (milliseconds) of the shared compiled ``detector.llr``.

    The callable timed here is exactly the one used for BLER scoring
    (:meth:`OtfsEvalModel.build_llr_fn`, compiled with ``torch.compile`` or the
    eager fallback if compilation failed), so the search times the same artifact
    it scores. Each call processes a batch of ``BATCH_SIZE`` frames; the mean
    per-call time is converted to milliseconds and returned as-is, so the
    returned value is the total latency of one ``BATCH_SIZE``-frame call.

    **Fresh random inputs every call** — a new channel realisation + transmit
    vector is drawn inside the timing loop for every timed call, so any per-frame
    channel-state cache inside the detector (commonly keyed on
    ``h_vals.data_ptr()``) cannot be exploited, exactly as frame-to-frame in a
    real receiver. Input generation is fenced out of the timed region with a
    ``cuda.synchronize`` before the CUDA-event start.
    """
    cuda = torch.cuda.is_available()
    fn = model.build_llr_fn(mid_snr_db)

    def _gen():
        return model.timing_sample(batch_size=BATCH_SIZE, snr_db=mid_snr_db)

    with torch.no_grad():
        for _ in range(NUM_WARMUP_RUNS):
            y, view, no = _gen()
            fn(y, view, no)
    if cuda:
        torch.cuda.synchronize()

    per_call_s = []
    with torch.no_grad():
        for _ in range(NUM_TIMING_RUNS):
            y, view, no = _gen()  # fresh random channel + data every call
            if cuda:
                torch.cuda.synchronize()  # fence input generation out of timing
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                fn(y, view, no)
                end.record()
                torch.cuda.synchronize()
                per_call_s.append(start.elapsed_time(end) * 1e-3)
            else:
                t0 = time.perf_counter()
                fn(y, view, no)
                per_call_s.append(time.perf_counter() - t0)

    # ``per_call_s`` holds per-batch-call seconds; report the total latency of
    # one ``BATCH_SIZE``-frame call, in milliseconds.
    mean_per_call_s = float(np.mean(per_call_s))
    return mean_per_call_s * 1e3


def evaluate_detector(source_file: str = "draft.py") -> str:
    """Main evaluation entry point.

    Output format::

        SUCCESS, <nve>, <avg_llr_runtime_ms_per_batch>
        FAILURE,
        <error details>

    ``<avg_llr_runtime_ms_per_batch>`` is the average total runtime of
    ``detector.llr`` alone for one ``BATCH_SIZE``-frame batch, in milliseconds
    (see ``NUM_TIMING_RUNS`` and ``_measure_latency``).
    """
    try:
        module = _load_module(source_file)
    except ImportError as e:
        return f"FAILURE,\nERROR: Could not import from {source_file}: {e}\n\n{_short_traceback()}"
    except SyntaxError as e:
        return f"FAILURE,\nERROR: Syntax error in {source_file}: {e}\n\n{_short_traceback()}"
    except Exception as e:
        return f"FAILURE,\nERROR: Failed to load {source_file}: {e}\n\n{_short_traceback()}"

    try:
        eq_cls = _find_equalizer_class(module)
    except Exception as e:
        return f"FAILURE,\nERROR: {e}\n\n{_short_traceback()}"

    try:
        detector = _build_detector(eq_cls)
    except Exception as e:
        return f"FAILURE,\nERROR: Could not instantiate detector: {e}\n\n{_short_traceback()}"

    try:
        torch.manual_seed(SEED)
        model = OtfsEvalModel(detector)

        baseline_snrs, baseline_blers = _load_baseline_bler()
        snr_points = baseline_snrs
        mid_snr_db = float(np.mean(snr_points))

        # Compile detector.llr once (or fall back to eager) BEFORE scoring, so
        # sim_ber and the latency measurement use the same artifact. Triggering
        # consumes RNG, so re-seed afterwards to keep sim_ber reproducible.
        model.build_llr_fn(mid_snr_db)
        torch.manual_seed(SEED)

        ebno_dbs = torch.tensor(snr_points.tolist(), dtype=model.dtype, device=model.device)

        with open(os.devnull, "w") as devnull, redirect_stdout(devnull):
            _, bler = sim_ber(
                mc_fun=model.mc_fun,
                ebno_dbs=ebno_dbs,
                batch_size=BATCH_SIZE,
                max_mc_iter=MAX_MC_ITER,
                num_target_block_errors=NUM_TARGET_BLOCK_ERRORS,
                target_bler=TARGET_BLER,
                early_stop=EARLY_STOP,
                soft_estimates=False,
                verbose=False,
            )

        bler = bler.detach().cpu().numpy().astype(np.float64)

        if np.all(bler >= 1.0 - 1e-12):
            return (
                "FAILURE,\n"
                "ERROR: Candidate BLER is 1.0 at every evaluation SNR point "
                f"({snr_points.tolist()} dB) - the LDPC decoder failed on every "
                "block. The detector's LLRs are not aligned with what the decoder "
                "expects, so the output carries no usable information; tuning "
                "hyperparameters will not change the result.\n"
                "Most common causes (check before iterating on the algorithm):\n"
                "  - hand-rolled 16-QAM constellation or bit-label table that "
                "does not match `sionna.phy.mapping.Mapper(\"qam\", "
                "num_bits_per_symbol)` on the TX side (Sionna's QAM uses an "
                "interleaved I/Q label layout, not (I-bits | Q-bits));\n"
                "  - LLR sign flip (Sionna expects "
                "`LLR(b) = log P(b=1|y) - log P(b=0|y)`);\n"
                "  - wrong output ordering (must be (batch, M*N*K) with bit k of "
                "symbol r at index r*K + k).\n"
                "Fix: use `otfs.make_demapper(num_bits_per_symbol)` (Path A) or "
                "`otfs.qam_constellation_with_labels(num_bits_per_symbol)` (Path B) "
                "from the staged otfs/ package - both are guaranteed to match the "
                "TX-side Mapper. See the 'LLR convention' section of the task "
                "prompt for full snippets."
            )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        torch.manual_seed(SEED + 10_000)
        elapsed = _measure_latency(model, mid_snr_db)

        mask = baseline_blers > 0
        if not np.any(mask):
            return "FAILURE,\nERROR: Baseline BLER is zero at all evaluation SNR points."

        nve = float(np.mean(bler[mask] / baseline_blers[mask]))
        if not np.isfinite(nve):
            return "FAILURE,\nERROR: NaN or Inf NVE"

        result = f"SUCCESS, {nve:.4f}, {elapsed:.3e}"
        report = model.compile_report()
        if report:
            result = f"{result}\n{report}"
        return result

    except Exception as e:
        return f"FAILURE,\nERROR: Runtime error during evaluation: {e}\n\n{_short_traceback()}"


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "draft.py"
    print(evaluate_detector(src))
