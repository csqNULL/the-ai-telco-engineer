# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
Pilotless Receiver Evaluation Script for Sionna.

This script evaluates an agent-provided pilotless OFDM receiver for a
single-antenna uplink over a 3GPP TDL-C channel.

The transmitter maps coded bits to a custom (pre-trained) constellation loaded
from ``constellation_points.pkl`` and sends them over a resource grid without
any DMRS pilots. The receiver must therefore recover per-bit LLRs directly from
the received resource grid.

The receiver must be implemented in a Python file (default ``draft.py``) in
the same directory, exposing a callable ``receiver(y, no)`` where

    y  : [batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size] complex
        Received resource grid.
    no : [batch_size] float
        Noise variance per batch element.

and returning LLRs on the transmitted coded bits with shape

    [batch_size, num_rx, num_streams_per_rx, num_ofdm_symbols, fft_size,
     num_bits_per_symbol]

i.e. the layout expected by ``ResourceGridDemapper``.

The metric reported is the Normalised BLER vs. Eb/N0 (NVE), defined as the
average ratio of the agent's BLER to a baseline BLER (loaded from
``baseline_bler.pkl``) at the evaluation SNR points. ``baseline_bler.pkl``
stores a ``(snrs, blers)`` tuple; the SNR grid embedded in that file is the
single source of truth for the evaluation SNR points.

The harness always wraps ``receiver`` with ``torch.compile`` before NVE and
latency measurement. The implementation must be compile-compatible (see the
task prompt).

Usage: python eval.py [source_file]
  source_file: path to the receiver module (default: draft.py).
"""
import os
import sys
import importlib.util
import time
import traceback
import tempfile
from contextlib import contextmanager

if os.getenv("CUDA_VISIBLE_DEVICES") is None:
    gpu_num = 0
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"

import torch
import numpy as np
import pickle
import sionna.phy
sionna.phy.config.seed = 42

import logging
logging.getLogger("torch._dynamo").setLevel(logging.ERROR)

from sionna.phy import Block
from sionna.phy.ofdm import ResourceGridMapper, ResourceGrid, ResourceGridDemapper
from sionna.phy.mimo import StreamManagement
from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
from sionna.phy.mapping import Mapper, BinarySource, Constellation
from sionna.phy.channel import OFDMChannel
from sionna.phy.channel.tr38901 import TDL
from sionna.phy.utils import ebnodb2no, sim_ber, expand_to_rank

from link_config import NUM_BITS_PER_SYMBOL, CODERATE, CARRIER_FREQUENCY

# Single-stream uplink: one RX observing one TX stream
STREAM_MANAGEMENT = StreamManagement(np.array([[1]]), 1)

# Resource grid *without* DMRS pilots (pilotless transmission)
RG = ResourceGrid(num_ofdm_symbols=14,
                  fft_size=72,
                  subcarrier_spacing=30e3,
                  num_tx=1,
                  num_streams_per_tx=1,
                  cyclic_prefix_length=6)

# Evaluation SNR points [dB] and Monte-Carlo settings
BATCH_SIZE = 10
MAX_MC_ITER = 100 # 1000
NUM_TARGET_BLOCK_ERRORS = 100
TARGET_BLER = 1e-3

# Untimed warm-up receiver calls (JIT / torch.compile) before latency measurement.
NUM_WARMUP_CALLS = 3

# Number of receiver calls used to measure the average per-call runtime.
NUM_TIMING_RUNS = 10

# RNG seeds: NVE uses a fixed seed; timing uses a separate stream so MC length
# does not affect latency inputs and so timing draws are reproducible.
NVE_SEED = 42
TIMING_SEED = NVE_SEED + 10_000

# Maximum UT speed [m/s] used by the TDL-C channel model
SPEED = 3.0

class PilotlessModel(Block):
    """End-to-end pilotless OFDM system used for evaluation.

    The transmitter encodes random bits with a 5G LDPC code, maps them to a
    custom constellation loaded from disk and places them on a resource grid
    with no DMRS pilots. The signal goes through a TDL-C channel with AWGN.
    The agent-provided ``receiver`` is then called to produce LLRs, which are
    decoded to recover the transmitted bits.
    """

    def __init__(self, agent_receiver):
        super().__init__()

        n = int(RG.num_data_symbols * NUM_BITS_PER_SYMBOL)
        k = int(n * CODERATE)
        self._k = k
        self._n = n

        self._binary_source = BinarySource()
        self._encoder = LDPC5GEncoder(k, n)
        self._rg_mapper = ResourceGridMapper(RG)

        # Pre-trained custom constellation (complex64 numpy array of shape
        # [2**NUM_BITS_PER_SYMBOL]).
        with open("constellation_points.pkl", "rb") as f:
            points = pickle.load(f)
        points = torch.as_tensor(np.asarray(points, dtype=np.complex64))
        self.constellation = Constellation(
            "custom",
            NUM_BITS_PER_SYMBOL,
            points=points,
            normalize=True,
            center=True,
        )
        self._mapper = Mapper(constellation=self.constellation)

        # TDL-C channel, matching the notebook used to train the agent.
        channel_model = TDL(
            model="C",
            delay_spread=100e-9,
            carrier_frequency=CARRIER_FREQUENCY,
            min_speed=0.0,
            max_speed=SPEED,
        )
        self._channel = OFDMChannel(
            channel_model,
            RG,
            add_awgn=True,
            normalize_channel=True,
            return_channel=False,
        )

        self._agent_receiver = agent_receiver
        self._rg_demapper = ResourceGridDemapper(RG, STREAM_MANAGEMENT)
        self._decoder = LDPC5GDecoder(self._encoder, hard_out=True)

    def _transmit(self, batch_size: int, ebno_db: torch.Tensor):
        """Run the TX + channel pipeline and return ``(y, no)``."""
        no = ebnodb2no(
            ebno_db,
            num_bits_per_symbol=NUM_BITS_PER_SYMBOL,
            coderate=CODERATE,
            resource_grid=RG,
        )
        if no.dim() == 0:
            no = no.expand(batch_size)

        bits = self._binary_source([batch_size, 1, 1, self._k])
        codewords = self._encoder(bits)

        x = self._mapper(codewords)
        x_rg = self._rg_mapper(x)

        no_expanded = expand_to_rank(no, x_rg.ndim)
        y = self._channel(x_rg, no_expanded)
        return bits, y, no

    def call(self, batch_size: int, ebno_db: torch.Tensor):
        bits, y, no = self._transmit(batch_size, ebno_db)

        llr = self._agent_receiver(y, no)
        llr = self._rg_demapper(llr)
        llr = llr.reshape(batch_size, 1, 1, self._n)

        bits_hat = self._decoder(llr)
        return bits, bits_hat


def _short_traceback(limit: int = -3) -> str:
    """Return a truncated traceback showing only the last ``abs(limit)`` frames."""
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


def _load_baseline_bler() -> tuple[np.ndarray, np.ndarray]:
    """Load the baseline BLER and the evaluation SNR grid.

    ``baseline_bler.pkl`` stores a ``(snrs, blers)`` tuple. The SNR grid
    embedded in the file is the single source of truth: callers should use
    the returned ``snrs`` rather than a separate constant.
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


def _force_cold_compile() -> None:
    """Clear compiler state/caches so every eval recompiles from scratch.

    torch.compile emits its diagnostic messages (the Inductor "complex
    operators" warning and the ``cudagraph partition`` notes) only on a cold
    compile; on a cache hit it stays silent. We reset dynamo and disable the
    Inductor caches so those warnings are surfaced on every evaluation. This
    costs one compile per eval (no cache reuse) — best-effort and tolerant of
    torch-version differences.
    """
    try:
        torch._dynamo.reset()
    except Exception:
        pass
    try:
        import torch._inductor.config as _inductor_config
        if hasattr(_inductor_config, "force_disable_caches"):
            _inductor_config.force_disable_caches = True
        if hasattr(_inductor_config, "fx_graph_cache"):
            _inductor_config.fx_graph_cache = False
    except Exception:
        pass


def _compile_receiver(receiver):
    """Wrap the agent ``receiver`` with ``torch.compile`` (required for evaluation)."""
    if not hasattr(torch, "compile"):
        raise RuntimeError("torch.compile requires PyTorch >= 2.0")
    _force_cold_compile()
    compile_kwargs = {"fullgraph": False}
    if torch.cuda.is_available():
        compile_kwargs["mode"] = "reduce-overhead"
    return torch.compile(receiver, **compile_kwargs)


def _measure_receiver_latency(
    model: PilotlessModel,
    receiver,
    snr_min: float,
    snr_max: float,
) -> float:
    """Average per-call wall-clock time of ``receiver`` on varied (y, no) samples.

    Pre-generates ``NUM_WARMUP_CALLS + NUM_TIMING_RUNS`` batches. Each batch draws
    Eb/N0 uniformly in ``[snr_min, snr_max]`` per batch element, then runs TX +
    channel via ``model._transmit``. Warm-up calls are untimed; only ``receiver``
    on the last ``NUM_TIMING_RUNS`` batches is timed.
    """
    with torch.no_grad():
        samples = []
        for _ in range(NUM_WARMUP_CALLS + NUM_TIMING_RUNS):
            ebno_db = torch.empty(BATCH_SIZE, dtype=torch.float32).uniform_(
                snr_min, snr_max
            )
            _, y, no = model._transmit(BATCH_SIZE, ebno_db)
            samples.append((y.clone(), no.clone()))

        for y, no in samples[:NUM_WARMUP_CALLS]:
            _ = receiver(y, no)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        for y, no in samples[NUM_WARMUP_CALLS:]:
            _ = receiver(y, no)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) / NUM_TIMING_RUNS


@contextmanager
def _capture_stderr():
    """Capture stderr (torch warnings / compile logs) and discard stdout.

    ``sim_ber`` prints its per-iteration BER table to stdout (suppressed here,
    as the original code did). The torch.compile feedback we want — the
    Inductor "complex operators" warning and the ``cudagraph partition``
    messages — goes to stderr. Capturing is at the file-descriptor level
    (``os.dup2``) so output from torch's logging handlers and compile worker
    threads/subprocesses is included. Captured text is in the yielded dict's
    ``"text"`` key.
    """
    box = {"text": ""}
    sys.stdout.flush()
    sys.stderr.flush()
    saved_out, saved_err = os.dup(1), os.dup(2)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    tmp = tempfile.TemporaryFile(mode="w+")
    try:
        os.dup2(devnull_fd, 1)
        os.dup2(tmp.fileno(), 2)
        yield box
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        os.close(saved_out)
        os.close(saved_err)
        os.close(devnull_fd)
        tmp.seek(0)
        box["text"] = tmp.read()
        tmp.close()


@contextmanager
def _suppress_stderr():
    """Send stderr to /dev/null (restore afterward)."""
    sys.stderr.flush()
    saved_err = os.dup(2)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull_fd, 2)
        yield
    finally:
        sys.stderr.flush()
        os.dup2(saved_err, 2)
        os.close(saved_err)
        os.close(devnull_fd)


def evaluate_receiver(source_file: str = "draft.py") -> str:
    """Main evaluation function.

    Loads the ``receiver`` callable from ``source_file`` and runs a
    Monte-Carlo BER simulation. Output format::

        SUCCESS, <nve>, <elapsed_seconds>
        FAILURE,
        <error details>

    ``<elapsed_seconds>`` is the average per-call runtime of the compiled
    ``receiver`` (see ``NUM_TIMING_RUNS``) in scientific notation. Evaluation only
    returns FAILURE on import/runtime errors, compile failures, or numerical
    blow-ups (NaN/Inf NVE).
    """
    try:
        module = _load_module(source_file)
    except ImportError as e:
        return f"FAILURE,\nERROR: Could not import from {source_file}: {e}\n\n{_short_traceback()}"
    except SyntaxError as e:
        return f"FAILURE,\nERROR: Syntax error in {source_file}: {e}\n\n{_short_traceback()}"
    except Exception as e:
        return f"FAILURE,\nERROR: Failed to load {source_file}: {e}\n\n{_short_traceback()}"

    receiver = getattr(module, "receiver", None)
    if not callable(receiver):
        return (
            f"FAILURE,\nERROR: {source_file} must define a callable 'receiver(y, no)' "
            "function.\n\n"
        )

    try:
        sionna.phy.config.seed = NVE_SEED
        torch.manual_seed(NVE_SEED)

        baseline_snrs, baseline_blers = _load_baseline_bler()
        snr_points = baseline_snrs

        # Capture all torch.compile stderr once (sim_ber + latency timing both
        # invoke the compiled receiver and emit the same diagnostics).
        with _capture_stderr() as cap:
            receiver = _compile_receiver(receiver)
            model = PilotlessModel(receiver)

            _, bler = sim_ber(
                model,
                snr_points,
                batch_size=BATCH_SIZE,
                max_mc_iter=MAX_MC_ITER,
                num_target_block_errors=NUM_TARGET_BLOCK_ERRORS,
                target_bler=TARGET_BLER,
            )

        torch_messages = cap["text"].strip()
        bler = bler.cpu().numpy().astype(np.float64)

        snr_min = float(snr_points.min())
        snr_max = float(snr_points.max())
        sionna.phy.config.seed = TIMING_SEED
        torch.manual_seed(TIMING_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(TIMING_SEED)
        # Timing re-invokes the compiled receiver; suppress duplicate stderr
        # (compile diagnostics are already in ``torch_messages`` from sim_ber).
        with _suppress_stderr():
            elapsed = _measure_receiver_latency(model, receiver, snr_min, snr_max)

        # Keep only SNR points where the baseline has non-zero BLER to avoid
        # a 0/0 situation (which would happen above the baseline error floor).
        mask = baseline_blers > 0
        if not np.any(mask):
            return "FAILURE,\nERROR: Baseline BLER is zero at all evaluation SNR points."

        nve = float(np.mean(bler[mask] / baseline_blers[mask]))
        if not np.isfinite(nve):
            return "FAILURE,\nERROR: NaN or Inf"

        result = f"SUCCESS, {nve:.4f}, {elapsed:.3e}"
        if torch_messages:
            result += "\n" + torch_messages
        return result

    except Exception as e:
        return f"FAILURE,\nERROR: Runtime error during evaluation: {e}\n\n{_short_traceback()}"


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "draft.py"
    print(evaluate_receiver(src))
