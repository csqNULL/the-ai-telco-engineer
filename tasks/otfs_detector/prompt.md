# OTFS Delay-Doppler Detector Implementation Task

## Objective

Implement an **OTFS DD-domain detector** that, from a received delay-Doppler vector, perfect channel-state information, and the noise variance, produces soft bit information (log-likelihood ratios) for the channel decoder. Your goal is to **minimise the Normalised Validation Error (NVE)** — the average ratio between the coded BLER achieved by your detector and a reference BLER measured with an Expectation Propagation (EP) detector, across a range of SNR points.

A secondary objective is **runtime**: the framework optimises NVE and the average total `detector.llr` wall-clock time for one batch of `BATCH_SIZE` frames (in milliseconds) jointly on a 2-D Pareto front, so cheaper detectors that match the EP curve are preferred over slower detectors that just barely beat it.

## Context

The system simulates an uncoded-pilot OTFS link entirely in the delay-Doppler (DD) domain using the **analytical rectangular-CP OTFS input-output relation**. There is no OFDM-style forward approximation:

* Information bits are 5G-LDPC-encoded, mapped to a standard QAM constellation, and placed directly on a DD grid of size `M*N` (delay-major order, i.e. flat index `l*N + k`).
* The DD signal goes through a continuous multipath channel `h(t,tau) = sum_i a_i exp(j 2 pi nu_i t) delta(tau - tau_i)`, with per-frame i.i.d. path gains, uniform random delays in `[0, l_max]` delay bins, and uniform random Doppler shifts in `[-k_sym_max, +k_sym_max]` times the subcarrier spacing.
* The receiver gets perfect (path-level) CSI, condensed by the harness into a per-output-row sparse DD channel representation (top-`k` strongest taps per row). Your detector picks both the channel form and `k`.
* The output of your detector is passed to a standard 5G LDPC decoder.

See `otfs/` in the workspace for the full toolkit (`ResourceGrid`, `MultipathChannel`, `RectCPOtfsIO`, `ChannelView`, `SparseDDChannel`, `DenseDDChannel`, `CIR`) and read their docstrings; the analytical IO model is documented inline.

### Scenario (frozen)

These values match the EP baseline you are benchmarked against. They are shown here so you do **not** need to read `link_config.py` during implementation: `link_config.py` is injected only while the evaluation harness is running and may be absent during normal file inspection or smoke tests. In detector code, prefer values passed at runtime (`grid.M`, `grid.N`, `num_bits_per_symbol`, `channel` shapes, and `no`) rather than module-level imports from `link_config.py`.

| Constant | Value | Meaning |
|----------|-------|---------|
| `M` | 64 | Number of delay bins |
| `N` | 64 | Number of Doppler bins |
| `SUBCARRIER_SPACING` | 15 kHz | Subcarrier spacing |
| `CARRIER_FREQUENCY` | 4 GHz | Carrier frequency |
| `CP_SAMPLES` | 16 | CP length in samples |
| `PROFILE` | `"continuous-p6"` | 6-path continuous Doppler profile, first path LoS |
| `NUM_PATHS` | 6 | |
| `DELAY_SPREAD_BINS` | 14.0 | Max delay in delay bins |
| `K_SYM_MAX` | 0.5 | Max normalised Doppler (high Doppler preset, ~7.5 kHz) |
| `FULL_PERIOD_IO` | `True` | Use exact full-period DD IO (no window truncation) |
| `TOP_K_DEFAULT` | 256 | Per-row sparsity used by the EP baseline |
| `NUM_BITS_PER_SYMBOL` | 4 | 16-QAM |
| `CODERATE` | 0.5 | LDPC rate |
| `LDPC_ITERS` | 20 | LDPC inner iterations |
| `SNR_DB` | `[13, 16]` dB | Evaluation SNR grid |

The noise variance the harness passes you obeys the unit-average-symbol-energy convention `N0 = 10**(-SNR_dB / 10)`.

## Requirements

### Class signature

Your solution must subclass `otfs.equalizers.base.Equalizer` and implement `llr`:

```python
import torch
from otfs.equalizers.base import Equalizer
from otfs.channel_view import ChannelView


class MyDetector(Equalizer):
    name = "MyDetector"        # optional, used only for logging
    channel_form = "sparse"    # "sparse" or "dense"
    top_k = 256                # design choice in [1, M*N]; None => 256

    def __init__(self, grid, *, num_bits_per_symbol=4,
                 precision="single", device="cuda:0", **kwargs):
        # grid is an otfs.grid.ResourceGrid with M, N, subcarrier_spacing, ...
        # Precompute anything constant (QAM constellation, bit masks, etc.) here.
        ...

    def llr(self,
            y_dd: torch.Tensor,        # (batch, M*N) complex
            channel: ChannelView,      # populated according to channel_form
            no: torch.Tensor,          # (batch,) real, per-frame noise variance
            ) -> torch.Tensor:
        # Return real LLRs of shape (batch, M*N * num_bits_per_symbol) in
        # Sionna sign convention (LLR > 0 means bit 1; natural log / nats).
        ...
```

The evaluation harness:

1. Discovers your `Equalizer` subclass by scanning `draft.py` (there must be **exactly one** subclass — define helper classes outside the `Equalizer` hierarchy or import them).
2. Instantiates it with `cls(grid, num_bits_per_symbol=4, precision="single", device="cuda:0")`. Accept `**kwargs` for forward compatibility.
3. Reads `channel_form` and `top_k` off the class **once**, then builds a single `ChannelView` per frame.

### Choosing `channel_form`

* `channel_form = "sparse"`: `channel.sparse.h_vals` (complex `(batch, M*N, top_k)`) and `channel.sparse.col_idx` (long `(batch, M*N, top_k)`) are populated. Convention: `y[r] ≈ sum_e h_vals[r, e] * x[col_idx[r, e]] + w[r]`. Best for message-passing / iterative detectors whose cost is dominated by the number of non-zero taps.
* `channel_form = "dense"`: `channel.dense.H` (complex `(batch, M*N, M*N)`) is populated. Same information as the sparse view — built by `scatter_add_` on the top-`k` entries — but laid out for matrix algebra. Best for LMMSE-style or matrix-factorisation detectors.

**Information content is identical** between the two forms for the same `top_k`. There is no `"cir"` form: handing the detector raw path parameters would give it strictly more information than the EP baseline saw and make the NVE comparison unfair.

### Choosing `top_k` (the sparsity knob)

`top_k` controls how many channel taps per output row the harness keeps when building the channel view. It is a real **design parameter**:

* **Higher `top_k`** keeps more channel taps → typically lower BLER, but **larger memory and more arithmetic per `llr` call**, so per-call runtime rises (often linearly or quadratically depending on the algorithm).
* **Lower `top_k`** drops the smallest-magnitude taps → cheaper detector, but the discarded taps may matter, so BLER can degrade.

Pick `top_k` accordingly: smaller for fast, lighter-weight detectors that target the BLER floor, larger for accuracy-first detectors aiming to match or beat EP.

Allowed range: `top_k in [1, M*N]` (i.e. up to 4096). If you don't set the attribute (or set it to `None`), the harness uses `TOP_K_DEFAULT = 256`, identical to the EP baseline.

### LLR convention

The transmit pipeline encodes bits with the 5G LDPC encoder and maps them with `sionna.phy.mapping.Mapper("qam", num_bits_per_symbol=4)`. The downstream LDPC decoder therefore expects LLRs in **Sionna's convention**, on three independent axes — sign, units, and the QAM bit-to-symbol labelling. Any single one of these wrong produces BLER ≈ 1.0 at every SNR and an NVE around 143 (see the sentinel-value warning below).

* **Sign**: `LLR(b) = log( P(b=1 | y) / P(b=0 | y) )`. **Positive** LLR means bit 1 is more likely; negative means bit 0.
* **Units**: natural log (nats), not bits.
* **Output shape**: `(batch, M*N * num_bits_per_symbol)`. Bit `k` of QAM symbol at flat DD position `r` (delay-major, `r = l*N + k_dop`) sits at index `r*num_bits_per_symbol + k`, identical to what `sionna.phy.mapping.Demapper` produces.
* **QAM bit labels**: the bit-label vector `[b_0, ..., b_{K-1}]` (`K = num_bits_per_symbol`, `b_0` is the MSB) maps to the constellation point at index `q = (b_0 << (K-1)) | ... | (b_{K-1} << 0)`. The actual point at index `q` is the one Sionna's `Constellation("qam", K).points[q]` returns — *do not* fabricate your own QAM points or your own bit-to-axis mapping; Sionna's QAM has a specific interleaved (I-sign, Q-sign, I-magnitude, Q-magnitude, ...) layout that is not reproducible by partitioning bits into independent I and Q halves.

Two equally valid ways to satisfy the above are supported. Pick whichever fits your detector:

**Path A — Equalise to symbol estimates, then demap.** Natural for LMMSE, matched-filter, AMP / VAMP, and other symbol-domain detectors. Produce a per-symbol estimate `x_hat: (batch, M*N)` complex and a per-symbol effective noise variance `var: (batch, M*N)` real, then hand them to a Sionna demapper:

```python
from otfs import make_demapper

class MyDetector(Equalizer):
    channel_form = "dense"
    def __init__(self, grid, *, num_bits_per_symbol=4, precision="single", device="cuda:0", **kw):
        self.demapper = make_demapper(num_bits_per_symbol, mode="maxlog",
                                      precision=precision, device=device)
    def llr(self, y_dd, channel, no):
        x_hat, var = ...  # whatever equalisation you want
        return self.demapper(x_hat, var)
```

**Path B — Per-bit detector that computes LLRs directly over constellation points.** Natural for EP, message-passing, BP, exact ML, and most iterative detectors. Use the helper to get the canonical points and label table; never hand-roll them:

```python
import torch
from otfs import qam_constellation_with_labels

class MyDetector(Equalizer):
    channel_form = "sparse"
    def __init__(self, grid, *, num_bits_per_symbol=4, precision="single", device="cuda:0", **kw):
        self.K = num_bits_per_symbol
        self.points, labels = qam_constellation_with_labels(
            num_bits_per_symbol, precision=precision, device=device
        )                                          # points: (Q,) complex, labels: (Q, K) bool
        self.bit_is_one = labels                   # (Q, K)
        self.bit_is_zero = ~labels                 # (Q, K)
    def llr(self, y_dd, channel, no):
        # Compute log p(x = a_q | y) for each symbol position r and point q.
        logp = ...                                                     # (batch, M*N, Q)
        llrs = []
        for k in range(self.K):
            m1 = torch.logsumexp(logp[..., self.bit_is_one[:, k]],  dim=-1)
            m0 = torch.logsumexp(logp[..., self.bit_is_zero[:, k]], dim=-1)
            llrs.append(m1 - m0)
        return torch.stack(llrs, dim=-1).reshape(y_dd.shape[0], -1)
```

> **Sentinel-value warning.** If your evaluation returns `NVE ≈ 85.12` regardless of what you change, your detector is achieving BLER ≈ 1.0 at every SNR (`mean(1 / baseline_bler)` on the evaluation grid is exactly `85.12`). This is far below random and almost always means the LDPC decoder is receiving LLRs that don't line up with the TX-side QAM labels — usually a hand-rolled constellation, an LSB/MSB confusion, or a sign flip. Verify against Sionna's `Demapper` / `Constellation` (or use the helpers above) before iterating on the algorithm itself.

### Technical constraints

1. **PyTorch only.** All computation inside `llr` must use `torch` operations. Do **not** convert tensors to NumPy or SciPy.
2. **No brittle hardcoded scenario values in the detector.** Do not read or import `link_config.py` while developing: it is evaluation-harness state and is injected only when `evaluate_otfs_detector` runs. Use `grid.M`, `grid.N`, the `num_bits_per_symbol` constructor argument, tensor shapes, and the passed noise variance `no`. The table above is for design context and sanity checks.
3. **Do not modify `baseline_bler.pkl`.** It is the reference BLER curve the harness divides into.
4. **Do not import or re-implement EP/MP/LMMSE from any external source.** You may inspect `otfs/` for channel/IO utilities, but the detector you submit must be your own work.
5. **Stateless interface.** `llr(y_dd, channel, no)` may rely on module-level / instance state initialised in `__init__`, but must be a pure function of its inputs once initialised.
6. **`torch.compile` compatibility.** The harness wraps your `llr` once with `torch.compile(mode="default", dynamic=False)` and uses that *same* compiled callable for **both** the NVE scoring and the runtime measurement. Write `llr` to be compile-friendly so it captures as a single graph and runs fast: avoid graph breaks and host–device syncs inside `llr` — no `.item()`, `.cpu()`, `.numpy()`, `float(...)`/`int(...)` on tensors, Python-side `print`, or `.data_ptr()`-keyed caching; avoid data-dependent Python control flow (loop bounds / `if` conditions that depend on tensor *values* rather than static shapes); and keep tensor shapes static across calls (input shapes are fixed by the scenario). Fixed-count Python `for` loops (e.g. a fixed number of detector iterations) and shape-based branching are fine. If compilation fails, the harness resets Dynamo and falls back to **eager for both paths**, so your detector is still scored — but the measured `detector.llr` runtime (the complexity objective minimised alongside NVE) will typically be **worse** than a compile-clean implementation. The evaluation output now reports compile status back to you: on success it appends a `[compile]` line stating whether `llr` captured as a single graph, hit graph breaks, or fell back to eager, plus a `[compile warnings]` summary of any PyTorch warnings emitted during compilation (e.g. *"does not support code generation for complex operators"*, which means those ops ran eagerly even though the graph compiled). Use these lines to drive your implementation toward a single fused graph; compile-friendliness is still a first-class design goal.

### How to spend your evaluation budget

Hyperparameters are tuned automatically **after** your run by an Optuna post-tuner that sweeps every `HP.get(...)` knob you declare. Don't replicate that work inside your own loop: between successive `evaluate_otfs_detector` calls, prefer **one structural change** (a different algorithmic component, a new iteration, a different variance estimate, a different decomposition) over tweaking a single numeric knob. Set sensible defaults plus reasonable ranges in `HP.get(...)` and let the post-tuner explore the box. Use your turns to *change the algorithm*, not to fine-tune scalars.

## Workspace files

* `otfs/` — pared-down toolkit (channel + IO + grid + equaliser base class + `demap` helpers). Available from turn 1; read it to understand the channel and the `ChannelView` API.
* `otfs/demap.py` — Sionna-compatible QAM helpers: `make_demapper(...)` (Path A) and `qam_constellation_with_labels(...)` (Path B). Use these instead of fabricating constellation points or bit labels.
* `link_config.py` — frozen scenario constants used by the evaluation harness; **not available for normal inspection** and injected only while `evaluate_otfs_detector` is running. Use the scenario table above plus runtime `grid`/argument values instead of trying to read it.
* `baseline_bler.pkl` — `(snr_db, bler)` tuple for the EP baseline (injected at evaluation time). Used by the harness to compute NVE; you should never need to touch it.
* `eval.py` — evaluation harness (injected at evaluation time).

## Baseline disclosure

The reference BLER curve in `baseline_bler.pkl` was produced by the analytical-IO OTFS benchmark in `tasks/otfs_detector/benchmark/` using the **sparse EP detector** of Li et al., 2021, with 15 EP iterations, damping 0.5, and `top_k = 256`. The same scenario constants as above were used; the run used `batch_size = 8`, `max_mc_iter = 200`, `num_target_block_errors = 50`, and `early_stop = True`. Your evaluation uses the same Monte-Carlo settings.

You are not required to match EP's `top_k`; you are free to pick a different sparsity and either match the curve more cheaply or beat it with more taps.

## Evaluation

Call the `evaluate_otfs_detector` tool to test your detector. It returns:

\[
\text{NVE} = \frac{1}{N_{\rm snr}} \sum_{i=1}^{N_{\rm snr}} \frac{\text{BLER}_{\text{cand}}(\text{SNR}_i)}{\text{BLER}_{\text{EP}}(\text{SNR}_i)}
\]

over the SNR points where the baseline BLER is non-zero. NVE is strictly positive. **The goal is NVE as close to 1 as possible, or lower.** Below 1 means you beat EP at the chosen SNR points; above 1 means you are worse.

The evaluation output is a single line of the form

```
SUCCESS, <nve>, <avg_llr_runtime_ms_per_batch>
```

or, on error,

```
FAILURE,
<error details>
```

The `<avg_llr_runtime_ms_per_batch>` is the average **total** wall-clock time of `detector.llr` only for one batch of `BATCH_SIZE` frames, in **milliseconds** (warm-up calls excluded, 10 timed calls, CUDA-synchronised). It is the complexity objective the framework minimises alongside NVE.

## Hints

* Start by **inspecting `otfs/`** in your workspace. The IO model and channel view are documented in their docstrings; in particular `SparseDDChannel.apply` shows you how `(h_vals, col_idx)` are used to compute `y = H x`.
* Use the supplied helpers (`otfs.make_demapper` for Path A, `otfs.qam_constellation_with_labels` for Path B) rather than fabricating a QAM constellation or label table — see the "LLR convention" section above. Sionna's QAM bit-to-symbol mapping is not what naïve `(I-bits, Q-bits)` constructions produce.
* The cheapest meaningful baseline is a single-shot **DD-domain LMMSE** built from the dense view: set `channel_form = "dense"`, solve `x_hat = (H^H H + N0 I)^{-1} H^H y` (via `torch.linalg.solve`, never an explicit inverse), use the per-symbol post-equalisation variance, and hand `(x_hat, var)` to `otfs.make_demapper`. It will not beat EP on BLER but it is the right smoke test for the harness and your LLR plumbing.
* Iterative detectors (variants of message passing, expectation propagation, gradient-based MAP, AMP/VAMP, ...) usually shine when `top_k` is moderate (the per-iteration cost is `O(top_k)`-bound). If you go that route, expose the number of iterations, damping, and any scaling factors via `HP.get(...)` so the framework's Optuna post-tuner can tune them after your agent run.
