# Pilotless OFDM Receiver Implementation Task

## Objective

Implement a **pilotless OFDM receiver** that, from the received resource grid and the noise variance, produces soft bit information (log-likelihood ratios) for the channel decoder. Your goal is to **minimize the Normalized Validation Error (NVE)** — the average ratio between the BLER achieved by your receiver and a reference BLER measured on the same link, across a range of SNR points. **The secondary goal is to keep complexity low**. **Do not trade NVE for speed.**

## Context

In the physical layer of wireless communication systems, the receiver processes the post-DFT resource grid to recover the transmitted data. Conventional receivers rely on **pilot symbols (DMRS)** placed inside the resource grid to estimate the channel before equalization and demapping.

In this task **the resource grid carries no pilots at all**: every resource element is a data-carrying symbol drawn from a pre-trained, non-uniform constellation. The receiver must therefore recover per-bit LLRs directly from the received resource grid and the noise variance, without any pilot-based channel estimate.

The link is **SISO**: a single transmit antenna and a single receive antenna. MIMO detection is not required.

The constellation has already been designed and is provided to you as a workspace file. **You must not retrain or modify it.** Your only job is the receiver.

**The constellation is the backbone of the receiver.** Because there are no pilots, the *only* structure the receiver can rely on to disentangle the channel from the data is the **geometry of the learned alphabet**. Any receiver that ignores it will underperform. Your receiver **must explicitly exploit the constellation points loaded from `constellation_points.pkl`**.

### Constellation description

The constellation is an **irregular `2**NUM_BITS_PER_SYMBOL`-point non-uniform QAM** stored in `constellation_points.pkl`. Its geometry is nothing like a square QAM grid:

- Points are scattered over an approximately **disc-shaped region** of the complex plane. The density is non-uniform: some neighbours are almost on top of each other while others are well separated.
- There is **no regular row/column structure**. Do not assume the alphabet factorises into independent I/Q PAMs.
- The **bit labelling is not standard Gray code**. Most (but not all) nearest-neighbour pairs differ by a single bit; some differ by more. A demapper that relies on Gray-coded QAM bit-slicing (e.g. per-bit thresholding on I and Q) will lose information.

Any demapper you build must operate on the **actual point cloud loaded from `constellation_points.pkl`**. Closed-form QAM demappers do not apply.

#### Constellation bit labelling

`constellation_points.pkl` is an array of length `2**NUM_BITS_PER_SYMBOL`. The bit label of the point at index `n` is the **MSB-first** binary expansion of `n`:

```python
# bit k (0 ≤ k < NUM_BITS_PER_SYMBOL) of constellation point n
bit_k = (n >> (NUM_BITS_PER_SYMBOL - 1 - k)) & 1
```

So bit 0 is the most significant bit of `n`, bit `NUM_BITS_PER_SYMBOL - 1` is the least significant. Equivalently: `np.binary_repr(n, NUM_BITS_PER_SYMBOL)` gives the bit label as a string, with the leftmost character being bit 0. This is the labelling Sionna's `Mapper`/`Demapper` use internally and the one the LDPC decoder expects — using the reverse ordering for any of the `NUM_BITS_PER_SYMBOL` bit positions is equivalent to permuting `NUM_BITS_PER_SYMBOL` of the LLR channels and will destroy decoding performance.

### Channel assumptions

The propagation channel is **not** flat. You must assume it is **selective in both dimensions of the resource grid**:

- **Frequency-selective**: the channel coefficient varies across subcarriers within the same OFDM symbol. Different subcarriers see different complex gains, with neighbouring subcarriers correlated.
- **Time-selective**: the channel coefficient varies across OFDM symbols within the same slot. Neighbouring OFDM symbols are correlated, but the channel is **not** constant over the slot.

Approaches that approximate the channel by a single complex coefficient — over the whole slot, per subcarrier, or per OFDM symbol — are not viable here, even as a "simple baseline". Any usable receiver needs a channel estimate that varies across the resource grid.

## Requirements

### Function Signature

Your solution must define a module-level callable with the following signature:

```python
def receiver(y, no):
    """
    Pilotless OFDM receiver.

    Args:
        y:  Received resource grid — a **PyTorch** complex64 tensor of shape
            [batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size]
            For this task num_rx = 1 and num_rx_ant = 1.
        no: Noise variance per batch element — a **PyTorch** float32 tensor of shape
            [batch_size]

    Returns:
        llr: Log-likelihood ratios on the transmitted coded bits — a **PyTorch** float32 tensor of shape
             [batch_size, num_rx, num_streams_per_rx, num_ofdm_symbols,
              fft_size, num_bits_per_symbol]
              For this task num_rx = 1 and num_streams_per_rx = 1.
    """
```

**IMPORTANT — PyTorch only.** Your receiver runs inside a Sionna simulation pipeline, which is built on **PyTorch**. All inputs (`y`, `no`) are PyTorch tensors and the output **must** be a PyTorch tensor. Use `torch` operations throughout — do **not** convert to numpy. Using numpy or scipy may cause errors or severe performance degradation.

The evaluation script wraps a `ResourceGridDemapper` and an LDPC decoder around your `receiver`, so the only tensor you need to produce is the per-resource-element LLR block above.

#### LLR convention

The LDPC decoder downstream of your `receiver` expects LLRs in **Sionna's convention**:

- **Sign**: `LLR(b) = log( P(b=1 | y) / P(b=0 | y) )`. A **positive** LLR means bit 1 is more likely; a negative LLR means bit 0 is more likely. Flipping this sign silently turns a working receiver into one near random.
- **Units**: natural log (nats), not bits.

### Configuration

**IMPORTANT: `link_config.py` is NOT available in your workspace. Do NOT try to read or find this file.**

The file `link_config.py` will be **automatically injected when you call the evaluation tool**. You must write your solution based on the documentation below, trusting that the imports will work during evaluation.

Simply write your code with the documented imports (e.g., `from link_config import NUM_BITS_PER_SYMBOL, FFT_SIZE, NUM_OFDM_SYMBOLS`) and they will resolve correctly when the evaluation tool runs.

**Do NOT:**
- Try to read `link_config.py` from the workspace
- Create your own `link_config.py`
- Wait for `link_config.py` to appear

**DO:**
- Call the evaluation tool to test your implementation
- The evaluation tool will provide `link_config.py` automatically

`link_config.py` exposes the following constants (for reference only — do not try to access this file):

**System Parameters:**
- `NUM_BITS_PER_SYMBOL`: bits per modulation symbol (defines the constellation size `2**NUM_BITS_PER_SYMBOL`)
- `NUM_OFDM_SYMBOLS`: OFDM symbols per slot
- `FFT_SIZE`: FFT size for OFDM
- `SUBCARRIER_SPACING`: subcarrier spacing in Hz
- `CYCLIC_PREFIX_LENGTH`: cyclic prefix length in samples
- `NUM_GUARD_CARRIERS`: guard carriers `[left, right]`
- `DC_NULL`: whether the DC subcarrier is nulled
- `CODERATE`: LDPC code rate

**Channel Parameters:**
- `CARRIER_FREQUENCY`: carrier frequency in Hz

### Provided Workspace Files

The following file is available in your workspace:

- `constellation_points.pkl`: a `numpy` complex64 array of shape `[2**NUM_BITS_PER_SYMBOL]` containing the constellation points used by the transmitter. You may load and inspect this file — it is the only way to know where the transmitter places its symbols — but **you must not modify it**.

### Technical Constraints

1. **PyTorch only.** All computation inside `receiver` must use `torch` operations. Do **not** convert inputs to numpy or use scipy — the pipeline requires PyTorch tensors throughout.
2. **No Hardcoded Values**: Import all system parameters from `link_config.py`. Your receiver must work for any valid configuration.
3. **Do not touch the constellation.** `constellation_points.pkl` is part of the system model. You may read it, but you must not alter it or replace it. Convert it to a `torch.Tensor` at import time for use in your receiver.
4. **Stateless interface.** `receiver(y, no)` may hold module-level state (e.g. loaded weights, cached tensors initialised at import time), but it must be a pure function of its inputs once initialised.
5. **`torch.compile` compatibility.** Evaluation always wraps `receiver` with `torch.compile` (on GPU it uses `mode="reduce-overhead"`, i.e. CUDA graphs). Load `constellation_points.pkl` and any other files at **module level** (before tracing), not inside `receiver`. Use `torch` ops in the hot path; avoid NumPy/scipy inside `receiver` and avoid Python loops over individual tensor elements. **Avoid per-call host→device copies** (e.g. calling `.to(device=...)` on module-level constants every call): each one breaks CUDA-graph capture and inflates the measured per-call runtime (the complexity metric). Make constants device-resident once and reuse them.

## Design Freedom

Your receiver has full design freedom on the *method*. What is **not** negotiable is the *information* you must use — the constellation geometry.

Concretely, your receiver must:

1. Take the received resource grid `y` and noise variance `no` as input.
2. **Exploit the geometry of the constellation loaded from `constellation_points.pkl`.** With no pilots, the only prior that ties the received samples to the transmitted bits is the shape of the learned alphabet. Your receiver must *use* those points rather than treat the modulation as unknown or as a generic QAM. A receiver that does not use the constellation cannot win.
3. **Treat the channel as selective in both time and frequency** (see *Channel assumptions* above). Recover an unknown channel coefficient *per resource element*, not a single scalar per slot, per subcarrier, or per OFDM symbol. A receiver that collapses the channel to a single complex gain over the grid cannot win.
4. Output LLRs with the correct shape and dtype.
5. Not modify or replace `constellation_points.pkl`.

## Starter Template

A minimal skeleton showing how to access `link_config.py` and load the constellation file. Add your own logic inside `receiver`.

```python
import pickle
import torch

from link_config import NUM_BITS_PER_SYMBOL  # plus any other constants you need

with open("constellation_points.pkl", "rb") as f:
    points = pickle.load(f)  # complex array of length 2**NUM_BITS_PER_SYMBOL

CONSTELLATION = torch.as_tensor(points, dtype=torch.complex64)

# ... initialise your receiver state here (precomputed bit masks, smoothing
# kernels, etc.) at import time ...

def receiver(y, no):
    # y and no are PyTorch tensors — use torch operations only.
    # TODO: compute LLRs from (y, no) and return them with the documented shape.
    raise NotImplementedError
```

Your submitted file must expose `receiver` at module scope. That is the only symbol the evaluation tool looks up.

## Hints

Before writing any receiver, **load the constellation and look at it**. Do not assume it is a 64-QAM (or any regular grid).

```python
import pickle
import numpy as np

with open("constellation_points.pkl", "rb") as f:
    pts = np.asarray(pickle.load(f), dtype=np.complex64)

print(pts.shape, pts.dtype)
print("abs min/mean/max:", np.abs(pts).min(), np.abs(pts).mean(), np.abs(pts).max())
```

## Evaluation

Use the provided evaluation tool to test your receiver. The harness compiles `receiver` with `torch.compile` before simulation. The tool returns the **Normalized Validation Error (NVE)**: the mean ratio of your receiver's BLER to a reference BLER, computed across a range of SNR points.

\[
\text{NVE} = \frac{1}{N} \sum_{i=1}^{N} \frac{\text{BLER}_{\text{agent}}(\text{SNR}_i)}{\text{BLER}_{\text{reference}}(\text{SNR}_i)}
\]

NVE is strictly positive. **The primary goal is to bring NVE as close to 1 as possible, or lower.** An NVE near 1 means your receiver matches the reference; an NVE well below 1 means it beats it; an NVE well above 1 means it is significantly worse. **The secondary goal is lower per-call runtime**.

On success, the tool returns **one parsed line** followed by optional **raw stderr** from `torch.compile`:

```
SUCCESS, <nve>, <avg_receiver_runtime_seconds>
<torch.compile stderr: warnings and diagnostic messages>
```

- **`<nve>`** — normalized validation error (lower is better; see above).
- **`<avg_receiver_runtime_seconds>`** — average per-call wall time of the compiled `receiver` after warm-up (**complexity**; lower is better).
- **Trailing stderr** — compile-time feedback. Clear `torch.compile` warnings and errors in the trailing stderr that come from your receiver and hurt compilation or runtime (see Technical Constraint 5). Ignore stderr that is not caused by your code.

On error:

```
FAILURE,
<error details>
```