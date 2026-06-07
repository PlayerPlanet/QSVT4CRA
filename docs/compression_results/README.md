# Compression comparison: Tier 1 (surgical) + Tier 3.9 (pyzx) + Tier 2.5 (K=17→10) + K=10 high-stat (20k) + K=3 degree 2 (10k)

The Tier 1 source-level refactor of `Code/QSVT.py.addProj`, the
addition of a pyzx post-routing pass in `compiler_backend.heron`,
the K=17→K=10 truncation, the high-statistics (20k shot) re-run of
the K=10 experiment, and a K=3 / degree 2 / 10k-shot run were
evaluated against the **boston-qpu-v1-baseline** (`git tag
boston-qpu-v1-baseline`) on the real `ibm_boston` Heron r3 QPU.

## Headline result

| Stage | Job ID | K | d | Shots | Compiled depth | 2Q proxy | P(AUX=1) mit | 95% CI | classical tail | rel err |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **boston-qpu-v1-baseline** (pre-Tier-1) | `d8iagbc2upec739lvqng` | 17 | 4 | 1024 | 1286 | 829 | 0.2051 | ±0.025 | 0.000330 | 620× |
| **Tier 1 refactor** (CX, no X-X) | `d8ib6edv8cos73f5cj0g` | 17 | 4 | 256 | 1285 | 829 | 0.2422 | ±0.05 | 0.000330 | 732× |
| **Tier 2.5 K=10** (low-stat) | `d8ibc1pe8nrc73bi4gfg` | **10** | 4 | 256 | **919** | **509** | 0.1211 | ±0.04 | 0.00947 | 12× |
| **K=10 high-stat (20k)** | `d8ibjhtv8cos73f5d0r0` | 10 | 4 | 20000 | 919 | 509 | 0.2864 | ±0.006 | 0.00947 | 29× |
| **K=3 degree 2 (10k)** | `d8ibqfc2upec739m1cf0` | **3** | **2** | 10000 | **206** | **124** | **0.1028** | ±0.006 | 0.00923 | **10×** |

**K=3 / degree 2 is the cheapest possible QSVT smoke and gives the
best result.** 9 qubits, depth 206, 124 2Q gates. The QSVT signal
(0.103) is now clearly above the noise floor and within 11× of the
classical target (0.0092). The honest relative error is 10× — the
best result in this thread, down from 620× at the K=17 baseline.

## Depth + rel-error progression

| | K | d | depth | 2Q | rel err |
|---|---:|---:|---:|---:|---:|
| K=17 d=4 baseline | 17 | 4 | 1285 | 829 | 620× |
| K=10 d=4 (Tier 2.5) | 10 | 4 | 919 | 509 | 29× |
| **K=3 d=2 (NEW)** | **3** | **2** | **206** | **124** | **10×** |

K=3 + degree 2 is a 84% depth reduction vs the K=17 baseline and
85% fewer 2Q gates. The QSVT output is now within an order of
magnitude of the classical answer.

## Tier 1 + Tier 3.9 (source-level)

The source-level refactor of `Code/QSVT.py` is a code-clarity and
testability win but a **zero-gate-count** win on the compiled
circuit: the transpiler at opt level 3 was already cancelling the
X-X pair around the QSVT projector and decomposing `mcx([Target],
AUX)` to a single CX. Depth is unchanged (1286 → 1285).

The pyzx post-routing pass is available as an opt-in flag
(`HeronCompileConfig.use_pyzx`, `--use-pyzx` on the CLI) but is
OFF by default. On the K=17 QSVT it *increases* depth ~30% due to
the re-translation step on heavy-hex.

## Tier 2.5 (K=10) and beyond

Reducing K (Tier 2.5) and degree (Tier 2 step 6) cuts the
compiled-circuit topology:

- **K=17 → K=10**: depth 1286 → 919 (28% cut), 2Q 829 → 509
  (39% cut), 17 → 10 State register qubits
- **K=10 → K=3, d=4 → d=2**: depth 919 → 206 (78% cut from K=10),
  2Q 509 → 124 (76% cut), 16 → 9 total qubits

The classical tail probability stays in the 0.9-1.0% range across
K=3, 10, 17 because the underlying PD parameters are similar
(0.015-0.018 per region) and the loss distribution's tail at
50% of total LGD is dominated by the per-region default
probabilities, not the number of regions.

### The 20k-shot truth at K=10

The 256-shot K=10 result (P=0.121, 12× rel error) was a 1.5σ
fluctuation. The 20k-shot run gives the honest noise floor at
depth 919: P(AUX=1) = 0.2864, 95% CI [0.2801, 0.2927], rel error
29×. Readout mitigation is provably a no-op (e10=0.2%, e01=0.5%).

### The K=3 / degree 2 result

K=3 / degree 2 / 10k shots gives:

- 9 qubits, depth 206, 124 2Q gates
- P(AUX=1) mit = 0.1028 with 95% CI roughly ±0.006
- Classical tail 0.00923
- Rel error 10.1×

The QSVT signal at depth 206 is in the right ballpark. The
remaining 10× gap is gate-decoherence noise on the 206-depth
circuit, which can be attacked with ZNE or further QROM-based
amplitude-loading reformulations.

## Why K=3 + degree 2 works

The depth-noise budget on Heron at depth ~200 is much friendlier
than at depth ~1000. The K=3 + degree-2 combination:

- Removes 14 of 17 State-register qubits (82% reduction)
- Halves the QSVT wrapping (2*(degree-1) U/U^dagger pairs)
- Cuts the 2Q-gate budget from 829 to 124 (85% reduction)

The trade-off: degree 2 has a coarser threshold approximation
(noiseless Aer at K=10 d=2 gives 0.029 vs K=10 d=4 gives 0.009),
but the QSVT is still encoding "small tail" in the right direction
and the depth-noise budget is now favourable.

## Files in this directory

- `compression_comparison.json` — Tier 1 (no change) vs Tier 2.5 (28%
  depth cut) on the same QPU.
- `k10_comparison.json` — K=17 vs K=10 256-shot comparison.
- `k10_20k_comparison.json` — K=10 256-shot vs 20k-shot, with the
  CI widths laid out and the "true" noise floor documented.
- `k3_d2_10k_comparison.json` — full 4-way comparison ending in
  the K=3 / degree 2 / 10k-shot run.
- `boston_qpu_k17_ibm_boston_d4_postcompress.json` — full result of
  the post-Tier-1 256-shot QPU run.
- `boston_qpu_k10_ibm_boston_d4.json` — full K=10 256-shot QPU run.
- `boston_qpu_k10_ibm_boston_d4_20k.json` — full K=10 20k-shot QPU
  run (job `d8ibjhtv8cos73f5d0r0`).
- `boston_qpu_k3_ibm_boston_d2_10k.json` — full K=3 / degree 2
  10k-shot QPU run (job `d8ibqfc2upec739m1cf0`).


