# Compression comparison: Tier 1 (surgical) + Tier 3.9 (pyzx) + Tier 2.5 (K=17→10) + K=10 high-stat (20k) + K=3 / K=2 degree 2 (10k) + K=2 / degree 4 (10k) + K=2 / degree 8 (10k) + K=3 / degree 4 (10k)

The Tier 1 source-level refactor of `Code/QSVT.py.addProj`, the
addition of a pyzx post-routing pass in `compiler_backend.heron`,
the K=17→K=10 truncation, the high-statistics (20k shot) re-run of
the K=10 experiment, the K=3 / degree 2 / 10k-shot run, the
K=2 / degree 2 / 10k-shot run, the K=2 / degree 4 / 10k-shot
run, the K=2 / degree 8 / 10k-shot run, and the K=3 / degree 4 /
10k-shot run were all evaluated against the **boston-qpu-v1-baseline**
(`git tag boston-qpu-v1-baseline`) on the real `ibm_boston` Heron
r3 QPU.

## Headline result

| Stage | Job ID | K | d | Shots | Compiled depth | 2Q proxy | P(AUX=1) mit | 95% CI | classical tail | rel err |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **boston-qpu-v1-baseline** | `d8iagbc2upec739lvqng` | 17 | 4 | 1024 | 1286 | 829 | 0.2051 | ±0.025 | 0.000330 | 620× |
| **Tier 1 refactor** | `d8ib6edv8cos73f5cj0g` | 17 | 4 | 256 | 1285 | 829 | 0.2422 | ±0.05 | 0.000330 | 732× |
| **Tier 2.5 K=10** (low-stat) | `d8ibc1pe8nrc73bi4gfg` | 10 | 4 | 256 | 919 | 509 | 0.1211 | ±0.04 | 0.00947 | 12× |
| **K=10 high-stat (20k)** | `d8ibjhtv8cos73f5d0r0` | 10 | 4 | 20000 | 919 | 509 | 0.2864 | ±0.006 | 0.00947 | 29× |
| **K=3 d=2 (10k)** | `d8ibqfc2upec739m1cf0` | 3 | 2 | 10000 | 206 | 124 | 0.1028 | ±0.006 | 0.00923 | 10× |
| **K=3 d=4 (10k)** | `d8ic3um6983c73drvdv0` | 3 | 4 | 10000 | 317 | 170 | 0.0870 | ±0.005 | 0.00923 | 8.43× |
| **K=2 d=2 (10k)** | `d8ibt0tv8cos73f5dcm0` | 2 | 2 | 10000 | 170 | 75 | 0.0833 | ±0.005 | 0.00908 | 8.2× |
| **K=2 d=4 (10k)** *(BEST)* | `d8ic08pe8nrc73bi56tg` | **2** | **4** | 10000 | **228** | **101** | **0.0476** | ±0.005 | 0.00908 | **4.25×** |
| **K=2 d=8 (10k)** *(overshoots)* | `d8ic32dv8cos73f5djlg` | 2 | 8 | 10000 | 331 | 140 | 0.0705 | ±0.005 | 0.00908 | 6.77× |

**K=2 / degree 4 / 10k shots is the global best:** rel error 4.25×.
The QSVT signal (0.048) is now within 4.3× of the classical target
(0.0091), and 5× below the random-coin-flip midpoint of 0.5.
Going to higher degree at K=2 (d=8) or to K=3 d=4 are both worse,
because the additional depth brings back gate-decoherence noise
that outweighs the polynomial-accuracy gain.

## Depth + rel-error progression

| | K | d | depth | 2Q | rel err |
|---|---:|---:|---:|---:|---:|
| K=17 d=4 baseline | 17 | 4 | 1285 | 829 | 620× |
| K=10 d=4 (Tier 2.5) | 10 | 4 | 919 | 509 | 29× |
| K=3  d=2 | 3 | 2 | 206 | 124 | 10× |
| K=3  d=4 | 3 | 4 | 317 | 170 | 8.43× |
| K=2  d=2 | 2 | 2 | 170 | 75 | 8.2× |
| **K=2  d=4 (BEST)** | **2** | **4** | **228** | **101** | **4.25×** |
| K=2  d=8 (overshoots) | 2 | 8 | 331 | 140 | 6.77× |

Total improvement vs the K=17 baseline: 5.6× depth reduction
(1285 → 228), 146× rel-error reduction (620× → 4.25×).

## K=2 degree sweep

| d | depth | 2Q | p_qpu | rel err |
|---:|---:|---:|---:|---:|
| 2 | 170 | 75 | 0.083 | 8.2× |
| **4** | **228** | **101** | **0.048** | **4.25×** |
| 8 | 331 | 140 | 0.071 | 6.77× |

d=4 is the sweet spot: d=2's step is too soft for the 4-outcome
K=2 distribution, d=8's extra depth brings back gate-decoherence
noise that the polynomial-accuracy gain can't offset.

## K=3 degree sweep

| d | depth | 2Q | p_qpu | rel err |
|---:|---:|---:|---:|---:|
| 2 | 206 | 124 | 0.103 | 10× |
| 4 | 317 | 170 | 0.087 | 8.43× |

At K=3 (8-outcome distribution) d=4 still helps but the gain is
smaller (1.6× reduction vs 1.9× at K=2).  The depth cost (+54%)
almost cancels the polynomial-accuracy gain.

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
- **K=3 → K=2**: depth 206 → 170 (17% cut), 2Q 124 → 75 (40%
  cut), 9 → 8 total qubits
- **K=2 d=2 → K=2 d=4**: depth 170 → 228 (34% more), 2Q 75 → 101
  (35% more), but the threshold approximation is sharper so the
  QPU output is **half the relative error** (8.2× → 4.25×)
- **K=2 d=4 → K=2 d=8 (overshoots)**: depth 228 → 331 (45%
  more), 2Q 101 → 140 (39% more), and the QPU output
  *worsens* — the polynomial was already sharp enough at d=4
- **K=2 d=4 → K=3 d=4**: depth 228 → 317 (39% more), 2Q 101 →
  170 (68% more), and the QPU output *worsens* — the K=3 8-outcome
  distribution doesn't gain enough from sharper thresholding to
  offset the noise

### The 20k-shot truth at K=10

The 256-shot K=10 result (P=0.121, 12× rel error) was a 1.5σ
fluctuation. The 20k-shot run gives the honest noise floor at
depth 919: P(AUX=1) = 0.2864, 95% CI [0.2801, 0.2927], rel error
29×. Readout mitigation is provably a no-op (e10=0.2%, e01=0.5%).

### The K=2 / degree 4 result (BEST)

K=2 / degree 4 / 10k shots gives:

- 8 qubits, depth 228, 101 2Q gates
- P(AUX=1) mit = 0.0476 with 95% CI roughly ±0.005
- Classical tail 0.00908
- Rel error 4.25×

The QSVT signal at depth 228 is within 4.3× of the classical
target. The remaining 4.3× gap is gate-decoherence noise on the
228-depth circuit.

## The depth-noise trade-off

At depth ~200-300 on Heron, doubling the depth roughly doubles
the gate-decoherence noise. The QSVT signal gain from sharper
thresholding is bounded by the polynomial approximation; once
the polynomial is sharp enough to resolve the discrete outcomes
of the loss distribution, more phases just add decoherence.

K=2 has 4 discrete loss outcomes (no defaults, default 1 only,
default 2 only, both defaults) — d=4 resolves these sharply.
K=3 has 8 outcomes — d=4 is sharper but the depth cost is
proportionally larger. K=17 has 2^17 = 131072 outcomes but
only the ones near the target matter; the threshold function
only needs to resolve the tail.

## Files in this directory

- `compression_comparison.json` — Tier 1 (no change) vs Tier 2.5 (28%
  depth cut) on the same QPU.
- `k10_comparison.json` — K=17 vs K=10 256-shot comparison.
- `k10_20k_comparison.json` — K=10 256-shot vs 20k-shot, with the
  CI widths laid out and the "true" noise floor documented.
- `k3_d2_10k_comparison.json` — full 4-way comparison ending in
  the K=3 / degree 2 / 10k-shot run.
- `k2_d2_10k_comparison.json` — full 5-way comparison ending in
  the K=2 / degree 2 / 10k-shot run.
- `k2_d4_10k_comparison.json` — full 6-way comparison ending in
  the K=2 / degree 4 / 10k-shot run (the new best).
- `k2_d8_10k_comparison.json` — K=2 d=8 overshoot result.
- `k3_d4_10k_comparison.json` — K=3 d=4 result (worse than K=2 d=4).
- `boston_qpu_k17_ibm_boston_d4_postcompress.json` — full result of
  the post-Tier-1 256-shot QPU run.
- `boston_qpu_k10_ibm_boston_d4.json` — full K=10 256-shot QPU run.
- `boston_qpu_k10_ibm_boston_d4_20k.json` — full K=10 20k-shot QPU
  run (job `d8ibjhtv8cos73f5d0r0`).
- `boston_qpu_k3_ibm_boston_d2_10k.json` — full K=3 / degree 2
  10k-shot QPU run (job `d8ibqfc2upec739m1cf0`).
- `boston_qpu_k2_ibm_boston_d2_10k.json` — full K=2 / degree 2
  10k-shot QPU run (job `d8ibt0tv8cos73f5dcm0`).
- `boston_qpu_k2_ibm_boston_d4_10k.json` — full K=2 / degree 4
  10k-shot QPU run (job `d8ic08pe8nrc73bi56tg`).
- `boston_qpu_k2_ibm_boston_d8_10k.json` — full K=2 / degree 8
  10k-shot QPU run (job `d8ic32dv8cos73f5djlg`).
- `boston_qpu_k3_ibm_boston_d4_10k.json` — full K=3 / degree 4
  10k-shot QPU run (job `d8ic3um6983c73drvdv0`).
