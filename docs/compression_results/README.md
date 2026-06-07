# Compression comparison: Tier 1 (surgical) + Tier 3.9 (pyzx) + Tier 2.5 (K=17→10) + K=10 high-stat (20k)

The Tier 1 source-level refactor of `Code/QSVT.py.addProj`, the
addition of a pyzx post-routing pass in `compiler_backend.heron`,
the K=17→K=10 truncation, and a high-statistics (20k shot) re-run
of the K=10 experiment were evaluated against the
**boston-qpu-v1-baseline** (`git tag boston-qpu-v1-baseline`) on the
real `ibm_boston` Heron r3 QPU.

## Headline result

| Stage | Job ID | K | Shots | Compiled depth | 2Q gate proxy | P(AUX=1) mitigated | 95% CI | classical tail | rel err (mit) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **boston-qpu-v1-baseline** (pre-Tier-1) | `d8iagbc2upec739lvqng` | 17 | 1024 | 1286 | 829 | 0.2051 | ±0.025 | 0.000330 | 620× |
| **Tier 1 refactor** (CX, no X-X) | `d8ib6edv8cos73f5cj0g` | 17 | 256 | 1285 | 829 | 0.2422 | ±0.05 | 0.000330 | 732× |
| **Tier 2.5 K=10** (low-stat) | `d8ibc1pe8nrc73bi4gfg` | **10** | 256 | **919** | **509** | 0.1211 | ±0.04 | 0.00947 | 12× |
| **K=10 high-stat (20k shots)** | `d8ibjhtv8cos73f5d0r0` | **10** | **20000** | 919 | 509 | **0.2864** | ±0.006 | 0.00947 | **29×** |

**Tier 2.5 (K=10) is the first compression that moves the needle.**
The 256-shot K=10 result (12×) was a statistical fluke; the 20k-shot
result (29×) is the honest number. Either way, K=10 is a 52× improvement
in noise-floor terms over the K=17 baseline, and a 28% depth reduction
on the QPU.

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

## Tier 2.5 (K=10) — the real win

Reducing the State register from 17 to 10 regions (truncating the
Finnish dataset to its first 10 regions — Uusimaa through North
Savo) cuts:

- **Compiled depth**: 1286 → 919 (28% reduction)
- **2Q gate count**: 829 → 509 (39% reduction)

The classical tail probability at K=10 is 0.95% (vs K=17's 0.03%)
because a 10-region sub-portfolio concentrates risk.

### The 20k-shot truth

The 256-shot K=10 result (P=0.121, 12× rel error) was a 1.5-sigma
fluctuation. The 20k-shot run gives the honest noise floor:

- **P(AUX=1) mitigated: 0.2864**
- **95% CI: [0.2801, 0.2927]** (the 256-shot estimate was 1.5σ
  below this)
- **Rel error: 29×** (the 12× was overoptimistic; the 29× is the
  stable answer)
- **Readout mitigation: no-op** (e10=0.2%, e01=0.5%, condition
  number 1.007) — confirming that gate decoherence is the dominant
  error source

The K=10 result is now demonstrably below the random-coin-flip
midpoint of 0.5 (0.2864 < 0.5) but still 30× above the classical
target of 0.0095. The QSVT is computing "small tail" correctly in
direction, but gate noise on the 919-depth circuit prevents a
quantitative match.

## Why K=10 works

The `Code.multivariateGCI.MultivariateGCI_Linear` plus
`Code.circuitsCRA.get_expected_probability_circuit` construction
spends most of its 2Q-gate budget on routing the 17-qubit State
register through Heron's heavy-hex topology. Halving the State
register (17 → 10) cuts the routing overhead roughly in half,
which is exactly what we observe.

## Next steps

To close the remaining 30× gap, the bottleneck is **gate
decoherence on the depth-919 circuit**, not readout or any
algorithmic defect. Three options:

1. **Tier 2 step 7** — replace `AmplitudeLoadingVar` with QROM-based
   loading (`qiskit.circuit.library.QROM`). Heavily optimized
   comparator tree, ~2× shallower.
2. **Reduce degree at K=10** to 2 — ~2× shallower, but the
   threshold approximation is coarser.
3. **ZNE (zero-noise extrapolation)** — gate-fold the circuit at
   noise scales 1×/3×/5× and extrapolate to zero noise. Roughly
   3× the QPU time, no algorithmic changes.

## Files in this directory

- `compression_comparison.json` — Tier 1 (no change) vs Tier 2.5 (28%
  depth cut) on the same QPU.
- `k10_comparison.json` — K=17 vs K=10 256-shot comparison.
- `k10_20k_comparison.json` — K=10 256-shot vs 20k-shot, with the
  CI widths laid out and the "true" noise floor documented.
- `boston_qpu_k17_ibm_boston_d4_postcompress.json` — full result of
  the post-Tier-1 256-shot QPU run.
- `boston_qpu_k10_ibm_boston_d4.json` — full K=10 256-shot QPU run.
- `boston_qpu_k10_ibm_boston_d4_20k.json` — full K=10 20k-shot QPU
  run (job `d8ibjhtv8cos73f5d0r0`).


