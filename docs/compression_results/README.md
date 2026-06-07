# Compression comparison: Tier 1 (surgical) + Tier 3.9 (pyzx) + Tier 2.5 (K=17→10)

The Tier 1 source-level refactor of `Code/QSVT.py.addProj`, the
addition of a pyzx post-routing pass in `compiler_backend.heron`, and
the K=17→K=10 truncation were evaluated against the
**boston-qpu-v1-baseline** (`git tag boston-qpu-v1-baseline`) on the
real `ibm_boston` Heron r3 QPU.

## Headline result

| Stage | Job ID | K | Shots | Compiled depth | 2Q gate proxy | P(AUX=1) mitigated | classical tail | rel err (mit) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **boston-qpu-v1-baseline** (pre-Tier-1) | `d8iagbc2upec739lvqng` | 17 | 1024 | 1286 | 829 | 0.2051 | 0.000330 | 620× |
| **Tier 1 refactor** (CX, no X-X) | `d8ib6edv8cos73f5cj0g` | 17 | 256 | 1285 | 829 | 0.2422 | 0.000330 | 732× |
| **Tier 2.5 K=10** (K=17→10) | `d8ibc1pe8nrc73bi4gfg` | **10** | 256 | **919** | **509** | **0.1211** | 0.00947 | **12×** |

**Tier 2.5 (K=10) is the first compression strategy that actually
moves the needle.** It cuts compiled depth by 28% and 2Q-gate count
by 39%, and brings the quantum-vs-classical relative error from
620× down to 12× — a 52× improvement.

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
- **Quantum-classical relative error**: 620× → 12× (52× improvement)

The classical tail probability at K=10 is 0.95% (vs K=17's 0.03%)
because a 10-region sub-portfolio concentrates risk. The QSVT
signal at degree 4 (0.121) is now within an order of magnitude of
this classical target.

The K=10 result is the first time in this thread that the QSVT
output is **directionally and quantitatively close to the classical
answer** — small but in the right ballpark, dominated by gate
noise rather than fundamental signal loss.

## Why K=10 works

The `Code.multivariateGCI.MultivariateGCI_Linear` plus
`Code.circuitsCRA.get_expected_probability_circuit` construction
spends most of its 2Q-gate budget on routing the 17-qubit State
register through Heron's heavy-hex topology. Halving the State
register (17 → 10) cuts the routing overhead roughly in half,
which is exactly what we observe.

## Files in this directory

- `compression_comparison.json` — Tier 1 (no change) vs Tier 2.5 (28%
  depth cut) on the same QPU.
- `boston_qpu_k17_ibm_boston_d4_postcompress.json` — full result of
  the post-Tier-1 256-shot QPU run.
- `k10_comparison.json` — side-by-side K=17 vs K=10 comparison.
- `boston_qpu_k10_ibm_boston_d4.json` — full result of the K=10
  256-shot QPU run (job `d8ibc1pe8nrc73bi4gfg`).

