# Compression comparison: Tier 1 (surgical) + Tier 3.9 (pyzx)

The Tier 1 source-level refactor of `Code/QSVT.py.addProj` and the
addition of a pyzx post-routing pass in `compiler_backend.heron` were
evaluated against the **boston-qpu-v1-baseline** (`git tag
boston-qpu-v1-baseline`) on the real `ibm_boston` Heron r3 QPU.

## Headline result

| Stage | Job ID | Shots | Compiled depth | 2Q gate proxy | P(AUX=1) mitigated |
|---|---|---:|---:|---:|---:|
| **boston-qpu-v1-baseline** (pre-compression) | `d8iagbc2upec739lvqng` | 1024 | 1286 | 829 | 0.2051 |
| **post-compression** (Tier 1 refactor) | `d8ib6edv8cos73f5cj0g` | 256 | **1285** | 829 | 0.2422 |

**Depth is unchanged.** The transpiler at Qiskit's preset-pass-manager
optimization level 3 was already cancelling the X-X pair around the
QSVT projector and decomposing `mcx([Target], AUX)` to a single CX.
The source-level refactor of `Code/QSVT.py` is therefore a code-clarity
and testability win but a **zero-gate-count** win on the compiled
circuit.

## What we did get

1. **`Code/QSVT.py` is cleaner and faster to compile.** The X-X
   sandwich is no longer emitted, and `mcx` is no longer called for
   a single-control case. The Qiskit transpiler used to do this
   optimization after the circuit was built; now it's explicit at
   source level.
2. **6 deterministic statevector tests** in
   `tests/test_code_qsvt_equivalence.py` lock in the math. They
   build the QSVT with K=4, degrees 2/4/8 and assert the
   statevector matches a saved baseline to within `1e-9`. Any
   future QSVT-construction refactor must pass these.
3. **A pyzx post-routing pass** is available as an **opt-in** flag
   in `HeronCompileConfig.use_pyzx` and `--use-pyzx` on the
   `boston_qpu` CLI. On small/medium circuits this can cut depth
   ~10-40 percent, but on the K=17 QSVT the re-translation step
   adds ~30 percent depth on Heron's heavy-hex. Default: off.

## Why pyzx is OFF by default

On the K=17 degree-4 QSVT (the production circuit), the pyzx
round-trip followed by re-translation increases depth:

| | depth | 2Q gates | notes |
|---|---:|---:|---|
| Without pyzx | 1769 | 965 | (FakeSherbrooke) |
| With pyzx    | 2306 | 1165 | re-translation adds heavy-hex routing |

The reason: `zx.basic_optimization` outputs `{h, cx, cz, rz, x}`
which is NOT Heron's native basis. The re-translation step has to
re-route the new circuit on the heavy-hex topology, and the new
routing is worse than what the original transpiler chose.

## What WOULD reduce the K=17 QSVT depth meaningfully

The source-level refactor + pyzx are the wrong places to look. The
real lever is the **circuit topology**:

1. **Reduce K from 17 to 10** — halves the State register qubits and
   the routing overhead. The rest of the codebase
   (`experiments/qsvt_sweep.py`) already uses K=10 with a
   Gaussian factor copula. The Finnish mortgage K=17 toy is the
   *data*, not the *algorithm*.
2. **Switch to a phase-kernel / QFT-based loss encoding** — the
   current `AmplitudeLoadingVar` adds a 2K-depth linear-add tree
   that's repeated 2*(degree-1) times. A QFT-based encoding would
   be O(K²) total.
3. **Use QROM** (`qiskit.circuit.library.QROM`) for the amplitude
   loading — Qiskit's QROM is a heavily optimized comparator tree.

These are **Tier 2** changes that the user explicitly deferred in
favour of the lower-risk Tier 1 + Tier 3.9. With Tier 2 in place,
the QPU noise floor at K=17 should drop below the QSVT signal at
degree 8.

## Files in this directory

- `boston_qpu_k17_ibm_boston_d4_postcompress.json` — full result of
  the post-Tier-1 256-shot QPU run.
- `compression_comparison.json` — the same data in a compact
  table format plus the structured `takeaways` list.
