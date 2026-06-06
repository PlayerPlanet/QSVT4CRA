# Hardware Notes: LUMI and IQM

> **Status**: Future-work / not yet exercised by the QSVT4CRA paper repo.
> **Source**: Lessons ported from `projects/JunctionHackathon` and `projects/JunctionHackathon-lumi` (2026-06).
> **Audience**: Anyone planning to run QSVT4CRA on real quantum hardware or on LUMI HPC.

This file is a port of infrastructure lessons from the JunctionHackathon family. It is not yet load-bearing for the paper itself (which is simulation-only) but is captured here so the next person does not re-derive these from scratch.

---

## 1. LUMI HPC (CSC, Finland)

LUMI is a EuroHPC pre-exascale system. Both `QSVT4CRA-research-run` (autonomous research run) and `JunctionHackathon` share the same allocation.

### 1.1 Account & access

| Item | Value |
|---|---|
| Project ID | `project_465003017` |
| Owner | `kkiirikk` (CSC account) |
| SSH alias | `lumi` in `~/.ssh/config` → `ssh kkiirikk@lumi.csc.fi` |
| LUMI web | https://www.lumi-supercomputer.eu/ |

### 1.2 Verified module stack (2026-06)

```bash
module --force purge
module load LUMI/25.09
module load partition/G          # GPU nodes (MI250X)
module load rocm                 # resolves to rocm/6.4.4
module load cray-python/3.11.7
```

The `LUMI/23.09` stack is **stale** and is missing `cray-python/3.11.5` — do not use it.

### 1.3 LUMI-G GPU node quirks

- 4× AMD MI250X per node = **8 GCDs visible to Slurm** (each MI250X has 2 dies)
- 56 CPU cores / node
- **No local disk** — `/tmp` is RAM-backed
- ROCm device targets `gfx90a` for MI250X
- `python -m venv` cannot bootstrap pip under `cray-python/3.11.7`

### 1.4 Three hard-won workarounds

#### a) Home directory quota is ~1 GB

torch alone is ~800 MB. **Never** install into `$HOME`. Always:

```bash
export SCRATCH=/scratch/$SLURM_JOB_ACCOUNT/$USER
mkdir -p "$SCRATCH"
```

#### b) `pip install` with `--target` and rsync

```bash
# On login node: stage to /tmp, then rsync to scratch
python -m pip install --no-cache-dir --target=/tmp/site-packages -r requirements.txt
rsync -a /tmp/site-packages/ "$SCRATCH/site-packages/"
```

#### c) Lustre cross-device rename

`mv` between filesystems fails with `EXDEV`. Always:

```bash
# stage → rsync → atomic rename
cp local_file.py "$SCRATCH/target/"     # slow but cross-FS safe
# or
rsync -a --remove-source-files src/ "$SCRATCH/dst/"
```

### 1.5 Slurm wrapper gotcha

`#SBATCH --account=${SLURM_JOB_ACCOUNT}` is **not expanded by sbatch**. Pin the literal:

```bash
#SBATCH --account=project_465003017
```

Or set `export SBATCH_ACCOUNT=project_465003017` in the calling environment and reference `$SBATCH_ACCOUNT` in the script. Validate with `bash -n script.sh` before submission.

### 1.6 Reference templates

See `lumi_deployment/` in this repo:
- `setup_lumi_env.sh` — module stack + scratch bootstrap
- `slurm_qsvt_risk_probes.sh` — Slurm wrapper for the QSVT risk probes
- `rsync_to_lumi.sh` — local → LUMI sync helper

---

## 2. IBM Heron r3 / `ibm_boston` simulation

> Relevant for the Finnish-mortgage K=17 toy dataset introduced in
> `Code/dataset_regions.py`.

The preferred non-QPU hardware-in-the-loop workflow is:

1. Build the K=17 Finnish regional mortgage circuit.
2. Fetch current IBM backend target/calibration data for `ibm_boston` using
   `$IBM_API_KEY`.
3. Compile with `compiler_backend.heron.compile_for_backend`, which applies a
   greedy calibration-aware initial layout over the Heron heavy-hex topology.
4. Simulate with `AerSimulator.from_backend(backend)` on LUMI.

Local command:

```bash
export IBM_API_KEY=...   # never echo this value into logs
python -m experiments.heron_simulation \
  --backend-name ibm_boston \
  --degree 4 \
  --shots 1024 \
  --output results/heron_k17_ibm_boston_d4.json
```

LUMI command after syncing the repo and staged Python packages:

```bash
export IBM_API_KEY=...
sbatch --export=ALL lumi_deployment/slurm_heron_simulation.sh
```

Use `DEGREE`, `SHOTS`, `BACKEND_NAME`, `TARGET_LOSS_FRACTION`, and `OUTPUT` to
override defaults in the Slurm wrapper.  The wrapper intentionally checks that
`IBM_API_KEY` is present but does not print it.

### 2.1 Known Heron caveats

| Issue | Symptom | Mitigation |
|---|---|---|
| Width is fine but depth is large | K=17 degree-4 smoke transpiles to hundreds of 2Q gates even before calibration-aware routing | Start with degree 4, then sweep 8/16 only after checking compiled depth and noisy result stability |
| Backend access from LUMI | Runtime service cannot fetch `ibm_boston` | Submit with `sbatch --export=ALL` and verify outbound network/token policy |
| `qiskit_finance` missing | `ModuleNotFoundError` in `Code.multivariateGCI` | `Code.multivariateGCI` now has a small internal `n_normal=2`-friendly normal-state-prep fallback |
| Calibration drift | Re-running compilation chooses a different physical patch | Store JSON output; it includes selected physical qubits and gate counts |

---

## 3. IQM Garnet / Emerald (real quantum hardware)

> Only relevant if/when QSVT4CRA targets IQM hardware. The paper itself is simulation-only.

### 3.1 Transpilation failure modes

| Issue | Symptom | Fix |
|---|---|---|
| `reset` gate on IQM backend | Transpiler error or runtime error | Omit with `transpile(qc, backend, include_reset=False)` or strip before transpile |
| `memory='x'` measurement | `KeyError` in `internal_helpers` | Add `MX`/`MRX`/`MY`/`MRY` measurement gate extensions (see JunctionHackathon `internal_helpers.py`) |
| SWAP insertion on dense→physical map | 24+ SWAPs per circuit | Use pre-computed dense→physical layout mapping (no SWAP) |

### 3.2 Dense→physical qubit map (FakeGarnet 54q surface code)

```
{0:1, 1:0, 2:3, 3:7, 4:5, 5:4, 6:9, 7:8, 8:13, 9:12, 10:6, 11:11,
 12:10, 13:15, 14:14, 15:18, 16:19}
```

This eliminates **all SWAP gates** when transpiling Stim→Garnet circuits, yielding 24 native CZ / 38 R / 17 measure / depth 9.

### 3.3 When to bring these to QSVT4CRA

Only when the experiment is **hardware-in-the-loop**. For pure-simulation quantum resource estimates (Qiskit Aer), none of this is needed.

---

## 4. Cross-references

- `projects/JunctionHackathon/internal_helpers.py` — measurement-gate extensions
- `projects/JunctionHackathon-lumi/lumi_deployment/` — LUMI Slurm wrappers
- `projects/JunctionHackathon/build_garnet_qubit_map.py` — dense→physical map
- Obsidian: `infrastructure/lumi/lumi-quantumhack-2026.md` (LUMI bible)
- Obsidian: `infrastructure/lumi/lumi-junctionhackathon-54q-2026.md` (LUMI install gotchas)
