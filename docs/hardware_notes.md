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

## 2. IQM Garnet / Emerald (real quantum hardware)

> Only relevant if/when QSVT4CRA targets IQM hardware. The paper itself is simulation-only.

### 2.1 Transpilation failure modes

| Issue | Symptom | Fix |
|---|---|---|
| `reset` gate on IQM backend | Transpiler error or runtime error | Omit with `transpile(qc, backend, include_reset=False)` or strip before transpile |
| `memory='x'` measurement | `KeyError` in `internal_helpers` | Add `MX`/`MRX`/`MY`/`MRY` measurement gate extensions (see JunctionHackathon `internal_helpers.py`) |
| SWAP insertion on dense→physical map | 24+ SWAPs per circuit | Use pre-computed dense→physical layout mapping (no SWAP) |

### 2.2 Dense→physical qubit map (FakeGarnet 54q surface code)

```
{0:1, 1:0, 2:3, 3:7, 4:5, 5:4, 6:9, 7:8, 8:13, 9:12, 10:6, 11:11,
 12:10, 13:15, 14:14, 15:18, 16:19}
```

This eliminates **all SWAP gates** when transpiling Stim→Garnet circuits, yielding 24 native CZ / 38 R / 17 measure / depth 9.

### 2.3 When to bring these to QSVT4CRA

Only when the experiment is **hardware-in-the-loop**. For pure-simulation quantum resource estimates (Qiskit Aer), none of this is needed.

---

## 3. Cross-references

- `projects/JunctionHackathon/internal_helpers.py` — measurement-gate extensions
- `projects/JunctionHackathon-lumi/lumi_deployment/` — LUMI Slurm wrappers
- `projects/JunctionHackathon/build_garnet_qubit_map.py` — dense→physical map
- Obsidian: `infrastructure/lumi/lumi-quantumhack-2026.md` (LUMI bible)
- Obsidian: `infrastructure/lumi/lumi-junctionhackathon-54q-2026.md` (LUMI install gotchas)
