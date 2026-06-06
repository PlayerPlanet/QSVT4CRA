# LUMI Deployment for QSVT4CRA-Research-Run

This directory contains the LUMI HPC deployment artifacts for the
**Posterior-Propagated Factor-Copula QSVT for Apartment Loan Portfolio Risk**
research run.

> Ported from `projects/JunctionHackathon` and `projects/JunctionHackathon-lumi`,
> which share the same LUMI allocation (`project_465003017`).

## Files

| File | Purpose |
|---|---|
| `setup_lumi_env.sh` | Module stack + scratch bootstrap |
| `rsync_to_lumi.sh` | Local → LUMI sync helper |
| `slurm_qsvt4cra_research.sh` | Generic Slurm wrapper, dispatch by `EXPERIMENT` env var |

## Workflow

```bash
# 1. Local: validate script syntax
bash -n lumi_deployment/setup_lumi_env.sh
bash -n lumi_deployment/rsync_to_lumi.sh
bash -n lumi_deployment/slurm_qsvt4cra_research.sh

# 2. Local: sync to LUMI
bash lumi_deployment/rsync_to_lumi.sh

# 3. LUMI: install Python deps once (uses scratch, not home)
ssh lumi
cd /scratch/project_465003017/$USER/qsvt4cra-research
bash lumi_deployment/setup_lumi_env.sh
pip install --no-cache-dir --target=./site-packages -r requirements.txt

# 4. LUMI: run experiments
EXPERIMENT=sbi_train sbatch lumi_deployment/slurm_qsvt4cra_research.sh
EXPERIMENT=mc_ground_truth sbatch lumi_deployment/slurm_qsvt4cra_research.sh
EXPERIMENT=qsvt_sweep sbatch lumi_deployment/slurm_qsvt4cra_research.sh
EXPERIMENT=ood_robustness sbatch lumi_deployment/slurm_qsvt4cra_research.sh
EXPERIMENT=resource_scaling sbatch lumi_deployment/slurm_qsvt4cra_research.sh
EXPERIMENT=figures sbatch lumi_deployment/slurm_qsvt4cra_research.sh
```

## Compute budget (per project brief)

| Resource | Available | Target |
|---|---|---|
| GPU-hours | 2000 | 50–150 |
| CPU-hours | Large | <500 |

Each experiment is sized to fit within budget. See `docs/hardware_notes.md`
for the full LUMI module stack and known gotchas.
