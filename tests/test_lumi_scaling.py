"""
Tests for the multi-node scale_runner and LUMI dispatching.

These tests verify:
- scale_runner environment detection
- Launcher instantiation
- Profile parsing from LUMI Slurm profiles
- Dispatch argument validation

No LUMI or Slurm is required; tests run on the local machine.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def clean_env(monkeypatch):
    """Remove all Slurm env vars for a clean baseline."""
    for k in list(os.environ.keys()):
        if k.startswith("SLURM_") or k == "CUDA_VISIBLE_DEVICES":
            monkeypatch.delenv(k, raising=False)
    return monkeypatch


# ---------------------------------------------------------------------------
# scale_runner.detect_environment
# ---------------------------------------------------------------------------


def test_detect_environment_default(clean_env):
    """Default env (no Slurm vars) → sequential launcher."""
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from scale_runner import detect_environment

    env = detect_environment()
    assert env["nnodes"] == 1
    assert env["ntasks"] == 1
    assert env["launcher_type"] == "sequential"
    assert env["has_gpu"] is False


def test_detect_environment_local_joblib(clean_env):
    """With cpus>4, should pick joblib_local."""
    clean_env.setenv("SLURM_CPUS_ON_NODE", "32")
    clean_env.setenv("SLURM_NTASKS", "1")
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from scale_runner import detect_environment

    env = detect_environment()
    assert env["cpus_per_node"] == 32
    assert env["launcher_type"] == "joblib_local"


def test_detect_environment_multi_node(clean_env):
    """Multi-node → srun joblib launcher."""
    clean_env.setenv("SLURM_NNODES", "4")
    clean_env.setenv("SLURM_NTASKS", "4")
    clean_env.setenv("SLURM_CPUS_ON_NODE", "128")
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from scale_runner import detect_environment

    env = detect_environment()
    assert env["nnodes"] == 4
    assert env["ntasks"] == 4
    assert env["launcher_type"] == "joblib_srun"


def test_detect_environment_with_gpu(clean_env):
    """CUDA_VISIBLE_DEVICES → has_gpu=True."""
    clean_env.setenv("CUDA_VISIBLE_DEVICES", "0")
    clean_env.setenv("SLURM_NTASKS", "2")
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from scale_runner import detect_environment

    env = detect_environment()
    assert env["has_gpu"] is True
    assert env["launcher_type"] in ("joblib_srun", "torch_ddp")


# ---------------------------------------------------------------------------
# scale_runner.get_launcher
# ---------------------------------------------------------------------------


def test_get_launcher_sequential(clean_env):
    """Sequential launcher can run a work fn."""
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from scale_runner import get_launcher

    launcher = get_launcher()
    assert launcher.get_world_size() == 1
    assert launcher.get_rank() == 0
    assert launcher.is_main()

    results = launcher.map(lambda x: x * 2, [1, 2, 3, 4])
    assert results == [2, 4, 6, 8]


def test_get_launcher_joblib(clean_env):
    """Joblib launcher can run a work fn in parallel."""
    clean_env.setenv("SLURM_CPUS_ON_NODE", "4")
    clean_env.setenv("SLURM_NTASKS", "1")
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from scale_runner import get_launcher

    launcher = get_launcher()
    assert launcher.get_world_size() == 1
    results = launcher.map(lambda x: x ** 2, [1, 2, 3, 4])
    assert sorted(results) == [1, 4, 9, 16]


# ---------------------------------------------------------------------------
# Profile parsing
# ---------------------------------------------------------------------------


PROFILE_DIR = Path(__file__).parent.parent / "lumi_deployment" / "profiles"


def _list_profiles() -> list[str]:
    """List all profile names."""
    if not PROFILE_DIR.exists():
        pytest.skip(f"Profile dir not found: {PROFILE_DIR}")
    return [p.stem for p in sorted(PROFILE_DIR.glob("*.sh"))]


def test_at_least_10_profiles():
    """Must have at least 10 profile presets."""
    profiles = _list_profiles()
    assert len(profiles) >= 10, f"Only {len(profiles)} profiles: {profiles}"


def test_profiles_have_sbatch_directives():
    """Every profile must have at least one #SBATCH directive."""
    profiles = _list_profiles()
    for p in profiles:
        path = PROFILE_DIR / f"{p}.sh"
        content = path.read_text()
        sbatch_lines = [l for l in content.splitlines() if l.startswith("#SBATCH")]
        assert len(sbatch_lines) > 0, f"Profile {p} has no #SBATCH directives"
        # Must have account, partition, nodes
        joined = "\n".join(sbatch_lines)
        assert "--account=project_465003017" in joined, f"{p}: missing account"
        assert "--partition=" in joined, f"{p}: missing partition"
        assert "--nodes=" in joined, f"{p}: missing nodes"


def test_profiles_source_dispatcher():
    """Every profile must source the shared dispatcher."""
    profiles = _list_profiles()
    for p in profiles:
        path = PROFILE_DIR / f"{p}.sh"
        content = path.read_text()
        assert "dispatcher.sh" in content, f"{p}: does not source dispatcher.sh"


def test_profiles_distinct_partitions():
    """Profiles declare valid partitions (standard or standard-g)."""
    profiles = _list_profiles()
    valid_partitions = {"standard", "standard-g"}
    for p in profiles:
        path = PROFILE_DIR / f"{p}.sh"
        content = path.read_text()
        match = re.search(r"--partition=(\S+)", content)
        assert match, f"{p}: no partition"
        assert match.group(1) in valid_partitions, (
            f"{p}: invalid partition {match.group(1)}"
        )


def test_cpu_profiles_no_gpus():
    """CPU profiles must not request GPUs."""
    profiles = _list_profiles()
    for p in profiles:
        if p.startswith("cpu_") or p == "hybrid_cpu":
            path = PROFILE_DIR / f"{p}.sh"
            content = path.read_text()
            assert "--gpus-per-node=" not in content or "--gpus-per-node=0" in content, (
                f"{p}: CPU profile requests GPU"
            )


def test_gpu_profiles_have_gpus():
    """GPU profiles must request at least 1 GPU."""
    profiles = _list_profiles()
    for p in profiles:
        if p.startswith("gpu_") or p == "hybrid_gpu":
            path = PROFILE_DIR / f"{p}.sh"
            content = path.read_text()
            match = re.search(r"--gpus-per-node=(\d+)", content)
            assert match, f"{p}: GPU profile has no --gpus-per-node"
            assert int(match.group(1)) >= 1, f"{p}: requests 0 GPUs"


def test_profile_time_limits_reasonable():
    """All profiles must request at least 5 min and at most 24h."""
    profiles = _list_profiles()
    for p in profiles:
        path = PROFILE_DIR / f"{p}.sh"
        content = path.read_text()
        match = re.search(r"--time=(\d{1,2}):(\d{2}):(\d{2})", content)
        assert match, f"{p}: no --time directive"
        h, m, s = (int(x) for x in match.groups())
        total_min = h * 60 + m + s / 60
        assert 5 <= total_min <= 24 * 60, (
            f"{p}: time {h}h{m}m out of range"
        )


# ---------------------------------------------------------------------------
# Dispatcher experiment case
# ---------------------------------------------------------------------------


DISPATCHER_PATH = Path(__file__).parent.parent / "lumi_deployment" / "dispatcher.sh"


def test_dispatcher_lists_all_experiments():
    """Dispatcher must list all 8 experiment types."""
    if not DISPATCHER_PATH.exists():
        pytest.skip("dispatcher.sh not found")
    content = DISPATCHER_PATH.read_text()
    expected = [
        "sbi_train",
        "mc_ground_truth",
        "mc_ground_truth_weak",
        "qsvt_sweep",
        "ood_robustness",
        "resource_scaling",
        "figures",
        "smoke_test",
    ]
    for exp in expected:
        assert exp in content, f"dispatcher.sh missing case for {exp}"


def test_dispatcher_handles_missing_checkpoint():
    """If no checkpoint, dispatcher should error out cleanly."""
    if not DISPATCHER_PATH.exists():
        pytest.skip("dispatcher.sh not found")
    content = DISPATCHER_PATH.read_text()
    assert "no checkpoint" in content or "Must provide" in content


# ---------------------------------------------------------------------------
# submit_from_windows.ps1
# ---------------------------------------------------------------------------


SUBMIT_PS1_PATH = Path(__file__).parent.parent / "lumi_deployment" / "submit_from_windows.ps1"


def test_submit_ps1_exists():
    """The Windows submit script must exist."""
    assert SUBMIT_PS1_PATH.exists(), f"Missing: {SUBMIT_PS1_PATH}"


def test_submit_ps1_handles_args():
    """The submit script must define the standard flag set."""
    if not SUBMIT_PS1_PATH.exists():
        pytest.skip("submit_from_windows.ps1 not found")
    content = SUBMIT_PS1_PATH.read_text()
    for flag in ("SmokeOnly", "SkipGpu", "SkipWeak", "DryRun", "LumiUser"):
        assert flag in content, f"submit_from_windows.ps1 missing param ${flag}"


def test_submit_ps1_uses_correct_user():
    """The submit script must default to user kkiirikk (or env-overrideable)."""
    if not SUBMIT_PS1_PATH.exists():
        pytest.skip("submit_from_windows.ps1 not found")
    content = SUBMIT_PS1_PATH.read_text()
    assert "kkiirikk" in content
    assert "lumi.csc.fi" in content
    assert "project_465003017" in content


# ---------------------------------------------------------------------------
# Experiments sbi_train CLI
# ---------------------------------------------------------------------------


def test_sbi_train_help_runs():
    """sbi_train --help must succeed (validates arg parser)."""
    result = subprocess.run(
        [sys.executable, "-m", "experiments.sbi_train", "--help"],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).parent.parent),
    )
    assert result.returncode == 0
    assert "--method" in result.stdout
    assert "--K" in result.stdout
    assert "--output" in result.stdout


def test_sbi_train_unknown_method_errors():
    """Invalid method should error out."""
    result = subprocess.run(
        [
            sys.executable, "-m", "experiments.sbi_train",
            "--method", "invalid_method_xyz",
            "--K", "5",
            "--n-simulations", "5",
            "--n-rounds", "1",
            "--device", "cpu",
            "--output", "_test_invalid.pt",
        ],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).parent.parent),
    )
    assert result.returncode != 0
    assert "invalid choice" in result.stderr
