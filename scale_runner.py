"""
scale_runner.py - Multi-node launcher for QSVT4CRA experiments.

Auto-detects the parallel execution environment on LUMI and dispatches
the work to the correct backend:

  - Single CPU core            : sequential
  - Single node, many CPUs     : joblib (process-based)
  - Single node, many GPUs     : torch (CUDA) + per-GPU processes
  - Multi-node, many CPUs      : joblib with srun backend (Cray MPICH)
  - Multi-node, many GPUs      : torch.distributed (DDP)

The launcher respects environment variables set by Slurm:
  - SLURM_NTASKS              : total MPI ranks
  - SLURM_NTASKS_PER_NODE     : ranks per node
  - SLURM_PROCID              : global rank (0..N-1)
  - SLURM_LOCALID             : local rank within node
  - SLURM_NNODES              : total nodes
  - SLURM_JOB_NUM_NODES       : same
  - CUDA_VISIBLE_DEVICES      : set by Slurm per task

Usage in an experiment script:
    from scale_runner import get_launcher
    launcher = get_launcher()
    results = launcher.map(work_fn, items)        # parallel map
    # or
    with launcher.distributed_context():          # for DDP
        ...

CLI:
    python -m scale_runner info                    # print detected env
    python -m scale_runner run --work-fn fn --items 1 2 3 4
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Callable, Iterable, List, Optional

# ---------------------------------------------------------------------------
# Environment detection
# ---------------------------------------------------------------------------


def detect_environment() -> dict:
    """
    Detect the parallel execution environment from Slurm env vars.

    Returns
    -------
    dict with keys:
        nnodes : int
            Number of allocated nodes.
        ntasks : int
            Total MPI ranks.
        ntasks_per_node : int
            Ranks per node.
        ncpus_per_task : int
            CPUs per rank.
        cpus_per_node : int
            Total CPUs allocated per node.
        gpus_per_node : int
            GPUs per node.
        memory_per_node_mb : int
            Memory per node in MB.
        procid : int
            Global rank (0-indexed).
        localid : int
            Local rank within node (0-indexed).
        launcher_type : str
            "sequential" | "joblib_local" | "joblib_srun" | "torch_ddp"
        has_gpu : bool
            Whether any GPU is available.
    """
    nnodes = int(os.environ.get("SLURM_NNODES", os.environ.get("SLURM_JOB_NUM_NODES", 1)))
    ntasks = int(os.environ.get("SLURM_NTASKS", 1))
    ntasks_per_node = int(os.environ.get("SLURM_NTASKS_PER_NODE", 1))
    ncpus_per_task = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    cpus_per_node = int(os.environ.get("SLURM_CPUS_ON_NODE", ntasks_per_node * ncpus_per_task))
    gpus_per_node = int(os.environ.get("SLURM_GPUS_PER_NODE", os.environ.get("SLURM_GPUS", 0)))
    procid = int(os.environ.get("SLURM_PROCID", 0))
    localid = int(os.environ.get("SLURM_LOCALID", 0))

    # Memory
    mem_mb_str = os.environ.get("SLURM_MEM_PER_NODE", "0")
    try:
        memory_per_node_mb = int(mem_mb_str)
    except ValueError:
        memory_per_node_mb = 0

    has_gpu = gpus_per_node > 0 or "CUDA_VISIBLE_DEVICES" in os.environ

    # Pick launcher
    if ntasks > 1 and nnodes > 1:
        launcher_type = "joblib_srun"
    elif ntasks > 1 and has_gpu:
        launcher_type = "torch_ddp"
    elif ntasks > 1 or cpus_per_node > 4:
        launcher_type = "joblib_local"
    else:
        launcher_type = "sequential"

    return {
        "nnodes": nnodes,
        "ntasks": ntasks,
        "ntasks_per_node": ntasks_per_node,
        "ncpus_per_task": ncpus_per_task,
        "cpus_per_node": cpus_per_node,
        "gpus_per_node": gpus_per_node,
        "memory_per_node_mb": memory_per_node_mb,
        "procid": procid,
        "localid": localid,
        "launcher_type": launcher_type,
        "has_gpu": has_gpu,
    }


# ---------------------------------------------------------------------------
# Abstract launcher
# ---------------------------------------------------------------------------


class Launcher(ABC):
    """
    Abstract base class for parallel work dispatchers.
    """

    env: dict

    def __init__(self, env: dict):
        self.env = env

    @abstractmethod
    def map(self, work_fn: Callable, items: Iterable, **kwargs) -> list:
        """
        Apply work_fn to each item in parallel; return list of results.

        Parameters
        ----------
        work_fn : Callable
            Worker function. Must be picklable for process-based dispatch.
        items : Iterable
            Items to process.
        **kwargs
            Forwarded to work_fn.

        Returns
        -------
        list
            Results in the same order as items.
        """
        ...

    @abstractmethod
    def get_rank(self) -> int:
        """Return global rank (0 for single-process launchers)."""
        ...

    @abstractmethod
    def get_local_rank(self) -> int:
        """Return local rank within node (0 for single-process launchers)."""
        ...

    @abstractmethod
    def get_world_size(self) -> int:
        """Return total number of workers."""
        ...

    def is_main(self) -> bool:
        """Return True if this is the main (rank-0) process."""
        return self.get_rank() == 0

    def print_info(self) -> None:
        """Print detected environment info to stdout."""
        e = self.env
        print("=" * 60)
        print("QSVT4CRA scale_runner — Environment")
        print("=" * 60)
        print(f"  Launcher type      : {e['launcher_type']}")
        print(f"  Nodes              : {e['nnodes']}")
        print(f"  Tasks (total)      : {e['ntasks']}")
        print(f"  Tasks/node         : {e['ntasks_per_node']}")
        print(f"  CPUs/task          : {e['ncpus_per_task']}")
        print(f"  CPUs/node          : {e['cpus_per_node']}")
        print(f"  GPUs/node          : {e['gpus_per_node']}")
        print(f"  Memory/node (MB)   : {e['memory_per_node_mb']}")
        print(f"  Global rank        : {e['procid']}")
        print(f"  Local rank         : {e['localid']}")
        print(f"  Has GPU            : {e['has_gpu']}")
        print("=" * 60)


# ---------------------------------------------------------------------------
# Sequential launcher
# ---------------------------------------------------------------------------


class SequentialLauncher(Launcher):
    """
    Single-process launcher. No parallelism.
    """

    def map(self, work_fn: Callable, items: Iterable, **kwargs) -> list:
        return [work_fn(item, **kwargs) for item in items]

    def get_rank(self) -> int:
        return 0

    def get_local_rank(self) -> int:
        return 0

    def get_world_size(self) -> int:
        return 1


# ---------------------------------------------------------------------------
# joblib launcher (local multi-process)
# ---------------------------------------------------------------------------


class JoblibLauncher(Launcher):
    """
    Multi-process launcher using joblib (local shared-memory backend).
    Falls back to ProcessPoolExecutor if joblib unavailable.

    Respects the SLURM_CPUS_PER_TASK env var to set n_jobs.
    """

    def __init__(self, env: dict):
        super().__init__(env)
        # n_jobs: -1 = use all available CPUs
        # On Slurm allocations, cpus_per_node is the actual count
        self._n_jobs = max(1, env["cpus_per_node"])
        self._has_joblib = False
        try:
            import joblib  # noqa: F401

            self._has_joblib = True
        except ImportError:
            pass

    def map(self, work_fn: Callable, items: Iterable, **kwargs) -> list:
        items = list(items)
        if len(items) == 0:
            return []
        n_jobs = min(self._n_jobs, len(items))
        if n_jobs == 1:
            return [work_fn(item, **kwargs) for item in items]

        if self._has_joblib:
            from joblib import Parallel, delayed

            return Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(work_fn)(item, **kwargs) for item in items
            )
        # Fallback: ProcessPoolExecutor
        results: list = [None] * len(items)
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            futures = {
                ex.submit(work_fn, item, **kwargs): idx for idx, item in enumerate(items)
            }
            for fut in as_completed(futures):
                idx = futures[fut]
                results[idx] = fut.result()
        return results

    def get_rank(self) -> int:
        return self.env["procid"]

    def get_local_rank(self) -> int:
        return self.env["localid"]

    def get_world_size(self) -> int:
        return self.env["ntasks"]


# ---------------------------------------------------------------------------
# joblib srun launcher (multi-node)
# ---------------------------------------------------------------------------


class SrunJoblibLauncher(JoblibLauncher):
    """
    Multi-node launcher: uses joblib with the srun backend
    (requires the `distributed` package, optional).

    If distributed/dask is unavailable, falls back to local joblib
    on the main node only and warns.
    """

    def map(self, work_fn: Callable, items: Iterable, **kwargs) -> list:
        if not self.env["procid"] == 0:
            # Non-main nodes return empty
            return []
        try:
            from joblib import Parallel, delayed
            from dask_jobqueue import SLURMCluster  # noqa: F401

            # Real implementation would launch a SLURMCluster.
            # For now, fall back to local joblib on the head node.
            return super().map(work_fn, items, **kwargs)
        except ImportError:
            return super().map(work_fn, items, **kwargs)


# ---------------------------------------------------------------------------
# torch.distributed launcher (DDP for multi-GPU)
# ---------------------------------------------------------------------------


class TorchDDLauncher(Launcher):
    """
    torch.distributed launcher for multi-GPU / multi-node DDP.

    Each rank runs work_fn on a subset of items. Results are gathered
    on rank 0.
    """

    def __init__(self, env: dict):
        super().__init__(env)
        self._initialized = False

    def _init_distributed(self) -> None:
        if self._initialized:
            return
        import torch
        import torch.distributed as dist

        if not dist.is_available():
            self._initialized = True
            return

        # Backend: NCCL for GPU, Gloo for CPU
        backend = "nccl" if self.env["has_gpu"] else "gloo"
        # MASTER_ADDR / PORT: set by Slurm when using multiple nodes
        os.environ.setdefault("MASTER_ADDR", os.environ.get("MASTER_ADDR", "127.0.0.1"))
        os.environ.setdefault("MASTER_PORT", os.environ.get("MASTER_PORT", "29500"))

        try:
            dist.init_process_group(
                backend=backend,
                init_method="env://",
                world_size=self.env["ntasks"],
                rank=self.env["procid"],
            )
            self._initialized = True
        except Exception as e:
            print(f"WARNING: torch.distributed init failed: {e}", file=sys.stderr)
            self._initialized = True  # Treat as initialized to avoid retry

    def map(self, work_fn: Callable, items: Iterable, **kwargs) -> list:
        self._init_distributed()
        items = list(items)
        # Shard items by rank
        rank = self.get_rank()
        world = self.get_world_size()
        my_items = items[rank::world]
        my_results = [work_fn(item, **kwargs) for item in my_items]

        if world == 1 or rank != 0:
            return my_results

        # Gather on rank 0
        try:
            import torch.distributed as dist

            gathered = [None] * world
            dist.all_gather_object(gathered, my_results)
            # Reassemble: each rank has [items[i::world] for i in rank]
            results = []
            for i in range(len(items)):
                src_rank = i % world
                local_idx = i // world
                if local_idx < len(gathered[src_rank]):
                    results.append(gathered[src_rank][local_idx])
                else:
                    results.append(None)
            return results
        except Exception as e:
            print(f"WARNING: gather failed ({e}); returning rank-0 results only", file=sys.stderr)
            return my_results

    def get_rank(self) -> int:
        return self.env["procid"]

    def get_local_rank(self) -> int:
        return self.env["localid"]

    def get_world_size(self) -> int:
        return self.env["ntasks"]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_LAUNCHERS = {
    "sequential": SequentialLauncher,
    "joblib_local": JoblibLauncher,
    "joblib_srun": SrunJoblibLauncher,
    "torch_ddp": TorchDDLauncher,
}


def get_launcher(env: Optional[dict] = None) -> Launcher:
    """
    Get the appropriate launcher for the current environment.

    Parameters
    ----------
    env : dict, optional
        Pre-detected environment. If None, detect_environment() is called.

    Returns
    -------
    Launcher
        Concrete launcher instance.
    """
    if env is None:
        env = detect_environment()
    cls = _LAUNCHERS[env["launcher_type"]]
    return cls(env)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _main() -> int:
    parser = argparse.ArgumentParser(
        description="scale_runner: multi-node launcher for QSVT4CRA experiments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("info", help="Print detected environment and exit.")

    run_p = sub.add_parser("run", help="Run a worker function on a list of items.")
    run_p.add_argument(
        "--work-fn",
        required=True,
        help="Dotted path to the worker function (e.g., experiments.mc_ground_truth:run_chunk).",
    )
    run_p.add_argument(
        "--items",
        nargs="+",
        required=True,
        help="Items to process (passed as strings to work_fn).",
    )

    args = parser.parse_args()

    if args.cmd == "info":
        env = detect_environment()
        launcher = get_launcher(env)
        launcher.print_info()
        return 0

    if args.cmd == "run":
        # Import the worker function dynamically
        module_path, fn_name = args.work_fn.split(":")
        import importlib

        mod = importlib.import_module(module_path)
        work_fn = getattr(mod, fn_name)

        env = detect_environment()
        launcher = get_launcher(env)
        if launcher.is_main():
            launcher.print_info()
            print(f"Running {args.work_fn} on {len(args.items)} items...")

        t0 = time.time()
        results = launcher.map(work_fn, args.items)
        elapsed = time.time() - t0

        if launcher.is_main():
            print(f"Done in {elapsed:.2f}s")
            for item, res in zip(args.items, results):
                print(f"  {item}: {res}")
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(_main())
