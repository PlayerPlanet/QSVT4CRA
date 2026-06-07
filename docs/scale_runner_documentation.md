# `scale_runner.py` — Multi-node launcher abstraction

> A small library that auto-detects the parallel execution
> environment (single CPU core, single-node multi-CPU, single-node
> multi-GPU, multi-node MPI, multi-node DDP) and dispatches the
> work to the right backend. The research run's LUMI jobs use it
> to abstract the Slurm allocation.

The file is a single module at the project root:
`scale_runner.py`.

## Usage

### Programmatic

```python
from scale_runner import detect_environment, get_launcher

env = detect_environment()
launcher = get_launcher(env)
launcher.print_info()
results = launcher.map(work_fn, items)  # parallel map
```

### CLI

```bash
# Print detected environment and exit
python -m scale_runner info

# Run a worker function on a list of items
python -m scale_runner run --work-fn experiments.mc_ground_truth:run_chunk --items 1 2 3 4
```

## `detect_environment() → dict`

Inspects the Slurm environment variables (`SLURM_NNODES`,
`SLURM_NTASKS`, `SLURM_NTASKS_PER_NODE`, `SLURM_CPUS_PER_TASK`,
`SLURM_CPUS_ON_NODE`, `SLURM_GPUS_PER_NODE`, `SLURM_PROCID`,
`SLURM_LOCALID`, `SLURM_MEM_PER_NODE`) and returns a dict with
keys:

| Key | Description |
|---|---|
| `nnodes` | Number of allocated nodes |
| `ntasks` | Total MPI ranks |
| `ntasks_per_node` | Ranks per node |
| `ncpus_per_task` | CPUs per rank |
| `cpus_per_node` | Total CPUs allocated per node |
| `gpus_per_node` | GPUs per node |
| `memory_per_node_mb` | Memory per node in MB |
| `procid` | Global rank (0-indexed) |
| `localid` | Local rank within node (0-indexed) |
| `launcher_type` | `sequential` / `joblib_local` / `joblib_srun` / `torch_ddp` |
| `has_gpu` | Whether any GPU is available |

### Launcher selection logic

```python
if ntasks > 1 and nnodes > 1:
    launcher_type = "joblib_srun"
elif ntasks > 1 and has_gpu:
    launcher_type = "torch_ddp"
elif ntasks > 1 or cpus_per_node > 4:
    launcher_type = "joblib_local"
else:
    launcher_type = "sequential"
```

## `Launcher` (abstract base)

```python
class Launcher(ABC):
    env: dict

    @abstractmethod
    def map(self, work_fn, items, **kwargs) -> list: ...

    @abstractmethod
    def get_rank(self) -> int: ...

    @abstractmethod
    def get_local_rank(self) -> int: ...

    @abstractmethod
    def get_world_size(self) -> int: ...

    def is_main(self) -> bool:
        return self.get_rank() == 0

    def print_info(self) -> None: ...
```

`work_fn` must be picklable for process-based dispatch
(`joblib_local`, `joblib_srun`, `torch_ddp`).

`print_info()` prints a one-shot summary of the detected
environment to stdout.

## `SequentialLauncher`

Single-process launcher. No parallelism.

```python
def map(self, work_fn, items, **kwargs) -> list:
    return [work_fn(item, **kwargs) for item in items]
```

`get_rank()`, `get_local_rank()` return 0; `get_world_size()`
returns 1.

## `JoblibLauncher`

Multi-process launcher using `joblib.Parallel` with the
`loky` backend. Falls back to `ProcessPoolExecutor` if
`joblib` is unavailable.

`n_jobs = min(cpus_per_node, len(items))`. If `n_jobs == 1`
(only one CPU), runs sequentially without spawning.

```python
def map(self, work_fn, items, **kwargs) -> list:
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
    # Fallback to ProcessPoolExecutor
    ...
```

`get_rank()`, `get_local_rank()` return `env["procid"]`,
`env["localid"]`; `get_world_size()` returns `env["ntasks"]`.

## `SrunJoblibLauncher`

Multi-node launcher. Tries to use `joblib` with a
`dask_jobqueue.SLURMCluster` backend. If `dask_jobqueue` is
not installed, falls back to the local `JoblibLauncher.map`
on the head node only and warns (the non-head nodes return
empty lists).

This is the launcher used by the LUMI multi-node CPU jobs
(`cpu_weak_2`, `cpu_weak_4`, `cpu_weak_8`).

## `TorchDDLauncher`

Multi-GPU / multi-node DDP launcher. Each rank runs
`work_fn` on a subset of items; results are gathered on
rank 0.

```python
def map(self, work_fn, items, **kwargs) -> list:
    self._init_distributed()  # init torch.distributed if not done
    items = list(items)
    rank = self.get_rank()
    world = self.get_world_size()
    my_items = items[rank::world]
    my_results = [work_fn(item, **kwargs) for item in my_items]
    if world == 1 or rank != 0:
        return my_results
    # Gather on rank 0 via torch.distributed.all_gather_object
    ...
```

### Backend selection

- `backend = "nccl"` if `has_gpu`, else `"gloo"`.
- `MASTER_ADDR` / `MASTER_PORT` are read from env if set
  (Slurm sets these for multi-node jobs).

If `torch.distributed` is unavailable or `init_process_group`
raises, the launcher treats itself as "initialized" and
falls back to running only on the current process (warning
printed to stderr).

This is the launcher used by the LUMI multi-GPU jobs
(`gpu_strong_4`, `gpu_multi_2`, `gpu_multi_4`).

## `get_launcher(env=None) → Launcher`

Factory. If `env is None`, calls `detect_environment()`.
Returns the concrete launcher instance for the detected
environment.

```python
_LAUNCHERS = {
    "sequential": SequentialLauncher,
    "joblib_local": JoblibLauncher,
    "joblib_srun": SrunJoblibLauncher,
    "torch_ddp": TorchDDLauncher,
}
```

## CLI

### `python -m scale_runner info`

Prints `Launcher.print_info()` output and exits with status 0.

### `python -m scale_runner run --work-fn MODULE:FN --items 1 2 3 4`

Imports the worker function from `MODULE` and looks up `FN`.
For example:

```bash
python -m scale_runner run \
    --work-fn experiments.mc_ground_truth:run_chunk \
    --items 1 2 3 4
```

Loops over the items, calling the launcher with the worker
function and the items. Prints a summary on rank 0
(`Launcher.is_main()`).

## Notes on robustness

- All launchers fail gracefully: if `joblib` is missing, use
  `ProcessPoolExecutor`; if `dask_jobqueue` is missing,
  use local joblib; if `torch.distributed` is missing,
  use sequential.
- The `SrunJoblibLauncher` and `TorchDDLauncher` both warn
  on fallback but do not abort, so the same code runs on
  a developer laptop and on a LUMI allocation.
- The `SequentialLauncher` is a deliberate zero-dependency
  fallback so that the import `from scale_runner import
  get_launcher` never fails.
