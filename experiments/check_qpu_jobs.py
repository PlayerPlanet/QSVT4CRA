"""Status checker for recently submitted Boston QPU jobs.

Use this to poll for completion of jobs that were submitted by
``experiments.boston_qpu.py`` (or ``experiments/heron_simulation.py``) and
have been sitting in the IBM Quantum queue.

Usage
-----

List recent jobs (default 10):

    python -m experiments.check_qpu_jobs

List recent jobs on a specific backend, with N entries:

    python -m experiments.check_qpu_jobs --backend ibm_boston --limit 20

Fetch and persist the result of a specific completed job, by ID:

    python -m experiments.check_qpu_jobs --fetch <job_id> \
        --output results/boston_qpu_k17_d8_result.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from compiler_backend.heron import DEFAULT_TOKEN_FILE, load_ibm_service


def list_recent_jobs(backend_name: str | None, limit: int) -> int:
    service = load_ibm_service()
    try:
        jobs = service.jobs(limit=limit, descending=True)
    except Exception as exc:
        print(f"jobs() query failed: {exc}", file=sys.stderr)
        return 1

    print(f"Recent jobs (limit={limit}):")
    print(f"  {'JOB_ID':<24} {'STATUS':<14} {'BACKEND':<20} {'NAME'}")
    print(f"  {'-'*24} {'-'*14} {'-'*20} {'-'*32}")
    shown = 0
    for j in jobs:
        try:
            status = j.status()
        except Exception as exc:
            status = f"ERR: {exc}"
        try:
            backend_obj = getattr(j, "backend", lambda: None)()
            if backend_obj is None:
                bname = "?"
            else:
                nm = getattr(backend_obj, "name", None)
                bname = nm() if callable(nm) else str(nm)
        except Exception:
            bname = "?"
        if backend_name and bname != backend_name:
            continue
        try:
            name = str(getattr(j, "name", lambda: "?")())[:32]
        except Exception:
            name = "?"
        try:
            jid = j.job_id()
        except Exception:
            jid = "?"
        print(f"  {jid:<24} {str(status):<14} {bname:<20} {name}")
        shown += 1
    if shown == 0:
        print(f"  (no jobs match backend={backend_name!r})")
    return 0


def fetch_and_persist(job_id: str, output: str) -> int:
    service = load_ibm_service()
    try:
        job = service.job(job_id)
    except Exception as exc:
        print(f"could not look up job {job_id}: {exc}", file=sys.stderr)
        return 2
    status = job.status()
    print(f"job {job_id} status: {status}")
    if str(status) not in {"DONE", "COMPLETED"}:
        print("Job is not done yet.  No result to fetch.", file=sys.stderr)
        return 3

    result = job.result()
    summary: dict[str, Any] = {
        "job_id": job_id,
        "status": str(status),
        "fetched_at": time.time(),
    }
    try:
        summary["backend"] = job.backend().name
    except Exception:
        pass
    try:
        pubs = job.inputs.get("pubs", [])
        summary["n_pubs"] = len(pubs)
    except Exception:
        summary["n_pubs"] = None

    pub_summaries = []
    for i, pub_result in enumerate(result):
        try:
            data = pub_result.data
            creg_names = list(getattr(data, "creg_names", []) or [])
            if creg_names:
                ba = getattr(data, creg_names[-1])
            else:
                ba = next(iter(getattr(data, "_data", {}).values()))
            int_counts = ba.get_int_counts()
            counts = {"0": 0, "1": 0}
            for k, v in int_counts.items():
                key = "1" if int(k) & 1 else "0"
                counts[key] += int(v)
            total = sum(v for v in counts.values())
            p1 = counts.get("1", 0) / total if total else 0.0
            pub_summaries.append(
                {"pub": i, "total_shots": total, "counts": counts, "p_bit_eq_1": p1}
            )
        except Exception as exc:
            pub_summaries.append({"pub": i, "error": str(exc)})
    summary["pubs"] = pub_summaries

    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Summary written to {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check status of, or fetch results from, recent IBM Quantum jobs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--backend", default=None, help="Filter list to this backend (e.g. ibm_boston).")
    parser.add_argument("--limit", type=int, default=10, help="Number of recent jobs to list.")
    parser.add_argument("--fetch", default=None, help="Fetch a specific completed job by ID and persist summary.")
    parser.add_argument("--output", default="results/qpu_job_summary.json")
    parser.add_argument(
        "--token-file",
        default=__import__("os").environ.get("IBM_API_KEY_FILE", DEFAULT_TOKEN_FILE),
        help="Path to a local file containing the IBM Quantum token.",
    )
    args = parser.parse_args(argv)

    if args.fetch:
        return fetch_and_persist(args.fetch, args.output)
    return list_recent_jobs(args.backend, args.limit)


if __name__ == "__main__":
    raise SystemExit(main())
