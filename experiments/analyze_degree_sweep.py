"""Analyse the degree-sweep QPU results and produce a trade-off table.

Reads ``results/degree_sweep_ids.json`` (written by the submit helper),
fetches each job's result, applies readout mitigation, compares to the
classical reference, and writes a summary JSON.

Usage
-----

    python -m experiments.analyze_degree_sweep \
        --output results/degree_sweep_summary.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

import experiments.boston_qpu as bq
from compiler_backend.heron import DEFAULT_TOKEN_FILE, load_ibm_service


def fetch_pubs(service, job_id: str) -> list[dict[str, int]]:
    job = service.job(job_id)
    status = str(job.status())
    if status not in {"DONE", "COMPLETED"}:
        return None
    result = job.result()
    out = []
    for i, pub_result in enumerate(result):
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
        out.append(counts)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fetch degree-sweep QPU results and build a trade-off table.",
    )
    parser.add_argument(
        "--ids-file",
        default="results/degree_sweep_ids.json",
        help="Path to the JSON with {degree_str: {job_id, aux_phys}}.",
    )
    parser.add_argument(
        "--output",
        default="results/degree_sweep_summary.json",
    )
    parser.add_argument(
        "--n-classical-scenarios",
        type=int,
        default=200_000,
    )
    args = parser.parse_args(argv)

    if not Path(args.ids_file).is_file():
        print(f"ids file not found: {args.ids_file}", flush=True)
        return 1

    ids = json.load(open(args.ids_file))
    service = load_ibm_service()

    rows: list[dict[str, Any]] = []
    for deg_str, info in sorted(ids.items(), key=lambda kv: int(kv[0])):
        degree = int(deg_str)
        job_id = info["job_id"]
        aux_phys = info.get("aux_phys")
        print(f"\n[degree={degree}] job_id={job_id}  aux_phys={aux_phys}", flush=True)

        pubs = fetch_pubs(service, job_id)
        if pubs is None:
            try:
                status = service.job(job_id).status()
            except Exception:
                status = "unknown"
            print(f"  not done yet (status={status}); skipping", flush=True)
            continue
        if len(pubs) != 3:
            print(f"  unexpected pub count {len(pubs)}; skipping", flush=True)
            continue

        counts_raw = pubs[0]
        c0 = pubs[1]
        c1 = pubs[2]
        cal_total_0 = c0["0"] + c0["1"]
        cal_total_1 = c1["0"] + c1["1"]
        if cal_total_0 == 0 or cal_total_1 == 0:
            print("  empty calibration; skipping", flush=True)
            continue

        e10 = c0["1"] / cal_total_0
        e01 = c1["0"] / cal_total_1
        confusion = np.array([[1 - e10, e01], [e10, 1 - e01]])
        counts_mit = bq._mitigate_single_qubit(counts_raw, confusion)
        total_raw = counts_raw["0"] + counts_raw["1"]
        total_mit = counts_mit["0"] + counts_mit["1"]
        p1_raw = counts_raw["1"] / total_raw
        p1_mit = counts_mit["1"] / total_mit

        # Noiseless Aer ground truth for this degree (run quickly)
        try:
            circuit, meta = bq.build_finnish_mortgage_qsvt_circuit(
                degree=degree, target_loss_fraction=0.5
            )
            from compiler_backend.heron import add_measurements as _am
            from qiskit_aer import AerSimulator
            decomposed = circuit.decompose(reps=10)
            aux_idx = int(meta["aux_qubit_index"])
            measured = _am(decomposed, qubits=[aux_idx])
            sim = AerSimulator()
            r = sim.run(measured, shots=2048).result()
            cnts = r.get_counts()
            aer_n1 = sum(c for k, c in cnts.items() if k.replace(" ", "")[-1] == "1")
            aer_total = sum(cnts.values())
            p_aer = aer_n1 / aer_total
        except Exception as exc:
            print(f"  Aer reference failed: {exc}", flush=True)
            p_aer = None

        # Classical reference (uses 1 build for the largest degree, reused)
        cl = bq.classical_reference(meta, n_scenarios=args.n_classical_scenarios, seed=0)
        tail = cl["tail_at_target_loss"]

        abs_raw = abs(p1_raw - tail)
        rel_raw = abs_raw / tail if tail > 0 else float("nan")
        abs_mit = abs(p1_mit - tail)
        rel_mit = abs_mit / tail if tail > 0 else float("nan")

        rows.append(
            {
                "degree": degree,
                "job_id": job_id,
                "aux_physical_qubit": aux_phys,
                "readout_e10": float(e10),
                "readout_e01": float(e01),
                "counts_raw": counts_raw,
                "counts_mitigated": {"0": counts_mit["0"], "1": counts_mit["1"]},
                "p_aer_noiseless": p_aer,
                "p_qpu_raw": float(p1_raw),
                "p_qpu_mitigated": float(p1_mit),
                "classical_tail": float(tail),
                "abs_error_raw": float(abs_raw),
                "rel_error_raw": float(rel_raw),
                "abs_error_mitigated": float(abs_mit),
                "rel_error_mitigated": float(rel_mit),
            }
        )
        print(
            f"  raw P={p1_raw:.4f}  mit P={p1_mit:.4f}  "
            f"Aer={p_aer if p_aer is None else f'{p_aer:.4f}'}  "
            f"classical={tail:.6f}",
            flush=True,
        )

    summary = {
        "fetched_at": time.time(),
        "rows": rows,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nWrote {out}")

    if rows:
        print("\n=== Trade-off table ===")
        print(f"  {'deg':<4} {'Aer_id':<10} {'QPU_raw':<10} {'QPU_mit':<10} {'class':<10} {'rel_err_mit':<12}")
        for r in rows:
            print(
                f"  {r['degree']:<4} "
                f"{(r['p_aer_noiseless'] if r['p_aer_noiseless'] is not None else float('nan')):<10.4f} "
                f"{r['p_qpu_raw']:<10.4f} "
                f"{r['p_qpu_mitigated']:<10.4f} "
                f"{r['classical_tail']:<10.6f} "
                f"{r['rel_error_mitigated']:<12.0f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
