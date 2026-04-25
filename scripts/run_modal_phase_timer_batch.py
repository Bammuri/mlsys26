from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_modal_phase_timer import (
    PACKED_SUBMIT_SOLUTION_SOURCE,
    TRACE_SET_BASELINE_SOLUTION_SOURCE,
    _load_solution_payload,
    app,
    time_prefill_workload,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch-run direct phase timing for selected prefill workloads on Modal B200.")
    parser.add_argument("--trace-set-path", type=Path, default=None)
    parser.add_argument("--definition", default="gdn_prefill_qk4_v8_d128_k_last")
    parser.add_argument(
        "--solution-source",
        choices=(PACKED_SUBMIT_SOLUTION_SOURCE, TRACE_SET_BASELINE_SOLUTION_SOURCE),
        default=PACKED_SUBMIT_SOLUTION_SOURCE,
    )
    parser.add_argument("--baseline-solution-index", type=int, default=0)
    parser.add_argument("--workload-uuids", default="", help="Comma-separated workload UUIDs.")
    parser.add_argument("--workload-uuids-file", type=Path, default=None, help="Optional file with one UUID per line.")
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--warmup-iterations", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest-json", type=Path, default=None)
    parser.add_argument("--label", default="phase-timer")
    return parser.parse_args(argv)


def _chunked(values: list[str], size: int) -> list[list[str]]:
    return [values[idx : idx + size] for idx in range(0, len(values), max(1, size))]


def _load_workload_uuids(args: argparse.Namespace) -> list[str]:
    uuids = [uuid.strip() for uuid in args.workload_uuids.split(",") if uuid.strip()]
    if args.workload_uuids_file is not None:
        file_uuids = [
            line.strip()
            for line in args.workload_uuids_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        uuids.extend(file_uuids)
    deduped: list[str] = []
    for uuid in uuids:
        if uuid not in deduped:
            deduped.append(uuid)
    return deduped


def _collect_results(workload_uuids: list[str], *, batch_size: int, spawn_fn) -> list[tuple[str, dict[str, Any]]]:
    results: list[tuple[str, dict[str, Any]]] = []
    for batch in _chunked(workload_uuids, batch_size):
        calls = [(uuid, spawn_fn(uuid)) for uuid in batch]
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(calls)) as executor:
            futures = {executor.submit(call.get): uuid for uuid, call in calls}
            for future in concurrent.futures.as_completed(futures):
                uuid = futures[future]
                results.append((uuid, future.result()))
    return results


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    workload_uuids = _load_workload_uuids(args)
    if not workload_uuids:
        raise SystemExit("No workloads selected")

    solution_payload, solution_provenance = _load_solution_payload(
        definition=args.definition,
        solution_source=args.solution_source,
        trace_set_path=args.trace_set_path,
        baseline_solution_index=args.baseline_solution_index,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    with app.run():
        results = _collect_results(
            workload_uuids,
            batch_size=args.batch_size,
            spawn_fn=lambda uuid: time_prefill_workload.spawn(
                solution_payload,
                args.definition,
                uuid,
                args.iterations,
                args.warmup_iterations,
            ),
        )

    for uuid, result in sorted(results, key=lambda item: item[0]):
        output_path = args.output_dir / f"{uuid}.json"
        output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        rows.append(
            {
                "uuid": uuid,
                "label": args.label,
                "result_path": output_path.name,
                "solution_name": result.get("solution", {}).get("name"),
                "entry_point": result.get("solution", {}).get("entry_point"),
                "destination_passing_style": result.get("solution", {}).get("destination_passing_style"),
            }
        )
        print(f"[ok] {uuid}", flush=True)

    manifest_path = args.manifest_json or args.output_dir / "manifest.json"
    manifest = {
        "metadata": {
            "definition": args.definition,
            "solution_source": args.solution_source,
            "solution_provenance": solution_provenance,
            "iterations": args.iterations,
            "warmup_iterations": args.warmup_iterations,
            "batch_size": args.batch_size,
            "workload_count": len(workload_uuids),
            "label": args.label,
        },
        "results": rows,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"saved_manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
