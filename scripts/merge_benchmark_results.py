from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.benchmark_results import extract_results_map, load_results_payload, save_results_json


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge multiple benchmark result JSON files by definition/workload UUID")
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--source", default="scripts/merge_benchmark_results.py")
    return parser.parse_args(argv)


def merge_results_payloads(input_paths: list[Path]) -> dict[str, Any]:
    merged: dict[str, dict[str, Any]] = {}
    metadata_inputs: list[str] = []
    trace_set_path: str | None = None
    for path in input_paths:
        payload = load_results_payload(path)
        metadata_inputs.append(str(path))
        metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
        if trace_set_path is None and isinstance(metadata, dict):
            candidate = metadata.get("trace_set_path")
            if isinstance(candidate, str) and candidate:
                trace_set_path = candidate
        results = extract_results_map(payload)
        for definition, workloads in results.items():
            definition_map = merged.setdefault(definition, {})
            for uuid, entry in workloads.items():
                definition_map[uuid] = entry
    return {
        "metadata": {
            "inputs": metadata_inputs,
            "trace_set_path": trace_set_path,
        },
        "results": merged,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = merge_results_payloads(args.inputs)
    save_results_json(
        args.output_json,
        payload["results"],
        source=args.source,
        trace_set_path=payload["metadata"].get("trace_set_path"),
        extra_metadata={"inputs": payload["metadata"]["inputs"]},
    )
    print(json.dumps({"output_json": str(args.output_json), "inputs": payload["metadata"]["inputs"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
