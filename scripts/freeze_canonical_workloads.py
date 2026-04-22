"""
Freeze canonical fast/mid/tail workload lanes from a cohort JSON artifact.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.benchmark_results import load_results_payload


def _cohort_results(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    results = payload.get("results", payload)
    if not isinstance(results, dict) or not results:
        raise ValueError("Expected a results dictionary in the cohort payload")
    definition_name, definition_payload = next(iter(results.items()))
    if not isinstance(definition_payload, dict):
        raise ValueError("Expected a definition payload dictionary")
    return definition_name, definition_payload


def _pick_entries(
    workload_rows: list[dict[str, Any]],
    cohort: str,
    *,
    count: int,
) -> list[dict[str, Any]]:
    cohort_rows = [row for row in workload_rows if row.get("cohort") == cohort]
    if cohort == "tail":
        cohort_rows = sorted(
            cohort_rows,
            key=lambda row: (-float(row["latency_ms"]), row["uuid"]),
        )
    else:
        cohort_rows = sorted(
            cohort_rows,
            key=lambda row: (float(row["latency_ms"]), row["uuid"]),
        )
    return cohort_rows[:count]


def build_canonical_lane_payload(
    payload: dict[str, Any],
    *,
    source_cohort_json: str | None,
    fast_count: int,
    mid_count: int,
    tail_count: int,
) -> dict[str, Any]:
    definition_name, definition_payload = _cohort_results(payload)
    workload_rows = definition_payload.get("workloads", [])
    thresholds = definition_payload.get("thresholds", {})

    lanes = {
        "fast_floor": _pick_entries(workload_rows, "fast", count=fast_count),
        "mid_transition": _pick_entries(workload_rows, "mid", count=mid_count),
        "tail": _pick_entries(workload_rows, "tail", count=tail_count),
    }

    return {
        "metadata": {
            "definition": definition_name,
            "source_cohort_json": source_cohort_json,
            "source_results_json": payload.get("metadata", {}).get("source_results_json"),
            "cohort_thresholds": thresholds,
        },
        "lanes": lanes,
    }


def render_markdown(lane_payload: dict[str, Any]) -> str:
    lines = [
        f"# Canonical workload lanes: {lane_payload['metadata']['definition']}",
        "",
        "## Source",
        f"- cohort json: {lane_payload['metadata'].get('source_cohort_json')}",
        f"- source results json: {lane_payload['metadata'].get('source_results_json')}",
        f"- thresholds: {lane_payload['metadata'].get('cohort_thresholds')}",
        "",
    ]
    for lane_name, entries in lane_payload["lanes"].items():
        lines.append(f"## {lane_name}")
        if not entries:
            lines.append("- (none)")
            lines.append("")
            continue
        for entry in entries:
            reason = f"axes={entry.get('axes')}" if entry.get("axes") else "no-axis-enrichment"
            lines.append(f"- {entry['uuid']}: {entry['latency_ms']:.4f} ms | {reason}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze canonical workload lanes from cohort JSON")
    parser.add_argument("cohort_json", type=Path, help="Path to a cohort JSON artifact")
    parser.add_argument("--fast-count", type=int, default=3)
    parser.add_argument("--mid-count", type=int, default=2)
    parser.add_argument("--tail-count", type=int, default=3)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-markdown", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = load_results_payload(args.cohort_json)
    lane_payload = build_canonical_lane_payload(
        payload,
        source_cohort_json=str(args.cohort_json),
        fast_count=args.fast_count,
        mid_count=args.mid_count,
        tail_count=args.tail_count,
    )
    markdown = render_markdown(lane_payload)
    print(markdown, end="")

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(lane_payload, indent=2, sort_keys=True) + "\n")
    if args.output_markdown is not None:
        args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
        args.output_markdown.write_text(markdown)


if __name__ == "__main__":
    main()
