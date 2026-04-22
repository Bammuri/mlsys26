"""
Evaluate Modal benchmark result pairs as a surrogate for bare-like structural policy tracking.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from statistics import mean
from typing import Any

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.benchmark_results import SUCCESS_STATUSES, extract_results_map, load_results_payload

DEFAULT_RELATIVE_TOLERANCE = 0.05
DEFAULT_ABSOLUTE_TOLERANCE_MS = 0.005
REQUIRED_SURFACE_ROLES = {"quick_broad", "official_like", "targeted_tail"}


def _load_definition_results(path: Path) -> tuple[str, dict[str, dict[str, Any]]]:
    payload = load_results_payload(path)
    results = extract_results_map(payload)
    if not isinstance(results, dict) or not results:
        raise ValueError(f"Expected result payload with at least one definition: {path}")
    definition_name, traces = next(iter(results.items()))
    if not isinstance(traces, dict):
        raise ValueError(f"Expected workload mapping for definition '{definition_name}'")
    return definition_name, traces


def _compare_latency_entry(
    baseline_entry: dict[str, Any] | None,
    candidate_entry: dict[str, Any] | None,
    *,
    relative_tolerance: float,
    absolute_tolerance_ms: float,
) -> dict[str, Any]:
    baseline_status = baseline_entry.get("status") if baseline_entry else "MISSING"
    candidate_status = candidate_entry.get("status") if candidate_entry else "MISSING"
    baseline_latency = baseline_entry.get("latency_ms") if baseline_entry else None
    candidate_latency = candidate_entry.get("latency_ms") if candidate_entry else None

    output = {
        "baseline_status": baseline_status,
        "candidate_status": candidate_status,
        "baseline_latency_ms": baseline_latency,
        "candidate_latency_ms": candidate_latency,
        "delta_ms": None,
        "delta_pct": None,
        "verdict": "unknown",
    }

    if (
        baseline_status not in SUCCESS_STATUSES
        or candidate_status not in SUCCESS_STATUSES
        or not isinstance(baseline_latency, (int, float))
        or not isinstance(candidate_latency, (int, float))
    ):
        return output

    baseline_latency = float(baseline_latency)
    candidate_latency = float(candidate_latency)
    delta_ms = candidate_latency - baseline_latency
    delta_pct = delta_ms / baseline_latency if baseline_latency != 0 else math.inf
    output["delta_ms"] = delta_ms
    output["delta_pct"] = delta_pct

    if abs(delta_ms) < absolute_tolerance_ms or abs(delta_pct) < relative_tolerance:
        output["verdict"] = "neutral"
    elif delta_ms < 0:
        output["verdict"] = "improved"
    else:
        output["verdict"] = "regressed"
    return output


def _lane_verdict(
    lane_name: str,
    rows: list[dict[str, Any]],
) -> tuple[str, str]:
    counts = {"improved": 0, "neutral": 0, "regressed": 0, "unknown": 0}
    for row in rows:
        counts[row["verdict"]] = counts.get(row["verdict"], 0) + 1

    known_rows = [row for row in rows if row["verdict"] != "unknown"]
    mean_delta_ms = mean([row["delta_ms"] for row in known_rows]) if known_rows else None

    if lane_name == "fast_floor":
        if counts["regressed"] > 0 and counts["regressed"] >= counts["improved"]:
            return "reject", "fast-floor regression suggests structural overhead on already efficient workloads"
        if counts["improved"] > 0 and counts["regressed"] == 0:
            return "pass", "fast-floor remained healthy or improved"
        return "hold", "fast-floor signal is mixed or insufficient"

    if lane_name == "tail":
        if counts["improved"] > counts["regressed"] and mean_delta_ms is not None and mean_delta_ms < 0:
            return "pass", "tail lane improved, consistent with selective underfilled-regime benefit"
        if counts["regressed"] >= counts["improved"] and counts["regressed"] > 0:
            return "reject", "tail lane does not show the intended benefit"
        return "hold", "tail signal is mixed or insufficient"

    if lane_name == "mid_transition":
        if counts["regressed"] > counts["improved"]:
            return "reject", "mid-transition regressions indicate broader instability"
        if counts["improved"] > counts["regressed"]:
            return "pass", "mid-transition lane improved"
        return "hold", "mid-transition lane is mixed or inconclusive"

    return "hold", "unknown lane"


def summarize_lane(
    lane_name: str,
    uuids: list[str],
    baseline_traces: dict[str, dict[str, Any]],
    candidate_traces: dict[str, dict[str, Any]],
    *,
    relative_tolerance: float,
    absolute_tolerance_ms: float,
) -> dict[str, Any]:
    rows = []
    for uuid in uuids:
        rows.append(
            {
                "uuid": uuid,
                **_compare_latency_entry(
                    baseline_traces.get(uuid),
                    candidate_traces.get(uuid),
                    relative_tolerance=relative_tolerance,
                    absolute_tolerance_ms=absolute_tolerance_ms,
                ),
            }
        )
    verdict, rationale = _lane_verdict(lane_name, rows)
    known = [row for row in rows if row["verdict"] != "unknown"]
    return {
        "lane": lane_name,
        "verdict": verdict,
        "rationale": rationale,
        "counts": {
            label: sum(1 for row in rows if row["verdict"] == label)
            for label in ("improved", "neutral", "regressed", "unknown")
        },
        "mean_delta_ms": mean([row["delta_ms"] for row in known]) if known else None,
        "mean_delta_pct": mean([row["delta_pct"] for row in known]) if known else None,
        "rows": rows,
    }


def _surface_gate(lane_summaries: dict[str, dict[str, Any]]) -> tuple[str, str]:
    fast_verdict = lane_summaries["fast_floor"]["verdict"]
    mid_verdict = lane_summaries["mid_transition"]["verdict"]
    tail_verdict = lane_summaries["tail"]["verdict"]

    if fast_verdict == "reject" or mid_verdict == "reject":
        return "reject", "fast-floor or mid-transition lane regressed structurally"
    if tail_verdict == "pass":
        return "pursue_selective", "tail lane improved without non-tail rejection"
    return "hold", "surface lacks sufficient selective upside"


def evaluate_surface(
    *,
    surface_label: str,
    surface_role: str,
    baseline_json: Path,
    candidate_json: Path,
    canonical_lanes: dict[str, Any],
    relative_tolerance: float,
    absolute_tolerance_ms: float,
) -> dict[str, Any]:
    baseline_definition, baseline_traces = _load_definition_results(baseline_json)
    candidate_definition, candidate_traces = _load_definition_results(candidate_json)
    if baseline_definition != candidate_definition:
        raise ValueError(
            f"Definition mismatch for surface '{surface_label}': "
            f"{baseline_definition} vs {candidate_definition}"
        )

    lane_summaries = {
        lane_name: summarize_lane(
            lane_name,
            [entry["uuid"] for entry in entries],
            baseline_traces,
            candidate_traces,
            relative_tolerance=relative_tolerance,
            absolute_tolerance_ms=absolute_tolerance_ms,
        )
        for lane_name, entries in canonical_lanes["lanes"].items()
    }
    surface_gate, rationale = _surface_gate(lane_summaries)
    return {
        "label": surface_label,
        "role": surface_role,
        "definition": baseline_definition,
        "baseline_json": str(baseline_json),
        "candidate_json": str(candidate_json),
        "lane_summaries": lane_summaries,
        "surface_gate": surface_gate,
        "rationale": rationale,
    }


def evaluate_surrogate_policy(
    *,
    canonical_lanes: dict[str, Any],
    surfaces: list[dict[str, Any]],
    relative_tolerance: float,
    absolute_tolerance_ms: float,
) -> dict[str, Any]:
    evaluated_surfaces = [
        evaluate_surface(
            surface_label=surface["label"],
            surface_role=surface["role"],
            baseline_json=Path(surface["baseline_json"]),
            candidate_json=Path(surface["candidate_json"]),
            canonical_lanes=canonical_lanes,
            relative_tolerance=relative_tolerance,
            absolute_tolerance_ms=absolute_tolerance_ms,
        )
        for surface in surfaces
    ]
    surfaces_by_role = {surface["role"]: surface for surface in evaluated_surfaces}
    missing_roles = sorted(REQUIRED_SURFACE_ROLES - set(surfaces_by_role))

    if missing_roles:
        recommendation = "hold"
        rationale = f"missing required surfaces: {missing_roles}"
    elif any(surface["surface_gate"] == "reject" for surface in evaluated_surfaces):
        recommendation = "reject"
        rationale = "at least one required surface rejected the candidate"
    elif surfaces_by_role["targeted_tail"]["lane_summaries"]["tail"]["verdict"] == "pass":
        recommendation = "pursue_selective"
        rationale = "targeted tail surface improved without broad surface rejection"
    else:
        recommendation = "hold"
        rationale = "candidate did not clear the selective-tail gate"

    return {
        "metadata": {
            "canonical_definition": canonical_lanes["metadata"]["definition"],
            "canonical_source_cohort_json": canonical_lanes["metadata"].get("source_cohort_json"),
            "relative_tolerance": relative_tolerance,
            "absolute_tolerance_ms": absolute_tolerance_ms,
        },
        "surfaces": evaluated_surfaces,
        "aggregate": {
            "recommendation": recommendation,
            "rationale": rationale,
            "missing_surface_roles": missing_roles,
        },
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        f"# Modal surrogate scorecard: {payload['metadata']['canonical_definition']}",
        "",
        "## Aggregate recommendation",
        f"- recommendation: {payload['aggregate']['recommendation']}",
        f"- rationale: {payload['aggregate']['rationale']}",
    ]
    missing_roles = payload["aggregate"].get("missing_surface_roles", [])
    if missing_roles:
        lines.append(f"- missing_surface_roles: {missing_roles}")
    lines.append("")
    for surface in payload["surfaces"]:
        lines.extend(
            [
                f"## Surface: {surface['label']} ({surface['role']})",
                f"- surface_gate: {surface['surface_gate']}",
                f"- rationale: {surface['rationale']}",
            ]
        )
        for lane_name, lane_summary in surface["lane_summaries"].items():
            lines.append(
                f"- {lane_name}: verdict={lane_summary['verdict']} counts={lane_summary['counts']} mean_delta_ms={lane_summary['mean_delta_ms']}"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def load_surface_manifest(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    surfaces = payload.get("surfaces")
    if not isinstance(surfaces, list) or not surfaces:
        raise ValueError("Surface manifest must contain a non-empty 'surfaces' list")
    normalized = []
    for surface in surfaces:
        for key in ("label", "role", "baseline_json", "candidate_json"):
            if key not in surface:
                raise ValueError(f"Surface entry missing required key '{key}'")
        normalized.append(surface)
    return normalized


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a Modal surrogate scorecard from saved result JSONs")
    parser.add_argument("canonical_lane_json", type=Path)
    parser.add_argument("--surface-manifest-json", type=Path, required=True)
    parser.add_argument("--relative-tolerance", type=float, default=DEFAULT_RELATIVE_TOLERANCE)
    parser.add_argument("--absolute-tolerance-ms", type=float, default=DEFAULT_ABSOLUTE_TOLERANCE_MS)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-markdown", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    canonical_lanes = json.loads(args.canonical_lane_json.read_text())
    surfaces = load_surface_manifest(args.surface_manifest_json)
    payload = evaluate_surrogate_policy(
        canonical_lanes=canonical_lanes,
        surfaces=surfaces,
        relative_tolerance=args.relative_tolerance,
        absolute_tolerance_ms=args.absolute_tolerance_ms,
    )
    markdown = render_markdown(payload)
    print(markdown, end="")
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if args.output_markdown is not None:
        args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
        args.output_markdown.write_text(markdown)


if __name__ == "__main__":
    main()
