from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SELECTOR_FIELDS = (
    "varlen",
    "batch_size",
    "max_seq_len",
    "total_seq_len",
    "h_q",
    "h_v",
    "head_dim",
)

HIGHER_IS_BETTER = {
    "issue_efficiency": True,
    "eligible_warps_per_scheduler": True,
    "issued_warps_per_scheduler": True,
    "achieved_occupancy_pct": True,
    "compute_utilization_pct": True,
    "memory_utilization_pct": True,
}

DEFAULT_GROUP_METRICS = {
    "scheduler_occupancy": (
        "issue_efficiency",
        "skipped_issue_slots",
        "eligible_warps_per_scheduler",
        "achieved_occupancy_pct",
    ),
    "memory_dependency": (
        "memory_utilization_pct",
        "compute_utilization_pct",
        "occupancy_gap_pct",
        "issue_efficiency",
    ),
    "mixed_or_unknown": (
        "issue_efficiency",
        "achieved_occupancy_pct",
        "compute_utilization_pct",
        "memory_utilization_pct",
    ),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build structural workload groups and per-group NCU gate tables")
    parser.add_argument("inventory_json", type=Path, help="Inventory JSON from scripts/build_workload_inventory.py")
    parser.add_argument(
        "--scorecard-json",
        type=Path,
        default=None,
        help="Optional NCU scorecard payload JSON from scripts/ncu_scorecard.py",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args(argv)


def _quantile_bounds(values: list[int | float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0], ordered[0]
    lower_idx = max(0, math.floor((len(ordered) - 1) * 0.33))
    upper_idx = max(lower_idx, math.floor((len(ordered) - 1) * 0.66))
    return ordered[lower_idx], ordered[upper_idx]


def _bucket_numeric(value: int | float | None, bounds: tuple[float | None, float | None]) -> str:
    lower, upper = bounds
    if value is None or lower is None or upper is None:
        return "unknown"
    numeric = float(value)
    if numeric <= lower:
        return "small"
    if numeric <= upper:
        return "medium"
    return "large"


def _row_structural_label(row: dict[str, Any], scorecard: dict[str, Any] | None) -> str:
    if scorecard is None:
        return "mixed_or_unknown"

    primary = scorecard.get("primary", {})
    scheduler = primary.get("scheduler_health", {}).get("metrics", {})
    occupancy = primary.get("occupancy_effectiveness", {}).get("metrics", {})
    bound = primary.get("bound_classification", {})

    issue_efficiency = scheduler.get("issue_efficiency")
    skipped_issue_slots = scheduler.get("skipped_issue_slots")
    eligible = scheduler.get("eligible_warps_per_scheduler")
    achieved_occupancy = occupancy.get("achieved_occupancy_pct")
    classification = bound.get("classification")

    if classification == "memory_bound":
        return "memory_dependency"

    if any(
        [
            isinstance(issue_efficiency, (int, float)) and issue_efficiency < 35.0,
            isinstance(skipped_issue_slots, (int, float)) and skipped_issue_slots > 55.0,
            isinstance(eligible, (int, float)) and eligible < 0.5,
            isinstance(achieved_occupancy, (int, float)) and achieved_occupancy < 25.0,
        ]
    ):
        return "scheduler_occupancy"

    return "mixed_or_unknown"


def _extract_metric_value(scorecard: dict[str, Any], metric_name: str) -> float | None:
    primary = scorecard.get("primary", {})
    for section in primary.values():
        metrics = section.get("metrics", {})
        value = metrics.get(metric_name)
        if isinstance(value, (int, float)):
            return float(value)
    if metric_name == "classification":
        classification = primary.get("bound_classification", {}).get("classification")
        return None if classification is None else None
    return None


def _build_threshold_entry(metric_name: str, values: list[float]) -> dict[str, Any]:
    median = statistics.median(values)
    minimum = min(values)
    maximum = max(values)
    higher_is_better = HIGHER_IS_BETTER.get(metric_name, False)
    if higher_is_better:
        pass_cutoff = round(median * 0.95, 4)
        hold_cutoff = round(median * 0.85, 4)
        reject_cutoff = round(median * 0.70, 4)
    else:
        pass_cutoff = round(median * 1.05, 4)
        hold_cutoff = round(median * 1.20, 4)
        reject_cutoff = round(median * 1.35, 4)
    return {
        "metric": metric_name,
        "baseline_min": round(minimum, 4),
        "baseline_max": round(maximum, 4),
        "baseline_median": round(median, 4),
        "higher_is_better": higher_is_better,
        "pass_cutoff": pass_cutoff,
        "hold_cutoff": hold_cutoff,
        "reject_cutoff": reject_cutoff,
        "confidence": "provisional-threshold",
    }


def _gate_rule(structural_label: str) -> dict[str, Any]:
    if structural_label == "scheduler_occupancy":
        return {
            "pass_rule": "promote when scheduler/occupancy metrics meet pass cutoffs and no severe fast-floor regression is present",
            "hold_rule": "hold when metrics are mixed, provisional, or between hold and pass cutoffs",
            "reject_rule": "reject when scheduler/occupancy metrics cross reject cutoffs or policy-risky regressions appear",
        }
    if structural_label == "memory_dependency":
        return {
            "pass_rule": "promote when memory-vs-compute balance improves without collapsing scheduler health",
            "hold_rule": "hold when memory signals improve but scheduler/occupancy evidence remains mixed",
            "reject_rule": "reject when memory-focused changes worsen scheduler health or widen occupancy imbalance past reject cutoffs",
        }
    return {
        "pass_rule": "no direct pass until the subgroup is stable and non-provisional",
        "hold_rule": "default hold while collecting more evidence",
        "reject_rule": "reject if exploratory changes create severe regressions in any primary metric",
    }


def _build_group_thresholds(group_rows: list[dict[str, Any]], scorecards: dict[str, dict[str, Any]], structural_label: str) -> list[dict[str, Any]]:
    metrics = DEFAULT_GROUP_METRICS[structural_label]
    tables: list[dict[str, Any]] = []
    for metric_name in metrics:
        values: list[float] = []
        for row in group_rows:
            scorecard = scorecards.get(row["uuid"])
            if scorecard is None:
                continue
            value = _extract_metric_value(scorecard, metric_name)
            if isinstance(value, float):
                values.append(value)
        if values:
            tables.append(_build_threshold_entry(metric_name, values))
        else:
            tables.append(
                {
                    "metric": metric_name,
                    "baseline_min": None,
                    "baseline_max": None,
                    "baseline_median": None,
                    "higher_is_better": HIGHER_IS_BETTER.get(metric_name, False),
                    "pass_cutoff": None,
                    "hold_cutoff": None,
                    "reject_cutoff": None,
                    "confidence": "provisional-threshold",
                }
            )
    return tables


def _optimization_family(structural_label: str) -> dict[str, Any]:
    if structural_label == "scheduler_occupancy":
        return {
            "optimization_family": "scheduler_latency_hiding",
            "allowed_runtime_action": "tune persistent branch thresholds or launch/scheduling branch policy",
            "rationale": "This group shows low issue efficiency / low eligible warps / low achieved occupancy, so the first lever is latency-hiding and scheduling behavior.",
            "expected_ncu_movement": "higher issue efficiency, fewer skipped issue slots, higher achieved occupancy",
            "risk_tradeoff": "Can improve tails while regressing fast-floor workloads if compile variants or launch overhead grow.",
        }
    if structural_label == "memory_dependency":
        return {
            "optimization_family": "memory_locality_and_dependency_reduction",
            "allowed_runtime_action": "keep one-kernel branch conservative; prioritize locality/coalescing-oriented tuning over aggressive branch expansion",
            "rationale": "This group looks memory-bound or dependency-bound, so scheduler-only changes are unlikely to be sufficient.",
            "expected_ncu_movement": "better memory-vs-compute balance, lower occupancy gap without worsening scheduler health",
            "risk_tradeoff": "May need deeper kernel work that can threaten one-kernel simplicity if the structural signature is heterogeneous.",
        }
    return {
        "optimization_family": "hold_and_collect_more_evidence",
        "allowed_runtime_action": "no new runtime branch until subgroup evidence stabilizes",
        "rationale": "This group does not yet show a stable single bottleneck class across enough representative workloads.",
        "expected_ncu_movement": "stabilized bottleneck labeling before branch design",
        "risk_tradeoff": "Moving too early risks encoding noise or current-policy artifacts into the selector.",
    }


def _threshold_provenance(scorecard_payload: dict[str, Any] | None) -> dict[str, Any]:
    metadata = scorecard_payload.get("metadata", {}) if isinstance(scorecard_payload, dict) else {}
    source = metadata.get("source") or metadata.get("source_report") or metadata.get("source_cohort_json")
    if source:
        return {
            "confidence_source": "surrogate_modal_sm100_or_non_authoritative_capture",
            "hardware_caveat": "Thresholds are derived from non-authoritative or surrogate captures unless later recalibrated on explicit sm_100a-proof hardware.",
            "evidence_source": source,
        }
    return {
        "confidence_source": "no_capture_source_recorded",
        "hardware_caveat": "Threshold provenance is incomplete; treat thresholds as provisional until source hardware is recorded and calibrated.",
        "evidence_source": None,
    }


def build_structural_groups(inventory_payload: dict[str, Any], scorecard_payload: dict[str, Any] | None = None) -> dict[str, Any]:
    rows = inventory_payload.get("rows", [])
    if not isinstance(rows, list):
        raise ValueError("Inventory payload must contain a 'rows' list")

    scorecards = {}
    if scorecard_payload is not None:
        scorecards = {item["uuid"]: item for item in scorecard_payload.get("scorecards", []) if isinstance(item, dict) and "uuid" in item}
    threshold_provenance = _threshold_provenance(scorecard_payload)

    batch_bounds = _quantile_bounds([
        row.get("runtime_summary", {}).get("batch_size") for row in rows if isinstance(row.get("runtime_summary", {}).get("batch_size"), (int, float))
    ])
    max_seq_bounds = _quantile_bounds([
        row.get("runtime_summary", {}).get("max_seq_len") for row in rows if isinstance(row.get("runtime_summary", {}).get("max_seq_len"), (int, float))
    ])
    total_seq_bounds = _quantile_bounds([
        row.get("runtime_summary", {}).get("total_seq_len") for row in rows if isinstance(row.get("runtime_summary", {}).get("total_seq_len"), (int, float))
    ])

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        summary = row.get("runtime_summary", {})
        selector_summary = {
            key: summary.get(key)
            for key in SELECTOR_FIELDS
        }
        selector_key = ":".join(
            [
                "varlen" if summary.get("varlen") else "fixed",
                f"batch={_bucket_numeric(summary.get('batch_size'), batch_bounds)}",
                f"maxseq={_bucket_numeric(summary.get('max_seq_len'), max_seq_bounds)}",
                f"totalseq={_bucket_numeric(summary.get('total_seq_len'), total_seq_bounds)}",
            ]
        )
        enriched = dict(row)
        enriched["selector_summary"] = selector_summary
        enriched["selector_key"] = selector_key
        grouped[selector_key].append(enriched)

    groups = []
    optimization_matrix = []
    for selector_key, group_rows in sorted(grouped.items()):
        label_counts = Counter(_row_structural_label(row, scorecards.get(row["uuid"])) for row in group_rows)
        dominant_label, _ = label_counts.most_common(1)[0]
        structural_subgroups = [
            {
                "structural_label": label,
                "member_count": count,
                "provisional": count < 2,
            }
            for label, count in sorted(label_counts.items())
        ]
        has_non_dominant_singleton = any(
            label != dominant_label and count < 2 for label, count in label_counts.items()
        )
        if has_non_dominant_singleton:
            effective_label = "mixed_or_unknown"
        else:
            effective_label = dominant_label
        structural_member_count = label_counts.get(effective_label, 0) if effective_label in label_counts else 0
        provisional = structural_member_count < 2 or has_non_dominant_singleton
        thresholds = _build_group_thresholds(group_rows, scorecards, effective_label)
        for threshold in thresholds:
            threshold.update(threshold_provenance)
        family = _optimization_family(effective_label)
        group_payload = {
            "selector_key": selector_key,
            "member_count": len(group_rows),
            "structural_member_count": structural_member_count,
            "provisional": provisional,
            "selector_schema": list(SELECTOR_FIELDS),
            "analysis_annotation_schema": [
                "fallback_hit",
                "persistent_policy_outcome",
                "structural_label",
                "ncu_lane",
                "ncu_classification",
            ],
            "structural_label": effective_label,
            "label_counts": dict(label_counts),
            "structural_subgroups": structural_subgroups,
            "uuids": [row["uuid"] for row in group_rows],
            "gate_tables": thresholds,
            "gate_rule": _gate_rule(effective_label),
            "optimization_family": family,
        }
        groups.append(group_payload)
        optimization_matrix.append(
            {
                "selector_key": selector_key,
                "member_count": len(group_rows),
                "structural_member_count": structural_member_count,
                "provisional": provisional,
                "structural_label": effective_label,
                "structural_subgroups": structural_subgroups,
                **family,
            }
        )

    return {
        "metadata": {
            **inventory_payload.get("metadata", {}),
            "selector_field_schema": list(SELECTOR_FIELDS),
            "analysis_annotation_schema": [
                "fallback_hit",
                "persistent_policy_outcome",
                "structural_label",
                "ncu_lane",
                "ncu_classification",
            ],
            "group_count": len(groups),
            "batch_bounds": batch_bounds,
            "max_seq_bounds": max_seq_bounds,
            "total_seq_bounds": total_seq_bounds,
        },
        "groups": groups,
        "optimization_family_matrix": optimization_matrix,
    }


def _write_payload(payload: dict[str, Any], output_json: Path | None) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    print(text, end="")
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    inventory_payload = json.loads(args.inventory_json.read_text(encoding="utf-8"))
    scorecard_payload = None
    if args.scorecard_json is not None:
        scorecard_payload = json.loads(args.scorecard_json.read_text(encoding="utf-8"))
    payload = build_structural_groups(inventory_payload, scorecard_payload)
    _write_payload(payload, args.output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
