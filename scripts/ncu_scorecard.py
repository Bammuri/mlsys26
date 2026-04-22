"""
Generate reusable NCU scorecard templates for canonical workload lanes.
"""

from __future__ import annotations

import argparse
import json
import numbers
import re
import sys
from pathlib import Path
from typing import Any

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


KNOWN_SECTIONS = [
    "SchedulerStats",
    "Occupancy",
    "WarpStateStats",
    "MemoryWorkloadAnalysis",
    "LaunchStats",
    "SpeedOfLight",
    "Roofline",
]

DEFAULT_CAPTURE_LABELS = ("baseline", "candidate")
COMPARISON_LABELS = {"baseline", "candidate"}

METRIC_PATTERNS = {
    "issue_efficiency": [
        r"Issue(?:\s+Slots)?\s+Busy\s+([0-9]+(?:\.[0-9]+)?)\s*%",
        r"Scheduler\s+Issue\s+Efficiency\s+([0-9]+(?:\.[0-9]+)?)\s*%",
    ],
    "skipped_issue_slots": [
        r"Skipped\s+Issue\s+Slots\s+([0-9]+(?:\.[0-9]+)?)\s*%",
        r"No\s+Eligible\s+([0-9]+(?:\.[0-9]+)?)\s*%",
    ],
    "eligible_warps_per_scheduler": [
        r"Eligible\s+Warps\s+Per\s+Scheduler\s+([0-9]+(?:\.[0-9]+)?)",
    ],
    "issued_warps_per_scheduler": [
        r"Issued\s+Warps\s+Per\s+Scheduler\s+([0-9]+(?:\.[0-9]+)?)",
    ],
    "theoretical_occupancy_pct": [
        r"Theoretical\s+Occupancy\s+([0-9]+(?:\.[0-9]+)?)\s*%",
    ],
    "achieved_occupancy_pct": [
        r"Achieved\s+Occupancy\s+([0-9]+(?:\.[0-9]+)?)\s*%",
    ],
    "compute_utilization_pct": [
        r"Compute\s+\(SM\)\s+Throughput\s+([0-9]+(?:\.[0-9]+)?)\s*%",
        r"SM\s+Throughput\s+([0-9]+(?:\.[0-9]+)?)\s*%",
    ],
    "memory_utilization_pct": [
        r"Memory\s+Throughput\s+([0-9]+(?:\.[0-9]+)?)\s*%",
        r"DRAM\s+Throughput\s+([0-9]+(?:\.[0-9]+)?)\s*%",
    ],
}


def detect_sections(raw_text: str) -> list[str]:
    found = []
    for section in KNOWN_SECTIONS:
        pattern = re.compile(re.escape(section), re.IGNORECASE)
        if pattern.search(raw_text):
            found.append(section)
    return found


def _extract_first_float(raw_text: str, patterns: list[str]) -> float | None:
    for pattern in patterns:
        match = re.search(pattern, raw_text, flags=re.IGNORECASE)
        if match:
            return float(match.group(1))
    return None


def parse_ncu_metrics(raw_text: str) -> dict[str, float | str | None]:
    metrics = {
        name: _extract_first_float(raw_text, patterns)
        for name, patterns in METRIC_PATTERNS.items()
    }
    achieved = metrics.get("achieved_occupancy_pct")
    theoretical = metrics.get("theoretical_occupancy_pct")
    metrics["occupancy_gap_pct"] = (
        theoretical - achieved
        if isinstance(theoretical, float) and isinstance(achieved, float)
        else None
    )

    compute = metrics.get("compute_utilization_pct")
    memory = metrics.get("memory_utilization_pct")
    if isinstance(compute, float) and isinstance(memory, float):
        if memory >= compute + 10.0:
            classification = "memory_bound"
        elif compute >= memory + 10.0:
            classification = "compute_bound"
        else:
            classification = "balanced_or_mixed"
    else:
        classification = "unknown"
    metrics["classification"] = classification
    return metrics


def build_scorecard_entry(
    lane_name: str,
    workload_entry: dict[str, Any],
    *,
    source_report: str | None = None,
    detected_sections: list[str] | None = None,
    parsed_metrics: dict[str, Any] | None = None,
    notes: list[str] | None = None,
) -> dict[str, Any]:
    parsed_metrics = parsed_metrics or {}
    return {
        "lane": lane_name,
        "uuid": workload_entry["uuid"],
        "latency_ms_reference": workload_entry.get("latency_ms"),
        "axes": workload_entry.get("axes"),
        "source_report": source_report,
        "detected_sections": detected_sections or [],
        "primary": {
            "scheduler_health": {
                "section": "SchedulerStats / WarpStateStats",
                "metrics": {
                    "issue_efficiency": parsed_metrics.get("issue_efficiency"),
                    "skipped_issue_slots": parsed_metrics.get("skipped_issue_slots"),
                    "eligible_warps_per_scheduler": parsed_metrics.get("eligible_warps_per_scheduler"),
                    "issued_warps_per_scheduler": parsed_metrics.get("issued_warps_per_scheduler"),
                },
                "interpretation": "Focus on stalls only if schedulers fail to issue consistently.",
                "verdict": "unknown",
            },
            "occupancy_effectiveness": {
                "section": "Occupancy / LaunchStats",
                "metrics": {
                    "theoretical_occupancy_pct": parsed_metrics.get("theoretical_occupancy_pct"),
                    "achieved_occupancy_pct": parsed_metrics.get("achieved_occupancy_pct"),
                    "occupancy_gap_pct": parsed_metrics.get("occupancy_gap_pct"),
                },
                "interpretation": "Low occupancy hurts latency hiding; large theory-vs-achieved gap implies imbalance.",
                "verdict": "unknown",
            },
            "bound_classification": {
                "section": "SpeedOfLight / MemoryWorkloadAnalysis / Roofline",
                "metrics": {
                    "compute_utilization_pct": parsed_metrics.get("compute_utilization_pct"),
                    "memory_utilization_pct": parsed_metrics.get("memory_utilization_pct"),
                },
                "classification": parsed_metrics.get("classification", "unknown"),
                "interpretation": "Use roofline + SOL to identify whether the kernel is memory- or compute/scheduler-limited.",
                "verdict": "unknown",
            },
            "cross_lane_consistency": {
                "section": "Cross-lane comparison",
                "interpretation": "Tail wins must not create fast-floor structural regressions.",
                "verdict": "unknown",
            },
        },
        "secondary": {
            "paired_latency_ms": None,
            "trial_1_vs_steady_state": "unknown",
            "broad_regression_screen": "unknown",
        },
        "captures": [],
        "notes": list(notes or []),
    }


def resolve_report_match(report_text: dict[str, str], uuid: str) -> tuple[str | None, str | None, list[str]]:
    full_matches = sorted(name for name in report_text if uuid in name)
    if len(full_matches) == 1:
        report_name = full_matches[0]
        return report_name, report_text[report_name], []
    if len(full_matches) > 1:
        return None, None, [f"ambiguous_report_matches={full_matches}"]

    prefix = uuid[:8]
    prefix_matches = sorted(name for name in report_text if prefix in name)
    if len(prefix_matches) == 1:
        report_name = prefix_matches[0]
        return report_name, report_text[report_name], []
    if len(prefix_matches) > 1:
        return None, None, [f"ambiguous_report_matches={prefix_matches}"]

    return None, None, []


def build_manifest_template(
    canonical_lanes: dict[str, Any],
    *,
    capture_labels: tuple[str, ...] = DEFAULT_CAPTURE_LABELS,
) -> dict[str, Any]:
    reports = []
    for lane_name, entries in canonical_lanes["lanes"].items():
        for entry in entries:
            for label in capture_labels:
                reports.append(
                    {
                        "uuid": entry["uuid"],
                        "lane": lane_name,
                        "label": label,
                        "report_path": None,
                        "notes": [],
                    }
                )
    return {
        "metadata": canonical_lanes["metadata"],
        "reports": reports,
    }


def load_report_manifest(manifest_path: Path) -> tuple[dict[str, list[dict[str, Any]]], Path]:
    payload = json.loads(manifest_path.read_text())
    if not isinstance(payload, dict) or not isinstance(payload.get("reports"), list):
        raise ValueError("Manifest must be a JSON object with a 'reports' list")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for entry in payload["reports"]:
        uuid = entry.get("uuid")
        if not isinstance(uuid, str):
            raise ValueError("Each manifest entry must include a string uuid")
        grouped.setdefault(uuid, []).append(entry)
    return grouped, manifest_path.parent


def _capture_from_raw_text(label: str, report_name: str, raw_text: str) -> dict[str, Any]:
    return {
        "label": label,
        "source_report": report_name,
        "detected_sections": detect_sections(raw_text),
        "metrics": parse_ncu_metrics(raw_text),
        "notes": [],
        "latency_ms": None,
        "trial_phase": None,
    }


def _metric_delta(
    baseline: float | None,
    candidate: float | None,
    *,
    higher_is_better: bool,
) -> dict[str, Any]:
    if not isinstance(baseline, numbers.Real) or not isinstance(candidate, numbers.Real):
        return {
            "baseline": baseline,
            "candidate": candidate,
            "delta": None,
            "verdict": "unknown",
            "higher_is_better": higher_is_better,
        }
    baseline = float(baseline)
    candidate = float(candidate)
    delta = candidate - baseline
    if delta == 0:
        verdict = "neutral"
    elif (delta > 0 and higher_is_better) or (delta < 0 and not higher_is_better):
        verdict = "improved"
    else:
        verdict = "regressed"
    return {
        "baseline": baseline,
        "candidate": candidate,
        "delta": delta,
        "verdict": verdict,
        "higher_is_better": higher_is_better,
    }


def compare_capture_metrics(captures: list[dict[str, Any]]) -> dict[str, Any] | None:
    indexed = {capture["label"]: capture for capture in captures if capture.get("label") in COMPARISON_LABELS}
    if "baseline" not in indexed or "candidate" not in indexed:
        return None

    baseline_capture = indexed["baseline"]
    candidate_capture = indexed["candidate"]
    baseline_metrics = baseline_capture.get("metrics", {})
    candidate_metrics = candidate_capture.get("metrics", {})

    scheduler = {
        "issue_efficiency": _metric_delta(
            baseline_metrics.get("issue_efficiency"),
            candidate_metrics.get("issue_efficiency"),
            higher_is_better=True,
        ),
        "skipped_issue_slots": _metric_delta(
            baseline_metrics.get("skipped_issue_slots"),
            candidate_metrics.get("skipped_issue_slots"),
            higher_is_better=False,
        ),
        "eligible_warps_per_scheduler": _metric_delta(
            baseline_metrics.get("eligible_warps_per_scheduler"),
            candidate_metrics.get("eligible_warps_per_scheduler"),
            higher_is_better=True,
        ),
        "issued_warps_per_scheduler": _metric_delta(
            baseline_metrics.get("issued_warps_per_scheduler"),
            candidate_metrics.get("issued_warps_per_scheduler"),
            higher_is_better=True,
        ),
    }
    occupancy = {
        "theoretical_occupancy_pct": _metric_delta(
            baseline_metrics.get("theoretical_occupancy_pct"),
            candidate_metrics.get("theoretical_occupancy_pct"),
            higher_is_better=True,
        ),
        "achieved_occupancy_pct": _metric_delta(
            baseline_metrics.get("achieved_occupancy_pct"),
            candidate_metrics.get("achieved_occupancy_pct"),
            higher_is_better=True,
        ),
        "occupancy_gap_pct": _metric_delta(
            baseline_metrics.get("occupancy_gap_pct"),
            candidate_metrics.get("occupancy_gap_pct"),
            higher_is_better=False,
        ),
    }
    latency = _metric_delta(
        baseline_capture.get("latency_ms"),
        candidate_capture.get("latency_ms"),
        higher_is_better=False,
    )

    scheduler_verdicts = {item["verdict"] for item in scheduler.values()}
    occupancy_verdicts = {item["verdict"] for item in occupancy.values()}

    def summarize(verdicts: set[str]) -> str:
        known = {verdict for verdict in verdicts if verdict != "unknown"}
        if not known:
            return "unknown"
        if "regressed" in known and "improved" in known:
            return "mixed"
        if known == {"regressed"}:
            return "regressed"
        if "improved" in known:
            return "improved"
        if known == {"neutral"}:
            return "neutral"
        return "mixed"

    return {
        "scheduler_health": {
            "metrics": scheduler,
            "verdict": summarize(scheduler_verdicts),
        },
        "occupancy_effectiveness": {
            "metrics": occupancy,
            "verdict": summarize(occupancy_verdicts),
        },
        "bound_classification": {
            "baseline": baseline_metrics.get("classification", "unknown"),
            "candidate": candidate_metrics.get("classification", "unknown"),
            "verdict": "changed"
            if baseline_metrics.get("classification") != candidate_metrics.get("classification")
            else "unchanged",
        },
        "secondary": {
            "paired_latency_ms": latency,
            "trial_phase": {
                "baseline": baseline_capture.get("trial_phase"),
                "candidate": candidate_capture.get("trial_phase"),
            },
        },
    }


def build_capture_entries(
    uuid: str,
    *,
    report_dir: Path | None = None,
    report_manifest: dict[str, list[dict[str, Any]]] | None = None,
    report_manifest_base_dir: Path | None = None,
) -> tuple[list[dict[str, Any]], str | None, list[str], dict[str, Any], list[str]]:
    captures: list[dict[str, Any]] = []
    notes: list[str] = []
    if report_manifest is not None:
        manifest_entries = report_manifest.get(uuid, [])
        if not manifest_entries:
            return captures, None, [], {}, notes
        seen_labels: set[str] = set()
        for entry in manifest_entries:
            label = entry.get("label", "capture")
            if label in seen_labels:
                notes.append(f"duplicate_capture_label={label}")
                continue
            seen_labels.add(label)
            report_path = entry.get("report_path")
            if not report_path:
                captures.append(
                    {
                        "label": label,
                        "source_report": None,
                        "detected_sections": [],
                        "metrics": {},
                        "notes": ["missing_report_path"],
                        "latency_ms": entry.get("latency_ms"),
                        "trial_phase": entry.get("trial_phase"),
                    }
                )
                continue
            path = Path(report_path)
            if not path.is_absolute():
                if report_manifest_base_dir is not None:
                    path = report_manifest_base_dir / path
                elif report_dir is not None:
                    path = report_dir / path
            if not path.exists():
                captures.append(
                    {
                        "label": label,
                        "source_report": str(path),
                        "detected_sections": [],
                        "metrics": {},
                        "notes": ["missing_report_file"],
                        "latency_ms": entry.get("latency_ms"),
                        "trial_phase": entry.get("trial_phase"),
                    }
                )
                continue
            raw_text = path.read_text(encoding="utf-8", errors="ignore")
            capture = _capture_from_raw_text(label, str(path), raw_text)
            capture["latency_ms"] = entry.get("latency_ms")
            capture["trial_phase"] = entry.get("trial_phase")
            capture["notes"].extend(entry.get("notes", []))
            captures.append(capture)
        return captures, None, [], {}, notes

    report_text = {}
    if report_dir is not None and report_dir.exists():
        for report_path in report_dir.glob("*.txt"):
            report_text[report_path.name] = report_path.read_text(encoding="utf-8", errors="ignore")
    if not report_text:
        return captures, None, [], {}, notes

    source_report, raw_text, notes = resolve_report_match(report_text, uuid)
    if raw_text is None:
        return captures, source_report, [], {}, notes
    detected = detect_sections(raw_text)
    parsed_metrics = parse_ncu_metrics(raw_text)
    captures.append(_capture_from_raw_text("report_dir_match", source_report or uuid, raw_text))
    return captures, source_report, detected, parsed_metrics, notes


def build_scorecard_payload(
    canonical_lanes: dict[str, Any],
    *,
    report_dir: Path | None = None,
    report_manifest: dict[str, list[dict[str, Any]]] | None = None,
    report_manifest_base_dir: Path | None = None,
) -> dict[str, Any]:
    scorecards = []

    for lane_name, entries in canonical_lanes["lanes"].items():
        for entry in entries:
            captures, source_report, detected, parsed_metrics, notes = build_capture_entries(
                entry["uuid"],
                report_dir=report_dir,
                report_manifest=report_manifest,
                report_manifest_base_dir=report_manifest_base_dir,
            )
            scorecard = build_scorecard_entry(
                lane_name,
                entry,
                source_report=source_report,
                detected_sections=detected,
                parsed_metrics=parsed_metrics,
                notes=notes,
            )
            scorecard["captures"] = captures
            comparison = compare_capture_metrics(captures)
            scorecard["comparison"] = comparison
            if comparison is not None:
                scorecard["primary"]["scheduler_health"]["verdict"] = comparison["scheduler_health"]["verdict"]
                scorecard["primary"]["occupancy_effectiveness"]["verdict"] = comparison["occupancy_effectiveness"]["verdict"]
                scorecard["primary"]["bound_classification"]["verdict"] = comparison["bound_classification"]["verdict"]
                scorecard["secondary"]["paired_latency_ms"] = comparison["secondary"]["paired_latency_ms"]
            scorecards.append(scorecard)

    return {
        "metadata": canonical_lanes["metadata"],
        "scorecards": scorecards,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        f"# NCU scorecard template: {payload['metadata']['definition']}",
        "",
        "Use this sheet to evaluate bare-metal NCU captures lane-by-lane.",
        "",
    ]
    for scorecard in payload["scorecards"]:
        lines.extend(
            [
                f"## {scorecard['lane']} — {scorecard['uuid']}",
                f"- reference latency: {scorecard['latency_ms_reference']}",
                f"- axes: {scorecard.get('axes')}",
                f"- source report: {scorecard.get('source_report')}",
                f"- detected sections: {', '.join(scorecard.get('detected_sections', [])) or '(none)'}",
                "",
                "### Capture entries",
            ]
        )
        captures = scorecard.get("captures", [])
        if captures:
            for capture in captures:
                lines.append(
                    f"- {capture['label']}: source={capture.get('source_report')} sections={', '.join(capture.get('detected_sections', [])) or '(none)'}"
                )
                metrics = capture.get("metrics", {})
                if metrics:
                    for key in [
                        "issue_efficiency",
                        "skipped_issue_slots",
                        "eligible_warps_per_scheduler",
                        "issued_warps_per_scheduler",
                        "theoretical_occupancy_pct",
                        "achieved_occupancy_pct",
                        "occupancy_gap_pct",
                        "compute_utilization_pct",
                        "memory_utilization_pct",
                        "classification",
                    ]:
                        lines.append(f"  - {key}: {metrics.get(key)}")
                for note in capture.get("notes", []):
                    lines.append(f"  - note: {note}")
        else:
            lines.append("- (none)")
        lines.extend(
            [
                "",
                "### Comparison summary",
            ]
        )
        comparison = scorecard.get("comparison")
        if comparison is not None:
            lines.append(f"- scheduler_health: {comparison['scheduler_health']['verdict']}")
            lines.append(f"- occupancy_effectiveness: {comparison['occupancy_effectiveness']['verdict']}")
            lines.append(
                f"- bound_classification: {comparison['bound_classification']['baseline']} -> {comparison['bound_classification']['candidate']} ({comparison['bound_classification']['verdict']})"
            )
            paired_latency = comparison["secondary"]["paired_latency_ms"]
            lines.append(
                f"- paired_latency_ms: baseline={paired_latency['baseline']} candidate={paired_latency['candidate']} delta={paired_latency['delta']} verdict={paired_latency['verdict']}"
            )
        else:
            lines.append("- (no baseline/candidate comparison available)")
        lines.extend(
            [
                "",
                "### Primary criteria",
            ]
        )
        for key, section in scorecard["primary"].items():
            lines.append(f"- **{key}** [{section['section']}]")
            lines.append(f"  - interpretation: {section['interpretation']}")
            lines.append(f"  - verdict: {section['verdict']}")
            for metric_name, metric_value in section.get("metrics", {}).items():
                lines.append(f"  - {metric_name}: {metric_value}")
            if "classification" in section:
                lines.append(f"  - classification: {section['classification']}")
        lines.extend(
            [
                "",
                "### Secondary criteria",
                f"- paired_latency_ms: {scorecard['secondary']['paired_latency_ms']}",
                f"- trial_1_vs_steady_state: {scorecard['secondary']['trial_1_vs_steady_state']}",
                f"- broad_regression_screen: {scorecard['secondary']['broad_regression_screen']}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate NCU scorecard templates from canonical lanes")
    parser.add_argument("canonical_lane_json", type=Path)
    parser.add_argument("--report-dir", type=Path, default=None, help="Optional directory of saved NCU text reports")
    parser.add_argument("--manifest-json", type=Path, default=None, help="Optional explicit manifest mapping UUIDs to baseline/candidate report paths")
    parser.add_argument("--output-manifest-template", type=Path, default=None, help="Optional path to save a baseline/candidate manifest template")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-markdown", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    canonical_lanes = json.loads(args.canonical_lane_json.read_text())
    if args.output_manifest_template is not None:
        manifest_template = build_manifest_template(canonical_lanes)
        args.output_manifest_template.parent.mkdir(parents=True, exist_ok=True)
        args.output_manifest_template.write_text(json.dumps(manifest_template, indent=2, sort_keys=True) + "\n")
    report_manifest = None
    report_manifest_base_dir = None
    if args.manifest_json is not None:
        report_manifest, report_manifest_base_dir = load_report_manifest(args.manifest_json)
    payload = build_scorecard_payload(
        canonical_lanes,
        report_dir=args.report_dir,
        report_manifest=report_manifest,
        report_manifest_base_dir=report_manifest_base_dir,
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
