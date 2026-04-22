import tempfile
import unittest
from pathlib import Path

from scripts.benchmark_results import (
    extract_results_map,
    filter_workloads_by_uuid,
    load_results_payload,
    save_results_json,
    select_workloads_evenly,
    summarize_trace_entries,
)
from scripts.freeze_canonical_workloads import build_canonical_lane_payload
from scripts.profile_workload import _resolve_trace_set_path
from scripts.ncu_scorecard import (
    build_scorecard_payload,
    detect_sections,
    parse_ncu_metrics,
    resolve_report_match,
)
from scripts.segment_benchmark_results import (
    auto_cluster_boundaries,
    segment_definition_rows,
    summarize_rows,
)


class BenchmarkResultsIOTest(unittest.TestCase):
    def test_save_and_load_results_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "results.json"
            results = {"demo": {"wl-fast": {"status": "OK", "latency_ms": 0.1}}}
            save_results_json(
                output,
                results,
                source="unit-test",
                extra_metadata={"trace_set_path": "/tmp/dataset"},
            )
            payload = load_results_payload(output)
            self.assertEqual(payload["metadata"]["source"], "unit-test")
            self.assertEqual(extract_results_map(payload), results)

    def test_workload_filtering_and_selection_helpers(self) -> None:
        class _Workload:
            def __init__(self, uuid: str) -> None:
                self.workload = type("WorkloadRef", (), {"uuid": uuid})()

        workloads = [_Workload("a"), _Workload("b"), _Workload("c"), _Workload("d")]

        filtered = filter_workloads_by_uuid(workloads, ["b", "d"])
        selected = select_workloads_evenly(workloads, 3)

        self.assertEqual([item.workload.uuid for item in filtered], ["b", "d"])
        self.assertEqual([item.workload.uuid for item in selected], ["a", "c", "d"])


class SegmentBenchmarkResultsTest(unittest.TestCase):
    def test_auto_cluster_boundaries_three_bands(self) -> None:
        fast_threshold, tail_threshold = auto_cluster_boundaries([0.04, 0.041, 0.09, 0.11, 0.4, 0.5])
        self.assertAlmostEqual(fast_threshold, 0.041)
        self.assertAlmostEqual(tail_threshold, 0.4)

    def test_segment_definition_rows_adds_cohorts_and_enrichment(self) -> None:
        rows = [
            {"uuid": "a", "status": "OK", "latency_ms": 0.04},
            {"uuid": "b", "status": "OK", "latency_ms": 0.10},
            {"uuid": "c", "status": "OK", "latency_ms": 0.50},
            {"uuid": "d", "status": "FAILED", "latency_ms": None},
        ]
        workload_index = {
            ("demo", "a"): {
                "axes": {"batch_size": 1},
                "shape_fingerprint": "batch_size=1",
                "input_names": ["q"],
                "has_initial_state": False,
            },
        }

        payload = segment_definition_rows("demo", rows, workload_index, None, None)
        workloads = {row["uuid"]: row for row in payload["workloads"]}

        self.assertEqual(workloads["a"]["cohort"], "fast")
        self.assertEqual(workloads["b"]["cohort"], "mid")
        self.assertEqual(workloads["c"]["cohort"], "tail")
        self.assertEqual(workloads["d"]["cohort"], "failed")
        self.assertEqual(workloads["a"]["axes"], {"batch_size": 1})
        self.assertEqual(payload["cohorts"]["tail"], ["c"])

    def test_summary_ignores_failed_rows_with_latency(self) -> None:
        rows = [
            {"uuid": "ok-fast", "status": "OK", "latency_ms": 0.04, "cohort": "fast"},
            {"uuid": "ok-tail", "status": "OK", "latency_ms": 0.50, "cohort": "tail"},
            {"uuid": "fail-timed", "status": "FAILED", "latency_ms": 9.99, "cohort": "failed"},
        ]

        summary = summarize_rows(rows)

        self.assertEqual(summary["status_counts"]["FAILED"], 1)
        self.assertEqual(summary["latency_ms"]["min"], 0.04)
        self.assertEqual(summary["latency_ms"]["max"], 0.50)


class TraceSummaryTest(unittest.TestCase):
    def test_summarize_trace_entries_ignores_failed_latency(self) -> None:
        summary = summarize_trace_entries(
            {
                "ok-fast": {"status": "OK", "latency_ms": 0.04, "speedup_factor": 3.0},
                "ok-tail": {"status": "OK", "latency_ms": 0.50, "speedup_factor": 2.0},
                "fail-timed": {"status": "FAILED", "latency_ms": 9.99, "speedup_factor": 0.1},
            }
        )

        self.assertEqual(summary["statuses"]["FAILED"], 1)
        self.assertEqual(summary["latency_values"], [0.04, 0.50])
        self.assertEqual(summary["speedup_values"], [3.0, 2.0])


class NcuTrackingArtifactsTest(unittest.TestCase):
    def test_build_canonical_lane_payload(self) -> None:
        payload = {
            "metadata": {"source_results_json": ".omx/results/modal-quick-32.json"},
            "results": {
                "demo": {
                    "thresholds": {"fast_max_latency_ms": 0.1, "tail_min_latency_ms": 0.5},
                    "workloads": [
                        {"uuid": "fast-a", "cohort": "fast", "latency_ms": 0.04},
                        {"uuid": "fast-b", "cohort": "fast", "latency_ms": 0.05},
                        {"uuid": "mid-a", "cohort": "mid", "latency_ms": 0.20},
                        {"uuid": "tail-a", "cohort": "tail", "latency_ms": 0.60},
                        {"uuid": "tail-b", "cohort": "tail", "latency_ms": 0.90},
                    ],
                }
            },
        }
        lane_payload = build_canonical_lane_payload(
            payload,
            source_cohort_json=None,
            fast_count=1,
            mid_count=1,
            tail_count=1,
        )
        self.assertEqual(
            lane_payload["metadata"]["source_cohort_json"],
            None,
        )
        self.assertEqual(lane_payload["lanes"]["fast_floor"][0]["uuid"], "fast-a")
        self.assertEqual(lane_payload["lanes"]["mid_transition"][0]["uuid"], "mid-a")
        self.assertEqual(lane_payload["lanes"]["tail"][0]["uuid"], "tail-b")

    def test_build_canonical_lane_payload_records_provenance(self) -> None:
        payload = {
            "metadata": {"source_results_json": ".omx/results/modal.json"},
            "results": {
                "demo": {
                    "thresholds": {},
                    "workloads": [{"uuid": "fast-a", "cohort": "fast", "latency_ms": 0.04}],
                }
            },
        }
        lane_payload = build_canonical_lane_payload(
            payload,
            source_cohort_json=".omx/results/modal.cohorts.json",
            fast_count=1,
            mid_count=0,
            tail_count=0,
        )
        self.assertEqual(
            lane_payload["metadata"]["source_cohort_json"],
            ".omx/results/modal.cohorts.json",
        )
        self.assertEqual(
            lane_payload["metadata"]["source_results_json"],
            ".omx/results/modal.json",
        )

    def test_ncu_scorecard_detects_sections(self) -> None:
        self.assertEqual(
            detect_sections("SchedulerStats Occupancy WarpStateStats"),
            ["SchedulerStats", "Occupancy", "WarpStateStats"],
        )

    def test_build_scorecard_payload(self) -> None:
        canonical_lanes = {
            "metadata": {"definition": "demo"},
            "lanes": {
                "fast_floor": [{"uuid": "fast-a", "latency_ms": 0.04}],
                "mid_transition": [],
                "tail": [{"uuid": "tail-b", "latency_ms": 0.90}],
            },
        }
        payload = build_scorecard_payload(canonical_lanes)
        self.assertEqual(len(payload["scorecards"]), 2)
        self.assertEqual(payload["scorecards"][0]["primary"]["scheduler_health"]["verdict"], "unknown")

    def test_profile_workload_trace_set_fallback(self) -> None:
        with unittest.mock.patch.dict("os.environ", {"FIB_DATASET_PATH": "/tmp/dataset"}, clear=False):
            self.assertEqual(_resolve_trace_set_path(None), "/tmp/dataset")
        self.assertEqual(_resolve_trace_set_path("/explicit/path"), "/explicit/path")

    def test_parse_ncu_metrics(self) -> None:
        raw_text = """
        SchedulerStats
        Issue Slots Busy 72.5%
        Skipped Issue Slots 12.0%
        Eligible Warps Per Scheduler 1.75
        Issued Warps Per Scheduler 0.95
        Occupancy
        Theoretical Occupancy 62.5%
        Achieved Occupancy 37.5%
        SpeedOfLight
        Compute (SM) Throughput 41.0%
        Memory Throughput 78.0%
        Roofline
        """
        metrics = parse_ncu_metrics(raw_text)
        self.assertEqual(metrics["classification"], "memory_bound")
        self.assertEqual(metrics["occupancy_gap_pct"], 25.0)
        self.assertEqual(metrics["issue_efficiency"], 72.5)

    def test_build_scorecard_payload_ingests_report_metrics(self) -> None:
        canonical_lanes = {
            "metadata": {"definition": "demo"},
            "lanes": {
                "tail": [{"uuid": "deadbeef-0000-0000-0000-000000000000", "latency_ms": 0.9}],
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            report_dir = Path(tmpdir)
            report_path = report_dir / "deadbeef-report.txt"
            report_path.write_text(
                "SchedulerStats\nIssue Slots Busy 70.0%\n"
                "Occupancy\nTheoretical Occupancy 60.0%\nAchieved Occupancy 40.0%\n"
                "SpeedOfLight\nCompute (SM) Throughput 65.0%\nMemory Throughput 20.0%\n"
            )
            payload = build_scorecard_payload(canonical_lanes, report_dir=report_dir)
        scorecard = payload["scorecards"][0]
        self.assertEqual(scorecard["source_report"], "deadbeef-report.txt")
        self.assertEqual(
            scorecard["primary"]["bound_classification"]["classification"],
            "compute_bound",
        )
        self.assertEqual(
            scorecard["primary"]["occupancy_effectiveness"]["metrics"]["occupancy_gap_pct"],
            20.0,
        )

    def test_resolve_report_match_rejects_ambiguous_matches(self) -> None:
        report_text = {
            "deadbeef-baseline.txt": "SchedulerStats",
            "deadbeef-candidate.txt": "SchedulerStats",
        }
        report_name, raw_text, notes = resolve_report_match(
            report_text,
            "deadbeef-0000-0000-0000-000000000000",
        )
        self.assertIsNone(report_name)
        self.assertIsNone(raw_text)
        self.assertTrue(notes)

    def test_build_scorecard_payload_leaves_ambiguous_report_uningested(self) -> None:
        canonical_lanes = {
            "metadata": {"definition": "demo"},
            "lanes": {
                "tail": [{"uuid": "deadbeef-0000-0000-0000-000000000000", "latency_ms": 0.9}],
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            report_dir = Path(tmpdir)
            (report_dir / "deadbeef-baseline.txt").write_text("SchedulerStats\nIssue Slots Busy 70.0%\n")
            (report_dir / "deadbeef-candidate.txt").write_text("SchedulerStats\nIssue Slots Busy 80.0%\n")
            payload = build_scorecard_payload(canonical_lanes, report_dir=report_dir)
        scorecard = payload["scorecards"][0]
        self.assertIsNone(scorecard["source_report"])
        self.assertEqual(scorecard["detected_sections"], [])
        self.assertTrue(scorecard["notes"])


if __name__ == "__main__":
    unittest.main()
