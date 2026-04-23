import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_structural_groups import build_structural_groups, main


class StructuralGroupsTest(unittest.TestCase):
    def _inventory_payload(self) -> dict:
        return {
            "metadata": {"definition": "demo", "workload_count": 3},
            "rows": [
                {
                    "uuid": "fast-a",
                    "runtime_summary": {
                        "varlen": False,
                        "batch_size": 1,
                        "max_seq_len": 32,
                        "total_seq_len": 32,
                        "h_q": 4,
                        "h_v": 8,
                        "head_dim": 128,
                    },
                },
                {
                    "uuid": "tail-a",
                    "runtime_summary": {
                        "varlen": True,
                        "batch_size": 6,
                        "max_seq_len": 768,
                        "total_seq_len": 2048,
                        "h_q": 4,
                        "h_v": 8,
                        "head_dim": 128,
                    },
                },
                {
                    "uuid": "tail-b",
                    "runtime_summary": {
                        "varlen": True,
                        "batch_size": 6,
                        "max_seq_len": 740,
                        "total_seq_len": 2100,
                        "h_q": 4,
                        "h_v": 8,
                        "head_dim": 128,
                    },
                },
            ],
        }

    def _scorecard_payload(self) -> dict:
        return {
            "metadata": {"definition": "demo"},
            "scorecards": [
                {
                    "uuid": "fast-a",
                    "primary": {
                        "scheduler_health": {
                            "metrics": {
                                "issue_efficiency": 72.0,
                                "skipped_issue_slots": 10.0,
                                "eligible_warps_per_scheduler": 1.2,
                                "issued_warps_per_scheduler": 0.9,
                            }
                        },
                        "occupancy_effectiveness": {
                            "metrics": {
                                "achieved_occupancy_pct": 44.0,
                                "occupancy_gap_pct": 8.0,
                            }
                        },
                        "bound_classification": {
                            "classification": "balanced_or_mixed",
                            "metrics": {
                                "compute_utilization_pct": 40.0,
                                "memory_utilization_pct": 42.0,
                            },
                        },
                    },
                },
                {
                    "uuid": "tail-a",
                    "primary": {
                        "scheduler_health": {
                            "metrics": {
                                "issue_efficiency": 20.0,
                                "skipped_issue_slots": 78.0,
                                "eligible_warps_per_scheduler": 0.22,
                                "issued_warps_per_scheduler": 0.20,
                            }
                        },
                        "occupancy_effectiveness": {
                            "metrics": {
                                "achieved_occupancy_pct": 12.0,
                                "occupancy_gap_pct": 45.0,
                            }
                        },
                        "bound_classification": {
                            "classification": "balanced_or_mixed",
                            "metrics": {
                                "compute_utilization_pct": 18.0,
                                "memory_utilization_pct": 24.0,
                            },
                        },
                    },
                },
                {
                    "uuid": "tail-b",
                    "primary": {
                        "scheduler_health": {
                            "metrics": {
                                "issue_efficiency": 22.0,
                                "skipped_issue_slots": 74.0,
                                "eligible_warps_per_scheduler": 0.30,
                                "issued_warps_per_scheduler": 0.24,
                            }
                        },
                        "occupancy_effectiveness": {
                            "metrics": {
                                "achieved_occupancy_pct": 15.0,
                                "occupancy_gap_pct": 40.0,
                            }
                        },
                        "bound_classification": {
                            "classification": "balanced_or_mixed",
                            "metrics": {
                                "compute_utilization_pct": 19.0,
                                "memory_utilization_pct": 23.0,
                            },
                        },
                    },
                },
            ],
        }

    def test_build_structural_groups_produces_numeric_gates_and_optimization_matrix(self) -> None:
        payload = build_structural_groups(self._inventory_payload(), self._scorecard_payload())

        self.assertEqual(payload["metadata"]["group_count"], 3)
        groups = {item["selector_key"]: item for item in payload["groups"]}
        self.assertTrue(any(group["structural_label"] == "scheduler_occupancy" for group in groups.values()))
        scheduler_group = next(group for group in groups.values() if group["structural_label"] == "scheduler_occupancy")
        self.assertEqual(scheduler_group["optimization_family"]["optimization_family"], "scheduler_latency_hiding")
        self.assertIn("hold_rule", scheduler_group["gate_rule"])
        self.assertEqual(len(scheduler_group["gate_tables"]), 4)
        self.assertTrue(all("pass_cutoff" in metric for metric in scheduler_group["gate_tables"]))
        self.assertTrue(all(metric["hold_cutoff"] != metric["reject_cutoff"] for metric in scheduler_group["gate_tables"]))
        self.assertTrue(all("confidence_source" in metric for metric in scheduler_group["gate_tables"]))
        self.assertEqual(len(payload["optimization_family_matrix"]), 3)

    def test_single_member_group_is_marked_provisional(self) -> None:
        payload = build_structural_groups(
            {
                "metadata": {"definition": "demo", "workload_count": 1},
                "rows": [self._inventory_payload()["rows"][0]],
            },
            self._scorecard_payload(),
        )
        self.assertTrue(payload["groups"][0]["provisional"])
        self.assertEqual(payload["groups"][0]["structural_member_count"], 1)

    def test_mixed_singleton_structural_label_downgrades_group(self) -> None:
        inventory = {
            "metadata": {"definition": "demo", "workload_count": 3},
            "rows": [
                {
                    "uuid": "m1",
                    "runtime_summary": {
                        "varlen": True,
                        "batch_size": 4,
                        "max_seq_len": 512,
                        "total_seq_len": 1600,
                        "h_q": 4,
                        "h_v": 8,
                        "head_dim": 128,
                    },
                },
                {
                    "uuid": "m2",
                    "runtime_summary": {
                        "varlen": True,
                        "batch_size": 4,
                        "max_seq_len": 512,
                        "total_seq_len": 1600,
                        "h_q": 4,
                        "h_v": 8,
                        "head_dim": 128,
                    },
                },
                {
                    "uuid": "m3",
                    "runtime_summary": {
                        "varlen": True,
                        "batch_size": 4,
                        "max_seq_len": 512,
                        "total_seq_len": 1600,
                        "h_q": 4,
                        "h_v": 8,
                        "head_dim": 128,
                    },
                },
            ],
        }
        scorecards = {
            "metadata": {"source": "modal-smoke"},
            "scorecards": [
                {
                    "uuid": "m1",
                    "primary": {
                        "scheduler_health": {"metrics": {"issue_efficiency": 20.0, "skipped_issue_slots": 70.0, "eligible_warps_per_scheduler": 0.2}},
                        "occupancy_effectiveness": {"metrics": {"achieved_occupancy_pct": 12.0, "occupancy_gap_pct": 38.0}},
                        "bound_classification": {"classification": "balanced_or_mixed", "metrics": {"compute_utilization_pct": 20.0, "memory_utilization_pct": 24.0}},
                    },
                },
                {
                    "uuid": "m2",
                    "primary": {
                        "scheduler_health": {"metrics": {"issue_efficiency": 22.0, "skipped_issue_slots": 68.0, "eligible_warps_per_scheduler": 0.3}},
                        "occupancy_effectiveness": {"metrics": {"achieved_occupancy_pct": 14.0, "occupancy_gap_pct": 34.0}},
                        "bound_classification": {"classification": "balanced_or_mixed", "metrics": {"compute_utilization_pct": 21.0, "memory_utilization_pct": 23.0}},
                    },
                },
                {
                    "uuid": "m3",
                    "primary": {
                        "scheduler_health": {"metrics": {"issue_efficiency": 55.0, "skipped_issue_slots": 15.0, "eligible_warps_per_scheduler": 1.5}},
                        "occupancy_effectiveness": {"metrics": {"achieved_occupancy_pct": 40.0, "occupancy_gap_pct": 8.0}},
                        "bound_classification": {"classification": "memory_bound", "metrics": {"compute_utilization_pct": 20.0, "memory_utilization_pct": 50.0}},
                    },
                },
            ],
        }
        payload = build_structural_groups(inventory, scorecards)
        group = payload["groups"][0]
        self.assertEqual(group["structural_label"], "mixed_or_unknown")
        self.assertTrue(group["provisional"])
        self.assertTrue(any(item["structural_label"] == "memory_dependency" and item["provisional"] for item in group["structural_subgroups"]))

    def test_cli_writes_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            inventory_json = tmpdir_path / "inventory.json"
            inventory_json.write_text(json.dumps(self._inventory_payload()), encoding="utf-8")
            scorecard_json = tmpdir_path / "scorecards.json"
            scorecard_json.write_text(json.dumps(self._scorecard_payload()), encoding="utf-8")
            output_json = tmpdir_path / "groups.json"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = main([
                    str(inventory_json),
                    "--scorecard-json",
                    str(scorecard_json),
                    "--output-json",
                    str(output_json),
                ])
            written = json.loads(output_json.read_text(encoding="utf-8"))
        self.assertEqual(exit_code, 0)
        self.assertEqual(written["metadata"]["definition"], "demo")
        self.assertIn("optimization_family_matrix", written)
        self.assertIn("groups", written)
        self.assertIn('"group_count": 3', stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
