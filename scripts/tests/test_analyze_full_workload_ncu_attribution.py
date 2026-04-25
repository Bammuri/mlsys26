import unittest

from scripts.analyze_full_workload_ncu_attribution import (
    classify_row,
    kernel_signal_from_ratios,
    parse_extended_ncu_metrics,
)


class FullWorkloadNcuAttributionTest(unittest.TestCase):
    def test_parse_extended_ncu_metrics_extracts_primary_cycle_metrics(self) -> None:
        raw_text = """
Duration                         us        65.38
Elapsed Cycles                cycle        75428
SM Active Cycles              cycle      3613.07
Compute (SM) Throughput           %         0.43
Memory Throughput                 %         0.70
Waves Per SM                                                0.05
"""
        metrics = parse_extended_ncu_metrics(raw_text)
        self.assertEqual(metrics["duration_us"], 65.38)
        self.assertEqual(metrics["elapsed_cycles"], 75428.0)
        self.assertEqual(metrics["sm_active_cycles"], 3613.07)
        self.assertAlmostEqual(metrics["sm_active_ratio"], 3613.07 / 75428.0)
        self.assertEqual(metrics["compute_utilization_pct"], 0.43)
        self.assertEqual(metrics["memory_utilization_pct"], 0.70)
        self.assertEqual(metrics["waves_per_sm"], 0.05)

    def test_classify_row_detects_path_win_only_with_authoritative_latency(self) -> None:
        kernel_signal = kernel_signal_from_ratios(0.99, 1.01)
        self.assertEqual(kernel_signal, "kernel_near_parity")
        self.assertEqual(
            classify_row(
                latency={"authoritative": True, "speedup_factor": 2.0},
                kernel_signal=kernel_signal,
                submit_metrics={"waves_per_sm": 0.05, "skipped_issue_slots": 90.0},
                kernel_duration_ratio=0.99,
                kernel_cycle_ratio=1.01,
            ),
            "path_win",
        )
        self.assertEqual(
            classify_row(
                latency={"authoritative": False, "speedup_factor": 2.0},
                kernel_signal=kernel_signal,
                submit_metrics={"waves_per_sm": 0.05, "skipped_issue_slots": 90.0},
                kernel_duration_ratio=0.99,
                kernel_cycle_ratio=1.01,
            ),
            "unknown",
        )


if __name__ == "__main__":
    unittest.main()
