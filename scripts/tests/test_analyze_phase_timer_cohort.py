import unittest

from scripts.analyze_phase_timer_cohort import build_summary


class AnalyzePhaseTimerCohortTest(unittest.TestCase):
    def test_build_summary(self) -> None:
        rows = [
            {
                "total_delta_us": 40.0,
                "prepare_delta_us": 30.0,
                "launch_delta_us": 25.0,
                "submit_nonlaunch_overhead_us": 20.0,
            },
            {
                "total_delta_us": 20.0,
                "prepare_delta_us": 10.0,
                "launch_delta_us": 15.0,
                "submit_nonlaunch_overhead_us": 5.0,
            },
        ]
        summary = build_summary(rows)
        self.assertEqual(summary["row_count"], 2)
        self.assertEqual(summary["mean_total_delta_us"], 30.0)
        self.assertEqual(summary["positive_total_delta_count"], 2)


if __name__ == "__main__":
    unittest.main()
