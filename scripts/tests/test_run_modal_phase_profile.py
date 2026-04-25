import unittest

from scripts.run_modal_phase_profile import summarize_profiler_events


class _FakeEvent:
    def __init__(self, key, count, self_cpu, cpu_total, cuda_total) -> None:
        self.key = key
        self.count = count
        self.self_cpu_time_total = self_cpu
        self.cpu_time_total = cpu_total
        self.cuda_time_total = cuda_total


class RunModalPhaseProfileTest(unittest.TestCase):
    def test_summarize_profiler_events_sorts_and_extracts_phases(self) -> None:
        events = [
            _FakeEvent("msinfer_entry.prepare_gate_beta", 3, 10.0, 20.0, 5.0),
            _FakeEvent("omx.profile.iteration", 3, 5.0, 100.0, 70.0),
            _FakeEvent("cudaLaunchKernel", 6, 1.0, 2.0, 30.0),
        ]
        summary = summarize_profiler_events(events, top_k=2)
        self.assertEqual(len(summary["top_events"]), 2)
        self.assertEqual(summary["top_events"][0]["key"], "omx.profile.iteration")
        self.assertIn("msinfer_entry.prepare_gate_beta", summary["phase_summary"])
        self.assertIn("omx.profile.iteration", summary["phase_summary"])


if __name__ == "__main__":
    unittest.main()
