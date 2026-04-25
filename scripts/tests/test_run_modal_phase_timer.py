import unittest

from scripts.run_modal_phase_timer import _entry_module_name, summarize_samples


class RunModalPhaseTimerTest(unittest.TestCase):
    def test_entry_module_name(self) -> None:
        self.assertEqual(_entry_module_name("msinfer_entry.py::run"), "msinfer_entry")
        self.assertEqual(_entry_module_name("pkg/foo.py::run"), "pkg.foo")

    def test_summarize_samples(self) -> None:
        summary = summarize_samples([10.0, 20.0, 30.0])
        self.assertEqual(summary["mean_us"], 20.0)
        self.assertEqual(summary["median_us"], 20.0)
        self.assertEqual(summary["min_us"], 10.0)
        self.assertEqual(summary["max_us"], 30.0)
        self.assertEqual(summary["count"], 3)


if __name__ == "__main__":
    unittest.main()
