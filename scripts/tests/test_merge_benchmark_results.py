import json
import tempfile
import unittest
from pathlib import Path

from scripts.benchmark_results import save_results_json
from scripts.merge_benchmark_results import main, merge_results_payloads


class MergeBenchmarkResultsTest(unittest.TestCase):
    def test_merge_results_payloads(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            shard0 = base / 'shard0.json'
            shard1 = base / 'shard1.json'
            save_results_json(shard0, {'demo': {'u1': {'status': 'PASSED', 'latency_ms': 1.0}}}, source='test', trace_set_path='/tmp/dataset')
            save_results_json(shard1, {'demo': {'u2': {'status': 'PASSED', 'latency_ms': 2.0}}}, source='test')
            payload = merge_results_payloads([shard0, shard1])
        self.assertEqual(payload['results']['demo']['u1']['latency_ms'], 1.0)
        self.assertEqual(payload['results']['demo']['u2']['latency_ms'], 2.0)
        self.assertEqual(payload['metadata']['trace_set_path'], '/tmp/dataset')

    def test_cli_writes_output_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            shard0 = base / 'shard0.json'
            shard1 = base / 'shard1.json'
            output = base / 'merged.json'
            save_results_json(shard0, {'demo': {'u1': {'status': 'PASSED', 'latency_ms': 1.0}}}, source='test')
            save_results_json(shard1, {'demo': {'u2': {'status': 'PASSED', 'latency_ms': 2.0}}}, source='test')
            exit_code = main([str(shard0), str(shard1), '--output-json', str(output)])
            merged = json.loads(output.read_text())
        self.assertEqual(exit_code, 0)
        self.assertEqual(merged['results']['demo']['u1']['latency_ms'], 1.0)
        self.assertEqual(merged['results']['demo']['u2']['latency_ms'], 2.0)

    def test_cli_embeds_solution_provenance_when_solution_json_is_given(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            shard0 = base / 'shard0.json'
            output = base / 'merged.json'
            solution_json = base / 'solution.json'
            save_results_json(shard0, {'demo': {'u1': {'status': 'PASSED', 'latency_ms': 1.0}}}, source='test')
            solution_json.write_text(json.dumps({
                'name': 'flashinfer_wrapper_123ca6',
                'definition': 'gdn_prefill_qk4_v8_d128_k_last',
                'author': 'flashinfer',
                'spec': {
                    'entry_point': 'main.py::run',
                    'destination_passing_style': False,
                },
            }))
            main([str(shard0), '--output-json', str(output), '--solution-json', str(solution_json)])
            merged = json.loads(output.read_text())
        provenance = merged['metadata']['solution_provenance']
        self.assertEqual(provenance['solution_name'], 'flashinfer_wrapper_123ca6')
        self.assertEqual(provenance['solution_author'], 'flashinfer')
        self.assertEqual(provenance['entry_point'], 'main.py::run')
        self.assertFalse(provenance['destination_passing_style'])


if __name__ == '__main__':
    unittest.main()
