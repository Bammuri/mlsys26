import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import safetensors.torch
import torch

from scripts.benchmark_results import save_results_json
from scripts.build_workload_inventory import build_inventory, main


VARLEN_DEFINITION = {
    "name": "demo_prefill",
    "description": "Minimal varlen prefill-like definition for workload inventory tests.",
    "op_type": "demo",
    "axes": {
        "total_seq_len": {"type": "var"},
        "num_seqs": {"type": "var"},
        "num_q_heads": {"type": "const", "value": 4},
        "num_v_heads": {"type": "const", "value": 8},
        "head_size": {"type": "const", "value": 128},
        "len_cu_seqlens": {"type": "var"},
    },
    "inputs": {
        "q": {"shape": ["total_seq_len", "num_q_heads", "head_size"], "dtype": "bfloat16"},
        "v": {"shape": ["total_seq_len", "num_v_heads", "head_size"], "dtype": "bfloat16"},
        "cu_seqlens": {"shape": ["len_cu_seqlens"], "dtype": "int64"},
    },
    "outputs": {
        "output": {"shape": ["total_seq_len", "num_v_heads", "head_size"], "dtype": "bfloat16"}
    },
    "reference": "def run(q, v, cu_seqlens):\n    return v\n",
}

FIXED_DEFINITION = {
    "name": "demo_fixed",
    "description": "Minimal fixed-shape definition for workload inventory tests.",
    "op_type": "demo",
    "axes": {
        "batch_size": {"type": "var"},
        "seq_len": {"type": "var"},
        "num_q_heads": {"type": "const", "value": 2},
        "num_v_heads": {"type": "const", "value": 4},
        "head_dim": {"type": "const", "value": 64},
    },
    "inputs": {
        "q": {"shape": ["batch_size", "seq_len", "num_q_heads", "head_dim"], "dtype": "bfloat16"},
        "v": {"shape": ["batch_size", "seq_len", "num_v_heads", "head_dim"], "dtype": "bfloat16"},
    },
    "outputs": {
        "output": {"shape": ["batch_size", "seq_len", "num_v_heads", "head_dim"], "dtype": "bfloat16"}
    },
    "reference": "def run(q, v):\n    return v\n",
}


class WorkloadInventoryTest(unittest.TestCase):
    def _write_definition(self, dataset_root: Path, definition: dict) -> None:
        path = dataset_root / "definitions" / definition["op_type"] / f"{definition['name']}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(definition, indent=2) + "\n")

    def _write_workload_trace(
        self,
        dataset_root: Path,
        *,
        definition_name: str,
        op_type: str,
        uuid: str,
        axes: dict,
        tensor_shapes: dict[str, tuple[int, ...]],
        cu_seqlens: list[int] | None = None,
    ) -> None:
        blob_relpath = f"blob/workloads/{op_type}/{definition_name}/{definition_name}_{uuid}.safetensors"
        blob_path = dataset_root / blob_relpath
        blob_path.parent.mkdir(parents=True, exist_ok=True)

        tensors = {
            name: torch.zeros(shape, dtype=torch.bfloat16)
            for name, shape in tensor_shapes.items()
        }
        if cu_seqlens is not None:
            tensors["cu_seqlens"] = torch.tensor(cu_seqlens, dtype=torch.int64)
        safetensors.torch.save_file(tensors, blob_path)

        workload_inputs = {
            name: {"type": "safetensors", "path": f"./{blob_relpath}", "tensor_key": name}
            for name in tensor_shapes
        }
        if cu_seqlens is not None:
            workload_inputs["cu_seqlens"] = {
                "type": "safetensors",
                "path": f"./{blob_relpath}",
                "tensor_key": "cu_seqlens",
            }

        workload_record = {
            "definition": definition_name,
            "solution": None,
            "workload": {
                "uuid": uuid,
                "axes": axes,
                "inputs": workload_inputs,
            },
            "evaluation": None,
        }
        workload_path = dataset_root / "workloads" / op_type / f"{definition_name}.jsonl"
        workload_path.parent.mkdir(parents=True, exist_ok=True)
        with workload_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(workload_record) + "\n")

    def _make_trace_set(self, dataset_root: Path) -> tuple[str, str, Path, Path]:
        self._write_definition(dataset_root, VARLEN_DEFINITION)
        self._write_definition(dataset_root, FIXED_DEFINITION)

        uuid_a = "11111111-1111-1111-1111-111111111111"
        uuid_b = "22222222-2222-2222-2222-222222222222"
        uuid_fixed = "33333333-3333-3333-3333-333333333333"

        self._write_workload_trace(
            dataset_root,
            definition_name="demo_prefill",
            op_type="demo",
            uuid=uuid_a,
            axes={"total_seq_len": 6, "num_seqs": 1, "len_cu_seqlens": 2},
            tensor_shapes={"q": (6, 4, 128), "v": (6, 8, 128)},
            cu_seqlens=[0, 6],
        )
        self._write_workload_trace(
            dataset_root,
            definition_name="demo_prefill",
            op_type="demo",
            uuid=uuid_b,
            axes={"total_seq_len": 10, "num_seqs": 3, "len_cu_seqlens": 4},
            tensor_shapes={"q": (10, 4, 128), "v": (10, 8, 128)},
            cu_seqlens=[0, 2, 7, 10],
        )
        self._write_workload_trace(
            dataset_root,
            definition_name="demo_fixed",
            op_type="demo",
            uuid=uuid_fixed,
            axes={"batch_size": 2, "seq_len": 4},
            tensor_shapes={"q": (2, 4, 2, 64), "v": (2, 4, 4, 64)},
        )

        results_json = dataset_root / "results.json"
        save_results_json(
            results_json,
            {
                "demo_prefill": {
                    uuid_a: {"status": "PASSED", "latency_ms": 0.25, "speedup_factor": 4.0},
                }
            },
            source="unit-test",
            trace_set_path=str(dataset_root),
            extra_metadata={"solution": {"definition": "demo_prefill", "name": "demo-solution"}},
        )

        ncu_json = dataset_root / "ncu.json"
        ncu_json.write_text(
            json.dumps(
                {
                    "reports": [
                        {"uuid": uuid_a, "label": "baseline", "status": "ok", "report_path": "a.txt"},
                        {"workload_uuid": uuid_b, "label": "candidate", "status": "missing"},
                        {"uuid": "deadbeef-dead-beef-dead-beefdeadbeef", "status": "ok"},
                    ]
                },
                indent=2,
            )
            + "\n"
        )

        return uuid_a, uuid_b, results_json, ncu_json

    def test_build_inventory_joins_results_and_ncu_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_root = Path(tmpdir) / "dataset"
            uuid_a, uuid_b, results_json, ncu_json = self._make_trace_set(dataset_root)

            payload = build_inventory(
                definition_name="demo_prefill",
                trace_set_path=dataset_root,
                results_json=results_json,
                ncu_json=ncu_json,
            )

        self.assertEqual(payload["metadata"]["definition"], "demo_prefill")
        self.assertEqual(payload["metadata"]["workload_count"], 2)
        self.assertEqual(payload["metadata"]["matched_benchmark_rows"], 1)
        self.assertEqual(payload["metadata"]["matched_ncu_rows"], 2)
        self.assertEqual(
            payload["metadata"]["unmatched_ncu_uuids"],
            ["deadbeef-dead-beef-dead-beefdeadbeef"],
        )

        rows = {row["uuid"]: row for row in payload["rows"]}
        row_a = rows[uuid_a]
        row_b = rows[uuid_b]

        self.assertEqual(row_a["benchmark_result"]["latency_ms"], 0.25)
        self.assertEqual(row_a["runtime_summary"]["batch_size"], 1)
        self.assertEqual(row_a["runtime_summary"]["max_seq_len"], 6)
        self.assertEqual(row_a["runtime_summary"]["total_seq_len"], 6)
        self.assertTrue(row_a["runtime_summary"]["varlen"])
        self.assertEqual(row_a["runtime_summary"]["h_q"], 4)
        self.assertEqual(row_a["runtime_summary"]["h_v"], 8)
        self.assertEqual(row_a["runtime_summary"]["head_dim"], 128)
        self.assertEqual(len(row_a["ncu_rows"]), 1)
        self.assertEqual(row_a["ncu_rows"][0]["label"], "baseline")

        self.assertIsNone(row_b["benchmark_result"])
        self.assertEqual(row_b["runtime_summary"]["batch_size"], 3)
        self.assertEqual(row_b["runtime_summary"]["max_seq_len"], 5)
        self.assertEqual(row_b["runtime_summary"]["total_seq_len"], 10)
        self.assertEqual(row_b["ncu_rows"][0]["uuid"], uuid_b)

    def test_build_inventory_supports_fixed_shape_definitions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_root = Path(tmpdir) / "dataset"
            _, _, _, _ = self._make_trace_set(dataset_root)

            payload = build_inventory(
                definition_name="demo_fixed",
                trace_set_path=dataset_root,
                results_json=None,
                ncu_json=None,
            )

        self.assertEqual(payload["metadata"]["definition"], "demo_fixed")
        self.assertEqual(payload["metadata"]["workload_count"], 1)
        row = payload["rows"][0]
        self.assertFalse(row["runtime_summary"]["varlen"])
        self.assertEqual(row["runtime_summary"]["batch_size"], 2)
        self.assertEqual(row["runtime_summary"]["max_seq_len"], 4)
        self.assertEqual(row["runtime_summary"]["total_seq_len"], 8)
        self.assertEqual(row["runtime_summary"]["h_q"], 2)
        self.assertEqual(row["runtime_summary"]["h_v"], 4)
        self.assertEqual(row["runtime_summary"]["head_dim"], 64)

    def test_cli_prints_and_writes_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_root = Path(tmpdir) / "dataset"
            _, _, results_json, ncu_json = self._make_trace_set(dataset_root)
            output_json = Path(tmpdir) / "inventory.json"

            stdout = io.StringIO()
            argv = [
                "build_workload_inventory.py",
                "--definition",
                "demo_prefill",
                "--trace-set-path",
                str(dataset_root),
                "--results-json",
                str(results_json),
                "--ncu-json",
                str(ncu_json),
                "--output-json",
                str(output_json),
            ]
            with mock.patch.object(sys, "argv", argv):
                with contextlib.redirect_stdout(stdout):
                    main()

            printed = json.loads(stdout.getvalue())
            written = json.loads(output_json.read_text())

        self.assertEqual(printed, written)
        self.assertEqual(printed["metadata"]["workload_count"], 2)

    def test_definition_and_trace_path_can_be_inferred_from_results_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_root = Path(tmpdir) / "dataset"
            uuid_a, _, results_json, _ = self._make_trace_set(dataset_root)

            payload = build_inventory(
                definition_name=None,
                trace_set_path=None,
                results_json=results_json,
                ncu_json=None,
            )

        self.assertEqual(payload["metadata"]["definition"], "demo_prefill")
        uuids = {row["uuid"] for row in payload["rows"]}
        self.assertIn(uuid_a, uuids)


if __name__ == "__main__":
    unittest.main()
