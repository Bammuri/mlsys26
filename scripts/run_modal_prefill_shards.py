from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from flashinfer_bench import TraceSet

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GDN prefill benchmark in workload shards via scripts/run_modal.py")
    parser.add_argument("--definition", default="gdn_prefill_qk4_v8_d128_k_last")
    parser.add_argument("--trace-set-path", type=Path, default=Path("/home/hyu/flashinfer/mlsys26-contest"))
    parser.add_argument("--solution-path", type=Path, default=None)
    parser.add_argument("--shard-size", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=Path(".omx/results/prefill-shards"))
    parser.add_argument("--merge-output", type=Path, default=Path(".omx/results/modal-official-full-100.json"))
    parser.add_argument("--start-shard", type=int, default=0)
    parser.add_argument("--limit-shards", type=int, default=0)
    return parser.parse_args(argv)


def resolve_workload_uuids(definition: str, trace_set_path: Path) -> list[str]:
    trace_set = TraceSet.from_path(str(trace_set_path))
    return [trace.workload.uuid for trace in trace_set.workloads.get(definition, [])]


def build_shards(workload_uuids: list[str], shard_size: int) -> list[list[str]]:
    return [workload_uuids[i : i + shard_size] for i in range(0, len(workload_uuids), shard_size)]


def run_shard(*, shard_index: int, shard: list[str], output_path: Path, solution_path: Path | None = None) -> None:
    cmd = [
        "modal",
        "run",
        "scripts/run_modal.py",
        "--summary-only",
        "--workload-uuids",
        ",".join(shard),
        "--save-json",
        str(output_path),
    ]
    if solution_path is not None:
        cmd.extend(["--solution-path", str(solution_path)])
    subprocess.run(cmd, check=True)


def merge_shards(shard_paths: list[Path], merge_output: Path, solution_path: Path | None = None) -> None:
    cmd = [
        sys.executable,
        "scripts/merge_benchmark_results.py",
        *[str(path) for path in shard_paths],
        "--output-json",
        str(merge_output),
    ]
    if solution_path is not None:
        cmd.extend(["--solution-json", str(solution_path)])
    subprocess.run(cmd, check=True)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    uuids = resolve_workload_uuids(args.definition, args.trace_set_path)
    shards = build_shards(uuids, args.shard_size)
    if args.start_shard >= len(shards):
        raise SystemExit(f"start shard {args.start_shard} >= shard count {len(shards)}")

    if args.limit_shards > 0:
        shards = shards[args.start_shard : args.start_shard + args.limit_shards]
        shard_offset = args.start_shard
    else:
        shards = shards[args.start_shard :]
        shard_offset = args.start_shard

    args.output_dir.mkdir(parents=True, exist_ok=True)
    shard_paths: list[Path] = []
    for offset, shard in enumerate(shards, start=shard_offset):
        output_path = args.output_dir / f"modal-official-shard-{offset}.json"
        run_shard(
            shard_index=offset,
            shard=shard,
            output_path=output_path,
            solution_path=args.solution_path,
        )
        shard_paths.append(output_path)

    if shard_paths:
        merge_shards(shard_paths, args.merge_output, solution_path=args.solution_path)
    print(json.dumps({"shard_count": len(shard_paths), "merge_output": str(args.merge_output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
