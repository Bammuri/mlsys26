"""Pack the current prefill solution with an overridden entrypoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from flashinfer_bench import BuildSpec
from flashinfer_bench.agents import pack_solution_from_files

from scripts.pack_solution import PROJECT_ROOT, load_config


def pack_variant(entry_point: str, output_path: Path, name_suffix: str = "") -> Path:
    config = load_config()
    solution_config = config["solution"]
    build_config = config["build"]

    language = build_config["language"]
    entry_file = entry_point.split("::", 1)[0]
    runtime_language = "python" if language == "cuda" and entry_file.endswith(".py") else language

    source_dir = PROJECT_ROOT / "solution" / "cuda"
    spec = BuildSpec(
        language=runtime_language,
        target_hardware=["cuda"],
        entry_point=entry_point,
        dependencies=build_config.get("dependencies", []),
        destination_passing_style=build_config.get("destination_passing_style", True),
        binding=None if runtime_language == "python" else build_config.get("binding"),
    )
    solution = pack_solution_from_files(
        path=str(source_dir),
        spec=spec,
        name=solution_config["name"] + name_suffix,
        definition=solution_config["definition"],
        author=solution_config["author"],
    )
    output_path.write_text(solution.model_dump_json(indent=2))
    print(f"packed {entry_point} -> {output_path}")
    print(f"runtime_language={runtime_language}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("entry_point")
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--name-suffix", default="")
    args = parser.parse_args()
    pack_variant(args.entry_point, args.output_path, args.name_suffix)
