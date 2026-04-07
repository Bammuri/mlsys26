"""
Pack solution source files into solution.json.

Reads configuration from config.toml and packs the appropriate source files
(Triton or CUDA) into a Solution JSON file for submission.
"""

import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import tomllib
except ImportError:
    import tomli as tomllib

from flashinfer_bench import BuildSpec
from flashinfer_bench.agents import pack_solution_from_files


def load_config() -> dict:
    """Load configuration from config.toml."""
    config_path = PROJECT_ROOT / "config.toml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "rb") as f:
        return tomllib.load(f)


def pack_solution(
    output_path: Path = None,
    *,
    definition: str | None = None,
    language: str | None = None,
    entry_point: str | None = None,
    destination_passing_style: bool | None = None,
    binding: str | None = None,
) -> Path:
    """Pack solution files into a Solution JSON."""
    config = load_config()

    solution_config = config["solution"]
    build_config = config["build"]

    solution_definition = definition or solution_config["definition"]
    build_language = language or build_config["language"]
    build_entry_point = entry_point or build_config["entry_point"]

    # Determine source directory based on language
    if build_language == "triton":
        source_dir = PROJECT_ROOT / "solution" / "triton"
    elif build_language == "cuda":
        source_dir = PROJECT_ROOT / "solution" / "cuda"
    else:
        raise ValueError(f"Unsupported language: {build_language}")

    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    # Create build spec
    dps = (
        destination_passing_style
        if destination_passing_style is not None
        else build_config.get("destination_passing_style", True)
    )
    spec = BuildSpec(
        language=build_language,
        target_hardware=["cuda"],
        entry_point=build_entry_point,
        destination_passing_style=dps,
        binding=binding or build_config.get("binding"),
    )

    # Pack the solution
    solution = pack_solution_from_files(
        path=str(source_dir),
        spec=spec,
        name=solution_config["name"],
        definition=solution_definition,
        author=solution_config["author"],
    )

    # Write to output file
    if output_path is None:
        output_path = PROJECT_ROOT / "solution.json"

    output_path.write_text(solution.model_dump_json(indent=2))
    print(f"Solution packed: {output_path}")
    print(f"  Name: {solution.name}")
    print(f"  Definition: {solution.definition}")
    print(f"  Author: {solution.author}")
    print(f"  Language: {build_language}")
    print(f"  Entry point: {build_entry_point}")
    print(f"  Binding: {spec.binding}")
    print(f"  DPS: {spec.destination_passing_style}")

    return output_path


def main():
    """Entry point for pack_solution script."""
    import argparse

    parser = argparse.ArgumentParser(description="Pack solution files into solution.json")
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output path for solution.json (default: ./solution.json)"
    )
    parser.add_argument("--definition", type=str, default=None, help="Override solution definition")
    parser.add_argument("--language", type=str, default=None, help="Override build language")
    parser.add_argument("--entry-point", type=str, default=None, help="Override build entry point")
    parser.add_argument(
        "--destination-passing-style",
        dest="destination_passing_style",
        action="store_true",
        help="Override destination passing style to true",
    )
    parser.add_argument(
        "--value-returning-style",
        dest="destination_passing_style",
        action="store_false",
        help="Override destination passing style to false",
    )
    parser.add_argument("--binding", type=str, default=None, help="Override CUDA/C++ binding")
    parser.set_defaults(destination_passing_style=None)
    args = parser.parse_args()

    try:
        pack_solution(
            args.output,
            definition=args.definition,
            language=args.language,
            entry_point=args.entry_point,
            destination_passing_style=args.destination_passing_style,
            binding=args.binding,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
