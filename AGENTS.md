# Repository Guidelines

## Project Structure & Module Organization
This starter repo is organized around one contest submission per repository. Edit `config.toml` first, then place kernel code under `solution/`:
- `solution/triton/kernel.py` for Triton submissions
- `solution/cuda/kernel.cu` and `solution/cuda/binding.py` for CUDA/FFI submissions
- `scripts/` contains helper entrypoints: `pack_solution.py`, `run_local.py`, and `run_modal.py`
- `images/` stores static logo assets used by the docs
- `solution.json` is the generated submission artifact; regenerate it instead of editing by hand
- `flashinfer-trace/` is a local dataset checkout used for development only

## Build, Test, and Development Commands
- `python scripts/pack_solution.py` — packs the active `solution/` sources plus `config.toml` into `solution.json`
- `python scripts/run_local.py` — runs the packed solution against the local dataset; requires `FIB_DATASET_PATH`
- `modal run scripts/run_modal.py` — runs the same benchmark flow on Modal B200
- `flashinfer-bench run --local /path/to/mlsys26-contest --definitions <definition> ...` — use for targeted manual evaluation, matching `EVALUATION.md`

## Coding Style & Naming Conventions
Use 4-space indentation in Python and keep module-level docstrings concise, matching the existing scripts. Follow PEP 8 for Python helpers (`snake_case` functions, `UPPER_SNAKE_CASE` constants). In CUDA/Triton, keep kernel entry names synchronized with `build.entry_point` in `config.toml` (for example, `kernel.cu::gdn_decode_qk4_v8_d128_k_last`). No formatter or linter config is committed here, so keep changes minimal and consistent with surrounding code.

## Testing Guidelines
There is no dedicated `tests/` suite in this starter kit. Minimum validation for every change is:
1. Repack with `python scripts/pack_solution.py`
2. Run `python scripts/run_local.py` on the relevant definition
3. Capture benchmark or correctness output in your PR notes
If you add reusable Python helpers, prefer `tests/test_<module>.py` with small smoke tests.

## Autonomous Loop Policy
For iterative kernel optimization work, do not stop between loop iterations unless the user explicitly says stop/cancel or a hard blocker prevents further progress. Use:
- a quick validation pass as a working gate
- a full 100-workload decision gate as the keep/revert criterion
- the full-workload arithmetic-mean latency as the primary keep/revert metric
- automatic continuation into the next loop after each result analysis

The current full decision-gate config is `warmup_runs=1, iterations=5, num_trials=1`. If an iteration does not produce a meaningful improvement and the kernel change is reverted, continue automatically into the next optimization loop. Only stop after 10 consecutive non-meaningful reverted iterations or on explicit user instruction.

## Commit & Pull Request Guidelines
Recent history uses short, imperative subjects (for example, `Add FAQ document`). For new work, use a descriptive imperative subject plus the repo’s required Lore trailers (`Constraint:`, `Confidence:`, `Tested:`, etc.). PRs should state the target definition/track, summarize code and `config.toml` changes, link any issue or evaluation thread, and include local or Modal benchmark evidence. Attach screenshots only when updating docs or visual assets.
