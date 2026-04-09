# Project: MLSys 2026 GDN Kernel Optimization

## What This Is
FlashInfer AI Kernel Generation Contest - Track C (Gated Delta Net).
Implementing CUDA kernels for GDN decode and prefill on NVIDIA B200 (sm100a).

## Environment Constraints
- **No local B200 GPU** - build/compile only locally, all execution on Modal B200
- Run benchmarks with: `/bench decode`, `/bench prefill`, or `/bench both`
- Never assume local GPU execution works

## Optimization Workflow
Use the iterative loop: `/research` → implement → `/bench` → evaluate → repeat.

Available skills:
- `/research [topic]` - Search for optimization ideas (web, papers, FlashInfer source)
- `/bench [decode|prefill|both]` - Pack + benchmark on Modal B200 + log results
- `/optimize [decode|prefill] [topic]` - Full optimization loop with research → implement → bench
- `/log-result [N]` - View benchmark history and trends

## Project Structure
```
gdn_decode_qk4_v8_d128_k_last/     # Decode kernel (B=1-64, T=1)
  config.toml
  solution/cuda/{kernel.cu, binding.py}

gdn_prefill_qk4_v8_d128_k_last/    # Prefill kernel (T=6-8192, variable length)
  config.toml
  solution/cuda/{kernel.cu, binding.py}

logs/
  decode/
    bench_history.jsonl             # Decode benchmark results (append-only)
    optimization_log.md             # Decode optimization iteration notes
  prefill/
    bench_history.jsonl             # Prefill benchmark results (append-only)
    optimization_log.md             # Prefill optimization iteration notes
```

## Key Technical Details
- State: [H=8, V=128, K=128] float32, k-last layout
- GVA: q_heads=4, k_heads=4, v_heads=8 (2x expansion)
- Precision: state/gates float32, inputs/outputs bfloat16
- DPS: outputs pre-allocated, write in-place
- Correctness: All workloads must PASS on Modal benchmark

## Profiling
- **NCU (Nsight Compute) is available on Modal B200** — FAQ.md says otherwise but this is outdated. Use NCU freely for kernel profiling and bottleneck analysis.

## Rules
- Correctness before performance — always
- One optimization per iteration
- Always benchmark before AND after changes
- Log every benchmark run to bench_history.jsonl
- **Target sm100a** (not sm100). FAQ.md says sm100 but we target sm100a to use full Blackwell instructions (WGMMA, TMA multicast, cluster features). Ignore any documentation that says sm100.
