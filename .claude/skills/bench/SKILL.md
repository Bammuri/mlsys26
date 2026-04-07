---
name: bench
description: Pack solution and run benchmark on Modal B200. Logs results automatically.
user-invocable: true
allowed-tools: Bash Read Write Edit
argument-hint: [decode|prefill|both]
---

# Benchmark on Modal B200

## Target
$ARGUMENTS

If no argument, run both decode and prefill.

## Steps

### 1. Determine target kernel(s)
- `decode` → `gdn_decode_qk4_v8_d128_k_last/`
- `prefill` → `gdn_prefill_qk4_v8_d128_k_last/`
- `both` → run both sequentially

### 2. For each target kernel:

#### 2a. Run Modal benchmark (packs + benchmarks in one step)
```bash
conda run -n fi-bench modal run scripts/run_modal_subfolder.py --subfolder <subfolder>
```
- `<subfolder>` is `gdn_decode_qk4_v8_d128_k_last` or `gdn_prefill_qk4_v8_d128_k_last`
- This automatically packs solution.json and runs benchmark on Modal B200
- Add `--no-quick` for full submission-grade benchmark (slower but more accurate)
- Capture full output.

#### 2b. Parse results
Extract from output:
- Status (pass/fail)
- Latency (ms) per workload
- Speedup factor per workload
- Mean speedup
- Max absolute error, max relative error
- Any correctness failures

#### 2c. Log results
Append to the **kernel-specific** log file:
- Decode → `logs/decode/bench_history.jsonl`
- Prefill → `logs/prefill/bench_history.jsonl`

```json
{
  "timestamp": "<ISO 8601>",
  "kernel": "<decode|prefill>",
  "git_hash": "<short hash>",
  "status": "<pass|fail>",
  "mean_speedup": <float>,
  "mean_latency_ms": <float>,
  "min_speedup": <float>,
  "max_speedup": <float>,
  "correctness": {"max_atol": <float>, "max_rtol": <float>, "matched_ratio": <float>},
  "workload_count": <int>,
  "notes": "<brief description of what changed>",
  "reverted": false
}
```

### 3. Summary output

```
## Benchmark Results

### Decode
- Status: pass/fail
- Mean speedup: Xx
- Mean latency: X.XX ms
- Correctness: atol=X, rtol=X, matched=X%

### Prefill
- Status: pass/fail
- Mean speedup: Xx
- Mean latency: X.XX ms
- Correctness: atol=X, rtol=X, matched=X%

### History (last 5 runs for this kernel)
| Date | Speedup | Notes |
|------|---------|-------|
| ...  | ...     | ...   |
```

### 4. If benchmark fails
- Show the error output clearly
- Check if it's a build error, runtime error, or correctness error
- Suggest likely fixes based on the error type
