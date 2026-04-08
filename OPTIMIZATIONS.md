# Decode Optimization Ledger

Benchmark target: `gdn_decode_qk4_v8_d128_k_last` on Modal `B200:1`

Benchmark config:

- `warmup_runs=1`
- `iterations=5`
- `num_trials=1`
- comparison metric: arithmetic mean of workload `latency_ms`

Current best decode implementation:

- source optimization commit: `e4bf14c` (`Prefer L1 for decode on B200`)
- current branch state: equivalent decode kernel restored after reverting the failed carveout experiment in `b3bc99a`
- current best average latency: `0.035059 ms`

| Order | Optimization | B200-specific | Commit | Kept | Avg latency (ms) | Delta vs naive | Delta vs prev best | Result JSON | Test log | Modal run | Notes |
| --- | --- | --- | --- | --- | ---: | --- | --- | --- | --- | --- | --- |
| 0 | Naive baseline | No | `48e2721` | Yes | `0.076361` | baseline | baseline | `.omx/benchmarks/gdn_decode_qk4_v8_d128_k_last/20260407T150106Z-baseline-naive.json` | `.omx/benchmarks/gdn_decode_qk4_v8_d128_k_last/logs/baseline-naive-48e2721.log` | `ap-ujktiqXnvboKzj9DNKetEy` | Benchmark artifact persistence added in this commit and baseline remeasured. |
| 1 | Stage decode `q/k/g/beta` in shared memory | No | `bbe50fd` | Yes | `0.064058` | `-0.012303 ms` (`-16.11%`) | `-0.012303 ms` (`-16.11%`) | `.omx/benchmarks/gdn_decode_qk4_v8_d128_k_last/20260407T150305Z-opt1-shared-qk.json` | `.omx/benchmarks/gdn_decode_qk4_v8_d128_k_last/logs/opt1-shared-qk-bbe50fd.log` | `ap-L01JSIHIEjAznr4q89LKaw` | Removed repeated global reads for per-block shared decode inputs. |
| 2 | Vectorize decode state update loops with `float4` chunks | No | `9e34313` | Yes | `0.036826` | `-0.039535 ms` (`-51.77%`) | `-0.027232 ms` (`-42.51%`) | `.omx/benchmarks/gdn_decode_qk4_v8_d128_k_last/20260407T150521Z-opt2-float4-state.json` | `.omx/benchmarks/gdn_decode_qk4_v8_d128_k_last/logs/opt2-float4-state-9e34313.log` | `ap-O3nPAdvogkLvIulSJj6ROi` | Best result so far. Kept as current decode implementation. |
| 3 | Tune decode `__launch_bounds__(128, 8)` for B200 | Yes | `e2da291` | No | `0.043963` | `-0.032398 ms` (`-42.43%`) | `+0.007137 ms` (`+19.38%`) | `.omx/benchmarks/gdn_decode_qk4_v8_d128_k_last/20260407T150733Z-opt3-launch-bounds.json` | `.omx/benchmarks/gdn_decode_qk4_v8_d128_k_last/logs/opt3-launch-bounds-e2da291.log` | `ap-3it2mMOGbLl8tCUkJdLGss` | Improved over naive but regressed against the current best. Reverted by `02bd564`. |
| 4 | Prefer L1 cache for decode launch | Yes | `e4bf14c` | Yes | `0.035059` | `-0.041302 ms` (`-54.09%`) | `-0.001767 ms` (`-4.80%`) | `.omx/benchmarks/gdn_decode_qk4_v8_d128_k_last/20260407T152055Z-opt4-prefer-l1.json` | `.omx/benchmarks/gdn_decode_qk4_v8_d128_k_last/logs/opt4-prefer-l1-e4bf14c.log` | `ap-2PGEzOLT8TUqt3RIINUYlm` | B200 cache-policy tuning beat the float4-only best and is now the current best. |
| 5 | Force decode shared-memory carveout to `0` | Yes | `d147d1c` | No | `0.036053` | `-0.040308 ms` (`-52.79%`) | `+0.000994 ms` (`+2.83%`) | `.omx/benchmarks/gdn_decode_qk4_v8_d128_k_last/20260407T152257Z-opt5-carveout-0.json` | `.omx/benchmarks/gdn_decode_qk4_v8_d128_k_last/logs/opt5-carveout-0-d147d1c.log` | `ap-NAR3qabvdSYL1j5lyjd1Xi` | Explicit carveout hurt the current best even though it still beat naive. Reverted by `b3bc99a`. |
| 6 | Combination: `64 threads x 2 rows/thread` on top of `float4 + shared q/k + prefer L1` | Yes | `39546de` | No | `0.035489` | `-0.040872 ms` (`-53.52%`) | `+0.000430 ms` (`+1.23%`) | `.omx/benchmarks/gdn_decode_qk4_v8_d128_k_last/20260407T152950Z-opt6-two-rows-thread.json` | `.omx/benchmarks/gdn_decode_qk4_v8_d128_k_last/logs/opt6-two-rows-thread-39546de.log` | `ap-vUXv1SG6qM59qhipgh9OhF` | Combination experiment. Very close, but still slower than the current best. Reverted by `4715a53`. |

# Prefill Optimization Ledger

Benchmark target: `gdn_prefill_qk4_v8_d128_k_last` on Modal `B200:1`

Benchmark config:

- `warmup_runs=1`
- `iterations=5`
- `num_trials=1`
- comparison metric: arithmetic mean of workload `latency_ms`

Current best prefill implementation:

- source optimization commit: `eb7d1f9` (`Try prefill state residency`)
- current branch state: equivalent prefill kernel restored after reverting the failed prefer-L1 experiment in `54e442a`
- current best average latency: `2.388144 ms`

Notes:

- `acd8ba7` (`Try prefill shared qk staging`) was attempted and later reverted by `1a3b4e1`, but no valid full-run artifact was produced because the previous Modal return path disconnected before artifact download. The stable remote-artifact path was added afterward.

| Order | Optimization | B200-specific | Commit | Kept | Avg latency (ms) | Delta vs naive | Delta vs prev best | Result JSON | Test log | Modal run | Notes |
| --- | --- | --- | --- | --- | ---: | --- | --- | --- | --- | --- | --- |
| 0 | Naive baseline | No | `cd5b3d8` | Yes | `30.959302` | baseline | baseline | `.omx/benchmarks/gdn_prefill_qk4_v8_d128_k_last/20260407T154057Z-prefill-baseline.json` | `.omx/benchmarks/gdn_prefill_qk4_v8_d128_k_last/logs/prefill-baseline-cd5b3d8.log` | `ap-T31nEnDj3JkrslVWANuKZb` | Baseline remeasured on B200 with the reduced benchmark config. |
| 1 | Vectorize prefill state updates with `float4` chunks | No | `6397b2c` | Yes | `5.755236` | `-25.204066 ms` (`-81.41%`) | `-25.204066 ms` (`-81.41%`) | `.omx/benchmarks/gdn_prefill_qk4_v8_d128_k_last/20260407T162242Z-prefill-opt1-float4.json` | `.omx/benchmarks/gdn_prefill_qk4_v8_d128_k_last/logs/prefill-opt1-float4.log` | `ap-k0YP4fK8b0gse4GWYDA83C` | First large prefill speedup. Kept. |
| 2 | Keep prefill state rows resident across the token loop | No | `eb7d1f9` | Yes | `2.388144` | `-28.571158 ms` (`-92.29%`) | `-3.367092 ms` (`-58.50%`) | `.omx/benchmarks/gdn_prefill_qk4_v8_d128_k_last/20260408T151628Z-prefill-opt3-state-residency.json` | `.omx/benchmarks/gdn_prefill_qk4_v8_d128_k_last/logs/prefill-opt3-state-residency.log` | `ap-UBQqDSrV2NtXaHJih1lqkW` | Current best. The key win is keeping each row state in registers across the whole sequence loop. |
| 3 | Prefer L1 cache for prefill launch on top of state residency | Yes | `84bac59` | No | `2.396034` | `-28.563268 ms` (`-92.26%`) | `+0.007890 ms` (`+0.33%`) | `.omx/benchmarks/gdn_prefill_qk4_v8_d128_k_last/20260408T152401Z-prefill-opt4-prefer-l1.json` | `.omx/benchmarks/gdn_prefill_qk4_v8_d128_k_last/logs/prefill-opt4-prefer-l1.log` | `ap-yIgLcndZ7RE9LyaddFhFPr` | Slight regression versus the current best. Reverted by `54e442a`. |
