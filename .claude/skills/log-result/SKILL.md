---
name: log-result
description: View optimization history and benchmark trends
user-invocable: true
allowed-tools: Read Bash
argument-hint: [decode|prefill] [last N]
---

# Optimization History & Trends

## Target
$ARGUMENTS

If first argument is `decode` or `prefill`, show only that kernel.
Otherwise show both. Second argument is number of entries (default: 10).

### Log file locations
- Decode bench history: `logs/decode/bench_history.jsonl`
- Decode optimization log: `logs/decode/optimization_log.md`
- Prefill bench history: `logs/prefill/bench_history.jsonl`
- Prefill optimization log: `logs/prefill/optimization_log.md`

### Steps

1. Read the relevant `bench_history.jsonl` and parse the last N entries
2. Read the relevant `optimization_log.md` for context on each change
3. Present:

```
## Benchmark History

### Decode Kernel
| # | Date       | Speedup | Latency (ms) | Git Hash | Notes              |
|---|------------|---------|--------------|----------|--------------------|
| 1 | 2026-04-06 | 396.4x  | 0.057        | abc1234  | Initial baseline   |

Trend: [improving/stagnating/regressing]
Best: #1 (396.4x) on 2026-04-06

### Prefill Kernel
| # | Date       | Speedup | Latency (ms) | Git Hash | Notes              |
|---|------------|---------|--------------|----------|--------------------|
| 1 | 2026-04-06 | 6.37x   | 34.28        | abc1234  | Initial baseline   |

Trend: [improving/stagnating/regressing]
Best: #1 (6.37x) on 2026-04-06

### Key Learnings
- [Aggregated insights from optimization log]
```
