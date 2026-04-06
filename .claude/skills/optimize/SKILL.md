---
name: optimize
description: Full optimization loop - research idea, implement kernel change, benchmark. Specify decode or prefill.
user-invocable: true
allowed-tools: WebSearch WebFetch Read Write Edit Grep Glob Bash Agent
argument-hint: [decode|prefill] [optimization topic]
model: claude-opus-4-6
effort: high
---

# GDN Kernel Optimization Loop

## Target & Topic
- Kernel: $0 (decode or prefill)
- Topic: $1 (optional - specific optimization focus)

## Context Files
- Decode kernel: `gdn_decode_qk4_v8_d128_k_last/solution/cuda/kernel.cu`
- Decode bench history: `logs/decode/bench_history.jsonl`
- Decode optimization log: `logs/decode/optimization_log.md`
- Prefill kernel: `gdn_prefill_qk4_v8_d128_k_last/solution/cuda/kernel.cu`
- Prefill bench history: `logs/prefill/bench_history.jsonl`
- Prefill optimization log: `logs/prefill/optimization_log.md`

## Agent Pipeline

Each step uses a specific agent. Spawn them via the Agent tool with the indicated `subagent_type`.

### Step 1: Assess Current State
Read the current kernel code and the target kernel's `logs/<kernel>/bench_history.jsonl` directly (no agent needed).
Identify the current bottleneck (compute-bound? memory-bound? latency?).

### Step 2: Research — spawn `researcher` agent
Spawn the **`researcher`** agent (subagent_type=researcher) with:
- The target kernel (decode/prefill) and topic
- Current kernel code summary and bottleneck analysis from Step 1
- Past optimization attempts from `logs/<kernel>/optimization_log.md`

The researcher will return 2-3 ranked optimization ideas with expected impact.

### Step 3: Select Optimization
Autonomously decide which optimization to implement:
1. Review `logs/<kernel>/optimization_log.md` and `logs/<kernel>/bench_history.jsonl` for past attempts and their outcomes
2. Avoid repeating ideas that already failed or regressed
3. Pick the highest-impact idea from Step 2 that hasn't been tried yet
4. Briefly state your choice and rationale (no user approval needed)

### Step 4: Implement — spawn `kernel-writer` agent
Spawn the **`kernel-writer`** agent (subagent_type=kernel-writer) with:
- The selected optimization idea and implementation sketch from Step 3
- The exact file path to modify
- Clear constraints (correctness first, surgical changes, one optimization only)

### Step 5: Benchmark — spawn `evaluator` agent
Spawn the **`evaluator`** agent (subagent_type=evaluator) with:
- The target subfolder name
- Instructions to run `conda run -n fi-bench modal run scripts/run_modal_subfolder.py --subfolder <subfolder>`
- Instructions to parse results and append to `logs/<kernel>/bench_history.jsonl`

### Step 6: Evaluate & Decide
Compare the evaluator's results with previous benchmarks:
- If improved: update optimization log (Step 7), then commit using `/conventional-commit`
- If regressed or no change: analyze why, consider reverting or adjusting
- If correctness fails: revert and try the next-ranked idea from Step 2

### Step 7: Update Optimization Log
Append to `logs/<kernel>/optimization_log.md`:

```markdown
## [Date] - [Optimization Name]
- **Idea**: [Brief description]
- **Result**: [speedup before] → [speedup after] ([+/-X%])
- **Git branch**: opt/<kernel>/<short-desc>
- **Status**: accepted/reverted/needs-work
- **Learnings**: [What we learned for future iterations]
```

## Important Rules
- NEVER break correctness for performance
- Always benchmark before AND after on Modal B200
- Keep changes atomic - one optimization per iteration
- Decide autonomously — do NOT ask the user for approval mid-loop
- If correctness fails after implementation, revert and try the next-ranked idea
