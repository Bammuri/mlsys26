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

**NCU Profile**: Run `modal run scripts/profile_kernel.py --kernel <decode|prefill>` to collect hardware-level metrics (throughput, occupancy, memory bandwidth, register usage, spills, cache hit rates). Parse the NCU output to identify the precise bottleneck:
- Compute-bound: high compute throughput %, low memory throughput %
- Memory-bound: high memory/DRAM throughput %, low compute %
- Latency-bound: low both, low waves/SM, low achieved occupancy
- Register pressure: high register count, low theoretical occupancy, spills > 0

Use these NCU metrics (not guesses) to drive the bottleneck analysis for Step 2.

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

### Step 6: Post-Benchmark Profiling & Evaluate
After benchmarking, run NCU profiling to understand **why** performance changed:

1. **Run NCU profile**: `modal run scripts/profile_kernel.py --kernel <decode|prefill>`
2. **Compare with Step 1 baseline profile** — check these key metrics:
   - Did the **targeted bottleneck** actually improve? (e.g., if we aimed to reduce register pressure, did register count/spills decrease?)
   - Did any **other metrics regress**? Common tradeoff patterns:
     - Occupancy ↑ but register spills ↑ → net negative from L1 cache thrashing
     - Compute throughput ↑ but memory throughput ↓ → data supply can't keep up
     - Shared memory usage ↓ but bank conflicts ↑ → access pattern degraded
     - Instruction count ↓ but warp stall cycles ↑ → introduced dependency chains
3. **Decide** based on both benchmark numbers AND profile analysis:
   - If improved AND profile confirms intended optimization worked cleanly: update optimization log (Step 7), then commit using `/conventional-commit`
   - If improved BUT profile shows a new bottleneck was introduced: still accept, but note the new bottleneck in the optimization log as a follow-up target
   - If regressed or no change: use profile diff to explain **why** (e.g., "occupancy improved 50%→75% but register spills increased 0→24, causing L1 thrashing"), update bench_history.jsonl entry with `"reverted": true`, revert kernel
   - If correctness fails: update bench_history.jsonl entry with `"reverted": true`, revert kernel, and try the next-ranked idea from Step 2

### Step 7: Update Optimization Log
Append to `logs/<kernel>/optimization_log.md`:

```markdown
## [Date] - [Optimization Name]
- **Idea**: [Brief description]
- **Result**: [speedup before] → [speedup after] ([+/-X%])
- **Profile diff**: [Key metric changes, e.g., "occupancy 50%→75%, registers 64→48, spills 0→0"]
- **Tradeoffs**: [Any metrics that regressed, or "none"]
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
