---
name: research
description: Search for CUDA kernel optimization ideas for GDN (Gated Delta Net). Uses web search and paper search.
user-invocable: true
allowed-tools: WebSearch WebFetch Read Grep Glob Bash Agent
argument-hint: [decode|prefill] [topic]
model: claude-opus-4-6
---

# GDN CUDA Kernel Optimization Research

## Goal
Find actionable optimization ideas for the GDN (Gated Delta Network) CUDA kernel targeting NVIDIA B200 (sm100a).

## Target & Topic
$ARGUMENTS

First argument should be `decode` or `prefill` to focus research.
If no topic specified, identify the most impactful optimization opportunity based on current kernel state.

## Context
Read the kernel files for details:
- Decode kernel: `gdn_decode_qk4_v8_d128_k_last/solution/cuda/kernel.cu`
- Decode bench history: `logs/decode/bench_history.jsonl`
- Decode optimization log: `logs/decode/optimization_log.md`
- Prefill kernel: `gdn_prefill_qk4_v8_d128_k_last/solution/cuda/kernel.cu`
- Prefill bench history: `logs/prefill/bench_history.jsonl`
- Prefill optimization log: `logs/prefill/optimization_log.md`

## Execution
**Spawn the `researcher` agent** (subagent_type=researcher) with the topic and context below.
The researcher agent has WebSearch and WebFetch access — delegate ALL web research to it.

## Research Process (for the researcher agent)

1. **Current state**: Read the target kernel code to understand what's already implemented
2. **Benchmark history**: Check the target kernel's `logs/<kernel>/bench_history.jsonl` for latest performance numbers
3. **Optimization history**: Check `logs/<kernel>/optimization_log.md` to avoid repeating past attempts
4. **Web search**: Search for relevant optimization techniques:
   - "CUDA kernel optimization [topic] sm100a Blackwell"
   - "linear attention CUDA kernel efficient"
   - "delta rule recurrent state update GPU"
   - "TMA tensor memory accelerator Blackwell"
   - FlashInfer source code for GDN implementation
5. **Paper search**: Search for relevant papers:
   - "Gated Delta Networks" (original paper)
   - "Flash Linear Attention" (chunked linear attention)
   - "CUTLASS sm100a" (Blackwell-specific GEMM)
6. **FlashInfer source**: Check https://github.com/flashinfer-ai/flashinfer for GDN kernel implementations

## Output Format

Output a structured research report:

```
## Research: [Topic] ([decode/prefill])

### Current Performance
- [kernel]: [latency] ms, [speedup]x

### Optimization Ideas (ranked by expected impact)

#### Idea 1: [Name]
- **Expected impact**: [high/medium/low], estimated [X]x improvement
- **Description**: [What to do]
- **Implementation sketch**: [Key code changes]
- **References**: [Papers, code links]
- **Risk**: [What could go wrong]

#### Idea 2: ...

### Recommendation
[Which idea to implement first and why]
```
