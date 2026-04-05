# MLSys 2026 Contest: NVIDIA B200 Optimization Directives

## 1. System and Hardware Context
The target execution environment is a high-performance bare-metal server. Code generation and kernel optimizations must be strictly tailored to this architecture.

* **Host CPU:** 2x Intel Xeon Platinum 8570 (x86_64 architecture)
* **Logical Cores:** 224 (112 physical cores, Hyperthreading enabled)
* **NUMA Topology:** 2 NUMA nodes
* **System Memory:** 2 TiB DDR5 RAM
* **GPU Configuration:** 8x NVIDIA B200 (Blackwell architecture)
* **GPU Interconnect:** NVLink
* **Host-to-Device Interconnect:** PCIe Gen5 x16

---

## 2. Evaluation Environment Constraints
The evaluation pipeline enforces specific runtime conditions. Do not optimize for scenarios outside these parameters.

* **Clocks:** GPU clocks are strictly locked. Performance is highly deterministic; focus on hardware-level metrics (e.g., L1/L2 cache hit rates, SM occupancy).
* **Isolation:** Each solution runs in an isolated subprocess concurrently with other evaluations.
* **Overhead Exclusions:** The benchmarking tool uses `--warmup-runs 1`. Do not sacrifice steady-state kernel throughput to optimize initialization, memory allocation, or JIT compilation times.

---

## 3. Target Workloads
The evaluation targets FlashInfer-based GDN algorithms. The standard evaluation commands are defined below.

### GDN Decode
Command: `flashinfer-bench run --local ./contest-dataset --definitions gdn_decode_qk4_v8_d128_k_last --save-results --use-isolated-runner --log-level INFO --resume --timeout 300`

### GDN Prefill
Command: `flashinfer-bench run --local ./contest-dataset --definitions gdn_prefill_qk4_v8_d128_k_last --save-results --use-isolated-runner --log-level INFO --resume --timeout 300 --warmup-runs 1 --iterations 5 --num-trials 3`

---

## 4. Primary Optimization Directives for Agent Code Generation
When generating, refactoring, or optimizing C++/CUDA code for this project, apply the following strategies:

* **Static Dimension Optimization:** The workload definitions explicitly specify `d128` (head dimension = 128) and `k_last` layout. Hardcode loop unrolling, shared memory padding, and register allocation specifically for $D=128$ at compile time to maximize 4th-generation Tensor Core utilization.
* **PCIe Bottleneck Mitigation:** The Host-to-Device link is PCIe Gen5 x16, which is slow relative to the B200 compute throughput. Strictly minimize Host-to-Device (H2D) and Device-to-Host (D2H) memory transfers.
* **Host CPU Pipelining:** Utilize asynchronous execution using CUDA Streams. Apply thread-pinning to bind host CPU threads to the same NUMA node as their target GPU to avoid QPI/UPI cross-socket latency.
* **Host-Side Preprocessing:** Since the host is a powerful x86_64 Intel Xeon system, utilize AVX-512 SIMD instructions if any CPU-side tensor preprocessing or scheduling logic is required.
* **Device-Side Fusion:** Maximize kernel fusion. Keep data on the device (GPU VRAM or SRAM) as long as possible to avoid hitting the PCIe bandwidth limit during concurrent subprocess execution.


# MLSys 2026 Contest: Expanded Knowledge Base & Reference Guide

This document categorizes the provided reference materials and outlines how the AI agent should utilize them for optimizing the GDN (Gated Delta Network) kernels on the NVIDIA B200 architecture.

## 1. Algorithmic Context: Gated Delta Networks (GDN)
The target workloads are based on Gated Delta Networks and DeepSeek Sparse Attention (DSA) mechanisms.
* **Core Logic:** The agent must reference the official PyTorch implementations (`NVlabs/GatedDeltaNet`, ICLR 2025) and standard formulations to ensure algorithmic equivalence during C++/CUDA translation.
* **Long-Context Optimization:** Utilize the mathematical properties of DeltaNets (e.g., chunk-wise parallelization, recurrent states) to optimize memory access patterns during the `Prefill` phase.

## 2. Blackwell (B200) Microarchitecture Utilization
To extract maximum performance, the agent must leverage Blackwell-specific hardware features detailed in the technical guides.
* **NVFP4 & Low-Precision Inference:** Evaluate the feasibility of using NVFP4 (NVIDIA's 4-bit floating-point format) to double the Tensor Core throughput and reduce memory bandwidth requirements, assuming the contest precision rules allow it.
* **Blackwell Decompression Engine:** Utilize `nvCOMP` to offload data decompression to the dedicated hardware decompression engine, reducing host CPU load and PCIe transfer times if data is streamed in compressed formats.
* **CUDA 13.x Features:** The agent must utilize CUDA Toolkit 13.0/13.1 features. Specifically, leverage **NVIDIA CUDA Tile** for advanced thread-block level data routing and synchronization, bypassing older, less efficient Shared Memory patterns.

## 3. Current Project Status & Baselines
The agent should be aware of the current codebase state to avoid regressing performance and to understand the target baseline.
* **GDN Decode Benchmark:** The current implementation (tracked in PR #7 by `hjunkim`) has already achieved an ultra-low latency of **0.013ms** for GDN Decode. The agent must use this 0.013ms boundary as the strict baseline. Any generated refactoring must not increase this latency.
* **Evaluation Tooling:** All kernel outputs and performance metrics must be strictly validated against `FlashInfer-Bench`.

## 4. Multi-GPU & System Considerations
* **GB200 NVL vs. Bare-metal HGX:** While some documents reference GB200/Grace systems, the agent must strictly apply the *HGX B200 (x86_64 host + PCIe Gen5)* constraints defined in the primary directives. Ignore ARM-specific (Grace) tuning suggestions.

* **MIG (Multi-Instance GPU):** Ensure the kernel configuration does not make assumptions about having the entire GPU exclusively if the evaluation environment partitions resources using MIG, though `locked GPU clocks` and `isolated subprocess` suggest full GPU allocation per run.

## Actionable Directives for the Agent
1.  **Do not reinvent the wheel:** Start by analyzing the 0.013ms GDN Decode PR to understand the winning memory access patterns before attempting to optimize the Prefill kernel.
2.  **Focus on Memory Tiering:** Map out the data flow from DDR5 (Host) -> PCIe -> HBM3e (Device) -> L2 Cache -> Shared Memory -> Registers. Eliminate any unnecessary round-trips.
3.  **Precision Exploitation:** Automatically insert PTX instructions for NVFP4/FP8 Tensor Core MMA (Matrix Multiply-Accumulate) operations where applicable.

upsteam base line

gdn_prefill_qk4_v8_d128_k_last:
    workloads: 100
    status counts: PASSED=100
    avg latency: 0.317 ms
    avg speedup: 576.79x
    worst abs error: 9.09e-03
    worst rel error: 3.29e+03
  end-to-endLocal entrypoint finished in 1496.64s

gdn_decode_qk4_v8_d128_k_last
  - workloads: 54
  - status counts: PASSED=54
  - avg latency: 0.034 ms
  - avg speedup: 843.12x
  - worst abs error: 3.05e-05
  - worst rel error: 3.48e-01
  - remote benchmark: 241.76s
  - end-to-end: 247.82s