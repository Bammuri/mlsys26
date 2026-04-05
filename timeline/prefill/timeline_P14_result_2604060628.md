# Prefill Timeline — P14 Result — 2604060628

## Optimization summary

이번 P14 iteration에서는 compile-time launch bounds를 명시해 compiler의 register allocation / occupancy tradeoff를 유도했다.

### Applied optimization

1. `gdn_prefill_kernel`에 `__launch_bounds__(128, 2)` 추가
2. `compute_gate_beta_kernel`에 `__launch_bounds__(256, 2)` 추가

핵심 의도:

> 수학이나 dataflow를 바꾸지 않고,
> compiler가 더 나은 register scheduling/occupancy 결정을 하도록 힌트를 주기

---

## Expected result vs actual result

### Expected before running

P11 full decision-gate baseline:

- avg latency: `4.732 ms`

예상:

1. quick 5 PASS 유지
2. full 100 PASS 유지
3. small positive gain 또는 no-op

### Actual results

#### 3-1. Quick working gate

- `PASSED=5/5`
- avg latency: `0.205 ms`

#### 3-2. Full decision gate

- `PASSED=100/100`
- avg latency: `4.725 ms`
- avg speedup: `48.54x`
- worst abs error: `1.22e-04`
- worst rel error: `3.20e-01`
- wall time: `275.64s`

대표 로그 예시:

- `Workload d0ce7b5d...: PASSED | 12.810 ms | 117.93x speedup | abs_err=1.53e-05, rel_err=1.75e-01`
- `Workload ce832e76...: PASSED | 2.376 ms | 33.24x speedup | abs_err=3.81e-06, rel_err=1.41e-02`
- `Workload 07aa7922...: PASSED | 29.665 ms | 32.93x speedup | abs_err=3.05e-05, rel_err=8.62e-02`

---

## Judgement

이번 iteration은 **좋아졌다**고 판단한다.

### Decision rule used

판단 기준은:

> **full 100-workload arithmetic-mean latency**

이다.

### Why this is a keep

P11 → P14:

- avg latency: `4.732 ms` → `4.725 ms`
- 약 **0.15% 개선**

개선 폭은 작지만,
현재 규칙에서는 full 평균 latency가 더 나아졌으므로 keep한다.

또한 100/100 correctness도 유지했다.

---

## Detailed analysis: why it likely improved

### 1. This is a compiler-guidance win, not a math win

이번 iteration은 kernel 수학을 전혀 바꾸지 않았다.
따라서 개선은:

- register allocation
- instruction scheduling
- occupancy tradeoff

같은 compiler/codegen 측면에서 나왔을 가능성이 크다.

### 2. The kernel is shared-heavy enough that launch-bounds hints can matter

현재 kernel은:

- shared state tile 사용량 큼
- compute와 memory가 모두 많은 편

이런 kernel에서는 compiler가 register를 과도하게 쓰면 occupancy가 떨어질 수 있다.
launch bounds는 그 선택을 조절할 수 있으므로,
작지만 실제 개선이 가능하다.

### 3. Why the gain is modest

launch bounds는 구조적 병목을 제거하는 최적화가 아니다.
즉,

- state update 방식
- long-tail workload shape
- row-per-thread structure

같은 근본 구조는 그대로다.

그래서 개선 폭이 작은 것은 자연스럽다.

### 4. Why it is still worth keeping

작지만 reproducible한 full 기준 개선이고,
정확도 손실 없이 baseline을 조금 더 좋게 만들었다.

이런 종류의 개선은 이후 큰 최적화 위에 누적될 수 있으므로 가치가 있다.

---

## Full 100-workload pref log

```text
✓ Initialized. View run at 
https://modal.com/apps/shjj1504/main/ap-8Xam4CEp7FmTsFbFVISSwP
✓ Created objects.
├── 🔨 Created mount /home/hyu/flashinfer/mlsys26/scripts/run_modal.py
└── 🔨 Created function run_benchmark.
[2026-04-06T06:28:43] Packing solution from source files...
Solution packed: /home/hyu/flashinfer/mlsys26/solution.json
  Name: my-team-solution-v1
  Definition: gdn_prefill_qk4_v8_d128_k_last
  Author: team-name
  Config language: cuda
  Runtime language: cuda
[2026-04-06T06:28:43] Validating solution JSON...
[2026-04-06T06:28:43] Loaded solution my-team-solution-v1 (gdn_prefill_qk4_v8_d128_k_last) in 0.00s
[2026-04-06T06:28:43] Decision-gate mode enabled: warmup_runs=1, iterations=5, num_trials=1, use_isolated_runner=False
[2026-04-06T06:28:43] Dispatching benchmark to Modal B200...

==========
== CUDA ==
==========

CUDA Version 13.0.2

Container image Copyright (c) 2016-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

This container image and its contents are governed by the NVIDIA Deep Learning Container License.
By pulling and using the container, you accept the terms and conditions of this license:
https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license

A copy of this license is made available in this container at /NGC-DL-CONTAINER-LICENSE for your convenience.

[2026-04-05T21:28:48] Remote benchmark start: solution=my-team-solution-v1, definition=gdn_prefill_qk4_v8_d128_k_last
[2026-04-05T21:28:48] BenchmarkConfig(warmup_runs=1, iterations=5, num_trials=1)
[2026-04-05T21:28:48] Loading trace set from /data/data/mlsys26-contest
[2026-04-05T21:28:49] Loaded trace set in 0.61s
[2026-04-05T21:28:49] Running benchmark across 100 workloads
[2026-04-05T21:33:18] Benchmark completed in 268.88s
[2026-04-06T06:33:18] Received benchmark results in 275.63s

gdn_prefill_qk4_v8_d128_k_last:
  workloads: 100
  status counts: PASSED=100
  avg latency: 4.725 ms
  avg speedup: 48.54x
  worst abs error: 1.22e-04
  worst rel error: 3.20e-01
  Workload 77daf91d...: PASSED | 0.073 ms | 20.54x speedup | abs_err=4.47e-08, rel_err=9.53e-03
  Workload ba08a83e...: PASSED | 0.749 ms | 33.87x speedup | abs_err=4.77e-07, rel_err=7.50e-03
  Workload c7846f96...: PASSED | 0.751 ms | 33.37x speedup | abs_err=3.81e-06, rel_err=1.48e-02
  Workload d0ce7b5d...: PASSED | 12.810 ms | 117.93x speedup | abs_err=1.53e-05, rel_err=1.75e-01
  Workload 5b8a0e4b...: PASSED | 19.919 ms | 75.94x speedup | abs_err=1.53e-05, rel_err=8.13e-02
  Workload 5d3fc66a...: PASSED | 19.944 ms | 74.24x speedup | abs_err=6.10e-05, rel_err=1.99e-01
  Workload 4b6143dd...: PASSED | 4.828 ms | 154.70x speedup | abs_err=1.53e-05, rel_err=2.84e-02
  Workload 5835a2bc...: PASSED | 26.540 ms | 56.76x speedup | abs_err=1.53e-05, rel_err=3.06e-02
  Workload cc310f94...: PASSED | 16.938 ms | 86.82x speedup | abs_err=6.10e-05, rel_err=1.24e-01
  Workload d49df0b2...: PASSED | 14.742 ms | 97.53x speedup | abs_err=1.53e-05, rel_err=1.19e-01
  Workload e9e1e445...: PASSED | 11.545 ms | 125.65x speedup | abs_err=7.63e-06, rel_err=2.95e-02
  Workload b8c8dc3c...: PASSED | 11.503 ms | 126.25x speedup | abs_err=6.10e-05, rel_err=2.06e-01
  Workload a9540651...: PASSED | 18.092 ms | 80.55x speedup | abs_err=6.10e-05, rel_err=6.65e-02
  Workload 06f21bb1...: PASSED | 9.244 ms | 156.52x speedup | abs_err=1.53e-05, rel_err=4.09e-02
  Workload c2931c92...: PASSED | 18.406 ms | 78.10x speedup | abs_err=1.22e-04, rel_err=1.62e-01
  Workload 618df04a...: PASSED | 16.581 ms | 88.07x speedup | abs_err=6.10e-05, rel_err=1.28e-01
  Workload 26244fb4...: PASSED | 15.680 ms | 91.31x speedup | abs_err=7.63e-06, rel_err=1.52e-01
  Workload a2629e02...: PASSED | 15.616 ms | 89.91x speedup | abs_err=6.10e-05, rel_err=3.20e-01
  Workload 9a5d694b...: PASSED | 10.865 ms | 129.44x speedup | abs_err=6.10e-05, rel_err=3.95e-02
  Workload 410794d4...: PASSED | 12.967 ms | 110.07x speedup | abs_err=1.53e-05, rel_err=1.71e-02
  Workload 7ba9d519...: PASSED | 15.141 ms | 44.73x speedup | abs_err=6.10e-05, rel_err=5.69e-02
  Workload 043e74e4...: PASSED | 0.135 ms | 43.13x speedup | abs_err=1.19e-07, rel_err=6.68e-03
  Workload ef9515b6...: PASSED | 0.120 ms | 25.91x speedup | abs_err=1.19e-07, rel_err=7.04e-03
  Workload f622a11d...: PASSED | 0.123 ms | 26.24x speedup | abs_err=4.77e-07, rel_err=1.98e-02
  Workload 1cf8e175...: PASSED | 0.160 ms | 45.10x speedup | abs_err=9.54e-07, rel_err=4.95e-03
  Workload 9343fd82...: PASSED | 0.297 ms | 31.03x speedup | abs_err=1.91e-06, rel_err=7.65e-03
  Workload f4926229...: PASSED | 7.260 ms | 31.81x speedup | abs_err=6.10e-05, rel_err=4.48e-02
  Workload 109addb1...: PASSED | 13.841 ms | 39.74x speedup | abs_err=6.10e-05, rel_err=1.76e-02
  Workload c5257f65...: PASSED | 4.995 ms | 35.43x speedup | abs_err=3.81e-06, rel_err=3.35e-02
  Workload f5619793...: PASSED | 4.997 ms | 33.27x speedup | abs_err=6.10e-05, rel_err=6.50e-02
  Workload fdf5f1f4...: PASSED | 0.189 ms | 44.16x speedup | abs_err=1.19e-07, rel_err=4.93e-03
  Workload 87bff084...: PASSED | 0.544 ms | 66.06x speedup | abs_err=3.81e-06, rel_err=6.89e-03
  Workload e92dafeb...: PASSED | 0.605 ms | 67.07x speedup | abs_err=9.54e-07, rel_err=2.57e-02
  Workload 1d0cc342...: PASSED | 1.237 ms | 40.66x speedup | abs_err=3.81e-06, rel_err=5.42e-02
  Workload 1b441950...: PASSED | 0.426 ms | 36.81x speedup | abs_err=9.54e-07, rel_err=1.07e-02
  Workload 19c6ab20...: PASSED | 0.428 ms | 37.39x speedup | abs_err=7.63e-06, rel_err=8.99e-03
  Workload 25d9c14d...: PASSED | 0.413 ms | 57.74x speedup | abs_err=1.91e-06, rel_err=1.41e-02
  Workload 3215fe5f...: PASSED | 1.675 ms | 35.25x speedup | abs_err=3.81e-06, rel_err=7.41e-03
  Workload 6f1ad833...: PASSED | 11.088 ms | 32.57x speedup | abs_err=6.10e-05, rel_err=5.49e-02
  Workload e44ba4d3...: PASSED | 5.960 ms | 45.79x speedup | abs_err=7.63e-06, rel_err=1.83e-02
  Workload fc7a2bcb...: PASSED | 0.548 ms | 44.95x speedup | abs_err=2.38e-07, rel_err=2.13e-02
  Workload 5d26ac5b...: PASSED | 0.549 ms | 44.16x speedup | abs_err=3.81e-06, rel_err=1.94e-02
  Workload ed66c791...: PASSED | 0.652 ms | 31.58x speedup | abs_err=1.91e-06, rel_err=9.77e-03
  Workload ba95d412...: PASSED | 0.864 ms | 37.32x speedup | abs_err=1.53e-05, rel_err=1.36e-02
  Workload 078a41ea...: PASSED | 0.329 ms | 42.46x speedup | abs_err=2.38e-07, rel_err=1.77e-02
  Workload d2b5a221...: PASSED | 0.202 ms | 73.24x speedup | abs_err=9.54e-07, rel_err=1.35e-02
  Workload aaa378be...: PASSED | 7.985 ms | 37.73x speedup | abs_err=7.63e-06, rel_err=1.39e-02
  Workload c2bb4f66...: PASSED | 7.985 ms | 37.22x speedup | abs_err=1.53e-05, rel_err=9.20e-02
  Workload f2f01c2c...: PASSED | 0.144 ms | 44.17x speedup | abs_err=8.94e-08, rel_err=4.96e-03
  Workload 15856e8c...: PASSED | 14.322 ms | 37.56x speedup | abs_err=7.63e-06, rel_err=1.59e-02
  Workload a39aa135...: PASSED | 0.361 ms | 30.87x speedup | abs_err=2.38e-07, rel_err=2.04e-02
  Workload 339a7ff4...: PASSED | 0.124 ms | 26.23x speedup | abs_err=4.77e-07, rel_err=5.52e-03
  Workload d8f4a9ae...: PASSED | 0.225 ms | 29.68x speedup | abs_err=1.19e-07, rel_err=9.66e-03
  Workload d3dc3577...: PASSED | 0.223 ms | 29.55x speedup | abs_err=3.58e-07, rel_err=7.72e-03
  Workload ce832e76...: PASSED | 2.376 ms | 33.24x speedup | abs_err=3.81e-06, rel_err=1.41e-02
  Workload 6fbc155c...: PASSED | 3.627 ms | 38.87x speedup | abs_err=7.63e-06, rel_err=7.84e-03
  Workload a87ded8a...: PASSED | 8.131 ms | 59.47x speedup | abs_err=1.22e-04, rel_err=5.11e-02
  Workload 62447caf...: PASSED | 0.197 ms | 26.74x speedup | abs_err=1.49e-07, rel_err=1.53e-02
  Workload fd072ba6...: PASSED | 0.395 ms | 49.40x speedup | abs_err=3.81e-06, rel_err=1.20e-02
  Workload 35ea9bbe...: PASSED | 0.398 ms | 48.19x speedup | abs_err=1.91e-06, rel_err=3.63e-02
  Workload 1aa8cf18...: PASSED | 0.133 ms | 50.65x speedup | abs_err=5.96e-08, rel_err=7.61e-03
  Workload d5f5c00c...: PASSED | 0.202 ms | 28.70x speedup | abs_err=1.49e-08, rel_err=1.75e-03
  Workload d5aa60dc...: PASSED | 1.130 ms | 38.89x speedup | abs_err=3.81e-06, rel_err=4.68e-02
  Workload 28b70283...: PASSED | 0.165 ms | 27.32x speedup | abs_err=1.19e-07, rel_err=8.86e-03
  Workload 73b8cc85...: PASSED | 0.552 ms | 30.94x speedup | abs_err=1.91e-06, rel_err=7.04e-03
  Workload 2683c087...: PASSED | 0.551 ms | 30.28x speedup | abs_err=7.63e-06, rel_err=1.63e-02
  Workload 4b94d568...: PASSED | 2.447 ms | 38.09x speedup | abs_err=3.81e-06, rel_err=1.64e-02
  Workload a0eb2dc2...: PASSED | 0.523 ms | 28.85x speedup | abs_err=3.81e-06, rel_err=7.63e-03
  Workload f3d30cb9...: PASSED | 0.425 ms | 35.14x speedup | abs_err=2.98e-07, rel_err=1.80e-02
  Workload 7a7deca8...: PASSED | 0.456 ms | 30.67x speedup | abs_err=3.81e-06, rel_err=7.69e-03
  Workload 977d19f8...: PASSED | 0.196 ms | 30.49x speedup | abs_err=1.19e-07, rel_err=7.04e-03
  Workload 02c1e5f0...: PASSED | 0.198 ms | 27.50x speedup | abs_err=7.63e-06, rel_err=6.94e-03
  Workload 5a91aa02...: PASSED | 1.486 ms | 46.56x speedup | abs_err=3.81e-06, rel_err=1.48e-02
  Workload 8e7ef744...: PASSED | 0.103 ms | 24.03x speedup | abs_err=2.98e-08, rel_err=2.39e-03
  Workload 85d7becb...: PASSED | 0.133 ms | 27.21x speedup | abs_err=3.58e-07, rel_err=6.89e-03
  Workload e286a4f4...: PASSED | 6.201 ms | 49.35x speedup | abs_err=3.81e-06, rel_err=3.27e-02
  Workload 08d4f2c4...: PASSED | 4.799 ms | 33.96x speedup | abs_err=7.63e-06, rel_err=7.79e-03
  Workload 9c1ef562...: PASSED | 4.794 ms | 34.98x speedup | abs_err=3.05e-05, rel_err=5.33e-02
  Workload bfd8f7b6...: PASSED | 1.277 ms | 35.45x speedup | abs_err=7.63e-06, rel_err=1.18e-02
  Workload c358edcd...: PASSED | 10.642 ms | 32.84x speedup | abs_err=1.53e-05, rel_err=7.81e-03
  Workload f203fdcd...: PASSED | 0.292 ms | 32.13x speedup | abs_err=2.38e-07, rel_err=5.39e-03
  Workload 33a38713...: PASSED | 0.411 ms | 51.26x speedup | abs_err=1.34e-07, rel_err=1.82e-02
  Workload 3a77dfec...: PASSED | 0.350 ms | 38.94x speedup | abs_err=9.54e-07, rel_err=7.04e-03
  Workload ea27be17...: PASSED | 0.352 ms | 37.55x speedup | abs_err=7.15e-07, rel_err=7.31e-02
  Workload 49ef89d2...: PASSED | 0.393 ms | 32.06x speedup | abs_err=1.91e-06, rel_err=5.34e-02
  Workload 056224b8...: PASSED | 0.397 ms | 32.70x speedup | abs_err=2.38e-07, rel_err=6.62e-03
  Workload 685d26ff...: PASSED | 0.106 ms | 25.53x speedup | abs_err=1.19e-07, rel_err=5.69e-03
  Workload 352c9ace...: PASSED | 2.797 ms | 30.38x speedup | abs_err=1.91e-06, rel_err=2.40e-02
  Workload 2c9693b4...: PASSED | 0.298 ms | 27.62x speedup | abs_err=2.38e-07, rel_err=1.84e-02
  Workload 27f44fd6...: PASSED | 0.297 ms | 30.12x speedup | abs_err=1.91e-06, rel_err=1.22e-02
  Workload 07aa7922...: PASSED | 29.589 ms | 35.56x speedup | abs_err=3.05e-05, rel_err=8.62e-02
  Workload eaa0fd47...: PASSED | 0.115 ms | 25.37x speedup | abs_err=9.54e-07, rel_err=6.45e-03
  Workload f105eda8...: PASSED | 2.210 ms | 53.53x speedup | abs_err=6.10e-05, rel_err=1.44e-02
  Workload cd979341...: PASSED | 0.358 ms | 40.25x speedup | abs_err=1.79e-07, rel_err=7.03e-03
  Workload 43bf9699...: PASSED | 4.720 ms | 37.21x speedup | abs_err=7.63e-06, rel_err=1.64e-02
  Workload 54856fec...: PASSED | 4.713 ms | 35.63x speedup | abs_err=3.05e-05, rel_err=9.22e-02
  Workload 2ba465c0...: PASSED | 0.113 ms | 48.70x speedup | abs_err=2.38e-07, rel_err=7.69e-03
  Workload 1efaf2a9...: PASSED | 0.181 ms | 44.61x speedup | abs_err=2.98e-08, rel_err=3.81e-03
  Workload a01a3f93...: PASSED | 11.156 ms | 35.20x speedup | abs_err=1.22e-04, rel_err=3.40e-02
  Workload cc241d2e...: PASSED | 0.161 ms | 28.04x speedup | abs_err=1.49e-07, rel_err=6.82e-03
[2026-04-06T06:33:18] Local entrypoint finished in 275.64s
[2026-04-05T21:33:18] Remote benchmark finished in 269.49s
Stopping app - local entrypoint completed.
✓ App completed. View run at 
https://modal.com/apps/shjj1504/main/ap-8Xam4CEp7FmTsFbFVISSwP

```
