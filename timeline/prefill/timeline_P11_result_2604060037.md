# Prefill Timeline — P11 Result — 2604060037

## Optimization summary

이번 P11 iteration에서는 gate/beta 계산을 main kernel 밖으로 분리해서 별도 GPU precompute kernel에서 미리 계산했다.

### Applied optimization

1. `compute_gate_beta_kernel` 추가
2. `gate` / `beta` temporary float tensors 생성
3. main kernel은 precomputed `gate/beta`만 읽음
4. token loop 안의 thread0 scalar exp/softplus/sigmoid 계산 제거

핵심 의도:

> token마다 CTA 안에서 반복되던 serial scalar math를 떼어내고,
> gate/beta 계산을 GPU-wide parallel precompute로 바꾸기

---

## Expected result vs actual result

### Expected before running

P6 full decision-gate baseline:

- avg latency: `4.913 ms`

예상:

1. quick 5 PASS 유지
2. full 100 PASS 유지
3. long-tail에서 의미 있는 개선 가능

### Actual results

#### 3-1. Quick working gate

- `PASSED=5/5`
- avg latency: `0.205 ms`
- avg speedup: `23.33x`
- worst abs error: `1.91e-06`
- worst rel error: `2.29e-02`

#### 3-2. Full decision gate

- `PASSED=100/100`
- avg latency: `4.732 ms`
- avg speedup: `51.12x`
- worst abs error: `1.22e-04`
- worst rel error: `2.46e-01`
- wall time: `285.50s`

대표 로그 예시:

- `Workload d0ce7b5d...: PASSED | 13.163 ms | 113.44x speedup | abs_err=1.53e-05, rel_err=2.02e-01`
- `Workload ce832e76...: PASSED | 2.339 ms | 36.15x speedup | abs_err=3.81e-06, rel_err=2.34e-02`
- `Workload 07aa7922...: PASSED | 29.142 ms | 35.80x speedup | abs_err=1.53e-05, rel_err=8.62e-02`

---

## Judgement

이번 iteration은 **좋아졌다**고 판단한다.

### Decision rule used

판단 기준은:

> **full 100-workload arithmetic-mean latency**

이다.

### Why this is a keep

P6 → P11:

- avg latency: `4.913 ms` → `4.732 ms`
- 약 **3.68% 개선**

즉,

> full 기준으로 유의미한 개선이 확인되었다.

또한:

- `PASSED=100/100`
- correctness stability 유지

이므로 keep 조건을 만족한다.

---

## Detailed analysis: why it likely improved

### 1. It removed the token-local serial scalar bottleneck from the main kernel

기존 kernel에서는 token마다 CTA의 thread0가:

- `softplus`
- `exp`
- `sigmoid`

를 직접 계산했다.

이건 token 수가 많아질수록 누적되며,
CTA 안에서는 사실상 serial phase였다.

P11은 이것을:

- main kernel 밖으로 분리
- 전체 `T * H` space에 대해 GPU-wide parallel precompute

로 바꿨다.

즉,

> main state-update kernel이 더 “순수한 state update + projection kernel”에 가까워졌다.

### 2. This targets exactly the kind of work previous loops suggested was still exposed

P7/P8/P9/P10이 실패한 반면,
P11은 성공했다.

이 패턴은 분명하다:

- staging micro-tuning: 효과 부족
- ILP micro-tuning: 효과 부족
- fast-math scalar tweak alone: 효과 부족
- runtime hint: 효과 부족
- **scalar gate/beta path를 구조적으로 분리**: 효과 있음

즉,

> 작은 미세조정보다, main loop에서 serial-looking scalar phase를 아예 제거하는 쪽이 훨씬 더 효과적이었다.

### 3. Why this helps long-tail workloads

prefill long-tail은 token 수가 많다.
main kernel 안에서 token마다 gate/beta math를 하면 그 비용이 그대로 누적된다.

반면 precompute kernel은:

- token×head 독립 계산
- massively parallel
- main kernel과 역할 분리

가 가능하다.

따라서,

> long-seq일수록 precompute 분리의 payoff가 커질 가능성이 높다.

### 4. Why this is more powerful than fast-math alone

P9는 fast intrinsics만 썼다.
하지만 P9는 여전히 같은 CTA 안에서 same serial phase를 유지했다.

P11은 수학 정확도를 크게 희생하지 않으면서도,
그 serial phase 자체를 main kernel에서 제거했다.

즉,

> P9는 “같은 일을 조금 빨리”
> P11은 “같은 일을 더 좋은 병렬 구조로” 바꾼 셈이다.

이 차이가 결과 차이를 만든 것으로 보인다.

### 5. Remaining limitations

비록 개선됐지만 아직 kernel의 큰 구조는 그대로다:

- full fp32 state tile shared resident
- row-per-thread structure
- long-tail workload still dominating average

즉,

> P11은 좋은 개선이지만 마지막 단계는 아니다.

---

## Full 100-workload pref log

```text
✓ Initialized. View run at 
https://modal.com/apps/shjj1504/main/ap-lIjhQNCiXG4Cdy8TocZoXS
✓ Created objects.
├── 🔨 Created mount /home/hyu/flashinfer/mlsys26/scripts/run_modal.py
└── 🔨 Created function run_benchmark.
[2026-04-06T00:39:42] Packing solution from source files...
Solution packed: /home/hyu/flashinfer/mlsys26/solution.json
  Name: my-team-solution-v1
  Definition: gdn_prefill_qk4_v8_d128_k_last
  Author: team-name
  Config language: cuda
  Runtime language: cuda
[2026-04-06T00:39:42] Validating solution JSON...
[2026-04-06T00:39:42] Loaded solution my-team-solution-v1 (gdn_prefill_qk4_v8_d128_k_last) in 0.01s
[2026-04-06T00:39:42] Decision-gate mode enabled: warmup_runs=1, iterations=5, num_trials=1, use_isolated_runner=False
[2026-04-06T00:39:42] Dispatching benchmark to Modal B200...

==========
== CUDA ==
==========

CUDA Version 13.0.2

Container image Copyright (c) 2016-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

This container image and its contents are governed by the NVIDIA Deep Learning Container License.
By pulling and using the container, you accept the terms and conditions of this license:
https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license

A copy of this license is made available in this container at /NGC-DL-CONTAINER-LICENSE for your convenience.

[2026-04-05T15:39:48] Remote benchmark start: solution=my-team-solution-v1, definition=gdn_prefill_qk4_v8_d128_k_last
[2026-04-05T15:39:48] BenchmarkConfig(warmup_runs=1, iterations=5, num_trials=1)
[2026-04-05T15:39:48] Loading trace set from /data/data/mlsys26-contest
[2026-04-05T15:39:48] Loaded trace set in 0.35s
[2026-04-05T15:39:48] Running benchmark across 100 workloads
[2026-04-05T15:44:27] Benchmark completed in 279.00s
[2026-04-06T00:44:27] Received benchmark results in 285.50s

gdn_prefill_qk4_v8_d128_k_last:
  workloads: 100
  status counts: PASSED=100
  avg latency: 4.732 ms
  avg speedup: 51.12x
  worst abs error: 1.22e-04
  worst rel error: 2.46e-01
  Workload 77daf91d...: PASSED | 0.072 ms | 19.70x speedup | abs_err=4.47e-08, rel_err=9.53e-03
  Workload ba08a83e...: PASSED | 0.735 ms | 35.00x speedup | abs_err=4.77e-07, rel_err=7.50e-03
  Workload c7846f96...: PASSED | 0.735 ms | 33.84x speedup | abs_err=3.81e-06, rel_err=9.91e-03
  Workload d0ce7b5d...: PASSED | 13.163 ms | 113.44x speedup | abs_err=1.53e-05, rel_err=2.02e-01
  Workload 5b8a0e4b...: PASSED | 20.300 ms | 74.42x speedup | abs_err=1.53e-05, rel_err=6.39e-02
  Workload 5d3fc66a...: PASSED | 20.366 ms | 73.74x speedup | abs_err=6.10e-05, rel_err=1.61e-01
  Workload 4b6143dd...: PASSED | 4.763 ms | 159.65x speedup | abs_err=7.63e-06, rel_err=3.14e-02
  Workload 5835a2bc...: PASSED | 26.320 ms | 57.28x speedup | abs_err=1.53e-05, rel_err=3.06e-02
  Workload cc310f94...: PASSED | 17.159 ms | 86.57x speedup | abs_err=6.10e-05, rel_err=9.95e-02
  Workload d49df0b2...: PASSED | 14.981 ms | 104.23x speedup | abs_err=3.05e-05, rel_err=8.77e-02
  Workload e9e1e445...: PASSED | 11.744 ms | 128.39x speedup | abs_err=1.53e-05, rel_err=3.65e-02
  Workload b8c8dc3c...: PASSED | 11.728 ms | 127.04x speedup | abs_err=6.10e-05, rel_err=2.46e-01
  Workload a9540651...: PASSED | 18.232 ms | 81.52x speedup | abs_err=6.10e-05, rel_err=6.65e-02
  Workload 06f21bb1...: PASSED | 9.518 ms | 157.50x speedup | abs_err=1.53e-05, rel_err=4.09e-02
  Workload c2931c92...: PASSED | 18.646 ms | 80.32x speedup | abs_err=1.22e-04, rel_err=1.10e-01
  Workload 618df04a...: PASSED | 16.768 ms | 89.01x speedup | abs_err=6.10e-05, rel_err=1.21e-01
  Workload 26244fb4...: PASSED | 16.109 ms | 94.32x speedup | abs_err=7.63e-06, rel_err=1.44e-01
  Workload a2629e02...: PASSED | 15.976 ms | 93.26x speedup | abs_err=6.10e-05, rel_err=2.43e-01
  Workload 9a5d694b...: PASSED | 11.154 ms | 134.31x speedup | abs_err=6.10e-05, rel_err=3.25e-02
  Workload 410794d4...: PASSED | 13.307 ms | 113.35x speedup | abs_err=1.53e-05, rel_err=1.71e-02
  Workload 7ba9d519...: PASSED | 14.883 ms | 49.35x speedup | abs_err=6.10e-05, rel_err=5.78e-02
  Workload 043e74e4...: PASSED | 0.134 ms | 46.79x speedup | abs_err=1.19e-07, rel_err=6.68e-03
  Workload ef9515b6...: PASSED | 0.121 ms | 27.58x speedup | abs_err=1.19e-07, rel_err=7.04e-03
  Workload f622a11d...: PASSED | 0.124 ms | 26.15x speedup | abs_err=4.77e-07, rel_err=1.92e-02
  Workload 1cf8e175...: PASSED | 0.157 ms | 50.71x speedup | abs_err=9.54e-07, rel_err=4.95e-03
  Workload 9343fd82...: PASSED | 0.296 ms | 32.66x speedup | abs_err=1.91e-06, rel_err=7.63e-03
  Workload f4926229...: PASSED | 7.150 ms | 36.45x speedup | abs_err=6.10e-05, rel_err=4.59e-02
  Workload 109addb1...: PASSED | 13.625 ms | 45.67x speedup | abs_err=6.10e-05, rel_err=2.49e-02
  Workload c5257f65...: PASSED | 4.922 ms | 36.83x speedup | abs_err=1.91e-06, rel_err=3.44e-02
  Workload f5619793...: PASSED | 4.902 ms | 37.01x speedup | abs_err=6.10e-05, rel_err=5.26e-02
  Workload fdf5f1f4...: PASSED | 0.183 ms | 48.30x speedup | abs_err=1.19e-07, rel_err=7.16e-03
  Workload 87bff084...: PASSED | 0.536 ms | 70.24x speedup | abs_err=3.81e-06, rel_err=6.89e-03
  Workload e92dafeb...: PASSED | 0.598 ms | 73.98x speedup | abs_err=4.77e-07, rel_err=2.58e-02
  Workload 1d0cc342...: PASSED | 1.214 ms | 44.88x speedup | abs_err=3.81e-06, rel_err=5.42e-02
  Workload 1b441950...: PASSED | 0.418 ms | 40.88x speedup | abs_err=9.54e-07, rel_err=1.07e-02
  Workload 19c6ab20...: PASSED | 0.417 ms | 41.06x speedup | abs_err=7.63e-06, rel_err=7.78e-03
  Workload 25d9c14d...: PASSED | 0.401 ms | 61.27x speedup | abs_err=1.91e-06, rel_err=9.87e-03
  Workload 3215fe5f...: PASSED | 1.649 ms | 38.02x speedup | abs_err=3.81e-06, rel_err=7.41e-03
  Workload 6f1ad833...: PASSED | 10.915 ms | 35.28x speedup | abs_err=6.10e-05, rel_err=2.62e-02
  Workload e44ba4d3...: PASSED | 5.858 ms | 50.03x speedup | abs_err=7.63e-06, rel_err=1.53e-02
  Workload fc7a2bcb...: PASSED | 0.538 ms | 48.32x speedup | abs_err=2.38e-07, rel_err=1.72e-02
  Workload 5d26ac5b...: PASSED | 0.537 ms | 48.29x speedup | abs_err=3.81e-06, rel_err=2.10e-02
  Workload ed66c791...: PASSED | 0.642 ms | 33.63x speedup | abs_err=3.81e-06, rel_err=3.70e-02
  Workload ba95d412...: PASSED | 0.847 ms | 40.32x speedup | abs_err=4.77e-07, rel_err=1.50e-02
  Workload 078a41ea...: PASSED | 0.327 ms | 46.49x speedup | abs_err=2.38e-07, rel_err=1.04e-02
  Workload d2b5a221...: PASSED | 0.205 ms | 75.47x speedup | abs_err=9.54e-07, rel_err=1.35e-02
  Workload aaa378be...: PASSED | 7.873 ms | 41.65x speedup | abs_err=7.63e-06, rel_err=1.84e-02
  Workload c2bb4f66...: PASSED | 7.864 ms | 42.08x speedup | abs_err=3.05e-05, rel_err=8.34e-02
  Workload f2f01c2c...: PASSED | 0.145 ms | 46.91x speedup | abs_err=8.94e-08, rel_err=4.96e-03
  Workload 15856e8c...: PASSED | 14.096 ms | 40.70x speedup | abs_err=7.63e-06, rel_err=1.59e-02
  Workload a39aa135...: PASSED | 0.355 ms | 33.53x speedup | abs_err=2.38e-07, rel_err=2.29e-02
  Workload 339a7ff4...: PASSED | 0.123 ms | 27.09x speedup | abs_err=4.77e-07, rel_err=5.72e-03
  Workload d8f4a9ae...: PASSED | 0.222 ms | 30.27x speedup | abs_err=1.19e-07, rel_err=7.04e-03
  Workload d3dc3577...: PASSED | 0.220 ms | 30.58x speedup | abs_err=3.58e-07, rel_err=1.04e-02
  Workload ce832e76...: PASSED | 2.339 ms | 36.15x speedup | abs_err=3.81e-06, rel_err=2.34e-02
  Workload 6fbc155c...: PASSED | 3.568 ms | 42.80x speedup | abs_err=7.63e-06, rel_err=7.75e-03
  Workload a87ded8a...: PASSED | 8.018 ms | 65.37x speedup | abs_err=1.22e-04, rel_err=4.93e-02
  Workload 62447caf...: PASSED | 0.196 ms | 29.49x speedup | abs_err=1.49e-07, rel_err=1.66e-02
  Workload fd072ba6...: PASSED | 0.392 ms | 54.37x speedup | abs_err=3.81e-06, rel_err=1.77e-02
  Workload 35ea9bbe...: PASSED | 0.391 ms | 54.04x speedup | abs_err=1.91e-06, rel_err=4.58e-02
  Workload 1aa8cf18...: PASSED | 0.132 ms | 51.29x speedup | abs_err=8.94e-08, rel_err=7.61e-03
  Workload d5f5c00c...: PASSED | 0.199 ms | 31.31x speedup | abs_err=1.49e-08, rel_err=1.75e-03
  Workload d5aa60dc...: PASSED | 1.107 ms | 42.76x speedup | abs_err=1.91e-06, rel_err=4.99e-02
  Workload 28b70283...: PASSED | 0.162 ms | 29.06x speedup | abs_err=1.19e-07, rel_err=8.86e-03
  Workload 73b8cc85...: PASSED | 0.543 ms | 33.35x speedup | abs_err=1.19e-07, rel_err=7.04e-03
  Workload 2683c087...: PASSED | 0.544 ms | 33.50x speedup | abs_err=7.63e-06, rel_err=1.63e-02
  Workload 4b94d568...: PASSED | 2.418 ms | 43.40x speedup | abs_err=3.81e-06, rel_err=1.12e-02
  Workload a0eb2dc2...: PASSED | 0.517 ms | 33.20x speedup | abs_err=3.81e-06, rel_err=6.37e-03
  Workload f3d30cb9...: PASSED | 0.417 ms | 38.29x speedup | abs_err=2.98e-07, rel_err=2.28e-02
  Workload 7a7deca8...: PASSED | 0.449 ms | 32.88x speedup | abs_err=3.81e-06, rel_err=7.69e-03
  Workload 977d19f8...: PASSED | 0.196 ms | 29.73x speedup | abs_err=1.19e-07, rel_err=7.04e-03
  Workload 02c1e5f0...: PASSED | 0.195 ms | 29.78x speedup | abs_err=7.63e-06, rel_err=6.94e-03
  Workload 5a91aa02...: PASSED | 1.468 ms | 50.42x speedup | abs_err=3.81e-06, rel_err=1.48e-02
  Workload 8e7ef744...: PASSED | 0.101 ms | 24.81x speedup | abs_err=2.98e-08, rel_err=3.76e-03
  Workload 85d7becb...: PASSED | 0.134 ms | 26.83x speedup | abs_err=3.58e-07, rel_err=6.89e-03
  Workload e286a4f4...: PASSED | 6.118 ms | 55.59x speedup | abs_err=7.63e-06, rel_err=3.52e-02
  Workload 08d4f2c4...: PASSED | 4.726 ms | 37.83x speedup | abs_err=7.63e-06, rel_err=7.79e-03
  Workload 9c1ef562...: PASSED | 4.726 ms | 37.75x speedup | abs_err=3.05e-05, rel_err=6.02e-02
  Workload bfd8f7b6...: PASSED | 1.259 ms | 36.37x speedup | abs_err=7.63e-06, rel_err=3.34e-02
  Workload c358edcd...: PASSED | 10.457 ms | 35.80x speedup | abs_err=1.53e-05, rel_err=7.69e-03
  Workload f203fdcd...: PASSED | 0.288 ms | 31.62x speedup | abs_err=2.38e-07, rel_err=1.20e-02
  Workload 33a38713...: PASSED | 0.402 ms | 56.84x speedup | abs_err=1.49e-07, rel_err=2.21e-02
  Workload 3a77dfec...: PASSED | 0.346 ms | 41.59x speedup | abs_err=9.54e-07, rel_err=7.25e-03
  Workload ea27be17...: PASSED | 0.346 ms | 41.45x speedup | abs_err=7.15e-07, rel_err=5.46e-02
  Workload 49ef89d2...: PASSED | 0.387 ms | 32.68x speedup | abs_err=1.91e-06, rel_err=5.21e-02
  Workload 056224b8...: PASSED | 0.393 ms | 32.76x speedup | abs_err=2.38e-07, rel_err=6.62e-03
  Workload 685d26ff...: PASSED | 0.109 ms | 24.76x speedup | abs_err=7.45e-08, rel_err=3.43e-03
  Workload 352c9ace...: PASSED | 2.753 ms | 36.46x speedup | abs_err=3.05e-05, rel_err=2.50e-02
  Workload 2c9693b4...: PASSED | 0.292 ms | 32.73x speedup | abs_err=1.19e-07, rel_err=1.84e-02
  Workload 27f44fd6...: PASSED | 0.293 ms | 31.74x speedup | abs_err=1.91e-06, rel_err=1.57e-02
  Workload 07aa7922...: PASSED | 29.142 ms | 35.80x speedup | abs_err=1.53e-05, rel_err=8.62e-02
  Workload eaa0fd47...: PASSED | 0.112 ms | 26.68x speedup | abs_err=9.54e-07, rel_err=6.45e-03
  Workload f105eda8...: PASSED | 2.179 ms | 57.81x speedup | abs_err=6.10e-05, rel_err=1.94e-02
  Workload cd979341...: PASSED | 0.350 ms | 40.84x speedup | abs_err=2.38e-07, rel_err=7.03e-03
  Workload 43bf9699...: PASSED | 4.660 ms | 38.08x speedup | abs_err=7.63e-06, rel_err=8.02e-03
  Workload 54856fec...: PASSED | 4.660 ms | 37.84x speedup | abs_err=3.05e-05, rel_err=9.25e-02
  Workload 2ba465c0...: PASSED | 0.114 ms | 48.78x speedup | abs_err=5.96e-08, rel_err=6.84e-03
  Workload 1efaf2a9...: PASSED | 0.186 ms | 45.09x speedup | abs_err=5.96e-08, rel_err=4.48e-03
  Workload a01a3f93...: PASSED | 10.984 ms | 38.30x speedup | abs_err=3.05e-05, rel_err=3.70e-02
  Workload cc241d2e...: PASSED | 0.159 ms | 28.72x speedup | abs_err=1.49e-07, rel_err=6.82e-03
[2026-04-06T00:44:27] Local entrypoint finished in 285.51s
[2026-04-05T15:44:27] Remote benchmark finished in 279.35s
Stopping app - local entrypoint completed.
✓ App completed. View run at 
https://modal.com/apps/shjj1504/main/ap-lIjhQNCiXG4Cdy8TocZoXS

```
