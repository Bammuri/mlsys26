# Prefill Timeline — P6 Result — 2604052357

## Optimization summary

이번 P6 iteration에서는 kernel 수학은 바꾸지 않고, compile target을 `sm_100a`로 명시하도록 runtime/build 경로를 조정했다.

### Applied optimization

1. background에 평가 타깃이 `sm_100a`임을 명시
2. background에 `sm_100a`-specific codegen 전략 추가
3. Modal/Torch extension runtime에 `TORCH_CUDA_ARCH_LIST=10.0a` 반영

핵심 의도:

> generic `sm_100` codegen이 아니라,
> 실제 평가 타깃인 `sm_100a`에 더 정확히 맞춘 code generation으로 작은 아키텍처 이득을 확보하기

---

## Expected result vs actual result

### Expected before running

이번 iteration의 비교 기준은 P5 full decision gate baseline이다.

P5 full decision gate baseline:

- config: `warmup_runs=1, iterations=5, num_trials=1`
- avg latency: `4.923 ms`
- `PASSED=100/100`

예상:

1. quick 5 PASS 유지
2. full 100 PASS 유지
3. 개선 폭은 크지 않을 수 있음
4. 하지만 실제 평가 타깃이 `sm_100a`라면, 작은 codegen 이득은 기대 가능

### Actual results

#### 3-1. Quick working check

실행:

```bash
modal run scripts/run_modal.py --quick --max-workloads 5
```

결과:

- `PASSED=5/5`
- avg latency: `0.204 ms`
- avg speedup: `28.49x`
- worst abs error: `1.91e-06`
- worst rel error: `2.29e-02`

#### 3-2. Full decision gate extraction

실행:

```bash
modal run scripts/run_modal.py --decision-gate --max-workloads 100
```

결과:

- `PASSED=100/100`
- avg latency: `4.913 ms`
- avg speedup: `51.06x`
- worst abs error: `1.22e-04`
- worst rel error: `2.46e-01`
- end-to-end wall time: `295.55s`

대표 로그 예시:

- `Workload d0ce7b5d...: PASSED | 13.602 ms | 109.34x speedup | abs_err=1.53e-05, rel_err=2.02e-01`
- `Workload ce832e76...: PASSED | 2.447 ms | 36.07x speedup | abs_err=3.81e-06, rel_err=2.34e-02`
- `Workload 07aa7922...: PASSED | 30.628 ms | 35.75x speedup | abs_err=1.53e-05, rel_err=8.62e-02`

---

## Judgement

이번 iteration은 **keep** 한다.

### Decision rule used

판단 기준은 사용자가 지정한 대로:

> **full 100-workload arithmetic-mean latency**

이다.

### Why this is a keep

P5 → P6 (full decision gate):

- avg latency: `4.923 ms` → `4.913 ms`
- 절대 차이: `0.010 ms`
- 상대 개선: 약 `0.20%`

폭은 매우 작다. 하지만:

1. full 100 기준으로 latency가 실제로 내려갔고
2. `PASSED=100/100`을 유지했고
3. `sm_100a`는 실제 평가 타깃이라는 새 사실과 일치한다

즉,

> 이번 loop는 공격적인 성능 leap는 아니지만,
> 타깃 아키텍처 정렬 측면에서도 맞는 방향이므로 keep할 가치가 있다.

---

## Detailed analysis: why the gain is small but plausible

### 1. This was a codegen alignment iteration, not an algorithmic one

P6는 kernel 수학/CTA 구조/shared layout을 바꾸지 않았다.
즉,

- state update 수학 동일
- q/k reuse 동일
- float4 vectorization 동일

따라서 크게 바뀔 수 있는 것은:

- compiler scheduling
- instruction selection
- arch-specific lowering

정도다.

그래서 개선 폭이 작아도 이상하지 않다.

### 2. `sm_100a` targeting is still strategically correct

사용자 업데이트로 인해,
실제 evaluation target은 generic `sm_100`이 아니라 `sm_100a`라는 점이 중요해졌다.

따라서 이번 loop의 의미는 단순히 `0.010 ms` 개선보다도,

> 이후 모든 loop가 실제 평가 아키텍처를 기준으로 codegen 되도록 baseline을 맞춘 것

에 있다.

### 3. Why the wall time is not the main keep/reject metric here

이번 full run wall time은 image/build/cache 상황 영향도 많이 받는다.
특히 `TORCH_CUDA_ARCH_LIST` 변경으로 인해 image/build path가 일부 다시 돈 영향이 있다.

사용자 규칙상 현재 keep/reject 기준은:

- **full workload avg latency arithmetic mean**

이므로, 이번 판단은 그 기준을 우선했다.

### 4. Why this should help future loops more than this loop itself

`sm_100a` targeting 자체는 즉시 큰 폭의 성능 향상을 주기보다는,

- 이후 kernel refactor
- arch-specific instruction selection
- codegen quality

의 바닥을 더 정확히 맞춘다.

즉,

> P6는 “작은 직접 개선 + 더 올바른 future baseline” 이라는 성격이 강하다.

---

## Full 100-workload pref log

```text
✓ Initialized. View run at 
https://modal.com/apps/shjj1504/main/ap-z5rQdDo5v0nQOIBOKcngtn
✓ Created objects.
├── 🔨 Created mount /home/hyu/flashinfer/mlsys26/scripts/run_modal.py
└── 🔨 Created function run_benchmark.
[2026-04-06T00:05:56] Packing solution from source files...
Solution packed: /home/hyu/flashinfer/mlsys26/solution.json
  Name: my-team-solution-v1
  Definition: gdn_prefill_qk4_v8_d128_k_last
  Author: team-name
  Config language: cuda
  Runtime language: cuda
[2026-04-06T00:05:56] Validating solution JSON...
[2026-04-06T00:05:56] Loaded solution my-team-solution-v1 (gdn_prefill_qk4_v8_d128_k_last) in 0.00s
[2026-04-06T00:05:56] Decision-gate mode enabled: warmup_runs=1, iterations=5, num_trials=1, use_isolated_runner=False
[2026-04-06T00:05:56] Dispatching benchmark to Modal B200...

==========
== CUDA ==
==========

CUDA Version 13.0.2

Container image Copyright (c) 2016-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

This container image and its contents are governed by the NVIDIA Deep Learning Container License.
By pulling and using the container, you accept the terms and conditions of this license:
https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license

A copy of this license is made available in this container at /NGC-DL-CONTAINER-LICENSE for your convenience.

[2026-04-05T15:06:09] Remote benchmark start: solution=my-team-solution-v1, definition=gdn_prefill_qk4_v8_d128_k_last
[2026-04-05T15:06:09] BenchmarkConfig(warmup_runs=1, iterations=5, num_trials=1)
[2026-04-05T15:06:09] Loading trace set from /data/data/mlsys26-contest
[2026-04-05T15:06:09] Loaded trace set in 0.13s
[2026-04-05T15:06:09] Running benchmark across 100 workloads
[2026-04-06T00:10:51] Received benchmark results in 295.55s

gdn_prefill_qk4_v8_d128_k_last:
  workloads: 100
  status counts: PASSED=100
  avg latency: 4.913 ms
  avg speedup: 51.06x
  worst abs error: 1.22e-04
  worst rel error: 2.46e-01
  Workload 77daf91d...: PASSED | 0.065 ms | 22.01x speedup | abs_err=4.47e-08, rel_err=9.53e-03
  Workload ba08a83e...: PASSED | 0.761 ms | 32.12x speedup | abs_err=4.77e-07, rel_err=7.50e-03
  Workload c7846f96...: PASSED | 0.763 ms | 31.97x speedup | abs_err=3.81e-06, rel_err=9.91e-03
  Workload d0ce7b5d...: PASSED | 13.602 ms | 109.34x speedup | abs_err=1.53e-05, rel_err=2.02e-01
  Workload 5b8a0e4b...: PASSED | 20.911 ms | 70.04x speedup | abs_err=1.53e-05, rel_err=6.39e-02
  Workload 5d3fc66a...: PASSED | 20.927 ms | 70.10x speedup | abs_err=6.10e-05, rel_err=1.61e-01
  Workload 4b6143dd...: PASSED | 4.986 ms | 148.52x speedup | abs_err=7.63e-06, rel_err=3.14e-02
  Workload 5835a2bc...: PASSED | 26.985 ms | 54.71x speedup | abs_err=1.53e-05, rel_err=3.06e-02
  Workload cc310f94...: PASSED | 17.803 ms | 82.73x speedup | abs_err=6.10e-05, rel_err=9.95e-02
  Workload d49df0b2...: PASSED | 15.343 ms | 96.58x speedup | abs_err=3.05e-05, rel_err=8.77e-02
  Workload e9e1e445...: PASSED | 12.167 ms | 121.42x speedup | abs_err=1.53e-05, rel_err=3.65e-02
  Workload b8c8dc3c...: PASSED | 12.175 ms | 120.98x speedup | abs_err=6.10e-05, rel_err=2.46e-01
  Workload a9540651...: PASSED | 18.983 ms | 77.66x speedup | abs_err=6.10e-05, rel_err=6.65e-02
  Workload 06f21bb1...: PASSED | 9.779 ms | 150.52x speedup | abs_err=1.53e-05, rel_err=4.09e-02
  Workload c2931c92...: PASSED | 19.199 ms | 77.01x speedup | abs_err=1.22e-04, rel_err=1.10e-01
  Workload 618df04a...: PASSED | 17.090 ms | 86.21x speedup | abs_err=6.10e-05, rel_err=1.21e-01
  Workload 26244fb4...: PASSED | 16.597 ms | 89.04x speedup | abs_err=7.63e-06, rel_err=1.44e-01
  Workload a2629e02...: PASSED | 16.506 ms | 92.43x speedup | abs_err=6.10e-05, rel_err=2.43e-01
  Workload 9a5d694b...: PASSED | 11.507 ms | 132.80x speedup | abs_err=6.10e-05, rel_err=3.25e-02
  Workload 410794d4...: PASSED | 13.716 ms | 107.69x speedup | abs_err=1.53e-05, rel_err=1.71e-02
  Workload 7ba9d519...: PASSED | 15.628 ms | 47.75x speedup | abs_err=6.10e-05, rel_err=5.78e-02
  Workload 043e74e4...: PASSED | 0.129 ms | 49.05x speedup | abs_err=1.19e-07, rel_err=6.68e-03
  Workload ef9515b6...: PASSED | 0.118 ms | 27.82x speedup | abs_err=1.19e-07, rel_err=7.04e-03
  Workload f622a11d...: PASSED | 0.123 ms | 26.74x speedup | abs_err=4.77e-07, rel_err=1.92e-02
  Workload 1cf8e175...: PASSED | 0.157 ms | 48.22x speedup | abs_err=9.54e-07, rel_err=4.95e-03
  Workload 9343fd82...: PASSED | 0.297 ms | 32.24x speedup | abs_err=1.91e-06, rel_err=7.63e-03
  Workload f4926229...: PASSED | 7.500 ms | 33.97x speedup | abs_err=6.10e-05, rel_err=4.59e-02
  Workload 109addb1...: PASSED | 14.307 ms | 43.42x speedup | abs_err=6.10e-05, rel_err=2.49e-02
  Workload c5257f65...: PASSED | 5.149 ms | 36.30x speedup | abs_err=1.91e-06, rel_err=3.44e-02
  Workload f5619793...: PASSED | 5.142 ms | 36.50x speedup | abs_err=6.10e-05, rel_err=5.26e-02
  Workload fdf5f1f4...: PASSED | 0.183 ms | 50.34x speedup | abs_err=1.19e-07, rel_err=7.16e-03
  Workload 87bff084...: PASSED | 0.552 ms | 70.82x speedup | abs_err=3.81e-06, rel_err=6.89e-03
  Workload e92dafeb...: PASSED | 0.618 ms | 78.06x speedup | abs_err=4.77e-07, rel_err=2.58e-02
  Workload 1d0cc342...: PASSED | 1.269 ms | 46.66x speedup | abs_err=3.81e-06, rel_err=5.42e-02
  Workload 1b441950...: PASSED | 0.429 ms | 41.45x speedup | abs_err=9.54e-07, rel_err=1.07e-02
  Workload 19c6ab20...: PASSED | 0.429 ms | 41.61x speedup | abs_err=7.63e-06, rel_err=7.78e-03
  Workload 25d9c14d...: PASSED | 0.417 ms | 68.38x speedup | abs_err=1.91e-06, rel_err=9.87e-03
  Workload 3215fe5f...: PASSED | 1.722 ms | 37.86x speedup | abs_err=3.81e-06, rel_err=7.41e-03
  Workload 6f1ad833...: PASSED | 11.456 ms | 35.22x speedup | abs_err=6.10e-05, rel_err=2.62e-02
  Workload e44ba4d3...: PASSED | 6.152 ms | 49.00x speedup | abs_err=7.63e-06, rel_err=1.53e-02
  Workload fc7a2bcb...: PASSED | 0.557 ms | 48.20x speedup | abs_err=2.38e-07, rel_err=1.72e-02
  Workload 5d26ac5b...: PASSED | 0.555 ms | 48.41x speedup | abs_err=3.81e-06, rel_err=2.10e-02
  Workload ed66c791...: PASSED | 0.663 ms | 33.48x speedup | abs_err=3.81e-06, rel_err=3.70e-02
  Workload ba95d412...: PASSED | 0.879 ms | 38.81x speedup | abs_err=4.77e-07, rel_err=1.50e-02
  Workload 078a41ea...: PASSED | 0.332 ms | 47.34x speedup | abs_err=2.38e-07, rel_err=1.04e-02
  Workload d2b5a221...: PASSED | 0.201 ms | 79.73x speedup | abs_err=9.54e-07, rel_err=1.35e-02
  Workload aaa378be...: PASSED | 8.252 ms | 41.63x speedup | abs_err=7.63e-06, rel_err=1.84e-02
  Workload c2bb4f66...: PASSED | 8.251 ms | 41.39x speedup | abs_err=3.05e-05, rel_err=8.34e-02
  Workload f2f01c2c...: PASSED | 0.141 ms | 49.75x speedup | abs_err=8.94e-08, rel_err=4.96e-03
  Workload 15856e8c...: PASSED | 14.812 ms | 40.16x speedup | abs_err=7.63e-06, rel_err=1.59e-02
  Workload a39aa135...: PASSED | 0.363 ms | 37.09x speedup | abs_err=2.38e-07, rel_err=2.29e-02
  Workload 339a7ff4...: PASSED | 0.117 ms | 28.66x speedup | abs_err=4.77e-07, rel_err=5.72e-03
  Workload d8f4a9ae...: PASSED | 0.222 ms | 31.35x speedup | abs_err=1.19e-07, rel_err=7.04e-03
  Workload d3dc3577...: PASSED | 0.223 ms | 31.62x speedup | abs_err=3.58e-07, rel_err=1.04e-02
  Workload ce832e76...: PASSED | 2.447 ms | 36.07x speedup | abs_err=3.81e-06, rel_err=2.34e-02
  Workload 6fbc155c...: PASSED | 3.742 ms | 43.44x speedup | abs_err=7.63e-06, rel_err=7.75e-03
  Workload a87ded8a...: PASSED | 8.411 ms | 64.91x speedup | abs_err=1.22e-04, rel_err=4.93e-02
  Workload 62447caf...: PASSED | 0.194 ms | 30.98x speedup | abs_err=1.49e-07, rel_err=1.66e-02
  Workload fd072ba6...: PASSED | 0.402 ms | 55.09x speedup | abs_err=3.81e-06, rel_err=1.77e-02
  Workload 35ea9bbe...: PASSED | 0.402 ms | 54.68x speedup | abs_err=1.91e-06, rel_err=4.58e-02
  Workload 1aa8cf18...: PASSED | 0.131 ms | 54.38x speedup | abs_err=8.94e-08, rel_err=7.61e-03
  Workload d5f5c00c...: PASSED | 0.201 ms | 31.14x speedup | abs_err=1.49e-08, rel_err=1.75e-03
  Workload d5aa60dc...: PASSED | 1.153 ms | 42.20x speedup | abs_err=1.91e-06, rel_err=4.99e-02
  Workload 28b70283...: PASSED | 0.162 ms | 33.77x speedup | abs_err=1.19e-07, rel_err=8.86e-03
  Workload 73b8cc85...: PASSED | 0.561 ms | 33.50x speedup | abs_err=1.19e-07, rel_err=7.04e-03
  Workload 2683c087...: PASSED | 0.559 ms | 33.70x speedup | abs_err=7.63e-06, rel_err=1.63e-02
  Workload 4b94d568...: PASSED | 2.529 ms | 47.08x speedup | abs_err=3.81e-06, rel_err=1.12e-02
  Workload a0eb2dc2...: PASSED | 0.531 ms | 35.23x speedup | abs_err=3.81e-06, rel_err=6.37e-03
  Workload f3d30cb9...: PASSED | 0.429 ms | 38.56x speedup | abs_err=2.98e-07, rel_err=2.28e-02
  Workload 7a7deca8...: PASSED | 0.460 ms | 34.18x speedup | abs_err=3.81e-06, rel_err=7.69e-03
  Workload 977d19f8...: PASSED | 0.196 ms | 31.13x speedup | abs_err=1.19e-07, rel_err=7.04e-03
  Workload 02c1e5f0...: PASSED | 0.195 ms | 29.93x speedup | abs_err=7.63e-06, rel_err=6.94e-03
  Workload 5a91aa02...: PASSED | 1.531 ms | 51.17x speedup | abs_err=3.81e-06, rel_err=1.48e-02
  Workload 8e7ef744...: PASSED | 0.097 ms | 26.83x speedup | abs_err=2.98e-08, rel_err=3.76e-03
  Workload 85d7becb...: PASSED | 0.130 ms | 29.10x speedup | abs_err=3.58e-07, rel_err=6.89e-03
  Workload e286a4f4...: PASSED | 6.411 ms | 53.66x speedup | abs_err=7.63e-06, rel_err=3.52e-02
  Workload 08d4f2c4...: PASSED | 4.950 ms | 37.24x speedup | abs_err=7.63e-06, rel_err=7.79e-03
  Workload 9c1ef562...: PASSED | 4.960 ms | 37.42x speedup | abs_err=3.05e-05, rel_err=6.02e-02
  Workload bfd8f7b6...: PASSED | 1.312 ms | 36.16x speedup | abs_err=7.63e-06, rel_err=3.34e-02
  Workload c358edcd...: PASSED | 10.986 ms | 35.40x speedup | abs_err=1.53e-05, rel_err=7.69e-03
  Workload f203fdcd...: PASSED | 0.292 ms | 33.81x speedup | abs_err=2.38e-07, rel_err=1.20e-02
  Workload 33a38713...: PASSED | 0.413 ms | 54.84x speedup | abs_err=1.49e-07, rel_err=2.21e-02
  Workload 3a77dfec...: PASSED | 0.355 ms | 42.57x speedup | abs_err=9.54e-07, rel_err=7.25e-03
  Workload ea27be17...: PASSED | 0.356 ms | 41.43x speedup | abs_err=7.15e-07, rel_err=5.46e-02
  Workload 49ef89d2...: PASSED | 0.398 ms | 32.62x speedup | abs_err=1.91e-06, rel_err=5.21e-02
  Workload 056224b8...: PASSED | 0.402 ms | 33.04x speedup | abs_err=2.38e-07, rel_err=6.62e-03
  Workload 685d26ff...: PASSED | 0.103 ms | 27.56x speedup | abs_err=7.45e-08, rel_err=3.43e-03
  Workload 352c9ace...: PASSED | 2.879 ms | 34.99x speedup | abs_err=3.05e-05, rel_err=2.50e-02
  Workload 2c9693b4...: PASSED | 0.300 ms | 39.55x speedup | abs_err=1.19e-07, rel_err=1.84e-02
  Workload 27f44fd6...: PASSED | 0.298 ms | 32.18x speedup | abs_err=1.91e-06, rel_err=1.57e-02
  Workload 07aa7922...: PASSED | 30.628 ms | 35.75x speedup | abs_err=1.53e-05, rel_err=8.62e-02
  Workload eaa0fd47...: PASSED | 0.108 ms | 32.47x speedup | abs_err=9.54e-07, rel_err=6.45e-03
  Workload f105eda8...: PASSED | 2.281 ms | 55.93x speedup | abs_err=6.10e-05, rel_err=1.94e-02
  Workload cd979341...: PASSED | 0.358 ms | 41.70x speedup | abs_err=2.38e-07, rel_err=7.03e-03
  Workload 43bf9699...: PASSED | 4.876 ms | 39.83x speedup | abs_err=7.63e-06, rel_err=8.02e-03
  Workload 54856fec...: PASSED | 4.876 ms | 39.11x speedup | abs_err=3.05e-05, rel_err=9.25e-02
  Workload 2ba465c0...: PASSED | 0.107 ms | 58.51x speedup | abs_err=5.96e-08, rel_err=6.84e-03
  Workload 1efaf2a9...: PASSED | 0.180 ms | 47.22x speedup | abs_err=5.96e-08, rel_err=4.48e-03
  Workload a01a3f93...: PASSED | 11.602 ms | 38.99x speedup | abs_err=3.05e-05, rel_err=3.70e-02
  Workload cc241d2e...: PASSED | 0.156 ms | 33.80x speedup | abs_err=1.49e-07, rel_err=6.82e-03
[2026-04-06T00:10:51] Local entrypoint finished in 295.55s
[2026-04-05T15:10:55] Benchmark completed in 285.98s
[2026-04-05T15:10:55] Remote benchmark finished in 286.11s
Stopping app - local entrypoint completed.
Runner terminated.
✓ App completed. View run at 
https://modal.com/apps/shjj1504/main/ap-z5rQdDo5v0nQOIBOKcngtn

```
