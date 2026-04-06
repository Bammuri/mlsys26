# Prefill Timeline — P6 Result — 2026-04-07

## Optimization summary

이번 P6 iteration에서는 current keep baseline(P3)에 대해,
**gate/beta를 separate float arrays 대신 packed `float2` buffer로 바꾸는 것**을 시험했다.

### Applied optimization

1. `compute_gate_beta_kernel` 출력 형식을 packed `float2`로 변경
2. host side temporary tensor도 packed layout으로 변경
3. main kernel은 lane 0가 `float2` 1회 load 후 gate/beta를 warp broadcast
4. launch shape / row-per-warp / no-shared / no-spill SRTP 구조는 그대로 유지

핵심 의도:

> current SRTP baseline의 scalar side-data path를 더 compact하게 만들어,
> per-token gate/beta read path를 줄이기

---

## Expected result before testing

baseline (keep baseline, full 100, `warmup_runs=1, iterations=5, num_trials=2`):

- avg latency: `0.511 ms`

예상:

1. quick 5 PASS 유지
2. full 100 PASS 유지
3. gain이 있더라도 small gain일 가능성이 높음
4. no-op 가능성도 높음

---

## Verification

### 1. pack

```bash
python scripts/pack_solution.py
```

결과:

- 성공

### 2. quick gate

```bash
modal run scripts/run_modal.py --quick --max-workloads 5
```

결과:

- `PASSED=5/5`
- avg latency: `0.057 ms`
- worst abs error: `1.91e-06`
- worst rel error: `1.93e-02`

### 3. full decision gate

```bash
modal run scripts/run_modal.py --decision-gate --max-workloads 100
```

결과:

- `PASSED=100/100`
- avg latency: `0.490 ms`
- avg speedup: `356.56x`
- worst abs error: `1.22e-04`
- worst rel error: `2.97e-01`

판단 기준은 사용자 지시대로 **full workload avg latency arithmetic mean**다.

---

## Comparison against baseline

### Baseline → P6

- avg latency: `0.511 ms` → `0.490 ms`
- 절대 개선: `0.021 ms`
- 상대 개선: 약 **4.11%**

즉 이번 iteration은 full 기준으로 의미 있는 개선이다.

---

## Representative workload comparison

대표 workload에서도 전반적으로 개선이 관찰된다.

- `77daf91d`: `0.035 ms` → `0.031 ms`
- `ba08a83e`: `0.092 ms` → `0.085 ms`
- `5835a2bc`: `2.169 ms` → `2.100 ms`
- `07aa7922`: `2.651 ms` → `2.545 ms`

특히 long-tail과 throughput-heavy에서 동시에 좋아졌다는 점이 중요하다.

---

## Detailed analysis: why it likely improved

### 1. P6 improved the scalar side-data path without disturbing the good parts of P3

P3 keep baseline의 강점은 이미 분명했다.

- row-per-warp 유지
- no-shared state
- no spill
- launch shape 개선

P6는 이 core structure를 전혀 건드리지 않았다.
대신, token마다 lane 0가 읽는 scalar pair인 gate/beta만 더 compact하게 바꿨다.

즉,

> P6는 current keep baseline의 핵심 장점을 유지한 채,
> 주변부 memory path만 더 효율적으로 만든 “good local optimization”이다.

### 2. Why packed `float2` is better than two separate scalar arrays here

현재 main kernel에서는 token마다 lane 0가:

- `gate[gate_idx]`
- `beta[gate_idx]`

를 각각 읽고 있었다.

이 둘은 항상 함께 소비되므로,
packed `float2`로 묶으면:

1. load instruction path 단순화
2. scalar metadata fetch path compact화
3. duplicated scalar read overhead 감소

효과를 기대할 수 있다.

이번 결과는 그 가설이 실제로 맞았음을 보여준다.

### 3. Why this succeeded where P5 failed

P5는 blanket cache hint로 hardware policy 전체를 건드렸다.
그 결과 full 평균이 `0.572 ms`로 악화됐다.

반면 P6는:

- cache policy 강제 없음
- launch shape 변경 없음
- just data representation cleanup

이었다.

즉,

> hardware의 전역 정책을 강제하는 것보다,
> 실제로 항상 같이 쓰는 데이터를 packed form으로 단순화하는 것이 더 안전하고 유효했다.

### 4. Why this is more than tiny noise

이번 개선은 tiny case뿐 아니라,
throughput-heavy / long-tail에서도 같이 나타난다.

- `5835a2bc`: `2.169 ms` → `2.100 ms`
- `07aa7922`: `2.651 ms` → `2.545 ms`

즉 이번 변화는 단순히 launch-sensitive tiny overfit이 아니라,
**full workload mix 전반에 걸친 small-but-real gain**이다.

### 5. What this says about the next search direction

P6는 현재 search space에 대해 좋은 교훈을 준다.

- P4: over-fragmented launch shape → 악화
- P5: blanket cache hint → 악화
- P6: compact scalar representation → 개선

즉 다음 루프에서도 더 믿을 만한 방향은:

1. current SRTP launch family 유지
2. always-consumed data의 representation cleanup
3. memory path simplification
4. long-tail/throughput 모두에 이득이 나는 local cleanup

이다.

---

## Judgement

이번 iteration은 **keep** 한다.

### Why this is a keep

1. `PASSED=100/100`
2. full avg latency가 `0.511 ms` → `0.490 ms`로 개선
3. tiny / medium / throughput / long-tail representative workloads 전반에서 positive trend
4. current SRTP family를 깨지 않고 얻은 안정적 개선

---

## Decision under workflow rule

- keep/revert 기준: **full workload avg latency arithmetic mean**
- 결과: **KEEP**

다음 단계:

1. commit
2. 다시 background + timeline 기반으로 다음 plan 생성

---

## Full pref log

full pref log is saved in:

- `timeline/prefill/perf_P6_2604070031.txt`

아래는 동일 로그 전문이다.

```text
✓ Initialized. View run at 
https://modal.com/apps/shjj1504/main/ap-zspCPqlZKLS1qx1jnh6MKL
✓ Created objects.
├── 🔨 Created mount /home/hyu/flashinfer/mlsys26/scripts/run_modal.py
└── 🔨 Created function run_benchmark.
[2026-04-07T00:31:43] Packing solution from source files...
Solution packed: /home/hyu/flashinfer/mlsys26/solution.json
  Name: my-team-solution-v1
  Definition: gdn_prefill_qk4_v8_d128_k_last
  Author: team-name
  Config language: cuda
  Runtime language: cuda
[2026-04-07T00:31:43] Validating solution JSON...
[2026-04-07T00:31:43] Loaded solution my-team-solution-v1 (gdn_prefill_qk4_v8_d128_k_last) in 0.01s
[2026-04-07T00:31:43] Decision-gate mode enabled: warmup_runs=1, iterations=5, num_trials=2, use_isolated_runner=False
[2026-04-07T00:31:43] Dispatching benchmark to Modal B200...

==========
== CUDA ==
==========

CUDA Version 13.0.2

Container image Copyright (c) 2016-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

This container image and its contents are governed by the NVIDIA Deep Learning Container License.
By pulling and using the container, you accept the terms and conditions of this license:
https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license

A copy of this license is made available in this container at /NGC-DL-CONTAINER-LICENSE for your convenience.

[2026-04-06T15:31:49] Remote benchmark start: solution=my-team-solution-v1, definition=gdn_prefill_qk4_v8_d128_k_last
[2026-04-06T15:31:49] BenchmarkConfig(warmup_runs=1, iterations=5, num_trials=2)
[2026-04-06T15:31:49] Loading trace set from /data/data/mlsys26-contest
[2026-04-06T15:31:50] Loaded trace set in 0.85s
[2026-04-06T15:31:50] Running benchmark across 100 workloads

==========
== CUDA ==
==========

CUDA Version 13.0.2

Container image Copyright (c) 2016-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

This container image and its contents are governed by the NVIDIA Deep Learning Container License.
By pulling and using the container, you accept the terms and conditions of this license:
https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license

A copy of this license is made available in this container at /NGC-DL-CONTAINER-LICENSE for your convenience.

[2026-04-06T15:39:16] Benchmark completed in 445.53s
[2026-04-06T15:39:16] Remote benchmark finished in 446.39s
[2026-04-07T00:39:16] Received benchmark results in 453.94s

gdn_prefill_qk4_v8_d128_k_last:
  workloads: 100
  status counts: PASSED=100
  avg latency: 0.490 ms
  avg speedup: 356.56x
  worst abs error: 1.22e-04
  worst rel error: 2.97e-01
  Workload 77daf91d...: PASSED | 0.031 ms | 39.25x speedup | abs_err=4.47e-08, rel_err=9.53e-03
  Workload ba08a83e...: PASSED | 0.085 ms | 251.88x speedup | abs_err=7.63e-06, rel_err=7.50e-03
  Workload c7846f96...: PASSED | 0.085 ms | 252.69x speedup | abs_err=3.81e-06, rel_err=6.89e-03
  Workload d0ce7b5d...: PASSED | 1.493 ms | 864.72x speedup | abs_err=6.10e-05, rel_err=1.21e-01
  Workload 5b8a0e4b...: PASSED | 2.174 ms | 594.29x speedup | abs_err=1.53e-05, rel_err=4.03e-02
  Workload 5d3fc66a...: PASSED | 2.151 ms | 599.25x speedup | abs_err=6.10e-05, rel_err=1.59e-01
  Workload 4b6143dd...: PASSED | 0.587 ms | 1100.92x speedup | abs_err=7.63e-06, rel_err=2.79e-02
  Workload 5835a2bc...: PASSED | 2.100 ms | 614.83x speedup | abs_err=1.53e-05, rel_err=1.73e-02
  Workload cc310f94...: PASSED | 1.952 ms | 656.73x speedup | abs_err=1.22e-04, rel_err=1.33e-01
  Workload d49df0b2...: PASSED | 1.595 ms | 806.53x speedup | abs_err=1.53e-05, rel_err=6.77e-02
  Workload e9e1e445...: PASSED | 1.645 ms | 781.73x speedup | abs_err=1.22e-04, rel_err=3.40e-02
  Workload b8c8dc3c...: PASSED | 1.631 ms | 786.69x speedup | abs_err=6.10e-05, rel_err=1.56e-01
  Workload a9540651...: PASSED | 2.060 ms | 621.19x speedup | abs_err=1.53e-05, rel_err=4.84e-02
  Workload 06f21bb1...: PASSED | 1.090 ms | 1170.00x speedup | abs_err=3.05e-05, rel_err=2.05e-02
  Workload c2931c92...: PASSED | 1.887 ms | 682.00x speedup | abs_err=1.22e-04, rel_err=1.25e-01
  Workload 618df04a...: PASSED | 1.588 ms | 807.36x speedup | abs_err=1.53e-05, rel_err=1.14e-01
  Workload 26244fb4...: PASSED | 1.975 ms | 649.10x speedup | abs_err=7.63e-06, rel_err=1.88e-01
  Workload a2629e02...: PASSED | 1.964 ms | 655.45x speedup | abs_err=6.10e-05, rel_err=2.97e-01
  Workload 9a5d694b...: PASSED | 1.371 ms | 932.70x speedup | abs_err=6.10e-05, rel_err=5.64e-02
  Workload 410794d4...: PASSED | 1.536 ms | 838.74x speedup | abs_err=1.53e-05, rel_err=1.67e-02
  Workload 7ba9d519...: PASSED | 1.362 ms | 458.01x speedup | abs_err=6.10e-05, rel_err=5.87e-02
  Workload 043e74e4...: PASSED | 0.031 ms | 170.43x speedup | abs_err=1.19e-07, rel_err=7.32e-03
  Workload ef9515b6...: PASSED | 0.030 ms | 93.37x speedup | abs_err=2.98e-08, rel_err=1.82e-03
  Workload f622a11d...: PASSED | 0.033 ms | 84.23x speedup | abs_err=5.96e-07, rel_err=1.16e-02
  Workload 1cf8e175...: PASSED | 0.034 ms | 195.49x speedup | abs_err=1.19e-07, rel_err=4.49e-03
  Workload 9343fd82...: PASSED | 0.045 ms | 177.91x speedup | abs_err=1.91e-06, rel_err=8.95e-03
  Workload f4926229...: PASSED | 0.633 ms | 340.70x speedup | abs_err=6.10e-05, rel_err=4.59e-02
  Workload 109addb1...: PASSED | 1.205 ms | 424.56x speedup | abs_err=6.10e-05, rel_err=1.80e-02
  Workload c5257f65...: PASSED | 0.442 ms | 348.51x speedup | abs_err=1.91e-06, rel_err=4.43e-02
  Workload f5619793...: PASSED | 0.432 ms | 357.95x speedup | abs_err=6.10e-05, rel_err=4.84e-02
  Workload fdf5f1f4...: PASSED | 0.038 ms | 196.81x speedup | abs_err=1.19e-07, rel_err=7.16e-03
  Workload 87bff084...: PASSED | 0.071 ms | 452.72x speedup | abs_err=1.91e-06, rel_err=6.62e-03
  Workload e92dafeb...: PASSED | 0.076 ms | 499.62x speedup | abs_err=1.91e-06, rel_err=2.63e-02
  Workload 1d0cc342...: PASSED | 0.124 ms | 374.10x speedup | abs_err=3.81e-06, rel_err=5.64e-02
  Workload 1b441950...: PASSED | 0.057 ms | 255.10x speedup | abs_err=1.19e-07, rel_err=1.07e-02
  Workload 19c6ab20...: PASSED | 0.059 ms | 246.95x speedup | abs_err=1.07e-06, rel_err=1.32e-02
  Workload 25d9c14d...: PASSED | 0.057 ms | 370.96x speedup | abs_err=4.77e-07, rel_err=1.41e-02
  Workload 3215fe5f...: PASSED | 0.163 ms | 331.26x speedup | abs_err=1.91e-06, rel_err=7.40e-03
  Workload 6f1ad833...: PASSED | 0.958 ms | 345.69x speedup | abs_err=6.10e-05, rel_err=4.11e-02
  Workload e44ba4d3...: PASSED | 0.532 ms | 469.30x speedup | abs_err=7.63e-06, rel_err=2.16e-02
  Workload fc7a2bcb...: PASSED | 0.067 ms | 331.04x speedup | abs_err=9.54e-07, rel_err=3.57e-02
  Workload 5d26ac5b...: PASSED | 0.070 ms | 320.19x speedup | abs_err=3.81e-06, rel_err=2.44e-02
  Workload ed66c791...: PASSED | 0.078 ms | 236.78x speedup | abs_err=1.91e-06, rel_err=3.70e-02
  Workload ba95d412...: PASSED | 0.094 ms | 314.11x speedup | abs_err=4.77e-07, rel_err=1.36e-02
  Workload 078a41ea...: PASSED | 0.047 ms | 276.80x speedup | abs_err=9.54e-07, rel_err=1.04e-02
  Workload d2b5a221...: PASSED | 0.042 ms | 316.36x speedup | abs_err=9.54e-07, rel_err=1.25e-02
  Workload aaa378be...: PASSED | 0.705 ms | 402.92x speedup | abs_err=3.81e-06, rel_err=1.22e-02
  Workload c2bb4f66...: PASSED | 0.707 ms | 397.41x speedup | abs_err=3.05e-05, rel_err=8.00e-02
  Workload f2f01c2c...: PASSED | 0.034 ms | 172.99x speedup | abs_err=5.96e-08, rel_err=4.96e-03
  Workload 15856e8c...: PASSED | 1.241 ms | 382.79x speedup | abs_err=7.63e-06, rel_err=1.94e-02
  Workload a39aa135...: PASSED | 0.052 ms | 188.80x speedup | abs_err=2.98e-07, rel_err=1.93e-02
  Workload 339a7ff4...: PASSED | 0.029 ms | 94.97x speedup | abs_err=1.19e-07, rel_err=4.82e-03
  Workload d8f4a9ae...: PASSED | 0.038 ms | 151.77x speedup | abs_err=2.24e-08, rel_err=6.33e-03
  Workload d3dc3577...: PASSED | 0.039 ms | 148.20x speedup | abs_err=3.58e-07, rel_err=1.52e-02
  Workload ce832e76...: PASSED | 0.221 ms | 326.54x speedup | abs_err=3.81e-06, rel_err=9.66e-03
  Workload 6fbc155c...: PASSED | 0.330 ms | 393.88x speedup | abs_err=3.81e-06, rel_err=7.63e-03
  Workload a87ded8a...: PASSED | 0.728 ms | 617.72x speedup | abs_err=1.22e-04, rel_err=4.99e-02
  Workload 62447caf...: PASSED | 0.037 ms | 135.04x speedup | abs_err=1.19e-07, rel_err=2.48e-02
  Workload fd072ba6...: PASSED | 0.056 ms | 325.69x speedup | abs_err=1.91e-06, rel_err=9.11e-03
  Workload 35ea9bbe...: PASSED | 0.053 ms | 340.73x speedup | abs_err=1.91e-06, rel_err=2.41e-02
  Workload 1aa8cf18...: PASSED | 0.035 ms | 165.09x speedup | abs_err=5.96e-08, rel_err=7.61e-03
  Workload d5f5c00c...: PASSED | 0.036 ms | 142.13x speedup | abs_err=1.49e-08, rel_err=1.71e-03
  Workload d5aa60dc...: PASSED | 0.115 ms | 351.26x speedup | abs_err=1.91e-06, rel_err=3.37e-02
  Workload 28b70283...: PASSED | 0.034 ms | 119.42x speedup | abs_err=9.54e-07, rel_err=7.52e-03
  Workload 73b8cc85...: PASSED | 0.069 ms | 223.93x speedup | abs_err=1.91e-06, rel_err=6.50e-03
  Workload 2683c087...: PASSED | 0.068 ms | 228.21x speedup | abs_err=7.63e-06, rel_err=1.17e-02
  Workload 4b94d568...: PASSED | 0.227 ms | 398.96x speedup | abs_err=7.63e-06, rel_err=1.64e-02
  Workload a0eb2dc2...: PASSED | 0.065 ms | 225.61x speedup | abs_err=3.81e-06, rel_err=6.16e-03
  Workload f3d30cb9...: PASSED | 0.056 ms | 245.32x speedup | abs_err=2.98e-07, rel_err=6.35e-03
  Workload 7a7deca8...: PASSED | 0.059 ms | 215.36x speedup | abs_err=3.81e-06, rel_err=7.69e-03
  Workload 977d19f8...: PASSED | 0.036 ms | 136.16x speedup | abs_err=1.19e-07, rel_err=5.71e-03
  Workload 02c1e5f0...: PASSED | 0.036 ms | 137.76x speedup | abs_err=4.77e-07, rel_err=5.65e-03
  Workload 5a91aa02...: PASSED | 0.148 ms | 426.92x speedup | abs_err=1.91e-06, rel_err=1.13e-02
  Workload 8e7ef744...: PASSED | 0.031 ms | 69.97x speedup | abs_err=2.98e-08, rel_err=3.73e-03
  Workload 85d7becb...: PASSED | 0.033 ms | 92.98x speedup | abs_err=3.58e-07, rel_err=5.67e-03
  Workload e286a4f4...: PASSED | 0.550 ms | 514.83x speedup | abs_err=7.63e-06, rel_err=2.26e-02
  Workload 08d4f2c4...: PASSED | 0.427 ms | 356.79x speedup | abs_err=3.81e-06, rel_err=1.11e-02
  Workload 9c1ef562...: PASSED | 0.425 ms | 360.15x speedup | abs_err=3.05e-05, rel_err=4.97e-02
  Workload bfd8f7b6...: PASSED | 0.129 ms | 303.24x speedup | abs_err=3.81e-06, rel_err=7.58e-03
  Workload c358edcd...: PASSED | 0.922 ms | 347.63x speedup | abs_err=1.53e-05, rel_err=7.67e-03
  Workload f203fdcd...: PASSED | 0.043 ms | 179.88x speedup | abs_err=2.38e-07, rel_err=9.32e-03
  Workload 33a38713...: PASSED | 0.058 ms | 320.73x speedup | abs_err=1.91e-06, rel_err=3.08e-02
  Workload 3a77dfec...: PASSED | 0.053 ms | 229.05x speedup | abs_err=5.96e-08, rel_err=7.50e-03
  Workload ea27be17...: PASSED | 0.049 ms | 247.34x speedup | abs_err=7.15e-07, rel_err=8.32e-03
  Workload 49ef89d2...: PASSED | 0.053 ms | 202.99x speedup | abs_err=1.91e-06, rel_err=3.87e-02
  Workload 056224b8...: PASSED | 0.054 ms | 202.11x speedup | abs_err=4.77e-07, rel_err=6.69e-03
  Workload 685d26ff...: PASSED | 0.030 ms | 76.21x speedup | abs_err=1.19e-07, rel_err=5.69e-03
  Workload 352c9ace...: PASSED | 0.255 ms | 322.55x speedup | abs_err=3.05e-05, rel_err=2.33e-02
  Workload 2c9693b4...: PASSED | 0.048 ms | 165.23x speedup | abs_err=2.98e-08, rel_err=5.83e-03
  Workload 27f44fd6...: PASSED | 0.046 ms | 173.18x speedup | abs_err=1.91e-06, rel_err=1.05e-02
  Workload 07aa7922...: PASSED | 2.545 ms | 352.83x speedup | abs_err=3.05e-05, rel_err=7.11e-02
  Workload eaa0fd47...: PASSED | 0.029 ms | 85.62x speedup | abs_err=9.54e-07, rel_err=6.45e-03
  Workload f105eda8...: PASSED | 0.211 ms | 491.00x speedup | abs_err=6.10e-05, rel_err=1.49e-02
  Workload cd979341...: PASSED | 0.051 ms | 239.35x speedup | abs_err=3.81e-06, rel_err=6.88e-03
  Workload 43bf9699...: PASSED | 0.413 ms | 366.38x speedup | abs_err=1.91e-06, rel_err=7.69e-03
  Workload 54856fec...: PASSED | 0.415 ms | 363.04x speedup | abs_err=3.05e-05, rel_err=1.01e-01
  Workload 2ba465c0...: PASSED | 0.035 ms | 135.33x speedup | abs_err=2.38e-07, rel_err=7.69e-03
  Workload 1efaf2a9...: PASSED | 0.037 ms | 187.30x speedup | abs_err=2.98e-08, rel_err=8.85e-03
  Workload a01a3f93...: PASSED | 0.971 ms | 368.64x speedup | abs_err=1.22e-04, rel_err=4.51e-02
  Workload cc241d2e...: PASSED | 0.034 ms | 110.98x speedup | abs_err=1.49e-07, rel_err=6.82e-03
[2026-04-07T00:39:16] Local entrypoint finished in 453.95s
Stopping app - local entrypoint completed.
Runner terminated.
✓ App completed. View run at 
https://modal.com/apps/shjj1504/main/ap-zspCPqlZKLS1qx1jnh6MKL

```
