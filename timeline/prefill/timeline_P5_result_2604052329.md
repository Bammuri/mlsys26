# Prefill Timeline — P5 Result — 2604052329

## Optimization summary

이번 P5 iteration에서는 token loop에서 반복되던 per-head scalar overhead를 loop 밖으로 hoist했다.

### Applied optimization

1. `exp(A_log[head_idx])`를 token loop 밖으로 hoist
2. `dt_bias[head_idx]`를 token loop 밖으로 hoist
3. `scale`를 float scalar로 한 번만 캐스팅
4. gate compute에서 반복되는 scalar global loads / exp 일부 제거
5. full decision gate config를 `warmup_runs=1, iterations=3, num_trials=2`로 바꿈

핵심 의도:

> long-seq에서 token마다 누적되는 작은 scalar 반복 비용을 줄이고,
> 이후 loop의 정식 판단 기준을 더 안정적인 full decision-gate config로 옮기기

---

## Expected result vs actual result

### Expected before running

이번 iteration부터 full 판단 기준은 다음으로 바뀌었다.

```bash
modal run scripts/run_modal.py --decision-gate --max-workloads 100
```

즉, P3의 `--quick --max-workloads 100`와는 정확히 같은 측정 프로토콜이 아니다.

예상:

1. quick 5는 PASS 유지
2. full decision gate도 PASS 유지
3. scalar-hoist는 개선 폭이 크지 않을 수 있지만, long-seq 누적 비용 감소로 small positive gain 가능
4. 이번 결과는 동시에 **새 decision-gate baseline**도 겸한다

### Actual results

#### 3-1. Quick working check

실행:

```bash
modal run scripts/run_modal.py --quick --max-workloads 5
```

결과:

- `PASSED=5/5`
- avg latency: `0.204 ms`
- avg speedup: `25.33x`
- worst abs error: `1.91e-06`
- worst rel error: `2.29e-02`

#### 3-2. Full decision gate extraction

실행:

```bash
modal run scripts/run_modal.py --decision-gate --max-workloads 100
```

결과:

- `PASSED=100/100`
- avg latency: `4.917 ms`
- avg speedup: `50.49x`
- worst abs error: `1.22e-04`
- worst rel error: `2.46e-01`
- end-to-end wall time: `881.54s`

대표 로그 예시:

- `Workload d0ce7b5d...: PASSED | 13.397 ms | 111.90x speedup | abs_err=1.53e-05, rel_err=2.02e-01`
- `Workload ce832e76...: PASSED | 2.441 ms | 36.02x speedup | abs_err=3.81e-06, rel_err=2.34e-02`
- `Workload 07aa7922...: PASSED | 30.601 ms | 34.70x speedup | abs_err=1.53e-05, rel_err=8.62e-02`

---

## Judgement

이번 iteration은 **keep** 한다.

### Decision rule used

사용자 지시에 따라, 최종 판단은 **full decision gate** 기준으로 수행했다.

### Why this is a keep

#### 1. Full decision gate is clean and stable

- `PASSED=100/100`
- compile/runtime failure 없음
- numerical quality는 P3 계열 수준 유지

즉,

> 새 decision-gate 기준선으로 쓰기에 충분히 안정적이다.

#### 2. Absolute latency remains slightly better than the previous full figure we had

직전 full-100 quick 기록(P3)은:

- avg latency: `4.935 ms`

이번 decision-gate full은:

- avg latency: `4.917 ms`

측정 프로토콜이 달라 strictly apples-to-apples는 아니지만,
더 무거운 gate 설정에서도 평균 latency가 여전히 좋게 나온 것은 긍정적이다.

#### 3. This iteration also establishes the new baseline for future loops

이번부터는 full 판단 설정이 바뀌었기 때문에,
P5 결과는 단순한 성능 개선 결과이기도 하지만 동시에:

> **향후 loop들의 official-ish development baseline**

역할도 한다.

---

## Detailed analysis: why this likely helped

### 1. The optimization targeted repeated scalar work in the longest loops

현재 kernel은 token loop 안에서 매번:

- `A_log[head_idx]` load
- `exp(A_log[head_idx])`
- `dt_bias[head_idx]` load
- `scale` float cast

같은 scalar work를 반복하고 있었다.

이 작업은 한 번당 작아 보이지만,
long-seq workload에서는 token 수만큼 누적된다.

즉,

> 아주 큰 구조 변경이 아니라도,
> long-tail에서 의미 있는 cumulative cost가 될 수 있다.

### 2. This is the kind of safe optimization P4 was not

P4는 CTA mapping 자체를 바꾸다가 correctness를 완전히 깨뜨렸다.

반면 P5는:

- math 불변
- layout 불변
- CTA mapping 불변
- shared state structure 불변

이다.

즉,

> baseline을 흔들지 않고 반복 scalar overhead만 줄이는 안전한 최적화였다.

### 3. Why the gain is small, not dramatic

이 최적화는 본질적으로 micro-optimization이다.

줄인 것은:

- per-token scalar exp/load/cast overhead

반면 아직 그대로인 것은:

- full fp32 state tile
- row-per-thread 구조
- token마다 full row sweep
- long-tail workload dominance

즉,

> 성능 ceiling을 움직이는 구조적 최적화는 아니고,
> 이미 개선된 kernel 위에 얹힌 low-risk cleanup gain에 가깝다.

### 4. Why it is still worth keeping

작은 최적화라도 keep할 가치가 있는 이유는:

1. correctness risk가 거의 없었고
2. full decision gate가 100/100 PASS였고
3. 앞으로 모든 loop가 이 gate를 기준으로 비교되기 때문이다

즉,

> P5는 “작은 개선 + 안정적인 새 기준선”이라는 의미가 있다.

---

## Relationship to future loops

이제 다음 loop부터는 full 판단 기준이 고정된다:

```bash
modal run scripts/run_modal.py --decision-gate --max-workloads 100
```

따라서 이후 P6+에서는:

- P5 decision-gate 결과를 직접 baseline으로 삼아
- full 100 평균 latency / wall time / representative workloads를 비교하면 된다

---

## Full 100-workload pref log

```text
✓ Initialized. View run at 
https://modal.com/apps/shjj1504/main/ap-0GjEoGitqHDozdkuqxkf6b
✓ Created objects.
├── 🔨 Created mount /home/hyu/flashinfer/mlsys26/scripts/run_modal.py
└── 🔨 Created function run_benchmark.
[2026-04-05T23:30:16] Packing solution from source files...
Solution packed: /home/hyu/flashinfer/mlsys26/solution.json
  Name: my-team-solution-v1
  Definition: gdn_prefill_qk4_v8_d128_k_last
  Author: team-name
  Config language: cuda
  Runtime language: cuda
[2026-04-05T23:30:16] Validating solution JSON...
[2026-04-05T23:30:16] Loaded solution my-team-solution-v1 (gdn_prefill_qk4_v8_d128_k_last) in 0.01s
[2026-04-05T23:30:16] Decision-gate mode enabled: warmup_runs=1, iterations=10, num_trials=2, use_isolated_runner=False
[2026-04-05T23:30:16] Dispatching benchmark to Modal B200...

==========
== CUDA ==
==========

CUDA Version 13.0.2

Container image Copyright (c) 2016-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

This container image and its contents are governed by the NVIDIA Deep Learning Container License.
By pulling and using the container, you accept the terms and conditions of this license:
https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license

A copy of this license is made available in this container at /NGC-DL-CONTAINER-LICENSE for your convenience.

[2026-04-05T14:30:26] Remote benchmark start: solution=my-team-solution-v1, definition=gdn_prefill_qk4_v8_d128_k_last
[2026-04-05T14:30:26] BenchmarkConfig(warmup_runs=1, iterations=10, num_trials=2)
[2026-04-05T14:30:26] Loading trace set from /data/data/mlsys26-contest
[2026-04-05T14:30:26] Loaded trace set in 0.20s
[2026-04-05T14:30:26] Running benchmark across 100 workloads
[2026-04-05T14:45:02] Benchmark completed in 875.65s
[2026-04-05T23:44:58] Received benchmark results in 881.54s

gdn_prefill_qk4_v8_d128_k_last:
  workloads: 100
  status counts: PASSED=100
  avg latency: 4.917 ms
  avg speedup: 50.49x
  worst abs error: 1.22e-04
  worst rel error: 2.46e-01
  Workload 77daf91d...: PASSED | 0.064 ms | 22.37x speedup | abs_err=4.47e-08, rel_err=9.53e-03
  Workload ba08a83e...: PASSED | 0.761 ms | 32.51x speedup | abs_err=4.77e-07, rel_err=7.50e-03
  Workload c7846f96...: PASSED | 0.760 ms | 32.43x speedup | abs_err=3.81e-06, rel_err=9.91e-03
  Workload d0ce7b5d...: PASSED | 13.397 ms | 111.90x speedup | abs_err=1.53e-05, rel_err=2.02e-01
  Workload 5b8a0e4b...: PASSED | 20.955 ms | 74.31x speedup | abs_err=1.53e-05, rel_err=6.39e-02
  Workload 5d3fc66a...: PASSED | 20.898 ms | 71.78x speedup | abs_err=6.10e-05, rel_err=1.61e-01
  Workload 4b6143dd...: PASSED | 4.988 ms | 156.53x speedup | abs_err=7.63e-06, rel_err=3.14e-02
  Workload 5835a2bc...: PASSED | 26.944 ms | 55.48x speedup | abs_err=1.53e-05, rel_err=3.06e-02
  Workload cc310f94...: PASSED | 17.852 ms | 86.66x speedup | abs_err=6.10e-05, rel_err=9.95e-02
  Workload d49df0b2...: PASSED | 15.518 ms | 97.74x speedup | abs_err=3.05e-05, rel_err=8.77e-02
  Workload e9e1e445...: PASSED | 12.190 ms | 124.30x speedup | abs_err=1.53e-05, rel_err=3.65e-02
  Workload b8c8dc3c...: PASSED | 12.202 ms | 127.61x speedup | abs_err=6.10e-05, rel_err=2.46e-01
  Workload a9540651...: PASSED | 19.049 ms | 78.61x speedup | abs_err=6.10e-05, rel_err=6.65e-02
  Workload 06f21bb1...: PASSED | 9.821 ms | 155.98x speedup | abs_err=1.53e-05, rel_err=4.09e-02
  Workload c2931c92...: PASSED | 19.153 ms | 80.61x speedup | abs_err=1.22e-04, rel_err=1.10e-01
  Workload 618df04a...: PASSED | 17.154 ms | 87.25x speedup | abs_err=6.10e-05, rel_err=1.21e-01
  Workload 26244fb4...: PASSED | 16.661 ms | 91.77x speedup | abs_err=7.63e-06, rel_err=1.44e-01
  Workload a2629e02...: PASSED | 16.630 ms | 93.52x speedup | abs_err=6.10e-05, rel_err=2.43e-01
  Workload 9a5d694b...: PASSED | 11.525 ms | 138.51x speedup | abs_err=6.10e-05, rel_err=3.25e-02
  Workload 410794d4...: PASSED | 13.721 ms | 112.65x speedup | abs_err=1.53e-05, rel_err=1.71e-02
  Workload 7ba9d519...: PASSED | 15.678 ms | 46.89x speedup | abs_err=6.10e-05, rel_err=5.78e-02
  Workload 043e74e4...: PASSED | 0.128 ms | 50.48x speedup | abs_err=1.19e-07, rel_err=6.68e-03
  Workload ef9515b6...: PASSED | 0.119 ms | 26.95x speedup | abs_err=1.19e-07, rel_err=7.04e-03
  Workload f622a11d...: PASSED | 0.118 ms | 27.32x speedup | abs_err=4.77e-07, rel_err=1.92e-02
  Workload 1cf8e175...: PASSED | 0.156 ms | 49.25x speedup | abs_err=9.54e-07, rel_err=4.95e-03
  Workload 9343fd82...: PASSED | 0.300 ms | 30.93x speedup | abs_err=1.91e-06, rel_err=7.63e-03
  Workload f4926229...: PASSED | 7.503 ms | 34.84x speedup | abs_err=6.10e-05, rel_err=4.59e-02
  Workload 109addb1...: PASSED | 14.316 ms | 43.48x speedup | abs_err=6.10e-05, rel_err=2.49e-02
  Workload c5257f65...: PASSED | 5.150 ms | 36.29x speedup | abs_err=1.91e-06, rel_err=3.44e-02
  Workload f5619793...: PASSED | 5.164 ms | 34.88x speedup | abs_err=6.10e-05, rel_err=5.26e-02
  Workload fdf5f1f4...: PASSED | 0.183 ms | 49.78x speedup | abs_err=1.19e-07, rel_err=7.16e-03
  Workload 87bff084...: PASSED | 0.550 ms | 68.13x speedup | abs_err=3.81e-06, rel_err=6.89e-03
  Workload e92dafeb...: PASSED | 0.613 ms | 72.62x speedup | abs_err=4.77e-07, rel_err=2.58e-02
  Workload 1d0cc342...: PASSED | 1.269 ms | 44.74x speedup | abs_err=3.81e-06, rel_err=5.42e-02
  Workload 1b441950...: PASSED | 0.428 ms | 40.20x speedup | abs_err=9.54e-07, rel_err=1.07e-02
  Workload 19c6ab20...: PASSED | 0.428 ms | 40.34x speedup | abs_err=7.63e-06, rel_err=7.78e-03
  Workload 25d9c14d...: PASSED | 0.415 ms | 61.59x speedup | abs_err=1.91e-06, rel_err=9.87e-03
  Workload 3215fe5f...: PASSED | 1.722 ms | 38.58x speedup | abs_err=3.81e-06, rel_err=7.41e-03
  Workload 6f1ad833...: PASSED | 11.455 ms | 34.45x speedup | abs_err=6.10e-05, rel_err=2.62e-02
  Workload e44ba4d3...: PASSED | 6.158 ms | 47.40x speedup | abs_err=7.63e-06, rel_err=1.53e-02
  Workload fc7a2bcb...: PASSED | 0.556 ms | 50.51x speedup | abs_err=2.38e-07, rel_err=1.72e-02
  Workload 5d26ac5b...: PASSED | 0.556 ms | 48.46x speedup | abs_err=3.81e-06, rel_err=2.10e-02
  Workload ed66c791...: PASSED | 0.664 ms | 32.51x speedup | abs_err=3.81e-06, rel_err=3.70e-02
  Workload ba95d412...: PASSED | 0.878 ms | 41.42x speedup | abs_err=4.77e-07, rel_err=1.50e-02
  Workload 078a41ea...: PASSED | 0.335 ms | 47.00x speedup | abs_err=2.38e-07, rel_err=1.04e-02
  Workload d2b5a221...: PASSED | 0.202 ms | 77.00x speedup | abs_err=9.54e-07, rel_err=1.35e-02
  Workload aaa378be...: PASSED | 8.262 ms | 41.65x speedup | abs_err=7.63e-06, rel_err=1.84e-02
  Workload c2bb4f66...: PASSED | 8.258 ms | 41.70x speedup | abs_err=3.05e-05, rel_err=8.34e-02
  Workload f2f01c2c...: PASSED | 0.140 ms | 50.69x speedup | abs_err=8.94e-08, rel_err=4.96e-03
  Workload 15856e8c...: PASSED | 14.814 ms | 38.82x speedup | abs_err=7.63e-06, rel_err=1.59e-02
  Workload a39aa135...: PASSED | 0.363 ms | 32.74x speedup | abs_err=2.38e-07, rel_err=2.29e-02
  Workload 339a7ff4...: PASSED | 0.118 ms | 27.07x speedup | abs_err=4.77e-07, rel_err=5.72e-03
  Workload d8f4a9ae...: PASSED | 0.221 ms | 30.35x speedup | abs_err=1.19e-07, rel_err=7.04e-03
  Workload d3dc3577...: PASSED | 0.222 ms | 30.39x speedup | abs_err=3.58e-07, rel_err=1.04e-02
  Workload ce832e76...: PASSED | 2.441 ms | 36.02x speedup | abs_err=3.81e-06, rel_err=2.34e-02
  Workload 6fbc155c...: PASSED | 3.744 ms | 42.55x speedup | abs_err=7.63e-06, rel_err=7.75e-03
  Workload a87ded8a...: PASSED | 8.420 ms | 63.62x speedup | abs_err=1.22e-04, rel_err=4.93e-02
  Workload 62447caf...: PASSED | 0.195 ms | 29.91x speedup | abs_err=1.49e-07, rel_err=1.66e-02
  Workload fd072ba6...: PASSED | 0.404 ms | 52.66x speedup | abs_err=3.81e-06, rel_err=1.77e-02
  Workload 35ea9bbe...: PASSED | 0.404 ms | 52.80x speedup | abs_err=1.91e-06, rel_err=4.58e-02
  Workload 1aa8cf18...: PASSED | 0.130 ms | 52.39x speedup | abs_err=8.94e-08, rel_err=7.61e-03
  Workload d5f5c00c...: PASSED | 0.200 ms | 29.93x speedup | abs_err=1.49e-08, rel_err=1.75e-03
  Workload d5aa60dc...: PASSED | 1.155 ms | 40.78x speedup | abs_err=1.91e-06, rel_err=4.99e-02
  Workload 28b70283...: PASSED | 0.161 ms | 30.27x speedup | abs_err=1.19e-07, rel_err=8.86e-03
  Workload 73b8cc85...: PASSED | 0.560 ms | 32.59x speedup | abs_err=1.19e-07, rel_err=7.04e-03
  Workload 2683c087...: PASSED | 0.559 ms | 32.47x speedup | abs_err=7.63e-06, rel_err=1.63e-02
  Workload 4b94d568...: PASSED | 2.528 ms | 41.82x speedup | abs_err=3.81e-06, rel_err=1.12e-02
  Workload a0eb2dc2...: PASSED | 0.532 ms | 32.29x speedup | abs_err=3.81e-06, rel_err=6.37e-03
  Workload f3d30cb9...: PASSED | 0.429 ms | 38.57x speedup | abs_err=2.98e-07, rel_err=2.28e-02
  Workload 7a7deca8...: PASSED | 0.461 ms | 32.24x speedup | abs_err=3.81e-06, rel_err=7.69e-03
  Workload 977d19f8...: PASSED | 0.195 ms | 29.89x speedup | abs_err=1.19e-07, rel_err=7.04e-03
  Workload 02c1e5f0...: PASSED | 0.194 ms | 29.98x speedup | abs_err=7.63e-06, rel_err=6.94e-03
  Workload 5a91aa02...: PASSED | 1.525 ms | 50.05x speedup | abs_err=3.81e-06, rel_err=1.48e-02
  Workload 8e7ef744...: PASSED | 0.097 ms | 25.39x speedup | abs_err=2.98e-08, rel_err=3.76e-03
  Workload 85d7becb...: PASSED | 0.129 ms | 28.70x speedup | abs_err=3.58e-07, rel_err=6.89e-03
  Workload e286a4f4...: PASSED | 6.412 ms | 51.64x speedup | abs_err=7.63e-06, rel_err=3.52e-02
  Workload 08d4f2c4...: PASSED | 4.955 ms | 37.47x speedup | abs_err=7.63e-06, rel_err=7.79e-03
  Workload 9c1ef562...: PASSED | 4.949 ms | 37.03x speedup | abs_err=3.05e-05, rel_err=6.02e-02
  Workload bfd8f7b6...: PASSED | 1.308 ms | 36.46x speedup | abs_err=7.63e-06, rel_err=3.34e-02
  Workload c358edcd...: PASSED | 10.978 ms | 35.59x speedup | abs_err=1.53e-05, rel_err=7.69e-03
  Workload f203fdcd...: PASSED | 0.293 ms | 31.27x speedup | abs_err=2.38e-07, rel_err=1.20e-02
  Workload 33a38713...: PASSED | 0.414 ms | 53.11x speedup | abs_err=1.49e-07, rel_err=2.21e-02
  Workload 3a77dfec...: PASSED | 0.353 ms | 40.48x speedup | abs_err=9.54e-07, rel_err=7.25e-03
  Workload ea27be17...: PASSED | 0.353 ms | 40.51x speedup | abs_err=7.15e-07, rel_err=5.46e-02
  Workload 49ef89d2...: PASSED | 0.396 ms | 31.83x speedup | abs_err=1.91e-06, rel_err=5.21e-02
  Workload 056224b8...: PASSED | 0.400 ms | 32.12x speedup | abs_err=2.38e-07, rel_err=6.62e-03
  Workload 685d26ff...: PASSED | 0.102 ms | 26.05x speedup | abs_err=7.45e-08, rel_err=3.43e-03
  Workload 352c9ace...: PASSED | 2.884 ms | 34.69x speedup | abs_err=3.05e-05, rel_err=2.50e-02
  Workload 2c9693b4...: PASSED | 0.299 ms | 32.44x speedup | abs_err=1.19e-07, rel_err=1.84e-02
  Workload 27f44fd6...: PASSED | 0.297 ms | 31.46x speedup | abs_err=1.91e-06, rel_err=1.57e-02
  Workload 07aa7922...: PASSED | 30.601 ms | 34.70x speedup | abs_err=1.53e-05, rel_err=8.62e-02
  Workload eaa0fd47...: PASSED | 0.107 ms | 27.66x speedup | abs_err=9.54e-07, rel_err=6.45e-03
  Workload f105eda8...: PASSED | 2.278 ms | 53.13x speedup | abs_err=6.10e-05, rel_err=1.94e-02
  Workload cd979341...: PASSED | 0.358 ms | 40.11x speedup | abs_err=2.38e-07, rel_err=7.03e-03
  Workload 43bf9699...: PASSED | 4.879 ms | 36.33x speedup | abs_err=7.63e-06, rel_err=8.02e-03
  Workload 54856fec...: PASSED | 4.880 ms | 36.17x speedup | abs_err=3.05e-05, rel_err=9.25e-02
  Workload 2ba465c0...: PASSED | 0.107 ms | 51.65x speedup | abs_err=5.96e-08, rel_err=6.84e-03
  Workload 1efaf2a9...: PASSED | 0.179 ms | 45.42x speedup | abs_err=5.96e-08, rel_err=4.48e-03
  Workload a01a3f93...: PASSED | 11.569 ms | 36.32x speedup | abs_err=3.05e-05, rel_err=3.70e-02
  Workload cc241d2e...: PASSED | 0.157 ms | 28.88x speedup | abs_err=1.49e-07, rel_err=6.82e-03
[2026-04-05T23:44:58] Local entrypoint finished in 881.54s
[2026-04-05T14:45:02] Remote benchmark finished in 875.85s
Stopping app - local entrypoint completed.
Runner terminated.
✓ App completed. View run at 
https://modal.com/apps/shjj1504/main/ap-0GjEoGitqHDozdkuqxkf6b

```
