# Prefill Timeline — P4 Result — 2026-04-07

## Optimization summary

이번 P4 iteration에서는 P3의 row-per-warp SRTP를 유지하면서,
**block을 1 warp까지 줄여 CTA 공급량을 극대화하는 variant**를 시험했다.

### Applied optimization

1. `kWarpsPerBlock = 2` → `1`
2. row-per-warp ownership 유지
3. block size `64` → `32`
4. `grid.x`를 2배로 늘려 더 많은 CTA를 공급
5. no-shared / no-spill / gate-beta precompute 구조 유지

핵심 의도:

> P3보다 더 많은 CTA를 공급해 tiny/long-tail underfill을 더 줄일 수 있는지 확인

---

## Expected result before testing

baseline (keep baseline, full 100, `warmup_runs=1, iterations=5, num_trials=2`):

- avg latency: `0.511 ms`

예상:

1. quick 5 PASS 유지
2. full 100 PASS 유지
3. tiny/long-tail에서 positive gain 가능
4. throughput-heavy에서는 launch overhead 때문에 no-op 또는 slight regression 가능

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
- avg latency: `0.056 ms`
- worst abs error: `1.91e-06`
- worst rel error: `1.93e-02`

### 3. full decision gate

```bash
modal run scripts/run_modal.py --decision-gate --max-workloads 100
```

결과:

- `PASSED=100/100`
- avg latency: `0.539 ms`
- avg speedup: `348.83x`
- worst abs error: `1.22e-04`
- worst rel error: `2.97e-01`

판단 기준은 사용자 지시대로 **quick이 아니라 full 100 arithmetic-mean latency**다.

---

## Comparison against baseline

### Baseline → P4

- avg latency: `0.511 ms` → `0.539 ms`
- 절대 악화: `+0.028 ms`
- 상대 악화: 약 **+5.48%**

즉 이번 iteration은 full 기준으로 명확히 나빠졌다.

---

## Representative workload comparison

대표 workload를 보면 underfill이 큰 tiny에서는 좋아졌지만,
full mix에서는 이득이 유지되지 않았다.

- `77daf91d`: `0.035 ms` → `0.034 ms`  (소폭 개선)
- `ba08a83e`: `0.092 ms` → `0.089 ms`  (소폭 개선)
- `5835a2bc`: `2.169 ms` → `2.190 ms`  (소폭 악화)
- `07aa7922`: `2.651 ms` → `2.636 ms`  (소폭 개선)

겉으로 보면 mixed result처럼 보이지만,
full 100 arithmetic mean은 분명히 나빠졌다.

즉,

> 일부 low-seq case의 미세 개선이 전체 workload distribution 평균을 이기지 못했다.

---

## Detailed analysis: why it likely got worse

### 1. P4 over-optimized CTA supply and started to lose on per-block efficiency

P3는 2 warps/block이었다.
P4는 여기서 더 나아가 1 warp/block으로 줄였다.

이 변화는 block 수를 늘려 tiny underfill을 줄이는 데는 도움이 될 수 있다.
하지만 그만큼:

- block launch/retirement overhead 증가
- block scheduling metadata overhead 증가
- same total work를 더 많은 아주 작은 block으로 나누는 비용 증가

가 생긴다.

즉,

> P3는 CTA 공급량과 per-block efficiency의 균형점에 가까웠고,
> P4는 그 균형을 넘어선 과분할(over-fragmentation)에 가까웠다.

### 2. Why tiny improved but full average got worse

`77daf91d`와 `ba08a83e`는 small/single-seq 계열이라,
block 수가 늘어나는 효과를 직접 받기 쉽다.

반면 full 100에는:

- throughput-heavy
- medium multi-seq
- long-tail
- mixed-shape

workload가 모두 포함된다.

이 분포에서는 tiny-case launch gain보다,
**많아진 block 관리 비용과 reduced per-block work efficiency**의 합이 더 크게 작용했을 가능성이 높다.

즉,

> P4는 작은 case에선 좋아 보일 수 있지만,
> 전체 distribution 관점에서는 overfit된 launch shape였다.

### 3. P3 already captured the useful part of this idea

P3에서 4 warps/block → 2 warps/block으로 줄였을 때는,
full 평균이 `0.517 ms -> 0.511 ms`로 좋아졌다.

즉,

- 4 → 2: 유익한 granularity tuning
- 2 → 1: 지나친 granularity tuning

이라는 해석이 자연스럽다.

따라서 P4는 “CTA를 더 늘리자” 아이디어 자체가 틀렸다기보다는,
**이미 좋은 방향을 지나치게 밀어붙인 케이스**라고 볼 수 있다.

### 4. Why this should be judged by full, not quick

quick gate는 여전히 PASS했고,
avg latency도 `0.056 ms`로 나쁘지 않아 보였다.

하지만 full에서는 `0.539 ms`로 baseline보다 악화됐다.

이번 결과는 다시 한번 다음 원칙을 확인해 준다.

> quick는 working check일 뿐,
> keep/revert는 full arithmetic mean으로만 해야 한다.

### 5. What this says about the next search direction

P2와 P4를 합쳐 보면,
현재까지의 negative lessons는 분명하다.

- P2: warp당 serial work 증가 → 악화
- P4: CTA granularity를 지나치게 쪼갬 → 악화

즉 다음 loop는

> P3 keep baseline의 균형을 유지한 채,
> 다른 종류의 small tuning (예: memory hierarchy hint, representation tweak 등)

을 봐야 한다.

---

## Judgement

이번 iteration은 **reject** 한다.

### Why this is a reject

1. `PASSED=100/100`
2. 하지만 full avg latency가 `0.511 ms` → `0.539 ms`로 악화
3. 일부 small case improvement가 전체 distribution mean을 이기지 못함
4. 사용자 규칙상 quick가 아니라 full 평균으로 판단해야 함

따라서 사용자 규칙에 따라:

- **commit하지 않는다**
- **kernel code만 revert한다**
- 다음 loop로 넘어간다

---

## Decision under workflow rule

- keep/revert 기준: **full workload avg latency arithmetic mean**
- 결과: **REVERT**

---

## Full pref log

full pref log is saved in:

- `timeline/prefill/perf_P4_2604070004.txt`

아래는 동일 로그 전문이다.

```text
✓ Initialized. View run at 
https://modal.com/apps/shjj1504/main/ap-UhazThnFhgKwINJuDGUFrI
✓ Created objects.
├── 🔨 Created mount /home/hyu/flashinfer/mlsys26/scripts/run_modal.py
└── 🔨 Created function run_benchmark.
[2026-04-07T00:08:28] Packing solution from source files...
Solution packed: /home/hyu/flashinfer/mlsys26/solution.json
  Name: my-team-solution-v1
  Definition: gdn_prefill_qk4_v8_d128_k_last
  Author: team-name
  Config language: cuda
  Runtime language: cuda
[2026-04-07T00:08:28] Validating solution JSON...
[2026-04-07T00:08:28] Loaded solution my-team-solution-v1 (gdn_prefill_qk4_v8_d128_k_last) in 0.01s
[2026-04-07T00:08:28] Decision-gate mode enabled: warmup_runs=1, iterations=5, num_trials=2, use_isolated_runner=False
[2026-04-07T00:08:28] Dispatching benchmark to Modal B200...

==========
== CUDA ==
==========

CUDA Version 13.0.2

Container image Copyright (c) 2016-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

This container image and its contents are governed by the NVIDIA Deep Learning Container License.
By pulling and using the container, you accept the terms and conditions of this license:
https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license

A copy of this license is made available in this container at /NGC-DL-CONTAINER-LICENSE for your convenience.

[2026-04-06T15:08:35] Remote benchmark start: solution=my-team-solution-v1, definition=gdn_prefill_qk4_v8_d128_k_last
[2026-04-06T15:08:35] BenchmarkConfig(warmup_runs=1, iterations=5, num_trials=2)
[2026-04-06T15:08:35] Loading trace set from /data/data/mlsys26-contest
[2026-04-06T15:08:36] Loaded trace set in 0.48s
[2026-04-06T15:08:36] Running benchmark across 100 workloads
[2026-04-06T15:16:36] Benchmark completed in 480.23s
[2026-04-06T15:16:36] Remote benchmark finished in 480.71s
[2026-04-07T00:16:37] Received benchmark results in 488.79s

gdn_prefill_qk4_v8_d128_k_last:
  workloads: 100
  status counts: PASSED=100
  avg latency: 0.539 ms
  avg speedup: 348.83x
  worst abs error: 1.22e-04
  worst rel error: 2.97e-01
  Workload 77daf91d...: PASSED | 0.034 ms | 39.26x speedup | abs_err=4.47e-08, rel_err=9.53e-03
  Workload ba08a83e...: PASSED | 0.089 ms | 261.34x speedup | abs_err=7.63e-06, rel_err=7.50e-03
  Workload c7846f96...: PASSED | 0.088 ms | 261.16x speedup | abs_err=3.81e-06, rel_err=6.89e-03
  Workload d0ce7b5d...: PASSED | 1.573 ms | 834.66x speedup | abs_err=6.10e-05, rel_err=1.21e-01
  Workload 5b8a0e4b...: PASSED | 2.485 ms | 547.66x speedup | abs_err=1.53e-05, rel_err=4.03e-02
  Workload 5d3fc66a...: PASSED | 2.478 ms | 547.74x speedup | abs_err=6.10e-05, rel_err=1.59e-01
  Workload 4b6143dd...: PASSED | 0.655 ms | 1048.26x speedup | abs_err=7.63e-06, rel_err=2.79e-02
  Workload 5835a2bc...: PASSED | 2.190 ms | 644.72x speedup | abs_err=1.53e-05, rel_err=1.73e-02
  Workload cc310f94...: PASSED | 2.130 ms | 749.79x speedup | abs_err=1.22e-04, rel_err=1.33e-01
  Workload d49df0b2...: PASSED | 1.837 ms | 717.87x speedup | abs_err=1.53e-05, rel_err=6.77e-02
  Workload e9e1e445...: PASSED | 2.063 ms | 673.90x speedup | abs_err=1.22e-04, rel_err=3.40e-02
  Workload b8c8dc3c...: PASSED | 2.053 ms | 636.36x speedup | abs_err=6.10e-05, rel_err=1.56e-01
  Workload a9540651...: PASSED | 2.327 ms | 670.15x speedup | abs_err=1.53e-05, rel_err=4.84e-02
  Workload 06f21bb1...: PASSED | 1.457 ms | 910.46x speedup | abs_err=3.05e-05, rel_err=2.05e-02
  Workload c2931c92...: PASSED | 1.952 ms | 667.31x speedup | abs_err=1.22e-04, rel_err=1.25e-01
  Workload 618df04a...: PASSED | 1.748 ms | 743.40x speedup | abs_err=1.53e-05, rel_err=1.14e-01
  Workload 26244fb4...: PASSED | 2.223 ms | 604.30x speedup | abs_err=7.63e-06, rel_err=1.88e-01
  Workload a2629e02...: PASSED | 2.224 ms | 593.43x speedup | abs_err=6.10e-05, rel_err=2.97e-01
  Workload 9a5d694b...: PASSED | 1.519 ms | 847.78x speedup | abs_err=6.10e-05, rel_err=5.64e-02
  Workload 410794d4...: PASSED | 1.846 ms | 802.02x speedup | abs_err=1.53e-05, rel_err=1.67e-02
  Workload 7ba9d519...: PASSED | 1.412 ms | 467.26x speedup | abs_err=6.10e-05, rel_err=5.87e-02
  Workload 043e74e4...: PASSED | 0.036 ms | 152.60x speedup | abs_err=1.19e-07, rel_err=7.32e-03
  Workload ef9515b6...: PASSED | 0.035 ms | 83.59x speedup | abs_err=2.98e-08, rel_err=1.82e-03
  Workload f622a11d...: PASSED | 0.036 ms | 79.56x speedup | abs_err=5.96e-07, rel_err=1.16e-02
  Workload 1cf8e175...: PASSED | 0.038 ms | 177.72x speedup | abs_err=1.19e-07, rel_err=4.49e-03
  Workload 9343fd82...: PASSED | 0.052 ms | 165.63x speedup | abs_err=1.91e-06, rel_err=8.95e-03
  Workload f4926229...: PASSED | 0.660 ms | 348.49x speedup | abs_err=6.10e-05, rel_err=4.59e-02
  Workload 109addb1...: PASSED | 1.251 ms | 427.18x speedup | abs_err=6.10e-05, rel_err=1.80e-02
  Workload c5257f65...: PASSED | 0.469 ms | 344.62x speedup | abs_err=1.91e-06, rel_err=4.43e-02
  Workload f5619793...: PASSED | 0.466 ms | 348.38x speedup | abs_err=6.10e-05, rel_err=4.84e-02
  Workload fdf5f1f4...: PASSED | 0.039 ms | 198.96x speedup | abs_err=1.19e-07, rel_err=7.16e-03
  Workload 87bff084...: PASSED | 0.074 ms | 465.89x speedup | abs_err=1.91e-06, rel_err=6.62e-03
  Workload e92dafeb...: PASSED | 0.079 ms | 517.00x speedup | abs_err=1.91e-06, rel_err=2.63e-02
  Workload 1d0cc342...: PASSED | 0.133 ms | 362.58x speedup | abs_err=3.81e-06, rel_err=5.64e-02
  Workload 1b441950...: PASSED | 0.060 ms | 260.15x speedup | abs_err=1.19e-07, rel_err=1.07e-02
  Workload 19c6ab20...: PASSED | 0.067 ms | 226.16x speedup | abs_err=1.07e-06, rel_err=1.32e-02
  Workload 25d9c14d...: PASSED | 0.059 ms | 377.95x speedup | abs_err=4.77e-07, rel_err=1.41e-02
  Workload 3215fe5f...: PASSED | 0.172 ms | 330.31x speedup | abs_err=1.91e-06, rel_err=7.40e-03
  Workload 6f1ad833...: PASSED | 0.995 ms | 335.17x speedup | abs_err=6.10e-05, rel_err=4.11e-02
  Workload e44ba4d3...: PASSED | 0.553 ms | 470.65x speedup | abs_err=7.63e-06, rel_err=2.16e-02
  Workload fc7a2bcb...: PASSED | 0.071 ms | 334.83x speedup | abs_err=9.54e-07, rel_err=3.57e-02
  Workload 5d26ac5b...: PASSED | 0.073 ms | 329.18x speedup | abs_err=3.81e-06, rel_err=2.44e-02
  Workload ed66c791...: PASSED | 0.083 ms | 225.99x speedup | abs_err=1.91e-06, rel_err=3.70e-02
  Workload ba95d412...: PASSED | 0.101 ms | 304.04x speedup | abs_err=4.77e-07, rel_err=1.36e-02
  Workload 078a41ea...: PASSED | 0.053 ms | 263.46x speedup | abs_err=9.54e-07, rel_err=1.04e-02
  Workload d2b5a221...: PASSED | 0.042 ms | 336.29x speedup | abs_err=9.54e-07, rel_err=1.25e-02
  Workload aaa378be...: PASSED | 0.738 ms | 411.72x speedup | abs_err=3.81e-06, rel_err=1.22e-02
  Workload c2bb4f66...: PASSED | 0.732 ms | 403.53x speedup | abs_err=3.05e-05, rel_err=8.00e-02
  Workload f2f01c2c...: PASSED | 0.037 ms | 169.05x speedup | abs_err=5.96e-08, rel_err=4.96e-03
  Workload 15856e8c...: PASSED | 1.288 ms | 384.11x speedup | abs_err=7.63e-06, rel_err=1.94e-02
  Workload a39aa135...: PASSED | 0.054 ms | 187.87x speedup | abs_err=2.98e-07, rel_err=1.93e-02
  Workload 339a7ff4...: PASSED | 0.034 ms | 84.05x speedup | abs_err=1.19e-07, rel_err=4.82e-03
  Workload d8f4a9ae...: PASSED | 0.044 ms | 141.33x speedup | abs_err=2.24e-08, rel_err=6.33e-03
  Workload d3dc3577...: PASSED | 0.043 ms | 142.12x speedup | abs_err=3.58e-07, rel_err=1.52e-02
  Workload ce832e76...: PASSED | 0.234 ms | 327.11x speedup | abs_err=3.81e-06, rel_err=9.66e-03
  Workload 6fbc155c...: PASSED | 0.345 ms | 394.96x speedup | abs_err=3.81e-06, rel_err=7.63e-03
  Workload a87ded8a...: PASSED | 0.755 ms | 631.16x speedup | abs_err=1.22e-04, rel_err=4.99e-02
  Workload 62447caf...: PASSED | 0.041 ms | 126.78x speedup | abs_err=1.19e-07, rel_err=2.48e-02
  Workload fd072ba6...: PASSED | 0.061 ms | 315.94x speedup | abs_err=1.91e-06, rel_err=9.11e-03
  Workload 35ea9bbe...: PASSED | 0.059 ms | 323.92x speedup | abs_err=1.91e-06, rel_err=2.41e-02
  Workload 1aa8cf18...: PASSED | 0.035 ms | 173.54x speedup | abs_err=5.96e-08, rel_err=7.61e-03
  Workload d5f5c00c...: PASSED | 0.043 ms | 128.56x speedup | abs_err=1.49e-08, rel_err=1.71e-03
  Workload d5aa60dc...: PASSED | 0.127 ms | 341.30x speedup | abs_err=1.91e-06, rel_err=3.37e-02
  Workload 28b70283...: PASSED | 0.037 ms | 115.92x speedup | abs_err=9.54e-07, rel_err=7.52e-03
  Workload 73b8cc85...: PASSED | 0.071 ms | 226.93x speedup | abs_err=1.91e-06, rel_err=6.50e-03
  Workload 2683c087...: PASSED | 0.072 ms | 227.39x speedup | abs_err=7.63e-06, rel_err=1.17e-02
  Workload 4b94d568...: PASSED | 0.240 ms | 383.36x speedup | abs_err=7.63e-06, rel_err=1.64e-02
  Workload a0eb2dc2...: PASSED | 0.068 ms | 229.35x speedup | abs_err=3.81e-06, rel_err=6.16e-03
  Workload f3d30cb9...: PASSED | 0.061 ms | 231.73x speedup | abs_err=2.98e-07, rel_err=6.35e-03
  Workload 7a7deca8...: PASSED | 0.065 ms | 200.75x speedup | abs_err=3.81e-06, rel_err=7.69e-03
  Workload 977d19f8...: PASSED | 0.042 ms | 120.42x speedup | abs_err=1.19e-07, rel_err=5.71e-03
  Workload 02c1e5f0...: PASSED | 0.041 ms | 128.77x speedup | abs_err=4.77e-07, rel_err=5.65e-03
  Workload 5a91aa02...: PASSED | 0.159 ms | 419.24x speedup | abs_err=1.91e-06, rel_err=1.13e-02
  Workload 8e7ef744...: PASSED | 0.031 ms | 73.45x speedup | abs_err=2.98e-08, rel_err=3.73e-03
  Workload 85d7becb...: PASSED | 0.036 ms | 94.10x speedup | abs_err=3.58e-07, rel_err=5.67e-03
  Workload e286a4f4...: PASSED | 0.572 ms | 519.37x speedup | abs_err=7.63e-06, rel_err=2.26e-02
  Workload 08d4f2c4...: PASSED | 0.442 ms | 361.21x speedup | abs_err=3.81e-06, rel_err=1.11e-02
  Workload 9c1ef562...: PASSED | 0.449 ms | 362.85x speedup | abs_err=3.05e-05, rel_err=4.97e-02
  Workload bfd8f7b6...: PASSED | 0.139 ms | 303.15x speedup | abs_err=3.81e-06, rel_err=7.58e-03
  Workload c358edcd...: PASSED | 0.953 ms | 360.64x speedup | abs_err=1.53e-05, rel_err=7.67e-03
  Workload f203fdcd...: PASSED | 0.049 ms | 165.98x speedup | abs_err=2.38e-07, rel_err=9.32e-03
  Workload 33a38713...: PASSED | 0.059 ms | 329.58x speedup | abs_err=1.91e-06, rel_err=3.08e-02
  Workload 3a77dfec...: PASSED | 0.056 ms | 235.17x speedup | abs_err=5.96e-08, rel_err=7.50e-03
  Workload ea27be17...: PASSED | 0.055 ms | 235.99x speedup | abs_err=7.15e-07, rel_err=8.32e-03
  Workload 49ef89d2...: PASSED | 0.057 ms | 200.03x speedup | abs_err=1.91e-06, rel_err=3.87e-02
  Workload 056224b8...: PASSED | 0.059 ms | 199.30x speedup | abs_err=4.77e-07, rel_err=6.69e-03
  Workload 685d26ff...: PASSED | 0.033 ms | 79.37x speedup | abs_err=1.19e-07, rel_err=5.69e-03
  Workload 352c9ace...: PASSED | 0.268 ms | 323.62x speedup | abs_err=3.05e-05, rel_err=2.33e-02
  Workload 2c9693b4...: PASSED | 0.051 ms | 161.60x speedup | abs_err=2.98e-08, rel_err=5.83e-03
  Workload 27f44fd6...: PASSED | 0.051 ms | 171.08x speedup | abs_err=1.91e-06, rel_err=1.05e-02
  Workload 07aa7922...: PASSED | 2.624 ms | 377.98x speedup | abs_err=3.05e-05, rel_err=7.11e-02
  Workload eaa0fd47...: PASSED | 0.034 ms | 80.16x speedup | abs_err=9.54e-07, rel_err=6.45e-03
  Workload f105eda8...: PASSED | 0.226 ms | 511.68x speedup | abs_err=6.10e-05, rel_err=1.49e-02
  Workload cd979341...: PASSED | 0.056 ms | 272.60x speedup | abs_err=3.81e-06, rel_err=6.88e-03
  Workload 43bf9699...: PASSED | 0.431 ms | 418.22x speedup | abs_err=1.91e-06, rel_err=7.69e-03
  Workload 54856fec...: PASSED | 0.429 ms | 401.74x speedup | abs_err=3.05e-05, rel_err=1.01e-01
  Workload 2ba465c0...: PASSED | 0.039 ms | 149.22x speedup | abs_err=2.38e-07, rel_err=7.69e-03
  Workload 1efaf2a9...: PASSED | 0.041 ms | 257.78x speedup | abs_err=2.98e-08, rel_err=8.85e-03
  Workload a01a3f93...: PASSED | 1.012 ms | 415.19x speedup | abs_err=1.22e-04, rel_err=4.51e-02
  Workload cc241d2e...: PASSED | 0.040 ms | 115.61x speedup | abs_err=1.49e-07, rel_err=6.82e-03
[2026-04-07T00:16:37] Local entrypoint finished in 488.82s
Stopping app - local entrypoint completed.
Runner terminated.
✓ App completed. View run at 
https://modal.com/apps/shjj1504/main/ap-UhazThnFhgKwINJuDGUFrI

```
