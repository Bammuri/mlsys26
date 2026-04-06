# Prefill Timeline — P5 Result — 2026-04-07

## Optimization summary

이번 P5 iteration에서는 current keep baseline(P3)에
**runtime cache hint `cudaFuncCachePreferL1`** 를 추가했다.

### Applied optimization

1. kernel math 불변
2. launch shape 불변 (P3 keep baseline 유지)
3. no-shared / no-spill SRTP dataflow 유지
4. `cudaFuncSetCacheConfig(gdn_prefill_kernel, cudaFuncCachePreferL1)` 추가

핵심 의도:

> shared memory를 거의 쓰지 않는 current SRTP kernel에서,
> L1-prefer cache policy가 q/k/state load 재사용에 미세한 이득을 줄 수 있는지 확인

---

## Expected result before testing

baseline (keep baseline, full 100, `warmup_runs=1, iterations=5, num_trials=2`):

- avg latency: `0.511 ms`

예상:

1. quick 5 PASS 유지
2. full 100 PASS 유지
3. 개선이 있다면 very small gain일 가능성 높음
4. no-op 또는 slight regression 가능

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
- avg latency: `0.049 ms`
- worst abs error: `1.91e-06`
- worst rel error: `1.93e-02`

### 3. full decision gate

```bash
modal run scripts/run_modal.py --decision-gate --max-workloads 100
```

결과:

- `PASSED=100/100`
- avg latency: `0.572 ms`
- avg speedup: `336.11x`
- worst abs error: `1.22e-04`
- worst rel error: `2.97e-01`

판단 기준은 사용자 지시대로 **full 100 arithmetic-mean latency**다.

---

## Comparison against baseline

### Baseline → P5

- avg latency: `0.511 ms` → `0.572 ms`
- 절대 악화: `+0.061 ms`
- 상대 악화: 약 **+11.9%**

즉 이번 iteration도 분명한 regression이다.

---

## Detailed analysis: why it likely got worse

### 1. The current kernel is already simple enough that forcing cache policy likely hurt more than it helped

current keep baseline(P3)은 이미:

- no shared state
- no spill
- low regs/thread
- row-per-warp
- modest block size

를 갖고 있었다.

이 상태에서는 hardware default cache policy가 이미 충분히 잘 작동할 가능성이 높다.

`cudaFuncCachePreferL1`를 강제로 주면,
workload mix 전체에 대해 균일한 정책을 강제하게 된다.

즉,

> 일부 case에선 맞을 수 있어도,
> 전체 distribution에서는 오히려 hardware의 기본 균형점을 망쳤을 가능성이 크다.

### 2. This is consistent with the broader lesson that blanket cache hints are fragile

background는 L1/L2 behavior를 보라고 했지만,
그 말이 곧 “항상 cache hint를 직접 강제하라”는 뜻은 아니다.

P5 결과는 오히려 다음을 보여준다.

> 현재 SRTP kernel에서는 cache hint보다 launch/dataflow 쪽이 더 본질적이고,
> blanket cache preference는 full workload 평균에서 불안정하다.

### 3. Why quick again was misleading

quick gate는 여전히 `0.049 ms`로 좋아 보일 수 있다.
하지만 full에서는 `0.572 ms`로 확실히 나빠졌다.

즉 P5도 다시 한번,

> quick는 keep 판단 기준이 될 수 없고,
> full arithmetic mean만이 최종 판단 기준이어야 함

을 보여줬다.

### 4. Why this matters for future loops

P4와 P5를 합치면:

- launch granularity를 지나치게 밀어붙여도 악화
- runtime cache hint를 blanket하게 줘도 악화

즉 current search space에서 더 믿을 만한 방향은:

1. P3 keep baseline 유지
2. more CTA 공급은 P3 수준까지만
3. cache hints보다는 representation/dataflow/shape 분기 쪽 탐색

이다.

---

## Judgement

이번 iteration은 **reject** 한다.

### Why this is a reject

1. `PASSED=100/100`
2. 하지만 full avg latency가 `0.511 ms` → `0.572 ms`로 분명히 악화
3. cache hint는 current SRTP kernel에선 full 기준 유효하지 않았다

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

- `timeline/prefill/perf_P5_2604070019.txt`

아래는 동일 로그 전문이다.

```text
✓ Initialized. View run at 
https://modal.com/apps/shjj1504/main/ap-1nEVUd3eo7FcQ7LKC9doof
✓ Created objects.
├── 🔨 Created mount /home/hyu/flashinfer/mlsys26/scripts/run_modal.py
└── 🔨 Created function run_benchmark.
[2026-04-07T00:19:24] Packing solution from source files...
Solution packed: /home/hyu/flashinfer/mlsys26/solution.json
  Name: my-team-solution-v1
  Definition: gdn_prefill_qk4_v8_d128_k_last
  Author: team-name
  Config language: cuda
  Runtime language: cuda
[2026-04-07T00:19:24] Validating solution JSON...
[2026-04-07T00:19:24] Loaded solution my-team-solution-v1 (gdn_prefill_qk4_v8_d128_k_last) in 0.00s
[2026-04-07T00:19:24] Decision-gate mode enabled: warmup_runs=1, iterations=5, num_trials=2, use_isolated_runner=False
[2026-04-07T00:19:24] Dispatching benchmark to Modal B200...

==========
== CUDA ==
==========

CUDA Version 13.0.2

Container image Copyright (c) 2016-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

This container image and its contents are governed by the NVIDIA Deep Learning Container License.
By pulling and using the container, you accept the terms and conditions of this license:
https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license

A copy of this license is made available in this container at /NGC-DL-CONTAINER-LICENSE for your convenience.

[2026-04-06T15:19:39] Remote benchmark start: solution=my-team-solution-v1, definition=gdn_prefill_qk4_v8_d128_k_last
[2026-04-06T15:19:39] BenchmarkConfig(warmup_runs=1, iterations=5, num_trials=2)
[2026-04-06T15:19:39] Loading trace set from /data/data/mlsys26-contest
[2026-04-06T15:19:40] Loaded trace set in 0.50s
[2026-04-06T15:19:40] Running benchmark across 100 workloads
[2026-04-06T15:27:50] Benchmark completed in 490.56s
[2026-04-07T00:27:51] Received benchmark results in 506.99s

gdn_prefill_qk4_v8_d128_k_last:
  workloads: 100
  status counts: PASSED=100
  avg latency: 0.572 ms
  avg speedup: 336.11x
  worst abs error: 1.22e-04
  worst rel error: 2.97e-01
  Workload 77daf91d...: PASSED | 0.036 ms | 45.78x speedup | abs_err=4.47e-08, rel_err=9.53e-03
  Workload ba08a83e...: PASSED | 0.091 ms | 241.52x speedup | abs_err=7.63e-06, rel_err=7.50e-03
  Workload c7846f96...: PASSED | 0.090 ms | 241.94x speedup | abs_err=3.81e-06, rel_err=6.89e-03
  Workload d0ce7b5d...: PASSED | 1.644 ms | 808.72x speedup | abs_err=6.10e-05, rel_err=1.21e-01
  Workload 5b8a0e4b...: PASSED | 2.492 ms | 563.19x speedup | abs_err=1.53e-05, rel_err=4.03e-02
  Workload 5d3fc66a...: PASSED | 2.562 ms | 581.65x speedup | abs_err=6.10e-05, rel_err=1.59e-01
  Workload 4b6143dd...: PASSED | 0.748 ms | 982.17x speedup | abs_err=7.63e-06, rel_err=2.79e-02
  Workload 5835a2bc...: PASSED | 2.396 ms | 584.73x speedup | abs_err=1.53e-05, rel_err=1.73e-02
  Workload cc310f94...: PASSED | 2.234 ms | 608.49x speedup | abs_err=1.22e-04, rel_err=1.33e-01
  Workload d49df0b2...: PASSED | 1.896 ms | 788.55x speedup | abs_err=1.53e-05, rel_err=6.77e-02
  Workload e9e1e445...: PASSED | 2.043 ms | 702.48x speedup | abs_err=1.22e-04, rel_err=3.40e-02
  Workload b8c8dc3c...: PASSED | 2.012 ms | 731.43x speedup | abs_err=6.10e-05, rel_err=1.56e-01
  Workload a9540651...: PASSED | 2.445 ms | 586.72x speedup | abs_err=1.53e-05, rel_err=4.84e-02
  Workload 06f21bb1...: PASSED | 1.507 ms | 936.58x speedup | abs_err=3.05e-05, rel_err=2.05e-02
  Workload c2931c92...: PASSED | 1.907 ms | 749.26x speedup | abs_err=1.22e-04, rel_err=1.25e-01
  Workload 618df04a...: PASSED | 1.879 ms | 748.51x speedup | abs_err=1.53e-05, rel_err=1.14e-01
  Workload 26244fb4...: PASSED | 2.240 ms | 650.24x speedup | abs_err=7.63e-06, rel_err=1.88e-01
  Workload a2629e02...: PASSED | 2.278 ms | 641.40x speedup | abs_err=6.10e-05, rel_err=2.97e-01
  Workload 9a5d694b...: PASSED | 1.506 ms | 973.03x speedup | abs_err=6.10e-05, rel_err=5.64e-02
  Workload 410794d4...: PASSED | 1.867 ms | 782.18x speedup | abs_err=1.53e-05, rel_err=1.67e-02
  Workload 7ba9d519...: PASSED | 1.446 ms | 455.44x speedup | abs_err=6.10e-05, rel_err=5.87e-02
  Workload 043e74e4...: PASSED | 0.044 ms | 133.40x speedup | abs_err=1.19e-07, rel_err=7.32e-03
  Workload ef9515b6...: PASSED | 0.037 ms | 82.90x speedup | abs_err=2.98e-08, rel_err=1.82e-03
  Workload f622a11d...: PASSED | 0.037 ms | 84.06x speedup | abs_err=5.96e-07, rel_err=1.16e-02
  Workload 1cf8e175...: PASSED | 0.045 ms | 165.48x speedup | abs_err=1.19e-07, rel_err=4.49e-03
  Workload 9343fd82...: PASSED | 0.051 ms | 175.28x speedup | abs_err=1.91e-06, rel_err=8.95e-03
  Workload f4926229...: PASSED | 0.663 ms | 371.09x speedup | abs_err=6.10e-05, rel_err=4.59e-02
  Workload 109addb1...: PASSED | 1.247 ms | 445.31x speedup | abs_err=6.10e-05, rel_err=1.80e-02
  Workload c5257f65...: PASSED | 0.505 ms | 348.35x speedup | abs_err=1.91e-06, rel_err=4.43e-02
  Workload f5619793...: PASSED | 0.500 ms | 357.75x speedup | abs_err=6.10e-05, rel_err=4.84e-02
  Workload fdf5f1f4...: PASSED | 0.048 ms | 174.41x speedup | abs_err=1.19e-07, rel_err=7.16e-03
  Workload 87bff084...: PASSED | 0.083 ms | 458.45x speedup | abs_err=1.91e-06, rel_err=6.62e-03
  Workload e92dafeb...: PASSED | 0.099 ms | 445.83x speedup | abs_err=1.91e-06, rel_err=2.63e-02
  Workload 1d0cc342...: PASSED | 0.165 ms | 332.75x speedup | abs_err=3.81e-06, rel_err=5.64e-02
  Workload 1b441950...: PASSED | 0.077 ms | 262.37x speedup | abs_err=1.19e-07, rel_err=1.07e-02
  Workload 19c6ab20...: PASSED | 0.080 ms | 198.48x speedup | abs_err=1.07e-06, rel_err=1.32e-02
  Workload 25d9c14d...: PASSED | 0.093 ms | 291.12x speedup | abs_err=4.77e-07, rel_err=1.41e-02
  Workload 3215fe5f...: PASSED | 0.176 ms | 366.16x speedup | abs_err=1.91e-06, rel_err=7.40e-03
  Workload 6f1ad833...: PASSED | 1.000 ms | 379.43x speedup | abs_err=6.10e-05, rel_err=4.11e-02
  Workload e44ba4d3...: PASSED | 0.760 ms | 338.02x speedup | abs_err=7.63e-06, rel_err=2.16e-02
  Workload fc7a2bcb...: PASSED | 0.086 ms | 291.68x speedup | abs_err=9.54e-07, rel_err=3.57e-02
  Workload 5d26ac5b...: PASSED | 0.085 ms | 287.07x speedup | abs_err=3.81e-06, rel_err=2.44e-02
  Workload ed66c791...: PASSED | 0.083 ms | 245.28x speedup | abs_err=1.91e-06, rel_err=3.70e-02
  Workload ba95d412...: PASSED | 0.100 ms | 342.33x speedup | abs_err=4.77e-07, rel_err=1.36e-02
  Workload 078a41ea...: PASSED | 0.058 ms | 253.14x speedup | abs_err=9.54e-07, rel_err=1.04e-02
  Workload d2b5a221...: PASSED | 0.057 ms | 297.33x speedup | abs_err=9.54e-07, rel_err=1.25e-02
  Workload aaa378be...: PASSED | 0.859 ms | 375.04x speedup | abs_err=3.81e-06, rel_err=1.22e-02
  Workload c2bb4f66...: PASSED | 0.802 ms | 398.47x speedup | abs_err=3.05e-05, rel_err=8.00e-02
  Workload f2f01c2c...: PASSED | 0.045 ms | 146.45x speedup | abs_err=5.96e-08, rel_err=4.96e-03
  Workload 15856e8c...: PASSED | 1.528 ms | 351.68x speedup | abs_err=7.63e-06, rel_err=1.94e-02
  Workload a39aa135...: PASSED | 0.057 ms | 196.58x speedup | abs_err=2.98e-07, rel_err=1.93e-02
  Workload 339a7ff4...: PASSED | 0.036 ms | 89.31x speedup | abs_err=1.19e-07, rel_err=4.82e-03
  Workload d8f4a9ae...: PASSED | 0.053 ms | 132.08x speedup | abs_err=2.24e-08, rel_err=6.33e-03
  Workload d3dc3577...: PASSED | 0.046 ms | 149.54x speedup | abs_err=3.58e-07, rel_err=1.52e-02
  Workload ce832e76...: PASSED | 0.271 ms | 309.42x speedup | abs_err=3.81e-06, rel_err=9.66e-03
  Workload 6fbc155c...: PASSED | 0.397 ms | 374.98x speedup | abs_err=3.81e-06, rel_err=7.63e-03
  Workload a87ded8a...: PASSED | 1.287 ms | 398.33x speedup | abs_err=1.22e-04, rel_err=4.99e-02
  Workload 62447caf...: PASSED | 0.043 ms | 127.05x speedup | abs_err=1.19e-07, rel_err=2.48e-02
  Workload fd072ba6...: PASSED | 0.076 ms | 267.87x speedup | abs_err=1.91e-06, rel_err=9.11e-03
  Workload 35ea9bbe...: PASSED | 0.076 ms | 283.75x speedup | abs_err=1.91e-06, rel_err=2.41e-02
  Workload 1aa8cf18...: PASSED | 0.046 ms | 141.40x speedup | abs_err=5.96e-08, rel_err=7.61e-03
  Workload d5f5c00c...: PASSED | 0.045 ms | 128.19x speedup | abs_err=1.49e-08, rel_err=1.71e-03
  Workload d5aa60dc...: PASSED | 0.151 ms | 296.90x speedup | abs_err=1.91e-06, rel_err=3.37e-02
  Workload 28b70283...: PASSED | 0.042 ms | 108.93x speedup | abs_err=9.54e-07, rel_err=7.52e-03
  Workload 73b8cc85...: PASSED | 0.073 ms | 239.55x speedup | abs_err=1.91e-06, rel_err=6.50e-03
  Workload 2683c087...: PASSED | 0.075 ms | 245.01x speedup | abs_err=7.63e-06, rel_err=1.17e-02
  Workload 4b94d568...: PASSED | 0.244 ms | 423.76x speedup | abs_err=7.63e-06, rel_err=1.64e-02
  Workload a0eb2dc2...: PASSED | 0.074 ms | 239.85x speedup | abs_err=3.81e-06, rel_err=6.16e-03
  Workload f3d30cb9...: PASSED | 0.071 ms | 213.88x speedup | abs_err=2.98e-07, rel_err=6.35e-03
  Workload 7a7deca8...: PASSED | 0.074 ms | 209.67x speedup | abs_err=3.81e-06, rel_err=7.69e-03
  Workload 977d19f8...: PASSED | 0.047 ms | 118.31x speedup | abs_err=1.19e-07, rel_err=5.71e-03
  Workload 02c1e5f0...: PASSED | 0.044 ms | 126.68x speedup | abs_err=4.77e-07, rel_err=5.65e-03
  Workload 5a91aa02...: PASSED | 0.173 ms | 413.47x speedup | abs_err=1.91e-06, rel_err=1.13e-02
  Workload 8e7ef744...: PASSED | 0.038 ms | 63.10x speedup | abs_err=2.98e-08, rel_err=3.73e-03
  Workload 85d7becb...: PASSED | 0.039 ms | 87.46x speedup | abs_err=3.58e-07, rel_err=5.67e-03
  Workload e286a4f4...: PASSED | 0.579 ms | 538.92x speedup | abs_err=7.63e-06, rel_err=2.26e-02
  Workload 08d4f2c4...: PASSED | 0.505 ms | 334.52x speedup | abs_err=3.81e-06, rel_err=1.11e-02
  Workload 9c1ef562...: PASSED | 0.504 ms | 324.32x speedup | abs_err=3.05e-05, rel_err=4.97e-02
  Workload bfd8f7b6...: PASSED | 0.140 ms | 307.36x speedup | abs_err=3.81e-06, rel_err=7.58e-03
  Workload c358edcd...: PASSED | 1.140 ms | 309.45x speedup | abs_err=1.53e-05, rel_err=7.67e-03
  Workload f203fdcd...: PASSED | 0.050 ms | 177.17x speedup | abs_err=2.38e-07, rel_err=9.32e-03
  Workload 33a38713...: PASSED | 0.083 ms | 250.95x speedup | abs_err=1.91e-06, rel_err=3.08e-02
  Workload 3a77dfec...: PASSED | 0.059 ms | 231.74x speedup | abs_err=5.96e-08, rel_err=7.50e-03
  Workload ea27be17...: PASSED | 0.059 ms | 234.00x speedup | abs_err=7.15e-07, rel_err=8.32e-03
  Workload 49ef89d2...: PASSED | 0.061 ms | 196.32x speedup | abs_err=1.91e-06, rel_err=3.87e-02
  Workload 056224b8...: PASSED | 0.063 ms | 195.46x speedup | abs_err=4.77e-07, rel_err=6.69e-03
  Workload 685d26ff...: PASSED | 0.037 ms | 70.07x speedup | abs_err=1.19e-07, rel_err=5.69e-03
  Workload 352c9ace...: PASSED | 0.275 ms | 331.24x speedup | abs_err=3.05e-05, rel_err=2.33e-02
  Workload 2c9693b4...: PASSED | 0.052 ms | 172.59x speedup | abs_err=2.98e-08, rel_err=5.83e-03
  Workload 27f44fd6...: PASSED | 0.051 ms | 171.21x speedup | abs_err=1.91e-06, rel_err=1.05e-02
  Workload 07aa7922...: PASSED | 2.636 ms | 378.28x speedup | abs_err=3.05e-05, rel_err=7.11e-02
  Workload eaa0fd47...: PASSED | 0.036 ms | 73.85x speedup | abs_err=9.54e-07, rel_err=6.45e-03
  Workload f105eda8...: PASSED | 0.224 ms | 505.55x speedup | abs_err=6.10e-05, rel_err=1.49e-02
  Workload cd979341...: PASSED | 0.067 ms | 207.03x speedup | abs_err=3.81e-06, rel_err=6.88e-03
  Workload 43bf9699...: PASSED | 0.482 ms | 341.83x speedup | abs_err=1.91e-06, rel_err=7.69e-03
  Workload 54856fec...: PASSED | 0.472 ms | 346.89x speedup | abs_err=3.05e-05, rel_err=1.01e-01
  Workload 2ba465c0...: PASSED | 0.043 ms | 123.59x speedup | abs_err=2.38e-07, rel_err=7.69e-03
  Workload 1efaf2a9...: PASSED | 0.046 ms | 168.28x speedup | abs_err=2.98e-08, rel_err=8.85e-03
  Workload a01a3f93...: PASSED | 1.193 ms | 328.23x speedup | abs_err=1.22e-04, rel_err=4.51e-02
  Workload cc241d2e...: PASSED | 0.042 ms | 103.54x speedup | abs_err=1.49e-07, rel_err=6.82e-03
[2026-04-07T00:27:51] Local entrypoint finished in 506.99s
Stopping app - local entrypoint completed.
[2026-04-06T15:27:50] Remote benchmark finished in 491.06s
✓ App completed. View run at 
https://modal.com/apps/shjj1504/main/ap-1nEVUd3eo7FcQ7LKC9doof

```
