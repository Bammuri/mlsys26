# Prefill Timeline — P2 Result — 2026-04-06

## Optimization summary

이번 P2 iteration에서는 P1의 SRTP row-per-warp 구조 위에서,
**warp 1개가 row 2개를 연속 처리하는 row-pair-per-warp variant**를 시험했다.

### Applied optimization

1. warp 1개가 row 1개 대신 row 2개를 담당
2. `kRowsPerBlock`을 4 rows/block → 8 rows/block으로 변경
3. q/k fragment load를 한 번 한 뒤 row 2개 update/output에 재사용
4. shared-state 없음, barrier 없음, gate/beta precompute 유지

핵심 의도:

> q/k load와 gate/beta broadcast를 더 많이 재사용해,
> row-per-warp보다 token당 유효 work를 늘리고 full 평균 latency를 더 낮추기

---

## Expected result before testing

baseline (current keep baseline, full 100, `warmup_runs=1, iterations=5, num_trials=2`):

- avg latency: `0.517 ms`

예상:

1. quick 5 PASS 유지
2. full 100 PASS 유지
3. throughput-heavy / long-tail에서 small positive gain 가능
4. tiny에서는 CTA 수가 줄어 약간 손해일 수 있음

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
- avg latency: `0.055 ms`
- worst abs error: `1.91e-06`
- worst rel error: `1.93e-02`

### 3. full decision gate

```bash
modal run scripts/run_modal.py --decision-gate --max-workloads 100
```

결과:

- `PASSED=100/100`
- avg latency: `0.782 ms`
- avg speedup: `277.70x`
- worst abs error: `1.22e-04`
- worst rel error: `2.97e-01`

판단 기준은 사용자 지시대로 **quick이 아니라 full 100 arithmetic-mean latency**를 사용한다.

---

## Comparison against baseline

### Baseline → P2

- avg latency: `0.517 ms` → `0.782 ms`
- 절대 악화: `+0.265 ms`
- 상대 악화: 약 **+51.3%**

즉 이번 iteration은 **명확한 regression**이다.

---

## Representative workload comparison

몇 개 대표 workload만 봐도 regression이 일관된다.

- `77daf91d`: `0.039 ms` → `0.043 ms`
- `ba08a83e`: `0.093 ms` → `0.128 ms`
- `5835a2bc`: `2.209 ms` → `3.829 ms`
- `07aa7922`: `2.674 ms` → `3.958 ms`

특히:

1. **throughput-heavy (`5835a2bc`)**: 약 `+73%` 악화
2. **long-tail (`07aa7922`)**: 약 `+48%` 악화

즉 이 변화는 tiny noise가 아니라,
**실제 full-distribution에서 구조적으로 나빠진 케이스**다.

---

## Detailed analysis: why it got worse

### 1. P2 traded away the most important thing P1 had just fixed: CTA supply

P1의 핵심 강점은 row-per-warp로 인해
`grid.x = 256`까지 늘어나며 CTA 공급량이 크게 좋아진 점이었다.

P2는 row-pair-per-warp로 바꾸면서:

- same block size (`128` threads)
- same warps/block (`4` warps)
- but **rows/block doubled**
- therefore **grid.x halved**

이 변화는 tiny case NCU에서 바로 보인다.

#### Tiny NCU: P1 vs P2 (`77daf91d`)

| Metric | P1 | P2 |
|---|---:|---:|
| `gdn_prefill_kernel` duration | `11.81 us` | `16.48 us` |
| grid size | `256` | `128` |
| achieved occupancy | `10.48%` | `6.30%` |
| waves/SM | `0.11` | `0.05` |
| memory throughput | `46.98 GB/s` | `33.69 GB/s` |
| regs/thread | `32` | `32` |
| spills | `0` | `0` |

핵심은 register pressure가 늘어서가 아니다.

> **CTA/warp 공급량이 줄어들어 latency hiding이 악화된 것**이 first-order effect다.

### 2. Reusing q/k twice per warp did not pay back the lost parallelism

P2의 가설은:

- q/k frag load 1회
- row 2개 update/output 2회
- therefore per-load work 증가

였지만, 실제로는 이득보다 손해가 컸다.

이유는 현재 kernel이 memory-bandwidth-bound라기보다,
**latency/parallelism-sensitive**하기 때문이다.

즉,

> q/k를 두 row에 재사용해서 얻는 이득보다,
> warp 하나가 두 row를 순차 처리하면서 잃는 scheduler-level concurrency가 더 컸다.

### 3. The precompute kernel stayed cheap, so the regression is squarely in the main kernel

tiny precompute NCU:

- P1: `5.89 us`
- P2: `5.50 us`

거의 비슷하거나 오히려 약간 좋아졌다.

즉 regression은 precompute가 아니라,
**main kernel work partitioning**에서 생겼다.

### 4. Long-tail workloads got worse because serial work per warp increased

`07aa7922`는 long-tail workload다.

- baseline: `2.674 ms`
- P2: `3.958 ms`

row-pair-per-warp는 warp 하나가 토큰마다 처리해야 할 row update/output를 2배로 만든다.
긴 sequence에서는 이 serial work 증가가 누적되어,
full average에 크게 불리하게 작용한다.

즉,

> long-tail에서는 q/k reuse보다,
> **token loop 안에서 warp가 얼마나 짧고 가볍게 끝나는가**가 더 중요했다.

### 5. Throughput-heavy workloads also regressed, which means the idea is not just “tiny-hostile”

만약 P2가 tiny만 손해였다면 full average 영향은 제한적일 수 있었다.
하지만 `5835a2bc`에서도:

- baseline: `2.209 ms`
- P2: `3.829 ms`

로 크게 악화됐다.

이 말은,

> 이 아이디어는 tiny launch underfill뿐 아니라,
> larger steady-state region에서도 kernel work granularity를 잘못 맞춘 것

이라는 뜻이다.

### 6. Why this is different from “just a little noise”

quick gate는 오히려 `0.055 ms`로 약간 좋아 보였다.
하지만 full에서는 `0.782 ms`로 크게 악화됐다.

이건 사용자가 강조한 판단 규칙을 정확히 보여준다.

> quick 결과는 working check일 뿐,
> keep/revert 판단 기준이 되어서는 안 된다.

이번 P2는 quick만 보면 착시가 있었지만,
full 100 arithmetic mean이 regression을 명확히 드러냈다.

---

## Judgement

이번 iteration은 **reject** 한다.

### Why this is a reject

1. `PASSED=100/100`이긴 하지만
2. full avg latency가 `0.517 ms` → `0.782 ms`로 크게 악화
3. regression이 tiny에만 국한되지 않고 medium / throughput / long-tail 전반에서 확인됨
4. tiny NCU도 main kernel underfill 악화를 명확히 보여줌

따라서 사용자 규칙에 따라:

- **commit하지 않는다**
- **kernel code만 삭제(revert)한다**
- 다음 loop로 넘어간다

---

## Decision under workflow rule

- keep/revert 기준: **full workload avg latency arithmetic mean**
- 결과: **REVERT**

다음 단계:

1. `solution/cuda/kernel.cu`를 P1 keep baseline으로 되돌린다
2. 다음 plan은 P1 결과 + background 기준으로 다시 수립한다

---

## Full pref log

full pref log is saved in:

- `timeline/prefill/perf_P2_2604062336.txt`

아래는 동일 로그 전문이다.

```text
✓ Initialized. View run at 
https://modal.com/apps/shjj1504/main/ap-9TXphEPxramHGWjc22KXRK
✓ Created objects.
├── 🔨 Created mount /home/hyu/flashinfer/mlsys26/scripts/run_modal.py
└── 🔨 Created function run_benchmark.
[2026-04-06T23:36:22] Packing solution from source files...
Solution packed: /home/hyu/flashinfer/mlsys26/solution.json
  Name: my-team-solution-v1
  Definition: gdn_prefill_qk4_v8_d128_k_last
  Author: team-name
  Config language: cuda
  Runtime language: cuda
[2026-04-06T23:36:22] Validating solution JSON...
[2026-04-06T23:36:22] Loaded solution my-team-solution-v1 (gdn_prefill_qk4_v8_d128_k_last) in 0.01s
[2026-04-06T23:36:22] Decision-gate mode enabled: warmup_runs=1, iterations=5, num_trials=2, use_isolated_runner=False
[2026-04-06T23:36:22] Dispatching benchmark to Modal B200...

==========
== CUDA ==
==========

CUDA Version 13.0.2

Container image Copyright (c) 2016-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

This container image and its contents are governed by the NVIDIA Deep Learning Container License.
By pulling and using the container, you accept the terms and conditions of this license:
https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license

A copy of this license is made available in this container at /NGC-DL-CONTAINER-LICENSE for your convenience.

[2026-04-06T14:36:28] Remote benchmark start: solution=my-team-solution-v1, definition=gdn_prefill_qk4_v8_d128_k_last
[2026-04-06T14:36:28] BenchmarkConfig(warmup_runs=1, iterations=5, num_trials=2)
[2026-04-06T14:36:28] Loading trace set from /data/data/mlsys26-contest
[2026-04-06T14:36:28] Loaded trace set in 0.47s
[2026-04-06T14:36:28] Running benchmark across 100 workloads
[2026-04-06T14:45:13] Benchmark completed in 524.71s
[2026-04-06T23:45:14] Received benchmark results in 532.36s

gdn_prefill_qk4_v8_d128_k_last:
  workloads: 100
  status counts: PASSED=100
  avg latency: 0.782 ms
  avg speedup: 277.70x
  worst abs error: 1.22e-04
  worst rel error: 2.97e-01
  Workload 77daf91d...: PASSED | 0.043 ms | 32.43x speedup | abs_err=4.47e-08, rel_err=9.53e-03
  Workload ba08a83e...: PASSED | 0.128 ms | 198.83x speedup | abs_err=7.63e-06, rel_err=7.50e-03
  Workload c7846f96...: PASSED | 0.128 ms | 198.83x speedup | abs_err=3.81e-06, rel_err=6.89e-03
  Workload d0ce7b5d...: PASSED | 2.462 ms | 599.27x speedup | abs_err=6.10e-05, rel_err=1.21e-01
  Workload 5b8a0e4b...: PASSED | 3.679 ms | 408.19x speedup | abs_err=1.53e-05, rel_err=4.03e-02
  Workload 5d3fc66a...: PASSED | 3.528 ms | 431.44x speedup | abs_err=6.10e-05, rel_err=1.59e-01
  Workload 4b6143dd...: PASSED | 0.718 ms | 1071.93x speedup | abs_err=7.63e-06, rel_err=2.79e-02
  Workload 5835a2bc...: PASSED | 3.829 ms | 395.43x speedup | abs_err=1.53e-05, rel_err=1.73e-02
  Workload cc310f94...: PASSED | 3.126 ms | 481.57x speedup | abs_err=1.22e-04, rel_err=1.33e-01
  Workload d49df0b2...: PASSED | 2.635 ms | 557.67x speedup | abs_err=1.53e-05, rel_err=6.77e-02
  Workload e9e1e445...: PASSED | 2.388 ms | 626.17x speedup | abs_err=1.22e-04, rel_err=3.40e-02
  Workload b8c8dc3c...: PASSED | 2.333 ms | 632.88x speedup | abs_err=6.10e-05, rel_err=1.56e-01
  Workload a9540651...: PASSED | 3.453 ms | 434.62x speedup | abs_err=1.53e-05, rel_err=4.84e-02
  Workload 06f21bb1...: PASSED | 1.545 ms | 963.94x speedup | abs_err=3.05e-05, rel_err=2.05e-02
  Workload c2931c92...: PASSED | 3.342 ms | 452.50x speedup | abs_err=1.22e-04, rel_err=1.25e-01
  Workload 618df04a...: PASSED | 2.668 ms | 561.28x speedup | abs_err=1.53e-05, rel_err=1.14e-01
  Workload 26244fb4...: PASSED | 3.232 ms | 456.67x speedup | abs_err=7.63e-06, rel_err=1.88e-01
  Workload a2629e02...: PASSED | 3.277 ms | 455.63x speedup | abs_err=6.10e-05, rel_err=2.97e-01
  Workload 9a5d694b...: PASSED | 2.245 ms | 658.22x speedup | abs_err=6.10e-05, rel_err=5.64e-02
  Workload 410794d4...: PASSED | 2.682 ms | 566.71x speedup | abs_err=1.53e-05, rel_err=1.67e-02
  Workload 7ba9d519...: PASSED | 2.076 ms | 344.98x speedup | abs_err=6.10e-05, rel_err=5.87e-02
  Workload 043e74e4...: PASSED | 0.044 ms | 138.81x speedup | abs_err=1.19e-07, rel_err=7.32e-03
  Workload ef9515b6...: PASSED | 0.043 ms | 76.38x speedup | abs_err=2.98e-08, rel_err=1.82e-03
  Workload f622a11d...: PASSED | 0.044 ms | 71.83x speedup | abs_err=5.96e-07, rel_err=1.16e-02
  Workload 1cf8e175...: PASSED | 0.047 ms | 161.08x speedup | abs_err=1.19e-07, rel_err=4.49e-03
  Workload 9343fd82...: PASSED | 0.065 ms | 138.93x speedup | abs_err=1.91e-06, rel_err=8.95e-03
  Workload f4926229...: PASSED | 0.987 ms | 249.94x speedup | abs_err=6.10e-05, rel_err=4.59e-02
  Workload 109addb1...: PASSED | 1.858 ms | 326.47x speedup | abs_err=6.10e-05, rel_err=1.80e-02
  Workload c5257f65...: PASSED | 0.689 ms | 268.34x speedup | abs_err=1.91e-06, rel_err=4.43e-02
  Workload f5619793...: PASSED | 0.690 ms | 271.08x speedup | abs_err=6.10e-05, rel_err=4.84e-02
  Workload fdf5f1f4...: PASSED | 0.051 ms | 175.14x speedup | abs_err=1.19e-07, rel_err=7.16e-03
  Workload 87bff084...: PASSED | 0.101 ms | 363.68x speedup | abs_err=1.91e-06, rel_err=6.62e-03
  Workload e92dafeb...: PASSED | 0.108 ms | 458.33x speedup | abs_err=1.91e-06, rel_err=2.63e-02
  Workload 1d0cc342...: PASSED | 0.192 ms | 315.88x speedup | abs_err=3.81e-06, rel_err=5.64e-02
  Workload 1b441950...: PASSED | 0.083 ms | 208.78x speedup | abs_err=1.19e-07, rel_err=1.07e-02
  Workload 19c6ab20...: PASSED | 0.083 ms | 235.77x speedup | abs_err=1.07e-06, rel_err=1.32e-02
  Workload 25d9c14d...: PASSED | 0.083 ms | 307.17x speedup | abs_err=4.77e-07, rel_err=1.41e-02
  Workload 3215fe5f...: PASSED | 0.249 ms | 251.40x speedup | abs_err=1.91e-06, rel_err=7.40e-03
  Workload 6f1ad833...: PASSED | 1.490 ms | 265.58x speedup | abs_err=6.10e-05, rel_err=4.11e-02
  Workload e44ba4d3...: PASSED | 0.830 ms | 359.11x speedup | abs_err=7.63e-06, rel_err=2.16e-02
  Workload fc7a2bcb...: PASSED | 0.099 ms | 257.52x speedup | abs_err=9.54e-07, rel_err=3.57e-02
  Workload 5d26ac5b...: PASSED | 0.130 ms | 203.46x speedup | abs_err=3.81e-06, rel_err=2.44e-02
  Workload ed66c791...: PASSED | 0.114 ms | 192.33x speedup | abs_err=1.91e-06, rel_err=3.70e-02
  Workload ba95d412...: PASSED | 0.141 ms | 236.87x speedup | abs_err=4.77e-07, rel_err=1.36e-02
  Workload 078a41ea...: PASSED | 0.072 ms | 214.22x speedup | abs_err=9.54e-07, rel_err=1.04e-02
  Workload d2b5a221...: PASSED | 0.057 ms | 279.83x speedup | abs_err=9.54e-07, rel_err=1.25e-02
  Workload aaa378be...: PASSED | 1.102 ms | 306.25x speedup | abs_err=3.81e-06, rel_err=1.22e-02
  Workload c2bb4f66...: PASSED | 1.100 ms | 310.36x speedup | abs_err=3.05e-05, rel_err=8.00e-02
  Workload f2f01c2c...: PASSED | 0.048 ms | 145.41x speedup | abs_err=5.96e-08, rel_err=4.96e-03
  Workload 15856e8c...: PASSED | 1.924 ms | 294.67x speedup | abs_err=7.63e-06, rel_err=1.94e-02
  Workload a39aa135...: PASSED | 0.074 ms | 156.78x speedup | abs_err=2.98e-07, rel_err=1.93e-02
  Workload 339a7ff4...: PASSED | 0.045 ms | 73.47x speedup | abs_err=1.19e-07, rel_err=4.82e-03
  Workload d8f4a9ae...: PASSED | 0.056 ms | 120.22x speedup | abs_err=2.24e-08, rel_err=6.33e-03
  Workload d3dc3577...: PASSED | 0.056 ms | 122.37x speedup | abs_err=3.58e-07, rel_err=1.52e-02
  Workload ce832e76...: PASSED | 0.339 ms | 243.12x speedup | abs_err=3.81e-06, rel_err=9.66e-03
  Workload 6fbc155c...: PASSED | 0.512 ms | 336.97x speedup | abs_err=3.81e-06, rel_err=7.63e-03
  Workload a87ded8a...: PASSED | 1.129 ms | 510.43x speedup | abs_err=1.22e-04, rel_err=4.99e-02
  Workload 62447caf...: PASSED | 0.054 ms | 107.51x speedup | abs_err=1.19e-07, rel_err=2.48e-02
  Workload fd072ba6...: PASSED | 0.082 ms | 252.28x speedup | abs_err=1.91e-06, rel_err=9.11e-03
  Workload 35ea9bbe...: PASSED | 0.082 ms | 263.84x speedup | abs_err=1.91e-06, rel_err=2.41e-02
  Workload 1aa8cf18...: PASSED | 0.044 ms | 152.81x speedup | abs_err=5.96e-08, rel_err=7.61e-03
  Workload d5f5c00c...: PASSED | 0.055 ms | 109.33x speedup | abs_err=1.49e-08, rel_err=1.71e-03
  Workload d5aa60dc...: PASSED | 0.176 ms | 261.07x speedup | abs_err=1.91e-06, rel_err=3.37e-02
  Workload 28b70283...: PASSED | 0.051 ms | 92.75x speedup | abs_err=9.54e-07, rel_err=7.52e-03
  Workload 73b8cc85...: PASSED | 0.102 ms | 173.12x speedup | abs_err=1.91e-06, rel_err=6.50e-03
  Workload 2683c087...: PASSED | 0.100 ms | 177.74x speedup | abs_err=7.63e-06, rel_err=1.17e-02
  Workload 4b94d568...: PASSED | 0.350 ms | 315.33x speedup | abs_err=7.63e-06, rel_err=1.64e-02
  Workload a0eb2dc2...: PASSED | 0.096 ms | 180.47x speedup | abs_err=3.81e-06, rel_err=6.16e-03
  Workload f3d30cb9...: PASSED | 0.082 ms | 198.78x speedup | abs_err=2.98e-07, rel_err=6.35e-03
  Workload 7a7deca8...: PASSED | 0.092 ms | 157.70x speedup | abs_err=3.81e-06, rel_err=7.69e-03
  Workload 977d19f8...: PASSED | 0.053 ms | 111.69x speedup | abs_err=1.19e-07, rel_err=5.71e-03
  Workload 02c1e5f0...: PASSED | 0.052 ms | 113.46x speedup | abs_err=4.77e-07, rel_err=5.65e-03
  Workload 5a91aa02...: PASSED | 0.225 ms | 332.51x speedup | abs_err=1.91e-06, rel_err=1.13e-02
  Workload 8e7ef744...: PASSED | 0.045 ms | 54.11x speedup | abs_err=2.98e-08, rel_err=3.73e-03
  Workload 85d7becb...: PASSED | 0.042 ms | 83.23x speedup | abs_err=3.58e-07, rel_err=5.67e-03
  Workload e286a4f4...: PASSED | 0.852 ms | 431.06x speedup | abs_err=7.63e-06, rel_err=2.26e-02
  Workload 08d4f2c4...: PASSED | 0.666 ms | 300.50x speedup | abs_err=3.81e-06, rel_err=1.11e-02
  Workload 9c1ef562...: PASSED | 0.667 ms | 273.04x speedup | abs_err=3.05e-05, rel_err=4.97e-02
  Workload bfd8f7b6...: PASSED | 0.196 ms | 228.60x speedup | abs_err=3.81e-06, rel_err=7.58e-03
  Workload c358edcd...: PASSED | 1.429 ms | 266.08x speedup | abs_err=1.53e-05, rel_err=7.67e-03
  Workload f203fdcd...: PASSED | 0.066 ms | 137.34x speedup | abs_err=2.38e-07, rel_err=9.32e-03
  Workload 33a38713...: PASSED | 0.085 ms | 310.51x speedup | abs_err=1.91e-06, rel_err=3.08e-02
  Workload 3a77dfec...: PASSED | 0.075 ms | 194.86x speedup | abs_err=5.96e-08, rel_err=7.50e-03
  Workload ea27be17...: PASSED | 0.075 ms | 217.60x speedup | abs_err=7.15e-07, rel_err=8.32e-03
  Workload 49ef89d2...: PASSED | 0.078 ms | 163.21x speedup | abs_err=1.91e-06, rel_err=3.87e-02
  Workload 056224b8...: PASSED | 0.080 ms | 183.20x speedup | abs_err=4.77e-07, rel_err=6.69e-03
  Workload 685d26ff...: PASSED | 0.040 ms | 67.24x speedup | abs_err=1.19e-07, rel_err=5.69e-03
  Workload 352c9ace...: PASSED | 0.395 ms | 263.15x speedup | abs_err=3.05e-05, rel_err=2.33e-02
  Workload 2c9693b4...: PASSED | 0.069 ms | 132.06x speedup | abs_err=2.98e-08, rel_err=5.83e-03
  Workload 27f44fd6...: PASSED | 0.066 ms | 160.47x speedup | abs_err=1.91e-06, rel_err=1.05e-02
  Workload 07aa7922...: PASSED | 3.958 ms | 270.12x speedup | abs_err=3.05e-05, rel_err=7.11e-02
  Workload eaa0fd47...: PASSED | 0.040 ms | 71.73x speedup | abs_err=9.54e-07, rel_err=6.45e-03
  Workload f105eda8...: PASSED | 0.325 ms | 385.92x speedup | abs_err=6.10e-05, rel_err=1.49e-02
  Workload cd979341...: PASSED | 0.074 ms | 197.74x speedup | abs_err=3.81e-06, rel_err=6.88e-03
  Workload 43bf9699...: PASSED | 0.652 ms | 277.00x speedup | abs_err=1.91e-06, rel_err=7.69e-03
  Workload 54856fec...: PASSED | 0.652 ms | 276.24x speedup | abs_err=3.05e-05, rel_err=1.01e-01
  Workload 2ba465c0...: PASSED | 0.044 ms | 122.44x speedup | abs_err=2.38e-07, rel_err=7.69e-03
  Workload 1efaf2a9...: PASSED | 0.051 ms | 155.64x speedup | abs_err=2.98e-08, rel_err=8.85e-03
  Workload a01a3f93...: PASSED | 1.508 ms | 271.59x speedup | abs_err=1.22e-04, rel_err=4.51e-02
  Workload cc241d2e...: PASSED | 0.047 ms | 98.11x speedup | abs_err=1.49e-07, rel_err=6.82e-03
[2026-04-06T23:45:14] Local entrypoint finished in 532.37s
[2026-04-06T14:45:13] Remote benchmark finished in 525.18s
Stopping app - local entrypoint completed.
Runner terminated.
✓ App completed. View run at 
https://modal.com/apps/shjj1504/main/ap-9TXphEPxramHGWjc22KXRK

```
