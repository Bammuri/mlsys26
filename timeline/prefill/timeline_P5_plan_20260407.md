# Prefill Timeline — P5 Plan — 2026-04-07

## Stage goal

P5의 목표는 **현재 no-shared SRTP kernel에 L1-prefer cache policy를 주는 것**이다.

이번 iteration의 핵심 질문:

> 현재 keep baseline(P3)은 shared memory를 거의 쓰지 않는 row-per-warp kernel이다.  
> 이 상황에서 runtime cache policy를 `cudaFuncCachePreferL1`로 명시하면,  
> q/k/state load 재사용에 더 유리해져 full 평균 latency가 조금이라도 내려갈까?

---

## Why this optimization was chosen

### 1. Current kernel characteristics changed completely from the old shared-heavy design

P1 이후 kernel은:

- full shared-state tile 없음
- dynamic shared memory 없음
- spill 제거 완료
- q/k/state loads는 cache hierarchy에 더 직접 의존

즉 과거 shared-heavy kernel에서의 cache-hint 실험 결과를
그대로 가져오면 안 된다.

### 2. Background alignment

background는 hardware-level metrics, L1/L2 behavior, SM occupancy를 보라고 했다.
현재 kernel은 launch shape는 많이 개선됐고,
이제 남은 작은 tuning 중 하나는 memory hierarchy hint다.

### 3. Why this is low-risk

이번 변경은:

- kernel math 불변
- launch shape 불변
- register ownership 불변
- correctness risk 거의 없음

즉 빠르게 full 기준으로 검증할 수 있는 low-risk pass다.

---

## Planned code change

수정 대상:

- `solution/cuda/kernel.cu`

변경 포인트:

1. `cudaFuncSetCacheConfig(gdn_prefill_kernel, cudaFuncCachePreferL1)` 추가
2. 필요하면 precompute kernel은 기본값 유지

핵심 의도:

> shared memory를 거의 쓰지 않는 현재 SRTP kernel에서
> L1 쪽 cache preference가 실제로 작은 positive gain을 줄 수 있는지 확인

---

## Expected result before testing

baseline (keep baseline, full 100, `warmup_runs=1, iterations=5, num_trials=2`):

- avg latency: `0.511 ms`

예상:

1. quick 5 PASS 유지
2. full 100 PASS 유지
3. 개선이 있다면 small single-digit % 또는 그 이하일 가능성이 높음
4. no-op 또는 slight regression 가능성도 있음

최종 판단은 **full workload avg latency arithmetic mean** 기준으로만 한다.

---

## Test plan

1. `python scripts/pack_solution.py`
2. `modal run scripts/run_modal.py --quick --max-workloads 5`
3. `modal run scripts/run_modal.py --decision-gate --max-workloads 100`
