# Prefill Timeline — P4 Plan — 2026-04-07

## Stage goal

P4의 목표는 **row-per-warp는 유지하되 block을 1 warp까지 더 줄여 CTA 공급량을 극대화하는 것**이다.

이번 iteration의 핵심 질문:

> 현재 keep baseline(P3)은 2 warps/block으로 full 평균 `0.511 ms`를 달성했다.  
> block을 1 warp로 더 줄여 `grid.x`를 다시 2배로 키우면,  
> tiny/long-tail underfill을 더 완화해 full 평균 latency를 추가로 낮출 수 있을까?

---

## Why this optimization was chosen

### 1. P2와 P3의 교훈

- P2 실패:
  - warp당 일을 늘리면 full 평균이 크게 악화됨 (`0.517 ms -> 0.782 ms`)
- P3 성공:
  - warp당 일은 유지하고 CTA 공급량을 늘리면 full 평균이 소폭 개선됨 (`0.517 ms -> 0.511 ms`)

즉 현재까지의 strongest signal은 다음이다.

> **reuse를 위해 warp당 serial work를 늘리는 것보다,  
> independent warp/CTA를 더 많이 공급하는 것이 낫다.**

### 2. Current baseline still shows underfill risk

P3 결과 기준 representative workloads:

- `77daf91d`: `0.035 ms`
- `07aa7922`: `2.651 ms`

이 둘은 여전히 low-seq / long-tail 관점에서,
launch shape 영향이 큰 구간이다.

따라서 다음 자연스러운 low-risk step은:

> row ownership을 건드리지 않고,
> block size만 더 줄여 launch granularity를 높이는 것

이다.

### 3. Why this is low-risk

이번 변경은:

- row-per-warp 유지
- no shared state 유지
- no spill 구조 유지 가능성 높음
- gate/beta precompute 유지
- math 동일

즉 P2처럼 per-warp serial work를 늘리는 실험이 아니라,
**P3의 same family에서 block granularity만 더 줄이는 tuning pass**다.

---

## Planned code change

수정 대상:

- `solution/cuda/kernel.cu`

변경 포인트:

1. `kWarpsPerBlock = 2` → `1`
2. row-per-warp ownership 유지
3. block size `64` → `32`
4. `grid.x = head * row_tiles` 증가

핵심 의도:

> row-per-warp / no-shared / no-spill dataflow는 그대로 두고,
> 가능한 한 많은 CTA를 공급해 latency hiding을 더 끌어내기

---

## Expected result before testing

baseline (keep baseline, full 100, `warmup_runs=1, iterations=5, num_trials=2`):

- avg latency: `0.511 ms`

예상:

1. quick 5 PASS 유지
2. full 100 PASS 유지
3. tiny/long-tail에서 small positive gain 가능
4. throughput-heavy에서는 launch overhead 때문에 no-op 또는 slight regression 가능

최종 판단은 **full 100 arithmetic-mean latency** 기준으로만 한다.

---

## Test plan

1. `python scripts/pack_solution.py`
2. `modal run scripts/run_modal.py --quick --max-workloads 5`
3. `modal run scripts/run_modal.py --decision-gate --max-workloads 100`
4. tiny representative workload (`77daf91d`) Modal NCU recapture
