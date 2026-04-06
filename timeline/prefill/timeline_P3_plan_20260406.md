# Prefill Timeline — P3 Plan — 2026-04-06

## Stage goal

P3의 목표는 **row-per-warp는 유지하되 block size를 더 줄여 CTA 공급량을 늘리는 것**이다.

이번 iteration의 핵심 질문:

> 현재 P1 keep baseline은 row-per-warp 구조로 잘 작동하지만, tiny/long-tail에서는 여전히 grid underfill이 남아 있다.  
> warps/block을 4 → 2로 줄여 `grid.x`를 256 → 512로 늘리면 full 평균 latency가 더 좋아질까?

---

## Why this optimization was chosen

### 1. P2 실패가 가르쳐준 것

P2는 row-pair-per-warp로 q/k reuse를 늘리려 했지만,
full 100 기준 `0.517 ms` → `0.782 ms`로 악화됐다.

핵심 교훈:

> 현재 단계에서는 warp당 일을 더 많이 주는 것보다,  
> 더 많은 CTA/warp를 공급해 latency hiding을 유지하는 것이 더 중요하다.

### 2. Current remaining issue in P1 baseline

P1 tiny NCU:

- `gdn_prefill_kernel` duration: `11.81 us`
- grid size: `256`
- achieved occupancy: `10.48%`
- waves/SM: `0.11`

즉 shared/spill 문제는 크게 줄였지만,
아직도 tiny에서는 GPU가 충분히 차지 않는다.

### 3. Why smaller blocks are a plausible next move

현재 kernel은:

- no shared state
- low regs/thread (`32`)
- no spill

즉 block size를 줄이더라도 resource pressure 때문에 망가질 가능성은 낮다.

따라서 다음 자연스러운 질문은:

> row ownership은 유지한 채,  
> block당 warp 수만 줄여 더 많은 block을 만들면 어떨까?

이다.

---

## Planned code change

수정 대상:

- `solution/cuda/kernel.cu`

변경 포인트:

1. `kWarpsPerBlock = 4` → `2`
2. row-per-warp 유지
3. block size `128` → `64`
4. `grid.x = head * row_tiles` 증가

핵심 의도:

> row-per-warp의 단순성과 no-shared/no-spill dataflow는 유지하면서,  
> tiny/long-tail에서 block 공급량을 더 늘리기

---

## Expected result before testing

baseline (keep baseline, full 100, `warmup_runs=1, iterations=5, num_trials=2`):

- avg latency: `0.517 ms`

예상:

1. quick 5 PASS 유지
2. full 100 PASS 유지
3. tiny/long-tail에서 positive gain 가능
4. throughput-heavy에서는 block당 work 축소 때문에 no-op 또는 slight regression 가능

최종 판단은 **full 100 arithmetic-mean latency** 기준으로만 한다.

---

## Test plan

1. `python scripts/pack_solution.py`
2. `modal run scripts/run_modal.py --quick --max-workloads 5`
3. `modal run scripts/run_modal.py --decision-gate --max-workloads 100`
4. tiny representative workload (`77daf91d`) NCU 재측정
