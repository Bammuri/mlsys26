# Prefill Timeline — P1 Plan — 2026-04-06

## Stage goal

P1의 목표는 P0에서 정의한 **SRTP (Streaming Register-Tiled Prefill)** 의
첫 번째 실행 가능한 prototype을 만드는 것이다.

핵심 질문:

> full shared-state CTA 구조를 버리고,  
> `(seq, head, v_tile)` 기반 warp-owned row update로 바꾸면  
> 실제 full decision-gate 평균 latency가 크게 내려갈까?

---

## Why this optimization was chosen

### 1. P0 planning direction

`plan/00_p0_restart.md`에서 새 알고리즘 방향으로 SRTP를 잡았다.

핵심 원칙:

- reference는 원리만 참고
- decode 구조는 그대로 복제하지 않음
- prefill long-sequence에 맞는 새 launch/dataflow를 만든다

### 2. NCU baseline evidence

baseline NCU는 다음을 보여줬다.

- `gdn_prefill_kernel` tiny case duration: `78.24 us`
- registers/thread: `255`
- dynamic shared memory: `66.56 KB`
- spill requests: `7552`
- shared excessive wavefronts: `83%`

즉 현재 구조의 병목은 분명했다:

1. full shared-state tile
2. 높은 register pressure + spill
3. tiny/long-tail에서의 grid underfill

### 3. Why this prototype is different from old failed repartition

예전 row-block repartition 실패는 correctness를 깨뜨렸다.  
이번 P1은 그와 다르게:

- row independence를 warp 단위 ownership으로 단순화
- inter-warp communication 제거
- token loop 내부 barrier 제거
- gate/beta precompute는 유지

즉 **더 작고 명시적인 ownership model**을 가진 구조다.

---

## Planned code change

수정 대상:

- `solution/cuda/kernel.cu`

변경 포인트:

1. full `128x128` shared-state tile 제거
2. warp 1개가 row 1개를 담당
3. thread 32개가 row의 K=128을 `float4` fragments로 register에 유지
4. `grid.x = kNumVHeads * kRowTilesPerHead` 로 확장
5. q/k는 warp별 vectorized bf16x4 load
6. gate/beta precompute kernel은 유지

---

## Test plan

1. `python scripts/pack_solution.py`
2. `modal run scripts/run_modal.py --quick --max-workloads 5`
3. `modal run scripts/run_modal.py --decision-gate --max-workloads 100`
4. tiny representative workload(`77daf91d`)에 대해 Modal NCU 재측정
