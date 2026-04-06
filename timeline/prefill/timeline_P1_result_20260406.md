# Prefill Timeline — P1 Result — 2026-04-06

## Optimization summary

이번 P1에서는 기존 full shared-state prefill kernel을
**SRTP-style warp-owned row kernel**로 바꿨다.

### Applied optimization

1. shared state tile 제거
2. warp 1개 = V-row 1개 ownership
3. 각 lane이 row의 4개 K 요소를 register에 보유
4. `(seq, head)`당 CTA 1개 대신 `(seq, head, row-tile)` 구조로 확장
5. q/k를 warp별 bf16x4 vectorized load
6. token loop 내부 `__syncthreads()` 제거
7. gate/beta precompute는 그대로 유지

핵심 의도:

> shared footprint와 barrier 비용을 제거하고,  
> CTA 수를 크게 늘려 latency hiding을 확보하면서  
> state update를 warp-local register dataflow로 바꾸기

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
- avg latency: `0.058 ms`
- worst abs error: `1.91e-06`
- worst rel error: `1.93e-02`

### 3. full decision gate

```bash
modal run scripts/run_modal.py --decision-gate --max-workloads 100
```

결과:

- `PASSED=100/100`
- avg latency: `0.525 ms`
- avg speedup: `390.63x`
- worst abs error: `1.22e-04`
- worst rel error: `2.97e-01`

---

## Tiny NCU comparison (`77daf91d`)

### Old baseline kernel

from `profiles/ncu/10_baseline_capture.md`:

- `gdn_prefill_kernel` duration: `78.24 us`
- registers/thread: `255`
- dynamic shared memory: `66.56 KB`
- spill requests: `7552`
- grid size: `8`
- waves/SM: `0.03`

### New P1 kernel

tiny workload re-capture after P1:

- `gdn_prefill_kernel` duration: `11.81 us`
- registers/thread: `32`
- dynamic shared memory: `0`
- spill requests: `0`
- grid size: `256`
- waves/SM: `0.11`
- achieved occupancy: `10.48%`

### What changed

1. duration dropped from `78.24 us` → `11.81 us`
2. register pressure collapsed from `255` → `32`
3. shared memory usage dropped from `66.56 KB` → `0`
4. spill requests dropped from `7552` → `0`
5. grid coverage increased from `8` blocks → `256` blocks

### Precompute kernel

precompute stayed small:

- old: `5.63 us`
- new: `5.89 us`

즉 성능 변화의 핵심은 precompute가 아니라 main kernel 구조 변경이다.

---

## Judgement

이번 iteration은 **강하게 개선되었다**고 판단한다.

### Why this is a keep

1. `PASSED=100/100`
2. quick gate와 full decision gate 모두 안정적으로 통과
3. full avg latency가 `0.525 ms`까지 내려감
4. tiny NCU에서도 구조 개선이 분명히 보임
   - no shared tile
   - no spill
   - much larger grid

즉 이번 P1은 단순 미세조정이 아니라,
**구조적 병목(shared tile / spill / underfill)을 실제로 제거한 성공적 rewrite**다.

---

## Key takeaway

이번 결과는 SRTP 가설을 강하게 지지한다.

> prefill에서 중요한 것은 gate math 미세조정보다  
> row ownership 단순화, shared-state 제거, 그리고 CTA 공급량 확대였다.

---

## Next

다음 루프에서는 아래를 우선 검토할 가치가 있다.

1. 현재 SRTP 위에서 q/k load path 추가 정리
2. row-per-warp와 row-pair-per-warp 비교
3. long-tail workload(`07aa7922`) 중심으로 launch shape 추가 튜닝
