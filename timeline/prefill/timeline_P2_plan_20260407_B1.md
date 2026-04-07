# Prefill Timeline — P2 Plan — 2026-04-07 (B1)

## Branch

- baseline family: **B1 contiguous-memory / wrapper-boundary cleanup**
- starting point: current keep baseline `77bd812` (`0.490 ms` full avg latency)
- decision rule: **full 100-workload arithmetic-mean latency only**
- fixed config: `warmup_runs=1`, `iterations=5`, `num_trials=2`

## Why this branch now

전체 100-workload NCU와 이전 timeline을 합치면 다음이 명확하다.

1. `F5 throughput-heavy`와 `F3 long-tail-low-seq`가 full mean을 지배한다.
2. main kernel은 이미 `F5`에서 occupancy가 꽤 올라와 있어, launch-only tuning의 한계가 보인다.
3. naive full fusion(B2-P1)은 precompute boundary를 없애는 대신 token hot path 안에 `threadIdx.x==0 + __syncthreads()` serial/sync section을 넣어서 크게 실패했다 (`0.490 -> 0.919 ms`).
4. tomas의 큰 wins 중 하나는 **kernel 밖의 불필요한 contiguous/transpose/copy/allocation 제거**였다.

즉 다음은 hot path를 다시 건드리기보다,
**current P6 kernel은 그대로 두고 wrapper boundary의 unnecessary work를 줄이는** 방향이 더 안전하다.

## Concrete variant

이번 variant는 다음 두 가지를 묶는다.

1. **redundant contiguous() no-op 제거**
   - 이미 `CHECK_CONTIGUOUS(...)`를 통과한 텐서들에 대해 다시 `.contiguous()`를 호출하지 않는다.
   - 대상: `q`, `k`, `v`, `A_log`, `a`, `dt_bias`, `b`, `cu_seqlens`, `state`

2. **state output allocation boundary 제거**
   - `has_state == true`일 때 `new_state`를 새로 allocate하지 않고 `state_in` storage를 그대로 output state로 재사용한다.
   - kernel은 same-buffer read/write(in-place state update)로 실행한다.
   - rationale: benchmark harness는 iteration마다 fresh input clones를 사용하므로, in-place state mutation은 contract-safe일 가능성이 높다.

핵심은:
> kernel 내부 recurrence/dataflow는 그대로 유지하면서,
> tomas 식으로 kernel 주변의 불필요한 wrapper work를 줄여 end-to-end latency를 깎기.

## Expected result before testing

baseline:
- current keep baseline: `0.490 ms`

예상:
1. quick 5 PASS 유지
2. full 100 PASS 유지
3. gain이 있다면 small-to-moderate gain일 가능성이 높음
4. worst-case는 no-op
5. correctness risk는 낮지만, input-state aliasing이 benchmark contract와 충돌하면 numerical/runtime failure 가능성 있음

## Why this is distinct from prior failed branches

- P4/P5는 launch/cache policy를 건드렸다.
- P8은 dtype contract를 건드렸다.
- P9는 side-data layout reorder였다.
- B2-P1은 token hot path 안에 serial/sync section을 집어넣었다.

이번 branch는 반대로,
- **kernel math / launch family / precision / row ownership은 그대로 유지**하고
- wrapper-level unnecessary work만 제거한다.

## Verification plan

1. `python scripts/pack_solution.py`
2. `modal run scripts/run_modal.py --quick --max-workloads 5`
3. `modal run scripts/run_modal.py --decision-gate --max-workloads 100`
4. full pref log 저장
5. full mean 기준 keep/revert 판단

## NCU plan

이번 branch는 kernel hot path 자체를 바꾸지 않으므로,
- **default는 NCU 생략**
- 다만 full 결과가 의미 있게 개선되거나 이상하면 representative workload 재프로파일링 검토

## Keep / revert rule

- improve on full avg latency: **KEEP + commit**
- no improvement / regression / contract issue: **REVERT without commit**
