# Prefill Timeline — P14 Plan — 2604060628

## Stage goal

P14의 목표는 **compile-time launch bounds 힌트**를 통해 register allocation / occupancy tradeoff를 개선하는 것이다.

이번 iteration의 핵심 질문:

> 현재 kernel이 shared-heavy이긴 하지만, launch bounds를 명시하면 compiler가 더 좋은 register scheduling을 선택해 full 평균 latency가 좋아질까?

---

## Why this optimization was chosen

### 1. Recent pattern

P11 성공 이후:

- P12 실패
- P13 실패

즉, 연속 실패 수는 2다.

지금 필요한 것은:

> correctness risk가 거의 없고, compiler behavior만 바꾸는 low-risk 탐색

이다.

### 2. Why launch bounds is a good candidate

현재 kernel은:

- block size 고정: 128 threads
- shared memory usage 큼
- main compute path는 어느 정도 정리됨

이 상황에서 compiler에게:

- launch shape
- blocks-per-SM 기대치

를 힌트 주면 register allocation을 다르게 할 수 있다.

즉,

> math는 그대로 두고 codegen/register pressure 쪽만 건드리는 loop다.

---

## Planned code change

이번 P14에서는 `solution/cuda/kernel.cu`만 수정한다.

변경 포인트:

1. `gdn_prefill_kernel`에 `__launch_bounds__(128, 2)` 추가
2. `compute_gate_beta_kernel`에도 적절한 launch bounds 추가

---

## Expected result before testing

현재 baseline(P11):

- full decision gate avg latency: `4.732 ms`

예상:

1. quick 5 PASS 유지
2. full 100 PASS 유지
3. small positive gain 또는 no-op 가능

판단 기준:

- full 100 arithmetic-mean latency

