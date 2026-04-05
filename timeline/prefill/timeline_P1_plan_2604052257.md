# Prefill Timeline — P1 — 2604052257

## Stage goal

P1의 목표는 첫 번째 native CUDA 최적화 iteration을 수행하는 것이다.

이번 iteration의 핵심 질문:

> 현재 kernel이 같은 token/head에 대해 q와 k를 thread마다 반복해서 global memory에서 읽고 있는데,  
> 이를 shared staging으로 줄이면 subset quick latency가 좋아질까?

---

## Why this optimization was chosen

background와 P0 baseline을 보면, 지금 가장 자연스러운 첫 최적화는 메모리 계층 정리다.

### 1. background 근거

- `d128`, `k_last`를 고정치로 적극 활용하라고 명시 (`background.md:39`)
- H2D/D2H 뿐 아니라 전체 memory tiering을 의식하라고 명시 (`background.md:40`, `background.md:72-73`)
- device-side fusion과 on-device residency를 최대화하라고 명시 (`background.md:43`)

즉, 첫 최적화는:

> “이미 kernel 안에 있는 반복적이고 명백한 메모리 낭비를 줄이는 것”

이 가장 background 친화적이다.

### 2. 현재 커널 구조 근거

현재 P0 kernel은:

- `(seq, v-head)` 별 CTA
- 각 thread가 `v_idx` 하나를 담당
- token loop 내부에서
  - `k[128]`을 old_v 계산용으로 반복해서 읽고
  - `k[128]`을 state update용으로 또 읽고
  - `q[128]`을 output 계산용으로 또 읽는다

이 구조는 correctness-first로는 좋지만,

> 같은 token/head에 대해 모든 thread가 같은 q/k vector를 중복으로 읽는다는 점이
> 첫 번째 병목 후보로 매우 자연스럽다.

### 3. 위험 대비 기대효과

이번 변경은 비교적 low-risk다.

변경 내용:

- token/head 단위 q[128], k[128]를 shared memory에 한 번 staged
- inner loop는 shared q/k를 재사용
- `head_size=128` 고정이므로 loop unroll을 적극 사용

장점:

1. 수학식은 거의 안 바뀜
2. correctness risk가 낮음
3. global load redundancy를 바로 줄일 수 있음

리스크:

1. shared memory 사용량이 약간 늘어남
2. sync가 추가됨
3. B200에서 실제 병목이 compute라면 이득이 작을 수 있음

---

## Expected result before testing

예상:

1. 5-workload quick는 계속 PASS 해야 한다
2. avg latency가 P0 baseline보다 개선될 가능성이 높다
3. 특히 짧은 workload보다 중간/긴 workload에서 이득이 더 보일 수 있다

P0 comparison target:

- 5 workloads quick
- avg latency: `0.386 ms`
- avg speedup: `13.51x`

---

## Planned code change

이번 P1에서는 `solution/cuda/kernel.cu`만 수정한다.

변경 포인트:

1. dynamic shared memory 안에 `state_sh` 뒤로 `q_sh`, `k_sh` 추가
2. token loop마다:
   - thread별로 q/k 한 element씩 load
   - shared q/k를 old_v / state update / output에서 재사용
3. fixed `128` loop에 `#pragma unroll` 추가

---

## Test plan

1. compile sanity:
   - `python3 scripts/pack_solution.py`
2. native quick benchmark:
   - `modal run scripts/run_modal.py --quick --max-workloads 5`
3. compare against P0:
   - pass/fail
   - avg latency
   - avg speedup
   - worst abs/rel error

---

## Result

_To be filled after benchmark._

