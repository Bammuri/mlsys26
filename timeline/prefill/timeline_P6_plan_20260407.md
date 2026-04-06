# Prefill Timeline — P6 Plan — 2026-04-07

## Stage goal

P6의 목표는 **gate/beta를 separate float arrays 대신 packed `float2` buffer로 바꾸는 것**이다.

이번 iteration의 핵심 질문:

> current keep baseline(P3)은 main kernel이 token마다 lane 0가 gate와 beta를 각각 읽고 broadcast한다.  
> 이 둘을 하나의 packed `float2` read로 바꾸면 full 100 평균 latency가 더 좋아질까?

---

## Why this optimization was chosen

### 1. P4/P5가 막힌 search directions

- P4: CTA granularity를 1 warp/block까지 줄였더니 full 평균 악화
- P5: blanket L1 cache hint도 full 평균 악화

즉 다음 후보는 launch shape나 cache hint보다,
**representation/data movement 자체를 조금 더 단순화하는 것**이 낫다.

### 2. Current kernel still loads gate and beta separately

현재 main kernel은 token마다 lane 0가:

- `gate[gate_idx]`
- `beta[gate_idx]`

를 각각 읽고 warp broadcast한다.

이 둘은 항상 같이 소비된다.

즉 다음 자연스러운 candidate는:

> per-token scalar pair를 packed form으로 바꿔,
> read path를 하나로 줄이기

이다.

### 3. Why this is low-risk

이번 변경은:

- kernel math 불변
- launch shape 불변 (P3 keep baseline 유지)
- no-shared / row-per-warp 구조 유지
- correctness risk 낮음

즉 current keep baseline을 크게 흔들지 않는 low-risk memory-path tuning이다.

---

## Planned code change

수정 대상:

- `solution/cuda/kernel.cu`

변경 포인트:

1. `compute_gate_beta_kernel` 출력 형식을 `float2` packed buffer로 변경
2. host side temporary tensor도 packed layout으로 변경
3. main kernel은 lane 0가 packed `float2` 1회 load 후 broadcast

핵심 의도:

> current SRTP baseline의 scalar side-data path를 더 compact하게 만들어,
> duplicated scalar global loads를 줄이기

---

## Expected result before testing

baseline (keep baseline, full 100, `warmup_runs=1, iterations=5, num_trials=2`):

- avg latency: `0.511 ms`

예상:

1. quick 5 PASS 유지
2. full 100 PASS 유지
3. gain이 있더라도 small gain일 가능성이 높음
4. no-op 가능성도 높음

최종 판단은 **full workload avg latency arithmetic mean** 기준으로만 한다.

---

## Test plan

1. `python scripts/pack_solution.py`
2. `modal run scripts/run_modal.py --quick --max-workloads 5`
3. `modal run scripts/run_modal.py --decision-gate --max-workloads 100`
