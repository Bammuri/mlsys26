# Prefill Timeline — P11 Plan — 2604060037

## Stage goal

P11의 목표는 **gate/beta scalar math를 main kernel 밖으로 빼는 것**이다.

이번 iteration의 핵심 질문:

> token마다 CTA의 thread0가 `softplus/exp/sigmoid`를 계산하는 대신,  
> 별도 GPU precompute kernel에서 gate/beta를 미리 만들어두면 full decision gate 기준으로 좋아질까?

---

## Why this optimization was chosen

### 1. Previous loop evidence

- P7 실패: staging micro-tuning은 효과 부족
- P8 실패: ILP micro-tuning은 효과 부족
- P9 실패: fast-math scalar tweak도 효과 부족
- P10 실패: cache/shared runtime hint도 효과 부족

즉,

> 이제 작은 micro-tuning보다, 남아 있는 serial-looking scalar work를 구조적으로 떼어내는 시도를 할 차례다.

### 2. Why gate/beta precompute is attractive

현재 main kernel에서는 token마다 thread0가:

- `x = a + dt_bias`
- `softplus(x)`
- `gate = exp(-exp(A_log) * softplus(x))`
- `beta = sigmoid(b)`

를 계산한다.

이건:

- token 수가 길수록 누적되고
- head마다 반복되고
- CTA 안에서 사실상 serial phase가 된다

따라서,

> 이 부분을 별도 GPU kernel로 batch-style로 계산한 뒤,
> main kernel은 precomputed gate/beta만 읽게 만들면 더 낫을 가능성이 있다.

### 3. Why this is still correctness-manageable

이번 변경은 구조 변화가 있긴 하지만:

- 수학식은 그대로
- state update math는 그대로
- CTA mapping은 그대로

즉,

> P4처럼 row ownership을 깨는 구조 변경은 아니고,
> main kernel에서 scalar precompute만 분리하는 수준이다.

---

## Planned code change

이번 P11에서는 `solution/cuda/kernel.cu`만 수정한다.

변경 포인트:

1. `compute_gate_beta_kernel` 추가
2. `gate` / `beta` temporary tensor를 `gdn_prefill_cuda` 안에서 생성
3. main `gdn_prefill_kernel`은 precomputed gate/beta를 입력으로 받음
4. main token loop에서는 thread0가 exp/softplus/sigmoid를 계산하지 않고 값만 읽음

---

## Expected result before testing

현재 baseline(P6):

- full decision gate avg latency: `4.913 ms`

예상:

1. quick 5 PASS 유지
2. full 100 PASS 유지
3. long-seq workloads에서 이득 가능
4. full 평균 latency가 내려가면 keep

---

## Test plan

### 3-1 quick working gate

```bash
modal run scripts/run_modal.py --quick --max-workloads 5
```

### 3-2 full decision gate

```bash
modal run scripts/run_modal.py --decision-gate --max-workloads 100
```

판단 기준은 full 100 arithmetic-mean latency이다.

