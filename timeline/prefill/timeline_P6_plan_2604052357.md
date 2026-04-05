# Prefill Timeline — P6 Plan — 2604052357

## Stage goal

P6의 목표는 **generic sm_100가 아니라 `sm_100a` codegen target을 직접 활용하는 것**이다.

이번 iteration의 핵심 질문:

> 같은 native CUDA kernel이라도, compile target을 `sm_100a`로 명시하면
> full decision gate 기준으로 measurable improvement가 생길까?

---

## Why this optimization was chosen

### 1. New background fact

사용자 업데이트에 따라:

> 평가 환경은 `sm_100a` 기준으로 본다

이제 background에도 이를 명시했다.

즉, 다음 자연스러운 전략은:

> 커널 수학을 크게 바꾸기 전에,
> compile target 자체를 평가 환경에 더 정확히 맞추는 것

이다.

### 2. Why this is a good next loop after P5

P5까지의 흐름을 보면:

- P1: q/k shared staging — 성공
- P2: barrier tweak — 실패
- P3: float4 vectorization — 성공
- P4: row-block repartition — 실패
- P5: scalar-hoist + decision gate 확립 — 성공

즉 현재는:

> correctness를 유지하는 작은/중간 규모의 최적화가 성과를 내고 있고,
> 위험한 구조 변경은 실패할 가능성이 높다.

`sm_100a` compile targeting은:

- low-risk
- architecture-aware
- current native path에 잘 맞는

다음 step이다.

### 3. Strategy being added

이번 iteration의 전략은:

1. compile target을 `sm_100a`로 명시
2. architecture-specific codegen을 우선 활용
3. 수학/CTA 구조는 그대로 유지
4. pure codegen benefit만 먼저 확인

즉,

> 알고리즘 refactor 전에 architecture match를 먼저 맞춘다

---

## Planned code change

이번 P6에서는 kernel math는 바꾸지 않는다.

변경 포인트:

1. Modal runtime / Torch extension compile path에 `TORCH_CUDA_ARCH_LIST=10.0a` 반영
2. background에 `sm_100a` targeting 전략 명시

---

## Expected result before testing

현재 P5 decision-gate baseline:

- full 100
  - avg latency: `4.923 ms`
  - avg speedup: `48.41x`
  - `PASSED=100/100`

예상:

1. quick 5는 PASS 유지
2. full 100 decision gate도 PASS 유지
3. 개선 폭은 작거나 0일 수도 있음
4. 하지만 만약 compiler가 `sm_100a`에서 더 좋은 codegen을 만들면 small positive gain 가능

---

## Test plan

### 3-1 quick working check

```bash
modal run scripts/run_modal.py --quick --max-workloads 5
```

### 3-2 full decision gate

```bash
modal run scripts/run_modal.py --decision-gate --max-workloads 100
```

판단은 full 100 decision gate 기준으로만 한다.

