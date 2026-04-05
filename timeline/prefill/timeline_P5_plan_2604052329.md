# Prefill Timeline — P5 Plan — 2604052329

## Stage goal

P5의 목표는 **long-seq에서 반복되는 per-token scalar overhead를 줄이는 것**이다.

이번 iteration의 핵심 질문:

> `exp(A_log[head])`, `dt_bias[head]`, `scale`, 그리고 token마다 반복되는 일부 scalar 작업을
> CTA 바깥/loop 바깥으로 hoist하면 full 100 decision gate에서 이득이 날까?

---

## Why this optimization was chosen

### 1. Previous loop history

- P1 성공: q/k shared staging
- P2 실패: barrier 미세조정은 충분한 개선 아님
- P3 성공: float4 vectorization
- P4 실패: row-block CTA repartition은 correctness를 깨뜨림

즉 현재 방향성은:

> 큰 구조를 흔드는 것보다,
> correctness-first baseline 위에서 안전하게 누적 비용을 줄이는 쪽이 낫다.

### 2. Background alignment

background는:

- `d128`, `k_last` 특화 (`background.md:39`)
- locked clocks 환경에서 hardware-level 효율을 보라 (`background.md:19-21`)
- memory tiering과 on-device reuse를 강화하라 (`background.md:40-43`)

이번 P5는:

> long-seq에서 token마다 반복되는 scalar 연산/로드를 줄여
> on-device reuse를 높이는 미세 최적화

라는 점에서 background와 일치한다.

### 3. Why this is low risk

이번 변경은:

- 수학식 불변
- CTA mapping 불변
- shared layout 불변
- row ownership 불변

즉 P4처럼 correctness를 깨뜨릴 구조 변경이 없다.

---

## Planned code change

이번 P5에서는 `solution/cuda/kernel.cu`만 수정한다.

변경 포인트:

1. `exp(A_log[head_idx])`를 token loop 밖으로 hoist
2. `dt_bias[head_idx]`를 token loop 밖으로 hoist
3. `scale`를 float로 한번만 캐스팅
4. gate compute에서 반복되는 scalar loads/exp 일부 제거

---

## Expected result before testing

Decision gate baseline is now:

```bash
modal run scripts/run_modal.py --decision-gate --max-workloads 100
```

P3 quick-100 reference:

- avg latency: `4.935 ms`
- wall time: `126.58s`
- `PASSED=100/100`

예상:

1. quick 5는 PASS 유지
2. full 100 decision gate도 PASS 유지
3. 개선 폭은 크지 않을 수 있지만, long-seq 누적 비용 감소로 small positive gain 가능

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

