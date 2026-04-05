# Prefill Timeline — P3 Plan — 2604052313

## Stage goal

P3의 목표는 현재 kernel의 **scalar 128-iteration inner loops**를 줄이는 것이다.

이번 iteration의 핵심 질문:

> q/k/state row 연산이 여전히 scalar loop 중심인데,  
> 이를 `float4` 벡터화로 바꾸면 quick와 full 100 workload 모두에서 더 강한 개선이 나올까?

---

## Why this optimization was chosen

### 1. P2 결과가 말해준 것

P2에서는 barrier 감소만으로는 충분한 이득이 나오지 않았다.

그 해석은 이미 정리했다:

> 현재 병목은 synchronization 자체보다  
> state row update / dot-product / shared-memory scalar sweep 쪽일 가능성이 높다.

즉, 다음 iteration은 coordination보다 **실제 inner-loop arithmetic / load-use pattern**에 손을 대는 게 맞다.

### 2. Current kernel shape is very vectorization-friendly

현재 커널은:

- `head_size = 128` 고정
- `q`, `k`, `row` 모두 128-float contiguous segments
- row stride가 128 floats 단위

즉,

> `128 = 32 * float4` 이기 때문에  
> scalar 128-step loop를 32-step vector loop로 바꾸기 좋은 구조다.

### 3. Why this is a better candidate than P2

P2는 barrier 최적화였고, 실제 hardware-visible work를 크게 줄이지 못했다.

반면 P3는:

1. dot accumulation loop 반복 횟수를 줄이고
2. state update loop 반복 횟수를 줄이고
3. state load/store loop 반복 횟수도 줄인다

즉,

> synchronization이 아니라 실제 instruction/load 반복 자체를 줄이는 최적화다.

---

## Planned code change

이번 P3에서는 `solution/cuda/kernel.cu`만 수정한다.

변경 포인트:

1. state load를 `float4` 단위로 로드
2. `old_v` 계산을 `float4` 단위 dot accumulation으로 변경
3. state update를 `float4` 단위로 변경
4. output projection도 `float4` 단위 accumulation으로 변경
5. final state writeback도 `float4` 단위로 저장

---

## Expected result before testing

P1 baseline:

- quick 5 workloads:
  - avg latency: `0.223 ms`
  - avg speedup: `24.24x`
- quick 100 workloads:
  - avg latency: `5.051 ms`
  - avg speedup: `46.83x`

P3 expectation:

1. quick 5 workloads는 P1보다 명확히 좋아질 가능성이 있다
2. quick 100 workloads 평균도 의미 있게 개선될 가능성이 있다
3. 특히 short/medium workloads에서 instruction pressure 감소 효과가 더 잘 보일 수 있다

---

## Test plan

### 3-1 quick working check

```bash
modal run scripts/run_modal.py --quick --max-workloads 5
```

### 3-2 all workload extraction

```bash
modal run scripts/run_modal.py --quick --max-workloads 100
```

full pref log는 result 문서에 포함한다.

