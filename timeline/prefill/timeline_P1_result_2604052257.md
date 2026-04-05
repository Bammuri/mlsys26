# Prefill Timeline — P1 Result — 2604052257

## Optimization summary

이번 P1 iteration에서는 `kernel.cu`에 다음 최적화를 적용했다.

### Applied optimization

1. token/head 단위로 `q[128]`, `k[128]`를 shared memory에 staged
2. `old_v` 계산, state update, output projection에서 shared `q/k` 재사용
3. fixed `head_size = 128` 루프에 `#pragma unroll` 적용

핵심 의도:

> 같은 token/head에 대해 모든 thread가 동일한 q/k vector를 반복적으로 global memory에서 읽는 낭비를 줄이기

---

## Expected result vs actual result

## Expected before running

예상했던 것은 다음과 같다.

1. 5-workload quick는 계속 PASS 해야 한다
2. 5-workload avg latency는 P0 baseline보다 개선될 가능성이 높다
3. 중간/긴 workload에서 이득이 더 눈에 띌 수 있다
4. shared memory 사용량 증가는 기존 128x128 state tile에 비해 상대적으로 작아서, 이득을 완전히 상쇄하지는 않을 가능성이 높다

P0 comparison floor:

- 5 workloads quick
- avg latency: `0.386 ms`
- avg speedup: `13.51x`
- worst abs error: `7.63e-06`
- worst rel error: `2.15e-02`

## Actual results

### 3-1. Quick working check

실행:

```bash
modal run scripts/run_modal.py --quick --max-workloads 5
```

결과:

- `PASSED=5/5`
- avg latency: `0.223 ms`
- avg speedup: `24.24x`
- worst abs error: `7.63e-06`
- worst rel error: `2.15e-02`

대표 상세 로그:

- `Workload 77daf91d...: PASSED | 0.078 ms | 21.03x speedup | abs_err=5.96e-08, rel_err=9.53e-03`
- `Workload 9343fd82...: PASSED | 0.323 ms | 25.64x speedup | abs_err=1.91e-06, rel_err=7.63e-03`
- `Workload a39aa135...: PASSED | 0.388 ms | 25.49x speedup | abs_err=7.63e-06, rel_err=2.15e-02`
- `Workload 85d7becb...: PASSED | 0.155 ms | 20.55x speedup | abs_err=2.98e-07, rel_err=5.83e-03`
- `Workload cc241d2e...: PASSED | 0.172 ms | 28.51x speedup | abs_err=1.49e-07, rel_err=4.81e-03`

### 3-2. All workload performance extraction

실행:

```bash
modal run scripts/run_modal.py --quick --max-workloads 100
```

결과:

- `PASSED=100/100`
- avg latency: `5.051 ms`
- avg speedup: `46.83x`
- worst abs error: `1.22e-04`
- worst rel error: `3.47e-01`
- end-to-end wall time: `147.86s`

대표 tail / notable workloads:

- `Workload d0ce7b5d...: PASSED | 14.002 ms | 108.35x speedup | abs_err=6.10e-05, rel_err=9.37e-02`
- `Workload 5b8a0e4b...: PASSED | 21.572 ms | 67.87x speedup | abs_err=1.53e-05, rel_err=8.48e-02`
- `Workload 5d3fc66a...: PASSED | 21.495 ms | 67.89x speedup | abs_err=6.10e-05, rel_err=1.82e-01`
- `Workload e9e1e445...: PASSED | 12.614 ms | 120.00x speedup | abs_err=1.22e-04, rel_err=4.33e-02`
- `Workload ce832e76...: PASSED | 2.519 ms | 33.89x speedup | abs_err=3.81e-06, rel_err=7.65e-03`

---

## Judgement

이번 iteration은 **좋아졌다**고 판단한다.

### Why this is an improvement

#### 1. Quick baseline comparison is clearly positive

P0 → P1 (5-workload quick):

- avg latency: `0.386 ms` → `0.223 ms`
  - 약 **42.2% 감소**
- avg speedup: `13.51x` → `24.24x`
  - 약 **79.4% 증가**
- worst abs/rel error:
  - essentially unchanged at the previous quality level

즉,

> correctness를 유지한 채 latency가 유의미하게 내려갔다

#### 2. All-workload extraction stayed fully green

이번 iteration은 5개 subset만 빨라진 게 아니라,

- **100/100 PASS**

를 유지했다.

즉,

> 이 최적화는 subset overfit가 아니라, 전체 workload 집합에서도 적어도 안정성은 해치지 않았다

#### 3. Improvement direction matches the hypothesis

원래 hypothesis는:

- q/k 중복 global loads를 줄이면 좋아질 것이다

였고, 실제로 quick benchmark가 크게 개선되었다.

즉,

> “memory redundancy removal”이라는 가설이 실제 측정에서 지지되었다

---

## Detailed analysis: why it likely improved

## 1. The old kernel had obvious q/k global-load duplication

P0 구조에서는 동일한 `(token, head)`에 대해 모든 thread가 같은 q/k vector를 다시 읽었다.

각 thread는 `v_idx` 하나만 다르기 때문에:

- q는 head 전체 thread가 동일하게 사용
- k도 head 전체 thread가 동일하게 사용

그런데 P0에서는 각 thread가 매 loop에서 q/k를 직접 global memory에서 읽고 있었다.

이건 사실상:

> **CTA 내부에서 공유 가능한 read-only vector를 thread별로 중복 로드**

하는 구조였다.

## 2. Shared staging turns duplicated global traffic into one cooperative load

P1에서는:

- 128 threads가 q/k 각 한 원소씩만 로드
- 이후 모든 thread가 shared q/k를 재사용

즉, 동일한 token/head 기준으로:

- q global reads
- k global reads

가 매우 크게 줄었다.

특히 현재 kernel은 token loop 안에서 q/k를 여러 계산 단계에서 재사용하므로:

1. old_v dot
2. state update
3. output dot

각 단계마다 q/k를 다시 읽던 것을,
한 번 staged해서 재사용하게 만든 것이 핵심이다.

## 3. The extra shared-memory cost was small relative to the existing state tile

이번 변경으로 증가한 shared memory는:

- `q_sh[128]`
- `k_sh[128]`

즉 대략 1KB 수준이다.

하지만 원래 baseline은 이미:

- `state_sh[128][128]` fp32

를 쓰고 있었기 때문에, shared memory footprint의 절대 증가는 상대적으로 작다.

즉,

> occupancy를 크게 더 깎아먹지 않으면서,
> 중복 global loads는 많이 줄이는 방향이었다

이 점이 이번 최적화가 “low risk, high return” 이었던 이유다.

## 4. Fixed-128 unroll likely helped the compiler schedule better

background는 `d128` 고정 최적화를 강하게 권장한다.

이번 iteration에서 `#pragma unroll`을 넣은 것은:

- head_size가 항상 128이라는 사실을 컴파일러에 더 강하게 반영
- dot/update loop scheduling에 도움

을 기대한 것이다.

shared staging이 주효과라면, unroll은 그 위에 얹힌 보조 이득일 가능성이 높다.

## 5. Why long/medium workloads also benefited

긴 sequence일수록 token loop 반복 횟수가 많다.

즉, q/k 중복 load 비용도 누적된다.

따라서:

- 짧은 workload는 launch/constant overhead에 가까운 이득
- 긴 workload는 token 반복에 따라 누적 이득

이 나타나는 것이 자연스럽다.

실제로 100-workload 결과에서도:

- 중간/긴 workload에서 높은 speedup이 다수 보였다

---

## Detailed analysis: why it is still far from the likely ceiling

이번 iteration이 좋아졌다고 해도, 아직 ceiling과는 거리가 있다.

### 1. State tile is still huge and row-wise scalarized

현재 kernel은 여전히:

- full fp32 state tile
- row-wise scalar loops
- thread당 128-element row update

구조다.

즉,

> q/k redundancy는 줄였지만, state update 자체는 아직 매우 naive하다

### 2. We are still not exploiting deeper Blackwell-specific features

background는 다음을 시사한다:

- CUDA 13.x / CUDA Tile
- deeper fusion
- better memory tiering
- Blackwell-specific optimization

하지만 현재 구현은 아직:

- correctness-first shared-memory baseline

일 뿐이다.

즉,

> 이번 P1은 “첫 obvious bottleneck 제거” 단계이지,
> Blackwell-specific tuning 단계는 아니다

### 3. Upstream reference still suggests large headroom

background에 적어둔 upstream prefill baseline 수치 기준으로 보면,

- upstream avg latency: `0.317 ms`

라는 reference가 있다.

현재 우리의 100-workload quick avg latency는:

- `5.051 ms`

이다.

정확히 apples-to-apples라고 단정할 수는 없지만,
대략적인 메시지는 분명하다:

> 아직 큰 최적화 여지가 남아 있다

---

## Decision

이번 P1 iteration은 **commit 대상**이다.

이유:

1. quick working check 통과
2. full workload extraction 통과
3. quick baseline 기준 수치 개선이 명확
4. correctness quality는 유지됨
5. 개선 이유가 설명 가능함

---

## Next iteration recommendation

다음 P2/P3 후보는:

1. state update loop의 shared-memory / register reuse 개선
2. CTA mapping 재설계
3. 긴 workload tail targeting

가장 자연스러운 다음 가설은:

> “현재 full state tile 방식이 occupancy를 너무 잡아먹고 있으므로,
> state update 타일링을 더 잘게 나누거나 register/shared 분담을 개선하면 추가 이득이 날 것”

