# Prefill Timeline — P2 Result — 2026-04-07 (B1)

## Optimization summary

이번 P2(B1) iteration에서는 current keep baseline(P6, `0.490 ms`) 위에서,
**kernel hot path는 그대로 두고 wrapper boundary의 unnecessary work를 줄이는 branch**를 시험했다.

### Applied optimization

1. 이미 `CHECK_CONTIGUOUS(...)`를 통과한 입력들에 대한 redundant `.contiguous()` 호출 제거
2. `has_state == true`일 때 `new_state`를 새로 allocate하지 않고 `state_in` storage를 그대로 재사용
3. main kernel의 row-per-warp / 2-warps-per-block / packed `float2` gate-beta path는 그대로 유지

핵심 의도:

> tomas가 decode에서 얻었던 “kernel 주변의 불필요한 work 제거” 원리를,
> prefill에서는 **hot path를 건드리지 않는 wrapper-boundary cleanup**으로 먼저 번역해보기.

---

## Expected result before testing

baseline:
- current keep baseline: `0.490 ms`

예상:
1. quick 5 PASS 유지
2. full 100 PASS 유지
3. gain이 있더라도 small-to-moderate gain일 가능성
4. improvement가 있다면 multi-seq / large-state-output workload에서 더 크게 나타날 가능성
5. aliasing contract가 잘못되면 runtime/numerical failure 가능성

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
- avg latency: `0.041 ms`
- worst abs error: `1.91e-06`
- worst rel error: `1.93e-02`

### 3. full decision gate

```bash
modal run scripts/run_modal.py --decision-gate --max-workloads 100
```

결과:
- `PASSED=100/100`
- avg latency: `0.484 ms`
- avg speedup: `420.86x`
- worst abs error: `1.22e-04`
- worst rel error: `2.97e-01`

판단 기준은 사용자 지시대로 **quick이 아니라 full 100 arithmetic-mean latency**다.

---

## Comparison against baseline

### Baseline → P2(B1)

- avg latency: `0.490 ms` → `0.484 ms`
- 절대 개선: `0.006 ms`
- 상대 개선: 약 **1.22%**

개선폭은 크지 않지만, full mean 기준으로는 분명한 positive change다.

---

## Representative workload comparison

- `77daf91d`: `0.031 ms` → `0.034 ms`
- `ba08a83e`: `0.085 ms` → `0.086 ms`
- `5835a2bc`: `2.100 ms` → `2.035 ms`
- `07aa7922`: `2.545 ms` → `2.579 ms`

즉 representative만 보면 mixed result다.
특히 long-tail 대표 `07aa7922`는 소폭 악화됐다.

그래서 이번 판단은 representative 한두 개가 아니라,
**100개 전체 family distribution**을 같이 봐야 한다.

---

## Family-level comparison

기존 `timeline/prefill/workload_catalog_20260407.csv` family 구분을 기준으로,
P6 baseline과 P2(B1) full log를 workload-by-workload로 비교했다.

| Family | Count | Baseline avg | P2(B1) avg | Delta | Interpretation |
|---|---:|---:|---:|---:|---|
| F1 tiny-single | 21 | 0.037 ms | 0.037 ms | `+0.001 ms` | 거의 no-op |
| F2 single-seq | 12 | 0.205 ms | 0.207 ms | `+0.002 ms` | 거의 no-op / 미세 악화 |
| F3 long-tail-low-seq | 9 | 0.985 ms | 0.998 ms | `+0.013 ms` | 소폭 악화 |
| F4 medium-multiseq | 42 | 0.206 ms | 0.210 ms | `+0.003 ms` | 미세 악화 |
| F5 throughput-heavy | 16 | 1.763 ms | 1.709 ms | `-0.055 ms` | **유의미한 개선** |

핵심:

> 이번 개선은 benchmark 전체를 고르게 좋게 만든 branch가 아니라,
> **full mean을 실제로 크게 끌어올리던 F5 throughput-heavy에 선택적으로 이득을 준 branch**다.

그리고 바로 그 이유 때문에,
전체 arithmetic mean도 `0.490 → 0.484 ms`로 내려갔다.

---

## Top movers

### Biggest improvements

- `cc310f94`: `1.952 → 1.827 ms` (`-0.125 ms`)
- `a9540651`: `2.060 → 1.941 ms` (`-0.119 ms`)
- `c2931c92`: `1.887 → 1.790 ms` (`-0.097 ms`)
- `5d3fc66a`: `2.151 → 2.061 ms` (`-0.090 ms`)
- `5b8a0e4b`: `2.174 → 2.086 ms` (`-0.088 ms`)
- `5835a2bc`: `2.100 → 2.035 ms` (`-0.065 ms`)

이 상위 개선 workload들은 모두 `F5 throughput-heavy`다.

### Biggest regressions

- `07aa7922`: `2.545 → 2.579 ms` (`+0.034 ms`)
- `15856e8c`: `1.241 → 1.269 ms` (`+0.028 ms`)
- `a01a3f93`: `0.971 → 0.995 ms` (`+0.024 ms`)
- `43bf9699`: `0.413 → 0.435 ms` (`+0.022 ms`)
- `54856fec`: `0.415 → 0.436 ms` (`+0.021 ms`)

---

## Detailed analysis: why it improved

### 1. This branch did not make the kernel math cheaper; it made the wrapper cheaper

이번 변화는 `gdn_prefill_kernel`의 token recurrence, launch shape, register dataflow를 건드리지 않았다.
즉 main kernel의 arithmetic structure는 그대로다.

따라서 이번 개선은:
- dot/update/output math가 빨라졌다기보다
- **kernel launch 주변의 tensor housekeeping이 줄었다**
고 보는 것이 맞다.

이 점에서 이번 iteration은 tomas의 다음 원리를 가장 직접적으로 반영한다.

> “The biggest wins came from eliminating unnecessary work around the kernel.”

### 2. The improvement lines up with `num_seqs`, not with token-loop depth

family별 평균 `num_seqs`와 expected `new_state` output footprint를 보면:

- F1 tiny-single: avg `num_seqs = 1.0`, avg `new_state ≈ 0.50 MB`
- F2 single-seq: avg `num_seqs = 1.0`, avg `new_state ≈ 0.50 MB`
- F3 long-tail-low-seq: avg `num_seqs = 2.67`, avg `new_state ≈ 1.33 MB`
- F4 medium-multiseq: avg `num_seqs = 3.07`, avg `new_state ≈ 1.54 MB`
- F5 throughput-heavy: avg `num_seqs = 38.19`, avg `new_state ≈ 19.09 MB`

즉 F5는 `new_state` output tensor 자체가 압도적으로 크다.

이번 branch는 `has_state == true`일 때 이 output allocation/storage boundary를 줄였기 때문에,
**num_seqs가 큰 F5에서 가장 강한 이득이 나타난 것**이 자연스럽다.

실제로 workload별 `num_seqs`와 latency improvement의 상관계수는 약 **`+0.65`**였다.

해석:

> 개선량은 token-loop depth보다도,
> **state output footprint가 큰 multi-seq workloads에서 더 강하게 나타났다.**

### 3. Why long-tail did not benefit much

all-workload NCU는 이미 F3 long-tail-low-seq의 병목이:
- 높은 L2 hit
- 낮은 occupancy
- 긴 recurrence depth

즉 **per-token recurrent cost** 쪽이라고 말해줬다.

이번 branch는 recurrence hot path를 거의 안 건드렸다.
그래서 long-tail은 큰 개선을 못 받았고,
대표 long-tail `07aa7922`는 오히려 약간 나빠졌다.

즉 이번 결과는 오히려 일관적이다.

> long-tail을 진짜 줄이려면 여전히 hot path / recurrence cost 쪽을 다시 봐야 하고,
> wrapper cleanup만으로는 충분하지 않다.

### 4. Why throughput-heavy improved enough to move the global mean

F5는 all-workload NCU 기준으로 이미 coverage가 꽤 좋았다.
즉 launch-only tuning 여지는 줄어든 상태였다.

이번 branch는 그 대신:
- large `new_state` output boundary
- redundant wrapper work

를 줄였다.

그리고 F5는 avg `num_seqs ≈ 38.2`라서 output-state boundary savings가 가장 크게 누적된다.

결과적으로:
- F1/F2/F3/F4의 small regression보다
- F5의 broad improvement가 더 크게 작용했고
- full arithmetic mean을 실제로 내릴 수 있었다.

### 5. Why this branch succeeded where B2-P1 failed

B2-P1은 precompute boundary를 없애는 대신 token hot path 안에 serial/sync section을 추가해서,
full mean을 지배하는 F5/F3를 동시에 망가뜨렸다.

반면 이번 branch는:
- hot path는 그대로 두고
- wrapper boundary만 줄였다.

즉,

> prefill에서는 “boundary removal”이 유효할 수 있지만,
> **그 위치가 token recurrence loop 안이면 안 되고,
> wrapper / allocation / aliasing boundary 쪽이면 유효할 수 있다**

는 중요한 교훈을 준다.

---

## NCU profiling note

이번 branch는 kernel hot path를 변경하지 않았기 때문에,
사용자와 합의한 운영 규칙대로 **추가 NCU는 생략**했다.

판단 근거:
1. `gdn_prefill_kernel` instruction/dataflow 자체는 동일
2. 이번 improvement signature는 kernel metric 변화보다도 **multi-seq output boundary 축소**와 더 잘 맞는다
3. all-workload NCU baseline은 이미 “F5는 hot path보다 주변 경계 비용 제거가 중요할 수 있다”는 해석을 뒷받침하고 있었다

즉 이번 keep는 **NCU로 kernel micro-metric이 좋아졌기 때문이 아니라,
full benchmark와 family-level deltas가 wrapper-boundary win을 설명하기 때문**이다.

---

## Judgement

이번 iteration은 **keep** 한다.

### Why this is a keep

1. `PASSED=100/100`
2. full avg latency가 `0.490 ms` → `0.484 ms`로 개선
3. improvement가 full mean을 실제로 지배하는 `F5 throughput-heavy`에서 집중적으로 나타남
4. kernel hot path를 불안정하게 바꾸지 않고 얻은 low-risk improvement

---

## Decision under workflow rule

- keep/revert 기준: **full workload avg latency arithmetic mean**
- 결과: **KEEP**

다음 단계:
1. commit
2. 다음 plan 생성
3. 이후에는 다시 long-tail(F3) recurrence cost를 건드리는 branch를 보되,
   **token-loop 안의 block-wide sync는 금지** 규칙을 유지

---

## Full pref log

full pref log is saved in:
- `timeline/prefill/perf_P2_B1_2604070848.txt`
