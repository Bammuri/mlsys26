# P0 Restart Plan — Original Prefill Algorithm

작성일: 2026-04-06

## 1. Reset Intent

이번 P0의 목표는 과거 문서 흐름을 이어 붙이는 것이 아니라,
**현재 코드 상태만 다시 읽고 새 알고리즘으로 재출발하는 것**이다.

사용자 지시:

> `tomas_reference`를 참고하되 완전히 베끼지 말고,
> 거기서 얻은 정보를 바탕으로 새로운 알고리즘으로 진행한다.

즉 이번 리셋의 핵심은:

1. reference의 성능 원리는 배운다
2. 하지만 decode 전용 구조나 launch mapping을 그대로 복제하지 않는다
3. prefill의 긴 sequence loop에 맞는 **독자적인 kernel shape**를 설계한다

## 2. Current Code Re-observation

현재 출발점은 오직 현 커널이다.

- gate/beta precompute kernel 존재
  - `solution/cuda/kernel.cu:51-71`
- main kernel이 full `128 x 128` state tile을 shared memory에 올림
  - `solution/cuda/kernel.cu:95-103`
- token loop 안에서 barrier가 반복됨
  - `solution/cuda/kernel.cu:118`, `133`, `140`, `171`
- CTA mapping은 사실상 `(seq, head)` 중심이고 V-tile 병렬화가 없음
  - `solution/cuda/kernel.cu:87-89`, `274-275`

따라서 현재 커널의 큰 문제 후보는 여전히 다음 셋이다.

1. shared-memory footprint가 너무 큼
2. barrier frequency가 높음
3. small/mid workload에서 CTA 공급량이 적을 수 있음

## 3. What We Take From `tomas_reference`

우리가 가져올 것은 **구체 코드가 아니라 원리**다.

### 가져올 원리

1. state를 가능한 한 register 가까이에 두려는 방향
2. V-axis 또는 row-axis 병렬화를 통해 CTA 수를 늘리는 방향
3. K-last/vectorized IO를 적극 활용하는 방향
4. math 미세조정보다 latency hiding과 구조 단순화를 먼저 보는 방향

### 가져오지 않을 것

1. decode 전용 T<=4 가정
2. reference의 kernel shape/launch shape 복제
3. gate math in-kernel fusion을 무조건 정답으로 보는 태도
4. BF16-state 중심 사고방식

특히 prefill에서는 긴 sequence loop가 핵심이므로,
reference의 decode 최적화는 **아이디어의 출처**일 뿐 정답 템플릿이 아니다.

## 4. New Algorithm Direction

이번 P0에서 새 메인 가설로 채택하는 방향은 다음이다.

# Streaming Register-Tiled Prefill (SRTP)

핵심 아이디어:

> 한 CTA가 head 전체 128개 row를 shared에 전부 올리는 대신,
> `(seq, head, v-tile)` 단위로 더 잘게 나누고,
> 각 warp가 자기 V-row(또는 소수 row)의 state를 register 중심으로 오래 들고 가면서
> sequence 전체를 stream 처리한다.

이 설계는 reference와 비슷한 문제의식을 가지지만,
아래 이유로 **동일 알고리즘이 아니다**.

1. decode가 아니라 **prefill long-sequence**를 직접 겨냥한다
2. state를 token마다 다시 구성하지 않고 **sequence streaming**으로 유지한다
3. gate/beta는 초기 버전에서 **외부 precompute 유지 가능**으로 둔다
4. shared memory는 full state tile이 아니라 **현재 token의 q/k staging 정도로만 축소**한다

## 5. Proposed Mapping (P0 draft)

초기 draft mapping은 아래를 기준으로 잡는다.

### Grid

- `grid.y = num_seqs`
- `grid.x = kNumVHeads * num_v_tiles_per_head`

즉 `(seq, head)`당 CTA 1개가 아니라,
`(seq, head, v_tile)`당 CTA 여러 개가 들어가도록 바꾼다.

### Block / Warp ownership

- 블록은 4~8 warps 후보로 시작
- warp 하나가 V-row 1개 또는 2개를 책임진다
- 각 warp는 자기 row의 K=128 성분을 register에 유지한다

### Per-token flow

1. q/k를 warp-cooperative 또는 tiny shared staging으로 로드
2. gate/beta를 head 기준으로 읽어 broadcast
3. 각 warp가 자기 row에 대해 `k · row` 계산
4. 각 warp가 자기 row를 update
5. 각 warp가 자기 output row를 계산 후 저장
6. sequence 끝에서 state writeback

## 6. Why This Is a Better P0 Than Small Tweaks

이 방향이 지금 더 맞는 이유:

1. 현재 병목 가설(shared footprint / barrier / CTA granularity)을 직접 건드린다
2. reference에서 배운 “parallelism + latency hiding” 원리를 쓰되 복제는 피한다
3. prefill의 본질인 긴 token loop를 설계 중심에 둔다
4. 실패하더라도 다음 variant 분기가 명확하다
   - warp당 1 row
   - warp당 2 rows
   - q/k shared staging 유무
   - gate/beta precompute 유지 vs 재통합

## 7. P0 Acceptance Criteria

P0 단계에서 바로 요구하는 것은 완성 코드가 아니라,
**새 알고리즘 브랜치의 시작 조건을 명확히 하는 것**이다.

완료 기준:

1. 새 알고리즘 이름과 핵심 매핑이 문서로 고정됨
2. reference에서 가져올 것 / 가져오지 않을 것이 분리됨
3. 첫 구현 순서가 정해짐

## 8. First Implementation Order After P0

### Step 1
기존 커널을 유지한 채, 새 브랜치에서
`(seq, head, v_tile)` CTA mapping prototype를 만든다.

### Step 2
full shared state tile을 없애고,
warp-owned register rows로 줄인다.

### Step 3
q/k staging을 tiny shared vs pure warp load 두 가지 중 더 단순한 쪽부터 시험한다.

### Step 4
quick gate → full decision gate로 keep/revert 판단한다.

## 9. Immediate Next Action

다음 액션은 문서가 아니라 구현 준비 기준으로는 이것이다.

> `solution/cuda/kernel.cu`를 새 알고리즘 브랜치 관점에서 다시 쪼개고,
> full-state-shared 구조 대신 **SRTP prototype**을 설계한다.

## 10. Parallel Support Track — NCU

구현과 병렬로, NCU support track은 `profiles/ncu/`에서 관리한다.

- `profiles/ncu/00_reference_analysis.md`
- `profiles/ncu/01_prefill_capture_plan.md`

이 track의 목적은 reference 복제가 아니라,
**SRTP가 겨냥하는 shared/barrier/warp 문제를 실제 metric으로 확인할 준비를 하는 것**이다.
