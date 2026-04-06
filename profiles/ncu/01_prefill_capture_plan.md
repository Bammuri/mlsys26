# Prefill NCU Capture Plan

작성일: 2026-04-06

## 1. 현재 상태

로컬 환경은 현재 NCU capture에 적합하지 않다.

- local: `ncu` 없음
- local torch/CUDA stack도 MX150(`sm_61`) 때문에 profiling target으로 부적합

사용자 지시에 따라 **Modal GPU 환경을 실제 capture 우선 경로**로 사용한다.  
따라서 이 문서는 Modal-based profiling plan 기준으로 읽는다.

## 2. 첫 번째 측정 대상

현재 커널 기준으로 먼저 분리 측정할 대상은 2개다.

1. `compute_gate_beta_kernel`
   - `solution/cuda/kernel.cu:51-71`
2. `gdn_prefill_kernel`
   - `solution/cuda/kernel.cu:74-179`

이 순서가 중요한 이유:

- precompute kernel이 실제로 의미 있는 시간을 먹는지
- main kernel의 진짜 병목이 shared/barrier/warp starvation인지

를 분리해서 보기 위해서다.

## 3. 첫 번째로 볼 metric

reference 분석을 바탕으로, 첫 pass에서는 아래 metric만 우선 본다.

### 공통

1. Duration
2. Achieved occupancy
3. Active warps / scheduler
4. Eligible warps / scheduler
5. Registers / thread

### main kernel에 특히 중요

6. Dynamic shared memory usage
7. Barrier-related stall
8. Long scoreboard stall
9. Shared memory bank conflicts
10. Global load bytes/sector 또는 load efficiency

## 4. 해석 규칙

### Case A — barrier/shared signal이 강함

다음이 동시에 보이면:

- dyn smem 큼
- barrier stall 큼
- eligible warps 낮음

해석:

> 현재 커널의 중심 병목은 full shared-state 구조일 가능성이 높다.

다음 액션:

- SRTP 우선순위 상승
- shared-state 유지형 미세조정 우선순위 하락

### Case B — warp starvation signal이 강함

다음이 보이면:

- active warps / scheduler 낮음
- eligible warps 낮음
- occupancy 낮음
- waves/SM도 낮게 관찰됨

해석:

> `(seq, head)` 중심 grid가 너무 작거나 block 내부 warp 수가 부족할 수 있다.

다음 액션:

- `(seq, head, v_tile)` CTA mapping 실험
- block당 warp 수 4~8 후보 시험

### Case C — q/k load efficiency가 나쁨

다음이 보이면:

- bytes/sector 낮음
- excessive sectors 높음

해석:

> q/k path에 vectorized load 정리가 필요하다.

다음 액션:

- `uint2`/`bfloat162` 기반 q/k load 경로 점검
- 단, 구조 변경보다 우선순위가 높지는 않음

### Case D — precompute kernel 비중이 큼

`compute_gate_beta_kernel` duration이 무시 못 할 수준이면:

해석:

> gate/beta 처리 방식이 다시 중요한 후보가 될 수 있다.

다음 액션:

- precompute 유지 vs reintegration을 profiling evidence로 재평가

## 5. Workload selection rule

리셋 직후 첫 profiling pass에서는 아래 representative set을 사용한다.

1. tiny launch-bound
   - `77daf91d-0660-4c4b-8c32-336a69281cd9`
2. single-seq medium
   - `ba08a83e-e151-4e16-bc70-abee6851604c`
3. throughput-heavy multi-seq
   - `5835a2bc-8d60-43fc-b1ed-d4729ea62693`
4. long-tail seq-heavy
   - `07aa7922-1848-48a9-830a-54216b5553b3`

선정 근거와 보조 후보는 `profiles/ncu/02_workload_selection.md`에 정리한다.

## 6. 실행 경로

실행 경로는 두 갈래다.

### A. Modal 경로

우선 경로는 `scripts/run_modal_ncu.py`다.

환경 확인:

```bash
modal run scripts/run_modal_ncu.py --check-env
```

실제 capture:

```bash
modal run scripts/run_modal_ncu.py --workload-uuid 77daf91d
```

### B. raw `ncu` 경로

Modal 경로가 막힐 때만, 별도 NCU host에서 repo 전용 harness `scripts/profile_ncu_prefill.py`로 아래 형태로 실행한다.

```bash
ncu --set detailed \
  --kernel-name regex:gdn_.* \
  --launch-skip 3 --launch-count 1 \
  python -m scripts.profile_ncu_prefill
```

## 7. 이 디렉터리에서 다음에 추가할 것

host가 준비되면 아래 파일을 순서대로 추가한다.

1. `profiles/ncu/10_baseline_capture.md`
2. `profiles/ncu/11_metric_summary.md`
3. `profiles/ncu/*.ncu-rep` 또는 export 결과
4. `scripts/run_modal_ncu.py`
5. 필요하면 `scripts/profile_ncu_prefill.py`

## 8. 지금 단계의 결론

지금은 actual capture보다도,
reference를 통해 **무엇을 측정해야 SRTP를 정당화할 수 있는지**를 먼저 고정하는 단계다.

이 문서 기준 첫 profiling 질문은 한 줄로 요약된다.

> 현재 prefill kernel의 진짜 병목은 gate math가 아니라  
> **shared-memory footprint / barrier cost / warp starvation** 인가?
