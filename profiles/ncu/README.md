# NCU Profiling Workspace

작성일: 2026-04-06

이 디렉터리는 GDN prefill용 **NCU profiling 작업 공간**이다.

현재 상태:

- 이 환경에서는 `ncu` 바이너리가 보이지 않는다 (`which ncu` 결과 없음)
- 사용자 지시에 따라, 실제 capture 경로는 **local 대신 Modal 우선**으로 전환한다
- 직접 probe 결과, Modal B200 컨테이너에서는 `ncu` 사용이 가능했다
  - env probe run: `ap-aMPQKAQt499ZGBZHCrUDSN`
- representative set 4개에 대한 Modal baseline capture를 완료했다
  - summary: `profiles/ncu/11_metric_summary.md`

따라서 지금 단계에서는:

1. `tomas_reference` 기반 **reference analysis**
2. 현재 prefill 커널에 맞는 **capture plan**
3. 나중에 NCU host가 준비되면 바로 실행할 **runbook**

만 먼저 정리한다.

## 파일

1. `00_reference_analysis.md`
   - `tomas_reference`에서 배울 점 / 가져오면 안 되는 점
2. `01_prefill_capture_plan.md`
   - 현재 prefill 커널에 대해 무엇을 어떤 순서로 측정할지
3. `02_workload_selection.md`
   - 대표 NCU capture 대상 UUID 선정
4. `10_baseline_capture.md`
   - 실제 baseline NCU 기록용 템플릿
5. `11_metric_summary.md`
   - workload 간 metric 비교 요약 템플릿
6. `../../scripts/profile_ncu_prefill.py`
   - raw `ncu` 실행용 repo-native harness
7. `../../scripts/run_modal_ncu.py`
   - Modal GPU 컨테이너에서 `ncu`를 실행하는 remote harness

## 원칙

- reference는 **원리만 참고**
- decode 전용 kernel shape는 복제하지 않음
- profiling 결과는 새 알고리즘 `SRTP` 설계 검증용으로만 사용
- 실제 baseline capture는 local보다 **Modal 경로를 우선 시도**한다
