# Prefill Timeline — P0 Restart — 2026-04-06

이번 P0에서는 이전 실험 기록을 이어가지 않고,
문서와 계획 흐름을 모두 초기화한 뒤 새 알고리즘 방향으로 다시 시작했다.

## 결정

- 이전 역사 문서는 제거
- `tomas_reference`는 원리 참고용으로만 사용
- 다음 알고리즘은 **Streaming Register-Tiled Prefill (SRTP)** 로 설정

## 새 핵심 질문

> full shared-state CTA 구조를 버리고,
> `(seq, head, v_tile)` 기반 register-tiled prefill로 바꾸면
> prefill long-sequence에서 더 나은 latency hiding을 얻을 수 있을까?

## 다음 단계

1. `solution/cuda/kernel.cu` 기준으로 SRTP prototype 설계
2. warp ownership / v_tile 크기 초안 확정
3. quick gate와 full decision gate로 첫 keep/revert 판단
