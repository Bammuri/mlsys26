# Representative Workload Selection for NCU

작성일: 2026-04-06

## 1. 목적

NCU는 모든 workload를 처음부터 다 찍는 용도가 아니라,
**구조적 병목을 빨리 드러내는 대표 workload 묶음**을 잡는 용도다.

이번 선택은 `scripts/profile_ncu_prefill.py --list-workloads`와
dataset axes(`total_seq_len`, `num_seqs`, `len_cu_seqlens`)를 기준으로 했다.

## 2. 선택 기준

첫 pass는 아래 네 클래스를 반드시 포함한다.

1. tiny launch-bound
2. single-seq medium
3. throughput-heavy multi-seq
4. long-tail seq-heavy

이 네 클래스가 있어야 다음 질문에 답할 수 있다.

- launch/driver 고정비가 큰가?
- single-seq에서 CTA 공급량이 부족한가?
- high-throughput에서 shared footprint가 문제인가?
- long-tail sequence에서 barrier/state streaming 비용이 커지는가?

## 3. Chosen Primary Set

### A. tiny launch-bound

- UUID: `77daf91d-0660-4c4b-8c32-336a69281cd9`
- axes:
  - `total_seq_len = 6`
  - `num_seqs = 1`

선정 이유:

> 가장 작은 축에 가까운 workload라 launch/dispatch overhead와 low-CTA shape를 보기 좋다.

---

### B. single-seq medium

- UUID: `ba08a83e-e151-4e16-bc70-abee6851604c`
- axes:
  - `total_seq_len = 134`
  - `num_seqs = 1`

선정 이유:

> single-seq 구조는 유지하면서 tiny보다 실제 kernel body가 충분히 보이는 대표 사례다.

비슷한 대체 후보:

- `c7846f96-c9ff-44d3-94ab-5eab99a1431b`

---

### C. throughput-heavy multi-seq

- UUID: `5835a2bc-8d60-43fc-b1ed-d4729ea62693`
- axes:
  - `total_seq_len = 8192`
  - `num_seqs = 32`

선정 이유:

> total이 크고 seq 수도 커서, shared-memory footprint / scheduler pressure / steady-state behavior를 보기 좋다.

비슷한 대체 후보:

- `d0ce7b5d-49e2-4a0b-b2ce-a087139b7d6b`
- `410794d4-6f70-4bb4-aed1-4669be9a610f`

---

### D. long-tail seq-heavy

- UUID: `07aa7922-1848-48a9-830a-54216b5553b3`
- axes:
  - `total_seq_len = 5709`
  - `num_seqs = 2`
  - avg seq len ≈ `2854.5`

선정 이유:

> sequence가 매우 길고 seq 수는 적어, prefill의 token-loop cost와 state streaming cost를 가장 직접적으로 드러낸다.

## 4. Secondary Support Set

필요하면 아래 두 workload를 보조로 추가한다.

### E. medium multi-seq

- UUID: `4b6143dd-0e5f-499f-93cb-076d9635bcd0`
- axes:
  - `total_seq_len = 4124`
  - `num_seqs = 15`

용도:

> single-seq와 32+ seq throughput-heavy 사이의 중간 구간을 메우는 완충 사례

### F. high-fanout multi-seq extreme

- UUID: `9a5d694b-7d4c-4ee6-8315-a13053ab6f92`
- axes:
  - `total_seq_len = 8192`
  - `num_seqs = 57`

용도:

> CTA 공급량은 많지만 per-seq 평균 길이는 짧은 extreme case

## 5. First Capture Order

첫 NCU pass 권장 순서:

1. `77daf91d...`
2. `ba08a83e...`
3. `5835a2bc...`
4. `07aa7922...`

이 순서는:

- launch-bound
- single-seq body
- throughput-heavy
- long-tail

순으로 해석 난이도를 올리는 방식이다.

## 6. Example Commands

### Workload list 확인

```bash
python scripts/profile_ncu_prefill.py --list-workloads --list-limit 10
```

### tiny

```bash
python scripts/profile_ncu_prefill.py --workload-uuid 77daf91d
```

### single-seq medium

```bash
python scripts/profile_ncu_prefill.py --workload-uuid ba08a83e
```

### throughput-heavy

```bash
python scripts/profile_ncu_prefill.py --workload-uuid 5835a2bc
```

### long-tail

```bash
python scripts/profile_ncu_prefill.py --workload-uuid 07aa7922
```

## 7. 결론

초기 NCU representative set은 다음 네 개로 고정한다.

1. `77daf91d`
2. `ba08a83e`
3. `5835a2bc`
4. `07aa7922`

이 네 개면 SRTP가 겨냥하는
**launch / warp starvation / shared footprint / long-tail token-loop cost**
를 충분히 점검할 수 있다.
