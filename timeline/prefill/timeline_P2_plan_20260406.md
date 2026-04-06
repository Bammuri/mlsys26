# Prefill Timeline — P2 Plan — 2026-04-06

## Stage goal

P2의 목표는 **row-pair-per-warp** 구조로 q/k/gate reuse를 늘리는 것이다.

이번 iteration의 핵심 질문:

> 현재 SRTP v1은 warp 1개가 row 1개만 처리한다.  
> warp 1개가 row 2개를 연속 처리하게 바꾸면,  
> q/k load와 gate broadcast를 더 잘 재사용하면서 full 100 평균 latency를 더 낮출 수 있을까?

---

## Why this optimization was chosen

### 1. P1이 이미 shared/spill 병목을 제거했다

P1 결과:

- full avg latency: `0.517 ms` (current baseline, `warmup_runs=1, iterations=5, num_trials=2`)
- tiny NCU:
  - main kernel duration: `11.81 us`
  - registers/thread: `32`
  - dynamic shared memory: `0`
  - spill requests: `0`

즉 P1 이후에는:

> shared-state tile 제거와 register-spill 제거는 이미 성공했다.

### 2. 남아 있는 자연스러운 다음 병목

현재 kernel은 row-per-warp라서,
같은 warp가 token마다 다음 데이터를 row 하나를 위해 읽는다:

- q fragment
- k fragment
- gate / beta broadcast

이 데이터들은 warp가 같은 token 안에서 **row 두 개 이상** 처리해도 그대로 재사용 가능하다.

즉 다음 후보는:

> CTA/warp 구조는 유지하되, warp당 work를 조금 더 늘려 load reuse와 per-row overhead를 줄이는 것

이다.

### 3. 왜 이게 low-risk인가

이번 변경은:

- P1의 launch philosophy 유지
- no shared-state 유지
- no barrier 유지
- 수학 동일

즉 새로운 structural family로 갈아타는 게 아니라,
**P1 위에서 warp work granularity만 조정하는 실험**이다.

---

## Planned code change

수정 대상:

- `solution/cuda/kernel.cu`

변경 포인트:

1. warp 1개가 row 1개 대신 row 2개를 담당
2. `kRowsPerBlock`을 4 → 8 rows/block으로 변경
3. q/k fragment load는 1회, row update/output은 2회 수행
4. state fragment를 row 2개 분량 register에 유지

핵심 의도:

> q/k load와 gate/beta broadcast를 더 많이 재사용하면서,  
> 여전히 no-shared / no-barrier / register-owned dataflow를 유지하기

---

## Expected result before testing

현재 baseline:

- full decision gate avg latency: `0.517 ms`

예상:

1. quick 5는 PASS 유지
2. full 100도 PASS 유지
3. throughput-heavy / long-tail에서 small positive gain 가능
4. tiny에서는 grid.x가 절반으로 줄어드는 만큼 약간 손해일 수도 있음
5. 최종 판단은 **full 100 arithmetic-mean latency** 기준으로만 한다

---

## Test plan

### 1. pack

```bash
python scripts/pack_solution.py
```

### 2. quick gate

```bash
modal run scripts/run_modal.py --quick --max-workloads 5
```

### 3. full decision gate

```bash
modal run scripts/run_modal.py --decision-gate --max-workloads 100
```

주의:

- decision gate config는 사용자 지시에 따라
  - `warmup_runs=1`
  - `iterations=5`
  - `num_trials=2`

### 4. post-change tiny NCU

```bash
modal run scripts/run_modal_ncu.py --workload-uuid 77daf91d --kernel-name gdn_prefill_kernel
modal run scripts/run_modal_ncu.py --workload-uuid 77daf91d --kernel-name compute_gate_beta_kernel
```
