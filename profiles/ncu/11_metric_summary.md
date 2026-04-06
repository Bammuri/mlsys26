# NCU Metric Summary

작성일: 2026-04-06

이 문서는 representative workload들의 NCU 결과를 **한 곳에 요약 비교**하는 문서다.

현재 상태:

- `77daf91d`, `ba08a83e`, `5835a2bc`, `07aa7922` 모두 **Modal에서 실제 capture 완료**

primary set:

1. `77daf91d` — tiny
2. `ba08a83e` — single-seq medium
3. `5835a2bc` — throughput-heavy
4. `07aa7922` — long-tail

---

## 1. Summary Table — `compute_gate_beta_kernel`

| Workload | Class | Duration | Regs/thread | Occupancy | Notes |
|---|---|---:|---:|---:|---|
| `77daf91d` | tiny | `5.63 us` | `16` | `6.97%` | Modal capture complete; grid size `1`, strong underfill |
| `ba08a83e` | single-seq medium | `6.53 us` | `16` | `11.07%` | grid size `5`, still underfilled |
| `5835a2bc` | throughput-heavy | `6.30 us` | `16` | `20.34%` | grid size `256`, precompute well-covered relative to other cases |
| `07aa7922` | long-tail | `6.78 us` | `16` | `14.14%` | grid size `179`, still under theoretical occupancy |

---

## 2. Summary Table — `gdn_prefill_kernel`

| Workload | Class | Duration | Regs/thread | Achv. Occ. | Active warps/sched | Eligible warps/sched | Dyn. SMEM | Barrier stall | Long scoreboard | Load bytes/sector | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---|---|---|---|
| `77daf91d` | tiny | `78.24 us` | `255` | `6.26%` | n/a in current extract | n/a in current extract | `66.56 KB` | strong signal via `83%` excessive shared wavefronts | not yet isolated from current page | strong issue via `49%` excessive sectors | grid size `8`, waves/SM `0.03`, spills `7552` |
| `ba08a83e` | single-seq medium | `1.26 ms` | `255` | `6.24%` | n/a | n/a | `66.56 KB` | strong signal via `82%` excessive shared wavefronts | not yet isolated from current page | `32%` excessive sectors | grid size `8`, waves/SM `0.03`, spills `163200` |
| `5835a2bc` | throughput-heavy | `44.76 ms` | `255` | `8.39%` | n/a | n/a | `66.56 KB` | strong signal via `82%` excessive shared wavefronts | not yet isolated from current page | `24%` excessive sectors | grid size `256`, waves/SM `0.86`, spills `9969664` |
| `07aa7922` | long-tail | `52.14 ms` | `255` | `6.25%` | n/a | n/a | `66.56 KB` | strong signal via `82%` excessive shared wavefronts | not yet isolated from current page | `4%` excessive sectors | grid size `16`, waves/SM `0.05`, spills `6942656` |

---

## 3. Cross-Workload Read

### A. launch-bound signal

- tiny workload observation:
  - capture succeeded on Modal
  - grid is drastically too small (`1` block for precompute, `8` blocks for main kernel)
  - main kernel is also constrained by registers + shared memory
- implication:
  - tiny case already supports the SRTP motivation: more CTAs and less shared-state coupling

### B. single-seq signal

- single-seq medium observation:
  - duration jumps from `78.24 us` (tiny) to `1.26 ms`
  - grid shape is still only `8` blocks, so launch underfill remains essentially unchanged
  - shared-wavefront excess stays at `82%`
- implication:
  - when sequence grows but grid does not, the current kernel scales mainly by token-loop work while preserving the same structural inefficiencies

### C. throughput signal

- throughput-heavy observation:
  - precompute remains cheap (`6.30 us`)
  - main kernel reaches `44.76 ms`
  - grid coverage improves (`256` blocks, waves/SM `0.86`) and achieved occupancy rises to `8.39%`
  - but shared-memory and register limits still cap theoretical occupancy at `12.5%`
- implication:
  - extra CTAs help, but the kernel still carries a heavy per-block structural cost from shared-state + registers

### D. long-tail signal

- long-tail observation:
  - precompute remains cheap (`6.78 us`)
  - main kernel is `52.14 ms`, worse than throughput-heavy despite fewer sequences
  - grid is only `16` blocks, waves/SM drops to `0.05`
  - global excessive sectors fall to `4%`, but shared-wavefront excess stays `82%`
- implication:
  - long-tail cost is much more about token-loop/state-update structure than q/k global load efficiency

---

## 4. Structural Conclusions

- Is precompute kernel cost material?
  - it is consistently secondary across the set (`5.63`–`6.78 us`) and does not scale like the main kernel
- Is full shared-state structure the dominant issue?
  - strong yes signal across all four workloads
- Is warp starvation clearly visible?
  - yes in tiny/single-seq/long-tail; partially relieved but not eliminated in throughput-heavy
- Is load efficiency secondary or primary?
  - global load inefficiency exists, but shared-structure and launch shape dominate more consistently
- Which SRTP design choice is most justified now?
  - prioritize `(seq, head, v_tile)` mapping plus register-tiling immediately; shared-state residency should be the first structural target

---

## 5. Design Impact on SRTP

### keep

- [x] register-tiled rows
- [x] `(seq, head, v_tile)` mapping
- [x] more warps/block
- [x] vectorized q/k load path

### reconsider

- [ ] gate/beta precompute
- [x] shared q/k staging
- [x] full shared-state tile

---

## 6. Follow-up Actions

1. Move from NCU capture to SRTP prototype implementation
2. Re-run the same four workloads after the first SRTP branch
3. Compare whether shared-wavefront excess, spill count, and waves/SM actually improve
