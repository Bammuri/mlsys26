# NCU Baseline Capture — Tiny Representative (`77daf91d`)

작성일: 2026-04-06

이 문서는 tiny representative workload `77daf91d`에 대해
**Modal 환경에서 실제 NCU baseline capture를 수행한 기록**이다.

local 환경은 `ncu` 부재와 CUDA stack mismatch로 부적합하므로,
이 baseline은 **Modal B200 + CUDA 13.0.2** 기준으로 잡는다.

---

## 1. Capture Metadata

- Date:
  - env probe: `2026-04-06T13:46:43+00:00`
  - main-kernel capture: `2026-04-06T13:48:35+00:00` 부근
  - precompute capture: `2026-04-06T13:51:49+00:00` 부근
- Operator: Codex
- Host machine: Modal container
- GPU: `NVIDIA B200`
- CUDA version:
  - toolkit/container: `13.0.2`
  - `nvcc`: `release 13.0, V13.0.88`
- Driver version: `580.95.05`
- PyTorch version: `2.11.0+cu130`
- `ncu --version`: `2025.3.1.0`
- Repo commit / worktree state:
  - HEAD: `0dc4f67`
  - branch: `decode_cuda`

---

## 2. Solution Under Test

- Definition: `gdn_prefill_qk4_v8_d128_k_last`
- Solution name: `my-team-solution-v1`
- Entry point: `binding.cpp::run`
- Kernel file:
  - `solution/cuda/kernel.cu`
- Build path:
  - PyTorch torch-extension build inside Modal container
- Notes:
  - `scripts/run_modal_ncu.py` used for remote capture
  - `scripts/profile_ncu_prefill.py` used as the in-container harness

---

## 3. Profiling Commands

### Environment probe

```bash
modal run scripts/run_modal_ncu.py --check-env
```

### Main kernel capture

```bash
modal run scripts/run_modal_ncu.py \
  --workload-uuid 77daf91d \
  --kernel-name gdn_prefill_kernel
```

### Precompute kernel capture

```bash
modal run scripts/run_modal_ncu.py \
  --workload-uuid 77daf91d \
  --kernel-name compute_gate_beta_kernel
```

### Report inspect

```bash
modal run scripts/run_modal_ncu.py \
  --inspect-report prefill-77daf91d-20260406-134835 \
  --page details

modal run scripts/run_modal_ncu.py \
  --inspect-report compute_gate_beta_kernel-77daf91d-20260406-135149 \
  --page details
```

---

## 4. Workload

- UUID: `77daf91d-0660-4c4b-8c32-336a69281cd9`
- Class: tiny launch-bound
- Axes:
  - `total_seq_len = 6`
  - `num_seqs = 1`
  - `len_cu_seqlens = 2`
- Why this workload matters:
  - the smallest representative workload
  - ideal for exposing launch/grid underfill and fixed overhead

---

## 5. Kernels Captured

이번 baseline은 두 커널을 **분리 capture** 했다.

1. `compute_gate_beta_kernel`
   - report: `/artifacts/compute_gate_beta_kernel-77daf91d-20260406-135149.ncu-rep`
2. `gdn_prefill_kernel`
   - report: `/artifacts/prefill-77daf91d-20260406-134835.ncu-rep`

Modal artifact volume:

- volume name: `mlsys26-ncu`

---

## 6. Quick Readout

### `compute_gate_beta_kernel`

- Duration: `5.63 us`
- Grid / Block:
  - grid size = `1`
  - block size = `256`
- Registers / thread: `16`
- Dynamic shared memory / block: `0 B`
- Theoretical occupancy: `100%`
- Achieved occupancy: `6.97%`
- Achieved active warps / SM: `4.46`
- Memory throughput: `1.09 GB/s`
- Local spill requests: `0`
- Primary NCU message:
  - **grid too small** (`1` block on `148` SMs)

### `gdn_prefill_kernel`

- Duration: `78.24 us`
- Grid / Block:
  - grid size = `8`
  - block size = `128`
- Registers / thread: `255`
- Dynamic shared memory / block: `66.56 KB`
- Theoretical occupancy: `12.50%`
- Achieved occupancy: `6.26%`
- Achieved active warps / SM: `4.01`
- Waves / SM: `0.03`
- Memory throughput: `7.27 GB/s`
- L1/TEX hit rate: `76.14%`
- L2 hit rate: `54.51%`
- Local spill requests: `7552`
- Shared-memory configuration size: `167.94 KB`
- Primary NCU messages:
  - grid too small (`8` blocks on `148` SMs)
  - theoretical occupancy limited by **registers + shared memory**
  - uncoalesced global accesses: `32768` excessive sectors (`49%`)
  - uncoalesced shared accesses: `401408` excessive wavefronts (`83%`)

---

## 7. Interpretation

### A. precompute kernel cost

Observation:

- precompute kernel duration is only `5.63 us`
- main kernel duration is `78.24 us`

Implication:

> tiny case에서도 precompute가 완전히 공짜는 아니지만,  
> 현재 baseline에서 main kernel이 훨씬 더 큰 병목이다.

### B. main kernel shared-heavy signal

Observation:

- dynamic shared memory per block = `66.56 KB`
- shared memory also appears in occupancy limiting factors
- shared accesses show `83%` excessive wavefronts

Implication:

> 현재 main kernel은 **강한 shared-memory 구조 비용**을 갖고 있다.

### C. warp/grid starvation signal

Observation:

- grid size = `8`
- waves / SM = `0.03`
- NCU가 grid too small 경고를 직접 출력

Implication:

> tiny workload에서는 `(seq, head)` 중심 launch shape가 극단적으로 underfill 상태다.

### D. register pressure signal

Observation:

- `gdn_prefill_kernel` registers/thread = `255`
- local spill requests = `7552`

Implication:

> 현재 커널은 register pressure도 높고 spill까지 발생한다.  
> shared-heavy 구조만이 아니라 row ownership / per-thread state shape도 재검토할 가치가 크다.

### E. load efficiency signal

Observation:

- uncoalesced global accesses `49%`
- uncoalesced shared accesses `83%`

Implication:

> vectorized q/k path와 shared layout 둘 다 다시 봐야 하지만,  
> 우선순위는 여전히 **launch shape + shared footprint** 쪽이 더 크다.

---

## 8. Decision

- Baseline valid?: yes — Modal에서 실제 NCU capture 성공
- Tiny-case dominant bottlenecks:
  1. **grid underfill / warp starvation**
  2. **shared-memory footprint**
  3. **register pressure + spill**
  4. **global/shared uncoalesced access**
- What this says about SRTP:
  - SRTP 방향은 더 강하게 정당화된다
  - 특히 아래 두 가설이 baseline evidence를 얻었다:
    1. `(seq, head, v_tile)`로 CTA 수를 늘려야 한다
    2. full shared-state tile을 줄이고 register-owned row/tile 구조로 옮겨야 한다

---

## 9. Raw Artifacts

- env probe run:
  - Modal app: `ap-aMPQKAQt499ZGBZHCrUDSN`
- main kernel capture run:
  - Modal app: `ap-GLsj6uzr759uBuM1lMLxdH`
  - report: `prefill-77daf91d-20260406-134835.ncu-rep`
- precompute kernel capture run:
  - Modal app: `ap-1C7ByVBcXiCB1tBMGCKY6c`
  - report: `compute_gate_beta_kernel-77daf91d-20260406-135149.ncu-rep`
- main report inspect:
  - Modal app: `ap-Ss6qmRAegEqVSlFuwjTKQW`

---

## 10. Next Action

- [x] tiny workload baseline captured on Modal
- [x] both kernels separated
- [x] summary doc updated
- [ ] capture `ba08a83e`
- [ ] capture `5835a2bc`
- [ ] capture `07aa7922`
