# Decode Optimization Log

## Setup
- Branch: `gdn-decode-opt`
- Baseline commit: `db64e0b`
- Decode config: `definition = "gdn_decode_qk4_v8_d128_k_last"`, `entry_point = "kernel.cu::gdn_decode"`
- Primary large-batch benchmark: workload `eaf0a285-447c-4432-8e68-d287acc3cb08` (`batch_size = 64`)
- Primary small-batch benchmark: workload `901e5104-dccb-4c3f-ae13-ef4d31a4d456` (`batch_size = 1`)
- Benchmark command:

```bash
docker exec mlsys26-dev bash -lc '
  cd /workspace/mlsys26 &&
  FIB_WORKLOAD_UUID=<uuid> /opt/conda/bin/conda run --no-capture-output -n fi-bench \
    modal run scripts/run_modal.py
'
```

## Executed Campaign

| ID | Commit | Composition | Category | Large Batch 64 | Small Batch 1 | Status | Optimization Summary | Notes |
|---|---|---|---|---|---|---|---|---|
| C0 | `db64e0b` | baseline | reference | `63.272 ms`, `1.05x`, `abs=0`, `rel=0` | N/R | Passed | ATen-based decode reference path plus stable Modal workload selection. | Starting point for all later comparisons. |
| C1 | `404d012` | C0 + fused CUDA kernel | kernelization | `0.134 ms`, `543.59x`, `abs=7.93e-04`, `rel=2.72e+01` | N/R | Passed | Replaced ATen decode loop with one CTA per `(batch, v_head)` CUDA kernel, fused gate/state/output math, and kept CPU fallback for local verification. | Large throughput jump; relative error increased but benchmark still passed. |
| C2 | `530b728` | C1 + PTX vector IO | PTX / memory | `0.058 ms`, `1285.33x`, `abs=4.77e-07`, `rel=2.63e-02` | N/R | Passed | Added inline PTX `ld/st.global.v4.f32` helpers and vectorized state-row update path. | This is the cleanest large-batch step-up in the campaign. |
| C3 | `e84c6da` | C2 + paired-head small-batch path | B200-specific | N/R | `0.032 ms`, `38.29x`, `abs=5.96e-08`, `rel=2.85e-03` | Passed | Added SM100-aware small-batch dispatch: one CTA handles two `v_head`s sharing the same `q/k` head when `batch_size <= 8`. | B200-focused path optimized for launch-limited decode. |
| C4 | `6ca96c0` | C2 + C3 + launch/FMA tuning | full stack combo | `0.053 ms`, `1374.06x`, `abs=4.77e-07`, `rel=3.04e-02` | `0.035 ms`, `43.31x`, `abs=5.96e-08`, `rel=2.85e-03` | Passed | Added launch bounds, FMA-based accumulation/update, and current-stream launch path on top of all earlier optimizations. | Best measured large-batch variant; slightly slower than C3 on the representative small-batch workload. |

## Combination Readout

| Combination | Commit | Result |
|---|---|---|
| `kernelize + fuse` | `404d012` | Converted the decode path from a correctness-first ATen implementation into a real CUDA kernel. |
| `kernelize + PTX vector IO` | `530b728` | Best standalone large-batch uplift before B200 specialization. |
| `kernelize + PTX vector IO + B200 small-batch` | `e84c6da` | Best small-batch result in this campaign. |
| `full stack` | `6ca96c0` | Best measured large-batch result and final recommended decode variant. |

## B200 / PTX Next Items

| Priority | Area | Idea | Expected Effect | Status |
|---|---|---|---|---|
| P1 | B200 | Persistent-CTA decode scheduler for `batch_size <= 4` | Further reduce launch overhead on the smallest decode shapes. | Queued |
| P1 | B200 | Cluster-aware launch experiment for paired-head decode | Improve CTA residency and cache reuse on SM100. | Queued |
| P2 | PTX | Add cache-policy or `prefetch.global.L2` hints to the state-row path | Potentially reduce long-latency state reads in the generic kernel. | Queued |
| P2 | PTX | Inspect generated PTX/SASS for spill points after `__launch_bounds__` | Validate whether more aggressive unroll is still safe. | Queued |
| P3 | B200 + math | Explore tensorcore-friendly reformulation of the row update | Only worthwhile if the decode update can be reshaped into stable fragment math. | Queued |
