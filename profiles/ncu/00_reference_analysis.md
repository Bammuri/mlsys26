# Reference Analysis for NCU Work — `tomas_reference`

작성일: 2026-04-06

## 1. 목적

이 문서의 목적은 `tomas_reference`를 그대로 따라 하는 것이 아니라,
**NCU 관점에서 어떤 원리를 배워야 하는지**를 추리는 것이다.

핵심 질문:

> decode reference가 어떤 metric을 통해 병목을 식별했고,  
> 그 중 prefill에도 그대로 유효한 원리는 무엇인가?

## 2. 참고한 소스

- `../tomas_reference/solution/cuda/kernel.cu`
- `../tomas_reference/findings/research.md`
- `../tomas_reference/findings/fi-gdn-decode-kernel.md`
- `../tomas_reference/scripts/profile_ncu.py`

## 3. Reference에서 배울 핵심 원리

### A. latency-bound kernel에서는 warp 수와 block 수가 함께 중요하다

reference는 v1과 v4 비교를 통해,
**같은 grid라도 block당 warp 수가 늘면 latency hiding이 크게 좋아질 수 있다**고 봤다.

- 근거: `../tomas_reference/findings/research.md:642-657`
- 특히 v4는 8 warps/block으로 active warps, eligible warps, occupancy를 끌어올렸다
  - `../tomas_reference/findings/research.md:665-685`

### prefill에 주는 시사점

현재 우리 커널은 `(seq, head)`당 CTA 하나에 가까운 구조라,
작은 workload에서 GPU scheduler가 놀 가능성이 있다.

즉 prefill에서도 단순히 instruction 수보다
**CTA 공급량과 block 내부 warp 수**를 같이 봐야 한다.

---

### B. vectorized load는 load efficiency를 고친다

reference는 bf16 scalar load를 `uint2`/`uint4` 기반 wide load로 바꾸면서
load sector efficiency를 크게 개선했다.

- 근거: `../tomas_reference/findings/research.md:715-767`
- 하지만 wall-clock gain은 제한적이었다
  - kernel이 여전히 fundamentally latency-bound였기 때문

### prefill에 주는 시사점

vectorized q/k load는 가치가 있다.  
하지만 그것만으로 큰 개선을 기대하면 안 된다.

즉:

1. load efficiency 문제는 **고쳐야 할 hygiene item**
2. 하지만 primary fix는 여전히 **구조와 latency hiding**

---

### C. shared-memory staging은 겉보기에 좋아 보여도 질 수 있다

reference는 여러 번의 실험 끝에,
SMEM staging이나 SMEM q/k sharing이 오히려 손해일 수 있다고 정리했다.

- 근거: `../tomas_reference/findings/research.md:845-852`
- 특히 `__syncthreads()`가 IPC와 eligible warps를 크게 깎았다고 본다
  - `../tomas_reference/findings/research.md:847-848`

### prefill에 주는 시사점

현재 우리 커널은 full state tile을 shared memory에 올리고,
token loop 안에서 barrier를 반복한다.

즉 reference의 lesson은 prefill에서 더 강하게 중요할 수 있다:

> full shared-state 구조를 유지한 채 미세조정하는 것보다,  
> shared footprint 자체를 줄이는 쪽이 더 유력하다.

---

### D. fast math는 우선순위가 아니다

reference는 `__expf/__logf`로 instruction 수는 줄었지만,
실제 wall-clock은 거의 안 움직였다고 정리했다.

- 근거: `../tomas_reference/findings/research.md:854-898`

### prefill에 주는 시사점

gate/beta math는 profiling 우선순위의 맨 앞이 아니다.  
prefill에서도 먼저 봐야 할 것은:

1. state load/store 구조
2. barrier stall
3. warp eligibility
4. CTA granularity

---

### E. benchmark time과 kernel time은 다르다

reference는 NCU kernel time과 end-to-end benchmark time 사이에
큰 차이가 있음을 명시적으로 분리했다.

- 근거: `../tomas_reference/findings/fi-gdn-decode-kernel.md:162-183`

### prefill에 주는 시사점

prefill도 decision-gate latency만 보면 kernel 구조 판단이 흔들릴 수 있다.
NCU가 필요한 이유는 단순하다:

> 실제 kernel 병목이 shared/barrier인지,  
> 아니면 host/runtime noise에 가려진 것인지를 분리해야 한다.

## 4. Reference에서 가져오면 안 되는 것

### A. decode 전용 launch shape 복제

reference kernel은 decode용으로 설계됐다.

- v1: `BV=8`, 1 warp/block
- v4: 1 V-row/warp, no shared memory
- 근거: `../tomas_reference/solution/cuda/kernel.cu:39-47`, `257-266`

prefill은 긴 sequence loop를 돈다.  
따라서 이 mapping을 그대로 가져오면 안 된다.

### B. in-kernel gate fusion을 default 정답으로 취급

reference decode는 gate를 in-kernel에서 계산한다
- `../tomas_reference/solution/cuda/kernel.cu:128-137`, `337-346`

하지만 prefill은 token 수가 길고,
우리 current baseline은 gate/beta precompute를 이미 갖고 있다.

따라서 prefill에서는 profiling 없이
“reference가 fused니까 우리도 fuse”라고 가면 안 된다.

### C. BF16-state 사고방식

reference 문서는 BF16 state path를 강하게 다루지만,
우리 현재 경로는 FP32 state다.

즉 register pressure, shared footprint, memory behavior 모두 다르게 볼 필요가 있다.

## 5. Prefill용 결론

reference로부터 지금 당장 가져와야 할 profiling 질문은 이 4개다.

1. 현재 prefill kernel은 **barrier-heavy** 인가?
2. 현재 prefill kernel은 **shared-memory-heavy** 인가?
3. 현재 prefill kernel은 **warp 수 부족 / eligible warps 부족** 상태인가?
4. q/k load는 실제로 **load efficiency 문제**가 있는가?

이 질문들에 답할 수 있으면,
새 알고리즘 `SRTP`가 왜 필요한지/아닌지를 더 명확히 판단할 수 있다.

## 6. SRTP와의 연결

`plan/00_p0_restart.md`의 SRTP 방향은
reference의 코드를 가져오는 것이 아니라 아래 원리만 계승한다.

1. state는 shared보다 register 쪽으로 옮길수록 유리할 수 있다
2. CTA/block 내부 warp 수를 늘려 latency hiding을 확보해야 한다
3. q/k는 vector-friendly path를 유지해야 한다
4. barrier는 줄일수록 좋다

즉 profiling의 목적은 reference 유사성 검증이 아니라,
**SRTP 가설을 measurement로 뒷받침할 수 있는지 확인하는 것**이다.
