# ADR 0007: Lower-Level Text Sequence Executor Required

## Status

Accepted, amended 2026-06-15

## Context

The promoted CUDA text path starts from the exact fast shape: 1024 columns,
64 column dimension, `k=10`, text-only CUDA checkpoint, `tick_tokens=128`,
`execution_quantum_tokens=16`, host-truth cadence `32`, and exact eight-token
native repeated-child CUDA Graph parent replay. It preserves sequential SNN
state updates and exposes Runtime Truth for graph replay, host-truth cadence,
fallbacks, failures, startup capture, and native compile cost.

The remaining target is the per-token graph/kernel boundary inside the ordered
text sequence. Local wrapper attempts have already been measured:

- Direct one-block route/vote fusion passed parity but regressed complete
  runtime and was deleted as hot-path code.
- A native C++ loop over `cudaGraphLaunch(graph_exec)` moved the loop below
  Python but still launched once per token and lost its long comparison.
- An eight-tick PyTorch graph body/nested sequence graph preserved parity but
  duplicated scheduling/evidence-copy work and regressed complete runtime.
- Partial native parent graphs for non-eight-token tails compiled extra parent
  graph counts but regressed complete runtime and exposed unaligned host-truth
  fallbacks.
- Wider event/truth cadences and wider Python burst groups reduced local
  counters without beating sustained complete-runtime evidence.
- Startup-warmed native16 parent graphs were safe and covered `131040` tokens
  with zero native/graph/burst failures, but the clean long run reached only
  `4887.767 tokens/sec`, below the retained native8 ceiling at
  `4992.049 tokens/sec`.
- Native32 under the maintained q16 execution quantum is not a valid native
  coverage benchmark. The run warmed `[32]` parent graphs, but q16 chunking
  made every burst partial relative to the parent graph, so Runtime Truth
  reported zero native successes and `131040` Python-loop tokens.

Current CUDA/PyTorch guidance does not provide a PyTorch Python API for the
device-owned conditional/time loop MARULHO needs. CUDA Graph conditional/device
launch features are lower-level CUDA features, while PyTorch CUDA Graphs still
favor stable fixed-address replay. FlashRNN, persistent-RNN, Triton persistent
kernels, PyTorch persistent grouped GEMM, and NeuronSpark fused PLIF kernels all
point to moving recurrent sequence ownership into a purpose-built C++/CUDA,
Triton, or hybrid executor rather than adding another host-side wrapper around
the same one-tick graph body.

On 2026-06-15, MARULHO added an opt-in CUDA conditional-WHILE parent graph
prototype selected by `cuda_graph_sequence_executor=conditional_while`,
`MARULHO_CUDA_GRAPH_SEQUENCE_EXECUTOR=conditional_while`, or the stress
benchmark `--sequence-executor conditional_while`. It keeps the retained
one-tick PyTorch CUDA Graph body, but moves the burst loop into a CUDA Graph
conditional node and a tiny device counter kernel. The executor preserves the
existing pre-mutation fallback and fail-closed launch behavior because failed
conditional parent construction returns to retained repeated-child replay, while
post-launch errors still raise.

The first clean long evidence made it a promotion candidate rather than another
rejected local wrapper. With the same text-only checkpoint and q16 shape,
`reports/conditional_sequence_20260615/native8-rerun-131072-i32.json` measured
the retained native8 executor at `5035.537 tokens/sec`, zero native failures,
and `velocity_environment.v1` contention `not_observed`. The opt-in
conditional-WHILE q8 run at
`reports/conditional_sequence_20260615/conditional-while-131072-i32.json`
measured `5277.975 tokens/sec`, and the opt-in conditional-WHILE q16 run at
`reports/conditional_sequence_20260615/conditional-while16-131072-i32.json`
measured `5559.473 tokens/sec`. The q16 conditional run covered `131040`
tokens with `8190` conditional parent launches, parent token counts `[16]`,
zero sequence/native fallbacks, zero sequence/native failures, host-truth
cadence `4097/126975`, and clean `velocity_environment.v1` evidence.

The follow-up promotion gate repeated the comparison in both orders and kept
startup/capture cost outside measured warm throughput. Pair A measured native8
at `5485.105 tokens/sec` and conditional q16 at `5883.805`. Pair B measured
conditional q16 at `6027.856` and native8 at `5816.477`. All four paired runs
reported `velocity_environment.v1` contention `not_observed`, host-truth
cadence `4097/126975`, and zero native/sequence fallbacks or failures. After
the default change, an explicit native8 opt-out reached `5329.542 tokens/sec`,
while the promoted default conditional q16 run reached `6116.646 tokens/sec`,
`train_compute=0.134167 ms/token`, `8190` conditional launches, `131040`
conditional-owned tokens, startup capture `5482.6059 ms`, conditional compile
`4970.7865 ms`, and no observed contention.

## Decision

Do not promote another local CUDA Graph wrapper, parent-capacity change,
truth-cadence change, Python burst grouping change, or route/vote wrapper as
the next executor boundary for the promoted text path.

Promote the CUDA conditional-WHILE sequence executor as the default for
eligible q16 CUDA text sequences. The retained repeated-child native parent
graph remains exact eight-token replay for fallback and explicit opt-out.
Benchmark/prototype knobs may remain for controlled rejection evidence, but
they must fail closed or fail early when they cannot exercise the requested
executor. In particular, native parent graph capacity probes must not exceed or
fail to divide the execution quantum.

The conditional-WHILE executor is the first accepted lower-level executor that
meets the directional requirement and beats the sustained native8 ceiling in
repeated clean long runs. The winning shape changes the effective sequence
capacity to `16`, so capacity ownership is split:
`cuda_graph_sequence_loop_tokens` sets the promoted conditional loop size,
while `cuda_graph_native_burst_tokens` continues to describe the repeated-child
parent capacity and remains `8` by default.

The next promotable executor must be lower level than the current
Python/CUDA Graph replay boundary. A candidate design must own a bounded
multi-tick sequence loop in C++/CUDA, Triton persistent kernels, CUDA Graph
conditional/device launch code, or a hybrid of those. It must preserve:

- exact sequential transition order and state parity against the retained path;
- host-truth cadence `32` or an explicitly re-ADR'd bounded truth contract;
- pre-mutation fallback and fail-closed post-launch behavior;
- device evidence for executor identity, token coverage, and failures;
- startup compile/capture cost outside measured warm throughput;
- no service-owned neural algorithms.

## Consequences

- The conditional-WHILE q16 parent graph is the promoted eligible CUDA text
  sequence executor.
- The native8 repeated-child parent graph remains the fallback and explicit
  opt-out path.
- Native16 remains a rejected safe prototype, not a default.
- Native32 is rejected under q16 unless a separate execution-quantum decision is
  made; the stress benchmark rejects misaligned capacity probes before startup.
- Runtime Truth now exposes `native_sequence_loop_*` fields for lower-level
  sequence-loop coverage separately from repeated-child parent-graph coverage,
  plus active/default repeated-child and sequence-loop capacity fields.
- Future performance work should start from a lower-level sequence-kernel or
  device-graph design, not another Python-side launcher, scalar report, or
  local route/vote wrapper.
- Runtime Truth for any future executor must prove which executor ran, token
  coverage, fallback/failure counters, device placement, host-truth cadence,
  startup cost, and parity or quality-gate status.

## Reversal

Revert by setting `cuda_graph_sequence_executor=native_repeated_child_graph` or
`MARULHO_CUDA_GRAPH_SEQUENCE_EXECUTOR=native_repeated_child_graph`, which returns
eligible bursts to the retained repeated-child native8 executor. This decision
should be revisited only if the promoted conditional executor loses repeated
clean 131072-token CUDA comparisons, weakens exact sequential state or
fail-closed fallback, or stops exposing complete Runtime Truth evidence. Short,
contended, or counter-only regressions are not enough.
