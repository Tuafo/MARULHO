# ADR 0007: Lower-Level Text Sequence Executor Required

## Status

Accepted

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

## Decision

Do not promote another local CUDA Graph wrapper, parent-capacity change,
truth-cadence change, Python burst grouping change, or route/vote wrapper as
the next executor boundary for the promoted text path.

The retained production path remains exact eight-token native parent graph
replay inside the q16 training-owned text sequence. Benchmark/prototype knobs
may remain for controlled rejection evidence, but they must fail closed or fail
early when they cannot exercise the requested executor. In particular, native
parent graph capacity probes must not exceed or fail to divide the execution
quantum.

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

- The native8 parent graph remains the sustained production ceiling until a
  lower-level executor beats it on comparable long CUDA runs.
- Native16 remains a rejected safe prototype, not a default.
- Native32 is rejected under q16 unless a separate execution-quantum decision is
  made; the stress benchmark rejects misaligned capacity probes before startup.
- Future performance work should start from a lower-level sequence-kernel or
  device-graph design, not another Python-side launcher, scalar report, or
  local route/vote wrapper.
- Runtime Truth for any future executor must prove which executor ran, token
  coverage, fallback/failure counters, device placement, host-truth cadence,
  startup cost, and parity or quality-gate status.

## Reversal

This decision can be revisited only if a local wrapper under the existing q16
architecture produces repeated clean 131072-token CUDA wins over
`4992.049 tokens/sec`, preserves exact sequential state and fail-closed
fallback, and exposes complete Runtime Truth evidence. Short, contended, or
counter-only wins are not enough.
