# Core

Use this with [../../../README.md](../../../README.md) and
[../../../CONTEXT.md](../../../CONTEXT.md).

`core` owns local SNN mechanisms: columns, context, binding, abstraction,
topography, plasticity, surprise, sparsity, and CUDA/Triton tensor semantics.

## Owns

- Tensor and state mechanisms that can report observed device/backend evidence.
- Competitive and predictive column behavior, including candidate-scoped
  homeostasis and prediction evidence.
- Binding and topology algorithms, including non-mutating topology proposals.
- Core CUDA/Triton kernels such as fused route/vote scoring.
- The LM-head RMSNorm Triton forward primitive in
  `language_rmsnorm_triton.py`, including forced parity/benchmark execution,
  PyTorch fallback, and runtime-use counters.
- The LM-head PLIF/adaptive-LIF Triton forward primitive in
  `language_plif_triton.py`, including forced parity/benchmark execution,
  PyTorch fallback, and runtime-use counters for membrane/spike/selective-state
  no-grad updates.
- The LM-head PLIF/adaptive-LIF Triton surrogate backward primitive in
  `language_plif_triton.py`, including `float32` gradient parity,
  PyTorch fallback, and runtime-use counters for surrogate training updates.
- The LM-head selective recurrent state scan Triton primitive in
  `language_selective_scan_triton.py`, including forced parity/benchmark
  execution, PyTorch fallback, and runtime-use counters for standalone
  `[batch,time,state_dim]` recurrent state scans.
- The LM-head local eligibility-trace update Triton primitive in
  `language_eligibility_trace_triton.py`, including forced parity/benchmark
  execution, PyTorch fallback, and runtime-use counters for final trace updates
  over `[batch,time,state_dim]` spike sequences.
- The LM-head route/vote top-k Triton primitive in
  `language_route_topk_triton.py`, including forced parity/benchmark
  execution, PyTorch fallback, and runtime-use counters for bounded
  routed-expert candidate scoring and selected expert IDs.
- The LM-head selected expert dispatch/combine Triton primitive in
  `language_expert_dispatch_triton.py`, including forced parity/benchmark
  execution, PyTorch fallback, and runtime-use counters for block-sparse
  routed expert rows.
- The LM-head bounded memory-slot retrieval Triton primitive in
  `language_memory_slots_triton.py`, including forced no-grad parity execution,
  PyTorch fallback, and runtime-use counters for selected memory-slot context
  rows.
- The LM-head sampled-vocabulary cross-entropy Triton primitive in
  `language_sampled_vocab_ce_triton.py`, including forced parity/benchmark
  execution, PyTorch fallback, forceable custom-autograd training probes, and
  runtime-use counters for selected vocabulary loss rows.

## Must Not Own

- HTTP/API behavior, service lifecycle, persistence policy, or UI contracts.
- Public capability claims. Core emits evidence; it does not promote claims.
- Hidden structural mutation inside ordinary live ticks.

## Runtime Rules

- Reporting must not force repeated scalar CUDA synchronization. Bounded column
  reports may take compact snapshots while live tensors remain on the runtime
  device.
- Disabled routing work should not launch CUDA kernels merely to multiply by
  zero.
- Language RMSNorm uses Triton only where the row count is large enough to beat
  the PyTorch CUDA expression. One-token streaming remains on the faster CUDA
  graph/PyTorch path and reports that fallback instead of silently regressing
  sustained throughput.
- Language PLIF forward uses Triton for no-grad/eval rows where the row count
  policy allows it. Gradient training may use the `float32` Triton surrogate
  backward path where the same policy allows it; half-precision backward stays
  on PyTorch until separate gradient parity evidence exists.
- Language selective scan uses Triton for CUDA recurrence tensors where the
  scan-size policy allows it. Standalone scan parity is not full state-block
  fusion; training-loop integration still needs separate complete-runtime
  evidence before promotion.
- Language eligibility-trace final update uses Triton for no-grad CUDA spike
  sequences where scan-size policy allows it. The matching no-grad PLIF
  variant skips inline eligibility load/store, but the current complete
  batch-16/seq-64 `524288` forward impact report is slower than inline PLIF, so
  the deferred state-block path remains off by default.
- Language route/vote top-k uses Triton for no-grad CUDA routed-expert rows
  where the row-count policy allows it. Gradient training stays on the PyTorch
  route-score/top-k path so route keys can receive gradients, and one-token
  streaming falls back unless the policy proves a real launch-cost win. The
  current complete no-grad LM forward impact report shows a `1.045x` win at
  the batch-16/seq-64 routed-expert shape while keeping broad hot-path
  promotion false.
- Language expert dispatch uses Triton for no-grad CUDA selected-expert rows
  where the token-count policy allows it. Current parity is `float32` only;
  half-precision dispatch falls back until separate numerical evidence exists.
  The current complete no-grad LM forward impact report shows a `1.057x` win at
  the batch-16/seq-64 routed-expert shape while keeping broad hot-path
  promotion false.
- Language memory-slot retrieval uses Triton for no-grad CUDA bounded memory
  rows where the row policy allows it. Supported `float32` gradient-training
  forward can opt into the Triton-forward/custom-autograd path with
  `MARULHO_LANGUAGE_MEMORY_SLOTS_TRITON_TRAINING=1`, but the maintained default
  stays on torch autograd because the full continual-learning window beat the
  Triton training path. The complete no-grad `524288` batch-16/seq-64 forward
  report improves bounded retrieval from `0.839x` to `0.969x` of disabled-memory
  control, and the isolated `524288` optimizer-token training report improved
  bounded training from `3076.582` to `3110.440` train tokens/sec with opt-in
  Triton. The stronger `524288` continual-window comparison rejected that as the
  default: torch autograd reached `3134.337` update tokens/sec and opt-in Triton
  reached `3074.512`, while both kept precomputed bounded candidates and
  nonzero memory gradients. The dedicated kernel report passes three CUDA
  `float32` shape sweeps with `4.950x` geometric microbenchmark speedup;
  `float16` remains fallback until separate parity evidence exists.
- Language sampled-vocab cross entropy uses Triton for `float32` CUDA hidden
  rows and selected vocabulary IDs that include all targets. Gradient training
  can force the Triton-forward/custom-autograd path with
  `MARULHO_LANGUAGE_SAMPLED_VOCAB_CE_TRITON_TRAINING=1`, but the maintained
  default stays on the selected-row PyTorch autograd path because complete
  b16/r8 CUDA evidence was faster there (`2675.442` versus `2622.292` train
  tokens/sec). Fixed-batch experiment runners can precompute sampled target
  positions once and pass them into the same loss helper, avoiding per-update
  target-position matching in the measured hot window. Keep the forceable
  Triton path as research evidence until it wins complete-runtime training
  impact.
- Learned-chunk routing should score exact retrieved candidates when possible;
  dense assembly stays active only where full assemblies define the key.
- `bind()` updates activation evidence only. Topology mutation belongs to an
  explicit checkpoint-backed maintenance transaction.
- Dead-column revival, growth, and pruning are explicit maintenance/deep-sleep
  operations, not automatic hot-path side effects.
