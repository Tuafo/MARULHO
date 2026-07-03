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
- Language PLIF forward uses Triton only for no-grad/eval rows where the row
  count policy allows it. Gradient training keeps the surrogate-gradient
  PyTorch path until a PLIF backward surrogate kernel has parity evidence.
- Learned-chunk routing should score exact retrieved candidates when possible;
  dense assembly stays active only where full assemblies define the key.
- `bind()` updates activation evidence only. Topology mutation belongs to an
  explicit checkpoint-backed maintenance transaction.
- Dead-column revival, growth, and pruning are explicit maintenance/deep-sleep
  operations, not automatic hot-path side effects.
