# Consolidation

Use this with [../../../README.md](../../../README.md) and
[../../../CONTEXT.md](../../../CONTEXT.md).

`consolidation` owns CPU archival memory, replay records, and explicit
consolidation metadata.

## Owns

- CPU-resident archival evidence.
- Memory-store admission, tags, PRP/STC scalar state, and replay records.
- Bucket-consolidation cache generation used by graph-safe pointer reuse.

## Must Not Own

- Device-local replay computation.
- Live plasticity application.
- Global scans hidden in live ticks.

## Runtime Rules

- Reservoir admission is decided before optional CUDA-to-CPU payload copies.
  Rejected archival rows should avoid unnecessary tensor copies.
- Capture tags, strong tags, and local PRP values use compact numeric storage
  for decay; checkpoint snapshots still serialize ordinary portable state.
- The archival boundary remains CPU-owned. Replay tensors move to the model
  device only when replay computation consumes them.
- Replay selection must be bounded, bucket-scoped where required, and explicit.
  Missing anchor scope should report a blocked/empty replay reason rather than
  falling back to a global scorer.
