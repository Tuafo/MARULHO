# Training

Use this with [../../../README.md](../../../README.md) and
[../../../CONTEXT.md](../../../CONTEXT.md).

`training` owns trainer execution, checkpointing, CUDA/native graph lifecycle,
developmental and consolidation runners, query runners, and long-run evidence.

## Owns

- `MarulhoTrainer.train_text_sequence` and ordered SNN token mutation.
- Persistent text-tick executor and CUDA graph route/transition lifecycle.
- Conditional-WHILE q16 native sequence execution, separately reported
  `torch_sequence_graph_*` q16 execution when native parent graph handles are
  unavailable, and exact fallback before mutation.
- Trainer-owned checkpoint state, route caches, strong-event rings, and burst
  evidence.

## Must Not Own

- HTTP lifecycle or UI contracts.
- Service-owned source selection and operator locks.
- Structural mutation authority outside explicit checkpoint/review gates.

## Runtime Rules

- The persistent text tick executor is not the whole brain loop. Source I/O,
  archival memory, replay review, service orchestration, and UI remain outside
  graph capture.
- The maintained service execution quantum is `16`. The promoted
  conditional-WHILE executor consumes q16 as one ordered native sequence loop
  when PyTorch exposes a raw child CUDA graph handle. If that handle is missing,
  `torch_sequence_graph_*` may consume full q16 chunks as one ordered Torch CUDA
  graph replay, but it must not increment native conditional-WHILE success
  counters. Repeated-child native parent graphs are internal fallback only.
- Boundary-aware text bursts must preserve exact token order and fail closed
  before mutation on drift, telemetry, sleep, slow-memory, metrics, sensory, or
  fallback boundaries.
- Trainer-stage profiling is evaluation evidence only. Do not add profiler
  synchronization to ordinary ticks.
- Slow replay-memory admission and ConceptStore observation are cadenced
  boundaries; they do not justify moving algorithms into service.
- `long_test_runner.py` is now a `MarulhoBrain` health runner. It feeds local
  preset text, starts/stops the brain loop, samples compact brain status, and
  checks feed/readout/tick progress through the active brain runtime.
