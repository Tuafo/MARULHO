# Reporting

Use this with [../../../README.md](../../../README.md) and
[../../../CONTEXT.md](../../../CONTEXT.md).

`reporting` owns human-readable summaries, benchmark plots, validation
summaries, and report helpers.

## Owns

- Formatting evidence for humans.
- Summaries of existing reports and benchmark outputs.
- Read-only inventories over saved JSON reports for service/UI display.
- Current language-evidence projections that condense saved benchmark,
  generation, repair, continual-learning throughput, forgetting/replay,
  structural-plasticity, checkpoint-lineage, sustained-throughput,
  active-compute, GPU-kernel, backend-decision, and checkpoint evidence without
  running the machinery.

## Must Not Own

- Primary Runtime Truth.
- Mutation decisions.
- Capability claims without source evidence.

## Runtime Rules

Reporting is a slow-path projection over evidence. It must not run hidden
benchmarks, mutate runtime state, or turn configured intent into a proven CUDA
or cognition claim.

`build_current_language_evidence_projection(...)` is the service/UI helper for
the current language runtime evidence card. It reads saved JSON reports only,
keeps `/brain/evidence/language` read-only, and preserves the claim boundary:
the projection can point at ready-for-review reports and protected checkpoints,
but it cannot create a runtime or generation-quality promotion claim by itself.
It also separates accepted training/inference throughput from backend rejection
evidence, so a tempting Triton or buffer variant is visible without becoming
the default until complete-window evidence supports it.
The backend bottleneck projection includes current training-window Triton
accounting from the raw installed-learning report when available, including
tracked PyTorch fallback calls, per-kernel fallback names, and whether
continual-learning batches were staged on the model device before the measured
update window. This makes remaining sampled-vocab, memory-slot, or other
fallback work visible without running a benchmark from the service path.
Memory-slot training impact reports can write partial JSON while long arms are
still running. The projection exposes the newest partial status, completed arm
names, and missing arm names, but backend-default decisions use the newest final
or legacy-complete comparison report so an incomplete run cannot erase the last
complete backend decision.
Installed-brain structural evidence remains a report projection: it can show a
checkpoint-backed route-bank expansion, rollback verification, and post-
structure sustained speed, but it does not make reporting or service code a
mutation owner.
