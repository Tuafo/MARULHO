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
  generation, repair, sustained-throughput, GPU-kernel, and checkpoint evidence
  without running the machinery.

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
