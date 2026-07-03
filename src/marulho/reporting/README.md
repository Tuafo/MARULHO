# Reporting

Use this with [../../../README.md](../../../README.md) and
[../../../CONTEXT.md](../../../CONTEXT.md).

`reporting` owns human-readable summaries, benchmark plots, validation
summaries, and report helpers.

## Owns

- Formatting evidence for humans.
- Summaries of existing reports and benchmark outputs.
- Read-only inventories over saved JSON reports for service/UI display.

## Must Not Own

- Primary Runtime Truth.
- Mutation decisions.
- Capability claims without source evidence.

## Runtime Rules

Reporting is a slow-path projection over evidence. It must not run hidden
benchmarks, mutate runtime state, or turn configured intent into a proven CUDA
or cognition claim.
