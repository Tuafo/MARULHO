---
type: retired
status: retired
related_code:
  - ../../../src/marulho/evaluation/artifact_io.py
  - ../../../src/marulho/service/api.py
related_docs:
  - ../../retired-paths.md
  - ../benchmarks/replay-cost.md
  - ../benchmarks/hot-path-latency.md
  - ../papers/replay-consolidation.md
related_benchmarks: []
---

# Replay Adapter Experiment Stack

## Retired Path

The isolated replay-adapter experiment stack kept a second active-looking
replay path in `src`: dry-run training approval, dry-run plan creation,
metadata-only adapter experiment output, experimental promotion gate, and
replay-to-adaptation experiment evidence. It did not mutate the live runtime,
but its report kinds were still service-visible through validation report
summaries and the dry-run plan carried an executable-looking stale adapter
command.

## Replacement

The stack is deleted. MARULHO keeps replay through the maintained
replay-sample, replay dataset, bounded readout replay/plasticity, and bounded
sleep replay windows. Generic JSON artifact loading and hashing moved to the
neutral `marulho.evaluation.artifact_io` module so autonomy and validation
tools no longer import replay approval code.

`REPORT_SUMMARY_KINDS` no longer includes
`terminus_replay_adapter_promotion_gate` or
`terminus_replay_adaptation_experiment_1`.

## Evidence

Historical reports record the adapter stack deletion and service report-kind
removal. Current repo tests keep only active `artifact_io` JSON object loading
and canonical SHA-256 hashing coverage for evaluation utilities; no
retirement-only adapter absence test remains.

## Revisit Condition

Do not restore a replay adapter stack as production or service-visible code.
Any future adapter experiment must start as a bounded offline proposal with
clear quality evidence, no hidden replay-text reasoning, no live/every-token
work, no runtime mutation, no service report allowlist exposure, and a long-run
hot-path check before promotion.
