---
type: retired-path
status: retired
related_code:
  - ../../../src/marulho/service/replay_runtime.py
  - ../../../src/marulho/evaluation/sleep_plasticity_ticket_queue_source_window_benchmark.py
related_docs:
  - ../../retired-paths.md
  - ../../../CONTEXT.md
  - ../papers/replay-consolidation.md
  - ../benchmarks/hot-path-latency.md
related_benchmarks:
  - reports/bounded_replay_window_20260620/sleep-plasticity-ticket-queue-source-window.json
  - reports/bounded_replay_window_20260620/hotpath-active-pressure-65536-524288-i32-sleep-plasticity-ticket-queue-source-window.json
---

# Full-Retained Sleep Plasticity Ticket Queue Verification

Full-retained verification for sleep-plasticity review-ticket queues is retired
as a production shape. The old queue shape capped returned tickets but still
made it easy to verify or materialize the entire retained deque before
autonomy, scheduler-design, or scheduler-installation proposals.

The active path is source-windowed:

- `bounded_snn_sleep_plasticity_review_ticket_queue_source_window.v1`
- `bounded_snn_sleep_plasticity_scheduler_design_review_ticket_queue_source_window.v1`

Each queue inspects at most `16` newest retained records, reports
`retained_count` separately from `source_window_inspected_count`, keeps
archival/source/score placement on CPU, and states no global candidate/score
scan, raw replay text, hidden language reasoning, live tick, every-token
cadence, scheduler install, mutation, or plasticity. A malformed newest ticket
blocks the bounded window instead of scanning past it to older retained records.

## Evidence

`reports/bounded_replay_window_20260620/sleep-plasticity-ticket-queue-source-window.json`
passed with `16/64` inspected records on both queues, latest-verified parity
against diagnostic full-retained scans, `4x` less source work, mean bounded
latency `3.326372 ms` for sleep-review tickets and `201.999864 ms` for
scheduler-design-review tickets, CPU placement, `0.072 MiB` traced Python peak,
and CUDA unused with `0.0 MiB` allocation/reservation.

`reports/bounded_replay_window_20260620/hotpath-active-pressure-65536-524288-i32-sleep-plasticity-ticket-queue-source-window.json`
processed `524288` tokens at `5997.714 tokens/sec`, p95 `21.621 ms`,
`train_compute=0.135466 ms/token`, bounded `12/65536` route rows, zero
graph/native sequence failures, CPU max `32%`, GPU max `20%`, and RTX 3060
memory `1779->1780 MiB`.

## Revisit Condition

Reintroduce full-retained ticket verification only as benchmark-local
diagnostic evidence or through a stronger indexed selector that proves better
prediction, grounding, or reconstruction under an explicit source budget,
device-placement report, and repeated long-run live-tick protection.
