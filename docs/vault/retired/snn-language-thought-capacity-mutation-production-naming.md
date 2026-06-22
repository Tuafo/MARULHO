---
type: retired-path
status: retired
related_code:
  - ../../../src/marulho/service/api.py
  - ../../../src/marulho/service/schemas.py
  - ../../../src/marulho/service/runtime_facade.py
  - ../../../src/marulho/service/snn_language_readout_ledger.py
  - ../../../src/marulho/service/snn_language_plasticity_executor.py
related_docs:
  - ../../retired-paths.md
related_benchmarks:
  - reports/bounded_replay_window_20260622/hotpath-active-pressure-65536-524288-i32-readout-capacity-canonical-noprofile-rerun.json
---

# SNN Language Thought Capacity Mutation Production Naming

The thought-era and generic capacity-mutation production path is retired.

The maintained path is `snn_language_readout_capacity_mutation_*` from public
route schema through facade, ledger, checkpoint-backed executor, runtime-state
snapshot, and event review. Public routes use
`snn-language-readout-capacity-mutation-*`, and request fields use
`snn_language_readout_capacity_mutation_*`.

The retired path included `thought_capacity` state keys,
`autonomous_snn_language_thought_capacity_mutation_*` call shapes, generic
`snn-language-capacity-mutation-*` routes, and
`snn_language_capacity_mutation_*` request fields. Keeping those around would
make the same resize transaction appear to have two active control surfaces.

Reintroduce only as a temporary migration adapter for a real external client,
with tests proving it cannot become a second production path.

The accepted live-tick protection run processed `524288` tokens at
`5826.031 tokens/sec`, kept route scoring bounded at `12/65536`, cached
`65526` transition rows, kept `state_transition_runs_all_columns=false`, and
observed no contention. The earlier trainer-stage-profiler run is diagnostic
only because it measured `5555.868 tokens/sec`.
