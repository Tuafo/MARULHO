---
type: capability
status: draft
related_code: []
related_docs: []
related_papers: []
related_benchmarks: []
---

            # Language Readout Status

            Language readout status covers grounded bounded surfaces, not free-form generation.

            ## Evidence Rule

            Do not claim this capability as live unless linked Runtime Evidence or benchmark output supports it.

            ## Current Runtime-State Evidence

            The SNN language plasticity snapshot exposes canonical `language_*` and `readout_*` evidence for growth/pruning evidence. Operators and new code should use the canonical names; retired `thought_*` naming is not live capability evidence and does not imply free-form thought generation or language autonomy.

            The surface, memory, consolidation, structural-plasticity, readout-capacity-mutation, newborn-neuron integration, critical-period learning, maturation review, and newborn-synapse pruning chains now use canonical `snn-language-*` routes. They preserve the existing reviewed/checkpoint-backed promotion gates and keep thought-era public route vocabulary out of the active API.

            The surface chain itself is canonical in production as
            `snn_language_readout_surface_design`,
            `snn_language_readout_surface_preflight`,
            `execute_snn_language_readout_surface`, and
            `snn_language_readout_surface_event_review`. Checkpoint load/save
            keeps canonical readout-surface fields and drops noncanonical
            readout-ledger state; status should not treat retired names as live
            capability evidence.

            The readout-memory and readout-consolidation chains are canonical in production as
            `snn_language_readout_memory_design`,
            `snn_language_readout_memory_preflight`,
            `execute_snn_language_readout_memory`, and
            `snn_language_readout_memory_event_review`, followed by
            `snn_language_readout_consolidation_design`,
            `snn_language_readout_consolidation_preflight`,
            `execute_snn_language_readout_consolidation`, and
            `snn_language_readout_consolidation_event_review`. The
            readout-structural-plasticity chain now continues that canonical
            path through `snn_language_readout_structural_plasticity_design`,
            `snn_language_readout_structural_plasticity_preflight`,
            `execute_snn_language_readout_structural_plasticity`, and
            `snn_language_readout_structural_plasticity_event_review`. The
            readout-capacity mutation chain is also canonical through
            `snn_language_readout_capacity_mutation_design`,
            `snn_language_readout_capacity_mutation_preflight`,
            `apply_snn_language_readout_capacity_mutation`, and
            `snn_language_readout_capacity_mutation_event_review`, with public
            routes under `snn-language-readout-capacity-mutation-*` and request
            fields named `snn_language_readout_capacity_mutation_*`.
            Checkpoint load/save keeps canonical readout ledger fields only;
            no active API/facade/ledger call alias keeps old production paths
            alive. The current
            source-window benchmarks are
            `reports/bounded_replay_window_20260622/snn-readout-ledger-normalization-readout-memory-canonical.json`
            and
            `reports/bounded_replay_window_20260622/snn-readout-ledger-normalization-readout-consolidation-canonical.json`,
            followed by
            `reports/bounded_replay_window_20260622/snn-readout-ledger-normalization-readout-structural-canonical.json`.
            The paired structural hot-path report
            `reports/bounded_replay_window_20260622/hotpath-active-pressure-65536-524288-i32-readout-structural-canonical.json`
            keeps measured throughput separate from prewarm setup and shows the
            live tick remains bounded at `12/65536` route rows.

            The readout-capacity mutation naming retirement is covered by
            `reports/bounded_replay_window_20260622/hotpath-active-pressure-65536-524288-i32-readout-capacity-canonical-noprofile-rerun.json`.
            It processed `524288` tokens at `5826.031 tokens/sec`, with
            bounded `12/65536` route rows, `65526` cached transition rows,
            `state_transition_runs_all_columns=false`, no observed contention,
            and RTX memory `1798->1796 MiB`.

            The readout-newborn developmental naming retirement extends that
            canonical chain through newborn-neuron integration,
            critical-period learning, maturation review, and newborn-synapse
            pruning. The hot-path report
            `reports/bounded_replay_window_20260622/hotpath-active-pressure-65536-524288-i32-readout-newborn-canonical.json`
            processed `524288` tokens at `5783.832 tokens/sec`, with
            bounded `12/65536` route rows, `65526` cached transition rows,
            `state_transition_runs_all_columns=false`, no observed contention,
            and RTX memory `1915->1913 MiB`.

            `snn_language_readout_corpus_evaluation.v1` is now the first bounded corpus-level report for next-readout trajectories. Runtime Truth exposes the latest saved report as `snn_language_readout_corpus_runtime_truth.v1`, including available/trained/grounded/device status, mutation absence, latency, memory/VRAM cost, and the promote/reject reason. Missing reports remain a rejection/collection state, not a hidden live generator.

            `snn_language_readout_corpus_checkpoint_review.v1` adds checkpoint/rollback truth for that same sparse readout path. Runtime Truth exposes checkpoint status, rollback status, restore verification, checkpoint bytes/hash, transition-weight counts, and production-runtime mutation absence from the latest saved checkpoint-review report.

            The first saved report used a bounded local fixture corpus and produced `promotion_decision=promote_bounded_readout_review` with `mean_mismatch_delta=0.166667`; the paired checkpoint review wrote an isolated sparse readout checkpoint, verified restore, wrote a rollback manifest, and kept `production_runtime_changed=false`. This is review evidence only, not live decoding. The paired 262144-token CUDA runtime check stayed at `6307.305 tokens/sec` with no observed contention, so the slow-path report did not lower the promoted runtime throughput bar.

            ## Links

            - [Runtime Truth](../concepts/runtime-truth.md)
            - [Capability Claim](../concepts/capability-claim.md)
            - [Language from Spikes](../concepts/language-from-spikes.md)
            - [Language Readout Speed](../benchmarks/language-readout-speed.md)
