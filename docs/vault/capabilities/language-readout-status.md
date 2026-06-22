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

            The SNN language plasticity snapshot exposes canonical `language_*` aliases for growth/pruning evidence while retaining legacy `thought_*` compatibility keys. Operators and new code should prefer the canonical names; compatibility keys do not imply free-form thought generation or language autonomy.

            The surface, memory, consolidation, structural-plasticity, capacity-mutation, newborn-neuron integration, critical-period learning, maturation review, and newborn-synapse pruning chains now use canonical `snn-language-*` routes. They preserve the existing reviewed/checkpoint-backed promotion gates and keep thought-era public route vocabulary out of the active API.

            The surface chain itself is canonical in production as
            `snn_language_readout_surface_design`,
            `snn_language_readout_surface_preflight`,
            `execute_snn_language_readout_surface`, and
            `snn_language_readout_surface_event_review`. Checkpoint load/save
            migrates legacy `autonomous_snn_language_thought_surface_*` state to
            readout-surface fields once; status should not treat those legacy
            names as live capability evidence.

            The readout-memory and readout-consolidation chains are canonical in production as
            `snn_language_readout_memory_design`,
            `snn_language_readout_memory_preflight`,
            `execute_snn_language_readout_memory`, and
            `snn_language_readout_memory_event_review`, followed by
            `snn_language_readout_consolidation_design`,
            `snn_language_readout_consolidation_preflight`,
            `execute_snn_language_readout_consolidation`, and
            `snn_language_readout_consolidation_event_review`. Checkpoint load/save
            migrates legacy `autonomous_snn_language_thought_memory_*` state to
            readout-memory fields and legacy
            `autonomous_snn_language_thought_consolidation_*` state to
            readout-consolidation fields once; no active API/facade/ledger call
            alias keeps either old production path alive. The current
            source-window benchmarks are
            `reports/bounded_replay_window_20260622/snn-readout-ledger-normalization-readout-memory-canonical.json`
            and
            `reports/bounded_replay_window_20260622/snn-readout-ledger-normalization-readout-consolidation-canonical.json`.

            `snn_language_readout_corpus_evaluation.v1` is now the first bounded corpus-level report for next-readout trajectories. Runtime Truth exposes the latest saved report as `snn_language_readout_corpus_runtime_truth.v1`, including available/trained/grounded/device status, mutation absence, latency, memory/VRAM cost, and the promote/reject reason. Missing reports remain a rejection/collection state, not a hidden live generator.

            `snn_language_readout_corpus_checkpoint_review.v1` adds checkpoint/rollback truth for that same sparse readout path. Runtime Truth exposes checkpoint status, rollback status, restore verification, checkpoint bytes/hash, transition-weight counts, and production-runtime mutation absence from the latest saved checkpoint-review report.

            The first saved report used a bounded local fixture corpus and produced `promotion_decision=promote_bounded_readout_review` with `mean_mismatch_delta=0.166667`; the paired checkpoint review wrote an isolated sparse readout checkpoint, verified restore, wrote a rollback manifest, and kept `production_runtime_changed=false`. This is review evidence only, not live decoding. The paired 262144-token CUDA runtime check stayed at `6307.305 tokens/sec` with no observed contention, so the slow-path report did not lower the promoted runtime throughput bar.

            ## Links

            - [Runtime Truth](../concepts/runtime-truth.md)
            - [Capability Claim](../concepts/capability-claim.md)
            - [Language from Spikes](../concepts/language-from-spikes.md)
            - [Language Readout Speed](../benchmarks/language-readout-speed.md)
