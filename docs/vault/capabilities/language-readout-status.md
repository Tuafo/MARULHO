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

            `snn_language_readout_corpus_evaluation.v1` is now the first bounded corpus-level report for next-readout trajectories. Runtime Truth exposes the latest saved report as `snn_language_readout_corpus_runtime_truth.v1`, including available/trained/grounded/device status, mutation absence, latency, memory/VRAM cost, and the promote/reject reason. Missing reports remain a rejection/collection state, not a hidden live generator.

            The first saved report used a bounded local fixture corpus and produced `promotion_decision=promote_bounded_readout_review` with `mean_mismatch_delta=0.166667`; this is review evidence only, not live decoding. The paired 262144-token CUDA runtime check stayed at `6307.305 tokens/sec` with no observed contention, so the slow-path report did not lower the promoted runtime throughput bar.

            ## Links

            - [Runtime Truth](../concepts/runtime-truth.md)
            - [Capability Claim](../concepts/capability-claim.md)
            - [Language from Spikes](../concepts/language-from-spikes.md)
            - [Language Readout Speed](../benchmarks/language-readout-speed.md)
