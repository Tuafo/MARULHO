---
type: concept
status: active
related_code: []
related_docs: []
related_papers: []
related_benchmarks: []
---

        # Language from Spikes

        ## Definition

        The research direction of MARULHO-owned language surfaces over sparse SNN state, with grounding and promotion gates.

        ## Relationships

        - [Subcortex](subcortex.md)
        - [Runtime Truth](runtime-truth.md)
        - [Runtime Evidence](runtime-evidence.md)
        - [Plasticity Gate](plasticity-gate.md)

        ## Runtime State Naming

        `snn_language_plasticity_runtime_state.v1` now exposes canonical `language_*` evidence names for capacity mutation, newborn-neuron integration, critical-period learning, and newborn-synapse pruning. Legacy `thought_*` keys remain compatibility aliases only; they are not a claim that MARULHO has a hidden text-thought substrate.

        Surface, memory, consolidation, structural plasticity, capacity mutation, newborn-neuron integration, critical-period learning, maturation review, and newborn-synapse pruning APIs now use canonical `snn-language-*` routes with `snn_language_*`/`surface_*` request fields. These routes delegate to the same reviewed ledger/executor gates and do not introduce a new mutation path. Thought-era public route vocabulary is no longer part of the active API.

        The readout-surface chain is now canonical inside production code:
        `snn_language_readout_surface_design`,
        `snn_language_readout_surface_preflight`,
        `execute_snn_language_readout_surface`, and
        `snn_language_readout_surface_event_review`. Legacy persisted
        `autonomous_snn_language_thought_surface_*` keys are load/save
        migration aliases only, so the surface remains bounded readout evidence
        rather than hidden language reasoning.

        ## Bounded Corpus Evaluation

        `snn_language_readout_corpus_evaluation.v1` is the slow-path runner that evaluates grounded next-readout trajectories over a bounded corpus. It wraps the transition-memory prediction gate and records dataset provenance, grounding support, device status, latency, memory/VRAM cost, mutation absence, and a promote/reject decision. Its paired `snn_language_readout_corpus_checkpoint_review.v1` can write an isolated sparse transition checkpoint plus rollback manifest for operator review. Runtime Truth projects only saved reports; it must not execute evaluation, mutate runtime state, or claim live generation.

        The first local checkpoint review wrote `reports/snn_language_readout_corpus/readout-corpus-checkpoint.json`, verified restore with hash `724a0f9bf46af65e66e7aca19b1bc90f7e05fc5d303051c9f16d2a20c83d8358`, wrote a rollback manifest, and kept `production_runtime_changed=false`. The first long service check after adding the evaluator used the existing active-pressure `65536`-column CUDA checkpoint for `262144` tokens and reached `6307.305 tokens/sec` with `contention=not_observed`, preserving the same-day throughput bar. This evidence keeps language-from-spikes on a bounded, report-backed path: promote review of the local sparse readout evaluation, reject live generation until a trained checkpointed readout passes grounding and rollback gates.

        ## Source Links

        - [CONTEXT.md](../../../CONTEXT.md)
        - [README.md](../../../README.md)
        - [Research notes](../../research-living-brain.md)
        - [Language readout status](../capabilities/language-readout-status.md)
        - [Language readout speed](../benchmarks/language-readout-speed.md)

        ## Ambiguity

        Keep claims evidence-gated. Do not widen this term into a generic programming or biology concept without updating [CONTEXT.md](../../../CONTEXT.md).
