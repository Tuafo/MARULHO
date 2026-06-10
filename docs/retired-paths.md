# Retired Paths

This file records paths that should not be revived without new evidence and, when appropriate, an ADR.

| Path/name | Status | Why retired | Replacement | Revisit condition |
| --- | --- | --- | --- | --- |
| External LLM/Cortex/ThoughtLoop runtime path | retired | Added external dependency and ambiguous cognition claims without being the living substrate. | Subcortex-owned runtime evidence, semantics/readout surfaces, replay, and Runtime Truth. | A future ADR proves a bounded dependency does not become the cognition substrate and preserves evidence gates. |
| Manager mixin compatibility aliases | retired | Preserved shallow ownership after the service split. | Explicit deep modules and RuntimeFacade/StatusReadModel ownership. | Only if a compatibility period is required by a released public API. |
| Static CUDA intent as capability proof | retired | Configuration can hide CPU fallback. | Observed tensor/device evidence in Runtime Truth and gates. | Never as proof; static intent may remain diagnostic only. |
| SNN language thought-era public HTTP route vocabulary | retired | Public readout-ledger routes now use canonical `snn-language-*` route names and `snn_language_*` request fields. Keeping the previous route families documented individually made old vocabulary look more important than the current API. | Canonical SNN language surface, memory, consolidation, structural plasticity, capacity mutation, newborn-neuron integration, critical-period learning, maturation review, and newborn-synapse pruning routes over the same reviewed ledger/executor gates. | Only if a real external integration appears; this local system does not keep compatibility routes by default. |
