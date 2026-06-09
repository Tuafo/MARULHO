# Retired Paths

This file records paths that should not be revived without new evidence and, when appropriate, an ADR.

| Path/name | Status | Why retired | Replacement | Revisit condition |
| --- | --- | --- | --- | --- |
| External LLM/Cortex/ThoughtLoop runtime path | retired | Added external dependency and ambiguous cognition claims without being the living substrate. | Subcortex-owned runtime evidence, semantics/readout surfaces, replay, and Runtime Truth. | A future ADR proves a bounded dependency does not become the cognition substrate and preserves evidence gates. |
| Manager mixin compatibility aliases | retired | Preserved shallow ownership after the service split. | Explicit deep modules and RuntimeFacade/StatusReadModel ownership. | Only if a compatibility period is required by a released public API. |
| Static CUDA intent as capability proof | retired | Configuration can hide CPU fallback. | Observed tensor/device evidence in Runtime Truth and gates. | Never as proof; static intent may remain diagnostic only. |
