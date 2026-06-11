---
type: module
status: active
related_code: []
related_docs: []
related_papers: []
related_benchmarks: []
---

        # marulho.core

        ## Responsibility

        Local SNN mechanisms: columns, context, binding, abstraction, topography, plasticity, surprise, and sparsity.

        ## Owns

        Tensor/state mechanisms and device-reportable substrate behavior.

        `core.column_runtime` owns the report-only Column Runtime control plane: a bounded awake-column scheduler summary, sampled registry evidence, cached vote evidence, disagreement, fail-closed repeated-surprise growth gate, pruning/homeostasis evidence, and bounded single-column associative recall derived from or colocated with existing competitive and predictive column tensors. `PredictiveColumnState` owns prediction failure streaks on the predictive tensor device so growth evidence is live model state rather than service projection. It does not mutate topology or control execution.

        CUDA-resident column reports take one bounded four-vector state snapshot, plus an optional prediction-failure streak vector when available, and compute the control-plane summary on CPU. This is an explicit reporting boundary that removes repeated scalar CUDA synchronization; live column tensors and execution remain on their resolved runtime device. Bounded associative recall keeps returned tensors on the caller's device and stays outside the always-on tick until separately promoted.

        `CompetitiveColumnLayer` skips input-drive matrix-vector work when `input_weight_blend` is exactly zero, and skips similarity blending when it is exactly one. Disabled routing evidence must not consume CUDA kernels merely to be multiplied by zero.

        For learned-chunk routing, `CompetitiveColumnLayer` prepares projected input before retrieval and scores only the exact retrieved candidate set. It reports the last scored count and fraction, while dense assembly remains active for representations that derive routing keys from the full assembly.

        `CompetitiveColumnLayer.process()` can also scope win-rate/threshold homeostasis to a caller-provided candidate set. The trainer enables this only after stale/deep-sleep counters can exist, so early learning remains all-column and deep-sleep columns can keep cached homeostasis state without structural mutation.

        `PredictiveColumnState` supports the same delayed candidate scope for prediction error, confidence, failure streaks, and high-prediction non-winner decay while preserving full-vector state shape. Runtime telemetry records update mode, count, fraction, and fallback reason; CUDA currently keeps dense predictive updates because scoped CUDA indexing measured launch-bound rather than faster.

        Adaptive context keeps neural state integration live on every observation while allowing its dense Hebbian input-weight update to be cadenced by the trainer. The layer reports state-update and plasticity-update counts; it does not own the always-on cadence policy.

        Hypercube binding separates hub evidence from topology mutation. `bind()` updates activation EMA but cannot change adjacency or structural mutation ledgers. `refresh_hub_topology(reason=...)` is an explicit maintenance mutation called only through the service-owned checkpoint transaction; core continues to own the topology algorithm and exact mutation ledger.

        `plan_candidate_hub_topology(...)` is the non-mutating precursor for repeated-failure growth evidence. It snapshots sparse adjacency, distributes a fixed edge budget across evidence-selected sources, hashes the baseline topology and exact proposal, and reports CPU-control/CUDA-transfer metabolism. It is never called by `bind()` or the trainer tick.

        Competitive processing no longer revives dead columns in the always-on tick. It records `steps_since_win` and spike-health stale evidence, while `force_revive_dead_columns` remains the explicit maintenance/deep-sleep mutation path.

        ## Should Not Own

        Operator HTTP surfaces, persistence policy, or language-facing claims.

        ## Hot-Path Relevance

        Treat runtime-critical tensor/state work as hot path only when it is required for live service behavior. Reporting, vault generation, and research-memory work stay slow path.

        ## Key Files

        - [src/marulho/core](../../../src/marulho/core)
        - [tests](../../../tests)

        ## Related Concepts

        [Subcortex](../concepts/subcortex.md), [Metabolism](../concepts/metabolism.md), [Column Runtime](../concepts/column-runtime.md), [Plasticity Gate](../concepts/plasticity-gate.md), [Dynamic Growth](../concepts/dynamic-growth.md), [Pruning](../concepts/pruning.md), [CUDA Evidence](../concepts/cuda-evidence.md)

        ## Graphify

        - Query: `"C:\Users\thiag\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\Scripts\graphify.exe" explain "core" --graph graphify-out/graph.json`
        - Generated module summary: [generated module index](../generated/module-index.md)
