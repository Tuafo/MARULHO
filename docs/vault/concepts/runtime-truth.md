---
type: concept
status: active
related_code: []
related_docs: []
related_papers: []
related_benchmarks: []
---

        # Runtime Truth

        ## Definition

        Operator-facing status that reports what the runtime can prove now, including liveness, device truth, and blocked mutation surfaces.

        The column-transition evidence names the requested and resolved executor, CUDA/tensor device, compile-only warmup result and candidate shapes, execution and failure counts, fallback reason, and the fail-closed post-launch policy. This proves which path executed; it does not convert hot-window throughput into full service throughput.

        When `cuda_graph_text` is active, the same evidence includes graph capture success and latency, graph names, fixed-address status, pre-route replay and sensory-bypass counts, transition replay/failure counts, and tensor device. Pointer changes disable the graph before mutation. These fields prove graph execution, not whole-service graph capture.

        Runtime Truth also exposes the last live tick's bounded stage timings for source selection/collection, training, concept observation, finalization, lock waits, and cooperative yields. These are CPU wall-clock orchestration measurements without per-token CUDA synchronization; CUDA kernel attribution still requires an explicit profiler slow path.

        Source collection latency can now distinguish cold local/file ingestion from cache-restored file-source ticks. A file-source cache hit should collapse `collect_source_queue` without being interpreted as faster neural cognition; it proves that deterministic source encoding was moved out of the tick.

        Memory-store device evidence includes archival update, reservoir admission/rejection, optional payload copy, and avoided-copy counters. These prove the CPU archival boundary and the executed rejection optimization; they do not prove replay quality or full-runtime throughput.

        The same evidence reports `stc_state_storage=zero_copy_array_buffer`, `stc_decay_zero_copy=true`, and the exact byte count for capture tags, local PRP, and strong-tag flags. These fields prove the representation executing in the live store. They do not claim that archival memory is CUDA-resident.

        Trainer metrics include `trainer_telemetry_interval_tokens` and `trainer_telemetry_due` so operators can distinguish freshly synchronized scalar metrics from cached values. Runtime Truth remains visible without forcing every cognitive tick to synchronize CUDA for display-only statistics.

        ## Relationships

        - [Subcortex](subcortex.md)
        - [Runtime Truth](runtime-truth.md)
        - [Runtime Evidence](runtime-evidence.md)

        ## Source Links

        - [CONTEXT.md](../../../CONTEXT.md)
        - [README.md](../../../README.md)
        - [Research notes](../../research-living-brain.md)

        ## Ambiguity

        Keep claims evidence-gated. Do not widen this term into a generic programming or biology concept without updating [CONTEXT.md](../../../CONTEXT.md).
