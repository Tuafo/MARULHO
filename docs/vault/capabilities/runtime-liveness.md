---
type: capability
status: draft
related_code: []
related_docs: []
related_papers: []
related_benchmarks: []
---

            # Runtime Liveness

            Behavioral liveness is evidence-gated by runtime progress, Runtime Truth, and absence of active retired paths.

            ## Evidence Rule

            Do not claim this capability as live unless linked Runtime Evidence or benchmark output supports it.

            Runtime Truth may include `benchmark_evidence_currency` as advisory evidence about saved benchmark reports. This indicates whether accepted baselines, fresh benchmark bundles, and regression gates are current, missing, stale, or failed. It is not a liveness verdict, does not run benchmarks, and does not create CUDA or speed claims by itself.

            ## Latest Local Runtime Evidence

            On 2026-06-10, the local API quick-start path `/terminus/quick-start?preset=curriculum` configured and started the maintained curriculum runtime from the UI-visible backend. Immediate evidence: `configured=true`, `running=true`, `source_count=3`, `tick_tokens=64`. A short status poll reported Runtime Truth `degraded`, recommended action `reduce_memory_pressure_before_extending_runtime`, `tick_count=3`, `background_tokens_processed=192`, and about `16.2` tokens/second. A later status read first exceeded a 10-second client timeout, then succeeded with `tick_count=4`, `background_tokens_processed=256`, and sampled `tokens_per_second=0.0`. CUDA evidence remained absent (`cuda=null`), and benchmark evidence was not attached to this live process.

            This proves bounded local runtime start/progress for the maintained preset, but it is not an `alive` capability claim. The next liveness step is to reduce or explain the memory-pressure degradation and attach current benchmark/device evidence without moving benchmark work into the hot path.

            Follow-up implementation on 2026-06-10 pinned the maintained quick-start preset to `memory_capacity=1000` instead of inheriting a possibly tiny checkpoint capacity. Focused tests prove the preset now rebuilds both trainer config and memory store capacity to `1000` on quick-start, preventing a 64-token cadence from immediately filling a small inherited memory store.

            After restarting the local backend with the same checkpoint and applying `/terminus/quick-start?preset=curriculum`, live status reported Runtime Truth `alive`, recommended action `continue_monitoring`, `configured=true`, `running=true`, `memory_capacity=1000`, and `memory_fill=0.051` after 64 background tokens. A later sample remained `alive` with `tick_count=5`, `background_tokens_processed=320`, about `9.1` tokens/second, `memory_fill=0.301`, and no last error. CUDA evidence remained absent (`cuda=null`), so this is a local liveness and memory-pressure fix, not a CUDA capability claim.

            On 2026-06-11, after replacing the default raw Wikipedia source with `open_textbooks`, the local backend was restarted on the same UI-visible port and `/terminus/quick-start?preset=curriculum` configured the maintained Source Bank as `open_textbooks,s2orc_arxiv_abstracts,fineweb_edu` with sensory sources `science_figures,environmental_audio`. A live status sample reported Runtime Truth `alive`, recommended action `continue_monitoring`, `token_count=681`, `tick_count=11`, `background_tokens_processed=657`, `source_count=3`, `configured=true`, and `running=true`. Device evidence reported `resolved_device=cuda`, `cuda_available=true`, routing search on `cuda`, and column-runtime execution `candidate_subset` scoring `10 / 1024` columns. This proves the new source bank is live with observed CUDA placement and sparse candidate execution; it does not prove source quality, factual promotion, or a CUDA speedup.

            A later 2026-06-11 UI-visible runtime check exposed active background tick execution evidence through `terminus_runtime.execution`. While `/status` may return a cached projection during CUDA lock contention, `/terminus` now provides fresh tick phase, source, target token, elapsed-time, and active-request evidence for the Systems UI. The live run showed CUDA selected, candidate-subset execution (`10 / 1024` columns), completed ticks, and a visible active phase/elapsed field. Tick latency is still weak: one observed 64-token HF tick took about `73.3 s`, so this is an observability/liveness improvement, not a throughput win.

            ## Binding Topology Mutation

            Binding-hub topology refresh is available only as an explicit slow-path transaction. Runtime evidence remains split: always-on `bind()` reports hub evidence updates without topology writes, while an operator-confirmed application must present a hash-bound target, method, reason, edge budget, current revision, and rollback checkpoint. Success reports exact edge/growth/prune deltas and a verified committed checkpoint; no-op, over-budget, tampered, or failed commits restore binding state and revision.

            This is not an autonomous-growth capability and not a CUDA speed claim. Synthetic 1024-column refresh measured substantially slower on CUDA because topology selection is still Python/control-bound; the default bounded transaction would reject the observed 572-edge delta.

            Repeated predictive failures can now produce a read-only binding-growth trial design before mutation review. The artifact uses server-owned candidate columns, exact baseline-topology and plan hashes, a fixed edge budget, observed tensor-device evidence, and no topology refresh. At 1024 columns, the optimized planner measured about `5.0 ms` median on CPU and `5.1 ms` on CUDA after replacing repeated CUDA scalar reads with two bounded adjacency snapshots. This proves a practical explicit planning surface, not cognitive improvement or autonomous growth.

            Live operator responsiveness was rechecked on 2026-06-11 after removing external stream `close()` calls from the manager-lock stop transition and requiring quiescent checkpoint saves. With the curriculum runtime active on the 1024-column CUDA checkpoint, `/checkpoint/save` returned HTTP `409` in `343.68 ms` and wrote no artifact. `/terminus/stop` then completed in `3967.76 ms`. A stopped checkpoint save completed in `14508.63 ms`, produced an `11,667,609` byte checkpoint, and restored `18` slow-memory input patterns. This proves bounded operator control and coherent checkpoint creation; stopped checkpoint serialization is still a slow path and needs further profiling.

            ## Links

            - [Runtime Truth](../concepts/runtime-truth.md)
            - [Capability Claim](../concepts/capability-claim.md)
