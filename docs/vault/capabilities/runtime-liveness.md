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

            ## Links

            - [Runtime Truth](../concepts/runtime-truth.md)
            - [Capability Claim](../concepts/capability-claim.md)
