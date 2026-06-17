---
type: paper
status: draft
related_code: []
related_docs: []
related_papers: []
related_benchmarks: []
---

        # NeuronSpark-V1

        ## Claim

        Emerging SNN language reference; inspiration-only until MARULHO owns training, grounding, and checkpoints.

        ## MARULHO Relevance

        Use this as research pressure only when it supports a MARULHO-owned mechanism, evidence gate, benchmark, or rejection note.

        ## Implementation Implication

        Do not import external runtime code or checkpoints unless a future ADR explicitly accepts that dependency. Prefer local probes, heldout gates, and rollback-aware experiments.

        The useful implementation pressure is not to load NeuronSpark as MARULHO's mind. It is to build MARULHO-owned sparse language neurons, bounded sequence/readout training, grounding verification, and device evidence. NeuronSpark's selective state-space spiking dynamics, adaptive timesteps, and fused Triton PLIF kernels support the direction of fast local spike-language modules.

        The current local step is `snn_language_readout_corpus_evaluation.v1`: a bounded corpus evaluator that measures next-readout trajectory quality, grounding, device placement, latency, memory/VRAM cost, and an explicit promote/reject decision without importing NeuronSpark code or checkpoints. The paired `snn_language_readout_corpus_checkpoint_review.v1` writes a MARULHO-owned isolated sparse transition checkpoint plus rollback manifest for operator review. NeuronSpark remains useful pressure toward sparse temporal state and fast kernels; Runtime Truth still requires MARULHO-owned report evidence and longer throughput runs before promotion. The first paired long run processed `262144` tokens at `6307.305 tokens/sec` with no observed contention, so the local report path did not buy language readiness by slowing the promoted CUDA runtime.

        ## Status

        inspiration-only

        ## Links

        - [Research notes](../../research-living-brain.md)
        - [Language from Spikes](../concepts/language-from-spikes.md)
        - [CUDA Evidence](../concepts/cuda-evidence.md)
        - [Next Throughput Goal Map](../maps/next-throughput-goal-map.md)
