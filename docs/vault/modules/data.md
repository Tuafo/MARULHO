---
type: module
status: active
related_code: []
related_docs: []
related_papers: []
related_benchmarks: []
---

        # marulho.data

        ## Responsibility

        Source loaders and sparse encoders for text, semantic, audio, event-camera, and multimodal evidence.

        ## Owns

        Input normalization and emitted encoder/device evidence.

        Terminus Source Bank defaults are data-plane configuration, not service algorithms. The maintained text bank now starts with `open_textbooks` from `izumi-lab/open-text-books` because Hugging Face Dataset Viewer evidence showed direct `text` rows with worked educational prose, letting MARULHO replace raw Wikipedia without adding a parser or hot-path work.

        Live RTF ingestion is inference-only. When the learned-chunk codebook is empty, Runtime Sources assembles at most 32 character windows and deterministic chunk signatures on the CPU control plane, constructs the emitted vectors as one device batch, and yields device-resident views. Mutation-enabled chunk learning remains in explicit training or remote-bootstrap work.

        Live source-cache persistence is deferred. The tick hashes and schedules bounded raw-window material; a Runtime Sources worker performs atomic `torch.save` writes and service shutdown flushes pending work. Runtime Truth exposes schedule, write, skip, failure, and pending counts.

        Source concept observation is bounded per service tick. Runtime Sources and Brain Runtime may train larger sequential source windows for throughput, but ConceptStore observation and structural maintenance are capped telemetry/knowledge-layer work with explicit max-per-tick and skipped-attempt evidence.

        Source tick width is evidence-tuned. The maintained default is `128` source tokens per tick after same-checkpoint CUDA service benchmarks rejected `256` and `512` as slower complete service windows.

        Slow-memory archival is cadenced for hot-path metabolism. The maintained default records first-token, strong-capture, and then every-64-token archival events, with archive/skip evidence surfaced through Runtime Truth instead of forcing replay-memory writes every eight source tokens.

        Cross-modal text-only background ticks use specialist sleep semantics. Once the sensory trace window expires, traces are cleared once and the idle specialist no longer decays tensors every token until sensory evidence wakes it again.

        ## Should Not Own

        Runtime Truth verdicts or promotion of observations into facts/actions.

        ## Hot-Path Relevance

        Treat runtime-critical tensor/state work as hot path only when it is required for live service behavior. Do not issue one scalar CUDA workflow per character when a bounded batch preserves the representation. Reporting, vault generation, research-memory work, and chunk-codebook mutation stay slow path.

        ## Key Files

        - [src/marulho/data](../../../src/marulho/data)
        - [tests](../../../tests)

        ## Related Concepts

        [Runtime Evidence](../concepts/runtime-evidence.md), [CUDA Evidence](../concepts/cuda-evidence.md), [Capability Claim](../concepts/capability-claim.md)

        ## Graphify

        - Query: `"C:\Users\thiag\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\Scripts\graphify.exe" explain "data" --graph graphify-out/graph.json`
        - Generated module summary: [generated module index](../generated/module-index.md)
