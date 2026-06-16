# ADR 0005: Prefer CUDA for tensor-heavy subcortex runtime work

## Status

Accepted

## Context

The project goal is shifting from a merely executable cortex-subcortex runtime toward a more continuous "living brain" system. The existing implementation already routes most subcortical computation through PyTorch tensors and `MarulhoConfig.resolve_device()`, with `device="auto"` selecting CUDA when available. Routing also has a `torch_topk` backend that becomes the automatic search path on CUDA, while FAISS HNSW and exact cosine remain CPU-friendly alternatives.

Unit tests intentionally force `MARULHO_DEVICE=cpu` in `conftest.py` so they remain deterministic and do not fail on machines without a GPU. That creates a useful split: production/runtime experiments should use CUDA when available, but correctness tests should not require CUDA.

Recent literature supports this direction: predictive coding with spiking neurons remains active work, and sparse SNN/GPU acceleration is a natural fit when the implementation can preserve event sparsity rather than densifying every operation.

## Decision

Adopt a CUDA-first posture for tensor-heavy subcortical runtime paths:

- Runtime and benchmark code should keep using `MarulhoConfig.resolve_device()` rather than hard-coding CPU.
- New tensor-heavy modules should accept an explicit `torch.device` and move persistent tensors onto that device at construction.
- CUDA claims must be proven by runtime telemetry or benchmark output that includes the actual search/device backend.
- Routing indexes must report actual cache placement, not just configured search intent. Tensor-backed backends should expose cache readiness and tensor devices in `stats()`.
- Predictive columns, competitive columns, plasticity circuits, and binding layers must expose live tensor device reports in the model runtime scope.
- Neuron dynamics modules such as AdEx must expose voltage, adaptation, and spike-timing tensor devices directly, and checkpoint restore must place live neuron tensors back on the selected runtime device.
- Checkpoint restore must select the runtime device during `torch.load`. Saving archival tensors on CPU is allowed for portability, but loading live Subcortex state through a hardcoded CPU map is not CUDA-first behavior.
- Cross-modal grounding and sparse binding variants, including spatial and hypercube binding, must expose live tensor and topology device reports when enabled.
- Context and abstraction layers must expose live tensor device reports because they control routing gain, curiosity pressure, and top-down feedback.
- Text encoders must keep tensor-heavy state on the runtime device, including learned chunking codebooks, semantic bucket embeddings, adapter tensors, emitted feature vectors, and spike traces. Python string parsing, segmentation lists, and hash-loop control flow remain CPU/control-plane work.
- CUDA-first text encoding does not permit per-character scalar launch/synchronization overhead to dominate a live tick. When the learned-chunk codebook is empty, Runtime Sources uses bounded CPU window/signature assembly plus batched device tensor construction with scalar-representation parity. Live ingestion is inference-only; mutation-enabled chunk learning remains an explicit training or remote-bootstrap path.
- The Memory Store must report its device boundary explicitly: archival replay records remain CPU storage, while trainer replay computation moves sampled tensors to the model device before use.
- Archival capture-tag, local-PRP, and strong-tag state remains CPU-resident and uses contiguous numeric buffers with zero-copy in-place decay. Small CUDA observation kernels and bulk asynchronous archival staging are not part of the CUDA-first claim unless complete-tick evidence beats the CPU control-plane implementation and preserves synchronization-safe checkpoint ownership.
- Sensory encoders must expose device reports, and real sensory episodes must carry encoder/device/spike-device metadata into grounded observations and preview metadata.
- Standalone sensory streams and multimodal loaders must resolve omitted devices the same way as the runtime posture: explicit device first, `MARULHO_DEVICE` second, CUDA when available, CPU only as the deterministic fallback. Directory-backed visual/audio tensors must load onto that resolved device before encoding.
- Unit tests should continue to default to CPU unless a test is explicitly marked as a CUDA scale or device test.
- The in-place steady-state column transition may be selected by checkpoint configuration only when CUDA, lightweight plasticity, and zero input-weight blend satisfy its proven execution boundary. Startup must compile all bounded candidate shapes without launching the mutating kernel. Unsupported configurations may fall back before mutation; launch-time failures must fail closed rather than retry after possible partial mutation.
- The trainer owns the in-place executor lifecycle, persistent work buffers, and execution counters. Runtime Truth must report requested and resolved mode, device, precompiled candidate counts, warmup result and latency, execution/failure counts, last execution mode, fallback reason, and the no-post-mutation-fallback policy.
- When that executor is active, its bounded single-winner decision stays on CUDA: a precompiled Triton selector writes winner, strength, and positive-activation evidence into persistent device buffers. The existing in-place transition consumes that evidence and applies fallback threshold decay, avoiding a separate host-visible positivity branch. Retained and CPU paths keep the existing PyTorch competition implementation.
- For the proven learned-chunk, zero-input-blend, one-winner shape with no context, abstraction, or binding gain, that selector also fuses predictive reference-frame voting and candidate prototype scoring. It owns a persistent previous-winner device scalar, reports execution/fallback counters, and falls back to the retained dense vote plus ordinary selector when any eligibility condition is absent.
- Exact torch-cache routing may join that selector only through checkpoint mode `predictive_route_vote_mode=fused_triton_text`. Retrieval exposes the current cache without copying, core owns the two Triton kernels, and `ColumnTransitionRuntime` owns compile-only warmup, persistent workspaces, cache-pointer refresh, execution/fallback counters, and fail-closed selection state.
- Fused route/vote is limited to text/idle ticks. Visual or audio evidence must use retained tensor routing because the global sensory variant did not establish a repeatable gain; Runtime Truth must count those sensory fallbacks. Cache shape changes disable the specialization before mutation.
- The wider checkpoint mode `predictive_route_vote_mode=cuda_graph_text` may capture a fixed-address text-tick island containing production input normalization/projection, exact fresh reconstruction distance, fused route/vote, and the in-place transition. It remains text/idle-only and must bypass graph pre-routing before any visual/audio tick.
- CUDA Graph capture must happen after checkpoint model state and routing caches are restored. Checkpoint loading defers this specialization during trainer construction, restores state, then creates `ColumnTransitionRuntime`; capturing earlier binds stale tensor addresses and must fail rather than execute.
- Retrieval cache rebuilds must preserve tensor addresses when shape and device are unchanged so fixed-address graphs remain valid. Shape/device changes or pointer replacement disable the graph before mutation; they do not trigger hidden recapture inside a cognitive tick.
- Runtime Truth must expose graph capture success/latency/count, fixed-address status, pre-route replay and sensory-bypass counts, transition replay/failure counts, graph names, and tensor device. The default remains `tensor`; graph use is checkpoint opt-in and rollback is the prior checkpoint configuration.
- The former Cortex execution path is retired from active runtime claims and is not part of the CUDA-first claim. NIM/LLM adapters were removed; future experiments must not return as runtime paths or living-substrate dependencies.
- Digital action execution records evidence in the Subcortex action ledger and must not initialize the retired external LLM/ThoughtLoop path for action-memory mirroring.
- Source and sensory observations are Subcortex-owned grounded evidence. They must be emitted from source/sensory runtime paths without requiring ThoughtLoop observation or surprise injection, and focus terms must come from Subcortex query gaps, autonomy, curiosity, concept state, or source metadata.
- Language generation, when present, should be exposed as a Subcortex Language Surface with explicit grounding/support/device evidence; it must not revive Cortex/ThoughtLoop as the active cognition substrate. The first accepted implementations are native-decode-bound responder language, which reports confidence, query overlap, evidence coverage, memory indices, and concept focus, and Cognitive Signal status language, which reports prediction error, confidence, neuromodulator pressure, and concept focus.
- Pure-SNN language projects such as NeuronSpark and Nord-AI may be used as implementation references, but MARULHO must own any language-neuron module, decoder, training loop, grounding evidence, and CUDA/sparsity telemetry before language generation can be promoted beyond a read-only readiness gate.
- A Spike Language Decoder Probe may report MARULHO-owned sparse tensor code, device placement, and grounded slot support from Subcortex Spike Readout Evidence, but it remains non-generative and must not satisfy `local_snn_language_generator_available` by itself.
- Runtime Truth may include a compact SNN-Native Language Readiness summary, but it must not embed research candidates, generation payloads, external dependency instructions, or decoder checkpoints.
- Developmental plasticity work, including growth and pruning, should report sparse structural changes and device placement before being treated as CUDA-first self-improvement evidence.
- Runtime evidence should include lightweight Subcortex Spike Health metrics from live tensor state: activity state, local spike fraction when available, win-rate EMA ranges, silence/saturation/stale-routing fractions, visible thresholds, bounded recent spike-window correlation, and explicit insufficiency status when the window is too small.
- Spike-health self-repair candidates may be surfaced to Living Loop and Policy Actuator status as advisory evidence with an explicit Self-Repair Promotion Gate, but they must not execute revival, pruning, growth, replay repair, or structural mutation from a status read.
- A dedicated Self-Repair Gate Artifact may be exposed for operator/replay/deep-sleep review, but it must remain separate from Replay Plan execution candidates and must not contain replay candidate IDs or suggested execution endpoints.
- Runtime Truth may include a compact summary of the Self-Repair Promotion Gate so liveness evidence can point to the review artifact without changing verdicts or executing repair.
- A Self-Repair Evaluation Artifact may describe isolated replay/deep-sleep evaluation requirements for ready repair candidates, but it must remain non-executable and require Runtime Truth improvement, rollback policy, and device evidence before any repair promotion.
- A Structural Plasticity Gate Artifact may combine ConceptStore growth pressure, hypercube/binding structural mutation ledgers, and CUDA/device placement into promotion readiness evidence, but it must not call observation, binding, growth, pruning, or topology refresh paths.
- Local plasticity evidence is part of Structural Plasticity readiness: local STDP eligibility traces, homeostatic synaptic scaling, inhibitory balance, spike-health state, synaptic validation, and device placement may support isolated growth/prune evaluation, but the read surface must not mutate synapses or topology.
- Structural readiness requires observed tensor placement. Concept growth pressure or binding mutation ledgers without binding/local-plasticity device keys must collect device evidence instead of opening isolated structural evaluation.
- Local plasticity readiness requires eligibility traces, homeostatic state, and non-failed synaptic validation; failed validation is pressure to monitor or repair, not evidence that growth/prune evaluation is ready.
- Runtime Truth may include a compact summary of the Structural Plasticity Gate Artifact so liveness reports expose structural-promotion pressure without embedding structural cases, device payloads, or mutation ledgers as executable instructions.
- Runtime status must expose Subcortex CUDA evidence directly so retired LLM adapters cannot be mistaken for the living-brain core.
- Runtime Truth does not use retired-path vocabulary. Active status and long-test report contracts must not emit `retired_runtime_path`, `cortex_*`, or retired evidence aliases as compatibility fields.
- Brain runtime, replay planning, and runtime evidence exports read active Subcortex evidence directly; `retired_runtime_path_snapshot` and `cortex_snapshot` must not be used for new internal call paths.
- Brain runtime snapshots must not expose `retired_runtime_path` or publish a `cortex` sibling payload for active internal consumers.
- Policy Actuator must read sleep/fatigue pressure from Subcortex Sleep Pressure and must not expose `retired_runtime_path` or `cortex_snapshot` as input arguments.
- Living Loop status and replay planning must read fatigue/sleep pressure from Subcortex Sleep Pressure; they must not publish or consume a `retired_runtime_path`, `cortex` sibling payload, or `cortex_loop_snapshot` capability.
- The retired Cortex controller name and the `retired_runtime_path` holder are deleted. No active module may expose ask/sleep/thought/action-intent helpers, factory references, query hints, lazy initialization hooks, or retired-runtime state snapshots.
- Digital action execution must keep evidence in the Subcortex Action Ledger only; retired ThoughtLoop mirroring is removed.
- `marulho.cortex.multi_cortex` is deleted. The codebase must not retain `NIMCortex`, `MultiCortex`, `create_cortex_from_env`, or `create_embedder_from_env` as importable external LLM adapter paths. Top-level `marulho.cortex` is deleted and must not export runtime, mock, memory, drive, language, ThoughtLoop, or external LLM factory entry points.
- `NIMEmbedder` is deleted from Episodic Memory. Memory indexing must not depend on remote embedding APIs, API keys, NIM request accounting, or external embedding clients; local sparse encoders may remain only as transitional machinery while SNN-native encoders mature.
- `marulho.cortex.rate_limit` is deleted with the external adapters. Remote API throttling is not a Cortex primitive and should return only as a maintained source/client concern if a real source requires it.
- `MockCortex` is deleted from production source. It must not be used in tests or docs as a language/thought substitute; retirement tests should assert factories and operator surfaces are absent or inert instead.
- Language/readout results belong to Subcortex/semantics ownership. Active modules should use `marulho.semantics.language_result.LanguageResult`; `ThoughtResult` aliases are deleted.
- Language/readout packets belong to Subcortex/semantics ownership. Active modules should use `marulho.semantics.language_packet` for `ContextPacket`, `MemoryItem`, `ReadoutMode`, and `DeliberationDepth`; `marulho.cortex.core` is deleted rather than kept as a retired compatibility shim.
- Cortex-owned static prompt templates are deleted. Future language/readout policy must live under semantics/Subcortex ownership and carry grounding evidence instead of reviving prompt-first Cortex machinery.
- The `marulho.cortex` package is deleted. New runtime, memory, drive, prompt, narrative, and language work must not add modules under that namespace.
- `ThoughtLoop()` and its module are deleted. The codebase must not keep hidden Cortex generation, sleep, or background-loop branches behind a constructor guard.
- Cognitive Signal state is a Subcortex/semantics primitive, not a ThoughtLoop primitive. Active status, language, and policy paths must import it from semantics, never from `marulho.cortex.thought_loop`.
- Active Exploration State is a Subcortex control primitive, not a ThoughtLoop-private field. ThalamicGate should store the canonical state object, normalize target text, and preserve reason/source/score so curiosity, source-focus, and future SNN deliberation modules can share it without reviving the retired loop.
- Grounding Diagnostics are Subcortex language evidence and must live outside retired `ThoughtLoop` so source/sensory support, coverage, alignment, and recovery can be reused by future SNN language/readout modules.
- Brain runtime metrics are telemetry primitives, not proof of liveness, and must live outside retired `ThoughtLoop` so output quality, grounding alignment, and latency can be reused by Subcortex-owned language/readout surfaces.
- Deterministic deliberation text merging is an active language-surface primitive and must live outside the retired `ThoughtLoop` class. Compatibility wrappers may delegate to the active helper only while historical code is being deleted.
- Feedback payloads from language-facing deliberation results are Subcortex control evidence, not Cortex ownership evidence. New code must call `emit_deliberation_feedback`; the old `emit_cortex_feedback` path is removed rather than kept as a dormant compatibility API.

The first acceleration targets are routing/index search, predictive column state updates, AdEx/neuron dynamics, binding/topographic or hypercube updates, plasticity traces, cross-modal grounding, text encoders, and sensory encoders.

## Consequences

### Positive

- The runtime can exploit GPU acceleration without making ordinary development require a GPU.
- Device behavior stays observable through existing runtime-scope and routing-index telemetry.
- CPU tests continue to protect correctness and portability.
- Future performance work has a clear target list instead of ad hoc device changes.
- The promoted in-place column executor removes repeated full-state output/writeback cost while preserving checkpoint-owned state and an explicit rollback image.

### Negative

- CUDA paths need dedicated benchmark validation because CPU-only tests cannot prove GPU performance.
- Some current Python loops around tensor operations may limit GPU benefit until vectorized.
- First-process Triton compilation can remain expensive even when disk-cache startup is fast, so deployment/install validation must distinguish cold compiler setup from steady runtime.
- Adding the selector specialization produced an empty-process Windows compile-only warmup of about `111 s` in one run, while populated-cache startup returned below one second. Runtime packaging must preserve or prewarm the compiler cache before startup is treated as production-ready.
- The fused text route/vote specialization added about `2.58 s` cold compile-only warmup in the measured process and about `0.206 ms` with a populated cache. It also keeps one 1024-score workspace and candidate buffers on CUDA, so checkpoint operators must account for startup/cache packaging and bounded extra VRAM.
- The CUDA Graph text-tick island increased the measured 1024-column checkpoint from about `10.76 MB` allocated / `26 MB` reserved VRAM to `18.88 MB` / `50 MB`. Cached capture measured about `125-207 ms`; the first uncached Triton specialization observed during development took about `56.6 s`. Deployment must prewarm or preserve compiler caches and must not hide capture behind a measured live tick.
- A fresh-process three-seed hot-window comparison improved mean throughput from `176.24` to `264.46 ticks/sec` (`1.501x`), while a real 24-token service source tick still took about `1.24 s`. This ADR therefore accepts the bounded graph island, not a claim that full service cognition is production-speed or near the maximum-throughput objective.
- Approximate routing backends need recall/latency evidence before they can replace exact search in evidence claims.
- CUDA-first does not mean moving small host-visible archival bookkeeping to the GPU. Measured launch, transfer, and synchronization cost can make CPU numeric buffers the faster production boundary.
- CUDA-first also does not mean issuing many tiny scalar encoder kernels. The bounded ingestion batch keeps emitted vectors on CUDA while amortizing construction and keeping plasticity outside source collection.
- CUDA-first does not make source-cache persistence part of the cognitive tick. Live ticks schedule bounded, content-addressed cache payloads; one Runtime Sources worker performs `torch.save` and atomic replacement, and service shutdown flushes that queue. Runtime Truth reports scheduling, completion, failure, and pending state without synchronizing CUDA.

### Neutral

- This ADR does not change public HTTP routes or runtime payload names.
- It records runtime posture, not a mandate to remove explicit CPU support for tests or CPU-only hosts.

## References

- `src/marulho/config/model_config.py`
- `src/marulho/retrieval/hnsw_index.py`
- Historical standalone `src/marulho/retrieval/ivf_router.py` was removed after a no-live-caller audit; future IVF/RaBitQ routing needs new bounded GPU-owned candidate-router evidence.
- `conftest.py`
- Lee, Dora, and Mejias, "Predictive coding with spiking neurons and feedforward gist signaling", Frontiers in Computational Neuroscience, 2024.
- "Predictive Coding with Spiking Neural Networks: a Survey", arXiv:2409.05386, 2024.
- "A flexible framework for structural plasticity in GPU-accelerated sparse spiking neural networks", arXiv:2510.19764, 2025.
- Johnson, Douze, and Jegou, "Billion-scale similarity search with GPUs", arXiv:1702.08734, 2017.
- Ootomo et al., "CAGRA: Highly Parallel Graph Construction and Approximate Nearest Neighbor Search for GPUs", arXiv:2308.15136, 2023/2024.
- Shi et al., "GPU-Native Approximate Nearest Neighbor Search with IVF-RaBitQ: Fast Index Build and Search", arXiv:2602.23999, 2026.
- Zandieh et al., "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate", arXiv:2504.19874, 2025.
- He et al., "Enhancing Audio-Visual Spiking Neural Networks through Semantic-Alignment and Cross-Modal Residual Learning", arXiv:2502.12488, 2025.
- Huang et al., "Crossmodal hierarchical predictive coding for audiovisual sequences in the human brain", Communications Biology, 2024.
- Zhou et al., "TDSNNs: Competitive Topographic Deep Spiking Neural Networks for Visual Cortex Modeling", AAAI, 2026.
- Reimann et al., "Cliques of Neurons Bound into Cavities Provide a Missing Link between Structure and Function", Frontiers in Computational Neuroscience, 2017.
- Jiang and Rao, "Dynamic predictive coding: A model of hierarchical sequence learning and prediction in the neocortex", PLOS Computational Biology, 2024.
- "HetSyn: Versatile Timescale Integration in Spiking Neural Networks via Heterogeneous Synapses", arXiv:2508.11644, 2025.
- "Spiking Neural Networks with Adaptive Membrane Time Constant for Event-Based Tracking", IEEE Transactions on Circuits and Systems for Video Technology, 2025.
- "Dynamics and Bifurcation Structure of a Mean-Field Model of Adaptive Exponential Integrate-and-Fire Networks", Neural Computation, 2025.
- "Event- and Time-Driven Techniques Using Parallel CPU-GPU Co-processing for Spiking Neural Networks", 2017.
- "Fast simulations of highly-connected spiking cortical models using GPUs", Frontiers in Computational Neuroscience, 2021.
- NVIDIA, "CUDA Graphs", CUDA Programming Guide: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs
- PyTorch, `torch.cuda.CUDAGraph`: https://docs.pytorch.org/docs/stable/generated/torch.cuda.CUDAGraph.html
- PyTorch, "Accelerating PyTorch with CUDA Graphs": https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/
