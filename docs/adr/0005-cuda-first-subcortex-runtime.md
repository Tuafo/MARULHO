# ADR 0005: Prefer CUDA for tensor-heavy subcortex runtime work

## Status

Accepted

## Context

The project goal is shifting from a merely executable cortex-subcortex runtime toward a more continuous "living brain" system. The existing implementation already routes most subcortical computation through PyTorch tensors and `HECSNConfig.resolve_device()`, with `device="auto"` selecting CUDA when available. Routing also has a `torch_topk` backend that becomes the automatic search path on CUDA, while FAISS HNSW and exact cosine remain CPU-friendly alternatives.

Unit tests intentionally force `HECSN_DEVICE=cpu` in `conftest.py` so they remain deterministic and do not fail on machines without a GPU. That creates a useful split: production/runtime experiments should use CUDA when available, but correctness tests should not require CUDA.

Recent literature supports this direction: predictive coding with spiking neurons remains active work, and sparse SNN/GPU acceleration is a natural fit when the implementation can preserve event sparsity rather than densifying every operation.

## Decision

Adopt a CUDA-first posture for tensor-heavy subcortical runtime paths:

- Runtime and benchmark code should keep using `HECSNConfig.resolve_device()` rather than hard-coding CPU.
- New tensor-heavy modules should accept an explicit `torch.device` and move persistent tensors onto that device at construction.
- CUDA claims must be proven by runtime telemetry or benchmark output that includes the actual search/device backend.
- Routing indexes must report actual cache placement, not just configured search intent. Tensor-backed backends should expose cache readiness and tensor devices in `stats()`.
- Predictive columns, competitive columns, plasticity circuits, and binding layers must expose live tensor device reports in the model runtime scope.
- Neuron dynamics modules such as AdEx must expose voltage, adaptation, and spike-timing tensor devices directly, and checkpoint restore must place live neuron tensors back on the selected runtime device.
- Cross-modal grounding and sparse binding variants, including spatial and hypercube binding, must expose live tensor and topology device reports when enabled.
- Context and abstraction layers must expose live tensor device reports because they control routing gain, curiosity pressure, and top-down feedback.
- Text encoders must keep tensor-heavy state on the runtime device, including learned chunking codebooks, semantic bucket embeddings, adapter tensors, emitted feature vectors, and spike traces. Python string parsing, segmentation lists, and hash-loop control flow remain CPU/control-plane work.
- The Memory Store must report its device boundary explicitly: archival replay records remain CPU storage, while trainer replay computation moves sampled tensors to the model device before use.
- Sensory encoders must expose device reports, and real sensory episodes must carry encoder/device/spike-device metadata into grounded observations and preview metadata.
- Unit tests should continue to default to CPU unless a test is explicitly marked as a CUDA scale or device test.
- The former Cortex execution path is retired from active runtime claims and is not part of the CUDA-first claim. NIM/LLM adapters are external compatibility or experiment code, not the living substrate.
- Runtime status must expose Subcortex CUDA evidence directly so retired LLM adapters cannot be mistaken for the living-brain core.

The first acceleration targets are routing/index search, predictive column state updates, AdEx/neuron dynamics, binding/topographic or hypercube updates, plasticity traces, cross-modal grounding, text encoders, and sensory encoders.

## Consequences

### Positive

- The runtime can exploit GPU acceleration without making ordinary development require a GPU.
- Device behavior stays observable through existing runtime-scope and routing-index telemetry.
- CPU tests continue to protect correctness and portability.
- Future performance work has a clear target list instead of ad hoc device changes.

### Negative

- CUDA paths need dedicated benchmark validation because CPU-only tests cannot prove GPU performance.
- Some current Python loops around tensor operations may limit GPU benefit until vectorized.
- Approximate routing backends need recall/latency evidence before they can replace exact search in evidence claims.

### Neutral

- This ADR does not change public HTTP routes or runtime payload names.
- It records runtime posture, not a mandate to remove CPU fallbacks.

## References

- `src/hecsn/config/model_config.py`
- `src/hecsn/retrieval/hnsw_index.py`
- `src/hecsn/retrieval/ivf_router.py`
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
