# Living Brain Research Notes

This file records research anchors for current architecture work. It is not a proof that HECSN is a biological brain; it is the paper trail for the engineering direction.

## Current Anchors

### Predictive spiking substrate

- Lee, Dora, and Mejias, "Predictive coding with spiking neurons and feedforward gist signaling", 2024: supports the project posture that prediction error can be represented with spiking mechanisms and local learning, rather than treating the subcortex as a dense language model.
- "Predictive Coding with Spiking Neural Networks: a Survey", 2024: useful taxonomy for future code reviews because it separates explicit error-neuron approaches, membrane-potential error encodings, and implicit prediction-error encodings.
- "Dynamic predictive coding: A model of hierarchical sequence learning and prediction in the neocortex", 2024: supports the existing emphasis on hierarchical temporal prediction, sequence learning, and context-sensitive dynamics.

### Sparse/GPU execution

- "A flexible framework for structural plasticity in GPU-accelerated sparse spiking neural networks", 2025: supports the idea that structural plasticity and sparse neural communication should remain first-class when optimizing for GPU.
- "Sparse Spiking Neural-like Membrane Systems on Graphics Processing Units", 2024: reinforces that sparse spike-style workloads need GPU-aware representation choices, not only generic dense matrix acceleration.
- "Prosperity: Accelerating Spiking Neural Networks via Product Sparsity", 2025: useful for future acceleration design because it frames SNN speedups around activation sparsity and product sparsity.
- "Event- and Time-Driven Techniques Using Parallel CPU-GPU Co-processing for Spiking Neural Networks", 2017: supports treating AdEx-style neuron dynamics as a real GPU simulation surface, with accuracy/performance trade-offs that depend on integration method.
- Knight et al., "Fast simulations of highly-connected spiking cortical models using GPUs", 2021: supports reporting neuron and spike-delivery placement separately; large AdEx networks can benefit from GPU execution, but performance claims require observed runtime evidence.

### Neuron dynamics

- "Dynamics and Bifurcation Structure of a Mean-Field Model of Adaptive Exponential Integrate-and-Fire Networks", 2025: reinforces that AdEx voltage/adaptation state is not just an implementation detail; it is the dynamical substrate being claimed.
- "Reproduction of AdEx dynamics on neuromorphic hardware through data embedding and simulation-based inference", 2024: supports keeping AdEx state inspectable and serializable so hardware/runtime differences can be compared against observed dynamics.

### GPU routing and vector search

- Johnson, Douze, and Jegou, "Billion-scale similarity search with GPUs", 2017: establishes GPU k-selection, brute-force search, approximate search, and compressed-domain product-quantization search as first-class designs for high-dimensional retrieval.
- Ootomo et al., "CAGRA: Highly Parallel Graph Construction and Approximate Nearest Neighbor Search for GPUs", 2023/2024: supports treating CPU HNSW as only one point in the design space; graph ANN construction and search can be redesigned around GPU parallelism.
- Shi et al., "GPU-Native Approximate Nearest Neighbor Search with IVF-RaBitQ: Fast Index Build and Search", 2026: supports the direction of fusing IVF-style cluster routing with low-bit quantization and GPU-native distance kernels for high-throughput routing.
- Zandieh et al., "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate", 2025: supports keeping online quantized routing as an experimental path, but it still needs observed device/backend telemetry before being used as CUDA evidence.

### Multimodal sensory grounding

- "Crossmodal hierarchical predictive coding for audiovisual sequences in the human brain", 2024: supports treating audio/visual sensory transitions as predictive evidence rather than detached media annotations.
- "Enhancing Audio-Visual Spiking Neural Networks through Semantic-Alignment and Cross-Modal Residual Learning", 2025: supports explicit semantic alignment between audio and visual SNN features.
- "Spike-HAR++: an energy-efficient and lightweight parallel spiking transformer for event-based human action recognition", 2024: reinforces event-based vision as a natural match for sparse spiking computation.

### Topographic and high-dimensional binding

- "TDSNNs: Competitive Topographic Deep Spiking Neural Networks for Visual Cortex Modeling", 2026: supports maintaining topographic binding as a first-class subcortical structure rather than treating all column interactions as dense all-to-all associations.
- Reimann et al., "Cliques of Neurons Bound into Cavities Provide a Missing Link between Structure and Function", 2017: supports keeping high-dimensional sparse binding structures observable, because the architecture's hypercube path is a topology claim as much as a performance claim.

### Context and abstraction dynamics

- Jiang and Rao, "Dynamic predictive coding: A model of hierarchical sequence learning and prediction in the neocortex", 2024: supports treating context as a dynamic sequence-prediction state that modulates lower-level gain rather than as passive metadata.
- "HetSyn: Versatile Timescale Integration in Spiking Neural Networks via Heterogeneous Synapses", 2025: supports the adaptive-context direction where heterogeneous temporal integration is a runtime mechanism, not merely a scalar hyperparameter.
- "Spiking Neural Networks with Adaptive Membrane Time Constant for Event-Based Tracking", 2025: supports keeping learned/adaptive timescale state observable when claiming temporal CUDA acceleration.

### Runtime self-model and observability

- "Active Inference for Learning and Development in Embodied Neuromorphic Agents", 2024: supports treating runtime status as part of the agent's developmental loop, not just external logging.
- "Active Inference as a Model of Agency", 2024: supports making differences in world model, uncertainty, and exploration/exploitation posture explicit in reports.
- "Self-evolving Embodied AI", 2026: supports requiring memory, task, environment, embodiment, and model adaptation evidence before claiming self-evolving or living-brain behavior.

### Backend-neutral deliberation

- Putra, Marchisio, and Shafique, "SNN4Agents: a framework for developing energy-efficient embodied spiking neural networks for autonomous agents", 2024: supports treating embodied SNN control as the primary agent substrate rather than assuming language-model inference is the runtime core.
- "Towards biologically plausible model-based reinforcement learning in recurrent spiking networks by dreaming new experiences", 2024: supports moving dream/replay work toward spiking world-model mechanisms that can improve policy learning without requiring an LLM loop.
- "World Models for Cognitive Agents: Transforming Edge Intelligence in Future Networks", 2025: supports considering world models as embedded cognitive engines for prediction, planning, and causal reasoning, with language backends treated as optional interaction adapters.
- "SPikE-SSM: A Sparse, Precise, and Efficient Spiking State Space Model for Long Sequences Learning", 2024: supports exploring spiking/state-space sequence backends for long-context deliberation surfaces where an LLM would be too slow or opaque.
- "SNN-BERT: Training-efficient Spiking Neural Networks for energy-efficient BERT", 2024: supports preserving language-facing capability as a possible SNN-backed path rather than equating language with external LLM services.

### Text encoding and consolidation

- Zhu et al., "SpikeGPT: Generative Pre-trained Language Model with Spiking Neural Networks", 2023: supports treating language input as a sparse/event-coded sequence problem rather than as a detached dense text embedding concern.
- Stoeckl et al., "Learning Long Sequences in Spiking Neural Networks", 2024: supports preserving temporal sequence structure when moving text-derived traces onto accelerated tensor devices.
- Lv, Xu, and Zheng, "Spiking Convolutional Neural Networks for Text Classification", 2024: supports keeping text encoders as first-class SNN-facing modules with explicit feature/spike tensors.
- "Online Continual Learning via Spiking Neural Networks with Sleep Enhanced Latent Replay", 2025: supports the existing distinction between CPU archival replay storage and device-local replay computation.
- "Compressed Latent Replays for Lightweight Continual Learning on Spiking Neural Networks", 2024: supports keeping replay representations compact and explicit rather than treating memory consolidation as hidden model state.

## Engineering Implications

- Keep **Subcortex** tensor state on the configured `torch.device`.
- Preserve sparse/event structure where possible before reaching for dense CUDA kernels.
- Record actual device/backend evidence in reports before claiming CUDA acceleration.
- For routing, separate configured intent (`search_device`) from observed cache placement (`torch_vector_cache_device`, `tq_*_device`) and benchmarked execution.
- For predictive/plasticity/binding work, local traces and eligibility state are the auditable CUDA surface. Device reports should prove where those live tensors reside before performance claims.
- For neuron dynamics, report membrane voltage, adaptation, and spike-time tensors directly, and preserve them across checkpoints without silently restoring to CPU.
- For cross-modal grounding, report the live devices of text/visual/audio traces, cross-modal weights, and confidence state.
- For spatial or hypercube binding, report topology tensor devices as well as learned binding-state devices; sparse topology claims need observable placement.
- For context and abstraction, report recurrent/context tensors, adaptive timescales, slow abstraction state, and top-down feedback tensors because they directly gate routing and curiosity.
- For text encoders, report learned chunking codebooks, semantic bucket embeddings, adapters, feature vectors, and spike traces on the selected device while keeping string parsing and segmentation as control-plane work.
- Keep memory consolidation storage CPU-resident unless the operation is active replay computation. Archival buffers, tags, timestamps, and raw windows are evidence records; replay tensors should move to the model device when consumed.
- Carry sensory encoder device metadata into grounded observations and previews so CUDA sensory claims are auditable.
- Surface trainer-owned encoder device reports in the operator-facing runtime evidence, because model-only CUDA scope is incomplete when the active input encoder is owned by the trainer/service layer.
- Treat **Living Brain** as an evidence-gated runtime target: tick, train, think, replay, act, sleep, and runtime truth must be observable.
- Treat the former **Cortex Backend** as retired from active runtime claims. LLM/NIM adapters can remain only as compatibility or experiment code while Subcortex/world-model paths become the real cognition surface.
- Make weak paths retireable: latency, external dependency, grounding quality, and liveness contribution should be visible before a path earns permanence.
