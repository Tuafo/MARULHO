# Living Brain Research Notes

This file records research anchors for current architecture work. It is not a proof that HECSN is a biological brain; it is the paper trail for the engineering direction.

## Current Anchors

### Predictive spiking substrate

- Lee, Dora, and Mejias, "Predictive coding with spiking neurons and feedforward gist signaling", 2024: supports the project posture that prediction error can be represented with spiking mechanisms and local learning, rather than treating the subcortex as a dense language model.
- "Predictive Coding with Spiking Neural Networks: a Survey", 2024: useful taxonomy for future code reviews because it separates explicit error-neuron approaches, membrane-potential error encodings, and implicit prediction-error encodings.
- "Confidence and second-order errors in cortical circuits", 2024: supports keeping predictive confidence in the Cognitive Signal alongside prediction error, because confidence-weighted errors are a plausible control variable rather than a display-only score.
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
- "Operational manifolds in spiking neural networks", 2026: supports adding lightweight health evidence for Subcortex stability, especially spike-train correlation statistics, drift indicators, reset/carry state policy, and energy/accuracy trade-off proxies.
- Grimaud, Longin, and Herzig, "SNN-Based Online Learning of Concepts and Action Laws in an Open World", 2026: supports keeping concepts, action laws, and runtime signals in the Subcortex control plane instead of routing active cognition through a static language backend.
- "Active inference and cognitive control: Balancing deliberation and habits through precision optimization", 2025: supports deriving deliberation pressure from surprise, uncertainty, control cost, and expected information gain rather than from free-form text generation.
- "Intermittent Active Inference", 2026: supports exposing control candidates as advisory replanning triggers, because replanning should fire when prediction error or expected free energy crosses a threshold rather than on every tick.

### Backend-neutral deliberation

- Putra, Marchisio, and Shafique, "SNN4Agents: a framework for developing energy-efficient embodied spiking neural networks for autonomous agents", 2024: supports treating embodied SNN control as the primary agent substrate rather than assuming language-model inference is the runtime core.
- "Towards biologically plausible model-based reinforcement learning in recurrent spiking networks by dreaming new experiences", 2024: supports moving dream/replay work toward spiking world-model mechanisms that can improve policy learning without requiring an LLM loop.
- "World Models for Cognitive Agents: Transforming Edge Intelligence in Future Networks", 2025: supports considering world models as embedded cognitive engines for prediction, planning, and causal reasoning, with language backends treated as optional interaction adapters.
- "SPikE-SSM: A Sparse, Precise, and Efficient Spiking State Space Model for Long Sequences Learning", 2024: supports exploring spiking/state-space sequence backends for long-context deliberation surfaces where an LLM would be too slow or opaque.
- "SNN-BERT: Training-efficient Spiking Neural Networks for energy-efficient BERT", 2024: supports preserving language-facing capability as a possible SNN-backed path rather than equating language with external LLM services.

### Text encoding and consolidation

- Zhu et al., "SpikeGPT: Generative Pre-trained Language Model with Spiking Neural Networks", 2023: supports treating language input as a sparse/event-coded sequence problem rather than as a detached dense text embedding concern.
- "NeuronSpark: A Spiking Neural Network Language Model with Selective State Space Dynamics", 2026: supports tracking SNN-native language generation as an emerging path, but its early loss/dialogue evidence means HECSN should first expose language as a grounded Subcortex state decoder rather than reintroducing an LLM-style cognition core.
- "SpikeMLLM: Spike-based Multimodal Large Language Models via Modality-Specific Temporal Scales and Temporal Compression", 2026: reinforces that multimodal spike-language work depends on modality-specific temporal encoding and compression; HECSN's language surface should preserve sensory timing evidence instead of flattening everything into text-first prompts.
- Liu, Liu, and Chen, "Scaling Natively-Trained Spiking Language Models to Multi-Domain Pre-training with 85% Global Activation Sparsity", 2026: supports the architectural bet that language-capable SNNs should preserve spike-gated associative lookup and sparsity, not collapse into dense transformer-style runtime claims.
- Stoeckl et al., "Learning Long Sequences in Spiking Neural Networks", 2024: supports preserving temporal sequence structure when moving text-derived traces onto accelerated tensor devices.
- Lv, Xu, and Zheng, "Spiking Convolutional Neural Networks for Text Classification", 2024: supports keeping text encoders as first-class SNN-facing modules with explicit feature/spike tensors.
- "Online Continual Learning via Spiking Neural Networks with Sleep Enhanced Latent Replay", 2025: supports the existing distinction between CPU archival replay storage and device-local replay computation.
- "Compressed Latent Replays for Lightweight Continual Learning on Spiking Neural Networks", 2024: supports keeping replay representations compact and explicit rather than treating memory consolidation as hidden model state.

### Developmental growth and pruning

- "Adaptive sparse structure development with pruning and regeneration for spiking neural networks", 2025: supports treating self-growth and pruning as a bounded structural-plasticity mechanism with explicit synaptic constraints, neuronal pruning, and regeneration rather than as open-ended architecture mutation.
- "A flexible framework for structural plasticity in GPU-accelerated sparse spiking neural networks", 2025: connects developmental plasticity to CUDA-first execution; growth/pruning evidence should include both sparse structural changes and observed device placement.
- "Self-Motivated Growing Neural Network for Adaptive Architecture via Local Structural Plasticity", 2025/2026: supports using local activation and weight-update statistics as growth/prune signals instead of central planner decisions, with reward stability as a promotion criterion.
- "Adaptively Pruned Spiking Neural Networks for Energy-Efficient Intracortical Neural Decoding", 2025: supports measuring pruning by retained task performance and efficiency gain, not just by lower parameter count.

### Novel correlations

- Spiking language models plus the retired Cortex decision imply a two-stage language roadmap: first, a grounded Subcortex Language Surface that decodes assemblies/replay/world-model state into auditable text; later, an SNN-native generator only after it can report sparsity, device placement, and support evidence. The responder's native-decode surface and the Cognitive Signal status surface are the first bridges because they attach confidence, memory support, evidence coverage, prediction error, neuromodulator pressure, and concept focus to language claims.
- Spike-based multimodal language work plus the Sensory Encoder direction imply language generation should be a decoder over time-preserving multimodal evidence. Operator UI should route "Subcortex" to spike dynamics and grounding evidence, not to a retired Cortex/Mind page.
- Text SNN work plus the CUDA-first evidence rule implies encoder reports must capture the observed placement of emitted feature vectors and spike traces, not just configured modules. This keeps language-facing telemetry grounded in runtime tensors instead of static declarations.
- Structural plasticity plus replay promotion gates imply a clean self-growth loop: propose local growth/prune candidates from surprise and replay failures, evaluate them in isolated replay experiments, then promote only if Runtime Truth, grounding support, and device telemetry improve.
- Self-motivated growth controllers plus adaptive pruning evidence imply HECSN's autonomy loop should grow topology from local stress signals and prune from sustained inactivity/redundancy, with replay-gated rollback. This is a cleaner path than a language planner that rewrites architecture.
- Structural-plasticity work plus HECSN's hypercube binding layer imply that sparse topology mutation must be ledgered at the edge level: added/removed hub outreach edges are the first auditable self-growth/pruning evidence before broader model self-modification is allowed.
- Homeostatic plasticity and adaptive pruning/regeneration work plus Subcortex Spike Health imply the first self-repair surface should be advisory: convert silence, saturation, stale routing, and over-correlation into review candidates, then require deep-sleep/replay/operator promotion before any revive, prune, or structural mutation occurs.
- GPU sparse-SNN work plus Terminus Runtime Truth implies CUDA claims should favor sparse state transitions and topology updates over dense "everything on GPU" rewrites; efficient living-brain behavior needs event sparsity preserved in the evidence report.
- Operational-manifold work plus Runtime Truth implies liveness should include internal stability evidence, not only endpoint availability: a running Subcortex can still be unhealthy if spike activity is silent, saturated, or over-correlated.
- Operational-manifold work plus HECSN's current competitive-column state implies a staged stability metric: expose silent/saturated/stale routing evidence from live EMAs and spike fractions, then add bounded windowed correlation evidence from recent winner vectors. This still avoids over-claiming full manifold health from scalar endpoint/status signals.
- Confidence-weighted predictive-coding work plus the retired Cortex decision implies the **Cognitive Signal** should be the active control packet for future deliberation: prediction error, confidence, neuromodulator mirrors, source, and concept candidates should be observable without requiring a ThoughtLoop consumer.
- Open-world SNN concept/action-law learning plus HECSN's retired Cortex decision implies naming matters in code: `cognitive_signal` should be the primary runtime contract, while `cortex_signal` can remain only as a temporary compatibility alias for retired internals.
- Active-inference cognitive-control work plus Cognitive Signal implies a clean first Subcortex Deliberation surface: convert prediction error, confidence, neuromodulator pressure, and concept focus into ranked control hypotheses, then require replay/policy/operator evidence before promotion.
- Intermittent-planning active inference plus HECSN Policy Actuator implies control candidates should be attached to advisory policy status as non-executable context. That lets operators and replay gates see why the Subcortex wants to replan without allowing a control candidate to become an action.
- Replay-gated adaptation work plus advisory control candidates implies every candidate needs an explicit promotion gate. Readiness for replay review is not readiness for action or fact promotion; those remain separate gates with operator/evidence requirements.

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
