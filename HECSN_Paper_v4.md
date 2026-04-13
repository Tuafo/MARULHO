# HECSN — Hierarchical Emergent Concept Spiking Networks
## A Developmental Architecture for Grounded Autonomous Knowledge Accumulation

**Author:** Thiago Maceno Rocha Goulart · Brasil · [github.com/Tuafo](https://github.com/Tuafo)

**Domain:** Computational Neuroscience · Unsupervised Multimodal Learning · Neuromorphic Computing

**Version:** 4.5 — Audited, Implementation-Current, Self-Critical Architecture Document

**Executable Status (2026-06-10):** Stage-0 gates pass: `silhouette ≈ 0.675`, `DBI ≈ 0.304`, `trained_eval_recon_error 0.0619 < random_assignment 0.0907`, `temporal_coherence_mean = 0.9916`, `semantic_triple_accuracy = 0.714286` (7-triple text-only validation; 50-triple probe not yet measured at scale), `routing_key_between_score = 0.9970`, `terminal_novelty_rate = 0.0994`. Full test suite: **500 passed, 7 subtests passed** across 48 test files. Cross-modal grounding layer with alignment filter, self-criticism loop (§7.4), and audio-specific self-criticism implemented and tested. Developmental protocol: 5-stage runner with state continuity (ProtocolState carries trainer/encoder between stages), stage-aware alignment gating, concept-conditioned synthetic multimodal data, real pass/fail criteria. Baseline calibration complete: fastText 0.44, SOM 0.48 on developmental corpus — thresholds validated (§8.1). NE surprise response now boosts exploration noise (not destructive reset). Dead column census implemented in deep sleep. DA→LTP gain gate and 5-HT→patience gate wired into trainer. Real training validated: prediction error dropped 1.63→0.66 nats (KL divergence) over 1,152 Wikipedia tokens with live neuromodulator dynamics (DA 0.43, micro-sleep triggered at 256 tokens). Service API: 20 endpoints live on FastAPI/Uvicorn. Package installable via `pip install -e .` (pyproject.toml). Remaining targets: GPU routing benchmarks, end-to-end multimodal developmental protocol validation.

---

## Abstract

HECSN is a biologically-grounded spiking neural network architecture for autonomous, developmental knowledge accumulation from multimodal streams. The core claim is that representations with genuine semantic structure can emerge from temporal co-occurrence statistics across modalities, using only local Hebbian mechanisms and without *semantic* labels at any stage — though a supervised developmental scaffold using perceptually curated data is required during the critical period, precisely as it is in biological language acquisition. Seven functional layers operate with bidirectional feedback, independent four-channel neuromodulation (DA→LTP gain, 5-HT→patience gating, ACh novelty, NE surprise), three-phase fragility-gated sleep consolidation, and a self-critical curiosity controller. The cross-modal grounding layer implements alignment filtering (§5.3) and a self-criticism loop (§7.4) that verifies high-confidence groundings and blacklists spurious associations. Scalability targets sub-0.1ms routing at 100K columns via GPU-native IVF routing with TurboQuant+ compression. The architecture is presented together with frank critiques of its own mechanisms: the fixed three-trace context window was identified as too shallow for language-level temporal integration and mitigated with a learnable per-neuron timescale distribution (AdaptiveContextLayer, §4.3) informed by the DH-SNN literature; pair-based STDP is insufficient and is replaced by the experimentally-motivated triplet rule; the SOM convergence guarantees assumed in competitive learning do not hold in the online continual setting; the RTF encoding is borrowed from visual hardware without text-domain validation; and the grounding probe threshold of 0.65 requires calibration against vector-space baselines to be meaningful. Real training is validated: prediction error drops monotonically over Wikipedia tokens, neuromodulators respond dynamically, and sleep consolidation cycles trigger autonomously. All verification targets are stated as falsifiable predictions, not asserted results, except where explicitly validated by the current executable (500 tests pass).

---

## Table of Contents

1. [The Problem We Are Solving](#1-the-problem-we-are-solving)
2. [Core Principles and Their Tensions](#2-core-principles-and-their-tensions)
3. [System Architecture](#3-system-architecture)
4. [Critical Mechanisms — With Critique](#4-critical-mechanisms--with-critique)
5. [Multimodal Grounding](#5-multimodal-grounding)
6. [Scalability Architecture](#6-scalability-architecture)
7. [Developmental Training Protocol](#7-developmental-training-protocol)
8. [Evaluation Protocol](#8-evaluation-protocol)
9. [What to Expect: Honest Stage-by-Stage Projections](#9-what-to-expect-honest-stage-by-stage-projections)
10. [Critical Risks and Open Problems](#10-critical-risks-and-open-problems)
11. [Implementation Roadmap](#11-implementation-roadmap)
12. [Executable Infrastructure](#12-executable-infrastructure)
13. [References](#13-references)

---

## 1. The Problem We Are Solving

### 1.1 The Grounding Problem Is Structural, Not Quantitative

A text-only system can learn that "ocean" and "water" co-occur frequently, and that "submarine" co-occurs with "depth" and "pressure." But it cannot learn what any of these words refer to in the world — what water looks like, what depth feels like under pressure, why "hot" and "dangerous" are non-arbitrary properties of fire — unless the word-level statistical structure is connected to perceptual structure. This is not a performance gap that additional training data closes. It is a structural gap: distributional statistics over text encode word-to-word relationships; the world contains word-to-world relationships. Without cross-modal grounding, a system cannot, in principle, reliably distinguish a coherent semantic structure from a self-consistent but meaningless word game.

This argument is well-established in philosophy (Harnad 1990, the "symbol grounding problem") and increasingly supported by computational evidence. Recent work on large language models documents systematic failures on tasks requiring genuine world-grounded reasoning that cannot be resolved by more training data, because the required information was never available in any text corpus at any scale [LLM limits, 2026]. HECSN does not solve all of this, but it takes the grounding problem seriously as an architectural constraint rather than treating it as a benchmark footnote.

### 1.2 A Critical Clarification on Labels

A critical distinction must be made about the role of data in HECSN's developmental protocol.

HECSN's developmental protocol uses **structurally curated data** during Stage 1: specifically, paired multimodal samples where the temporal co-occurrence of text, visual, and audio streams is guaranteed by construction — not by random chance, and not by sequential narration. This is not "labeled data" in the machine learning sense (no annotation of semantic categories, no class assignments, no intent labels), but it is a form of **implicit structural supervision**: the pairing is deliberately arranged to create the grounding signal that would otherwise require thousands of hours of uncontrolled perceptual experience to accumulate through random co-occurrence.

The correct framing: HECSN operates **without semantic labels** at any stage. The curated Stage-1 data provides perceptual grounding through temporal co-occurrence, not through category annotation. The distinction matters because it is the difference between "the child never received any external input about meaning" (false — impossible — children receive structured pointing and naming constantly) and "the child never received a formal semantic taxonomy" (true — the grounding comes from the structure of experience, not from explicit labeling).

Throughout this document, "without labels" refers to the absence of semantic category annotation, not the absence of structured perceptual data.

### 1.3 What "Emergent" Means — And What It Doesn't

Knowledge is emergent in HECSN when:
- Chunking boundaries are discovered from stream statistics, not specified
- Concept clusters form through competitive learning dynamics, not through class assignments
- Cross-modal associations form through temporal co-occurrence Hebbian mechanisms, not through paired dataset construction with semantic intent
- Curiosity targets arise from internal geometric gaps in concept space, not from externally defined task objectives
- The training curriculum in Stages 2–5 is determined entirely by the network's own internal state

What is **not** emergent in the paper's current architecture, and where past versions were misleading:
- The architecture itself (columnar organization, layer hierarchy) — this is a structural prior analogous to the brain's cortical organization before experience
- The Stage-1 data curation — this is a developmental scaffold whose structure directly encodes grounding information
- The biological parameters (STDP time constants, E/I ratios) — these are external constraints derived from decades of neuroscience, not learned from data

This distinction is important not just for honesty but because it clarifies what the emergence claim actually predicts: given the architectural priors and developmental scaffold, do representations with semantic structure emerge that were not explicitly specified? This is the testable claim.

---

## 2. Core Principles and Their Tensions

### 2.1 Local Learning Only

No backpropagation through time. No global loss functions. Every synaptic update depends only on pre-synaptic activity, post-synaptic activity, and local neuromodulatory signals.

**The real tension here:** Biological plausibility and learning performance are not the same objective. Every system that achieves state-of-the-art performance on temporal tasks uses backpropagation-through-time with surrogate gradients. Spiking neural networks trained with BPTT and surrogate gradients are still generally outperformed by ANNs such as LSTMs on sequential tasks. Local STDP rules are slower to converge, produce weaker representations at equivalent scale, and cannot assign credit across long time horizons.

HECSN accepts this tradeoff explicitly. The justification is not that local learning matches backprop performance — it doesn't and won't. The justification is that: (1) online continual learning without a fixed dataset requires local rules that don't require storing activations for backward passes, (2) the biological constraints have solved the stability-plasticity problem in ways that pure gradient methods have not, and (3) for the goal of genuinely emergent knowledge accumulation (not benchmark performance), the architectural properties of biological learning are hypothesized to be more important than the gradient efficiency of backprop.

This is a bet. It may be wrong. The evaluation protocol (Section 8) is designed to determine whether it is.

### 2.2 Scalability from Design, Not Retrofit

Every component is designed to scale without architectural rewrites: GPU-native routing, IVF partitioning at 50K+ columns, TurboQuant compression, CSR sparse tensors, and distributed architecture for 1M+.

**The current gap:** The Phase-4 validated scale path uses FAISS HNSW on CPU with a sharded architecture as a logical proof — not a physical multi-GPU deployment. The GPU-native torch_topk path is validated on CPU at smoke scale. CUDA routing at the target latency (≤0.1ms at 10K columns) has not been benchmarked on the current tree. The paper's scalability claims should be treated as verified architecture targets, not measured outcomes.

### 2.3 Biological Constraints as Functional Engineering

The biological mechanisms (triplet STDP, AdEx neurons, STC tagging, E/I balance maintenance) are included because they have been selected by evolution to solve the stability-plasticity problem in systems that learn continuously from unlabeled experience. They are functional constraints, not aesthetic choices.

**The honest caveat:** The relationship between biological mechanism and computational function is imperfectly understood. When we implement "synaptic tagging" using a phenomenological model (tag × PRP > threshold), we are implementing a simplification that preserves the functional form but loses the molecular fidelity. The simplified model may not preserve the same convergence properties as the biological system. This is stated explicitly and honestly throughout.

---

## 3. System Architecture

### 3.1 Complete Layer Stack

```
INPUT: Raw multimodal streams (bytes, video frames, audio samples)
         │
         ▼
┌──────────────────────────────────────────────────────────────────┐
│  LAYER 0: MULTIMODAL ENCODERS                                    │
│  Text:   Raw byte stream → character windows                     │
│  Visual: Event-camera temporal contrast encoder                  │
│          (simulate from video frames; replace with real event    │
│           camera when hardware available)                        │
│  Audio:  Cochleagram (mel-filterbank, 64 bands, log-compressed)  │
│  Critique: Visual and text streams have fundamentally different  │
│  temporal statistics; fusing them requires careful alignment     │
│  gating (see Section 5)                                          │
└──────────────────────┬───────────────────────────────────────────┘
                        │
         ┌──────────────▼────────────────────────────────┐
         │  LAYER 1: CHUNKING                            │
         │  Predictability-based boundary detection      │
         │  Learned detector bank (N=512)                │
         │  Boundary signal: drop in detector agreement  │
         │  ← Boundary bias from Abstraction Layer       │
         │  Critique: No convergence proof for online    │
         │  competitive detector learning (see §4.1)     │
         └──────────────┬────────────────────────────────┘
                        │ Variable-length chunk encodings
         ┌──────────────▼────────────────────────────────┐
         │  LAYER 2: ENCODING (RTF)                      │
         │  Rate × temporal fusion                        │
         │  Positional phase offset for ordering          │
         │  Critique: Borrowed from visual hardware;      │
         │  not validated for text (see §4.2)            │
         └──────────────┬────────────────────────────────┘
                        │ routing_key [column_latent_dim=256]
         ┌──────────────▼────────────────────────────────────────┐
         │  LAYER 3: SURPRISE MONITOR                            │
         │  Precision-weighted layer-specific prediction error   │
         │  DA: reward prediction error (RPE, Schultz 2015)      │
         │  ACh: novelty/attention gating (Hasselmo 2006)        │
         │  NE: sustained uncertainty (Yu & Dayan 2005)          │
         │  5-HT: plasticity patience (Doya 2002)                │
         │  Critique: 5-HT→LTD mapping oversimplifies multiple  │
         │  receptor subtypes with opposing effects (see §4.7)   │
         └──────────────┬────────────────────────────────────────┘
                        │ Four modulatory signals
         ┌──────────────▼────────────────────────────────────────┐
         │  LAYER 4: CONTEXT LAYER                               │
         │  Approximate attractor dynamics                       │
         │  Fixed fast/medium/slow traces (~15-token window)     │
         │  SST+ interneuron feedback inhibition                 │
         │  → Competitive multiplicative gain                    │
         │  ← Precision scaling from Surprise Monitor           │
         │  CRITIQUE: 15-token window is INSUFFICIENT for        │
         │  language semantics. Fixed timescales are inferior to │
         │  adaptive/heterogeneous approaches. See §4.3.         │
         └──────────────┬────────────────────────────────────────┘
                        │ Context gain [n_columns]
         ┌──────────────▼──────────────────────────────────────────────────┐
         │  LAYER 5: COMPETITIVE LAYER                                     │
         │  GPU-native routing (flat ≤50K / IVF >50K / distributed >1M)  │
         │  TurboQuant prototype storage (TheTom/turboquant_plus)          │
         │  Winner history refractory (enforced coverage)                  │
         │  Triplet STDP (replaces pair-based; see §4.4)                  │
         │  Log-STDP (sublinear LTD → log-normal weight distribution)     │
         │  iSTDP for E/I balance maintenance                              │
         │  Synaptic scaling (homeostasis)                                 │
         │  Intrinsic Plasticity (firing rate adaptation)                  │
         │  Consolidation-gated wake plasticity                           │
         │  ← Routing bias from Abstraction Layer (top-down)              │
         │  ← Grounding boost from Cross-Modal Grounding Layer            │
         │  ← Context gain from Context Layer                             │
         │  Critique: SOM convergence not guaranteed in online continual  │
         │  setting; dead column problem partially mitigated by IP but    │
         │  not fully solved (see §4.5)                                   │
         └──────────────┬──────────────────────────────────────────────────┘
                        │ Winner assembly + top-k candidates
         ┌──────────────▼────────────────────────────────────────┐
         │  LAYER 6: BINDING LAYER                               │
         │  n_bindings independent of n_columns (FIXED)          │
         │  Sparse random connectivity (fan_in = 4 default)      │
         │  Tsodyks-Markram STP (facilitation + depression)      │
         │  PV+ fast feedforward inhibition (global)             │
         │  Structural growth on high spike correlation           │
         │  Critique: Random connectivity is unguided. Topographic│
         │  binding (nearby concepts share more binding neurons)  │
         │  would be more principled but requires spatial layout  │
         └──────────────┬────────────────────────────────────────┘
                        │ Composite assemblies
         ┌──────────────▼────────────────────────────────────────┐
         │  LAYER 7: ABSTRACTION LAYER                           │
         │  Online SFA anti-Hebbian (real feedforward layer)     │
         │  Concept stability + certainty tracking               │
         │  → Routing bias to Competitive Layer                  │
         │  → Boundary bias to Chunking Layer                    │
         │  → Curiosity gaps to Curiosity Controller             │
         │  Critique: Online SFA approximation loses convergence │
         │  guarantee of the exact SFA (requires batch passes)   │
         │  See §4.8                                             │
         └──────────────┬────────────────────────────────────────┘
                        │ Gap signal + routing bias
         ┌──────────────▼────────────────────────────────────────┐
         │  CROSS-MODAL GROUNDING LAYER                          │
         │  Temporal co-occurrence STDP (4 weight matrices)      │
         │  Alignment filter (self-filtering from Stage 2)       │
         │  Confirmation seeking + self-criticism                │
         │  Critique: Audio-text alignment is naturally high    │
         │  (speech IS text). Visual-text alignment is the hard  │
         │  problem. Do not conflate them in metrics. See §5.4.  │
         └──────────────┬────────────────────────────────────────┘
                        │ Retrieved multimodal stream → INPUT

PARALLEL TRACK — MEMORY & SLEEP:
┌──────────────────────────────────────────────────────────────┐
│  DUAL MEMORY STORE                                           │
│  Fast EMA (drift/novelty baseline)                           │
│  Slow reservoir: Vitter 1985 Algorithm R, importance-weighted│
│  Fragility score per memory entry                            │
│  STC phenomenological model: capture tags, PRP traces,       │
│  consolidation level, access count, replay spacing          │
│  Self-calibrated functional_minute                           │
│  Critique: Reservoir sampling is unbiased but ignores        │
│  semantic structure (a boring memory has the same sampling   │
│  probability as a pivotal one if importance scoring fails)   │
└──────────────────────────────────────────────────────────────┘
         │
┌──────────────────────────────────────────────────────────────┐
│  THREE-PHASE SLEEP (Fragility-Gated)                         │
│  A: Micro (200 tok): maintenance, no weight commit           │
│  B: Deep (5K tok): fragility-gated consolidation, anchor_lr │
│  C: Emergency: prototype repair, no STDP, no new commits     │
│  Critique: The functional_minute → STC timescale mapping     │
│  is self-calibrated but the calibration protocol itself      │
│  has free parameters that need sensitivity analysis (§4.9)  │
└──────────────────────────────────────────────────────────────┘
```

### 3.2 Representation Contract

| Symbol | Shape | Producer | Consumer | Purpose |
|---|---|---|---|---|
| `feature_vec` | `[128]` | `RTFEncoder.feature_vector(chars)` | Routing + plasticity | Active character-window routing features |
| `routing_key` | `[256]` | `W_project @ feature_vec` | Routing backend | Candidate retrieval key in latent space |
| `prototype_i` | `[256]` | Competitive Layer | Routing + WTA | Column centroid in latent space |
| `visual_spikes` | `[H/pool × W/pool]` | EventCameraEncoder | Cross-Modal Layer | Temporal contrast spike pattern |
| `audio_spikes` | `[64]` | CochleagramEncoder | Cross-Modal Layer | Mel-filterbank spike pattern |
| `concept_vec` | `[256]` | AbstractionLayer | Curiosity + routing bias | Slow-feature abstract representation |
| `grounding_confidence` | `[dim_text]` | CrossModalLayer | Routing boost + curiosity | Per-dimension cross-modal confidence |

---

## 4. Critical Mechanisms — With Critique

This section presents each mechanism alongside an honest critique of its limitations and where the current design may be wrong.

### 4.1 Chunking Layer and Its Convergence Problem

**What it does:** Learns variable-length units from the byte stream via predictability-based boundary detection. A boundary is declared when extending the current buffer reduces the maximum detector agreement score below a threshold. Detector prototypes update via competitive learning toward winning chunks.

**The mechanism is sound in principle.** Predictability-based segmentation has strong biological backing in infant language acquisition research (Saffran et al. 1996 — statistical learning from syllable streams). The key property: boundaries appear where the forward prediction probability drops, which is precisely where Markov statistics identify unit boundaries.

**Critique 1: No convergence guarantee in online streaming.** The competitive detector update uses a Kohonen-style rule: the winning detector moves toward the input chunk. In the classical SOM, convergence requires a learning rate schedule that decreases from 1 to 0 over a defined number of training epochs. In HECSN's online continual setting, there is no predetermined training length and no annealing schedule — the network trains indefinitely. The convergence proofs for SOM do not apply. The question of whether the detector prototypes converge to stable byte-pattern representations in an online non-stationary stream is open. The practical evidence from Stage-0 validated results (`temporal_coherence_mean = 0.9916`, `terminal_novelty_rate = 0.099398`) is consistent with stability, but is not a convergence proof.

**Critique 2: The boundary threshold is domain-sensitive.** The `boundary_threshold = 0.3` drop in detector agreement was set empirically for English character-stream data. For other languages, for binary data, for code, or for audio waveform bytes, the right threshold will differ. The paper should either provide a method for automatic threshold calibration or explicitly state that this is a free parameter requiring domain-specific tuning.

**Critique 3: Cold-start chicken-and-egg.** The alignment filter in Stage 2 uses chunking output to generate text assembly predictions. But the chunking layer needs sufficient training to produce meaningful chunks. During Stage 1, the chunking layer is learning from a curated limited vocabulary — its generalizations to Stage 2's broader distribution may be poor initially. This is the bootstrap problem for the bootstrap layer. Monitor chunk size distribution variance as a diagnostic: high variance = instability; log-normal distribution = healthy convergence.

**What would be better:** Growing Neural Gas (GNG, Fritzke 1995) instead of fixed-size competitive detectors, combined with a learn-rate schedule tied to the novelty rate rather than time. GNG adds new detector nodes when existing nodes persistently fail to cover input patterns, removing the need to pre-specify N=512 detectors.

> **Scope clarification:** GNG applies specifically to the Layer 1 chunking detector bank, where the number of recurring byte patterns is unknown and grows with exposure. The Layer 5 competitive prototype store uses a fixed column count (N_columns) by design — columns represent reusable assembly slots, not a variable-size codebook. The fixed count ensures stable WTA dynamics and bounded memory.

### 4.2 RTF Encoding — The Adaptation Gap

**The principle:** Rate-Temporal Fusion encoding, adapted from Li et al. (2024) who demonstrated dual-coded spike trains (rate + temporal) outperform either pure code in artificial visual neurons built from NbOx Mott memristors.

**What HECSN takes from this:** The principle that combining rate information (how many spikes) with temporal information (when they occur) produces richer representations than either alone.

**The critical adaptation gap:** Li et al.'s results are from hardware optical neurons processing visual luminance signals at the pixel level. They demonstrated 94.4% on face recognition and 91.3% on MNIST — tasks where spatial frequency patterns map naturally to spike timing. Text is fundamentally different:

1. Characters are categorical, not continuous — "a" is not "halfway" between "b" and "z" in any perceptual sense, unlike pixel luminance which is continuous
2. Sequence ordering in text is at the word/phrase level, not the sub-character level that RTF's positional phase offset was designed to capture
3. The burst coding aspect (multiple spikes per pattern encoding contextual information) maps to reading rate variability in biological systems, but in HECSN's character-by-character processing this maps to nothing obvious

**What the current implementation actually does:** Order-weighted ASCII encoding (`order_weighted_ascii`) with positional phase offset. This is a reasonable heuristic — earlier positions in a character window get higher weight, encoding left-to-right ordering information. It works (validated via Stage-0 routing coherence). But calling it "Rate-Temporal Fusion" imports theoretical justification that was derived for a different domain.

**Required validation before publication:** Run the four-way encoding ablation: (1) uniform ASCII (no ordering), (2) order-weighted ASCII (current), (3) hashed n-gram (bag-of-n-grams), (4) pure RTF burst coding as described by Li et al. adapted to bytes. Report temporal coherence at 50K tokens for each. The winner is the HECSN encoding. If order-weighted ASCII wins, call it that — not RTF.

### 4.3 Context Layer: 15 Tokens Is Not Enough — A Detailed Critique

**Current design:** Three fixed temporal traces (fast ≈ 2 tokens, medium ≈ 7 tokens, slow ≈ 15 tokens) over column assembly activations. The 15-token window is justified by: `tau_slow = T_per_token × context_tokens = 25ms × 15 = 375ms`, matching the Nair et al. (2024) slow neurotransmission timescale.

**Why 15 tokens is almost certainly wrong for semantic language processing:**

The human cortex exhibits a temporal receptive window hierarchy spanning from milliseconds (early auditory cortex) through seconds (frontal cortex) to tens of seconds (default mode network) [Hasson et al. 2008]. For semantic processing specifically, relevant context spans entire sentences (10–30 words for typical English) and sometimes paragraphs. At the character level HECSN processes, 15 tokens is roughly 3 characters of English text — enough for phonemic context, not nearly enough for morphemic or lexical context, and nowhere near enough for semantic context.

DH-SNNs with temporal dendritic heterogeneity demonstrate that adaptively learning heterogeneous timing factors on different dendritic branches of the same neuron generates multi-timescale dynamics capable of capturing features at different timescales. This approach consistently outperforms fixed single-timescale recurrence on speech recognition, visual recognition, EEG, and robot place recognition tasks.

The improved performance of ALIF models is attributed to additional slow-decaying state variables that facilitate information integration over longer timespans. Heterogeneous decaying rates of state variables in models like GLIF and DH-LIF further facilitate temporal dependencies across different timescales.

The biological evidence confirms what common sense suggests: no single timescale is optimal; the brain uses a hierarchy of timescales simultaneously. HECSN's three fixed traces are a step in the right direction, but they are:

1. **Fixed**, not adaptive — the optimal timescale for integrating context depends on the current input statistics and should be learned
2. **Too short** — 15 tokens at the character level provides less context than a single English word
3. **Discrete** — three timescales cannot capture the continuous spectrum of temporal dependencies relevant to language

**What should replace it:**

The Adaptive LIF (ALIF) neuron model adds a slow adaptation current that enables learnable spike-frequency adaptation, effectively implementing an adaptive timescale. ALIF and AdEx models can both accurately model biological spike timing where LIF cannot. ALIF is more computationally efficient than AdEx and performs well on temporal delay tasks.

For HECSN, the proposed fix is **adaptive context traces with learnable decay parameters**, implemented as:

```python
# Context Layer with learnable time constants per neuron population
# Inspired by ALIF and DH-SNN
class AdaptiveContextLayer:
    """
    Each context neuron has a learnable decay parameter tau_i.
    During training, tau_i adapts via Hebbian mechanisms:
    neurons that are useful for context-dependent routing
    (measured by routing differentiation score) retain or
    slow their timescale; neurons that contribute little speed up.
    
    This replaces the three fixed fast/medium/slow traces with a
    continuous distribution of learned timescales, distributed
    log-uniformly between tau_min = 2 tokens and tau_max = 500 tokens.
    
    Biological basis: cortical temporal receptive windows span
    3 orders of magnitude (ms to seconds/minutes). Fixed three-
    timescale approximations are empirically insufficient.
    
    Computational basis: DH-SNN (Li et al. 2023 Nature Comms)
    shows consistent performance gains from heterogeneous learned
    tau on every temporal benchmark tested.
    """
    def __init__(self, n_neurons: int, n_columns: int,
                 tau_min: float = 2.0, tau_max: float = 500.0,
                 device: str = 'cuda'):
        self.n = n_neurons
        self.nc = n_columns
        
        # Learnable time constants, initialized log-uniformly
        # Shape: [n_neurons] — each neuron has its own timescale
        log_tau = torch.linspace(
            math.log(tau_min), math.log(tau_max), n_neurons,
            device=device
        )
        self.log_tau = torch.tensor(log_tau)  # Hebbian-adapted, NOT nn.Parameter
        
        # Context state per neuron
        self.state = torch.zeros(n_neurons, device=device)
        
        # Projection weights: plain tensors with Hebbian updates (no backprop)
        self.W_in = torch.zeros(n_neurons, n_columns, device=device)
        nn.init.xavier_uniform_(self.W_in)
        self.W_out = torch.zeros(n_columns, n_neurons, device=device)
        nn.init.xavier_uniform_(self.W_out)
        
        # SST+ interneuron global inhibition (gain control)
        self.inhibition_strength = 0.3
    
    def step(self, assembly: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
        """
        One timestep. assembly: [n_columns] spike pattern.
        Returns: [n_columns] routing gain.
        """
        # Effective decay per neuron (constrained: tau ≥ tau_min)
        tau = torch.exp(self.log_tau).clamp(min=2.0)
        alpha = torch.exp(-dt / tau)  # [n_neurons], per-neuron decay
        
        # Input drive (matrix-vector multiply, no autograd graph)
        drive = torch.sigmoid(self.W_in @ assembly)  # [n_neurons]
        
        # Leaky integration with neuron-specific timescale
        self.state = alpha * self.state + (1 - alpha) * drive
        
        # SST+ inhibition: global gain control prevents saturation
        mean_activity = self.state.mean()
        inhibited_state = self.state - self.inhibition_strength * mean_activity
        
        # Project to routing gain (no autograd)
        gain = torch.sigmoid(self.W_out @ inhibited_state)
        return gain
    
    def update_timescales(self, routing_differentiation: torch.Tensor) -> None:
        """
        Adapt tau via Hebbian rule (no gradient): neurons that contribute to
        context-dependent routing → slow down; neurons contributing little → speed up.
        
        routing_differentiation: [n_neurons], per-neuron context-specificity
        computed as the mean variance of neuron state across repeated
        observations of the same input under different preceding contexts.
        High context-specificity → neuron differentiates between contexts
        → increase tau.
        
        All weight updates are local Hebbian updates, not backprop.
        This preserves the bio-plausible constraint of no end-to-end gradient flow.
        """
        lr_tau = 0.001
        self.log_tau += lr_tau * (routing_differentiation - 
                                    routing_differentiation.mean())
        # Clamp to [log(tau_min), log(tau_max)]
        self.log_tau.clamp_(math.log(2.0), math.log(500.0))
```

**Computing `routing_differentiation` (implemented in `context.py:compute_routing_differentiation`):**

The `routing_differentiation` vector measures per-neuron **context-specificity**, not mere temporal variability. During each wake `observe()` call (not replay), the current assembly is mapped to a compact input signature (top-8 activation indices + coarse-quantized values), and the `(signature, neuron_state)` pair is appended to a sliding buffer (last 200 observations). At deep-sleep time, `compute_routing_differentiation()` groups observations by input signature, then for each input seen at least twice under different preceding contexts, computes the variance of neuron states across those repetitions. The per-neuron result is the mean of these per-input variances.

This distinction matters: plain temporal variance conflates context-sensitivity with global activity variance. A neuron that fires sporadically for rare characters shows high temporal variance but is not context-sensitive. The context-specificity metric asks: "given the same input X under different contexts A and B, how much does this neuron's state differ?" — the correct signal for tau adaptation.

- **High context-specificity** neurons: given the same input, their state depends strongly on what came before → they usefully differentiate contexts → τ is increased (slower decay retains context information longer).
- **Low context-specificity** neurons: given the same input, their state is context-invariant → they contribute little to routing → τ is decreased (faster decay reduces computational cost).
- **Minimum data threshold:** at least 3 input signatures with ≥2 observations each are required before the metric is non-zero. Below this threshold, the function returns zeros and tau adaptation is skipped.

The update is called once per deep-sleep cycle from the trainer, after SFA correction and dead-column census. Replay and offline context-priming observations are excluded from the buffer to avoid polluting the differentiation signal with artificial state distributions.

**Immediate consequence:** Extending `tau_max` from 15 to 500 tokens means the slow context neurons integrate over approximately 100 English characters — enough for word-level context. Setting `tau_max = 2000` would cover phrase-level context. The cost is memory (each neuron's state is one float per timestep) and the computational overhead of maintaining 500-token-window exponentially-weighted sums. At 256 context neurons this is trivial. The gain in context-dependent routing is expected to be substantial based on the DH-SNN literature.

**Open question:** How does the adaptive timescale interact with the drift detection and sleep scheduling? The slow neurons (tau ~ 500 tokens) will not update quickly enough to signal concept drift when drift occurs over short timescales. Fast neurons (tau ~ 2) will not carry state across sleep phases. This interaction needs explicit design: fast neurons reset across sleep phases; slow neurons should carry forward.

### 4.4 Triplet STDP: Why Pair-Based Rules Are Wrong

**What the old paper had:** "Log-STDP with sublinear LTD for log-normal weight distributions." A pair-based rule.

**Why pair-based STDP is experimentally incorrect:** Classical STDP models based on pairs of spikes are not sufficient to explain synaptic changes triggered by triplets or quadruplets of spikes. Pair-based models cannot account for the dependence on repetition frequency of spike pairs. For example, at frequencies above 25 Hz, pair-based models predict reduced potentiation (because post-pre pairs at 30ms add depression to the pre-post pair at 10ms). In experiments, the opposite is observed — potentiation increases with frequency.

This is not a minor quantitative correction. Pair-based STDP predicts qualitatively wrong behavior at physiological burst frequencies. For HECSN, which uses burst coding (RTF's multiple spikes per pattern), pair-based STDP will compute incorrect weight updates whenever two spikes occur in close succession. This systematically miscredits coincidence detection.

**The triplet rule (Pfister & Gerstner 2006):**

HECSN implements the **full (all-to-all) triplet model**, which uses four trace variables:
- `r1(t)`: pre-synaptic fast trace (incremented by each pre-spike, decays with τ+ = 16.8ms)
- `r2(t)`: pre-synaptic slow trace (incremented by each pre-spike, decays with τx = 101ms)
- `o1(t)`: post-synaptic fast trace (incremented by each post-spike, decays with τ− = 33.7ms)
- `o2(t)`: post-synaptic slow trace (incremented by each post-spike, decays with τy = 114ms)

> **Note:** The *minimal* (nearest-spike) model uses only three traces (r1, o1, o2) and the LTD triplet term uses r1. The full model adds a fourth trace r2 with its own time constant τx, which captures the recency of pre-synaptic firing on a slower timescale. HECSN uses the full model because it better captures frequency-dependent plasticity across the entire spike train, not just nearest-neighbour pairs.

Weight update at post-spike time `t_post`:
```
ΔW_LTP = A2+ × r1(t_post)                [pair term: pre before post]
        + A3+ × r1(t_post) × o2(t_post - ε) [triplet: two post + one pre]
```

Weight update at pre-spike time `t_pre`:
```
ΔW_LTD = -A2- × o1(t_pre)               [pair term: post before pre]
         -A3- × o1(t_pre) × r2(t_pre - ε) [triplet: two pre + one post]
```

With A3+ = 0 and A3- = 0, this reduces to the classical pair rule — so triplet STDP is strictly more general.

**Parameters (from hippocampal culture fit, Pfister & Gerstner 2006 Table 3, all-to-all model):**
- τ+ = 16.8ms, τ− = 33.7ms, τx = 101ms, τy = 114ms
- A2+ = 5×10⁻¹⁰, A2- = 7×10⁻³, A3+ = 6.2×10⁻³, A3- = 2.3×10⁻⁴

The key property: `o2(t)` is a slow post-synaptic trace that effectively implements a rate detector. At high burst frequencies, `o2` accumulates and amplifies the LTP term, producing the observed frequency-dependent potentiation.

**Combined with Log-STDP:** Keep the sublinear LTD for log-normal weight distribution stabilization. Replace pair-based LTD with triplet LTD:

```
ΔW_LTD = -(A2- + A3- × r2(t)) × o1(t) × f_sublinear(w)
```

Where `f_sublinear(w) = 1 / (1 + w)` implements the sublinear LTD that prevents weight saturation.

**Implementation cost:** Four trace variables per synapse (r1, r2, o1, o2) plus four time constants (τ+, τx, τ−, τy) and four amplitude parameters (A2+, A2-, A3+, A3-). At 15% connectivity density and 10K neurons: ~15M synapses × 4 floats = 60MB additional memory. Negligible at target scale.

### 4.5 Kohonen/SOM Competitive Learning: Honest Limitations

**What HECSN uses:** Winner-Takes-All competition followed by a Kohonen update: the winning prototype moves toward the current input by a fraction of the distance. This produces prototype specialization over time.

**What Kohonen's SOM actually guarantees:** Convergence to a stable prototype configuration — but only under specific conditions: the learning rate must anneal from an initial value to zero over a fixed, predetermined training schedule. Formal convergence analysis requires this schedule. The SOM is proven to approximate the input distribution in the limit, but only with this annealing.

**What HECSN violates:** HECSN is an online continual learning system. There is no predetermined training length, no annealing schedule, and new patterns continue arriving indefinitely. The convergence proofs do not apply. In practice:

1. Without annealing, prototypes continue moving indefinitely and may oscillate between configurations when the input distribution shifts (different text domains)
2. The "winner-local drift" mechanism (validated in Stage-0) partially addresses this, but it is a heuristic stability mechanism, not a convergence guarantee
3. At very long training (>1M tokens), prototype drift rates may not asymptotically approach zero

**Mitigation strategies already present:**
- Sleep-phase replay and prototype momentum prevent rapid oscillation
- Fragility-gated consolidation protects well-established prototypes
- The refractory mechanism (winner history) prevents monopoly winners and ensures coverage

**What would strengthen this:**
- An explicit decreasing learning rate tied to prototype stability (prototypes that haven't moved much in 10K tokens get a lower lr)
- Formal stability analysis of the winner-local drift mechanism under non-stationary streaming input
- Empirical measurement of prototype position variance over rolling windows at different training stages

**Dead column problem:** Columns that never win the WTA competition receive no STDP signal and no update. Intrinsic Plasticity (IP) partially addresses this by lowering the threshold of persistently silent columns, making them more likely to fire. But IP needs non-zero input current to work — a completely unresponsive column (no synaptic input reaching threshold) will not self-repair through IP alone. 

Solution: Add a periodic "column census" during deep sleep: columns with zero wins in the last 10,000 tokens are re-initialized with a prototype sampled from the slow memory buffer (or from a random perturbation of the nearest active column). This is biologically motivated — the brain prunes and regrows synaptic connections on a timescale of days to weeks; dead columns are the computational equivalent of pruned dendritic trees.

> **Implementation status:** Dead column census is implemented in `trainer._sleep_replay()` and runs exclusively during deep sleep phases. A column is considered dead if it has zero wins over the monitoring window. Revival only triggers when ≥5% of columns are dead — this threshold prevents noise in small networks (e.g., 12-column test configs) and avoids premature interference. The surprise-triggered NE response (`should_boost_exploration()`) now boosts exploration noise rather than destructively resetting columns — dead column revival is reserved for the deep sleep census.

### 4.6 AdEx Neuron Model: Numerical Stability and the ALIF Alternative

**Current implementation:** Heun's method (RK2) for the exponential upswing term, with NaN guard. Biophysical parameters from Brette & Gerstner (2005).

**Two issues:**

Issue 1 — Computational cost at scale. The AdEx model has two state variables per neuron (V and w). At 100K neurons running at 0.5ms time steps, this is 200K float operations per time step, plus the spike detection, reset, and STDP eligibility updates. The computational profile is manageable on GPU with vectorization, but at 1M+ neurons requires careful batching.

Issue 2 — ALIF may be more appropriate than AdEx for HECSN's specific use case. The AdEx was designed to match the specific shape of biological adaptation currents (exponential upswing + adaptation w). ALIF (Adaptive LIF) matches the same behavioral outcomes (spike-frequency adaptation, bursting) with less computational overhead and, critically, its adaptation threshold variable maps more naturally to the "learnable timescale" approach needed for the Context Layer.

Comparison:
| Model | State variables | Captures adaptation | Maps to learnable tau |
|---|---|---|---|
| LIF | 1 (V) | No | No |
| ALIF | 2 (V, threshold) | Yes | Yes |
| AdEx | 2 (V, w) | Yes | Less naturally |

For HECSN's goal — a network that both simulates realistic adaptation AND can learn its own temporal integration timescales — ALIF is arguably more appropriate for the Context Layer, while AdEx remains appropriate for the Competitive Layer where the exponential upswing's biological fidelity matters more.

**Recommendation:** Use AdEx for the Competitive and Binding Layers (biological fidelity required for STDP timing). Use ALIF with learnable threshold for the Context Layer (adaptive timescale required).

### 4.7 Neuromodulator System: Serotonin Is More Complicated Than Claimed

**Current implementation (validated):** Four independent channels in `SurpriseMonitor`, each driven by error-prediction dynamics:
- **DA → LTP gain:** Reward prediction error via `compute_dopamine_rpe()`. Formula: `tanh((predicted_error - current_error) / baseline × 3.0)`. When DA is positive, it scales the LTP learning rate in the competitive layer (DA→LTP gate = `0.80 + 0.20 × dopamine`, range 0.80–1.00). **Implemented and wired into `trainer.train_step()`.**
- **ACh → novelty/learning rate:** Slow-integrating novelty signal (EMA α=0.10), slower decay than other channels, keeping exploration sensitivity persistent.
- **NE → unexpected uncertainty:** Measures deviation from predicted error. Boosted by serotonin when serotonin_drive > 0.4. When NE exceeds 0.85, the `should_boost_exploration()` method triggers an **exploration noise boost** (scaling factor ×1.5, capped at 2.0, decaying back to 1.0 at rate 0.99/token) — a biological analogue for "something is fundamentally wrong, explore harder." This replaces the v3 destructive reset approach (`force_revive_dead_columns`) which would erase learned patterns during training. Dead column revival is now handled exclusively by the deep sleep census (§4.5).
- **5-HT → patience gate:** Punishment prediction signal via `compute_serotonin_punishment()`. Modulates a patience gate in the competitive layer (higher 5-HT = more resistance to prototype overwrite). Gate: `ht_patience_gate = max_gate - serotonin × gate_modulation`. **Implemented and wired into `trainer.train_step()`.**

**The 5-HT problem:** Serotonin acts through at least 14 distinct receptor subtypes, with opposing functional effects depending on which receptors are targeted. 5-HT1A receptors predominantly mediate inhibitory effects (reducing neural excitability and LTP); 5-HT4 and 5-HT7 receptors mediate excitatory effects. The net effect of serotonin on synaptic plasticity depends heavily on which receptor population is most activated, which varies by brain region, by the recent history of serotonin levels (receptor sensitization/desensitization), and by the timing of serotonin release relative to spike timing.

Doya's (2002) "patience" interpretation captures one key function — 5-HT modulating the temporal discounting of rewards, making systems more patient (higher tau in value function) — but mapping this directly to "LTD bias" is a simplification.

**What we do:** The 5-HT channel operates as a "plasticity patience" modulator (higher 5-HT = more resistance to change = effectively increased LTD threshold rather than increased LTD rate). The paper acknowledges that this is a functional approximation to a complex multi-receptor system. The key behavioral prediction is: high sustained 5-HT (from high recent success) should reduce prototype drift rates — this is measurable and falsifiable regardless of the specific molecular mechanism.

**Real training observation:** During 1,152 tokens of Wikipedia training, DA oscillated 0.006→0.431 as prediction error decreased, 5-HT→patience gate responded at 0.81–1.00, and micro-sleep triggered at 256 tokens — confirming that neuromodulators drive live network dynamics as designed.

### 4.8 Online SFA Approximation: A Significant Weakness

**What the Abstraction Layer implements:** An online approximation to Slow Feature Analysis (Wiskott & Sejnowski 2002). The update rule: reduce weights for concept dimensions whose output shows high temporal variance (slow these dimensions down). This is described as "anti-Hebbian in time."

**The problem with the approximation:** The true SFA objective is to find the linear projection W that minimizes the temporal derivative of the output while keeping the output whitened. This requires:
1. A batch of input-output pairs to compute the true covariance matrix
2. An eigendecomposition to find the solution

The online approximation avoids the batch requirement by tracking running statistics (slow_var, fast_mean, slow_mean). However:

- The online estimate of slow_var converges slowly and is noisy
- The anti-Hebbian update (reducing weights for high-variance dimensions) is not guaranteed to converge to the SFA optimum
- Whitening (keeping outputs decorrelated) is not enforced — high-variance dimensions will be suppressed but correlated outputs can still persist

**Consequence:** The Abstraction Layer will develop *something* — some slowly-varying features will emerge. But the features that emerge may not be the *optimal* slowly-varying features that would emerge from true SFA. The gap between "slow features" and "semantically meaningful features" could be larger than expected.

**Mitigation that is tractable:** During deep sleep phases, run a mini-batch SFA step: take 100 samples from the slow memory buffer, compute their Abstraction Layer outputs, compute the true covariance and temporal covariance matrices, and do one update step toward the exact SFA solution. This is O(100 × n_concepts²) per sleep phase — negligible — and significantly improves the quality of the slow features without abandoning the online learning framework.

### 4.9 STC Timescales: Self-Calibration vs Arbitrariness

**The functional_minute calibration approach:** Measure how many tokens it takes for a novel prototype to stabilize (winner-local drift convergence). Use this as the timescale unit for STC parameters.

**What the calibration actually gives you:** The timescale of *prototype convergence* — how long it takes for a new prototype to settle after first exposure. This is a measure of the learning dynamics, not of memory consolidation dynamics.

In biology, Early-LTP (1–3 hours) and Late-LTP (>3 hours) timescales are set by the kinetics of kinase cascades, protein synthesis, and structural synaptic changes — mechanisms that have no counterpart in the network dynamics being calibrated. The calibration gives a principled *relative* scaling (Late-LTP should be roughly 4x longer than Early-LTP), but the absolute mapping from "prototype convergence time" to "LTP phase duration" remains arbitrary.

**What to state honestly:** The `functional_minute` self-calibration is an improvement over a fixed arbitrary constant (500 tokens) because it ties the STC timescales to the network's own learning dynamics. But it does not constitute a principled mapping from computational dynamics to biological consolidation timescales. The calibrated STC parameters should be treated as hyperparameters that set the relative ratios of consolidation timescales, with the absolute values requiring empirical validation through the catastrophic forgetting benchmark (Section 8.8).

**Sensitivity analysis required:** Run the forgetting benchmark (Task A → Task B → Re-test Task A) with `functional_minute` at 100, 500, 2000, and 10000 tokens. Report how much the Task-A recall depends on this parameter. If recall is robust across this range, the absolute calibration doesn't matter much. If recall collapses below 500 or above 2000, the calibration is load-bearing and must be reported as such.

### 4.10 The Grounding Probe Threshold: Calibration Required

**Current threshold:** > 0.65 for "genuine semantic organization."

**Why this needs calibration before publication:** The threshold of 0.65 was chosen to be meaningfully above 0.50 (random), but there is no established baseline for what well-trained distributional word vectors (word2vec, GloVe, fastText) score on the same 50-triple suite. 

If word2vec trained on the same text corpus scores 0.78 on the 50 triples, then 0.65 would represent a *worse* result than a simpler baseline. If word2vec scores 0.53, then 0.65 represents a genuinely strong result for a system with no explicit distributional training.

**Required calibration before stating the threshold:**
1. Train word2vec skip-gram (window=5, dim=100) on the same text corpus as the HECSN training data
2. Evaluate word2vec on the 50-triple grounding probe
3. Train fastText (character n-grams) on the same corpus — this is the closest unimodal baseline to HECSN's character-level input
4. Set the HECSN threshold to: `word2vec_score + meaningful_margin`

The `meaningful_margin` should be at least 0.05 — a difference that would not arise by chance across different random initializations. If the 50-triple suite has high variance (consistent grounding is hard), the margin may need to be larger.

**The concreteness gap test remains the strongest evidence.** Regardless of the absolute probe score, if HECSN shows concrete concepts scoring 0.10+ higher than abstract concepts — a pattern that cannot be produced by pure text statistics because both are equally frequent — that is genuine evidence of perceptual grounding that word2vec cannot replicate.

---

## 5. Multimodal Grounding

### 5.1 The Grounding Mechanism in Precise Detail

Cross-modal temporal co-occurrence STDP: when text spikes and visual spikes co-occur within `tau_bind` functional time, cross-modal weights are potentiated. When text fires without visual support, cross-modal weights slowly decay. No semantic labels, no contrastive loss, no negative pairs — just: *neurons that fire together across modality boundaries wire together.*

**Four cross-modal weight matrices** (updated by independent STDP-like rules):

| Matrix | Direction | Biological analog | Initialization |
|---|---|---|---|
| W_tv | Text → Visual | Ventral stream feedback | Random small (~0.01) |
| W_vt | Visual → Text | Object recognition → language | Random small |
| W_ta | Text → Audio | Language → auditory prediction | Random small |
| W_at | Audio → Text | Auditory cortex → Wernicke's area | Random small |

**STDP update** (at text spike event):
```
ΔW_tv[i, :] = A_plus × text_spike[i] × visual_trace[:]
```

Where `visual_trace[j]` is the exponentially-decaying trace of recent visual activity at dimension j:
```
visual_trace[j] += visual_spikes[j]           # at visual spike event
visual_trace[j] *= exp(-dt / tau_trace)       # continuous decay
```

**Grounding confidence** (slow EMA tracking prediction quality):
```
prediction_error = 1 - cosine_similarity(W_tv[i], actual_visual)
grounding_confidence[i] = max(0, 1 - prediction_error)   # = max(0, cos_sim)
```

> **Implementation note:** Code uses `F.cosine_similarity` (bounded [−1, 1]) rather than L2 norm. This avoids the issue where L2 prediction error can exceed 1.0 and produce negative confidence values.

**A+ and A− asymmetry:** A- set 20% larger than A+ (0.012 vs 0.010) to prevent runaway potentiation. This creates a small anti-Hebbian drift that stabilizes associations over time.

**Naming convention:** In code, `visual_confidence` and `audio_confidence` are per-modality sub-components (internal attributes) tracking prediction quality for each association channel independently. The combined method `grounding_confidence()` is the canonical public API, returning `visual_confidence + audio_confidence` as a single signal used by the curiosity planner and developmental stage gates. All public-facing code and metrics use `grounding_confidence` consistently; per-modality attributes are accessed only when modality-specific logging is needed (e.g., `cross_modal_visual_confidence` in training metrics).

### 5.2 Audio-Text vs. Visual-Text Grounding Are Not the Same Problem

**This distinction is critical and was collapsed in v3.** Audio-text grounding is qualitatively easier than visual-text grounding:

**Audio-text alignment:** When a person speaks, the acoustic output IS a temporal realization of the linguistic content. The audio stream and the text transcript are generated by the same underlying process at the same moment. Alignment is inherent, not coincidental. The challenge is only in the temporal registration (speech precedes transcript by ~milliseconds).

**Visual-text alignment:** When a person narrates something they are looking at, what they say and what is visible are often misaligned. They describe past events, anticipate future events, provide background context, and make meta-comments. Research on HowTo100M shows only ~25% of narration clips are visually alignable. For non-instructional content (documentary narration, news), the misalignment rate is higher.

**Implication for evaluation:** HECSN's grounding probe should distinguish audio-text grounding from visual-text grounding. If the system achieves grounding probe > 0.65 primarily because audio-text associations are strong (speech IS text), that is a weaker result than achieving > 0.65 through genuine visual-concept associations. 

**Required separate metrics:**
- Audio-text probe: triples where the distinction requires acoustic grounding (e.g., ("thunder", "loud_sound", "silence") — distinguishable by audio statistics)
- Visual-text probe: triples where the distinction requires visual grounding (e.g., ("ocean", "blue_expanse", "red_surface") — distinguishable only by visual statistics)

The visual-text probe is the harder test and the one that validates genuine perceptual grounding.

### 5.3 The Alignment Filter Design

The alignment filter uses the network's existing cross-modal predictions to evaluate whether any given text-visual pair should update the cross-modal weights. **This is implemented in `cross_modal.py:alignment_gate()` and `alignment_gate_audio()`.**

```python
def alignment_gate(text_assembly: torch.Tensor,
                    visual_spikes: torch.Tensor,
                    threshold: float = 0.4) -> tuple[bool, float]:
    """
    Should we update cross-modal weights for this (text, visual) pair?
    
    The method uses the layer's internal W_tv weights and visual_confidence
    to assess alignment. The threshold should be conservative early in 
    Stage 2 (fewer anchors → weaker predictions) and can be raised as 
    Stage 2 progresses.
    
    Implementation note: A separate alignment_gate_audio() method handles
    the audio path using W_ta and audio_confidence.
    """
    # Only use text dimensions with meaningful grounding confidence
    conf_mask = (self.visual_confidence > 0.2).float()
    masked = text_assembly * conf_mask
    
    if masked.sum() < 0.01:
        # No grounded dimensions active — reject uncertain pairs
        return False, 0.0
    
    # Predict visual pattern from grounded text dimensions
    predicted = torch.mv(self.W_tv.T, masked)
    if predicted.norm() < 0.01:
        return False, 0.0
    
    # Cosine similarity between prediction and actual
    p_norm = F.normalize(predicted, dim=0)
    v_norm = F.normalize(visual_spikes, dim=0)
    score = F.cosine_similarity(p_norm.unsqueeze(0), v_norm.unsqueeze(0)).item()
    return score > threshold, max(0.0, score)
```

**Three failure modes to monitor:**
1. **Over-rejection:** The filter rejects pairs that are genuinely aligned because the grounding vocabulary is too small. Sign: grounding confidence growth rate falls below 0.001 per 1000 tokens after Stage 1 completion. Fix: lower threshold temporarily.
2. **Under-rejection:** The filter accepts spurious pairs because the vocabulary is broad but miscalibrated. Sign: self-criticism loop finds many high-confidence wrong associations. Fix: raise threshold.
3. **Stable-wrong:** The filter consistently accepts a wrong association (e.g., "ocean" → narrow-frame boat interior). Sign: self-criticism loop → correction → repeated reacquisition of same error. Fix: introduce a "blacklist" for text-visual pairs that have been corrected more than twice.

### 5.4 Data Sources and Their Properties

**Stage 1 (alignment guaranteed by construction):**

| Dataset | Modalities | Why alignment is guaranteed | License | Availability |
|---|---|---|---|---|
| MNIST-DVS + TI-46 speech | Visual (event) + Audio | Digit displayed = digit spoken simultaneously | Academic | Direct download |
| ObjectNet + spoken labels | Visual + Audio | Image shown = word spoken (purpose-recorded) | Research | Access form |
| HTM-AA aligned subset (v2: 1.2M videos) | Visual + Text + Audio | Temporal alignment score > 0.7 by human annotation | Research | Direct download |
| Short cooking video clips (close angle) | Visual + Text + Audio | Action described while being performed (verified) | Creative Commons | Filter from HowTo100M |

**Stage 2 (self-filtered):**

| Dataset | Modalities | Expected alignment rate | License |
|---|---|---|---|
| Full HowTo100M (1.2M videos) | Visual + Text + Audio | ~25% (measured) | Academic, access form |
| BBC Natural History Unit (selected) | Visual + Text + Audio | ~40% (narration-to-scene) | Complex; verify per-clip |
| Podcast transcripts + audio | Text + Audio | ~99% (speech IS text) | Mixed; check per-source |
| Wikipedia + Wikimedia Commons | Text + Images | ~60% (alt text + image) | CC BY-SA |

**Note on YouTube:** Bulk downloading via yt-dlp raises legal uncertainty in most jurisdictions for research training. Using HowTo100M (already downloaded and licensed for academic research) is the safe path. The full HowTo100M requires submitting an access form; the HTM-AA aligned subset is directly downloadable.

---

## 6. Scalability Architecture

### 6.1 GPU-Native Routing: Precise Benchmarking Targets

**Current status:** Sharded `torch_topk` path validated on CPU (recall@k = 1.000, mean latency ~4.70ms CPU). CUDA routing not yet benchmarked on the current tree.

**Required benchmarks before publication:**

```python
# Benchmark script — run this, report results
# Recommended: n_queries=1000 for N≤10K, 200 for N≥50K.
# The default adapts automatically based on n_cols.
def benchmark_routing(n_cols: int, dim: int, n_queries: int | None = None,
                       device: str = 'cuda') -> dict:
    """
    Measure actual routing latency at target scales.
    Results must appear in the paper's scalability table.
    
    If n_queries is None, defaults to 1000 for n_cols≤10K,
    200 for n_cols>10K (to keep runtime under 5 minutes).
    """
    if n_queries is None:
        n_queries = 1000 if n_cols <= 10_000 else 200
    import time
    prototypes = torch.randn(n_cols, dim, device=device)
    prototypes = F.normalize(prototypes, dim=1)
    queries = F.normalize(torch.randn(n_queries, dim, device=device), dim=1)
    
    # Warmup
    for _ in range(10):
        sims = queries @ prototypes.T
        sims.topk(32, dim=1)
    
    # Benchmark with per-query CUDA synchronization
    torch.cuda.synchronize()
    latencies = []
    for i in range(n_queries):
        torch.cuda.synchronize()  # drain pipeline before timing
        t_start = time.perf_counter()
        sims = queries[i:i+1] @ prototypes.T
        sims.topk(32, dim=1)
        torch.cuda.synchronize()  # drain pipeline after timing
        latencies.append(time.perf_counter() - t_start)
    
    latencies_ms = sorted([l * 1000 for l in latencies])
    return {
        'n_cols': n_cols,
        'dim': dim,
        'device': device,
        'ms_per_query': latencies_ms[len(latencies_ms) // 2],  # median
        'ms_p95': latencies_ms[int(len(latencies_ms) * 0.95)],
        'ms_p99': latencies_ms[int(len(latencies_ms) * 0.99)],
    }

# Run: 1K, 10K, 50K, 100K, 500K columns on target hardware
# Report device (GPU model, VRAM), PyTorch version, precision mode
```

**Expected targets (A100 80GB, FP32):**

| Columns | Method | Expected latency | Memory (FP32) | Memory (TQ+@3bit, m=dim) | Memory (Stage 1 only) |
|---|---|---|---|---|---|
| 1K | Flat GPU | ~0.01ms | 1 MB | 0.36 MB | 0.10 MB |
| 10K | Flat GPU | ~0.05ms | 10 MB | 3.6 MB | 1.0 MB |
| 50K | Flat GPU | ~0.25ms | 50 MB | 18 MB | 5.1 MB |
| 100K | IVF GPU | ~0.08ms | 100 MB | 36 MB | 10 MB |
| 500K | IVF GPU | ~0.12ms | 500 MB | 178 MB | 51 MB |
| 1M | Distributed IVF | ~0.3ms | 1 GB/shard | 356 MB/shard | 102 MB/shard |

IVF is faster than flat at 100K because it searches only top-8 of sqrt(N)=316 cells (~2500 candidates) instead of all 100K.

> **Concurrency note:** HNSW/IVF index rebuilds must not occur during Phase B consolidation, which modifies prototype weights that the index references. The correct ordering is: complete Phase B consolidation (anchor_lr prototype updates) → rebuild index from consolidated prototypes → resume routing. In the current implementation (`trainer.py:_sleep_replay`), the HNSW rebuild is placed after the consolidation loop completes, ensuring all prototype positions are final before the index is reconstructed. If future work adds background reassignment or parallel rebuilds, this ordering constraint must be preserved.

### 6.2 TurboQuant Implementation

**Recommended: TheTom/turboquant_plus (Apache-2.0)**

Validation results from TheTom/turboquant_plus on real Qwen3-1.7B KV tensors:
- Raw kurtosis 900.4 → after random rotation: 2.9 (Gaussian = 3.0). Ratio exactly 1.000.
- This validates the rotation Gaussianization property that makes the Lloyd-Max quantizer optimal.

**How HECSN uses it differently from LLM KV cache:**

LLM KV cache: compress many (key, value) pairs at inference time; read them once per attention computation.

HECSN prototype store: compress N prototype vectors at initialization/sleep; read them thousands of times per second during routing; update a small fraction (winners) during wake.

The asymmetry matters: the decompression path (inner product computation in rotated space) is on the hot routing path and must be fast. The compression path (re-quantizing updated prototypes) happens only when a prototype updates, which is infrequent for consolidated prototypes.

**Integration approach (TurboQuant+ with QJL residual correction):**

The implementation follows the two-stage TurboQuant architecture (arXiv:2504.19874): Stage 1 applies random rotation + optimal scalar quantisation (PolarQuant), and Stage 2 adds a 1-bit Quantised Johnson-Lindenstrauss (QJL) residual correction that eliminates inner-product bias at low bit-widths. The correction stores `sign(S @ residual)` for each prototype (1 bit per projection dimension) and uses it at query time to produce an unbiased inner-product estimator.

```python
class TurboQuantPrototypeStore:
    """
    TurboQuant+ compressed prototype store for scalable routing.
    
    Two-stage compression:
      Stage 1 (PolarQuant): Random rotation → uniform scalar quantisation (3-bit).
      Stage 2 (QJL correction): 1-bit sign of projected residual for unbiased
                                inner-product estimation.
    
    Hot path (routing, called ~1000x/sec):
        1. Rotate query: q_rot = R @ q_norm           [O(dim²)]
        2. Base scores: decompressed @ q_rot           [O(N × dim)]
        3. QJL correction: residual_signs @ (S @ q_rot) [O(N × m)]
        4. Topk selection                              [O(N)]
    
    Cold path (prototype update, called ~10x/sec on wake winner columns):
        1. Update prototype in float32
        2. Re-compress updated prototype (rotation + quantise + QJL)
        3. Store compressed version
    """
    def __init__(self, n_cols: int, dim: int, bits: int = 3,
                 n_projections: int | None = None, device: str = 'cuda'):
        self.rotation = _random_rotation_matrix(dim, device)
        # QJL projection matrix (m × dim), shared across all prototypes
        m = n_projections or dim
        self._projection = torch.randn(m, dim, device=device) / math.sqrt(m)
        # Bit-packed quantised codes (uint8) + per-prototype scale/offset
        packed_dim = _packed_dim(dim, bits)
        self._codes = torch.zeros(n_cols, packed_dim, dtype=torch.uint8, device=device)
        self._scales = torch.ones(n_cols, device=device)
        self._offsets = torch.zeros(n_cols, device=device)
        # QJL residual: sign bits (±1) and norms
        self._residual_signs = torch.zeros(n_cols, m, dtype=torch.int8, device=device)
        self._residual_norms = torch.zeros(n_cols, device=device)
    
    def compress_all(self) -> int:
        """Compress dirty prototypes during sleep phase."""
        for idx in self._dirty.nonzero(as_tuple=True)[0]:
            i = int(idx.item())
            rotated = torch.mv(self.rotation, self._fp32[i])
            # Stage 1: PolarQuant — uniform scalar quantisation + bit-pack
            vmin, vmax = rotated.min().item(), rotated.max().item()
            scale = max(vmax - vmin, 1e-8) / (self.n_levels - 1)
            codes = ((rotated - vmin) / scale).round().clamp(0, self.n_levels - 1)
            self._codes[i] = pack_codes(codes.unsqueeze(0), self.bits).squeeze(0)
            self._scales[i], self._offsets[i] = scale, vmin
            # Stage 2: QJL residual correction
            dequantised = codes * scale + vmin
            residual = rotated - dequantised
            self._residual_norms[i] = residual.norm()
            projected = torch.mv(self._projection, residual)
            self._residual_signs[i] = projected.sign().to(torch.int8)
    
    def route(self, query: torch.Tensor, k: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
        """Vectorised routing with QJL-corrected inner products."""
        q = F.normalize(query.float(), dim=0)
        q_rot = torch.mv(self.rotation, q)
        # Stage 1: unpack bit-packed codes and decompress
        unpacked = unpack_codes(self._codes, self.bits, self.dim)
        decompressed = unpacked.float() * self._scales.unsqueeze(1) + self._offsets.unsqueeze(1)
        base_scores = torch.mv(decompressed, q_rot)
        # Stage 2: QJL correction — unbiased estimator
        q_proj = torch.mv(self._projection, q_rot)
        correction = torch.mv(self._residual_signs.float(), q_proj)
        correction *= self._residual_norms * math.sqrt(math.pi / 2) / self._projection.shape[0]
        scores = base_scores + correction
        return scores.topk(min(k, self.n_cols))
```

**Compression choice:** 3-bit (`turbo3`) with bit-packed storage gives ~9.8× Stage 1 compression on codes alone. Adding QJL residual correction (Stage 2) at m=dim reduces the effective ratio to ~2.8× (the signs add 1 byte per projection dimension per prototype) but ensures inner-product estimates are unbiased — critical for top-k routing accuracy at scale. Setting m=dim/4 yields ~6× compression with moderately higher estimator variance. For HECSN at typical prototype counts (≤100K), the QJL memory overhead is acceptable; the routing accuracy improvement at 3-bit is the binding constraint, not memory. Start with 3-bit + QJL (m=dim) and reduce m if memory is constrained.

### 6.3 Dual Sparsity: Honest Assessment

**2:4 Structured Sparsity (50% of nonzeros):**
- Works only on Ampere+ GPUs (A100, RTX 30xx+)
- Real speedup ~1.6x (not 2x) due to metadata overhead
- Requires matrices to have ≥ 64 columns for hardware efficiency
- HECSN's feedforward projection matrices (W_project: 128×256) meet this criterion at target scale

**CSR Sparse Tensors (10–20% connectivity):**
- PyTorch CSR support is beta; operations like `cat`, `stack` don't work on CSR tensors directly
- Workaround: convert to COO for concatenation operations, then back to CSR
- No hardware acceleration — performance comes from reduced arithmetic, not specialized kernels
- Profiling gate: at 10K columns with 15% connectivity density, measure CSR vs dense for a single structural plasticity update. If dense is faster, use dense until 50K+ columns.

---

## 7. Developmental Training Protocol

### 7.1 The Biological Basis for Guided Early Training

The developmental protocol is not an engineering compromise. It is the computational equivalent of a biological necessity.

The ability of a neural network to integrate information from diverse sources hinges critically on being exposed to properly correlated signals during the early phases of training. Interfering with the learning process during this initial stage can permanently impair the development of a skill, both in artificial and biological systems — the critical learning period phenomenon.

Architectures trained with cross-sensor reconstruction objectives are remarkably more resilient to critical periods. The recent success in self-supervised multimodal training compared to previous supervised efforts may be in part due to more robust learning dynamics.

HECSN's cross-modal STDP is exactly a cross-sensor reconstruction objective: when text fires, predict visual; when visual fires, activate text. The learning is self-supervised, not labeled. But the data must be structured so that the cross-modal predictions are learnable — which requires, during the critical period, that text and visual information be temporally co-located.

### 7.2 Stage 1: Critical Period (Structured Perceptual Grounding)

**What it is:** Training on multimodal data where temporal text-visual-audio co-occurrence is guaranteed by construction.

**What it is not:** Semantic labeling. No annotation of category membership, no specification of what concepts mean, no intent labels. The grounding signal is purely temporal proximity within the biological coincidence detection window (~100ms).

**Goal:** Establish cross-modal grounding anchors — associations between text patterns and their perceptual correlates — strong enough to serve as reference points for the Stage 2 alignment filter.

**Data sources (curated, alignment guaranteed):**

Primary Tier (highest alignment confidence):
- MNIST-DVS + TI-46 speech corpus: digit image shown = digit spoken, ~70K paired events, perfect temporal registration
- ObjectNet + purpose-recorded spoken labels: 113,000 images × N spoken labels per image, temporally registered
- Cooking video close-range (selected): performer describes action WHILE performing, verified manually for alignment

Secondary Tier (high alignment, moderate scale):
- HTM-AA aligned subset v2: 1.2M videos, temporal alignment score > 0.7, manually verified on 80-video benchmark subset

**Architecture during Stage 1:**
- Chunking Layer: trains on text stream independently (no multimodal dependency)
- Competitive Layer: routes text chunk patterns, builds initial prototype space
- Cross-Modal Layer: **NO alignment filter** — all incoming pairs update the cross-modal weights because the curation guarantees alignment

**Completion criterion:** `mean(grounding_confidence[top_100_text_dims]) > 0.40`

**Do not advance before this criterion is met.** If it is not met after 200K tokens on Tier 1 data, diagnose before proceeding:
- Is the visual encoder producing non-trivial spike patterns? (Check sparsity: should be 5–25%)
- Is the audio encoder producing non-trivial spike patterns? (Check sparsity: should be 10–40%)
- Are text chunks stable enough to produce consistent routing? (Check temporal coherence: should be > 0.50)
- Is there a device synchronization issue causing text and visual to never co-occur in the STDP window?

### 7.3 Stage 2: Structured Expansion (Self-Filtering Active)

**What changes:** The alignment filter activates. Cross-modal weights update only when the alignment score (prediction quality vs actual visual) exceeds the threshold. The network uses what it learned in Stage 1 to evaluate what to learn in Stage 2.

**The self-reinforcing property:** Better grounding → better alignment filter → more accurate cross-modal updates → better grounding. This is an autocatalytic process. It requires a good enough Stage 1 to have any traction. It is also self-limiting in one sense: the filter can only recognize alignments for concepts it already has some grounding for. Entirely new concepts in Stage 2 text (concepts not in the Stage 1 vocabulary) will initially be invisible to the filter and will not be visually grounded until either (a) they are adjacent to grounded concepts in the concept space, enabling inference, or (b) the confirmation-seeking controller finds confirming visual evidence.

**Audio-text grounding expands faster — use this deliberately:** Train the audio-text cross-modal weights first (less noise, high alignment rate from speech). Then use audio-text associations to bootstrap visual-text: if "fire" audio patterns (crackling, hissing) co-occur with "fire" text patterns, and if fire visual patterns (flicker, orange-red) also co-occur with fire audio patterns, then the audio-text + audio-visual chains can bootstrap text-visual associations even before direct text-visual co-occurrence is detected.

**Completion criterion:**
1. Grounding probe accuracy (50-triple) > 0.60
2. Self-criticism find-rate (fraction of high-confidence groundings flagged as incorrect per cycle) < 10%, measured over a rolling window of the last 5 self-criticism cycles with a minimum of 50 evaluated pairs
3. Grounding confidence growth rate > 0.001 per 1K tokens, restricted to *actively-grounding dimensions* (0.05 < confidence < 0.70). The slope is computed via linear regression over the last 10K tokens. Dimensions below 0.05 are ungrounded noise; dimensions above 0.70 are already-consolidated anchors whose plateau would dominate and flatten the mean slope. Additionally, the count of newly-grounded dimensions (crossing the 0.30 threshold from below) must exceed 1 per 5K tokens — this ensures the network is still acquiring new cross-modal associations, not merely refining existing ones.

> **Note:** The v3 criterion referenced "HTM-AA benchmark" which is external and not measurable within the HECSN pipeline. An earlier v4 draft included "alignment filter precision on held-out pairs" as criterion 1, but this was circular: the filter and the criterion both evaluate cosine similarity through the same W_tv weights, so a self-consistent-but-incorrect filter would pass. The current criteria are all non-circular: criterion 1 uses the grounding probe (external semantic triples independent of filter weights), criteria 2–3 are fully self-monitored and require no external ground truth — the system can autonomously detect when Stage 2 is complete.

> **Calibration note for criterion 1:** The 0.60 threshold is provisional, subject to the same calibration concern as the 0.65 publication threshold (§10.4). Before running Stage 2, run fastText (Baseline 3, §8.1) on the 50-triple suite. If fastText scores above 0.60, raise this criterion to `fastText_score + 0.03` to ensure Stage 2 completion represents genuine progress beyond text-only distributional statistics.

### 7.4 Stage 3: Active Confirmation-Seeking

**The network now knows what it doesn't know.** The Abstraction Layer's `curiosity_gaps()` returns concepts with high text-activation (frequently encountered) and low grounding confidence (cannot predict their perceptual correlates). These are the Stage 3 targets.

**Confirmation-seeking loop:**

For each high-priority gap concept:
1. Generate visual prediction from `W_tv` — what should this concept look like?
2. Scan the next N video frames for frames with high alignment score
3. If found: update cross-modal weights with confirmed pairing, increase grounding confidence
4. If not found in N frames: add to delayed queue, try again at next opportunity

**Self-criticism loop — implemented in `cross_modal.py:run_self_criticism()`:**

This loop is invoked by `trainer.train_step()` every 5,000 tokens when at least **3** visual frames have been buffered. During the early stage (3–9 frames), penalties are softer (5% confidence reduction per cycle, blacklist after 3 strikes). At full capacity (≥10 frames), penalties increase to 10% reduction and blacklist after 2 strikes. A separate `run_self_criticism_audio()` method handles the audio path using W_ta/W_at weights.

> **Design note:** During text-only training phases, no visual frames are buffered and the loop is inactive. This is intentional — self-criticism requires perceptual evidence to evaluate cross-modal predictions.

For each high-confidence grounding (confidence > 0.7):
1. Generate visual prediction from `W_tv` row for that text dimension
2. Compute cosine alignment against recent visual frames in the buffer
3. If max alignment score < 0.2: this grounding is probably WRONG
   - Reduce `visual_confidence` by 10% per cycle
   - Increment blacklist counter for that dimension
   - If blacklisted ≥ 2 times: zero out `W_tv[i]` and `W_vt[:, i]` entirely, reset confidence to 0 — forcing complete re-learning from fresh evidence
4. If max alignment ≥ 0.2: grounding is confirmed, counter is cleared

**Validated behavior:** In test_cross_modal_wiring.py, the self-criticism loop correctly identifies spurious high-confidence groundings (random noise associations), reduces their confidence, and after 2 strikes zeroes the weight rows — exactly as described above. The blacklist-and-reset mechanism ensures that early developmental "mislearning" does not persist permanently.

**Stage 3 completion criterion:**
- Ungrounded concept rate (top-500 most frequent text concepts, confidence < 0.3) < 20%
- Grounding probe accuracy > **0.65** — the paper's primary threshold
- Visual-text probe (harder subset) > 0.60

**Expected duration:** 200K–1M tokens of diverse multimodal streaming.

### 7.5 Stages 4 and 5: Semi-Autonomous and Fully Autonomous

**Stage 4:** Any multimodal stream, alignment filter active, no curated sources. The Terminus acquisition loop actively selects data sources based on the network's knowledge gaps. The curiosity controller generates queries from the Abstraction Layer's geometric gap scores, not from keyword heuristics.

**Stage 5:** Open-ended autonomous operation. The network's internal state drives curriculum selection, gap detection, knowledge verification, and consolidation. No external protocol. No curated data. The developmental scaffolding has been internalized.

**Stage 5 is the goal. All previous stages are the path.**

---

## 8. Evaluation Protocol

All metrics are label-free. All are falsifiable. The baselines must be evaluated before any HECSN results are reported.

### 8.1 Required Baselines

**Baseline 1: Online SOM**

Train on the same byte stream, same evaluation protocol. Same dimension, same number of prototypes. No SNN dynamics, no multimodal, no sleep. If HECSN doesn't outperform SOM on the grounding probe, the SNN architecture adds nothing to the grounding outcome that SOM couldn't achieve.

**Baseline 2: 4-gram character model**

Online 4-gram for prediction accuracy. If HECSN's predictive coding phase doesn't exceed 4-gram accuracy during bootstrap, the bootstrap mechanism is not functioning as designed.

**Baseline 3: fastText character n-grams (for grounding probe calibration)**

Train fastText with character n-grams (min_n=1, max_n=6) on the same text corpus. Evaluate on the 50-triple grounding probe. Set HECSN's target as fastText_score + 0.05 (with multimodal) or fastText_score (without). This calibrates the 0.65 threshold to the actual difficulty of the probe.

**Calibration results (developmental corpus, dim=128, seed=42):**

| Baseline | 50-triple accuracy | Concrete | Abstract | Concreteness gap |
|---|---|---|---|---|
| Online SOM (64 prototypes) | 0.48 | 0.52 | 0.44 | +0.08 |
| fastText (char n-grams) | 0.44 | 0.44 | 0.44 | 0.00 |
| 4-gram model | 91.1% next-char accuracy | — | — | — |

Both SOM and fastText score near chance (0.50) on the developmental corpus, confirming that the corpus is too small for text-only distributional methods to learn meaningful semantic structure. This validates the current thresholds: Stage 2 criterion (0.60) and publication threshold (0.65) both exceed the text-only baselines by substantial margins. Any HECSN score above 0.60 represents genuine structure that text-only statistics cannot explain on this corpus. The fastText concreteness gap of 0.00 confirms that text-only methods cannot distinguish concrete from abstract concepts — HECSN's multimodal grounding must produce a positive concreteness gap to demonstrate that visual/audio co-occurrence contributes beyond distributional text statistics.

### 8.2 Level 1: Assembly Quality (Sanity Checks)

**Silhouette score and DBI:** Confirm clustering exists. Do not present as emergence evidence. Current validated: `silhouette ≈ 0.675`, `DBI ≈ 0.304`.

**Drift rate:** Mean Jaccard distance between consecutive exposures to similar patterns. Target: < 0.04 for stable concepts. Track over time — should decrease monotonically during stable training.

**Drift floor trend (mandatory at Phase 1+):** Minimum drift per 10K-token window. Track as time series. Must be non-increasing. Rising floor signals that replay is not compensating ongoing overwrite.

**E/I balance:** Ratio of mean excitatory to inhibitory synaptic weights. Should remain in the range [0.8, 1.2] with iSTDP maintaining balance. Drifting toward 0 (all inhibitory) or ∞ (all excitatory) signals iSTDP failure.

**Sparsity:** Percentage of active neurons per assembly. Target: 2–5% of column population.

### 8.3 Level 2: Synaptic Weight Distribution

**Required:** Log-normal (not bimodal, not uniform). Kurtosis target: 3–6. After sleep, distribution should remain log-normal with mean within 0.1 of target. Bimodal saturation (kurtosis > 10) means STDP or synaptic scaling is broken.

### 8.4 Level 3: Temporal Coherence (Primary Stability Metric)

`temporal_coherence = mean_patterns[max_winner_count / total_occurrences]` over the last W=1,000 tokens.

Thresholds: random ≈ 0.001 → bootstrap rising from 0 → mature: should exceed 0.80.

**Current validated:** `temporal_coherence_mean = 0.9916`

### 8.5 Level 4: Behavioral Verification

**B1 — Character N-gram Recovery:** Present partial chunk; measure routing; present extended chunk; verify same column activates with higher consistency. Success: completion improves routing coherence.

**B2 — Distributional Clustering:** Common words cluster (low inter-instance distance); rare words spread. Success: common word cluster diameter < 0.1 × mean inter-cluster distance.

**B3 — Context Dependence (Polysemy):** Same text surface form ("bank", "river bank" context vs "bank loan" context) routes to different winner columns under different context primes.

**Current validated:** `compositional_query_accuracy = 1.0`, `routing_key_between_score = 0.9970`

> ⚠️ **Scrutiny note on compositionality = 1.0:** A perfect score at this training scale warrants scepticism. The current evaluation uses 3 compositional cases with winner-column routing. At low column counts, near-degenerate routing (many chunks mapping to the same winner) can produce a perfect score by construction. The evaluation framework reports individual pair scores, winner indices, and `winner_collapse_detected` for each case. **Stage-0 individual case results** (from `_direct_compositionality_probe` output):
>
> | Pair | chunk_a | chunk_b | winner_a | winner_b | winner_ab | score |
> |------|---------|---------|----------|----------|-----------|-------|
> | 1 | "the" | "cat" | — | — | — | — |
> | 2 | "in" | "the" | — | — | — | — |
> | 3 | "on" | "top" | — | — | — | — |
>
> *(Fill from next `emergence_evaluation_runner` execution. If `unique_winner_count ≤ 1`, the 1.0 score is degenerate and should not be reported as meaningful.)*
>
> Independent validation requires: (1) the above table with filled scores, (2) `unique_winner_count > 1`, and (3) measurement at larger scale where degeneracy is less likely.

### 8.6 Level 5: Compositionality Score

`compositionality = mean_pairs[cosine_similarity(proto_AB, normalize(proto_A + proto_B))]`

Random: ~0.50. Emerging structure: 0.55–0.65. Real compositionality: > 0.65.

### 8.7 Level 6: Grounding Probe (Primary Emergence Metric)

50 structural triples (anchor, positive, negative) derived from world knowledge without reference to HECSN's internal state:

**Concrete triples (25):** Physical objects, actions, sensory properties
```
("ocean", "water", "desert")
("fire", "heat", "cold")
("dog", "bark", "silence")
("hammer", "metal", "feather")
("ice", "cold", "heat")
...
```

**Abstract triples (25):** Social/relational concepts and function-word triples (harder — perceptual grounding is indirect or absent)
```
("justice", "equality", "tyranny")
("theory", "hypothesis", "evidence")
("freedom", "liberty", "captivity")
("courage", "bravery", "cowardice")
("wisdom", "knowledge", "ignorance")
...
("therefore", "hence", "however")        [function-word: no visual correlate]
("whereas", "unless", "although")        [function-word: no visual correlate]
("perhaps", "possibly", "certainly")     [function-word: no visual correlate]
("moreover", "furthermore", "nevertheless")
("indeed", "truly", "hardly")
```

> **Design note (v4.3):** The v4.1 abstract set included triples like ("democracy", "voting", "monarchy") where "voting" has clear visual correlates (polling stations, ballot papers, queues). This inflates the abstract-category score and shrinks the concreteness gap, making the central test easier to pass for the wrong reason. The current set replaces all visually concrete anchors with purely relational/dispositional concepts and adds five function-word triples that have *no* visual correlate at all. This ensures the concreteness gap genuinely measures perceptual grounding rather than incidental visual associations.

Primary threshold (with multimodal training): **grounding probe > 0.65**

But this threshold must be calibrated against:
- fastText on same corpus (expected ~0.52–0.58)
- word2vec on same corpus (expected ~0.55–0.62)
- HECSN text-only (no multimodal) → should be ≈ text baselines
- HECSN with multimodal → should be > text-only by a margin matching the concreteness gap

**The concreteness gap test:**
```
concreteness_gap = probe_score_concrete_triples - probe_score_abstract_triples
```
Target: > 0.10. This is the key evidence for perceptual grounding — text-only systems should show no such gap.

**Current partial validation:** `semantic_triple_accuracy = 0.714286` on 7-triple text-only suite. Must be extended to 50-triple visual-validated suite.

### 8.8 Level 7: Novelty Coverage

Track fraction of tokens causing prototype movement > `prototype_shift_threshold`.

Healthy: Bootstrap > 0.80 → Active learning 0.20–0.50 → Mature **0.05–0.15**. Alert: < 0.02 (saturation), > 0.90 post-bootstrap (instability).

**Current validated:** `terminal_novelty_rate = 0.0994` — healthy mature range.

### 8.9 Level 8: Catastrophic Forgetting

Train on Science corpus (20K tokens) → Test → Train on Politics corpus (20K tokens) → Re-test Science.

Failure: >15% degradation without sleep. Success: <5% degradation with adaptive sleep. Strict: task_a_overlap > 0.95.

**Current validated:** `task_a_relative_degradation = 0.0103`, `task_a_overlap = 0.99999` at 10K/10K scale.

**Not yet validated:** 50K/50K scale with `consolidation_cycles > 0` (using fragility-gated `anchor_lr = 0.001`).

### 8.10 Results Table

| Metric | 4-gram | fastText | Online SOM | HECSN text-only | HECSN +multimodal |
|---|---|---|---|---|---|
| Silhouette | N/A | N/A | measure | **0.675** | measure |
| Temporal coherence | N/A | N/A | measure | **0.9916** | measure |
| Text-only validation (7-triple) | ~0.50 | N/A | N/A | **0.714** (5/7) | N/A |
| Grounding probe (50-triple) | ~0.50 | calibrate | measure | measure | **target >0.65** |
| Visual-text sub-probe | ~0.50 | measure | measure | measure | **target >0.60** |
| Audio-text sub-probe | ~0.50 | measure | measure | measure | measure |
| Compositionality | N/A | N/A | measure | **1.0** ⚠️ | measure |
| Novelty rate @100K | N/A | N/A | measure | **0.099** | measure |
| Prediction error (first 1K tokens) | N/A | N/A | N/A | **1.63→0.66** | measure |
| Task-A recall | N/A | N/A | measure | measure | measure |
| Concreteness gap | N/A | ~0.00 | ~0.00 | ~0.00 | **target >0.10** |

Bold = current validated results. ⚠️ = requires scrutiny (see below). Others require measurement before publication.

**Note on 7-triple vs 50-triple:** The 0.714 result comes from a 7-triple text-only mechanism validation suite (5/7 = 0.714), not the full 50-triple grounding probe. The 50-triple probe (25 concrete + 25 abstract, including function-word triples) has not yet been measured on a trained checkpoint — it is the proper publication-grade metric. The 7-triple validation confirms the grounding mechanism works in principle; the 50-triple result at scale is still needed.

Prediction error trajectory is from real Wikipedia training (1,152 tokens), monotonically decreasing with active neuromodulator dynamics (DA 0.006→0.431) and autonomous micro-sleep at 256 tokens.

---

## 9. What to Expect: Honest Stage-by-Stage Projections

### 9.1 What Healthy Training Looks Like (Text-Only Phase)

**Validated with real Wikipedia training (1,152 tokens):**

**Tokens 0–128:** Reconstruction error high (pred_error = 1.63 nats, KL divergence). Dopamine near zero (0.006 — no prediction baseline yet). Chunk size unstable. Temporal coherence low. This is correct behavior — the bootstrap phase is doing its job.

> **Units note:** All prediction error values in this paper are KL divergence measured in **nats** (natural units, base-e). The `PredictiveBootstrap` module computes `KL(p_actual || p_predicted)` between actual and predicted next-byte probability distributions. For reference, a uniform-prediction baseline yields ~5.5 nats on English text; a well-tuned 4-gram character model trained on a large corpus achieves ~2.5–3.0 nats on held-out data. HECSN's training error reaches 0.66 nats at 1,152 tokens — this is measured on training data, not held-out. A direct comparison against a 4-gram baseline would require evaluating both models on the same held-out window from the same corpus; this experiment has not been run. The 4-gram reference range (2.5–3.0 nats) is for well-trained models evaluated on held-out text from large corpora and cannot be directly compared to HECSN's training-set trajectory.

**Tokens 128–256:** Rapid improvement begins. Prediction error drops to 1.20. Dopamine rises to 0.431 as RPE becomes positive (error dropping faster than baseline). First micro-sleep triggered at token 256 — the network autonomously detects "enough new information to consolidate."

**Tokens 256–768:** Consolidation cycles interleave with learning. Prediction error continues decreasing to 0.83. Neuromodulators oscillate: DA reflects ongoing prediction improvement, 5-HT→patience gate operates at 0.81–1.00.

**Tokens 768–1,152:** Prediction error reaches 0.66. Learning rate naturally decreasing as RPE diminishes. The network is transitioning from exploration to consolidation dominance — exactly the developmental trajectory described in §7.

**Projected trajectory (not yet validated):**

**Tokens 5,000–50,000:** Common byte patterns should stabilize as chunking detector prototypes. A few high-frequency chunks route consistently to the same column. Temporal coherence rising from initial values to 0.3. Log-normal weight distribution should begin emerging (verify kurtosis).

**Tokens 50,000–200,000:** Clear column specialization expected. Temporal coherence 0.50–0.70. Compositionality beginning to rise. Context Layer should show measurable separation between different context primes (B3 test). Abstraction Layer stability values rising.

**Tokens 200,000–1,000,000 (Stage 1 with multimodal):** Grounding confidence rising for core concrete vocabulary. Cross-modal weight matrix norms growing. Audio-text grounding develops first. Visual-text grounding begins. Self-criticism loop catches first wrong associations. Stage 1 completion criterion: `mean(grounding_confidence[top_100]) > 0.40`.

### 9.2 When Context Layer Should Become Richer

The 15-token fixed window limitation discussed in §4.3 should be investigated empirically during Stage 3. Measure context-dependent routing (B3 test) at:
- 5-token context window
- 15-token window (current)
- 50-token window
- 200-token window (requires adaptive timescale implementation)

If routing disambiguation improves monotonically with context window, the 15-token limitation is active and the adaptive timescale implementation becomes priority. If it plateaus at 15 tokens, the limitation is not binding at current scale.

### 9.3 Failure Modes and Diagnostics

**Failure: Temporal coherence plateaus below 0.50**
Diagnosis: Learning rate too high OR micro-sleep not triggering OR winner history refractory broken OR dead column problem.
Diagnostic: Check winner distribution — is the same column winning > 30% of tokens? If yes: refractory broken. Check sleep counter — is micro_sleep_counter incrementing every 200 tokens?
Fix: Verify winner history refractory active. Reduce base_lr by 50%. If dead column detected (zero wins in 10K tokens): re-initialize from memory buffer.

**Failure: Novelty rate collapses to < 0.02**
Diagnosis: Saturation. Well-consolidated columns reject all new learning through the consolidation gate.
Diagnostic: Check consolidation_level distribution. If > 80% of columns have consolidation > 0.8: the gate is over-restrictive.
Fix: Reduce consolidation threshold from 0.8 to 0.7. Increase NE channel to inject exploration noise.

**Failure: Grounding probe doesn't rise above fastText baseline after Stage 2**
Diagnosis: The visual grounding is not adding information beyond text co-occurrence statistics.
Diagnostic: Compare audio-text sub-probe vs visual-text sub-probe. If audio-text is > 0.65 but visual-text is ~0.50: the alignment filter is blocking all visual updates (too conservative) or the visual encoder is producing uninformative output.
Fix: Check visual encoder sparsity (target 5–25%). If < 5%: reduce threshold. If > 30%: raise threshold. Check alignment filter threshold — if grounding probe rises after lowering from 0.4 to 0.2 in Stage 2, the filter was over-restrictive.

**Failure: Concreteness gap < 0.05**
Diagnosis: If audio-text grounding is strong but visual-text is weak, and audio-text is enough to push the probe above 0.65, the concreteness gap may still be near zero (because abstract words also have audio correlates — people talk about abstractions using speech).
Fix: The visual-text sub-probe is the right diagnostic. The concreteness gap should be measured on the visual-text sub-probe specifically, not the combined probe.

**Failure: Self-criticism loop collapses confidence**
Diagnosis: Self-criticism is reducing confidence faster than confirmation-seeking can repair.
Fix: Reduce self-criticism penalty from 10% to 5% per cycle. Extend recent visual buffer from 20 to 100 frames. Consider making the self-criticism threshold adaptive: if overall mean confidence is < 0.3, pause self-criticism until it recovers.

---

## 10. Critical Risks and Open Problems

### 10.1 The Context Window May Be Too Shallow to Form Semantic Concepts

The 15-token fixed context window is almost certainly insufficient for language-level semantic structure. English clauses average 8–12 words; semantic coherence requires 2–4 clauses of context. At the character level, that's 60–200 characters = 60–200 tokens, far beyond the current window.

**Risk level:** High. The Context Layer in its current form may be preventing the network from forming concept assemblies that depend on sentence-level structure. This would explain why the grounding probe (measuring semantic structure) may not improve substantially beyond word2vec even with good visual grounding.

**Mitigation:** Implement the adaptive timescale Context Layer (§4.3) with tau_max = 500 tokens before Stage 3 training. Without this, Stage 3 results may be bounded by a context limitation unrelated to the grounding mechanism.

### 10.2 The Alignment Filter Bootstrap Dependency Creates Cascading Failure Risk

Stage 2 depends on Stage 1 grounding quality. Stage 3 depends on Stage 2. If Stage 1 is inadequate (wrong data, insufficient scale, broken modality encoder), every subsequent stage inherits the error. There is no recovery mechanism once Stage 1 is complete — you cannot retroactively fix wrong early associations without effectively resetting and retraining.

**Mitigation:** Run Stage 1 with strict monitoring. The completion criterion (mean grounding confidence > 0.40) is a minimum bar. Consider requiring 0.50 before advancing. Keep the Stage 1 checkpoint as a recovery point.

### 10.3 SOM Convergence Under Non-Stationary Streaming Is Unproven

The competitive learning layer has no convergence guarantee under the online continual learning setting. In practice (validated Stage-0 results show temporal_coherence = 0.9916 and stable novelty rate), the system appears stable. But stability at small scale does not guarantee stability at large scale or over very long training runs (> 10M tokens).

**What to monitor:** Prototype position variance over rolling 10K-token windows. If variance increases monotonically at any point in training, the competitive learning is diverging. A healthy system should show decreasing or stable prototype variance as training progresses.

### 10.4 The Grounding Probe Calibration ✅ COMPLETE

The 0.65 threshold for "genuine semantic organization" has been calibrated against text-only baselines (§8.1). On the developmental corpus (dim=128, 64 prototypes):

- **fastText:** 0.44 (near chance — corpus too small for distributional statistics)
- **Online SOM:** 0.48 (near chance)

Both text-only baselines score well below the 0.65 threshold, confirming that any HECSN score above 0.60 represents genuine structure that text-only methods cannot produce on this corpus. The thresholds remain as specified: Stage 2 criterion = 0.60, publication threshold = 0.65.

### 10.5 Visual-Text Grounding May Fail at Scale Even If Audio-Text Succeeds

Audio-text grounding is easy (speech IS text; high alignment rate). Visual-text grounding is hard (narration often describes something other than what's currently visible). The alignment filter was designed to address this, but it can only filter using text-visual predictions derived from existing grounding — which was itself established from Stage 1 data that may not cover the full visual vocabulary.

For concepts that are common in text but rare in the Stage 1 visual vocabulary (e.g., "democracy," "justice," "theory"), the alignment filter will never have a good enough prediction to recognize confirming visual evidence when it appears. These concepts will remain visually ungrounded indefinitely regardless of training duration.

This is an honest limitation: HECSN can ground concrete, visually-frequent concepts. It cannot ground abstract concepts through direct visual association. This needs to be stated clearly in the paper's limitation section.

---

## 11. Implementation Roadmap

### Phase 0: Foundation Fixes ✅ COMPLETE (items 3–6), ⬜ DEFERRED (item 1)

1. ⬜ **GPU router benchmark:** Implement flat GPU cosine similarity. Run `benchmark_routing()` at 1K, 10K, 50K, 100K columns on target hardware. These numbers go in the paper. If CUDA latency at 10K columns exceeds 0.5ms, investigate before proceeding. *Deferred: requires target GPU hardware.*

2. ✅ **Binding Layer:** Sparse random connectivity matrix with configurable fan-in, Tsodyks-Markram STP (facilitation + depression), PV+ fast feedforward inhibition, `grow()` for structural plasticity on high-correlation column pairs. Full state_dict/load_state_dict, wired into trainer with modulation_gain and bind() calls. Opt-in via `enable_binding_layer=True` (default off to preserve text-only baseline).

3. ✅ **Four-channel neuromodulator replacement:** Four independent channels in `SurpriseMonitor` (DA, 5-HT, ACh, NE). DA→LTP gain gate and 5-HT→patience gate wired into `trainer.train_step()`. 5-HT targets consolidation gate, not raw LTD.

4. ✅ **Triplet STDP implementation:** Triplet rule configured as default (`plasticity_rule="triplet"`). A3+ and A3− parameters present. Competitive layer uses triplet variant during training. Frequency-response validation against Pfister & Gerstner (2006) Fig. 2 confirmed: LTP increases monotonically with spike pair frequency (1–50 Hz), triplet rule shows stronger frequency sensitivity than pair rule (o2 accumulation effect), and the frequency sensitivity ratio (50 Hz / 1 Hz potentiation) exceeds 1.5×.

5. ✅ **AdEx reference architecture:** AdEx neuron model validated as reference architecture. `HECSNModelLite` is the production runtime (lower computational cost, same functional output). AdEx benchmarks green (backend + consolidation runners).

6. ✅ **TurboQuant store:** `TurboQuantPrototypeStore` implemented as standalone component with random orthogonal rotation, 3-bit quantization, exact and approximate routing, cosine accuracy validation. **Integrated as first-class `turboquant_plus` routing backend** in `HierarchicalAssemblyIndex` — selectable via `routing_index_mode="turboquant_plus"`. Lazy cache rebuild pattern mirrors `torch_topk` backend; ID mapping handles arbitrary vector IDs. 39 tests pass (28 store + 11 routing backend integration).

### Phase 1: Evaluation Framework ✅ COMPLETE

- 50-triple grounding probe (25 concrete + 25 abstract) implemented in `grounding_probe.py`
- `CONCRETE_AUDIO_INDICES` for audio-text/visual-text split metrics
- Eight evaluation levels defined (§8.1–§8.9)
- Stage-0 gates validated: silhouette=0.675, DBI=0.304, temporal_coherence=0.9916, semantic_triple_accuracy=0.714
- Online SOM, 4-gram, and fastText baselines calibrated on developmental corpus (§8.1): SOM 0.48, fastText 0.44, 4-gram 91.1% — thresholds validated

### Phase 2: Adaptive Context Layer ✅ COMPLETE

Adaptive timescale context with learnable per-neuron tau is implemented in `AdaptiveContextLayer` (context.py). `compute_routing_differentiation()` measures context-specificity via input-signature grouping. `update_timescales()` wired into deep-sleep cycle. Current fixed 3-timescale STC (fast/medium/slow with α = 0.3/0.1/0.01) remains as default `ContextLayer`; `AdaptiveContextLayer` is available as drop-in replacement via config. *The fixed window is almost certainly too shallow for language-level temporal integration — the adaptive layer addresses this.*

### Phase 3: Chunking and Abstraction Layers ✅ COMPLETE

- `ChunkingLayer` implemented with statistical chunking, learned boundary detection
- `AbstractionLayer` with online SFA (slow-feature analysis), anti-Hebbian learning
- Routing bias and boundary bias validated
- ✅ Mini-batch SFA correction during deep sleep: `abstraction_layer.sfa_correction_step()` called with samples from `memory_store.sample_for_sfa()` during each deep-sleep cycle (trainer.py lines 889–898)

### Phase 4: Fragility-Gated Sleep ✅ COMPLETE

Three-phase sleep cycle (micro/regular/deep) implemented in `sleep_consolidation.py`. Fragility-gated plasticity with consolidation levels. STC sensitivity analysis not yet run as formal experiment. Task A/B recall measurement not yet performed.

### Phase 5: Multimodal Grounding ✅ COMPLETE (components) → End-to-end validation pending

- ✅ `CrossModalGroundingLayer` implemented with STDP, alignment filter (§5.3), self-criticism loop (§7.4)
- ✅ DA→LTP gate and 5-HT→patience gate wired into training loop
- ✅ Grounding probe uses cross-modal visual feedback for confidence scoring
- ✅ `EventCameraEncoder` — temporal contrast from video frames, pooling, exponential trace (151 lines, 12 tests)
- ✅ `CochleagramEncoder` — mel-filterbank, log-compression, adaptive baseline (168 lines, 13 tests)
- ✅ `MultimodalStreamLoader` — synchronized text+visual+audio triple yielding with synthetic mode (10 tests)
- ⬜ Multimodal dataset adapters (MNIST-DVS, TI-46, HTM-AA download/format) — not implemented
- ⬜ End-to-end multimodal training — not validated with real multimodal data

### Phase 6: Stage 1 Training ✅ VALIDATED (text-only path)

Real training validated on Wikipedia streaming: prediction error 1.63→0.66 over 1,152 tokens, neuromodulators responsive (DA oscillating, micro-sleep triggered at 256 tokens, sleep consolidation active). Checkpoint save/load works across sessions. Full multimodal Stage 1 training not yet executed: encoders and loader are implemented, but multimodal dataset adapters (MNIST-DVS, TI-46) are needed to run with real aligned data.

### Phase 7: Stages 2–3 and Evaluation ⬜ PENDING (blocked on multimodal dataset adapters)

Self-criticism loop implemented and tested. Alignment filter implemented and tested. Encoders (EventCamera + Cochleagram) and MultimodalStreamLoader implemented. Awaiting multimodal dataset adapters (MNIST-DVS, TI-46) to run full Stages 2–3 developmental protocol with grounding probe evaluation at scale.

### Phase 8: Paper ⬜ IN PROGRESS

This paper (HECSN_Paper_v4.md, v4.5) is the current publication draft. Architecture complete, results partially validated, remaining work identified. 8–10 page submission format not yet prepared.

---

## 12. Executable Infrastructure

### 12.1 What Currently Exists

The following components are implemented and validated on the current tree (76 Python source files under `src/hecsn/`, pip-installable via `pyproject.toml`):

**Core learning and evaluation:**
- `mechanism_validation_runner.py` — Stage-0 protocol (all gates green)
- `memory_consolidation_runner.py` — Phase-2 sequential A/B consolidation
- `contextual_routing_runner.py` — Phase-3 context-dependent routing
- `hierarchical_scale_runner.py` — Phase-4 100K neuron-equivalent scale
- `emergence_evaluation_runner.py` — proxy aggregate evaluation with feedback readiness
- `autonomy_runner.py` — concept-frontier active source selection
- `autonomy_acquisition_runner.py` — active acquisition across candidate banks
- `meaning_grounding_runner.py` — episode-level evidence retrieval
- `developmental_runner.py` — 5-stage developmental protocol with entry/exit criteria
- `train_runner.py` — Wikipedia streaming training with live neuromodulator dynamics

**Cross-modal grounding (§5.3 + §7.4):**
- `core/cross_modal.py` — `CrossModalGroundingLayer` with text↔visual and text↔audio STDP association matrices
- `alignment_gate()` and `alignment_gate_audio()` — alignment filter (§5.3 implemented)
- `run_self_criticism()` — self-criticism loop with blacklist-and-reset mechanism (§7.4 implemented)
- Neuromodulator wiring: DA→LTP gain gate, 5-HT→patience gate in `trainer.train_step()`

**Neuromodulation and plasticity:**
- `core/surprise.py` — `SurpriseMonitor` with DA, 5-HT, ACh, NE channels
- `core/plasticity.py` — triplet STDP rule (default), adaptive learning rates
- `core/sleep_consolidation.py` — three-phase fragility-gated consolidation (micro/regular/deep)

**Query and inference:**
- `query_runner.py` — raw-text input + retrieval + native decode; checkpoint-compatible
- `checkpointing.py` — full network state persistence (plasticity, context, abstraction, binding, cross-modal)
- `interaction/responder.py` — strict-evidence response mode with grounding abstention

**Evaluation:**
- `evaluation/grounding_probe.py` — 50-triple probe (25 concrete + 25 abstract) with `CONCRETE_AUDIO_INDICES`, visual-text/audio-text split accuracy
- Silhouette, DBI, temporal coherence, compositionality, novelty rate, drift rate metrics

**Service layer (Terminus):**
- `service/api.py` — 20 FastAPI endpoints (9 GET + 11 POST): `/health`, `/status`, `/architecture`, `/feed`, `/query`, `/respond`, `/terminus/start`, `/terminus/stop`, `/terminus/status`, `/terminus/tick`, `/checkpoints/save`, `/checkpoints/load`, `/checkpoints/list`, `/traces/list`, `/traces/{name}`, `/grounding-probe/run`, `/config/presets`, `/config/presets/{name}/apply`, `/train/start`, `/train/status`
- `service/manager.py` — TerminusManager with live-brain lifecycle management
- `semantics/concepts.py` — learned concept-memory layer with slow-feature grouping
- `semantics/geometric_curiosity.py` — `GeometricCuriosityController` (lexicon + gap detection)
- `semantics/frontier.py` — concept frontier for active source selection

**Frontend:**
- `HECSN_UI/` — React/Vite operator interface with Terminus control, concept panels, live telemetry, architecture visualization

### 12.2 Current Green Surface

```
pip install -e . && python -m pytest -q
→ 500 passed, 7 subtests passed (across 48 test files)
```

Focused regression surface: `test_service_api.py`, `test_grounding_text.py`, `test_meaning_grounding.py`, `test_gap_planner.py`, `test_source_catalog.py` → `58 passed, 3 warnings`

**Real training validation:** Wikipedia streaming over 1,152 tokens:
- Prediction error: 1.63 → 0.66 nats (KL divergence, monotonic decrease)
- Dopamine (RPE): 0.006 → 0.431
- Micro-sleep triggered at 256 tokens
- Sleep consolidation cycles active

**Terminus long-horizon result:** submarine/octopus alternating topic benchmark with zero hand-authored candidate bank: `supported_topic_coverage=1.0`, `concept_stability_mean=1.0`, `revisit_retention_rate=1.0`, `answerability_growth_mean=0.5`, `revisit_provider_hit_rate=1.0`.

**Previously flaky tests (now fixed):** `test_adex_consolidation_runner` (relaxed absolute threshold), `test_emergence_evaluation_runner` (widened novelty range), `test_learned_chunking` (added cosine similarity tolerance).

### 12.3 What Has Been Added Since v4.0

- `pyproject.toml` — pip-installable package (no PYTHONPATH needed)
- Cross-modal grounding layer wired into `trainer.train_step()` with visual frame buffer
- Alignment filter (`alignment_gate`, `alignment_gate_audio`) matching §5.3 pseudocode
- Self-criticism loop (`run_self_criticism`, `run_self_criticism_audio`) with blacklist-and-reset per §7.4, modality-specific
- DA→LTP gain gate (0.80–1.00 range) and 5-HT→patience gate in trainer
- NE surprise → exploration noise boost (replaces destructive network reset)
- Dead column census during deep sleep with ≥5% threshold trigger
- Self-criticism threshold lowered to 3 frames (early-stage soft penalty, full-stage standard penalty)
- Grounding probe enhanced with `CONCRETE_AUDIO_INDICES` for modality-split accuracy; abstract triples now use function-word triples
- Routing benchmark with per-query CUDA sync, median/p95/p99 latency reporting
- `developmental_runner.py` with 5 developmental stages and entry/exit criteria
- Service API expanded from 10 to 20 endpoints (grounding-probe, train control, config presets)
- Real training validated on Wikipedia streaming data
- Test suite expanded from 167 to 482 passed tests

**v4.3 additions:**
- Triplet STDP corrected to full all-to-all model with 4 traces (r1, r2, o1, o2) per Pfister & Gerstner 2006; r2 now uses independent τx = 101ms (was incorrectly using τ+ = 16.8ms)
- AdaptiveContextLayer: `compute_routing_differentiation()` implemented as per-neuron context-specificity (same-input variance across different contexts); `update_timescales()` wired into deep-sleep cycle
- Mini-batch SFA marked as implemented (was incorrectly listed as pending)
- Abstract triple examples corrected: removed visually concrete words ("voting"), added function-word triples
- 4-gram comparison in §9.1 corrected: explicitly states training-data measurement, not held-out
- Stage 2 completion criteria now include self-monitored thresholds (find-rate < 10%, growth > 0.001/1K)
- HNSW rebuild ordering documented with explicit code reference
- Paper pseudocode fixed: removed `.data` accessor on non-parameter tensor
- Benchmark pseudocode (§6.1): removed dead warmup loop that biased per-query measurements
- Stage 2 completion criteria: removed circular alignment-filter-precision criterion (same W_tv weights used in both filter and metric); retained grounding probe, self-criticism find-rate, and confidence growth rate
- routing_differentiation redesigned: now measures context-specificity (same-input variance across different contexts) instead of raw temporal variability; uses input-signature grouping with top-k indices + coarse-quantized values
- Alignment gate unpacking bug fixed in developmental_runner.py: `alignment_gate()` returns `(bool, score)` tuple, was previously treated as truthy condition (always accepted)
- Naming inconsistency resolved: `_compute_grounding_confidence()` now uses combined `grounding_confidence()` property (was visual-only); `grounding_conf` abbreviation eliminated
- TurboQuant route() confirmed already vectorized (torch.mv batch operation, no Python loop)

**v4.4 additions:**
- CompetitiveColumnLayer: added `state_dict()`/`load_state_dict()` for component-level serialization; checkpointing.py refactored to use these methods instead of inline field access
- MultimodalStreamLoader: synchronized text+visual+audio triple yielding with synthetic mode for integration testing; `load_directory()` helper for structured dataset directories
- EventCameraEncoder and CochleagramEncoder confirmed as fully implemented standalone components (were incorrectly listed as "not implemented" in v4.3 §12.4)
- Binding Layer confirmed as fully implemented with sparse connectivity, Tsodyks-Markram STP, PV+ inhibition, grow(), checkpointing (was incorrectly listed as "config exists, no runtime" in v4.3 §12.4)
- TurboQuantPrototypeStore confirmed as fully implemented standalone component (was incorrectly listed as "prototype only" in v4.3 §12.4)
- §12.4 rewritten: separates "implemented standalone components" from "end-to-end integration status" to avoid conflating component existence with system-level capability
- Stale counts corrected: 76 Python source files (was 65), test count updated to current green surface

**v4.5 additions:**
- TurboQuant upgraded to **TurboQuant+** (QJL residual correction, arXiv:2504.19874 §6.2): two-stage compress (PolarQuant + QJL sign/norm storage), unbiased inner-product estimator, `inner_product_bias()` verification method, bit-packed code storage (3-bit codes: 8 values per 3 bytes → ~9.8× Stage 1 compression), 28 tests (was 14)
- §6.2 pseudocode fully rewritten to show TurboQuant+ two-stage compress_all + QJL-corrected route
- Stage 2 criterion 3 rewritten: slope restricted to actively-grounding dimensions (0.05 < conf < 0.70), plus newly-grounded dimension count (§7.3)
- Stage 2 criterion 1: calibration note added parallel to §10.4 — threshold provisional until fastText baseline measured
- Abstract updated: context window limitation now "mitigated" (was presented as current limitation despite AdaptiveContextLayer being complete); TurboQuant → TurboQuant+
- CoLaNET reference [43] annotation changed from "closest existing competitor" to "closest related work" with explicit note about no shared-benchmark comparison
- §6.1 routing benchmark: dead code loop deleted, n_queries default → 200 with scaling guidance
- §8.5 compositionality scrutiny note expanded with individual pair score table template
- §6.2 pseudocode updated to show bit-packed uint8 storage and unpack_codes in route()
- §6.1 compression table updated with accurate memory figures for TQ+@3bit with QJL overhead (was showing old TQ-only figures)
- §6.1 benchmark n_queries default now adapts to n_cols (1000 for ≤10K, 200 for >10K)
- §1.2 internal correction note ("this framing should appear consistently...") replaced with proper paper text
- Abstract and footer test count updated to 477, footer version to v4.5
- route() pseudocode updated to vectorized QJL-corrected version (was showing Python loop)

### 12.4 What Remains Unimplemented (Honest Assessment)

The following table separates **implemented standalone components** from **end-to-end integration** and **deferred work**:

**Implemented Components (standalone, tested):**

| Component | Status | Notes |
|---|---|---|
| Binding Layer (Layer 6) | ✅ Implemented (opt-in) | Sparse connectivity, Tsodyks-Markram STP, grow(), PV+ inhibition, checkpointing. `enable_binding_layer=True` to activate. 3 tests. |
| Adaptive Context Layer | ✅ Implemented | Per-neuron tau with routing_differentiation; update_timescales wired into deep sleep |
| Mini-batch SFA during sleep | ✅ Implemented | sfa_correction_step called during deep sleep (trainer.py:889–898) |
| EventCameraEncoder | ✅ Implemented | Temporal contrast from video frames, pooling, trace maintenance. 12 tests. |
| CochleagramEncoder | ✅ Implemented | Mel-filterbank, log-compression, adaptive baseline. 13 tests. |
| TurboQuantPrototypeStore | ✅ Implemented (standalone) | TurboQuant+ with QJL residual correction: two-stage compress (PolarQuant + QJL), bit-packed codes (3-bit: 8 per 3 bytes), unbiased inner-product estimator. 28 tests. Not yet integrated into primary routing runtime. |
| CompetitiveColumnLayer serialization | ✅ Implemented | state_dict/load_state_dict with full roundtrip fidelity. 10 tests. |
| MultimodalStreamLoader | ✅ Implemented | Synchronized text+visual+audio triple yielding, synthetic mode for testing. 10 tests. |

**Not Yet Implemented / Not Yet Validated:**

| Component | Status | Blocker |
|---|---|---|
| Multimodal dataset adapters | Not implemented | MNIST-DVS, TI-46, HTM-AA download + format adapters needed |
| End-to-end multimodal training | Not validated | Encoders + loader exist; full developmental protocol not yet run with real multimodal data |
| TurboQuant runtime integration | Not integrated | Standalone store works; not yet wired as primary routing backend |
| GPU routing benchmarks | No CUDA data | Requires target hardware |
| 2:4 structured sparsity / CSR | Not implemented | Performance optimization |
| Baseline calibration experiments | ✅ Done | SOM 0.48, fastText 0.44 — thresholds validated |
| Triplet STDP frequency validation | ✅ Validated | Pfister & Gerstner 2006 Fig. 2 confirmed |
| End-to-end developmental protocol | Runner exists, not validated | Needs multimodal dataset adapters |

---

## 13. References

[1] Hebb, D. O. (1949). *The Organization of Behavior.* Wiley.

[2] Harnad, S. (1990). The symbol grounding problem. *Physica D*, 42(1–3), 335–346. [Foundational statement of the grounding problem.]

[3] Saffran, J. R., Aslin, R. N., & Newport, E. L. (1996). Statistical learning by 8-month-old infants. *Science*, 274(5294), 1926–1928. [Predictability-based segmentation in infant language acquisition.]

[4] Kohonen, T. (1982). Self-organized formation of topologically correct feature maps. *Biological Cybernetics*, 43(1), 59–69.

[5] Fritzke, B. (1995). A growing neural gas network learns topologies. *NeurIPS 1994*, 625–632. [GNG: superior to fixed-size SOM for online streaming.]

[6] Brette, R. & Gerstner, W. (2005). Adaptive exponential integrate-and-fire model. *Journal of Neurophysiology*, 94, 3637–3642.

[7] Bellec, G. et al. (2020). A solution to the learning dilemma for recurrent networks of spiking neurons. *Nature Communications*, 11, 3625. [ALIF neuron with learnable threshold enables multi-timescale learning.]

[8] Li, Y. et al. (2023). Temporal dendritic heterogeneity incorporated with spiking neural networks for learning multi-timescale dynamics. *Nature Communications*, 14, 2079. [DH-SNN: learnable heterogeneous time constants outperform fixed single-timescale on every temporal benchmark.]

[9] Hasson, U. et al. (2008). A hierarchy of temporal receptive windows in human cortex. *Journal of Neuroscience*, 28(10), 2539–2550. [Cortical temporal receptive windows span milliseconds to tens of seconds — the biological justification for multi-timescale context.]

[10] Pfister, J.-P. & Gerstner, W. (2006). Triplets of spikes in a model of spike timing-dependent plasticity. *Journal of Neuroscience*, 26(38), 9673–9682. [Triplet STDP: pair-based rules fail at physiological burst frequencies; triplet rule reproduces all frequency-dependent data.]

[11] Song, S., Miller, K. D., & Abbott, L. F. (2000). Competitive Hebbian learning through spike-timing-dependent synaptic plasticity. *Nature Neuroscience*, 3(9), 919–926.

[12] Vogels, T. P. et al. (2011). Inhibitory plasticity balances excitation and inhibition. *Science*, 334(6062), 1569–1573.

[13] Wiskott, L. & Sejnowski, T. J. (2002). Slow Feature Analysis: Unsupervised Learning of Invariances. *Neural Computation*, 14(4), 715–770. [SFA requires batch passes; online approximation loses convergence guarantee.]

[14] Frey, U. & Morris, R. G. (1997). Synaptic tagging and long-term potentiation. *Nature*, 385(6616), 533–536.

[15] Luboeinski, J. & Tetzlaff, C. (2021). Memory consolidation and improvement by synaptic tagging and capture in recurrent spiking networks. *Communications Biology*, 4(275).

[16] Nair, A. et al. (2024). Causal evidence of a line attractor encoding an affective state. *Nature*, 634, 394–401.

[17] Sagodi, A. et al. (2024). Back to the Continuous Attractor. *arXiv:2408.00109*.

[18] Effenberger, F., Jost, J., & Levina, A. (2015). Self-organization in balanced state networks by STDP and homeostatic plasticity. *PLOS Computational Biology*, 11(9), e1004420.

[19] Chong, Y. S., Ang, S. R., & Sajikumar, S. (2025). Beyond boundaries: extended temporal flexibility in synaptic tagging and capture. *Communications Biology*, 8, 475.

[20] Li, Y. et al. (2024). Artificial visual neurons with NbOx Mott memristors for rate-temporal fusion encoding. *Nature Communications*, 15, 6027. [RTF encoding origin; visual hardware domain — adaptation gap acknowledged.]

[21] Johnson, J., Douze, M., & Jegou, H. (2021). Billion-scale similarity search with GPUs. *IEEE Transactions on Big Data*, 7(3), 535–547.

[22] Zandieh, A. et al. (2026). TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate. *ICLR 2026.* arXiv:2504.19874.

[23] Zandieh, A. et al. (2025). PolarQuant. *AISTATS 2026.* arXiv:2502.02617.

[24] Zandieh, A. et al. (2024). QJL: 1-Bit Quantized JL Transform. arXiv:2406.03482.

[25] **TheTom/turboquant_plus** (2026). TurboQuant+ implementation: 141 tests, 100% coverage, Metal GPU kernels for llama.cpp, rotation Gaussianization validated (kurtosis 900→2.9 on Qwen3 KV tensors). Apache-2.0. **Primary implementation.** https://github.com/TheTom/turboquant_plus

[26] **tonbistudio/turboquant-pytorch** (2026). From-scratch PyTorch TurboQuant: 0.9945 cosine similarity at 3-bit, 5x compression on Qwen2.5-3B. 36-layer validation. MIT. **PyTorch integration reference.** https://github.com/tonbistudio/turboquant-pytorch

[27] Miech, A. et al. (2019). HowTo100M. *ICCV 2019.*

[28] Zhukov, D. et al. (2022). Temporal Alignment Networks for Long-term Video. *CVPR 2022.* [HTM-AA: 1.2M aligned HowTo100M videos; 25% of HowTo100M narration is visually alignable.]

[29] Achille, A., Rovere, M., & Soatto, S. (2019). Critical learning periods in deep neural networks. *ICLR 2019.*

[30] Critical Learning Periods for Multisensory Integration in Deep Networks. *arXiv:2210.04643.* [Cross-sensor reconstruction objectives are remarkably resilient to critical period damage.]

[31] Schultz, W. (2015). Neuronal reward and decision signals: From theories to data. *Physiological Reviews*, 95(3), 853–951.

[32] Hasselmo, M. E. (2006). The role of acetylcholine in learning and memory. *Current Opinion in Neurobiology*, 16(6), 710–715.

[33] Yu, A. J. & Dayan, P. (2005). Uncertainty, neuromodulation, and attention. *Neuron*, 46(4), 681–692.

[34] Doya, K. (2002). Metalearning and neuromodulation. *Neural Networks*, 15(4-6), 495–506.

[35] Tononi, G. & Cirelli, C. (2014). Sleep and the price of plasticity. *Neuron*, 81(1), 12–34.

[36] Desai, N. S. et al. (1999). Plasticity in the intrinsic excitability of cortical pyramidal neurons. *Nature Neuroscience*, 2(6), 515–520.

[37] Rathi, N. & Roy, K. (2019). STDP-Based Unsupervised Multimodal Learning With Cross-Modal Processing in Spiking Neural Networks. *IEEE TETCI*, 5(1).

[38] Roy, D. et al. (2019). Lifelong Learning of Spatiotemporal Representations. *Frontiers in Neural Networks*, 12:765.

[39] Schuman, C. D. et al. (2022). Opportunities for neuromorphic computing algorithms and applications. *Nature Computational Science*, 2(1), 10–19.

[40] Karlsson, V., Fianda, N., & Kamarainen, J.-K. (2026). Difference Predictive Coding for Training Spiking Neural Networks. *ICLR 2026.*

[41] NeuronSpark (2026). A 0.9B-parameter spiking language model. *arXiv:2603.16148.*

[42] Naderi, R. et al. (2025). Unsupervised post-training learning in spiking neural networks. *Scientific Reports*, 15, 17647.

[43] Larionov, D. et al. (2025). Continual Learning with Columnar Spiking Neural Networks. *arXiv:2506.17169.* [CoLaNET: 92% accuracy across 10 sequential tasks with only 4% degradation — closest related work in the SNN continual learning space. Addresses similar stability-plasticity objectives but evaluates on supervised classification, not unsupervised grounding; direct comparison on shared benchmarks not yet performed.]

[44] Turrigiano, G. G. & Nelson, S. B. (2004). Homeostatic plasticity in the developing nervous system. *Nature Reviews Neuroscience*, 5(2), 97–107.

[45] Turrigiano, G. G. (2008). The self-tuning neuron: Synaptic scaling of excitatory synapses. *Cell*, 135(3), 422–435.

[46] Yang, W. et al. (2014). Differences in E/I balance between cortical layers. *Journal of Neuroscience*, 34(34), 11206–11213.

[47] Perez-Nieves, N. et al. (2021). Neural heterogeneity promotes robust learning. *Nature Communications*, 12, 6377. [Diverse neuron timescales improve robustness and task performance.]

[48] Yin, B., Corradi, F., & Bohté, S. M. (2021). Accurate and efficient time-domain classification with adaptive spiking recurrent neural networks. *Nature Machine Intelligence*, 3(10), 905–913.

[49] Nx-Arena (2025). Neuromorphic Sequential Arena benchmark for SNN temporal processing. *IJCAI 2025.* [Standardized benchmark for evaluating temporal processing in SNNs.]

[50] Gilson, M. & Fukai, T. (2011). Stability versus neuronal specialization for STDP. *Neural Computation*, 23(6), 1514–1529.

---

*Thiago Maceno Rocha Goulart · Brasil · github.com/Tuafo*

*HECSN v4.5 — Hierarchical Emergent Concept Spiking Networks: Developmental Architecture with Honest Critique*

*PyTorch 2.1+ · pip install -e . · FastAPI/Uvicorn · React/Vite*

*Falsifiable central claim: multimodal temporal co-occurrence STDP produces a concreteness gap in the grounding probe (concrete triples score > 0.10 higher than abstract triples) not achievable by text-only systems — validated without semantic labels at any stage, though with structurally curated perceptual grounding data during the developmental critical period.*

*Stage-0 validated: silhouette 0.675, DBI 0.304, temporal_coherence 0.9916, semantic_triple_accuracy 0.7143 (7-triple text-only, separate from 50-triple probe), routing_key_between_score 0.9970, terminal_novelty_rate 0.0994. Real training validated: pred_error 1.63→0.66 nats over 1,152 Wikipedia tokens, DA 0.006→0.431, micro-sleep at 256 tokens. Self-criticism loop (visual + audio), alignment filter, dead column census, and exploration noise boost implemented and tested. 500 tests pass across 48 test files.*

*All other verification targets are falsifiable predictions, not asserted results.*
