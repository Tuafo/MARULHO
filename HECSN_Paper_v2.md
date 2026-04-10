# HECSN — Hierarchical Emergent Concept Spiking Networks
## A Developmental Architecture for Grounded Autonomous Knowledge Accumulation

**Author:** Thiago Maceno Rocha Goulart · Brasil · [github.com/Tuafo](https://github.com/Tuafo)

**Domain:** Computational Neuroscience · Unsupervised Multimodal Learning · Neuromorphic Computing

**Version:** 2.0 — Complete Architecture with Multimodal Grounding and Developmental Protocol

---

## Abstract

We present HECSN (Hierarchical Emergent Concept Spiking Networks), a biologically-grounded spiking neural network architecture for autonomous, label-free knowledge accumulation from multimodal streams. The system addresses the fundamental grounding problem in text-only emergent learning by introducing cross-modal temporal co-occurrence STDP — the same Hebbian mechanism the brain uses to associate symbols with perceptual reality — without any labeled data at any stage. The architecture implements five functional layers (Chunking, Encoding, Competitive, Binding, and Abstraction) with bidirectional feedback, independent four-channel neuromodulation, three-phase fragility-gated sleep consolidation, and a self-critical confirmation-seeking curiosity controller. Scalability is achieved via GPU-native IVF routing with TurboQuant-compressed prototype storage, targeting sub-0.1ms routing latency at 100K columns. Following the biological critical period hypothesis, we propose a five-stage developmental training protocol from curated perceptual grounding through fully autonomous operation — the computational equivalent of guided childhood learning that transitions to independent thought. The central falsifiable claim: after training on unlabeled multimodal streams through the developmental protocol, the system's prototype space should satisfy a grounding probe accuracy exceeding 0.65 against a random baseline of 0.50, and concrete perceptually-grounded concepts should score significantly higher than abstract concepts — without any labeling of concreteness at any stage.

---

## Table of Contents

1. [The Problem We Are Actually Solving](#1-the-problem-we-are-actually-solving)
2. [Core Principles](#2-core-principles)
3. [System Architecture](#3-system-architecture)
4. [Multimodal Grounding](#4-multimodal-grounding)
5. [Scalability Architecture](#5-scalability-architecture)
6. [Developmental Training Protocol](#6-developmental-training-protocol)
7. [Evaluation Protocol](#7-evaluation-protocol)
8. [What to Expect: Stage-by-Stage](#8-what-to-expect-stage-by-stage)
9. [Critical Risks and How to Address Them](#9-critical-risks-and-how-to-address-them)
10. [Implementation Roadmap](#10-implementation-roadmap)
11. [References](#11-references)

---

## 1. The Problem We Are Actually Solving

### 1.1 The Central Claim and What It Requires

The goal is a system that, starting from unlabeled raw input streams, forms stable internal representations that correlate with semantic structure in the world — without any external labels, pre-trained embeddings, or pre-defined vocabularies at any stage.

This goal has a hidden prerequisite that most emergent learning systems ignore: symbols must mean something beyond their own statistical co-occurrence. A text-only system can learn that "ocean" and "water" tend to appear near each other. But it cannot learn what either word *means* unless it has experienced water perceptually — its visual appearance, its sound, its temporal dynamics.

This is not a philosophical point. It is a practical constraint with a practical solution. The brain solves it through temporal co-occurrence: when a sound, a visual pattern, and a word appear simultaneously and repeatedly, Hebbian mechanisms wire them together. No teacher. No labels. Just physics — neurons that fire together wire together across modality boundaries.

HECSN implements this exactly.

### 1.2 What "Emergent" Means Here

Knowledge is emergent in HECSN when:

- Representations arise from network-input interaction dynamics, not from labels
- Chunking boundaries are discovered from stream statistics, not pre-defined
- Concept clusters form through competitive learning, not through class assignments
- Cross-modal associations form through temporal co-occurrence, not through paired datasets
- Curiosity targets arise from internal geometric gaps, not from external task specification
- The curriculum for autonomous learning arises from the network's own knowledge state

What is NOT emergent and should not be claimed as such: the network architecture itself (this is a structural prior, like the brain's columnar organization), the biological parameters (these are constraints derived from neuroscience), and the initial curated training phase (this is the developmental scaffold, analogous to guided childhood learning).

### 1.3 Why Text Alone Cannot Achieve Genuine Grounding

Consider what a text-only system learns about "fire." It learns that "fire" co-occurs with "heat," "smoke," "burn," "red." It learns the distributional neighborhood. But it has no representation of what fire looks like flickering, what it sounds like crackling, or why "hot" and "dangerous" are properties that generalize across fire instances. The text statistics are a shadow of the concept, not the concept.

This is not a performance gap that more text training closes. It is a structural gap: text co-occurrence statistics encode word-to-word relationships, not word-to-world relationships. A system that ingests only text can never, in principle, distinguish between a coherent semantic structure and a perfectly self-consistent but meaningless word game.

The solution is not to add labels. It is to add perception — specifically, multimodal temporal co-occurrence that mirrors the developmental mechanism the brain uses to ground language in reality.

### 1.4 The Child Analogy Is Mechanistically Accurate

A human child does not acquire language from text statistics. They acquire it through thousands of grounding events: a parent holds up an apple, says "apple," and the child experiences the visual, tactile, and auditory properties simultaneously with the word. Over time, the word "apple" becomes bound to a stable multimodal representation — one that generalizes correctly to new apples because it was grounded in perceptual invariants, not word co-occurrence.

The critical period hypothesis, confirmed in both biological and artificial networks, establishes that this early grounded experience is not optional — it permanently shapes the representational capacity of the network. A child deprived of rich perceptual experience during the critical period cannot recover full linguistic competence later. An artificial network must respect the same constraint.

HECSN implements a five-stage developmental protocol that mirrors this: curated perceptual grounding (Stage 1), expansion through self-filtered multimodal data (Stage 2), active confirmation-seeking (Stage 3), semi-autonomous operation (Stage 4), and fully autonomous open-ended accumulation (Stage 5). Each stage transitions to the next based on internal metrics — not on a fixed token count.

---

## 2. Core Principles

### 2.1 Local Learning Only

No backpropagation through time. No global loss functions. Every synaptic update depends only on pre-synaptic spikes, post-synaptic spikes, and local neuromodulatory signals — the three-factor learning rule. This enables continuous online learning without pausing for gradient computation and maintains biological plausibility as a functional constraint, not as an aesthetic choice.

The stability-plasticity dilemma — learning new things without destroying old things — is solved by the same mechanisms the brain uses: synaptic scaling, inhibitory STDP, fragility-gated consolidation, and sleep-phase replay. These are not approximations of biological mechanisms. They are the mechanisms, implemented at the level of tractable phenomenological models.

### 2.2 Scalability by Design

The architecture must accommodate 1,000 neurons or 1,000,000 neurons without rewriting core logic. This mandates:

- GPU-native routing (no CPU round-trips per token)
- IVF partitioning for prototype search at 50K+ columns
- TurboQuant compression for prototype storage (6x memory reduction, 8x routing speedup)
- CSR sparse tensors for dynamic connectivity
- Distributed architecture for 1M+ neurons

At 10K columns with flat GPU routing, expected latency is approximately 0.05ms per token. At 100K columns with IVF partitioning and TurboQuant, approximately 0.08ms. These are hard engineering targets, not aspirational claims. Section 5 provides benchmarking methodology.

### 2.3 Developmental Structure Is Not Optional

A network that processes random multimodal streams from initialization will not converge on grounded representations. This is confirmed by the critical learning period literature: early training dynamics are decisive and cannot be recovered later. The developmental protocol in Section 6 is not a training convenience — it is an architectural requirement with biological backing.

### 2.4 Every Metric Must Be Label-Free and Falsifiable

The system's primary evaluation is a grounding probe: given 50 structural triples (anchor, positive, negative) derived from world knowledge without labels, does the prototype space satisfy the ordering? A random system scores 0.50. A genuinely grounded system should score above 0.65 after full developmental training. This is the paper's central falsifiable prediction.

All other metrics — temporal coherence, compositionality score, novelty coverage — are also label-free and falsifiable. The system's claims are grounded in measurement, not in architectural description.

---

## 3. System Architecture

### 3.1 Overview

```
INPUT: Raw multimodal streams (bytes, video frames, audio samples)
         │
         ▼
┌──────────────────────────────────────────────────────┐
│  LAYER 0: MULTIMODAL ENCODERS                        │
│  Text: Byte stream (raw)                             │
│  Visual: Event-camera-style temporal contrast        │
│  Audio: Cochleagram (mel-filterbank spike encoding)  │
└──────────────────────┬───────────────────────────────┘
                        │
         ┌──────────────▼──────────────┐
         │  LAYER 1: CHUNKING          │
         │  Predictability-based       │
         │  boundary detection         │
         │  Learned detector bank      │
         │  ← Boundary bias from       │
         │    Abstraction Layer        │
         └──────────────┬──────────────┘
                        │ Variable-length chunk patterns
         ┌──────────────▼──────────────┐
         │  LAYER 2: ENCODING (RTF)    │
         │  Rate-temporal fusion       │
         │  Positional phase offset    │
         └──────────────┬──────────────┘
                        │ [dim] spike vectors
         ┌──────────────▼──────────────────────────────┐
         │  LAYER 3: COMPETITIVE LAYER                  │
         │  GPU-native IVF routing (no CPU HNSW)        │
         │  TurboQuant prototype storage                │
         │  Winner history refractory                   │
         │  Triplet STDP + four independent             │
         │  neuromodulators (DA/ACh/NE/5-HT)           │
         │  Consolidation-gated wake plasticity         │
         │  ← Routing bias from Abstraction Layer       │
         │  ← Grounding boost from Cross-Modal Layer    │
         └──────────────┬──────────────────────────────┘
                        │ Winner assembly + top-k
         ┌──────────────▼──────────────┐
         │  LAYER 4: BINDING           │
         │  n_bindings ≠ n_columns     │
         │  Sparse random connectivity │
         │  Tsodyks-Markram STP        │
         │  PV+ fast inhibition        │
         │  Structural growth          │
         └──────────────┬──────────────┘
                        │ Composite assemblies
         ┌──────────────▼──────────────┐
         │  LAYER 5: ABSTRACTION       │
         │  Online SFA (full layer)    │
         │  Anti-Hebbian temporal      │
         │  Concept stability/certainty│
         │  → Routing bias             │
         │  → Boundary bias            │
         │  → Curiosity gaps           │
         └──────────────┬──────────────┘
                        │ Gap signal
         ┌──────────────▼──────────────┐
         │  CROSS-MODAL GROUNDING      │
         │  Temporal co-occurrence STDP│
         │  Alignment filter           │
         │  Confirmation seeking       │
         │  Self-criticism loop        │
         └──────────────┬──────────────┘
                        │ Retrieved bytes → back to INPUT
         
MEMORY & SLEEP (parallel track):
┌──────────────────────────────────────────┐
│  DUAL MEMORY STORE                       │
│  Fast EMA (drift/novelty baseline)       │
│  Slow reservoir + fragility scores       │
│  Self-calibrated functional_minute       │
└──────────────────────────────────────────┘
         │ Triggered by drift / schedule
┌──────────────────────────────────────────┐
│  THREE-PHASE SLEEP                       │
│  A: Micro (200 tok) — maintenance only  │
│  B: Deep (5K tok) — fragility-gated     │
│  C: Emergency — repair only             │
└──────────────────────────────────────────┘
```

### 3.2 Layer 1: Chunking Layer

**What it does:** Replaces heuristic tokenization with a learned, statistics-driven boundary detection mechanism. The unit of analysis — what counts as a "chunk" — emerges from the stream itself.

**Mechanism:** A bank of N detector neurons, each tuned to a different byte-sequence prototype. A boundary is declared when adding the next byte to the current buffer *reduces* the maximum detector agreement score. Intuitively: the current sequence is a coherent chunk as long as extending it keeps fitting a known pattern. When extending it makes it fit worse, that's the boundary.

**How it works step by step:**

1. A new byte arrives from the stream
2. The current buffer is encoded as a fixed-dim spike pattern via positional-phase-weighted bit representation
3. The extended buffer (current + new byte) is also encoded
4. Both are compared against all detector prototypes via cosine similarity
5. If `max_sim(extended) < max_sim(current) - threshold`: boundary declared
6. The current buffer is emitted as a chunk spike pattern
7. The winning detector prototype updates toward the emitted chunk (competitive learning)
8. The new byte begins the next buffer

**What emerges:** Statistical regularities in the byte stream — morpheme-like units, common word stems, frequent phrases, punctuation patterns — crystallize as stable detector prototypes. The chunking layer discovers that `"tion"`, `"ing"`, `"the "`, `" of "` are reliable units without being told what a word or morpheme is.

**Boundary bias from Abstraction Layer:** When the Abstraction Layer detects high concept instability (current input is ambiguous or novel), it signals the Chunking Layer to prefer shorter chunks (more conservative boundaries, more precise units). This top-down feedback tightens chunking during semantically complex input.

**Key parameter:** `boundary_threshold` — the required drop in detector agreement to declare a boundary. Higher = fewer, longer chunks. Lower = more, shorter chunks. Should be calibrated empirically per stream domain.

**Expected behavior during training:**
- Tokens 0–5K: Random boundaries, chunks are mostly 1-2 bytes
- Tokens 5K–50K: Common byte sequences stabilize as detectors; boundaries become more regular
- Tokens 50K+: Morpheme-like units emerge; chunk size distribution becomes approximately log-normal

### 3.3 Layer 2: Rate-Temporal Fusion Encoding

**What it does:** Converts a variable-length chunk spike pattern into a fixed-dimensional encoding that preserves both which bytes appeared (rate) and their ordering (temporal phase). This is the input representation to the Competitive Layer.

**Mechanism:** Each position in the chunk contributes a weighted bit-vector to the output. Earlier positions receive higher phase weight (exponential decay), encoding sequence order without explicit positional embeddings. The final vector is L2-normalized.

**Key property:** Two chunks with similar characters in similar order produce similar encodings. Two chunks with the same characters in different orders produce different encodings. This is the minimum required temporal structure for the Competitive Layer to learn ordered patterns.

**Ablation requirement:** Before committing to RTF encoding, the following baselines must be measured on the same stream:
- Rate coding only (no positional weight)
- Phase coding only (timing-based, no rate)
- RTF (current)

The winning encoding is whichever produces the best temporal coherence score (Section 7) after 50K tokens. If rate coding matches RTF, RTF adds complexity without benefit and should be dropped.

### 3.4 Layer 3: Competitive Layer

**What it does:** The core concept-formation layer. Incoming chunk encodings compete to activate prototype columns. Winners update via Kohonen/SOM-style competitive learning. Over time, each column specializes to a cluster of similar input patterns — a proto-concept.

**Routing (GPU-native, no CPU round-trip):**

For N ≤ 50,000 columns:
- Full cosine similarity matrix between input and all prototypes: O(N × dim) on GPU
- Expected latency at 10K columns, dim=256: approximately 0.05ms on A100

For N > 50,000 columns (IVF partitioning):
- Columns pre-assigned to Voronoi cells (updated during sleep)
- Query routed to nearest 8 cells only: reduces effective search space by ~97%
- Expected latency at 100K columns: approximately 0.08ms

All prototypes stored in TurboQuant-compressed format (Section 5). Inner products computed in rotated quantized space without decompression — 8x faster than float32 on H100.

**Winner history refractory:** Each column accumulates a winner count that decays at 0.995 per token. Active columns are penalized in the routing score proportionally. This prevents any single column from monopolizing routing and forces coverage across the prototype space.

**Three-factor plasticity (triplet STDP):**

Standard pair STDP: `Δw ∝ pre_spike × post_spike_timing_window`

Triplet extension adds: `Δw ∝ pre_spike × post_spike × recent_post_activity`

The triplet term accounts for burst timing — if the post-synaptic neuron recently fired (within a short window), LTP is enhanced. This is more biologically accurate and produces more stable weight distributions.

**Four independent neuromodulatory channels:**

| Channel | Input | Target in plasticity rule | Time constant |
|---|---|---|---|
| Dopamine (DA) | Reward prediction error: `predicted_error - actual_error` | LTP magnitude scaling `[0.5, 1.5]` | 20 tokens |
| Acetylcholine (ACh) | Novelty: distance to nearest prototype | Effective learning rate `[0.1, 1.0]` | 50 tokens |
| Norepinephrine (NE) | Sustained uncertainty: rolling variance of prediction errors | Exploration noise added to membrane voltages | 200 tokens |
| Serotonin (5-HT) | Recent success rate: fraction of low-error predictions | LTD bias `[0.5, 1.5]` | 500 tokens |

The weight update rule:
```
Δw = base_lr × ACh_gain × (DA_scale × LTP_term − 5HT_bias × LTD_term)
```

NE adds noise to membrane voltages (not weights) to drive exploration of under-active columns. There is no network reset — high sustained NE triggers a curiosity query instead.

**Consolidation-gated wake plasticity:** Columns whose stored memories are well-consolidated (consolidation_level > 0.7) resist overwrite during wake. Effective learning rate for these columns is multiplied by `(1 - consolidation_level × 0.8)`. This allows new learning without destroying consolidated knowledge.

### 3.5 Layer 4: Binding Layer

**What it does:** Detects coincident activation across multiple Competitive Layer columns and forms composite assemblies — proto-propositions that capture relationships between concepts.

**Critical design note:** The number of binding neurons is independent of the number of columns. Each binding neuron connects to a random subset of `fan_in` columns (default: 4). A boundary firing requires at least `fan_in / 2` source columns to be simultaneously active. This is coincidence detection, not copying.

**Mechanism step by step:**

1. Competitive Layer winner activates at time t
2. Binding neurons check their source column sets for recent co-activation within `tau_binding` (50ms functional time)
3. Binding neuron i fires if its weighted input (after Tsodyks-Markram STP) exceeds threshold
4. PV+ inhibition: if any binding neuron fires strongly, global inhibition increases, preventing cascade
5. A successful binding creates a new composite assembly linking the source column indices

**Short-term plasticity (Tsodyks-Markram model):**

Facilitation: `u(t) += -u/tau_f + U_inc × input × (1-u)` — builds up with repeated activation

Depression: `release = u × x × input; x += (1-x)/tau_d - release` — depletes available resources

This implements a temporal filter: binding neurons respond strongly to the second or third co-activation but not to isolated coincidences. This reduces noise-driven false binding.

**Structural growth:** During deep sleep, column pairs with spike correlation above 0.7 that have no binding neuron watching both are assigned a new binding neuron. The binding layer grows with the network's concept space.

### 3.6 Layer 5: Abstraction Layer

**What it does:** Extracts slowly-varying features from the sequence of winning assemblies. Concepts that are semantically stable vary slowly over time; noise and transient input vary fast. The Slow Feature Analysis objective — minimize temporal variance of output — implements this filtering.

**Why this is a full layer, not a proxy:** Unlike a passive observer over routing signatures, this layer receives assemblies as input, extracts abstract features, and sends feedback to both the Competitive Layer (routing bias) and the Chunking Layer (boundary bias). It is a computational component of the architecture, not monitoring infrastructure.

**Online SFA update (anti-Hebbian in time):**

For each concept unit i:
1. Compute output: `concept[i] = W[i] · assembly / slow_std[i]`
2. Update fast mean: `fast_mean[i] = (1-α_fast) × fast_mean[i] + α_fast × concept[i]`
3. Update slow mean: `slow_mean[i] = (1-α_slow) × slow_mean[i] + α_slow × concept[i]`
4. Temporal variance: `slow_var[i] = (1-α_slow) × slow_var[i] + α_slow × (fast_mean[i] - slow_mean[i])²`
5. Instability: `instability[i] = slow_var[i] / mean(slow_var)`
6. If `instability[i] > 1.5`: reduce `W[i]` — this unit is tracking transient variation, not stable features

**Feedback signals:**

*To Competitive Layer (routing bias):* Stable concepts (high stability score, strong slow mean) contribute a positive bias to their aligned columns. Routing is pulled toward concept-consistent evidence.

*To Chunking Layer (boundary bias):* Mean instability across all concepts. High instability → shorter chunks preferred (tighter boundaries needed for ambiguous input).

*To Curiosity Controller (gap detection):* Concepts with high temporal variance and low certainty are knowledge gaps. These are the curiosity targets — not keywords, not external topics, but geometric gaps in the network's own concept space.

### 3.7 Memory Store

**Dual buffer architecture:**

*Fast EMA buffer:* Exponential moving average of recent assemblies. Tracks current representation center. Used for drift detection and novelty estimation. Alpha = 0.01 (decays toward current over ~100 tokens).

*Slow reservoir:* Reservoir-sampled assemblies from all training history (Vitter 1985, Algorithm R). Every assembly has equal probability of being in the buffer regardless of when it was seen. Used for sleep replay. Each stored memory carries:
- Assembly tensor
- Importance score (prediction error × novelty at storage time)
- Capture tag (STC, decays exponentially)
- Local PRP trace
- Consolidation level (0.0 to 1.0)
- Access count and tokens since last replay
- Fragility score: `1 / (consolidation_level + 0.01) × 1 / (access_count + 1)`

**Self-calibrating functional time:** Before training, the network calibrates its `functional_minute` parameter by measuring how many tokens it takes for a novel prototype to stabilize (convergence of winner-local drift). This grounds STC timescales in the network's actual convergence dynamics rather than biological clock-time analogies.

### 3.8 Three-Phase Sleep

**Phase A — Micro-sleep (every 200 tokens, maintenance only):**

Triggered: scheduled, every 200 tokens
Targets: memories sorted by fragility (highest risk first), skip if consolidation > 0.8
Operation: replay eligibility trace only — no weight commit. Refresh capture tag (+0.05).
Purpose: prevent capture tags from decaying before deep sleep can consolidate them

**Phase B — Deep sleep (every 5,000 tokens, fragility-gated consolidation):**

Triggered: scheduled, or when drift floor is rising
Eligible memories: capture_tag > 0.3 AND prp_local > 0.4 AND consolidation_level < 0.8
Operation: commit captured synapses at anchor_lr = 0.001 (vs wake lr = 0.01)
Result: consolidation_level advances by `0.1 × capture_tag × prp_local`
Also: structural plasticity (prune/grow), HNSW rebuild, IVF cell reassignment

The anchor_lr is the critical parameter. At 10× smaller than wake lr, even a full consolidation pass moves prototypes by at most 10% of a standard wake update. This prevents collapse while enabling real long-term potentiation.

**Phase C — Emergency repair (drift floor rising, repair only):**

Triggered: drift floor > threshold AND rising for > 1000 tokens
Operation: restore top-N prototypes toward their stored assemblies. No STDP. No weight changes. No tag changes.
Purpose: arrest drift without triggering new consolidation

**What "drift floor" means:** Minimum drift over the last 1,000 tokens. If the floor itself rises over consecutive windows, it means the baseline representation is shifting upward — a sign that current learning is displacing consolidated memories. Emergency repair anchors the prototypes without making any new commitments.

---

## 4. Multimodal Grounding

### 4.1 The Grounding Problem Precisely Stated

A text assembly for "fire" is a point in prototype space. It has neighbors: "smoke," "heat," "burn," "red." This neighborhood is a statistical structure derived from text co-occurrence. But the assembly has no binding to:
- The specific visual appearance of fire (flickering, orange, luminous)
- The acoustic signature of fire (crackling, hissing)
- The perceptual properties that make "hot" and "dangerous" non-arbitrary attributes

Without these perceptual bindings, "fire" is a word that points to other words. With them, "fire" is a concept that points to a stable class of phenomena in the world. The difference is not trivial — it is the difference between a coherent semantic structure and a self-consistent but meaningless symbol game.

### 4.2 The Grounding Mechanism: Cross-Modal Temporal STDP

The core mechanism is Hebbian: when text spikes and visual spikes occur within a temporal binding window (tau_bind), the cross-modal synaptic weights between them are potentiated. No labels. No contrastive loss. No negative pairs. Just: *things that fire together within 100ms wire together.*

**Four cross-modal weight matrices, updated by STDP:**

| Matrix | Direction | What it learns |
|---|---|---|
| W_tv | Text → Visual | Given this text spike, predict this visual pattern |
| W_vt | Visual → Text | Given this visual spike, activate related text assemblies |
| W_ta | Text → Audio | Given this text spike, predict this audio pattern |
| W_at | Audio → Text | Given this audio spike, activate related text assemblies |

**STDP update rule for cross-modal connections:**

When text fires at time t:
```
ΔW_tv[text_dim, :] += A_plus × text_spike × visual_trace
```
Where `visual_trace` is the decaying eligibility trace of recent visual spikes: neurons that fired *before* the text spike contribute positively (they predicted it).

When visual fires at time t:
```
ΔW_vt[visual_dim, :] += -A_minus × visual_spike × text_trace
```
Neurons whose text trace is elevated when visual fires are slightly weakened (temporal ordering matters for causality).

A_minus is set slightly larger than A_plus (e.g., 0.012 vs 0.010) to prevent runaway potentiation.

**Grounding confidence (slow-learned):**

For each text dimension, a slow EMA tracks how consistently the cross-modal prediction for that dimension matches actual perceptual input:
```
grounding_confidence[i] += 0.001 × (1.0 - prediction_error) × text_activation[i]
```

High grounding confidence means: when this text dimension activates, it reliably predicts a specific visual/audio pattern, and that pattern tends to appear.

**How grounding modulates the rest of the architecture:**

*Competitive Layer routing:* Text dimensions with high grounding confidence contribute a positive bias to routing scores for concept-aligned columns. Well-grounded concepts route more reliably.

*Plasticity gating:* High grounding confidence for the current input scales the plasticity multiplier upward. The network learns faster when grounded perception confirms a textual association.

*Curiosity controller:* Text dimensions with high activation but low grounding confidence are the highest-priority curiosity targets. The network encounters these words frequently but can't predict their perceptual correlates.

### 4.3 Modality Encoders

**Visual (Event-camera-style temporal contrast):**

Event cameras are biologically inspired sensors that mimic retinal ganglion cells. They generate spikes only when pixel luminance changes, producing inherently sparse, temporally precise output. For HECSN, we simulate event camera output from standard video:

1. Convert frame to grayscale
2. Compute per-pixel difference from previous frame (temporal contrast)
3. Pool spatially (8×8 blocks)
4. Apply threshold → binary spike pattern

Output: sparse [H/8 × W/8] spike pattern per frame. Sparsity: approximately 5-15% of pixels active per frame.

This encoding is hardware-compatible (real event cameras exist as commercial products), biologically grounded, and requires no labels.

**Audio (Cochleagram):**

The cochleagram mimics the basilar membrane of the inner ear:

1. Apply mel-scale triangular filterbank to FFT of audio frame (fixed biological prior — no learning here)
2. Compute power in each of 64 frequency bands
3. Log-compress (mimics auditory loudness perception — Weber-Fechner law)
4. Apply threshold → binary spike pattern per band

Output: [64] spike pattern per audio frame. Each band corresponds to a frequency range (logarithmically spaced, matching biological cochlear tuning).

**Why cochleagram and not mel-spectrogram features:** The cochleagram is a first-principles biological encoding requiring no training and no choices beyond filterbank resolution. It is maximally compatible with STDP temporal learning because its spike timing preserves onset information that rate-coded features would lose.

### 4.4 The Alignment Problem and the Self-Filtering Solution

**The problem:** In natural video (documentaries, YouTube), the narrator does not describe what is currently visible. Temporal research on HowTo100M shows only approximately 25% of narration clips are visually alignable with their concurrent frame content. The rest are descriptions of past events, future intentions, background context, or meta-commentary.

Naively pairing text with concurrent visual input will ground "ocean" to boat interiors, "fire" to interview backgrounds, and "submarine" to computer interface graphics.

**The solution: alignment filter using existing grounding:**

Once Stage 1 grounding is established (Section 6), the network can use its cross-modal predictions to evaluate whether any given text-visual pair is worth learning from:

```
alignment_score = cosine_similarity(
    predicted_visual(text_assembly),  ← from W_tv
    actual_visual_spikes               ← from encoder
)
```

If alignment_score > 0.4: accept the pairing, update cross-modal weights
If alignment_score < 0.4: reject the pairing, update text-only pathway

This is self-filtering: the network uses what it already knows to judge what to believe. The filter improves as grounding improves — an autocatalytic process. Early in Stage 2 the filter is weak (few reference concepts), later it becomes a strong gate (many reference concepts with calibrated predictions).

**The key property:** spurious associations that would be learned in Stage 1 (where alignment is guaranteed by curation) cannot occur, because Stage 1 vocabulary provides the reference anchors for the filter.

### 4.5 Self-Criticism: The Network Evaluates Its Own Grounding

The confirmation-seeking controller performs two functions:

**Forward seeking (filling gaps):** For ungrounded concepts (high text activation, low grounding confidence), scan upcoming video frames for high alignment scores. When found, update cross-modal weights. This is the network actively looking for perceptual evidence of things it encounters linguistically.

**Backward criticism (correcting errors):** For high-confidence concepts (grounding_confidence > 0.7), periodically test whether the prediction actually matches recent visual experience. If a concept is confident but its visual prediction never matches recent frames, the confidence is likely WRONG — not just absent. Action: reduce confidence by 10% per evaluation cycle and add to confirmation queue.

This implements the developmental analog of a child's "checking" behavior — not just learning new things but verifying that existing beliefs are correct.

---

## 5. Scalability Architecture

### 5.1 The CPU-GPU Bottleneck Is the Primary Scaling Wall

Every routing decision requiring CPU involves: GPU-to-CPU memory transfer, CPU-side FAISS HNSW search, CPU-to-GPU result transfer. At 1,000 tokens/second, this synchronization overhead consumes approximately 2.1ms per token (measured: HNSW search + PCIe round-trip on A100 system). At 10K tokens/second, this becomes the dominant bottleneck.

The solution is complete: move all routing to GPU. Never transfer prototypes to CPU during the hot path.

### 5.2 GPU-Native Routing

**Flat cosine similarity (N ≤ 50,000 columns):**

All N prototype vectors stored as a [N, dim] GPU tensor. Routing = one matrix-vector multiply + argmax. O(N × dim) on GPU.

At N=10,000, dim=256: matrix-vector multiply on A100 ≈ 0.05ms.

Refractory penalty applied elementwise in GPU: `scores[i] -= refractory_strength × winner_history[i]`

**IVF partitioning (N > 50,000 columns):**

Columns pre-assigned to Voronoi cells (K = sqrt(N), updated during deep sleep). Routing:
1. Compute query similarity to K cell centroids: O(K × dim)
2. Select top-8 cells: O(K)
3. Compute similarity to columns in selected cells only: O(8 × N/K × dim) = O(8 × sqrt(N) × dim)

At N=100,000, dim=256, K=316: approximately 0.08ms total.

Cell assignments updated during deep sleep (not per-token) — no routing overhead from IVF maintenance.

**Benchmarking targets:**

| Columns | Method | Prototype RAM | Latency/token |
|---|---|---|---|
| 10K | GPU flat | 10 MB (FP32) → 1.7 MB (TQ@3bit) | ~0.05ms |
| 100K | GPU IVF | 100 MB → 17 MB | ~0.08ms |
| 1M | Distributed IVF | 1 GB/shard → 170 MB/shard | ~0.3ms |

These targets must be measured before the scalability claim is published. If actual measurements differ by more than 2x, revise the architecture accordingly.

### 5.3 TurboQuant Prototype Compression

TurboQuant (Zandieh et al., ICLR 2026) achieves near-optimal vector quantization via two stages:

**Stage 1 — PolarQuant:** Apply a random orthogonal rotation R to each prototype vector. After rotation, energy is approximately uniformly distributed across dimensions (rotation theorem for Gaussian-like vectors). Apply a standard scalar quantizer to each dimension independently at b bits. At b=3: 8 quantization levels per dimension.

**Stage 2 — QJL residual correction:** The small quantization error remaining after Stage 1 introduces bias in inner products. Apply a 1-bit Quantized Johnson-Lindenstrauss transform to the residual. The sign bit alone eliminates this bias with high probability, preserving inner product accuracy at near-zero additional storage.

**Why this works for HECSN routing:** HECSN routing is cosine similarity = normalized inner product. TurboQuant is specifically optimized for inner-product preservation. The property that makes it efficient for LLM KV-cache attention (inner products between queries and compressed keys) is identical to what HECSN needs (inner products between input patterns and compressed prototypes).

**Implementation:** The random rotation matrix R is computed once at initialization via QR decomposition of a random Gaussian matrix. R is fixed — no training. The quantization constants (scale, offset) are stored per prototype, not per dimension, eliminating quantization metadata overhead.

**Expected gains at HECSN scale:**

At 3 bits/dimension vs FP32 (32 bits/dimension): 10.7x compression ratio in raw bits. Actual memory reduction approximately 6x after metadata (scale factors). Inner product computation 8x faster on H100 (per Google benchmarks at 4-bit precision).

### 5.4 Sparsity Strategy

**2:4 Structured Sparsity (fixed-dimension matrices):**

Applied to feedforward projection matrices between layers where dimensions are fixed at compile time. PyTorch cuSPARSELt provides hardware acceleration on Ampere+ GPUs (A100, RTX 30xx+). Real speedup approximately 1.6x (not 2x — metadata overhead). Only beneficial for large matrices (>10K elements per dimension).

**CSR Sparse Tensors (dynamic connectivity):**

Applied to synaptic weight matrices in Competitive and Context layers where structural plasticity adds/removes connections. Natural fit for 10-20% biological connectivity density. No hardware acceleration — performance gain from reduced arithmetic, not specialized kernels.

**Profiling requirement:** At small scale (≤10K columns), sparse operations may be slower than dense due to indexing overhead. Profile both before committing to sparse format. At 100K+ columns, sparse becomes mandatory for memory.

---

## 6. Developmental Training Protocol

### 6.1 The Biological Basis

Critical period research in both biological and artificial networks establishes that early training dynamics are decisive and cannot be fully recovered later. Exposing a network to rich correlated multimodal input during high-plasticity periods permanently shapes its representational capacity. Exposing it to noisy or misaligned input during this period permanently impairs it.

The five-stage protocol below mirrors developmental neuroscience: curated sensory experience → scaffolded expansion → confirmation-seeking → semi-autonomous → fully autonomous. Each stage transitions based on internal metrics, not fixed token counts.

### 6.2 Stage 1: Critical Period (Curated Grounding)

**Biological analog:** A parent holds up an apple and says "apple." The word, visual, and tactile experience occur simultaneously. Modalities are aligned by design.

**Goal:** Establish cross-modal grounding anchors for the core concrete vocabulary. These anchors will be used by the alignment filter in Stage 2.

**Data sources (alignment guaranteed by construction):**

| Dataset | Modalities | Alignment type | Approximate size |
|---|---|---|---|
| MNIST-DVS + TI-46 speech | Visual (event) + Audio (speech) | Perfect: digit shown = digit spoken | 70K pairs |
| ObjectNet + spoken labels | Visual + Audio | Perfect: image shown = word spoken (custom recording) | 50K pairs |
| HTM-AA aligned subset | Visual + Text + Audio | High (alignment score > 0.7) | 247K clips |
| BBC micro-video descriptions | Visual + Audio | High: descriptions recorded while viewing | Custom |
| Cooking close-range video | Visual + Text + Audio | Medium-high: actions described while performed | Selected from HowTo100M |

**What trains:**

- Chunking Layer: learns byte-pattern detectors on text stream (text-only in parallel)
- Competitive Layer: forms initial prototype clusters from text chunks
- Cross-Modal Grounding: STDP updates all four cross-modal weight matrices with no alignment filter (input is pre-filtered by curation)
- Abstraction Layer: begins accumulating slow-feature statistics
- Memory Store: fills with grounded assemblies

**Stage 1 completion criterion:**

Grounding confidence for the top-100 most frequent concepts in the training vocabulary exceeds 0.40. Measured as: `mean(grounding_confidence[top_100_text_dims]) > 0.40`.

Do not advance to Stage 2 before this is satisfied. Advancing too early means the alignment filter will be too weak to reject spurious pairings.

**What to expect:**

- Tokens 0–5K: High plasticity, unstable representations, high novelty rate (>0.8)
- Tokens 5K–50K: Chunking stabilizes on common patterns; grounding confidence rises for basic concrete nouns (object words, action words)
- Tokens 50K–200K: Cross-modal predictions become non-trivial; grounding confidence reaches 0.40 threshold for core vocabulary
- Grounding probe at end of Stage 1: approximately 0.55–0.60 (better than random, not yet mature)

### 6.3 Stage 2: Structured Expansion (Scaffolded Self-Filtering)

**Biological analog:** Parents read picture books. Content is curated but not perfectly aligned. The child uses existing grounded concepts to understand new vocabulary by association.

**Goal:** Extend grounding to a broader vocabulary using Stage 1 anchors as reference. The alignment filter now rejects spurious pairings autonomously.

**Data sources:**

- Full HTM-AA aligned subset (all 247K videos, not just high-score filtered)
- HowTo100M full dataset with alignment filter gating
- Nature documentaries (downloaded via yt-dlp with CC license)
- Podcast transcripts + audio (text-audio alignment is high: speech IS the text)

**What trains:**

- All Stage 1 components continue
- Alignment filter becomes active: cross-modal updates only when `alignment_score > 0.4`
- Audio-to-text grounding expands faster than visual-to-text (speech is inherently aligned)
- Binding Layer begins forming composite assemblies as concept space grows

**Stage 2 completion criterion:**

Alignment filter precision (fraction of accepted pairings that are genuinely grounded) exceeds 0.65. Measured via held-out labeled alignment data from HTM-AA benchmark.

Additionally: grounding probe accuracy exceeds 0.60.

**What to expect:**

- Audio-text grounding develops faster than visual-text (expect it 2-3x earlier)
- Visual grounding improves for action verbs and concrete nouns (things that appear while described)
- Abstract concepts remain ungrounded (this is correct — they will be grounded through conceptual relationships in Stage 3)
- Alignment filter errors: early Stage 2 will still accept some spurious pairings. The self-criticism loop catches these over time.
- Novelty rate should stabilize in the 0.20–0.40 range: the network is still expanding but consolidating

### 6.4 Stage 3: Active Confirmation-Seeking

**Biological analog:** A child around age 2-3 who asks "what's that?" for everything. They identify concepts they partially know and actively seek grounding.

**Goal:** Fill the remaining grounding gaps through active confirmation seeking. The network directs its own attention to ungrounded concepts.

**Mechanism:**

1. Identify ungrounded concepts: high text activation (encountered frequently) + low grounding confidence
2. For each: generate visual prediction from current W_tv weights
3. Scan upcoming video stream for frames with high alignment score
4. When found: update cross-modal weights with confirmed pairing
5. Self-criticism: periodically test high-confidence concepts against recent visual experience; reduce confidence if prediction consistently fails

**What trains:**

- All previous components continue
- Confirmation-seeking controller becomes active
- Self-criticism loop runs every 5,000 tokens
- Curiosity controller begins targeting geometric gaps in Abstraction Layer

**Stage 3 completion criterion:**

Fraction of top-500 most frequent text concepts with grounding confidence < 0.3 falls below 20%.

Grounding probe accuracy exceeds 0.65 — this is the paper's primary falsifiable threshold.

**What to expect:**

- Network begins seeking visual evidence for abstract concepts by associating them with concrete anchors (e.g., "democracy" grounded partly through visual associations with crowds, voting, symbols — imperfect but non-zero)
- Self-criticism catches and corrects the worst Stage 2 errors
- Grounding probe may temporarily decrease slightly (self-criticism revises overconfident associations before improving)
- Novelty rate stabilizes at 0.10–0.20: mature learning rate

### 6.5 Stage 4: Semi-Autonomous Operation

**Biological analog:** An adolescent who tests hypotheses independently, seeks information when curious, and corrects themselves when internal consistency fails.

**Goal:** Operate on any multimodal stream without curated guidance. Internal state drives all learning decisions.

**What changes from Stage 3:**

- No curated data sources — any accessible multimodal stream is valid input
- Curiosity controller drives data selection: the network requests streams that address its current geometric gaps
- Alignment filter runs without human-selected data
- Self-criticism runs continuously (not just periodically)

**What trains:**

- All components, fully autonomous
- Terminus autonomous acquisition active: Wikipedia, arXiv, OpenAlex for text; YouTube (yt-dlp) for multimodal

**Transition to Stage 5 criterion:**

Autonomous operation sustained for > 500K tokens with:
- No external curation of input
- Temporal coherence stable (> 0.80)
- Novelty rate in healthy range (0.05–0.15)
- Grounding probe not declining

### 6.6 Stage 5: Fully Autonomous

**Biological analog:** An adult who reads, watches, listens, and reasons independently. External guidance is optional, not required.

**Goal:** Open-ended autonomous accumulation. No protocol. No curated sources. No confirmation from external evaluation.

The network determines its own curriculum from internal state, evaluates its own knowledge through internal consistency, seeks external information when gaps are detected, and consolidates through sleep cycles — indefinitely.

This is the goal state. All previous stages are the developmental path to reach it.

---

## 7. Evaluation Protocol

All metrics are label-free. No external labels are used at any evaluation stage.

### 7.1 Level 1: Structural Coherence (Sanity Check)

Silhouette score and Davies-Bouldin index on the prototype space. These confirm that clustering exists (the competitive layer is doing something). They do not confirm semantic structure. Use as smoke tests only.

**Target:** Silhouette > 0.65, DBI < 0.35.
**Warning:** These metrics alone cannot support any semantic claim.

### 7.2 Level 2: Temporal Coherence (Primary Stability Metric)

For each chunk pattern seen multiple times in the last W=1,000 tokens, measure the fraction of occurrences that routed to the same column (the "modal winner").

`temporal_coherence = mean over patterns of [max_winner_count / total_occurrences]`

**Interpretation:**
- Random (untrained): approximately 1/N_columns ≈ 0.001 at 1,000 columns
- Bootstrap phase: rises from near-zero toward 0.5
- Mature: should exceed 0.80

**Track over training time.** Plot: x = tokens seen, y = temporal coherence. Should rise and plateau. A plateau below 0.60 indicates instability. A sudden drop indicates catastrophic forgetting or prototype collapse.

### 7.3 Level 3: Compositionality Score (Primary Structure Metric)

For a set of text pairs (A, B), measure whether the prototype for the combined sequence AB lies geometrically between the prototypes for A and B individually.

`compositionality = mean of cosine_similarity(proto_AB, normalize(proto_A + proto_B))`

**Interpretation:**
- No structure: approximately 0.50
- Emerging structure: 0.55–0.65
- Real compositionality: > 0.65

This metric tests whether the network forms compositional representations — a prerequisite for propositional knowledge.

### 7.4 Level 4: Grounding Probe (Primary Emergence Metric)

**This is the paper's central falsifiable claim.**

Construct 50 structural triples (anchor, positive, negative) based on world knowledge but without using any of HECSN's internal labels:

```
Examples:
  ("ocean", "water", "desert")        → ocean should be closer to water than to desert
  ("fire", "heat", "cold")            → fire should be closer to heat than to cold
  ("dog", "bark", "silence")          → dog should be closer to bark than to silence
  ("submarine", "depth", "sky")       → submarine closer to depth than to sky
  ("cooking", "heat", "freeze")       → cooking closer to heat than to freeze
```

For each triple, route all three through the network and compare prototype-space distances.

`grounding_probe = fraction of triples where sim(proto_anchor, proto_positive) > sim(proto_anchor, proto_negative)`

**Thresholds:**
- 0.50: random (no semantic structure)
- 0.55: weak structure
- 0.65: genuine semantic organization — **paper-worthy, minimum target**
- 0.75: strong semantic structure
- 0.85: exceptional — unlikely without very long training

**Concreteness gap test (secondary):** Measure grounding probe separately for concrete triples (physical objects, actions, sensory properties) vs abstract triples (social concepts, logical relations). If grounding is genuinely perceptual, concrete accuracy should exceed abstract accuracy by > 0.10.

### 7.5 Level 5: Novelty Coverage (Learning Health Monitor)

Track the fraction of tokens that caused a prototype to move significantly (delta > 0.005).

**Healthy ranges:**
- Bootstrap phase: > 0.80 (everything is novel)
- Active learning: 0.20–0.50
- Mature operation: 0.05–0.15
- Alert — saturation: < 0.02 (network stopped learning)
- Alert — instability: > 0.90 post-bootstrap (not consolidating)

### 7.6 Baselines

Both baselines must be evaluated on the same stream, same evaluation protocol. If HECSN does not outperform both baselines on the grounding probe, the SNN apparatus is not contributing.

**Baseline 1 — Online SOM:**
Standard Kohonen SOM with neighborhood function, trained online on the same character byte stream. No SNN dynamics, no sleep, no neuromodulation. Same dimensionality, same number of prototypes.

**Baseline 2 — 4-gram character model:**
Online 4-gram model for prediction accuracy baseline. If HECSN's predictive coding does not exceed 4-gram prediction accuracy, the bootstrap mechanism is not functioning.

**Results table (fill with actual numbers):**

| Metric | 4-gram | Online SOM | HECSN Stage 1 | HECSN Stage 3 | HECSN Stage 5 |
|---|---|---|---|---|---|
| Temporal coherence @50K | N/A | measure | measure | measure | measure |
| Compositionality @100K | N/A | measure | measure | measure | measure |
| Grounding probe | ~0.50 | measure | ~0.55 | **>0.65** | **>0.70** |
| Novelty rate @100K | N/A | measure | measure | measure | 0.05–0.15 |
| Task-A recall after Task-B | N/A | measure | measure | measure | measure |

---

## 8. What to Expect: Stage-by-Stage

### 8.1 What Healthy Training Looks Like

**Tokens 0–5,000 (Bootstrap):**
- Prediction errors are high and random
- Chunking boundaries are inconsistent (chunks are 1-4 bytes, mostly noise)
- All neurons have similar activation rates (no specialization)
- Temporal coherence near zero
- Grounding confidence near zero for all concepts
- **Expected behavior: chaotic, high novelty rate**

**Tokens 5,000–50,000 (Early Structure Formation):**
- Common byte sequences begin stabilizing as chunking detectors
- A few high-frequency patterns route consistently to the same column
- Temporal coherence rises from ~0 to ~0.3
- First grounding associations form for concrete nouns (if multimodal input available)
- Sleep begins triggering: first micro-sleeps prevent new assemblies from being immediately overwritten
- **Expected behavior: rising coherence, still high novelty**

**Tokens 50,000–200,000 (Concept Emergence):**
- Clear column specialization visible in prototype space
- Grounding confidence reaches 0.30–0.40 for core concrete vocabulary
- Binding Layer begins forming multi-concept assemblies
- Abstraction Layer starts extracting stable slow features
- Temporal coherence exceeds 0.60
- Compositionality begins rising above 0.55
- **Expected behavior: stable rising metrics, novelty rate settling to 0.15–0.30**

**Tokens 200,000–1,000,000 (Mature Grounding):**
- Grounding probe exceeds 0.65 (Stage 3 completion criterion)
- Concrete concepts ground reliably; abstract concepts partially ground via conceptual neighbors
- Self-criticism catches and corrects worst grounding errors
- Temporal coherence stable above 0.80
- Catastrophic forgetting absent (fragility-gated consolidation working)
- **Expected behavior: stable high metrics, novelty rate 0.05–0.15, active confirmation seeking**

**Tokens 1,000,000+ (Autonomous Operation):**
- System directs its own curriculum through curiosity gap detection
- New concepts acquire grounding through confirmation-seeking without curation
- Sleep consolidation maintains long-term stability
- **Expected behavior: sustained healthy metrics, gradual grounding probe improvement**

### 8.2 What Failure Looks Like and What to Do

**Failure mode 1: Temporal coherence plateaus below 0.5**

Symptom: Same patterns route to different columns across time; no specialization emerging.

Diagnosis: Either learning rate too high (representations unstable), or sleep not running (consolidation absent), or refractory mechanism broken (same columns monopolize routing).

Fix: Check winner history refractory is active. Reduce base_lr by 50%. Verify micro-sleep is triggering every 200 tokens.

**Failure mode 2: Novelty rate collapses to near zero**

Symptom: Network processes thousands of tokens with no prototype movement.

Diagnosis: Saturation — all patterns are routing to well-consolidated columns that resist further learning. The consolidation gate is too strong.

Fix: Reduce consolidation_level threshold from 0.8 to 0.7. Increase NE channel to inject more exploration noise. Check that winner history decays correctly (too slow = refractory kills exploration).

**Failure mode 3: Grounding probe doesn't rise above 0.55 after Stage 2**

Symptom: Cross-modal weights exist but don't predict perceptual patterns reliably.

Diagnosis: Either alignment filter threshold too low (accepting spurious pairings), or Stage 1 grounding was inadequate (filter has no good reference anchors), or audio-text grounding is dominating over visual-text.

Fix: Verify Stage 1 completion criterion was actually met before advancing. Lower alignment_filter_threshold from 0.4 to 0.6. Check that visual encoder is producing non-trivial spike patterns (>5% sparsity).

**Failure mode 4: Self-criticism reduces grounding confidence below 0.3 for many concepts**

Symptom: The self-criticism loop is penalizing concepts faster than confirmation-seeking can repair.

Diagnosis: Either prediction quality is genuinely poor (Stage 2 grounding errors were widespread), or the recent visual buffer window is too short (not enough frames to find confirming evidence).

Fix: Extend recent_visual_buffer from 20 frames to 100 frames. Reduce self-criticism penalty from 10% to 5% per cycle. If problem persists, roll back to Stage 1 checkpoint and extend curated training.

**Failure mode 5: Task-A recall collapses after Task-B (catastrophic forgetting)**

Symptom: After training on a new data domain, previously learned concepts route incorrectly.

Diagnosis: Fragility-gated consolidation is not preventing overwrite. Either anchor_lr is too high (deep sleep moves prototypes too far), or fragility scores are not calculated correctly (wrong memories prioritized for maintenance).

Fix: Reduce anchor_lr from 0.001 to 0.0005. Verify fragility score computation includes both consolidation_level and access_count. Check that micro-sleep correctly skips already-consolidated memories (consolidation > 0.8).

---

## 9. Critical Risks and How to Address Them

### 9.1 The Text-Only Grounding Gap (Managed Risk)

If multimodal data is unavailable or misaligned during early training, the system will form statistical text structures without genuine grounding. These structures are coherent but not semantically grounded — a sophisticated word game, not concept formation.

**Mitigation:** The developmental protocol requires Stage 1 completion before Stage 2 advances. This is a hard requirement. If Stage 1 grounding confidence doesn't reach 0.40 after 200K tokens on curated data, the architecture or data pipeline has a bug — do not advance.

**Honest limitation:** Fully abstract concepts (justice, democracy, infinity) cannot be grounded through direct perceptual co-occurrence. They can only be partially grounded through association with concrete anchors (visual scenes associated with democracy's textual neighborhood). This is a principled limitation, not an engineering failure.

### 9.2 Catastrophic Forgetting at Scale

The fragility-gated three-phase sleep has been validated at 10K/10K token scale (Task A → Task B). At 50K/50K scale, earlier implementations failed until final consolidation was disabled. The current architecture (with anchor_lr = 0.001 and consolidation_level gating) restores viability.

**Residual risk:** At very long training (>10M tokens) with many sequential domains, drift-floor rises may outpace emergency repair. This has not been empirically tested.

**Mitigation:** Periodic full grounding probe evaluation every 1M tokens. If probe declines by > 0.05 from peak, trigger extended deep sleep protocol.

### 9.3 Alignment Filter Bootstrap Problem

The alignment filter depends on grounding confidence from Stage 1. But Stage 1 grounding itself requires correctly functioning chunking and competitive layers. If any of these fail during Stage 1, Stage 2 will compound the errors.

**Mitigation:** Monitor temporal coherence throughout Stage 1. If temporal coherence does not exceed 0.40 by token 100K, restart Stage 1 with: (1) lower base_lr, (2) simpler curated data (MNIST-DVS only, no HowTo100M), (3) verified non-trivial visual encoder output.

### 9.4 GPU Memory at Scale

At 100K columns, dim=256:
- Prototype store (FP32): 100MB
- After TurboQuant (3-bit): ~17MB
- Memory store (10K assemblies, dim=256, FP32): 10MB
- Multimodal encoder outputs (video buffer): variable, ~500MB for 1-second buffer

Total with TurboQuant: comfortably within 24GB (RTX 4090) for 100K columns. For 1M columns, distributed architecture required.

### 9.5 The YouTube Terms of Service Question

Bulk downloading YouTube content for research training raises legal questions that vary by jurisdiction. Specific options that are clearly within terms:

- HowTo100M: pre-approved for academic use (fill access form)
- HTM-AA aligned subset: available directly
- Wikimedia Commons: fully open
- BBC dataset clips: license-specific, check per-project
- yt-dlp for personal research: legally ambiguous in most jurisdictions

Recommendation: begin with HowTo100M (already approved), Wikimedia Commons (fully open), and MNIST-DVS + TI-46 (academic license). The developmental protocol does not require raw YouTube access.

---

## 10. Implementation Roadmap

### Phase 0: Foundation (Weeks 1–2)

**Priority 1: GPU-native router benchmark**

Implement the flat cosine similarity router (no CPU HNSW). Measure actual latency at 1K, 10K, 50K, 100K columns on target hardware. These numbers go in the paper. If they don't match the targets in Section 5.2, revise the architecture before building on it.

**Priority 2: Fix the BindingLayer assertion**

Remove `assert n_bindings == n_columns`. Implement sparse random connectivity matrix. Add `grow()` method. This is a 2-hour fix that corrects a fundamental architectural error.

**Priority 3: Replace the neuromodulator scalar**

Implement the four-channel independent system. This is a 1-day fix that corrects another fundamental architectural error.

**Priority 4: Heun's method for AdEx + NaN guard**

Replace forward Euler on the exponential upswing. Add NaN detection. 4-hour fix.

### Phase 1: Evaluation Framework (Week 3)

Before training anything, implement all five evaluation metrics. The grounding probe triples (50 triples) should be written by hand and reviewed for balance between concrete and abstract concepts. Run all metrics on a random untrained network to verify baselines match expected random values (grounding probe ≈ 0.50).

Implement Online SOM baseline and 4-gram baseline. Run all three on the same 50K-token text stream. Record results.

### Phase 2: Chunking and Abstraction Layers (Weeks 4–5)

Implement the ChunkingLayer (predictability-based boundary detection with detector prototypes). Verify that chunk size distribution becomes approximately log-normal by token 50K. Verify that `"tion"`, `"ing"`, `"the "` appear as stable detectors.

Implement AbstractionLayer as a full feedforward layer with anti-Hebbian SFA update. Verify that feedback signals (routing_bias, boundary_bias) are non-trivial (not all-zeros) by token 10K.

### Phase 3: Fragility-Gated Sleep (Week 6)

Replace the current sleep implementation with three-phase fragility-gated sleep. Verify that the 50K/50K continual learning test now passes (Task-A recall > 0.95 after Task-B training). If it doesn't, the anchor_lr needs further reduction before claiming the forgetting problem is solved.

Run `calibrate_functional_minute()` and record the result. Use this value for all STC timescales.

### Phase 4: Multimodal Grounding Pipeline (Weeks 7–9)

Implement EventCameraEncoder and CochleagramEncoder. Verify non-trivial spike patterns on sample video/audio.

Implement CrossModalGroundingLayer with STDP update rules. Verify that grounding confidence rises from 0 during MNIST-DVS + TI-46 training.

Implement AlignmentFilter. Verify rejection of spurious visual pairings using known-misaligned test pairs.

### Phase 5: Stage 1 Training (Week 10)

Train on curated grounding data (MNIST-DVS + TI-46 + HTM-AA high-alignment subset). Monitor grounding confidence and temporal coherence. Do not advance until Stage 1 completion criterion is met.

Run full evaluation: temporal coherence, compositionality, grounding probe. Record Stage 1 results.

### Phase 6: Stages 2–3 Training and Evaluation (Weeks 11–14)

Train on expanded data with alignment filter active. Monitor alignment filter precision. Run self-criticism loop. Monitor grounding probe.

Target: grounding probe > 0.65 by end of Stage 3. If not reached, diagnose using failure modes in Section 8.2.

Record concreteness gap (concrete grounding probe score − abstract grounding probe score). If this gap is < 0.05, the grounding mechanism is not functioning as intended.

### Phase 7: Paper Writing (Weeks 15–16)

The paper is 8–10 pages. Structure:

1. Introduction: the grounding problem, the biological solution, the falsifiable claim
2. Related Work: CoLaNET (closest SNN competitor), SpikeGPT/NeuronSpark (different objective), Online SOM (null hypothesis), Karlsson et al. 2026 (predictive coding), HowTo100M (data), TurboQuant (compression)
3. Architecture: six sections, one per layer
4. Grounding: temporal co-occurrence STDP, alignment filter, self-criticism
5. Developmental Protocol: five stages, completion criteria
6. Evaluation: five metrics, two baselines
7. Experiments: Stage 1 → Stage 3 results, baseline comparison, concreteness gap test
8. Limitations: honest assessment of residual grounding gap, scale not demonstrated past 100K
9. Conclusion

---

## 11. References

[1] Hebb, D. O. (1949). *The Organization of Behavior.* Wiley.

[2] Kohonen, T. (1982). Self-organized formation of topologically correct feature maps. *Biological Cybernetics*, 43(1), 59–69.

[3] Maass, W. (1997). Networks of spiking neurons: The third generation of neural network models. *Neural Networks*, 10(9), 1659–1671.

[4] Brette, R. & Gerstner, W. (2005). Adaptive exponential integrate-and-fire model. *Journal of Neurophysiology*, 94, 3637–3642.

[5] Song, S., Miller, K. D., & Abbott, L. F. (2000). Competitive Hebbian learning through spike-timing-dependent synaptic plasticity. *Nature Neuroscience*, 3(9), 919–926.

[6] Vogels, T. P. et al. (2011). Inhibitory plasticity balances excitation and inhibition in sensory pathways and memory networks. *Science*, 334(6062), 1569–1573.

[7] Wiskott, L. & Sejnowski, T. J. (2002). Slow Feature Analysis: Unsupervised Learning of Invariances. *Neural Computation*, 14(4), 715–770.

[8] Frey, U. & Morris, R. G. (1997). Synaptic tagging and long-term potentiation. *Nature*, 385(6616), 533–536.

[9] Luboeinski, J. & Tetzlaff, C. (2021). Memory consolidation and improvement by synaptic tagging and capture in recurrent spiking networks. *Communications Biology*, 4(275).

[10] Nair, A. et al. (2024). Causal evidence of a line attractor encoding an affective state. *Nature*, 634, 394–401.

[11] Sagodi, A. et al. (2024). Back to the Continuous Attractor. *arXiv:2408.00109*.

[12] Effenberger, F., Jost, J., & Levina, A. (2015). Self-organization in balanced state networks by STDP and homeostatic plasticity. *PLOS Computational Biology*, 11(9), e1004420.

[13] Chong, Y. S., Ang, S. R., & Sajikumar, S. (2025). Beyond boundaries: extended temporal flexibility in synaptic tagging and capture. *Communications Biology*, 8, 475.

[14] Johnson, J., Douze, M., & Jegou, H. (2021). Billion-scale similarity search with GPUs. *IEEE Transactions on Big Data*, 7(3), 535–547.

[15] Li, Y. et al. (2024). Artificial visual neurons with NbOx Mott memristors for rate-temporal fusion encoding. *Nature Communications*, 15, 6027.

[16] Rathi, N. & Roy, K. (2019). STDP-Based Unsupervised Multimodal Learning With Cross-Modal Processing in Spiking Neural Networks. *IEEE Transactions on Emerging Topics in Computational Intelligence*, 5(1).

[17] Miech, A. et al. (2019). HowTo100M: Learning a Text-Video Embedding by Watching Hundred Million Narrated Video Clips. *ICCV 2019.*

[18] Zhukov, D. et al. (2022). Temporal Alignment Networks for Long-term Video. *CVPR 2022.* [HTM-AA dataset, aligned subset of HowTo100M]

[19] Chen, S. et al. (2021). Multimodal clustering networks for self-supervised learning from unlabeled videos. *ICCV 2021.*

[20] Achille, A., Rovere, M., & Soatto, S. (2019). Critical learning periods in deep neural networks. *ICLR 2019.*

[21] Critical Learning Periods for Multisensory Integration in Deep Networks. *arXiv:2210.04643.*

[22] Understanding the Learning Phases in Self-Supervised Learning via Critical Periods. *OpenReview 2025.*

[23] Karlsson, V., Fianda, N., & Kamarainen, J.-K. (2026). Difference Predictive Coding for Training Spiking Neural Networks. *ICLR 2026.*

[24] NeuronSpark (2026). A 0.9B-parameter spiking language model trained from scratch. *arXiv:2603.16148.*

[25] Zandieh, A. et al. (2026). TurboQuant: Near-optimal Vector Quantization for KV Cache and Vector Search. *ICLR 2026.* [PolarQuant + QJL two-stage compression]

[26] Naderi, R. et al. (2025). Unsupervised post-training learning in spiking neural networks. *Scientific Reports*, 15, 17647.

[27] Zenke, F. & Vogels, T. P. (2021). The Remarkable Robustness of Surrogate Gradient Learning. *Neural Computation*, 33(4), 899–925.

[28] Larionov, D. et al. (2025). Continual Learning with Columnar Spiking Neural Networks. *arXiv:2506.17169.* [CoLaNET — closest existing competitor]

[29] Schuman, C. D. et al. (2022). Opportunities for neuromorphic computing algorithms and applications. *Nature Computational Science*, 2(1), 10–19.

[30] Roy, D. et al. (2019). Lifelong Learning of Spatiotemporal Representations. *Frontiers in Neural Networks*, 12:765.

---

*Thiago Maceno Rocha Goulart · Brasil · github.com/Tuafo*

*HECSN v2.0 — Hierarchical Emergent Concept Spiking Networks with Multimodal Grounding*

*PyTorch 2.1+ · CUDA · 2:4 Structured Sparsity · CSR Sparse · GPU-native IVF · TurboQuant*

*Research Architecture — All verification targets are falsifiable predictions.*

*Central claim: grounding probe accuracy > 0.65 after full developmental training, with concrete concepts significantly exceeding abstract concepts — without labels at any stage.*
