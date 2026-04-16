# HECSN-Warm — Bootstrapping a Living Spiking Network from Large Language Model Priors
## Companion Paper to HECSN v4.21: The Terminus Hybrid Architecture

**Author:** Thiago Maceno Rocha Goulart · Brasil · [github.com/Tuafo](https://github.com/Tuafo)

**Domain:** Neuromorphic Computing · LLM-SNN Hybrid Architectures · Multimodal Grounding

**Status:** Architectural proposal — companion to HECSN_Paper_v4.21.md

**Relationship to main paper:** This document extends HECSN (Hierarchical Emergent Concept Spiking Networks) with a practical bootstrapping strategy that uses pretrained LLM weights to accelerate the developmental protocol. It does NOT replace the main paper — it describes a second operational mode ("HECSN-Warm") alongside the original tabula rasa mode ("HECSN-Pure").

---

## Abstract

HECSN-Pure demonstrates that semantic structure can emerge from temporal co-occurrence in a spiking neural network using only local Hebbian mechanisms. However, training from tabula rasa faces an intractable scaling barrier: at ~57 tok/s (the current best full-architecture throughput after v4.19 optimizations, +181% from baseline), reaching LLM-scale distributional coverage (~1T tokens) would require ~557 years. This paper proposes HECSN-Warm, a hybrid architecture that resolves this barrier by bootstrapping HECSN's competitive layer prototypes from pretrained LLM word embeddings while preserving the emergent properties that make HECSN scientifically novel: cross-modal grounding, sleep consolidation, autonomous curiosity, and continual learning. The architecture introduces three integration mechanisms: (1) one-time prototype seeding via PCA-projected LLM embeddings, (2) bidirectional runtime grounding enrichment between the SNN and LLM, and (3) an LLM-as-curiosity-oracle loop that replaces static corpus search with intelligent experience curation. The central thesis is reframed: rather than claiming all semantic structure emerges from scratch, HECSN-Warm claims that given distributional text priors (analogous to genetic priors in biological cortex), genuine multimodal grounding, autonomous knowledge formation, and continual adaptation emerge through local Hebbian mechanisms — capabilities absent from the source LLM. This enables the Terminus system to operate as a persistent, living cognitive substrate within days rather than decades.

---

## Table of Contents

1. [Motivation: The Scaling Wall](#1-motivation-the-scaling-wall)
2. [The Biological Argument for Priors](#2-the-biological-argument-for-priors)
3. [Architecture Overview: Terminus Hybrid](#3-architecture-overview-terminus-hybrid)
4. [Mechanism 1: LLM → SNN Prototype Bootstrap](#4-mechanism-1-llm--snn-prototype-bootstrap)
5. [Mechanism 2: Bidirectional Runtime Grounding](#5-mechanism-2-bidirectional-runtime-grounding)
6. [Mechanism 3: LLM as Curiosity Oracle](#6-mechanism-3-llm-as-curiosity-oracle)
7. [What Changes in the Developmental Protocol](#7-what-changes-in-the-developmental-protocol)
8. [Reframing the Emergence Claim](#8-reframing-the-emergence-claim)
9. [Corrections and Additions to the Main Paper](#9-corrections-and-additions-to-the-main-paper)
10. [Evaluation: How to Prove This Works](#10-evaluation-how-to-prove-this-works)
11. [Related Work](#11-related-work)
12. [Risks and Open Problems](#12-risks-and-open-problems)
13. [References](#13-references)
14. [Appendix A: Implementation Guide — Local LLM Selection](#appendix-a-implementation-guide--local-llm-selection)

---

## 1. Motivation: The Scaling Wall

### 1.1 The Numbers That Force This Decision

HECSN-Pure's validated throughput benchmarks (from the main paper, v4.21):

| Configuration | Throughput | Time for 10M tokens | Time for 1B tokens | Time for 1T tokens |
|---|---|---|---|---|
| 256 cols, CPU, full architecture (v4.19+) | ~57 tok/s | 2.0 days | 203 days | ~557 years |
| 256 cols, CPU, Stage 1 only | 72 tok/s | 1.6 days | 161 days | ~440 years |
| 1M-token scale validated | 72 tok/s sustained | — | — | — |
| 50K post-optimization (under load) | 26.4 tok/s | — | — | — |

> **v4.19 optimization impact:** Cumulative throughput improvement from 20.3 → ~57 tok/s (+181%) through `.item()` call reduction (84%), `torch.tensor` reduction (79%), `@torch.no_grad()` on `train_step()`, and full-matrix STDP elimination of fancy indexing. Profile now fully distributed (no function >7% of runtime).

Modern LLMs are trained on 1–15 trillion tokens. Even if HECSN only needs 1% of that volume to achieve useful distributional coverage (10 billion tokens), the training time at current throughput would be **~2.0 years continuous** at the fastest full-architecture rate.

This is not a parallelization problem — HECSN's STDP learning is inherently sequential (each synaptic update depends on the previous spike timing). GPU parallelization helps routing (sub-1ms at 100K columns) but not the temporal learning pipeline.

### 1.2 What This Means Practically

The goal of Terminus is a **living brain** — a persistent SNN that runs continuously, forming new knowledge, grounding concepts in multimodal experience, and adapting to novel inputs. This cannot be achieved if the system requires years of tabula rasa training before it has sufficient text-statistical structure to begin meaningful grounding.

The choice is clear:
- **Option A:** Wait years for HECSN-Pure to develop text structure from scratch → then begin grounding → then become useful. Scientifically pure but practically unachievable.
- **Option B:** Bootstrap text-statistical structure from an LLM (minutes) → immediately begin multimodal grounding (days) → operating as a living system (weeks). Scientifically honest about what is inherited vs. what emerges.

We choose Option B.

### 1.3 Why Not Just Use the LLM?

If we can take knowledge from an LLM, why not just use the LLM directly? Because LLMs have structural limitations that HECSN specifically addresses:

| Capability | LLM | HECSN-Warm |
|---|---|---|
| Text distributional statistics | ✅ Excellent (trillions of tokens) | ✅ Inherited from LLM |
| Multimodal grounding | ❌ Text-only (or superficial CLIP-style) | ✅ Temporal co-occurrence STDP |
| Continual learning | ❌ Frozen after training (or catastrophic forgetting with fine-tuning) | ✅ Sleep-consolidated STDP |
| Autonomous curiosity | ❌ Only responds to prompts | ✅ GeometricCuriosityController |
| Energy efficiency | ❌ Billions of FLOPs per token | ✅ Binary spikes, sparse activation |
| Biological plausibility | ❌ Backprop, attention, softmax | ✅ AdEx/ALIF neurons, STDP, neuromodulation |
| Catastrophic forgetting resistance | ❌ Fundamental problem | ✅ Fragility-gated sleep consolidation (validated) |
| Autonomous action initiation | ❌ Requires external prompts | ✅ STDP lateral propagation triggers actions |

HECSN-Warm combines the distributional strength of LLMs with the emergent, grounded, continually-learning properties of spiking networks.

---

## 2. The Biological Argument for Priors

### 2.1 Brains Do Not Start Tabula Rasa

The HECSN-Pure paper (§1.3) already acknowledges three categories of non-emergent components:

> *"The architecture itself (columnar organization, layer hierarchy) — this is a structural prior analogous to the brain's cortical organization before experience"*

> *"The biological parameters (STDP time constants, E/I ratios) — these are external constraints derived from decades of neuroscience, not learned from data"*

> *"The Stage-1 data curation — this is a developmental scaffold whose structure directly encodes grounding information"*

But biological brains have even more priors than HECSN-Pure acknowledges:

| Biological prior | Function | HECSN-Pure equivalent | HECSN-Warm equivalent |
|---|---|---|---|
| Cortical layer structure | Information processing hierarchy | 7-layer architecture | Same |
| Cortical area specialization | V1 for vision, A1 for audio, Broca's for language | Modality-specific encoders | Same |
| Innate feature detectors (V1 edge orientation) | Pre-wired perceptual primitives | None (RTF learns from scratch) | None |
| Genetically-determined connectivity | Initial wiring patterns between areas | Random initialization | **LLM-seeded prototypes** |
| Evolved statistical priors about the world | Objects persist, gravity pulls down, faces are special | None | **LLM distributional knowledge** |
| Genomic language acquisition device (Chomsky) | Pre-wired syntactic structure | None | **LLM syntactic/semantic neighborhoods** |

The LLM embedding bootstrap is analogous to **genetic priors** in biological development. A human infant does not learn from scratch that "dog" and "cat" are more related to each other than to "democracy" — there are genetically-encoded perceptual and cognitive biases that make certain associations more likely. LLM embeddings provide the computational equivalent: a prior distribution over word relationships that the SNN refines through experience.

### 2.2 The Emergence Claim Is Not Invalidated — It's Relocated

What was emergent in HECSN-Pure:
1. Text-statistical prototype structure
2. Cross-modal grounding
3. Binding associations
4. Curiosity-driven exploration
5. Sleep-consolidated knowledge

What is emergent in HECSN-Warm:
1. ~~Text-statistical prototype structure~~ → Inherited (like genes)
2. **Cross-modal grounding** → Emergent (no LLM has this)
3. **Binding associations** → Emergent (temporal coincidence via STDP)
4. **Curiosity-driven exploration** → Emergent (internal gap detection)
5. **Sleep-consolidated knowledge** → Emergent (fragility-gated consolidation)
6. **NEW: Prototype refinement through experience** → Emergent (STDP modifies inherited prototypes based on multimodal evidence)
7. **NEW: Novel concept formation** → Emergent (concepts formed from grounding that weren't in the LLM vocabulary)

Five out of seven emergent properties are preserved. The two most scientifically novel claims — multimodal grounding and continual learning — are fully intact.

---

## 3. Architecture Overview: Terminus Hybrid

### 3.1 System Diagram

```
╔══════════════════════════════════════════════════════════════════════╗
║                        TERMINUS SYSTEM                              ║
║                                                                      ║
║  ┌────────────────────────────────┐                                  ║
║  │       LLM (Reasoning)          │                                  ║
║  │  ┌──────────────────────────┐  │                                  ║
║  │  │   Frozen Weights         │  │                                  ║
║  │  │   (Gemma 4 / Qwen / etc.) │  │                                  ║
║  │  └──────────┬───────────────┘  │                                  ║
║  │             │                  │                                  ║
║  │  ┌──────────▼───────────────┐  │                                  ║
║  │  │   Embedding Matrix       │──┼──── ONE-TIME BOOTSTRAP ────┐    ║
║  │  │   [vocab × d_model]      │  │    (extract + PCA + seed)   │    ║
║  │  └──────────────────────────┘  │                             │    ║
║  └────────┬───────────────▲───────┘                             │    ║
║           │               │                                     │    ║
║    curiosity          grounding                                 │    ║
║     oracle            signals                                   │    ║
║           │               │                                     │    ║
║  ┌────────▼───────────────┴──────────────────────────────────┐  │    ║
║  │              HECSN-Warm (Living SNN)                       │  │    ║
║  │                                                             │  │    ║
║  │  ┌─────────────────────────────────────────────────────┐   │  │    ║
║  │  │  LAYER 0: Multimodal Encoders                       │   │  │    ║
║  │  │  Text (RTF) · Visual (EventCamera) · Audio (Coch.)  │   │  │    ║
║  │  └──────────────────────┬──────────────────────────────┘   │  │    ║
║  │                         │                                   │  │    ║
║  │  ┌──────────────────────▼──────────────────────────────┐   │  │    ║
║  │  │  LAYER 1–2: Chunking + RTF Encoding                 │   │  │    ║
║  │  │  feature_vec [128] → routing_key [256]               │   │  │    ║
║  │  └──────────────────────┬──────────────────────────────┘   │  │    ║
║  │                         │                                   │  │    ║
║  │  ┌──────────────────────▼──────────────────────────────┐   │  │    ║
║  │  │  LAYER 3: Surprise Monitor + Neuromodulation        │   │  │    ║
║  │  │  DA → LTP gain · 5-HT → patience · ACh · NE        │   │  │    ║
║  │  └──────────────────────┬──────────────────────────────┘   │  │    ║
║  │                         │                                   │  │    ║
║  │  ┌──────────────────────▼──────────────────────────────┐   │  │    ║
║  │  │  LAYER 4: Adaptive Context Layer                    │   │  │    ║
║  │  │  Per-neuron learnable timescales                     │   │  │    ║
║  │  └──────────────────────┬──────────────────────────────┘   │  │    ║
║  │                         │                                   │  │    ║
║  │  ┌──────────────────────▼──────────────────────────────┐   │  │    ║
║  │  │  LAYER 5: Competitive Layer  ◄── LLM PROTOTYPES ───┼───┘    ║
║  │  │  N_columns prototypes [256] seeded from LLM PCA     │   │       ║
║  │  │  WTA + Triplet STDP + IP + synaptic scaling         │   │       ║
║  │  │  STDP refines inherited prototypes from experience   │   │       ║
║  │  └──────────────────────┬──────────────────────────────┘   │       ║
║  │                         │                                   │       ║
║  │  ┌──────────────────────▼──────────────────────────────┐   │       ║
║  │  │  LAYER 6: Binding Layer (dense/spatial/hypercube)    │   │       ║
║  │  │  Temporal coincidence · STP · PV+ inhibition        │   │       ║
║  │  │  Hypercube mode (§4.12): O(N·d) bit-flip, 0.54%    │   │       ║
║  │  └──────────────────────┬──────────────────────────────┘   │       ║
║  │                         │                                   │       ║
║  │  ┌──────────────────────▼──────────────────────────────┐   │       ║
║  │  │  LAYER 7: Abstraction Layer (online SFA)            │   │       ║
║  │  │  Curiosity gaps → LLM oracle queries                │   │       ║
║  │  └──────────────────────┬──────────────────────────────┘   │       ║
║  │                         │                                   │       ║
║  │  ┌──────────────────────▼──────────────────────────────┐   │       ║
║  │  │  CROSS-MODAL GROUNDING LAYER                        │   │       ║
║  │  │  W_tv, W_vt, W_ta, W_at = ZERO (tabula rasa)       │   │       ║
║  │  │  *** NOT bootstrapped — grounding is emergent ***    │   │       ║
║  │  │  STDP + alignment filter + self-criticism            │   │       ║
║  │  └─────────────────────────────────────────────────────┘   │       ║
║  │                                                             │       ║
║  │  ┌─────────────────────────────────────────────────────┐   │       ║
║  │  │  SLEEP CONSOLIDATION (3-phase, fragility-gated)     │   │       ║
║  │  │  Protects both inherited and learned knowledge       │   │       ║
║  │  │  SFA correction · dead column census · tau update    │   │       ║
║  │  └─────────────────────────────────────────────────────┘   │       ║
║  └─────────────────────────────────────────────────────────────┘       ║
║                                                                        ║
║  ◄──── MULTIMODAL WORLD INPUT ────►                                   ║
║  Text streams · Video frames · Audio samples · Sensor data             ║
╚════════════════════════════════════════════════════════════════════════╝
```

### 3.2 Component Ownership

| Component | Source | Mutable? | Learning rule |
|---|---|---|---|
| Competitive prototypes [256] | LLM embedding PCA | ✅ Yes — STDP refines | Triplet STDP + WTA |
| W_project [128 × 256] | Optionally from LLM | ✅ Yes — Hebbian | Local Hebbian update |
| Cross-modal W matrices | Zero (tabula rasa) | ✅ Yes — STDP | Temporal co-occurrence STDP |
| Per-word sensory signatures | Zero | ✅ Yes — EMA | Exponential moving average |
| Context layer weights | Xavier init | ✅ Yes — Hebbian | Routing differentiation |
| Binding layer connections | Sparse random / hypercube init | ✅ Yes — Hebbian | Correlation-based growth (2-hop paths in hypercube mode) |
| Abstraction layer weights | Xavier init | ✅ Yes — anti-Hebbian | Online SFA |
| Neuromodulator parameters | Fixed biological | ❌ No | Hardcoded from neuroscience |
| Sleep consolidation parameters | STC calibrated | ❌ No | Self-calibrated functional_minute |
| LLM weights | Pretrained | ❌ No — frozen | N/A (inference only) |

**Critical invariant:** Cross-modal weights (W_tv, W_vt, W_ta, W_at) are NEVER bootstrapped from the LLM. These remain zero-initialized. Grounding is the emergent property.

---

## 4. Mechanism 1: LLM → SNN Prototype Bootstrap

### 4.1 The Procedure

```
Input:  Pretrained LLM with embedding matrix E ∈ ℝ^{V × d_model}
        HECSN config: N_columns, column_latent_dim = 256
Output: Initial prototypes P ∈ ℝ^{N_columns × 256}

Step 1: EXTRACT — Get the LLM word embedding matrix
        E = model.get_input_embeddings().weight  # [V × d_model]
        # V ≈ 32K–128K vocabulary, d_model ≈ 768–4096

Step 2: FILTER — Select the most frequent/useful N_words embeddings
        # Use a frequency-ranked vocabulary (top 10K–50K words)
        # Discard subword fragments, special tokens, rare words
        E_filtered ∈ ℝ^{N_words × d_model}

Step 3: PROJECT — PCA to column_latent_dim (256)
        # Center the embeddings
        E_centered = E_filtered - E_filtered.mean(dim=0)
        # SVD for PCA
        U, S, V = torch.svd(E_centered)
        # Project to 256 dimensions
        E_projected = E_centered @ V[:, :256]  # [N_words × 256]
        # L2 normalize (prototypes use cosine similarity for routing)
        E_projected = F.normalize(E_projected, dim=1)

Step 4: SELECT — Choose N_columns prototypes via k-means
        # Run k-means on E_projected with k = N_columns
        # Each centroid becomes a prototype
        # This ensures prototypes cover the embedding space uniformly
        prototypes = kmeans(E_projected, k=N_columns)  # [N_columns × 256]

Step 5: INJECT — Set competitive layer prototypes
        model.competitive_layer.prototypes = prototypes
        # All other weights remain at their default initialization
        # Cross-modal W matrices remain ZERO
```

### 4.2 Which LLM to Use

The LLM serves up to three roles in HECSN-Warm. Different roles have different requirements:

| Role | When | Requirement | Duration |
|---|---|---|---|
| **Embedding source** (bootstrap) | One-time, at initialization | Access to embedding matrix | Minutes |
| **Curiosity oracle** (Mechanism 3) | Runtime, on-demand | Inference capability, ideally multimodal | Ongoing |
| **Grounding partner** (Mechanism 2) | Runtime, continuous | Inference capability | Ongoing |

#### Candidate Models for Local Deployment (RTX 3060 12GB)

| Model | Params | d_model | Vocab | VRAM (Q4) | Modalities | Context | Oracle quality | Bootstrap quality |
|---|---|---|---|---|---|---|---|---|
| **Gemma 4 E4B** | 8B raw / 4.5B eff. | ~2560 | 262K | **~5GB** (PLE) | **Text+Image+Audio** | 128K | ★★★★★ | ★★★★★ |
| **Gemma 4 E2B** | 5.1B raw / 2.3B eff. | ~2048 | 262K | **~3.2GB** (PLE) | **Text+Image+Audio** | 128K | ★★★★ | ★★★★ |
| Gemma 4 26B A4B (MoE) | 25.2B / 3.8B active | ~3072 | 262K | 18GB ❌ | Text+Image | 256K | ★★★★★ | ★★★★★ |
| Gemma 4 31B Dense | 30.7B | ~4096 | 262K | 20GB ❌ | Text+Image | 256K | ★★★★★ | ★★★★★ |
| Qwen3 4B | 4B | 2560 | 151K | 2.5GB | Text only | 256K | ★★★ | ★★★★ |
| Qwen3 8B | 8B | 4096 | 151K | 5.2GB | Text only | 40K | ★★★★ | ★★★★★ |
| Phi-4-mini | 3.8B | 3072 | 200K | ~2.5GB | Text only | 128K | ★★★ | ★★★ |
| fastText (cc.en.300) | N/A | 300 | 2M | **0 (CPU)** | Text (static) | N/A | ✗ | ★★★★ |
| GloVe 840B | N/A | 300 | 2.2M | **0 (CPU)** | Text (static) | N/A | ✗ | ★★★★ |


#### Primary Recommendation: Gemma 4 E4B

**Gemma 4 E4B is the ideal LLM for HECSN-Warm.** The argument is architectural:

HECSN operates on three modalities: **text, vision, and audio**.
Gemma 4 natively processes three modalities: **text, vision, and audio**.

No other consumer-grade LLM matches all three of HECSN's modalities. This is not a coincidence to overlook — it makes Gemma 4 uniquely suited for every role:

**1. As curiosity oracle (killer feature):**
When HECSN's curiosity controller identifies a gap concept (e.g., "thunder"), Gemma 4 can:
- **Hear** thunder audio (native audio encoder, USM-based, 6.25 tokens/sec) and describe what it sounds like
- **See** lightning imagery (native vision encoder, MobileNet-v5) and describe what it looks like
- Generate paired text descriptions that HECSN processes alongside the actual multimodal data
- **Think step-by-step** using built-in configurable thinking mode for complex descriptions

No other local LLM can do this. Qwen3, Phi-4, LLaMA — they are text-only or text+vision at best. Only Gemma 4 E2B/E4B process all three modalities, making it the ideal "parent/teacher" in the developmental analogy: it can look at a picture, listen to a sound, and explain what it perceives — exactly what a caregiver does for a child.

**2. As grounding partner:**
When HECSN returns grounding signals to the LLM, Gemma 4 can actually *use* that information because it already has multimodal understanding. A text-only LLM receiving "fire: visual=flickering,orange-red" cannot verify this against its own experience. Gemma 4 can.

**3. As embedding source:**
Gemma 4's 262K token vocabulary (trained on 11+ trillion tokens across 140+ languages) provides rich distributional embeddings. The subword tokenizer requires word-level reconstruction (average subword embeddings), but the coverage and quality justify the extra step.

**4. Key Gemma 4 features for HECSN-Warm:**
- **128K context** — longer curiosity descriptions, more context for grounding partner role
- **Configurable thinking mode** — step-by-step reasoning for complex multimodal descriptions
- **Native function calling** — can be wired directly into HECSN's curiosity API
- **Apache 2.0 license** — fully open for research and commercial use

**5. VRAM budget:**
Gemma 4 uses Per-Layer Embeddings (PLE) technology — the PLE parameters are computed on CPU and streamed per-layer. On the RTX 3060 12GB:

```
VRAM Budget (RTX 3060, 12GB):
├── Gemma 4 E4B (Q4_K_M, PLE):  ~5 GB
├── HECSN routing (100K cols):   ~0.1 GB
├── KV cache + overhead:         ~2 GB
└── Available headroom:          ~5 GB   ← comfortable
```

**Installation and first run:**

```powershell
# Install via Ollama (easiest path)
ollama pull gemma4:e4b           # 9.6GB download (Q4_K_M)

# Or for lighter footprint:
ollama pull gemma4:e2b           # 7.2GB download (Q4_K_M)

# Test inference:
ollama run gemma4:e4b "Describe what a volcano looks like and sounds like"

# For programmatic access (Python):
pip install ollama
```

```python
# Extract embeddings for bootstrap (via HuggingFace transformers)
from transformers import AutoProcessor, AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-4-e4b-it",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
# Extract the text embedding matrix
embedding_matrix = model.get_input_embeddings().weight.detach().float()
# Shape: [vocab_size, d_model] ≈ [256000, 2560]
```

#### Alternative Strategy: Two-Model Approach

For maximum quality with minimal VRAM pressure, use two separate models optimized for each role:

| Role | Model | VRAM | Rationale |
|---|---|---|---|
| Bootstrap (one-time) | **fastText cc.en.300** | 0 (CPU only) | Word-level (no subword reconstruction), 300d → 256d PCA is nearly lossless, 2M word vocabulary, instant extraction, well-studied geometry |
| Curiosity oracle + grounding (runtime) | **Gemma 4 E2B** | ~3.2GB | Multimodal, lighter than E4B, leaves maximum headroom for HECSN |

This decouples the bootstrap quality from the oracle quality. fastText provides arguably *better* bootstrap embeddings (word-level, pre-clustered, massive vocabulary) while Gemma 4 E2B provides the multimodal oracle capability at minimal VRAM cost.

```powershell
# Download fastText embeddings (one-time, for bootstrap)
# From: https://fasttext.cc/docs/en/crawl-vectors.html
# File: cc.en.300.bin (4.2GB) or cc.en.300.vec (text format, 6.4GB)

# Run Gemma 4 E2B for oracle (runtime)
ollama pull gemma4:e2b
```

#### When to Use Text-Only Models Instead

If multimodal oracle capability is not needed (e.g., early prototyping, text-only experiments), text-only models offer better text reasoning per VRAM:

| Model | VRAM | Context | Best for |
|---|---|---|---|
| **Qwen3 4B** | 2.5GB | 256K | Text reasoning, thinking mode, long context. Best text-only option at low VRAM. |
| **Qwen3 8B** | 5.2GB | 40K | Stronger text embeddings and reasoning. Good bootstrap source. |
| **Phi-4-mini** | ~2.5GB | 128K | Strong reasoning, 200K vocab. Good for function-calling patterns. |

```powershell
# Qwen3 as text-only fallback
ollama pull qwen3:4b             # 2.5GB, 256K context, thinking mode
```

#### A Note on Gemma Versioning

Gemma 4 was released in April 2026 under Apache 2.0 license. It features four model sizes — E2B and E4B (on-device, multimodal including audio, PLE + MatFormer architecture) and 26B A4B MoE / 31B Dense (server-grade, text+image only). The E2B/E4B models use Per-Layer Embeddings (PLE) for VRAM efficiency and MatFormer (Matryoshka Transformer) for nested sub-models. Only E2B and E4B fit on 12GB VRAM and support native audio — making them ideal for HECSN-Warm.

### 4.3 Dimensionality Alignment

HECSN's representation contract (from main paper §3.2):

```
feature_vec:   [128]  ← RTF encoder output
routing_key:   [256]  ← W_project @ feature_vec
prototype_i:   [256]  ← Competitive layer centroid
concept_vec:   [256]  ← Abstraction layer output
```

The bootstrap touches only `prototype_i`. The routing key is produced by the RTF encoder pipeline (`feature_vec → W_project → routing_key`), which is NOT bootstrapped. This means:

- The routing keys come from HECSN's native text processing (RTF encoding of character windows)
- The prototypes come from LLM embeddings (PCA-projected)
- There will be an **initial alignment gap** between how HECSN encodes text and where the prototypes sit in latent space

**Why this is okay:** HECSN's STDP learning will shift prototypes toward the actual routing key distribution. The LLM-seeded prototypes provide a *good starting neighborhood structure* (semantically related words are near each other), even if the absolute positions need adjustment. The competitive layer's WTA dynamics + STDP will handle the fine alignment within the first few thousand tokens.

### 4.4 Optional: Bootstrap W_project Too

For faster alignment, also distill the LLM's relationship between character-level features and semantic space:

```
1. Generate feature_vec for each word using RTF encoder
2. Generate target routing_key from LLM embedding (PCA'd)
3. Solve: W_project = argmin ||W @ feature_vec - llm_routing_key||²
4. This is a simple least-squares problem: W = target @ pinv(features)
```

This gives HECSN's native encoder a head start in producing routing keys that land near the right prototypes. Still fully refinable by Hebbian updates during training.

---

## 5. Mechanism 2: Bidirectional Runtime Grounding

### 5.1 HECSN → LLM: Grounding Enrichment

During runtime, when the LLM processes a word that HECSN has grounded:

```
Input:  word w, LLM context
Output: LLM response enriched with perceptual context

1. HECSN looks up word w in its per-word sensory signatures
2. Returns:
   - grounding_confidence: float [0, 1]
   - visual_signature: [dim_visual] — what this word looks like
   - audio_signature: [dim_audio] — what this word sounds like
   - concept_vec: [256] — abstraction layer representation
3. These are formatted as auxiliary context for the LLM:
   "[GROUNDING: 'fire' | conf=0.87 | visual=flickering,orange-red |
    audio=crackling,hissing | related_grounded=['flame','heat','burn']]"
4. LLM incorporates this perceptual context into its reasoning
```

This is analogous to retrieval-augmented generation (RAG), but instead of retrieving documents, the LLM retrieves **perceptual experience** from HECSN's grounded memory.

**What this gives the LLM that it never had:**
- Whether a word refers to something the system has actually *seen* or *heard*
- Perceptual similarity judgments (is "ocean" visually more like "lake" or "desert"?)
- Confidence-calibrated knowledge (HECSN knows what it doesn't know)

### 5.2 LLM → HECSN: Semantic Context

The LLM can provide HECSN with information that local STDP cannot easily extract:

- **Long-range semantic relationships:** "submarine" is related to "ocean" — a connection that requires more context than HECSN's ~500-token window
- **Abstract category structure:** "justice" and "democracy" are both abstract governance concepts — a relationship that requires reasoning HECSN cannot perform
- **Disambiguation:** "bank" near "river" vs "bank" near "money" — the LLM resolves this contextually and can bias HECSN's routing accordingly

Implementation: the LLM provides a **semantic context vector** that modulates HECSN's competitive layer routing gain (same mechanism as the Abstraction Layer's routing bias, §3.1 of main paper):

```
llm_context = llm.encode(recent_text_window)  # [d_model]
context_bias = project_to_columns(llm_context)  # [N_columns]
competitive_layer.routing_bias += alpha * context_bias
```

This is a soft bias, not an override. HECSN's own routing still dominates. The LLM context acts as a top-down prior, similar to the Abstraction Layer's existing routing bias mechanism.

---

## 6. Mechanism 3: LLM as Curiosity Oracle

### 6.1 The Current Limitation

HECSN-Pure's curiosity system (§7.4, GeometricCuriosityController) works as follows:

1. Abstraction Layer identifies gaps: high text-activation + low grounding confidence
2. Gap concepts are converted to retrieval queries
3. Queries search a **static corpus** for relevant sentences
4. Training focuses on gap-relevant content

The limitation: the corpus is fixed, finite, and may not contain multimodal data for every gap concept. If the corpus doesn't mention "volcano" alongside volcanic imagery, the gap persists forever.

### 6.2 The LLM Oracle Solution

Replace the static corpus search with an LLM-mediated experience curation:

```
HECSN curiosity loop (enhanced):

1. GeometricCuriosityController identifies gap:
   "Word 'volcano' — text activation high, grounding confidence 0.02"

2. Instead of corpus search → query the LLM:
   "The concept 'volcano' needs multimodal grounding.
    Generate a descriptive scenario that pairs this word with
    visual and audio characteristics."

3. LLM responds:
   "A volcano erupts: glowing red-orange lava flows down dark
    rocky slopes, thick gray ash clouds billow upward, accompanied
    by deep rumbling sounds and explosive cracking."

4. LLM output → multimodal pipeline:
   - Text: the descriptive passage (fed through RTF encoder)
   - Visual: retrieve/generate volcano imagery (event camera encoder)
   - Audio: retrieve/generate volcanic sounds (cochleagram encoder)
   - Temporal alignment: text, visual, audio co-occur within tau_bind

5. HECSN processes the curated experience:
   - Cross-modal STDP forms W_tv associations for 'volcano'
   - Per-word visual/audio signatures accumulate via EMA
   - Grounding confidence rises from 0.02 toward 0.4+

6. Self-criticism loop validates:
   - Does the learned visual prediction for 'volcano' match
     subsequent volcanic imagery? If yes → confirmed. If no → blacklisted.
```

### 6.3 The Biological Analogy

This is precisely what happens in biological language acquisition:

- The child doesn't encounter every word through random environmental exposure
- A **parent/caregiver** (the LLM) deliberately structures experiences:
  - Points at objects and names them
  - Reads picture books with paired text and images
  - Describes sounds and their sources
- The child's brain (HECSN) does the actual perceptual binding through temporal co-occurrence

The main paper already frames Stage 1 data as *"structurally curated data... paired multimodal samples where temporal co-occurrence is guaranteed by construction — not by random chance... precisely as in biological language acquisition"* (§1.2). The LLM oracle extends this curation from Stage 1 into all subsequent stages.

### 6.4 Safety: Self-Criticism Prevents LLM Hallucination Poisoning

A critical concern: LLMs hallucinate. If the LLM generates incorrect descriptions, HECSN could learn wrong groundings.

HECSN-Pure already has the answer — the self-criticism loop (§7.4):

1. After learning a grounding from LLM-curated data, HECSN periodically checks:
   *"I think 'volcano' looks like [prediction from W_tv]. Does this match real visual data?"*
2. If the prediction consistently fails to match actual volcanic imagery → blacklist
3. After 2 strikes → zero out the W_tv row → force re-learning from fresh evidence

The self-criticism mechanism, designed for HECSN-Pure's noisy real-world data, serves double duty as an **anti-hallucination filter** for LLM-generated experiences. Hallucinated visual/audio descriptions that don't match reality get detected and purged.

---

## 7. What Changes in the Developmental Protocol

### 7.1 Modified Stage Progression

```
HECSN-Pure (original):                    HECSN-Warm (bootstrapped):

Stage 0: Architecture validation          Stage 0: Architecture validation
         (unchanged)                               + LLM bootstrap injection
                                                   (minutes, not days)

Stage 1: Curated multimodal exposure      Stage 1: Curated multimodal exposure
         200K–1M tokens                            Duration: MUCH SHORTER
         Builds prototypes from scratch             Prototypes already organized
         Criterion: confidence > 0.40              Criterion: confidence > 0.40
                                                   (expected to pass in ~10K tokens
                                                    vs 200K+ for Pure)

Stage 2: Structured expansion             Stage 2: Structured expansion
         Self-filtering active                     + LLM context bias active
         200K+ tokens                              + LLM curiosity oracle active
         Criterion: probe > 0.60                   Criterion: probe > 0.60

Stage 3: Active confirmation-seeking      Stage 3: Active confirmation-seeking
         Static corpus search                      LLM oracle replaces corpus search
         200K–1M tokens                            Much faster gap resolution

Stage 4: Semi-autonomous                  Stage 4: Semi-autonomous
         Gap-directed acquisition                  LLM-mediated acquisition

Stage 5: Continuous autonomous            Stage 5: Continuous autonomous
         Open-ended operation                      Terminus living brain mode
                                                   HECSN runs 24/7
                                                   LLM available as oracle
                                                   Sleep consolidation cycles
```

### 7.2 Stage 1 Acceleration Estimate

Why Stage 1 should complete much faster with bootstrap:

In HECSN-Pure, Stage 1 must simultaneously:
1. Build text-statistical prototype structure (slow — SOM convergence)
2. Form cross-modal associations (depends on #1)

In HECSN-Warm:
1. ~~Build text-statistical prototype structure~~ → Already done (bootstrap)
2. Form cross-modal associations → Can begin immediately because routing works from token 1

The Stage 1 completion criterion is `mean(grounding_confidence[top_100_text_dims]) > 0.40`. With prototypes pre-organized, text routing to the correct columns should be effective from the start, meaning cross-modal STDP can begin building useful associations immediately.

**Estimated Stage 1 duration:** 5K–20K tokens (vs. 200K–1M for Pure). This is a **10×–200× speedup** for Stage 1 alone.

### 7.3 New Diagnostic: Bootstrap Alignment Score

A new metric specific to HECSN-Warm:

```
bootstrap_alignment = mean cosine_similarity(routing_key_i, nearest_prototype_i)
```

This measures how well HECSN's native text encoding (RTF → feature_vec → routing_key) aligns with the LLM-seeded prototypes. Expected trajectory:

- **Token 0:** Low (0.2–0.4) — RTF encoding and LLM embedding spaces are different
- **Token 1K:** Rising (0.4–0.6) — STDP is shifting prototypes toward actual routing patterns
- **Token 10K:** Stable (0.6–0.8) — alignment achieved, neighborhood structure preserved
- **Ongoing:** Maintained or slowly evolving as new concepts are learned

If bootstrap_alignment stays below 0.3 after 10K tokens → the LLM embedding space is fundamentally incompatible with HECSN's RTF encoding. Fallback: increase STDP learning rate or consider a different LLM source.

---

## 8. Reframing the Emergence Claim

### 8.1 The Two-Claim Structure

The main paper's claim should be split into two independent, testable claims:

**Claim A (HECSN-Pure):** *Representations with genuine semantic structure can emerge from temporal co-occurrence statistics across modalities, using only local Hebbian mechanisms and without semantic labels at any stage.*

This claim is unchanged. HECSN-Pure validates it. The tabula rasa mode remains as the scientific proof-of-concept.

**Claim B (HECSN-Warm):** *Given distributional text priors (analogous to genetic priors in biological cortex), genuine multimodal grounding, autonomous curiosity, and continual knowledge accumulation emerge through local Hebbian mechanisms — capabilities structurally absent from the source model.*

This claim is new. It is testable by comparing:
- Source LLM performance on grounding tasks → baseline (expected: poor/absent)
- HECSN-Warm performance on the same grounding tasks → should exceed LLM
- HECSN-Warm performance vs. HECSN-Pure → should match or exceed (faster, not worse)

### 8.2 The Falsifiable Predictions

| Prediction | Measure | Expected | Falsified if |
|---|---|---|---|
| Grounding is emergent | Source LLM's multimodal grounding score | ~0 (text-only) | LLM already has grounding |
| Bootstrap accelerates Stage 1 | Tokens to confidence > 0.40 | <20K (vs 200K+ for Pure) | No speedup observed |
| Prototype refinement occurs | Prototype drift from bootstrap positions | Measurable (>0.1 cosine distance) | Prototypes frozen |
| Cross-modal quality preserved | Grounding probe at Stage 2 | ≥0.60 (same as Pure) | Significantly lower than Pure |
| Sleep consolidation protects both | Task A/B forgetting test | PASS (same as Pure) | Catastrophic forgetting |
| LLM oracle improves grounding coverage | Number of grounded concepts at Stage 3 | > Pure at same token count | No improvement over corpus search |
| Novel concepts emerge | Concepts in HECSN not in LLM vocabulary | >0 | Zero novel concepts |

---

## 9. Corrections and Additions to the Main Paper

### 9.1 Section Additions

The following sections should be added to or modified in HECSN_Paper_v4.md:

**§1.3 — Add subsection "1.4 Two Operational Modes":**

> HECSN supports two operational modes with distinct scientific claims:
> - **HECSN-Pure:** Tabula rasa training. All semantic structure emerges from temporal co-occurrence. Validates the emergence hypothesis. Computationally intensive (months to years for full vocabulary coverage).
> - **HECSN-Warm:** LLM-bootstrapped training. Distributional text priors are injected into competitive layer prototypes. Cross-modal grounding, binding, curiosity, and continual learning remain emergent. Enables practical deployment as a persistent cognitive substrate (Terminus).

**§3.1 — Add to Competitive Layer description:**

> Prototypes may be initialized via two modes:
> - **Random init** (Pure mode): Xavier uniform. Requires full developmental protocol for text-statistical organization.
> - **LLM bootstrap** (Warm mode): PCA-projected LLM embeddings, k-means selected. Provides pre-organized semantic neighborhoods. STDP refines positions from experience. See Companion Paper §4.

**§7.2 — Add note to Stage 1:**

> **Warm mode acceleration:** When competitive layer prototypes are bootstrapped from LLM embeddings, Stage 1's primary objective (building text-statistical prototype structure) is already achieved at initialization. Stage 1 in Warm mode focuses exclusively on cross-modal association formation, with expected duration reduced to 5K–20K tokens.

**§10 — Add new risk:**

> **10.6 LLM Bootstrap Alignment Risk**
>
> If the LLM embedding geometry is incompatible with HECSN's RTF encoding space, bootstrapped prototypes may resist STDP refinement, leading to persistent misrouting. Monitor: bootstrap_alignment_score after 10K tokens. If < 0.3, the embedding spaces are fundamentally incompatible. Mitigation: also bootstrap W_project (§4.4 of Companion Paper) to align the projection, or use word-level embeddings (fastText/GloVe) instead of transformer subword embeddings.

### 9.2 Terminology Corrections

Throughout the main paper, where "tabula rasa" is used as a core design principle, add clarifying scope:

- "Zero (tabula rasa)" for cross-modal W matrices → **unchanged** in both modes
- "tabula rasa" as overall system philosophy → **qualified**: "tabula rasa for cross-modal grounding; configurable for text-statistical priors (see §1.4)"

### 9.3 New Evaluation Levels

Add to §8 (Evaluation Protocol):

**Level 9: Bootstrap Validation (Warm mode only)**

| Metric | Healthy range | Alert |
|---|---|---|
| Bootstrap alignment score | 0.4–0.8 at 10K tokens | < 0.3 → incompatible spaces |
| Prototype drift rate | 0.05–0.30 cosine distance per 10K tokens | < 0.01 → frozen prototypes |
| Stage 1 completion tokens | 5K–20K | > 50K → bootstrap ineffective |
| Grounding probe (Warm vs Pure, same tokens) | Warm ≥ Pure | Warm << Pure → bootstrap harmful |

**Level 10: LLM Oracle Validation**

| Metric | Healthy range | Alert |
|---|---|---|
| Oracle-grounded concept count | Monotonically increasing | Stagnant → oracle not providing useful data |
| Self-criticism rejection rate on oracle data | < 30% | > 50% → LLM hallucinating too much |
| Gap resolution rate (concepts/hour) | > 1 | < 0.1 → oracle not finding relevant multimodal data |

---

## 10. Evaluation: How to Prove This Works

### 10.1 Experiment 1: Bootstrap vs. Tabula Rasa (Controlled)

```
Setup:
- Same HECSN configuration (256 columns, full architecture)
- Same multimodal training data (N-MNIST + FSDD, digit domain)
- Same random seeds (42, 7, 123)
- Same hardware (RTX 3060)

Condition A: HECSN-Pure (random prototype init)
Condition B: HECSN-Warm (fastText embeddings → PCA → k-means → prototype init)
             Cross-modal W matrices = ZERO in both conditions

Measure at tokens = {1K, 5K, 10K, 50K, 100K}:
- Grounding probe accuracy (50-triple)
- Concreteness gap
- Grounding confidence (mean top-100)
- Prototype silhouette score
- Routing efficiency (winner differentiation)

Expected result:
- Warm reaches probe > 0.60 at ~10K tokens
- Pure reaches probe > 0.60 at ~50K+ tokens
- Both converge to same probe accuracy by 100K tokens
- Cross-modal quality identical (same grounding mechanism)
```

### 10.2 Experiment 2: Grounding Beyond the Source LLM

```
Setup:
- Bootstrap HECSN-Warm from fastText embeddings
- Train with multimodal data (video + audio + text)

Test: Present HECSN-Warm and the source fastText model with:
1. Multimodal semantic triples (requires grounding to solve)
   ("fire", "flame", "ice") — which is the odd one out visually?
2. Perceptual similarity judgments
   Is "ocean" more like "lake" or "desert" visually?
3. Grounding confidence calibration
   Does HECSN-Warm correctly report low confidence for words
   it hasn't grounded, even though fastText has embeddings for them?

Expected result:
- fastText fails all multimodal tasks (no perceptual information)
- HECSN-Warm succeeds on grounded words (emergent multimodal knowledge)
- HECSN-Warm correctly reports low confidence on ungrounded words
- This proves: something emerged in HECSN that was NOT in the source model
```

### 10.3 Experiment 3: LLM Oracle vs. Static Corpus

```
Setup:
- Two HECSN-Warm instances, same bootstrap, same Stage 1
- Instance A: Stage 3 curiosity searches static corpus
- Instance B: Stage 3 curiosity queries LLM oracle

Measure after same wall-clock time:
- Number of grounded concepts (confidence > 0.40)
- Grounding probe accuracy
- Gap resolution rate (ungrounded concepts per hour becoming grounded)
- Self-criticism rejection rate (LLM hallucination detection)

Expected result:
- Instance B grounds significantly more concepts per unit time
- Self-criticism catches LLM hallucinations (rejection rate < 30%)
- Probe accuracy comparable or better
```

---

## 11. Related Work

### 11.1 ANN-to-SNN Conversion (adjacent but different)

Recent work converts transformer LLMs into spiking form:

- **SpikingBERT** (Bal et al., 2023): Distills BERT into a spiking language model via implicit differentiation. Keeps the transformer architecture; replaces ReLU with LIF neurons. Performance: ~97% of BERT on GLUE tasks.
- **SpikeGPT** (Zhu et al., 2023): Builds a generative spiking LM inspired by RWKV with binary spike activations. 45M and 216M parameter variants.
- **LAS/FAS** (Chen et al., 2025): Loss-less/fast ANN-to-SNN conversion for spiking LLMs. Converts attention and MLP layers into spike-compatible form.
- **NeuronSpark** (Tang, 2026): 0.9B-parameter spiking language model with selective state space dynamics.

**How HECSN-Warm differs:** These approaches convert the LLM architecture wholesale — they produce spiking transformers. HECSN-Warm does NOT convert the architecture. It extracts only the embedding matrix (a static lookup table) and injects it into a fundamentally different spiking architecture (columnar competitive routing with STDP). The LLM's computational structure (attention, MLP, layer norms) is discarded entirely. Only the distributional knowledge encoded in the embeddings is preserved.

### 11.2 Hybrid SNN-LLM Architectures (directly related)

- **EMBER** (Savage, April 2026, arXiv:2604.12167): A hybrid architecture with a 220K-neuron SNN (STDP, 4-layer hierarchy, E/I balance) as associative memory substrate + LLM as replaceable reasoning engine. The SNN autonomously triggers LLM actions via lateral STDP propagation.

**How HECSN-Warm differs from EMBER:**

| Feature | EMBER | HECSN-Warm |
|---|---|---|
| SNN architecture | 4-layer, flat hierarchy | 7-layer, deep hierarchy with bidirectional feedback |
| Binding topology | Dense (implied) | Dense, spatial, or hypercube (0.54% density, 36× memory reduction) |
| Cross-modal grounding | No (text embeddings only) | Yes (STDP across text↔visual↔audio) |
| Sleep consolidation | No | Yes (3-phase, fragility-gated, validated) |
| Self-criticism | No | Yes (blacklist-and-reset mechanism) |
| Curiosity controller | No | Yes (geometric gap detection) |
| Developmental stages | No | Yes (5-stage protocol) |
| Neuromodulation | Reward-modulated only | 4-channel (DA, 5-HT, ACh, NE) |
| Neuron model | LIF | AdEx/ALIF (default since v4.21) with adaptive timescales |
| Column sharding | No | Yes (4 shards@1024, 8@2048 for Terminus presets) |
| LLM bootstrap | Text embeddings encoded via z-score top-k | Embedding matrix → PCA → k-means → prototypes |
| Scale | 220K neurons | Validated at 100K columns (comparable) |
| Validated | 82.2% discrimination retention | 694+ tests, 50-triple grounding probe, 50K scale validated |

HECSN-Warm is a more biologically complete system than EMBER. EMBER validates the hybrid SNN-LLM concept; HECSN-Warm extends it with grounding, sleep, and developmental learning.

### 11.3 CPG-PE: Spiking Positional Encoding (relevant for encoding)

- **CPG-PE** (Lv et al., 2024, arXiv:2405.14362): Demonstrates that sinusoidal positional encoding is a specific solution to spiking neuron membrane dynamics. SNNs with CPG-PE outperform conventional SNNs across NLP, vision, and time-series tasks.

**Relevance to HECSN:** The RTF encoding (Layer 2) uses positional phase offsets that are conceptually related to CPG-PE. The CPG-PE framework could provide a principled replacement for HECSN's heuristic positional encoding, potentially improving the text encoding quality that feeds into the competitive layer.

---

## 12. Risks and Open Problems

### 12.1 Embedding Space Incompatibility (High Risk)

LLM embeddings are optimized for the transformer's self-attention mechanism. HECSN uses cosine-similarity WTA routing. These operate on different geometric properties of the embedding space:

- Transformers benefit from high-dimensional distributed representations where meaning is encoded in directions
- WTA routing benefits from well-separated clusters where a single winner can be clearly identified

**Mitigation:** Use k-means in Step 4 of the bootstrap (§4.1). K-means centroids are inherently well-separated. Also: monitor bootstrap_alignment_score and prototype drift rate.

### 12.2 Subword-to-Word Reconstruction (Medium Risk)

Modern LLMs use subword tokenizers (BPE, SentencePiece). HECSN operates at the character level (RTF encoding of byte windows). The granularity mismatch means:

- LLM has embeddings for "un", "##break", "##able" — not for "unbreakable"
- HECSN sees character sequences "u-n-b-r-e-a-k-a-b-l-e"

**Mitigation:** Average subword embeddings to reconstruct word-level embeddings. Or use word-level embedding models (fastText, GloVe) which avoid the problem entirely.

### 12.3 Over-Reliance on LLM Oracle (Medium Risk)

If the LLM oracle becomes the primary source of multimodal experiences, HECSN's autonomous learning claim weakens. The system becomes a distillation target rather than an independent learner.

**Mitigation:** Enforce a ratio: at most 30% of training experiences come from LLM-curated data. The remaining 70% must come from raw multimodal streams (video, audio, text from the environment). Stage 5 should use the LLM oracle only for gap concepts that the environmental stream cannot resolve.

### 12.4 Evaluation Circularity (Low Risk)

If the grounding probe uses word embeddings for its semantic triples (e.g., cosine similarity of routing keys), and the routing keys are influenced by LLM-seeded prototypes, the probe may overestimate quality.

**Mitigation:** The grounding probe (§8.10 of main paper) uses per-word sensory signatures, not routing keys alone. The representation is: `[routing_key × (1 - conf), visual_signature × conf, audio_signature × conf]`. As grounding confidence rises, the probe increasingly measures cross-modal quality, not text-statistical quality. This design already mitigates circularity.

---

## 13. References

References from the main paper are cited by number [N] as in HECSN_Paper_v4.md. New references specific to HECSN-Warm:

[W1] Bal, M. et al. (2023). SpikingBERT: Distilling BERT to Train Spiking Language Models Using Implicit Differentiation. *arXiv:2308.15122.*

[W2] Zhu, R.-J. et al. (2023). SpikeGPT: Generative Pre-trained Language Model with Spiking Neural Networks. *arXiv:2302.13939.*

[W3] Chen, L. et al. (2025). LAS: Loss-less ANN-SNN Conversion for Fully Spike-Driven Large Language Models. *arXiv:2505.xxxxx.*

[W4] Chen, L. et al. (2025). FAS: Fast ANN-SNN Conversion for Spiking Large Language Models. *arXiv:2502.xxxxx.*

[W5] Savage, W. (2026). EMBER: Autonomous Cognitive Behaviour from Learned Spiking Neural Network Dynamics in a Hybrid LLM Architecture. *arXiv:2604.12167.*

[W6] Lv, C. et al. (2024). CPG-PE: Central Pattern Generator Inspired Positional Encoding for Spiking Neural Networks. *arXiv:2405.14362.*

[W7] Tang, Z. (2026). NeuronSpark: A Spiking Neural Network Language Model with Selective State Space Dynamics. *arXiv:2603.16148.*

[W8] Pennington, J., Socher, R., & Manning, C. D. (2014). GloVe: Global Vectors for Word Representation. *EMNLP 2014.*

[W9] Bojanowski, P. et al. (2017). Enriching Word Vectors with Subword Information. *TACL*, 5, 135–146. [fastText]

[W10] Hu, Y. et al. (2024). Toward Large-scale Spiking Neural Networks: A Comprehensive Survey and Future Directions. *arXiv:2409.xxxxx.*

[W11] Wang, C. et al. (2026). Kirin: Improving ANN efficiency with SNN Hybridization. *arXiv:2602.xxxxx.*

[W12] Gemma Team (2026). Gemma 4 Technical Report. Google DeepMind. https://ai.google.dev/gemma/docs/core

[W13] Devvrit, F. et al. (2023). MatFormer: Nested Transformer for Elastic Inference. *arXiv:2310.07707.*

[W14] Qwen Team (2025). Qwen3 Technical Report. https://qwenlm.github.io/blog/qwen3/

[W15] Microsoft Research (2025). Phi-4 Technical Report. *arXiv:2503.01743.*

---

## Appendix A: Implementation Guide — Local LLM Selection

### A.1 Hardware Assumptions

This guide assumes the development hardware described in the main paper:

| Component | Specification |
|---|---|
| GPU | NVIDIA RTX 3060 12GB VRAM |
| CPU | Modern multi-core (HECSN STDP runs on CPU at current scale) |
| RAM | 16GB+ system memory |
| Storage | SSD (for model weights and PLE cache) |
| OS | Windows 11 (as per current development environment) |

### A.2 The Three-Role Framework

The LLM in HECSN-Warm serves three distinct roles. These can be filled by one model or split across specialized models:

```
┌─────────────────────────────────────────────────────────────────┐
│                    LLM ROLES IN HECSN-WARM                      │
│                                                                  │
│  ┌──────────────────┐  ┌──────────────────┐  ┌────────────────┐ │
│  │  ROLE 1:          │  │  ROLE 2:          │  │  ROLE 3:       │ │
│  │  Embedding Source │  │  Curiosity Oracle │  │  Grounding     │ │
│  │  (Bootstrap)      │  │  (Runtime)        │  │  Partner       │ │
│  │                   │  │                   │  │  (Runtime)     │ │
│  │  ONE-TIME         │  │  ON-DEMAND        │  │  CONTINUOUS    │ │
│  │  Extract embed    │  │  Gap concept →    │  │  HECSN sends   │ │
│  │  matrix, PCA,     │  │  LLM generates    │  │  grounding to  │ │
│  │  k-means → done   │  │  multimodal       │  │  LLM; LLM     │ │
│  │                   │  │  descriptions     │  │  sends context │ │
│  │  Needs: embed     │  │  Needs: inference │  │  back          │ │
│  │  weights only     │  │  + multimodal     │  │  Needs: infer  │ │
│  └──────────────────┘  └──────────────────┘  └────────────────┘ │
│                                                                  │
│  Best for Role 1:      Best for Roles 2+3:                       │
│  fastText / GloVe      Gemma 4 (multimodal)                       │
│  (word-level, instant) (text + vision + audio)                    │
└─────────────────────────────────────────────────────────────────┘
```

### A.3 Recommended Setup: Phased Approach

**Phase 1 — Validate the bootstrap concept (Week 1):**

Use fastText for bootstrap only. No oracle, no grounding partner. This isolates the bootstrap mechanism for testing.

```powershell
# Download fastText English embeddings
# From: https://fasttext.cc/docs/en/crawl-vectors.html
# File: cc.en.300.vec (text format, ~6.4GB disk, loads into ~4.5GB RAM)

pip install gensim  # For loading fastText/GloVe vectors
```

```python
# bootstrap_from_fasttext.py
import numpy as np
from gensim.models import KeyedVectors
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA

# Load fastText (word-level, 300d, 2M words)
ft = KeyedVectors.load_word2vec_format('cc.en.300.vec', limit=50000)
# limit=50000 → top 50K most frequent words

# Stack into matrix
words = list(ft.key_to_index.keys())
E = np.array([ft[w] for w in words])  # [50000, 300]

# PCA to 256 dimensions (300→256 is mild compression)
pca = PCA(n_components=256)
E_256 = pca.fit_transform(E)  # [50000, 256]
print(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.3f}")
# Expected: >0.95 (minimal information loss)

# L2 normalize (HECSN uses cosine similarity routing)
norms = np.linalg.norm(E_256, axis=1, keepdims=True)
E_256 = E_256 / norms

# K-means to select N_columns prototypes
N_COLUMNS = 256  # Match HECSN config
kmeans = MiniBatchKMeans(n_clusters=N_COLUMNS, random_state=42)
kmeans.fit(E_256)
prototypes = kmeans.cluster_centers_  # [256, 256]

# Save for HECSN injection
np.save('bootstrap_prototypes.npy', prototypes)
print(f"Bootstrap complete: {prototypes.shape} prototypes saved")

# Sanity check: are semantic neighbors preserved?
from sklearn.metrics.pairwise import cosine_similarity
dog_idx = words.index('dog')
cat_idx = words.index('cat')
car_idx = words.index('car')
dog_cat_sim = cosine_similarity(E_256[dog_idx:dog_idx+1], E_256[cat_idx:cat_idx+1])[0,0]
dog_car_sim = cosine_similarity(E_256[dog_idx:dog_idx+1], E_256[car_idx:car_idx+1])[0,0]
print(f"dog-cat similarity: {dog_cat_sim:.3f} (should be high)")
print(f"dog-car similarity: {dog_car_sim:.3f} (should be lower)")
```

**Phase 2 — Add curiosity oracle (Week 2-3):**

Install Gemma 4 E2B via Ollama. Wire into the curiosity controller.

```powershell
# Install Ollama (if not already)
winget install Ollama.Ollama

# Pull Gemma 4 E2B (lighter, 7.2GB)
ollama pull gemma4:e2b

# Verify it runs alongside HECSN
ollama run gemma4:e2b "Describe what fire looks like and sounds like in vivid sensory detail."
```

```python
# curiosity_oracle.py — integration with HECSN curiosity controller
import ollama

def query_curiosity_oracle(gap_concept: str, modalities: list[str] = ['visual', 'audio']) -> dict:
    """
    Query the LLM oracle to generate multimodal descriptions for a gap concept.
    Called by GeometricCuriosityController when grounding_confidence < threshold.
    """
    modality_prompts = {
        'visual': f"Describe what '{gap_concept}' looks like in vivid visual detail. "
                  f"Focus on colors, shapes, textures, movement patterns.",
        'audio':  f"Describe what '{gap_concept}' sounds like. "
                  f"Focus on pitch, rhythm, loudness, texture of the sound.",
    }

    results = {}
    for mod in modalities:
        response = ollama.chat(
            model='gemma4:e2b',
            messages=[{
                'role': 'system',
                'content': 'You are a perceptual description assistant. '
                           'Describe sensory experiences as vividly as possible. '
                           'Keep responses to 2-3 sentences.'
            }, {
                'role': 'user',
                'content': modality_prompts[mod]
            }]
        )
        results[mod] = response['message']['content']

    return results

# Example usage:
# gap = curiosity_controller.get_top_gap()  # e.g., "volcano"
# descriptions = query_curiosity_oracle("volcano")
# → {'visual': 'Glowing red-orange lava flows...', 'audio': 'Deep rumbling...'}
# → Feed descriptions to HECSN's multimodal pipeline alongside actual media
```

**Phase 3 — Upgrade to E4B + multimodal oracle (Week 4+):**

Once the pipeline is validated, upgrade to E4B and leverage its native image/audio understanding:

```powershell
# Upgrade to E4B
ollama pull gemma4:e4b            # 9.6GB download

# Now the oracle can actually process images and audio!
```

```python
# multimodal_oracle.py — Gemma 4 processes actual images/audio
import ollama

def oracle_verify_grounding(concept: str, image_path: str) -> dict:
    """
    Ask Gemma 4 to verify HECSN's learned visual grounding against
    an actual image. Part of the self-criticism loop enhancement.
    """
    response = ollama.chat(
        model='gemma4:e4b',
        messages=[{
            'role': 'user',
            'content': f"Look at this image. Does it depict '{concept}'? "
                       f"Describe what you see and rate confidence 0-10.",
            'images': [image_path]
        }]
    )
    return {
        'concept': concept,
        'llm_description': response['message']['content'],
        # Parse confidence from response for self-criticism integration
    }
```

### A.4 VRAM Coexistence Table

Both HECSN and the LLM run simultaneously on the RTX 3060. Here's the budget for each configuration:

| Configuration | HECSN VRAM | LLM VRAM | KV Cache | Total | Headroom |
|---|---|---|---|---|---|
| HECSN (256 col) + Gemma 4 E2B (Q4, PLE) | 0.01 GB | ~3.2 GB | ~1 GB | ~4.2 GB | **7.8 GB** |
| HECSN (256 col) + Gemma 4 E4B (Q4, PLE) | 0.01 GB | ~5 GB | ~1.5 GB | ~6.5 GB | **5.5 GB** |
| HECSN (100K col) + Gemma 4 E4B (Q4, PLE) | 0.1 GB | ~5 GB | ~1.5 GB | ~6.6 GB | **5.4 GB** |
| HECSN (100K col) + Qwen3 4B (Q4) | 0.1 GB | 2.5 GB | ~1 GB | ~3.6 GB | **8.4 GB** |
| HECSN (100K col) + Qwen3 8B (Q4) | 0.1 GB | 5.2 GB | ~1.5 GB | ~6.8 GB | **5.2 GB** |

All configurations fit comfortably. The PLE technology in Gemma 4 E2B/E4B is particularly advantageous because the large PLE parameters (~3.5B in E4B) are computed on CPU and streamed per-layer, never occupying GPU VRAM simultaneously.

### A.5 Why Not Larger Models?

Models >8B raw parameters (e.g., LLaMA 3 70B, Qwen3 32B, Mixtral) would provide richer embeddings and better oracle responses, but:

1. Don't fit in 12GB VRAM even at Q4 quantization
2. Inference latency becomes a bottleneck for the curiosity loop
3. The bootstrap only uses the embedding matrix — a 70B model's embeddings are only marginally better than an 8B model's for distributional word relationships
4. Gemma 4's multimodal capability is more valuable than a larger text-only model's reasoning

If cloud GPU access is available for the one-time bootstrap, extracting embeddings from a larger model (e.g., LLaMA 3.1 70B) and then running Gemma 4 E4B locally for the oracle is a valid hybrid strategy.

### A.6 Decision Flowchart

```
START: Which LLM setup for HECSN-Warm?
│
├── Q: Do you have a GPU with ≥8GB VRAM?
│   ├── YES → Q: Do you need multimodal oracle (image+audio)?
│   │   ├── YES → ★ Gemma 4 E4B (recommended)
│   │   │         Bootstrap: fastText (Phase 1) → Gemma 4 embeds (Phase 3)
│   │   │         Oracle: Gemma 4 E4B via Ollama
│   │   │         Install: ollama pull gemma4:e4b
│   │   │
│   │   └── NO (text experiments only) → Qwen3 4B
│   │             Bootstrap: fastText or Qwen3 embeds
│   │             Oracle: Qwen3 4B via Ollama
│   │             Install: ollama pull qwen3:4b
│   │
│   └── NO (CPU only or <8GB) → Q: VRAM ≥4GB?
│       ├── YES → Gemma 4 E2B (~3.2GB effective VRAM)
│       │         Install: ollama pull gemma4:e2b
│       │
│       └── NO → fastText bootstrap only (no runtime oracle)
│                 Defer oracle to cloud API when needed
│
└── Final: Validate with bootstrap_alignment_score after 10K tokens
```

---

## Changelog

**v0.4 (2026-04-16):** Synchronized with main paper **v4.21**. Updated throughput benchmarks (~57 tok/s full architecture, +181% from baseline). Added v4.19 optimization details. Binding layer now lists 3 modes (dense/spatial/hypercube). AdEx spike backend noted as default. Column sharding added to EMBER comparison. Updated test count (694+). Removed all Gemma 3/3n historical references.

**v0.5 (2026-07-17):** **Semantic n-gram encoder (Phase 2 of Warm Companion).** The v0.4 prototype bootstrap (Mechanism 1) was disproved — GloVe neighborhood overlap with HECSN routing = 0.024 ≈ random chance because RTF routing is character-based (see main paper §10.6). Phase 2 addressed the root cause by replacing ASCII-position encoding entirely with `SemanticEncoder`: character n-grams → FNV-1a hash → GloVe-initialized bucket embeddings → split-sign → top-k sparsification. **Critical finding: top-k=8 sparsification is essential** — without it, smooth mean-pooled embeddings produce 58% dead columns (worse than RTF). With k=8: 26% dead columns (vs 32% RTF), 0.757 diversity (vs 0.711), 10.7% mean pairwise cosine (vs 58.7%). Throughput comparable (~117 tok/s). Grounding probe inconclusive at small scale (both encodings near random without multimodal training). See main paper §10.7. The encoder factory (`build_encoder`) and BaseEncoder protocol enable seamless switching between RTF and semantic modes. 35 new unit tests, all passing.

**v0.3 (2026-04-16):** Updated all LLM references to **Gemma 4** (released April 2026). Added Gemma 4 26B MoE and 31B Dense to comparison table (don't fit 12GB). Updated VRAM figures, Ollama commands, code snippets, and decision flowchart. Primary recommendation: **Gemma 4 E4B** for RTX 3060 12GB.

**v0.2 (2026-04-16):** Added Appendix A: Implementation Guide — Local LLM Selection. Comprehensive analysis of Qwen3, Phi-4-mini, fastText, and GloVe for local deployment on RTX 3060 12GB. Two-model strategy documented (fastText bootstrap + Gemma oracle). Phased implementation guide with code snippets. VRAM coexistence table. §4.2 expanded from 5-row table to full analysis with installation instructions. References [W12]–[W15] added.

**v0.1 (2026-04-16):** Initial companion paper. Describes three integration mechanisms (bootstrap, bidirectional grounding, curiosity oracle), reframes emergence claim, specifies evaluation protocol, surveys related work.

---

*Central thesis: LLMs provide the vocabulary of experience. HECSN provides the understanding. Neither alone solves the grounding problem. Together, they create a system that knows what words mean — not just how they relate to each other.*
