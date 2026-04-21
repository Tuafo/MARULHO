# Terminus
## A Cortex–Subcortex Architecture Built on a Grounded Predictive Concept Spiking Network

**Author:** Thiago Maceno Rocha Goulart · Brasil · [github.com/Tuafo](https://github.com/Tuafo)

**Updated:** 2026-04-21

---

## Abstract

Terminus is a hybrid cognitive architecture that combines a predictive spiking substrate with a cloud-hosted language cortex. The underlying spiking system — described here as a **Grounded Predictive Concept Spiking Network (GPCSN)** — is responsible for sparse routing, multimodal grounding, predictive error, neuromodulation, memory replay, and curiosity-driven pressure over future cognition. The cortical layer is responsible for language, reasoning, question formation, narrative continuity, answer generation, and dream-style hypothesis composition. The central claim of the architecture is not that spikes should replace large language models, but that a grounded predictive spiking system can provide the functions that large language models lack in isolation: continuous multimodal grounding, local online adaptation, replay-driven stabilization, and persistent uncertainty pressure.

The current runtime is strictly NVIDIA NIM-only. Local LLM backends are no longer part of the production path. The cortex now includes working memory, narrative self, compositional dreams, active exploration, prediction-error-aware depth selection, richer neuromodulation channels, and shared request-budget control across both chat and embedding endpoints. The GPCSN side includes predictive columns, awake ripple tagging, hub-aware binding, multimodal grounding, and curriculum-oriented concept pressure. Current validation confirms that the architecture is executable and coherent: the developmental spiking pipeline remains strong, the hybrid cortex stack is stable under real NIM calls, and recent real-runtime validations show meaningful topic diversity under a constrained API budget. The system remains incomplete: dream verification is weak in short runs, output quality still drifts, and broader multimodal training sources are still needed. Even so, the present implementation is now best described as a functioning cortex–subcortex architecture rather than a collection of disconnected mechanisms.

**Keywords:** hybrid cognitive architectures, spiking neural networks, grounded cognition, predictive processing, multimodal learning, memory consolidation, autonomous agents

---

## 1. Introduction

The present generation of large language models is strong at language production, latent world knowledge, and broad reasoning over text. Yet they remain weak in several areas that matter for persistent autonomous cognition:

- they do not natively maintain grounded multimodal bindings through local learning
- they do not naturally regulate cognition through predictive error, fatigue, boredom, and novelty pressure
- they do not perform sleep-like replay and consolidation in a biologically meaningful sense
- they do not preserve explicit provenance boundaries between observation, inference, dream, contradiction, and verification
- they do not choose future topics under a persistent, embodied-like uncertainty economy without substantial scaffolding

Biologically-inspired spiking systems have complementary strengths. They support local plasticity, temporal continuity, novelty response, sparse routing, replay, and explicit control signals. Their weakness is expressive symbolic and linguistic reasoning. Terminus is built on the premise that these two systems should not be forced into the same role.

The core architectural claim is therefore:

> **Use the spiking system as the grounded predictive subcortex and the language model as the expressive cortex.**

Under this design, the spiking substrate decides:
- when the cortex should think
- how deeply it should think
- which memories and tensions should enter the current context window
- what uncertainty should be explored next

The cortex decides:
- how to transform that state into language, explanation, hypotheses, narrative continuity, and answers.

---

## 2. Design Principles

Terminus is organized around five design principles.

### 2.1 Grounding is not optional
The architecture does not treat language-only competence as sufficient for cognition. The spiking substrate is expected to learn cross-modal structure and to supply grounded pressure to the cortex.

### 2.2 Prediction error is a first-class signal
Surprise and predictive mismatch are not auxiliary metrics. They are part of the system’s control economy, influencing learning pressure, thought depth, curiosity, and future topic choice.

### 2.3 Memory must have multiple forms
The architecture now distinguishes replay memory, episodic memory, working memory, narrative self, and dream-origin hypotheses. This multi-form memory design is essential for stable cognition.

### 2.4 The cortex must be gated, not free-running
The language layer does not think arbitrarily. It is triggered and shaped by subcortical pressure, uncertainty, and externally requested goals.

### 2.5 Runtime budgets are part of the scientific design
Because the cortex depends on external inference, API budgets materially shape the architecture. Request limits affect thought depth, sleep-phase design, chain length, and routing policy.

---

## 3. System Overview

### 3.1 High-level architecture

```text
┌──────────────────────────────────────────────────────────────────────────┐
│                               TERMINUS                                  │
│                                                                          │
│  Cortex: fast/deep NVIDIA NIM routing                                   │
│    ├─ ThoughtLoop                                                       │
│    ├─ DriveSystem + ThalamicGate                                        │
│    ├─ WorkingMemory + NarrativeSelf                                     │
│    ├─ EpisodicMemory                                                    │
│    └─ Dream compose/test + active exploration                           │
│                                                                          │
└───────────────────────────────▲──────────────────────────────────────────┘
                                │
                                │ prediction error, concepts, memory
                                │ pressure, active topics, curriculum
                                │
┌───────────────────────────────┴──────────────────────────────────────────┐
│                    GPCSN SUBCORTICAL SUBSTRATE                           │
│                                                                          │
│  Encoders → Competitive Columns → Adaptive Context → Hypercube Binding   │
│           → Abstraction → Cross-Modal Grounding → Surprise Monitor       │
│           → Dual Memory Store → Predictive Columns → Concept Pressure    │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Division of labor

| Subsystem | Main responsibilities |
|---|---|
| **GPCSN subcortex** | sensory transformation, sparse routing, local plasticity, predictive error, novelty, replay, grounding, concept pressure |
| **Cortex** | language, synthesis, question formation, answer generation, narrative continuity, dream recombination |

This division is the clearest way to understand the present system.

---

## 4. The Grounded Predictive Concept Spiking Network

The spiking core is best described as a **Grounded Predictive Concept Spiking Network**.

### 4.1 Main components

| Component | Current role |
|---|---|
| **CompetitiveColumnLayer** | sparse competitive routing and representation separation |
| **AdaptiveContextLayer** | temporal context with learnable timescale behavior |
| **HypercubeBindingLayer** | sparse structural binding with hub tracking and hub boost |
| **AbstractionLayer** | slow-feature abstraction and concept support |
| **CrossModalGroundingLayer** | text–vision and text–audio alignment via local plasticity |
| **SurpriseMonitor** | dopamine / serotonin / norepinephrine / acetylcholine mirrors |
| **DualMemoryStore** | replay-capable reservoir + consolidation memory |
| **PredictiveColumnState** | prediction error, local location-like state, and consensus support |
| **ConceptStore / Curiosity controller** | weak-concept detection and curriculum targeting |

### 4.2 Implemented upgrades in the current system

| Upgrade | Function |
|---|---|
| **Awake ripple tagging** | tags recent salient memories for replay priority [6] |
| **Hub-aware hypercube binding** | increases influence of structurally important high-usage nodes |
| **Predictive columns** | adds local prediction state and consensus signals inspired by reference-frame theories [5] |
| **Prediction-error-aware modulation** | adjusts STDP and context behavior using predictive mismatch |
| **Multimodal default embedding alignment** | makes memory indexing consistent with the NIM multimodal embedding model |

### 4.3 What the spiking substrate is now for

The spiking system should not be understood as “a worse text learner than the LLM.” It is now specialized for:
- multimodal grounding
- novelty and salience extraction
- local online adaptation
- replay and consolidation
- curiosity pressure over future cognition
- predictive control of cortical depth and topic selection

---

## 5. The Terminus Cortex

### 5.1 Model routing

| Role | Model |
|---|---|
| Fast cortex | `nvidia/llama-3.1-nemotron-nano-8b-v1` |
| Deep cortex | `meta/llama-3.3-70b-instruct` |
| Embedder | `nvidia/llama-nemotron-embed-vl-1b-v2` |

### 5.2 Routing policy

| Situation | Route |
|---|---|
| routine wake thought | fast model |
| reflect | fast model |
| answer / generic dream | deep model if healthy, else fast |
| chain continuation (`question`, `reason`, `synthesize`) | fast model only |
| dream compose / dream test | fast model only |

This policy is explicitly budget-aware and prompt-obedience-aware.

### 5.3 Main cortical modules

| Component | Current role |
|---|---|
| **MultiCortex** | backend routing and health-aware model selection |
| **ThoughtLoop** | continuous cognition, sleep, thought-depth control |
| **DriveSystem** | converts predictive error and surprise into cognitive pressure |
| **ThalamicGate** | assembles budgeted context packets |
| **EpisodicMemory** | provenance-aware similarity memory |
| **WorkingMemory** | chain-local global workspace |
| **NarrativeSelf** | persistent autobiographical summary |
| **Dream compose / dream test** | sleep-time hypothesis generation and validation |
| **Active exploration** | next-topic selection under uncertainty |

### 5.4 Cortex-side innovations currently implemented

| Feature | Current function |
|---|---|
| **Working memory** | maintains within-chain continuity |
| **Narrative self** | provides cross-session continuity without dominating wakeful chains |
| **Thought depth selection** | chooses between quick, standard, and deep deliberation |
| **Compositional dreams** | generates lineage-tracked hypotheses from replayed memories |
| **Prediction-error-aware drives** | turns uncertainty into curiosity, anxiety, and exploration pressure |
| **Richer neuromodulation channels** | derives reward/novelty/salience/attention-like channels from scalar monitors |
| **Shared budget control** | limits chat and embedding traffic together |
| **Statement-oriented merge logic** | reduces question spam and repeated phrase echo |
| **Evidence-aware dream verdict parsing** | combines explicit verdict markers with lexical evidence support |
| **Exploration-target filtering** | penalizes weak fragmentary SNN labels using lexical quality and recent-memory grounding |

---

## 6. Memory Architecture

A major advance in the current system is the explicit separation of memory forms.

### 6.1 SNN-side memory
The spiking side maintains replay-capable memory and consolidation state. Awake ripple tagging adds a second layer of replay prioritization.

### 6.2 Episodic memory
Episodic memory carries provenance tags:
- `observed`
- `inferred`
- `dreamed`
- `verified`
- `contradicted`

This supports safer reasoning because the system does not have to treat all remembered text as equivalent.

### 6.3 Working memory
Working memory is a chain-local workspace. It is not a raw transcript of recent thought. Instead, it holds the active observation, question, contradiction, or insight across a single deliberation chain.

### 6.4 Narrative self
Narrative self is a persistent compressed account of what the system has been exploring across time. It supports answer, reflection, and dream contexts.

### 6.5 Dream lineage
Dream-origin episodes remain marked even if they are later verified or contradicted. This is important for auditability and for computing dream verification behavior.

### 6.6 Memory hierarchy overview

```text
SNN replay/consolidation memory
        ↓
Cortex episodic memory (provenance-aware)
        ↓
Working memory (active chain-local workspace)
        ↓
Narrative self (cross-session compressed continuity)
```

---

## 7. Cognitive Operation

### 7.1 Deliberation depths

| Depth | Structure | Intended role |
|---|---|---|
| `quick` | one call | rapid factual scan / pattern response |
| `standard` | observe → question | compact uncertainty formation |
| `deep` | observe → question → reason → synthesize | controlled multi-step reasoning |

### 7.2 Active exploration

Active exploration is currently selected from:
- working-memory questions
- working-memory tensions
- wake tensions from dreams
- low-confidence recent thought topics
- concept candidates under predictive mismatch

Recent improvements emphasize:
- linguistic quality
- lexical grounding against recent memory
- priority for unresolved working-memory questions

### 7.3 Sleep and dreams

A current sleep cycle:
1. retrieves replay-worthy episodes
2. forms a cross-memory dream hypothesis
3. validates that hypothesis against a broader evidence set
4. marks it as supported, unresolved, or contradicted
5. converts unresolved contradictions into future wake tensions

This is one of the strongest pieces of the present architecture because it gives sleep a real computational role.

---

## 8. Runtime Constraints

### 8.1 Strict NIM-only startup
The runtime now fails clearly if NIM is unavailable.

- no production local-LLM path
- no silent runtime mock cortex
- no silent embedder downgrade by default

### 8.2 Shared request budget
NVIDIA NIM free tier allows **40 requests/minute** per key. Terminus now uses a **shared 20 RPM limiter** across:
- chat completions
- embeddings

This matters because both endpoints draw from the same real external budget.

### 8.3 Why the budget matters scientifically
The budget constrains:
- the frequency of deeper thought chains
- how aggressively the cortex can embed and recall memory
- how often dreams can be tested in practice
- how the architecture balances quality and quantity of cognition

---

## 9. Training Strategy and Data

### 9.1 Current recommended strategy
The present system no longer treats raw Wikipedia-style text as the preferred path. The language cortex already brings broad text priors; the spiking system should focus on what the cortex lacks.

### 9.2 Current recommended presets

| Preset | Function |
|---|---|
| `curriculum` | default text + NIM-guided curriculum |
| `multimodal` | curriculum + N-MNIST + FSDD |
| `multimodal_fast` | faster multimodal configuration |
| `wikipedia` | legacy compatibility only |

### 9.3 Current data sources
- **AG News** for broad topic seeding
- **N-MNIST** for event-based visual grounding
- **FSDD** for spoken-digit grounding
- **NIM-generated curriculum segments** for targeted text episodes

### 9.4 Remaining problem
These datasets are still too narrow for broad grounded cognition. The current architecture is ready for richer episodes, but the data regime needs to grow.

---

## 10. Current Findings

### 10.1 Historical validated findings for the spiking core
The spiking developmental results remain important:

| Validation | Current status |
|---|---|
| 5-stage developmental protocol | validated end-to-end |
| Grounding probe accuracy | approximately **0.64–0.68** |
| FastText / SOM baselines | approximately **0.46** |
| Real multimodal 100K-step run | validated at roughly **31.2 steps/s** |
| Large text-scale run | validated up to **1M tokens** |

These findings support the claim that the spiking core is not merely decorative.

### 10.2 Current hybrid validation surface

| Metric | Current value |
|---|---|
| Passed tests | **911** |
| Skipped | **3** |
| Deselected | **1** |
| Warnings | **1** |
| Subtests passed | **7** |

The single deselected item is the known stochastic learned-chunking flake.

### 10.3 Recent hybrid-system confirmations
Recent validation confirms:
- strict NIM startup
- shared chat+embedding limiting
- working-memory chain continuity
- persistent narrative self
- explicit dream verdict parsing
- exploration-target filtering by lexical quality and memory grounding
- UTF-8-safe report generation
- corrected final-sample timing in long-test validation

### 10.4 Recent real-runtime behavior
Recent short real-NIM runs show:
- topic diversity typically in the range of roughly **3.0–4.7 topics per thought**
- average latency in short smoke tests typically around **5–7 seconds**
- healthy embedder telemetry with no fallback and no observed rate-limit hits in the latest smoke runs
- exploration targets such as **origami mathematics** and **biotechnology** appearing where earlier fragmentary targets frequently dominated

---

## 11. Related Work

Terminus sits at the intersection of several research traditions rather than cleanly inside one.

### 11.1 Related-work summary

| Tradition / system | Main idea | Relation to Terminus |
|---|---|---|
| **Working memory / global workspace** [1,2,3] | cognition depends on globally available active content | motivates working memory and broadcasted chain state |
| **Predictive processing / active inference** [4] | prediction error governs perception and action | motivates predictive columns, uncertainty-driven depth, and exploration |
| **Thousand Brains / reference-frame cognition** [5] | cortical-style modules track local frames and consensus | motivates predictive columns and consensus-style integration |
| **Sharp-wave ripple memory selection** [6] | salient experience is tagged for replay | motivates awake ripple tagging |
| **Small-world spiking systems** [7] | sparse graph structure with hubs improves integration | motivates hub-aware binding behavior |
| **Narrative identity / self-modeling** [8] | persistent selfhood is story-like, not raw state-like | motivates narrative self rather than raw thread replay |
| **Hybrid SNN–LLM systems such as EMBER** [9] | spiking systems can trigger and shape external language modules | provides a nearby hybrid comparison point |
| **Thousand Brains Project / Monty** [10] | object-centric, reference-frame, sensorimotor modeling | closest conceptual analogue for predictive modularity |

### 11.2 How Terminus differs

Terminus differs from ordinary LLM agent wrappers because it is not built primarily around tools, planners, or prompt loops. Its central control variables come from a continuously learning predictive spiking substrate.

It differs from standalone SNN cognitive proposals because it does not require the spiking substrate to perform the full burden of language competence.

It differs from purely symbolic cognitive architectures because the key control pressures — predictive error, novelty, replay, fatigue, grounding, and multi-timescale memory — arise from learned dynamical state rather than a hand-authored symbolic controller.

---

## 12. Limitations and Open Problems

The current architecture is substantially stronger than earlier versions, but it still has serious limitations.

### 12.1 Dream verification is still weak
The dream architecture is now structurally meaningful, but short runs still show low support rates.

### 12.2 Thought quality is still uneven
Even after recent improvements, some outputs remain awkward, generic, or overly question-heavy.

### 12.3 Multimodal breadth remains narrow
The benchmark datasets are enough to validate mechanisms, not enough to support broad claims of grounded cognition in the wild.

### 12.4 Affect is still simplified
The richer neuromodulation channels are useful, but they are not yet a fully differentiated affective or motivational model.

### 12.5 One stochastic surface failure remains
The learned-chunking flake remains a known issue on the spiking side.

---

## 13. Terminology Used in This Paper

This paper uses the following names:
- **Terminus** — the full cortex–subcortex architecture
- **Grounded Predictive Concept Spiking Network (GPCSN)** — the spiking substrate
- **cortex** — the NIM-based language-and-reasoning layer
- **subcortex** — the predictive spiking grounding and control layer

These names are chosen because they describe the current architecture more accurately than older labels centered on generic hierarchy or emergence.

---

## 14. Conclusion

Terminus has evolved into a clear cortex–subcortex architecture rather than remaining a standalone concept-learning SNN or a shallow LLM wrapper. The spiking substrate now supplies grounding, prediction, salience, replay, and curiosity pressure, while the cortical layer supplies explicit language, synthesis, questions, answers, and dream-like recombination. The result is not a finished artificial mind, but it is now a coherent and defensible research architecture for grounded autonomous cognition.

---

## References

[1] A. Baddeley. Working memory and the episodic buffer.

[2] B. Baars. *A Cognitive Theory of Consciousness*. 1988.

[3] S. Dehaene and J.-P. Changeux. Experimental and theoretical approaches to conscious processing and global workspace dynamics.

[4] K. Friston. The free-energy principle: a unified brain theory? 2010.

[5] J. Hawkins, S. Lewis, M. Klukas, S. Purdy, and S. Ahmad. A framework for intelligence and cortical function based on grid cells in the neocortex. 2019.

[6] W. Yang and G. Buzsáki. Selection of experience for memory by hippocampal sharp wave ripples. 2024.

[7] W. Pan et al. Emergence of brain-inspired small-world spiking neural network through neuroevolution. 2024.

[8] D. McAdams and K. McLean. Narrative identity.

[9] W. Savage. EMBER: autonomous cognitive behaviour from learned spiking neural network dynamics in a hybrid LLM architecture. 2026.

[10] Thousand Brains Project / Monty documentation and codebase.
