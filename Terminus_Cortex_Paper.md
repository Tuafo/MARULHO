# Terminus Cortex
## Architecture, Functionality, Related Work, and Current Findings

**Author:** Thiago Maceno Rocha Goulart · Brasil · [github.com/Tuafo](https://github.com/Tuafo)

**Updated:** 2026-04-21

---

## Abstract

Terminus Cortex is the language-and-reasoning layer of a hybrid cortex–subcortex cognitive architecture. A predictive spiking substrate supplies multimodal grounding, predictive error, novelty, fatigue, replay pressure, and exploration pressure, while the cortex converts that state into language, reasoning, narrative continuity, question formation, dream-style recombination, and answer synthesis. The current cortex runtime is strictly NVIDIA NIM-only, budgeted under a shared limiter for both chat and embeddings, and organized around a `ThoughtLoop` that manages wake cognition, sleep, working memory, narrative self, and active exploration. Recent validation shows that the cortex stack is materially real rather than a thin prompt wrapper: it now has multiple memory forms, depth-aware deliberation, explicit dream verdict handling, shared API-budget control, and improved exploration-target filtering. The architecture remains incomplete: dream verification is weak in short runs and some wake outputs remain awkward. Even so, Terminus Cortex is now a credible platform for studying language-level cognition under grounded predictive control.

---

## 1. Design Goal

Terminus Cortex exists to solve a specific problem: large language models are good at language and broad reasoning, but poor at grounded ongoing cognition when left alone. They need a control substrate that can decide:
- when thought should happen
- what the next important topic is
- whether the system should stay shallow or reason more deeply
- which memories matter right now
- which contradictions should persist into the future

The spiking substrate supplies that pressure. Terminus Cortex supplies the expressive machinery.

---

## 2. Architectural Overview

```text
┌──────────────────────────────────────────────────────────────────────────┐
│                            TERMINUS CORTEX                              │
│                                                                          │
│  MultiCortex                                                             │
│    ├─ fast model                                                         │
│    ├─ deep model                                                         │
│    └─ shared-rate-limited embedder                                       │
│                                                                          │
│  ThoughtLoop                                                             │
│    ├─ DriveSystem                                                        │
│    ├─ ThalamicGate                                                       │
│    ├─ EpisodicMemory                                                     │
│    ├─ WorkingMemory                                                      │
│    ├─ NarrativeSelf                                                      │
│    └─ Dream compose/test + active exploration                            │
│                                                                          │
└───────────────────────────────▲──────────────────────────────────────────┘
                                │
                                │ predictive error, concepts, replay,
                                │ neuromodulators, curriculum pressure
                                │
┌───────────────────────────────┴──────────────────────────────────────────┐
│                    PREDICTIVE SPIKING SUBSTRATE                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### Cortex module summary

| Module | Main function |
|---|---|
| **MultiCortex** | fast/deep backend routing |
| **ThoughtLoop** | continuous cognition and sleep control |
| **DriveSystem** | converts predictive error and surprise into cognitive pressure |
| **ThalamicGate** | builds budgeted context packets |
| **EpisodicMemory** | provenance-aware memory retrieval |
| **WorkingMemory** | chain-local global workspace |
| **NarrativeSelf** | cross-session autobiographical continuity |
| **Dream compose/test** | sleep-time hypothesis generation and validation |
| **Active exploration** | next-topic selection under uncertainty |

---

## 3. Runtime Stack

### 3.1 Current backend models

| Role | Model |
|---|---|
| Fast cortex | `nvidia/llama-3.1-nemotron-nano-8b-v1` |
| Deep cortex | `meta/llama-3.3-70b-instruct` |
| Embedder | `nvidia/llama-nemotron-embed-vl-1b-v2` |

### 3.2 Routing policy

| Situation | Route |
|---|---|
| routine wake thought | fast model |
| reflection | fast model |
| answer / generic dream | deep model if healthy, else fast |
| `question`, `reason`, `synthesize` | fast model only |
| `dream_compose`, `dream_test` | fast model only |

This policy is intentionally shaped by budget, obedience, and latency considerations.

### 3.3 Strict runtime behavior

Current runtime guarantees:
- no production local-LLM path
- no silent runtime mock cortex
- no silent embedder downgrade by default
- strict startup failure when NIM is unavailable
- shared request budgeting across chat and embedding endpoints

---

## 4. ThoughtLoop and Deliberation

The central runtime process is `ThoughtLoop`.

### 4.1 Thought depths

| Depth | Structure | Purpose |
|---|---|---|
| `quick` | one call | rapid factual scan |
| `standard` | observe → question | compact uncertainty formation |
| `deep` | observe → question → reason → synthesize | multi-step deliberate reasoning |

### 4.2 Depth selection signals
Thought depth is selected from:
- prediction error mean/max
- predictive confidence
- external query presence
- working-memory questions and tensions
- norepinephrine-like alerting and salience-like signals
- cooldown logic required by the runtime budget

### 4.3 Short-chain merge behavior
Recent work changed how short chains are surfaced. Instead of emitting raw question spam, short observe/question chains are merged into more declarative uncertainty statements with overlap suppression.

This reduced one major failure mode of early hybrid runs.

---

## 5. Drive System and Thalamic Gating

### 5.1 Core drive variables
The current drive state includes:
- curiosity
- anxiety
- boredom
- fatigue
- prediction error
- uncertainty
- exploration urgency

### 5.2 Neuromodulation

#### Raw scalar mirrors
- dopamine
- serotonin
- norepinephrine
- acetylcholine

#### Derived channels
- `da_reward`
- `da_novelty`
- `da_salience`
- `ne_alerting`
- `ne_orienting`
- `ach_learning`
- `ach_attention`
- `serotonin_patience`

These channels affect:
- whether the system should think
- how deep the next thought should be
- which exploration target should dominate
- whether the system should remain in wake mode or drift toward sleep

### 5.3 ThalamicGate
The gate decides what the cortex sees.

It selects from:
- episodic memories
- working-memory narrative
- narrative self
- active exploration target
- external query
- anti-rumination avoidance terms

This is the most important “attention interface” in the current system.

---

## 6. Memory Hierarchy

### 6.1 Episodic memory
Episodes are stored with provenance:
- `observed`
- `inferred`
- `dreamed`
- `verified`
- `contradicted`

This enables safer cognition than a flat untyped memory store.

### 6.2 Working memory
Working memory is a short-lived global workspace that preserves continuity within a deliberation chain.

It is used to hold:
- current observation
- current question
- unresolved contradiction
- emerging insight

### 6.3 Narrative self
Narrative self is a compressed content-level summary across runs. It is most useful in answer, reflection, and dream contexts.

### 6.4 Memory diagram

```text
Recent spiking pressure / replay
           ↓
Episodic memory (typed, provenance-aware)
           ↓
Working memory (active chain-local content)
           ↓
Narrative self (cross-session continuity)
```

---

## 7. Sleep, Dreams, and Validation

Sleep is not decorative in Terminus Cortex.

### 7.1 Dream cycle
1. retrieve replay-worthy episodes
2. compose a cross-memory hypothesis
3. validate the hypothesis against a broader evidence set
4. assign a verdict
5. turn unresolved contradictions into wake tensions

### 7.2 Current verdict channel
The current dream validator expects explicit prefixes:
- `SUPPORTED:`
- `UNRESOLVED:`
- `CONTRADICTED:`

Verdicts are then interpreted using:
- explicit prefix
- confidence
- lexical evidence support

This makes dream evaluation more inspectable and less brittle than confidence-only logic.

---

## 8. Active Exploration

Active exploration is the current bridge from uncertainty to future thought.

### 8.1 Candidate sources
- working-memory questions
- working-memory tensions
- wake tensions from dream contradiction
- low-confidence recent cortex topics
- SNN concept candidates associated with prediction error

### 8.2 Recent improvements
Recent filtering now uses:
- lexical quality
- recent-memory grounding
- penalties for weak fragmentary labels
- stronger priority for unresolved working-memory questions

This improved the quality of some exploration targets, though the issue is not fully solved.

### 8.3 Why this matters
The selected exploration target strongly influences the quality of the next thought. In practice, improving target quality has had outsized effect on overall output quality.

---

## 9. Budget and Runtime Constraints

### 9.1 Shared request budget
NVIDIA NIM free tier allows **40 requests/minute** per key.

Terminus currently uses a shared **20 RPM** budget across:
- chat completions
- embeddings

### 9.2 Why the shared limiter matters
Without a shared limiter, the cortex could appear stable while the embedder silently consumed the remaining request budget. The current design prevents that mismatch.

### 9.3 Practical consequence
Budgeting shapes:
- how often deeper chains can be run
- how aggressively memory can be embedded and recalled
- how often dream validation can happen in short windows

Budget is therefore not just an implementation detail. It is part of the architecture.

---

## 10. Current Functionality

### Wake cognition
- generate factual thoughts
- express uncertainty declaratively
- choose thought depth under pressure
- choose exploration targets
- feed topics/confidence back into curiosity control

### Answer mode
- accept external queries asynchronously
- answer using memory and narrative context
- preserve cortex–subcortex separation

### Reflection mode
- engage when boredom/anxiety/social conditions call for it
- operate under the same budget-aware routing system

### Sleep mode
- replay memory
- form dream hypotheses
- validate them
- return unresolved contradictions as future wake pressure

### Telemetry
The cortex exposes:
- drive state
- neuromodulation state
- cognitive signals
- depth policy
- active exploration target/reason
- episodic-memory embedder health
- long-test JSON and markdown reports

---

## 11. Current Findings

### 11.1 Stable executable surface
Current stable validation state:
- **911 passed**
- **3 skipped**
- **1 deselected**
- **1 warning**
- **7 subtests passed**

### 11.2 What recent testing confirms
Recent hybrid validation confirms:
- strict NIM startup behavior
- no silent local-runtime fallback
- shared chat+embedding limiting
- working-memory continuity
- narrative self persistence
- explicit dream verdict handling
- lexical/memory-grounded exploration-target filtering
- UTF-8-safe report writing
- corrected final-sample timing in long-test reporting

### 11.3 Representative current runtime behavior
Recent short real-NIM runs show:
- meaningful topic diversity in short windows
- healthy embedder telemetry with no fallback and no observed rate-limit hits in the latest smoke validations
- exploration targets such as `origami mathematics` and `biotechnology` appearing where earlier fragmentary targets often dominated

### 11.4 Main findings

| Finding | Interpretation |
|---|---|
| Working memory materially improved continuity | the cortex behaves less like a one-shot prompt wrapper |
| Narrative self is useful only when scoped carefully | unrestricted self-context easily reintroduces fixation loops |
| Dream lineage matters | sleep becomes auditable instead of opaque |
| Exploration target quality is a major output lever | poor targets produce poor thoughts |
| Shared budgeting improves runtime honesty | chat and embeddings must be treated as one resource |

---

## 12. Related Work

### 12.1 Positioning
Terminus Cortex sits between several traditions:
- LLM agent wrappers
- hybrid SNN–LLM systems
- predictive-processing architectures
- working-memory and global-workspace theories
- narrative-self and autobiographical continuity models

### 12.2 Related-work table

| Related area | Core idea | Relation to Terminus Cortex |
|---|---|---|
| **Global workspace / working memory** [1,2,3] | cognition depends on globally available active content | motivates working memory and gated broadcast |
| **Predictive processing** [4] | error drives future inference and action | motivates prediction-error-aware depth and exploration |
| **Narrative identity** [5] | selfhood is story-like continuity, not raw state | motivates narrative self |
| **Thousand Brains / reference-frame cognition** [6] | modular predictive units maintain local hypotheses | motivates predictive columns and consensus influence |
| **Sharp-wave ripple memory selection** [7] | salient experience is selected for replay | motivates awake ripple tagging and replay priority |
| **Small-world SNNs** [8] | sparse structured connectivity improves integration | motivates hub-aware binding |
| **Hybrid SNN–LLM systems such as EMBER** [9] | spiking systems can govern when LLM-like systems act | nearest modern hybrid comparison |
| **Monty / Thousand Brains Project** [10] | reference-frame structured perception and learning | conceptual neighbour on predictive modularity |

### 12.3 Main difference from standard LLM wrappers
Most LLM agent systems wrap the model with tools, plans, and prompt state. Terminus Cortex is different because its control pressure comes from a continuously learning predictive substrate rather than from only symbolic orchestration.

---

## 13. Limitations

### 13.1 Dream verification remains weak
The mechanism exists, but support rates are still low in short validations.

### 13.2 Some wake outputs remain awkward
Recent improvements reduced repetition and poor exploration targets, but output quality is not yet consistently strong.

### 13.3 Multimodal breadth is still narrow
The current datasets are enough to validate mechanisms, not enough to support broad real-world claims.

### 13.4 Affective control is still simplified
The richer neuromodulation channels are useful, but still only an initial approximation.

---

## 14. Conclusion

Terminus Cortex is now best understood as a genuine language-and-reasoning layer under grounded predictive control. Its importance lies not in claiming solved cognition, but in demonstrating that a hybrid architecture can move past two weak extremes:
- past shallow LLM wrappers with no grounded control substrate
- past isolated spiking systems with little expressive capacity

As a result, Terminus Cortex is now a credible research platform for language-level cognition under predictive grounded control.

---

## References

[1] A. Baddeley. Working memory and the episodic buffer.

[2] B. Baars. *A Cognitive Theory of Consciousness*. 1988.

[3] S. Dehaene and J.-P. Changeux. Global workspace approaches to conscious processing.

[4] K. Friston. The free-energy principle and predictive processing.

[5] D. McAdams and K. McLean. Narrative identity.

[6] J. Hawkins et al. A framework for intelligence and cortical function based on grid cells in the neocortex.

[7] W. Yang and G. Buzsáki. Selection of experience for memory by hippocampal sharp wave ripples.

[8] W. Pan et al. Small-world spiking neural network emergence and performance.

[9] W. Savage. EMBER: autonomous cognitive behaviour from learned spiking neural network dynamics in a hybrid LLM architecture.

[10] Thousand Brains Project / Monty documentation and codebase.
