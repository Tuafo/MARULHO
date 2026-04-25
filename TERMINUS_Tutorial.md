# HECSN Tutorial — Current Operator & Developer Guide

> **Current runtime truth:** HECSN/Terminus is a hybrid SNN–LLM system with a predictive spiking substrate plus a **strict NVIDIA NIM** cortex path. There is **no production Ollama path** and no local Gemma runtime in the maintained system surface.

---

## Table of Contents

1. [What HECSN is now](#1-what-hecsn-is-now)
2. [Current runtime stack](#2-current-runtime-stack)
3. [Installation and environment setup](#3-installation-and-environment-setup)
4. [Training a checkpoint](#4-training-a-checkpoint)
5. [Running the service](#5-running-the-service)
6. [Building and serving the dashboard](#6-building-and-serving-the-dashboard)
7. [Terminus runtime](#7-terminus-runtime)
8. [Cortex, action loop, and sleep control](#8-cortex-action-loop-and-sleep-control)
9. [API reference](#9-api-reference)
10. [Validation and testing](#10-validation-and-testing)
11. [Extension points](#11-extension-points)
12. [Current limitations](#12-current-limitations)

---

## 1. What HECSN is now

HECSN is a **hybrid cognitive architecture**:

- a **predictive spiking substrate** handles grounding, prediction error, salience, replay, buffering, and local online adaptation
- a **cloud-hosted NVIDIA NIM cortex** handles language, reasoning, narrative continuity, question formation, answer generation, and dream-style recombination

The architectural rule is:

> the SNN side controls **when**, **why**, and **how deeply** the cortex thinks; the cortex does not free-run by itself.

The maintained system is now centered around five backend truths:

1. **runtime correctness**
2. **grounded evidence flow**
3. **real gating**
4. **maintained action/verification loops**
5. **explicit acceptance/health classification**

---

## 2. Current runtime stack

### Cortex / LLM side

Production cortex path:
- **fast model:** `nvidia/llama-3.1-nemotron-nano-8b-v1`
- **deep model:** `meta/llama-3.3-70b-instruct`
- **embedder:** `nvidia/llama-nemotron-embed-vl-1b-v2`

Important:
- production cortex is **NIM-only**
- tests may use `MockCortex`
- `CorticalCore` is the abstract interface, **not** an Ollama wrapper

### Text / multimodal runtime side

The maintained quick-start runtime uses one canonical preset:
- **`curriculum`**

Current runtime sources:
- `wikimedia/wikipedia` (`20231101.en`)
- `AlgorithmicResearchGroup/s2orc_arxiv` (`abstract`)
- `HuggingFaceFW/fineweb-edu` (`sample-10BT`)
- `ScienceOne-AI/S1-MMAlign`
- `OpenSound/AudioCaps`
- focus-aware allocation across maintained background text sources
- autonomy-guided real-source acquisition over maintained source catalogs with adaptive focus-pressure/provider-alignment budgeting, persistent source/provider utility calibration, grounded answer/action-outcome utility calibration, response-evidence provenance credit, delayed multi-turn consequence tracking, contradiction/decay-aware long-horizon utility penalties, explicit recovery/forgiveness scheduling for mixed long-horizon evidence, and age-sensitive retirement/cooling, compaction/aggregation of repeated long-horizon consequence records, trajectory-sensitive summaries for aggregated long-horizon consequence families, divergence-sensitive splitting of mixed long-horizon consequence families, lineage-aware remerge of split long-horizon consequence families, and grounded family-summary calibration of long-horizon consequence utility
- real multimodal grounding stays on the Hugging Face sensory streams

### Maintained action loop

The maintained digital action surface currently supports:
- `workspace_search`
- `workspace_read`
- `web_fetch`
- `api_request`

These all share the same maintained path for:
- execution
- verification
- provenance
- persistence
- replay

### Maintained sleep control

Explicit sleep control also now has a maintained manager-visible path:
- operator-triggered via `manager.cortex_sleep()` or `POST /terminus/cortex/sleep`
- cortex-intent-triggered via `action_intent="sleep"`

### Acceptance / long-test path

The maintained long-test runner now:
- runs a deterministic local acceptance harness
- classifies runs as:
  - `alive`
  - `degraded`
  - `dead`
- emits non-zero exit codes for degraded/dead runs

---

## 3. Installation and environment setup

### Prerequisites

- Python 3.10+
- Node.js 20+
- PyTorch 2.x
- a valid **NVIDIA NIM API key**
- Hugging Face access if you want the live HF-backed runtime to behave well

### Python dependencies

The minimal maintained dependency surface is defined in `pyproject.toml`.

Typical install:

```bash
pip install -e .
```

or, if you need a quick direct install:

```bash
pip install torch numpy fastapi uvicorn[standard] datasets httpx
```

### Environment

Create a `.env` in the repo root:

```bash
NVIDIA_API_KEY=your_nim_key_here
HF_TOKEN=your_hf_token_here
```

Recommended optional overrides:

```bash
NIM_FAST_MODEL=nvidia/llama-3.1-nemotron-nano-8b-v1
NIM_DEEP_MODEL=meta/llama-3.3-70b-instruct
NIM_MAX_RPM=20
HECSN_CORS_ORIGINS=http://127.0.0.1:5173,http://localhost:5173
```

### Runtime behavior when env is missing

- if `NVIDIA_API_KEY` is missing or NIM is unreachable, the **service can still start**, but cortex-dependent features are disabled
- if `HF_TOKEN` is missing, the live HF runtime may still work, but with worse limits / slower behavior

This is different from older documentation that implied local Ollama/Gemma operation. That is no longer the production path.

---

## 4. Training a checkpoint

Training still begins from the SNN side. The normal entrypoint is the developmental runner.

### Developmental training

```bash
PYTHONPATH=src python -m hecsn.training.developmental_runner \
  --output-dir checkpoints/terminus \
  --dataset-name wikitext \
  --dataset-config wikitext-103-raw-v1 \
  --text-field text \
  --max-tokens 500000
```

### Developmental validation protocol

```bash
PYTHONPATH=src python -m hecsn.training.developmental_runner \
  --output-dir reports/developmental \
  --n-tokens 5000
```

### Supported evaluation and health checks

```bash
PYTHONPATH=src python -m hecsn.training.meaning_grounding_runner \
  --output-dir reports/meaning_grounding
```

For runtime acceptance/health reporting, use:

```bash
PYTHONPATH=src python -m hecsn.training.long_test_runner \
  --duration 0.2 \
  --interval 2.0 \
  --preset curriculum \
  --output reports/long_test
```

The old `hecsn.training.emergence_evaluation_runner` module is no longer available.
Checkpoint output is a `.pt` file used by the local service.

---

## 5. Running the service

### Start the API service

```bash
PYTHONPATH=src python -m hecsn.service.server \
  --checkpoint checkpoints/terminus/model.pt \
  --port 8000 \
  --trace-dir reports/service/traces
```

### Useful flags

| Flag | Meaning |
|---|---|
| `--checkpoint` | checkpoint `.pt` file to load |
| `--host` | bind host |
| `--port` | bind port |
| `--trace-history-limit` | in-memory trace cap |
| `--trace-dir` | persisted trace output directory |
| `--web-dist-dir` | built frontend directory to serve at `/app`; defaults to `HECSN_UI/dist` |
| `--log-level` | uvicorn log level |
| `--reload` | dev autoreload |

### Verify it is up

```bash
curl http://localhost:8000/health
```

Expected:

```json
{"status":"ok"}
```

---

## 6. Building and serving the dashboard

The active frontend source currently lives in:
- `HECSN_UI/`

### Development mode

```bash
cd HECSN_UI
npm install
npm run dev
```

### Production build

```bash
cd HECSN_UI
npm run build
```

The service defaults to this build directory:

```bash
PYTHONPATH=src python -m hecsn.service.server \
  --checkpoint checkpoints/terminus/model.pt
```

If the build directory exists, the service mounts it at:
- `/app`

---

## 7. Terminus runtime

Terminus is the maintained long-running backend runtime. The canonical current path is:

`observe -> predict -> error/salience/drives -> reason/act -> verify -> typed memory -> replay/consolidation -> self-model update`

In practice this means:

1. observe configured sources, sensory previews, operator questions, and action outcomes
2. predict likely outcomes before acting or answering
3. convert prediction error, salience, uncertainty, novelty, fatigue, and drive pressure into control signals
4. reason or act through the cortex and maintained digital action loop
5. verify action/answer outcomes against grounded evidence
6. write typed memory with provenance such as `observed`, `inferred`, `dreamed`, `verified`, and `contradicted`
7. replay and consolidate memories, delayed consequences, source utility, and long-horizon evidence
8. update the operational self-model so capabilities, limits, budgets, memory health, and grounding health stay visible

### Current quick-start preset

There is currently **one maintained quick-start preset**:
- `curriculum`

It configures:
- 3 HF background text sources
- 2 live sensory sources
- focus-aware allocation across the maintained background text registry
- autonomy-guided targeted acquisition over the maintained real-source registry with adaptive focus-pressure/provider-alignment budgeting, persistent source/provider utility calibration, grounded answer/action-outcome utility calibration, response-evidence provenance credit, delayed multi-turn consequence tracking, contradiction/decay-aware long-horizon utility penalties, explicit recovery/forgiveness scheduling for mixed long-horizon evidence, and age-sensitive retirement/cooling, compaction/aggregation of repeated long-horizon consequence records, trajectory-sensitive summaries for aggregated long-horizon consequence families, divergence-sensitive splitting of mixed long-horizon consequence families, lineage-aware remerge of split long-horizon consequence families, and grounded family-summary calibration of long-horizon consequence utility
- bounded `tick_tokens` for better short-run visibility
- hypercube binding + cross-modal runtime support via model overrides

### Terminus runtime commands

#### List presets

```bash
curl http://localhost:8000/terminus/presets
```

#### Quick-start the maintained runtime

```bash
curl -X POST "http://localhost:8000/terminus/quick-start?preset=curriculum"
```

#### Inspect runtime state

```bash
curl http://localhost:8000/terminus
```

#### Inspect the living loop/self-model surface

```bash
curl http://localhost:8000/terminus/living-loop
```

`/terminus/living-loop` returns the same living-loop snapshot embedded in `/status` under `terminus_runtime.living_loop`, plus wrapper status fields such as `dirty_state`, `state_revision`, and `token_count`.

Key telemetry fields:
- `prediction_count` — number of surfaced predictions in the current snapshot
- `action_count` — number of surfaced action records in the current snapshot
- `world_model_lite` — derived prediction/verification scoring, contradiction rate, policy score, and recommended next action
- `skill_memories` — action/tool memories derived from verified, contradicted, and mixed action history
- `capabilities` — currently observed operational abilities such as action execution, prediction tracking, verification tracking, and world-model-lite policy scoring
- `limits` — supported actions, snapshot/history counts, truncation state, memory capacity/fill, revision, and runtime flags
- `budgets` — action-history use, snapshot use, world-model policy budget/cost/risk/uncertainty, and memory use
- `memory_health` — capacity/fill/provenance health for the surfaced memory snapshot
- `grounding_health` — verification, evidence, contradiction, and grounding status for the current loop state

#### Stop runtime

```bash
curl -X POST http://localhost:8000/terminus/stop
```

### Custom runtime configuration

If you do not want the canonical preset, use `/terminus/configure` with explicit `source_bank`, `ingestion`, `sensory`, and `autonomy` payloads.

Example local file configuration:

```bash
curl -X POST http://localhost:8000/terminus/configure \
  -H "Content-Type: application/json" \
  -d '{
    "source_bank": [
      {
        "name": "local_text",
        "source": "C:/data/notes.txt",
        "source_type": "file",
        "text_field": "text",
        "topic_terms": ["systems neuroscience memory plasticity"],
        "metadata": {"label": "systems neuroscience notes"}
      }
    ],
    "tick_tokens": 20,
    "sleep_interval_seconds": 0.01,
    "repeat_sources": false,
    "ingestion": {
      "enabled": true,
      "queue_target_tokens": 40,
      "prewarm_on_startup": false,
      "prewarm_max_seconds": 0.2
    }
  }'
```

Manual tick:

```bash
curl -X POST http://localhost:8000/terminus/tick \
  -H "Content-Type: application/json" \
  -d '{"steps": 2}'
```

---

## 8. Cortex, action loop, and sleep control

### Cortex runtime

The cortex side is organized around:
- `MultiCortex`
- `ThoughtLoop`
- `DriveSystem`
- `ThalamicGate`
- `WorkingMemory`
- `NarrativeSelf`
- `EpisodicMemory`

### Programmatic cortex creation

Production path:

```python
from hecsn.cortex import ThoughtLoop, create_cortex_from_env

cortex = create_cortex_from_env()
brain = ThoughtLoop(cortex=cortex)
```

Test path:

```python
from hecsn.cortex import MockCortex, ThoughtLoop

brain = ThoughtLoop(cortex=MockCortex())
```

### Maintained action loop

The maintained action loop is backend-first and verification-first.

Supported actions:
- `workspace_search`
- `workspace_read`
- `web_fetch`
- `api_request`

API examples:

```bash
curl -X POST http://localhost:8000/terminus/action \
  -H "Content-Type: application/json" \
  -d '{
    "action_type": "workspace_search",
    "query_text": "cats chase mice",
    "predicted_outcome": "I expect grounded workspace evidence about cats chasing mice."
  }'
```

```bash
curl http://localhost:8000/terminus/actions
```

### Maintained sleep control

Operator-triggered explicit sleep:

```bash
curl -X POST http://localhost:8000/terminus/cortex/sleep \
  -H "Content-Type: application/json" \
  -d '{"reason": "Operator requested a consolidation cycle."}'
```

Thought-loop query submission:

```bash
curl -X POST "http://localhost:8000/terminus/ask?query=What%20do%20cats%20chase%20at%20night%3F"
```

Recent thought stream:

```bash
curl http://localhost:8000/terminus/thoughts
curl http://localhost:8000/terminus/cortex
```

---

## 9. API reference

### Core service endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | liveness check |
| `GET` | `/status` | full service/runtime snapshot |
| `GET` | `/architecture` | architecture summary |
| `GET` | `/traces` | recent trace history |
| `GET` | `/datasets` | current runtime datasets |
| `GET` | `/stream/status` | SSE telemetry stream |
| `POST` | `/feed` | feed text into the checkpoint-backed runtime |
| `POST` | `/query` | routing/evidence query without answer synthesis |
| `POST` | `/respond` | grounded answer generation |
| `POST` | `/grounding-probe/run` | grounding probe |
| `GET` | `/checkpoints` | list checkpoints |
| `POST` | `/checkpoint/save` | save checkpoint |
| `POST` | `/checkpoint/restore` | restore checkpoint |

### Terminus runtime endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/terminus` | current Terminus runtime snapshot |
| `GET` | `/terminus/living-loop` | living-loop telemetry and operational self-model snapshot |
| `GET` | `/terminus/presets` | list maintained quick-start presets |
| `POST` | `/terminus/quick-start` | apply preset and start runtime |
| `POST` | `/terminus/configure` | explicit runtime configuration |
| `POST` | `/terminus/start` | start background runtime |
| `POST` | `/terminus/stop` | stop background runtime |
| `POST` | `/terminus/tick` | manual step-based tick |
| `POST` | `/terminus/ask` | queue cortex query |
| `GET` | `/terminus/thoughts` | recent thoughts |
| `GET` | `/terminus/cortex` | full cortex snapshot |
| `POST` | `/terminus/cortex/sleep` | explicit maintained sleep request |
| `GET` | `/terminus/actions` | action history |
| `POST` | `/terminus/action` | execute maintained digital action |
| `GET` | `/terminus/sensory/recent` | recent sensory previews |

---

## 10. Validation and testing

### Full suite

```bash
python -m pytest -q
```

### Useful slices

```bash
# service + runtime
python -m pytest tests/test_service_api.py tests/test_service_manager.py -q

# cortex / thought loop
python -m pytest tests/test_cortical_core.py tests/test_thought_loop.py -q

# action loop
python -m pytest tests/test_action_loop.py -q

# long-test / acceptance path
python -m pytest tests/test_long_test_runner.py -q
```

### Long-test runner

```bash
PYTHONPATH=src python -m hecsn.training.long_test_runner \
  --duration 0.2 \
  --interval 2.0 \
  --preset curriculum \
  --output reports/wp06_validation
```

The maintained long-test path now:
- runs the deterministic local acceptance harness
- emits `alive` / `degraded` / `dead`
- writes markdown + JSON reports

### Benchmark posture: ARC-AGI

ARC-AGI should remain a **separate benchmark path**, not a claim attached to the current Terminus runtime. A credible ARC-AGI path would need, at minimum:
- an object parser for grids and relational scene structure
- a DSL/program synthesis layer for candidate transformations
- a verifier that executes candidates against examples
- search/refinement over candidates and partial programs
- optional LLM candidates as proposals, not as the scorer
- exact-match scoring against held-out task outputs

Current Terminus telemetry validates living-loop, grounding, action verification, memory, and operational self-model posture. It does **not** imply that Terminus already solves ARC-AGI.

### Current validation reports

Key backend reports live in:
- `reports/runtime_correctness_validation.md`
- `reports/grounding_bridge_validation.md`
- `reports/gating_validation.md`
- `reports/ingestion_latency_validation.md`
- `reports/action_loop_validation.md`
- `reports/acceptance_harness_validation.md`

---

## 11. Extension points

### Adding a new maintained action

1. extend `src/hecsn/service/action_loop.py`
2. extend `src/hecsn/service/schemas.py`
3. keep execution/verification/provenance on the same maintained path
4. add manager/service/API tests
5. add controlled validation and update reports

### Adding a new source type

1. add the source runtime/stream logic in `src/hecsn/service/manager.py`
2. keep it on the same ingestion/warm-queue/runtime path
3. extend schemas/API only if needed
4. add tests for configuration, runtime progress, and shutdown behavior

### Adding a new sensory modality

1. add encoder/runtime support in `src/hecsn/data/` and service wiring
2. integrate with the same sensory buffering and observation path
3. add tests and validation

### Adding a new API endpoint

1. define schemas in `src/hecsn/service/schemas.py`
2. add manager method
3. wire route in `src/hecsn/service/api.py`
4. add tests in `tests/test_service_api.py`

---

## 12. Current limitations

The active docs should state current runtime limitations plainly.

### Runtime limitations
- production cortex depends on NVIDIA NIM availability and budget
- live Hugging Face sources can still introduce real latency and transient stalls
- short smoke runs are now correctly classified, but they remain sensitive to real remote performance

### Cognitive limitations
- dream verification remains weak in short runs
- some wake thoughts are still awkward or generic
- thought quality is not yet consistent across domains

### Data limitations
- current multimodal sources are materially better than old digit-only scaffolding, but they are still not broad embodied experience
- focus-aware background source allocation plus autonomy-guided real-source acquisition with adaptive focus-pressure/provider-alignment budgeting, persistent source/provider utility calibration, grounded answer/action-outcome utility calibration, response-evidence provenance credit, delayed multi-turn consequence tracking, contradiction/decay-aware long-horizon utility penalties, explicit recovery/forgiveness scheduling for mixed long-horizon evidence, and age-sensitive retirement/cooling, compaction/aggregation of repeated long-horizon consequence records, trajectory-sensitive summaries for aggregated long-horizon consequence families, divergence-sensitive splitting of mixed long-horizon consequence families, lineage-aware remerge of split long-horizon consequence families, and grounded family-summary calibration of long-horizon consequence utility is still only a first step, not a substitute for richer real-world grounding data

### Documentation truth
If you find any operator-facing documentation that still tells you to:
- start Ollama
- pull Gemma locally
- use `FakeCortex`
- use removed quick-start presets like `multimodal`

then that documentation is stale and should be treated as incorrect.

---

## Quick reference

```bash
# Train a checkpoint
PYTHONPATH=src python -m hecsn.training.developmental_runner \
  --output-dir checkpoints/terminus \
  --dataset-name wikitext \
  --dataset-config wikitext-103-raw-v1 \
  --text-field text

# Start service
PYTHONPATH=src python -m hecsn.service.server \
  --checkpoint checkpoints/terminus/model.pt

# Build frontend
cd HECSN_UI && npm install && npm run build

# Run tests
python -m pytest -q

# Run long-test acceptance/health path
PYTHONPATH=src python -m hecsn.training.long_test_runner \
  --duration 0.2 \
  --interval 2.0 \
  --preset curriculum \
  --output reports/wp06_validation

# Quick-start Terminus
curl -X POST "http://localhost:8000/terminus/quick-start?preset=curriculum"

# Ask cortex
curl -X POST "http://localhost:8000/terminus/ask?query=What%20is%20the%20sun%3F"

# Explicit cortex sleep
curl -X POST http://localhost:8000/terminus/cortex/sleep \
  -H "Content-Type: application/json" \
  -d '{"reason": "Operator requested a consolidation cycle."}'
```
