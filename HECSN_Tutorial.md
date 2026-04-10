# HECSN Tutorial — Operator & Developer Guide

> **Hierarchical Encoding with Competitive Spiking Networks**
> A biologically-grounded spiking neural network for autonomous knowledge accumulation.

---

## Table of Contents

1. [What is HECSN?](#1-what-is-hecsn)
2. [Architecture Overview](#2-architecture-overview)
3. [Installation & Setup](#3-installation--setup)
4. [Training a Checkpoint](#4-training-a-checkpoint)
5. [Running the Service](#5-running-the-service)
6. [Using the Dashboard UI](#6-using-the-dashboard-ui)
7. [Terminus — The Live Brain](#7-terminus--the-live-brain)
8. [API Reference](#8-api-reference)
9. [Validation & Testing](#9-validation--testing)
10. [Extension Points](#10-extension-points)

---

## 1. What is HECSN?

HECSN is a **spiking neural network** that learns continuously from streaming text without gradient descent. It uses biologically-inspired mechanisms:

- **Competitive columnar routing** — input patterns are routed to winner columns via nearest-neighbor lookup.
- **Local STDP plasticity** — synaptic weights update via spike-timing-dependent rules with eligibility traces.
- **Neuromodulated surprise** — dopamine, serotonin, acetylcholine, and norepinephrine modulate learning rate based on prediction error.
- **Sleep-based memory consolidation** — tag/PRP replay events transfer short-term traces into long-term memory.
- **Cross-modal grounding** — visual and audio grounding channels bind multi-sensory representations.
- **Slow-feature abstraction** — a feedback layer learns stable abstract features from fast-changing inputs.
- **Sparse binding** — coincidence-detection with PV-cell inhibition binds co-occurring patterns.

The system accumulates knowledge by processing text, stores it in a memory buffer with importance-weighted replay, and answers natural-language queries by retrieving relevant memories and routing through learned prototypes.

---

## 2. Architecture Overview

### Processing Pipeline

```
Text Input
    │
    ▼
┌────────────────┐
│  Character      │   Character n-gram encoding (or learned chunking)
│  Encoder        │   Converts text → fixed-dim input vector
└───────┬────────┘
        │
        ▼
┌────────────────┐
│  Competitive    │   n_columns prototypes compete via WTA
│  Layer          │   Winner determined by HNSW nearest-neighbor
└───────┬────────┘
        │
        ▼
┌────────────────┐
│  Context Layer  │   Multi-scale recurrent attractor with adaptive τ
│  (optional)     │   Maintains temporal context across inputs
└───────┬────────┘
        │
        ▼
┌────────────────┐
│  Abstraction    │   Slow-feature analysis feedback layer
│  Layer          │   Extracts stable abstract representations
└───────┬────────┘
        │
        ▼
┌────────────────┐
│  Binding Layer  │   Sparse subset STP coincidence detection
│  (optional)     │   Binds co-occurring patterns with PV inhibition
└───────┬────────┘
        │
        ▼
┌────────────────┐
│  Cross-Modal    │   Visual & audio grounding channels
│  Grounding      │   Builds multi-sensory associations
└───────┬────────┘
        │
        ▼
┌────────────────┐
│  Memory Store   │   Importance-weighted buffer with replay
│                 │   Tag/PRP consolidation (micro-sleep, deep-sleep)
└────────────────┘
```

### Key Modules

| Module | File | Purpose |
|--------|------|---------|
| `HECSNModelLite` | `src/hecsn/training/trainer.py` | Core model holding all layers |
| `HECSNTrainer` | `src/hecsn/training/trainer.py` | Training loop with sleep events |
| `HECSNConfig` | `src/hecsn/training/config.py` | All hyperparameters |
| `HECSNServiceManager` | `src/hecsn/service/manager.py` | Runtime service orchestration |
| `create_app` | `src/hecsn/service/api.py` | FastAPI server factory |
| `CompetitiveLayer` | `src/hecsn/model/competitive.py` | Columnar WTA competition |
| `ContextLayer` | `src/hecsn/model/context.py` | Recurrent attractor context |
| `AbstractionLayer` | `src/hecsn/model/abstraction.py` | SFA feedback layer |
| `BindingLayer` | `src/hecsn/model/binding.py` | Sparse coincidence binding |
| `CrossModalGrounding` | `src/hecsn/model/cross_modal.py` | Multi-sensory grounding |
| `MemoryStore` | `src/hecsn/model/memory.py` | Buffer + replay + consolidation |
| `SurpriseModule` | `src/hecsn/model/surprise.py` | Neuromodulator signals |

---

## 3. Installation & Setup

### Prerequisites

- Python 3.11+
- Node.js 20+ (for the dashboard UI)
- PyTorch 2.x

### Install Python Dependencies

```bash
pip install torch numpy pydantic uvicorn fastapi datasets
```

### Install UI Dependencies

```bash
cd HECSN_UI
npm install
```

### Build the UI

```bash
cd HECSN_UI
npm run build
```

The built files land in `HECSN_UI/dist/`. The server can serve them as static files.

---

## 4. Training a Checkpoint

Training creates a `.pt` checkpoint file that stores the model's learned weights, prototypes, and memory buffer.

### Quick Training Run

```bash
PYTHONPATH=src python -m hecsn.training.train_runner \
  --output-dir checkpoints/my_first_run \
  --dataset-name wikitext \
  --dataset-config wikitext-2-raw-v1 \
  --text-field text \
  --max-tokens 500000
```

### What Happens During Training

1. **Bootstrap phase** — Columns initialize prototypes from early input patterns.
2. **Memory warm-up** — After `slow_memory_start_tokens`, the memory store begins accepting traces.
3. **Micro-sleep events** — Periodic short replay bursts consolidate recent memories.
4. **Deep-sleep events** — Less frequent, more thorough replay across the full buffer.
5. **Drift monitoring** — Rolling drift floor tracks how much the representation is shifting.

### Evaluating a Checkpoint

Run the emergence evaluation protocol to verify the checkpoint passes quality gates:

```bash
PYTHONPATH=src python -m hecsn.training.emergence_evaluation_runner \
  --output-dir reports/my_evaluation
```

This runs the full gate battery: routing quality, memory retention, grounding accuracy, hierarchical scale.

---

## 5. Running the Service

The HECSN service is a FastAPI server that loads a checkpoint and exposes REST + SSE endpoints.

### Start the Server

```bash
PYTHONPATH=src python -m hecsn.service.server \
  --checkpoint checkpoints/my_first_run/model.pt \
  --port 8000 \
  --trace-dir reports/service/traces
```

### Server Options

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint` | (required) | Path to `.pt` checkpoint file |
| `--host` | `127.0.0.1` | Bind address |
| `--port` | `8000` | Port number |
| `--trace-history-limit` | `200` | Max traces kept in memory |
| `--trace-dir` | `reports/service/traces` | Where trace JSON files are written |
| `--web-dist-dir` | `web/dist` | Path to built UI static files |
| `--log-level` | `info` | Uvicorn log level |

### Verify It's Running

```bash
curl http://localhost:8000/health
# → {"status": "ok"}
```

---

## 6. Using the Dashboard UI

The React dashboard connects to the running service and provides real-time monitoring and interaction.

### Development Mode

```bash
cd HECSN_UI
npm run dev
```

Then open `http://localhost:5173`. Set the API base URL in the sidebar to point at your running service (default: `http://localhost:8000`).

### Dashboard Sections

#### Overview
Live summary cards showing tokens processed, memory fill fraction, answer support score, and loaded checkpoint. Includes:
- **Token trend chart** — incoming text processing rate over time.
- **State drift chart** — memory fill and drift floor side by side.
- **Novelty signals chart** — dopamine, serotonin, acetylcholine, norepinephrine traces.
- **V4 indicators** — cross-modal grounding confidence, adaptive context τ distribution, and plasticity circuit badges.

#### Architecture
SVG diagram of the active model layers driven by the `/architecture` endpoint. Each layer node shows its configuration; only layers enabled in the current config appear. Animated connection arrows show data flow between layers.

#### Activity (Animation)
Live visualization consuming SSE animation events:
- **Column activation grid** — each column as a cell, colored by threshold intensity, winner highlighted with a pulse ring.
- **Neuromodulator bars** — real-time dopamine/serotonin/acetylcholine/norepinephrine levels.
- **Memory fill arc** — radial gauge showing buffer occupancy.
- **Cross-modal confidence** — visual and audio channel strength when the grounding layer is active.

#### Ask
Interactive query workspace:
1. Type a natural-language question.
2. The system routes through learned prototypes, retrieves evidence from memory, and generates a grounded response.
3. View the routing path, evidence snippets, support score, and response mode.

#### Grounding Probe
Run the 50-triple grounding probe from the UI:
- Hit "Run probe" to evaluate concrete vs. abstract accuracy.
- View overall accuracy, concrete accuracy, and abstract accuracy bars.
- See the concreteness gap (difference between concrete and abstract performance).

#### Runtime
Detailed operational internals:
- Model type, revision, context norm, estimated neurons.
- Checkpoint metadata (protocol, source, token counts).
- Memory store (capacity, fill, importance, replay stats).
- Routing index (type, shards, balance ratio, rebuild count).
- Weight distribution (mean, std, skewness, kurtosis).
- **Capability flags** — all v4 features displayed as on/off badges (contextual routing, binding, log-STDP, iSTDP balance, AdEx spikes, STC consolidation, abstraction, attractor context) plus architecture descriptions.

#### Developmental
Stage pipeline showing runtime maturity:
- **Overall progress bar** — fraction of stages complete.
- **Plasticity details** — active mode and spike backend.
- **Consolidation counters** — sleep, micro-sleep, and deep-sleep events.
- **Stage list** — bootstrap → memory warm-up → consolidation → contextual → cross-modal → abstraction → binding, each with active/complete status.

#### Checkpoints
Save the current runtime state or restore a previously saved checkpoint.

#### Traces
Browse stored query traces with full routing, evidence, and response details.

---

## 7. Terminus — The Live Brain

Terminus is the autonomous brain runtime that continuously learns from configured data sources without human intervention.

### How It Works

1. **Configure sources** — Tell Terminus which HuggingFace datasets or text streams to consume.
2. **Start the brain** — Terminus begins processing text, one tick at a time.
3. **Autonomy** — When enabled, Terminus can self-direct which sources to focus on based on novelty signals and gap analysis.
4. **Checkpointing** — Terminus periodically saves checkpoints so learning persists across restarts.

### Using Terminus from the UI

The Ask section includes a "Terminus" panel where you can:
- **Configure** data sources (dataset name, config, text field, split, max tokens).
- **Start/Stop** the brain loop.
- **Tick** manually for single-step debugging.
- **Monitor** source progress, exhaustion status, and autonomy decisions.

### Using Terminus via API

```bash
# Check current brain status
curl http://localhost:8000/terminus

# Configure a source
curl -X POST http://localhost:8000/terminus/configure \
  -H "Content-Type: application/json" \
  -d '{"sources": [{"name": "wiki", "dataset": "wikitext", "config": "wikitext-2-raw-v1", "text_field": "text"}]}'

# Start autonomous learning
curl -X POST http://localhost:8000/terminus/start

# Single tick (for debugging)
curl -X POST http://localhost:8000/terminus/tick \
  -H "Content-Type: application/json" \
  -d '{"n_tokens": 200}'

# Stop the brain
curl -X POST http://localhost:8000/terminus/stop
```

---

## 8. API Reference

### Core Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check → `{"status": "ok"}` |
| `GET` | `/status` | Full runtime status snapshot |
| `GET` | `/architecture` | Layer topology diagram data |
| `GET` | `/stream/status?interval=1.0` | SSE stream of telemetry + animation data |
| `GET` | `/checkpoints` | List saved checkpoints |
| `GET` | `/traces?limit=20` | Recent query traces |
| `POST` | `/feed` | Feed text into the runtime for learning |
| `POST` | `/query` | Run a query and get routing + evidence |
| `POST` | `/respond` | Full question-answering with grounded response |
| `POST` | `/grounding-probe/run` | Run the 50-triple grounding probe |
| `POST` | `/checkpoint/save` | Save current state to disk |
| `POST` | `/checkpoint/restore` | Load a checkpoint from disk |

### Terminus Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/terminus` | Current brain runtime status |
| `POST` | `/terminus/configure` | Set data sources and autonomy config |
| `POST` | `/terminus/start` | Start the autonomous brain loop |
| `POST` | `/terminus/stop` | Stop the brain loop |
| `POST` | `/terminus/tick` | Execute one tick manually |

### SSE Stream Format

The `/stream/status` endpoint pushes events every `interval` seconds:

```
event: status
data: {"token_count": 42000, "last_winner": 7, "dopamine": 0.52, ..., "animation": {"n_columns": 64, "winner_id": 7, "activations": [...], "spike_counts": [...], "cross_modal": {"visual_confidence": 0.3, "audio_confidence": 0.2}, "context_tau": [0.9, 1.1, ...], "memory_fill": 0.45}}
```

The `animation` sub-object is designed for real-time visualization and contains:
- `n_columns` — number of competitive columns
- `winner_id` — currently winning column
- `activations` — threshold values per column
- `spike_counts` — cumulative spikes per column
- `cross_modal` — visual/audio confidence (if active)
- `context_tau` — adaptive time constants (if context layer active)
- `memory_fill` — buffer occupancy fraction

---

## 9. Validation & Testing

### Running the Full Test Suite

```bash
PYTHONPATH=src python -m pytest -q
```

Current baseline: **425 tests passed**, 7 subtests passed.

### Focused Test Slices

```bash
# Service API tests only
PYTHONPATH=src python -m pytest tests/test_service_api.py -q

# Grounding and query tests
PYTHONPATH=src python -m pytest tests/test_grounding_text.py tests/test_meaning_grounding.py tests/test_gap_planner.py -q

# Emergence evaluation
PYTHONPATH=src python -m pytest tests/test_emergence_evaluation_runner.py -q

# Long-horizon autonomy
PYTHONPATH=src python -m pytest tests/test_terminus_long_horizon_runner.py -q
```

### UI Build Verification

```bash
cd HECSN_UI
npm run build
```

Expect clean build with ~2519 modules and no errors.

### Emergence Evaluation Protocol

The full quality gate battery:

```bash
PYTHONPATH=src python -m hecsn.training.emergence_evaluation_runner \
  --output-dir reports/my_evaluation
```

This evaluates:
- **Routing quality** — winner diversity, prototype collapse detection
- **Memory retention** — recall accuracy after consolidation
- **Grounding accuracy** — concrete vs. abstract triple accuracy
- **Hierarchical scale** — multi-level routing consistency
- **Cross-modal binding** — visual/audio association quality

Results are written to `summary.json` in the output directory.

---

## 10. Extension Points

### Adding a New Layer

1. Create your layer module in `src/hecsn/model/`.
2. Register it in `HECSNModelLite.__init__()` (in `trainer.py`).
3. Wire it into the forward pass in the trainer's step method.
4. Add config flags in `HECSNConfig` (in `config.py`).
5. Expose it in `runtime_scope_report()` and `architecture_summary()`.
6. Add tests.

### Adding a New Data Source Type

Terminus data sources are configured via the `/terminus/configure` endpoint. To add a new source type:

1. Create a source runtime class similar to existing HuggingFace source runtimes.
2. Register it in the manager's source resolution logic.
3. The UI's Terminus panel will automatically pick up new sources.

### Adding a New Dashboard Section

1. Create `HECSN_UI/src/components/dashboard/YourSection.jsx`.
2. Add a lazy import in `App.jsx`.
3. Add an entry to the `SECTIONS` array with id, label, icon, and help text.
4. Add the id to `SECTION_TITLES`.
5. Add a `case` in `renderActiveSection`.

### Adding a New API Endpoint

1. Define request/response schemas in `src/hecsn/service/schemas.py`.
2. Add the handler method in `HECSNServiceManager`.
3. Wire the route in `create_app()` in `api.py`.
4. Add tests in `tests/test_service_api.py`.

---

## Quick Reference Card

```
# Start service
PYTHONPATH=src python -m hecsn.service.server --checkpoint checkpoints/model.pt

# Start UI dev server
cd HECSN_UI && npm run dev

# Run tests
PYTHONPATH=src python -m pytest -q

# Run emergence gates
PYTHONPATH=src python -m hecsn.training.emergence_evaluation_runner --output-dir reports/eval

# Feed text to the service
curl -X POST http://localhost:8000/feed -H "Content-Type: application/json" -d '{"text": "The sun is a star."}'

# Ask a question
curl -X POST http://localhost:8000/respond -H "Content-Type: application/json" -d '{"query_text": "What is the sun?"}'
```
