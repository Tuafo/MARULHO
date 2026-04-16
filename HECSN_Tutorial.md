# HECSN Tutorial — Operator & Developer Guide

> **Hierarchical Encoding with Competitive Spiking Networks**
> A biologically-grounded spiking neural network for autonomous multimodal knowledge accumulation.

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

HECSN is a **multimodal spiking neural network** that learns continuously from streaming text, visual events, and audio without gradient descent. It uses biologically-inspired mechanisms:

- **Competitive columnar routing** — input patterns are routed to winner columns via HNSW nearest-neighbor lookup (256 columns by default).
- **Local STDP plasticity** — synaptic weights update via log-domain spike-timing-dependent rules with eligibility traces.
- **AdEx spike dynamics** — adaptive-exponential integrate-and-fire neurons with realistic spike generation.
- **Neuromodulated surprise** — dopamine, serotonin, acetylcholine, and norepinephrine modulate learning rate based on prediction error.
- **Sleep-based memory consolidation** — tag/PRP replay events with synaptic tag capture transfer short-term traces into long-term memory.
- **Cross-modal grounding** — visual (event camera) and audio (cochleagram) grounding channels bind multi-sensory representations.
- **Slow-feature abstraction** — a feedback layer learns stable abstract features from fast-changing inputs.
- **Hypercube binding** — 11-dimensional hypercube topology binds co-occurring patterns with O(N·d) memory, scaling to 100K+ columns where dense/spatial binding is impossible.

The system accumulates knowledge by processing text, images, and audio, stores it in a memory buffer with importance-weighted replay, and answers natural-language queries by retrieving relevant memories and routing through learned prototypes.

---

## 2. Architecture Overview

### Processing Pipeline

```
Text / Visual / Audio Input
    │
    ▼
┌────────────────┐
│  Encoders       │   Character n-gram / learned chunking (text)
│                 │   EventCameraEncoder (N-MNIST visual events)
│                 │   CochleagramEncoder (FSDD audio)
└───────┬────────┘
        │
        ▼
┌────────────────┐
│  Competitive    │   256 columns (default) compete via WTA
│  Layer          │   Winner determined by HNSW nearest-neighbor
└───────┬────────┘
        │
        ▼
┌────────────────┐
│  Context Layer  │   Multi-scale recurrent attractor with adaptive τ
│                 │   Maintains temporal context across inputs
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
│  Binding Layer  │   Three modes: dense, spatial, or hypercube
│                 │   Hypercube uses 11D topology for O(N·d) scaling
│                 │   Binds co-occurring patterns with PV inhibition
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
| `HECSNModel` | `src/hecsn/training/trainer.py` | Core model holding all layers |
| `HECSNTrainer` | `src/hecsn/training/trainer.py` | Training loop with sleep events |
| `HECSNConfig` | `src/hecsn/config/model_config.py` | All hyperparameters |
| `HECSNServiceManager` | `src/hecsn/service/manager.py` | Runtime service orchestration |
| `create_app` | `src/hecsn/service/api.py` | FastAPI server factory |
| `CompetitiveLayer` | `src/hecsn/core/columns.py` | Columnar WTA competition |
| `ContextLayer` | `src/hecsn/core/context.py` | Recurrent attractor context |
| `AbstractionLayer` | `src/hecsn/core/abstraction.py` | SFA feedback layer |
| `BindingLayer` | `src/hecsn/core/context.py` | Dense coincidence binding |
| `SpatialBindingLayer` | `src/hecsn/core/topographic.py` | Spatially-organized binding |
| `HypercubeBindingLayer` | `src/hecsn/core/hypercube.py` | 11D hypercube binding (default) |
| `CrossModalGrounding` | `src/hecsn/core/cross_modal.py` | Multi-sensory grounding |
| `MemoryStore` | `src/hecsn/consolidation/memory_store.py` | Buffer + replay + consolidation |
| `SurpriseModule` | `src/hecsn/core/surprise.py` | Neuromodulator signals |
| `EventCameraEncoder` | `src/hecsn/data/event_camera_encoder.py` | N-MNIST visual event encoding |
| `CochleagramEncoder` | `src/hecsn/data/cochleagram_encoder.py` | Audio cochleagram encoding |
| `DevelopmentalRunner` | `src/hecsn/training/developmental_runner.py` | Multi-stage developmental protocol |

---

## 3. Installation & Setup

### Prerequisites

- Python 3.11+
- Node.js 20+ (for the dashboard UI)
- PyTorch 2.x (CUDA optional — CPU is faster for ≤2048 columns)

### Install Python Dependencies

```bash
pip install torch numpy pydantic uvicorn fastapi datasets librosa soundfile hnswlib
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

The built files land in `HECSN_UI/dist/`. The server serves them as static files automatically.

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

### Developmental Protocol

For full multi-stage training (bootstrap → memory → consolidation → contextual → cross-modal → abstraction → binding):

```bash
PYTHONPATH=src python -m hecsn.training.developmental_runner \
  --output-dir checkpoints/developmental \
  --dataset-name wikitext \
  --dataset-config wikitext-103-raw-v1 \
  --text-field text
```

The developmental runner automatically advances through stages based on probe metrics and growth-rate criteria, enabling layers one at a time.

### Key Config Defaults

| Parameter | Default | Notes |
|-----------|---------|-------|
| `n_columns` | 256 | Number of competitive columns |
| `binding_mode` | `"hypercube"` | Binding topology (dense, spatial, or hypercube) |
| `enable_binding_layer` | `False` | Enable after bootstrapping |
| `enable_context_layer` | `True` | Recurrent attractor context |

### What Happens During Training

1. **Bootstrap phase** — Columns initialize prototypes from early input patterns.
2. **Memory warm-up** — After `slow_memory_start_tokens`, the memory store begins accepting traces.
3. **Micro-sleep events** — Periodic short replay bursts consolidate recent memories.
4. **Deep-sleep events** — Less frequent, more thorough replay across the full buffer.
5. **Drift monitoring** — Rolling drift floor tracks how much the representation is shifting.

### Multimodal Training

To train with text, visual (N-MNIST), and audio (FSDD) simultaneously:

```python
from hecsn.data.dataset_adapters import iter_episode_steps
from hecsn.data.event_camera_encoder import EventCameraEncoder
from hecsn.data.cochleagram_encoder import CochleagramEncoder

visual_enc = EventCameraEncoder(output_dim=128)
audio_enc = CochleagramEncoder(output_dim=128)

for step in iter_episode_steps(nmnist_dir="N-MNIST",
                                fsdd_dir="free-spoken-digit-dataset-master/recordings",
                                n_steps=10):
    trainer.train_step(step.input_vector,
                       visual_input=step.visual,
                       audio_input=step.audio)
```

The Terminus quick-start presets handle this automatically (see §7).

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

Terminus is the autonomous brain runtime that continuously learns from configured data sources without human intervention. It runs with 1024 columns (2048 in fast mode) and hypercube binding for production-scale learning.

### How It Works

1. **Quick-start presets** — Choose a preset to instantly configure sources and model scale.
2. **Configure sources** — Tell Terminus which HuggingFace datasets to consume (text, visual, audio).
3. **Start the brain** — Terminus begins processing data, one tick at a time.
4. **Autonomy** — When enabled, Terminus can self-direct which sources to focus on based on novelty signals and gap analysis.
5. **Checkpointing** — Terminus periodically saves checkpoints so learning persists across restarts.

### Quick-Start Presets

Terminus includes six pre-configured presets accessible from the UI or API:

| Preset | Description | Columns | Modalities |
|--------|-------------|---------|------------|
| **Wikipedia** | General knowledge from wikitext-103 | 1024 | Text |
| **Wikipedia + News** | Wikipedia paired with AG News | 1024 | Text |
| **Diverse** | Wiki + News + IMDB reviews | 1024 | Text |
| **Diverse — Fast** | Same three domains, larger batches | 2048 | Text |
| **Multimodal** | Wiki + N-MNIST visual + FSDD audio | 1024 | Text + Visual + Audio |
| **Multimodal — Fast** | Faster multimodal training | 2048 | Text + Visual + Audio |

All presets use **hypercube binding** and will rebuild the model to the specified column count on activation.

### Using Terminus from the UI

The Dashboard includes a **Terminus** panel where you can:
- **Select a quick-start preset** from a dropdown and activate it with one click.
- **Configure** custom data sources (dataset name, config, text field, split, max tokens).
- **Start/Stop** the brain loop.
- **Tick** manually for single-step debugging.
- **Monitor** source progress, exhaustion status, and autonomy decisions.
- **Chat** with the brain in the Ask tab — see how it routes your questions through learned prototypes.

### Using Terminus via API

```bash
# List available presets
curl http://localhost:8000/terminus/presets

# Quick-start with a preset
curl -X POST http://localhost:8000/terminus/quick-start \
  -H "Content-Type: application/json" \
  -d '{"preset": "multimodal"}'

# Check current brain status
curl http://localhost:8000/terminus

# Configure a custom source
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
| `GET` | `/datasets` | List available HuggingFace datasets |
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
| `GET` | `/terminus/presets` | List available quick-start presets |
| `POST` | `/terminus/configure` | Set data sources and autonomy config |
| `POST` | `/terminus/quick-start` | Activate a preset (rebuilds model if needed) |
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

Current baseline: **~654 tests passed**, 7 subtests passed.

### Focused Test Slices

```bash
# Service API & manager tests
PYTHONPATH=src python -m pytest tests/test_service_api.py tests/test_service_manager.py -q

# Hypercube binding tests
PYTHONPATH=src python -m pytest tests/test_hypercube.py -q

# Topographic & spatial binding tests
PYTHONPATH=src python -m pytest tests/test_topographic.py -q

# Multimodal & cross-modal tests
PYTHONPATH=src python -m pytest tests/test_cross_modal.py tests/test_dataset_adapters.py -q

# Developmental runner tests
PYTHONPATH=src python -m pytest tests/test_developmental_runner.py -q

# Grounding and query tests
PYTHONPATH=src python -m pytest tests/test_grounding_text.py tests/test_meaning_grounding.py tests/test_gap_planner.py -q

# Emergence evaluation
PYTHONPATH=src python -m pytest tests/test_emergence_evaluation_runner.py -q
```

### UI Build Verification

```bash
cd HECSN_UI
npm run build
```

Expect clean build with no errors.

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

1. Create your layer module in `src/hecsn/core/`.
2. Register it in `HECSNModel.__init__()` (in `src/hecsn/training/trainer.py`).
3. Wire it into the forward pass in the trainer's `train_step` method.
4. Add config flags in `HECSNConfig` (in `src/hecsn/config/model_config.py`).
5. Expose it in `runtime_scope_report()` and `architecture_summary()`.
6. Add a developmental stage in `DevelopmentalRunner` if needed.
7. Add tests.

### Adding a New Binding Mode

1. Create your binding layer in `src/hecsn/core/` (see `hypercube.py` as reference).
2. Implement `bind(context_prediction, assembly) → bound_output` method.
3. Add the mode to `binding_mode` validation in `HECSNConfig.__post_init__`.
4. Wire it into `HECSNModel.__init__` and `HECSNTrainer.train_step`.
5. Add tests.

### Adding a New Sensory Modality

1. Create an encoder in `src/hecsn/data/` (see `event_camera_encoder.py` or `cochleagram_encoder.py`).
2. The encoder should implement `encode(raw_data) → spike_vector` and `reset()`.
3. Add a grounding channel in `CrossModalGrounding` (`src/hecsn/core/cross_modal.py`).
4. Wire it into `iter_episode_steps` in `src/hecsn/data/dataset_adapters.py`.
5. Add the modality to `HECSNTrainer.train_step`.

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

# Quick-start Terminus with multimodal preset
curl -X POST http://localhost:8000/terminus/quick-start -H "Content-Type: application/json" -d '{"preset": "multimodal"}'
```

---

## Appendix A: Binding Modes

HECSN supports three binding topologies, configured via `binding_mode` in `HECSNConfig`:

| Mode | Memory | Speed at 1K cols | Max columns | Best for |
|------|--------|-------------------|-------------|----------|
| `dense` | O(N²) | 77 steps/s | ~4K | Small experiments |
| `spatial` | O(N²) | similar | ~8K | Topographic research |
| `hypercube` | O(N·d) | 750 steps/s | 131K+ tested | **Default — production scale** |

**Hypercube** uses an 11-dimensional hypercube topology where each column connects to its `d` nearest neighbors in hypercube space. This gives O(N·d) memory instead of O(N²), making it possible to run networks with 100K+ columns on a single machine.

At 256 columns (the default), all three modes are functionally equivalent. The advantage of hypercube emerges at scale: it is 10× faster at 1K columns, 49× faster at 4K, and the only option that works beyond ~8K columns.

## Appendix B: Performance Notes

- **CPU vs GPU**: CPU is faster than GPU for HECSN at ≤2048 columns. GPU only benefits at 10K+ columns due to per-step SNN ops being too small for GPU kernel launch overhead.
- **Throughput**: ~72 tok/s sustained at 256 columns on CPU (1M token scale test). Throughput improves over time as prototypes stabilize.
- **Scaling**: Hypercube build time is a one-time cost at model initialization (~2 min at 131K columns). Per-step cost is O(1) with no degradation over time.
