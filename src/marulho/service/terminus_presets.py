"""Terminus quick-start presets.

The live Terminus runtime is now Hugging Face + real-source autonomy:
- Hugging Face open-textbook / educational / scientific text streams
- focus-aware background source allocation across those text streams
- real Hugging Face visual/audio grounding episodes
- autonomy-driven targeted acquisition over maintained real source catalogs
- adaptive autonomy cadence/budget retuning under strong focus pressure
- persistent source/provider utility calibration on that maintained path
- grounded answer/action-outcome utility calibration on that same path
- response-evidence provenance credit on that same path
- delayed multi-turn consequence tracking on that same path
- contradiction/decay-aware long-horizon utility penalties on that same path
- explicit recovery/forgiveness scheduling for mixed long-horizon evidence on that same path
- age-sensitive retirement/cooling, compaction/aggregation of repeated long-horizon consequence records, trajectory-sensitive summaries for aggregated long-horizon consequence families, divergence-sensitive splitting of mixed long-horizon consequence families, lineage-aware remerge of split long-horizon consequence families, and grounded family-summary calibration of long-horizon consequence utility on that same path

The preset surface is intentionally narrow so the runtime stays aligned with the
current architecture.
"""

from __future__ import annotations

from typing import Any

from marulho.service.terminus_hf_sources import (
    current_runtime_autonomy_config,
    current_runtime_sensory_config,
    current_runtime_source_bank,
)


TERMINUS_QUICK_START_PRESETS: dict[str, dict[str, Any]] = {
    "curriculum": {
        "label": "Curriculum (HF background + adaptive autonomy guidance)",
        "default": True,
        "description": (
            "Uses a Hugging Face source mixture for steady background training "
            "(OpenStax open textbooks, S2ORC ArXiv abstracts, FineWeb-Edu) with focus-aware background "
            "source allocation, real Hugging Face multimodal grounding episodes "
            "(S1-MMAlign + AudioCaps) on a balanced, confidence-aware, semantically "
            "routed schedule, plus autonomy-driven targeted acquisition over the maintained "
            "real source registry with adaptive focus-pressure/provider-alignment budgeting, "
            "persistent source/provider utility calibration, grounded answer/action-outcome "
            "utility calibration, response-evidence provenance credit, delayed multi-turn "
            "consequence tracking, contradiction/decay-aware long-horizon utility penalties, "
            "explicit recovery/forgiveness scheduling for mixed long-horizon evidence, and "
            "age-sensitive retirement/cooling, compaction/aggregation of repeated long-horizon consequence records, trajectory-sensitive summaries for aggregated long-horizon consequence families, divergence-sensitive splitting of mixed long-horizon consequence families, lineage-aware remerge of split long-horizon consequence families, and grounded family-summary calibration of long-horizon consequence utility. Real multimodal grounding "
            "is carried by the maintained Hugging Face sensory path."
        ),
        "source_bank": current_runtime_source_bank(),
        "autonomy": current_runtime_autonomy_config(),
        "sensory": current_runtime_sensory_config(),
        "tick_tokens": 128,
        "source_concept_observation_tick_interval": 4,
        "sleep_interval_seconds": 0.05,
        "execution_quantum_tokens": 8,
        "execution_yield_seconds": 0.0,
        "repeat_sources": True,
        "model_overrides": {
            "n_columns": 1024,
            "memory_capacity": 1000,
            "enable_context_layer": True,
            "enable_binding_layer": True,
            "binding_mode": "hypercube",
            "routing_shards": 4,
            "plasticity_spike_backend": "adex",
            "slow_memory_archive_interval_tokens": 256,
            "enable_cross_modal": True,
            "cross_modal_dim_visual": 64,
            "cross_modal_dim_audio": 64,
        },
    },
}
