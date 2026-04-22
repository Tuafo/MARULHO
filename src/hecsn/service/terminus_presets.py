"""Terminus quick-start presets.

The live Terminus runtime is now Hugging Face + NIM curriculum:
- Hugging Face educational / encyclopedic / scientific text streams
- real Hugging Face visual/audio grounding episodes
- NIM-generated curriculum episodes for targeted learning
- curriculum-derived visual/audio hints as an additional lightweight path

The preset surface is intentionally narrow so the runtime stays aligned with the
current architecture.
"""

from __future__ import annotations

from typing import Any

from hecsn.service.terminus_hf_sources import current_runtime_sensory_config, current_runtime_source_bank


TERMINUS_QUICK_START_PRESETS: dict[str, dict[str, Any]] = {
    "curriculum": {
        "label": "Curriculum (HF background + NIM guidance)",
        "default": True,
        "description": (
            "Uses a Hugging Face source mixture for steady background training "
            "(Wikipedia, S2ORC ArXiv abstracts, FineWeb-Edu), real Hugging Face "
            "multimodal grounding episodes (S1-MMAlign + AudioCaps) on a balanced, "
            "confidence-aware, semantically routed schedule, and NIM-generated "
            "curriculum episodes for targeted learning. Curriculum hints remain as a "
            "lightweight auxiliary multimodal path."
        ),
        "source_bank": current_runtime_source_bank(),
        "curriculum": {
            "enabled": True,
            "topics_per_cycle": 3,
            "episode_length_tokens": 256,
            "diversity_threshold": 0.7,
            "trigger_interval_tokens": 1024,
            "cooldown_seconds": 30.0,
        },
        "sensory": current_runtime_sensory_config(),
        "tick_tokens": 512,
        "sleep_interval_seconds": 0.05,
        "repeat_sources": True,
        "model_overrides": {
            "n_columns": 1024,
            "enable_binding_layer": True,
            "binding_mode": "hypercube",
            "routing_shards": 4,
            "plasticity_spike_backend": "adex",
            "enable_cross_modal": True,
            "cross_modal_dim_visual": 64,
            "cross_modal_dim_audio": 64,
        },
    },
}
