"""Terminus quick-start presets — predefined configurations for brain training.

Each preset specifies:
- source_bank: HuggingFace datasets for text training
- multimodal: optional N-MNIST + FSDD episode configuration
- tick_tokens: tokens processed per brain tick
- model_overrides: HECSNConfig overrides (columns, binding, spike backend)

The default preset is 'multimodal' — trains all three modalities (text + visual + audio).
Wikipedia-only presets are retained for backwards compatibility but are not recommended
for production Terminus use (the LLM cortex already has text knowledge; the SNN should
focus on multimodal grounding).
"""

from __future__ import annotations

from typing import Any


TERMINUS_QUICK_START_PRESETS: dict[str, dict[str, Any]] = {
    "wikipedia": {
        "label": "Wikipedia (wikitext-103)",
        "description": "General knowledge from English Wikipedia. Good all-around starting point.",
        "source_bank": [
            {"name": "wiki", "source": "wikitext", "source_type": "hf", "hf_config": "wikitext-103-raw-v1", "text_field": "text"},
        ],
        "tick_tokens": 512,
        "sleep_interval_seconds": 0.05,
        "repeat_sources": True,
        "model_overrides": {"n_columns": 1024, "enable_binding_layer": True, "binding_mode": "hypercube", "routing_shards": 4, "plasticity_spike_backend": "adex"},
    },
    "wikipedia_news": {
        "label": "Wikipedia + News",
        "description": "Wikipedia paired with AG News for broader topic coverage.",
        "source_bank": [
            {"name": "wiki", "source": "wikitext", "source_type": "hf", "hf_config": "wikitext-103-raw-v1", "text_field": "text"},
            {"name": "news", "source": "ag_news", "source_type": "hf", "hf_config": None, "text_field": "text"},
        ],
        "tick_tokens": 512,
        "sleep_interval_seconds": 0.05,
        "repeat_sources": True,
        "model_overrides": {"n_columns": 1024, "enable_binding_layer": True, "binding_mode": "hypercube", "routing_shards": 4, "plasticity_spike_backend": "adex"},
    },
    "diverse": {
        "label": "Diverse (Wiki + News + Reviews)",
        "description": "Three domains for maximum coverage: Wikipedia, AG News, and IMDB reviews.",
        "source_bank": [
            {"name": "wiki", "source": "wikitext", "source_type": "hf", "hf_config": "wikitext-103-raw-v1", "text_field": "text"},
            {"name": "news", "source": "ag_news", "source_type": "hf", "hf_config": None, "text_field": "text"},
            {"name": "reviews", "source": "imdb", "source_type": "hf", "hf_config": None, "text_field": "text"},
        ],
        "tick_tokens": 512,
        "sleep_interval_seconds": 0.05,
        "repeat_sources": True,
        "model_overrides": {"n_columns": 1024, "enable_binding_layer": True, "binding_mode": "hypercube", "routing_shards": 4, "plasticity_spike_backend": "adex"},
    },
    "diverse_fast": {
        "label": "Diverse — Fast (Wiki + News + Reviews)",
        "description": "Same three domains but with larger batches and shorter sleep for faster throughput.",
        "source_bank": [
            {"name": "wiki", "source": "wikitext", "source_type": "hf", "hf_config": "wikitext-103-raw-v1", "text_field": "text"},
            {"name": "news", "source": "ag_news", "source_type": "hf", "hf_config": None, "text_field": "text"},
            {"name": "reviews", "source": "imdb", "source_type": "hf", "hf_config": None, "text_field": "text"},
        ],
        "tick_tokens": 1024,
        "sleep_interval_seconds": 0.02,
        "repeat_sources": True,
        "model_overrides": {"n_columns": 2048, "enable_binding_layer": True, "binding_mode": "hypercube", "routing_shards": 8, "plasticity_spike_backend": "adex"},
    },
    "multimodal": {
        "label": "Multimodal (Wiki + N-MNIST + FSDD)",
        "description": "Text interleaved with N-MNIST visual and FSDD audio digit episodes. Trains all three modalities.",
        "source_bank": [
            {"name": "wiki", "source": "wikitext", "source_type": "hf", "hf_config": "wikitext-103-raw-v1", "text_field": "text"},
        ],
        "multimodal": {
            "enabled": True,
            "nmnist_dir": "N-MNIST",
            "fsdd_dir": "free-spoken-digit-dataset-master",
            "episode_interval_tokens": 256,
            "n_steps": 10,
        },
        "tick_tokens": 512,
        "sleep_interval_seconds": 0.05,
        "repeat_sources": True,
        "model_overrides": {"n_columns": 1024, "enable_binding_layer": True, "binding_mode": "hypercube", "routing_shards": 4, "plasticity_spike_backend": "adex", "enable_cross_modal": True, "cross_modal_dim_visual": 64, "cross_modal_dim_audio": 64},
    },
    "multimodal_fast": {
        "label": "Multimodal — Fast",
        "description": "Faster multimodal training with larger batches. All three modalities active.",
        "source_bank": [
            {"name": "wiki", "source": "wikitext", "source_type": "hf", "hf_config": "wikitext-103-raw-v1", "text_field": "text"},
        ],
        "multimodal": {
            "enabled": True,
            "nmnist_dir": "N-MNIST",
            "fsdd_dir": "free-spoken-digit-dataset-master",
            "episode_interval_tokens": 128,
            "n_steps": 10,
        },
        "tick_tokens": 1024,
        "sleep_interval_seconds": 0.02,
        "repeat_sources": True,
        "model_overrides": {"n_columns": 2048, "enable_binding_layer": True, "binding_mode": "hypercube", "routing_shards": 8, "plasticity_spike_backend": "adex", "enable_cross_modal": True, "cross_modal_dim_visual": 64, "cross_modal_dim_audio": 64},
    },
}
