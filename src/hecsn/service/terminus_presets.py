"""Terminus quick-start presets -- predefined configurations for brain training.

Each preset specifies:
- source_bank: HuggingFace datasets for text training
- multimodal: optional visual + audio episode configuration
- curriculum: NIM-driven curriculum generation (replaces static datasets)
- tick_tokens: tokens processed per brain tick
- model_overrides: HECSNConfig overrides (columns, binding, spike backend)

The default preset is 'curriculum' -- uses NIM cortex to generate diverse
training episodes on demand, covering science, nature, technology, history,
and more. The SNN learns grounding from these rich multimodal episodes.

Wikipedia-only presets are retained for backwards compatibility but are not
recommended (the LLM cortex already has text knowledge; the SNN should focus
on multimodal grounding, not memorizing text facts).
"""

from __future__ import annotations

from typing import Any


TERMINUS_QUICK_START_PRESETS: dict[str, dict[str, Any]] = {
    # ===== RECOMMENDED PRESETS =====

    "curriculum": {
        "label": "Curriculum (NIM-driven, recommended)",
        "default": True,
        "description": (
            "NIM cortex generates diverse training episodes on demand. "
            "Covers science, nature, technology, history, art, and more. "
            "No static dataset download needed."
        ),
        "source_bank": [
            # AG News provides broad topic seeds for curriculum generation
            {"name": "news", "source": "ag_news", "source_type": "hf", "hf_config": None, "text_field": "text"},
        ],
        "curriculum": {
            "enabled": True,
            "topics_per_cycle": 3,
            "episode_length_tokens": 256,
            "diversity_threshold": 0.7,
        },
        "tick_tokens": 512,
        "sleep_interval_seconds": 0.05,
        "repeat_sources": True,
        "model_overrides": {
            "n_columns": 1024,
            "enable_binding_layer": True,
            "binding_mode": "hypercube",
            "routing_shards": 4,
            "plasticity_spike_backend": "adex",
        },
    },

    "multimodal": {
        "label": "Multimodal (Curriculum + Visual + Audio)",
        "description": (
            "NIM-driven curriculum text interleaved with visual and audio episodes. "
            "Trains all three modalities with cross-modal grounding."
        ),
        "source_bank": [
            # Diverse text seeds -- NOT wikitext (cortex already has that knowledge)
            {"name": "news", "source": "ag_news", "source_type": "hf", "hf_config": None, "text_field": "text"},
            {"name": "reviews", "source": "imdb", "source_type": "hf", "hf_config": None, "text_field": "text"},
        ],
        "curriculum": {
            "enabled": True,
            "topics_per_cycle": 3,
            "episode_length_tokens": 256,
            "diversity_threshold": 0.7,
        },
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

    "multimodal_fast": {
        "label": "Multimodal -- Fast",
        "description": "Faster multimodal training with larger batches. All three modalities active.",
        "source_bank": [
            {"name": "news", "source": "ag_news", "source_type": "hf", "hf_config": None, "text_field": "text"},
        ],
        "curriculum": {
            "enabled": True,
            "topics_per_cycle": 4,
            "episode_length_tokens": 192,
            "diversity_threshold": 0.6,
        },
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
        "model_overrides": {
            "n_columns": 2048,
            "enable_binding_layer": True,
            "binding_mode": "hypercube",
            "routing_shards": 8,
            "plasticity_spike_backend": "adex",
            "enable_cross_modal": True,
            "cross_modal_dim_visual": 64,
            "cross_modal_dim_audio": 64,
        },
    },

    # ===== LEGACY PRESETS (backwards compatibility) =====

    "wikipedia": {
        "label": "Wikipedia (wikitext-103, legacy)",
        "legacy": True,
        "description": "General knowledge from English Wikipedia. Legacy -- prefer 'curriculum' preset.",
        "source_bank": [
            {"name": "wiki", "source": "wikitext", "source_type": "hf", "hf_config": "wikitext-103-raw-v1", "text_field": "text"},
        ],
        "tick_tokens": 512,
        "sleep_interval_seconds": 0.05,
        "repeat_sources": True,
        "model_overrides": {
            "n_columns": 1024,
            "enable_binding_layer": True,
            "binding_mode": "hypercube",
            "routing_shards": 4,
            "plasticity_spike_backend": "adex",
        },
    },

    "diverse": {
        "label": "Diverse (News + Reviews + Sci, no wiki)",
        "legacy": True,
        "description": "Three diverse domains without Wikipedia. Better for SNN grounding.",
        "source_bank": [
            {"name": "news", "source": "ag_news", "source_type": "hf", "hf_config": None, "text_field": "text"},
            {"name": "reviews", "source": "imdb", "source_type": "hf", "hf_config": None, "text_field": "text"},
            {"name": "sci", "source": "scientific_papers", "source_type": "hf", "hf_config": "arxiv", "text_field": "abstract"},
        ],
        "tick_tokens": 512,
        "sleep_interval_seconds": 0.05,
        "repeat_sources": True,
        "model_overrides": {
            "n_columns": 1024,
            "enable_binding_layer": True,
            "binding_mode": "hypercube",
            "routing_shards": 4,
            "plasticity_spike_backend": "adex",
        },
    },

    "diverse_fast": {
        "label": "Diverse -- Fast",
        "legacy": True,
        "description": "Fast diverse training. Larger batches, shorter sleep.",
        "source_bank": [
            {"name": "news", "source": "ag_news", "source_type": "hf", "hf_config": None, "text_field": "text"},
            {"name": "reviews", "source": "imdb", "source_type": "hf", "hf_config": None, "text_field": "text"},
        ],
        "curriculum": {
            "enabled": True,
            "topics_per_cycle": 5,
            "episode_length_tokens": 128,
            "diversity_threshold": 0.5,
        },
        "tick_tokens": 1024,
        "sleep_interval_seconds": 0.02,
        "repeat_sources": True,
        "model_overrides": {
            "n_columns": 2048,
            "enable_binding_layer": True,
            "binding_mode": "hypercube",
            "routing_shards": 8,
            "plasticity_spike_backend": "adex",
        },
    },
}
