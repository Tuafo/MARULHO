from __future__ import annotations

from copy import deepcopy
from typing import Any
from urllib.parse import quote


def _preset(*parts: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for part in parts:
        merged.update(deepcopy(part))
    return merged


def _wikipedia_revision_url(title: str, *, oldid: int) -> str:
    page_title = quote(title.replace(" ", "_"), safe="_")
    return f"https://en.wikipedia.org/w/index.php?title={page_title}&oldid={int(oldid)}&action=raw"


def _wikipedia_revision_source(
    *,
    name: str,
    title: str,
    oldid: int,
    summary: str,
    tags: list[str],
    catalog_priority: float,
) -> dict[str, Any]:
    return {
        "name": name,
        "source": _wikipedia_revision_url(title, oldid=oldid),
        "source_type": "web",
        "hf_config": None,
        "text_field": "text",
        "title": title,
        "summary": summary,
        "tags": [*tags, "wikipedia", "revision"],
        "catalog_priority": catalog_priority,
        "provider": "wikipedia_revision",
    }


def _wikipedia_revision_seed(
    *,
    name: str,
    title: str,
    oldid: int,
) -> dict[str, Any]:
    return {
        "name": name,
        "source": _wikipedia_revision_url(title, oldid=oldid),
        "source_type": "web",
        "hf_config": None,
        "text_field": "text",
    }


_WIKITEXT_HF_SOURCE = {
    "source": "wikitext",
    "hf_config": "wikitext-103-raw-v1",
    "text_field": "text",
}

_WIKITEXT_HF_STREAM_SOURCE = _preset(
    _WIKITEXT_HF_SOURCE,
    {"source_type": "hf"},
)

_NEWS_WIKI_HF_TASKS = {
    "task_a_source": "ag_news",
    "task_a_hf_config": None,
    "task_a_text_field": "text",
    "task_b_source": "wikitext",
    "task_b_hf_config": "wikitext-103-raw-v1",
    "task_b_text_field": "text",
}

_COMMON_MARULHO_PRESET = {
    "seed": 7,
    "n_columns": 256,
    "column_latent_dim": 256,
    "memory_capacity": 1000,
    "input_weight_blend": 0.02,
    "input_synapse_ltp": 0.02,
    "input_synapse_ltd": 0.01,
    "input_weight_row_target": 1.0,
    "homeostasis_beta": 0.01,
    "homeostasis_lr": 0.2,
    "enable_binding_layer": True,
    "binding_mode": "hypercube",
}

_COMMON_SLEEP_PRESET = {
    "slow_mean_decay": 0.9999,
    "use_winner_local_drift": True,
    "drift_threshold": 0.02,
    "micro_sleep_interval_tokens": 200,
    "micro_sleep_replay_steps": 10,
    "micro_sleep_candidate_pool": 5,
    "micro_sleep_memory_blend": 0.05,
    "deep_sleep_interval_tokens": 2500,
    "deep_sleep_replay_steps": 200,
    "deep_sleep_candidate_pool": 100,
    "deep_sleep_memory_blend": 0.20,
    "deep_sleep_cooldown_tokens": 1000,
    "emergency_deep_sleep_cooldown_tokens": 1000,
    "drift_floor_history_tokens": 1000,
    "drift_floor_check_interval_tokens": 200,
    "drift_floor_window_tokens": 10000,
    "drift_floor_trigger_min_tokens": 1000,
    "drift_floor_rise_tolerance": 0.0,
    "prototype_momentum": 0.85,
}

_BASE_MARULHO_PRESET = _preset(_COMMON_MARULHO_PRESET, _COMMON_SLEEP_PRESET)

_BASE_MECHANISM_VALIDATION_PRESET = _preset(
    _BASE_MARULHO_PRESET,
    {"slow_memory_start_tokens": 0},
)

_CONTEXT_BINDING_PRESET = {
    "enable_context_layer": True,
    "enable_binding_layer": True,
    "context_mode": "adaptive",
    "plasticity_mode": "local_stdp",
    "plasticity_rule": "triplet",
    "context_decay": 0.92,
    "context_transition_lr": 0.05,
    "context_modulation_strength": 0.60,
    "binding_threshold": 0.02,
    "binding_association_lr": 0.20,
    "binding_association_decay": 0.995,
    "binding_gain_strength": 0.80,
}

_BASE_CONTEXT_MARULHO_PRESET = _preset(_BASE_MARULHO_PRESET, _CONTEXT_BINDING_PRESET)

_MEMORY_CONSOLIDATION_SMOKE_SETTINGS = {
    "task_boundary_tag_strength": 3.0,  # Increased from 1.5 to match baseline - tag must survive decay
    "task_boundary_anchor_strength": 8.0,  # Increased from 2.0 to drive blend to cap (0.35)
    "task_boundary_consolidation_cycles": 6,  # Increased from 2 for stronger anchor reinforcement
    "consolidation_mode": "deep",
    "consolidation_cycles": 12,  # Increased from 5 for more post-B recovery
    "task_boundary_replay_repair_strength_schedule": [0.1, 0.05, 0.02, 0.01],
    "consolidation_replay_repair_strength_schedule": [0.1, 0.05, 0.02, 0.01],
}

_MEMORY_CONSOLIDATION_BASELINE_SETTINGS = {
    "task_boundary_tag_strength": 3.0,
    "task_boundary_anchor_strength": 8.0,
    "task_boundary_consolidation_cycles": 6,
    "consolidation_mode": "deep",
    "consolidation_cycles": 12,
    "task_boundary_replay_repair_strength_schedule": [0.1, 0.05, 0.02, 0.01],
    "consolidation_replay_repair_strength_schedule": [0.1, 0.05, 0.02, 0.01],
}

_HIERARCHICAL_SCALE_BASE_PRESET = _preset(
    _BASE_MARULHO_PRESET,
    {
        "index_rebuild_threshold": 128,
        "shard_candidate_factor": 2,
        "neurons_per_column_assumption": 100,
    },
)

_AUTONOMY_TUNING_PRESET = {
    "warmup_rounds": 1,
    "gap_exploration_bonus": 0.02,
    "gap_ambiguity_weight": 0.08,
    "gap_switch_weight": 0.08,
    "gap_margin_reference": 0.22,  # Increased from 0.12 to reactivate ambiguity signal
    "coverage_balance_penalty": 0.02,
    "gap_focus_margin": 0.02,
}


MECHANISM_VALIDATION_PRESETS: dict[str, dict[str, Any]] = {
    "mechanism_validation_active_20k_hf": _preset(
        _WIKITEXT_HF_STREAM_SOURCE,
        _BASE_MECHANISM_VALIDATION_PRESET,
        {
            "train_tokens": 20000,
            "eval_tokens": 2000,
            "log_every": 5000,
        },
    ),
    "mechanism_validation_active_20k_hf_unigram": _preset(
        _WIKITEXT_HF_STREAM_SOURCE,
        _BASE_MECHANISM_VALIDATION_PRESET,
        {
            "train_tokens": 20000,
            "eval_tokens": 2000,
            "log_every": 5000,
            "input_representation": "unigram_ascii",
        },
    ),
    "mechanism_validation_active_20k_hf_hashed": _preset(
        _WIKITEXT_HF_STREAM_SOURCE,
        _BASE_MECHANISM_VALIDATION_PRESET,
        {
            "train_tokens": 20000,
            "eval_tokens": 2000,
            "log_every": 5000,
            "input_representation": "hashed_ngram",
            "hashed_ngram_dim": 2048,
            "hashed_ngram_min_n": 2,
            "hashed_ngram_max_n": 3,
        },
    ),
    "mechanism_validation_active_100k_hf": _preset(
        _WIKITEXT_HF_STREAM_SOURCE,
        _BASE_MECHANISM_VALIDATION_PRESET,
        {
            "train_tokens": 100000,
            "eval_tokens": 5000,
            "log_every": 5000,
        },
    ),
}


MEMORY_CONSOLIDATION_PRESETS: dict[str, dict[str, Any]] = {
    "memory_consolidation_hf_smoke": _preset(
        _NEWS_WIKI_HF_TASKS,
        _BASE_MARULHO_PRESET,
        _MEMORY_CONSOLIDATION_SMOKE_SETTINGS,
        {
            "task_a_train_tokens": 2000,
            "task_b_train_tokens": 2000,
            "eval_tokens": 500,
        },
    ),
    "memory_consolidation_hf_baseline": _preset(
        _NEWS_WIKI_HF_TASKS,
        _BASE_MARULHO_PRESET,
        _MEMORY_CONSOLIDATION_BASELINE_SETTINGS,
        {
            "task_a_train_tokens": 10000,
            "task_b_train_tokens": 10000,
            "eval_tokens": 2000,
        },
    ),
    "memory_consolidation_hf_scale_robust": _preset(
        _NEWS_WIKI_HF_TASKS,
        _BASE_MARULHO_PRESET,
        _MEMORY_CONSOLIDATION_BASELINE_SETTINGS,
        {
            "task_a_train_tokens": 10000,
            "task_b_train_tokens": 10000,
            "eval_tokens": 2000,
            "consolidation_cycles": 0,
        },
    ),
}


CONTEXTUAL_ROUTING_PRESETS: dict[str, dict[str, Any]] = {
    "contextual_routing_hf_smoke": _preset(
        _NEWS_WIKI_HF_TASKS,
        _BASE_CONTEXT_MARULHO_PRESET,
        {
            "task_a_train_tokens": 4000,
            "task_b_train_tokens": 4000,
            "eval_tokens": 1000,
            "context_block_tokens": 250,
            "prime_tokens": 128,
            "probe_tokens": 256,
        },
    ),
    "contextual_routing_hf_baseline": _preset(
        _NEWS_WIKI_HF_TASKS,
        _BASE_CONTEXT_MARULHO_PRESET,
        {
            "task_a_train_tokens": 10000,
            "task_b_train_tokens": 10000,
            "eval_tokens": 2000,
            "context_block_tokens": 250,
            "prime_tokens": 256,
            "probe_tokens": 512,
        },
    ),
}


HIERARCHICAL_SCALE_PRESETS: dict[str, dict[str, Any]] = {
    "hierarchical_scale_hf_smoke": _preset(
        _WIKITEXT_HF_SOURCE,
        _HIERARCHICAL_SCALE_BASE_PRESET,
        {
            "train_tokens": 12000,
            "eval_tokens": 1500,
            "routing_eval_queries": 256,
            "latency_eval_queries": 128,
            "n_columns": 256,
            "k_routing": 12,
            "routing_shards": 4,
        },
    ),
    "hierarchical_scale_hf_baseline": _preset(
        _WIKITEXT_HF_SOURCE,
        _HIERARCHICAL_SCALE_BASE_PRESET,
        {
            "train_tokens": 30000,
            "eval_tokens": 4000,
            "routing_eval_queries": 512,
            "latency_eval_queries": 256,
            "n_columns": 1024,
            "k_routing": 16,
            "routing_shards": 8,
            "memory_capacity": 2000,
        },
    ),
}


_AUTONOMY_HF_SOURCE_REGISTRY: list[dict[str, Any]] = [
    {
        "name": "news",
        "source": "ag_news",
        "source_type": "hf",
        "hf_config": None,
        "text_field": "text",
        "title": "AG News",
        "summary": "News headlines and articles with temporal events, entities, and topic shifts.",
        "tags": ["news", "events", "temporal", "entities"],
        "catalog_priority": 0.75,
    },
    {
        "name": "wiki",
        "source": "wikitext",
        "source_type": "hf",
        "hf_config": "wikitext-103-raw-v1",
        "text_field": "text",
        "title": "WikiText",
        "summary": "Encyclopedic long-form writing with definitions, relations, and broad factual structure.",
        "tags": ["knowledge", "definitions", "relations", "encyclopedic"],
        "catalog_priority": 0.90,
    },
    {
        "name": "reviews",
        "source": "imdb",
        "source_type": "hf",
        "hf_config": None,
        "text_field": "text",
        "title": "IMDB Reviews",
        "summary": "Long subjective movie reviews with opinion-rich narrative context and sentiment drift.",
        "tags": ["reviews", "sentiment", "narrative", "subjective"],
        "catalog_priority": 0.35,
    },
    {
        "name": "dbpedia",
        "source": "dbpedia_14",
        "source_type": "hf",
        "hf_config": None,
        "text_field": "content",
        "title": "DBPedia 14",
        "summary": "Entity-centric encyclopedia descriptions organized by taxonomy and factual type structure.",
        "tags": ["entities", "taxonomy", "facts", "knowledge"],
        "catalog_priority": 0.80,
    },
    {
        "name": "yelp",
        "source": "yelp_polarity",
        "source_type": "hf",
        "hf_config": None,
        "text_field": "text",
        "title": "Yelp Polarity",
        "summary": "Restaurant reviews with grounded consumer language, sentiment, and local detail.",
        "tags": ["reviews", "sentiment", "restaurants", "grounded"],
        "catalog_priority": 0.30,
    },
    {
        "name": "amazon",
        "source": "amazon_polarity",
        "source_type": "hf",
        "hf_config": None,
        "text_field": "content",
        "title": "Amazon Polarity",
        "summary": "Product reviews with broad topical coverage, grounded descriptions, and preference signals.",
        "tags": ["reviews", "products", "sentiment", "grounded"],
        "catalog_priority": 0.30,
    },
]

_AUTONOMY_HF_SOURCE_BANK: list[dict[str, Any]] = [
    {"name": "news", "source": "ag_news", "source_type": "hf", "hf_config": None, "text_field": "text"},
    {"name": "wiki", "source": "wikitext", "source_type": "hf", "hf_config": "wikitext-103-raw-v1", "text_field": "text"},
    {"name": "reviews", "source": "imdb", "source_type": "hf", "hf_config": None, "text_field": "text"},
]

_AUTONOMY_ACQUISITION_HF_SEED_BANK: list[dict[str, Any]] = [
    {"name": "news", "source": "ag_news", "source_type": "hf", "hf_config": None, "text_field": "text"},
    {"name": "wiki", "source": "wikitext", "source_type": "hf", "hf_config": "wikitext-103-raw-v1", "text_field": "text"},
]

_AUTONOMY_ACQUISITION_HF_PAIR_CANDIDATE_BANK: list[dict[str, Any]] = [
    {
        "catalog_mode": "semantic_registry",
        "catalog_entries": [entry for entry in _AUTONOMY_HF_SOURCE_REGISTRY if entry["name"] in {"yelp", "reviews"}],
        "catalog_limit": 2,
        "catalog_semantic_weight": 1.2,
        "catalog_prior_weight": 1.0,
        "catalog_diversity_weight": 0.15,
    }
]

_AUTONOMY_ACQUISITION_HF_ALLOCATION_CANDIDATE_BANK: list[dict[str, Any]] = [
    {
        "catalog_mode": "semantic_registry",
        "catalog_entries": [entry for entry in _AUTONOMY_HF_SOURCE_REGISTRY if entry["name"] in {"yelp", "dbpedia", "reviews"}],
        "catalog_limit": 3,
        "catalog_semantic_weight": 1.2,
        "catalog_prior_weight": 1.0,
        "catalog_diversity_weight": 0.18,
    }
]

_AUTONOMY_ACQUISITION_HF_CATALOG_CANDIDATE_BANK: list[dict[str, Any]] = [
    {
        "catalog_mode": "semantic_registry",
        "catalog_entries": [entry for entry in _AUTONOMY_HF_SOURCE_REGISTRY if entry["name"] in {"yelp", "amazon", "dbpedia", "reviews"}],
        "catalog_limit": 3,
        "catalog_probe_pool_limit": 4,
        "catalog_probe_tokens": 48,
        "catalog_scout_tokens": 192,
        "catalog_semantic_weight": 1.25,
        "catalog_prior_weight": 1.0,
        "catalog_diversity_weight": 0.20,
    }
]

_AUTONOMY_CURATED_WEB_SOURCE_REGISTRY: list[dict[str, Any]] = [
    _wikipedia_revision_source(
        name="wiki_synaptic_plasticity",
        title="Synaptic plasticity",
        oldid=1341547585,
        summary="Pinned Wikipedia revision covering synapse-level adaptation, learning, and memory consolidation.",
        tags=["neuroscience", "plasticity", "learning", "memory"],
        catalog_priority=0.92,
    ),
    _wikipedia_revision_source(
        name="wiki_hebbian_theory",
        title="Hebbian theory",
        oldid=1335815717,
        summary="Pinned Wikipedia revision covering associative learning, co-activation, and synaptic strengthening.",
        tags=["hebbian", "learning", "associations", "synapses"],
        catalog_priority=0.82,
    ),
    _wikipedia_revision_source(
        name="wiki_long_term_potentiation",
        title="Long-term potentiation",
        oldid=1313766320,
        summary="Pinned Wikipedia revision covering durable synaptic strengthening and hippocampal memory experiments.",
        tags=["potentiation", "hippocampus", "memory", "plasticity"],
        catalog_priority=0.78,
    ),
    _wikipedia_revision_source(
        name="wiki_long_term_depression",
        title="Long-term depression",
        oldid=1343045585,
        summary="Pinned Wikipedia revision covering durable synaptic weakening, plasticity tradeoffs, and memory updating.",
        tags=["depression", "plasticity", "synapses", "memory"],
        catalog_priority=0.76,
    ),
    _wikipedia_revision_source(
        name="wiki_spike_timing_dependent_plasticity",
        title="Spike-timing-dependent plasticity",
        oldid=1319959459,
        summary="Pinned Wikipedia revision covering spike-timing rules that link precise temporal structure to synaptic change.",
        tags=["timing", "spikes", "plasticity", "learning"],
        catalog_priority=0.74,
    ),
]

_AUTONOMY_ACQUISITION_CURATED_WEB_SEED_BANK: list[dict[str, Any]] = [
    _wikipedia_revision_seed(
        name="wiki_neuroscience",
        title="Neuroscience",
        oldid=1344865459,
    ),
    _wikipedia_revision_seed(
        name="wiki_memory",
        title="Memory",
        oldid=1345761939,
    ),
]

_AUTONOMY_ACQUISITION_CURATED_WEB_CANDIDATE_BANK: list[dict[str, Any]] = [
    {
        "catalog_mode": "semantic_registry",
        "catalog_entries": [entry for entry in _AUTONOMY_CURATED_WEB_SOURCE_REGISTRY],
        "catalog_limit": 3,
        "catalog_probe_pool_limit": 5,
        "catalog_probe_tokens": 48,
        "catalog_scout_tokens": 192,
        "catalog_semantic_weight": 1.25,
        "catalog_prior_weight": 1.0,
        "catalog_diversity_weight": 0.20,
    }
]

_AUTONOMY_HF_BASE_PRESET = _preset(
    {"source_bank": _AUTONOMY_HF_SOURCE_BANK},
    _BASE_CONTEXT_MARULHO_PRESET,
    _AUTONOMY_TUNING_PRESET,
)

_AUTONOMY_ACQUISITION_HF_PAIR_PRESET: dict[str, Any] = _preset(
    {
        "seed_bank": _AUTONOMY_ACQUISITION_HF_SEED_BANK,
        "candidate_bank": _AUTONOMY_ACQUISITION_HF_PAIR_CANDIDATE_BANK,
    },
    _BASE_CONTEXT_MARULHO_PRESET,
    _AUTONOMY_TUNING_PRESET,
    {
        "seed_train_tokens": 8000,
        "candidate_train_tokens": 8000,
        "probe_tokens": 256,
        "acquisition_tokens": 2000,
        "acquisition_slots": 1,
    },
)

_AUTONOMY_ACQUISITION_HF_ALLOCATION_PRESET: dict[str, Any] = _preset(
    {
        "seed_bank": _AUTONOMY_ACQUISITION_HF_SEED_BANK,
        "candidate_bank": _AUTONOMY_ACQUISITION_HF_ALLOCATION_CANDIDATE_BANK,
    },
    _BASE_CONTEXT_MARULHO_PRESET,
    _AUTONOMY_TUNING_PRESET,
    {
        "seed_train_tokens": 8000,
        "candidate_train_tokens": 6000,
        "probe_tokens": 256,
        "acquisition_tokens": 2000,
        "acquisition_slots": 2,
    },
)

_AUTONOMY_ACQUISITION_HF_CATALOG_PRESET: dict[str, Any] = _preset(
    {
        "seed_bank": _AUTONOMY_ACQUISITION_HF_SEED_BANK,
        "candidate_bank": _AUTONOMY_ACQUISITION_HF_CATALOG_CANDIDATE_BANK,
    },
    _BASE_CONTEXT_MARULHO_PRESET,
    _AUTONOMY_TUNING_PRESET,
    {
        "seed_train_tokens": 8000,
        "candidate_train_tokens": 4000,
        "probe_tokens": 160,
        "acquisition_tokens": 1500,
        "acquisition_slots": 3,
        "semantic_shortlist_size": 0,
        "semantic_shortlist_gap_weight": 0.35,
        "semantic_shortlist_affinity_weight": 0.65,
    },
)

_AUTONOMY_ACQUISITION_CURATED_WEB_SMOKE_PRESET: dict[str, Any] = _preset(
    {
        "seed_bank": _AUTONOMY_ACQUISITION_CURATED_WEB_SEED_BANK,
        "candidate_bank": _AUTONOMY_ACQUISITION_CURATED_WEB_CANDIDATE_BANK,
    },
    _BASE_CONTEXT_MARULHO_PRESET,
    _AUTONOMY_TUNING_PRESET,
    {
        "seed_train_tokens": 4000,
        "candidate_train_tokens": 2000,
        "probe_tokens": 96,
        "acquisition_tokens": 800,
        "acquisition_slots": 2,
        "semantic_shortlist_size": 0,
        "semantic_shortlist_gap_weight": 0.35,
        "semantic_shortlist_affinity_weight": 0.65,
    },
)

REPRESENTATION_PRESETS: dict[str, dict[str, Any]] = {
    "representation_hf_smoke": _preset(
        _WIKITEXT_HF_STREAM_SOURCE,
        {
            "train_tokens": 4000,
            "eval_tokens": 1000,
            "window_size": 10,
            "representations": ["order_weighted_ascii", "unigram_ascii", "hashed_ngram"],
            "hashed_ngram_dim": 2048,
            "hashed_ngram_min_n": 2,
            "hashed_ngram_max_n": 3,
            "n_columns": 64,
            "column_latent_dim": 256,
            "memory_capacity": 256,
            "baseline_clusters": 32,
            "probe_samples": 128,
            "seed": 7,
        },
    ),
}


AUTONOMY_PRESETS: dict[str, dict[str, Any]] = {
    "autonomy_hf_smoke": _preset(
        _AUTONOMY_HF_BASE_PRESET,
        {
            "source_train_tokens": 4000,
            "probe_tokens": 128,
            "episode_tokens": 500,
            "seek_episodes": 8,
            "coverage_balance_penalty": 0.01,
            "gap_focus_margin": 0.01,
        },
    ),
    "autonomy_hf_baseline": _preset(
        _AUTONOMY_HF_BASE_PRESET,
        {
            "source_train_tokens": 12000,
            "probe_tokens": 256,
            "episode_tokens": 1000,
            "seek_episodes": 9,
        },
    ),
}


AUTONOMY_ACQUISITION_PRESETS: dict[str, dict[str, Any]] = {
    "autonomy_acquisition_curated_web_smoke": deepcopy(_AUTONOMY_ACQUISITION_CURATED_WEB_SMOKE_PRESET),
    "autonomy_acquisition_hf_baseline": deepcopy(_AUTONOMY_ACQUISITION_HF_PAIR_PRESET),  # Merged from _smoke (was identical)
    "autonomy_acquisition_hf_allocation": deepcopy(_AUTONOMY_ACQUISITION_HF_ALLOCATION_PRESET),
    "autonomy_acquisition_hf_catalog": deepcopy(_AUTONOMY_ACQUISITION_HF_CATALOG_PRESET),
}


def _get_preset(presets: dict[str, dict[str, Any]], name: str | None, kind: str) -> dict[str, Any]:
    if name is None:
        return {}
    if name not in presets:
        raise KeyError(f"Unknown {kind} preset: {name}")
    return deepcopy(presets[name])


def get_mechanism_validation_preset(name: str | None) -> dict[str, Any]:
    return _get_preset(MECHANISM_VALIDATION_PRESETS, name, "mechanism validation")


def get_representation_preset(name: str | None) -> dict[str, Any]:
    return _get_preset(REPRESENTATION_PRESETS, name, "representation")


def get_memory_consolidation_preset(name: str | None) -> dict[str, Any]:
    return _get_preset(MEMORY_CONSOLIDATION_PRESETS, name, "memory consolidation")


def get_contextual_routing_preset(name: str | None) -> dict[str, Any]:
    return _get_preset(CONTEXTUAL_ROUTING_PRESETS, name, "contextual routing")


def get_hierarchical_scale_preset(name: str | None) -> dict[str, Any]:
    return _get_preset(HIERARCHICAL_SCALE_PRESETS, name, "hierarchical scale")


def get_autonomy_preset(name: str | None) -> dict[str, Any]:
    return _get_preset(AUTONOMY_PRESETS, name, "autonomy")


def get_autonomy_acquisition_preset(name: str | None) -> dict[str, Any]:
    return _get_preset(AUTONOMY_ACQUISITION_PRESETS, name, "autonomy acquisition")


def mechanism_validation_preset_names() -> list[str]:
    return sorted(MECHANISM_VALIDATION_PRESETS)


def representation_preset_names() -> list[str]:
    return sorted(REPRESENTATION_PRESETS)


def memory_consolidation_preset_names() -> list[str]:
    return sorted(MEMORY_CONSOLIDATION_PRESETS)


def contextual_routing_preset_names() -> list[str]:
    return sorted(CONTEXTUAL_ROUTING_PRESETS)


def hierarchical_scale_preset_names() -> list[str]:
    return sorted(HIERARCHICAL_SCALE_PRESETS)


def autonomy_preset_names() -> list[str]:
    return sorted(AUTONOMY_PRESETS)


def autonomy_acquisition_preset_names() -> list[str]:
    return sorted(AUTONOMY_ACQUISITION_PRESETS)
