"""Terminus runtime configuration normalization helpers.

This module keeps source-bank and runtime configuration validation separate
from the service manager facade. It does not start training, call tools, or
mutate replay datasets; it only turns operator config into normalized dicts.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Mapping, Sequence, cast

from marulho.service.terminus_autonomy import (
    AUTO_REMOTE_PROVIDER_QUERY_FAMILY_LIMIT,
    AUTO_REMOTE_PROVIDER_TOPIC_TERM_LIMIT,
    DEFAULT_AUTONOMY_REMOTE_CATALOG_LIMIT,
    DEFAULT_AUTONOMY_REMOTE_PROBE_POOL_LIMIT,
    DEFAULT_AUTONOMY_REMOTE_PROVIDERS,
    DEFAULT_AUTONOMY_REMOTE_PROVIDER_RESULT_LIMIT,
    DEFAULT_AUTONOMY_REMOTE_QUERIES_PER_PROVIDER,
    _canonical_provider_term,
)

PUBLIC_ACQUISITION_POLICIES: tuple[str, ...] = ("active", "round_robin")
DEFAULT_BRAIN_TICK_TOKENS = 128
DEFAULT_BRAIN_SLEEP_INTERVAL_SECONDS = 0.01
DEFAULT_EXECUTION_QUANTUM_TOKENS = 8
MAX_EXECUTION_QUANTUM_TOKENS = 128
DEFAULT_EXECUTION_YIELD_SECONDS = 0.0
DEFAULT_AUTONOMY_TRIGGER_INTERVAL_TOKENS = 4096
DEFAULT_INGESTION_QUEUE_MULTIPLIER = 2


class RuntimeConfig:
    """Stateless normalization gate for operator runtime configuration."""

    def __init__(
        self,
        *,
        provider_query_family_priority: Callable[[Mapping[str, Any]], float] | None = None,
        provider_topic_family_priority: Callable[[Mapping[str, Any]], float] | None = None,
    ) -> None:
        self._provider_query_family_priority = provider_query_family_priority
        self._provider_topic_family_priority = provider_topic_family_priority

    def _provider_query_family_priority_locked(self, family_entry: Mapping[str, Any]) -> float:
        if self._provider_query_family_priority is None:
            return 0.0
        return float(self._provider_query_family_priority(family_entry))

    def _provider_topic_family_priority_locked(self, family_entry: Mapping[str, Any]) -> float:
        if self._provider_topic_family_priority is None:
            return 0.0
        return float(self._provider_topic_family_priority(family_entry))

    def _normalize_brain_source_spec(self, spec: Any, index: int) -> dict[str, Any]:
        if not isinstance(spec, dict):
            raise ValueError("Each Terminus source must be an object")
        source = str(spec.get("source", "")).strip()
        if not source:
            raise ValueError("Each Terminus source requires a non-empty source")
        source_type = str(spec.get("source_type", "auto")).strip() or "auto"
        if source_type not in {"auto", "file", "hf", "web"}:
            raise ValueError("Terminus sources only support source_type auto/file/hf/web")
        name = str(spec.get("name", f"source_{index + 1}")).strip() or f"source_{index + 1}"
        text_field = str(spec.get("text_field", "text")).strip() or "text"
        hf_config_raw = spec.get("hf_config")
        hf_config = None if hf_config_raw in (None, "", "None") else str(hf_config_raw)
        normalized = {
            "name": name,
            "source": source,
            "source_type": source_type,
            "text_field": text_field,
            "hf_config": hf_config,
        }
        topic_terms = spec.get("topic_terms")
        if isinstance(topic_terms, Sequence) and not isinstance(topic_terms, (str, bytes)):
            normalized["topic_terms"] = [
                _canonical_provider_term(term)
                for term in list(topic_terms)
                if _canonical_provider_term(term)
            ]
        metadata = spec.get("metadata")
        if isinstance(metadata, dict) and metadata:
            normalized["metadata"] = deepcopy(metadata)
        return normalized

    def _normalize_sensory_source_spec(self, spec: Any, index: int) -> dict[str, Any]:
        if not isinstance(spec, dict):
            raise ValueError("Each Terminus sensory source must be an object")
        adapter = str(spec.get("adapter", "")).strip().lower()
        if adapter not in {"s1_mmalign", "audiocaps"}:
            raise ValueError("Terminus sensory sources require adapter 's1_mmalign' or 'audiocaps'")
        source = str(spec.get("source", "")).strip()
        if not source:
            source = "ScienceOne-AI/S1-MMAlign" if adapter == "s1_mmalign" else "OpenSound/AudioCaps"
        name = str(spec.get("name", f"sensory_{index + 1}")).strip() or f"sensory_{index + 1}"
        split = str(spec.get("split", "train")).strip() or "train"
        normalized: dict[str, Any] = {
            "name": name,
            "adapter": adapter,
            "source": source,
            "split": split,
        }
        if adapter == "s1_mmalign":
            year_prefixes = spec.get("year_prefixes")
            if isinstance(year_prefixes, Sequence) and not isinstance(year_prefixes, (str, bytes)):
                normalized["year_prefixes"] = [
                    str(item).zfill(2)[:2]
                    for item in list(year_prefixes)
                    if str(item).strip()
                ] or ["07", "08", "09"]
            else:
                normalized["year_prefixes"] = ["07", "08", "09"]
            normalized["max_text_chars"] = max(64, int(spec.get("max_text_chars", 480)))
        else:
            normalized["sample_rate"] = max(1000, int(spec.get("sample_rate", 16000)))
            normalized["n_fft"] = max(64, int(spec.get("n_fft", 512)))
            normalized["max_text_chars"] = max(32, int(spec.get("max_text_chars", 240)))
            normalized["audio_candidates_per_item"] = max(1, int(spec.get("audio_candidates_per_item", 6)))
        topic_terms = spec.get("topic_terms")
        if isinstance(topic_terms, Sequence) and not isinstance(topic_terms, (str, bytes)):
            normalized["topic_terms"] = [
                " ".join(str(term).split()).strip().lower()
                for term in list(topic_terms)
                if " ".join(str(term).split()).strip()
            ]
        metadata = spec.get("metadata")
        if isinstance(metadata, dict) and metadata:
            normalized["metadata"] = deepcopy(metadata)
        return normalized

    def _normalize_catalog_candidate_spec(self, spec: Any, index: int) -> dict[str, Any]:
        if not isinstance(spec, dict):
            raise ValueError("Each Terminus autonomy candidate must be an object")
        catalog_mode = str(spec.get("catalog_mode", "")).strip().lower()
        if catalog_mode not in {"semantic_registry", "live_remote_search"}:
            raise ValueError(
                "Catalog-backed candidate specs require catalog_mode "
                "'semantic_registry' or 'live_remote_search'"
            )
        name = str(spec.get("name", f"{catalog_mode}_{index + 1}")).strip() or f"{catalog_mode}_{index + 1}"
        normalized: dict[str, Any] = {
            "name": name,
            "catalog_mode": catalog_mode,
            "catalog_limit": max(1, int(spec.get("catalog_limit", 8))),
            "catalog_diversity_weight": float(spec.get("catalog_diversity_weight", 0.20)),
            "catalog_semantic_weight": float(spec.get("catalog_semantic_weight", 1.0)),
            "catalog_prior_weight": float(spec.get("catalog_prior_weight", 1.0)),
            "catalog_provider_timeout_seconds": max(
                1.0,
                float(spec.get("catalog_provider_timeout_seconds", 15.0)),
            ),
        }
        if "catalog_probe_pool_limit" in spec and spec.get("catalog_probe_pool_limit") is not None:
            normalized["catalog_probe_pool_limit"] = max(1, int(spec.get("catalog_probe_pool_limit", 1)))
        focus_text = " ".join(str(spec.get("catalog_focus_text", "")).split()).strip()
        if focus_text:
            normalized["catalog_focus_text"] = focus_text
        focus_terms = spec.get("catalog_focus_terms")
        if isinstance(focus_terms, Sequence) and not isinstance(focus_terms, (str, bytes)):
            normalized["catalog_focus_terms"] = [
                str(term).strip()
                for term in list(focus_terms)
                if str(term).strip()
            ]
        exclude_sources = spec.get("catalog_exclude_sources")
        if isinstance(exclude_sources, Sequence) and not isinstance(exclude_sources, (str, bytes)):
            normalized["catalog_exclude_sources"] = [
                str(item).strip()
                for item in list(exclude_sources)
                if str(item).strip()
            ]
        exclude_names = spec.get("catalog_exclude_names")
        if isinstance(exclude_names, Sequence) and not isinstance(exclude_names, (str, bytes)):
            normalized["catalog_exclude_names"] = [
                str(item).strip()
                for item in list(exclude_names)
                if str(item).strip()
            ]
        if catalog_mode == "semantic_registry":
            entries = list(spec.get("catalog_entries") or [])
            if not entries:
                raise ValueError("semantic_registry candidate specs require catalog_entries")
            normalized_entries: list[dict[str, Any]] = []
            for entry in entries:
                if not isinstance(entry, Mapping):
                    raise ValueError("catalog_entries items must be objects")
                normalized_entry: dict[str, Any] = {
                    "name": str(entry.get("name", "")).strip(),
                    "source": str(entry.get("source", "")).strip(),
                    "source_type": str(entry.get("source_type", "auto")).strip() or "auto",
                    "text_field": str(entry.get("text_field", "text")).strip() or "text",
                }
                if normalized_entry["source_type"] not in {"auto", "hf", "web"}:
                    raise ValueError("catalog_entries source_type must be auto/hf/web")
                if not normalized_entry["name"] or not normalized_entry["source"]:
                    raise ValueError("catalog_entries items require non-empty name and source")
                hf_config_raw = entry.get("hf_config")
                if hf_config_raw not in (None, "", "None"):
                    normalized_entry["hf_config"] = str(hf_config_raw)
                for key in (
                    "summary",
                    "title",
                    "description",
                    "query_text",
                    "provider",
                ):
                    value = " ".join(str(entry.get(key, "")).split()).strip()
                    if value:
                        normalized_entry[key] = value
                for key in ("tags", "terms"):
                    values = entry.get(key)
                    if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
                        normalized_entry[key] = [
                            str(item).strip()
                            for item in list(values)
                            if str(item).strip()
                        ]
                for key in ("catalog_priority", "prior_weight"):
                    value = entry.get(key)
                    if value is not None:
                        normalized_entry[key] = float(cast(Any, value))
                normalized_entries.append(normalized_entry)
            normalized["catalog_entries"] = normalized_entries
        else:
            providers = spec.get("catalog_providers")
            if isinstance(providers, Sequence) and not isinstance(providers, (str, bytes)):
                normalized["catalog_providers"] = [
                    str(provider).strip()
                    for provider in list(providers)
                    if str(provider).strip()
                ]
            normalized["catalog_queries_per_provider"] = max(
                1,
                int(spec.get("catalog_queries_per_provider", 2)),
            )
            normalized["catalog_provider_result_limit"] = max(
                1,
                int(spec.get("catalog_provider_result_limit", 4)),
            )
        return normalized

    def _normalize_autonomy_candidate_spec(self, spec: Any, index: int) -> dict[str, Any]:
        if isinstance(spec, dict) and str(spec.get("catalog_mode", "")).strip():
            return self._normalize_catalog_candidate_spec(spec, index)
        return self._normalize_brain_source_spec(spec, index)

    def _normalize_provider_curriculum(self, value: Any) -> dict[str, dict[str, Any]]:
        if not isinstance(value, Mapping):
            return {}

        def _safe_int(raw_value: Any) -> int:
            try:
                return max(0, int(raw_value))
            except (TypeError, ValueError):
                return 0

        def _safe_float(raw_value: Any) -> float:
            try:
                return max(0.0, float(raw_value))
            except (TypeError, ValueError):
                return 0.0

        normalized: dict[str, dict[str, Any]] = {}
        for raw_provider, raw_entry in value.items():
            provider = " ".join(str(raw_provider).split()).strip().lower()
            if not provider or not isinstance(raw_entry, Mapping):
                continue
            topic_terms: dict[str, float] = {}
            raw_topic_terms = raw_entry.get("topic_terms")
            if isinstance(raw_topic_terms, Mapping):
                for raw_term, raw_weight in raw_topic_terms.items():
                    term = _canonical_provider_term(raw_term)
                    if not term:
                        continue
                    weight = _safe_float(raw_weight)
                    if weight > 0.0:
                        topic_terms[term] = float(weight)
            topic_families: dict[str, dict[str, Any]] = {}
            raw_topic_families = raw_entry.get("topic_families")
            if isinstance(raw_topic_families, Mapping):
                for raw_family, raw_family_entry in raw_topic_families.items():
                    family = _canonical_provider_term(raw_family)
                    if not family or not isinstance(raw_family_entry, Mapping):
                        continue
                    topic_families[family] = {
                        "commits": _safe_int(raw_family_entry.get("commits", 0)),
                        "successes": _safe_int(raw_family_entry.get("successes", 0)),
                        "semantic_relevance_ema": _safe_float(raw_family_entry.get("semantic_relevance_ema", 0.0)),
                        "answerability_gain_ema": _safe_float(raw_family_entry.get("answerability_gain_ema", 0.0)),
                        "uncertainty_reduction_ema": _safe_float(
                            raw_family_entry.get("uncertainty_reduction_ema", 0.0)
                        ),
                        "weak_concept_stabilization_ema": _safe_float(
                            raw_family_entry.get("weak_concept_stabilization_ema", 0.0)
                        ),
                        "last_selected_at": " ".join(
                            str(raw_family_entry.get("last_selected_at", "")).split()
                        ).strip(),
                    }
            query_families: dict[str, dict[str, Any]] = {}
            raw_query_families = raw_entry.get("query_families")
            if isinstance(raw_query_families, Mapping):
                for raw_family, raw_family_entry in raw_query_families.items():
                    family = _canonical_provider_term(raw_family)
                    if not family or not isinstance(raw_family_entry, Mapping):
                        continue
                    query_families[family] = {
                        "commits": _safe_int(raw_family_entry.get("commits", 0)),
                        "successes": _safe_int(raw_family_entry.get("successes", 0)),
                        "semantic_relevance_ema": _safe_float(raw_family_entry.get("semantic_relevance_ema", 0.0)),
                        "answerability_gain_ema": _safe_float(raw_family_entry.get("answerability_gain_ema", 0.0)),
                        "uncertainty_reduction_ema": _safe_float(
                            raw_family_entry.get("uncertainty_reduction_ema", 0.0)
                        ),
                        "weak_concept_stabilization_ema": _safe_float(
                            raw_family_entry.get("weak_concept_stabilization_ema", 0.0)
                        ),
                        "last_selected_at": " ".join(
                            str(raw_family_entry.get("last_selected_at", "")).split()
                        ).strip(),
                    }
            normalized[provider] = {
                "attempts": _safe_int(raw_entry.get("attempts", 0)),
                "commits": _safe_int(raw_entry.get("commits", 0)),
                "successes": _safe_int(raw_entry.get("successes", 0)),
                "gap_gain_ema": _safe_float(raw_entry.get("gap_gain_ema", 0.0)),
                "diagnostic_gain_ema": _safe_float(raw_entry.get("diagnostic_gain_ema", 0.0)),
                "semantic_relevance_ema": _safe_float(raw_entry.get("semantic_relevance_ema", 0.0)),
                "answerability_gain_ema": _safe_float(raw_entry.get("answerability_gain_ema", 0.0)),
                "uncertainty_reduction_ema": _safe_float(raw_entry.get("uncertainty_reduction_ema", 0.0)),
                "weak_concept_stabilization_ema": _safe_float(
                    raw_entry.get("weak_concept_stabilization_ema", 0.0)
                ),
                "utility_ema": _safe_float(raw_entry.get("utility_ema", 0.0)),
                "focus_alignment_ema": _safe_float(raw_entry.get("focus_alignment_ema", 0.0)),
                "grounded_outcome_ema": _safe_float(raw_entry.get("grounded_outcome_ema", 0.0)),
                "grounded_family_summary_ema": _safe_float(raw_entry.get("grounded_family_summary_ema", 0.0)),
                "delayed_consequence_ema": _safe_float(raw_entry.get("delayed_consequence_ema", 0.0)),
                "contradiction_decay_ema": _safe_float(raw_entry.get("contradiction_decay_ema", 0.0)),
                "last_query_text": " ".join(str(raw_entry.get("last_query_text", "")).split()).strip(),
                "last_selected_at": " ".join(str(raw_entry.get("last_selected_at", "")).split()).strip(),
                "topic_terms": dict(
                    sorted(
                        topic_terms.items(),
                        key=lambda item: (-float(item[1]), item[0]),
                    )[:AUTO_REMOTE_PROVIDER_TOPIC_TERM_LIMIT]
                ),
                "topic_families": dict(
                    sorted(
                        topic_families.items(),
                        key=lambda item: (-self._provider_topic_family_priority_locked(item[1]), item[0]),
                    )[:AUTO_REMOTE_PROVIDER_TOPIC_TERM_LIMIT]
                ),
                "query_families": dict(
                    sorted(
                        query_families.items(),
                        key=lambda item: (-self._provider_query_family_priority_locked(item[1]), item[0]),
                    )[:AUTO_REMOTE_PROVIDER_QUERY_FAMILY_LIMIT]
                ),
            }
        return normalized

    def _default_autonomy_candidate_bank(self) -> list[dict[str, Any]]:
        return [
            self._normalize_catalog_candidate_spec(
                {
                    "name": "autonomy_live_remote_search",
                    "catalog_mode": "live_remote_search",
                    "catalog_providers": list(DEFAULT_AUTONOMY_REMOTE_PROVIDERS),
                    "catalog_queries_per_provider": DEFAULT_AUTONOMY_REMOTE_QUERIES_PER_PROVIDER,
                    "catalog_provider_result_limit": DEFAULT_AUTONOMY_REMOTE_PROVIDER_RESULT_LIMIT,
                    "catalog_limit": DEFAULT_AUTONOMY_REMOTE_CATALOG_LIMIT,
                    "catalog_probe_pool_limit": DEFAULT_AUTONOMY_REMOTE_PROBE_POOL_LIMIT,
                },
                0,
            )
        ]

    def _normalize_autonomy_config(self, autonomy: Any) -> dict[str, Any] | None:
        if autonomy is None:
            return None
        if not isinstance(autonomy, dict):
            raise ValueError("Terminus autonomy configuration must be an object")
        candidate_specs = [
            self._normalize_autonomy_candidate_spec(item, index)
            for index, item in enumerate(list(autonomy.get("candidate_bank") or []))
        ]
        enabled = bool(autonomy.get("enabled", bool(candidate_specs)))
        using_default_remote_search = False
        if enabled and not candidate_specs:
            candidate_specs = self._default_autonomy_candidate_bank()
            using_default_remote_search = True
        policy = str(autonomy.get("policy", "active")).strip() or "active"
        if policy not in PUBLIC_ACQUISITION_POLICIES:
            raise ValueError(
                "Unsupported Terminus autonomy policy. "
                f"Supported policies: {', '.join(PUBLIC_ACQUISITION_POLICIES)}"
            )
        shortlist_size_raw = autonomy.get("semantic_shortlist_size")
        shortlist_gap_weight_raw = autonomy.get("semantic_shortlist_gap_weight")
        shortlist_affinity_weight_raw = autonomy.get("semantic_shortlist_affinity_weight")
        if using_default_remote_search:
            shortlist_size = max(
                1,
                int(1 if shortlist_size_raw in (None, 0, "0") else shortlist_size_raw),
            )
            if shortlist_gap_weight_raw in (None, 0.5, "0.5") and shortlist_affinity_weight_raw in (None, 0.5, "0.5"):
                shortlist_gap_weight = 0.0
                shortlist_affinity_weight = 1.0
            else:
                shortlist_gap_weight = float(
                    0.0 if shortlist_gap_weight_raw in (None, "", "None") else shortlist_gap_weight_raw
                )
                shortlist_affinity_weight = float(
                    1.0 if shortlist_affinity_weight_raw in (None, "", "None") else shortlist_affinity_weight_raw
                )
        else:
            shortlist_size = max(0, int(autonomy.get("semantic_shortlist_size", 0)))
            shortlist_gap_weight = float(autonomy.get("semantic_shortlist_gap_weight", 0.5))
            shortlist_affinity_weight = float(autonomy.get("semantic_shortlist_affinity_weight", 0.5))
        return {
            "enabled": enabled,
            "policy": policy,
            "candidate_bank": candidate_specs,
            "provider_curriculum": self._normalize_provider_curriculum(autonomy.get("provider_curriculum")),
            "trigger_interval_tokens": max(
                1,
                int(autonomy.get("trigger_interval_tokens", DEFAULT_AUTONOMY_TRIGGER_INTERVAL_TOKENS)),
            ),
            "candidate_train_tokens": max(1, int(autonomy.get("candidate_train_tokens", 768))),
            "probe_tokens": max(1, int(autonomy.get("probe_tokens", 96))),
            "acquisition_tokens": max(1, int(autonomy.get("acquisition_tokens", 512))),
            "acquisition_slots": max(1, int(autonomy.get("acquisition_slots", 1))),
            "gap_exploration_bonus": float(autonomy.get("gap_exploration_bonus", 0.03)),
            "gap_ambiguity_weight": float(autonomy.get("gap_ambiguity_weight", 0.4)),
            "gap_switch_weight": float(autonomy.get("gap_switch_weight", 0.2)),
            "gap_margin_reference": float(autonomy.get("gap_margin_reference", 0.12)),
            "coverage_balance_penalty": float(autonomy.get("coverage_balance_penalty", 0.2)),
            "gap_focus_margin": float(autonomy.get("gap_focus_margin", 0.05)),
            "scout_commit_tokens": max(0, int(autonomy.get("scout_commit_tokens", 0))),
            "scout_top_k": max(1, int(autonomy.get("scout_top_k", 1))),
            "semantic_shortlist_size": shortlist_size,
            "semantic_shortlist_gap_weight": shortlist_gap_weight,
            "semantic_shortlist_affinity_weight": shortlist_affinity_weight,
        }

    def _normalize_brain_config(self, config: Any) -> dict[str, Any]:
        if config is None:
            tick_tokens = DEFAULT_BRAIN_TICK_TOKENS
            return {
                "source_bank": [],
                "tick_tokens": tick_tokens,
                "sleep_interval_seconds": DEFAULT_BRAIN_SLEEP_INTERVAL_SECONDS,
                "execution_quantum_tokens": min(
                    tick_tokens,
                    DEFAULT_EXECUTION_QUANTUM_TOKENS,
                ),
                "execution_yield_seconds": DEFAULT_EXECUTION_YIELD_SECONDS,
                "repeat_sources": True,
                "autonomy": None,
                "sensory": None,
                "ingestion": self._normalize_ingestion_config(None, tick_tokens=tick_tokens),
            }
        if not isinstance(config, dict):
            raise ValueError("Terminus runtime configuration must be an object")
        source_bank = [
            self._normalize_brain_source_spec(item, index)
            for index, item in enumerate(list(config.get("source_bank") or []))
        ]
        tick_tokens = max(1, int(config.get("tick_tokens", DEFAULT_BRAIN_TICK_TOKENS)))
        execution_quantum_tokens = min(
            tick_tokens,
            MAX_EXECUTION_QUANTUM_TOKENS,
            max(
                1,
                int(
                    config.get(
                        "execution_quantum_tokens",
                        DEFAULT_EXECUTION_QUANTUM_TOKENS,
                    )
                ),
            ),
        )
        normalized = {
            "source_bank": source_bank,
            "tick_tokens": tick_tokens,
            "sleep_interval_seconds": max(
                0.01,
                float(config.get("sleep_interval_seconds", DEFAULT_BRAIN_SLEEP_INTERVAL_SECONDS)),
            ),
            "execution_quantum_tokens": execution_quantum_tokens,
            "execution_yield_seconds": max(
                0.0,
                min(
                    1.0,
                    float(
                        config.get(
                            "execution_yield_seconds",
                            DEFAULT_EXECUTION_YIELD_SECONDS,
                        )
                    ),
                ),
            ),
            "repeat_sources": bool(config.get("repeat_sources", True)),
            "autonomy": self._normalize_autonomy_config(config.get("autonomy")),
            "sensory": self._normalize_sensory_config(config.get("sensory")),
            "ingestion": self._normalize_ingestion_config(config.get("ingestion"), tick_tokens=tick_tokens),
        }
        return normalized

    @staticmethod
    def _normalize_ingestion_config(config: Any, *, tick_tokens: int) -> dict[str, Any]:
        raw = config if isinstance(config, dict) else {}
        enabled = bool(raw.get("enabled", True))
        default_queue_target = max(
            int(tick_tokens),
            int(tick_tokens) * DEFAULT_INGESTION_QUEUE_MULTIPLIER,
        )
        queue_target_tokens = max(
            int(tick_tokens),
            int(raw.get("queue_target_tokens", default_queue_target)),
        )
        return {
            "enabled": enabled,
            "queue_target_tokens": queue_target_tokens,
            "prewarm_on_startup": bool(raw.get("prewarm_on_startup", False)),
            "prewarm_max_seconds": max(0.05, float(raw.get("prewarm_max_seconds", 5.0))),
        }

    def _normalize_sensory_config(self, config: Any) -> dict[str, Any] | None:
        if config is None or not isinstance(config, dict):
            return None
        if not config.get("enabled"):
            return None
        source_bank = [
            self._normalize_sensory_source_spec(item, index)
            for index, item in enumerate(list(config.get("source_bank") or []))
        ]
        if not source_bank:
            return None
        base_windows = max(1, int(config.get("base_windows_per_item", 4)))
        max_windows = max(base_windows, int(config.get("max_windows_per_item", 10)))
        items_per_episode = max(1, int(config.get("items_per_episode", 2)))
        lookahead = max(1, int(config.get("item_retrieval_lookahead", 6)))
        queue_target_items = max(
            1,
            int(config.get("queue_target_items", max(items_per_episode, lookahead))),
        )
        return {
            "enabled": True,
            "source_bank": source_bank,
            "episode_interval_tokens": max(256, int(config.get("episode_interval_tokens", 1536))),
            "items_per_episode": items_per_episode,
            "base_windows_per_item": base_windows,
            "max_windows_per_item": max_windows,
            "confidence_window_gain": max(0.0, float(config.get("confidence_window_gain", 3.0))),
            "semantic_window_gain": max(0.0, float(config.get("semantic_window_gain", 3.0))),
            "item_retrieval_lookahead": lookahead,
            "item_retrieval_semantic_weight": max(0.0, min(1.0, float(config.get("item_retrieval_semantic_weight", 0.72)))),
            "modality_target_confidence": max(0.1, min(1.0, float(config.get("modality_target_confidence", 0.70)))),
            "observation_salience": max(0.1, min(1.0, float(config.get("observation_salience", 0.82)))),
            "cooldown_seconds": max(1.0, float(config.get("cooldown_seconds", 8.0))),
            "repeat_sources": bool(config.get("repeat_sources", True)),
            "queue_target_items": queue_target_items,
            "prewarm_on_startup": bool(config.get("prewarm_on_startup", False)),
            "prewarm_max_seconds": max(0.05, float(config.get("prewarm_max_seconds", 5.0))),
        }

