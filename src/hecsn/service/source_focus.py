"""Source focus and utility scoring helpers for Terminus.

This mixin chooses useful text sources and updates source utility summaries. It
keeps source scoring separate from brain-loop orchestration and from the delayed
consequence learner.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from hecsn.semantics.grounding_text import salient_query_terms
from hecsn.service.runtime_sources import _BrainSourceRuntime
from hecsn.service.terminus_autonomy import _canonical_provider_term

DEFAULT_BRAIN_TICK_TOKENS = 512
DEFAULT_UTILITY_PENALTY_WEIGHT = 0.65


class SourceFocusMixin:
    def _focus_gap_terms_locked(self, limit: int = 4) -> list[str]:
        terms: list[str] = []

        exploration_target = ""
        if self._thought_loop_actual is not None and hasattr(self._thought_loop_actual, "gate"):
            exploration_target = str(getattr(self._thought_loop_actual.gate, "active_exploration_target", "")).strip()
        if exploration_target:
            terms.append(exploration_target)

        try:
            plan = self._geometric_curiosity.focus_plan(top_n=max(1, limit))
            for item in list((plan or {}).get("geometric_gaps", []))[:limit]:
                concept = " ".join(str(item.get("concept", "")).split()).strip()
                if concept:
                    terms.append(concept)
        except Exception:
            pass

        try:
            snap = self._concept_store.snapshot(limit=max(1, limit))
            for concept in list(snap.get("top_concepts", []))[:limit]:
                if not isinstance(concept, dict):
                    continue
                label = " ".join(str(concept.get("label", "")).split()).strip()
                if label:
                    terms.append(label)
                for term in list(concept.get("top_terms", []))[:2]:
                    cleaned = " ".join(str(term).split()).strip()
                    if cleaned:
                        terms.append(cleaned)
        except Exception:
            pass

        normalized: list[str] = []
        seen: set[str] = set()
        for term in terms:
            cleaned = " ".join(term.replace("/", " ").replace("|", " ").split()).strip().lower()
            if cleaned and cleaned not in seen:
                normalized.append(cleaned)
                seen.add(cleaned)
            if len(normalized) >= max(1, limit):
                break
        return normalized

    def _background_focus_terms_locked(
        self,
        limit: int = 12,
        *,
        focus_plan: Mapping[str, Any] | None = None,
    ) -> list[str]:
        plan = focus_plan if focus_plan is not None else self._autonomy_focus_plan_locked()
        phrases: list[str] = []
        if isinstance(plan, Mapping):
            phrases.extend(str(item) for item in list(plan.get("query_terms") or []) if str(item).strip())
            phrases.extend(str(item) for item in list(plan.get("unsupported_terms") or []) if str(item).strip())
            phrases.extend(
                str(item.get("term", ""))
                for item in list(plan.get("gap_terms") or [])
                if isinstance(item, Mapping) and str(item.get("term", "")).strip()
            )
            phrases.extend(str(item) for item in list(plan.get("retrieval_queries") or []) if str(item).strip())
            for raw_concept in list(plan.get("weak_concepts") or []):
                if not isinstance(raw_concept, Mapping):
                    continue
                phrases.append(str(raw_concept.get("label", "")))
                phrases.extend(
                    str(item)
                    for item in list(raw_concept.get("top_terms") or [])
                    if str(item).strip()
                )
        if not phrases and self._brain_recent_query_gaps:
            recent_gap = self._brain_recent_query_gaps[0]
            phrases.append(str(recent_gap.get("query_text", "")))
            phrases.extend(str(term) for term in list(recent_gap.get("unsupported_terms") or [])[:4])
        if not phrases:
            phrases.extend(self._focus_gap_terms_locked(limit=max(4, limit // 2)))

        ordered: list[str] = []
        seen: set[str] = set()
        for phrase in phrases:
            for term in salient_query_terms(str(phrase)):
                cleaned = _canonical_provider_term(term)
                if len(cleaned) < 4 or cleaned in seen:
                    continue
                seen.add(cleaned)
                ordered.append(cleaned)
                if len(ordered) >= max(1, limit):
                    return ordered
        return ordered

    @staticmethod
    def _brain_source_memory_metadata(runtime: _BrainSourceRuntime) -> dict[str, Any]:
        metadata = runtime.spec.get("metadata") if isinstance(runtime.spec.get("metadata"), Mapping) else {}
        topic_terms = [
            _canonical_provider_term(term)
            for term in list(runtime.spec.get("topic_terms") or [])
            if _canonical_provider_term(term)
        ]
        catalog_terms = [str(term) for term in topic_terms]
        raw_catalog_terms = metadata.get("catalog_terms") if isinstance(metadata, Mapping) else None
        if isinstance(raw_catalog_terms, Sequence) and not isinstance(raw_catalog_terms, (str, bytes)):
            catalog_terms = list(
                dict.fromkeys(
                    [
                        *catalog_terms,
                        *[
                            _canonical_provider_term(term)
                            for term in list(raw_catalog_terms)
                            if _canonical_provider_term(term)
                        ],
                    ]
                )
            )
        memory_metadata: dict[str, Any] = {
            "observation_kind": "source",
            "source_name": runtime.name,
            "source_type": runtime.source_type,
            "source": str(runtime.spec.get("source", "")),
            "catalog_terms": catalog_terms[:8],
        }
        for key in ("provider", "query_text", "catalog_title", "catalog_summary"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                memory_metadata[str(key)] = value.strip()
        return memory_metadata

    @staticmethod
    def _brain_source_topic_terms(runtime: _BrainSourceRuntime) -> set[str]:
        terms: set[str] = set()
        for raw in list(runtime.spec.get("topic_terms") or []):
            for term in salient_query_terms(str(raw)):
                cleaned = _canonical_provider_term(term)
                if len(cleaned) >= 4:
                    terms.add(cleaned)
        for raw in (
            runtime.spec.get("name", ""),
            runtime.spec.get("source", ""),
            runtime.spec.get("hf_config", ""),
        ):
            for term in salient_query_terms(str(raw)):
                cleaned = _canonical_provider_term(term)
                if len(cleaned) >= 4:
                    terms.add(cleaned)
        metadata = runtime.spec.get("metadata")
        if isinstance(metadata, Mapping):
            for key in ("role", "label", "why", "summary", "description", "title"):
                for term in salient_query_terms(str(metadata.get(key, ""))):
                    cleaned = _canonical_provider_term(term)
                    if len(cleaned) >= 4:
                        terms.add(cleaned)
        return terms

    def _brain_source_semantic_match_locked(
        self,
        runtime: _BrainSourceRuntime,
        focus_terms: Sequence[str] | None = None,
    ) -> float:
        normalized_focus = [
            _canonical_provider_term(term)
            for term in list(focus_terms or self._background_focus_terms_locked())
            if _canonical_provider_term(term)
        ]
        source_terms = self._brain_source_topic_terms(runtime)
        if not normalized_focus or not source_terms:
            return 0.0
        focus_set = set(normalized_focus)
        overlap = len(focus_set & source_terms) / max(1.0, min(float(len(focus_set)), float(len(source_terms))))
        head_hits = sum(1 for term in normalized_focus[:3] if term in source_terms)
        head_bonus = min(1.0, 0.5 * head_hits)
        metadata = runtime.spec.get("metadata") if isinstance(runtime.spec.get("metadata"), Mapping) else {}
        combined_text = " ".join(
            part
            for part in [
                str(runtime.spec.get("name", "")),
                str(runtime.spec.get("source", "")),
                str(runtime.spec.get("hf_config", "")),
                *(str(metadata.get(key, "")) for key in ("role", "label", "why", "summary", "description", "title")),
            ]
            if part
        ).lower()
        phrase_hits = sum(1 for term in normalized_focus[:4] if term and term in combined_text)
        phrase_bonus = min(1.0, 0.34 * phrase_hits)
        return max(0.0, min(1.0, 0.55 * overlap + 0.30 * head_bonus + 0.15 * phrase_bonus))

    def _brain_source_selection_score_locked(
        self,
        runtime: _BrainSourceRuntime,
        *,
        focus_terms: Sequence[str],
        focus_pressure: float,
        tick_tokens: int,
    ) -> tuple[float, float, float, float, float]:
        semantic_match = self._brain_source_semantic_match_locked(runtime, focus_terms)
        source_count = max(1, len(self._brain_source_runtimes))
        min_visits = min((rt.tick_visits for rt in self._brain_source_runtimes), default=0)
        fairness = max(
            0.0,
            min(
                1.0,
                1.0 - max(0, runtime.tick_visits - min_visits) / float(source_count + 1),
            ),
        )
        readiness = max(
            0.0,
            min(1.0, float(len(runtime.buffered_patterns)) / float(max(1, int(tick_tokens)))),
        )
        freshness = 1.0 if runtime.last_activity_at is None else 0.0
        focus_factor = max(0.0, min(1.0, float(focus_pressure)))
        source_utility = self._background_source_utility_entry_locked(runtime)
        utility_ema = max(0.0, min(1.0, float(source_utility.get("utility_ema", 0.0))))
        grounded_family_summary = max(0.0, min(1.0, float(source_utility.get("grounded_family_summary_ema", 0.0))))
        contradiction_decay = max(0.0, min(1.0, float(source_utility.get("contradiction_decay_ema", 0.0))))
        net_utility = max(
            0.0,
            max(utility_ema, grounded_family_summary) - DEFAULT_UTILITY_PENALTY_WEIGHT * contradiction_decay,
        )
        effective_utility = float(
            max(net_utility, grounded_family_summary)
            * max(0.0, min(1.0, max(float(semantic_match), 0.35 * focus_factor)))
        )
        semantic_weight = 0.10 + 0.24 * focus_factor
        utility_weight = 0.04 + 0.34 * focus_factor
        fairness_weight = max(0.16, 0.54 - 0.43 * focus_factor)
        readiness_weight = 0.10
        freshness_weight = 0.10
        score = (
            semantic_weight * semantic_match
            + utility_weight * effective_utility
            + fairness_weight * fairness
            + readiness_weight * readiness
            + freshness_weight * freshness
        )
        runtime.last_semantic_match = float(semantic_match)
        runtime.last_selection_score = float(score)
        runtime.last_fairness_score = float(fairness)
        runtime.last_buffer_readiness = float(readiness)
        runtime.last_utility_score = float(effective_utility)
        return score, semantic_match, fairness, readiness, effective_utility

    def _background_focus_overlap_locked(
        self,
        focus_terms: Sequence[str],
        grounded_observation: Mapping[str, Any] | None,
    ) -> float:
        normalized_focus = [
            _canonical_provider_term(term)
            for term in list(focus_terms or [])
            if _canonical_provider_term(term)
        ]
        if not normalized_focus:
            return 0.0
        overlap_sources: list[str] = []
        if isinstance(grounded_observation, Mapping):
            overlap_sources.append(str(grounded_observation.get("content", "")))
            overlap_sources.extend(str(item) for item in list(grounded_observation.get("topics") or []) if str(item).strip())
        combined = " ".join(part for part in overlap_sources if part).strip()
        if not combined:
            return 0.0
        overlap = self._source_text_overlap(" ".join(normalized_focus), combined)
        phrase_hits = sum(1 for term in normalized_focus[:4] if term and term in combined.lower())
        phrase_bonus = min(1.0, 0.34 * phrase_hits)
        return max(0.0, min(1.0, 0.70 * overlap + 0.30 * phrase_bonus))

    def _update_background_source_utility_locked(
        self,
        *,
        runtime: _BrainSourceRuntime,
        grounded_observation: Mapping[str, Any] | None,
        total_trained: int,
    ) -> None:
        entry = self._background_source_utility_entry_locked(runtime)
        focus_plan = self._autonomy_focus_plan_locked()
        focus_terms = self._background_focus_terms_locked(focus_plan=focus_plan)
        semantic_alignment = max(0.0, min(1.0, float(runtime.last_semantic_match)))
        grounding_signal = 0.0
        if isinstance(grounded_observation, Mapping):
            grounding_signal = max(0.0, min(1.0, float(grounded_observation.get("grounding_signal", 0.0) or 0.0)))
        focus_overlap = self._background_focus_overlap_locked(focus_terms, grounded_observation)
        token_fraction = min(
            1.0,
            float(max(0, int(total_trained))) / float(max(1, int(self._brain_config.get("tick_tokens", DEFAULT_BRAIN_TICK_TOKENS)))),
        )
        utility_sample = max(
            0.0,
            min(
                1.0,
                0.50 * semantic_alignment
                + 0.20 * grounding_signal
                + 0.20 * focus_overlap
                + 0.10 * token_fraction,
            ),
        )

        entry["attempts"] = int(entry.get("attempts", 0)) + 1
        entry["selections"] = int(entry.get("selections", 0)) + 1
        entry["tokens_trained_total"] = int(entry.get("tokens_trained_total", 0)) + max(0, int(total_trained))
        alpha = 0.30
        for key, sample in (
            ("utility_ema", utility_sample),
            ("semantic_alignment_ema", semantic_alignment),
            ("grounding_signal_ema", grounding_signal),
            ("focus_overlap_ema", focus_overlap),
        ):
            previous = max(0.0, min(1.0, float(entry.get(key, 0.0) or 0.0)))
            entry[key] = float(sample if int(entry["selections"]) <= 1 else (1.0 - alpha) * previous + alpha * float(sample))
        entry["last_selected_at"] = datetime.now(timezone.utc).isoformat()
        self._mark_mutated()

    @staticmethod
    def _selected_evidence_weight_map(
        response: Mapping[str, Any],
        *,
        singular_field: str,
        plural_field: str,
    ) -> dict[str, float]:
        weighted: dict[str, float] = {}
        for rank, raw_item in enumerate(list(response.get("selected_evidence") or [])):
            if not isinstance(raw_item, Mapping):
                continue
            rank_weight = 1.0 / float(rank + 1)
            term_coverage = max(0.0, min(1.0, float(raw_item.get("term_coverage", 0.0) or 0.0)))
            score_weight = max(0.0, min(1.0, float(raw_item.get("score", 0.0) or 0.0)))
            item_weight = max(0.20, min(1.0, 0.40 * rank_weight + 0.30 * term_coverage + 0.30 * score_weight))
            names: list[str] = []
            single_value = " ".join(str(raw_item.get(singular_field, "")).split()).strip()
            if single_value:
                names.append(single_value)
            raw_values = raw_item.get(plural_field)
            if isinstance(raw_values, Sequence) and not isinstance(raw_values, (str, bytes)):
                names.extend(" ".join(str(item).split()).strip() for item in list(raw_values) if " ".join(str(item).split()).strip())
            for name in names:
                key = name.lower() if singular_field == "provider" else name
                weighted[key] = max(float(weighted.get(key, 0.0)), float(item_weight))
        return weighted

    def _response_grounded_outcome_score_locked(
        self,
        *,
        query_result: Mapping[str, Any],
        response: Mapping[str, Any],
        action_assist: Mapping[str, Any] | None,
    ) -> float:
        gap_plan = query_result.get("gap_plan") if isinstance(query_result.get("gap_plan"), Mapping) else {}
        grounded_fraction = max(0.0, min(1.0, float(gap_plan.get("grounded_fraction", 0.0) or 0.0)))
        evidence_coverage = max(0.0, min(1.0, float(response.get("evidence_coverage", 0.0) or 0.0)))
        selected_evidence_count = int(len(list(response.get("selected_evidence") or [])))
        selected_evidence_bonus = min(1.0, float(selected_evidence_count) / 2.0)
        response_mode = self._normalize_action_text(response.get("response_mode", "")).lower()
        unsupported_terms = [
            str(item).strip().lower()
            for item in list(response.get("unsupported_terms") or gap_plan.get("unsupported_terms") or [])
            if str(item).strip()
        ]
        unsupported_penalty = min(1.0, float(len(unsupported_terms)) / 4.0)
        score = max(
            0.0,
            min(
                1.0,
                0.36 * grounded_fraction
                + 0.34 * evidence_coverage
                + 0.15 * selected_evidence_bonus
                + 0.15 * (0.0 if response_mode == "insufficient_evidence" else 1.0)
                - 0.25 * unsupported_penalty,
            ),
        )
        if isinstance(action_assist, Mapping):
            record = action_assist.get("result") if isinstance(action_assist.get("result"), Mapping) else {}
            verification = record.get("verification") if isinstance(record.get("verification"), Mapping) else {}
            if bool(verification.get("success", False)):
                confidence = max(0.0, min(1.0, float(verification.get("confidence", 0.0) or 0.0)))
                score = max(score, max(0.0, min(1.0, 0.55 + 0.35 * confidence)))
            elif bool(verification.get("contradiction", False)):
                score *= 0.25
        return float(max(0.0, min(1.0, score)))
