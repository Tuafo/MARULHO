"""Terminus autonomy intelligence -- focus planning and provider curriculum.

This mixin provides the methods for:
- Gap-based focus planning (which concepts need grounding)
- Provider curriculum prioritization (which external sources to query)
- Autonomy candidate selection and shortlist building
- Query family scoring and topic matching

These methods are mixed into HECSNServiceManager and operate on its
internal state (self._brain_*, self._trainer, self._concept_store, etc.)
via the manager's RLock.
"""

from __future__ import annotations

import math
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence, cast

from hecsn.semantics.grounding_text import salient_query_terms

# Autonomy constants (used by provider curriculum and focus planning)
DEFAULT_AUTONOMY_REMOTE_PROVIDERS: tuple[str, ...] = ("wikipedia", "arxiv", "openalex")
DEFAULT_AUTONOMY_REMOTE_CATALOG_LIMIT = 4
DEFAULT_AUTONOMY_REMOTE_PROBE_POOL_LIMIT = 4
DEFAULT_AUTONOMY_REMOTE_QUERIES_PER_PROVIDER = 2
DEFAULT_AUTONOMY_REMOTE_PROVIDER_RESULT_LIMIT = 4
AUTO_REMOTE_QUERY_BUDGET_MAX = 4
AUTO_REMOTE_PROVIDER_PRIORITY_WEIGHT = 0.35
AUTO_REMOTE_PROVIDER_TOPIC_TERM_LIMIT = 8
AUTO_REMOTE_PROVIDER_QUERY_FAMILY_LIMIT = 4
AUTO_FOCUS_SHORTLIST_MAX_SIZE = 3
AUTO_FOCUS_SHORTLIST_GAP_WEIGHT = 0.2
AUTO_FOCUS_SHORTLIST_AFFINITY_WEIGHT = 0.8
AUTO_FOCUS_TRIGGER_INTERVAL_FLOOR = 0.5
AUTO_FOCUS_ACQUISITION_TOKEN_SCALE_MAX = 1.75
AUTO_FOCUS_SLOT_PRESSURE_THRESHOLD = 0.72

# Irregular plural → singular mappings for topic normalization
_IRREGULAR_TOPIC_SINGULARS = {
    "octopi": "octopus",
    "octopuses": "octopus",
}


def _canonical_provider_term(value: Any) -> str:
    """Normalize a topic term: lowercase, singularize, strip whitespace."""
    normalized = " ".join(str(value).split()).strip().lower()
    if not normalized:
        return ""

    def _canonical_token(token: str) -> str:
        if token in _IRREGULAR_TOPIC_SINGULARS:
            return _IRREGULAR_TOPIC_SINGULARS[token]
        if token in {"species", "series"} or len(token) <= 3:
            return token
        if token.endswith("ies") and len(token) > 4:
            return token[:-3] + "y"
        if token.endswith(("ses", "xes", "zes", "ches", "shes")) and len(token) > 4:
            return token[:-2]
        if token.endswith("oes") and len(token) > 4:
            return token[:-2]
        if token.endswith(("us", "is")):
            return token
        if token.endswith("s") and not token.endswith("ss") and len(token) > 3:
            return token[:-1]
        return token

    return " ".join(_canonical_token(part) for part in normalized.split() if part).strip()


class TerminusAutonomyMixin:
    """Mixin providing autonomy focus-planning and provider curriculum methods.

    Mixed into HECSNServiceManager. All methods assume the caller holds
    self._lock (hence the _locked suffix convention).
    """

    def _autonomy_focus_plan_locked(self) -> dict[str, Any] | None:
        recent_query_focus = self._recent_query_focus_plan_locked()
        abstraction_query = ""
        if recent_query_focus is not None:
            abstraction_query = " ".join(
                [
                    *[
                        str(value)
                        for value in list(recent_query_focus.get("query_terms") or [])[:4]
                        if str(value).strip()
                    ],
                    *[
                        str(value)
                        for value in list(recent_query_focus.get("unsupported_terms") or [])[:2]
                        if str(value).strip()
                    ],
                ]
            ).strip()
        concept_focus = (
            self._concept_store.focus_plan(
                query_text=abstraction_query,
                min_observations=1,
            )
            if abstraction_query
            else self._concept_store.focus_plan()
        )
        geometric_focus = self._geometric_curiosity.focus_plan(
            query_text=abstraction_query or None,
        )
        plans = [
            plan
            for plan in (recent_query_focus, concept_focus, geometric_focus)
            if isinstance(plan, Mapping)
        ]
        if not plans:
            return None
        merged: dict[str, Any] = deepcopy(dict(plans[0]))
        for plan in plans[1:]:
            merged = self._merge_focus_plans_locked(merged, plan)
        return merged

    def _merge_focus_plans_locked(
        self,
        primary: Mapping[str, Any],
        secondary: Mapping[str, Any],
    ) -> dict[str, Any]:
        def _dedupe(values: Sequence[str], limit: int) -> list[str]:
            seen: set[str] = set()
            ordered: list[str] = []
            for raw_value in values:
                value = " ".join(str(raw_value).split()).strip()
                if not value:
                    continue
                lowered = value.lower()
                if lowered in seen:
                    continue
                seen.add(lowered)
                ordered.append(value)
                if len(ordered) >= max(1, int(limit)):
                    break
            return ordered

        gap_weights: Counter[str] = Counter()
        unsupported_weights: Counter[str] = Counter()
        query_terms: list[str] = []
        retrieval_queries: list[str] = []
        follow_up_questions: list[str] = []
        weak_concept_scores: dict[str, dict[str, Any]] = {}
        geometric_gaps: list[dict[str, Any]] = []

        for plan, plan_weight in ((primary, 1.0), (secondary, 0.75)):
            query_terms.extend(str(term) for term in list(plan.get("query_terms") or []) if str(term).strip())
            retrieval_queries.extend(str(item) for item in list(plan.get("retrieval_queries") or []) if str(item).strip())
            follow_up_questions.extend(str(item) for item in list(plan.get("follow_up_questions") or []) if str(item).strip())
            for raw_gap in list(plan.get("geometric_gaps") or []):
                if isinstance(raw_gap, Mapping):
                    geometric_gaps.append(deepcopy(dict(raw_gap)))

            for raw_term in list(plan.get("unsupported_terms") or []):
                term = str(raw_term).strip().lower()
                if not term:
                    continue
                unsupported_weights[term] += float(plan_weight)
                gap_weights[term] += float(plan_weight)

            for raw_gap in list(plan.get("gap_terms") or []):
                if not isinstance(raw_gap, Mapping):
                    continue
                term = str(raw_gap.get("term", "")).strip().lower()
                if not term:
                    continue
                gap_weights[term] += float(plan_weight) * max(0.0, float(raw_gap.get("weight", 0.0)))

            for raw_concept in list(plan.get("weak_concepts") or []):
                if not isinstance(raw_concept, Mapping):
                    continue
                label = " ".join(str(raw_concept.get("label", "")).split()).strip()
                top_terms = [
                    " ".join(str(value).split()).strip().lower()
                    for value in list(raw_concept.get("top_terms") or [])
                    if " ".join(str(value).split()).strip()
                ]
                if not label and not top_terms:
                    continue
                key = label.lower() if label else "|".join(top_terms[:3])
                if not key:
                    continue
                aggregate = weak_concept_scores.setdefault(
                    key,
                    {
                        "label": label,
                        "top_terms": [],
                        "weight_sum": 0.0,
                        "weakness_sum": 0.0,
                        "uncertainty_sum": 0.0,
                        "drift_sum": 0.0,
                        "match_count": 0,
                    },
                )
                aggregate["label"] = str(aggregate["label"] or label)
                aggregate["top_terms"] = list(dict.fromkeys([*list(aggregate["top_terms"]), *top_terms]))[:4]
                aggregate["weight_sum"] = float(aggregate["weight_sum"]) + float(plan_weight)
                aggregate["weakness_sum"] = float(aggregate["weakness_sum"]) + float(plan_weight) * max(
                    0.0,
                    float(raw_concept.get("weakness", 0.0)),
                )
                aggregate["uncertainty_sum"] = float(aggregate["uncertainty_sum"]) + float(plan_weight) * max(
                    0.0,
                    float(raw_concept.get("uncertainty", 0.0)),
                )
                aggregate["drift_sum"] = float(aggregate["drift_sum"]) + float(plan_weight) * max(
                    0.0,
                    float(raw_concept.get("drift", 0.0)),
                )
                aggregate["match_count"] = max(
                    int(aggregate["match_count"]),
                    max(0, int(raw_concept.get("match_count", 0))),
                )

        unsupported_terms = [
            term
            for term, _weight in sorted(
                unsupported_weights.items(),
                key=lambda item: (-float(item[1]), item[0]),
            )[:8]
        ]
        if not retrieval_queries and unsupported_terms:
            retrieval_queries.append(" ".join(unsupported_terms[:3]))

        weak_concepts = [
            {
                "label": str(values["label"]),
                "weakness": float(values["weakness_sum"]) / max(1e-8, float(values["weight_sum"])),
                "uncertainty": float(values["uncertainty_sum"]) / max(1e-8, float(values["weight_sum"])),
                "drift": float(values["drift_sum"]) / max(1e-8, float(values["weight_sum"])),
                "top_terms": list(values["top_terms"])[:4],
                "match_count": int(values["match_count"]),
            }
            for _key, values in sorted(
                weak_concept_scores.items(),
                key=lambda item: (
                    -(float(item[1]["weakness_sum"]) / max(1e-8, float(item[1]["weight_sum"]))),
                    -(float(item[1]["uncertainty_sum"]) / max(1e-8, float(item[1]["weight_sum"]))),
                    str(item[1]["label"] or "|".join(list(item[1]["top_terms"]))),
                ),
            )[:4]
        ]
        structural_growth = None
        secondary_growth = secondary.get("structural_growth")
        primary_growth = primary.get("structural_growth")
        if isinstance(secondary_growth, Mapping):
            structural_growth = deepcopy(dict(secondary_growth))
        elif isinstance(primary_growth, Mapping):
            structural_growth = deepcopy(dict(primary_growth))

        merged = {
            "planner_mode": "merged_runtime_abstraction_focus",
            "query_terms": _dedupe(query_terms, 8),
            "unsupported_terms": unsupported_terms,
            "gap_terms": [
                {"term": term, "weight": float(weight)}
                for term, weight in sorted(
                    gap_weights.items(),
                    key=lambda item: (-float(item[1]), item[0]),
                )[:8]
            ],
            "retrieval_queries": _dedupe(retrieval_queries, 4),
            "follow_up_questions": _dedupe(follow_up_questions, 4),
            "weak_concepts": weak_concepts,
        }
        if structural_growth is not None:
            merged["structural_growth"] = structural_growth
        if geometric_gaps:
            merged["geometric_gaps"] = geometric_gaps[:4]
        return merged

    def _recent_query_focus_plan_locked(self) -> dict[str, Any] | None:
        if not self._brain_recent_query_gaps:
            return None
        gap_weights: Counter[str] = Counter()
        unsupported_weights: Counter[str] = Counter()
        retrieval_queries: list[str] = []
        follow_up_questions: list[str] = []
        weak_concept_scores: dict[str, dict[str, Any]] = {}
        query_terms: list[str] = []
        seen_queries: set[str] = set()
        seen_questions: set[str] = set()
        seen_terms: set[str] = set()
        for index, item in enumerate(list(self._brain_recent_query_gaps)):
            recency_weight = 1.0 / float(index + 1)
            for raw_term in salient_query_terms(str(item.get("query_text", ""))):
                term = str(raw_term).strip().lower()
                if not term or term in seen_terms:
                    continue
                seen_terms.add(term)
                query_terms.append(term)
            for raw_gap in list(item.get("gap_terms") or []):
                if not isinstance(raw_gap, dict):
                    continue
                term = str(raw_gap.get("term", "")).strip().lower()
                if not term:
                    continue
                gap_weights[term] += recency_weight * max(0.0, float(raw_gap.get("weight", 0.0)))
            for raw_term in list(item.get("unsupported_terms") or []):
                term = str(raw_term).strip().lower()
                if not term:
                    continue
                unsupported_weights[term] += recency_weight
                gap_weights[term] += 2.0 * recency_weight
                if term not in seen_terms:
                    seen_terms.add(term)
                    query_terms.append(term)
            for raw_query in list(item.get("retrieval_queries") or []):
                retrieval_query = " ".join(str(raw_query).split()).strip()
                if not retrieval_query:
                    continue
                lowered = retrieval_query.lower()
                if lowered in seen_queries:
                    continue
                seen_queries.add(lowered)
                retrieval_queries.append(retrieval_query)
            for raw_question in list(item.get("follow_up_questions") or []):
                question = " ".join(str(raw_question).split()).strip()
                if not question:
                    continue
                lowered = question.lower()
                if lowered in seen_questions:
                    continue
                seen_questions.add(lowered)
                follow_up_questions.append(question)
            for raw_concept in list(item.get("weak_concepts") or []):
                if not isinstance(raw_concept, dict):
                    continue
                label = " ".join(str(raw_concept.get("label", "")).split()).strip()
                top_terms = [
                    " ".join(str(value).split()).strip().lower()
                    for value in list(raw_concept.get("top_terms") or [])
                    if " ".join(str(value).split()).strip()
                ]
                if not label and not top_terms:
                    continue
                key = label.lower() if label else "|".join(top_terms[:3])
                if not key:
                    continue
                aggregate = weak_concept_scores.setdefault(
                    key,
                    {
                        "label": label,
                        "top_terms": [],
                        "weight_sum": 0.0,
                        "weakness_sum": 0.0,
                        "uncertainty_sum": 0.0,
                        "drift_sum": 0.0,
                        "match_count": 0,
                    },
                )
                aggregate["label"] = str(aggregate["label"] or label)
                aggregate["top_terms"] = list(
                    dict.fromkeys([*list(aggregate["top_terms"]), *top_terms])
                )[:4]
                aggregate["weight_sum"] = float(aggregate["weight_sum"]) + recency_weight
                aggregate["weakness_sum"] = float(aggregate["weakness_sum"]) + recency_weight * max(
                    0.0,
                    float(raw_concept.get("weakness", 0.0)),
                )
                aggregate["uncertainty_sum"] = float(aggregate["uncertainty_sum"]) + recency_weight * max(
                    0.0,
                    float(raw_concept.get("uncertainty", 0.0)),
                )
                aggregate["drift_sum"] = float(aggregate["drift_sum"]) + recency_weight * max(
                    0.0,
                    float(raw_concept.get("drift", 0.0)),
                )
                aggregate["match_count"] = max(
                    int(aggregate["match_count"]),
                    max(0, int(raw_concept.get("match_count", 0))),
                )
        if not gap_weights and not unsupported_weights and not retrieval_queries and not follow_up_questions and not weak_concept_scores:
            return None
        unsupported_terms = [
            term
            for term, _weight in sorted(
                unsupported_weights.items(),
                key=lambda item: (-float(item[1]), item[0]),
            )[:8]
        ]
        if not retrieval_queries and unsupported_terms:
            retrieval_queries.append(" ".join(unsupported_terms[:3]))
        weak_concepts = [
            {
                "label": str(values["label"]),
                "weakness": (
                    float(values["weakness_sum"]) / max(1e-8, float(values["weight_sum"]))
                ),
                "uncertainty": (
                    float(values["uncertainty_sum"]) / max(1e-8, float(values["weight_sum"]))
                ),
                "drift": (
                    float(values["drift_sum"]) / max(1e-8, float(values["weight_sum"]))
                ),
                "top_terms": list(values["top_terms"])[:4],
                "match_count": int(values["match_count"]),
            }
            for _key, values in sorted(
                weak_concept_scores.items(),
                key=lambda item: (
                    -(
                        float(item[1]["weakness_sum"])
                        / max(1e-8, float(item[1]["weight_sum"]))
                    ),
                    -(
                        float(item[1]["uncertainty_sum"])
                        / max(1e-8, float(item[1]["weight_sum"]))
                    ),
                    str(item[1]["label"] or "|".join(list(item[1]["top_terms"]))),
                ),
            )[:4]
        ]
        return {
            "planner_mode": "recent_query_gap_focus",
            "query_terms": query_terms[:8],
            "unsupported_terms": unsupported_terms,
            "gap_terms": [
                {"term": term, "weight": float(weight)}
                for term, weight in sorted(
                    gap_weights.items(),
                    key=lambda item: (-float(item[1]), item[0]),
                )[:8]
            ],
            "retrieval_queries": retrieval_queries[:4],
            "follow_up_questions": follow_up_questions[:4],
            "weak_concepts": weak_concepts,
        }

    def _autonomy_candidate_specs_locked(
        self,
        *,
        candidate_bank: list[dict[str, Any]],
        focus_plan: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        specs = deepcopy(candidate_bank)
        if focus_plan is None:
            return specs
        retrieval_target_count = min(2, len(list(focus_plan.get("retrieval_queries") or [])))
        follow_up_target_count = 1 if list(focus_plan.get("follow_up_questions") or []) else 0
        curiosity_ready_weak_concepts = self._curiosity_ready_weak_concept_count_locked(focus_plan)
        focus_text = " ".join(
            [
                *[str(item) for item in list(focus_plan.get("query_terms") or [])[:3]],
                *[str(item) for item in list(focus_plan.get("retrieval_queries") or [])[:4]],
                *[str(item) for item in list(focus_plan.get("unsupported_terms") or [])[:3]],
            ]
        ).strip()
        if not focus_text:
            return specs
        for spec in specs:
            if str(spec.get("catalog_mode", "")).strip():
                existing_focus = " ".join(str(spec.get("catalog_focus_text", "")).split()).strip()
                if existing_focus and existing_focus.lower() != "none":
                    spec["catalog_focus_text"] = f"{existing_focus} {focus_text}".strip()
                else:
                    spec["catalog_focus_text"] = focus_text
                if str(spec.get("catalog_mode", "")).strip().lower() == "live_remote_search":
                    current_queries_per_provider = max(
                        1,
                        int(spec.get("catalog_queries_per_provider", DEFAULT_AUTONOMY_REMOTE_QUERIES_PER_PROVIDER)),
                    )
                    desired_queries_per_provider = max(
                        current_queries_per_provider,
                        min(
                            AUTO_REMOTE_QUERY_BUDGET_MAX,
                            retrieval_target_count
                            + min(2, curiosity_ready_weak_concepts)
                            + follow_up_target_count,
                        ),
                    )
                    spec["catalog_queries_per_provider"] = int(desired_queries_per_provider)
                    self._apply_provider_curriculum_locked(spec, focus_plan=focus_plan)
                continue
            metadata = dict(spec.get("metadata") or {})
            existing_query_text = " ".join(str(metadata.get("query_text", "")).split()).strip()
            if existing_query_text and existing_query_text.lower() != "none":
                metadata["query_text"] = f"{existing_query_text} {focus_text}".strip()
            else:
                metadata["query_text"] = focus_text
            metadata["semantic_relevance"] = float(metadata.get("semantic_relevance", 0.0))
            spec["metadata"] = metadata
        return specs

    def _curiosity_ready_weak_concept_count_locked(self, focus_plan: Mapping[str, Any] | None) -> int:
        if focus_plan is None:
            return 0
        ready_count = 0
        for raw_concept in list(focus_plan.get("weak_concepts") or []):
            if not isinstance(raw_concept, Mapping):
                continue
            label = " ".join(str(raw_concept.get("label", "")).split()).strip()
            top_terms = [
                " ".join(str(value).split()).strip()
                for value in list(raw_concept.get("top_terms") or [])
                if " ".join(str(value).split()).strip()
            ]
            if not label and not top_terms:
                continue
            weakness = max(0.0, min(1.0, float(raw_concept.get("weakness", 0.0))))
            uncertainty = max(0.0, min(1.0, float(raw_concept.get("uncertainty", 0.0))))
            intermediate_uncertainty = max(0.0, 1.0 - min(1.0, abs(uncertainty - 0.5) / 0.5))
            curiosity_score = 0.65 * weakness + 0.35 * intermediate_uncertainty
            if curiosity_score >= 0.45:
                ready_count += 1
        return ready_count

    def _provider_curriculum_focus_terms_locked(self, focus_plan: Mapping[str, Any] | None) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []

        def _extend(values: Sequence[str]) -> None:
            for raw_value in values:
                for term in salient_query_terms(str(raw_value)):
                    normalized = _canonical_provider_term(term)
                    if not normalized or normalized in seen:
                        continue
                    seen.add(normalized)
                    ordered.append(normalized)
                    if len(ordered) >= AUTO_REMOTE_PROVIDER_TOPIC_TERM_LIMIT:
                        return

        if focus_plan is None:
            return []

        explicit_focus_signals = [
            str(item)
            for item in list(focus_plan.get("query_terms") or [])
            if str(item).strip()
        ]
        explicit_focus_signals.extend(
            str(item)
            for item in list(focus_plan.get("unsupported_terms") or [])
            if str(item).strip()
        )
        explicit_focus_signals.extend(
            str(item.get("term", ""))
            for item in list(focus_plan.get("gap_terms") or [])
            if isinstance(item, Mapping) and str(item.get("term", "")).strip()
        )
        _extend(explicit_focus_signals)
        if ordered:
            return ordered[:AUTO_REMOTE_PROVIDER_TOPIC_TERM_LIMIT]

        fallback_focus_signals = [
            str(item)
            for item in list(focus_plan.get("focus_terms") or [])
            if str(item).strip()
        ]
        fallback_focus_signals.extend(
            str(item)
            for item in list(focus_plan.get("retrieval_queries") or [])
            if str(item).strip()
        )
        _extend(fallback_focus_signals)
        for raw_concept in list(focus_plan.get("weak_concepts") or []):
            if not isinstance(raw_concept, Mapping):
                continue
            _extend([str(item) for item in list(raw_concept.get("top_terms") or []) if str(item).strip()])
            if len(ordered) >= AUTO_REMOTE_PROVIDER_TOPIC_TERM_LIMIT:
                break
        return ordered[:AUTO_REMOTE_PROVIDER_TOPIC_TERM_LIMIT]

    def _autonomy_focus_pressure_locked(self, focus_plan: Mapping[str, Any] | None) -> tuple[float, dict[str, Any]]:
        if focus_plan is None:
            return 0.0, {
                "unsupported_term_count": 0,
                "retrieval_query_count": 0,
                "follow_up_question_count": 0,
                "gap_term_weight": 0.0,
                "weak_concept_pressure": 0.0,
                "geometric_gap_count": 0,
                "score": 0.0,
            }

        unsupported_term_count = int(len(list(focus_plan.get("unsupported_terms") or [])))
        retrieval_query_count = int(len(list(focus_plan.get("retrieval_queries") or [])))
        follow_up_question_count = int(len(list(focus_plan.get("follow_up_questions") or [])))
        gap_term_weight = float(
            sum(
                max(0.0, float(item.get("weight", 0.0)))
                for item in list(focus_plan.get("gap_terms") or [])
                if isinstance(item, Mapping)
            )
        )
        weak_concept_pressures = [
            max(
                0.0,
                min(
                    1.0,
                    0.55 * float(raw_concept.get("weakness", 0.0))
                    + 0.35 * float(raw_concept.get("uncertainty", 0.0))
                    + 0.10 * float(raw_concept.get("drift", 0.0)),
                ),
            )
            for raw_concept in list(focus_plan.get("weak_concepts") or [])
            if isinstance(raw_concept, Mapping)
        ]
        weak_concept_pressure = float(
            sum(weak_concept_pressures) / float(len(weak_concept_pressures))
        ) if weak_concept_pressures else 0.0
        geometric_gap_count = int(len(list(focus_plan.get("geometric_gaps") or [])))

        unsupported_pressure = min(1.0, float(unsupported_term_count) / 3.0)
        retrieval_pressure = min(1.0, float(retrieval_query_count) / 2.0)
        follow_up_pressure = min(1.0, float(follow_up_question_count) / 2.0)
        gap_pressure = min(1.0, gap_term_weight / 4.0)
        geometric_pressure = min(1.0, float(geometric_gap_count) / 2.0)

        pressure = float(
            min(
                1.0,
                0.24 * unsupported_pressure
                + 0.20 * retrieval_pressure
                + 0.10 * follow_up_pressure
                + 0.22 * gap_pressure
                + 0.18 * weak_concept_pressure
                + 0.06 * geometric_pressure,
            )
        )
        return pressure, {
            "unsupported_term_count": unsupported_term_count,
            "retrieval_query_count": retrieval_query_count,
            "follow_up_question_count": follow_up_question_count,
            "gap_term_weight": float(gap_term_weight),
            "weak_concept_pressure": float(weak_concept_pressure),
            "geometric_gap_count": geometric_gap_count,
            "score": float(pressure),
        }

    def _provider_topic_family_priority_locked(self, family_entry: Mapping[str, Any]) -> float:
        commits = max(0, int(family_entry.get("commits", 0)))
        successes = max(0, int(family_entry.get("successes", 0)))
        success_rate = 0.0 if commits <= 0 else float(successes) / float(commits)
        priority = float(
            0.32 * max(0.0, min(1.0, float(family_entry.get("answerability_gain_ema", 0.0))))
            + 0.22 * max(0.0, min(1.0, float(family_entry.get("uncertainty_reduction_ema", 0.0))))
            + 0.18 * max(0.0, min(1.0, float(family_entry.get("weak_concept_stabilization_ema", 0.0))))
            + 0.13 * max(0.0, min(1.0, float(family_entry.get("semantic_relevance_ema", 0.0))))
            + 0.10 * success_rate
            + 0.05 * min(1.0, float(commits) / 3.0)
        )
        return max(0.0, min(1.0, priority))

    def _provider_topic_family_match_score_locked(self, family_term: str, focus_terms: Sequence[str]) -> float:
        normalized_family = _canonical_provider_term(family_term)
        if not normalized_family:
            return 0.0
        family_tokens = {term.lower() for term in salient_query_terms(normalized_family)}
        if not family_tokens:
            family_tokens = {part for part in normalized_family.split() if part}
        if not family_tokens:
            return 0.0
        best = 0.0
        for raw_focus in focus_terms:
            normalized_focus = _canonical_provider_term(raw_focus)
            if not normalized_focus:
                continue
            if normalized_focus == normalized_family:
                return 1.0
            focus_tokens = {term.lower() for term in salient_query_terms(normalized_focus)}
            if not focus_tokens:
                focus_tokens = {part for part in normalized_focus.split() if part}
            if not focus_tokens:
                continue
            overlap = float(len(family_tokens & focus_tokens)) / float(max(len(family_tokens), len(focus_tokens)))
            if normalized_focus in normalized_family or normalized_family in normalized_focus:
                overlap = max(overlap, 0.75)
            best = max(best, overlap)
        return max(0.0, min(1.0, best))

    def _provider_topic_family_details_locked(
        self,
        entry: Mapping[str, Any],
        focus_terms: Sequence[str],
    ) -> tuple[float, list[str], int, dict[str, float], float]:
        raw_topic_families = entry.get("topic_families")
        if not focus_terms or not isinstance(raw_topic_families, Mapping):
            return 0.0, [], 0, {}, 0.0
        focus_index = {term: index for index, term in enumerate(focus_terms)}
        ranked_matches: list[tuple[float, str, int]] = []
        for raw_family, raw_family_entry in raw_topic_families.items():
            family = " ".join(str(raw_family).split()).strip().lower()
            if not family or not isinstance(raw_family_entry, Mapping):
                continue
            match_score = self._provider_topic_family_match_score_locked(family, focus_terms)
            if match_score <= 0.0:
                continue
            family_priority = self._provider_topic_family_priority_locked(raw_family_entry)
            if family_priority <= 0.0:
                continue
            commits = max(0, int(raw_family_entry.get("commits", 0)))
            ranked_matches.append((float(match_score * family_priority), family, commits))
        if not ranked_matches:
            return 0.0, [], 0, {}, 0.0
        ranked_matches.sort(
            key=lambda item: (
                -float(item[0]),
                int(focus_index.get(item[1], AUTO_REMOTE_PROVIDER_TOPIC_TERM_LIMIT)),
                item[1],
            )
        )
        top_matches = ranked_matches[: min(3, len(ranked_matches))]
        strength = max(
            0.0,
            min(
                1.0,
                sum(float(score) for score, _family, _commits in top_matches) / float(len(top_matches)),
            ),
        )
        best_score, _best_family, best_commits = top_matches[0]
        query_bonus = 0
        if best_commits >= 2 and best_score >= 0.28:
            query_bonus = 1
        if best_commits >= 4 and best_score >= 0.45:
            query_bonus = 2
        return (
            strength,
            [family for _score, family, _commits in top_matches],
            query_bonus,
            {family: float(score) for score, family, _commits in top_matches},
            float(best_score),
        )

    def _provider_query_family_priority_locked(self, family_entry: Mapping[str, Any]) -> float:
        commits = max(0, int(family_entry.get("commits", 0)))
        successes = max(0, int(family_entry.get("successes", 0)))
        success_rate = 0.0 if commits <= 0 else float(successes) / float(commits)
        priority = float(
            0.30 * max(0.0, min(1.0, float(family_entry.get("answerability_gain_ema", 0.0))))
            + 0.25 * max(0.0, min(1.0, float(family_entry.get("uncertainty_reduction_ema", 0.0))))
            + 0.20 * max(0.0, min(1.0, float(family_entry.get("weak_concept_stabilization_ema", 0.0))))
            + 0.15 * max(0.0, min(1.0, float(family_entry.get("semantic_relevance_ema", 0.0))))
            + 0.10 * success_rate
        )
        return max(0.0, min(1.0, priority))

    def _provider_query_family_match_score_locked(
        self,
        query_family: str,
        focus_terms: Sequence[str],
        focus_queries: Sequence[str],
    ) -> float:
        normalized_family = " ".join(str(query_family).split()).strip().lower()
        if not normalized_family:
            return 0.0
        family_tokens = {term.lower() for term in salient_query_terms(normalized_family)}
        if not family_tokens:
            family_tokens = {part for part in normalized_family.split() if part}
        if not family_tokens:
            return 0.0
        best = 0.0
        for raw_query in focus_queries:
            normalized_query = " ".join(str(raw_query).split()).strip().lower()
            if not normalized_query:
                continue
            if normalized_query == normalized_family:
                return 1.0
            query_tokens = {term.lower() for term in salient_query_terms(normalized_query)}
            if not query_tokens:
                query_tokens = {part for part in normalized_query.split() if part}
            if not query_tokens:
                continue
            overlap = float(len(family_tokens & query_tokens)) / float(max(len(family_tokens), len(query_tokens)))
            if normalized_query in normalized_family or normalized_family in normalized_query:
                overlap = max(overlap, 0.80)
            best = max(best, overlap)
        for raw_focus in focus_terms:
            normalized_focus = " ".join(str(raw_focus).split()).strip().lower()
            if not normalized_focus:
                continue
            focus_tokens = {term.lower() for term in salient_query_terms(normalized_focus)}
            if not focus_tokens:
                focus_tokens = {part for part in normalized_focus.split() if part}
            if not focus_tokens:
                continue
            overlap = float(len(family_tokens & focus_tokens)) / float(max(len(family_tokens), len(focus_tokens)))
            best = max(best, overlap)
        return max(0.0, min(1.0, best))

    def _provider_query_family_details_locked(
        self,
        entry: Mapping[str, Any],
        focus_plan: Mapping[str, Any] | None,
        focus_terms: Sequence[str],
    ) -> tuple[float, list[str], int, dict[str, float], float]:
        raw_query_families = entry.get("query_families")
        if not focus_terms or not isinstance(raw_query_families, Mapping):
            return 0.0, [], 0, {}, 0.0
        if not isinstance(focus_plan, Mapping) or not list(focus_plan.get("geometric_gaps") or []):
            return 0.0, [], 0, {}, 0.0
        focus_queries = [
            " ".join(str(item).split()).strip().lower()
            for item in list(focus_plan.get("retrieval_queries") or [])
            if " ".join(str(item).split()).strip()
        ]
        if not focus_queries:
            focus_queries = [
                " ".join(str(item).split()).strip().lower()
                for item in list(focus_plan.get("follow_up_questions") or [])
                if " ".join(str(item).split()).strip()
            ]
        ranked_matches: list[tuple[float, str, int]] = []
        for raw_family, raw_family_entry in raw_query_families.items():
            family = " ".join(str(raw_family).split()).strip().lower()
            if not family or not isinstance(raw_family_entry, Mapping):
                continue
            match_score = self._provider_query_family_match_score_locked(family, focus_terms, focus_queries)
            if match_score <= 0.0:
                continue
            family_priority = self._provider_query_family_priority_locked(raw_family_entry)
            if family_priority <= 0.0:
                continue
            commits = max(0, int(raw_family_entry.get("commits", 0)))
            ranked_matches.append((float(match_score * family_priority), family, commits))
        if not ranked_matches:
            return 0.0, [], 0, {}, 0.0
        ranked_matches.sort(key=lambda item: (-float(item[0]), item[1]))
        top_matches = ranked_matches[: min(3, len(ranked_matches))]
        strength = max(
            0.0,
            min(
                1.0,
                sum(float(score) for score, _family, _commits in top_matches) / float(len(top_matches)),
            ),
        )
        best_score, _best_family, best_commits = top_matches[0]
        query_bonus = 0
        if best_commits >= 1 and best_score >= 0.18:
            query_bonus = 1
        if best_commits >= 2 and best_score >= 0.35:
            query_bonus = 2
        return (
            strength,
            [family for _score, family, _commits in top_matches],
            query_bonus,
            {family: float(score) for score, family, _commits in top_matches},
            float(best_score),
        )

    def _provider_curriculum_priority_locked(
        self,
        provider: str,
        focus_plan: Mapping[str, Any] | None,
        *,
        autonomy: Mapping[str, Any],
    ) -> tuple[float, dict[str, Any]]:
        normalized_provider = " ".join(str(provider).split()).strip().lower()
        curriculum = self._normalize_provider_curriculum(autonomy.get("provider_curriculum"))
        entry = curriculum.get(normalized_provider, {})
        attempts = max(0, int(entry.get("attempts", 0)))
        commits = max(0, int(entry.get("commits", 0)))
        successes = max(0, int(entry.get("successes", 0)))
        success_rate = 0.0 if attempts <= 0 else float(successes) / float(attempts)
        commit_rate = 0.0 if attempts <= 0 else float(commits) / float(attempts)
        diagnostic_gain = max(0.0, min(1.0, float(entry.get("diagnostic_gain_ema", 0.0))))
        semantic_relevance = max(0.0, min(1.0, float(entry.get("semantic_relevance_ema", 0.0))))
        answerability_gain = max(0.0, min(1.0, float(entry.get("answerability_gain_ema", 0.0))))
        uncertainty_reduction = max(0.0, min(1.0, float(entry.get("uncertainty_reduction_ema", 0.0))))
        weak_concept_stabilization = max(
            0.0,
            min(1.0, float(entry.get("weak_concept_stabilization_ema", 0.0))),
        )
        grounded_outcome = max(0.0, min(1.0, float(entry.get("grounded_outcome_ema", 0.0))))
        grounded_family_summary = max(0.0, min(1.0, float(entry.get("grounded_family_summary_ema", 0.0))))
        delayed_consequence = max(0.0, min(1.0, float(entry.get("delayed_consequence_ema", 0.0))))
        contradiction_decay = max(0.0, min(1.0, float(entry.get("contradiction_decay_ema", 0.0))))
        focus_terms = self._provider_curriculum_focus_terms_locked(focus_plan)
        focus_pressure, _focus_pressure_details = self._autonomy_focus_pressure_locked(focus_plan)
        topic_terms = {
            str(term).strip().lower(): float(weight)
            for term, weight in dict(entry.get("topic_terms") or {}).items()
            if str(term).strip() and float(weight) > 0.0
        }
        (
            topic_family_strength,
            matched_topic_families,
            topic_family_query_bonus,
            topic_family_scores,
            topic_family_focus_score,
        ) = self._provider_topic_family_details_locked(entry, focus_terms)
        (
            query_family_strength,
            matched_query_families,
            query_family_query_bonus,
            query_family_scores,
            query_family_focus_score,
        ) = self._provider_query_family_details_locked(entry, focus_plan, focus_terms)
        topic_overlap = 0.0
        if focus_terms and topic_terms:
            denominator = sum(float(weight) for weight in topic_terms.values())
            if denominator > 0.0:
                topic_overlap = max(
                    0.0,
                    min(
                        1.0,
                        sum(float(topic_terms.get(term, 0.0)) for term in focus_terms) / float(denominator),
                    ),
                )
        focus_alignment = max(topic_overlap, topic_family_focus_score, query_family_focus_score)
        focus_alignment_ema = max(0.0, min(1.0, float(entry.get("focus_alignment_ema", 0.0))))
        utility_ema = max(0.0, min(1.0, float(entry.get("utility_ema", 0.0))))
        effective_family_summary = max(0.0, grounded_family_summary - 0.35 * contradiction_decay)
        effective_utility_ema = max(0.0, max(utility_ema, effective_family_summary) - 0.65 * contradiction_decay)
        provider_effectiveness = max(
            diagnostic_gain,
            answerability_gain,
            uncertainty_reduction,
            weak_concept_stabilization,
            grounded_outcome,
            effective_family_summary,
            effective_utility_ema,
        )
        exploration_bonus = 0.0 if attempts > 0 else 0.15
        exploration_bonus += 0.10 / math.sqrt(float(attempts) + 1.0)
        if focus_terms:
            exploration_bonus *= max(0.35, 1.0 - 0.65 * focus_pressure)
        base_priority = float(
            0.20 * success_rate
            + 0.13 * commit_rate
            + 0.15 * diagnostic_gain
            + 0.09 * semantic_relevance
            + 0.10 * answerability_gain
            + 0.07 * uncertainty_reduction
            + 0.08 * weak_concept_stabilization
            + 0.08 * topic_overlap
            + 0.10 * topic_family_strength
            + 0.20 * topic_family_focus_score
            + 0.06 * query_family_strength
            + 0.14 * query_family_focus_score
            + 0.10 * effective_utility_ema
            + 0.08 * grounded_outcome
            + 0.08 * effective_family_summary
            + 0.07 * focus_alignment_ema
            - 0.10 * contradiction_decay
            + exploration_bonus
        )
        focus_alignment_bonus = float(0.18 * focus_pressure * max(focus_alignment, focus_alignment_ema))
        off_topic_penalty = 0.0
        if focus_terms and focus_alignment < 0.20:
            off_topic_penalty = float(0.16 * focus_pressure * ((0.20 - focus_alignment) / 0.20))
        priority = max(0.0, float(base_priority + focus_alignment_bonus - off_topic_penalty))
        return priority, {
            "attempts": attempts,
            "commits": commits,
            "successes": successes,
            "success_rate": float(success_rate),
            "commit_rate": float(commit_rate),
            "diagnostic_gain_ema": float(entry.get("diagnostic_gain_ema", 0.0)),
            "semantic_relevance_ema": float(entry.get("semantic_relevance_ema", 0.0)),
            "answerability_gain_ema": float(entry.get("answerability_gain_ema", 0.0)),
            "uncertainty_reduction_ema": float(entry.get("uncertainty_reduction_ema", 0.0)),
            "weak_concept_stabilization_ema": float(entry.get("weak_concept_stabilization_ema", 0.0)),
            "utility_ema": float(utility_ema),
            "effective_utility_ema": float(effective_utility_ema),
            "focus_alignment_ema": float(focus_alignment_ema),
            "grounded_outcome_ema": float(grounded_outcome),
            "grounded_family_summary_ema": float(grounded_family_summary),
            "effective_family_summary_ema": float(effective_family_summary),
            "delayed_consequence_ema": float(delayed_consequence),
            "contradiction_decay_ema": float(contradiction_decay),
            "focus_pressure": float(focus_pressure),
            "focus_alignment": float(focus_alignment),
            "provider_effectiveness": float(provider_effectiveness),
            "exploration_bonus": float(exploration_bonus),
            "focus_alignment_bonus": float(focus_alignment_bonus),
            "off_topic_penalty": float(off_topic_penalty),
            "topic_overlap": float(topic_overlap),
            "topic_family_strength": float(topic_family_strength),
            "topic_family_focus_score": float(topic_family_focus_score),
            "topic_family_query_bonus": int(topic_family_query_bonus),
            "matched_topic_families": list(matched_topic_families),
            "topic_family_scores": dict(topic_family_scores),
            "query_family_strength": float(query_family_strength),
            "query_family_focus_score": float(query_family_focus_score),
            "query_family_query_bonus": int(query_family_query_bonus),
            "matched_query_families": list(matched_query_families),
            "query_family_scores": dict(query_family_scores),
            "last_query_text": str(entry.get("last_query_text", "")),
            "last_selected_at": str(entry.get("last_selected_at", "")),
            "topic_terms": dict(entry.get("topic_terms") or {}),
            "topic_families": dict(entry.get("topic_families") or {}),
            "query_families": dict(entry.get("query_families") or {}),
        }

    def _provider_curriculum_snapshot_locked(
        self,
        autonomy: Mapping[str, Any],
        focus_plan: Mapping[str, Any] | None,
    ) -> dict[str, Any] | None:
        curriculum = self._normalize_provider_curriculum(autonomy.get("provider_curriculum"))
        if not curriculum:
            return None
        ranked: list[dict[str, Any]] = []
        for provider in curriculum:
            priority, details = self._provider_curriculum_priority_locked(
                provider,
                focus_plan,
                autonomy=autonomy,
            )
            ranked.append(
                {
                    "provider": provider,
                    "priority": float(priority),
                    **details,
                }
            )
        ranked.sort(
            key=lambda item: (
                -float(item["priority"]),
                -int(item["successes"]),
                str(item["provider"]),
            )
        )
        return {
            "focus_terms": self._provider_curriculum_focus_terms_locked(focus_plan),
            "ranked_providers": ranked[: max(1, len(ranked))],
        }

    def _provider_curriculum_signal_locked(
        self,
        autonomy: Mapping[str, Any],
        focus_plan: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        curriculum = self._normalize_provider_curriculum(autonomy.get("provider_curriculum"))
        if not curriculum:
            return {
                "provider": None,
                "priority": 0.0,
                "signal": 0.0,
                "alignment": 0.0,
                "effectiveness": 0.0,
            }
        ranked: list[dict[str, Any]] = []
        for provider in curriculum:
            priority, details = self._provider_curriculum_priority_locked(
                provider,
                focus_plan,
                autonomy=autonomy,
            )
            alignment = max(
                float(details.get("topic_overlap", 0.0)),
                float(details.get("topic_family_focus_score", 0.0)),
                float(details.get("query_family_focus_score", 0.0)),
            )
            effectiveness = max(
                float(details.get("diagnostic_gain_ema", 0.0)),
                float(details.get("answerability_gain_ema", 0.0)),
                float(details.get("uncertainty_reduction_ema", 0.0)),
                float(details.get("weak_concept_stabilization_ema", 0.0)),
                float(details.get("effective_utility_ema", details.get("utility_ema", 0.0))),
                float(details.get("grounded_outcome_ema", 0.0)),
            )
            signal = min(1.0, 0.60 * alignment + 0.40 * effectiveness)
            ranked.append(
                {
                    "provider": provider,
                    "priority": float(priority),
                    "signal": float(signal),
                    "alignment": float(alignment),
                    "effectiveness": float(effectiveness),
                }
            )
        ranked.sort(
            key=lambda item: (
                -float(item["priority"]),
                -float(item["signal"]),
                str(item["provider"]),
            )
        )
        return ranked[0]

    def _adaptive_autonomy_settings_locked(
        self,
        autonomy: Mapping[str, Any],
        focus_plan: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        base_trigger_interval = max(1, int(autonomy.get("trigger_interval_tokens", 1)))
        base_acquisition_tokens = max(1, int(autonomy.get("acquisition_tokens", 1)))
        base_acquisition_slots = max(1, int(autonomy.get("acquisition_slots", 1)))
        focus_pressure, focus_pressure_details = self._autonomy_focus_pressure_locked(focus_plan)
        provider_signal = self._provider_curriculum_signal_locked(autonomy, focus_plan)
        provider_priority_signal = max(0.0, min(1.0, float(provider_signal.get("signal", 0.0))))
        provider_budget_signal = provider_priority_signal if focus_pressure > 0.0 else 0.0

        trigger_scale = max(
            AUTO_FOCUS_TRIGGER_INTERVAL_FLOOR,
            1.0 - 0.45 * focus_pressure - 0.20 * provider_budget_signal,
        )
        effective_trigger_interval = max(1, int(math.ceil(float(base_trigger_interval) * float(trigger_scale))))

        acquisition_token_scale = min(
            AUTO_FOCUS_ACQUISITION_TOKEN_SCALE_MAX,
            1.0 + 0.55 * focus_pressure + 0.20 * provider_budget_signal,
        )
        effective_acquisition_tokens = max(
            1,
            int(math.ceil(float(base_acquisition_tokens) * float(acquisition_token_scale))),
        )

        slot_pressure = 0.70 * focus_pressure + 0.30 * provider_budget_signal
        effective_acquisition_slots = int(base_acquisition_slots)
        if base_acquisition_slots < 2 and slot_pressure >= AUTO_FOCUS_SLOT_PRESSURE_THRESHOLD:
            effective_acquisition_slots = 2

        targeted_learning_share_target = float(
            effective_acquisition_tokens
            / float(max(1, effective_acquisition_tokens + effective_trigger_interval))
        )
        return {
            "focus_pressure": float(focus_pressure),
            "focus_pressure_details": dict(focus_pressure_details),
            "provider_priority_signal": float(provider_priority_signal),
            "provider_budget_signal": float(provider_budget_signal),
            "provider_priority_details": dict(provider_signal),
            "base_trigger_interval_tokens": int(base_trigger_interval),
            "effective_trigger_interval_tokens": int(effective_trigger_interval),
            "base_acquisition_tokens": int(base_acquisition_tokens),
            "effective_acquisition_tokens": int(effective_acquisition_tokens),
            "base_acquisition_slots": int(base_acquisition_slots),
            "effective_acquisition_slots": int(effective_acquisition_slots),
            "targeted_learning_share_target": float(targeted_learning_share_target),
        }

    def _apply_provider_response_outcome_calibration_locked(
        self,
        *,
        autonomy: dict[str, Any],
        response: Mapping[str, Any],
        outcome_score: float,
    ) -> bool:
        curriculum = self._normalize_provider_curriculum(autonomy.get("provider_curriculum"))
        weighted_providers = self._selected_evidence_weight_map(
            response,
            singular_field="provider",
            plural_field="providers",
        )
        if not curriculum or not weighted_providers:
            return False
        applied = False
        for provider, weight in weighted_providers.items():
            normalized_provider = " ".join(str(provider).split()).strip().lower()
            if not normalized_provider:
                continue
            entry = curriculum.get(normalized_provider)
            if not isinstance(entry, Mapping):
                continue
            sample = max(0.0, min(1.0, float(outcome_score) * float(weight)))
            previous_outcome = max(0.0, min(1.0, float(entry.get("grounded_outcome_ema", 0.0) or 0.0)))
            entry["grounded_outcome_ema"] = float(
                sample if previous_outcome <= 0.0 else 0.70 * previous_outcome + 0.30 * sample
            )
            previous_utility = max(0.0, min(1.0, float(entry.get("utility_ema", 0.0) or 0.0)))
            reinforced_utility = max(previous_utility, float(entry["grounded_outcome_ema"]))
            entry["utility_ema"] = float(
                reinforced_utility if previous_utility <= 0.0 else 0.75 * previous_utility + 0.25 * reinforced_utility
            )
            previous_alignment = max(0.0, min(1.0, float(entry.get("focus_alignment_ema", 0.0) or 0.0)))
            entry["focus_alignment_ema"] = float(
                float(weight)
                if previous_alignment <= 0.0
                else 0.75 * previous_alignment + 0.25 * max(float(weight), previous_alignment)
            )
            applied = True
        if applied:
            autonomy["provider_curriculum"] = curriculum
        return applied

    def _apply_provider_outcome_calibration_locked(
        self,
        *,
        autonomy: dict[str, Any],
        query_text: str,
        outcome_score: float,
    ) -> None:
        curriculum = self._normalize_provider_curriculum(autonomy.get("provider_curriculum"))
        normalized_query = " ".join(str(query_text).split()).strip()
        calibrated_score = max(0.0, min(1.0, float(outcome_score)))
        if not curriculum or not normalized_query or calibrated_score <= 0.0:
            return
        focus_plan = self._autonomy_focus_plan_locked()
        focus_terms = list(
            dict.fromkeys(
                [
                    *self._provider_curriculum_focus_terms_locked(focus_plan),
                    *[
                        _canonical_provider_term(term)
                        for term in salient_query_terms(normalized_query)
                        if _canonical_provider_term(term)
                    ],
                ]
            )
        )[:AUTO_REMOTE_PROVIDER_TOPIC_TERM_LIMIT]
        ranked: list[tuple[float, float, float, str]] = []
        for provider, entry in curriculum.items():
            priority, details = self._provider_curriculum_priority_locked(
                provider,
                focus_plan,
                autonomy=autonomy,
            )
            last_query_text = " ".join(str(entry.get("last_query_text", "")).split()).strip().lower()
            query_overlap = 0.0 if not last_query_text else self._source_text_overlap(normalized_query.lower(), last_query_text)
            focus_alignment = max(
                float(details.get("focus_alignment", 0.0)),
                float(details.get("focus_alignment_ema", 0.0)),
                float(query_overlap),
            )
            utility_signal = max(
                float(details.get("effective_utility_ema", details.get("utility_ema", 0.0))),
                float(details.get("grounded_outcome_ema", 0.0)),
            )
            ranking_score = max(float(priority), focus_alignment) * max(0.35, utility_signal if utility_signal > 0.0 else 0.35)
            ranked.append((float(ranking_score), float(focus_alignment), float(utility_signal), str(provider)))
        if not ranked:
            return
        ranked.sort(key=lambda item: (-float(item[0]), -float(item[1]), -float(item[2]), item[3]))
        _ranking_score, focus_alignment, _utility_signal, provider = ranked[0]
        if float(focus_alignment) <= 0.0:
            return
        entry = curriculum.get(provider)
        if not isinstance(entry, Mapping):
            return
        outcome_sample = max(0.0, min(1.0, calibrated_score * max(float(focus_alignment), 0.35)))
        previous_outcome = max(0.0, min(1.0, float(entry.get("grounded_outcome_ema", 0.0) or 0.0)))
        entry["grounded_outcome_ema"] = float(
            outcome_sample if previous_outcome <= 0.0 else 0.70 * previous_outcome + 0.30 * outcome_sample
        )
        previous_utility = max(0.0, min(1.0, float(entry.get("utility_ema", 0.0) or 0.0)))
        reinforced_utility = max(previous_utility, float(entry["grounded_outcome_ema"]))
        entry["utility_ema"] = float(
            reinforced_utility if previous_utility <= 0.0 else 0.75 * previous_utility + 0.25 * reinforced_utility
        )
        previous_alignment = max(0.0, min(1.0, float(entry.get("focus_alignment_ema", 0.0) or 0.0)))
        entry["focus_alignment_ema"] = float(
            max(float(focus_alignment), float(previous_alignment))
            if previous_alignment <= 0.0
            else 0.75 * previous_alignment + 0.25 * max(float(focus_alignment), float(previous_alignment))
        )
        autonomy["provider_curriculum"] = curriculum

    def _apply_provider_curriculum_locked(
        self,
        spec: dict[str, Any],
        *,
        focus_plan: Mapping[str, Any] | None,
    ) -> None:
        autonomy = cast(dict[str, Any], self._brain_config.get("autonomy") or {})
        curriculum = self._normalize_provider_curriculum(autonomy.get("provider_curriculum"))
        if not curriculum:
            return
        providers = [
            str(provider).strip()
            for provider in list(spec.get("catalog_providers") or [])
            if str(provider).strip()
        ]
        if not providers:
            return
        ranked: list[tuple[int, str, float]] = []
        priority_map: dict[str, float] = {}
        provider_topic_terms: dict[str, list[str]] = {}
        provider_query_families: dict[str, list[str]] = {}
        topic_family_query_bonus = 0
        query_family_query_bonus = 0
        for index, provider in enumerate(providers):
            priority, details = self._provider_curriculum_priority_locked(
                provider,
                focus_plan,
                autonomy=autonomy,
            )
            ranked.append((index, provider, float(priority)))
            priority_map[str(provider)] = float(priority)
            curriculum_entry = curriculum.get(str(provider).strip().lower()) or {}
            matched_topic_families = [
                str(term).strip()
                for term in list(details.get("matched_topic_families") or [])
                if str(term).strip()
            ]
            topic_terms = [
                str(term).strip()
                for term in dict(curriculum_entry.get("topic_terms") or {}).keys()
                if str(term).strip()
            ]
            if matched_topic_families:
                ordered_terms = list(dict.fromkeys([*matched_topic_families, *topic_terms]))
            elif dict(curriculum_entry.get("topic_families") or {}):
                ordered_terms = []
            else:
                ordered_terms = list(topic_terms)
            if ordered_terms:
                provider_topic_terms[str(provider)] = ordered_terms[:AUTO_REMOTE_PROVIDER_TOPIC_TERM_LIMIT]
            topic_family_query_bonus = max(topic_family_query_bonus, int(details.get("topic_family_query_bonus", 0)))
            matched_query_families = [
                str(query).strip()
                for query in list(details.get("matched_query_families") or [])
                if str(query).strip()
            ]
            if matched_query_families:
                provider_query_families[str(provider)] = matched_query_families[:AUTO_REMOTE_PROVIDER_QUERY_FAMILY_LIMIT]
            query_family_query_bonus = max(query_family_query_bonus, int(details.get("query_family_query_bonus", 0)))
        ranked.sort(key=lambda item: (-float(item[2]), int(item[0])))
        spec["catalog_providers"] = [provider for _index, provider, _priority in ranked]
        spec["catalog_provider_priority_map"] = dict(priority_map)
        if provider_topic_terms:
            spec["catalog_provider_topic_terms"] = dict(provider_topic_terms)
        if provider_query_families:
            spec["catalog_provider_query_families"] = dict(provider_query_families)
        spec["catalog_topic_family_budget_bonus"] = int(topic_family_query_bonus)
        spec["catalog_query_family_budget_bonus"] = int(query_family_query_bonus)
        total_query_bonus = int(topic_family_query_bonus) + int(query_family_query_bonus)
        if total_query_bonus > 0:
            spec["catalog_queries_per_provider"] = int(
                min(
                    AUTO_REMOTE_QUERY_BUDGET_MAX,
                    max(1, int(spec.get("catalog_queries_per_provider", 1))) + total_query_bonus,
                )
            )
        spec["catalog_provider_priority_weight"] = float(
            max(
                float(spec.get("catalog_provider_priority_weight", 0.0)),
                AUTO_REMOTE_PROVIDER_PRIORITY_WEIGHT,
            )
        )

    def _update_provider_curriculum_locked(
        self,
        *,
        autonomy: dict[str, Any],
        result: Mapping[str, Any],
        candidate_specs: Sequence[dict[str, Any]],
        focus_plan: Mapping[str, Any] | None,
    ) -> None:
        curriculum = self._normalize_provider_curriculum(autonomy.get("provider_curriculum"))

        def _ensure(provider: str) -> dict[str, Any]:
            normalized_provider = " ".join(str(provider).split()).strip().lower()
            if not normalized_provider:
                return {}
            entry = curriculum.setdefault(
                normalized_provider,
                {
                    "attempts": 0,
                    "commits": 0,
                    "successes": 0,
                    "gap_gain_ema": 0.0,
                    "diagnostic_gain_ema": 0.0,
                    "semantic_relevance_ema": 0.0,
                    "answerability_gain_ema": 0.0,
                    "uncertainty_reduction_ema": 0.0,
                    "weak_concept_stabilization_ema": 0.0,
                    "utility_ema": 0.0,
                    "focus_alignment_ema": 0.0,
                    "grounded_outcome_ema": 0.0,
                    "grounded_family_summary_ema": 0.0,
                    "delayed_consequence_ema": 0.0,
                    "contradiction_decay_ema": 0.0,
                    "last_query_text": "",
                    "last_selected_at": "",
                    "topic_terms": {},
                    "topic_families": {},
                    "query_families": {},
                },
            )
            entry["topic_terms"] = dict(entry.get("topic_terms") or {})
            entry["topic_families"] = dict(entry.get("topic_families") or {})
            entry["query_families"] = dict(entry.get("query_families") or {})
            return entry

        attempted_providers: list[str] = []
        for spec in candidate_specs:
            if str(spec.get("catalog_mode", "")).strip().lower() != "live_remote_search":
                continue
            attempted_providers.extend(
                str(provider).strip().lower()
                for provider in list(spec.get("catalog_providers") or [])
                if str(provider).strip()
            )
        for provider in dict.fromkeys(attempted_providers):
            entry = _ensure(provider)
            if entry:
                entry["attempts"] = int(entry.get("attempts", 0)) + 1

        current_focus_terms = self._provider_curriculum_focus_terms_locked(focus_plan)
        weak_concepts = [
            item
            for item in list((focus_plan or {}).get("weak_concepts") or [])
            if isinstance(item, Mapping)
        ]
        weak_focus_scale = 0.0
        if weak_concepts:
            weak_focus_scale = max(
                0.0,
                min(
                    1.0,
                    sum(
                        max(
                            0.0,
                            min(
                                1.0,
                                0.5 * float(item.get("weakness", 0.0))
                                + 0.5 * float(item.get("uncertainty", 0.0)),
                            ),
                        )
                        for item in weak_concepts
                    )
                    / float(len(weak_concepts)),
                ),
            )
        for raw_row in list(result.get("acquisition_history") or []):
            if not isinstance(raw_row, Mapping):
                continue
            provider = " ".join(str(raw_row.get("selected_provider", "")).split()).strip().lower()
            selected_metadata = raw_row.get("selected_metadata")
            if not provider and isinstance(selected_metadata, Mapping):
                provider = " ".join(str(selected_metadata.get("provider", "")).split()).strip().lower()
            entry = _ensure(provider)
            if not entry:
                continue
            entry["commits"] = int(entry.get("commits", 0)) + 1
            gap_gain = max(0.0, float(raw_row.get("selected_gap_reduction", 0.0)))
            diagnostic_gain = max(0.0, float(raw_row.get("selected_diagnostic_gap_reduction", 0.0)))
            semantic_relevance = max(0.0, min(1.0, float(raw_row.get("selected_semantic_relevance", 0.0))))
            selected_source = " ".join(str(raw_row.get("selected_source", "")).split()).strip()
            candidate_snapshot = raw_row.get("candidate_snapshot")
            before_metrics = {}
            if (
                selected_source
                and isinstance(candidate_snapshot, Mapping)
                and isinstance(candidate_snapshot.get(selected_source), Mapping)
            ):
                before_metrics = cast(Mapping[str, Any], candidate_snapshot.get(selected_source))
            answerability_before = max(
                0.0,
                min(1.0, float(before_metrics.get("semantic_answerability", 0.0) or 0.0)),
            )
            answerability_after = max(
                0.0,
                min(
                    1.0,
                    float(raw_row.get("selected_semantic_answerability_after", answerability_before) or answerability_before),
                ),
            )
            answerability_gain = max(0.0, answerability_after - answerability_before)
            uncertainty_before = max(
                0.0,
                min(1.0, float(before_metrics.get("concept_uncertainty", 0.0) or 0.0)),
            )
            uncertainty_after = max(
                0.0,
                min(
                    1.0,
                    float(raw_row.get("selected_concept_uncertainty_after", uncertainty_before) or uncertainty_before),
                ),
            )
            uncertainty_reduction = max(0.0, uncertainty_before - uncertainty_after)
            support_before = max(
                0.0,
                min(1.0, float(before_metrics.get("concept_support", 0.0) or 0.0)),
            )
            support_after = max(
                0.0,
                min(1.0, float(raw_row.get("selected_concept_support_after", support_before) or support_before)),
            )
            support_gain = max(0.0, support_after - support_before)
            weak_pressure_before = max(
                0.0,
                min(1.0, float(before_metrics.get("semantic_weak_concept_pressure", 0.0) or 0.0)),
            )
            weak_pressure_after = max(
                0.0,
                min(
                    1.0,
                    float(raw_row.get("selected_weak_concept_pressure_after", weak_pressure_before) or weak_pressure_before),
                ),
            )
            weak_pressure_reduction = max(0.0, weak_pressure_before - weak_pressure_after)
            weak_concept_stabilization = max(
                0.0,
                min(
                    1.0,
                    weak_focus_scale
                    * (
                        0.50 * uncertainty_reduction
                        + 0.30 * support_gain
                        + 0.20 * weak_pressure_reduction
                    ),
                ),
            )
            entry["gap_gain_ema"] = float(
                gap_gain
                if int(entry["commits"]) <= 1
                else 0.70 * float(entry.get("gap_gain_ema", 0.0)) + 0.30 * gap_gain
            )
            entry["diagnostic_gain_ema"] = float(
                diagnostic_gain
                if int(entry["commits"]) <= 1
                else 0.70 * float(entry.get("diagnostic_gain_ema", 0.0)) + 0.30 * diagnostic_gain
            )
            entry["semantic_relevance_ema"] = float(
                semantic_relevance
                if int(entry["commits"]) <= 1
                else 0.75 * float(entry.get("semantic_relevance_ema", 0.0)) + 0.25 * semantic_relevance
            )
            entry["answerability_gain_ema"] = float(
                answerability_gain
                if int(entry["commits"]) <= 1
                else 0.75 * float(entry.get("answerability_gain_ema", 0.0)) + 0.25 * answerability_gain
            )
            entry["uncertainty_reduction_ema"] = float(
                uncertainty_reduction
                if int(entry["commits"]) <= 1
                else 0.75 * float(entry.get("uncertainty_reduction_ema", 0.0)) + 0.25 * uncertainty_reduction
            )
            entry["weak_concept_stabilization_ema"] = float(
                weak_concept_stabilization
                if int(entry["commits"]) <= 1
                else 0.75 * float(entry.get("weak_concept_stabilization_ema", 0.0))
                + 0.25 * weak_concept_stabilization
            )
            if gap_gain > 0.0 or diagnostic_gain > 0.0 or answerability_gain > 0.0 or weak_concept_stabilization > 0.0:
                entry["successes"] = int(entry.get("successes", 0)) + 1
            query_text = " ".join(str(raw_row.get("selected_query_text", "")).split()).strip()
            if not query_text and isinstance(selected_metadata, Mapping):
                query_text = " ".join(str(selected_metadata.get("query_text", "")).split()).strip()
            if query_text:
                entry["last_query_text"] = query_text
            entry["last_selected_at"] = datetime.now(timezone.utc).isoformat()
            topic_terms = {
                str(term).strip().lower(): float(weight)
                for term, weight in dict(entry.get("topic_terms") or {}).items()
                if str(term).strip() and float(weight) > 0.0
            }
            for term in list(topic_terms):
                topic_terms[term] = float(topic_terms[term]) * 0.85
                if topic_terms[term] < 0.05:
                    topic_terms.pop(term, None)
            metadata_terms: list[str] = []
            if isinstance(selected_metadata, Mapping):
                metadata_terms = [
                    _canonical_provider_term(term)
                    for term in list(selected_metadata.get("catalog_terms") or [])
                    if _canonical_provider_term(term)
                ]
            focus_alignment_sample = 0.0
            selected_terms = list(metadata_terms)
            if not selected_terms and query_text:
                selected_terms = [
                    normalized
                    for normalized in (_canonical_provider_term(term) for term in salient_query_terms(query_text))
                    if normalized
                ]
            focus_term_tokens = {
                _canonical_provider_term(term)
                for raw_term in current_focus_terms
                for term in salient_query_terms(str(raw_term))
                if _canonical_provider_term(term)
            }
            selected_term_tokens = {
                _canonical_provider_term(term)
                for raw_term in selected_terms
                for term in salient_query_terms(str(raw_term))
                if _canonical_provider_term(term)
            }
            if query_text:
                selected_term_tokens.update(
                    _canonical_provider_term(term)
                    for term in salient_query_terms(query_text)
                    if _canonical_provider_term(term)
                )
            if focus_term_tokens and selected_term_tokens:
                token_overlap = len(focus_term_tokens & selected_term_tokens) / max(
                    1.0,
                    min(float(len(focus_term_tokens)), float(len(selected_term_tokens))),
                )
                phrase_hits = sum(1 for term in list(current_focus_terms)[:4] if term and term in query_text.lower())
                phrase_bonus = min(1.0, 0.34 * phrase_hits)
                focus_alignment_sample = max(0.0, min(1.0, 0.75 * token_overlap + 0.25 * phrase_bonus))
            utility_sample = max(
                0.0,
                min(
                    1.0,
                    0.30 * semantic_relevance
                    + 0.22 * answerability_gain
                    + 0.18 * uncertainty_reduction
                    + 0.12 * weak_concept_stabilization
                    + 0.18 * focus_alignment_sample,
                ),
            )
            entry["focus_alignment_ema"] = float(
                focus_alignment_sample
                if int(entry["commits"]) <= 1
                else 0.75 * float(entry.get("focus_alignment_ema", 0.0)) + 0.25 * focus_alignment_sample
            )
            entry["utility_ema"] = float(
                utility_sample
                if int(entry["commits"]) <= 1
                else 0.75 * float(entry.get("utility_ema", 0.0)) + 0.25 * utility_sample
            )
            update_terms = list(dict.fromkeys([*current_focus_terms, *metadata_terms]))
            if not update_terms and query_text:
                update_terms = [
                    normalized
                    for normalized in (_canonical_provider_term(term) for term in salient_query_terms(query_text))
                    if normalized
                ]
            for rank, term in enumerate(update_terms[:AUTO_REMOTE_PROVIDER_TOPIC_TERM_LIMIT]):
                topic_terms[term] = float(topic_terms.get(term, 0.0)) + 1.0 / float(rank + 1)
            entry["topic_terms"] = dict(
                sorted(
                    topic_terms.items(),
                    key=lambda item: (-float(item[1]), item[0]),
                )[:AUTO_REMOTE_PROVIDER_TOPIC_TERM_LIMIT]
            )
            topic_families: dict[str, dict[str, Any]] = {}
            for raw_family, raw_family_entry in dict(entry.get("topic_families") or {}).items():
                family = _canonical_provider_term(raw_family)
                if not family or not isinstance(raw_family_entry, Mapping):
                    continue
                topic_families[family] = {
                    "commits": max(0, int(raw_family_entry.get("commits", 0))),
                    "successes": max(0, int(raw_family_entry.get("successes", 0))),
                    "semantic_relevance_ema": max(
                        0.0,
                        float(raw_family_entry.get("semantic_relevance_ema", 0.0)),
                    ),
                    "answerability_gain_ema": max(
                        0.0,
                        float(raw_family_entry.get("answerability_gain_ema", 0.0)),
                    ),
                    "uncertainty_reduction_ema": max(
                        0.0,
                        float(raw_family_entry.get("uncertainty_reduction_ema", 0.0)),
                    ),
                    "weak_concept_stabilization_ema": max(
                        0.0,
                        float(raw_family_entry.get("weak_concept_stabilization_ema", 0.0)),
                    ),
                    "last_selected_at": " ".join(str(raw_family_entry.get("last_selected_at", "")).split()).strip(),
                }
            for family_term in update_terms[:AUTO_REMOTE_PROVIDER_TOPIC_TERM_LIMIT]:
                topic_family = topic_families.setdefault(
                    family_term,
                    {
                        "commits": 0,
                        "successes": 0,
                        "semantic_relevance_ema": 0.0,
                        "answerability_gain_ema": 0.0,
                        "uncertainty_reduction_ema": 0.0,
                        "weak_concept_stabilization_ema": 0.0,
                        "last_selected_at": "",
                    },
                )
                topic_family["commits"] = int(topic_family.get("commits", 0)) + 1
                if gap_gain > 0.0 or diagnostic_gain > 0.0 or answerability_gain > 0.0 or weak_concept_stabilization > 0.0:
                    topic_family["successes"] = int(topic_family.get("successes", 0)) + 1
                topic_family["semantic_relevance_ema"] = float(
                    semantic_relevance
                    if int(topic_family["commits"]) <= 1
                    else 0.75 * float(topic_family.get("semantic_relevance_ema", 0.0)) + 0.25 * semantic_relevance
                )
                topic_family["answerability_gain_ema"] = float(
                    answerability_gain
                    if int(topic_family["commits"]) <= 1
                    else 0.75 * float(topic_family.get("answerability_gain_ema", 0.0)) + 0.25 * answerability_gain
                )
                topic_family["uncertainty_reduction_ema"] = float(
                    uncertainty_reduction
                    if int(topic_family["commits"]) <= 1
                    else 0.75 * float(topic_family.get("uncertainty_reduction_ema", 0.0))
                    + 0.25 * uncertainty_reduction
                )
                topic_family["weak_concept_stabilization_ema"] = float(
                    weak_concept_stabilization
                    if int(topic_family["commits"]) <= 1
                    else 0.75 * float(topic_family.get("weak_concept_stabilization_ema", 0.0))
                    + 0.25 * weak_concept_stabilization
                )
                topic_family["last_selected_at"] = entry["last_selected_at"]
            entry["topic_families"] = dict(
                sorted(
                    topic_families.items(),
                    key=lambda item: (-self._provider_topic_family_priority_locked(item[1]), item[0]),
                )[:AUTO_REMOTE_PROVIDER_TOPIC_TERM_LIMIT]
            )
            query_families: dict[str, dict[str, Any]] = {}
            for raw_family, raw_family_entry in dict(entry.get("query_families") or {}).items():
                family = _canonical_provider_term(raw_family)
                if not family or not isinstance(raw_family_entry, Mapping):
                    continue
                query_families[family] = {
                    "commits": max(0, int(raw_family_entry.get("commits", 0))),
                    "successes": max(0, int(raw_family_entry.get("successes", 0))),
                    "semantic_relevance_ema": max(
                        0.0,
                        float(raw_family_entry.get("semantic_relevance_ema", 0.0)),
                    ),
                    "answerability_gain_ema": max(
                        0.0,
                        float(raw_family_entry.get("answerability_gain_ema", 0.0)),
                    ),
                    "uncertainty_reduction_ema": max(
                        0.0,
                        float(raw_family_entry.get("uncertainty_reduction_ema", 0.0)),
                    ),
                    "weak_concept_stabilization_ema": max(
                        0.0,
                        float(raw_family_entry.get("weak_concept_stabilization_ema", 0.0)),
                    ),
                    "last_selected_at": " ".join(str(raw_family_entry.get("last_selected_at", "")).split()).strip(),
                }
            geometric_queries = [
                " ".join(str(query).split()).strip().lower()
                for query in list((focus_plan or {}).get("retrieval_queries") or [])
                if " ".join(str(query).split()).strip()
            ]
            update_query_families = list(dict.fromkeys([query_text.lower(), *geometric_queries])) if query_text else list(dict.fromkeys(geometric_queries))
            if not update_query_families and query_text:
                update_query_families = [query_text.lower()]
            for family_query in update_query_families[:AUTO_REMOTE_PROVIDER_QUERY_FAMILY_LIMIT]:
                query_family = query_families.setdefault(
                    family_query,
                    {
                        "commits": 0,
                        "successes": 0,
                        "semantic_relevance_ema": 0.0,
                        "answerability_gain_ema": 0.0,
                        "uncertainty_reduction_ema": 0.0,
                        "weak_concept_stabilization_ema": 0.0,
                        "last_selected_at": "",
                    },
                )
                query_family["commits"] = int(query_family.get("commits", 0)) + 1
                if gap_gain > 0.0 or diagnostic_gain > 0.0 or answerability_gain > 0.0 or weak_concept_stabilization > 0.0:
                    query_family["successes"] = int(query_family.get("successes", 0)) + 1
                query_family["semantic_relevance_ema"] = float(
                    semantic_relevance
                    if int(query_family["commits"]) <= 1
                    else 0.75 * float(query_family.get("semantic_relevance_ema", 0.0)) + 0.25 * semantic_relevance
                )
                query_family["answerability_gain_ema"] = float(
                    answerability_gain
                    if int(query_family["commits"]) <= 1
                    else 0.75 * float(query_family.get("answerability_gain_ema", 0.0)) + 0.25 * answerability_gain
                )
                query_family["uncertainty_reduction_ema"] = float(
                    uncertainty_reduction
                    if int(query_family["commits"]) <= 1
                    else 0.75 * float(query_family.get("uncertainty_reduction_ema", 0.0))
                    + 0.25 * uncertainty_reduction
                )
                query_family["weak_concept_stabilization_ema"] = float(
                    weak_concept_stabilization
                    if int(query_family["commits"]) <= 1
                    else 0.75 * float(query_family.get("weak_concept_stabilization_ema", 0.0))
                    + 0.25 * weak_concept_stabilization
                )
                query_family["last_selected_at"] = entry["last_selected_at"]
            entry["query_families"] = dict(
                sorted(
                    query_families.items(),
                    key=lambda item: (-self._provider_query_family_priority_locked(item[1]), item[0]),
                )[:AUTO_REMOTE_PROVIDER_QUERY_FAMILY_LIMIT]
            )

        autonomy["provider_curriculum"] = curriculum

    def _candidate_pool_size_hint(self, candidate_bank: Sequence[dict[str, Any]]) -> int:
        estimated_pool_size = 0
        for spec in candidate_bank:
            if not isinstance(spec, dict):
                continue
            catalog_mode = str(spec.get("catalog_mode", "")).strip().lower()
            if not catalog_mode:
                estimated_pool_size += 1
                continue
            catalog_entries = spec.get("catalog_entries")
            entry_count = 0
            if isinstance(catalog_entries, Sequence) and not isinstance(catalog_entries, (str, bytes)):
                entry_count = len(list(catalog_entries))
            catalog_limit = max(1, int(spec.get("catalog_limit", max(1, entry_count or 1))))
            probe_pool_limit = int(spec.get("catalog_probe_pool_limit", 0) or 0)
            if probe_pool_limit > 0:
                estimated_pool_size += max(catalog_limit, probe_pool_limit, entry_count)
                continue
            if catalog_mode == "live_remote_search":
                provider_count = max(
                    1,
                    len(
                        [
                            str(item).strip()
                            for item in list(spec.get("catalog_providers") or [])
                            if str(item).strip()
                        ]
                    ),
                )
                query_count = max(1, int(spec.get("catalog_queries_per_provider", 2)))
                result_limit = max(1, int(spec.get("catalog_provider_result_limit", catalog_limit)))
                estimated_pool_size += max(catalog_limit, provider_count * query_count * result_limit)
                continue
            estimated_pool_size += max(catalog_limit, entry_count)
        return estimated_pool_size

    def _autonomy_shortlist_settings_locked(
        self,
        *,
        candidate_bank: list[dict[str, Any]],
        config: dict[str, Any],
        focus_plan: dict[str, Any] | None,
    ) -> tuple[int, float, float]:
        shortlist_size = max(0, int(config.get("semantic_shortlist_size", 0)))
        gap_weight = float(config.get("semantic_shortlist_gap_weight", 0.5))
        affinity_weight = float(config.get("semantic_shortlist_affinity_weight", 0.5))
        if shortlist_size > 0:
            return shortlist_size, gap_weight, affinity_weight
        if focus_plan is None:
            return shortlist_size, gap_weight, affinity_weight

        focus_signal_count = int(len(list(focus_plan.get("unsupported_terms") or [])))
        focus_signal_count += int(len(list(focus_plan.get("retrieval_queries") or [])))
        focus_signal_count += int(len(list(focus_plan.get("gap_terms") or [])))
        focus_signal_count += int(len(list(focus_plan.get("weak_concepts") or [])))
        if focus_signal_count <= 0:
            return shortlist_size, gap_weight, affinity_weight

        estimated_pool_size = self._candidate_pool_size_hint(candidate_bank)
        if estimated_pool_size <= 1:
            return shortlist_size, gap_weight, affinity_weight
        auto_size = max(1, min(AUTO_FOCUS_SHORTLIST_MAX_SIZE, (estimated_pool_size + 1) // 2))
        return auto_size, AUTO_FOCUS_SHORTLIST_GAP_WEIGHT, AUTO_FOCUS_SHORTLIST_AFFINITY_WEIGHT

