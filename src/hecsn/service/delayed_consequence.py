"""Delayed-consequence tracking helpers for Terminus.

This tracker owns the long-horizon consequence record state machines for
sources/providers and response provenance. Source utility mutation is owned by
Brain Runtime through an explicit interface so the learning rules stay
separable from runtime control.
"""

from __future__ import annotations

from collections import Counter, deque
from copy import deepcopy
from datetime import datetime, timezone
import math
from typing import Any, Mapping, Sequence, cast
from uuid import uuid4

from hecsn.semantics.grounding_text import salient_query_terms
from hecsn.service.manager_bound_module import ManagerBoundModule
from hecsn.service.runtime_sources import _BrainSourceRuntime
from hecsn.service.terminus_autonomy import _canonical_provider_term

DEFAULT_BRAIN_TICK_TOKENS = 512
DEFAULT_DELAYED_CONSEQUENCE_RECORDS = 24
DEFAULT_DELAYED_CONSEQUENCE_MATCH_THRESHOLD = 0.34
DEFAULT_DELAYED_CONSEQUENCE_DELTA_THRESHOLD = 0.08
DEFAULT_DELAYED_CONTRADICTION_DECAY_THRESHOLD = 0.18
DEFAULT_DELAYED_CONTRADICTION_UNSUPPORTED_THRESHOLD = 0.34
DEFAULT_DELAYED_CONSEQUENCE_COOLING_START_TOKENS = 512
DEFAULT_DELAYED_CONSEQUENCE_COOLING_WINDOW_TOKENS = 1024
DEFAULT_DELAYED_CONSEQUENCE_RETIREMENT_TOKENS = 4096
DEFAULT_DELAYED_CONSEQUENCE_RETIREMENT_BALANCE_THRESHOLD = 0.05
DEFAULT_DELAYED_CONSEQUENCE_AGGREGATION_MATCH_THRESHOLD = 0.52
DEFAULT_DELAYED_CONSEQUENCE_AGGREGATION_PROVENANCE_THRESHOLD = 0.60
DEFAULT_DELAYED_CONSEQUENCE_AGGREGATION_QUERY_EXAMPLE_LIMIT = 6
DEFAULT_DELAYED_CONSEQUENCE_AGGREGATION_TERM_LIMIT = 12
DEFAULT_DELAYED_CONSEQUENCE_AGGREGATION_SUPPORT_SCALE = 0.18
DEFAULT_DELAYED_CONSEQUENCE_AGGREGATION_SUPPORT_MAX = 1.35
DEFAULT_DELAYED_CONSEQUENCE_TRAJECTORY_STATE_THRESHOLD = 0.12
DEFAULT_DELAYED_CONSEQUENCE_TRAJECTORY_SUPPORT_MAX = 1.25
DEFAULT_DELAYED_CONSEQUENCE_TRAJECTORY_RECENT_ALPHA = 0.50
DEFAULT_DELAYED_CONSEQUENCE_TRAJECTORY_SCORE_LIMIT = 4.0
DEFAULT_DELAYED_CONSEQUENCE_SPLIT_MAX_BRANCH_OVERLAP = 0.70
DEFAULT_DELAYED_CONSEQUENCE_SPLIT_MIN_BRANCH_OCCURRENCES = 1
DEFAULT_DELAYED_CONSEQUENCE_REMERGE_MIN_CROSS_OCCURRENCES = 1
DEFAULT_FORGIVENESS_RECOVERY_RATIO = 0.80
DEFAULT_UTILITY_PENALTY_WEIGHT = 0.65


def _build_delayed_consequence_initial_state() -> dict[str, Any]:
    return {
        "_delayed_consequence_records": deque(maxlen=DEFAULT_DELAYED_CONSEQUENCE_RECORDS),
        "_delayed_consequence_cooled_total": 0,
        "_delayed_consequence_retired_total": 0,
        "_delayed_consequence_compacted_total": 0,
        "_delayed_consequence_split_total": 0,
        "_delayed_consequence_remerged_total": 0,
    }


DELAYED_CONSEQUENCE_STATE_FIELDS = frozenset(_build_delayed_consequence_initial_state())


def _restore_non_negative_int(state: dict[str, Any], key: str) -> int:
    """Restore a non-negative integer total from checkpoint state."""
    return max(0, int(state.get(key, 0) or 0))


class DelayedConsequenceTracker(ManagerBoundModule):

    def __init__(self, manager: Any | None = None) -> None:
        object.__setattr__(self, "_manager", manager)
        for field_name, initial_value in _build_delayed_consequence_initial_state().items():
            object.__setattr__(self, field_name, initial_value)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "_manager" or name in DELAYED_CONSEQUENCE_STATE_FIELDS:
            object.__setattr__(self, name, value)
            return
        manager = object.__getattribute__(self, "_manager")
        if manager is None or manager is self:
            object.__setattr__(self, name, value)
            return
        setattr(manager, name, value)

    @staticmethod
    def _consequence_query_terms(value: Any) -> list[str]:
        normalized_text = " ".join(str(value).split()).strip()
        if not normalized_text:
            return []
        ordered: list[str] = []
        seen: set[str] = set()
        for raw_term in salient_query_terms(normalized_text):
            term = _canonical_provider_term(raw_term)
            if not term or term in seen:
                continue
            seen.add(term)
            ordered.append(term)
            if len(ordered) >= 8:
                break
        return ordered

    def _query_progress_snapshot_locked(
        self,
        query_result: Mapping[str, Any],
    ) -> dict[str, Any]:
        query_summary = query_result.get("query_summary") if isinstance(query_result.get("query_summary"), Mapping) else {}
        gap_plan = query_result.get("gap_plan") if isinstance(query_result.get("gap_plan"), Mapping) else {}
        query_text = self._normalize_action_text(query_summary.get("query_text", ""))
        query_terms = [
            _canonical_provider_term(term)
            for term in list(gap_plan.get("query_terms") or [])
            if _canonical_provider_term(term)
        ]
        if not query_terms:
            query_terms = self._consequence_query_terms(query_text)
        candidate_items = list(query_summary.get("memory_episodes") or query_summary.get("memory_matches") or [])
        query_term_count = max(1, len(query_terms))
        top_similarity = 0.0
        top_query_overlap_ratio = 0.0
        supported_episode_hits = 0
        for raw_item in candidate_items[:3]:
            if not isinstance(raw_item, Mapping):
                continue
            similarity = max(0.0, min(1.0, float(raw_item.get("similarity", 0.0) or 0.0)))
            query_overlap = max(0, int(raw_item.get("query_overlap", 0) or 0))
            top_similarity = max(top_similarity, similarity)
            top_query_overlap_ratio = max(
                top_query_overlap_ratio,
                min(1.0, float(query_overlap) / float(query_term_count)),
            )
            if query_overlap > 0:
                supported_episode_hits += 1
        grounded_fraction = max(0.0, min(1.0, float(gap_plan.get("grounded_fraction", 0.0) or 0.0)))
        support_episode_bonus = min(1.0, float(supported_episode_hits) / 2.0)
        query_score = max(
            0.0,
            min(
                1.0,
                0.60 * grounded_fraction
                + 0.20 * top_query_overlap_ratio
                + 0.10 * top_similarity
                + 0.10 * support_episode_bonus,
            ),
        )
        return {
            "query_text": query_text,
            "query_terms": list(query_terms),
            "grounded_fraction": float(grounded_fraction),
            "query_score": float(query_score),
            "top_similarity": float(top_similarity),
            "top_query_overlap_ratio": float(top_query_overlap_ratio),
            "supported_episode_hits": int(supported_episode_hits),
            "memory_episode_count": int(len(candidate_items)),
            "unsupported_terms": [
                _canonical_provider_term(term)
                for term in list(gap_plan.get("unsupported_terms") or [])
                if _canonical_provider_term(term)
            ],
        }

    @staticmethod
    def _delayed_consequence_query_examples(record: Mapping[str, Any]) -> list[str]:
        ordered: list[str] = []
        seen: set[str] = set()

        def _append(raw_value: Any) -> None:
            text = " ".join(str(raw_value).split()).strip()
            if not text:
                return
            key = text.lower()
            if key in seen:
                return
            seen.add(key)
            ordered.append(text)

        _append(record.get("query_text", ""))
        raw_examples = record.get("query_examples")
        if isinstance(raw_examples, Sequence) and not isinstance(raw_examples, (str, bytes)):
            for raw_value in list(raw_examples):
                _append(raw_value)
                if len(ordered) >= DEFAULT_DELAYED_CONSEQUENCE_AGGREGATION_QUERY_EXAMPLE_LIMIT:
                    break
        return ordered[:DEFAULT_DELAYED_CONSEQUENCE_AGGREGATION_QUERY_EXAMPLE_LIMIT]

    def _delayed_consequence_match_score_locked(
        self,
        record: Mapping[str, Any],
        query_snapshot: Mapping[str, Any],
    ) -> float:
        record_queries = [text.lower() for text in self._delayed_consequence_query_examples(record) if text]
        current_query = self._normalize_action_text(query_snapshot.get("query_text", "")).lower()
        if not record_queries or not current_query:
            return 0.0
        record_terms = {
            _canonical_provider_term(term)
            for term in list(record.get("query_terms") or [])
            if _canonical_provider_term(term)
        }
        if not record_terms:
            record_terms = {
                _canonical_provider_term(term)
                for query_text in record_queries
                for term in self._consequence_query_terms(query_text)
                if _canonical_provider_term(term)
            }
        current_terms = {
            _canonical_provider_term(term)
            for term in list(query_snapshot.get("query_terms") or self._consequence_query_terms(current_query))
            if _canonical_provider_term(term)
        }
        term_overlap = 0.0
        if record_terms and current_terms:
            term_overlap = float(len(record_terms & current_terms)) / float(max(1, min(len(record_terms), len(current_terms))))
        text_overlap = max(self._source_text_overlap(record_query, current_query) for record_query in record_queries)
        return max(0.0, min(1.0, max(term_overlap, 0.65 * term_overlap + 0.35 * text_overlap, text_overlap)))

    def _recent_action_contradiction_signal_locked(self, query_text: str) -> tuple[float, int]:
        contradicted_records = self._recent_relevant_action_records_locked(
            query_text,
            statuses=["contradicted"],
            limit=3,
        )
        if not contradicted_records:
            return 0.0, 0
        best_signal = 0.0
        for record in contradicted_records:
            verification = record.get("verification") if isinstance(record.get("verification"), Mapping) else {}
            confidence = max(0.0, min(1.0, float(verification.get("confidence", 0.0) or 0.0)))
            relevance = max(0.0, min(1.0, self._action_record_relevance_score_locked(record, query_text)))
            signal = max(0.0, min(1.0, relevance * max(0.35, 0.55 + 0.45 * confidence)))
            best_signal = max(best_signal, signal)
        return float(best_signal), int(len(contradicted_records))

    @staticmethod
    def _delayed_consequence_support_multiplier(record: Mapping[str, Any]) -> float:
        aggregate_count = max(1, int(record.get("aggregate_count", 1) or 1))
        if aggregate_count <= 1:
            return 1.0
        return float(
            min(
                DEFAULT_DELAYED_CONSEQUENCE_AGGREGATION_SUPPORT_MAX,
                1.0 + DEFAULT_DELAYED_CONSEQUENCE_AGGREGATION_SUPPORT_SCALE * math.log1p(float(aggregate_count - 1)),
            )
        )

    @staticmethod
    def _delayed_consequence_trajectory_totals(record: Mapping[str, Any]) -> tuple[float, float, float, float]:
        credit_total = max(0.0, float(record.get("trajectory_credit_total", 0.0) or 0.0))
        penalty_total = max(0.0, float(record.get("trajectory_penalty_total", 0.0) or 0.0))
        forgiveness_total = max(0.0, float(record.get("trajectory_forgiveness_total", 0.0) or 0.0))
        raw_net = float(record.get("trajectory_net_score", credit_total + forgiveness_total - penalty_total) or 0.0)
        net_score = max(
            -DEFAULT_DELAYED_CONSEQUENCE_TRAJECTORY_SCORE_LIMIT,
            min(DEFAULT_DELAYED_CONSEQUENCE_TRAJECTORY_SCORE_LIMIT, raw_net),
        )
        return float(credit_total), float(penalty_total), float(forgiveness_total), float(net_score)

    @classmethod
    def _delayed_consequence_trajectory_balance(cls, record: Mapping[str, Any]) -> float:
        credit_total, penalty_total, forgiveness_total, _net_score = cls._delayed_consequence_trajectory_totals(record)
        positive_total = max(0.0, credit_total + forgiveness_total)
        negative_total = max(0.0, penalty_total)
        total = positive_total + negative_total
        if total <= 1e-6:
            return 0.0
        return float(max(-1.0, min(1.0, (positive_total - negative_total) / total)))

    @staticmethod
    def _delayed_consequence_trajectory_recent_signal(record: Mapping[str, Any]) -> float:
        return float(max(-1.0, min(1.0, float(record.get("trajectory_recent_delta_ema", 0.0) or 0.0))))

    @classmethod
    def _delayed_consequence_trajectory_state(cls, record: Mapping[str, Any]) -> str:
        credit_total, penalty_total, forgiveness_total, net_score = cls._delayed_consequence_trajectory_totals(record)
        balance = cls._delayed_consequence_trajectory_balance(record)
        recent_signal = cls._delayed_consequence_trajectory_recent_signal(record)
        unresolved_penalty_balance = max(0.0, min(1.0, float(record.get("unresolved_penalty_balance", 0.0) or 0.0)))
        last_event_type = " ".join(str(record.get("last_trajectory_event_type", "")).split()).strip().lower()
        trajectory_floor = max(
            -DEFAULT_DELAYED_CONSEQUENCE_TRAJECTORY_SCORE_LIMIT,
            min(
                DEFAULT_DELAYED_CONSEQUENCE_TRAJECTORY_SCORE_LIMIT,
                float(record.get("trajectory_floor_score", net_score) or net_score),
            ),
        )
        if (credit_total + penalty_total + forgiveness_total) <= 1e-6:
            return "neutral"
        if (
            last_event_type in {"credit", "forgiveness"}
            and penalty_total > 0.0
            and unresolved_penalty_balance > 0.0
            and float(net_score) > float(trajectory_floor) + 0.05
        ):
            return "recovering"
        if (
            recent_signal >= DEFAULT_DELAYED_CONSEQUENCE_TRAJECTORY_STATE_THRESHOLD
            and balance < 0.0
            and unresolved_penalty_balance > 0.0
        ):
            return "recovering"
        if balance >= DEFAULT_DELAYED_CONSEQUENCE_TRAJECTORY_STATE_THRESHOLD:
            return "positive"
        if balance <= -DEFAULT_DELAYED_CONSEQUENCE_TRAJECTORY_STATE_THRESHOLD:
            return "negative"
        return "mixed"

    @classmethod
    def _delayed_consequence_trajectory_support_multiplier(
        cls,
        record: Mapping[str, Any],
        *,
        mode: str,
    ) -> float:
        balance = cls._delayed_consequence_trajectory_balance(record)
        recent_signal = cls._delayed_consequence_trajectory_recent_signal(record)
        normalized_mode = " ".join(str(mode).split()).strip().lower()
        if normalized_mode == "penalty":
            aligned_signal = 0.70 * max(0.0, -balance) + 0.30 * max(0.0, -recent_signal)
            opposing_signal = 0.70 * max(0.0, balance) + 0.30 * max(0.0, recent_signal)
        else:
            aligned_signal = 0.70 * max(0.0, balance) + 0.30 * max(0.0, recent_signal)
            opposing_signal = 0.70 * max(0.0, -balance) + 0.30 * max(0.0, -recent_signal)
        factor = 1.0 + 0.18 * float(aligned_signal) - 0.10 * float(opposing_signal)
        return float(max(0.85, min(DEFAULT_DELAYED_CONSEQUENCE_TRAJECTORY_SUPPORT_MAX, factor)))

    @classmethod
    def _delayed_consequence_family_support_multiplier(
        cls,
        record: Mapping[str, Any],
        *,
        mode: str,
    ) -> float:
        aggregate_support = cls._delayed_consequence_support_multiplier(record)
        trajectory_support = cls._delayed_consequence_trajectory_support_multiplier(record, mode=mode)
        return float(
            max(
                0.80,
                min(
                    DEFAULT_DELAYED_CONSEQUENCE_AGGREGATION_SUPPORT_MAX * DEFAULT_DELAYED_CONSEQUENCE_TRAJECTORY_SUPPORT_MAX,
                    float(aggregate_support) * float(trajectory_support),
                ),
            )
        )

    @classmethod
    def _grounded_family_summary_score(cls, record: Mapping[str, Any]) -> float:
        best_grounded_fraction = max(0.0, min(1.0, float(record.get("best_grounded_fraction", 0.0) or 0.0)))
        baseline_grounded_fraction = max(
            0.0,
            min(1.0, float(record.get("baseline_grounded_fraction", 0.0) or 0.0)),
        )
        best_query_score = max(0.0, min(1.0, float(record.get("best_query_score", 0.0) or 0.0)))
        baseline_query_score = max(0.0, min(1.0, float(record.get("baseline_query_score", 0.0) or 0.0)))
        grounded_gain = max(0.0, best_grounded_fraction - baseline_grounded_fraction)
        query_gain = max(0.0, best_query_score - baseline_query_score)
        aggregate_count = max(1, int(record.get("aggregate_count", 1) or 1))
        aggregate_support = min(1.0, math.log1p(float(aggregate_count)) / math.log1p(4.0))
        trajectory_balance = cls._delayed_consequence_trajectory_balance(record)
        recent_signal = cls._delayed_consequence_trajectory_recent_signal(record)
        unresolved_penalty_balance = max(
            0.0,
            min(1.0, float(record.get("unresolved_penalty_balance", 0.0) or 0.0)),
        )
        trajectory_state = cls._delayed_consequence_trajectory_state(record)
        state_bonus = {
            "positive": 0.12,
            "recovering": 0.08,
            "mixed": 0.02,
            "negative": -0.12,
            "neutral": 0.0,
        }.get(trajectory_state, 0.0)
        split_branch = " ".join(str(record.get("split_branch", "")).split()).strip().lower()
        branch_bonus = 0.0
        if split_branch == "supportive":
            branch_bonus = 0.08
        elif split_branch == "adverse":
            branch_bonus = -0.12
        remerge_bonus = 0.08 if int(record.get("remerge_events", 0) or 0) > 0 else 0.0
        score = (
            0.26 * best_grounded_fraction
            + 0.18 * best_query_score
            + 0.16 * grounded_gain
            + 0.10 * query_gain
            + 0.10 * aggregate_support
            + 0.12 * max(0.0, trajectory_balance)
            + 0.08 * max(0.0, recent_signal)
            + float(state_bonus)
            + float(branch_bonus)
            + float(remerge_bonus)
            - 0.22 * unresolved_penalty_balance
            - 0.10 * max(0.0, -trajectory_balance)
        )
        return float(max(0.0, min(1.0, score)))

    def _update_delayed_consequence_trajectory_locked(
        self,
        record: dict[str, Any],
        *,
        event_type: str,
        event_score: float,
        timestamp: str,
        current_token: int,
    ) -> None:
        score = max(0.0, min(1.0, float(event_score)))
        if score <= 0.0:
            return
        credit_total, penalty_total, forgiveness_total, _net_score = self._delayed_consequence_trajectory_totals(record)
        normalized_event_type = " ".join(str(event_type).split()).strip().lower()
        signed_delta = score
        if normalized_event_type == "penalty":
            penalty_total += score
            signed_delta = -score
        elif normalized_event_type == "forgiveness":
            forgiveness_total += score
        else:
            credit_total += score
            normalized_event_type = "credit"
        raw_net = credit_total + forgiveness_total - penalty_total
        net_score = max(
            -DEFAULT_DELAYED_CONSEQUENCE_TRAJECTORY_SCORE_LIMIT,
            min(DEFAULT_DELAYED_CONSEQUENCE_TRAJECTORY_SCORE_LIMIT, raw_net),
        )
        previous_recent = self._delayed_consequence_trajectory_recent_signal(record)
        event_count = max(0, int(record.get("trajectory_event_count", 0) or 0)) + 1
        record["trajectory_credit_total"] = float(credit_total)
        record["trajectory_penalty_total"] = float(penalty_total)
        record["trajectory_forgiveness_total"] = float(forgiveness_total)
        record["trajectory_event_count"] = int(event_count)
        record["trajectory_net_score"] = float(net_score)
        alpha = float(DEFAULT_DELAYED_CONSEQUENCE_TRAJECTORY_RECENT_ALPHA)
        record["trajectory_recent_delta_ema"] = float(
            signed_delta if event_count <= 1 else (1.0 - alpha) * float(previous_recent) + alpha * float(signed_delta)
        )
        record["trajectory_peak_score"] = float(
            max(float(record.get("trajectory_peak_score", net_score) or net_score), float(net_score))
        )
        record["trajectory_floor_score"] = float(
            min(float(record.get("trajectory_floor_score", net_score) or net_score), float(net_score))
        )
        record["last_trajectory_event_type"] = str(normalized_event_type)
        record["last_trajectory_event_score"] = float(score)
        record["last_trajectory_event_at"] = str(timestamp)
        record["last_trajectory_event_token_count"] = int(current_token)

    @staticmethod
    def _delayed_consequence_branch_examples(
        record: Mapping[str, Any],
        *,
        field: str,
    ) -> list[str]:
        ordered: list[str] = []
        seen: set[str] = set()
        raw_values = record.get(field)
        if not isinstance(raw_values, Sequence) or isinstance(raw_values, (str, bytes)):
            return ordered
        for raw_value in list(raw_values):
            text = " ".join(str(raw_value).split()).strip()
            if not text:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            ordered.append(text)
            if len(ordered) >= DEFAULT_DELAYED_CONSEQUENCE_AGGREGATION_QUERY_EXAMPLE_LIMIT:
                break
        return ordered

    def _update_delayed_consequence_branch_partition_locked(
        self,
        record: dict[str, Any],
        *,
        event_type: str,
        query_text: str,
    ) -> None:
        normalized_query = self._normalize_action_text(query_text)
        if not normalized_query:
            return
        normalized_event_type = " ".join(str(event_type).split()).strip().lower()
        if normalized_event_type == "penalty":
            field = "adverse_query_examples"
            count_field = "adverse_occurrence_count"
        else:
            field = "supportive_query_examples"
            count_field = "supportive_occurrence_count"
        examples = self._delayed_consequence_branch_examples(record, field=field)
        lowered = {item.lower() for item in examples}
        if normalized_query.lower() not in lowered:
            examples.append(normalized_query)
            examples = examples[-DEFAULT_DELAYED_CONSEQUENCE_AGGREGATION_QUERY_EXAMPLE_LIMIT :]
            record[field] = list(examples)
        elif field not in record:
            record[field] = list(examples)
        record[count_field] = max(int(record.get(count_field, 0) or 0), len(examples))

    def _delayed_consequence_query_text_overlap_locked(self, left_text: str, right_text: str) -> float:
        left = self._normalize_action_text(left_text).lower()
        right = self._normalize_action_text(right_text).lower()
        if not left or not right:
            return 0.0
        left_terms = {
            _canonical_provider_term(term)
            for term in self._consequence_query_terms(left)
            if _canonical_provider_term(term)
        }
        right_terms = {
            _canonical_provider_term(term)
            for term in self._consequence_query_terms(right)
            if _canonical_provider_term(term)
        }
        term_overlap = 0.0
        if left_terms and right_terms:
            term_overlap = float(len(left_terms & right_terms)) / float(max(1, min(len(left_terms), len(right_terms))))
        text_overlap = self._source_text_overlap(left, right)
        return float(max(0.0, min(1.0, max(term_overlap, 0.65 * term_overlap + 0.35 * text_overlap, text_overlap))))

    def _delayed_consequence_branch_overlap_locked(self, record: Mapping[str, Any]) -> float:
        supportive_examples = self._delayed_consequence_branch_examples(record, field="supportive_query_examples")
        adverse_examples = self._delayed_consequence_branch_examples(record, field="adverse_query_examples")
        if not supportive_examples or not adverse_examples:
            return 1.0
        return float(
            max(
                self._delayed_consequence_query_text_overlap_locked(left_text, right_text)
                for left_text in supportive_examples
                for right_text in adverse_examples
            )
        )

    def _build_delayed_consequence_split_child_locked(
        self,
        parent: Mapping[str, Any],
        *,
        branch: str,
        split_group_id: str,
        timestamp: str,
    ) -> dict[str, Any] | None:
        normalized_branch = " ".join(str(branch).split()).strip().lower()
        if normalized_branch not in {"supportive", "adverse"}:
            return None

        def _safe_int(raw_value: Any) -> int:
            try:
                return max(0, int(raw_value))
            except (TypeError, ValueError):
                return 0

        def _safe_float(raw_value: Any) -> float:
            try:
                return max(0.0, min(1.0, float(raw_value)))
            except (TypeError, ValueError):
                return 0.0

        supportive_examples = self._delayed_consequence_branch_examples(parent, field="supportive_query_examples")
        adverse_examples = self._delayed_consequence_branch_examples(parent, field="adverse_query_examples")
        supportive_count = max(1, int(parent.get("supportive_occurrence_count", 0) or 0), len(supportive_examples))
        adverse_count = max(1, int(parent.get("adverse_occurrence_count", 0) or 0), len(adverse_examples))
        trajectory_credit_total, trajectory_penalty_total, trajectory_forgiveness_total, _trajectory_net = (
            self._delayed_consequence_trajectory_totals(parent)
        )
        current_token = int(self._trainer.token_count)
        split_generation = max(1, int(parent.get("split_generation", 0) or 0) + 1)
        split_parent_record_id = str(parent.get("record_id", "")) or str(uuid4())
        baseline_query_score = float(parent.get("baseline_query_score", 0.0) or 0.0)
        baseline_grounded_fraction = float(parent.get("baseline_grounded_fraction", 0.0) or 0.0)
        if normalized_branch == "supportive":
            query_examples = supportive_examples
            aggregate_count = int(supportive_count)
            best_query_score = max(baseline_query_score, float(parent.get("best_query_score", baseline_query_score) or baseline_query_score))
            best_grounded_fraction = max(
                baseline_grounded_fraction,
                float(parent.get("best_grounded_fraction", baseline_grounded_fraction) or baseline_grounded_fraction),
            )
            credit_events = int(parent.get("credit_events", 0) or 0)
            penalty_events = 0
            forgiveness_events = int(parent.get("forgiveness_events", 0) or 0)
            unresolved_penalty_balance = 0.0
            resolved_improvement = float(parent.get("resolved_improvement", 0.0) or 0.0)
            max_regression = 0.0
            max_contradiction_signal = 0.0
            last_credit_score = float(parent.get("last_credit_score", 0.0) or 0.0)
            last_forgiveness_score = float(parent.get("last_forgiveness_score", 0.0) or 0.0)
            last_penalty_score = 0.0
            last_penalty_reason = ""
            branch_recent = max(
                0.20,
                float(parent.get("last_forgiveness_score", 0.0) or 0.0),
                float(parent.get("last_credit_score", 0.0) or 0.0),
                max(0.0, self._delayed_consequence_trajectory_recent_signal(parent)),
            )
            last_event_type = (
                "forgiveness"
                if float(parent.get("last_forgiveness_score", 0.0) or 0.0) > 0.0
                else "credit"
            )
            last_event_score = max(
                float(parent.get("last_forgiveness_score", 0.0) or 0.0),
                float(parent.get("last_credit_score", 0.0) or 0.0),
            )
            last_event_token = max(
                int(parent.get("last_forgiveness_token_count", 0) or 0),
                int(parent.get("last_credit_token_count", 0) or 0),
            )
            branch_credit_total = float(trajectory_credit_total)
            branch_penalty_total = 0.0
            branch_forgiveness_total = float(trajectory_forgiveness_total)
            branch_event_count = max(1, credit_events + forgiveness_events)
            branch_net_score = max(
                -DEFAULT_DELAYED_CONSEQUENCE_TRAJECTORY_SCORE_LIMIT,
                min(
                    DEFAULT_DELAYED_CONSEQUENCE_TRAJECTORY_SCORE_LIMIT,
                    branch_credit_total + branch_forgiveness_total - branch_penalty_total,
                ),
            )
            branch_peak_score = max(0.0, branch_net_score)
            branch_floor_score = min(0.0, branch_net_score)
            supportive_branch_examples = list(query_examples)
            adverse_branch_examples: list[str] = []
            supportive_occurrence_count = int(aggregate_count)
            adverse_occurrence_count = 0
            cumulative_cooling_delta = 0.0
            cooling_events = 0
        else:
            query_examples = adverse_examples
            aggregate_count = int(adverse_count)
            best_query_score = float(parent.get("baseline_query_score", 0.0) or 0.0)
            best_grounded_fraction = float(parent.get("baseline_grounded_fraction", 0.0) or 0.0)
            credit_events = 0
            penalty_events = int(parent.get("penalty_events", 0) or 0)
            forgiveness_events = 0
            unresolved_penalty_balance = float(parent.get("unresolved_penalty_balance", 0.0) or 0.0)
            resolved_improvement = 0.0
            max_regression = float(parent.get("max_regression", 0.0) or 0.0)
            max_contradiction_signal = float(parent.get("max_contradiction_signal", 0.0) or 0.0)
            last_credit_score = 0.0
            last_forgiveness_score = 0.0
            last_penalty_score = float(parent.get("last_penalty_score", 0.0) or 0.0)
            last_penalty_reason = str(parent.get("last_penalty_reason", "") or "")
            branch_recent = -max(
                0.20,
                float(parent.get("last_penalty_score", 0.0) or 0.0),
                abs(self._delayed_consequence_trajectory_recent_signal(parent)),
            )
            last_event_type = "penalty"
            last_event_score = float(parent.get("last_penalty_score", 0.0) or 0.0)
            last_event_token = int(parent.get("last_penalty_token_count", 0) or 0)
            branch_credit_total = 0.0
            branch_penalty_total = float(trajectory_penalty_total)
            branch_forgiveness_total = 0.0
            branch_event_count = max(1, penalty_events)
            branch_net_score = max(
                -DEFAULT_DELAYED_CONSEQUENCE_TRAJECTORY_SCORE_LIMIT,
                min(DEFAULT_DELAYED_CONSEQUENCE_TRAJECTORY_SCORE_LIMIT, -branch_penalty_total),
            )
            branch_peak_score = max(0.0, branch_net_score)
            branch_floor_score = min(0.0, branch_net_score)
            supportive_branch_examples = []
            adverse_branch_examples = list(query_examples)
            supportive_occurrence_count = 0
            adverse_occurrence_count = int(aggregate_count)
            cumulative_cooling_delta = float(parent.get("cumulative_cooling_delta", 0.0) or 0.0)
            cooling_events = int(parent.get("cooling_events", 0) or 0)
        if not query_examples:
            return None
        child = self._normalize_delayed_consequence_record(
            {
                "record_id": str(uuid4()),
                "created_at": str(parent.get("created_at") or timestamp),
                "created_token_count": int(parent.get("created_token_count", current_token) or current_token),
                "origin": str(parent.get("origin", "response_selected_evidence") or "response_selected_evidence"),
                "query_text": query_examples[0],
                "query_examples": list(query_examples),
                "baseline_query_score": float(baseline_query_score),
                "best_query_score": float(best_query_score),
                "baseline_grounded_fraction": float(baseline_grounded_fraction),
                "best_grounded_fraction": float(best_grounded_fraction),
                "outcome_score": float(parent.get("outcome_score", 0.0) or 0.0),
                "source_weights": deepcopy(dict(parent.get("source_weights") or {})),
                "provider_weights": deepcopy(dict(parent.get("provider_weights") or {})),
                "credit_events": int(credit_events),
                "penalty_events": int(penalty_events),
                "forgiveness_events": int(forgiveness_events),
                "cooling_events": int(cooling_events),
                "aggregate_count": int(aggregate_count),
                "aggregation_events": max(0, int(aggregate_count) - 1),
                "supportive_query_examples": list(supportive_branch_examples),
                "adverse_query_examples": list(adverse_branch_examples),
                "supportive_occurrence_count": int(supportive_occurrence_count),
                "adverse_occurrence_count": int(adverse_occurrence_count),
                "trajectory_credit_total": float(branch_credit_total),
                "trajectory_penalty_total": float(branch_penalty_total),
                "trajectory_forgiveness_total": float(branch_forgiveness_total),
                "trajectory_event_count": int(branch_event_count),
                "trajectory_net_score": float(branch_net_score),
                "trajectory_recent_delta_ema": float(max(-1.0, min(1.0, branch_recent))),
                "trajectory_peak_score": float(branch_peak_score),
                "trajectory_floor_score": float(branch_floor_score),
                "unresolved_penalty_balance": float(_safe_float(unresolved_penalty_balance)),
                "resolved_improvement": float(_safe_float(resolved_improvement)),
                "max_regression": float(_safe_float(max_regression)),
                "max_contradiction_signal": float(_safe_float(max_contradiction_signal)),
                "cumulative_cooling_delta": float(_safe_float(cumulative_cooling_delta)),
                "last_match_score": float(_safe_float(parent.get("last_match_score", 0.0))),
                "last_credit_score": float(_safe_float(last_credit_score)),
                "last_penalty_score": float(_safe_float(last_penalty_score)),
                "last_forgiveness_score": float(_safe_float(last_forgiveness_score)),
                "last_penalty_reason": str(last_penalty_reason),
                "last_activity_token_count": int(parent.get("last_activity_token_count", current_token) or current_token),
                "last_evaluated_token_count": int(parent.get("last_evaluated_token_count", current_token) or current_token),
                "last_cooling_token_count": int(parent.get("last_cooling_token_count", current_token) or current_token),
                "last_credit_token_count": int(parent.get("last_credit_token_count", 0) or 0) if normalized_branch == "supportive" else 0,
                "last_penalty_token_count": int(parent.get("last_penalty_token_count", 0) or 0) if normalized_branch == "adverse" else 0,
                "last_forgiveness_token_count": int(parent.get("last_forgiveness_token_count", 0) or 0) if normalized_branch == "supportive" else 0,
                "last_trajectory_event_type": str(last_event_type),
                "last_trajectory_event_score": float(_safe_float(last_event_score)),
                "last_trajectory_event_at": str(parent.get("last_evaluated_at", timestamp) or timestamp),
                "last_trajectory_event_token_count": int(last_event_token),
                "split_generation": int(split_generation),
                "split_parent_record_id": str(split_parent_record_id),
                "split_group_id": str(split_group_id),
                "split_branch": str(normalized_branch),
                "last_split_at": str(timestamp),
                "last_aggregated_at": str(parent.get("last_aggregated_at", "") or ""),
                "last_cooled_at": str(parent.get("last_cooled_at", "") or ""),
                "last_evaluated_at": str(parent.get("last_evaluated_at", timestamp) or timestamp),
                "last_evaluated_query_text": str(parent.get("last_evaluated_query_text", query_examples[0]) or query_examples[0]),
            }
        )
        return cast(dict[str, Any] | None, child)

    def _split_divergent_delayed_consequence_families_locked(self) -> dict[str, Any]:
        records = list(self._delayed_consequence_records)
        if not records:
            return {
                "split_records": 0,
                "max_branch_overlap": 1.0,
                "record_ids": [],
            }
        updated_records: list[dict[str, Any]] = []
        split_records = 0
        max_branch_overlap = 1.0
        split_record_ids: list[str] = []
        timestamp = datetime.now(timezone.utc).isoformat()
        for record in records:
            if int(record.get("aggregate_count", 1) or 1) < 2:
                updated_records.append(record)
                continue
            if str(record.get("split_branch", "") or ""):
                updated_records.append(record)
                continue
            supportive_examples = self._delayed_consequence_branch_examples(record, field="supportive_query_examples")
            adverse_examples = self._delayed_consequence_branch_examples(record, field="adverse_query_examples")
            supportive_occurrence_count = max(int(record.get("supportive_occurrence_count", 0) or 0), len(supportive_examples))
            adverse_occurrence_count = max(int(record.get("adverse_occurrence_count", 0) or 0), len(adverse_examples))
            if supportive_occurrence_count < DEFAULT_DELAYED_CONSEQUENCE_SPLIT_MIN_BRANCH_OCCURRENCES:
                updated_records.append(record)
                continue
            if adverse_occurrence_count < DEFAULT_DELAYED_CONSEQUENCE_SPLIT_MIN_BRANCH_OCCURRENCES:
                updated_records.append(record)
                continue
            trajectory_credit_total, trajectory_penalty_total, trajectory_forgiveness_total, _trajectory_net = (
                self._delayed_consequence_trajectory_totals(record)
            )
            if (trajectory_credit_total + trajectory_forgiveness_total) <= 0.0 or trajectory_penalty_total <= 0.0:
                updated_records.append(record)
                continue
            branch_overlap = self._delayed_consequence_branch_overlap_locked(record)
            max_branch_overlap = min(max_branch_overlap, float(branch_overlap))
            if branch_overlap > DEFAULT_DELAYED_CONSEQUENCE_SPLIT_MAX_BRANCH_OVERLAP:
                updated_records.append(record)
                continue
            split_group_id = str(record.get("split_group_id", "") or record.get("record_id", "") or uuid4())
            supportive_child = self._build_delayed_consequence_split_child_locked(
                record,
                branch="supportive",
                split_group_id=split_group_id,
                timestamp=timestamp,
            )
            adverse_child = self._build_delayed_consequence_split_child_locked(
                record,
                branch="adverse",
                split_group_id=split_group_id,
                timestamp=timestamp,
            )
            if supportive_child is None or adverse_child is None:
                updated_records.append(record)
                continue
            updated_records.extend([supportive_child, adverse_child])
            split_records += 1
            split_record_ids.append(str(record.get("record_id", "")))
        if split_records <= 0:
            return {
                "split_records": 0,
                "max_branch_overlap": 1.0 if max_branch_overlap >= 1.0 else float(max_branch_overlap),
                "record_ids": [],
            }
        self._delayed_consequence_records = deque(updated_records[:DEFAULT_DELAYED_CONSEQUENCE_RECORDS], maxlen=DEFAULT_DELAYED_CONSEQUENCE_RECORDS)
        self._delayed_consequence_split_total += int(split_records)
        self._record_brain_event_locked(
            {
                "type": "delayed_consequence_state_split",
                "timestamp": timestamp,
                "split_records": int(split_records),
                "record_ids": split_record_ids[:8],
                "max_branch_overlap": float(max_branch_overlap),
            }
        )
        self._runtime_state.mark_mutated()
        return {
            "split_records": int(split_records),
            "max_branch_overlap": float(max_branch_overlap),
            "record_ids": split_record_ids[:8],
        }

    def _should_remerge_delayed_consequence_split_group_locked(
        self,
        group_records: Sequence[Mapping[str, Any]],
    ) -> bool:
        branches = {
            self._normalize_action_text(record.get("split_branch", "")).lower()
            for record in list(group_records)
            if self._normalize_action_text(record.get("split_branch", "")).lower() in {"supportive", "adverse"}
        }
        if branches != {"supportive", "adverse"}:
            return False
        for record in list(group_records):
            branch = self._normalize_action_text(record.get("split_branch", "")).lower()
            if branch != "adverse":
                continue
            supportive_cross = max(
                int(record.get("supportive_occurrence_count", 0) or 0),
                len(self._delayed_consequence_branch_examples(record, field="supportive_query_examples")),
            )
            if supportive_cross < DEFAULT_DELAYED_CONSEQUENCE_REMERGE_MIN_CROSS_OCCURRENCES:
                continue
            recent_signal = self._delayed_consequence_trajectory_recent_signal(record)
            trajectory_state = self._delayed_consequence_trajectory_state(record)
            net_score = float(record.get("trajectory_net_score", 0.0) or 0.0)
            floor_score = float(record.get("trajectory_floor_score", net_score) or net_score)
            if recent_signal > 0.0 or trajectory_state in {"recovering", "positive", "mixed"} or net_score > floor_score + 0.05:
                return True
        return False

    def _build_remerged_delayed_consequence_family_locked(
        self,
        group_records: Sequence[Mapping[str, Any]],
        *,
        split_group_id: str,
        timestamp: str,
    ) -> dict[str, Any] | None:
        ordered_records = [cast(dict[str, Any], deepcopy(record)) for record in list(group_records) if isinstance(record, Mapping)]
        if not ordered_records:
            return None
        merged = cast(dict[str, Any], ordered_records[0])
        for record in ordered_records[1:]:
            merged = self._merge_delayed_consequence_records_locked(merged, record)
        merged["supportive_query_examples"] = []
        merged["adverse_query_examples"] = []
        merged["supportive_occurrence_count"] = 0
        merged["adverse_occurrence_count"] = 0
        merged["split_group_id"] = str(split_group_id)
        merged["split_branch"] = ""
        merged["split_generation"] = max(
            int(record.get("split_generation", 0) or 0)
            for record in ordered_records
        )
        merged["split_parent_record_id"] = self._normalize_action_text(merged.get("split_parent_record_id", "")) or str(
            ordered_records[0].get("split_parent_record_id", "") or ordered_records[0].get("record_id", "")
        )
        merged["remerge_events"] = (
            sum(int(record.get("remerge_events", 0) or 0) for record in ordered_records) + 1
        )
        merged["last_remerged_at"] = str(timestamp)
        normalized = self._normalize_delayed_consequence_record(merged)
        return cast(dict[str, Any] | None, normalized)

    def _remerge_converged_delayed_consequence_families_locked(self) -> dict[str, Any]:
        records = list(self._delayed_consequence_records)
        if len(records) < 2:
            return {
                "remerged_records": 0,
                "record_ids": [],
            }
        groups: dict[str, list[dict[str, Any]]] = {}
        for record in records:
            split_group_id = self._normalize_action_text(record.get("split_group_id", ""))
            split_branch = self._normalize_action_text(record.get("split_branch", "")).lower()
            if not split_group_id or split_branch not in {"supportive", "adverse"}:
                continue
            groups.setdefault(split_group_id, []).append(record)
        if not groups:
            return {
                "remerged_records": 0,
                "record_ids": [],
            }
        remerge_map: dict[str, dict[str, Any]] = {}
        remerged_record_ids: list[str] = []
        timestamp = datetime.now(timezone.utc).isoformat()
        for split_group_id, group_records in groups.items():
            if not self._should_remerge_delayed_consequence_split_group_locked(group_records):
                continue
            merged = self._build_remerged_delayed_consequence_family_locked(
                group_records,
                split_group_id=split_group_id,
                timestamp=timestamp,
            )
            if merged is None:
                continue
            remerge_map[split_group_id] = merged
            remerged_record_ids.extend(str(record.get("record_id", "")) for record in group_records)
        if not remerge_map:
            return {
                "remerged_records": 0,
                "record_ids": [],
            }
        updated_records: list[dict[str, Any]] = []
        inserted_groups: set[str] = set()
        for record in records:
            split_group_id = self._normalize_action_text(record.get("split_group_id", ""))
            split_branch = self._normalize_action_text(record.get("split_branch", "")).lower()
            if split_group_id in remerge_map and split_branch in {"supportive", "adverse"}:
                if split_group_id in inserted_groups:
                    continue
                updated_records.append(remerge_map[split_group_id])
                inserted_groups.add(split_group_id)
                continue
            updated_records.append(record)
        self._delayed_consequence_records = deque(updated_records[:DEFAULT_DELAYED_CONSEQUENCE_RECORDS], maxlen=DEFAULT_DELAYED_CONSEQUENCE_RECORDS)
        self._delayed_consequence_remerged_total += int(len(remerge_map))
        self._record_brain_event_locked(
            {
                "type": "delayed_consequence_state_remerged",
                "timestamp": timestamp,
                "remerged_records": int(len(remerge_map)),
                "record_ids": remerged_record_ids[:8],
            }
        )
        self._runtime_state.mark_mutated()
        return {
            "remerged_records": int(len(remerge_map)),
            "record_ids": remerged_record_ids[:8],
        }

    @staticmethod
    def _delayed_consequence_weight_overlap(
        left: Mapping[str, Any],
        right: Mapping[str, Any],
    ) -> float:
        def _normalized(value: Mapping[str, Any]) -> dict[str, float]:
            result: dict[str, float] = {}
            for raw_key, raw_weight in value.items():
                key = " ".join(str(raw_key).split()).strip().lower()
                if not key:
                    continue
                try:
                    weight = max(0.0, min(1.0, float(raw_weight)))
                except (TypeError, ValueError):
                    continue
                if weight <= 0.0:
                    continue
                result[key] = weight
            return result

        normalized_left = _normalized(left)
        normalized_right = _normalized(right)
        if not normalized_left or not normalized_right:
            return 0.0
        all_keys = set(normalized_left) | set(normalized_right)
        shared_keys = set(normalized_left) & set(normalized_right)
        if not all_keys or not shared_keys:
            return 0.0
        weighted_overlap = sum(min(normalized_left.get(key, 0.0), normalized_right.get(key, 0.0)) for key in all_keys) / max(
            1e-6,
            sum(max(normalized_left.get(key, 0.0), normalized_right.get(key, 0.0)) for key in all_keys),
        )
        key_overlap = float(len(shared_keys)) / float(max(1, len(all_keys)))
        return float(max(0.0, min(1.0, max(weighted_overlap, key_overlap))))

    def _delayed_consequence_provenance_overlap_locked(
        self,
        left: Mapping[str, Any],
        right: Mapping[str, Any],
    ) -> float:
        overlap_scores: list[float] = []
        left_sources = cast(Mapping[str, Any], left.get("source_weights") or {})
        right_sources = cast(Mapping[str, Any], right.get("source_weights") or {})
        if left_sources and right_sources:
            overlap_scores.append(self._delayed_consequence_weight_overlap(left_sources, right_sources))
        left_providers = cast(Mapping[str, Any], left.get("provider_weights") or {})
        right_providers = cast(Mapping[str, Any], right.get("provider_weights") or {})
        if left_providers and right_providers:
            overlap_scores.append(self._delayed_consequence_weight_overlap(left_providers, right_providers))
        if not overlap_scores:
            return 0.0
        if len(overlap_scores) == 1:
            return float(overlap_scores[0])
        return float(min(overlap_scores))

    def _delayed_consequence_aggregation_score_locked(
        self,
        existing: Mapping[str, Any],
        candidate: Mapping[str, Any],
    ) -> float:
        existing_split_group = self._normalize_action_text(existing.get("split_group_id", ""))
        candidate_split_group = self._normalize_action_text(candidate.get("split_group_id", ""))
        existing_split_branch = self._normalize_action_text(existing.get("split_branch", "")).lower()
        candidate_split_branch = self._normalize_action_text(candidate.get("split_branch", "")).lower()
        if (
            existing_split_group
            and candidate_split_group
            and existing_split_group == candidate_split_group
            and existing_split_branch
            and candidate_split_branch
            and existing_split_branch != candidate_split_branch
        ):
            return 0.0
        provenance_overlap = self._delayed_consequence_provenance_overlap_locked(existing, candidate)
        if provenance_overlap < DEFAULT_DELAYED_CONSEQUENCE_AGGREGATION_PROVENANCE_THRESHOLD:
            return 0.0
        query_score = self._delayed_consequence_match_score_locked(
            existing,
            {
                "query_text": str(candidate.get("query_text", "")),
                "query_terms": list(candidate.get("query_terms") or []),
            },
        )
        if query_score < DEFAULT_DELAYED_CONSEQUENCE_AGGREGATION_MATCH_THRESHOLD:
            return 0.0
        return float(max(0.0, min(1.0, 0.55 * float(query_score) + 0.45 * float(provenance_overlap))))

    def _merge_delayed_consequence_records_locked(
        self,
        primary: Mapping[str, Any],
        secondary: Mapping[str, Any],
    ) -> dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()

        def _safe_int(raw_value: Any) -> int:
            try:
                return max(0, int(raw_value))
            except (TypeError, ValueError):
                return 0

        def _safe_float(raw_value: Any) -> float:
            try:
                return max(0.0, min(1.0, float(raw_value)))
            except (TypeError, ValueError):
                return 0.0

        def _safe_total(raw_value: Any) -> float:
            try:
                return max(0.0, float(raw_value))
            except (TypeError, ValueError):
                return 0.0

        def _safe_signed(raw_value: Any, *, limit: float = DEFAULT_DELAYED_CONSEQUENCE_TRAJECTORY_SCORE_LIMIT) -> float:
            try:
                return max(-float(limit), min(float(limit), float(raw_value)))
            except (TypeError, ValueError):
                return 0.0

        def _merged_weights(*values: Mapping[str, Any]) -> dict[str, float]:
            merged: dict[str, float] = {}
            for raw_value in values:
                for raw_name, raw_weight in dict(raw_value).items():
                    name = " ".join(str(raw_name).split()).strip()
                    if not name:
                        continue
                    merged[name] = max(float(merged.get(name, 0.0)), _safe_float(raw_weight))
            return {name: weight for name, weight in merged.items() if weight > 0.0}

        merged_examples: list[str] = []
        seen_examples: set[str] = set()
        for record in (primary, secondary):
            for example in self._delayed_consequence_query_examples(record):
                key = example.lower()
                if key in seen_examples:
                    continue
                seen_examples.add(key)
                merged_examples.append(example)
                if len(merged_examples) >= DEFAULT_DELAYED_CONSEQUENCE_AGGREGATION_QUERY_EXAMPLE_LIMIT:
                    break
            if len(merged_examples) >= DEFAULT_DELAYED_CONSEQUENCE_AGGREGATION_QUERY_EXAMPLE_LIMIT:
                break

        term_counter: Counter[str] = Counter()
        first_seen: dict[str, int] = {}
        for record in (primary, secondary):
            for query_text in self._delayed_consequence_query_examples(record):
                for raw_term in self._consequence_query_terms(query_text):
                    term = _canonical_provider_term(raw_term)
                    if not term:
                        continue
                    term_counter[term] += 1
                    first_seen.setdefault(term, len(first_seen))
            for raw_term in list(record.get("query_terms") or []):
                term = _canonical_provider_term(raw_term)
                if not term:
                    continue
                term_counter[term] += 1
                first_seen.setdefault(term, len(first_seen))
        merged_terms = [
            term
            for term, _count in sorted(
                term_counter.items(),
                key=lambda item: (-int(item[1]), int(first_seen.get(item[0], 0)), item[0]),
            )
        ][:DEFAULT_DELAYED_CONSEQUENCE_AGGREGATION_TERM_LIMIT]

        merged_supportive_examples: list[str] = []
        merged_adverse_examples: list[str] = []
        for field, target in (
            ("supportive_query_examples", merged_supportive_examples),
            ("adverse_query_examples", merged_adverse_examples),
        ):
            seen_branch: set[str] = set()
            for record in (primary, secondary):
                for example in self._delayed_consequence_branch_examples(record, field=field):
                    key = example.lower()
                    if key in seen_branch:
                        continue
                    seen_branch.add(key)
                    target.append(example)
                    if len(target) >= DEFAULT_DELAYED_CONSEQUENCE_AGGREGATION_QUERY_EXAMPLE_LIMIT:
                        break
                if len(target) >= DEFAULT_DELAYED_CONSEQUENCE_AGGREGATION_QUERY_EXAMPLE_LIMIT:
                    break
        merged_supportive_occurrence_count = max(
            len(merged_supportive_examples),
            _safe_int(primary.get("supportive_occurrence_count", 0)),
            _safe_int(secondary.get("supportive_occurrence_count", 0)),
        )
        merged_adverse_occurrence_count = max(
            len(merged_adverse_examples),
            _safe_int(primary.get("adverse_occurrence_count", 0)),
            _safe_int(secondary.get("adverse_occurrence_count", 0)),
        )
        merged_split_generation = max(
            _safe_int(primary.get("split_generation", 0)),
            _safe_int(secondary.get("split_generation", 0)),
        )
        merged_split_parent_record_id = self._normalize_action_text(primary.get("split_parent_record_id", "")) or self._normalize_action_text(
            secondary.get("split_parent_record_id", "")
        )
        merged_split_group_id = self._normalize_action_text(primary.get("split_group_id", "")) or self._normalize_action_text(
            secondary.get("split_group_id", "")
        )
        primary_split_branch = self._normalize_action_text(primary.get("split_branch", ""))
        secondary_split_branch = self._normalize_action_text(secondary.get("split_branch", ""))
        merged_split_branch = (
            ""
            if primary_split_branch and secondary_split_branch and primary_split_branch != secondary_split_branch
            else primary_split_branch or secondary_split_branch
        )
        merged_last_split_at = self._normalize_action_text(primary.get("last_split_at", "")) or self._normalize_action_text(
            secondary.get("last_split_at", "")
        )
        merged_remerge_events = _safe_int(primary.get("remerge_events", 0)) + _safe_int(secondary.get("remerge_events", 0))
        merged_last_remerged_at = self._normalize_action_text(primary.get("last_remerged_at", "")) or self._normalize_action_text(
            secondary.get("last_remerged_at", "")
        )

        primary_created_token = _safe_int(primary.get("created_token_count", 0))
        secondary_created_token = _safe_int(secondary.get("created_token_count", 0))
        if primary_created_token <= 0 and secondary_created_token > 0:
            family_created_token = secondary_created_token
            family_created_at = str(secondary.get("created_at") or now)
        elif secondary_created_token <= 0 and primary_created_token > 0:
            family_created_token = primary_created_token
            family_created_at = str(primary.get("created_at") or now)
        elif 0 < secondary_created_token < primary_created_token:
            family_created_token = secondary_created_token
            family_created_at = str(secondary.get("created_at") or now)
        else:
            family_created_token = max(0, primary_created_token or secondary_created_token)
            family_created_at = str(primary.get("created_at") or secondary.get("created_at") or now)

        primary_credit_total, primary_penalty_total, primary_forgiveness_total, _primary_net = self._delayed_consequence_trajectory_totals(primary)
        secondary_credit_total, secondary_penalty_total, secondary_forgiveness_total, _secondary_net = self._delayed_consequence_trajectory_totals(secondary)
        merged_credit_total = float(primary_credit_total + secondary_credit_total)
        merged_penalty_total = float(primary_penalty_total + secondary_penalty_total)
        merged_forgiveness_total = float(primary_forgiveness_total + secondary_forgiveness_total)
        merged_net_score = float(
            max(
                -DEFAULT_DELAYED_CONSEQUENCE_TRAJECTORY_SCORE_LIMIT,
                min(
                    DEFAULT_DELAYED_CONSEQUENCE_TRAJECTORY_SCORE_LIMIT,
                    merged_credit_total + merged_forgiveness_total - merged_penalty_total,
                ),
            )
        )
        primary_trajectory_events = max(0, int(primary.get("trajectory_event_count", 0) or 0))
        secondary_trajectory_events = max(0, int(secondary.get("trajectory_event_count", 0) or 0))
        merged_trajectory_events = int(primary_trajectory_events + secondary_trajectory_events)
        if merged_trajectory_events > 0:
            merged_recent_delta = (
                float(primary_trajectory_events)
                * _safe_signed(primary.get("trajectory_recent_delta_ema", 0.0), limit=1.0)
                + float(secondary_trajectory_events)
                * _safe_signed(secondary.get("trajectory_recent_delta_ema", 0.0), limit=1.0)
            ) / float(max(1, merged_trajectory_events))
        else:
            merged_recent_delta = 0.0
        primary_peak = _safe_signed(primary.get("trajectory_peak_score", primary.get("trajectory_net_score", 0.0)))
        secondary_peak = _safe_signed(secondary.get("trajectory_peak_score", secondary.get("trajectory_net_score", 0.0)))
        primary_floor = _safe_signed(primary.get("trajectory_floor_score", primary.get("trajectory_net_score", 0.0)))
        secondary_floor = _safe_signed(secondary.get("trajectory_floor_score", secondary.get("trajectory_net_score", 0.0)))
        primary_event_token = max(
            _safe_int(primary.get("last_trajectory_event_token_count", 0)),
            _safe_int(primary.get("last_activity_token_count", 0)),
        )
        secondary_event_token = max(
            _safe_int(secondary.get("last_trajectory_event_token_count", 0)),
            _safe_int(secondary.get("last_activity_token_count", 0)),
        )
        latest_trajectory_record = primary if primary_event_token >= secondary_event_token else secondary

        normalized = self._normalize_delayed_consequence_record(
            {
                "record_id": str(primary.get("record_id", "")) or str(uuid4()),
                "created_at": str(family_created_at),
                "created_token_count": int(family_created_token),
                "origin": self._normalize_action_text(primary.get("origin", "response_selected_evidence"))
                or "response_selected_evidence",
                "query_text": merged_examples[0] if merged_examples else str(primary.get("query_text", "")),
                "query_examples": merged_examples,
                "query_terms": merged_terms,
                "baseline_query_score": max(
                    _safe_float(primary.get("baseline_query_score", 0.0)),
                    _safe_float(secondary.get("baseline_query_score", 0.0)),
                ),
                "best_query_score": max(
                    _safe_float(primary.get("best_query_score", 0.0)),
                    _safe_float(secondary.get("best_query_score", 0.0)),
                ),
                "baseline_grounded_fraction": max(
                    _safe_float(primary.get("baseline_grounded_fraction", 0.0)),
                    _safe_float(secondary.get("baseline_grounded_fraction", 0.0)),
                ),
                "best_grounded_fraction": max(
                    _safe_float(primary.get("best_grounded_fraction", 0.0)),
                    _safe_float(secondary.get("best_grounded_fraction", 0.0)),
                ),
                "outcome_score": max(
                    _safe_float(primary.get("outcome_score", 0.0)),
                    _safe_float(secondary.get("outcome_score", 0.0)),
                ),
                "source_weights": _merged_weights(
                    cast(Mapping[str, Any], primary.get("source_weights") or {}),
                    cast(Mapping[str, Any], secondary.get("source_weights") or {}),
                ),
                "provider_weights": _merged_weights(
                    cast(Mapping[str, Any], primary.get("provider_weights") or {}),
                    cast(Mapping[str, Any], secondary.get("provider_weights") or {}),
                ),
                "credit_events": _safe_int(primary.get("credit_events", 0)) + _safe_int(secondary.get("credit_events", 0)),
                "penalty_events": _safe_int(primary.get("penalty_events", 0)) + _safe_int(secondary.get("penalty_events", 0)),
                "forgiveness_events": _safe_int(primary.get("forgiveness_events", 0)) + _safe_int(secondary.get("forgiveness_events", 0)),
                "cooling_events": _safe_int(primary.get("cooling_events", 0)) + _safe_int(secondary.get("cooling_events", 0)),
                "aggregate_count": max(1, _safe_int(primary.get("aggregate_count", 1)))
                + max(1, _safe_int(secondary.get("aggregate_count", 1))),
                "aggregation_events": _safe_int(primary.get("aggregation_events", 0))
                + _safe_int(secondary.get("aggregation_events", 0))
                + 1,
                "supportive_query_examples": list(merged_supportive_examples),
                "adverse_query_examples": list(merged_adverse_examples),
                "supportive_occurrence_count": int(merged_supportive_occurrence_count),
                "adverse_occurrence_count": int(merged_adverse_occurrence_count),
                "trajectory_credit_total": float(merged_credit_total),
                "trajectory_penalty_total": float(merged_penalty_total),
                "trajectory_forgiveness_total": float(merged_forgiveness_total),
                "trajectory_event_count": int(merged_trajectory_events),
                "trajectory_net_score": float(merged_net_score),
                "trajectory_recent_delta_ema": float(_safe_signed(merged_recent_delta, limit=1.0)),
                "trajectory_peak_score": float(max(primary_peak, secondary_peak, merged_net_score)),
                "trajectory_floor_score": float(min(primary_floor, secondary_floor, merged_net_score)),
                "unresolved_penalty_balance": min(
                    1.0,
                    _safe_float(primary.get("unresolved_penalty_balance", 0.0))
                    + _safe_float(secondary.get("unresolved_penalty_balance", 0.0)),
                ),
                "resolved_improvement": max(
                    _safe_float(primary.get("resolved_improvement", 0.0)),
                    _safe_float(secondary.get("resolved_improvement", 0.0)),
                ),
                "max_regression": max(
                    _safe_float(primary.get("max_regression", 0.0)),
                    _safe_float(secondary.get("max_regression", 0.0)),
                ),
                "max_contradiction_signal": max(
                    _safe_float(primary.get("max_contradiction_signal", 0.0)),
                    _safe_float(secondary.get("max_contradiction_signal", 0.0)),
                ),
                "cumulative_cooling_delta": min(
                    1.0,
                    _safe_float(primary.get("cumulative_cooling_delta", 0.0))
                    + _safe_float(secondary.get("cumulative_cooling_delta", 0.0)),
                ),
                "last_match_score": max(
                    _safe_float(primary.get("last_match_score", 0.0)),
                    _safe_float(secondary.get("last_match_score", 0.0)),
                ),
                "last_credit_score": max(
                    _safe_float(primary.get("last_credit_score", 0.0)),
                    _safe_float(secondary.get("last_credit_score", 0.0)),
                ),
                "last_penalty_score": max(
                    _safe_float(primary.get("last_penalty_score", 0.0)),
                    _safe_float(secondary.get("last_penalty_score", 0.0)),
                ),
                "last_forgiveness_score": max(
                    _safe_float(primary.get("last_forgiveness_score", 0.0)),
                    _safe_float(secondary.get("last_forgiveness_score", 0.0)),
                ),
                "last_penalty_reason": self._normalize_action_text(primary.get("last_penalty_reason", ""))
                or self._normalize_action_text(secondary.get("last_penalty_reason", "")),
                "last_activity_token_count": max(
                    _safe_int(primary.get("last_activity_token_count", 0)),
                    _safe_int(secondary.get("last_activity_token_count", 0)),
                ),
                "last_evaluated_token_count": max(
                    _safe_int(primary.get("last_evaluated_token_count", 0)),
                    _safe_int(secondary.get("last_evaluated_token_count", 0)),
                ),
                "last_cooling_token_count": max(
                    _safe_int(primary.get("last_cooling_token_count", 0)),
                    _safe_int(secondary.get("last_cooling_token_count", 0)),
                ),
                "last_credit_token_count": max(
                    _safe_int(primary.get("last_credit_token_count", 0)),
                    _safe_int(secondary.get("last_credit_token_count", 0)),
                ),
                "last_penalty_token_count": max(
                    _safe_int(primary.get("last_penalty_token_count", 0)),
                    _safe_int(secondary.get("last_penalty_token_count", 0)),
                ),
                "last_forgiveness_token_count": max(
                    _safe_int(primary.get("last_forgiveness_token_count", 0)),
                    _safe_int(secondary.get("last_forgiveness_token_count", 0)),
                ),
                "last_trajectory_event_type": self._normalize_action_text(
                    latest_trajectory_record.get("last_trajectory_event_type", "")
                ),
                "last_trajectory_event_score": float(
                    _safe_total(latest_trajectory_record.get("last_trajectory_event_score", 0.0))
                ),
                "last_trajectory_event_at": self._normalize_action_text(
                    latest_trajectory_record.get("last_trajectory_event_at", "")
                ),
                "last_trajectory_event_token_count": int(
                    _safe_int(latest_trajectory_record.get("last_trajectory_event_token_count", 0))
                ),
                "split_generation": int(merged_split_generation),
                "split_parent_record_id": str(merged_split_parent_record_id),
                "split_group_id": str(merged_split_group_id),
                "split_branch": str(merged_split_branch),
                "remerge_events": int(merged_remerge_events),
                "last_split_at": str(merged_last_split_at),
                "last_remerged_at": str(merged_last_remerged_at),
                "last_cooled_at": self._normalize_action_text(primary.get("last_cooled_at", ""))
                or self._normalize_action_text(secondary.get("last_cooled_at", "")),
                "last_evaluated_at": self._normalize_action_text(primary.get("last_evaluated_at", ""))
                or self._normalize_action_text(secondary.get("last_evaluated_at", "")),
                "last_evaluated_query_text": self._normalize_action_text(primary.get("last_evaluated_query_text", ""))
                or self._normalize_action_text(secondary.get("last_evaluated_query_text", "")),
                "last_aggregated_at": now,
            }
        )
        return cast(dict[str, Any], normalized if normalized is not None else dict(primary))

    def _upsert_delayed_consequence_record_locked(
        self,
        candidate: Mapping[str, Any],
    ) -> dict[str, Any]:
        existing_records = list(self._delayed_consequence_records)
        best_index: int | None = None
        best_score = 0.0
        for index, record in enumerate(existing_records):
            score = self._delayed_consequence_aggregation_score_locked(record, candidate)
            if score > best_score:
                best_score = float(score)
                best_index = index
        if best_index is None:
            self._delayed_consequence_records = deque(
                [cast(dict[str, Any], candidate), *existing_records][:DEFAULT_DELAYED_CONSEQUENCE_RECORDS],
                maxlen=DEFAULT_DELAYED_CONSEQUENCE_RECORDS,
            )
            self._runtime_state.mark_mutated()
            return cast(dict[str, Any], candidate)
        merged = self._merge_delayed_consequence_records_locked(candidate, existing_records[best_index])
        self._delayed_consequence_records = deque(
            [merged, *(record for index, record in enumerate(existing_records) if index != best_index)],
            maxlen=DEFAULT_DELAYED_CONSEQUENCE_RECORDS,
        )
        self._delayed_consequence_compacted_total += 1
        self._record_brain_event_locked(
            {
                "type": "delayed_consequence_state_compacted",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "compacted_records": 1,
                "max_aggregate_count": int(merged.get("aggregate_count", 1) or 1),
                "record_id": str(merged.get("record_id", "")),
            }
        )
        self._runtime_state.mark_mutated()
        return merged

    def _compact_delayed_consequence_records_locked(self) -> dict[str, Any]:
        records = list(self._delayed_consequence_records)
        if len(records) < 2:
            return {
                "compacted_records": 0,
                "max_aggregate_count": max(
                    1,
                    max((int(record.get("aggregate_count", 1) or 1) for record in records), default=1),
                ),
            }
        compacted: list[dict[str, Any]] = []
        compacted_records = 0
        max_aggregate_count = 1
        for record in records:
            best_index: int | None = None
            best_score = 0.0
            for index, existing in enumerate(compacted):
                score = self._delayed_consequence_aggregation_score_locked(existing, record)
                if score > best_score:
                    best_score = float(score)
                    best_index = index
            if best_index is None:
                compacted.append(record)
                max_aggregate_count = max(max_aggregate_count, int(record.get("aggregate_count", 1) or 1))
                continue
            compacted[best_index] = self._merge_delayed_consequence_records_locked(compacted[best_index], record)
            compacted_records += 1
            max_aggregate_count = max(max_aggregate_count, int(compacted[best_index].get("aggregate_count", 1) or 1))
        if compacted_records <= 0:
            return {
                "compacted_records": 0,
                "max_aggregate_count": int(max_aggregate_count),
            }
        self._delayed_consequence_records = deque(compacted, maxlen=DEFAULT_DELAYED_CONSEQUENCE_RECORDS)
        self._delayed_consequence_compacted_total += int(compacted_records)
        self._record_brain_event_locked(
            {
                "type": "delayed_consequence_state_compacted",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "compacted_records": int(compacted_records),
                "max_aggregate_count": int(max_aggregate_count),
            }
        )
        self._runtime_state.mark_mutated()
        return {
            "compacted_records": int(compacted_records),
            "max_aggregate_count": int(max_aggregate_count),
        }

    def _cool_delayed_consequence_records_locked(
        self,
        *,
        current_token: int | None = None,
    ) -> dict[str, Any]:
        token = int(self._trainer.token_count if current_token is None else current_token)
        if not self._delayed_consequence_records:
            return {
                "cooled_records": 0,
                "retired_records": 0,
                "max_cooling_delta": 0.0,
                "retired_record_ids": [],
            }
        now = datetime.now(timezone.utc).isoformat()
        remaining: list[dict[str, Any]] = []
        cooled_records = 0
        retired_records = 0
        max_cooling_delta = 0.0
        retired_record_ids: list[str] = []
        for record in list(self._delayed_consequence_records):
            created_token_count = max(0, int(record.get("created_token_count", token)))
            last_activity_token_count = max(
                created_token_count,
                int(record.get("last_activity_token_count", created_token_count)),
            )
            last_cooling_token_count = max(
                created_token_count,
                int(record.get("last_cooling_token_count", last_activity_token_count)),
            )
            unresolved_penalty_balance = max(
                0.0,
                min(1.0, float(record.get("unresolved_penalty_balance", 0.0) or 0.0)),
            )
            inactivity_tokens = max(0, token - last_activity_token_count)
            if (
                unresolved_penalty_balance > 0.0
                and inactivity_tokens >= DEFAULT_DELAYED_CONSEQUENCE_COOLING_START_TOKENS
            ):
                cooling_anchor = max(
                    last_cooling_token_count,
                    last_activity_token_count + DEFAULT_DELAYED_CONSEQUENCE_COOLING_START_TOKENS,
                )
                cooling_delta_tokens = max(0, token - cooling_anchor)
                if cooling_delta_tokens > 0:
                    decay = math.exp(
                        -float(cooling_delta_tokens)
                        / float(max(1, DEFAULT_DELAYED_CONSEQUENCE_COOLING_WINDOW_TOKENS))
                    )
                    cooled_balance = max(0.0, unresolved_penalty_balance * float(decay))
                    cooling_delta = max(0.0, unresolved_penalty_balance - cooled_balance)
                    if cooling_delta > 1e-6:
                        record["unresolved_penalty_balance"] = float(cooled_balance)
                        record["last_cooling_token_count"] = int(token)
                        record["cooling_events"] = int(record.get("cooling_events", 0) or 0) + 1
                        record["cumulative_cooling_delta"] = float(
                            max(0.0, float(record.get("cumulative_cooling_delta", 0.0) or 0.0)) + cooling_delta
                        )
                        record["last_cooled_at"] = now
                        cooled_records += 1
                        max_cooling_delta = max(max_cooling_delta, cooling_delta)
                        self._delayed_consequence_cooled_total += 1
                        unresolved_penalty_balance = cooled_balance
            retirement_age_tokens = max(
                max(0, token - created_token_count),
                max(0, token - last_activity_token_count),
            )
            if (
                retirement_age_tokens >= DEFAULT_DELAYED_CONSEQUENCE_RETIREMENT_TOKENS
                and unresolved_penalty_balance <= DEFAULT_DELAYED_CONSEQUENCE_RETIREMENT_BALANCE_THRESHOLD
            ):
                retired_records += 1
                self._delayed_consequence_retired_total += 1
                retired_record_ids.append(str(record.get("record_id", "")))
                continue
            remaining.append(record)
        mutated = cooled_records > 0 or retired_records > 0
        if mutated:
            self._delayed_consequence_records = deque(remaining, maxlen=DEFAULT_DELAYED_CONSEQUENCE_RECORDS)
            if cooled_records > 0:
                self._record_brain_event_locked(
                    {
                        "type": "delayed_consequence_state_cooled",
                        "timestamp": now,
                        "cooled_records": int(cooled_records),
                        "max_cooling_delta": float(max_cooling_delta),
                    }
                )
            if retired_records > 0:
                self._record_brain_event_locked(
                    {
                        "type": "delayed_consequence_state_retired",
                        "timestamp": now,
                        "retired_records": int(retired_records),
                        "record_ids": retired_record_ids[:8],
                    }
                )
            self._runtime_state.mark_mutated()
        return {
            "cooled_records": int(cooled_records),
            "retired_records": int(retired_records),
            "max_cooling_delta": float(max_cooling_delta),
            "retired_record_ids": retired_record_ids[:8],
        }

    def _apply_background_source_delayed_penalty_locked(
        self,
        *,
        source_weights: Mapping[str, Any],
        penalty_score: float,
    ) -> list[str]:
        applied: list[str] = []
        calibrated_score = max(0.0, min(1.0, float(penalty_score)))
        if calibrated_score <= 0.0:
            return applied
        for runtime in self._brain_source_runtimes:
            weight = max(0.0, min(1.0, float(source_weights.get(runtime.name, 0.0) or 0.0)))
            if weight <= 0.0:
                continue
            entry = self._background_source_utility_entry_locked(runtime)
            sample = max(0.0, min(1.0, calibrated_score * weight))
            previous_penalty = max(0.0, min(1.0, float(entry.get("contradiction_decay_ema", 0.0) or 0.0)))
            entry["contradiction_decay_ema"] = float(
                sample if previous_penalty <= 0.0 else 0.75 * previous_penalty + 0.25 * sample
            )
            applied.append(runtime.name)
        return applied

    def _apply_background_source_forgiveness_locked(
        self,
        *,
        source_weights: Mapping[str, Any],
        forgiveness_score: float,
    ) -> list[str]:
        applied: list[str] = []
        calibrated_score = max(0.0, min(1.0, float(forgiveness_score)))
        if calibrated_score <= 0.0:
            return applied
        for runtime in self._brain_source_runtimes:
            weight = max(0.0, min(1.0, float(source_weights.get(runtime.name, 0.0) or 0.0)))
            if weight <= 0.0:
                continue
            entry = self._background_source_utility_entry_locked(runtime)
            previous_penalty = max(0.0, min(1.0, float(entry.get("contradiction_decay_ema", 0.0) or 0.0)))
            if previous_penalty <= 0.0:
                continue
            reduction = min(previous_penalty, float(calibrated_score) * float(weight))
            if reduction <= 0.0:
                continue
            entry["contradiction_decay_ema"] = float(max(0.0, previous_penalty - reduction))
            applied.append(runtime.name)
        return applied

    def _apply_background_source_delayed_consequence_locked(
        self,
        *,
        source_weights: Mapping[str, Any],
        consequence_score: float,
    ) -> list[str]:
        applied: list[str] = []
        calibrated_score = max(0.0, min(1.0, float(consequence_score)))
        if calibrated_score <= 0.0:
            return applied
        for runtime in self._brain_source_runtimes:
            weight = max(0.0, min(1.0, float(source_weights.get(runtime.name, 0.0) or 0.0)))
            if weight <= 0.0:
                continue
            entry = self._background_source_utility_entry_locked(runtime)
            sample = max(0.0, min(1.0, calibrated_score * weight))
            previous_delayed = max(0.0, min(1.0, float(entry.get("delayed_consequence_ema", 0.0) or 0.0)))
            entry["delayed_consequence_ema"] = float(
                sample if previous_delayed <= 0.0 else 0.75 * previous_delayed + 0.25 * sample
            )
            previous_utility = max(0.0, min(1.0, float(entry.get("utility_ema", 0.0) or 0.0)))
            reinforced_utility = max(
                previous_utility,
                float(entry.get("grounded_outcome_ema", 0.0) or 0.0),
                float(entry.get("delayed_consequence_ema", 0.0) or 0.0),
            )
            entry["utility_ema"] = float(
                reinforced_utility if previous_utility <= 0.0 else 0.80 * previous_utility + 0.20 * reinforced_utility
            )
            applied.append(runtime.name)
        return applied

    def _apply_background_source_family_summary_locked(
        self,
        *,
        source_weights: Mapping[str, Any],
        family_summary_score: float,
    ) -> list[str]:
        applied: list[str] = []
        calibrated_score = max(0.0, min(1.0, float(family_summary_score)))
        for runtime in self._brain_source_runtimes:
            weight = max(0.0, min(1.0, float(source_weights.get(runtime.name, 0.0) or 0.0)))
            if weight <= 0.0:
                continue
            entry = self._background_source_utility_entry_locked(runtime)
            sample = max(0.0, min(1.0, calibrated_score * weight))
            previous_summary = max(0.0, min(1.0, float(entry.get("grounded_family_summary_ema", 0.0) or 0.0)))
            entry["grounded_family_summary_ema"] = float(
                sample if previous_summary <= 0.0 else 0.70 * previous_summary + 0.30 * sample
            )
            previous_utility = max(0.0, min(1.0, float(entry.get("utility_ema", 0.0) or 0.0)))
            reinforced_utility = max(
                previous_utility,
                float(entry.get("grounded_outcome_ema", 0.0) or 0.0),
                float(entry.get("delayed_consequence_ema", 0.0) or 0.0),
                float(entry.get("grounded_family_summary_ema", 0.0) or 0.0),
            )
            entry["utility_ema"] = float(
                reinforced_utility if previous_utility <= 0.0 else 0.80 * previous_utility + 0.20 * reinforced_utility
            )
            applied.append(runtime.name)
        return applied

    def _apply_provider_delayed_penalty_locked(
        self,
        *,
        autonomy: dict[str, Any],
        provider_weights: Mapping[str, Any],
        penalty_score: float,
    ) -> list[str]:
        curriculum = self._normalize_provider_curriculum(autonomy.get("provider_curriculum"))
        calibrated_score = max(0.0, min(1.0, float(penalty_score)))
        if not curriculum or calibrated_score <= 0.0:
            return []
        applied: list[str] = []
        for raw_provider, raw_weight in dict(provider_weights).items():
            provider = " ".join(str(raw_provider).split()).strip().lower()
            weight = max(0.0, min(1.0, float(raw_weight or 0.0)))
            if not provider or weight <= 0.0:
                continue
            entry = curriculum.get(provider)
            if not isinstance(entry, Mapping):
                continue
            sample = max(0.0, min(1.0, calibrated_score * weight))
            previous_penalty = max(0.0, min(1.0, float(entry.get("contradiction_decay_ema", 0.0) or 0.0)))
            entry["contradiction_decay_ema"] = float(
                sample if previous_penalty <= 0.0 else 0.75 * previous_penalty + 0.25 * sample
            )
            applied.append(provider)
        if applied:
            autonomy["provider_curriculum"] = curriculum
        return applied

    def _apply_provider_forgiveness_locked(
        self,
        *,
        autonomy: dict[str, Any],
        provider_weights: Mapping[str, Any],
        forgiveness_score: float,
    ) -> list[str]:
        curriculum = self._normalize_provider_curriculum(autonomy.get("provider_curriculum"))
        calibrated_score = max(0.0, min(1.0, float(forgiveness_score)))
        if not curriculum or calibrated_score <= 0.0:
            return []
        applied: list[str] = []
        for raw_provider, raw_weight in dict(provider_weights).items():
            provider = " ".join(str(raw_provider).split()).strip().lower()
            weight = max(0.0, min(1.0, float(raw_weight or 0.0)))
            if not provider or weight <= 0.0:
                continue
            entry = curriculum.get(provider)
            if not isinstance(entry, Mapping):
                continue
            previous_penalty = max(0.0, min(1.0, float(entry.get("contradiction_decay_ema", 0.0) or 0.0)))
            if previous_penalty <= 0.0:
                continue
            reduction = min(previous_penalty, float(calibrated_score) * float(weight))
            if reduction <= 0.0:
                continue
            entry["contradiction_decay_ema"] = float(max(0.0, previous_penalty - reduction))
            applied.append(provider)
        if applied:
            autonomy["provider_curriculum"] = curriculum
        return applied

    def _apply_provider_delayed_consequence_locked(
        self,
        *,
        autonomy: dict[str, Any],
        provider_weights: Mapping[str, Any],
        consequence_score: float,
    ) -> list[str]:
        curriculum = self._normalize_provider_curriculum(autonomy.get("provider_curriculum"))
        calibrated_score = max(0.0, min(1.0, float(consequence_score)))
        if not curriculum or calibrated_score <= 0.0:
            return []
        applied: list[str] = []
        for raw_provider, raw_weight in dict(provider_weights).items():
            provider = " ".join(str(raw_provider).split()).strip().lower()
            weight = max(0.0, min(1.0, float(raw_weight or 0.0)))
            if not provider or weight <= 0.0:
                continue
            entry = curriculum.get(provider)
            if not isinstance(entry, Mapping):
                continue
            sample = max(0.0, min(1.0, calibrated_score * weight))
            previous_delayed = max(0.0, min(1.0, float(entry.get("delayed_consequence_ema", 0.0) or 0.0)))
            entry["delayed_consequence_ema"] = float(
                sample if previous_delayed <= 0.0 else 0.75 * previous_delayed + 0.25 * sample
            )
            previous_utility = max(0.0, min(1.0, float(entry.get("utility_ema", 0.0) or 0.0)))
            reinforced_utility = max(
                previous_utility,
                float(entry.get("grounded_outcome_ema", 0.0) or 0.0),
                float(entry.get("delayed_consequence_ema", 0.0) or 0.0),
            )
            entry["utility_ema"] = float(
                reinforced_utility if previous_utility <= 0.0 else 0.80 * previous_utility + 0.20 * reinforced_utility
            )
            applied.append(provider)
        if applied:
            autonomy["provider_curriculum"] = curriculum
        return applied

    def _apply_provider_family_summary_locked(
        self,
        *,
        autonomy: dict[str, Any],
        provider_weights: Mapping[str, Any],
        family_summary_score: float,
    ) -> list[str]:
        curriculum = self._normalize_provider_curriculum(autonomy.get("provider_curriculum"))
        calibrated_score = max(0.0, min(1.0, float(family_summary_score)))
        if not curriculum:
            return []
        applied: list[str] = []
        for raw_provider, raw_weight in dict(provider_weights).items():
            provider = " ".join(str(raw_provider).split()).strip().lower()
            weight = max(0.0, min(1.0, float(raw_weight or 0.0)))
            if not provider or weight <= 0.0:
                continue
            entry = curriculum.get(provider)
            if not isinstance(entry, Mapping):
                continue
            sample = max(0.0, min(1.0, calibrated_score * weight))
            previous_summary = max(0.0, min(1.0, float(entry.get("grounded_family_summary_ema", 0.0) or 0.0)))
            entry["grounded_family_summary_ema"] = float(
                sample if previous_summary <= 0.0 else 0.70 * previous_summary + 0.30 * sample
            )
            previous_utility = max(0.0, min(1.0, float(entry.get("utility_ema", 0.0) or 0.0)))
            reinforced_utility = max(
                previous_utility,
                float(entry.get("grounded_outcome_ema", 0.0) or 0.0),
                float(entry.get("delayed_consequence_ema", 0.0) or 0.0),
                float(entry.get("grounded_family_summary_ema", 0.0) or 0.0),
            )
            entry["utility_ema"] = float(
                reinforced_utility if previous_utility <= 0.0 else 0.80 * previous_utility + 0.20 * reinforced_utility
            )
            applied.append(provider)
        if applied:
            autonomy["provider_curriculum"] = curriculum
        return applied

    def _apply_delayed_query_consequence_locked(
        self,
        *,
        query_result: Mapping[str, Any],
    ) -> dict[str, Any]:
        remerge = self._remerge_converged_delayed_consequence_families_locked()
        split = self._split_divergent_delayed_consequence_families_locked()
        compaction = self._compact_delayed_consequence_records_locked()
        maintenance = self._cool_delayed_consequence_records_locked()
        summary = {
            "enabled": True,
            "record_count": int(len(self._delayed_consequence_records)),
            "matched_records": 0,
            "credited_records": 0,
            "penalized_records": 0,
            "forgiven_records": 0,
            "remerged_records": int(remerge.get("remerged_records", 0) or 0),
            "split_records": int(split.get("split_records", 0) or 0),
            "max_split_branch_overlap": float(split.get("max_branch_overlap", 1.0) or 1.0),
            "compacted_records": int(compaction.get("compacted_records", 0) or 0),
            "max_aggregate_count": int(compaction.get("max_aggregate_count", 1) or 1),
            "cooled_records": int(maintenance.get("cooled_records", 0) or 0),
            "retired_records": int(maintenance.get("retired_records", 0) or 0),
            "credited_source_names": [],
            "credited_providers": [],
            "penalized_source_names": [],
            "penalized_providers": [],
            "forgiven_source_names": [],
            "forgiven_providers": [],
            "max_improvement": 0.0,
            "max_penalty": 0.0,
            "max_regression": 0.0,
            "max_forgiveness": 0.0,
            "max_family_summary_score": 0.0,
            "max_cooling_delta": float(maintenance.get("max_cooling_delta", 0.0) or 0.0),
            "contradicted_action_count": 0,
            "contradiction_signal": 0.0,
        }
        if not self._delayed_consequence_records:
            return summary
        query_snapshot = self._query_progress_snapshot_locked(query_result)
        if not query_snapshot.get("query_text") or not list(query_snapshot.get("query_terms") or []):
            return summary
        autonomy = cast(dict[str, Any] | None, self._brain_config.get("autonomy"))
        contradiction_signal, contradicted_action_count = self._recent_action_contradiction_signal_locked(
            str(query_snapshot.get("query_text", "")),
        )
        summary["contradicted_action_count"] = int(contradicted_action_count)
        summary["contradiction_signal"] = float(contradiction_signal)
        credited_sources: set[str] = set()
        credited_providers: set[str] = set()
        penalized_sources: set[str] = set()
        penalized_providers: set[str] = set()
        forgiven_sources: set[str] = set()
        forgiven_providers: set[str] = set()
        mutated = False
        timestamp = datetime.now(timezone.utc).isoformat()
        current_token = int(self._trainer.token_count)
        current_query_score = max(0.0, min(1.0, float(query_snapshot.get("query_score", 0.0) or 0.0)))
        current_grounded_fraction = max(0.0, min(1.0, float(query_snapshot.get("grounded_fraction", 0.0) or 0.0)))
        current_top_similarity = max(0.0, min(1.0, float(query_snapshot.get("top_similarity", 0.0) or 0.0)))
        current_top_query_overlap_ratio = max(
            0.0,
            min(1.0, float(query_snapshot.get("top_query_overlap_ratio", 0.0) or 0.0)),
        )
        current_supported_episode_hits = max(0, int(query_snapshot.get("supported_episode_hits", 0) or 0))
        query_terms = list(query_snapshot.get("query_terms") or [])
        unsupported_terms = list(query_snapshot.get("unsupported_terms") or [])
        unsupported_ratio = min(1.0, float(len(unsupported_terms)) / float(max(1, len(query_terms))))
        current_query_text = self._normalize_action_text(query_snapshot.get("query_text", "")).lower()
        for record in list(self._delayed_consequence_records):
            match_score = self._delayed_consequence_match_score_locked(record, query_snapshot)
            adverse_examples = {
                example.lower()
                for example in self._delayed_consequence_branch_examples(record, field="adverse_query_examples")
            }
            supportive_examples = {
                example.lower()
                for example in self._delayed_consequence_branch_examples(record, field="supportive_query_examples")
            }
            if current_query_text and (current_query_text in supportive_examples or current_query_text in adverse_examples):
                match_score = max(match_score, DEFAULT_DELAYED_CONSEQUENCE_MATCH_THRESHOLD)
            if match_score < DEFAULT_DELAYED_CONSEQUENCE_MATCH_THRESHOLD:
                continue
            summary["matched_records"] = int(summary["matched_records"]) + 1
            best_query_score = max(
                0.0,
                min(
                    1.0,
                    max(
                        float(record.get("baseline_query_score", 0.0) or 0.0),
                        float(record.get("best_query_score", 0.0) or 0.0),
                    ),
                ),
            )
            best_grounded_fraction = max(
                0.0,
                min(
                    1.0,
                    max(
                        float(record.get("baseline_grounded_fraction", 0.0) or 0.0),
                        float(record.get("best_grounded_fraction", 0.0) or 0.0),
                    ),
                ),
            )
            unresolved_penalty_balance = max(
                0.0,
                min(1.0, float(record.get("unresolved_penalty_balance", 0.0) or 0.0)),
            )
            score_improvement = max(0.0, current_query_score - best_query_score)
            grounded_improvement = max(0.0, current_grounded_fraction - best_grounded_fraction)
            improvement = max(score_improvement, 0.85 * grounded_improvement)
            split_branch = " ".join(str(record.get("split_branch", "")).split()).strip().lower()
            primary_query_text = self._normalize_action_text(record.get("query_text", "")).lower()
            primary_query_recovery = (
                unresolved_penalty_balance > 0.0
                and current_query_text
                and current_query_text == primary_query_text
                and current_query_text not in adverse_examples
                and current_top_similarity >= 0.90
                and current_top_query_overlap_ratio >= 0.40
                and current_supported_episode_hits >= 2
            )
            supportive_recovery = (
                unresolved_penalty_balance > 0.0
                and current_query_text
                and current_query_text not in adverse_examples
                and (split_branch != "supportive" or not supportive_examples or current_query_text in supportive_examples)
                and (
                    (
                        current_grounded_fraction >= max(0.60, best_grounded_fraction - 0.05)
                        and unsupported_ratio < DEFAULT_DELAYED_CONTRADICTION_UNSUPPORTED_THRESHOLD
                    )
                    or primary_query_recovery
                )
            )
            effective_improvement = max(
                improvement,
                max(
                    DEFAULT_DELAYED_CONSEQUENCE_DELTA_THRESHOLD,
                    0.15 * unresolved_penalty_balance,
                )
                if supportive_recovery
                else 0.0,
            )
            if effective_improvement >= DEFAULT_DELAYED_CONSEQUENCE_DELTA_THRESHOLD:
                support_multiplier = self._delayed_consequence_family_support_multiplier(record, mode="credit")
                delayed_sample = max(
                    0.0,
                    min(
                        1.0,
                        1.5
                        * float(record.get("outcome_score", 0.0) or 0.0)
                        * float(match_score)
                        * float(effective_improvement)
                        * float(support_multiplier),
                    ),
                )
                if delayed_sample > 0.0:
                    applied_sources = self._apply_background_source_delayed_consequence_locked(
                        source_weights=cast(Mapping[str, Any], record.get("source_weights") or {}),
                        consequence_score=delayed_sample,
                    )
                    applied_providers = []
                    if autonomy is not None:
                        applied_providers = self._apply_provider_delayed_consequence_locked(
                            autonomy=autonomy,
                            provider_weights=cast(Mapping[str, Any], record.get("provider_weights") or {}),
                            consequence_score=delayed_sample,
                        )
                    if applied_sources or applied_providers:
                        mutated = True
                        credited_sources.update(applied_sources)
                        credited_providers.update(applied_providers)
                        summary["credited_records"] = int(summary["credited_records"]) + 1
                        summary["max_improvement"] = max(float(summary["max_improvement"]), float(effective_improvement))
                        record["best_query_score"] = max(best_query_score, current_query_score)
                        record["best_grounded_fraction"] = max(best_grounded_fraction, current_grounded_fraction)
                        record["credit_events"] = int(record.get("credit_events", 0)) + 1
                        record["resolved_improvement"] = max(
                            float(record.get("resolved_improvement", 0.0) or 0.0),
                            float(effective_improvement),
                        )
                        record["last_match_score"] = float(match_score)
                        record["last_evaluated_at"] = timestamp
                        record["last_evaluated_query_text"] = str(query_snapshot.get("query_text", ""))
                        record["last_evaluated_token_count"] = int(current_token)
                        record["last_activity_token_count"] = int(current_token)
                        record["last_credit_token_count"] = int(current_token)
                        record["last_cooling_token_count"] = int(current_token)
                        record["last_credit_score"] = float(delayed_sample)
                        record["unresolved_penalty_balance"] = float(unresolved_penalty_balance)
                        self._update_delayed_consequence_trajectory_locked(
                            record,
                            event_type="credit",
                            event_score=delayed_sample,
                            timestamp=timestamp,
                            current_token=current_token,
                        )
                        self._update_delayed_consequence_branch_partition_locked(
                            record,
                            event_type="credit",
                            query_text=str(query_snapshot.get("query_text", "")),
                        )
                        forgiveness_budget = min(
                            unresolved_penalty_balance,
                            float(DEFAULT_FORGIVENESS_RECOVERY_RATIO) * float(delayed_sample),
                        )
                        if forgiveness_budget > 0.0:
                            forgiven_sources_now = self._apply_background_source_forgiveness_locked(
                                source_weights=cast(Mapping[str, Any], record.get("source_weights") or {}),
                                forgiveness_score=forgiveness_budget,
                            )
                            forgiven_providers_now = []
                            if autonomy is not None:
                                forgiven_providers_now = self._apply_provider_forgiveness_locked(
                                    autonomy=autonomy,
                                    provider_weights=cast(Mapping[str, Any], record.get("provider_weights") or {}),
                                    forgiveness_score=forgiveness_budget,
                                )
                            if forgiven_sources_now or forgiven_providers_now:
                                forgiven_sources.update(forgiven_sources_now)
                                forgiven_providers.update(forgiven_providers_now)
                                summary["forgiven_records"] = int(summary["forgiven_records"]) + 1
                                summary["max_forgiveness"] = max(
                                    float(summary["max_forgiveness"]),
                                    float(forgiveness_budget),
                                )
                                record["forgiveness_events"] = int(record.get("forgiveness_events", 0)) + 1
                                record["last_forgiveness_score"] = float(forgiveness_budget)
                                record["last_forgiveness_token_count"] = int(current_token)
                                record["last_activity_token_count"] = int(current_token)
                                record["last_cooling_token_count"] = int(current_token)
                                record["unresolved_penalty_balance"] = float(
                                    max(0.0, unresolved_penalty_balance - forgiveness_budget)
                                )
                                self._update_delayed_consequence_trajectory_locked(
                                    record,
                                    event_type="forgiveness",
                                    event_score=forgiveness_budget,
                                    timestamp=timestamp,
                                    current_token=current_token,
                                )
                                self._update_delayed_consequence_branch_partition_locked(
                                    record,
                                    event_type="forgiveness",
                                    query_text=str(query_snapshot.get("query_text", "")),
                                )
                        family_summary_score = self._grounded_family_summary_score(record)
                        summary["max_family_summary_score"] = max(
                            float(summary["max_family_summary_score"]),
                            float(family_summary_score),
                        )
                        self._apply_background_source_family_summary_locked(
                            source_weights=cast(Mapping[str, Any], record.get("source_weights") or {}),
                            family_summary_score=family_summary_score,
                        )
                        if autonomy is not None:
                            self._apply_provider_family_summary_locked(
                                autonomy=autonomy,
                                provider_weights=cast(Mapping[str, Any], record.get("provider_weights") or {}),
                                family_summary_score=family_summary_score,
                            )
                        continue
            expectation_score = max(
                float(record.get("outcome_score", 0.0) or 0.0),
                float(best_query_score),
                float(best_grounded_fraction),
            )
            regression = max(
                0.0,
                expectation_score - max(current_query_score, current_grounded_fraction),
                best_query_score - current_query_score,
                best_grounded_fraction - current_grounded_fraction,
            )
            contradiction_decay = max(
                float(regression),
                0.75 * unsupported_ratio
                if expectation_score >= 0.60 and unsupported_ratio >= DEFAULT_DELAYED_CONTRADICTION_UNSUPPORTED_THRESHOLD
                else 0.0,
                0.85 * float(contradiction_signal),
            )
            if contradiction_decay < DEFAULT_DELAYED_CONTRADICTION_DECAY_THRESHOLD:
                continue
            support_multiplier = self._delayed_consequence_family_support_multiplier(record, mode="penalty")
            penalty_sample = max(
                0.0,
                min(
                    1.0,
                    1.35
                    * max(0.35, float(record.get("outcome_score", 0.0) or 0.0))
                    * float(match_score)
                    * float(contradiction_decay)
                    * float(support_multiplier),
                ),
            )
            if penalty_sample <= 0.0:
                continue
            penalized_sources_now = self._apply_background_source_delayed_penalty_locked(
                source_weights=cast(Mapping[str, Any], record.get("source_weights") or {}),
                penalty_score=penalty_sample,
            )
            penalized_providers_now = []
            if autonomy is not None:
                penalized_providers_now = self._apply_provider_delayed_penalty_locked(
                    autonomy=autonomy,
                    provider_weights=cast(Mapping[str, Any], record.get("provider_weights") or {}),
                    penalty_score=penalty_sample,
                )
            if not penalized_sources_now and not penalized_providers_now:
                continue
            mutated = True
            penalized_sources.update(penalized_sources_now)
            penalized_providers.update(penalized_providers_now)
            summary["penalized_records"] = int(summary["penalized_records"]) + 1
            summary["max_penalty"] = max(float(summary["max_penalty"]), float(penalty_sample))
            summary["max_regression"] = max(float(summary["max_regression"]), float(regression))
            record["penalty_events"] = int(record.get("penalty_events", 0)) + 1
            record["max_regression"] = max(float(record.get("max_regression", 0.0) or 0.0), float(regression))
            record["max_contradiction_signal"] = max(
                float(record.get("max_contradiction_signal", 0.0) or 0.0),
                float(contradiction_signal),
            )
            record["unresolved_penalty_balance"] = float(min(1.0, unresolved_penalty_balance + penalty_sample))
            record["last_match_score"] = float(match_score)
            record["last_evaluated_at"] = timestamp
            record["last_evaluated_query_text"] = str(query_snapshot.get("query_text", ""))
            record["last_evaluated_token_count"] = int(current_token)
            record["last_activity_token_count"] = int(current_token)
            record["last_penalty_token_count"] = int(current_token)
            record["last_cooling_token_count"] = int(current_token)
            record["last_penalty_score"] = float(penalty_sample)
            self._update_delayed_consequence_trajectory_locked(
                record,
                event_type="penalty",
                event_score=penalty_sample,
                timestamp=timestamp,
                current_token=current_token,
            )
            self._update_delayed_consequence_branch_partition_locked(
                record,
                event_type="penalty",
                query_text=str(query_snapshot.get("query_text", "")),
            )
            record["last_penalty_reason"] = (
                "contradicted_action"
                if contradiction_signal >= max(float(regression), 0.75 * unsupported_ratio)
                else "unsupported_decay"
                if unsupported_ratio >= DEFAULT_DELAYED_CONTRADICTION_UNSUPPORTED_THRESHOLD
                else "regression_decay"
            )
            family_summary_score = self._grounded_family_summary_score(record)
            summary["max_family_summary_score"] = max(
                float(summary["max_family_summary_score"]),
                float(family_summary_score),
            )
            self._apply_background_source_family_summary_locked(
                source_weights=cast(Mapping[str, Any], record.get("source_weights") or {}),
                family_summary_score=family_summary_score,
            )
            if autonomy is not None:
                self._apply_provider_family_summary_locked(
                    autonomy=autonomy,
                    provider_weights=cast(Mapping[str, Any], record.get("provider_weights") or {}),
                    family_summary_score=family_summary_score,
                )
        if credited_sources:
            summary["credited_source_names"] = sorted(credited_sources)
        if credited_providers:
            summary["credited_providers"] = sorted(credited_providers)
        if penalized_sources:
            summary["penalized_source_names"] = sorted(penalized_sources)
        if penalized_providers:
            summary["penalized_providers"] = sorted(penalized_providers)
        if forgiven_sources:
            summary["forgiven_source_names"] = sorted(forgiven_sources)
        if forgiven_providers:
            summary["forgiven_providers"] = sorted(forgiven_providers)
        if mutated:
            if int(summary["credited_records"]) > 0:
                self._record_brain_event_locked(
                    {
                        "type": "delayed_consequence_applied",
                        "timestamp": timestamp,
                        "query_text": str(query_snapshot.get("query_text", "")),
                        "credited_records": int(summary["credited_records"]),
                        "credited_source_names": sorted(credited_sources),
                        "credited_providers": sorted(credited_providers),
                        "max_improvement": float(summary["max_improvement"]),
                    }
                )
            if int(summary["penalized_records"]) > 0:
                self._record_brain_event_locked(
                    {
                        "type": "delayed_consequence_penalized",
                        "timestamp": timestamp,
                        "query_text": str(query_snapshot.get("query_text", "")),
                        "penalized_records": int(summary["penalized_records"]),
                        "penalized_source_names": sorted(penalized_sources),
                        "penalized_providers": sorted(penalized_providers),
                        "max_penalty": float(summary["max_penalty"]),
                        "max_regression": float(summary["max_regression"]),
                        "contradiction_signal": float(contradiction_signal),
                    }
                )
            if int(summary["forgiven_records"]) > 0:
                self._record_brain_event_locked(
                    {
                        "type": "delayed_consequence_forgiven",
                        "timestamp": timestamp,
                        "query_text": str(query_snapshot.get("query_text", "")),
                        "forgiven_records": int(summary["forgiven_records"]),
                        "forgiven_source_names": sorted(forgiven_sources),
                        "forgiven_providers": sorted(forgiven_providers),
                        "max_forgiveness": float(summary["max_forgiveness"]),
                    }
                )
            remerge_post = self._remerge_converged_delayed_consequence_families_locked()
            summary["remerged_records"] = int(summary["remerged_records"]) + int(remerge_post.get("remerged_records", 0) or 0)
            split_post = self._split_divergent_delayed_consequence_families_locked()
            summary["split_records"] = int(summary["split_records"]) + int(split_post.get("split_records", 0) or 0)
            summary["max_split_branch_overlap"] = min(
                float(summary["max_split_branch_overlap"]),
                float(split_post.get("max_branch_overlap", 1.0) or 1.0),
            )
            summary["record_count"] = int(len(self._delayed_consequence_records))
            self._runtime_state.mark_mutated()
        return summary

    def _record_response_consequence_candidate_locked(
        self,
        *,
        query_result: Mapping[str, Any],
        response: Mapping[str, Any],
        outcome_score: float,
    ) -> dict[str, Any] | None:
        source_weights = self._selected_evidence_weight_map(
            response,
            singular_field="source_name",
            plural_field="source_names",
        )
        provider_weights = self._selected_evidence_weight_map(
            response,
            singular_field="provider",
            plural_field="providers",
        )
        if not source_weights and not provider_weights:
            return None
        query_snapshot = self._query_progress_snapshot_locked(query_result)
        current_token = int(self._trainer.token_count)
        normalized = self._normalize_delayed_consequence_record(
            {
                "record_id": str(uuid4()),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "created_token_count": int(current_token),
                "origin": "response_selected_evidence",
                "query_text": str(query_snapshot.get("query_text", "")),
                "query_terms": list(query_snapshot.get("query_terms") or []),
                "baseline_grounded_fraction": float(query_snapshot.get("grounded_fraction", 0.0) or 0.0),
                "best_grounded_fraction": float(query_snapshot.get("grounded_fraction", 0.0) or 0.0),
                "baseline_query_score": float(query_snapshot.get("query_score", 0.0) or 0.0),
                "best_query_score": float(query_snapshot.get("query_score", 0.0) or 0.0),
                "outcome_score": float(outcome_score),
                "source_weights": dict(source_weights),
                "provider_weights": dict(provider_weights),
                "last_activity_token_count": int(current_token),
                "last_evaluated_token_count": int(current_token),
                "last_cooling_token_count": int(current_token),
            }
        )
        if normalized is None:
            return None
        merged_record = self._upsert_delayed_consequence_record_locked(normalized)
        return {
            "record_id": str(merged_record.get("record_id", "")),
            "query_text": str(merged_record.get("query_text", "")),
            "query_examples": self._delayed_consequence_query_examples(merged_record),
            "aggregate_count": int(merged_record.get("aggregate_count", 1) or 1),
            "source_names": sorted(dict(merged_record.get("source_weights") or {}).keys()),
            "providers": sorted(dict(merged_record.get("provider_weights") or {}).keys()),
            "baseline_query_score": float(merged_record.get("baseline_query_score", 0.0) or 0.0),
            "baseline_grounded_fraction": float(merged_record.get("baseline_grounded_fraction", 0.0) or 0.0),
        }

    def _apply_background_source_response_provenance_locked(
        self,
        *,
        response: Mapping[str, Any],
        outcome_score: float,
    ) -> bool:
        weighted_sources = self._selected_evidence_weight_map(
            response,
            singular_field="source_name",
            plural_field="source_names",
        )
        if not weighted_sources:
            return False
        applied = False
        for runtime in self._brain_source_runtimes:
            if runtime.name not in weighted_sources:
                continue
            entry = self._background_source_utility_entry_locked(runtime)
            sample = max(0.0, min(1.0, float(outcome_score) * float(weighted_sources[runtime.name])))
            previous_outcome = max(0.0, min(1.0, float(entry.get("grounded_outcome_ema", 0.0) or 0.0)))
            entry["grounded_outcome_ema"] = float(sample if previous_outcome <= 0.0 else 0.70 * previous_outcome + 0.30 * sample)
            previous_utility = max(0.0, min(1.0, float(entry.get("utility_ema", 0.0) or 0.0)))
            reinforced_utility = max(previous_utility, float(entry["grounded_outcome_ema"]))
            entry["utility_ema"] = float(
                reinforced_utility if previous_utility <= 0.0 else 0.75 * previous_utility + 0.25 * reinforced_utility
            )
            applied = True
        if applied:
            self._runtime_state.mark_mutated()
        return applied

    def _apply_background_source_outcome_calibration_locked(
        self,
        *,
        query_text: str,
        outcome_score: float,
    ) -> None:
        calibrated_score = max(0.0, min(1.0, float(outcome_score)))
        if calibrated_score <= 0.0 or not self._brain_source_runtimes:
            return
        focus_terms = self._background_focus_terms_locked(limit=12)
        query_terms = [
            _canonical_provider_term(term)
            for term in salient_query_terms(query_text)
            if _canonical_provider_term(term)
        ]
        combined_focus_terms = list(dict.fromkeys([*query_terms, *focus_terms]))[:12]
        ranked: list[tuple[float, float, float, _BrainSourceRuntime]] = []
        for runtime in self._brain_source_runtimes:
            entry = self._background_source_utility_entry_locked(runtime)
            if int(entry.get("selections", 0)) <= 0 and float(entry.get("utility_ema", 0.0) or 0.0) <= 0.0:
                continue
            semantic_alignment = self._brain_source_semantic_match_locked(runtime, combined_focus_terms)
            historical_alignment = max(
                float(entry.get("semantic_alignment_ema", 0.0) or 0.0),
                float(entry.get("focus_overlap_ema", 0.0) or 0.0),
            )
            utility_signal = max(
                0.0,
                float(entry.get("utility_ema", 0.0) or 0.0)
                - DEFAULT_UTILITY_PENALTY_WEIGHT * float(entry.get("contradiction_decay_ema", 0.0) or 0.0),
            )
            priority = max(semantic_alignment, historical_alignment) * max(0.35, utility_signal if utility_signal > 0.0 else 0.35)
            ranked.append((float(priority), float(semantic_alignment), float(utility_signal), runtime))
        if not ranked:
            return
        ranked.sort(key=lambda item: (-float(item[0]), -float(item[1]), -float(item[2]), item[3].name))
        priority, semantic_alignment, _utility_signal, runtime = ranked[0]
        if float(priority) <= 0.0:
            return
        entry = self._background_source_utility_entry_locked(runtime)
        sample = max(0.0, min(1.0, calibrated_score * max(float(priority), float(semantic_alignment), 0.35)))
        previous_outcome = max(0.0, min(1.0, float(entry.get("grounded_outcome_ema", 0.0) or 0.0)))
        entry["grounded_outcome_ema"] = float(sample if previous_outcome <= 0.0 else 0.70 * previous_outcome + 0.30 * sample)
        previous_utility = max(0.0, min(1.0, float(entry.get("utility_ema", 0.0) or 0.0)))
        reinforced_utility = max(previous_utility, float(entry["grounded_outcome_ema"]))
        entry["utility_ema"] = float(
            reinforced_utility
            if previous_utility <= 0.0
            else 0.75 * previous_utility + 0.25 * reinforced_utility
        )
        self._runtime_state.mark_mutated()

    def _delayed_consequence_summary_locked(self, limit: int = 4) -> dict[str, Any]:
        records = list(self._delayed_consequence_records)
        current_token = int(self._trainer.token_count)
        credited_count = sum(1 for record in records if int(record.get("credit_events", 0) or 0) > 0)
        penalized_count = sum(1 for record in records if int(record.get("penalty_events", 0) or 0) > 0)
        forgiven_count = sum(1 for record in records if int(record.get("forgiveness_events", 0) or 0) > 0)
        aggregated_count = sum(1 for record in records if int(record.get("aggregate_count", 1) or 1) > 1)
        aggregate_occurrence_count = sum(max(1, int(record.get("aggregate_count", 1) or 1)) for record in records)
        trajectory_state_counts = Counter(self._delayed_consequence_trajectory_state(record) for record in records)
        recent_records: list[dict[str, Any]] = []
        for record in records[: max(1, int(limit))]:
            recent_records.append(
                {
                    "record_id": str(record.get("record_id", "")),
                    "origin": str(record.get("origin", "response_selected_evidence")),
                    "created_at": str(record.get("created_at", "")),
                    "query_text": str(record.get("query_text", "")),
                    "query_examples": self._delayed_consequence_query_examples(record),
                    "aggregate_count": int(record.get("aggregate_count", 1) or 1),
                    "aggregation_events": int(record.get("aggregation_events", 0) or 0),
                    "supportive_query_examples": self._delayed_consequence_branch_examples(record, field="supportive_query_examples"),
                    "adverse_query_examples": self._delayed_consequence_branch_examples(record, field="adverse_query_examples"),
                    "supportive_occurrence_count": int(record.get("supportive_occurrence_count", 0) or 0),
                    "adverse_occurrence_count": int(record.get("adverse_occurrence_count", 0) or 0),
                    "aggregate_support_multiplier": float(self._delayed_consequence_support_multiplier(record)),
                    "family_support_multiplier": float(self._delayed_consequence_family_support_multiplier(record, mode="credit")),
                    "trajectory_support_multiplier": float(
                        self._delayed_consequence_trajectory_support_multiplier(record, mode="credit")
                    ),
                    "trajectory_penalty_multiplier": float(
                        self._delayed_consequence_trajectory_support_multiplier(record, mode="penalty")
                    ),
                    "grounded_family_summary_score": float(self._grounded_family_summary_score(record)),
                    "source_names": sorted(dict(record.get("source_weights") or {}).keys()),
                    "providers": sorted(dict(record.get("provider_weights") or {}).keys()),
                    "baseline_query_score": float(record.get("baseline_query_score", 0.0) or 0.0),
                    "best_query_score": float(record.get("best_query_score", 0.0) or 0.0),
                    "baseline_grounded_fraction": float(record.get("baseline_grounded_fraction", 0.0) or 0.0),
                    "best_grounded_fraction": float(record.get("best_grounded_fraction", 0.0) or 0.0),
                    "credit_events": int(record.get("credit_events", 0) or 0),
                    "penalty_events": int(record.get("penalty_events", 0) or 0),
                    "forgiveness_events": int(record.get("forgiveness_events", 0) or 0),
                    "cooling_events": int(record.get("cooling_events", 0) or 0),
                    "trajectory_state": self._delayed_consequence_trajectory_state(record),
                    "trajectory_event_count": int(record.get("trajectory_event_count", 0) or 0),
                    "trajectory_credit_total": float(record.get("trajectory_credit_total", 0.0) or 0.0),
                    "trajectory_penalty_total": float(record.get("trajectory_penalty_total", 0.0) or 0.0),
                    "trajectory_forgiveness_total": float(record.get("trajectory_forgiveness_total", 0.0) or 0.0),
                    "trajectory_net_score": float(record.get("trajectory_net_score", 0.0) or 0.0),
                    "trajectory_signal_balance": float(self._delayed_consequence_trajectory_balance(record)),
                    "trajectory_recent_delta_ema": float(self._delayed_consequence_trajectory_recent_signal(record)),
                    "trajectory_peak_score": float(record.get("trajectory_peak_score", 0.0) or 0.0),
                    "trajectory_floor_score": float(record.get("trajectory_floor_score", 0.0) or 0.0),
                    "unresolved_penalty_balance": float(record.get("unresolved_penalty_balance", 0.0) or 0.0),
                    "resolved_improvement": float(record.get("resolved_improvement", 0.0) or 0.0),
                    "max_regression": float(record.get("max_regression", 0.0) or 0.0),
                    "max_contradiction_signal": float(record.get("max_contradiction_signal", 0.0) or 0.0),
                    "cumulative_cooling_delta": float(record.get("cumulative_cooling_delta", 0.0) or 0.0),
                    "created_token_count": int(record.get("created_token_count", 0) or 0),
                    "last_activity_token_count": int(record.get("last_activity_token_count", 0) or 0),
                    "last_cooling_token_count": int(record.get("last_cooling_token_count", 0) or 0),
                    "age_tokens": int(max(0, current_token - int(record.get("created_token_count", current_token)))),
                    "activity_age_tokens": int(max(0, current_token - int(record.get("last_activity_token_count", current_token)))),
                    "last_credit_score": float(record.get("last_credit_score", 0.0) or 0.0),
                    "last_penalty_score": float(record.get("last_penalty_score", 0.0) or 0.0),
                    "last_forgiveness_score": float(record.get("last_forgiveness_score", 0.0) or 0.0),
                    "last_penalty_reason": str(record.get("last_penalty_reason", "")),
                    "last_trajectory_event_type": str(record.get("last_trajectory_event_type", "")),
                    "last_trajectory_event_score": float(record.get("last_trajectory_event_score", 0.0) or 0.0),
                    "last_trajectory_event_at": str(record.get("last_trajectory_event_at", "")),
                    "split_generation": int(record.get("split_generation", 0) or 0),
                    "split_parent_record_id": str(record.get("split_parent_record_id", "")),
                    "split_group_id": str(record.get("split_group_id", "")),
                    "split_branch": str(record.get("split_branch", "")),
                    "remerge_events": int(record.get("remerge_events", 0) or 0),
                    "last_split_at": str(record.get("last_split_at", "")),
                    "last_remerged_at": str(record.get("last_remerged_at", "")),
                    "last_aggregated_at": str(record.get("last_aggregated_at", "")),
                    "last_cooled_at": str(record.get("last_cooled_at", "")),
                    "last_evaluated_at": str(record.get("last_evaluated_at", "")),
                    "last_evaluated_query_text": str(record.get("last_evaluated_query_text", "")),
                }
            )
        return {
            "enabled": True,
            "record_count": int(len(records)),
            "credited_record_count": int(credited_count),
            "penalized_record_count": int(penalized_count),
            "forgiven_record_count": int(forgiven_count),
            "aggregated_record_count": int(aggregated_count),
            "aggregate_occurrence_count": int(aggregate_occurrence_count),
            "trajectory_state_counts": {str(state): int(count) for state, count in dict(trajectory_state_counts).items()},
            "max_grounded_family_summary_score": float(
                max((self._grounded_family_summary_score(record) for record in records), default=0.0)
            ),
            "cooled_record_count_total": int(self._delayed_consequence_cooled_total),
            "retired_record_count_total": int(self._delayed_consequence_retired_total),
            "compacted_record_count_total": int(self._delayed_consequence_compacted_total),
            "split_record_count_total": int(self._delayed_consequence_split_total),
            "remerged_record_count_total": int(self._delayed_consequence_remerged_total),
            "pending_record_count": int(
                sum(
                    1
                    for record in records
                    if int(record.get("credit_events", 0) or 0) <= 0
                    and int(record.get("penalty_events", 0) or 0) <= 0
                    and int(record.get("forgiveness_events", 0) or 0) <= 0
                )
            ),
            "recent_records": recent_records,
        }

    def _normalize_delayed_consequence_record(self, item: Any) -> dict[str, Any] | None:
        if not isinstance(item, Mapping):
            return None

        def _safe_float(raw_value: Any) -> float:
            try:
                return max(0.0, min(1.0, float(raw_value)))
            except (TypeError, ValueError):
                return 0.0

        def _safe_int(raw_value: Any) -> int:
            try:
                return max(0, int(raw_value))
            except (TypeError, ValueError):
                return 0

        def _safe_total(raw_value: Any) -> float:
            try:
                return max(0.0, float(raw_value))
            except (TypeError, ValueError):
                return 0.0

        def _safe_signed(raw_value: Any, *, limit: float = DEFAULT_DELAYED_CONSEQUENCE_TRAJECTORY_SCORE_LIMIT) -> float:
            try:
                return max(-float(limit), min(float(limit), float(raw_value)))
            except (TypeError, ValueError):
                return 0.0

        current_token = int(self._trainer.token_count)
        query_text = self._normalize_action_text(item.get("query_text", ""))
        if not query_text:
            return None
        query_examples: list[str] = []
        seen_query_examples: set[str] = set()
        raw_query_examples = item.get("query_examples")
        query_example_values = [query_text]
        if isinstance(raw_query_examples, Sequence) and not isinstance(raw_query_examples, (str, bytes)):
            query_example_values.extend(list(raw_query_examples))
        for raw_value in query_example_values:
            text = self._normalize_action_text(raw_value)
            if not text:
                continue
            key = text.lower()
            if key in seen_query_examples:
                continue
            seen_query_examples.add(key)
            query_examples.append(text)
            if len(query_examples) >= DEFAULT_DELAYED_CONSEQUENCE_AGGREGATION_QUERY_EXAMPLE_LIMIT:
                break
        query_terms = [
            _canonical_provider_term(term)
            for term in list(
                item.get("query_terms")
                or [term for text in query_examples for term in self._consequence_query_terms(text)]
                or self._consequence_query_terms(query_text)
            )
            if _canonical_provider_term(term)
        ]

        def _weight_map(raw_value: Any, *, lowercase_keys: bool) -> dict[str, float]:
            weighted: dict[str, float] = {}
            if not isinstance(raw_value, Mapping):
                return weighted
            for raw_name, raw_weight in raw_value.items():
                name = " ".join(str(raw_name).split()).strip()
                if not name:
                    continue
                key = name.lower() if lowercase_keys else name
                weight = _safe_float(raw_weight)
                if weight <= 0.0:
                    continue
                weighted[key] = weight
            return weighted

        source_weights = _weight_map(item.get("source_weights"), lowercase_keys=False)
        provider_weights = _weight_map(item.get("provider_weights"), lowercase_keys=True)
        if not source_weights and not provider_weights:
            return None
        baseline_query_score = _safe_float(item.get("baseline_query_score", 0.0))
        best_query_score = max(baseline_query_score, _safe_float(item.get("best_query_score", baseline_query_score)))
        baseline_grounded_fraction = _safe_float(item.get("baseline_grounded_fraction", 0.0))
        best_grounded_fraction = max(
            baseline_grounded_fraction,
            _safe_float(item.get("best_grounded_fraction", baseline_grounded_fraction)),
        )
        credit_events = _safe_int(item.get("credit_events", 0))
        penalty_events = _safe_int(item.get("penalty_events", 0))
        forgiveness_events = _safe_int(item.get("forgiveness_events", 0))
        trajectory_credit_total = max(
            _safe_total(item.get("trajectory_credit_total", 0.0)),
            _safe_total(item.get("resolved_improvement", 0.0)) if credit_events > 0 else 0.0,
        )
        trajectory_penalty_total = max(
            _safe_total(item.get("trajectory_penalty_total", 0.0)),
            _safe_total(item.get("max_regression", 0.0)) if penalty_events > 0 else 0.0,
            _safe_total(item.get("unresolved_penalty_balance", 0.0)) if penalty_events > 0 else 0.0,
        )
        trajectory_forgiveness_total = max(
            _safe_total(item.get("trajectory_forgiveness_total", 0.0)),
            _safe_total(item.get("last_forgiveness_score", 0.0)) if forgiveness_events > 0 else 0.0,
        )
        trajectory_event_count = max(
            _safe_int(item.get("trajectory_event_count", 0)),
            credit_events + penalty_events + forgiveness_events,
        )
        raw_trajectory_net = item.get(
            "trajectory_net_score",
            trajectory_credit_total + trajectory_forgiveness_total - trajectory_penalty_total,
        )
        trajectory_net_score = _safe_signed(raw_trajectory_net)
        trajectory_peak_score = max(
            trajectory_net_score,
            _safe_signed(item.get("trajectory_peak_score", trajectory_net_score)),
        )
        trajectory_floor_score = min(
            trajectory_net_score,
            _safe_signed(item.get("trajectory_floor_score", trajectory_net_score)),
        )
        split_branch = self._normalize_action_text(item.get("split_branch", "")).lower()
        supportive_query_examples: list[str] = []
        seen_supportive_examples: set[str] = set()
        raw_supportive_examples = item.get("supportive_query_examples")
        if isinstance(raw_supportive_examples, Sequence) and not isinstance(raw_supportive_examples, (str, bytes)):
            for raw_value in list(raw_supportive_examples):
                text = self._normalize_action_text(raw_value)
                if not text:
                    continue
                key = text.lower()
                if key in seen_supportive_examples:
                    continue
                seen_supportive_examples.add(key)
                supportive_query_examples.append(text)
                if len(supportive_query_examples) >= DEFAULT_DELAYED_CONSEQUENCE_AGGREGATION_QUERY_EXAMPLE_LIMIT:
                    break
        if split_branch == "supportive" and not supportive_query_examples:
            supportive_query_examples = list(query_examples)[:DEFAULT_DELAYED_CONSEQUENCE_AGGREGATION_QUERY_EXAMPLE_LIMIT]
        adverse_query_examples: list[str] = []
        seen_adverse_examples: set[str] = set()
        raw_adverse_examples = item.get("adverse_query_examples")
        if isinstance(raw_adverse_examples, Sequence) and not isinstance(raw_adverse_examples, (str, bytes)):
            for raw_value in list(raw_adverse_examples):
                text = self._normalize_action_text(raw_value)
                if not text:
                    continue
                key = text.lower()
                if key in seen_adverse_examples:
                    continue
                seen_adverse_examples.add(key)
                adverse_query_examples.append(text)
                if len(adverse_query_examples) >= DEFAULT_DELAYED_CONSEQUENCE_AGGREGATION_QUERY_EXAMPLE_LIMIT:
                    break
        if split_branch == "adverse" and not adverse_query_examples:
            adverse_query_examples = list(query_examples)[:DEFAULT_DELAYED_CONSEQUENCE_AGGREGATION_QUERY_EXAMPLE_LIMIT]
        supportive_occurrence_count = max(
            len(supportive_query_examples),
            _safe_int(item.get("supportive_occurrence_count", 0)),
        )
        adverse_occurrence_count = max(
            len(adverse_query_examples),
            _safe_int(item.get("adverse_occurrence_count", 0)),
        )
        remerge_events = _safe_int(item.get("remerge_events", 0))
        return {
            "record_id": self._normalize_action_text(item.get("record_id", "")) or str(uuid4()),
            "created_at": str(item.get("created_at") or datetime.now(timezone.utc).isoformat()),
            "created_token_count": _safe_int(item.get("created_token_count", current_token)),
            "origin": self._normalize_action_text(item.get("origin", "response_selected_evidence")) or "response_selected_evidence",
            "query_text": query_text,
            "query_examples": list(query_examples)[:DEFAULT_DELAYED_CONSEQUENCE_AGGREGATION_QUERY_EXAMPLE_LIMIT],
            "query_terms": list(dict.fromkeys(query_terms))[:DEFAULT_DELAYED_CONSEQUENCE_AGGREGATION_TERM_LIMIT],
            "baseline_query_score": float(baseline_query_score),
            "best_query_score": float(best_query_score),
            "baseline_grounded_fraction": float(baseline_grounded_fraction),
            "best_grounded_fraction": float(best_grounded_fraction),
            "outcome_score": float(_safe_float(item.get("outcome_score", 0.0))),
            "source_weights": dict(source_weights),
            "provider_weights": dict(provider_weights),
            "credit_events": int(credit_events),
            "penalty_events": int(penalty_events),
            "forgiveness_events": int(forgiveness_events),
            "cooling_events": _safe_int(item.get("cooling_events", 0)),
            "aggregate_count": max(1, _safe_int(item.get("aggregate_count", 1))),
            "aggregation_events": _safe_int(item.get("aggregation_events", 0)),
            "supportive_query_examples": list(supportive_query_examples),
            "adverse_query_examples": list(adverse_query_examples),
            "supportive_occurrence_count": int(supportive_occurrence_count),
            "adverse_occurrence_count": int(adverse_occurrence_count),
            "trajectory_credit_total": float(trajectory_credit_total),
            "trajectory_penalty_total": float(trajectory_penalty_total),
            "trajectory_forgiveness_total": float(trajectory_forgiveness_total),
            "trajectory_event_count": int(trajectory_event_count),
            "trajectory_net_score": float(trajectory_net_score),
            "trajectory_recent_delta_ema": float(_safe_signed(item.get("trajectory_recent_delta_ema", 0.0), limit=1.0)),
            "trajectory_peak_score": float(trajectory_peak_score),
            "trajectory_floor_score": float(trajectory_floor_score),
            "unresolved_penalty_balance": float(_safe_float(item.get("unresolved_penalty_balance", 0.0))),
            "resolved_improvement": float(_safe_float(item.get("resolved_improvement", 0.0))),
            "max_regression": float(_safe_float(item.get("max_regression", 0.0))),
            "max_contradiction_signal": float(_safe_float(item.get("max_contradiction_signal", 0.0))),
            "cumulative_cooling_delta": float(_safe_float(item.get("cumulative_cooling_delta", 0.0))),
            "last_match_score": float(_safe_float(item.get("last_match_score", 0.0))),
            "last_credit_score": float(_safe_float(item.get("last_credit_score", 0.0))),
            "last_penalty_score": float(_safe_float(item.get("last_penalty_score", 0.0))),
            "last_forgiveness_score": float(_safe_float(item.get("last_forgiveness_score", 0.0))),
            "last_penalty_reason": self._normalize_action_text(item.get("last_penalty_reason", "")),
            "last_activity_token_count": _safe_int(item.get("last_activity_token_count", current_token)),
            "last_evaluated_token_count": _safe_int(item.get("last_evaluated_token_count", current_token)),
            "last_cooling_token_count": _safe_int(item.get("last_cooling_token_count", current_token)),
            "last_credit_token_count": _safe_int(item.get("last_credit_token_count", 0)),
            "last_penalty_token_count": _safe_int(item.get("last_penalty_token_count", 0)),
            "last_forgiveness_token_count": _safe_int(item.get("last_forgiveness_token_count", 0)),
            "last_trajectory_event_type": self._normalize_action_text(item.get("last_trajectory_event_type", "")),
            "last_trajectory_event_score": float(_safe_total(item.get("last_trajectory_event_score", 0.0))),
            "last_trajectory_event_at": self._normalize_action_text(item.get("last_trajectory_event_at", "")),
            "last_trajectory_event_token_count": _safe_int(item.get("last_trajectory_event_token_count", 0)),
            "split_generation": _safe_int(item.get("split_generation", 0)),
            "split_parent_record_id": self._normalize_action_text(item.get("split_parent_record_id", "")),
            "split_group_id": self._normalize_action_text(item.get("split_group_id", "")),
            "split_branch": split_branch,
            "remerge_events": int(remerge_events),
            "last_split_at": self._normalize_action_text(item.get("last_split_at", "")),
            "last_remerged_at": self._normalize_action_text(item.get("last_remerged_at", "")),
            "last_aggregated_at": self._normalize_action_text(item.get("last_aggregated_at", "")),
            "last_cooled_at": self._normalize_action_text(item.get("last_cooled_at", "")),
            "last_evaluated_at": self._normalize_action_text(item.get("last_evaluated_at", "")),
            "last_evaluated_query_text": self._normalize_action_text(item.get("last_evaluated_query_text", "")),
        }


    def restore_state(self, terminus_state: dict[str, Any]) -> None:
        """Restore consequence records and totals from checkpoint state."""
        self._delayed_consequence_records = deque(
            (
                item
                for item in (
                    self._normalize_delayed_consequence_record(raw_item)
                    for raw_item in list(terminus_state.get("delayed_consequence_records") or [])
                )
                if item is not None
            ),
            maxlen=DEFAULT_DELAYED_CONSEQUENCE_RECORDS,
        )
        self._delayed_consequence_cooled_total = _restore_non_negative_int(
            terminus_state, "delayed_consequence_cooled_total",
        )
        self._delayed_consequence_retired_total = _restore_non_negative_int(
            terminus_state, "delayed_consequence_retired_total",
        )
        self._delayed_consequence_compacted_total = _restore_non_negative_int(
            terminus_state, "delayed_consequence_compacted_total",
        )
        self._delayed_consequence_split_total = _restore_non_negative_int(
            terminus_state, "delayed_consequence_split_total",
        )
        self._delayed_consequence_remerged_total = _restore_non_negative_int(
            terminus_state, "delayed_consequence_remerged_total",
        )

DelayedConsequenceMixin = DelayedConsequenceTracker
