from __future__ import annotations

from collections import deque
from copy import deepcopy
import unittest
from types import SimpleNamespace

from marulho.service.delayed_consequence import (
    DEFAULT_DELAYED_CONSEQUENCE_RECORDS,
    DelayedConsequenceDependencies,
    DelayedConsequenceTracker,
)


class _FakeRuntimeState:
    def __init__(self) -> None:
        self.mutated = 0

    def mark_mutated(self) -> None:
        self.mutated += 1


class _FakeManager:
    def __init__(self) -> None:
        self._trainer = SimpleNamespace(token_count=128)
        self._runtime_state = _FakeRuntimeState()
        self._delayed_consequence_records = deque(maxlen=DEFAULT_DELAYED_CONSEQUENCE_RECORDS)
        self._delayed_consequence_cooled_total = 0
        self._delayed_consequence_retired_total = 0
        self._delayed_consequence_compacted_total = 0
        self._delayed_consequence_split_total = 0
        self._delayed_consequence_remerged_total = 0
        self._brain_source_utility: dict[str, dict[str, object]] = {}
        self._brain_source_runtimes = []
        self._brain_config = {"tick_tokens": 8, "autonomy": {"provider_curriculum": {}}}
        self.events: list[dict[str, object]] = []

    @staticmethod
    def _normalize_action_text(value):
        return " ".join(str(value).split()).strip()

    @staticmethod
    def _source_text_overlap(left: str, right: str) -> float:
        left_terms = {term for term in left.lower().split() if term}
        right_terms = {term for term in right.lower().split() if term}
        if not left_terms or not right_terms:
            return 0.0
        return float(len(left_terms & right_terms)) / float(max(1, min(len(left_terms), len(right_terms))))

    def _record_brain_event_locked(self, event: dict[str, object]) -> None:
        self.events.append(event)

    def _action_record_relevance_score_locked(self, record, query_text: str) -> float:
        return 0.0

    def _background_focus_terms_locked(self, **kwargs):
        return []

    def _background_source_utility_entry_locked(self, runtime):
        return self._brain_source_utility.setdefault(runtime.name, {})

    def _brain_source_semantic_match_locked(self, runtime, focus_terms=None) -> float:
        return 0.0

    @staticmethod
    def _normalize_provider_curriculum(value):
        return dict(value or {}) if isinstance(value, dict) else {}

    def _recent_relevant_action_records_locked(self, query_text: str, **kwargs):
        return []

    @staticmethod
    def _selected_evidence_weight_map(response, **kwargs):
        return {}


def _delayed_consequence_tracker(fake: _FakeManager | None = None) -> DelayedConsequenceTracker:
    fake = fake or _FakeManager()
    return DelayedConsequenceTracker(
        DelayedConsequenceDependencies(
            action_record_relevance_score=fake._action_record_relevance_score_locked,
            background_focus_terms=fake._background_focus_terms_locked,
            background_source_utility_entry=fake._background_source_utility_entry_locked,
            brain_config=lambda: fake._brain_config,
            brain_source_runtimes=lambda: fake._brain_source_runtimes,
            brain_source_semantic_match=fake._brain_source_semantic_match_locked,
            normalize_action_text=fake._normalize_action_text,
            normalize_provider_curriculum=fake._normalize_provider_curriculum,
            recent_relevant_action_records=fake._recent_relevant_action_records_locked,
            record_brain_event=fake._record_brain_event_locked,
            runtime_state=fake._runtime_state,
            selected_evidence_weight_map=fake._selected_evidence_weight_map,
            source_text_overlap=fake._source_text_overlap,
            trainer=lambda: fake._trainer,
        )
    )


class DelayedConsequenceTrackerSeamTests(unittest.TestCase):
    def _build_record(self):
        tracker = _delayed_consequence_tracker()
        return tracker._normalize_delayed_consequence_record(
            {
                "record_id": "record-1",
                "created_at": "2026-05-10T00:00:00Z",
                "created_token_count": 12,
                "origin": "response_selected_evidence",
                "query_text": "cats chase mice",
                "query_examples": ["cats chase mice", "felines chase mice"],
                "query_terms": ["cats", "mice"],
                "baseline_query_score": 0.4,
                "best_query_score": 0.8,
                "baseline_grounded_fraction": 0.35,
                "best_grounded_fraction": 0.75,
                "outcome_score": 0.6,
                "source_weights": {"science_source": 1.0},
                "provider_weights": {"web": 1.0},
                "credit_events": 2,
                "penalty_events": 1,
                "forgiveness_events": 0,
                "aggregate_count": 1,
                "supportive_query_examples": ["cats chase mice"],
                "adverse_query_examples": ["dogs bark loudly"],
                "supportive_occurrence_count": 1,
                "adverse_occurrence_count": 1,
                "trajectory_credit_total": 0.6,
                "trajectory_penalty_total": 0.25,
                "trajectory_forgiveness_total": 0.0,
                "trajectory_event_count": 2,
                "trajectory_net_score": 0.35,
                "trajectory_recent_delta_ema": 0.15,
                "trajectory_peak_score": 0.35,
                "trajectory_floor_score": -0.25,
                "unresolved_penalty_balance": 0.3,
                "resolved_improvement": 0.2,
                "max_regression": 0.1,
                "max_contradiction_signal": 0.2,
                "cumulative_cooling_delta": 0.0,
                "last_match_score": 0.4,
                "last_credit_score": 0.6,
                "last_penalty_score": 0.25,
                "last_forgiveness_score": 0.0,
                "last_penalty_reason": "regression_decay",
                "last_activity_token_count": 12,
                "last_evaluated_token_count": 12,
                "last_cooling_token_count": 12,
                "last_credit_token_count": 12,
                "last_penalty_token_count": 12,
                "last_forgiveness_token_count": 0,
                "last_trajectory_event_type": "credit",
                "last_trajectory_event_score": 0.6,
                "last_trajectory_event_at": "2026-05-10T00:00:00Z",
                "last_trajectory_event_token_count": 12,
            }
        )

    def test_state_machines_split_remerge_compact_and_cool(self) -> None:
        tracker = _delayed_consequence_tracker()
        base_record = self._build_record()
        self.assertIsNotNone(base_record)
        assert base_record is not None

        inserted = tracker._upsert_delayed_consequence_record_locked(base_record)
        merged = tracker._upsert_delayed_consequence_record_locked(deepcopy(base_record))

        self.assertEqual(inserted["record_id"], merged["record_id"])
        self.assertEqual(len(tracker._delayed_consequence_records), 1)
        self.assertEqual(tracker._delayed_consequence_compacted_total, 1)
        self.assertEqual(int(merged["aggregate_count"]), 2)

        split_result = tracker._split_divergent_delayed_consequence_families_locked()
        self.assertEqual(split_result["split_records"], 1)
        self.assertEqual(tracker._delayed_consequence_split_total, 1)
        self.assertEqual(len(tracker._delayed_consequence_records), 2)

        adverse_record = next(record for record in tracker._delayed_consequence_records if record["split_branch"] == "adverse")
        adverse_record["supportive_occurrence_count"] = 1
        adverse_record["trajectory_recent_delta_ema"] = 0.3

        remerge_result = tracker._remerge_converged_delayed_consequence_families_locked()
        self.assertEqual(remerge_result["remerged_records"], 1)
        self.assertEqual(tracker._delayed_consequence_remerged_total, 1)
        self.assertEqual(len(tracker._delayed_consequence_records), 1)

        tracker._trainer.token_count = 8192
        stale_record = tracker._normalize_delayed_consequence_record(
            {
                "record_id": "record-stale",
                "created_at": "2026-01-01T00:00:00Z",
                "created_token_count": 0,
                "origin": "response_selected_evidence",
                "query_text": "stale record",
                "source_weights": {"science_source": 1.0},
                "provider_weights": {"web": 1.0},
                "unresolved_penalty_balance": 1.0,
                "last_activity_token_count": 0,
                "last_cooling_token_count": 0,
            }
        )
        self.assertIsNotNone(stale_record)
        assert stale_record is not None
        tracker._delayed_consequence_records = deque(
            [stale_record],
            maxlen=DEFAULT_DELAYED_CONSEQUENCE_RECORDS,
        )

        cool_result = tracker._cool_delayed_consequence_records_locked()
        self.assertEqual(cool_result["cooled_records"], 1)
        self.assertEqual(cool_result["retired_records"], 1)
        self.assertGreater(tracker._delayed_consequence_cooled_total, 0)
        self.assertGreater(tracker._delayed_consequence_retired_total, 0)
        self.assertEqual(len(tracker._delayed_consequence_records), 0)


if __name__ == "__main__":
    unittest.main()
