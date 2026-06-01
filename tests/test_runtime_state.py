"""Tests for the private Runtime State module."""

from __future__ import annotations

from pathlib import Path
import unittest

from hecsn.service.runtime_state import RuntimeState


class RuntimeStateTests(unittest.TestCase):
    def test_initial_state_defaults(self) -> None:
        state = RuntimeState()

        self.assertFalse(state.dirty_state)
        self.assertEqual(state.state_revision, 0)
        self.assertIsNone(state.last_event)
        self.assertEqual(state.recent_events, [])
        self.assertEqual(state.snapshot()["recent_events"], [])

    def test_mark_mutated_sets_dirty_and_increments_revision_once(self) -> None:
        state = RuntimeState()

        state.mark_mutated()

        self.assertTrue(state.dirty_state)
        self.assertEqual(state.state_revision, 1)

    def test_mutation_summary_reports_dirty_state_and_revision(self) -> None:
        state = RuntimeState()

        state.mark_mutated()

        self.assertEqual(
            state.mutation_summary(),
            {
                "dirty_state": True,
                "state_revision": 1,
            },
        )

    def test_dirty_without_revision_preserves_revision(self) -> None:
        state = RuntimeState()

        state.mark_mutated()
        state.mark_dirty_without_revision()

        self.assertTrue(state.dirty_state)
        self.assertEqual(state.state_revision, 1)

    def test_mark_clean_clears_dirty_without_incrementing_revision(self) -> None:
        state = RuntimeState()

        state.mark_mutated()
        state.mark_clean()

        self.assertFalse(state.dirty_state)
        self.assertEqual(state.state_revision, 1)

    def test_restore_clean_clears_dirty_and_increments_revision_once(self) -> None:
        state = RuntimeState()

        state.restore_clean()

        self.assertFalse(state.dirty_state)
        self.assertEqual(state.state_revision, 1)

    def test_commit_restored_revision_advances_beyond_live_and_persisted_histories(self) -> None:
        state = RuntimeState()
        state.state_revision = 4
        state.mark_dirty_without_revision()

        state.commit_restored_revision(9)

        self.assertFalse(state.dirty_state)
        self.assertEqual(state.state_revision, 10)

    def test_hydrate_persisted_revision_restores_exact_clean_revision(self) -> None:
        state = RuntimeState()
        state.mark_mutated()

        state.hydrate_persisted_revision(7)

        self.assertFalse(state.dirty_state)
        self.assertEqual(state.state_revision, 7)

    def test_record_event_normalizes_payload_and_keeps_defensive_copies(self) -> None:
        state = RuntimeState()
        payload = {
            1: {
                "path": Path("reports/runtime/event.json"),
                "items": ["alpha", Path("nested/item.txt")],
            },
            "type": "brain_event_recorded",
        }

        recorded = state.record_event(payload)
        payload[1]["items"].append("mutated")
        recorded["1"]["items"].append("mutated_again")
        last_event = state.last_event
        snapshot_last_event = state.snapshot()["last_event"]

        self.assertEqual(Path(recorded["1"]["path"]).as_posix(), "reports/runtime/event.json")
        self.assertEqual(recorded["1"]["items"][0], "alpha")
        self.assertEqual(Path(recorded["1"]["items"][1]).as_posix(), "nested/item.txt")
        self.assertEqual(recorded["1"]["items"][2], "mutated_again")
        self.assertIsNotNone(last_event)
        self.assertIsNotNone(snapshot_last_event)
        assert last_event is not None
        assert snapshot_last_event is not None
        self.assertEqual(Path(last_event["1"]["path"]).as_posix(), "reports/runtime/event.json")
        self.assertEqual(last_event["1"]["items"], ["alpha", str(Path("nested/item.txt"))])
        self.assertEqual(Path(snapshot_last_event["1"]["path"]).as_posix(), "reports/runtime/event.json")

    def test_record_event_history_is_newest_first_and_bounded_to_sixteen(self) -> None:
        state = RuntimeState()

        for index in range(20):
            state.record_event({"type": f"event-{index}", "index": index})

        self.assertEqual(len(state.recent_events), 16)
        self.assertEqual(state.recent_events[0]["type"], "event-19")
        self.assertEqual(state.recent_events[-1]["type"], "event-4")
        last_event = state.last_event
        self.assertIsNotNone(last_event)
        assert last_event is not None
        self.assertEqual(last_event["type"], "event-19")
        self.assertEqual(state.snapshot()["recent_events"][0]["type"], "event-19")

    def test_restore_event_history_normalizes_payload_and_preserves_last_event_head(self) -> None:
        state = RuntimeState(history_limit=2)
        recent_events = [
            {"type": "older", "path": Path("older/event.json")},
            {"type": "oldest"},
            {"type": "trimmed"},
        ]
        last_event = {"type": "latest", "path": Path("latest/event.json")}

        state.restore_event_history(last_event=last_event, recent_events=recent_events)
        recent_events[0]["type"] = "mutated"
        last_event["type"] = "mutated"

        restored_events = state.recent_events
        restored_last_event = state.last_event

        self.assertEqual(len(restored_events), 2)
        self.assertEqual(restored_events[0]["type"], "latest")
        self.assertEqual(Path(restored_events[0]["path"]).as_posix(), "latest/event.json")
        self.assertEqual(restored_events[1]["type"], "oldest")
        self.assertIsNotNone(restored_last_event)
        assert restored_last_event is not None
        self.assertEqual(restored_last_event, restored_events[0])
