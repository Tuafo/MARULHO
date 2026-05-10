from __future__ import annotations

import unittest
from types import SimpleNamespace

from hecsn.service.brain_runtime import BrainRuntime, BrainRuntimeMixin
from hecsn.service.runtime_sources import _BrainSourceRuntime


class _FakeRuntimeState:
    def __init__(self) -> None:
        self.mutated = 0

    def mark_mutated(self) -> None:
        self.mutated += 1


class _FakeManager:
    def __init__(self) -> None:
        self._brain_config = {"tick_tokens": 8}
        self._brain_source_utility: dict[str, dict[str, object]] = {}
        self._runtime_state = _FakeRuntimeState()
        self._trainer = SimpleNamespace(token_count=16)

    def _autonomy_focus_plan_locked(self):
        return {"query_terms": ["cats", "mice"]}

    def _background_focus_terms_locked(self, *, focus_plan=None):
        return ["cats", "mice"]

    def _background_focus_overlap_locked(self, focus_terms, grounded_observation):
        return 0.5


class BrainRuntimeSeamTests(unittest.TestCase):
    def test_alias_points_to_constructed_module(self) -> None:
        self.assertIs(BrainRuntimeMixin, BrainRuntime)

    def test_update_background_source_utility_uses_brain_runtime_interface(self) -> None:
        module = BrainRuntime(_FakeManager())
        runtime = _BrainSourceRuntime(
            spec={
                "name": "science_source",
                "source": "science.txt",
                "topic_terms": ["cats", "mice"],
            },
            stream=iter([]),
        )
        runtime.last_semantic_match = 0.6

        module._update_background_source_utility_locked(
            runtime=runtime,
            grounded_observation={"content": "cats and mice", "grounding_signal": 0.75},
            total_trained=4,
        )

        entry = module._background_source_utility_entry_locked(runtime)
        self.assertEqual(module._runtime_state.mutated, 1)
        self.assertEqual(entry["attempts"], 1)
        self.assertEqual(entry["selections"], 1)
        self.assertEqual(entry["tokens_trained_total"], 4)
        self.assertAlmostEqual(float(entry["utility_ema"]), 0.6)
        self.assertTrue(entry["last_selected_at"])


if __name__ == "__main__":
    unittest.main()
