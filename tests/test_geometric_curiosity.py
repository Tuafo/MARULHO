from __future__ import annotations

import unittest

import torch

from hecsn.core import AbstractionLayer
from hecsn.semantics import GeometricCuriosityController


class GeometricCuriosityControllerTests(unittest.TestCase):
    def test_focus_plan_uses_neighbor_lexicon_terms(self) -> None:
        layer = AbstractionLayer(
            n_columns=4,
            n_concepts=3,
            device=torch.device("cpu"),
        )
        layer.feedforward = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.9, 0.1, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
        layer.concept_stability = torch.tensor([0.2, 0.9, 0.8], dtype=torch.float32)
        layer.concept_certainty = torch.tensor([0.1, 0.9, 0.8], dtype=torch.float32)
        layer.slow_var = torch.tensor([0.9, 0.1, 0.1], dtype=torch.float32)
        layer.slow_state = torch.tensor([0.4, 0.3, 0.3], dtype=torch.float32)

        controller = GeometricCuriosityController(layer, gap_threshold=0.05)
        controller.lexicon = {
            1: ["river", "stream", "water"],
            2: ["finance", "credit"],
        }

        focus_plan = controller.focus_plan()

        self.assertIsNotNone(focus_plan)
        assert focus_plan is not None
        self.assertEqual(focus_plan["planner_mode"], "geometric_abstraction_gap_focus")
        self.assertTrue(focus_plan["retrieval_queries"])
        self.assertIn("river", " ".join(focus_plan["retrieval_queries"]))
        self.assertTrue(focus_plan["geometric_gaps"])

    def test_state_round_trip_preserves_lexicon(self) -> None:
        controller = GeometricCuriosityController(None)
        controller.lexicon = {1: ["river", "stream"]}

        restored = GeometricCuriosityController.from_state_dict(None, controller.state_dict())

        self.assertEqual(restored.lexicon, controller.lexicon)


if __name__ == "__main__":
    unittest.main()
