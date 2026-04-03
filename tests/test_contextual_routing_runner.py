from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from hecsn.training import contextual_routing_runner
from hecsn.training.contextual_routing_runner import (
    bank_polysemy_probe,
    leave_one_out_family_accuracy,
    mean_cross_distance,
    mean_pairwise_distance,
)


class ContextualRoutingRunnerMetricTests(unittest.TestCase):
    def test_leave_one_out_family_accuracy_separates_families(self) -> None:
        families = {
            "river": [
                torch.tensor([1.0, 0.0, 0.0]),
                torch.tensor([0.95, 0.05, 0.0]),
                torch.tensor([0.9, 0.1, 0.0]),
            ],
            "money": [
                torch.tensor([0.0, 1.0, 0.0]),
                torch.tensor([0.0, 0.95, 0.05]),
                torch.tensor([0.1, 0.9, 0.0]),
            ],
        }

        accuracy = leave_one_out_family_accuracy(families)

        self.assertGreaterEqual(accuracy, 1.0)

    def test_cross_distance_exceeds_within_distance_for_separated_families(self) -> None:
        river = [
            torch.tensor([1.0, 0.0, 0.0]),
            torch.tensor([0.95, 0.05, 0.0]),
            torch.tensor([0.9, 0.1, 0.0]),
        ]
        money = [
            torch.tensor([0.0, 1.0, 0.0]),
            torch.tensor([0.0, 0.95, 0.05]),
            torch.tensor([0.1, 0.9, 0.0]),
        ]

        within_distance = (mean_pairwise_distance(river) + mean_pairwise_distance(money)) / 2.0
        cross_distance = mean_cross_distance(river, money)

        self.assertGreater(cross_distance, within_distance)

    def test_bank_polysemy_probe_uses_signature_readout(self) -> None:
        def fake_text_examples(text: str, encoder: object, window_size: int) -> list[tuple[str, torch.Tensor]]:
            del encoder, window_size
            if text == "bank":
                return [
                    ("b", torch.tensor([1.0, 0.0, 0.0])),
                    ("ba", torch.tensor([2.0, 0.0, 0.0])),
                    ("ban", torch.tensor([3.0, 0.0, 0.0])),
                    ("bank", torch.tensor([4.0, 0.0, 0.0])),
                ]
            if "river" in text:
                return [(text.strip(), torch.tensor([0.0, 1.0, 0.0]))]
            if any(token in text for token in ("money", "cash", "finance", "credit", "loan", "account", "deposit", "branch", "savings")):
                return [(text.strip(), torch.tensor([0.0, 2.0, 0.0]))]
            return [(text.strip(), torch.tensor([0.0, 0.0, 1.0]))]

        class FakeTrainer:
            def __init__(self) -> None:
                self.family = "river"

            def prime_context_with_signatures(
                self,
                patterns: list[torch.Tensor],
                update_weights: bool = False,
                *,
                blend_context_state: bool = False,
                readout_mode: str = "softmax",
            ) -> None:
                del update_weights, blend_context_state, readout_mode
                marker = float(patterns[0][1].item())
                self.family = "river" if marker < 1.5 else "money"

            def context_state(self) -> torch.Tensor:
                if self.family == "river":
                    return torch.tensor([1.0, 0.0, 0.0])
                return torch.tensor([0.0, 1.0, 0.0])

            def contextual_signature_for_pattern(
                self,
                pattern_vec: torch.Tensor,
                *,
                blend_context_state: bool = False,
                readout_mode: str = "softmax",
            ) -> torch.Tensor:
                del blend_context_state, readout_mode
                prefix_step = float(pattern_vec[0].item())
                if self.family == "river":
                    base = torch.tensor([0.60, 0.30, 0.10]) + prefix_step * torch.tensor([0.00, 0.01, -0.01])
                else:
                    base = torch.tensor([0.60, 0.10, 0.30]) + prefix_step * torch.tensor([0.00, -0.01, 0.01])
                base = torch.clamp(base, min=0.0)
                return base / (base.sum() + 1e-8)

            def prime_context(self, *args: object, **kwargs: object) -> None:
                raise AssertionError("bank_polysemy_probe should not use sparse prime_context")

            def contextual_winner_for_pattern(self, pattern_vec: torch.Tensor) -> int:
                del pattern_vec
                raise AssertionError("bank_polysemy_probe should not use sparse winners")

            def contextual_assembly_for_pattern(self, pattern_vec: torch.Tensor) -> torch.Tensor:
                del pattern_vec
                raise AssertionError("bank_polysemy_probe should not use sparse assemblies")

        with patch.object(contextual_routing_runner, "text_examples", side_effect=fake_text_examples):
            probe = bank_polysemy_probe(FakeTrainer(), object(), 6)

        self.assertGreaterEqual(probe["family_classification_accuracy"], 1.0)
        self.assertGreater(probe["signature_separation_margin"], 0.0)
        self.assertEqual(probe["winner_sequence_difference_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
