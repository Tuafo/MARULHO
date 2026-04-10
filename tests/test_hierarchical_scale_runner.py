from __future__ import annotations

import types
import unittest

import torch

from hecsn.training.hierarchical_scale_runner import evaluate_index_integrity


class _DuplicateReturningIndex:
    def __init__(self, prototypes: torch.Tensor) -> None:
        self._prototypes = prototypes

    def search(self, query: torch.Tensor, k: int = 5):  # noqa: ANN001 - test double
        del k
        if torch.allclose(query, self._prototypes[0:1]) or torch.allclose(query, self._prototypes[1:2]):
            return [[1]], torch.zeros((1, 1), dtype=torch.float32).numpy()
        return [[2]], torch.zeros((1, 1), dtype=torch.float32).numpy()


class _OffsetReturningIndex:
    def __init__(self, prototypes: torch.Tensor) -> None:
        self._prototypes = prototypes

    def search(self, query: torch.Tensor, k: int = 5):  # noqa: ANN001 - test double
        del k
        if torch.allclose(query, self._prototypes[0:1]):
            return [[1]], torch.zeros((1, 1), dtype=torch.float32).numpy()
        if torch.allclose(query, self._prototypes[1:2]):
            return [[0]], torch.zeros((1, 1), dtype=torch.float32).numpy()
        return [[1]], torch.zeros((1, 1), dtype=torch.float32).numpy()


class HierarchicalScaleIntegrityTests(unittest.TestCase):
    def test_evaluate_index_integrity_treats_duplicate_equivalent_candidates_as_reachable(self) -> None:
        prototypes = torch.tensor(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=torch.float32,
        )
        trainer = types.SimpleNamespace(
            config=types.SimpleNamespace(n_columns=3),
            model=types.SimpleNamespace(
                competitive=types.SimpleNamespace(prototypes=prototypes),
                hnsw_index=_DuplicateReturningIndex(prototypes),
            ),
        )

        metrics = evaluate_index_integrity(trainer, k=1)

        self.assertAlmostEqual(float(metrics["strict_unreachable_fraction"]), 1.0 / 3.0, places=6)
        self.assertAlmostEqual(float(metrics["unreachable_fraction"]), 0.0, places=6)
        self.assertEqual(int(metrics["equivalent_recoveries"]), 1)
        self.assertAlmostEqual(float(metrics["self_recall"]), 1.0, places=6)

    def test_evaluate_index_integrity_keeps_non_equivalent_candidates_unreachable(self) -> None:
        prototypes = torch.tensor(
            [
                [1.0, 0.0],
                [0.8, 0.2],
                [0.0, 1.0],
            ],
            dtype=torch.float32,
        )
        trainer = types.SimpleNamespace(
            config=types.SimpleNamespace(n_columns=3),
            model=types.SimpleNamespace(
                competitive=types.SimpleNamespace(prototypes=prototypes),
                hnsw_index=_OffsetReturningIndex(prototypes),
            ),
        )

        metrics = evaluate_index_integrity(trainer, k=1, equivalent_similarity_min=0.99999)

        self.assertGreater(float(metrics["strict_unreachable_fraction"]), 0.0)
        self.assertGreater(float(metrics["unreachable_fraction"]), 0.0)
        self.assertEqual(int(metrics["equivalent_recoveries"]), 0)
        self.assertLess(float(metrics["self_recall"]), 1.0)


if __name__ == "__main__":
    unittest.main()
