"""Tests for GPU-native IVF routing (§6.1)."""

from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from hecsn.retrieval.ivf_router import IVFRouter, benchmark_routing


class TestIVFRouterInit(unittest.TestCase):
    def test_untrained_raises(self) -> None:
        router = IVFRouter(dim=64)
        with self.assertRaises(RuntimeError):
            router.search(torch.randn(64))

    def test_train_sets_centroids(self) -> None:
        protos = F.normalize(torch.randn(200, 64), dim=1)
        router = IVFRouter(dim=64)
        router.train(protos)
        self.assertGreater(router.n_cells, 0)
        self.assertTrue(router._is_trained)


class TestIVFRouterSearch(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(42)
        self.protos = F.normalize(torch.randn(500, 128), dim=1)
        self.router = IVFRouter(dim=128, nprobe=8)
        self.router.train(self.protos)

    def test_search_single_query(self) -> None:
        q = torch.randn(128)
        indices, scores = self.router.search(q, k=10)
        self.assertEqual(indices.shape, (10,))
        self.assertEqual(scores.shape, (10,))

    def test_search_batch_query(self) -> None:
        q = torch.randn(5, 128)
        indices, scores = self.router.search(q, k=10)
        self.assertEqual(indices.shape, (5, 10))
        self.assertEqual(scores.shape, (5, 10))

    def test_scores_descending(self) -> None:
        q = torch.randn(128)
        _, scores = self.router.search(q, k=20)
        for i in range(len(scores) - 1):
            self.assertGreaterEqual(scores[i].item(), scores[i + 1].item())

    def test_flat_matches_brute_force(self) -> None:
        q = F.normalize(torch.randn(128), dim=0)
        flat_idx, _ = self.router.search_flat(q, k=5)
        # Brute force
        all_sims = self.protos @ q
        _, bf_idx = all_sims.topk(5)
        self.assertTrue(torch.equal(flat_idx, bf_idx))

    def test_k_larger_than_n(self) -> None:
        small_protos = F.normalize(torch.randn(10, 128), dim=1)
        router = IVFRouter(dim=128)
        router.train(small_protos)
        q = torch.randn(128)
        indices, scores = router.search(q, k=20)
        self.assertEqual(indices.shape, (10,))


class TestIVFRecall(unittest.TestCase):
    def test_recall_at_32_reasonable(self) -> None:
        """IVF with nprobe=12 on 500 prototypes should have decent recall."""
        torch.manual_seed(42)
        protos = F.normalize(torch.randn(500, 128), dim=1)
        router = IVFRouter(dim=128, nprobe=12)
        router.train(protos)
        recall = router.recall_at_k(n_queries=50, k=32)
        self.assertGreater(recall, 0.55, f"recall@32 = {recall:.3f}, expected > 0.55")

    def test_higher_nprobe_better_recall(self) -> None:
        torch.manual_seed(42)
        protos = F.normalize(torch.randn(500, 128), dim=1)

        router_low = IVFRouter(dim=128, nprobe=2)
        router_low.train(protos)

        router_high = IVFRouter(dim=128, nprobe=16)
        router_high.train(protos)

        recall_low = router_low.recall_at_k(n_queries=30, k=32)
        recall_high = router_high.recall_at_k(n_queries=30, k=32)
        self.assertGreaterEqual(recall_high, recall_low - 0.05)


class TestBenchmarkRouting(unittest.TestCase):
    def test_flat_benchmark(self) -> None:
        result = benchmark_routing(n_cols=100, dim=64, n_queries=10, device="cpu")
        self.assertEqual(result.method, "flat")
        self.assertEqual(result.n_cols, 100)
        self.assertGreater(result.ms_per_query, 0)
        self.assertEqual(result.recall_at_k, 1.0)

    def test_benchmark_result_fields(self) -> None:
        result = benchmark_routing(n_cols=50, dim=32, n_queries=5, device="cpu")
        self.assertIsInstance(result.ms_per_query, float)
        self.assertIsInstance(result.device, str)


class TestNCells(unittest.TestCase):
    def test_auto_ncells(self) -> None:
        protos = F.normalize(torch.randn(1000, 64), dim=1)
        router = IVFRouter(dim=64)
        router.train(protos)
        # sqrt(1000) ≈ 31.6, so ~32 cells
        self.assertGreater(router.n_cells, 20)
        self.assertLess(router.n_cells, 50)

    def test_manual_ncells(self) -> None:
        protos = F.normalize(torch.randn(200, 64), dim=1)
        router = IVFRouter(dim=64, n_cells=10)
        router.train(protos)
        self.assertEqual(router.n_cells, 10)


if __name__ == "__main__":
    unittest.main()
