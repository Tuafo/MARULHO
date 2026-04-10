"""Tests for TurboQuantPrototypeStore (§6.2)."""

from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from hecsn.retrieval.turboquant_store import TurboQuantPrototypeStore


class TestTurboQuantInit(unittest.TestCase):
    def test_dimensions(self) -> None:
        store = TurboQuantPrototypeStore(n_cols=100, dim=256)
        self.assertEqual(store.n_cols, 100)
        self.assertEqual(store.dim, 256)
        self.assertEqual(store._fp32.shape, (100, 256))

    def test_rotation_orthogonal(self) -> None:
        store = TurboQuantPrototypeStore(n_cols=10, dim=64)
        R = store.rotation
        eye = torch.eye(64)
        product = R @ R.T
        self.assertTrue(torch.allclose(product, eye, atol=1e-5))


class TestCompression(unittest.TestCase):
    def setUp(self) -> None:
        self.store = TurboQuantPrototypeStore(n_cols=50, dim=128)
        # Set random prototypes
        protos = F.normalize(torch.randn(50, 128), dim=1)
        self.store.set_all(protos)

    def test_compress_all_returns_count(self) -> None:
        count = self.store.compress_all()
        self.assertEqual(count, 50)

    def test_second_compress_returns_zero(self) -> None:
        self.store.compress_all()
        count = self.store.compress_all()
        self.assertEqual(count, 0)

    def test_update_marks_dirty(self) -> None:
        self.store.compress_all()
        self.store.update(5, torch.randn(128))
        self.assertTrue(self.store._dirty[5].item())
        count = self.store.compress_all()
        self.assertEqual(count, 1)

    def test_codes_in_range(self) -> None:
        self.store.compress_all()
        self.assertTrue((self.store._codes >= 0).all())
        self.assertTrue((self.store._codes < self.store.n_levels).all())


class TestRouting(unittest.TestCase):
    def setUp(self) -> None:
        self.store = TurboQuantPrototypeStore(n_cols=100, dim=256)
        protos = F.normalize(torch.randn(100, 256), dim=1)
        self.store.set_all(protos)
        self.store.compress_all()

    def test_route_returns_k(self) -> None:
        query = torch.randn(256)
        indices, scores = self.store.route(query, k=10)
        self.assertEqual(indices.shape, (10,))
        self.assertEqual(scores.shape, (10,))

    def test_route_scores_descending(self) -> None:
        query = torch.randn(256)
        _, scores = self.store.route(query, k=20)
        for i in range(len(scores) - 1):
            self.assertGreaterEqual(scores[i].item(), scores[i + 1].item())

    def test_exact_routing_matches_brute_force(self) -> None:
        query = F.normalize(torch.randn(256), dim=0)
        idx_exact, scores_exact = self.store.route_exact(query, k=5)
        # Brute force
        all_scores = torch.mv(self.store._fp32, query)
        _, bf_idx = torch.topk(all_scores, 5)
        self.assertTrue(torch.equal(idx_exact, bf_idx))

    def test_approximate_top1_recall(self) -> None:
        """Top-1 recall should be high for quantised routing."""
        hits = 0
        n_trials = 50
        for _ in range(n_trials):
            query = F.normalize(torch.randn(256), dim=0)
            exact_idx, _ = self.store.route_exact(query, k=1)
            approx_idx, _ = self.store.route(query, k=5)
            if exact_idx[0] in approx_idx:
                hits += 1
        recall = hits / n_trials
        self.assertGreater(recall, 0.5, f"Top-1 recall@5 = {recall:.2f}, expected > 0.5")


class TestCosineAccuracy(unittest.TestCase):
    def test_cosine_accuracy_reasonable(self) -> None:
        store = TurboQuantPrototypeStore(n_cols=50, dim=128, bits=3)
        protos = F.normalize(torch.randn(50, 128), dim=1)
        store.set_all(protos)
        acc = store.cosine_accuracy(n_queries=20)
        self.assertGreater(acc, 0.8, f"Cosine accuracy = {acc:.4f}, expected > 0.8")

    def test_higher_bits_better_accuracy(self) -> None:
        protos = F.normalize(torch.randn(30, 64), dim=1)
        shared_rotation = TurboQuantPrototypeStore(n_cols=1, dim=64).rotation.clone()

        store_3 = TurboQuantPrototypeStore(n_cols=30, dim=64, bits=3)
        store_3.rotation = shared_rotation.clone()
        store_3.set_all(protos)

        store_8 = TurboQuantPrototypeStore(n_cols=30, dim=64, bits=8)
        store_8.rotation = shared_rotation.clone()
        store_8.set_all(protos)

        acc_3 = store_3.cosine_accuracy(n_queries=20)
        acc_8 = store_8.cosine_accuracy(n_queries=20)
        self.assertGreaterEqual(acc_8, acc_3 - 0.05)


class TestMemory(unittest.TestCase):
    def test_compression_ratio(self) -> None:
        store = TurboQuantPrototypeStore(n_cols=1000, dim=256, bits=3)
        mem = store.memory_bytes()
        self.assertGreater(mem["compression_ratio"], 1.0)
        # At 3-bit, compressed should be much smaller than FP32
        self.assertLess(mem["compressed"], mem["fp32"])


class TestSerialization(unittest.TestCase):
    def test_state_dict_roundtrip(self) -> None:
        store = TurboQuantPrototypeStore(n_cols=20, dim=64)
        protos = F.normalize(torch.randn(20, 64), dim=1)
        store.set_all(protos)
        store.compress_all()

        state = store.state_dict()
        store2 = TurboQuantPrototypeStore(n_cols=20, dim=64)
        store2.load_state_dict(state)

        self.assertTrue(torch.allclose(store._fp32, store2._fp32))
        self.assertTrue(torch.equal(store._codes, store2._codes))


if __name__ == "__main__":
    unittest.main()
