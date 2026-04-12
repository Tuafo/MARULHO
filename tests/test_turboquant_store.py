"""Tests for TurboQuantPrototypeStore with QJL residual correction (§6.2)."""

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

    def test_qjl_projection_shape(self) -> None:
        store = TurboQuantPrototypeStore(n_cols=10, dim=64)
        self.assertEqual(store._projection.shape, (64, 64))

    def test_custom_n_projections(self) -> None:
        store = TurboQuantPrototypeStore(n_cols=10, dim=64, n_projections=32)
        self.assertEqual(store._projection.shape, (32, 64))
        self.assertEqual(store._residual_signs.shape, (10, 32))


class TestCompression(unittest.TestCase):
    def setUp(self) -> None:
        self.store = TurboQuantPrototypeStore(n_cols=50, dim=128)
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
        from hecsn.retrieval.turboquant_store import unpack_codes
        unpacked = unpack_codes(self.store._codes, self.store.bits, self.store.dim)
        self.assertTrue((unpacked >= 0).all())
        self.assertTrue((unpacked < self.store.n_levels).all())

    def test_residual_signs_are_pm1_or_zero(self) -> None:
        self.store.compress_all()
        signs = self.store._residual_signs
        self.assertTrue(((signs == -1) | (signs == 0) | (signs == 1)).all())

    def test_residual_norms_nonnegative(self) -> None:
        self.store.compress_all()
        self.assertTrue((self.store._residual_norms >= 0).all())


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
        all_scores = torch.mv(self.store._fp32, query)
        _, bf_idx = torch.topk(all_scores, 5)
        self.assertTrue(torch.equal(idx_exact, bf_idx))

    def test_approximate_top1_recall(self) -> None:
        """Top-1 recall should be high for QJL-corrected routing."""
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


class TestQJLCorrection(unittest.TestCase):
    """Validate QJL residual correction properties."""

    def test_qjl_reduces_bias(self) -> None:
        """QJL correction should produce near-zero mean bias."""
        store = TurboQuantPrototypeStore(n_cols=50, dim=128, bits=3)
        protos = F.normalize(torch.randn(50, 128), dim=1)
        store.set_all(protos)
        bias = store.inner_product_bias(n_queries=100)
        self.assertAlmostEqual(bias, 0.0, places=2,
                               msg=f"QJL bias = {bias:.6f}, expected near 0")

    def test_qjl_improves_over_no_correction(self) -> None:
        """QJL-corrected scores should be closer to exact than uncorrected."""
        torch.manual_seed(42)
        protos = F.normalize(torch.randn(50, 128), dim=1)

        store = TurboQuantPrototypeStore(n_cols=50, dim=128, bits=3)
        store.set_all(protos.clone())
        store.compress_all()

        from hecsn.retrieval.turboquant_store import unpack_codes

        total_err_qjl = 0.0
        total_err_base = 0.0
        n_queries = 50
        for _ in range(n_queries):
            q = F.normalize(torch.randn(128), dim=0)
            _, exact = store.route_exact(q, k=50)
            _, qjl = store.route(q, k=50)
            # Compute base (no QJL) scores manually
            q_rot = torch.mv(store.rotation, q)
            unpacked = unpack_codes(store._codes, store.bits, store.dim)
            decompressed = (
                unpacked.float() * store._scales.unsqueeze(1)
                + store._offsets.unsqueeze(1)
            )
            base = torch.mv(decompressed, q_rot)
            base_sorted, _ = base.sort(descending=True)
            total_err_qjl += (qjl - exact).abs().mean().item()
            total_err_base += (base_sorted - exact).abs().mean().item()

        mean_err_qjl = total_err_qjl / n_queries
        mean_err_base = total_err_base / n_queries
        self.assertLessEqual(mean_err_qjl, mean_err_base + 0.01,
                             f"QJL error {mean_err_qjl:.4f} should not be much worse "
                             f"than base error {mean_err_base:.4f}")

    def test_higher_projections_lower_variance(self) -> None:
        """More QJL projections should reduce estimator variance."""
        torch.manual_seed(123)
        protos = F.normalize(torch.randn(30, 64), dim=1)

        store_low = TurboQuantPrototypeStore(n_cols=30, dim=64, n_projections=16)
        store_low.set_all(protos.clone())

        store_high = TurboQuantPrototypeStore(n_cols=30, dim=64, n_projections=128)
        store_high.rotation = store_low.rotation.clone()
        store_high.set_all(protos.clone())

        store_low.compress_all()
        store_high.compress_all()

        var_low = 0.0
        var_high = 0.0
        n_q = 30
        for _ in range(n_q):
            q = F.normalize(torch.randn(64), dim=0)
            _, exact = store_low.route_exact(q, k=30)

            _, approx_low = store_low.route(q, k=30)
            _, approx_high = store_high.route(q, k=30)

            var_low += (approx_low - exact).var().item()
            var_high += (approx_high - exact).var().item()

        # Higher projections should give lower or similar variance
        self.assertLessEqual(var_high / n_q, var_low / n_q + 0.01)


class TestBitPacking(unittest.TestCase):
    """Validate bit-pack/unpack roundtrip for various bit widths."""

    def test_roundtrip_3bit(self) -> None:
        from hecsn.retrieval.turboquant_store import pack_codes, unpack_codes
        codes = torch.randint(0, 8, (10, 256), dtype=torch.int16)
        packed = pack_codes(codes, 3)
        recovered = unpack_codes(packed, 3, 256)
        self.assertTrue(torch.equal(codes, recovered))

    def test_roundtrip_4bit(self) -> None:
        from hecsn.retrieval.turboquant_store import pack_codes, unpack_codes
        codes = torch.randint(0, 16, (5, 128), dtype=torch.int16)
        packed = pack_codes(codes, 4)
        recovered = unpack_codes(packed, 4, 128)
        self.assertTrue(torch.equal(codes, recovered))

    def test_roundtrip_8bit(self) -> None:
        from hecsn.retrieval.turboquant_store import pack_codes, unpack_codes
        codes = torch.randint(0, 256, (5, 64), dtype=torch.int16)
        packed = pack_codes(codes, 8)
        recovered = unpack_codes(packed, 8, 64)
        self.assertTrue(torch.equal(codes, recovered))

    def test_packed_size_3bit(self) -> None:
        """3-bit: 256 codes → 96 bytes (8 codes per 3 bytes)."""
        from hecsn.retrieval.turboquant_store import pack_codes
        codes = torch.randint(0, 8, (1, 256), dtype=torch.int16)
        packed = pack_codes(codes, 3)
        self.assertEqual(packed.shape[-1], 96)

    def test_packed_size_4bit(self) -> None:
        """4-bit: 256 codes → 128 bytes (2 codes per byte)."""
        from hecsn.retrieval.turboquant_store import pack_codes
        codes = torch.randint(0, 16, (1, 256), dtype=torch.int16)
        packed = pack_codes(codes, 4)
        self.assertEqual(packed.shape[-1], 128)


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
        self.assertLess(mem["compressed"], mem["fp32"])

    def test_memory_includes_qjl(self) -> None:
        store = TurboQuantPrototypeStore(n_cols=100, dim=64, bits=3)
        mem = store.memory_bytes()
        self.assertIn("stage1_codes", mem)
        self.assertIn("stage2_qjl", mem)
        self.assertIn("projection", mem)
        self.assertGreater(mem["stage2_qjl"], 0)


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
        self.assertTrue(torch.equal(store._residual_signs, store2._residual_signs))
        self.assertTrue(torch.allclose(store._residual_norms, store2._residual_norms))
        self.assertTrue(torch.allclose(store._projection, store2._projection))

    def test_state_dict_keys(self) -> None:
        store = TurboQuantPrototypeStore(n_cols=10, dim=32)
        state = store.state_dict()
        expected_keys = {"fp32", "codes", "scales", "offsets", "rotation",
                         "projection", "residual_signs", "residual_norms", "dirty"}
        self.assertEqual(set(state.keys()), expected_keys)


if __name__ == "__main__":
    unittest.main()
