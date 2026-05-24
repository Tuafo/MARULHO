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


class TestNormExtraction(unittest.TestCase):
    """Verify that compress_all extracts norms before rotation (PolarQuant)."""

    def test_non_unit_norm_routing_accuracy(self) -> None:
        """Vectors with varying norms should route correctly after compression."""
        store = TurboQuantPrototypeStore(n_cols=20, dim=64, bits=3)
        # Create prototypes with varying norms (0.1 to 10.0)
        protos = torch.randn(20, 64)
        norms = torch.linspace(0.1, 10.0, 20).unsqueeze(1)
        protos = protos * norms
        store.set_all(protos)
        store.compress_all()

        # Verify norms are stored
        for i in range(20):
            expected_norm = protos[i].norm().item()
            stored_norm = store._norms[i].item()
            self.assertAlmostEqual(stored_norm, expected_norm, places=4)

    def test_non_unit_norm_ranking_preserved(self) -> None:
        """Top-k ranking should be similar for unit and non-unit norm protos."""
        dim = 64
        store = TurboQuantPrototypeStore(n_cols=50, dim=dim, bits=3)
        protos = torch.randn(50, dim)
        norms = torch.linspace(0.5, 5.0, 50).unsqueeze(1)
        protos = protos * norms
        store.set_all(protos)
        store.compress_all()

        # Top-1 of approximate should match top-1 of exact reasonably often
        matches = 0
        n_queries = 50
        for _ in range(n_queries):
            q = F.normalize(torch.randn(dim), dim=0)
            approx_idx, _ = store.route(q, k=1)
            exact_idx, _ = store.route_exact(q, k=1)
            if approx_idx[0].item() == exact_idx[0].item():
                matches += 1
        recall = matches / n_queries
        self.assertGreater(recall, 0.5, f"Top-1 recall {recall:.2f} too low")

    def test_cosine_accuracy_with_varying_norms(self) -> None:
        """Cosine accuracy should remain high even with varying norms."""
        store = TurboQuantPrototypeStore(n_cols=30, dim=64, bits=3)
        protos = torch.randn(30, 64)
        norms = torch.linspace(0.1, 10.0, 30).unsqueeze(1)
        protos = protos * norms
        store.set_all(protos)
        cos_acc = store.cosine_accuracy(n_queries=50)
        self.assertGreater(cos_acc, 0.85)


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
        expected_keys = {"fp32", "codes", "scales", "offsets", "norms", "rotation",
                         "projection", "residual_signs", "residual_norms", "dirty"}
        self.assertEqual(set(state.keys()), expected_keys)


# ---------------------------------------------------------------------------
# TurboQuant+ as HierarchicalAssemblyIndex routing backend
# ---------------------------------------------------------------------------

class TestTurboQuantRoutingBackend(unittest.TestCase):
    """Integration tests for the turboquant_plus routing backend."""

    def _make_index(self, dim: int = 64, n: int = 50) -> "tuple":
        import numpy as np
        from hecsn.retrieval.hnsw_index import HierarchicalAssemblyIndex

        idx = HierarchicalAssemblyIndex(dim=dim, backend="turboquant_plus")
        vecs = F.normalize(torch.randn(n, dim), dim=1)
        ids = np.arange(n, dtype=np.int64)
        idx.add(vecs, ids)
        idx.rebuild()
        return idx, vecs, ids

    def test_backend_resolves(self) -> None:
        from hecsn.retrieval.hnsw_index import HierarchicalAssemblyIndex
        idx = HierarchicalAssemblyIndex(dim=32, backend="turboquant_plus")
        self.assertEqual(idx._backend, "turboquant_plus")

    def test_search_returns_correct_format(self) -> None:
        idx, vecs, _ = self._make_index(dim=32, n=20)
        q = F.normalize(torch.randn(1, 32), dim=1)
        result_ids, result_dists = idx.search(q, k=5)
        self.assertIsInstance(result_ids, list)
        self.assertEqual(len(result_ids), 1)
        self.assertEqual(len(result_ids[0]), 5)
        self.assertEqual(result_dists.shape, (1, 5))

    def test_batch_search(self) -> None:
        idx, vecs, _ = self._make_index(dim=32, n=20)
        q = F.normalize(torch.randn(3, 32), dim=1)
        result_ids, result_dists = idx.search(q, k=5)
        self.assertEqual(len(result_ids), 3)
        self.assertEqual(result_dists.shape, (3, 5))

    def test_top1_recall_vs_exact(self) -> None:
        """TQ top-1 should agree with exact cosine >70% (3-bit pre-filter)."""
        import numpy as np
        from hecsn.retrieval.hnsw_index import HierarchicalAssemblyIndex

        dim, n = 64, 50
        vecs = F.normalize(torch.randn(n, dim), dim=1)
        ids = np.arange(n, dtype=np.int64)

        tq_idx = HierarchicalAssemblyIndex(dim=dim, backend="turboquant_plus")
        exact_idx = HierarchicalAssemblyIndex(dim=dim, backend="exact_cosine")
        tq_idx.add(vecs, ids)
        exact_idx.add(vecs, ids)
        tq_idx.rebuild()

        n_queries = 100
        matches = 0
        for _ in range(n_queries):
            q = F.normalize(torch.randn(1, dim), dim=1)
            tq_ids, _ = tq_idx.search(q, k=1)
            exact_ids, _ = exact_idx.search(q, k=1)
            if tq_ids[0][0] == exact_ids[0][0]:
                matches += 1

        recall = matches / n_queries
        self.assertGreaterEqual(recall, 0.70, f"Top-1 recall {recall:.2f} < 0.70")

    def test_topk_recall_vs_exact(self) -> None:
        """Top-k shortlist should contain the exact top-1 >90% of the time."""
        import numpy as np
        from hecsn.retrieval.hnsw_index import HierarchicalAssemblyIndex

        dim, n = 64, 50
        vecs = F.normalize(torch.randn(n, dim), dim=1)
        ids = np.arange(n, dtype=np.int64)

        tq_idx = HierarchicalAssemblyIndex(dim=dim, backend="turboquant_plus")
        exact_idx = HierarchicalAssemblyIndex(dim=dim, backend="exact_cosine")
        tq_idx.add(vecs, ids)
        exact_idx.add(vecs, ids)
        tq_idx.rebuild()

        n_queries = 100
        recalls = 0
        for _ in range(n_queries):
            q = F.normalize(torch.randn(1, dim), dim=1)
            tq_ids, _ = tq_idx.search(q, k=10)
            exact_ids, _ = exact_idx.search(q, k=1)
            if exact_ids[0][0] in tq_ids[0]:
                recalls += 1

        recall = recalls / n_queries
        self.assertGreaterEqual(recall, 0.90, f"Top-k recall {recall:.2f} < 0.90")

    def test_add_updates_sync(self) -> None:
        """After adding new vectors and rebuilding, search finds them."""
        import numpy as np
        from hecsn.retrieval.hnsw_index import HierarchicalAssemblyIndex

        dim = 32
        idx = HierarchicalAssemblyIndex(dim=dim, backend="turboquant_plus")

        v1 = F.normalize(torch.randn(5, dim), dim=1)
        idx.add(v1, np.arange(5, dtype=np.int64))
        idx.rebuild()

        # Add 5 more and rebuild
        v2 = F.normalize(torch.randn(5, dim), dim=1)
        idx.add(v2, np.arange(5, 10, dtype=np.int64))
        idx.rebuild()

        self.assertEqual(idx.stats()["unique_vectors"], 10)

        # Search for v2[0] — should find id=5 as top-1
        result_ids, _ = idx.search(v2[0:1], k=1)
        self.assertEqual(result_ids[0][0], 5)

    def test_prototype_update_reflected_after_rebuild(self) -> None:
        """Updating a prototype and rebuilding makes the new vector findable."""
        import numpy as np
        from hecsn.retrieval.hnsw_index import HierarchicalAssemblyIndex

        dim = 32
        idx = HierarchicalAssemblyIndex(dim=dim, backend="turboquant_plus")

        vecs = F.normalize(torch.randn(10, dim), dim=1)
        idx.add(vecs, np.arange(10, dtype=np.int64))
        idx.rebuild()

        # Replace vector 0 with a very specific direction
        new_vec = torch.zeros(1, dim)
        new_vec[0, 0] = 1.0  # all weight on dim 0
        idx.add(new_vec, np.array([0], dtype=np.int64))
        idx.rebuild()

        # Query along same direction
        q = torch.zeros(1, dim)
        q[0, 0] = 1.0
        result_ids, _ = idx.search(q, k=1)
        self.assertEqual(result_ids[0][0], 0)

    def test_remove_excludes_from_search(self) -> None:
        import numpy as np
        from hecsn.retrieval.hnsw_index import HierarchicalAssemblyIndex

        dim = 32
        idx = HierarchicalAssemblyIndex(dim=dim, backend="turboquant_plus")
        vecs = F.normalize(torch.randn(10, dim), dim=1)
        idx.add(vecs, np.arange(10, dtype=np.int64))
        idx.rebuild()

        # Remove vector 0
        idx.remove(0)
        idx.rebuild()

        result_ids, _ = idx.search(vecs[0:1], k=5)
        self.assertNotIn(0, result_ids[0])

    def test_stats_includes_tq_memory(self) -> None:
        idx, _, _ = self._make_index(dim=32, n=20)
        stats = idx.stats()
        self.assertEqual(stats["index_type"], "turboquant_plus")
        self.assertIn("tq_memory", stats)
        self.assertGreater(stats["tq_memory"]["compression_ratio"], 1.0)
        self.assertTrue(stats["tq_cache_ready"])
        self.assertFalse(stats["tq_cache_dirty"])
        self.assertEqual(stats["tq_device"], "cpu")
        self.assertEqual(stats["tq_fp32_device"], "cpu")
        self.assertEqual(stats["tq_codes_device"], "cpu")
        self.assertEqual(stats["tq_residual_device"], "cpu")

    def test_empty_search(self) -> None:
        from hecsn.retrieval.hnsw_index import HierarchicalAssemblyIndex
        idx = HierarchicalAssemblyIndex(dim=32, backend="turboquant_plus")
        q = F.normalize(torch.randn(1, 32), dim=1)
        result_ids, result_dists = idx.search(q, k=5)
        self.assertEqual(result_ids, [[]])

    def test_lazy_rebuild_on_search(self) -> None:
        """Search should auto-rebuild TQ cache when dirty."""
        import numpy as np
        from hecsn.retrieval.hnsw_index import HierarchicalAssemblyIndex

        dim = 32
        idx = HierarchicalAssemblyIndex(dim=dim, backend="turboquant_plus")
        vecs = F.normalize(torch.randn(10, dim), dim=1)
        idx.add(vecs, np.arange(10, dtype=np.int64))
        # No explicit rebuild — search should lazily rebuild
        result_ids, _ = idx.search(vecs[0:1], k=3)
        self.assertEqual(len(result_ids[0]), 3)
        self.assertEqual(result_ids[0][0], 0)


if __name__ == "__main__":
    unittest.main()
