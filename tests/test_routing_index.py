from __future__ import annotations

import unittest

import numpy as np
import torch

from marulho.config.model_config import MarulhoConfig
from marulho.retrieval.hnsw_index import HierarchicalAssemblyIndex, ShardedHierarchicalAssemblyIndex
from marulho.training.model import MarulhoModel


class RoutingIndexTests(unittest.TestCase):
    def test_torch_topk_index_returns_exact_topk(self) -> None:
        index = HierarchicalAssemblyIndex(
            dim=3,
            rebuild_threshold=2,
            device=torch.device("cpu"),
            backend="torch_topk",
        )
        vectors = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )
        ids = np.array([0, 1, 2], dtype=np.int64)
        index.add(vectors, ids)

        found_ids, dists = index.search(torch.tensor([[0.9, 0.1, 0.0]], dtype=torch.float32), k=2)

        self.assertEqual(found_ids[0], [0, 1])
        self.assertEqual(dists.shape, (1, 2))
        stats = index.stats()
        self.assertEqual(stats["index_type"], "torch_topk")
        self.assertTrue(stats["torch_cache_ready"])
        self.assertFalse(stats["torch_cache_dirty"])
        self.assertEqual(stats["torch_vector_cache_device"], "cpu")
        self.assertEqual(stats["torch_id_cache_device"], "cpu")
        self.assertEqual(stats["torch_vector_cache_count"], 3)
        self.assertFalse(stats["torch_cache_cuda"])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA device required")
    def test_auto_cuda_index_reports_actual_cache_devices(self) -> None:
        index = HierarchicalAssemblyIndex(
            dim=3,
            rebuild_threshold=2,
            device=torch.device("cuda"),
            backend="auto",
        )
        vectors = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
            device=torch.device("cuda"),
        )
        ids = np.array([0, 1, 2], dtype=np.int64)
        index.add(vectors, ids)

        query = torch.tensor(
            [[0.9, 0.1, 0.0]],
            dtype=torch.float32,
            device=torch.device("cuda"),
        )
        found_ids, _ = index.search(query, k=2)

        stats = index.stats()
        self.assertEqual(found_ids[0], [0, 1])
        self.assertEqual(stats["index_type"], "torch_topk")
        self.assertEqual(stats["search_device"], "cuda")
        self.assertTrue(str(stats["torch_vector_cache_device"]).startswith("cuda"))
        self.assertTrue(str(stats["torch_id_cache_device"]).startswith("cuda"))
        self.assertTrue(stats["torch_cache_cuda"])

    def test_sharded_torch_topk_merges_global_topk(self) -> None:
        index = ShardedHierarchicalAssemblyIndex(
            dim=2,
            n_shards=2,
            rebuild_threshold=2,
            shard_candidate_factor=2,
            device=torch.device("cpu"),
            backend="torch_topk",
        )
        vectors = torch.tensor(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.0, 1.0],
                [0.1, 0.9],
            ],
            dtype=torch.float32,
        )
        ids = np.array([0, 1, 2, 3], dtype=np.int64)
        index.add(vectors, ids)

        found_ids, _ = index.search(torch.tensor([[0.95, 0.05]], dtype=torch.float32), k=2)

        self.assertEqual(found_ids[0], [0, 1])
        stats = index.stats()
        self.assertEqual(stats["index_type"], "sharded_torch_topk")
        self.assertEqual(stats["per_shard_search_device"], ["cpu", "cpu"])
        self.assertEqual(stats["per_shard_torch_vector_cache_device"], ["cpu", "cpu"])
        self.assertEqual(stats["per_shard_torch_id_cache_device"], ["cpu", "cpu"])
        self.assertEqual(stats["per_shard_torch_cache_ready"], [True, True])

    def test_sharded_torch_topk_search_tensors_merges_global_topk_on_device(self) -> None:
        index = ShardedHierarchicalAssemblyIndex(
            dim=2,
            n_shards=2,
            rebuild_threshold=2,
            shard_candidate_factor=2,
            device=torch.device("cpu"),
            backend="torch_topk",
        )
        vectors = torch.tensor(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.0, 1.0],
                [0.1, 0.9],
            ],
            dtype=torch.float32,
        )
        ids = np.array([0, 1, 2, 3], dtype=np.int64)
        index.add(vectors, ids)

        found_ids, distances = index.search_tensors(
            torch.tensor([[0.95, 0.05]], dtype=torch.float32),
            k=2,
        )

        self.assertEqual(found_ids.device.type, "cpu")
        self.assertEqual(distances.device.type, "cpu")
        self.assertEqual(found_ids.tolist()[0], [0, 1])
        self.assertEqual(tuple(found_ids.shape), (1, 2))
        self.assertEqual(tuple(distances.shape), (1, 2))
        stats = index.stats()
        self.assertEqual(stats["last_search_mode"], "tensor")
        self.assertEqual(stats["tensor_search_count"], 1)
        self.assertEqual(stats["list_search_count"], 0)
        self.assertTrue(stats["merged_torch_search_enabled"])
        self.assertTrue(stats["merged_torch_cache_ready"])
        self.assertEqual(stats["merged_torch_vector_cache_count"], 4)

    def test_merged_torch_cache_matches_per_shard_tensor_search(self) -> None:
        vectors = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.9, 0.1, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.9, 0.1],
                [0.0, 0.0, 1.0],
                [0.1, 0.0, 0.9],
            ],
            dtype=torch.float32,
        )
        ids = np.arange(6, dtype=np.int64)
        merged = ShardedHierarchicalAssemblyIndex(
            dim=3,
            n_shards=3,
            device=torch.device("cpu"),
            backend="torch_topk",
            merge_torch_shards=True,
        )
        per_shard = ShardedHierarchicalAssemblyIndex(
            dim=3,
            n_shards=3,
            device=torch.device("cpu"),
            backend="torch_topk",
            merge_torch_shards=False,
        )
        merged.add(vectors, ids)
        per_shard.add(vectors, ids)
        queries = torch.tensor(
            [
                [0.95, 0.05, 0.0],
                [0.0, 0.85, 0.15],
                [0.05, 0.0, 0.95],
            ],
            dtype=torch.float32,
        )

        merged_ids, merged_dists = merged.search_tensors(queries, k=3)
        shard_ids, shard_dists = per_shard.search_tensors(queries, k=3)

        self.assertTrue(torch.equal(merged_ids, shard_ids))
        self.assertTrue(torch.allclose(merged_dists, shard_dists, atol=1e-6))

    def test_merged_torch_cache_invalidates_after_update_and_remove(self) -> None:
        index = ShardedHierarchicalAssemblyIndex(
            dim=2,
            n_shards=2,
            device=torch.device("cpu"),
            backend="torch_topk",
        )
        index.add(
            torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
            np.array([0, 1], dtype=np.int64),
        )
        query = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
        first_ids, _ = index.search_tensors(query, k=1)
        self.assertEqual(first_ids.tolist(), [[0]])
        self.assertTrue(index.stats()["merged_torch_cache_ready"])

        index.add(
            torch.tensor([[1.0, 0.0]], dtype=torch.float32),
            np.array([1], dtype=np.int64),
        )
        self.assertTrue(index.stats()["merged_torch_cache_dirty"])
        updated_ids, _ = index.search_tensors(query, k=2)
        self.assertEqual(set(updated_ids.tolist()[0]), {0, 1})

        index.remove(0)
        self.assertTrue(index.stats()["merged_torch_cache_dirty"])
        remaining_ids, _ = index.search_tensors(query, k=1)
        self.assertEqual(remaining_ids.tolist(), [[1]])

    def test_model_runtime_scope_reports_requested_torch_topk_backend(self) -> None:
        cfg = MarulhoConfig(
            n_columns=4,
            column_latent_dim=8,
            routing_index_mode="torch_topk",
        )
        model = MarulhoModel(cfg)
        scope = model.runtime_scope_report()

        self.assertEqual(scope["routing_backend_mode"], "torch_topk")
        self.assertEqual(scope["routing_index"]["index_type"], "torch_topk")

    def test_model_can_disable_merged_torch_routing_cache(self) -> None:
        cfg = MarulhoConfig(
            n_columns=8,
            column_latent_dim=4,
            routing_index_mode="torch_topk",
            routing_shards=2,
            merge_torch_routing_shards=False,
            device="cpu",
        )
        model = MarulhoModel(cfg)
        model.hnsw_index.search_tensors(torch.rand(1, cfg.column_latent_dim), k=2)
        routing_stats = model.runtime_scope_report()["routing_index"]

        self.assertFalse(routing_stats["merged_torch_search_enabled"])
        self.assertFalse(routing_stats["merged_torch_cache_ready"])
        self.assertEqual(routing_stats["merged_torch_cache_bytes"], 0)

    def test_model_runtime_scope_reports_cuda_first_device_evidence(self) -> None:
        cfg = MarulhoConfig(
            n_columns=4,
            column_latent_dim=8,
            routing_index_mode="torch_topk",
            device="cpu",
        )
        model = MarulhoModel(cfg)
        scope = model.runtime_scope_report()

        self.assertEqual(scope["device"]["requested_device"], "cpu")
        self.assertEqual(scope["device"]["resolved_device"], "cpu")
        self.assertEqual(scope["cuda_first_runtime"]["tensor_device"], "cpu")
        self.assertEqual(scope["cuda_first_runtime"]["routing_search_device"], "cpu")
        self.assertTrue(scope["cuda_first_runtime"]["routing_backend_cuda_capable"])

    def test_faiss_search_exact_fills_missing_unique_ids(self) -> None:
        class FakeFaissIndex:
            ntotal = 3

            def search(self, query: np.ndarray, search_k: int) -> tuple[np.ndarray, np.ndarray]:
                del query, search_k
                return (
                    np.asarray([[0.2, 0.2, 1.8]], dtype=np.float32),
                    np.asarray([[0, 0, 1]], dtype=np.int64),
                )

        index = HierarchicalAssemblyIndex(
            dim=2,
            rebuild_threshold=2,
            device=torch.device("cpu"),
            backend="exact_cosine",
        )
        index._backend = "faiss_hnsw"
        index._use_faiss = True
        index.index = FakeFaissIndex()
        index._vector_store = {
            0: np.asarray([1.0, 0.0], dtype=np.float32),
            1: np.asarray([0.0, 1.0], dtype=np.float32),
            2: np.asarray([0.0, 1.0], dtype=np.float32),
        }

        found_ids, distances = index.search(torch.tensor([[0.0, 1.0]], dtype=torch.float32), k=3)

        self.assertEqual(found_ids[0], [2, 0, 1])
        self.assertEqual(distances.shape, (1, 3))

    def test_sharded_hnsw_search_sweeps_full_small_shards(self) -> None:
        class RecordingShard:
            def __init__(self, ntotal: int, index_type: str) -> None:
                self.ntotal = ntotal
                self._index_type = index_type
                self.calls: list[int] = []

            def search(self, query: torch.Tensor, k: int = 5):
                self.calls.append(int(k))
                width = min(int(k), int(self.ntotal))
                ids = [list(range(width)) for _ in range(query.shape[0])]
                dists = np.zeros((query.shape[0], width), dtype=np.float32)
                return ids, dists

            def stats(self) -> dict[str, object]:
                return {"index_type": self._index_type}

        index = ShardedHierarchicalAssemblyIndex(
            dim=2,
            n_shards=2,
            rebuild_threshold=2,
            shard_candidate_factor=2,
            device=torch.device("cpu"),
            backend="exact_cosine",
        )
        small_hnsw_shard = RecordingShard(ntotal=128, index_type="faiss_hnsw")
        large_hnsw_shard = RecordingShard(ntotal=512, index_type="faiss_hnsw")
        index.shards = [small_hnsw_shard, large_hnsw_shard]  # type: ignore[assignment]

        index.search(torch.tensor([[1.0, 0.0]], dtype=torch.float32), k=16)

        self.assertEqual(small_hnsw_shard.calls, [128])
        self.assertEqual(large_hnsw_shard.calls, [32])


if __name__ == "__main__":
    unittest.main()
