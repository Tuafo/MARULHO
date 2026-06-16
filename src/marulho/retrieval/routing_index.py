from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

class HierarchicalAssemblyIndex:
    """Torch-backed exact top-k routing index.

    The promoted scheduler route depends on this exact tensor cache. Older
    CPU FAISS and numpy cosine modes were retired because they cannot feed the
    CUDA graph route/vote boundary without falling back out of the real path.
    """

    def __init__(
        self,
        dim: int,
        rebuild_threshold: int = 1000,
        *,
        device: torch.device | None = None,
    ) -> None:
        self.dim = int(dim)
        self.rebuild_threshold = int(rebuild_threshold)
        self.device = torch.device("cpu") if device is None else torch.device(device)
        self.insertion_count = 0
        self.rebuild_count = 0
        self.tombstones: set[int] = set()
        self._vector_store: Dict[int, np.ndarray] = {}
        self._torch_ids = torch.empty(0, dtype=torch.long, device=self.device)
        self._torch_vectors = torch.empty((0, self.dim), dtype=torch.float32, device=self.device)
        self._torch_cache_dirty = True
        self._torch_cache_generation = 0
        self.tensor_search_count = 0
        self.last_search_mode: str | None = None

    @property
    def ntotal(self) -> int:
        return len(self._vector_store)

    def add(self, vectors: torch.Tensor, ids: np.ndarray) -> None:
        norms = torch.norm(vectors, dim=1, keepdim=True)
        norm_t = vectors / (norms + 1e-8)
        if norm_t.device.type == "cpu":
            normalized = norm_t.numpy().astype(np.float32)
        else:
            normalized = norm_t.detach().cpu().numpy().astype(np.float32)
        self._torch_cache_dirty = True
        self._torch_cache_generation += 1

        updated_count = 0

        for i, vec_id in enumerate(ids):
            key = int(vec_id)
            self.tombstones.discard(key)
            if key in self._vector_store:
                updated_count += 1
            self._vector_store[key] = normalized[i].copy()

        self.insertion_count += int(len(ids))
        if updated_count > 0 and self.insertion_count >= self.rebuild_threshold:
            self.rebuild()
        elif self.insertion_count >= self.rebuild_threshold:
            self.rebuild()

    def synchronize_host_store(
        self,
        vectors: torch.Tensor,
        ids: np.ndarray,
    ) -> None:
        """Refresh the slow host mirror without invalidating live device caches."""
        if int(vectors.shape[0]) != int(len(ids)):
            raise ValueError("vectors and ids must have matching rows")
        normalized = F.normalize(vectors.detach().float(), dim=1)
        normalized_cpu = normalized.to(device="cpu").numpy().astype(
            np.float32,
            copy=False,
        )
        self._vector_store = {
            int(vec_id): normalized_cpu[position].copy()
            for position, vec_id in enumerate(ids.tolist())
        }
        self.tombstones.clear()

    def remove(self, vec_id: int) -> None:
        self.tombstones.add(int(vec_id))
        self._vector_store.pop(int(vec_id), None)
        self._torch_cache_dirty = True
        self._torch_cache_generation += 1

    def _rebuild_torch_cache(self) -> None:
        valid_ids = [idx for idx in self._vector_store.keys() if idx not in self.tombstones]
        if not valid_ids:
            self._torch_ids = torch.empty(0, dtype=torch.long, device=self.device)
            self._torch_vectors = torch.empty((0, self.dim), dtype=torch.float32, device=self.device)
            self._torch_cache_dirty = False
            return

        vectors = np.stack([self._vector_store[idx] for idx in valid_ids], axis=0)
        next_ids = torch.tensor(valid_ids, dtype=torch.long, device=self.device)
        next_vectors = torch.from_numpy(vectors).to(self.device)
        if (
            tuple(self._torch_ids.shape) == tuple(next_ids.shape)
            and tuple(self._torch_vectors.shape) == tuple(next_vectors.shape)
            and self._torch_ids.device == next_ids.device
            and self._torch_vectors.device == next_vectors.device
        ):
            self._torch_ids.copy_(next_ids)
            self._torch_vectors.copy_(next_vectors)
        else:
            self._torch_ids = next_ids
            self._torch_vectors = next_vectors
        self._torch_cache_dirty = False

    def rebuild(self) -> None:
        self._rebuild_torch_cache()
        self.insertion_count = 0
        self.tombstones.clear()
        self.rebuild_count += 1

    def search_tensors(self, query: torch.Tensor, k: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return candidate ids and distances as tensors on the index device.

        This avoids the Python-list and CPU numpy boundary for torch-backed
        routing, which is required before routing can become a measured hot-path
        component instead of a preparation bottleneck.
        """

        self.tensor_search_count += 1
        self.last_search_mode = "tensor"
        query_batch = query if query.dim() == 2 else query.unsqueeze(0)
        batch_size = int(query_batch.shape[0])
        target_k = max(1, int(k))
        if self.ntotal == 0:
            empty_ids = torch.empty((batch_size, 0), dtype=torch.long, device=self.device)
            empty_dists = torch.empty((batch_size, 0), dtype=torch.float32, device=self.device)
            return empty_ids, empty_dists

        if self._torch_cache_dirty:
            self._rebuild_torch_cache()
        if int(self._torch_vectors.shape[0]) <= 0:
            empty_ids = torch.empty((batch_size, 0), dtype=torch.long, device=self.device)
            empty_dists = torch.empty((batch_size, 0), dtype=torch.float32, device=self.device)
            return empty_ids, empty_dists
        normalized_query = F.normalize(query_batch.to(self.device), dim=1)
        sims = normalized_query @ self._torch_vectors.T
        topk = min(target_k, int(self._torch_vectors.shape[0]))
        values, indices = torch.topk(sims, k=topk, dim=1)
        ids = self._torch_ids[indices]
        dists = 1.0 - values
        return ids.long(), dists.float()

    def routing_tensor_cache(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the current exact torch routing cache without copying it."""

        if self._torch_cache_dirty:
            self._rebuild_torch_cache()
        return self._torch_vectors, self._torch_ids

    def routing_tensor_cache_is_dirty(self) -> bool:
        """Return whether the exact torch routing cache must be rebuilt."""

        return bool(self._torch_cache_dirty)

    def routing_tensor_cache_generation(self) -> int:
        """Return a monotonic stamp for cache-invalidating routing changes."""

        return int(self._torch_cache_generation)

    def stats(self) -> dict[str, Any]:
        info: dict[str, Any] = {
            "index_type": "torch_topk",
            "raw_entries": int(self.ntotal),
            "unique_vectors": int(len(self._vector_store)),
            "tombstones": int(len(self.tombstones)),
            "insertion_count": int(self.insertion_count),
            "rebuild_count": int(self.rebuild_count),
            "rebuild_threshold": int(self.rebuild_threshold),
            "search_device": self.device.type,
            "tensor_search_count": int(self.tensor_search_count),
            "last_search_mode": self.last_search_mode,
        }
        info["torch_cache_dirty"] = bool(self._torch_cache_dirty)
        info["torch_cache_ready"] = not bool(self._torch_cache_dirty)
        info["torch_cache_generation"] = int(self._torch_cache_generation)
        info["torch_vector_cache_device"] = str(self._torch_vectors.device)
        info["torch_id_cache_device"] = str(self._torch_ids.device)
        info["torch_vector_cache_count"] = int(self._torch_vectors.shape[0])
        info["torch_cache_cuda"] = bool(
            self._torch_vectors.is_cuda and self._torch_ids.is_cuda
        )
        return info


class ShardedHierarchicalAssemblyIndex:
    """Logical column-sharded wrapper over exact torch-cache routing indices."""

    def __init__(
        self,
        dim: int,
        n_shards: int = 2,
        rebuild_threshold: int = 1000,
        shard_candidate_factor: int = 2,
        *,
        device: torch.device | None = None,
    ) -> None:
        self.dim = int(dim)
        self.n_shards = max(1, int(n_shards))
        self.rebuild_threshold = int(rebuild_threshold)
        self.shard_candidate_factor = max(1, int(shard_candidate_factor))
        self.shards = [
            HierarchicalAssemblyIndex(
                dim=self.dim,
                rebuild_threshold=self.rebuild_threshold,
                device=device,
            )
            for _ in range(self.n_shards)
        ]
        self.tensor_search_count = 0
        self.last_search_mode: str | None = None
        merged_device = self.shards[0].device if self.shards else torch.device("cpu")
        self._merged_torch_ids = torch.empty(0, dtype=torch.long, device=merged_device)
        self._merged_torch_vectors = torch.empty(
            (0, self.dim),
            dtype=torch.float32,
            device=merged_device,
        )
        self._merged_torch_cache_dirty = True
        self._merged_torch_cache_generation = 0

    @property
    def ntotal(self) -> int:
        return int(sum(shard.ntotal for shard in self.shards))

    def shard_for_id(self, vec_id: int) -> int:
        return int(vec_id) % self.n_shards

    def add(self, vectors: torch.Tensor, ids: np.ndarray) -> None:
        if len(ids) == 0:
            return
        self._merged_torch_cache_dirty = True
        self._merged_torch_cache_generation += 1

        shard_positions: Dict[int, List[int]] = {}
        for position, vec_id in enumerate(ids.tolist()):
            shard_id = self.shard_for_id(int(vec_id))
            shard_positions.setdefault(shard_id, []).append(position)

        for shard_id, positions in shard_positions.items():
            shard_vectors = vectors[positions]
            shard_ids = ids[np.asarray(positions, dtype=np.int64)]
            self.shards[shard_id].add(shard_vectors, shard_ids.astype(np.int64))

    def synchronize_host_store(
        self,
        vectors: torch.Tensor,
        ids: np.ndarray,
    ) -> None:
        """Refresh shard host mirrors while preserving the live merged cache."""
        if int(vectors.shape[0]) != int(len(ids)):
            raise ValueError("vectors and ids must have matching rows")
        shard_positions: Dict[int, List[int]] = {
            shard_id: [] for shard_id in range(self.n_shards)
        }
        for position, vec_id in enumerate(ids.tolist()):
            shard_positions[self.shard_for_id(int(vec_id))].append(position)
        for shard_id, positions in shard_positions.items():
            shard_ids = ids[np.asarray(positions, dtype=np.int64)]
            shard_vectors = vectors[positions]
            self.shards[shard_id].synchronize_host_store(
                shard_vectors,
                shard_ids.astype(np.int64),
            )

    def remove(self, vec_id: int) -> None:
        self._merged_torch_cache_dirty = True
        self._merged_torch_cache_generation += 1
        self.shards[self.shard_for_id(vec_id)].remove(vec_id)

    def rebuild(self) -> None:
        self._merged_torch_cache_dirty = True
        self._merged_torch_cache_generation += 1
        for shard in self.shards:
            shard.rebuild()

    def _rebuild_merged_torch_cache(self) -> None:
        ids: list[torch.Tensor] = []
        vectors: list[torch.Tensor] = []
        for shard in self.shards:
            if shard._torch_cache_dirty:
                shard._rebuild_torch_cache()
            if int(shard._torch_ids.numel()) <= 0:
                continue
            ids.append(shard._torch_ids)
            vectors.append(shard._torch_vectors)

        if not ids:
            device = self.shards[0].device if self.shards else torch.device("cpu")
            self._merged_torch_ids = torch.empty(0, dtype=torch.long, device=device)
            self._merged_torch_vectors = torch.empty(
                (0, self.dim),
                dtype=torch.float32,
                device=device,
            )
        else:
            self._merged_torch_ids = torch.cat(ids, dim=0)
            self._merged_torch_vectors = torch.cat(vectors, dim=0)
        self._merged_torch_cache_dirty = False

    def search_tensors(self, query: torch.Tensor, k: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
        self.tensor_search_count += 1
        self.last_search_mode = "tensor"
        query_batch = query if query.dim() == 2 else query.unsqueeze(0)
        batch_size = int(query_batch.shape[0])
        target_k = max(1, int(k))
        if self.ntotal == 0:
            device = self.shards[0].device if self.shards else torch.device("cpu")
            empty_ids = torch.empty((batch_size, 0), dtype=torch.long, device=device)
            empty_dists = torch.empty((batch_size, 0), dtype=torch.float32, device=device)
            return empty_ids, empty_dists

        if self._merged_torch_cache_dirty:
            self._rebuild_merged_torch_cache()
        if int(self._merged_torch_ids.numel()) <= 0:
            device = self.shards[0].device if self.shards else torch.device("cpu")
            empty_ids = torch.empty((batch_size, 0), dtype=torch.long, device=device)
            empty_dists = torch.empty((batch_size, 0), dtype=torch.float32, device=device)
            return empty_ids, empty_dists
        normalized_query = F.normalize(query_batch.to(self._merged_torch_vectors.device), dim=1)
        similarities = normalized_query @ self._merged_torch_vectors.T
        topk = min(target_k, int(self._merged_torch_ids.shape[0]))
        values, positions = torch.topk(similarities, k=topk, dim=1)
        return self._merged_torch_ids[positions].long(), (1.0 - values).float()

    def routing_tensor_cache(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return one merged exact torch cache for fused routing."""

        if self._merged_torch_cache_dirty:
            self._rebuild_merged_torch_cache()
        return self._merged_torch_vectors, self._merged_torch_ids

    def routing_tensor_cache_is_dirty(self) -> bool:
        """Return whether the merged exact torch routing cache must be rebuilt."""

        return bool(self._merged_torch_cache_dirty)

    def routing_tensor_cache_generation(self) -> int:
        """Return a monotonic stamp for merged cache-invalidating changes."""

        return int(self._merged_torch_cache_generation)

    def stats(self) -> dict[str, Any]:
        shard_stats = [shard.stats() for shard in self.shards]
        shard_sizes = [int(stat["unique_vectors"]) for stat in shard_stats]
        active_sizes = [size for size in shard_sizes if size > 0]
        balance_ratio = 1.0
        if len(active_sizes) >= 2:
            balance_ratio = float(max(active_sizes) / max(1, min(active_sizes)))
        elif not active_sizes:
            balance_ratio = 0.0

        index_type = "sharded_torch_topk"

        return {
            "index_type": index_type,
            "n_shards": int(self.n_shards),
            "raw_entries": int(sum(int(stat["raw_entries"]) for stat in shard_stats)),
            "unique_vectors": int(sum(shard_sizes)),
            "rebuild_count": int(sum(int(stat["rebuild_count"]) for stat in shard_stats)),
            "rebuild_threshold": int(self.rebuild_threshold),
            "shard_candidate_factor": int(self.shard_candidate_factor),
            "per_shard_unique_vectors": shard_sizes,
            "per_shard_raw_entries": [int(stat["raw_entries"]) for stat in shard_stats],
            "per_shard_tombstones": [int(stat["tombstones"]) for stat in shard_stats],
            "shard_balance_ratio": balance_ratio,
            "search_device": shard_stats[0].get("search_device", "cpu") if shard_stats else "cpu",
            "tensor_search_count": int(self.tensor_search_count),
            "last_search_mode": self.last_search_mode,
            "merged_torch_search_enabled": True,
            "merged_torch_cache_dirty": bool(self._merged_torch_cache_dirty),
            "merged_torch_cache_generation": int(
                self._merged_torch_cache_generation
            ),
            "merged_torch_cache_ready": not bool(self._merged_torch_cache_dirty),
            "merged_torch_vector_cache_device": str(self._merged_torch_vectors.device),
            "merged_torch_id_cache_device": str(self._merged_torch_ids.device),
            "merged_torch_vector_cache_count": int(self._merged_torch_vectors.shape[0]),
            "merged_torch_cache_bytes": int(
                self._merged_torch_vectors.numel() * self._merged_torch_vectors.element_size()
                + self._merged_torch_ids.numel() * self._merged_torch_ids.element_size()
            ),
            "per_shard_search_device": [
                str(stat.get("search_device", "cpu")) for stat in shard_stats
            ],
            "per_shard_torch_cache_ready": [
                bool(stat.get("torch_cache_ready", False)) for stat in shard_stats
            ],
            "per_shard_torch_vector_cache_device": [
                str(stat.get("torch_vector_cache_device", "cpu")) for stat in shard_stats
            ],
            "per_shard_torch_id_cache_device": [
                str(stat.get("torch_id_cache_device", "cpu")) for stat in shard_stats
            ],
        }
