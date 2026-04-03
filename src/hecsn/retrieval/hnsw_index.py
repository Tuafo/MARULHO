from __future__ import annotations

from typing import Any, Dict, List, Tuple, cast
import numpy as np
import torch

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover
    faiss = None


class HierarchicalAssemblyIndex:
    """Two-mode ANN index.

    - If FAISS is available: CPU HNSW index.
    - Otherwise: exact cosine fallback over in-memory vectors.
    """

    def __init__(self, dim: int, rebuild_threshold: int = 1000) -> None:
        self.dim = int(dim)
        self.rebuild_threshold = int(rebuild_threshold)
        self.insertion_count = 0
        self.rebuild_count = 0
        self.tombstones: set[int] = set()
        self._vector_store: Dict[int, np.ndarray] = {}

        self._use_faiss = faiss is not None
        self.index = None
        if self._use_faiss:
            self.index = self._create_faiss_index()

    def _create_faiss_index(self) -> Any:
        faiss_module = cast(Any, faiss)
        base_index = cast(Any, faiss_module.IndexHNSWFlat(self.dim, 32))
        base_index.hnsw.efConstruction = 200
        base_index.hnsw.efSearch = 128
        return cast(Any, faiss_module.IndexIDMap(base_index))

    @property
    def ntotal(self) -> int:
        if self._use_faiss:
            assert self.index is not None
            return int(self.index.ntotal)
        return len(self._vector_store)

    def add(self, vectors: torch.Tensor, ids: np.ndarray) -> None:
        norms = torch.norm(vectors, dim=1, keepdim=True)
        normalized = (vectors / (norms + 1e-8)).detach().cpu().numpy().astype(np.float32)

        new_positions: List[int] = []
        updated_count = 0

        for i, vec_id in enumerate(ids):
            key = int(vec_id)
            self.tombstones.discard(int(vec_id))
            if key not in self._vector_store:
                new_positions.append(i)
            else:
                updated_count += 1
            self._vector_store[key] = normalized[i].copy()

        if self._use_faiss and new_positions:
            index = cast(Any, self.index)
            new_vectors = normalized[new_positions]
            new_ids = ids[np.asarray(new_positions, dtype=np.int64)].astype(np.int64)
            index.add_with_ids(new_vectors, new_ids)

        self.insertion_count += int(len(ids))
        if updated_count > 0 and self.insertion_count >= self.rebuild_threshold:
            self.rebuild()
        elif self.insertion_count >= self.rebuild_threshold:
            self.rebuild()

    def remove(self, vec_id: int) -> None:
        self.tombstones.add(int(vec_id))
        self._vector_store.pop(int(vec_id), None)

    def rebuild(self) -> None:
        if not self._use_faiss:
            self.insertion_count = 0
            self.tombstones.clear()
            self.rebuild_count += 1
            return

        valid_ids = [idx for idx in self._vector_store.keys() if idx not in self.tombstones]
        self.index = self._create_faiss_index()

        if valid_ids:
            vectors = np.stack([self._vector_store[idx] for idx in valid_ids], axis=0)
            ids = np.array(valid_ids, dtype=np.int64)
            self.index.add_with_ids(vectors, ids)

        self.insertion_count = 0
        self.tombstones.clear()
        self.rebuild_count += 1

    @staticmethod
    def _pad_distance_rows(rows: List[List[float]], batch_size: int) -> np.ndarray:
        width = max((len(row) for row in rows), default=0)
        if width == 0:
            return np.empty((batch_size, 0), dtype=np.float32)

        padded = np.full((batch_size, width), np.inf, dtype=np.float32)
        for row_idx, row in enumerate(rows):
            if row:
                padded[row_idx, : len(row)] = np.asarray(row, dtype=np.float32)
        return padded

    def search(self, query: torch.Tensor, k: int = 5) -> Tuple[List[List[int]], np.ndarray]:
        if self.ntotal == 0:
            return [[] for _ in range(query.shape[0])], np.empty((query.shape[0], 0), dtype=np.float32)

        norms = torch.norm(query, dim=1, keepdim=True)
        normalized = (query / (norms + 1e-8)).detach().cpu().numpy().astype(np.float32)

        if self._use_faiss:
            index = cast(Any, self.index)
            search_k = max(k + len(self.tombstones), k * 4)
            dists, ids = index.search(normalized, search_k)
            valid_ids: List[List[int]] = []
            valid_dists: List[List[float]] = []
            for row_ids, row_dists in zip(ids, dists):
                row_valid_ids: List[int] = []
                row_valid_dists: List[float] = []
                seen: set[int] = set()
                for candidate_id, candidate_dist in zip(row_ids.tolist(), row_dists.tolist()):
                    idx = int(candidate_id)
                    if idx < 0 or idx in self.tombstones or idx in seen:
                        continue
                    seen.add(idx)
                    row_valid_ids.append(idx)
                    row_valid_dists.append(float(candidate_dist))
                    if len(row_valid_ids) >= k:
                        break
                valid_ids.append(row_valid_ids)
                valid_dists.append(row_valid_dists)
            return valid_ids, self._pad_distance_rows(valid_dists, query.shape[0])

        all_ids = np.array(list(self._vector_store.keys()), dtype=np.int64)
        all_vecs = np.stack([self._vector_store[i] for i in all_ids], axis=0)
        sim = normalized @ all_vecs.T
        order = np.argsort(-sim, axis=1)

        out_ids: List[List[int]] = []
        out_rows: List[List[float]] = []
        for r in range(order.shape[0]):
            chosen = order[r, : min(k, all_vecs.shape[0])]
            ids_row = [int(all_ids[i]) for i in chosen]
            out_ids.append(ids_row)
            out_rows.append((1.0 - sim[r, chosen]).astype(np.float32).tolist())

        return out_ids, self._pad_distance_rows(out_rows, normalized.shape[0])

    def stats(self) -> dict[str, Any]:
        return {
            "index_type": "faiss_hnsw" if self._use_faiss else "exact_cosine",
            "raw_entries": int(self.ntotal),
            "unique_vectors": int(len(self._vector_store)),
            "tombstones": int(len(self.tombstones)),
            "insertion_count": int(self.insertion_count),
            "rebuild_count": int(self.rebuild_count),
            "rebuild_threshold": int(self.rebuild_threshold),
        }


class ShardedHierarchicalAssemblyIndex:
    """Logical column-sharded wrapper over multiple ANN indices."""

    def __init__(
        self,
        dim: int,
        n_shards: int = 2,
        rebuild_threshold: int = 1000,
        shard_candidate_factor: int = 2,
    ) -> None:
        self.dim = int(dim)
        self.n_shards = max(1, int(n_shards))
        self.rebuild_threshold = int(rebuild_threshold)
        self.shard_candidate_factor = max(1, int(shard_candidate_factor))
        self.shards = [
            HierarchicalAssemblyIndex(dim=self.dim, rebuild_threshold=self.rebuild_threshold)
            for _ in range(self.n_shards)
        ]

    @property
    def ntotal(self) -> int:
        return int(sum(shard.ntotal for shard in self.shards))

    def shard_for_id(self, vec_id: int) -> int:
        return int(vec_id) % self.n_shards

    def add(self, vectors: torch.Tensor, ids: np.ndarray) -> None:
        if len(ids) == 0:
            return

        shard_positions: Dict[int, List[int]] = {}
        for position, vec_id in enumerate(ids.tolist()):
            shard_id = self.shard_for_id(int(vec_id))
            shard_positions.setdefault(shard_id, []).append(position)

        for shard_id, positions in shard_positions.items():
            shard_vectors = vectors[positions]
            shard_ids = ids[np.asarray(positions, dtype=np.int64)]
            self.shards[shard_id].add(shard_vectors, shard_ids.astype(np.int64))

    def remove(self, vec_id: int) -> None:
        self.shards[self.shard_for_id(vec_id)].remove(vec_id)

    def rebuild(self) -> None:
        for shard in self.shards:
            shard.rebuild()

    def search(self, query: torch.Tensor, k: int = 5) -> Tuple[List[List[int]], np.ndarray]:
        if self.ntotal == 0:
            return [[] for _ in range(query.shape[0])], np.empty((query.shape[0], 0), dtype=np.float32)

        local_k = max(1, int(k) * self.shard_candidate_factor)
        shard_results = [shard.search(query, k=local_k) for shard in self.shards]

        merged_ids: List[List[int]] = []
        merged_dists: List[List[float]] = []
        for row_idx in range(query.shape[0]):
            candidates: List[Tuple[float, int]] = []
            seen: set[int] = set()
            for shard_ids, shard_dists in shard_results:
                ids_row = shard_ids[row_idx]
                for candidate_pos, candidate_id in enumerate(ids_row):
                    if candidate_id in seen:
                        continue
                    seen.add(candidate_id)
                    distance = float(shard_dists[row_idx, candidate_pos]) if candidate_pos < shard_dists.shape[1] else float("inf")
                    candidates.append((distance, int(candidate_id)))

            candidates.sort(key=lambda item: item[0])
            top_candidates = candidates[:k]
            merged_ids.append([candidate_id for _, candidate_id in top_candidates])
            merged_dists.append([distance for distance, _ in top_candidates])

        return merged_ids, HierarchicalAssemblyIndex._pad_distance_rows(merged_dists, query.shape[0])

    def stats(self) -> dict[str, Any]:
        shard_stats = [shard.stats() for shard in self.shards]
        shard_sizes = [int(stat["unique_vectors"]) for stat in shard_stats]
        active_sizes = [size for size in shard_sizes if size > 0]
        balance_ratio = 1.0
        if len(active_sizes) >= 2:
            balance_ratio = float(max(active_sizes) / max(1, min(active_sizes)))
        elif not active_sizes:
            balance_ratio = 0.0

        return {
            "index_type": "sharded_hnsw" if shard_stats and shard_stats[0]["index_type"] == "faiss_hnsw" else "sharded_exact",
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
        }
