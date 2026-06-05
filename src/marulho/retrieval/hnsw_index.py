from __future__ import annotations

from typing import Any, Dict, List, Tuple, cast

import numpy as np
import torch
import torch.nn.functional as F

from marulho.retrieval.turboquant_store import TurboQuantPrototypeStore

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover
    faiss = None


class HierarchicalAssemblyIndex:
    """Multi-backend routing index.

    - torch_topk: exact top-k search on the configured torch device.
    - faiss_hnsw: CPU HNSW index when FAISS is available.
    - exact_cosine: numpy cosine fallback over in-memory vectors.
    """

    def __init__(
        self,
        dim: int,
        rebuild_threshold: int = 1000,
        *,
        device: torch.device | None = None,
        backend: str = "auto",
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

        self._backend = self._resolve_backend(backend)
        self._use_faiss = self._backend == "faiss_hnsw"
        self.index = None
        if self._use_faiss:
            self.index = self._create_faiss_index()

        # TurboQuant+ state (lazy-built on first search after add)
        self._tq_store: TurboQuantPrototypeStore | None = None
        self._tq_id_list: List[int] = []
        self._tq_cache_dirty = True

    def _resolve_backend(self, backend: str) -> str:
        requested = str(backend).strip().lower() or "auto"
        if requested == "auto":
            if self.device.type == "cuda":
                return "torch_topk"
            return "faiss_hnsw" if faiss is not None else "exact_cosine"
        if requested == "faiss_hnsw":
            if faiss is None:
                raise ValueError("faiss_hnsw backend requested but faiss is unavailable")
            return requested
        if requested in {"torch_topk", "exact_cosine", "turboquant_plus"}:
            return requested
        raise ValueError(f"Unsupported routing backend: {backend!r}")

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
        norm_t = vectors / (norms + 1e-8)
        if norm_t.device.type == "cpu":
            normalized = norm_t.numpy().astype(np.float32)
        else:
            normalized = norm_t.detach().cpu().numpy().astype(np.float32)
        self._torch_cache_dirty = True
        self._tq_cache_dirty = True

        new_positions: List[int] = []
        updated_count = 0

        for i, vec_id in enumerate(ids):
            key = int(vec_id)
            self.tombstones.discard(key)
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
        self._torch_cache_dirty = True
        self._tq_cache_dirty = True

    def _rebuild_torch_cache(self) -> None:
        valid_ids = [idx for idx in self._vector_store.keys() if idx not in self.tombstones]
        if not valid_ids:
            self._torch_ids = torch.empty(0, dtype=torch.long, device=self.device)
            self._torch_vectors = torch.empty((0, self.dim), dtype=torch.float32, device=self.device)
            self._torch_cache_dirty = False
            return

        vectors = np.stack([self._vector_store[idx] for idx in valid_ids], axis=0)
        self._torch_ids = torch.tensor(valid_ids, dtype=torch.long, device=self.device)
        self._torch_vectors = torch.from_numpy(vectors).to(self.device)
        self._torch_cache_dirty = False

    def _rebuild_tq_cache(self) -> None:
        """Rebuild TurboQuant+ compressed store from current vector_store."""
        valid_ids = [idx for idx in self._vector_store.keys() if idx not in self.tombstones]
        if not valid_ids:
            self._tq_store = None
            self._tq_id_list = []
            self._tq_cache_dirty = False
            return

        vectors = np.stack([self._vector_store[idx] for idx in valid_ids], axis=0)
        protos = torch.from_numpy(vectors).float().to(self.device)

        n_cols = len(valid_ids)
        n_proj = max(2 * self.dim, 64)  # more projections → lower QJL variance
        self._tq_store = TurboQuantPrototypeStore(
            n_cols=n_cols, dim=self.dim, bits=3,
            n_projections=n_proj, device=self.device,
        )
        self._tq_store.set_all(protos)
        self._tq_store.compress_all()
        self._tq_id_list = valid_ids
        self._tq_cache_dirty = False

    def rebuild(self) -> None:
        if self._backend == "torch_topk":
            self._rebuild_torch_cache()
            self.insertion_count = 0
            self.tombstones.clear()
            self.rebuild_count += 1
            return

        if self._backend == "turboquant_plus":
            self._rebuild_tq_cache()
            self.insertion_count = 0
            self.tombstones.clear()
            self.rebuild_count += 1
            return

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

    def _complete_faiss_row(
        self,
        normalized_query: np.ndarray,
        row_valid_ids: List[int],
        row_valid_dists: List[float],
        target_k: int,
    ) -> Tuple[List[int], List[float]]:
        if len(row_valid_ids) >= target_k:
            pairs = sorted(zip(row_valid_dists, row_valid_ids), key=lambda item: item[0])[:target_k]
            return [int(candidate_id) for _, candidate_id in pairs], [float(distance) for distance, _ in pairs]

        seen = set(row_valid_ids)
        remaining_ids = [
            idx
            for idx in self._vector_store.keys()
            if idx not in self.tombstones and idx not in seen
        ]
        if not remaining_ids:
            pairs = sorted(zip(row_valid_dists, row_valid_ids), key=lambda item: item[0])[:target_k]
            return [int(candidate_id) for _, candidate_id in pairs], [float(distance) for distance, _ in pairs]

        remaining_vectors = np.stack([self._vector_store[idx] for idx in remaining_ids], axis=0)
        similarities = remaining_vectors @ normalized_query
        distances = np.maximum(0.0, 2.0 - 2.0 * similarities)
        combined = list(zip(row_valid_dists, row_valid_ids))
        combined.extend((float(distance), int(candidate_id)) for distance, candidate_id in zip(distances.tolist(), remaining_ids))
        combined.sort(key=lambda item: item[0])
        top = combined[:target_k]
        return [int(candidate_id) for _, candidate_id in top], [float(distance) for distance, _ in top]

    def search(self, query: torch.Tensor, k: int = 5) -> Tuple[List[List[int]], np.ndarray]:
        query_batch = query if query.dim() == 2 else query.unsqueeze(0)
        if self.ntotal == 0:
            return [[] for _ in range(query_batch.shape[0])], np.empty((query_batch.shape[0], 0), dtype=np.float32)

        if self._backend == "torch_topk":
            if self._torch_cache_dirty:
                self._rebuild_torch_cache()
            if int(self._torch_vectors.shape[0]) <= 0:
                return [[] for _ in range(query_batch.shape[0])], np.empty((query_batch.shape[0], 0), dtype=np.float32)
            normalized_query = F.normalize(query_batch.to(self.device), dim=1)
            sims = normalized_query @ self._torch_vectors.T
            topk = min(max(1, int(k)), int(self._torch_vectors.shape[0]))
            values, indices = torch.topk(sims, k=topk, dim=1)
            _on_cpu = (self.device == torch.device("cpu"))
            ids = self._torch_ids[indices].tolist() if _on_cpu else self._torch_ids[indices].cpu().tolist()
            dists_t = 1.0 - values
            dists = dists_t.numpy().astype(np.float32) if _on_cpu else dists_t.detach().cpu().numpy().astype(np.float32)
            return [[int(candidate_id) for candidate_id in row] for row in ids], dists

        if self._backend == "turboquant_plus":
            if self._tq_cache_dirty:
                self._rebuild_tq_cache()
            if self._tq_store is None or not self._tq_id_list:
                return [[] for _ in range(query_batch.shape[0])], np.empty((query_batch.shape[0], 0), dtype=np.float32)
            out_ids: List[List[int]] = []
            out_dists: List[List[float]] = []
            for row_idx in range(query_batch.shape[0]):
                q = query_batch[row_idx].to(self.device)
                k_actual = min(k, len(self._tq_id_list))
                tq_indices, tq_scores = self._tq_store.route(q, k=k_actual)
                row_ids = [self._tq_id_list[int(i)] for i in tq_indices.cpu().tolist()]
                row_dist = (1.0 - tq_scores).detach().cpu().numpy().astype(np.float32).tolist()
                out_ids.append(row_ids)
                out_dists.append(row_dist)
            return out_ids, self._pad_distance_rows(out_dists, query_batch.shape[0])

        norms = torch.norm(query_batch, dim=1, keepdim=True)
        normalized = (query_batch / (norms + 1e-8)).detach().cpu().numpy().astype(np.float32)

        if self._use_faiss:
            index = cast(Any, self.index)
            search_k = max(k + len(self.tombstones), k * 4)
            dists, ids = index.search(normalized, search_k)
            valid_ids: List[List[int]] = []
            valid_dists: List[List[float]] = []
            target_k = min(max(1, int(k)), len(self._vector_store) - len(self.tombstones))
            for row_idx, (row_ids, row_dists) in enumerate(zip(ids, dists)):
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
                    if len(row_valid_ids) >= target_k:
                        break
                row_valid_ids, row_valid_dists = self._complete_faiss_row(
                    normalized[row_idx],
                    row_valid_ids,
                    row_valid_dists,
                    target_k,
                )
                valid_ids.append(row_valid_ids)
                valid_dists.append(row_valid_dists)
            return valid_ids, self._pad_distance_rows(valid_dists, query_batch.shape[0])

        all_ids = np.array(list(self._vector_store.keys()), dtype=np.int64)
        all_vecs = np.stack([self._vector_store[i] for i in all_ids], axis=0)
        sim = normalized @ all_vecs.T
        order = np.argsort(-sim, axis=1)

        out_ids: List[List[int]] = []
        out_rows: List[List[float]] = []
        for row_idx in range(order.shape[0]):
            chosen = order[row_idx, : min(k, all_vecs.shape[0])]
            ids_row = [int(all_ids[i]) for i in chosen]
            out_ids.append(ids_row)
            out_rows.append((1.0 - sim[row_idx, chosen]).astype(np.float32).tolist())

        return out_ids, self._pad_distance_rows(out_rows, normalized.shape[0])

    def stats(self) -> dict[str, Any]:
        info: dict[str, Any] = {
            "index_type": self._backend,
            "raw_entries": int(self.ntotal),
            "unique_vectors": int(len(self._vector_store)),
            "tombstones": int(len(self.tombstones)),
            "insertion_count": int(self.insertion_count),
            "rebuild_count": int(self.rebuild_count),
            "rebuild_threshold": int(self.rebuild_threshold),
            "search_device": self.device.type,
        }
        if self._backend == "torch_topk":
            info["torch_cache_dirty"] = bool(self._torch_cache_dirty)
            info["torch_cache_ready"] = not bool(self._torch_cache_dirty)
            info["torch_vector_cache_device"] = str(self._torch_vectors.device)
            info["torch_id_cache_device"] = str(self._torch_ids.device)
            info["torch_vector_cache_count"] = int(self._torch_vectors.shape[0])
            info["torch_cache_cuda"] = bool(
                self._torch_vectors.is_cuda and self._torch_ids.is_cuda
            )
        if self._backend == "turboquant_plus":
            info["tq_cache_dirty"] = bool(self._tq_cache_dirty)
            info["tq_cache_ready"] = (
                self._tq_store is not None and not bool(self._tq_cache_dirty)
            )
            tq_device = self._tq_store.device if self._tq_store is not None else self.device
            info["tq_device"] = str(tq_device)
            if self._tq_store is not None:
                info["tq_memory"] = self._tq_store.memory_bytes()
                info["tq_fp32_device"] = str(self._tq_store._fp32.device)
                info["tq_codes_device"] = str(self._tq_store._codes.device)
                info["tq_residual_device"] = str(self._tq_store._residual_signs.device)
        return info


class ShardedHierarchicalAssemblyIndex:
    """Logical column-sharded wrapper over multiple ANN indices."""

    def __init__(
        self,
        dim: int,
        n_shards: int = 2,
        rebuild_threshold: int = 1000,
        shard_candidate_factor: int = 2,
        *,
        device: torch.device | None = None,
        backend: str = "auto",
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
                backend=backend,
            )
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

    def _local_k_for_shard(self, shard: HierarchicalAssemblyIndex, requested_k: int) -> int:
        local_k = max(1, int(requested_k) * self.shard_candidate_factor)
        shard_size = int(shard.ntotal)
        if shard_size <= 0:
            return local_k
        if shard.stats()["index_type"] == "faiss_hnsw" and shard_size <= max(local_k * 4, 128):
            return shard_size
        return local_k

    def search(self, query: torch.Tensor, k: int = 5) -> Tuple[List[List[int]], np.ndarray]:
        query_batch = query if query.dim() == 2 else query.unsqueeze(0)
        if self.ntotal == 0:
            return [[] for _ in range(query_batch.shape[0])], np.empty((query_batch.shape[0], 0), dtype=np.float32)

        shard_results = [
            shard.search(query_batch, k=self._local_k_for_shard(shard, k))
            for shard in self.shards
        ]

        merged_ids: List[List[int]] = []
        merged_dists: List[List[float]] = []
        for row_idx in range(query_batch.shape[0]):
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

        return merged_ids, HierarchicalAssemblyIndex._pad_distance_rows(merged_dists, query_batch.shape[0])

    def stats(self) -> dict[str, Any]:
        shard_stats = [shard.stats() for shard in self.shards]
        shard_sizes = [int(stat["unique_vectors"]) for stat in shard_stats]
        active_sizes = [size for size in shard_sizes if size > 0]
        balance_ratio = 1.0
        if len(active_sizes) >= 2:
            balance_ratio = float(max(active_sizes) / max(1, min(active_sizes)))
        elif not active_sizes:
            balance_ratio = 0.0

        base_type = shard_stats[0]["index_type"] if shard_stats else "exact_cosine"
        if base_type == "faiss_hnsw":
            index_type = "sharded_hnsw"
        elif base_type == "torch_topk":
            index_type = "sharded_torch_topk"
        else:
            index_type = "sharded_exact"

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
