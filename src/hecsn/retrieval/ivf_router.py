"""GPU-native IVF routing for >50K columns (§6.1).

Implements Inverted File (IVF) index using pure PyTorch:
- Partition prototypes into sqrt(N) cells via mini-batch k-means
- At query time: find top-nprobe cells, search only those candidates
- All operations on GPU when available

Falls back to flat cosine search at small scale (< 1K prototypes).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F


@dataclass
class RoutingBenchmarkResult:
    """Result of a routing benchmark run."""
    n_cols: int
    dim: int
    device: str
    method: str
    ms_per_query: float
    ms_p95: float = 0.0
    ms_p99: float = 0.0
    recall_at_k: float = 1.0
    k: int = 32
    n_queries: int = 1000


class IVFRouter:
    """Inverted File Index router using pure PyTorch.

    Args:
        dim: Prototype dimensionality.
        n_cells: Number of Voronoi cells (default: auto = sqrt(n_protos)).
        nprobe: Number of cells to search at query time (default: 8).
        device: Torch device.
    """

    def __init__(
        self,
        dim: int,
        n_cells: int | None = None,
        nprobe: int = 8,
        device: torch.device | None = None,
    ) -> None:
        self.dim = int(dim)
        self._n_cells_hint = n_cells
        self.nprobe = int(nprobe)
        self.device = device or torch.device("cpu")

        # Centroids: (n_cells, dim)
        self._centroids: torch.Tensor | None = None
        # Inverted lists: cell_id → list of (prototype_idx, vector)
        self._inv_lists: dict[int, list[int]] = {}
        self._prototypes: torch.Tensor | None = None
        self._n_protos = 0
        self._is_trained = False

    def train(self, prototypes: torch.Tensor, n_iters: int = 20) -> None:
        """Train IVF centroids via mini-batch k-means."""
        n = prototypes.shape[0]
        protos = F.normalize(prototypes.to(self.device).float(), dim=1)
        self._prototypes = protos
        self._n_protos = n

        n_cells = self._n_cells_hint or max(4, int(math.sqrt(n)))
        n_cells = min(n_cells, n)

        # Initialize centroids from random prototypes
        perm = torch.randperm(n, device=self.device)[:n_cells]
        centroids = protos[perm].clone()

        for _ in range(n_iters):
            # Assign each prototype to nearest centroid
            sims = protos @ centroids.T  # (n, n_cells)
            assignments = sims.argmax(dim=1)  # (n,)

            # Update centroids
            new_centroids = torch.zeros_like(centroids)
            counts = torch.zeros(n_cells, device=self.device)
            for c in range(n_cells):
                mask = assignments == c
                if mask.any():
                    new_centroids[c] = protos[mask].mean(dim=0)
                    counts[c] = mask.sum()
                else:
                    # Dead centroid — reinitialize from random prototype
                    new_centroids[c] = protos[torch.randint(n, (1,), device=self.device)]
                    counts[c] = 1

            centroids = F.normalize(new_centroids, dim=1)

        self._centroids = centroids

        # Build inverted lists
        sims = protos @ centroids.T
        assignments = sims.argmax(dim=1)
        self._inv_lists = {}
        for i in range(n):
            cell = int(assignments[i].item())
            self._inv_lists.setdefault(cell, []).append(i)

        self._is_trained = True

    def search(self, query: torch.Tensor, k: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
        """Search for top-k nearest prototypes.

        Args:
            query: (dim,) or (batch, dim) query tensor.
            k: Number of nearest prototypes to return.

        Returns:
            (indices, scores) — both of shape (batch, k).
        """
        if self._prototypes is None or not self._is_trained:
            raise RuntimeError("IVFRouter not trained. Call train() first.")

        single = query.dim() == 1
        if single:
            query = query.unsqueeze(0)

        q = F.normalize(query.to(self.device).float(), dim=1)
        batch_size = q.shape[0]
        k_actual = min(k, self._n_protos)

        assert self._centroids is not None
        # Find top-nprobe cells
        cell_sims = q @ self._centroids.T  # (batch, n_cells)
        nprobe = min(self.nprobe, self._centroids.shape[0])
        _, top_cells = cell_sims.topk(nprobe, dim=1)  # (batch, nprobe)

        all_indices = []
        all_scores = []

        for b in range(batch_size):
            # Gather candidate indices from top cells
            candidates: list[int] = []
            for cell_idx in top_cells[b].tolist():
                candidates.extend(self._inv_lists.get(int(cell_idx), []))

            if not candidates:
                all_indices.append(torch.zeros(k_actual, dtype=torch.long, device=self.device))
                all_scores.append(torch.zeros(k_actual, device=self.device))
                continue

            cand_tensor = torch.tensor(candidates, dtype=torch.long, device=self.device)
            cand_vectors = self._prototypes[cand_tensor]
            sims = (q[b] @ cand_vectors.T)

            k_local = min(k_actual, len(candidates))
            top_scores, top_local = sims.topk(k_local)

            result_indices = cand_tensor[top_local]

            # Pad if needed
            if k_local < k_actual:
                pad = k_actual - k_local
                result_indices = torch.cat([result_indices, torch.zeros(pad, dtype=torch.long, device=self.device)])
                top_scores = torch.cat([top_scores, torch.full((pad,), -1.0, device=self.device)])

            all_indices.append(result_indices)
            all_scores.append(top_scores)

        indices = torch.stack(all_indices)
        scores = torch.stack(all_scores)

        if single:
            indices = indices.squeeze(0)
            scores = scores.squeeze(0)

        return indices, scores

    def search_flat(self, query: torch.Tensor, k: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
        """Exact flat search (for recall comparison)."""
        if self._prototypes is None:
            raise RuntimeError("No prototypes loaded.")

        single = query.dim() == 1
        if single:
            query = query.unsqueeze(0)

        q = F.normalize(query.to(self.device).float(), dim=1)
        sims = q @ self._prototypes.T
        k_actual = min(k, self._n_protos)
        top_scores, top_indices = sims.topk(k_actual, dim=1)

        if single:
            top_indices = top_indices.squeeze(0)
            top_scores = top_scores.squeeze(0)

        return top_indices, top_scores

    def recall_at_k(self, n_queries: int = 100, k: int = 32) -> float:
        """Measure recall@k of IVF vs flat search."""
        if self._prototypes is None:
            return 0.0

        queries = F.normalize(torch.randn(n_queries, self.dim, device=self.device), dim=1)
        total_recall = 0.0

        for i in range(n_queries):
            ivf_idx, _ = self.search(queries[i], k=k)
            flat_idx, _ = self.search_flat(queries[i], k=k)

            ivf_set = set(ivf_idx.tolist())
            flat_set = set(flat_idx.tolist())
            overlap = len(ivf_set & flat_set)
            total_recall += overlap / max(1, len(flat_set))

        return total_recall / n_queries

    @property
    def n_cells(self) -> int:
        return 0 if self._centroids is None else self._centroids.shape[0]


def benchmark_routing(
    n_cols: int,
    dim: int = 256,
    n_queries: int = 200,
    k: int = 32,
    device: str | None = None,
) -> RoutingBenchmarkResult:
    """Benchmark routing latency at given scale.

    Automatically selects flat or IVF based on n_cols threshold.
    """
    if device is None:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)

    prototypes = F.normalize(torch.randn(n_cols, dim, device=dev), dim=1)
    queries = F.normalize(torch.randn(n_queries, dim, device=dev), dim=1)

    use_ivf = n_cols > 50_000
    method = "ivf" if use_ivf else "flat"

    if use_ivf:
        router = IVFRouter(dim=dim, device=dev)
        router.train(prototypes)

        # Warmup
        for i in range(min(10, n_queries)):
            router.search(queries[i], k=k)

        if dev.type == "cuda":
            torch.cuda.synchronize()

        latencies: list[float] = []
        for i in range(n_queries):
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t_start = time.perf_counter()
            router.search(queries[i], k=k)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t_end = time.perf_counter()
            latencies.append((t_end - t_start) * 1000)

        recall = router.recall_at_k(n_queries=min(50, n_queries), k=k)
    else:
        # Flat cosine search
        # Warmup
        for i in range(min(10, n_queries)):
            sims = queries[i:i+1] @ prototypes.T
            sims.topk(min(k, n_cols), dim=1)

        if dev.type == "cuda":
            torch.cuda.synchronize()

        latencies = []
        for i in range(n_queries):
            q = queries[i:i+1]
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t_start = time.perf_counter()
            sims = q @ prototypes.T
            sims.topk(min(k, n_cols), dim=1)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t_end = time.perf_counter()
            latencies.append((t_end - t_start) * 1000)

        recall = 1.0  # Flat is exact

    lat_tensor = torch.tensor(latencies)
    ms_per_query = float(lat_tensor.median().item())
    ms_p95 = float(lat_tensor.quantile(0.95).item())
    ms_p99 = float(lat_tensor.quantile(0.99).item())

    return RoutingBenchmarkResult(
        n_cols=n_cols,
        dim=dim,
        device=str(dev),
        method=method,
        ms_per_query=ms_per_query,
        ms_p95=ms_p95,
        ms_p99=ms_p99,
        recall_at_k=recall,
        k=k,
        n_queries=n_queries,
    )
