"""Hypercube topology for column connectivity.

Implements 11D-style hypercube connectivity inspired by the Blue Brain
Project's discovery of directed cliques up to 11 dimensions in neocortex
(Reimann et al. 2017) and the snn-llm project (hafufu-stack/snn-llm)
which demonstrated optimal performance at 11D with 186× parameter reduction.

Key components:
  - HypercubeTopology: reusable graph primitive with bit-flip neighbors
  - HypercubeBindingLayer: binding via hypercube-structured sparse connectivity

References:
  - Reimann et al. (2017): Cliques of neurons in neocortex, up to 11D
  - Gorban & Tyukin (2018, PMC6874527): High-dimensional selectivity
  - hafufu-stack/snn-llm: 11D hypercube SNN achieving PPL 15.9
"""

from __future__ import annotations

from itertools import combinations
import math
from typing import Any

import torch


class HypercubeTopology:
    """Hypercube graph primitive with O(N·d) bit-flip neighbor generation.

    Each node has exactly ``dim`` neighbors, defined by flipping a single
    bit in the node's binary address. This gives:
      - Regular graph: every node has identical degree
      - Logarithmic diameter: max distance = dim hops
      - Structured sparsity: d connections vs N-1 full connectivity

    Supports non-power-of-2 column counts by masking unused vertices with
    per-node degree renormalization.

    Optional small-world shortcuts add deterministic long-range edges to
    reduce average path length. The maintained policy uses a target total
    degree: fully populated vertices aim for ``dim + shortcuts_per_node``
    neighbors, while masked-boundary vertices may receive extra long-range
    edges to compensate for lost bit-flip neighbors. Shortcut selection still
    prefers the largest Hamming-distance masks first, making the topology
    reproducible and explicitly long-range. Long-range edge weights are
    calibrated row-by-row so rows with many long-range edges keep direct
    hypercube neighbors dominant.
    """

    def __init__(
        self,
        n_columns: int,
        shortcuts_per_node: int = 0,
        device: torch.device | None = None,
    ) -> None:
        self.n_columns = int(n_columns)
        self.device = device or torch.device("cpu")
        self.shortcuts_per_node = max(0, int(shortcuts_per_node))

        # Compute hypercube dimension: smallest d such that 2^d >= n_columns
        self.dim = max(1, math.ceil(math.log2(max(2, n_columns))))
        self.n_vertices = 1 << self.dim  # 2^dim
        self.shortcut_strategy = "deterministic_long_range"
        self.shortcut_budget_policy = "target_degree_compensated"
        self.long_range_weight_policy = "bounded_relative_mass"
        self.long_range_mass_ratio = 0.5
        self.target_degree = min(max(0, self.n_columns - 1), self.dim + self.shortcuts_per_node)

        # Which vertices are active (have a real column assigned)
        self.active_mask = torch.zeros(self.n_vertices, dtype=torch.bool, device=self.device)
        self.active_mask[:self.n_columns] = True
        self._shortcut_masks = self._build_shortcut_masks()

        # Build neighbor structure
        (
            self._neighbor_ids,
            self._neighbor_weights,
            self._degree,
            self._direct_degree,
            self._shortcut_degree,
        ) = self._build_neighbors()

    def _build_shortcut_masks(self) -> tuple[int, ...]:
        """Build deterministic long-range shortcut masks ordered by span.

        Masks are ordered by descending Hamming weight, so earlier masks jump
        farther across the cube. Weight-1 masks are excluded because they are
        already covered by direct hypercube neighbors.
        """
        full_mask = (1 << self.dim) - 1
        masks: list[int] = []
        for omitted_bits in range(0, max(0, self.dim - 1)):
            for omitted in combinations(range(self.dim), omitted_bits):
                mask = full_mask
                for bit in omitted:
                    mask &= ~(1 << bit)
                if mask.bit_count() <= 1:
                    continue
                masks.append(mask)
        return tuple(masks)

    def _effective_shortcut_budget(self, base_degree: int) -> int:
        return max(0, self.target_degree - int(base_degree))

    def _long_range_raw_weight(self, *, direct_degree: int, long_range_count: int) -> float:
        if long_range_count <= 0 or direct_degree <= 0:
            return 0.0
        return float(self.long_range_mass_ratio * min(1.0, float(direct_degree) / float(long_range_count)))

    def structural_row_weights(self, row: int, sources: list[int]) -> list[float]:
        if not sources:
            return []
        direct_mask = [self.hamming_distance(row, int(source)) == 1 for source in sources]
        direct_degree = sum(1 for is_direct in direct_mask if is_direct)
        long_range_count = len(sources) - direct_degree
        long_range_raw_weight = self._long_range_raw_weight(
            direct_degree=direct_degree,
            long_range_count=long_range_count,
        )
        raw_weights = [1.0 if is_direct else long_range_raw_weight for is_direct in direct_mask]
        total = sum(raw_weights)
        if total <= 0.0:
            return [0.0 for _ in sources]
        return [float(weight / total) for weight in raw_weights]

    def _reconstruct_degree_breakdown(self) -> None:
        self._direct_degree = torch.zeros(self.n_columns, dtype=torch.long, device=self.device)
        self._shortcut_degree = torch.zeros(self.n_columns, dtype=torch.long, device=self.device)
        for col in range(self.n_columns):
            d = int(self._degree[col].item())
            direct_degree = 0
            for target in self._neighbor_ids[col, :d].tolist():
                if int(target) >= 0 and self.hamming_distance(col, int(target)) == 1:
                    direct_degree += 1
            self._direct_degree[col] = direct_degree
            self._shortcut_degree[col] = max(0, d - direct_degree)

    def _build_neighbors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build neighbor lists via bit-flip in O(N·d). No O(N²) distance matrix.

        Returns:
            neighbor_ids: [n_columns, max_degree] — padded neighbor indices
            neighbor_weights: [n_columns, max_degree] — normalized weights
            degree: [n_columns] — actual degree per node
            direct_degree: [n_columns] — active one-hop hypercube neighbors
            shortcut_degree: [n_columns] — effective long-range shortcuts per node
        """
        dim = self.dim
        n = self.n_columns
        max_degree = self.target_degree

        neighbor_ids = torch.full((n, max_degree), -1, dtype=torch.long, device=self.device)
        neighbor_weights = torch.zeros(n, max_degree, device=self.device)
        degree = torch.zeros(n, dtype=torch.long, device=self.device)
        direct_degree = torch.zeros(n, dtype=torch.long, device=self.device)
        shortcut_degree = torch.zeros(n, dtype=torch.long, device=self.device)

        # Hypercube neighbors: flip each bit
        for col in range(n):
            idx = 0
            for d in range(dim):
                neighbor = col ^ (1 << d)
                if neighbor < n:  # Only connect to active columns
                    if idx < max_degree:
                        neighbor_ids[col, idx] = neighbor
                    idx += 1
            capped_degree = min(idx, max_degree)
            degree[col] = capped_degree
            direct_degree[col] = capped_degree

        # Deterministic long-range shortcuts with degree-deficit compensation.
        if self.shortcuts_per_node > 0 and max_degree > 0:
            for col in range(n):
                base_degree = int(direct_degree[col].item())
                shortcut_budget = self._effective_shortcut_budget(base_degree)
                if shortcut_budget <= 0:
                    continue
                existing = set(neighbor_ids[col, :degree[col]].tolist())
                existing.add(col)
                added = 0
                for mask in self._shortcut_masks:
                    target = col ^ mask
                    if target >= n or target in existing:
                        continue
                    idx = base_degree + added
                    if idx >= max_degree:
                        break
                    neighbor_ids[col, idx] = target
                    existing.add(target)
                    added += 1
                    if added >= shortcut_budget:
                        break
                degree[col] = min(max_degree, base_degree + added)
                shortcut_degree[col] = max(0, int(degree[col].item()) - base_degree)

        # Calibrate long-range edge weights row-by-row so over-augmented
        # rows do not let long-range mass dominate direct hypercube support.
        for col in range(n):
            d = int(degree[col].item())
            if d <= 0:
                continue
            sources = [int(v) for v in neighbor_ids[col, :d].tolist()]
            weights = self.structural_row_weights(col, sources)
            if weights:
                neighbor_weights[col, :d] = torch.tensor(
                    weights,
                    device=self.device,
                    dtype=neighbor_weights.dtype,
                )

        return neighbor_ids, neighbor_weights, degree, direct_degree, shortcut_degree

    @property
    def avg_degree(self) -> float:
        """Average number of active neighbors per column."""
        return float(self._degree.float().mean().item())

    @property
    def max_degree(self) -> int:
        """Target maximum total degree after shortcut compensation."""
        return self.target_degree

    def neighbors(self, column_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (neighbor_ids, weights) for a column (only valid entries)."""
        d = int(self._degree[column_id].item())
        return self._neighbor_ids[column_id, :d], self._neighbor_weights[column_id, :d]

    def hamming_distance(self, col_a: int, col_b: int) -> int:
        """Hamming distance (number of differing bits) between two columns."""
        return bin(col_a ^ col_b).count("1")

    def is_neighbor(self, col_a: int, col_b: int) -> bool:
        """Check if two columns are direct hypercube neighbors."""
        d = int(self._degree[col_a].item())
        return col_b in self._neighbor_ids[col_a, :d].tolist()

    def two_hop_neighbors(self, column_id: int) -> set[int]:
        """Return set of columns reachable in exactly 2 hops."""
        one_hop = set(self.neighbors(column_id)[0].tolist())
        two_hop: set[int] = set()
        for n1 in one_hop:
            if 0 <= n1 < self.n_columns:
                n1_neighbors = set(self.neighbors(n1)[0].tolist())
                two_hop.update(n1_neighbors)
        two_hop -= one_hop
        two_hop.discard(column_id)
        return two_hop

    def topology_stats(self) -> dict[str, Any]:
        """Summary statistics for the topology."""
        degrees = self._degree[:self.n_columns].float()
        effective_shortcuts = self._shortcut_degree[:self.n_columns].float()
        shortcut_distances: list[int] = []
        long_range_weights: list[float] = []
        long_range_weight_shares: list[float] = []
        for col in range(self.n_columns):
            d = int(self._degree[col].item())
            long_range_share = 0.0
            if d > 0:
                row_sources = [int(v) for v in self._neighbor_ids[col, :d].tolist()]
                row_weights = [float(v) for v in self._neighbor_weights[col, :d].tolist()]
                direct_degree = int(self._direct_degree[col].item())
                long_range_count = max(0, d - direct_degree)
                raw_long_range_weight = self._long_range_raw_weight(
                    direct_degree=direct_degree,
                    long_range_count=long_range_count,
                )
                if long_range_count > 0:
                    long_range_weights.append(raw_long_range_weight)
                for target, weight in zip(row_sources, row_weights):
                    if target < 0:
                        continue
                    distance = self.hamming_distance(col, int(target))
                    if distance > 1:
                        shortcut_distances.append(distance)
                        long_range_share += float(weight)
                if long_range_count > 0:
                    long_range_weight_shares.append(long_range_share)
        return {
            "dim": self.dim,
            "n_vertices": self.n_vertices,
            "n_columns": self.n_columns,
            "avg_degree": float(degrees.mean().item()),
            "degree_std": float(degrees.std(unbiased=False).item()) if self.n_columns > 0 else 0.0,
            "min_degree": int(degrees.min().item()) if self.n_columns > 0 else 0,
            "max_degree": int(degrees.max().item()) if self.n_columns > 0 else 0,
            "shortcuts_per_node": self.shortcuts_per_node,
            "shortcut_strategy": self.shortcut_strategy,
            "shortcut_budget_policy": self.shortcut_budget_policy,
            "long_range_weight_policy": self.long_range_weight_policy,
            "long_range_mass_ratio": self.long_range_mass_ratio,
            "target_degree": self.target_degree,
            "shortcut_edges": int(len(shortcut_distances)),
            "avg_effective_shortcuts_per_node": float(effective_shortcuts.mean().item()) if self.n_columns > 0 else 0.0,
            "min_effective_shortcuts_per_node": int(effective_shortcuts.min().item()) if self.n_columns > 0 else 0,
            "max_effective_shortcuts_per_node": int(effective_shortcuts.max().item()) if self.n_columns > 0 else 0,
            "avg_long_range_raw_weight": float(sum(long_range_weights) / max(1, len(long_range_weights))),
            "min_long_range_raw_weight": float(min(long_range_weights, default=0.0)),
            "max_long_range_raw_weight": float(max(long_range_weights, default=0.0)),
            "avg_long_range_weight_share": float(sum(long_range_weight_shares) / max(1, len(long_range_weight_shares))),
            "max_long_range_weight_share": float(max(long_range_weight_shares, default=0.0)),
            "avg_shortcut_hamming_distance": float(sum(shortcut_distances) / max(1, len(shortcut_distances))),
            "max_shortcut_hamming_distance": int(max(shortcut_distances, default=0)),
            "total_edges": int(degrees.sum().item()),
            "sparsity_ratio": float(degrees.sum().item()) / max(1, self.n_columns * (self.n_columns - 1)),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "n_columns": self.n_columns,
            "dim": self.dim,
            "shortcuts_per_node": self.shortcuts_per_node,
            "shortcut_strategy": self.shortcut_strategy,
            "shortcut_budget_policy": self.shortcut_budget_policy,
            "long_range_weight_policy": self.long_range_weight_policy,
            "long_range_mass_ratio": self.long_range_mass_ratio,
            "target_degree": self.target_degree,
            "neighbor_ids": self._neighbor_ids.cpu(),
            "neighbor_weights": self._neighbor_weights.cpu(),
            "degree": self._degree.cpu(),
            "direct_degree": self._direct_degree.cpu(),
            "shortcut_degree": self._shortcut_degree.cpu(),
        }

    def load_state_dict(self, snapshot: dict[str, Any]) -> None:
        self.shortcut_strategy = str(snapshot.get("shortcut_strategy", self.shortcut_strategy))
        self.shortcut_budget_policy = str(snapshot.get("shortcut_budget_policy", self.shortcut_budget_policy))
        self.long_range_weight_policy = str(snapshot.get("long_range_weight_policy", self.long_range_weight_policy))
        self.long_range_mass_ratio = float(snapshot.get("long_range_mass_ratio", self.long_range_mass_ratio))
        self.target_degree = int(snapshot.get("target_degree", self.target_degree))
        nids = snapshot.get("neighbor_ids")
        if isinstance(nids, torch.Tensor):
            self._neighbor_ids = nids.to(self.device)
        nw = snapshot.get("neighbor_weights")
        if isinstance(nw, torch.Tensor):
            self._neighbor_weights = nw.to(self.device)
        deg = snapshot.get("degree")
        if isinstance(deg, torch.Tensor):
            self._degree = deg.to(self.device)
        direct_degree = snapshot.get("direct_degree")
        shortcut_degree = snapshot.get("shortcut_degree")
        if isinstance(direct_degree, torch.Tensor) and isinstance(shortcut_degree, torch.Tensor):
            self._direct_degree = direct_degree.to(self.device)
            self._shortcut_degree = shortcut_degree.to(self.device)
        else:
            self._reconstruct_degree_breakdown()


def _normalize(x: torch.Tensor) -> torch.Tensor:
    s = x.sum()
    if s <= 0.0:
        return x
    return x / s


class HypercubeBindingLayer:
    """Binding layer using hypercube-structured sparse connectivity.

    Drop-in replacement for BindingLayer / SpatialBindingLayer.
    Each column connects to its ``dim`` hypercube neighbors (plus optional
    shortcuts). Binding operations are O(N·d) instead of O(N²).

    Key differences from SpatialBindingLayer:
    - Topology is hypercube (bit-flip) instead of 2D grid
    - Regular degree: every interior node has exactly ``dim`` neighbors
    - grow_binding() constrained to 2-hop hypercube paths only
    - No O(N²) distance matrix — pure O(N·d) construction
    """

    def __init__(
        self,
        n_columns: int,
        device: torch.device,
        topology: HypercubeTopology | None = None,
        shortcuts_per_node: int = 0,
        threshold: float = 0.02,
        gain_strength: float = 0.35,
        tau_binding: float = 6.0,
        stp_u_inc: float = 0.15,
        stp_tau_f: float = 12.0,
        stp_tau_d: float = 4.0,
        pv_threshold: float = 0.12,
        pv_gain: float = 0.60,
        association_lr: float = 0.10,
        association_decay: float = 0.995,
    ) -> None:
        self.n_columns = int(n_columns)
        self.device = device
        self.threshold = float(threshold)
        self.gain_strength = float(gain_strength)
        self.tau_binding = float(tau_binding)
        self.stp_u_inc = float(stp_u_inc)
        self.stp_tau_f = float(stp_tau_f)
        self.stp_tau_d = float(stp_tau_d)
        self.pv_threshold = float(pv_threshold)
        self.pv_gain = float(pv_gain)
        self.association_lr = float(association_lr)
        self.association_decay = float(association_decay)

        # Use provided topology or create one
        if topology is not None:
            self.topology = topology
        else:
            self.topology = HypercubeTopology(
                n_columns, shortcuts_per_node=shortcuts_per_node, device=device,
            )

        # State tensors — per-column
        self.binding_state = torch.zeros(n_columns, device=device)
        self.coincidence_trace = torch.zeros(n_columns, device=device)
        self.facilitation = torch.zeros(n_columns, device=device)
        self.resources = torch.ones(n_columns, device=device)
        self.binding_usage = torch.zeros(n_columns, device=device)
        self.pv_inhibition = torch.tensor(0.0, device=device)

        # Hub tracking: sustained high-usage columns are expressed through
        # structural outreach rather than a direct signal multiplier. The
        # maintained policy targets the top 5% of columns by activation EMA
        # and gives them up to ~2x outgoing structural connections.
        self._hub_activation_ema = torch.zeros(n_columns, device=device)
        self._hub_ema_alpha = 0.01
        self._hub_target_fraction = 0.05
        self._hub_strength = torch.zeros(n_columns, device=device)
        self._hub_connection_multiplier = torch.ones(n_columns, device=device)
        self._hub_extra_connections = torch.zeros(n_columns, dtype=torch.long, device=device)
        self._hub_mask = torch.zeros(n_columns, dtype=torch.bool, device=device)

        self._base_neighbor_ids = torch.empty((n_columns, 0), dtype=torch.long, device=device)
        self._base_degree = torch.zeros(n_columns, dtype=torch.long, device=device)
        self._base_neighbor_weights = torch.empty((n_columns, 0), device=device)
        self._base_outgoing_targets: list[list[int]] = [[] for _ in range(n_columns)]
        self._hub_target_candidates: list[list[int]] = [[] for _ in range(n_columns)]

        # Sparse connectivity from topology
        self._sync_base_connectivity_from_topology()
        self.neighbor_ids = self._base_neighbor_ids.clone()
        self.degree = self._base_degree.clone()
        self.neighbor_weights = self._base_neighbor_weights.clone()
        self.learned_weights = self.neighbor_weights.clone()

        # Compatibility attributes
        self.n_bindings = n_columns
        self.fan_in = self.topology.dim

    def _outgoing_targets_from_rows(
        self,
        neighbor_ids: torch.Tensor,
        degree: torch.Tensor,
    ) -> list[list[int]]:
        outgoing: list[list[int]] = [[] for _ in range(self.n_columns)]
        for target in range(self.n_columns):
            d = int(degree[target].item())
            for source in neighbor_ids[target, :d].tolist():
                source_id = int(source)
                if 0 <= source_id < self.n_columns:
                    outgoing[source_id].append(target)
        return outgoing

    def _edge_weight_map(
        self,
        neighbor_ids: torch.Tensor,
        degree: torch.Tensor,
        weights: torch.Tensor,
    ) -> list[dict[int, float]]:
        row_maps: list[dict[int, float]] = [dict() for _ in range(self.n_columns)]
        for row in range(self.n_columns):
            d = int(degree[row].item())
            for source, weight in zip(neighbor_ids[row, :d].tolist(), weights[row, :d].tolist()):
                source_id = int(source)
                if source_id >= 0:
                    row_maps[row][source_id] = float(weight)
        return row_maps

    def _sync_base_connectivity_from_topology(self) -> None:
        max_deg = self.topology.max_degree
        self._base_neighbor_ids = self.topology._neighbor_ids[:self.n_columns, :max_deg].to(self.device).clone()
        self._base_degree = self.topology._degree[:self.n_columns].to(self.device).clone()
        self._base_neighbor_weights = self.topology._neighbor_weights[:self.n_columns, :max_deg].to(self.device).clone()
        self._base_outgoing_targets = self._outgoing_targets_from_rows(self._base_neighbor_ids, self._base_degree)
        self._hub_target_candidates = []
        for source in range(self.n_columns):
            existing = set(self._base_outgoing_targets[source])
            existing.add(source)
            candidates: list[int] = []
            for mask in self.topology._shortcut_masks:
                target = source ^ mask
                if target >= self.n_columns or target in existing:
                    continue
                candidates.append(target)
                existing.add(target)
            self._hub_target_candidates.append(candidates)

    def _refresh_structural_hub_connectivity(
        self,
        preserved_weights: list[dict[int, float]] | None = None,
    ) -> None:
        if preserved_weights is None:
            preserved_weights = self._edge_weight_map(self.neighbor_ids, self.degree, self.learned_weights)

        row_sources: list[list[int]] = []
        row_existing: list[set[int]] = []
        for row in range(self.n_columns):
            d = int(self._base_degree[row].item())
            sources = [int(v) for v in self._base_neighbor_ids[row, :d].tolist()]
            row_sources.append(sources)
            row_existing.append(set(sources))

        for source in range(self.n_columns):
            extra_connections = int(self._hub_extra_connections[source].item())
            if extra_connections <= 0:
                continue
            for target in self._hub_target_candidates[source][:extra_connections]:
                if source in row_existing[target]:
                    continue
                row_sources[target].append(source)
                row_existing[target].add(source)

        max_deg = max((len(sources) for sources in row_sources), default=0)
        neighbor_ids = torch.full((self.n_columns, max_deg), -1, dtype=torch.long, device=self.device)
        neighbor_weights = torch.zeros(self.n_columns, max_deg, device=self.device)
        learned_weights = torch.zeros(self.n_columns, max_deg, device=self.device)
        degree = torch.zeros(self.n_columns, dtype=torch.long, device=self.device)

        for row in range(self.n_columns):
            sources = row_sources[row]
            d = len(sources)
            degree[row] = d
            if d <= 0:
                continue
            normalized_structural = self.topology.structural_row_weights(row, sources)
            learned_row = torch.tensor(
                [preserved_weights[row].get(source, normalized_structural[idx]) for idx, source in enumerate(sources)],
                device=self.device,
                dtype=neighbor_weights.dtype,
            )
            if float(learned_row.sum().item()) > 0.0:
                learned_row = learned_row / learned_row.sum().clamp(min=1e-8)
            else:
                learned_row = torch.tensor(normalized_structural, device=self.device, dtype=neighbor_weights.dtype)
            neighbor_ids[row, :d] = torch.tensor(sources, device=self.device, dtype=torch.long)
            neighbor_weights[row, :d] = torch.tensor(
                normalized_structural,
                device=self.device,
                dtype=neighbor_weights.dtype,
            )
            learned_weights[row, :d] = learned_row

        self.neighbor_ids = neighbor_ids
        self.neighbor_weights = neighbor_weights
        self.learned_weights = learned_weights
        self.degree = degree

    def _refresh_hub_profile(
        self,
        *,
        preserved_weights: list[dict[int, float]] | None = None,
        force_structure_refresh: bool = False,
    ) -> None:
        """Refresh structural hub outreach from the current activation EMA."""
        new_hub_strength = torch.zeros_like(self._hub_strength)
        new_hub_mask = torch.zeros_like(self._hub_mask)
        new_hub_extra_connections = torch.zeros_like(self._hub_extra_connections)
        new_hub_connection_multiplier = torch.ones_like(self._hub_connection_multiplier)

        if self.n_columns > 0:
            max_ema = float(self._hub_activation_ema.max().item())
            if max_ema > 1e-8:
                hub_count = min(self.n_columns, max(1, math.ceil(self._hub_target_fraction * self.n_columns)))
                top_values, top_indices = torch.topk(self._hub_activation_ema, hub_count)
                selected_strength = torch.clamp(top_values / max_ema, min=0.0, max=1.0)
                new_hub_strength[top_indices] = selected_strength
                new_hub_mask[top_indices] = True
                for idx, source in enumerate(top_indices.tolist()):
                    base_out_degree = len(self._base_outgoing_targets[int(source)])
                    if base_out_degree <= 0:
                        continue
                    extra_cap = len(self._hub_target_candidates[int(source)])
                    requested_extra = int(math.ceil(base_out_degree * float(selected_strength[idx].item())))
                    actual_extra = min(extra_cap, requested_extra)
                    new_hub_extra_connections[int(source)] = actual_extra
                    new_hub_connection_multiplier[int(source)] = (
                        float(base_out_degree + actual_extra) / float(max(1, base_out_degree))
                    )

        profile_changed = (
            force_structure_refresh
            or not torch.equal(new_hub_mask, self._hub_mask)
            or not torch.equal(new_hub_extra_connections, self._hub_extra_connections)
        )

        self._hub_strength = new_hub_strength
        self._hub_mask = new_hub_mask
        self._hub_extra_connections = new_hub_extra_connections
        self._hub_connection_multiplier = new_hub_connection_multiplier

        if profile_changed:
            self._refresh_structural_hub_connectivity(preserved_weights=preserved_weights)

    def hub_stats(self) -> dict[str, Any]:
        hub_indices = torch.nonzero(self._hub_mask, as_tuple=False).flatten().tolist()
        return {
            "hub_target_fraction": self._hub_target_fraction,
            "hub_count": len(hub_indices),
            "hub_indices": [int(idx) for idx in hub_indices],
            "hub_extra_edges": int(self._hub_extra_connections.sum().item()) if self.n_columns > 0 else 0,
            "max_hub_extra_connections": int(self._hub_extra_connections.max().item()) if self.n_columns > 0 else 0,
            "max_hub_strength": float(self._hub_strength.max().item()) if self.n_columns > 0 else 0.0,
            "mean_hub_strength": float(self._hub_strength.mean().item()) if self.n_columns > 0 else 0.0,
            "max_hub_connection_multiplier": float(self._hub_connection_multiplier.max().item()) if self.n_columns > 0 else 1.0,
            "mean_hub_connection_multiplier": float(self._hub_connection_multiplier.mean().item()) if self.n_columns > 0 else 1.0,
        }

    def _sparse_drive(self, signal: torch.Tensor, *, already_normalized: bool = False) -> torch.Tensor:
        """Compute local drive from hypercube neighbors. O(N*d).

        Hub influence is expressed through the maintained structural adjacency,
        not a direct source-amplitude multiplier.
        """
        normed = signal if already_normalized else _normalize(signal.to(self.device))
        # Clamp neighbor_ids: replace -1 (padding) with 0, then zero out via weights
        safe_ids = self.neighbor_ids.clamp(min=0)
        neighbor_acts = normed[safe_ids]  # [n_columns, max_degree]
        drive = (neighbor_acts * self.learned_weights).sum(dim=1)
        return drive

    def _update_stp(self, drive: torch.Tensor) -> torch.Tensor:
        """Short-term plasticity: facilitation + depression."""
        self.facilitation = torch.clamp(
            self.facilitation * (1.0 - 1.0 / self.stp_tau_f)
            + self.stp_u_inc * drive * (1.0 - self.facilitation),
            min=0.0, max=1.0,
        )
        release = torch.clamp(self.facilitation * self.resources * drive, min=0.0)
        self.resources = torch.clamp(
            self.resources + (1.0 - self.resources) / self.stp_tau_d - release,
            min=0.0, max=1.0,
        )
        return release

    def bind(
        self,
        context_prediction: torch.Tensor,
        assembly: torch.Tensor,
        update_weights: bool = True,
    ) -> tuple[torch.Tensor, float]:
        """Hypercube binding: coincidence detection via bit-flip neighbors.

        Same interface as BindingLayer.bind() / SpatialBindingLayer.bind().
        """
        context = _normalize(context_prediction.to(self.device))
        current = _normalize(assembly.to(self.device))

        if context.sum() <= 0.0 or current.sum() <= 0.0:
            self.coincidence_trace *= max(0.0, 1.0 - 1.0 / self.tau_binding)
            self.binding_state.zero_()
            return self.binding_state, 0.0

        # Local drive from hypercube neighbors
        ctx_drive = self._sparse_drive(context, already_normalized=True)
        cur_drive = self._sparse_drive(current, already_normalized=True)

        # Coincidence: both context and current active in neighborhood
        joint_drive = torch.min(ctx_drive, cur_drive)

        # STP
        release = self._update_stp(joint_drive)

        # Coincidence trace with exponential decay
        decay = max(0.0, 1.0 - 1.0 / self.tau_binding)
        self.coincidence_trace = self.coincidence_trace * decay + release

        # PV inhibition
        mean_activity = self.coincidence_trace.mean()
        if mean_activity > self.pv_threshold:
            self.pv_inhibition = self.pv_gain * (mean_activity - self.pv_threshold)
        else:
            self.pv_inhibition = torch.tensor(0.0, device=self.device)

        # Binding output
        output = torch.relu(self.coincidence_trace - self.threshold - self.pv_inhibition)
        self.binding_usage += (output > 0.0).float()

        # Column support: weighted sum of neighbor outputs
        safe_ids = self.neighbor_ids.clamp(min=0)
        neighbor_outputs = output[safe_ids]  # [n, max_degree]
        support = (neighbor_outputs * self.learned_weights).sum(dim=1)
        self.binding_state = _normalize(support)

        strength = float(output.sum().item())

        # Hebbian weight update
        if update_weights and strength > 0.0:
            active_mask = output > 0.0
            if active_mask.any():
                target_neighbors = current[safe_ids]  # [n_columns, max_degree]
                update = (self.association_lr
                          * output.unsqueeze(1)
                          * target_neighbors)
                self.learned_weights = torch.where(
                    active_mask.unsqueeze(1),
                    torch.clamp(
                        self.learned_weights * self.association_decay + update,
                        min=0.0, max=1.0,
                    ),
                    self.learned_weights,
                )
                # Re-zero padded positions
                pad_mask = self.neighbor_ids < 0
                self.learned_weights[pad_mask] = 0.0
                # Renormalize per-column
                weight_sums = self.learned_weights.sum(dim=1, keepdim=True).clamp(min=1e-8)
                self.learned_weights = self.learned_weights / weight_sums

        # Update hub activation tracking after the current bind step has fully
        # consumed the existing structural graph. The refreshed hub structure
        # applies to the next bind step.
        active_float = (output > 0.0).float()
        self._hub_activation_ema = (
            self._hub_ema_alpha * active_float
            + (1.0 - self._hub_ema_alpha) * self._hub_activation_ema
        )
        self._refresh_hub_profile()

        return self.binding_state, strength

    def binding_prediction(self, context_prediction: torch.Tensor) -> torch.Tensor:
        """Predict column activation from hypercube binding context."""
        context = _normalize(context_prediction.to(self.device))
        if context.sum() <= 0.0:
            return torch.zeros(self.n_columns, device=self.device)
        if self.binding_usage.max() <= 1e-6:
            return torch.zeros(self.n_columns, device=self.device)
        predicted = self._sparse_drive(context, already_normalized=True) * self.binding_usage
        return _normalize(predicted)

    def modulation_gain(self, context_prediction: torch.Tensor) -> torch.Tensor:
        return self.modulation_gain_for_context(context_prediction)

    def modulation_gain_for_context(self, context_prediction: torch.Tensor) -> torch.Tensor:
        prediction = self.binding_prediction(context_prediction)
        if prediction.sum() <= 0.0:
            return torch.ones(self.n_columns, device=self.device)
        centered = prediction - prediction.mean()
        scale = torch.clamp(centered.abs().max(), min=1e-8)
        gain = 1.0 + self.gain_strength * (centered / scale)
        return torch.clamp(gain, min=0.70, max=1.35)

    def grow_binding(self, high_correlation_columns: list[tuple[int, int, float]]) -> int:
        """Grow binding constrained to 2-hop hypercube neighbors only.

        Unlike dense BindingLayer (which adds new binding neurons) or spatial
        (which only strengthens existing 1-hop), hypercube grow_binding can
        extend to 2-hop neighbors — columns that share a common hypercube
        neighbor. This preserves the topology's structural sparsity.
        """
        strengthened = 0
        for col_a, col_b, corr in high_correlation_columns:
            a, b = int(col_a), int(col_b)
            if corr <= 0.7 or a == b:
                continue
            if not (0 <= a < self.n_columns and 0 <= b < self.n_columns):
                continue

            # Check 1-hop: direct neighbor
            d_a = int(self.degree[a].item())
            nids_a = self.neighbor_ids[a, :d_a].tolist()
            if b in nids_a:
                idx = nids_a.index(b)
                self.learned_weights[a, idx] = min(1.0, self.learned_weights[a, idx] + 0.1 * corr)
                strengthened += 1
                # Reciprocal
                d_b = int(self.degree[b].item())
                nids_b = self.neighbor_ids[b, :d_b].tolist()
                if a in nids_b:
                    idx_b = nids_b.index(a)
                    self.learned_weights[b, idx_b] = min(1.0, self.learned_weights[b, idx_b] + 0.1 * corr)
                    strengthened += 1
                continue

            # Check 2-hop: shares a common neighbor
            two_hop = self.topology.two_hop_neighbors(a)
            if b in two_hop:
                # Strengthen all intermediate edges on shortest 2-hop path
                d_b = int(self.degree[b].item())
                nids_b_set = set(self.neighbor_ids[b, :d_b].tolist())
                common = set(nids_a) & nids_b_set
                for mid in common:
                    if mid < 0:
                        continue
                    # Strengthen a→mid
                    if mid in nids_a:
                        idx = nids_a.index(mid)
                        self.learned_weights[a, idx] = min(
                            1.0, self.learned_weights[a, idx] + 0.05 * corr
                        )
                        strengthened += 1
                    # Strengthen mid→b
                    d_mid = int(self.degree[mid].item())
                    nids_mid = self.neighbor_ids[mid, :d_mid].tolist()
                    if b in nids_mid:
                        idx_mid = nids_mid.index(b)
                        self.learned_weights[mid, idx_mid] = min(
                            1.0, self.learned_weights[mid, idx_mid] + 0.05 * corr
                        )
                        strengthened += 1

        # Renormalize
        if strengthened > 0:
            pad_mask = self.neighbor_ids < 0
            self.learned_weights[pad_mask] = 0.0
            weight_sums = self.learned_weights.sum(dim=1, keepdim=True).clamp(min=1e-8)
            self.learned_weights = self.learned_weights / weight_sums

        return strengthened

    def reset_state(self) -> None:
        self.binding_state.zero_()
        self.coincidence_trace.zero_()
        self.facilitation.zero_()
        self.resources.fill_(1.0)
        self.pv_inhibition.zero_()
        self._hub_activation_ema.zero_()
        self._hub_strength.zero_()
        self._hub_connection_multiplier.fill_(1.0)
        self._hub_extra_connections.zero_()
        self._hub_mask.zero_()
        self._refresh_structural_hub_connectivity()

    def state_dict(self) -> dict[str, Any]:
        return {
            "n_bindings": self.n_bindings,
            "fan_in": self.fan_in,
            "binding_state": self.binding_state.detach().clone().cpu(),
            "coincidence_trace": self.coincidence_trace.detach().clone().cpu(),
            "facilitation": self.facilitation.detach().clone().cpu(),
            "resources": self.resources.detach().clone().cpu(),
            "binding_usage": self.binding_usage.detach().clone().cpu(),
            "pv_inhibition": self.pv_inhibition.detach().clone().cpu(),
            "learned_weights": self.learned_weights.detach().clone().cpu(),
            "neighbor_ids": self.neighbor_ids.detach().clone().cpu(),
            "degree": self.degree.detach().clone().cpu(),
            "hub_activation_ema": self._hub_activation_ema.detach().clone().cpu(),
            "hub_extra_connections": self._hub_extra_connections.detach().clone().cpu(),
            "hub_connection_multiplier": self._hub_connection_multiplier.detach().clone().cpu(),
            "topology_state": self.topology.state_dict(),
            # Compatibility keys for BindingLayer format
            "connectivity": torch.eye(self.n_columns),
            "output_weights": torch.eye(self.n_columns),
            "binding_outputs": self.coincidence_trace.detach().clone().cpu(),
        }

    def load_state_dict(self, snapshot: dict[str, Any]) -> None:
        topo_state = snapshot.get("topology_state")
        if isinstance(topo_state, dict):
            self.topology.load_state_dict(topo_state)
        self._sync_base_connectivity_from_topology()
        self.neighbor_ids = self._base_neighbor_ids.clone()
        self.degree = self._base_degree.clone()
        self.neighbor_weights = self._base_neighbor_weights.clone()
        self.learned_weights = self.neighbor_weights.clone()

        snapshot_weight_map: list[dict[int, float]] | None = None
        learned = snapshot.get("learned_weights")
        neighbor_ids_snapshot = snapshot.get("neighbor_ids")
        degree_snapshot = snapshot.get("degree")
        if (
            isinstance(learned, torch.Tensor)
            and isinstance(neighbor_ids_snapshot, torch.Tensor)
            and isinstance(degree_snapshot, torch.Tensor)
        ):
            snapshot_weight_map = self._edge_weight_map(
                neighbor_ids_snapshot.to(self.device),
                degree_snapshot.to(self.device),
                learned.to(self.device),
            )

        for attr, default_fn in [
            ("binding_state", lambda: torch.zeros(self.n_columns, device=self.device)),
            ("coincidence_trace", lambda: torch.zeros(self.n_columns, device=self.device)),
            ("facilitation", lambda: torch.zeros(self.n_columns, device=self.device)),
            ("resources", lambda: torch.ones(self.n_columns, device=self.device)),
            ("binding_usage", lambda: torch.zeros(self.n_columns, device=self.device)),
        ]:
            value = snapshot.get(attr)
            if isinstance(value, torch.Tensor):
                setattr(self, attr, value.to(self.device))
            else:
                setattr(self, attr, default_fn())

        pv = snapshot.get("pv_inhibition")
        if isinstance(pv, torch.Tensor):
            self.pv_inhibition = pv.to(self.device)
        else:
            self.pv_inhibition = torch.tensor(0.0, device=self.device)

        hub_ema = snapshot.get("hub_activation_ema")
        if isinstance(hub_ema, torch.Tensor):
            self._hub_activation_ema = hub_ema.to(self.device)
        else:
            self._hub_activation_ema.zero_()
        self._refresh_hub_profile(
            preserved_weights=snapshot_weight_map,
            force_structure_refresh=True,
        )
