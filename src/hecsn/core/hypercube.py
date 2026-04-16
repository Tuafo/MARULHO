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

    Optional small-world shortcuts add a fixed number of random long-range
    edges per node (default 0) to reduce average path length.
    """

    def __init__(
        self,
        n_columns: int,
        shortcuts_per_node: int = 0,
        device: torch.device | None = None,
    ) -> None:
        self.n_columns = int(n_columns)
        self.device = device or torch.device("cpu")
        self.shortcuts_per_node = int(shortcuts_per_node)

        # Compute hypercube dimension: smallest d such that 2^d >= n_columns
        self.dim = max(1, math.ceil(math.log2(max(2, n_columns))))
        self.n_vertices = 1 << self.dim  # 2^dim

        # Which vertices are active (have a real column assigned)
        self.active_mask = torch.zeros(self.n_vertices, dtype=torch.bool, device=self.device)
        self.active_mask[:self.n_columns] = True

        # Build neighbor structure
        self._neighbor_ids, self._neighbor_weights, self._degree = self._build_neighbors()

    def _build_neighbors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build neighbor lists via bit-flip in O(N·d). No O(N²) distance matrix.

        Returns:
            neighbor_ids: [n_columns, max_degree] — padded neighbor indices
            neighbor_weights: [n_columns, max_degree] — normalized weights
            degree: [n_columns] — actual degree per node
        """
        dim = self.dim
        n = self.n_columns
        max_degree = dim + self.shortcuts_per_node

        neighbor_ids = torch.full((n, max_degree), -1, dtype=torch.long, device=self.device)
        neighbor_weights = torch.zeros(n, max_degree, device=self.device)
        degree = torch.zeros(n, dtype=torch.long, device=self.device)

        # Hypercube neighbors: flip each bit
        for col in range(n):
            idx = 0
            for d in range(dim):
                neighbor = col ^ (1 << d)
                if neighbor < n:  # Only connect to active columns
                    neighbor_ids[col, idx] = neighbor
                    neighbor_weights[col, idx] = 1.0
                    idx += 1
            degree[col] = idx

        # Small-world shortcuts (fixed count per node, not percentage)
        if self.shortcuts_per_node > 0:
            for col in range(n):
                existing = set(neighbor_ids[col, :degree[col]].tolist())
                existing.add(col)
                added = 0
                attempts = 0
                while added < self.shortcuts_per_node and attempts < self.shortcuts_per_node * 10:
                    target = int(torch.randint(0, n, (1,)).item())
                    attempts += 1
                    if target not in existing:
                        idx = int(degree[col].item()) + added
                        if idx < max_degree:
                            neighbor_ids[col, idx] = target
                            # Shortcuts get lower weight than hypercube edges
                            neighbor_weights[col, idx] = 0.5
                            existing.add(target)
                            added += 1
                degree[col] += added

        # Normalize weights per node
        for col in range(n):
            d = int(degree[col].item())
            if d > 0:
                total = neighbor_weights[col, :d].sum()
                if total > 0:
                    neighbor_weights[col, :d] /= total

        return neighbor_ids, neighbor_weights, degree

    @property
    def avg_degree(self) -> float:
        """Average number of active neighbors per column."""
        return float(self._degree.float().mean().item())

    @property
    def max_degree(self) -> int:
        """Maximum degree (dim + shortcuts)."""
        return self.dim + self.shortcuts_per_node

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
        return {
            "dim": self.dim,
            "n_vertices": self.n_vertices,
            "n_columns": self.n_columns,
            "avg_degree": float(degrees.mean().item()),
            "min_degree": int(degrees.min().item()),
            "max_degree": int(degrees.max().item()),
            "shortcuts_per_node": self.shortcuts_per_node,
            "total_edges": int(degrees.sum().item()),
            "sparsity_ratio": float(degrees.sum().item()) / max(1, self.n_columns * (self.n_columns - 1)),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "n_columns": self.n_columns,
            "dim": self.dim,
            "shortcuts_per_node": self.shortcuts_per_node,
            "neighbor_ids": self._neighbor_ids.cpu(),
            "neighbor_weights": self._neighbor_weights.cpu(),
            "degree": self._degree.cpu(),
        }

    def load_state_dict(self, snapshot: dict[str, Any]) -> None:
        nids = snapshot.get("neighbor_ids")
        if isinstance(nids, torch.Tensor):
            self._neighbor_ids = nids.to(self.device)
        nw = snapshot.get("neighbor_weights")
        if isinstance(nw, torch.Tensor):
            self._neighbor_weights = nw.to(self.device)
        deg = snapshot.get("degree")
        if isinstance(deg, torch.Tensor):
            self._degree = deg.to(self.device)


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

        # Sparse connectivity from topology
        max_deg = self.topology.max_degree
        self.neighbor_ids = self.topology._neighbor_ids[:n_columns, :max_deg].to(device)
        self.degree = self.topology._degree[:n_columns].to(device)
        self.neighbor_weights = self.topology._neighbor_weights[:n_columns, :max_deg].to(device)
        self.learned_weights = self.neighbor_weights.clone()

        # Compatibility attributes
        self.n_bindings = n_columns
        self.fan_in = self.topology.dim

    def _sparse_drive(self, signal: torch.Tensor, *, already_normalized: bool = False) -> torch.Tensor:
        """Compute local drive from hypercube neighbors. O(N·d)."""
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
            max_deg = self.topology.max_degree
            self.neighbor_ids = self.topology._neighbor_ids[:self.n_columns, :max_deg].to(self.device)
            self.degree = self.topology._degree[:self.n_columns].to(self.device)

        learned = snapshot.get("learned_weights")
        if isinstance(learned, torch.Tensor):
            self.learned_weights = learned.to(self.device)

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
