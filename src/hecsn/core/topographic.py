"""Topographic column organization: 2D spatial grid with local binding.

Implements biologically-inspired topographic organization where columns
are arranged on a 2D grid. Semantically similar concepts self-organize
to occupy nearby grid positions via SOM-like neighborhood learning.

Key components:
  - TopographicGrid: 2D flat grid with precomputed neighbor lists
  - SpatialBindingLayer: local binding via grid proximity (replaces dense BindingLayer)

References:
  - Hebb (1949): Cell assemblies and cortical columns
  - Zhou et al. (2025, AAAI): TDSNNs — topographic deep SNNs
  - Gao et al. (2025): SG-SNN — temporal self-organizing maps
  - Dehghani et al. (2025, ICLR): Credit-based deep topographic networks
"""

from __future__ import annotations

import math
from typing import Any

import torch


class TopographicGrid:
    """2D flat grid layout for N columns with precomputed neighbor structure.

    Arranges ``n_columns`` on a rectangular grid of size ``(n_rows, n_cols)``.
    Provides precomputed neighbor lists and Gaussian distance weights for
    efficient local operations.

    The grid is flat (not toroidal) — edge columns have fewer neighbors,
    which is honest about the topology and avoids artificial wrap-around.
    """

    def __init__(
        self,
        n_columns: int,
        k_neighbors: int = 8,
        sigma: float = 1.5,
        device: torch.device | None = None,
    ) -> None:
        self.n_columns = int(n_columns)
        self.k_neighbors = min(int(k_neighbors), n_columns - 1)
        self.sigma = float(sigma)
        self.device = device or torch.device("cpu")

        self.n_rows, self.n_cols = self._compute_grid_dims(n_columns)
        self.grid_positions = self._assign_grid_positions()

        # Precompute distance matrix and neighbor lists
        self._distance_matrix = self._compute_distances()
        self._neighbor_ids, self._neighbor_weights = self._compute_neighbors()

    @staticmethod
    def _compute_grid_dims(n: int) -> tuple[int, int]:
        """Find the most square grid that fits >= n cells."""
        side = int(math.ceil(math.sqrt(n)))
        rows = side
        cols = side
        # Trim rows if we have too many cells
        while rows * cols >= n and (rows - 1) * cols >= n:
            rows -= 1
        return rows, cols

    def _assign_grid_positions(self) -> torch.Tensor:
        """Assign each column a (row, col) position on the grid."""
        positions = torch.zeros(self.n_columns, 2, device=self.device)
        for i in range(self.n_columns):
            positions[i, 0] = float(i // self.n_cols)  # row
            positions[i, 1] = float(i % self.n_cols)   # col
        return positions

    def _compute_distances(self) -> torch.Tensor:
        """Compute pairwise Euclidean distances on the grid."""
        # positions: [n_columns, 2]
        diff = self.grid_positions.unsqueeze(0) - self.grid_positions.unsqueeze(1)
        return torch.sqrt((diff ** 2).sum(dim=-1))  # [n_columns, n_columns]

    def _compute_neighbors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Precompute K nearest neighbors and their Gaussian weights.

        Returns:
            neighbor_ids: [n_columns, k_neighbors] — indices of neighbors
            neighbor_weights: [n_columns, k_neighbors] — Gaussian decay weights
        """
        k = self.k_neighbors
        # Set self-distance to infinity to exclude self
        dist = self._distance_matrix.clone()
        dist.fill_diagonal_(float("inf"))

        # Get K nearest neighbors
        _, indices = torch.topk(dist, k, dim=1, largest=False)  # [n, k]

        # Compute Gaussian weights
        neighbor_dists = torch.gather(self._distance_matrix, 1, indices)
        weights = torch.exp(-neighbor_dists ** 2 / (2.0 * self.sigma ** 2))
        # Normalize weights per column
        weight_sums = weights.sum(dim=1, keepdim=True).clamp(min=1e-8)
        weights = weights / weight_sums

        return indices, weights

    def grid_position(self, column_id: int) -> tuple[int, int]:
        """Return (row, col) grid position for a column."""
        pos = self.grid_positions[column_id]
        return int(pos[0].item()), int(pos[1].item())

    def neighbors(self, column_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (neighbor_ids, weights) for a column."""
        return self._neighbor_ids[column_id], self._neighbor_weights[column_id]

    def distance(self, col_a: int, col_b: int) -> float:
        """Euclidean grid distance between two columns."""
        return float(self._distance_matrix[col_a, col_b].item())

    def neighborhood_kernel(self, center: int, sigma: float | None = None) -> torch.Tensor:
        """Gaussian neighborhood kernel centered on a column.

        Returns: [n_columns] weights with Gaussian decay from center.
        """
        s = sigma if sigma is not None else self.sigma
        dists = self._distance_matrix[center]
        kernel = torch.exp(-dists ** 2 / (2.0 * s ** 2))
        kernel[center] = 1.0  # center gets full weight
        return kernel

    def neighbor_purity(self, prototypes: torch.Tensor, k: int | None = None) -> float:
        """Measure topographic quality: average cosine similarity of grid neighbors.

        Higher values mean that spatially nearby columns have similar prototypes.
        """
        if k is None:
            k = self.k_neighbors
        # Normalize prototypes
        norms = prototypes.norm(dim=1, keepdim=True).clamp(min=1e-8)
        normed = prototypes / norms
        similarities = torch.mm(normed, normed.t())  # [n, n]

        # Average similarity of K nearest neighbors
        total = 0.0
        count = 0
        for i in range(self.n_columns):
            nids = self._neighbor_ids[i, :k]
            total += float(similarities[i, nids].mean().item())
            count += 1
        return total / max(count, 1)

    def topographic_error(self, prototypes: torch.Tensor) -> float:
        """Fraction of columns whose best-matching prototype is NOT a grid neighbor.

        Lower = better topographic organization. 0.0 = perfect.
        """
        norms = prototypes.norm(dim=1, keepdim=True).clamp(min=1e-8)
        normed = prototypes / norms
        similarities = torch.mm(normed, normed.t())
        similarities.fill_diagonal_(-float("inf"))

        best_match = similarities.argmax(dim=1)  # [n_columns]
        errors = 0
        for i in range(self.n_columns):
            bm = int(best_match[i].item())
            if bm not in self._neighbor_ids[i].tolist():
                errors += 1
        return errors / max(self.n_columns, 1)

    def state_dict(self) -> dict[str, Any]:
        return {
            "n_columns": self.n_columns,
            "k_neighbors": self.k_neighbors,
            "sigma": self.sigma,
            "grid_positions": self.grid_positions.cpu(),
        }

    def load_state_dict(self, snapshot: dict[str, Any]) -> None:
        positions = snapshot.get("grid_positions")
        if isinstance(positions, torch.Tensor):
            self.grid_positions = positions.to(self.device)
            self._distance_matrix = self._compute_distances()
            self._neighbor_ids, self._neighbor_weights = self._compute_neighbors()


def _normalize(x: torch.Tensor) -> torch.Tensor:
    s = x.sum()
    if s <= 0.0:
        return x
    return x / s


class SpatialBindingLayer:
    """Binding via local spatial connectivity on a topographic grid.

    Replaces the dense BindingLayer with sparse local operations.
    Each column connects to its K nearest grid neighbors. Binding
    strength decays with grid distance (Gaussian profile).

    Maintains the same interface as BindingLayer for drop-in replacement:
    bind(), modulation_gain(), reset_state(), state_dict(), load_state_dict().
    """

    def __init__(
        self,
        n_columns: int,
        device: torch.device,
        grid: TopographicGrid | None = None,
        k_neighbors: int = 8,
        sigma: float = 1.5,
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
        self.k_neighbors = min(int(k_neighbors), n_columns - 1)
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

        # Use provided grid or create one
        if grid is not None:
            self.grid = grid
        else:
            self.grid = TopographicGrid(
                n_columns, k_neighbors=k_neighbors, sigma=sigma, device=device,
            )

        # State tensors — per-column (not per-binding-neuron like dense version)
        self.binding_state = torch.zeros(n_columns, device=device)
        self.coincidence_trace = torch.zeros(n_columns, device=device)
        self.facilitation = torch.zeros(n_columns, device=device)
        self.resources = torch.ones(n_columns, device=device)
        self.binding_usage = torch.zeros(n_columns, device=device)
        self.pv_inhibition = torch.tensor(0.0, device=device)

        # Sparse connectivity: precomputed neighbor indices + learned weights
        # neighbor_ids: [n_columns, k_neighbors] — which columns each connects to
        # learned_weights: [n_columns, k_neighbors] — learned association strengths
        self.neighbor_ids = self.grid._neighbor_ids.to(device)
        self.neighbor_weights = self.grid._neighbor_weights.to(device)
        self.learned_weights = self.neighbor_weights.clone()

        # For compatibility: n_bindings (used by some diagnostics)
        self.n_bindings = n_columns
        self.fan_in = k_neighbors

    def _sparse_drive(self, signal: torch.Tensor) -> torch.Tensor:
        """Compute local drive for each column from its neighbors.

        Instead of dense matvec (160×128), uses sparse gather from K neighbors.
        """
        normed = _normalize(signal.to(self.device))
        # Gather neighbor activations: [n_columns, k_neighbors]
        neighbor_acts = normed[self.neighbor_ids]
        # Weight by learned spatial weights
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
        """Spatial binding: coincidence detection via local grid connectivity.

        Same interface as BindingLayer.bind().
        """
        context = _normalize(context_prediction.to(self.device))
        current = _normalize(assembly.to(self.device))
        ctx_sum = context.sum()
        cur_sum = current.sum()

        if ctx_sum <= 0.0 or cur_sum <= 0.0:
            self.coincidence_trace *= max(0.0, 1.0 - 1.0 / self.tau_binding)
            self.binding_state.zero_()
            return self.binding_state, 0.0

        # Local drive from neighbors
        ctx_drive = self._sparse_drive(context)
        cur_drive = self._sparse_drive(current)

        # Coincidence: both context and current must be active in neighborhood
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

        # Column support: each column gets weighted sum of neighbor outputs
        neighbor_outputs = output[self.neighbor_ids]  # [n, k]
        support = (neighbor_outputs * self.learned_weights).sum(dim=1)
        self.binding_state = _normalize(support)

        strength = float(output.sum().item())

        # Update learned weights via Hebbian rule
        if update_weights and strength > 0.0:
            target = _normalize(current)
            for i in range(self.n_columns):
                if output[i] > 0.0:
                    nids = self.neighbor_ids[i]
                    neighbor_activity = target[nids]
                    self.learned_weights[i] = torch.clamp(
                        self.learned_weights[i] * self.association_decay
                        + self.association_lr * output[i] * neighbor_activity,
                        min=1e-6, max=1.0,
                    )
            # Renormalize
            weight_sums = self.learned_weights.sum(dim=1, keepdim=True).clamp(min=1e-8)
            self.learned_weights = self.learned_weights / weight_sums

        return self.binding_state, strength

    def binding_prediction(self, context_prediction: torch.Tensor) -> torch.Tensor:
        """Predict column activation from spatial binding context."""
        context = _normalize(context_prediction.to(self.device))
        if context.sum() <= 0.0:
            return torch.zeros(self.n_columns, device=self.device)
        if self.binding_usage.max() <= 1e-6:
            return torch.zeros(self.n_columns, device=self.device)
        predicted = self._sparse_drive(context) * self.binding_usage
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
        """Spatial grow_binding: boost learned weights for correlated neighbor pairs.

        Unlike dense BindingLayer which adds new neurons, spatial binding
        strengthens existing neighbor connections for correlated columns.
        Returns count of strengthened connections.
        """
        strengthened = 0
        for col_a, col_b, corr in high_correlation_columns:
            a, b = int(col_a), int(col_b)
            if corr <= 0.7 or a == b:
                continue
            if not (0 <= a < self.n_columns and 0 <= b < self.n_columns):
                continue
            # Check if b is a neighbor of a
            nids_a = self.neighbor_ids[a].tolist()
            if b in nids_a:
                idx = nids_a.index(b)
                self.learned_weights[a, idx] = min(1.0, self.learned_weights[a, idx] + 0.1 * corr)
                strengthened += 1
            # Also check a as neighbor of b
            nids_b = self.neighbor_ids[b].tolist()
            if a in nids_b:
                idx = nids_b.index(a)
                self.learned_weights[b, idx] = min(1.0, self.learned_weights[b, idx] + 0.1 * corr)
                strengthened += 1
        # Renormalize
        if strengthened > 0:
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
            "grid_state": self.grid.state_dict(),
            # Compatibility keys for BindingLayer format
            "connectivity": torch.eye(self.n_columns),
            "output_weights": torch.eye(self.n_columns),
            "binding_outputs": self.coincidence_trace.detach().clone().cpu(),
        }

    def load_state_dict(self, snapshot: dict[str, Any]) -> None:
        grid_state = snapshot.get("grid_state")
        if isinstance(grid_state, dict):
            self.grid.load_state_dict(grid_state)
            self.neighbor_ids = self.grid._neighbor_ids.to(self.device)
            self.neighbor_weights = self.grid._neighbor_weights.to(self.device)

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
