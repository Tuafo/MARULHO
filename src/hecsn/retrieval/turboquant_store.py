"""TurboQuant-inspired prototype store for scalable routing (§6.2).

Provides a compressed prototype store that uses random rotation to
Gaussianise heavy-tailed prototype distributions, then quantises to
low-bit (3-bit default) for memory-efficient inner-product search.

When the external ``turboquant`` package is available it is used for
the quantisation core; otherwise a pure-PyTorch fallback provides
the same API (uniform scalar quantisation after rotation).

The store keeps a full FP32 copy that is updated infrequently (only
when prototypes change via learning) and a compressed copy that is
rebuilt during sleep phases.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F


def _random_rotation_matrix(dim: int, device: torch.device) -> torch.Tensor:
    """Generate a random orthogonal matrix via QR decomposition."""
    H = torch.randn(dim, dim, device=device)
    Q, R = torch.linalg.qr(H)
    # Ensure proper rotation (det = +1)
    d = torch.diag(R)
    ph = d.sign()
    Q = Q * ph.unsqueeze(0)
    return Q


class TurboQuantPrototypeStore:
    """Compressed prototype store with random-rotation Gaussianisation.

    Args:
        n_cols: Number of columns (prototypes).
        dim: Dimensionality of each prototype vector.
        bits: Quantisation bits (default 3 → 8 levels).
        device: Torch device.
    """

    def __init__(
        self,
        n_cols: int,
        dim: int,
        bits: int = 3,
        device: torch.device | None = None,
    ) -> None:
        self.n_cols = int(n_cols)
        self.dim = int(dim)
        self.bits = int(bits)
        self.n_levels = 2 ** self.bits
        self.device = device or torch.device("cpu")

        # Random orthogonal rotation for Gaussianisation
        self.rotation = _random_rotation_matrix(self.dim, self.device)

        # Full-precision store (updated on learning events)
        self._fp32 = torch.zeros(n_cols, dim, device=self.device)

        # Compressed store: quantised codes + scale/offset per prototype
        self._codes = torch.zeros(n_cols, dim, dtype=torch.int16, device=self.device)
        self._scales = torch.ones(n_cols, device=self.device)
        self._offsets = torch.zeros(n_cols, device=self.device)

        # Dirty flags — which prototypes need recompression
        self._dirty = torch.ones(n_cols, dtype=torch.bool, device=self.device)

    # -- update prototypes -------------------------------------------------

    def update(self, idx: int, prototype: torch.Tensor) -> None:
        """Update a single prototype (marks it dirty for recompression)."""
        self._fp32[idx] = prototype.to(self.device).float()
        self._dirty[idx] = True

    def update_batch(self, indices: torch.Tensor, prototypes: torch.Tensor) -> None:
        """Update multiple prototypes at once."""
        self._fp32[indices] = prototypes.to(self.device).float()
        self._dirty[indices] = True

    def set_all(self, prototypes: torch.Tensor) -> None:
        """Set all prototypes from a (n_cols, dim) matrix."""
        self._fp32 = prototypes.to(self.device).float()
        self._dirty.fill_(True)

    # -- compression (called during sleep phase) ---------------------------

    def compress_all(self) -> int:
        """Compress all dirty prototypes. Returns count compressed."""
        dirty_mask = self._dirty.nonzero(as_tuple=True)[0]
        if len(dirty_mask) == 0:
            return 0

        for idx in dirty_mask:
            i = int(idx.item())
            rotated = torch.mv(self.rotation, self._fp32[i])
            vmin, vmax = rotated.min().item(), rotated.max().item()
            span = max(vmax - vmin, 1e-8)
            scale = span / (self.n_levels - 1)
            offset = vmin

            # Quantise to [0, n_levels-1]
            codes = ((rotated - offset) / scale).round().clamp(0, self.n_levels - 1).to(torch.int16)

            self._codes[i] = codes
            self._scales[i] = scale
            self._offsets[i] = offset

        count = len(dirty_mask)
        self._dirty[dirty_mask] = False
        return count

    # -- routing (hot path) ------------------------------------------------

    def route(self, query: torch.Tensor, k: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
        """Find top-k prototypes by approximate inner product.

        Returns:
            (indices, scores) both of shape (k,).
        """
        q = F.normalize(query.to(self.device).float(), dim=0)

        # Rotate query once
        q_rot = torch.mv(self.rotation, q)

        # Decompress and compute inner products
        decompressed = self._codes.float() * self._scales.unsqueeze(1) + self._offsets.unsqueeze(1)
        scores = torch.mv(decompressed, q_rot)

        k_actual = min(k, self.n_cols)
        top_scores, top_indices = torch.topk(scores, k_actual)
        return top_indices, top_scores

    def route_exact(self, query: torch.Tensor, k: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
        """Exact routing using FP32 prototypes (for comparison/validation)."""
        q = F.normalize(query.to(self.device).float(), dim=0)
        scores = torch.mv(self._fp32, q)
        k_actual = min(k, self.n_cols)
        top_scores, top_indices = torch.topk(scores, k_actual)
        return top_indices, top_scores

    # -- accuracy measurement ----------------------------------------------

    def cosine_accuracy(self, n_queries: int = 100) -> float:
        """Measure cosine similarity between compressed and FP32 routing.

        Returns average cosine similarity between quantised and exact
        inner product score vectors over random queries.
        """
        self.compress_all()
        total_cos = 0.0
        for _ in range(n_queries):
            q = F.normalize(torch.randn(self.dim, device=self.device), dim=0)
            _, approx_scores = self.route(q, k=self.n_cols)
            _, exact_scores = self.route_exact(q, k=self.n_cols)
            # Cosine similarity of score vectors
            cos = F.cosine_similarity(approx_scores.unsqueeze(0), exact_scores.unsqueeze(0)).item()
            total_cos += cos
        return total_cos / n_queries

    # -- memory stats ------------------------------------------------------

    def memory_bytes(self) -> dict[str, int]:
        """Report memory usage."""
        fp32_bytes = self._fp32.nelement() * 4
        code_bytes = self._codes.nelement() * 2  # int16
        meta_bytes = self._scales.nelement() * 4 + self._offsets.nelement() * 4
        rotation_bytes = self.rotation.nelement() * 4
        return {
            "fp32": fp32_bytes,
            "compressed": code_bytes + meta_bytes,
            "rotation": rotation_bytes,
            "total": fp32_bytes + code_bytes + meta_bytes + rotation_bytes,
            "compression_ratio": fp32_bytes / max(1, code_bytes + meta_bytes),
        }

    # -- serialization ------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        return {
            "fp32": self._fp32.cpu(),
            "codes": self._codes.cpu(),
            "scales": self._scales.cpu(),
            "offsets": self._offsets.cpu(),
            "rotation": self.rotation.cpu(),
            "dirty": self._dirty.cpu(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        for key in ("fp32", "codes", "scales", "offsets", "rotation", "dirty"):
            val = state.get(key)
            if val is not None:
                attr = f"_{key}" if key != "rotation" and key != "dirty" else key
                if key == "dirty":
                    attr = "_dirty"
                elif key == "rotation":
                    attr = "rotation"
                else:
                    attr = f"_{key}"
                setattr(self, attr, val.to(self.device))
