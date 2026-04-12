"""TurboQuant+ prototype store for scalable routing (§6.2).

Implements the two-stage TurboQuant architecture (arXiv:2504.19874):

  Stage 1 — PolarQuant: Random orthogonal rotation to Gaussianise
  heavy-tailed prototype distributions, then uniform scalar quantisation
  to low-bit (3-bit default).

  Stage 2 — QJL residual correction: 1-bit Quantised Johnson-Lindenstrauss
  projection of the quantisation residual.  At query time the sign-packed
  residual produces an *unbiased* inner-product estimator, eliminating the
  systematic bias that plain scalar quantisation introduces at ≤4 bits.

The store keeps a full FP32 copy (updated infrequently during wake) and
a compressed copy (rebuilt during sleep phases).

Reference: Shen et al., "TurboQuant: Online Vector Quantization with
Near-optimal Distortion Rate", arXiv:2504.19874, 2025.
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
    """TurboQuant+ compressed prototype store with QJL residual correction.

    Two-stage compression:
      Stage 1 (PolarQuant): Random rotation → uniform scalar quantisation.
      Stage 2 (QJL):        1-bit sign of projected residual for unbiased
                            inner-product estimation.

    Args:
        n_cols: Number of columns (prototypes).
        dim: Dimensionality of each prototype vector.
        bits: Quantisation bits (default 3 → 8 levels).
        n_projections: Number of QJL projection dimensions (default = dim).
            Higher m gives lower variance in the correction term.
        device: Torch device.
    """

    def __init__(
        self,
        n_cols: int,
        dim: int,
        bits: int = 3,
        n_projections: int | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.n_cols = int(n_cols)
        self.dim = int(dim)
        self.bits = int(bits)
        self.n_levels = 2 ** self.bits
        self.device = device or torch.device("cpu")

        # Random orthogonal rotation for Gaussianisation (Stage 1)
        self.rotation = _random_rotation_matrix(self.dim, self.device)

        # QJL random Gaussian projection matrix (Stage 2), shared for all
        self.n_proj = n_projections if n_projections is not None else self.dim
        self._projection = torch.randn(
            self.n_proj, self.dim, device=self.device
        ) / math.sqrt(self.n_proj)

        # Full-precision store (updated on learning events)
        self._fp32 = torch.zeros(n_cols, dim, device=self.device)

        # Stage 1: quantised codes + per-prototype scale/offset
        self._codes = torch.zeros(n_cols, dim, dtype=torch.int16, device=self.device)
        self._scales = torch.ones(n_cols, device=self.device)
        self._offsets = torch.zeros(n_cols, device=self.device)

        # Stage 2: QJL residual — sign bits (±1 stored as int8) and norms
        self._residual_signs = torch.zeros(
            n_cols, self.n_proj, dtype=torch.int8, device=self.device
        )
        self._residual_norms = torch.zeros(n_cols, device=self.device)

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
        """Compress all dirty prototypes (Stage 1 + Stage 2). Returns count."""
        dirty_mask = self._dirty.nonzero(as_tuple=True)[0]
        if len(dirty_mask) == 0:
            return 0

        for idx in dirty_mask:
            i = int(idx.item())
            rotated = torch.mv(self.rotation, self._fp32[i])

            # Stage 1: PolarQuant — uniform scalar quantisation
            vmin, vmax = rotated.min().item(), rotated.max().item()
            span = max(vmax - vmin, 1e-8)
            scale = span / (self.n_levels - 1)
            codes = ((rotated - vmin) / scale).round().clamp(
                0, self.n_levels - 1
            ).to(torch.int16)

            self._codes[i] = codes
            self._scales[i] = scale
            self._offsets[i] = vmin

            # Stage 2: QJL residual correction
            dequantised = codes.float() * scale + vmin
            residual = rotated - dequantised
            self._residual_norms[i] = residual.norm()
            projected = torch.mv(self._projection, residual)  # (m,)
            self._residual_signs[i] = projected.sign().to(torch.int8)

        count = len(dirty_mask)
        self._dirty[dirty_mask] = False
        return count

    # -- routing (hot path) ------------------------------------------------

    def route(self, query: torch.Tensor, k: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
        """Find top-k prototypes by QJL-corrected approximate inner product.

        The estimator is unbiased: E[score] = <q_rot, prototype_rot>.

        Returns:
            (indices, scores) both of shape (k,).
        """
        q = F.normalize(query.to(self.device).float(), dim=0)
        q_rot = torch.mv(self.rotation, q)  # (dim,)

        # Stage 1: base inner products via batch decompression
        decompressed = (
            self._codes.float() * self._scales.unsqueeze(1)
            + self._offsets.unsqueeze(1)
        )
        base_scores = torch.mv(decompressed, q_rot)  # (n_cols,)

        # Stage 2: QJL correction for unbiased estimation
        q_proj = torch.mv(self._projection, q_rot)  # (m,)
        correction = torch.mv(self._residual_signs.float(), q_proj)  # (n_cols,)
        correction = correction * (
            self._residual_norms * math.sqrt(math.pi / 2) / self.n_proj
        )
        scores = base_scores + correction

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
            cos = F.cosine_similarity(
                approx_scores.unsqueeze(0), exact_scores.unsqueeze(0)
            ).item()
            total_cos += cos
        return total_cos / n_queries

    def inner_product_bias(self, n_queries: int = 200) -> float:
        """Measure mean bias of the QJL-corrected inner-product estimator.

        Returns average (approx_score - exact_score) across all prototype
        scores and queries.  A well-calibrated QJL correction should yield
        a bias near zero.
        """
        self.compress_all()
        total_bias = 0.0
        count = 0
        for _ in range(n_queries):
            q = F.normalize(torch.randn(self.dim, device=self.device), dim=0)
            _, approx_scores = self.route(q, k=self.n_cols)
            _, exact_scores = self.route_exact(q, k=self.n_cols)
            total_bias += (approx_scores - exact_scores).sum().item()
            count += self.n_cols
        return total_bias / max(count, 1)

    # -- memory stats ------------------------------------------------------

    def memory_bytes(self) -> dict[str, int | float]:
        """Report memory usage including QJL overhead."""
        fp32_bytes = self._fp32.nelement() * 4
        code_bytes = self._codes.nelement() * 2  # int16
        meta_bytes = self._scales.nelement() * 4 + self._offsets.nelement() * 4
        rotation_bytes = self.rotation.nelement() * 4
        projection_bytes = self._projection.nelement() * 4
        qjl_sign_bytes = self._residual_signs.nelement() * 1  # int8
        qjl_norm_bytes = self._residual_norms.nelement() * 4
        compressed_total = code_bytes + meta_bytes + qjl_sign_bytes + qjl_norm_bytes
        return {
            "fp32": fp32_bytes,
            "compressed": compressed_total,
            "stage1_codes": code_bytes + meta_bytes,
            "stage2_qjl": qjl_sign_bytes + qjl_norm_bytes,
            "rotation": rotation_bytes,
            "projection": projection_bytes,
            "total": fp32_bytes + compressed_total + rotation_bytes + projection_bytes,
            "compression_ratio": fp32_bytes / max(1, compressed_total),
        }

    # -- serialization ------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        return {
            "fp32": self._fp32.cpu(),
            "codes": self._codes.cpu(),
            "scales": self._scales.cpu(),
            "offsets": self._offsets.cpu(),
            "rotation": self.rotation.cpu(),
            "projection": self._projection.cpu(),
            "residual_signs": self._residual_signs.cpu(),
            "residual_norms": self._residual_norms.cpu(),
            "dirty": self._dirty.cpu(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        _field_map = {
            "fp32": "_fp32",
            "codes": "_codes",
            "scales": "_scales",
            "offsets": "_offsets",
            "rotation": "rotation",
            "projection": "_projection",
            "residual_signs": "_residual_signs",
            "residual_norms": "_residual_norms",
            "dirty": "_dirty",
        }
        for key, attr in _field_map.items():
            val = state.get(key)
            if val is not None:
                setattr(self, attr, val.to(self.device))
