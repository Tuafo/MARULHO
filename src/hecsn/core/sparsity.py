"""Dual sparsity utilities for HECSN (§6.3).

Provides:
- 2:4 structured sparsity: keep 2 largest-magnitude values in every group of 4.
  This is a *regularization / compression* utility.  On Ampere+ GPUs with
  a supported semi-structured kernel path the mask can unlock ~1.6× speedup,
  but simply zeroing elements in a dense tensor does **not** engage hardware
  acceleration — callers must export to NVIDIA's semi-structured format for
  that.  In HECSN this is used for offline weight compression and as a
  structural regularizer during training.
- CSR sparse tensor conversion: profiling-only utility for read-mostly
  matmul paths (e.g. routing queries).  Not suitable for matrices that
  undergo frequent in-place updates, normalization, or indexing.
- Profiling gate: benchmark CSR vs dense matmul for given dimensions.
- SparsityManager: register tensors and enforce sparsity patterns periodically.
  Callers must re-apply any invariants (row-normalization, projection norm
  targets) after enforcement.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import torch


# ---------------------------------------------------------------------------
# 2:4 Structured Sparsity
# ---------------------------------------------------------------------------


def apply_2_4_mask(weight: torch.Tensor) -> torch.Tensor:
    """Enforce 2:4 structured sparsity on a 2-D weight matrix.

    For each contiguous group of 4 elements along the last dimension,
    keep only the 2 with the largest absolute value and zero the rest.
    If the last dimension is not divisible by 4, the remainder elements
    are left unchanged (no pruning applied to the tail).

    Args:
        weight: 2-D tensor (rows × cols).

    Returns:
        A new tensor with the 2:4 mask applied.
    """
    if weight.ndim != 2:
        raise ValueError(f"apply_2_4_mask requires a 2-D tensor, got {weight.ndim}-D")

    rows, cols = weight.shape
    full_groups = cols // 4
    remainder = cols % 4

    if full_groups == 0:
        return weight.clone()

    # Reshape the groupable portion into (rows, full_groups, 4)
    groupable = weight[:, : full_groups * 4].reshape(rows, full_groups, 4)

    # For each group of 4, find the top-2 by magnitude
    abs_vals = groupable.abs()
    _, top_indices = abs_vals.topk(2, dim=2)  # (rows, full_groups, 2)

    mask = torch.zeros_like(groupable, dtype=torch.bool)
    mask.scatter_(2, top_indices, True)

    pruned_groupable = groupable * mask.to(groupable.dtype)

    # Reassemble
    if remainder > 0:
        tail = weight[:, full_groups * 4 :]
        result = torch.cat([pruned_groupable.reshape(rows, full_groups * 4), tail], dim=1)
    else:
        result = pruned_groupable.reshape(rows, cols)

    return result


def compute_2_4_mask(weight: torch.Tensor) -> torch.Tensor:
    """Compute a boolean mask for 2:4 structured sparsity without applying it.

    Returns a boolean tensor of the same shape: True where values are kept.
    """
    if weight.ndim != 2:
        raise ValueError(f"compute_2_4_mask requires a 2-D tensor, got {weight.ndim}-D")

    rows, cols = weight.shape
    full_groups = cols // 4
    remainder = cols % 4

    mask = torch.ones(rows, cols, dtype=torch.bool, device=weight.device)

    if full_groups == 0:
        return mask

    groupable = weight[:, : full_groups * 4].reshape(rows, full_groups, 4)
    abs_vals = groupable.abs()
    _, top_indices = abs_vals.topk(2, dim=2)

    group_mask = torch.zeros(rows, full_groups, 4, dtype=torch.bool, device=weight.device)
    group_mask.scatter_(2, top_indices, True)

    mask[:, : full_groups * 4] = group_mask.reshape(rows, full_groups * 4)
    # Remainder cols stay True (no pruning)

    return mask


# ---------------------------------------------------------------------------
# CSR Sparse Tensor Utilities
# ---------------------------------------------------------------------------


def density(tensor: torch.Tensor) -> float:
    """Compute the fraction of nonzero elements."""
    return float((tensor != 0).sum().item()) / max(tensor.numel(), 1)


def to_csr_if_sparse(
    tensor: torch.Tensor,
    density_threshold: float = 0.3,
) -> torch.Tensor:
    """Convert a 2-D dense tensor to CSR if its density is below *density_threshold*.

    Returns the tensor unchanged (dense) if density is above the threshold,
    or if the tensor is not 2-D, or if CSR conversion fails.
    """
    if tensor.ndim != 2:
        return tensor
    d = density(tensor)
    if d >= density_threshold:
        return tensor
    try:
        return tensor.to_sparse_csr()
    except Exception:
        return tensor


def from_csr(tensor: torch.Tensor) -> torch.Tensor:
    """Convert a CSR (or any sparse) tensor back to dense."""
    if tensor.is_sparse or tensor.layout == torch.sparse_csr:
        return tensor.to_dense()
    return tensor


def csr_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Matrix multiply that handles CSR inputs transparently.

    If either operand is CSR, performs sparse matmul via torch.mm.
    Falls back to dense matmul if sparse mm fails.
    """
    a_sparse = a.is_sparse or (hasattr(a, "layout") and a.layout == torch.sparse_csr)
    b_sparse = b.is_sparse or (hasattr(b, "layout") and b.layout == torch.sparse_csr)

    if a_sparse or b_sparse:
        try:
            return torch.mm(a, b)
        except Exception:
            return torch.mm(from_csr(a), from_csr(b))
    return torch.mm(a, b)


# ---------------------------------------------------------------------------
# Profiling Gate (§6.3)
# ---------------------------------------------------------------------------


@dataclass
class ProfilingResult:
    """Result of a CSR-vs-dense profiling run."""

    n_rows: int
    n_cols: int
    density_pct: float
    dense_mean_us: float
    csr_mean_us: float
    speedup: float  # dense_mean / csr_mean; >1 means CSR is faster
    recommendation: str  # "csr" or "dense"


def profiling_gate(
    n_rows: int = 100,
    n_cols: int = 10_000,
    density_frac: float = 0.15,
    n_trials: int = 20,
    warmup: int = 5,
) -> ProfilingResult:
    """Benchmark CSR vs dense matmul for given matrix dimensions and density.

    Creates a random matrix of shape (n_rows, n_cols) with the specified
    density, converts to CSR, and times matrix-vector products.

    Args:
        n_rows: Number of rows.
        n_cols: Number of columns.
        density_frac: Fraction of nonzero elements (0 < d ≤ 1).
        n_trials: Number of timed iterations (after warmup).
        warmup: Number of warmup iterations (not timed).

    Returns:
        ProfilingResult with timing data and recommendation.
    """
    device = torch.device("cpu")

    # Build sparse matrix
    mask = (torch.rand(n_rows, n_cols, device=device) < density_frac).float()
    values = torch.randn(n_rows, n_cols, device=device) * mask
    x = torch.randn(n_cols, 1, device=device)

    csr = values.to_sparse_csr()

    # Warmup
    for _ in range(warmup):
        torch.mm(values, x)
        torch.mm(csr, x)

    # Time dense
    t0 = time.perf_counter()
    for _ in range(n_trials):
        torch.mm(values, x)
    dense_total = time.perf_counter() - t0
    dense_mean_us = (dense_total / n_trials) * 1e6

    # Time CSR
    t0 = time.perf_counter()
    for _ in range(n_trials):
        torch.mm(csr, x)
    csr_total = time.perf_counter() - t0
    csr_mean_us = (csr_total / n_trials) * 1e6

    speedup = dense_mean_us / max(csr_mean_us, 1e-9)
    recommendation = "csr" if speedup > 1.1 else "dense"

    return ProfilingResult(
        n_rows=n_rows,
        n_cols=n_cols,
        density_pct=round(density_frac * 100, 1),
        dense_mean_us=round(dense_mean_us, 2),
        csr_mean_us=round(csr_mean_us, 2),
        speedup=round(speedup, 3),
        recommendation=recommendation,
    )


# ---------------------------------------------------------------------------
# SparsityManager
# ---------------------------------------------------------------------------


@dataclass
class _RegisteredTensor:
    name: str
    tensor_ref: torch.Tensor
    mode: str  # "2_4", "csr", "both"
    density_threshold: float
    post_enforce: Optional[callable]  # normalization callback


class SparsityManager:
    """Manages sparsity enforcement for a collection of weight tensors.

    Many HECSN weight matrices carry invariants (positive-only values,
    row-normalization targets, projection norm targets).  After masking,
    these invariants may be violated.  Use the *post_enforce* callback
    when registering a tensor to re-apply the invariant after enforcement.

    Usage::

        mgr = SparsityManager(mode="2_4")
        mgr.register("W_project", model.W_project,
                      post_enforce=lambda t: F.normalize(t, dim=0) * norm_target)
        mgr.register("recurrent", model.recurrent, density_threshold=0.35)
        # After each training step or periodically:
        updated = mgr.enforce()  # returns dict of name → new tensor
    """

    def __init__(self, mode: str = "none"):
        """
        Args:
            mode: Default sparsity mode for registered tensors.
                  "none" — no sparsity enforcement.
                  "2_4"  — 2:4 structured sparsity.
                  "csr"  — CSR conversion for sparse matrices.
                  "both" — apply 2:4 first, then CSR if still sparse.
        """
        if mode not in {"none", "2_4", "csr", "both"}:
            raise ValueError(f"Unknown sparsity mode: {mode!r}")
        self.default_mode = mode
        self._registered: dict[str, _RegisteredTensor] = {}

    def register(
        self,
        name: str,
        tensor: torch.Tensor,
        mode: Optional[str] = None,
        density_threshold: float = 0.3,
        post_enforce: Optional[callable] = None,
    ) -> None:
        """Register a tensor for sparsity management.

        Args:
            name: Identifier for the tensor.
            tensor: The weight tensor to manage.
            mode: Override the manager's default mode for this tensor.
            density_threshold: CSR conversion threshold.
            post_enforce: Optional callback ``fn(tensor) -> tensor`` to
                re-apply invariants (normalization, clamping, etc.)
                after sparsity enforcement.
        """
        self._registered[name] = _RegisteredTensor(
            name=name,
            tensor_ref=tensor,
            mode=mode or self.default_mode,
            density_threshold=density_threshold,
            post_enforce=post_enforce,
        )

    def enforce(self) -> dict[str, torch.Tensor]:
        """Apply sparsity patterns to all registered tensors.

        Returns a dict mapping name → sparsified tensor.
        If a post_enforce callback was registered, it is applied after
        sparsity masking to restore weight invariants.
        Callers must copy the result back into the model parameter
        (this function does not modify in-place for safety).
        """
        results: dict[str, torch.Tensor] = {}
        for name, entry in self._registered.items():
            t = entry.tensor_ref
            if entry.mode == "none" or t.ndim != 2:
                results[name] = t
                continue

            if entry.mode in ("2_4", "both"):
                t = apply_2_4_mask(t)

            if entry.post_enforce is not None:
                t = entry.post_enforce(t)

            if entry.mode in ("csr", "both"):
                t = to_csr_if_sparse(t, entry.density_threshold)

            results[name] = t
        return results

    @property
    def registered_names(self) -> list[str]:
        return list(self._registered.keys())

    def summary(self) -> dict[str, dict[str, float]]:
        """Return density statistics for each registered tensor."""
        stats: dict[str, dict[str, float]] = {}
        for name, entry in self._registered.items():
            t = entry.tensor_ref
            if t.ndim == 2:
                d = density(t)
                stats[name] = {
                    "density": round(d, 4),
                    "nonzeros": int((t != 0).sum().item()),
                    "total": t.numel(),
                    "mode": entry.mode,
                }
        return stats
