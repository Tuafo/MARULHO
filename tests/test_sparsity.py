"""Tests for hecsn.core.sparsity — 2:4 structured sparsity, CSR, profiling."""

from __future__ import annotations

import torch
import pytest

from hecsn.core.sparsity import (
    apply_2_4_mask,
    compute_2_4_mask,
    csr_matmul,
    density,
    from_csr,
    profiling_gate,
    to_csr_if_sparse,
    SparsityManager,
)


# ---------------------------------------------------------------------------
# 2:4 Structured Sparsity
# ---------------------------------------------------------------------------


class Test24Mask:
    """Tests for apply_2_4_mask and compute_2_4_mask."""

    def test_basic_2_4_pattern(self):
        """Each group of 4 should have exactly 2 nonzeros."""
        w = torch.randn(8, 16)
        masked = apply_2_4_mask(w)
        # Check each group of 4 in each row
        for row in range(8):
            for g in range(4):  # 16 / 4 = 4 groups
                group = masked[row, g * 4 : g * 4 + 4]
                assert (group != 0).sum().item() == 2

    def test_keeps_largest_two(self):
        """The two kept values should be the largest by magnitude."""
        w = torch.tensor([[1.0, -5.0, 3.0, 2.0, 0.1, 0.2, 0.9, -0.8]])
        masked = apply_2_4_mask(w)
        # Group 0: [-5.0, 3.0] are largest → kept
        assert masked[0, 1] == -5.0
        assert masked[0, 2] == 3.0
        assert masked[0, 0] == 0.0
        assert masked[0, 3] == 0.0
        # Group 1: [0.9, -0.8] are largest
        assert masked[0, 6] == 0.9
        assert masked[0, 7] == -0.8

    def test_remainder_cols_preserved(self):
        """Columns not divisible by 4 should be kept unchanged."""
        w = torch.randn(4, 7)  # 1 group of 4 + 3 remainder
        masked = apply_2_4_mask(w)
        # Last 3 cols unchanged
        assert torch.equal(masked[:, 4:], w[:, 4:])

    def test_too_few_cols_unchanged(self):
        """Matrix with <4 cols should be returned unchanged."""
        w = torch.randn(3, 3)
        masked = apply_2_4_mask(w)
        assert torch.equal(masked, w)

    def test_rejects_non_2d(self):
        """Should raise ValueError for non-2D input."""
        with pytest.raises(ValueError, match="2-D"):
            apply_2_4_mask(torch.randn(8))
        with pytest.raises(ValueError, match="2-D"):
            apply_2_4_mask(torch.randn(2, 3, 4))

    def test_compute_mask_matches_apply(self):
        """compute_2_4_mask should produce the same pattern as apply_2_4_mask."""
        w = torch.randn(6, 12)
        mask = compute_2_4_mask(w)
        applied = apply_2_4_mask(w)
        reconstructed = w * mask.float()
        assert torch.allclose(applied, reconstructed)

    def test_mask_is_boolean(self):
        w = torch.randn(4, 8)
        mask = compute_2_4_mask(w)
        assert mask.dtype == torch.bool

    def test_density_after_mask(self):
        """After 2:4, density should be exactly 50% for divisible dimensions."""
        w = torch.randn(10, 64)
        masked = apply_2_4_mask(w)
        d = density(masked)
        assert abs(d - 0.5) < 0.01

    def test_does_not_modify_original(self):
        """apply_2_4_mask should return a new tensor, not modify in-place."""
        w = torch.randn(4, 8)
        original = w.clone()
        _ = apply_2_4_mask(w)
        assert torch.equal(w, original)

    def test_all_zeros_group(self):
        """A group that is all zeros should stay all zeros."""
        w = torch.zeros(2, 8)
        masked = apply_2_4_mask(w)
        assert torch.equal(masked, w)


# ---------------------------------------------------------------------------
# CSR Utilities
# ---------------------------------------------------------------------------


class TestCSRUtilities:
    """Tests for CSR conversion helpers."""

    def test_density_computation(self):
        t = torch.tensor([[1.0, 0.0, 0.0, 2.0], [0.0, 0.0, 3.0, 0.0]])
        d = density(t)
        assert abs(d - 3 / 8) < 1e-6

    def test_density_empty(self):
        t = torch.zeros(3, 3)
        assert density(t) == 0.0

    def test_density_full(self):
        t = torch.ones(3, 3)
        assert density(t) == 1.0

    def test_to_csr_converts_sparse(self):
        """Matrix with density below threshold should convert to CSR."""
        t = torch.zeros(10, 10)
        t[0, 0] = 1.0
        t[5, 5] = 2.0
        result = to_csr_if_sparse(t, density_threshold=0.3)
        assert result.layout == torch.sparse_csr

    def test_to_csr_keeps_dense_above_threshold(self):
        """Dense matrix should stay dense."""
        t = torch.randn(4, 4)
        result = to_csr_if_sparse(t, density_threshold=0.3)
        assert not result.is_sparse
        assert result.layout != torch.sparse_csr

    def test_to_csr_rejects_1d(self):
        """Non-2D tensors should be returned unchanged."""
        t = torch.randn(10)
        result = to_csr_if_sparse(t)
        assert torch.equal(result, t)

    def test_from_csr_roundtrip(self):
        """CSR → dense roundtrip should preserve values."""
        t = torch.zeros(5, 5)
        t[0, 1] = 3.0
        t[2, 4] = -1.0
        csr = t.to_sparse_csr()
        dense = from_csr(csr)
        assert torch.equal(dense, t)

    def test_from_csr_on_dense_is_noop(self):
        t = torch.randn(3, 3)
        assert torch.equal(from_csr(t), t)

    def test_csr_matmul_correctness(self):
        """CSR matmul should match dense matmul."""
        a = torch.randn(10, 20)
        b = torch.randn(20, 5)
        expected = torch.mm(a, b)

        a_csr = a.to_sparse_csr()
        result = csr_matmul(a_csr, b)
        assert torch.allclose(result, expected, atol=1e-5)

    def test_csr_matmul_both_dense(self):
        """Should work fine with both dense inputs."""
        a = torch.randn(4, 6)
        b = torch.randn(6, 3)
        result = csr_matmul(a, b)
        expected = torch.mm(a, b)
        assert torch.allclose(result, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# Profiling Gate
# ---------------------------------------------------------------------------


class TestProfilingGate:
    """Tests for the CSR-vs-dense profiling benchmark."""

    def test_returns_profiling_result(self):
        result = profiling_gate(n_rows=10, n_cols=100, density_frac=0.1, n_trials=3, warmup=1)
        assert result.n_rows == 10
        assert result.n_cols == 100
        assert result.density_pct == 10.0
        assert result.dense_mean_us > 0
        assert result.csr_mean_us > 0
        assert result.recommendation in ("csr", "dense")

    def test_high_density_recommends_dense(self):
        """Very dense matrix should prefer dense matmul."""
        result = profiling_gate(n_rows=10, n_cols=50, density_frac=0.95, n_trials=3, warmup=1)
        # At 95% density, CSR overhead makes dense faster
        assert result.recommendation == "dense"

    def test_speedup_is_positive(self):
        result = profiling_gate(n_rows=5, n_cols=20, n_trials=2, warmup=1)
        assert result.speedup > 0


# ---------------------------------------------------------------------------
# SparsityManager
# ---------------------------------------------------------------------------


class TestSparsityManager:
    """Tests for the SparsityManager class."""

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown sparsity mode"):
            SparsityManager(mode="invalid")

    def test_register_and_enforce_2_4(self):
        mgr = SparsityManager(mode="2_4")
        w = torch.randn(4, 8)
        mgr.register("test", w)
        results = mgr.enforce()
        assert "test" in results
        # Should have 2:4 pattern
        masked = results["test"]
        for row in range(4):
            for g in range(2):
                group = masked[row, g * 4 : g * 4 + 4]
                assert (group != 0).sum().item() == 2

    def test_mode_none_noop(self):
        mgr = SparsityManager(mode="none")
        w = torch.randn(4, 8)
        mgr.register("test", w)
        results = mgr.enforce()
        assert torch.equal(results["test"], w)

    def test_post_enforce_callback(self):
        """Post-enforce callback should be applied after masking."""
        mgr = SparsityManager(mode="2_4")
        w = torch.randn(4, 8)

        # Callback: make all values positive
        def make_positive(t):
            return t.abs()

        mgr.register("test", w, post_enforce=make_positive)
        results = mgr.enforce()
        assert (results["test"] >= 0).all()

    def test_post_enforce_renormalize(self):
        """Post-enforce renormalization should restore column norms."""
        import torch.nn.functional as F

        mgr = SparsityManager(mode="2_4")
        w = torch.randn(8, 16)
        norm_target = 1.0

        def renorm(t):
            return F.normalize(t, dim=0) * norm_target

        mgr.register("W_project", w, post_enforce=renorm)
        results = mgr.enforce()
        col_norms = results["W_project"].norm(dim=0)
        # All non-zero columns should have norm close to target
        nonzero_cols = col_norms > 0.01
        if nonzero_cols.any():
            assert torch.allclose(col_norms[nonzero_cols], torch.ones_like(col_norms[nonzero_cols]) * norm_target, atol=0.01)

    def test_register_with_mode_override(self):
        mgr = SparsityManager(mode="2_4")
        w = torch.randn(4, 8)
        mgr.register("skip_this", w, mode="none")
        results = mgr.enforce()
        # mode=none should skip masking
        assert torch.equal(results["skip_this"], w)

    def test_registered_names(self):
        mgr = SparsityManager()
        mgr.register("a", torch.randn(2, 4))
        mgr.register("b", torch.randn(3, 6))
        assert set(mgr.registered_names) == {"a", "b"}

    def test_summary(self):
        mgr = SparsityManager(mode="2_4")
        w = torch.randn(4, 8)
        mgr.register("test", w)
        stats = mgr.summary()
        assert "test" in stats
        assert "density" in stats["test"]
        assert stats["test"]["total"] == 32

    def test_csr_mode(self):
        """CSR mode should convert sparse tensors."""
        mgr = SparsityManager(mode="csr")
        w = torch.zeros(10, 10)
        w[0, 0] = 1.0
        w[5, 5] = 2.0
        mgr.register("sparse_w", w, density_threshold=0.5)
        results = mgr.enforce()
        assert results["sparse_w"].layout == torch.sparse_csr

    def test_multiple_tensors(self):
        mgr = SparsityManager(mode="2_4")
        w1 = torch.randn(4, 8)
        w2 = torch.randn(6, 12)
        mgr.register("w1", w1)
        mgr.register("w2", w2)
        results = mgr.enforce()
        assert len(results) == 2
        # Both should have 2:4 pattern
        for name in ("w1", "w2"):
            t = results[name]
            rows, cols = t.shape
            for row in range(rows):
                for g in range(cols // 4):
                    group = t[row, g * 4 : g * 4 + 4]
                    assert (group != 0).sum().item() == 2

    def test_1d_tensor_skipped(self):
        """1-D tensors should pass through unchanged."""
        mgr = SparsityManager(mode="2_4")
        v = torch.randn(10)
        mgr.register("bias", v)
        results = mgr.enforce()
        assert torch.equal(results["bias"], v)
