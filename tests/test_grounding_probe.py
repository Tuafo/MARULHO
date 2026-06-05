"""Tests for the 50-triple grounding probe evaluation module."""

from __future__ import annotations

import torch
import pytest

from marulho.evaluation.grounding_probe import (
    GROUNDING_PROBE_TRIPLES_50,
    CONCRETE_TRIPLES,
    ABSTRACT_TRIPLES,
    GroundingProbeResult,
    evaluate_grounding_probe,
)


class TestTripleDefinitions:
    """Verify the triple definitions match v4 paper §8.7 requirements."""

    def test_concrete_count(self) -> None:
        assert len(CONCRETE_TRIPLES) == 25

    def test_abstract_count(self) -> None:
        assert len(ABSTRACT_TRIPLES) == 25

    def test_total_count(self) -> None:
        assert len(GROUNDING_PROBE_TRIPLES_50) == 50

    def test_first_25_are_concrete(self) -> None:
        for i in range(25):
            assert GROUNDING_PROBE_TRIPLES_50[i] == CONCRETE_TRIPLES[i]

    def test_last_25_are_abstract(self) -> None:
        for i in range(25):
            assert GROUNDING_PROBE_TRIPLES_50[25 + i] == ABSTRACT_TRIPLES[i]

    def test_each_triple_has_three_strings(self) -> None:
        for triple in GROUNDING_PROBE_TRIPLES_50:
            assert len(triple) == 3
            assert all(isinstance(s, str) for s in triple)
            assert all(len(s) > 0 for s in triple)

    def test_no_duplicate_anchors_within_category(self) -> None:
        concrete_anchors = [t[0] for t in CONCRETE_TRIPLES]
        abstract_anchors = [t[0] for t in ABSTRACT_TRIPLES]
        assert len(concrete_anchors) == len(set(concrete_anchors))
        assert len(abstract_anchors) == len(set(abstract_anchors))


class TestEvaluateGroundingProbe:
    """Test the evaluate_grounding_probe function."""

    @staticmethod
    def _random_vector_fn(dim: int = 64) -> callable:
        """Return a vector_fn that produces random vectors (expected ~0.50 accuracy)."""
        cache: dict[str, torch.Tensor] = {}

        def fn(text: str) -> torch.Tensor:
            if text not in cache:
                cache[text] = torch.randn(dim)
            return cache[text]

        return fn

    @staticmethod
    def _perfect_vector_fn() -> callable:
        """Return a vector_fn where anchor ≈ positive, negative is distant (100% accuracy)."""
        torch.manual_seed(12345)
        cache: dict[str, torch.Tensor] = {}

        # Pre-populate: for each triple, make anchor ≈ positive, negative orthogonal
        for anchor, positive, negative in GROUNDING_PROBE_TRIPLES_50:
            base = torch.randn(32)
            base = base / base.norm()
            cache[anchor] = base
            cache[positive] = base + 0.001 * torch.randn(32)
            # Make negative orthogonal to anchor
            neg = torch.randn(32)
            neg = neg - (neg @ base) * base
            neg = neg / neg.norm()
            cache[negative] = neg

        def fn(text: str) -> torch.Tensor:
            if text not in cache:
                cache[text] = torch.randn(32)
            return cache[text]

        return fn

    def test_result_type(self) -> None:
        result = evaluate_grounding_probe(self._random_vector_fn())
        assert isinstance(result, GroundingProbeResult)

    def test_result_counts(self) -> None:
        result = evaluate_grounding_probe(self._random_vector_fn())
        assert result.total_count == 50
        assert result.concrete_count == 25
        assert result.abstract_count == 25

    def test_perfect_accuracy(self) -> None:
        result = evaluate_grounding_probe(self._perfect_vector_fn())
        assert result.total_accuracy == 1.0
        assert result.concrete_accuracy == 1.0
        assert result.abstract_accuracy == 1.0
        assert result.probe_pass is True

    def test_perfect_concreteness_gap_is_zero(self) -> None:
        result = evaluate_grounding_probe(self._perfect_vector_fn())
        assert result.concreteness_gap == 0.0

    def test_per_triple_output(self) -> None:
        result = evaluate_grounding_probe(self._random_vector_fn())
        assert len(result.per_triple) == 50
        first = result.per_triple[0]
        assert "anchor" in first
        assert "positive" in first
        assert "negative" in first
        assert "positive_similarity" in first
        assert "negative_similarity" in first
        assert "margin" in first
        assert "correct" in first
        assert first["category"] == "concrete"
        assert result.per_triple[25]["category"] == "abstract"

    def test_to_dict(self) -> None:
        result = evaluate_grounding_probe(self._random_vector_fn())
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "total_accuracy" in d
        assert "concreteness_gap" in d
        assert "per_triple" in d
        assert len(d["per_triple"]) == 50

    def test_custom_triples(self) -> None:
        small = (
            ("cat", "dog", "car"),
            ("sun", "star", "fish"),
        )
        result = evaluate_grounding_probe(
            self._random_vector_fn(), triples=small, concrete_count=1
        )
        assert result.total_count == 2
        assert result.concrete_count == 1
        assert result.abstract_count == 1

    def test_zero_vector_handling(self) -> None:
        def zero_fn(text: str) -> torch.Tensor:
            return torch.zeros(16)

        result = evaluate_grounding_probe(zero_fn)
        assert result.total_count == 50
        # All similarities should be 0, all margins 0, no correct
        assert result.total_accuracy == 0.0


class TestGroundingProbeResult:
    """Test GroundingProbeResult properties."""

    def test_probe_pass_below_threshold(self) -> None:
        r = GroundingProbeResult(
            total_accuracy=0.50,
            concrete_accuracy=0.60,
            abstract_accuracy=0.40,
            concreteness_gap=0.20,
            total_count=50,
            concrete_count=25,
            abstract_count=25,
            mean_margin=0.05,
            concrete_mean_margin=0.10,
            abstract_mean_margin=0.0,
        )
        assert r.probe_pass is False

    def test_probe_pass_above_threshold(self) -> None:
        r = GroundingProbeResult(
            total_accuracy=0.70,
            concrete_accuracy=0.80,
            abstract_accuracy=0.60,
            concreteness_gap=0.20,
            total_count=50,
            concrete_count=25,
            abstract_count=25,
            mean_margin=0.15,
            concrete_mean_margin=0.20,
            abstract_mean_margin=0.10,
        )
        assert r.probe_pass is True

    def test_concreteness_gap_pass(self) -> None:
        r = GroundingProbeResult(
            total_accuracy=0.70,
            concrete_accuracy=0.80,
            abstract_accuracy=0.60,
            concreteness_gap=0.20,
            total_count=50,
            concrete_count=25,
            abstract_count=25,
            mean_margin=0.15,
            concrete_mean_margin=0.20,
            abstract_mean_margin=0.10,
        )
        assert r.concreteness_gap_pass is True

    def test_concreteness_gap_fail(self) -> None:
        r = GroundingProbeResult(
            total_accuracy=0.70,
            concrete_accuracy=0.72,
            abstract_accuracy=0.68,
            concreteness_gap=0.04,
            total_count=50,
            concrete_count=25,
            abstract_count=25,
            mean_margin=0.15,
            concrete_mean_margin=0.16,
            abstract_mean_margin=0.14,
        )
        assert r.concreteness_gap_pass is False


class TestHeldOutProbe:
    """Test the held-out concrete probe for transfer evaluation."""

    def test_held_out_triple_count(self) -> None:
        from marulho.evaluation.grounding_probe import HELD_OUT_CONCRETE_TRIPLES
        assert len(HELD_OUT_CONCRETE_TRIPLES) == 10

    def test_held_out_no_overlap_with_concept_vocabulary(self) -> None:
        from marulho.evaluation.grounding_probe import HELD_OUT_CONCRETE_TRIPLES
        from marulho.training.developmental_runner import CONCEPT_VOCABULARY
        vocab = set(CONCEPT_VOCABULARY)
        for anchor, pos, neg in HELD_OUT_CONCRETE_TRIPLES:
            for word in (anchor, pos, neg):
                assert word not in vocab, f"'{word}' overlaps with CONCEPT_VOCABULARY"

    def test_extended_probe_60_triples(self) -> None:
        from marulho.evaluation.grounding_probe import (
            GROUNDING_PROBE_TRIPLES_60,
            evaluate_grounding_probe_extended,
        )
        assert len(GROUNDING_PROBE_TRIPLES_60) == 60

        def random_fn(text: str, _c: dict = {}) -> torch.Tensor:
            if text not in _c:
                _c[text] = torch.randn(64)
            return _c[text]

        result = evaluate_grounding_probe_extended(random_fn)
        assert result.total_count == 60
        assert result.concrete_count == 25
        assert result.held_out_concrete_count == 10
        assert result.abstract_count == 25

    def test_held_out_per_triple_category(self) -> None:
        from marulho.evaluation.grounding_probe import evaluate_grounding_probe_extended

        def random_fn(text: str, _c: dict = {}) -> torch.Tensor:
            if text not in _c:
                _c[text] = torch.randn(64)
            return _c[text]

        result = evaluate_grounding_probe_extended(random_fn)
        categories = [t["category"] for t in result.per_triple]
        assert categories[:25] == ["concrete"] * 25
        assert categories[25:35] == ["held_out_concrete"] * 10
        assert categories[35:60] == ["abstract"] * 25

    def test_held_out_in_to_dict(self) -> None:
        from marulho.evaluation.grounding_probe import evaluate_grounding_probe_extended

        def random_fn(text: str, _c: dict = {}) -> torch.Tensor:
            if text not in _c:
                _c[text] = torch.randn(64)
            return _c[text]

        d = evaluate_grounding_probe_extended(random_fn).to_dict()
        assert "held_out_concrete_accuracy" in d
        assert "held_out_concrete_count" in d
        assert "held_out_concreteness_gap" in d
        assert d["held_out_concrete_count"] == 10

    def test_backward_compatible_50_probe(self) -> None:
        """Original 50-triple probe should still work with held_out_count=0."""

        def random_fn(text: str, _c: dict = {}) -> torch.Tensor:
            if text not in _c:
                _c[text] = torch.randn(64)
            return _c[text]

        result = evaluate_grounding_probe(random_fn)
        assert result.total_count == 50
        assert result.held_out_concrete_count == 0
        assert result.held_out_concrete_accuracy == 0.0
