"""Tests for the semantic n-gram encoder."""

from __future__ import annotations

import torch
import pytest

from hecsn.config.model_config import HECSNConfig
from hecsn.data.semantic_encoder import SemanticEncoder
from hecsn.data.base_encoder import BaseEncoder
from hecsn.data.encoder_factory import build_encoder


# ── Construction & protocol ────────────────────────────────────────────


class TestSemanticEncoderBasic:
    def test_construction_default(self):
        enc = SemanticEncoder()
        report = enc.device_report()
        assert enc.output_dim == 128
        assert enc.n_buckets == 10_000
        assert enc.embed_dim == 64
        assert report["bucket_embeddings_device"] == "cpu"
        assert report["last_feature_vector_device"] is None
        assert report["last_spike_trace_device"] is None

    def test_implements_base_encoder_protocol(self):
        enc = SemanticEncoder()
        assert isinstance(enc, BaseEncoder)

    def test_output_dim_with_custom_embed(self):
        enc = SemanticEncoder(embed_dim=32)
        assert enc.output_dim == 64  # 2 * 32

    def test_output_dim_concat_chunking(self):
        enc = SemanticEncoder(
            enable_learned_chunking=True,
            learned_chunk_feature_mode="concat",
            learned_chunk_concat_dim=64,
        )
        assert enc.output_dim == 128 + 64  # base + concat

    def test_from_config(self):
        cfg = HECSNConfig(input_representation="semantic")
        enc = SemanticEncoder.from_config(cfg)
        assert enc.output_dim == 128
        assert enc.n_buckets == cfg.semantic_n_buckets


# ── Feature vector properties ──────────────────────────────────────────


class TestFeatureVectorProperties:
    def test_feature_vector_shape(self):
        enc = SemanticEncoder()
        vec = enc.feature_vector([ord(c) for c in "hello"])
        assert vec.shape == (128,)

    def test_feature_vector_nonneg(self):
        enc = SemanticEncoder()
        vec = enc.feature_vector([ord(c) for c in "hello world"])
        assert (vec >= -1e-7).all(), f"Nonnegativity violated: min={vec.min().item()}"

    def test_feature_vector_l2_normalized(self):
        enc = SemanticEncoder()
        vec = enc.feature_vector([ord(c) for c in "test"])
        norm = torch.norm(vec, p=2).item()
        assert abs(norm - 1.0) < 1e-4, f"L2 norm = {norm}, expected ~1.0"

    def test_empty_input_returns_zero(self):
        enc = SemanticEncoder()
        vec = enc.feature_vector([])
        assert vec.shape == (128,)
        assert float(vec.sum().item()) == 0.0

    def test_different_words_different_vectors(self):
        enc = SemanticEncoder()
        v1 = enc.feature_vector([ord(c) for c in "dog"])
        v2 = enc.feature_vector([ord(c) for c in "cat"])
        cos = float(torch.dot(v1, v2).item())
        assert cos < 0.99, f"Different words should produce different vectors, cos={cos}"

    def test_same_word_same_vector(self):
        enc = SemanticEncoder()
        v1 = enc.feature_vector([ord(c) for c in "hello"])
        v2 = enc.feature_vector([ord(c) for c in "hello"])
        cos = float(torch.dot(v1, v2).item())
        assert cos > 0.999, f"Same word should produce same vector, cos={cos}"


# ── Split-sign encoding ────────────────────────────────────────────────


class TestSplitSignEncoding:
    def test_split_sign_preserves_info(self):
        enc = SemanticEncoder()
        raw = torch.randn(64)
        encoded = enc._split_sign_encode(raw)
        assert encoded.shape == (128,)
        assert (encoded >= -1e-7).all()

    def test_split_sign_opposite_vectors(self):
        enc = SemanticEncoder()
        raw = torch.randn(64)
        e1 = enc._split_sign_encode(raw)
        e2 = enc._split_sign_encode(-raw)
        cos = float(torch.dot(e1, e2).item())
        # Opposite vectors should have low cosine similarity after split-sign
        assert cos < 0.5, f"Opposite vectors should differ after split-sign, cos={cos}"

    def test_top_k_sparsification(self):
        enc = SemanticEncoder(top_k_sparse=32)
        vec = enc.feature_vector([ord(c) for c in "hello"])
        nonzero = (vec > 1e-8).sum().item()
        assert nonzero <= 32, f"Expected ≤32 nonzero entries, got {nonzero}"


# ── N-gram hashing ─────────────────────────────────────────────────────


class TestNgramHashing:
    def test_hash_deterministic(self):
        enc = SemanticEncoder()
        b1 = enc._hash_ngram_to_bucket([100, 111, 103])
        b2 = enc._hash_ngram_to_bucket([100, 111, 103])
        assert b1 == b2

    def test_hash_in_range(self):
        enc = SemanticEncoder(n_buckets=1000)
        for word in ["hello", "world", "test", "xyz"]:
            codes = [ord(c) for c in word]
            bucket = enc._hash_ngram_to_bucket(codes)
            assert 0 <= bucket < 1000

    def test_collect_ngrams(self):
        enc = SemanticEncoder(ngram_min_n=2, ngram_max_n=3)
        codes = [100, 111, 103]  # "dog"
        buckets = enc._collect_ngram_buckets(codes)
        # bigrams: (d,o), (o,g) = 2; trigrams: (d,o,g) = 1 → 3 buckets
        assert len(buckets) == 3

    def test_single_char_produces_buckets(self):
        enc = SemanticEncoder(ngram_min_n=2, ngram_max_n=4)
        buckets = enc._collect_ngram_buckets([65])  # "A"
        assert len(buckets) >= 1  # Should get at least unigram hash


# ── Streaming iteration ────────────────────────────────────────────────


class TestIterCharPatterns:
    def test_yields_per_character(self):
        enc = SemanticEncoder()
        text = "hello"
        patterns = list(enc.iter_char_patterns(text, window_size=10))
        assert len(patterns) == len(text)

    def test_yields_correct_display_text(self):
        enc = SemanticEncoder()
        patterns = list(enc.iter_char_patterns("hi", window_size=10))
        assert patterns[0][0] == "h"
        assert patterns[1][0] == "hi"

    def test_all_vectors_nonneg_normalized(self):
        enc = SemanticEncoder()
        for _, vec in enc.iter_char_patterns("hello world", window_size=10):
            assert (vec >= -1e-7).all(), "Nonnegativity violated in streaming"
            norm = torch.norm(vec, p=2).item()
            assert abs(norm - 1.0) < 1e-3 or norm == 0.0

    def test_token_boundary_resets(self):
        """Vectors should differ across token boundaries (space resets token)."""
        enc = SemanticEncoder()
        patterns = list(enc.iter_char_patterns("ab cd", window_size=10))
        # After space, token_codes resets → different semantic base
        vec_b = patterns[1][1]  # 'b' in "ab"
        vec_c = patterns[3][1]  # 'c' in "cd" (after space)
        cos = float(torch.dot(vec_b, vec_c).item())
        assert cos < 0.99


# ── Segmentation ───────────────────────────────────────────────────────


class TestSegmentation:
    def test_segment_simple(self):
        enc = SemanticEncoder()
        segments = enc.segment_text("hello world")
        assert segments == ["hello", "world"]

    def test_segment_empty(self):
        enc = SemanticEncoder()
        assert enc.segment_text("") == []

    def test_segment_with_chunking(self):
        enc = SemanticEncoder(enable_learned_chunking=True)
        segments = enc.segment_text("hello world")
        assert len(segments) >= 2  # At least two segments


# ── Spike trace ────────────────────────────────────────────────────────


class TestSpikeTrace:
    def test_spike_trace_shape(self):
        enc = SemanticEncoder()
        trace = enc.spike_trace([ord(c) for c in "hello"], 0.5)
        assert trace.shape == (128,)

    def test_spike_trace_nonneg(self):
        enc = SemanticEncoder()
        trace = enc.spike_trace([ord(c) for c in "test"], 0.8)
        assert (trace >= -1e-7).all()

    def test_spike_trace_confidence_modulation(self):
        enc = SemanticEncoder()
        codes = [ord(c) for c in "hello"]
        t_high = enc.spike_trace(codes, 1.0)
        t_low = enc.spike_trace(codes, 0.1)
        assert float(t_high.sum().item()) >= float(t_low.sum().item())


class TestDeviceReport:
    def test_reports_last_emitted_tensor_devices(self):
        enc = SemanticEncoder()
        chars = [ord(c) for c in "device evidence"]

        feature = enc.feature_vector(chars)
        feature_report = enc.device_report()
        trace = enc.spike_trace(chars, 0.75)
        trace_report = enc.device_report()

        assert feature_report["last_feature_vector_device"] == str(feature.device)
        assert feature_report["last_feature_vector_shape"] == tuple(feature.shape)
        assert trace_report["last_spike_trace_device"] == str(trace.device)
        assert trace_report["last_spike_trace_shape"] == tuple(trace.shape)


# ── State serialization ───────────────────────────────────────────────


class TestSerialization:
    def test_state_dict_roundtrip(self):
        enc1 = SemanticEncoder()
        vec1 = enc1.feature_vector([ord(c) for c in "hello"])

        state = enc1.state_dict()
        assert state["bucket_embeddings"].device.type == "cpu"
        assert state["adapter"].device.type == "cpu"
        enc2 = SemanticEncoder()
        enc2.load_state_dict(state)
        vec2 = enc2.feature_vector([ord(c) for c in "hello"])

        assert torch.allclose(vec1, vec2, atol=1e-6)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
    def test_load_state_dict_restores_live_tensors_to_encoder_device(self):
        enc1 = SemanticEncoder(n_buckets=128, enable_learned_chunking=True)
        state = enc1.state_dict()

        enc2 = SemanticEncoder(n_buckets=128, enable_learned_chunking=True, device="cuda")
        enc2.load_state_dict(state)

        assert enc2.bucket_embeddings.device.type == "cuda"
        assert enc2.adapter.device.type == "cuda"
        assert enc2.learned_chunking is not None
        assert enc2.learned_chunking.prototypes.device.type == "cuda"


# ── Config integration ─────────────────────────────────────────────────


class TestConfigIntegration:
    def test_semantic_config_input_dim(self):
        cfg = HECSNConfig(input_representation="semantic")
        assert cfg.input_dim == 128  # 2 * 64

    def test_semantic_config_custom_embed_dim(self):
        cfg = HECSNConfig(input_representation="semantic", semantic_embed_dim=32)
        assert cfg.input_dim == 64  # 2 * 32

    def test_semantic_config_concat_chunking(self):
        cfg = HECSNConfig(
            input_representation="semantic",
            enable_learned_chunking=True,
            learned_chunk_feature_mode="concat",
            learned_chunk_concat_dim=64,
        )
        assert cfg.input_dim == 128 + 64


# ── Encoder factory ────────────────────────────────────────────────────


class TestEncoderFactory:
    def test_factory_returns_rtf_by_default(self):
        cfg = HECSNConfig()
        enc = build_encoder(cfg)
        from hecsn.data.rtf_encoder import RTFEncoder
        assert isinstance(enc, RTFEncoder)

    def test_factory_returns_semantic(self):
        cfg = HECSNConfig(input_representation="semantic")
        enc = build_encoder(cfg)
        assert isinstance(enc, SemanticEncoder)

    def test_factory_semantic_implements_protocol(self):
        cfg = HECSNConfig(input_representation="semantic")
        enc = build_encoder(cfg)
        assert isinstance(enc, BaseEncoder)

    def test_factory_accepts_explicit_device(self):
        cfg = HECSNConfig(input_representation="semantic", semantic_n_buckets=128)
        enc = build_encoder(cfg, device=torch.device("cpu"))
        assert isinstance(enc, SemanticEncoder)
        assert enc.device_report()["bucket_embeddings_device"] == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
    def test_semantic_encoder_can_keep_outputs_and_chunking_on_cuda(self):
        enc = SemanticEncoder(n_buckets=128, enable_learned_chunking=True, device="cuda")
        chars = [ord(c) for c in "semantic cuda"]

        assert enc.bucket_embeddings.device.type == "cuda"
        assert enc.adapter.device.type == "cuda"
        assert enc.feature_vector(chars).device.type == "cuda"
        assert enc.spike_trace(chars, 0.75).device.type == "cuda"
        assert next(enc.iter_char_patterns("hi", window_size=2))[1].device.type == "cuda"
        assert enc.learned_chunking is not None
        assert enc.learned_chunking.prototypes.device.type == "cuda"
        report = enc.device_report()
        assert report["last_feature_vector_device"] == "cuda:0"
        assert report["last_spike_trace_device"] == "cuda:0"
