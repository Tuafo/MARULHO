"""Tests for BaseEncoder protocol conformance (Stage 1A)."""

from __future__ import annotations

import unittest

import torch

from hecsn.data.base_encoder import BaseEncoder
from hecsn.data.rtf_encoder import RTFEncoder


class BaseEncoderProtocolTests(unittest.TestCase):
    """Verify RTFEncoder satisfies the BaseEncoder protocol."""

    def test_rtf_encoder_is_base_encoder(self) -> None:
        """RTFEncoder must be recognized as a BaseEncoder at runtime."""
        encoder = RTFEncoder()
        self.assertIsInstance(encoder, BaseEncoder)

    def test_output_dim_property(self) -> None:
        encoder = RTFEncoder()
        self.assertIsInstance(encoder.output_dim, int)
        self.assertGreater(encoder.output_dim, 0)

    def test_feature_vector_shape(self) -> None:
        encoder = RTFEncoder()
        chars = [ord(c) for c in "hello"]
        vec = encoder.feature_vector(chars)
        self.assertEqual(vec.shape, (encoder.output_dim,))

    def test_feature_vector_normalized(self) -> None:
        encoder = RTFEncoder()
        chars = [ord(c) for c in "test input"]
        vec = encoder.feature_vector(chars)
        norm = float(torch.norm(vec).item())
        self.assertAlmostEqual(norm, 1.0, places=3)

    def test_iter_char_patterns_yields_tuples(self) -> None:
        encoder = RTFEncoder()
        patterns = list(encoder.iter_char_patterns("hello world", window_size=5))
        self.assertGreater(len(patterns), 0)
        for text, vec in patterns:
            self.assertIsInstance(text, str)
            self.assertIsInstance(vec, torch.Tensor)

    def test_segment_text_returns_list(self) -> None:
        encoder = RTFEncoder()
        segments = encoder.segment_text("hello world")
        self.assertIsInstance(segments, list)
        self.assertGreater(len(segments), 0)

    def test_spike_trace_shape(self) -> None:
        encoder = RTFEncoder()
        chars = [ord(c) for c in "test"]
        trace = encoder.spike_trace(chars, context_confidence=0.8)
        self.assertIsInstance(trace, torch.Tensor)
        self.assertGreater(trace.numel(), 0)

    def test_state_dict_roundtrip(self) -> None:
        encoder = RTFEncoder()
        state = encoder.state_dict()
        self.assertIsInstance(state, dict)
        # Should not raise
        encoder.load_state_dict(state)

    def test_protocol_type_checking(self) -> None:
        """Verify the protocol works for type checking at runtime."""

        class NotAnEncoder:
            pass

        self.assertNotIsInstance(NotAnEncoder(), BaseEncoder)


class EncoderFactoryTests(unittest.TestCase):
    """Test that code can work with BaseEncoder type hint."""

    def _process_with_encoder(self, encoder: BaseEncoder, text: str) -> torch.Tensor:
        """Example function that accepts any BaseEncoder."""
        chars = [ord(c) for c in text]
        return encoder.feature_vector(chars)

    def test_rtf_encoder_as_base_encoder_arg(self) -> None:
        encoder = RTFEncoder()
        result = self._process_with_encoder(encoder, "test")
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape[0], encoder.output_dim)


if __name__ == "__main__":
    unittest.main()
