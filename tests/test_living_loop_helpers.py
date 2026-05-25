"""Tests for the extracted living_loop_helpers module.

These tests verify that the 12 cross-layer private helper functions
are importable from the helpers module and produce identical behaviour
to their original inline definitions in living_loop.py.
"""
from __future__ import annotations

import unittest

from hecsn.semantics.provenance import Provenance
from hecsn.service.living_loop import VerificationStatus, WorldModelLiteSummary
from hecsn.service.living_loop_helpers import (
    _as_mapping,
    _clean_text,
    _clamp01,
    _coerce_world_model_lite,
    _enum_value,
    _latest_text,
    _limited_unique_clean_text,
    _provenance_value,
    _safe_float,
    _safe_ratio,
    _stable_id,
    _verification_status_from_payload,
)


class StableIdTests(unittest.TestCase):
    def test_deterministic(self) -> None:
        result = _stable_id("pred", "alpha", 42)
        self.assertEqual(result, _stable_id("pred", "alpha", 42))

    def test_prefix(self) -> None:
        result = _stable_id("rec", "x")
        self.assertTrue(result.startswith("rec-"))

    def test_different_inputs_different_ids(self) -> None:
        self.assertNotEqual(_stable_id("a", 1), _stable_id("a", 2))


class CleanTextTests(unittest.TestCase):
    def test_strips_whitespace(self) -> None:
        self.assertEqual(_clean_text("  hello   world  "), "hello world")

    def test_none_returns_empty(self) -> None:
        self.assertEqual(_clean_text(None), "")

    def test_empty_string(self) -> None:
        self.assertEqual(_clean_text(""), "")

    def test_number_conversion(self) -> None:
        self.assertEqual(_clean_text(42), "42")


class Clamp01Tests(unittest.TestCase):
    def test_clamp_high(self) -> None:
        self.assertEqual(_clamp01(1.5), 1.0)

    def test_clamp_low(self) -> None:
        self.assertEqual(_clamp01(-0.5), 0.0)

    def test_in_range(self) -> None:
        self.assertEqual(_clamp01(0.5), 0.5)

    def test_invalid_returns_zero(self) -> None:
        self.assertEqual(_clamp01("bad"), 0.0)


class SafeRatioTests(unittest.TestCase):
    def test_normal(self) -> None:
        self.assertAlmostEqual(_safe_ratio(3.0, 4.0), 0.75)

    def test_zero_denominator(self) -> None:
        self.assertEqual(_safe_ratio(5.0, 0.0), 0.0)

    def test_negative_denominator(self) -> None:
        self.assertEqual(_safe_ratio(5.0, -1.0), 0.0)


class LimitedUniqueCleanTextTests(unittest.TestCase):
    def test_deduplication(self) -> None:
        result = _limited_unique_clean_text(["hello", "  hello  ", "world"])
        self.assertEqual(result, ("hello", "world"))

    def test_limit(self) -> None:
        result = _limited_unique_clean_text(["a", "b", "c", "d"], limit=2)
        self.assertEqual(len(result), 2)

    def test_lower(self) -> None:
        result = _limited_unique_clean_text(["Hello", "HELLO", "World"], lower=True)
        self.assertEqual(result, ("hello", "world"))

    def test_empty_input(self) -> None:
        self.assertEqual(_limited_unique_clean_text([]), ())


class LatestTextTests(unittest.TestCase):
    def test_returns_max(self) -> None:
        self.assertEqual(_latest_text(["alpha", "beta", "gamma"]), "gamma")

    def test_empty(self) -> None:
        self.assertEqual(_latest_text([]), "")

    def test_strips_whitespace(self) -> None:
        self.assertEqual(_latest_text(["  z  ", "  a  "]), "z")


class AsMappingTests(unittest.TestCase):
    def test_mapping_passthrough(self) -> None:
        data = {"key": "value"}
        self.assertIs(_as_mapping(data), data)

    def test_to_payload(self) -> None:
        class FakePayload:
            def to_payload(self) -> dict:
                return {"x": 1}
        result = _as_mapping(FakePayload())
        self.assertEqual(result, {"x": 1})

    def test_non_mapping_returns_empty(self) -> None:
        self.assertEqual(_as_mapping(42), {})


class EnumValueTests(unittest.TestCase):
    def test_already_enum(self) -> None:
        result = _enum_value(VerificationStatus, VerificationStatus.VERIFIED, VerificationStatus.UNKNOWN)
        self.assertIs(result, VerificationStatus.VERIFIED)

    def test_by_value(self) -> None:
        result = _enum_value(VerificationStatus, "verified", VerificationStatus.UNKNOWN)
        self.assertIs(result, VerificationStatus.VERIFIED)

    def test_by_name(self) -> None:
        result = _enum_value(VerificationStatus, "VERIFIED", VerificationStatus.UNKNOWN)
        self.assertIs(result, VerificationStatus.VERIFIED)

    def test_unknown_returns_default(self) -> None:
        result = _enum_value(VerificationStatus, "nonsense", VerificationStatus.UNKNOWN)
        self.assertIs(result, VerificationStatus.UNKNOWN)


class ProvenanceValueTests(unittest.TestCase):
    def test_already_provenance(self) -> None:
        result = _provenance_value(Provenance.INFERRED)
        self.assertIs(result, Provenance.INFERRED)

    def test_by_value(self) -> None:
        result = _provenance_value("inferred")
        self.assertIs(result, Provenance.INFERRED)

    def test_default(self) -> None:
        result = _provenance_value("nonsense")
        self.assertIs(result, Provenance.INFERRED)

    def test_custom_default(self) -> None:
        result = _provenance_value("nonsense", default=Provenance.OBSERVED)
        self.assertIs(result, Provenance.OBSERVED)


class VerificationStatusFromPayloadTests(unittest.TestCase):
    def test_verified(self) -> None:
        self.assertIs(_verification_status_from_payload("verified"), VerificationStatus.VERIFIED)

    def test_contradicted(self) -> None:
        self.assertIs(_verification_status_from_payload("contradicted"), VerificationStatus.CONTRADICTED)

    def test_unverified(self) -> None:
        self.assertIs(_verification_status_from_payload("unverified"), VerificationStatus.UNVERIFIED)

    def test_pending(self) -> None:
        self.assertIs(_verification_status_from_payload("pending"), VerificationStatus.UNVERIFIED)

    def test_unknown(self) -> None:
        self.assertIs(_verification_status_from_payload("nonsense"), VerificationStatus.UNKNOWN)

    def test_whitespace_stripped(self) -> None:
        self.assertIs(_verification_status_from_payload("  verified  "), VerificationStatus.VERIFIED)


class SafeFloatTests(unittest.TestCase):
    def test_valid(self) -> None:
        self.assertAlmostEqual(_safe_float(3.14), 3.14)

    def test_string_number(self) -> None:
        self.assertAlmostEqual(_safe_float("2.5"), 2.5)

    def test_negative_returns_none(self) -> None:
        self.assertIsNone(_safe_float(-1.0))

    def test_invalid_returns_none(self) -> None:
        self.assertIsNone(_safe_float("bad"))

    def test_none_returns_none(self) -> None:
        self.assertIsNone(_safe_float(None))

    def test_zero(self) -> None:
        self.assertAlmostEqual(_safe_float(0), 0.0)


class CoerceWorldModelLiteTests(unittest.TestCase):
    def test_none_returns_default(self) -> None:
        result = _coerce_world_model_lite(None)
        self.assertIsInstance(result, WorldModelLiteSummary)

    def test_passthrough(self) -> None:
        wml = WorldModelLiteSummary(prediction_count=5)
        result = _coerce_world_model_lite(wml)
        self.assertIs(result, wml)

    def test_mapping(self) -> None:
        data = {"prediction_count": 10}
        result = _coerce_world_model_lite(data)
        self.assertIsInstance(result, WorldModelLiteSummary)
        self.assertEqual(result.prediction_count, 10)


class ReExportShimTests(unittest.TestCase):
    """Verify that living_loop.py still re-exports all the helpers so that
    any internal callers within the file still resolve."""

    def test_helpers_importable_from_living_loop(self) -> None:
        from hecsn.service.living_loop import (
            _as_mapping,
            _clean_text,
            _clamp01,
            _coerce_world_model_lite,
            _enum_value,
            _latest_text,
            _limited_unique_clean_text,
            _provenance_value,
            _safe_float,
            _safe_ratio,
            _stable_id,
            _verification_status_from_payload,
        )
        # Just verify they're callable
        self.assertTrue(callable(_stable_id))
        self.assertTrue(callable(_clean_text))
        self.assertTrue(callable(_clamp01))
        self.assertTrue(callable(_safe_ratio))
        self.assertTrue(callable(_limited_unique_clean_text))
        self.assertTrue(callable(_latest_text))
        self.assertTrue(callable(_as_mapping))
        self.assertTrue(callable(_enum_value))
        self.assertTrue(callable(_provenance_value))
        self.assertTrue(callable(_verification_status_from_payload))
        self.assertTrue(callable(_safe_float))
        self.assertTrue(callable(_coerce_world_model_lite))


if __name__ == "__main__":
    unittest.main()
