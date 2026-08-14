from __future__ import annotations

from marulho.evaluation.language_dclm_materialization import (
    _filter_reason,
    _normalized_text_sha256,
)


def _row(text: str) -> dict:
    return {
        "text": text,
        "language": "en",
        "language_score": 0.95,
        "edu_int_score": 3,
    }


def test_v79_dclm_filter_rejects_formulaic_synthetic_text() -> None:
    assert _filter_reason(_row("This chapter will" + " useful text" * 300)) == (
        "contains_rejected_template_phrase"
    )
    assert _filter_reason(_row("Natural prose " * 300)) is None


def test_v79_dclm_filter_enforces_language_quality_and_length() -> None:
    short = _row("short")
    assert _filter_reason(short) == "text_below_2000_characters"
    low_score = _row("Natural prose " * 300)
    low_score["language_score"] = 0.89
    assert _filter_reason(low_score) == "language_score_below_0_90"
    low_edu = _row("Natural prose " * 300)
    low_edu["edu_int_score"] = 2
    assert _filter_reason(low_edu) == "edu_int_score_below_3"


def test_v79_normalized_hash_deduplicates_whitespace_and_case() -> None:
    assert _normalized_text_sha256("Hello\n WORLD") == _normalized_text_sha256(
        " hello world "
    )
