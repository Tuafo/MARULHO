from __future__ import annotations

import pytest

from marulho.evaluation.language_billion_generation import (
    COSMOPEDIA_PROMPTS,
    DCLM_ROWS_AND_PROMPTS,
    FINEWEB_PROMPTS,
    PANEL_SPECS,
    _cases_for_spec,
    _expected_decode_controls,
    _read_marked_documents,
    generation_decision,
)
from marulho.evaluation.language_quality_continuation import DOCUMENT_MARKER


def test_v80_generation_panels_freeze_old_and_new_prompts() -> None:
    assert len(PANEL_SPECS) == 5
    assert FINEWEB_PROMPTS == (
        "The Heilwood Company",
        "Heilwood Company store",
        "Company store was",
        "Being a rural,",
    )
    assert COSMOPEDIA_PROMPTS == (
        "In today's digital",
        "today's digital age,",
        "digital age, computers",
        "One critical aspect",
    )
    assert DCLM_ROWS_AND_PROMPTS == (
        (0, "Numerous industries utilize"),
        (2, "Summer in the"),
        (4, "Prevent Direct Execution"),
        (6, "In just one"),
    )


def test_marked_document_reader_selects_only_frozen_rows(tmp_path) -> None:
    source = tmp_path / "documents.txt"
    source.write_text(
        "\n".join(
            [
                DOCUMENT_MARKER,
                "zero first",
                "zero second",
                DOCUMENT_MARKER,
                "one",
                DOCUMENT_MARKER,
                "two",
            ]
        ),
        encoding="utf-8",
    )
    assert _read_marked_documents(source, (0, 2)) == {
        0: "zero first\nzero second",
        2: "two",
    }
    with pytest.raises(RuntimeError, match="missing frozen rows"):
        _read_marked_documents(source, (3,))


def test_panel_cases_bind_each_dclm_prompt_to_its_own_document() -> None:
    spec = next(spec for spec in PANEL_SPECS if spec.name == "dclm_greedy")
    sources = {
        "fineweb": "unused",
        "cosmopedia": "unused",
        "dclm": {
            row: f"{prompt} and row {row} continues here."
            for row, prompt in DCLM_ROWS_AND_PROMPTS
        },
    }
    cases = _cases_for_spec(spec, sources)
    assert [case.prompt_text for case in cases] == list(spec.prompts)
    assert [case.source_text for case in cases] == [
        f"{prompt} and row {row} continues here."
        for row, prompt in DCLM_ROWS_AND_PROMPTS
    ]


def test_decode_controls_and_human_decision_are_explicit() -> None:
    greedy = next(spec for spec in PANEL_SPECS if spec.name == "dclm_greedy")
    controlled = next(
        spec for spec in PANEL_SPECS if spec.name == "dclm_controlled"
    )
    assert _expected_decode_controls(greedy)["decode_controls_requested"] is False
    assert _expected_decode_controls(controlled) == {
        "repetition_penalty": 1.1,
        "repetition_penalty_applied": True,
        "no_repeat_ngram_size": 3,
        "no_repeat_ngram_applied": True,
        "decode_controls_requested": True,
    }
    assert generation_decision(validity_passed=False, human_review_coherent=True) == (
        "reject_v80_unseen_generation_invalid_evidence"
    )
    assert generation_decision(validity_passed=True, human_review_coherent=True) == (
        "advance_v80_to_continual_and_grounded_self_challenge_validation"
    )
    assert generation_decision(validity_passed=True, human_review_coherent=False) == (
        "redesign_v80_objective_or_tokenizer_after_scale_failure"
    )
