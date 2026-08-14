from __future__ import annotations

import json

import pytest

from marulho.evaluation.language_billion_generation import (
    COSMOPEDIA_PROMPTS,
    DCLM_ROWS_AND_PROMPTS,
    FINEWEB_PROMPTS,
    PANEL_SPECS,
    _cases_for_spec,
    _expected_decode_controls,
    _panel_checks,
    _read_marked_documents,
    _training_admission,
    generation_decision,
)
from marulho.evaluation.language_generation_coherence import (
    LanguageGenerationPromptCase,
    prompt_case_metadata,
)
from marulho.evaluation.language_quality_continuation import DOCUMENT_MARKER
from marulho.evaluation.language_quality_continuation import file_sha256


def test_v80_generation_panels_freeze_old_and_new_prompts() -> None:
    assert len(PANEL_SPECS) == 6
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
    sampled = next(spec for spec in PANEL_SPECS if spec.name == "dclm_sampled")
    assert _expected_decode_controls(greedy)["decode_controls_requested"] is False
    assert _expected_decode_controls(controlled) == {
        "decode_strategy": "greedy_argmax",
        "repetition_penalty": 1.1,
        "repetition_penalty_applied": True,
        "no_repeat_ngram_size": 3,
        "no_repeat_ngram_applied": True,
        "temperature": 0.0,
        "top_p": 1.0,
        "sampling_seed": None,
        "top_p_applied": False,
        "decode_controls_requested": True,
    }
    assert _expected_decode_controls(sampled) == {
        "decode_strategy": "nucleus_sampling",
        "repetition_penalty": 1.05,
        "repetition_penalty_applied": True,
        "no_repeat_ngram_size": 0,
        "no_repeat_ngram_applied": False,
        "temperature": 0.8,
        "top_p": 0.9,
        "sampling_seed": 80080,
        "top_p_applied": True,
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


def test_training_admission_requires_quality_fidelity_and_owned_checkpoint(
    tmp_path,
) -> None:
    import marulho.evaluation.language_billion_generation as module

    checkpoint = tmp_path / "candidate.pt"
    checkpoint.write_bytes(b"qualified checkpoint fixture")
    report_path = tmp_path / "training.json"
    report = {
        "surface": "marulho_language_billion_continuation.v80",
        "passed": True,
        "decision": "admit_v80_checkpoint_to_unseen_generation",
        "owned_by_marulho": True,
        "external_llm_used": False,
        "quality_checks": {"quality": True},
        "configuration": {
            "target_cumulative_processed_positions": (
                module.TARGET_CUMULATIVE_POSITIONS
            )
        },
        "final_evaluation": {"later_segment_loss": 2.5},
        "checkpoint": {
            "saved": True,
            "sha256": file_sha256(checkpoint),
            "fidelity": {"passed": True},
        },
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")
    admission, checkpoint_hash = _training_admission(report_path, checkpoint)
    assert all(admission["checks"].values())
    assert checkpoint_hash == file_sha256(checkpoint)

    report["external_llm_used"] = True
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(RuntimeError, match="generation is not admitted"):
        _training_admission(report_path, checkpoint)


def test_panel_checks_cover_prompt_source_decode_and_ownership() -> None:
    spec = next(spec for spec in PANEL_SPECS if spec.name == "fineweb_greedy")
    source = " ".join(f"{prompt} continues." for prompt in spec.prompts)
    cases = tuple(
        LanguageGenerationPromptCase(prompt_text=prompt, source_text=source)
        for prompt in spec.prompts
    )
    report = {
        "prompt_suite": {
            "prompt_cases": [prompt_case_metadata(case) for case in cases],
            "generation_decode_controls": _expected_decode_controls(spec),
        },
        "source": {"sha256": spec.source_sha256},
        "checkpoint": {
            "sha256": "checkpoint",
            "tokenizer_hash": (
                "faca1e26aa29e897bef4e4335a0300f90e3996723d556a681b4495240f660715"
            ),
        },
        "owned_by_marulho": True,
        "external_llm_used": False,
        "cases": [{}, {}, {}, {}],
    }
    checks = _panel_checks(
        report,
        spec=spec,
        cases=cases,
        checkpoint_sha256="checkpoint",
    )
    assert all(checks.values())
    report["external_llm_used"] = True
    assert not _panel_checks(
        report,
        spec=spec,
        cases=cases,
        checkpoint_sha256="checkpoint",
    )["external_llm_absent"]
