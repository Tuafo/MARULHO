from __future__ import annotations

import json

import torch

from marulho.data.language_tokenizer import ByteLevelLanguageTokenizer
from marulho.evaluation.language_grounding_support import (
    SURFACE,
    run_language_grounding_support_report,
)
from marulho.training.language_model import LanguageModelConfig, MarulhoLanguageModel


def _model(tokenizer: ByteLevelLanguageTokenizer) -> MarulhoLanguageModel:
    return MarulhoLanguageModel(
        LanguageModelConfig(
            vocab_size=tokenizer.vocab_size,
            embedding_dim=12,
            state_dim=20,
            expert_count=2,
            active_expert_count=1,
            route_candidate_count=2,
            expert_hidden_dim=24,
        )
    )


def test_language_grounding_support_report_passes_source_term_gate(tmp_path) -> None:
    torch.manual_seed(33)
    tokenizer = ByteLevelLanguageTokenizer()
    output = tmp_path / "grounding-support.json"

    report = run_language_grounding_support_report(
        _model(tokenizer),
        tokenizer,
        prompt_text="runtime truth replay evidence",
        source_text=(
            "Runtime truth records replay evidence. "
            "Checkpointed source windows keep evidence reviewable."
        ),
        required_terms=("runtime", "truth", "replay", "evidence"),
        max_new_tokens=0,
        output_path=output,
    )
    written = json.loads(output.read_text(encoding="utf-8"))

    assert report["surface"] == SURFACE
    assert written["surface"] == SURFACE
    assert report["owned_by_marulho"] is True
    assert report["external_llm_used"] is False
    assert report["source_terms"]["source_term_coverage"] == 1.0
    assert report["source_terms"]["missing_required_terms"] == []
    assert report["promotion_gate"]["grounding_support_available"] is True
    assert report["promotion_gate"]["source_term_coverage_gate_passed"] is True
    assert report["promotion_gate"]["promotes_generation_quality_claim"] is False
    assert report["generation"]["active_language_path"] == "marulho_lm_head"
    assert report["generation"]["external_llm_used"] is False
    assert report["generation"]["new_token_count"] == 0
    assert report["generation"]["unsupported_generated_terms"] == []
    assert (tmp_path / "README.md").exists()


def test_language_grounding_support_report_blocks_missing_source_terms() -> None:
    torch.manual_seed(34)
    tokenizer = ByteLevelLanguageTokenizer()

    report = run_language_grounding_support_report(
        _model(tokenizer),
        tokenizer,
        prompt_text="runtime truth missingconcept",
        source_text="Runtime truth records replay evidence.",
        required_terms=("runtime", "truth", "missingconcept"),
        max_new_tokens=0,
    )

    assert report["source_terms"]["source_term_coverage"] < 1.0
    assert report["source_terms"]["missing_required_terms"] == ["missingconcept"]
    assert report["promotion_gate"]["grounding_support_available"] is False
    assert report["promotion_gate"]["status"] == "blocked_missing_source_terms"
