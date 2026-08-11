from __future__ import annotations

import pytest

from marulho.data.language_tokenizer import ByteLevelLanguageTokenizer
from marulho.evaluation.language_sustained_runtime_evidence import (
    SURFACE,
    run_language_sustained_runtime_evidence,
)
from marulho.training.language_model import (
    LanguageModelConfig,
    MarulhoLanguageModel,
    save_language_model_checkpoint,
)


def test_transformer_sustained_report_uses_same_checkpoint(tmp_path) -> None:
    tokenizer = ByteLevelLanguageTokenizer()
    model = MarulhoLanguageModel(
        LanguageModelConfig(
            vocab_size=tokenizer.vocab_size,
            embedding_dim=16,
            state_dim=16,
            state_layers=1,
            attention_heads=4,
            transformer_context_length=32,
            transformer_mlp_ratio=2.0,
        )
    )
    checkpoint = save_language_model_checkpoint(
        tmp_path / "model.pt",
        model,
        tokenizer,
    )
    report = run_language_sustained_runtime_evidence(
        model,
        tokenizer,
        output_path=tmp_path / "sustained.json",
        target_tokens=4,
        checkpoint_path=checkpoint,
        prompt="MARULHO",
        timeout_seconds=30.0,
    )

    assert report["surface"] == SURFACE
    assert report["success"] is True
    assert report["token_delta"] == 4
    assert report["checkpoint_sha256"]
    assert report["model_state_immutable"] is True
    assert report["qualifies_sustained_runtime_contract"] is False
    assert report["active_compute"]["executed_parameter_fraction"] == 1.0
    assert report["runtime"]["state_core"] == "transformer"
    assert report["runtime"]["routing_present"] is False
    assert report["external_llm_used"] is False


def test_sustained_report_separates_aggregate_and_per_stream_tokens(tmp_path) -> None:
    tokenizer = ByteLevelLanguageTokenizer()
    model = MarulhoLanguageModel(
        LanguageModelConfig(
            vocab_size=tokenizer.vocab_size,
            embedding_dim=16,
            state_dim=16,
            state_layers=1,
            attention_heads=4,
            transformer_context_length=16,
            transformer_mlp_ratio=2.0,
        )
    )
    report = run_language_sustained_runtime_evidence(
        model,
        tokenizer,
        output_path=tmp_path / "multi.json",
        target_tokens=12,
        stream_count=3,
        prompt="runtime",
        timeout_seconds=30.0,
    )

    assert report["success"] is True
    assert report["stream_count"] == 3
    assert report["tokens_per_stream"] == 4
    assert report["token_delta"] == 12
    assert report["qualifies_sustained_runtime_contract"] is False
    assert report["single_stream_524288"] is False
    assert len(report["stream_continuation_sha256"]) == 3
    assert report["runtime"]["output_storage"] == (
        "aggregate_hashes_and_bounded_previews"
    )


def test_sustained_report_rejects_fractional_stream_work(tmp_path) -> None:
    tokenizer = ByteLevelLanguageTokenizer()
    model = MarulhoLanguageModel(
        LanguageModelConfig(
            vocab_size=tokenizer.vocab_size,
            embedding_dim=16,
            state_dim=16,
            state_layers=1,
            attention_heads=4,
            transformer_context_length=16,
            transformer_mlp_ratio=2.0,
        )
    )
    with pytest.raises(ValueError, match="divisible"):
        run_language_sustained_runtime_evidence(
            model,
            tokenizer,
            output_path=tmp_path / "invalid.json",
            target_tokens=10,
            stream_count=3,
        )
