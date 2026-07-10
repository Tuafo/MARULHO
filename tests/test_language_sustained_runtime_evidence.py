from __future__ import annotations

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
    assert report["runtime"]["state_core"] == "transformer"
    assert report["runtime"]["routing_present"] is False
    assert report["external_llm_used"] is False
