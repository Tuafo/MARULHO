from pathlib import Path

from marulho.data.language_tokenizer import ByteLevelLanguageTokenizer
from marulho.evaluation.language_decode_comparison import (
    SURFACE,
    DecodePolicy,
    run_language_decode_comparison,
)
from marulho.training.language_model import (
    LanguageModelConfig,
    MarulhoLanguageModel,
    save_language_model_checkpoint,
)


def test_decode_comparison_records_greedy_and_seeded_nucleus(tmp_path: Path) -> None:
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
    checkpoint = save_language_model_checkpoint(
        tmp_path / "model.pt",
        model,
        tokenizer,
        metadata={"cumulative_update_tokens": 32},
    )
    report = run_language_decode_comparison(
        checkpoint_path=checkpoint,
        output_path=tmp_path / "decode.json",
        prompts=("Test",),
        policies=(
            DecodePolicy("greedy", 0.0, 1.0, 5),
            DecodePolicy("nucleus", 0.8, 0.9, 5),
        ),
        max_new_tokens=2,
        device="cpu",
    )

    assert report["surface"] == SURFACE
    assert report["checkpoint"]["cumulative_update_tokens"] == 32
    assert report["generation_count"] == 2
    assert report["generations"][0]["generation_decode"]["decode_strategy"] == (
        "greedy_argmax"
    )
    assert report["generations"][1]["generation_decode"]["decode_strategy"] == (
        "nucleus_sampling"
    )
    assert report["quality_boundary"]["promotes_generation_quality_claim"] is False
