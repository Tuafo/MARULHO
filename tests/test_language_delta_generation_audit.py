from __future__ import annotations

from pathlib import Path

from marulho.data.language_tokenizer import ByteLevelLanguageTokenizer
from marulho.evaluation.language_delta_generation_audit import (
    run_delta_generation_audit,
)
from marulho.training.language_delta import (
    DeltaLanguageConfig,
    MarulhoDeltaLanguageModel,
    save_delta_language_checkpoint,
)


def test_delta_generation_audit_records_unseen_owned_outputs(tmp_path: Path) -> None:
    tokenizer = ByteLevelLanguageTokenizer()
    model = MarulhoDeltaLanguageModel(
        DeltaLanguageConfig(
            vocab_size=tokenizer.vocab_size,
            width=16,
            layers=2,
            memory_heads=2,
            memory_head_dim=4,
            attention_heads=2,
            local_attention_every=2,
            context_length=8,
            mlp_dim=32,
        )
    )
    checkpoint = save_delta_language_checkpoint(
        tmp_path / "delta.pt",
        model,
        tokenizer,
        metadata={"cumulative_update_tokens": 10, "optimizer_steps": 2},
    )
    source = tmp_path / "source.txt"
    source.write_text("Completely unrelated source material.", encoding="utf-8")
    report = run_delta_generation_audit(
        checkpoint_path=checkpoint,
        output_path=tmp_path / "audit.json",
        prompts=("An original prompt",),
        source_paths=(source,),
        max_new_tokens=3,
        device="cpu",
    )
    assert report["owned_by_marulho"] is True
    assert report["external_llm_used"] is False
    assert report["prompt_contract"]["exact_prompt_absence_verified"] is True
    assert len(report["generations"]) == 2
    assert report["decision_boundary"]["promotes_unseen_generation_quality"] is False
