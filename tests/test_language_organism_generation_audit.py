from __future__ import annotations

from pathlib import Path

from marulho.data.language_tokenizer import ByteLevelLanguageTokenizer
from marulho.evaluation.language_organism_generation_audit import (
    GenerationAuditPrompt,
    run_organism_generation_audit,
)
from marulho.training.language_organism import (
    DistributedLanguageConfig,
    MarulhoDistributedLanguageModel,
    save_distributed_language_checkpoint,
)


def test_organism_generation_audit_records_source_absent_owned_outputs(
    tmp_path: Path,
) -> None:
    tokenizer = ByteLevelLanguageTokenizer()
    model = MarulhoDistributedLanguageModel(
        DistributedLanguageConfig(
            vocab_size=tokenizer.vocab_size,
            width=16,
            layers=2,
            attention_heads=2,
            context_length=8,
            unit_groups=2,
            workspace_slots=2,
            episodic_slots=4,
            state_update_interval=4,
            mlp_dim=32,
            counterfactual_rate=0.0,
        )
    )
    checkpoint = save_distributed_language_checkpoint(
        tmp_path / "organism.pt",
        model,
        tokenizer,
        metadata={"cumulative_update_tokens": 10, "optimizer_steps": 2},
    )
    source = tmp_path / "source.txt"
    source.write_text(
        "x" * (1024 * 1024 - 8) + "boundary phrase" + " unrelated material",
        encoding="utf-8",
    )
    prompts = (
        GenerationAuditPrompt(
            prompt_id="unseen",
            capability="test",
            text="An original prompt",
            review_question="Is this coherent?",
        ),
        GenerationAuditPrompt(
            prompt_id="boundary",
            capability="test",
            text="boundary phrase",
            review_question="Was the boundary match detected?",
        ),
    )
    report = run_organism_generation_audit(
        checkpoint_path=checkpoint,
        output_path=tmp_path / "audit.json",
        prompts=prompts,
        source_paths=(source,),
        max_new_tokens=3,
        device="cpu",
    )
    assert report["owned_by_marulho"] is True
    assert report["external_llm_used"] is False
    assert report["prompt_contract"]["exact_prompt_absence_verified"] is False
    assert len(report["generations"]) == 4
    unseen = [
        row for row in report["generations"] if row["prompt_id"] == "unseen"
    ]
    boundary = [
        row for row in report["generations"] if row["prompt_id"] == "boundary"
    ]
    assert all(not row["prompt_exactly_present_in_sources"] for row in unseen)
    assert all(row["prompt_exactly_present_in_sources"] for row in boundary)
    assert all(row["review_question_used_for_generation"] is False for row in unseen)
    assert report["decision_boundary"]["promotes_unseen_generation_quality"] is False
