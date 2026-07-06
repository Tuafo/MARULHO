from __future__ import annotations

import json
from pathlib import Path

from marulho.brain import MarulhoBrain
from marulho.config.model_config import MarulhoConfig
from marulho.data.language_tokenizer import ByteLevelLanguageTokenizer
from marulho.evaluation.language_brain_generation_evidence import (
    SURFACE,
    BrainInstalledGenerationEvidenceConfig,
    build_language_brain_installed_generation_evidence,
)
from marulho.evaluation.language_generation_coherence import (
    LanguageGenerationPromptCase,
)
from marulho.training.language_model import (
    LanguageModelConfig,
    MarulhoLanguageModel,
    build_language_model_splits,
    evaluate_language_model,
)


def _tiny_config() -> MarulhoConfig:
    return MarulhoConfig(
        n_columns=16,
        column_latent_dim=16,
        window_size=4,
        bootstrap_tokens=0,
        k_routing=4,
        memory_capacity=128,
        enable_learned_chunking=False,
        micro_sleep_interval_tokens=10_000,
        deep_sleep_interval_tokens=10_000,
        trainer_telemetry_interval_tokens=10_000,
        device="cpu",
    )


def _write_installed_brain_checkpoint(tmp_path: Path) -> Path:
    brain = MarulhoBrain.fresh(_tiny_config())
    tokenizer = ByteLevelLanguageTokenizer()
    split = build_language_model_splits(
        ["brain installed generation evidence stays MARULHO owned. " * 8],
        tokenizer,
        sequence_length=10,
        eval_fraction=0.25,
        stride=5,
        batch_size=2,
    )
    model = MarulhoLanguageModel(
        LanguageModelConfig(
            vocab_size=tokenizer.vocab_size,
            embedding_dim=8,
            state_dim=12,
            expert_count=3,
            active_expert_count=1,
            route_candidate_count=2,
            expert_hidden_dim=16,
        )
    )
    brain.install_language_model(
        model,
        tokenizer,
        evaluation_report=evaluate_language_model(model, split.eval),
    )
    saved = brain.save(tmp_path / "installed-brain.pt")
    return Path(saved["path"])


def test_brain_installed_generation_runs_through_public_brain_generate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("MARULHO_DEVICE", "cpu")
    checkpoint = _write_installed_brain_checkpoint(tmp_path)
    output = tmp_path / "brain-installed-generation.json"

    report = build_language_brain_installed_generation_evidence(
        output_path=output,
        brain_checkpoint_path=checkpoint,
        prompt_cases=(
            LanguageGenerationPromptCase(
                prompt_text="brain",
                source_text="brain installed generation evidence stays MARULHO owned.",
                max_new_tokens=4,
                min_new_tokens=0,
                min_prefix_match_chars=0,
                min_prefix_match_fraction=0.0,
                min_printable_fraction=0.0,
                min_distinct_bigram_fraction=0.0,
                max_token_run_length=999,
            ),
        ),
        config=BrainInstalledGenerationEvidenceConfig(
            min_case_pass_rate=0.0,
            generation_repetition_penalty=1.05,
            generation_no_repeat_ngram_size=2,
        ),
    )
    written = json.loads(output.read_text(encoding="utf-8"))

    assert written["surface"] == SURFACE
    assert report["report_status"] == "final"
    assert report["runtime_owner"] == "MarulhoBrain"
    assert report["active_language_path"] == "marulho_lm_head"
    assert report["status_read_mutation"] is False
    assert report["owned_by_marulho"] is True
    assert report["external_llm_used"] is False
    assert report["loads_external_checkpoint"] is False
    assert report["service_owned_cognition"] is False
    assert report["generation_summary"]["case_count"] == 1
    assert report["generation_coherence"]["active_language_path"] == "marulho_lm_head"
    assert report["promotion_gate"]["generation_runs_through_marulho_brain"] is True
    assert report["promotion_gate"]["status_read_mutation_absent"] is True
    assert report["promotion_gate"]["promotes_runtime_claim"] is False
    assert report["promotion_gate"]["promotes_generation_quality_claim"] is False
    assert (tmp_path / "README.md").exists()


def test_brain_installed_generation_blocks_missing_language_runtime(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("MARULHO_DEVICE", "cpu")
    brain = MarulhoBrain.fresh(_tiny_config())
    saved = brain.save(tmp_path / "no-language-brain.pt")

    report = build_language_brain_installed_generation_evidence(
        output_path=tmp_path / "blocked.json",
        brain_checkpoint_path=saved["path"],
    )

    assert report["status"] == "blocked_brain_installed_generation_evidence"
    assert report["report_status"] == "partial"
    assert report["failure_reason"] == "brain_language_runtime_missing"
