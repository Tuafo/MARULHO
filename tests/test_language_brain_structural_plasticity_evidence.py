from __future__ import annotations

import json
from pathlib import Path

from marulho.brain import MarulhoBrain
from marulho.config.model_config import MarulhoConfig
from marulho.data.language_tokenizer import ByteLevelLanguageTokenizer
from marulho.evaluation.language_brain_structural_plasticity_evidence import (
    SURFACE,
    BrainInstalledStructuralPlasticityEvidenceConfig,
    build_language_brain_installed_structural_plasticity_evidence,
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
        ["brain installed structural route-bank evidence needs rollback. " * 8],
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
            expert_count=5,
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


def test_brain_installed_structural_plasticity_applies_and_restores(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("MARULHO_DEVICE", "cpu")
    checkpoint = _write_installed_brain_checkpoint(tmp_path)
    output = tmp_path / "brain-installed-structure.json"

    report = build_language_brain_installed_structural_plasticity_evidence(
        output_path=output,
        brain_checkpoint_path=checkpoint,
        operator_approved=True,
        config=BrainInstalledStructuralPlasticityEvidenceConfig(
            sequence_length=10,
            stride=5,
            batch_size=2,
            max_eval_batches=1,
            route_candidate_growth=2,
            run_post_structure_sustained=True,
            sustained_target_tokens=4,
            sustained_tick_tokens=4,
            sustained_quantum_tokens=2,
            sustained_timeout_seconds=60.0,
            generation_no_repeat_ngram_size=2,
        ),
    )
    written = json.loads(output.read_text(encoding="utf-8"))

    assert written["surface"] == SURFACE
    assert report["report_status"] == "final"
    assert report["status_read_mutation"] is False
    assert report["proposal"]["proposal"]["proposal_kind"] == "route_bank_expansion"
    assert report["proposal_read_only"]["mutates_runtime_state"] is False
    summary = report["structural_transaction_summary"]
    assert summary["applied"] is True
    assert summary["trace_event"] == "language_structure"
    assert summary["source_route_candidate_count"] == 2
    assert summary["target_route_candidate_count"] == 4
    assert summary["checkpoint_restore_verified"] is True
    assert summary["rollback_verified"] is True
    assert report["post_structural_brain_checkpoint"]["restore_verified"] is True
    assert report["post_structure_sustained_window"]["success"] is True
    assert report["promotion_gate"]["records_reviewed_structural_mutation"] is True
    assert report["promotion_gate"]["post_structure_sustained_target_reached"] is True
    assert report["promotion_gate"]["promotes_runtime_claim"] is False
    assert (tmp_path / "README.md").exists()


def test_brain_installed_structural_plasticity_blocks_missing_language_runtime(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("MARULHO_DEVICE", "cpu")
    brain = MarulhoBrain.fresh(_tiny_config())
    saved = brain.save(tmp_path / "no-language-brain.pt")

    report = build_language_brain_installed_structural_plasticity_evidence(
        output_path=tmp_path / "blocked.json",
        brain_checkpoint_path=saved["path"],
        operator_approved=True,
        config=BrainInstalledStructuralPlasticityEvidenceConfig(
            max_eval_batches=1,
            run_post_structure_sustained=False,
        ),
    )

    assert report["status"] == "blocked_brain_installed_structural_plasticity_evidence"
    assert report["report_status"] == "partial"
    assert report["failure_reason"] == "brain_language_runtime_missing"
