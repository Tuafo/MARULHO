from __future__ import annotations

import json
from pathlib import Path

from marulho.brain import MarulhoBrain
from marulho.config.model_config import MarulhoConfig
from marulho.data.language_tokenizer import ByteLevelLanguageTokenizer
from marulho.evaluation.language_brain_generation_repair_evidence import (
    SURFACE,
    SWEEP_SURFACE,
    BrainInstalledGenerationRepairEvidenceConfig,
    build_language_brain_installed_generation_repair_evidence,
    build_language_brain_installed_generation_repair_sweep,
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
        ["brain repair evidence keeps learning inside MARULHO. " * 8],
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


def test_brain_installed_generation_repair_learns_and_rescores(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("MARULHO_DEVICE", "cpu")
    checkpoint = _write_installed_brain_checkpoint(tmp_path)
    output = tmp_path / "brain-installed-generation-repair.json"

    report = build_language_brain_installed_generation_repair_evidence(
        output_path=output,
        brain_checkpoint_path=checkpoint,
        repaired_brain_checkpoint_path=tmp_path / "repaired-brain.pt",
        prompt_cases=(
            LanguageGenerationPromptCase(
                prompt_text="brain",
                source_text="brain repair evidence keeps learning inside MARULHO.",
                max_new_tokens=4,
                min_new_tokens=0,
                min_prefix_match_chars=0,
                min_prefix_match_fraction=0.0,
                min_printable_fraction=0.0,
                min_distinct_bigram_fraction=0.0,
                max_token_run_length=999,
            ),
        ),
        config=BrainInstalledGenerationRepairEvidenceConfig(
            sequence_length=10,
            stride=5,
            batch_size=2,
            hard_prompt_repeat=2,
            hard_prompt_context_chars=64,
            max_new_batches=1,
            max_replay_batches=1,
            max_old_eval_batches=1,
            max_new_eval_batches=1,
            max_steps=1,
            learning_rate=1e-3,
            replay_loss_weight=0.25,
            repair_pass_count=2,
            stop_when_generation_coherence_available=False,
            min_case_pass_rate=0.0,
            run_post_repair_sustained=False,
            device="cpu",
        ),
    )
    written = json.loads(output.read_text(encoding="utf-8"))

    assert written["surface"] == SURFACE
    assert report["report_status"] == "final"
    assert report["runtime_owner"] == "MarulhoBrain"
    assert report["active_language_path"] == "marulho_lm_head"
    assert report["status_read_mutation"] is False
    assert report["learning_summary"]["trace_event"] == "language_learn"
    assert report["learning_summary"]["update_token_count"] > 0
    learning_accounting = report["learning_summary"][
        "training_window_triton_accounting"
    ]
    assert isinstance(learning_accounting["tracked_torch_fallback_calls"], int)
    assert learning_accounting["tracked_torch_fallback_call_count"] == (
        learning_accounting["tracked_torch_fallback_calls"]
    )
    assert report["aggregate_learning_summary"]["repair_pass_count"] == 2
    assert report["aggregate_learning_summary"]["update_token_count"] >= (
        report["learning_summary"]["update_token_count"]
    )
    aggregate_accounting = report["aggregate_learning_summary"][
        "training_window_triton_accounting"
    ]
    assert isinstance(aggregate_accounting["tracked_torch_fallback_calls"], int)
    assert aggregate_accounting["tracked_torch_fallback_call_count"] == (
        aggregate_accounting["tracked_torch_fallback_calls"]
    )
    assert report["executed_repair_pass_count"] == 2
    assert len(report["repair_passes"]) == 2
    assert report["promotion_gate"]["repair_pass_count"] == 2
    assert report["promotion_gate"]["executed_repair_pass_count"] == 2
    assert report["repaired_brain_checkpoint"]["restore_verified"] is True
    assert report["pre_generation_coherence"]["active_language_path"] == (
        "marulho_lm_head"
    )
    assert report["post_generation_coherence"]["active_language_path"] == (
        "marulho_lm_head"
    )
    assert report["promotion_gate"]["learning_runs_through_marulho_brain"] is True
    assert report["promotion_gate"]["post_generation_runs_through_marulho_brain"] is True
    assert report["promotion_gate"]["promotes_runtime_claim"] is False
    assert report["promotion_gate"]["promotes_generation_quality_claim"] is False
    assert (tmp_path / "README.md").exists()


def test_brain_installed_generation_repair_sweep_selects_candidate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("MARULHO_DEVICE", "cpu")
    checkpoint = _write_installed_brain_checkpoint(tmp_path)
    output = tmp_path / "brain-installed-generation-repair-sweep.json"

    report = build_language_brain_installed_generation_repair_sweep(
        output_path=output,
        brain_checkpoint_path=checkpoint,
        prompt_cases=(
            LanguageGenerationPromptCase(
                prompt_text="brain",
                source_text="brain repair evidence keeps learning inside MARULHO.",
                max_new_tokens=4,
                min_new_tokens=0,
                min_prefix_match_chars=0,
                min_prefix_match_fraction=0.0,
                min_printable_fraction=0.0,
                min_distinct_bigram_fraction=0.0,
                max_token_run_length=999,
            ),
        ),
        config=BrainInstalledGenerationRepairEvidenceConfig(
            sequence_length=10,
            stride=5,
            batch_size=2,
            hard_prompt_repeat=2,
            hard_prompt_context_chars=64,
            max_new_batches=1,
            max_replay_batches=1,
            max_old_eval_batches=1,
            max_new_eval_batches=1,
            max_steps=1,
            learning_rate=1e-3,
            replay_loss_weight=0.25,
            candidate_learning_rates=(1e-3, 5e-4),
            candidate_replay_loss_weights=(0.25, 0.75),
            candidate_max_steps=(1, 1),
            candidate_repair_pass_counts=(1, 2),
            min_case_pass_rate=0.0,
            run_post_repair_sustained=False,
            device="cpu",
        ),
    )
    written = json.loads(output.read_text(encoding="utf-8"))
    selection = report["candidate_selection"]
    selected = selection["selected_candidate_id"]
    selected_candidates = [
        candidate for candidate in selection["candidates"] if candidate["selected"]
    ]

    assert written["surface"] == SWEEP_SURFACE
    assert report["report_status"] == "final"
    assert selection["candidate_count"] == 2
    assert len(selection["candidates"]) == 2
    assert len(selected_candidates) == 1
    assert selected in {"candidate-00", "candidate-01"}
    assert selection["mutates_parent_checkpoint"] is False
    assert selection["runs_sustained_runtime_only_for_selected_child"] is True
    assert Path(selection["selected_repair_report_path"]).exists()
    assert Path(selection["selected_repaired_brain_checkpoint_path"]).exists()
    assert report["promotion_gate"]["candidate_count"] == 2
    assert report["promotion_gate"]["selected_report_final"] is True
    assert report["promotion_gate"]["promotes_runtime_claim"] is False
    assert report["promotion_gate"]["promotes_generation_quality_claim"] is False
    assert (tmp_path / "brain-installed-generation-repair-sweep-candidate-00-repair.json").exists()
    assert (tmp_path / "brain-installed-generation-repair-sweep-candidate-01-repair.json").exists()


def test_brain_installed_generation_repair_blocks_missing_language_runtime(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("MARULHO_DEVICE", "cpu")
    brain = MarulhoBrain.fresh(_tiny_config())
    saved = brain.save(tmp_path / "no-language-brain.pt")

    report = build_language_brain_installed_generation_repair_evidence(
        output_path=tmp_path / "blocked.json",
        brain_checkpoint_path=saved["path"],
        config=BrainInstalledGenerationRepairEvidenceConfig(device="cpu"),
    )

    assert report["status"] == "blocked_brain_installed_generation_repair_evidence"
    assert report["report_status"] == "partial"
    assert report["failure_reason"] == "brain_language_runtime_missing"
