from __future__ import annotations

import hashlib
import json
from pathlib import Path

from marulho.data.language_tokenizer import ByteLevelLanguageTokenizer
from marulho.evaluation.language_brain_continual_learning_evidence import (
    SURFACE,
    BrainInstalledContinualLearningEvidenceConfig,
    build_language_brain_installed_continual_learning_evidence,
)
from marulho.training.language_model import (
    LanguageModelConfig,
    MarulhoLanguageModel,
    save_language_model_checkpoint,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_review(tmp_path: Path, *, approved_hash: bool = True) -> Path:
    tokenizer = ByteLevelLanguageTokenizer()
    model = MarulhoLanguageModel(
        LanguageModelConfig(
            vocab_size=tokenizer.vocab_size + 8,
            embedding_dim=8,
            state_dim=12,
            sampled_vocab_size=16,
            sampled_vocab_sparse_lm_head_gradient=True,
            sparse_token_embedding_gradients=True,
            expert_count=2,
            active_expert_count=1,
            route_candidate_count=2,
            expert_hidden_dim=16,
            generation_vocab_size=tokenizer.vocab_size,
        )
    )
    checkpoint = tmp_path / "selected-child.pt"
    save_language_model_checkpoint(
        checkpoint,
        model,
        tokenizer,
        metadata={"source": "brain-continual-learning-evidence-test"},
    )
    checkpoint_hash = _sha256_file(checkpoint)
    recorded_hash = checkpoint_hash if approved_hash else "wrong"
    review = {
        "surface": "marulho_language_checkpoint_promotion_review.v1",
        "status": "ready_for_operator_parent_promotion_review",
        "ready": True,
        "candidate_checkpoint": {
            "surface": "marulho_language_checkpoint_promotion_candidate.v1",
            "selected_candidate_id": "candidate-03",
            "checkpoint_path": checkpoint.name,
            "checkpoint_sha256": recorded_hash,
            "checkpoint_file_exists": True,
            "checkpoint_file_sha256": recorded_hash,
            "checkpoint_hash_verified": True,
        },
        "lineage": {
            "surface": "marulho_language_checkpoint_promotion_lineage.v1",
            "quality_parent_matches_evolved_child_hash": True,
            "rollback_to_evolution_parent_verified": True,
        },
        "promotion_gate": {
            "surface": "marulho_language_checkpoint_promotion_gate.v1",
            "status": "ready_for_operator_parent_promotion_review",
            "eligible_for_operator_parent_promotion_review": True,
            "eligible_for_live_parent_replacement": False,
            "operator_approval_recorded": False,
            "writes_live_checkpoint": False,
            "mutates_runtime_state": False,
            "promotes_runtime_claim": False,
        },
    }
    review_path = tmp_path / "promotion-review.json"
    review_path.write_text(json.dumps(review), encoding="utf-8")
    return review_path


def test_brain_installed_continual_learning_evidence_learns_and_restores(
    tmp_path: Path,
) -> None:
    review_path = _write_review(tmp_path)
    output = tmp_path / "brain-installed-learning.json"

    report = build_language_brain_installed_continual_learning_evidence(
        output_path=output,
        promotion_review_path=review_path,
        operator_approved=True,
        operator_id="pytest-operator",
        approval_note="exercise brain-owned continual learning evidence",
        artifact_base_dir=tmp_path,
        config=BrainInstalledContinualLearningEvidenceConfig(
            sequence_length=10,
            stride=5,
            batch_size=2,
            max_old_eval_batches=1,
            max_new_eval_batches=1,
            max_new_batches=1,
            max_replay_batches=1,
            learning_rate=2e-2,
            max_steps=1,
            gradient_clip_interval=1,
            run_post_learning_sustained=True,
            sustained_target_tokens=4,
            sustained_tick_tokens=4,
            sustained_quantum_tokens=2,
            sustained_timeout_seconds=60.0,
            generation_no_repeat_ngram_size=2,
            device="cpu",
        ),
    )
    written = json.loads(output.read_text(encoding="utf-8"))

    assert written["surface"] == SURFACE
    assert report["report_status"] == "final"
    assert report["installation"]["installed"] is True
    assert report["candidate_checkpoint"][
        "tokenizer_hash_matches_installed_runtime"
    ] is True
    assert report["pre_learning_brain_checkpoint"]["restore_verified"] is True
    assert report["learned_brain_checkpoint"]["restore_verified"] is True
    assert report["status_read_mutation"] is False
    assert report["learning_window"]["surface"] == (
        "marulho_brain_language_learning_window.v1"
    )
    assert report["learning_window"]["trace"]["event"] == "language_learn"
    assert report["learning_window"]["report"]["mutates_language_model_weights"] is True
    assert report["learning_summary"]["update_token_count"] > 0
    assert report["learning_summary"]["tokens_per_second"] > 0.0
    assert report["learning_summary"][
        "measured_update_loop_caller_device_transfer_calls"
    ] == 0
    assert report["learning_summary"]["batch_device_staging"][
        "staged_before_measured_update_window"
    ] is True
    learning_accounting = report["learning_summary"][
        "training_window_triton_accounting"
    ]
    assert isinstance(learning_accounting["tracked_torch_fallback_calls"], int)
    assert learning_accounting["tracked_torch_fallback_call_count"] == (
        learning_accounting["tracked_torch_fallback_calls"]
    )
    assert "old_domain_forgetting" in report["learning_evidence"]
    assert "general_replay_retention_delta" in report["learning_evidence"]
    assert report["post_learning_sustained_window"]["success"] is True
    assert report["promotion_gate"]["learning_runs_through_marulho_brain"] is True
    assert report["promotion_gate"]["learned_brain_checkpoint_restore_verified"] is True
    assert report["promotion_gate"]["promotes_runtime_claim"] is False
    assert (tmp_path / "README.md").exists()


def test_brain_installed_continual_learning_evidence_blocks_bad_hash(
    tmp_path: Path,
) -> None:
    review_path = _write_review(tmp_path, approved_hash=False)

    report = build_language_brain_installed_continual_learning_evidence(
        output_path=tmp_path / "blocked.json",
        promotion_review_path=review_path,
        operator_approved=True,
        operator_id="pytest-operator",
        artifact_base_dir=tmp_path,
        config=BrainInstalledContinualLearningEvidenceConfig(
            sequence_length=10,
            stride=5,
            batch_size=2,
            max_new_batches=1,
            max_replay_batches=1,
            max_steps=1,
            run_post_learning_sustained=False,
            device="cpu",
        ),
    )

    assert report["status"] == "blocked_brain_installed_continual_learning_evidence"
    assert report["report_status"] == "partial"
    assert report["installation"]["installed"] is False
    assert report["failure_reason"] == "language_checkpoint_installation_blocked"
    assert "candidate_checkpoint_hash_matches_file" in report["installation"][
        "missing_evidence"
    ]
