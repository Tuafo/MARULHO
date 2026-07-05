from __future__ import annotations

import hashlib
import json
from pathlib import Path

from marulho.data.language_tokenizer import ByteLevelLanguageTokenizer
from marulho.evaluation.language_brain_checkpoint_runtime_evidence import (
    SURFACE,
    build_language_brain_checkpoint_runtime_evidence,
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
            generation_vocab_size=tokenizer.vocab_size,
        )
    )
    checkpoint = tmp_path / "selected-child.pt"
    save_language_model_checkpoint(
        checkpoint,
        model,
        tokenizer,
        metadata={"source": "brain-runtime-evidence-test"},
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


def test_language_brain_checkpoint_runtime_evidence_installs_restores_and_generates(
    tmp_path: Path,
) -> None:
    review_path = _write_review(tmp_path)
    output = tmp_path / "brain-runtime-evidence.json"

    report = build_language_brain_checkpoint_runtime_evidence(
        output_path=output,
        promotion_review_path=review_path,
        brain_checkpoint_path=tmp_path / "installed-brain.pt",
        operator_approved=True,
        operator_id="pytest-operator",
        artifact_base_dir=tmp_path,
        device="cpu",
        prompt="MARULHO",
        target_tokens=8,
        chunk_tokens=4,
        timeout_seconds=60.0,
    )
    written = json.loads(output.read_text(encoding="utf-8"))

    assert written["surface"] == SURFACE
    assert report["report_status"] == "final"
    assert report["installation"]["installed"] is True
    assert report["brain_checkpoint"]["restore_verified"] is True
    assert report["active_language_path"] == "marulho_lm_head"
    assert report["token_delta"] == 8
    assert report["tokens_per_second"] > 0.0
    assert report["training_owned_sustained_window"]["success"] is True
    assert report["training_owned_sustained_window"]["token_delta"] == 8
    assert report["training_owned_tokens_per_second"] > 0.0
    assert report["status_read_mutation"] is False
    assert report["promotion_gate"]["target_tokens_reached"] is True
    assert report["promotion_gate"]["training_owned_sustained_target_reached"] is True
    assert report["promotion_gate"]["ready_for_runtime_claim_review"] is False
    assert report["promotion_gate"]["promotes_runtime_claim"] is False
    assert report["restored_brain"]["language_model"]["vocab_policy"][
        "padded_vocab_rows"
    ] == 8
    assert report["generation_window"]["last_generation"]["generation_decode"][
        "padded_vocab_rows_masked"
    ] == 8
    assert (tmp_path / "README.md").exists()


def test_language_brain_checkpoint_runtime_evidence_blocks_bad_review_hash(
    tmp_path: Path,
) -> None:
    review_path = _write_review(tmp_path, approved_hash=False)

    report = build_language_brain_checkpoint_runtime_evidence(
        output_path=tmp_path / "blocked.json",
        promotion_review_path=review_path,
        brain_checkpoint_path=tmp_path / "blocked-brain.pt",
        operator_approved=True,
        operator_id="pytest-operator",
        artifact_base_dir=tmp_path,
        device="cpu",
        target_tokens=8,
        chunk_tokens=4,
    )

    assert report["status"] == "blocked_language_brain_checkpoint_runtime_evidence"
    assert report["report_status"] == "partial"
    assert report["installation"]["installed"] is False
    assert report["failure_reason"] == "language_checkpoint_installation_blocked"
    assert "candidate_checkpoint_hash_matches_file" in report["installation"][
        "missing_evidence"
    ]
