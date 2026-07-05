from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import time

from fastapi.testclient import TestClient
import torch

from marulho.brain import MarulhoBrain
from marulho.config.model_config import MarulhoConfig
from marulho.data.language_tokenizer import ByteLevelLanguageTokenizer
from marulho.service.api import create_app
from marulho.training.checkpointing import save_trainer_checkpoint
from marulho.training.language_checkpoint_evolution import LanguageCheckpointEvolutionConfig
from marulho.training.language_continual_learning import LanguageContinualLearningConfig
from marulho.training.language_model import (
    LanguageModelConfig,
    MarulhoLanguageModel,
    build_language_model_splits,
    evaluate_language_model,
    save_language_model_checkpoint,
)
from marulho.training.language_structural_plasticity import (
    LanguageStructuralPlasticityConfig,
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


def _trained_brain() -> MarulhoBrain:
    brain = MarulhoBrain.fresh(_tiny_config())
    feed = brain.feed("marulho learns local sparse loops.", source="test")
    assert feed["accepted_tokens"] > 0
    tick = brain.tick(tokens=feed["accepted_tokens"], source="test")
    assert tick["trained_tokens"] == feed["accepted_tokens"]
    return brain


def _language_model_fixture() -> tuple[
    MarulhoLanguageModel,
    ByteLevelLanguageTokenizer,
    dict[str, object],
]:
    torch.manual_seed(20260703)
    tokenizer = ByteLevelLanguageTokenizer()
    split = build_language_model_splits(
        [
            "marulho language head predicts byte tokens from owned state. " * 4,
            "checkpointed runtime evidence keeps external llms absent. " * 4,
        ],
        tokenizer,
        sequence_length=10,
        eval_fraction=0.25,
    )
    model = MarulhoLanguageModel(
        LanguageModelConfig(
            vocab_size=tokenizer.vocab_size,
            embedding_dim=8,
            state_dim=12,
        )
    )
    report = evaluate_language_model(model, split.eval)
    return model, tokenizer, report


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_ready_language_checkpoint_promotion_review(
    tmp_path: Path,
    *,
    padded_vocab_rows: int = 0,
) -> tuple[Path, Path, str]:
    if padded_vocab_rows > 0:
        tokenizer = ByteLevelLanguageTokenizer()
        model = MarulhoLanguageModel(
            LanguageModelConfig(
                vocab_size=tokenizer.vocab_size + int(padded_vocab_rows),
                embedding_dim=8,
                state_dim=12,
                generation_vocab_size=tokenizer.vocab_size,
            )
        )
    else:
        model, tokenizer, _report = _language_model_fixture()
    checkpoint_path = tmp_path / "selected-child.pt"
    save_language_model_checkpoint(
        checkpoint_path,
        model,
        tokenizer,
        metadata={
            "source": "test_reviewed_child",
            "owned_by_marulho": True,
        },
    )
    checkpoint_hash = _sha256_file(checkpoint_path)
    review_path = tmp_path / "promotion-review.json"
    review_path.write_text(
        json.dumps(
            {
                "surface": "marulho_language_checkpoint_promotion_review.v1",
                "status": "ready_for_operator_parent_promotion_review",
                "ready": True,
                "candidate_checkpoint": {
                    "surface": "marulho_language_checkpoint_promotion_candidate.v1",
                    "selected_candidate_id": "candidate-03",
                    "checkpoint_path": checkpoint_path.name,
                    "checkpoint_sha256": checkpoint_hash,
                    "checkpoint_file_exists": True,
                    "checkpoint_file_sha256": checkpoint_hash,
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
                    "promotes_parent_promotion": False,
                    "promotes_generation_quality_claim": False,
                    "promotes_runtime_claim": False,
                },
            }
        ),
        encoding="utf-8",
    )
    return review_path, checkpoint_path, checkpoint_hash


def test_marulho_brain_feed_tick_generate_and_trace() -> None:
    brain = _trained_brain()

    generation = brain.generate(prompt="mar", max_tokens=12)
    status = brain.status()

    assert generation["owned_by_marulho"] is True
    assert generation["external_llm_used"] is False
    assert generation["thought_loop_used"] is False
    assert generation["cortex_used"] is False
    assert generation["emitted_tokens"] > 0
    assert status["surface"] == "marulho_brain_runtime.v1"
    assert status["readout"]["observed_transition_count"] > 0
    assert status["last_trace"]["executor"] == brain.trainer.config.cuda_graph_sequence_executor
    assert status["retired_brain_surfaces"]["external_llm_used"] is False


def test_marulho_brain_tick_uses_trainer_winner_for_readout_keys() -> None:
    brain = MarulhoBrain.fresh(_tiny_config())
    feed = brain.feed("marulho readout keys stay off the hot path.", source="test")
    assert feed["accepted_tokens"] > 0

    def fail_offline_winner(_pattern: object) -> int:
        raise AssertionError("tick should not recompute offline winners per token")

    brain.trainer.winner_for_pattern = fail_offline_winner  # type: ignore[method-assign]
    tick = brain.tick(tokens=feed["accepted_tokens"], source="test")

    assert tick["trained_tokens"] == feed["accepted_tokens"]
    assert brain.status()["readout"]["observed_transition_count"] > 0


def test_marulho_brain_feed_does_not_learn_chunks_unless_requested() -> None:
    config = _tiny_config()
    config.enable_learned_chunking = True
    brain = MarulhoBrain.fresh(config)
    chunking = brain.encoder.learned_chunking
    assert chunking is not None

    feed = brain.feed("chunk learning must stay explicit.", source="test", learn=False)

    assert feed["accepted_tokens"] > 0
    assert feed["learned_immediately"] is False
    assert float(chunking.usage.sum().item()) == 0.0


def test_marulho_brain_checkpoint_roundtrip_preserves_readout(tmp_path: Path) -> None:
    brain = _trained_brain()
    before = brain.generate(max_tokens=12)
    checkpoint_path = tmp_path / "brain.pt"

    saved = brain.save(checkpoint_path)
    restored = MarulhoBrain.load(saved["path"])
    after = restored.generate(max_tokens=12)

    assert saved["path"] == str(checkpoint_path)
    assert restored.status()["token_count"] == brain.status()["token_count"]
    assert restored.status()["readout"]["observed_transition_count"] > 0
    assert after["text"] == before["text"]


def test_marulho_brain_uses_checkpointed_lm_head_when_installed(tmp_path: Path) -> None:
    brain = MarulhoBrain.fresh(_tiny_config())
    model, tokenizer, report = _language_model_fixture()

    install = brain.install_language_model(
        model,
        tokenizer,
        evaluation_report=report,
    )
    generation = brain.generate(
        prompt="marulho",
        max_tokens=4,
        generation_repetition_penalty=1.2,
        generation_no_repeat_ngram_size=1,
    )
    status = brain.status()
    saved = brain.save(tmp_path / "brain-lm.pt")
    restored = MarulhoBrain.load(saved["path"])
    restored_generation = restored.generate(
        prompt="marulho",
        max_tokens=4,
        generation_repetition_penalty=1.2,
        generation_no_repeat_ngram_size=1,
    )

    assert install["surface"] == "marulho_brain_language_model_install.v1"
    assert install["active_language_path"] == "marulho_lm_head"
    assert generation["surface"] == "marulho_brain_language_model_generation.v1"
    assert generation["active_language_path"] == "marulho_lm_head"
    assert generation["transition_readout_fallback_used"] is False
    assert generation["checkpointed_language_components"] is True
    assert generation["external_llm_used"] is False
    assert generation["thought_loop_used"] is False
    assert generation["cortex_used"] is False
    assert generation["emitted_tokens"] > 0
    assert generation["generation_decode"]["repetition_penalty_applied"] is True
    assert generation["generation_decode"]["repetition_penalty"] == 1.2
    assert generation["generation_decode"]["no_repeat_ngram_applied"] is True
    assert generation["generation_decode"]["no_repeat_ngram_size"] == 1
    assert generation["generation_decode"]["decode_controls_backend"] == (
        "torch_device_tensor"
    )
    assert generation["generation_decode"]["decode_controls_cpu_token_copy"] is False
    assert generation["tokenizer_hash"] == tokenizer.vocabulary_hash()
    assert status["active_language_path"] == "marulho_lm_head"
    assert status["language_model"]["checkpointed_language_components"] is True
    assert status["language_model"]["heldout_evaluation_available"] is True
    assert status["last_trace"]["active_language_path"] == "marulho_lm_head"
    assert restored.status()["active_language_path"] == "marulho_lm_head"
    assert restored_generation["active_language_path"] == "marulho_lm_head"
    assert restored_generation["generated_token_ids"] == generation["generated_token_ids"]


def test_marulho_brain_installs_reviewed_language_checkpoint(tmp_path: Path) -> None:
    brain = MarulhoBrain.fresh(_tiny_config())
    review_path, _checkpoint_path, checkpoint_hash = (
        _write_ready_language_checkpoint_promotion_review(
            tmp_path,
            padded_vocab_rows=16,
        )
    )

    install = brain.install_language_checkpoint_from_promotion_review(
        review_path,
        operator_approved=True,
        operator_id="pytest-operator",
        approval_note="install reviewed child into brain-owned runtime",
        artifact_base_dir=tmp_path,
    )
    generation = brain.generate(prompt="marulho", max_tokens=4)
    status = brain.status()
    saved = brain.save(tmp_path / "reviewed-language-brain.pt")
    restored = MarulhoBrain.load(saved["path"])
    restored_status = restored.status()

    assert install["surface"] == "marulho_brain_language_checkpoint_installation.v1"
    assert install["status"] == "installed_reviewed_language_checkpoint"
    assert install["installed"] is True
    assert install["runtime_owner"] == "MarulhoBrain"
    assert install["active_language_path"] == "marulho_lm_head"
    assert install["model_vocab_size"] == status["language_model"]["model_vocab_size"]
    assert status["language_model"]["vocab_policy"]["padded_vocab_rows"] == 16
    assert status["language_model"]["vocab_policy"]["padded_vocab_decode_policy"] == (
        "limit_generation_to_tokenizer_vocab_rows"
    )
    assert install["candidate_checkpoint"]["hash_verified"] is True
    assert install["candidate_checkpoint"]["actual_sha256"] == checkpoint_hash
    assert install["approval"]["operator_approved"] is True
    assert install["mutates_runtime_state"] is True
    assert install["writes_live_checkpoint"] is False
    assert install["status_read_mutation"] is False
    assert install["service_owned_cognition"] is False
    assert install["external_llm_used"] is False
    assert install["loads_external_checkpoint"] is False
    assert install["promotes_runtime_claim"] is False
    assert install["trace"]["event"] == "language_checkpoint_install"
    assert install["language_model"]["checkpoint_installation_count"] == 1
    assert generation["active_language_path"] == "marulho_lm_head"
    assert generation["checkpointed_language_components"] is True
    assert generation["generation_decode"]["generation_vocab_size"] == (
        status["language_model"]["vocab_size"]
    )
    assert generation["generation_decode"]["padded_vocab_rows_masked"] == 16
    assert status["language_model"]["checkpoint_installation_count"] == 1
    assert status["language_model"]["last_checkpoint_installation"]["status"] == (
        "installed_reviewed_language_checkpoint"
    )
    assert restored_status["active_language_path"] == "marulho_lm_head"
    assert restored_status["language_model"]["checkpoint_installation_count"] == 1
    assert restored_status["language_model"]["last_checkpoint_installation"][
        "candidate_checkpoint"
    ]["hash_verified"] is True


def test_marulho_brain_blocks_reviewed_language_checkpoint_without_operator_approval(
    tmp_path: Path,
) -> None:
    brain = MarulhoBrain.fresh(_tiny_config())
    review_path, _checkpoint_path, _checkpoint_hash = (
        _write_ready_language_checkpoint_promotion_review(tmp_path)
    )

    install = brain.install_language_checkpoint_from_promotion_review(
        review_path,
        operator_approved=False,
        artifact_base_dir=tmp_path,
    )

    assert install["status"] == "blocked_language_checkpoint_installation"
    assert install["installed"] is False
    assert install["mutates_runtime_state"] is False
    assert "operator_approval_recorded" in install["missing_evidence"]
    assert brain.status()["active_language_path"] == "local_transition_readout"
    assert brain.status()["language_model"]["checkpointed_language_components"] is False


def test_marulho_brain_blocks_reviewed_language_checkpoint_hash_mismatch(
    tmp_path: Path,
) -> None:
    brain = MarulhoBrain.fresh(_tiny_config())
    review_path, _checkpoint_path, _checkpoint_hash = (
        _write_ready_language_checkpoint_promotion_review(tmp_path)
    )
    review = json.loads(review_path.read_text(encoding="utf-8"))
    review["candidate_checkpoint"]["checkpoint_sha256"] = "wrong"
    review["candidate_checkpoint"]["checkpoint_file_sha256"] = "wrong"
    review_path.write_text(json.dumps(review), encoding="utf-8")

    install = brain.install_language_checkpoint_from_promotion_review(
        review_path,
        operator_approved=True,
        operator_id="pytest-operator",
        artifact_base_dir=tmp_path,
    )

    assert install["status"] == "blocked_language_checkpoint_installation"
    assert install["installed"] is False
    assert install["mutates_runtime_state"] is False
    assert install["candidate_checkpoint"]["hash_verified"] is False
    assert "candidate_checkpoint_hash_matches_file" in install["missing_evidence"]
    assert brain.status()["active_language_path"] == "local_transition_readout"


def test_brain_service_uses_restored_lm_head_without_service_owner(tmp_path: Path) -> None:
    brain = MarulhoBrain.fresh(_tiny_config())
    model, tokenizer, report = _language_model_fixture()
    brain.install_language_model(model, tokenizer, evaluation_report=report)
    checkpoint_path = tmp_path / "service-lm-brain.pt"
    brain.save(checkpoint_path)
    app = create_app(
        checkpoint_path=checkpoint_path,
        trace_dir=tmp_path / "traces",
        env_root=tmp_path,
    )

    with TestClient(app) as client:
        generate = client.post(
            "/brain/generate",
            json={
                "prompt": "marulho",
                "max_tokens": 4,
                "generation_repetition_penalty": 1.2,
                "generation_no_repeat_ngram_size": 1,
            },
        )
        status = client.get("/brain/status")

    assert generate.status_code == 200
    assert generate.json()["active_language_path"] == "marulho_lm_head"
    assert generate.json()["external_llm_used"] is False
    assert generate.json()["generation_decode"]["repetition_penalty_applied"] is True
    assert generate.json()["generation_decode"]["no_repeat_ngram_applied"] is True
    assert generate.json()["generation_decode"]["decode_controls_backend"] == (
        "torch_device_tensor"
    )
    assert generate.json()["generation_decode"]["decode_controls_cpu_token_copy"] is False
    assert generate.json()["tokenizer_hash"] == tokenizer.vocabulary_hash()
    assert status.status_code == 200
    assert status.json()["active_language_path"] == "marulho_lm_head"
    assert status.json()["language_model"]["checkpointed_language_components"] is True


def test_marulho_brain_language_learning_window_is_checkpointed(tmp_path: Path) -> None:
    brain = MarulhoBrain.fresh(_tiny_config())
    model, tokenizer, report = _language_model_fixture()
    brain.install_language_model(model, tokenizer, evaluation_report=report)
    old_split = build_language_model_splits(
        ["old runtime evidence keeps replay protected. " * 6],
        tokenizer,
        sequence_length=10,
        eval_fraction=0.25,
    )
    new_split = build_language_model_splits(
        ["new online language window updates owned weights. " * 6],
        tokenizer,
        sequence_length=10,
        eval_fraction=0.25,
    )

    learning = brain.learn_language_window(
        new_batches=new_split.train[:2],
        old_eval_batches=old_split.eval,
        new_eval_batches=new_split.train[:2],
        replay_batches=old_split.train[:1],
        config=LanguageContinualLearningConfig(
            learning_rate=2e-2,
            max_steps=3,
            forgetting_tolerance=100.0,
            replay_retention_tolerance=100.0,
        ),
    )
    status = brain.status()
    saved = brain.save(tmp_path / "brain-language-learn.pt")
    restored = MarulhoBrain.load(saved["path"])
    restored_status = restored.status()

    assert learning["surface"] == "marulho_brain_language_learning_window.v1"
    assert learning["active_language_path"] == "marulho_lm_head"
    assert learning["external_llm_used"] is False
    assert learning["report"]["mutates_language_model_weights"] is True
    assert learning["report"]["learning_evidence"]["final_parameter_delta_l2"] > 0.0
    assert learning["trace"]["event"] == "language_learn"
    assert status["language_model"]["continual_learning_window_count"] == 1
    assert status["language_model"]["last_continual_learning"]["surface"] == (
        "marulho_language_continual_learning_window.v1"
    )
    assert restored_status["active_language_path"] == "marulho_lm_head"
    assert restored_status["language_model"]["continual_learning_window_count"] == 1
    assert restored_status["language_model"]["last_continual_learning"]["rollback_evidence"][
        "restore_verified"
    ] is True


def test_marulho_brain_language_structural_transaction_is_checkpointed(tmp_path: Path) -> None:
    brain = MarulhoBrain.fresh(_tiny_config())
    tokenizer = ByteLevelLanguageTokenizer()
    split = build_language_model_splits(
        ["structural expert growth needs checkpoint rollback evidence. " * 6],
        tokenizer,
        sequence_length=10,
        eval_fraction=0.25,
    )
    model = MarulhoLanguageModel(
        LanguageModelConfig(
            vocab_size=tokenizer.vocab_size,
            embedding_dim=8,
            state_dim=12,
            expert_count=2,
            active_expert_count=1,
            route_candidate_count=2,
        )
    )
    brain.install_language_model(
        model,
        tokenizer,
        evaluation_report=evaluate_language_model(model, split.eval),
    )
    proposal = brain.propose_language_structure(
        routing_evidence={
            "surface": "marulho_routed_language_experts.v1",
            "total_columns": 2,
            "active_columns": 2,
            "candidate_rows_scored": 20,
            "runs_all_columns": False,
        },
        config=LanguageStructuralPlasticityConfig(route_saturation_threshold=0.5),
    )
    transaction = brain.apply_language_structure(
        proposal,
        eval_batches=split.eval,
        checkpoint_path=tmp_path / "brain-language-structure-baseline.pt",
        operator_approved=True,
        config=LanguageStructuralPlasticityConfig(max_eval_loss_delta=10.0),
    )
    status = brain.status()
    saved = brain.save(tmp_path / "brain-language-structure.pt")
    restored = MarulhoBrain.load(saved["path"])

    assert proposal["mutates_runtime_state"] is False
    assert transaction["surface"] == "marulho_brain_language_structural_transaction.v1"
    assert transaction["report"]["applied"] is True
    assert transaction["report"]["mutation"]["target_expert_count"] > 2
    assert transaction["report"]["checkpoint"]["checkpoint_restore_verified"] is True
    assert transaction["trace"]["event"] == "language_structure"
    assert status["language_model"]["structural_transaction_count"] == 1
    assert status["language_model"]["last_structural_transaction"]["surface"] == (
        "marulho_language_structural_plasticity_transaction.v1"
    )
    restored_status = restored.status()
    assert restored_status["language_model"]["structural_transaction_count"] == 1
    assert restored_status["language_model"]["last_structural_transaction"]["promotion_gate"][
        "checkpoint_backed"
    ] is True


def test_marulho_brain_language_column_split_transaction_is_checkpointed(
    tmp_path: Path,
) -> None:
    brain = MarulhoBrain.fresh(_tiny_config())
    tokenizer = ByteLevelLanguageTokenizer()
    split = build_language_model_splits(
        ["structural column split needs overload evidence and rollback. " * 6],
        tokenizer,
        sequence_length=10,
        eval_fraction=0.25,
    )
    model = MarulhoLanguageModel(
        LanguageModelConfig(
            vocab_size=tokenizer.vocab_size,
            embedding_dim=8,
            state_dim=12,
            expert_count=4,
            active_expert_count=1,
            route_candidate_count=2,
        )
    )
    brain.install_language_model(
        model,
        tokenizer,
        evaluation_report=evaluate_language_model(model, split.eval),
    )
    proposal = brain.propose_language_structure(
        routing_evidence={
            "surface": "marulho_routed_language_experts.v1",
            "total_columns": 4,
            "active_columns": 2,
            "split_candidate_expert_ids": [1],
            "expert_loads": [0.1, 0.95, 0.2, 0.3],
            "candidate_rows_scored": 40,
            "runs_all_columns": False,
        },
        config=LanguageStructuralPlasticityConfig(
            max_added_experts=2,
            max_split_experts=1,
            split_load_threshold=0.8,
        ),
        mutation_kind="column_split",
    )
    transaction = brain.apply_language_structure(
        proposal,
        eval_batches=split.eval,
        checkpoint_path=tmp_path / "brain-language-column-split-baseline.pt",
        operator_approved=True,
        config=LanguageStructuralPlasticityConfig(
            max_added_experts=2,
            max_split_experts=1,
            max_eval_loss_delta=100.0,
        ),
    )
    status = brain.status()
    saved = brain.save(tmp_path / "brain-language-column-split.pt")
    restored = MarulhoBrain.load(saved["path"])

    assert proposal["proposal"]["proposal_kind"] == "column_split"
    assert proposal["mutates_runtime_state"] is False
    assert transaction["surface"] == "marulho_brain_language_structural_transaction.v1"
    assert transaction["report"]["applied"] is True
    assert transaction["report"]["mutation"]["target_expert_count"] == 5
    assert transaction["report"]["mutation"]["parent_child_expert_pairs"] == [[1, 4]]
    assert transaction["report"]["promotion_gate"][
        "eligible_for_reviewed_column_split_promotion"
    ] is True
    assert transaction["report"]["checkpoint"]["checkpoint_restore_verified"] is True
    assert transaction["trace"]["event"] == "language_structure"
    assert status["language_model"]["last_structural_transaction"]["mutation"][
        "proposal_kind"
    ] == "column_split"
    restored_status = restored.status()
    assert restored_status["language_model"]["last_structural_transaction"]["mutation"][
        "target_expert_count"
    ] == 5


def test_marulho_brain_language_structural_prune_transaction_is_checkpointed(
    tmp_path: Path,
) -> None:
    brain = MarulhoBrain.fresh(_tiny_config())
    tokenizer = ByteLevelLanguageTokenizer()
    split = build_language_model_splits(
        ["structural expert prune needs checkpoint rollback evidence. " * 6],
        tokenizer,
        sequence_length=10,
        eval_fraction=0.25,
    )
    model = MarulhoLanguageModel(
        LanguageModelConfig(
            vocab_size=tokenizer.vocab_size,
            embedding_dim=8,
            state_dim=12,
            expert_count=3,
            active_expert_count=1,
            route_candidate_count=2,
        )
    )
    brain.install_language_model(
        model,
        tokenizer,
        evaluation_report=evaluate_language_model(model, split.eval),
    )
    proposal = brain.propose_language_structure(
        routing_evidence={
            "surface": "marulho_routed_language_experts.v1",
            "total_columns": 3,
            "active_columns": 1,
            "active_expert_ids": [0],
            "inactive_expert_ids": [2],
            "expert_utilities": [0.8, 0.2, 0.0],
            "candidate_rows_scored": 30,
            "runs_all_columns": False,
        },
        config=LanguageStructuralPlasticityConfig(
            min_expert_count=2,
            max_pruned_experts=1,
            prune_utility_threshold=0.05,
        ),
        mutation_kind="prune",
    )
    transaction = brain.apply_language_structure(
        proposal,
        eval_batches=split.eval,
        checkpoint_path=tmp_path / "brain-language-prune-baseline.pt",
        operator_approved=True,
        config=LanguageStructuralPlasticityConfig(
            min_expert_count=2,
            max_pruned_experts=1,
            max_eval_loss_delta=10.0,
        ),
    )
    status = brain.status()
    saved = brain.save(tmp_path / "brain-language-prune.pt")
    restored = MarulhoBrain.load(saved["path"])

    assert proposal["proposal"]["proposal_kind"] == "expert_prune"
    assert proposal["mutates_runtime_state"] is False
    assert transaction["surface"] == "marulho_brain_language_structural_transaction.v1"
    assert transaction["report"]["applied"] is True
    assert transaction["report"]["mutation"]["target_expert_count"] == 2
    assert transaction["report"]["mutation"]["pruned_expert_ids"] == [2]
    assert transaction["report"]["promotion_gate"]["eligible_for_reviewed_prune_promotion"] is True
    assert transaction["report"]["checkpoint"]["checkpoint_restore_verified"] is True
    assert transaction["trace"]["event"] == "language_structure"
    assert status["language_model"]["last_structural_transaction"]["mutation"][
        "proposal_kind"
    ] == "expert_prune"
    restored_status = restored.status()
    assert restored_status["language_model"]["last_structural_transaction"]["mutation"][
        "target_expert_count"
    ] == 2


def test_marulho_brain_language_retire_transaction_is_checkpointed(
    tmp_path: Path,
) -> None:
    brain = MarulhoBrain.fresh(_tiny_config())
    tokenizer = ByteLevelLanguageTokenizer()
    split = build_language_model_splits(
        ["structural expert retire needs terminal evidence and rollback. " * 6],
        tokenizer,
        sequence_length=10,
        eval_fraction=0.25,
    )
    model = MarulhoLanguageModel(
        LanguageModelConfig(
            vocab_size=tokenizer.vocab_size,
            embedding_dim=8,
            state_dim=12,
            expert_count=4,
            active_expert_count=1,
            route_candidate_count=2,
        )
    )
    brain.install_language_model(
        model,
        tokenizer,
        evaluation_report=evaluate_language_model(model, split.eval),
    )
    proposal = brain.propose_language_structure(
        routing_evidence={
            "surface": "marulho_routed_language_experts.v1",
            "total_columns": 4,
            "active_columns": 1,
            "active_expert_ids": [0],
            "retire_candidate_expert_ids": [3],
            "dead_spike_expert_ids": [3],
            "expert_utilities": [0.8, 0.4, 0.3, 0.0],
            "candidate_rows_scored": 40,
            "runs_all_columns": False,
        },
        config=LanguageStructuralPlasticityConfig(
            min_expert_count=2,
            max_retired_experts=1,
            prune_utility_threshold=0.05,
        ),
        mutation_kind="retire",
    )
    transaction = brain.apply_language_structure(
        proposal,
        eval_batches=split.eval,
        checkpoint_path=tmp_path / "brain-language-retire-baseline.pt",
        operator_approved=True,
        config=LanguageStructuralPlasticityConfig(
            min_expert_count=2,
            max_retired_experts=1,
            max_eval_loss_delta=100.0,
        ),
    )
    status = brain.status()
    saved = brain.save(tmp_path / "brain-language-retire.pt")
    restored = MarulhoBrain.load(saved["path"])

    assert proposal["proposal"]["proposal_kind"] == "expert_retire"
    assert proposal["mutates_runtime_state"] is False
    assert transaction["surface"] == "marulho_brain_language_structural_transaction.v1"
    assert transaction["report"]["applied"] is True
    assert transaction["report"]["mutation"]["target_expert_count"] == 3
    assert transaction["report"]["mutation"]["retired_expert_ids"] == [3]
    assert transaction["report"]["promotion_gate"][
        "eligible_for_reviewed_retire_promotion"
    ] is True
    assert transaction["report"]["checkpoint"]["checkpoint_restore_verified"] is True
    assert transaction["trace"]["event"] == "language_structure"
    assert status["language_model"]["last_structural_transaction"]["mutation"][
        "proposal_kind"
    ] == "expert_retire"
    restored_status = restored.status()
    assert restored_status["language_model"]["last_structural_transaction"]["mutation"][
        "retired_expert_ids"
    ] == [3]


def test_marulho_brain_language_structural_merge_transaction_is_checkpointed(
    tmp_path: Path,
) -> None:
    brain = MarulhoBrain.fresh(_tiny_config())
    tokenizer = ByteLevelLanguageTokenizer()
    split = build_language_model_splits(
        ["structural expert merge needs duplicate evidence and rollback. " * 6],
        tokenizer,
        sequence_length=10,
        eval_fraction=0.25,
    )
    model = MarulhoLanguageModel(
        LanguageModelConfig(
            vocab_size=tokenizer.vocab_size,
            embedding_dim=8,
            state_dim=12,
            expert_count=4,
            active_expert_count=1,
            route_candidate_count=2,
        )
    )
    brain.install_language_model(
        model,
        tokenizer,
        evaluation_report=evaluate_language_model(model, split.eval),
    )
    proposal = brain.propose_language_structure(
        routing_evidence={
            "surface": "marulho_routed_language_experts.v1",
            "total_columns": 4,
            "active_columns": 2,
            "duplicate_expert_pairs": [[1, 2]],
            "expert_pair_similarities": {"1,2": 0.99},
            "candidate_rows_scored": 40,
            "runs_all_columns": False,
        },
        config=LanguageStructuralPlasticityConfig(
            min_expert_count=2,
            max_merged_expert_pairs=1,
            merge_similarity_threshold=0.95,
        ),
        mutation_kind="merge",
    )
    transaction = brain.apply_language_structure(
        proposal,
        eval_batches=split.eval,
        checkpoint_path=tmp_path / "brain-language-merge-baseline.pt",
        operator_approved=True,
        config=LanguageStructuralPlasticityConfig(
            min_expert_count=2,
            max_merged_expert_pairs=1,
            max_eval_loss_delta=10.0,
        ),
    )
    status = brain.status()
    saved = brain.save(tmp_path / "brain-language-merge.pt")
    restored = MarulhoBrain.load(saved["path"])

    assert proposal["proposal"]["proposal_kind"] == "expert_merge"
    assert proposal["mutates_runtime_state"] is False
    assert transaction["surface"] == "marulho_brain_language_structural_transaction.v1"
    assert transaction["report"]["applied"] is True
    assert transaction["report"]["mutation"]["target_expert_count"] == 3
    assert transaction["report"]["mutation"]["merged_expert_groups"] == [[1, 2]]
    assert transaction["report"]["promotion_gate"]["eligible_for_reviewed_merge_promotion"] is True
    assert transaction["report"]["checkpoint"]["checkpoint_restore_verified"] is True
    assert transaction["trace"]["event"] == "language_structure"
    assert status["language_model"]["last_structural_transaction"]["mutation"][
        "proposal_kind"
    ] == "expert_merge"
    restored_status = restored.status()
    assert restored_status["language_model"]["last_structural_transaction"]["mutation"][
        "target_expert_count"
    ] == 3


def test_marulho_brain_language_structural_deep_sleep_transaction_is_checkpointed(
    tmp_path: Path,
) -> None:
    brain = MarulhoBrain.fresh(_tiny_config())
    tokenizer = ByteLevelLanguageTokenizer()
    split = build_language_model_splits(
        ["structural expert deep sleep needs stale evidence and rollback. " * 6],
        tokenizer,
        sequence_length=10,
        eval_fraction=0.25,
    )
    model = MarulhoLanguageModel(
        LanguageModelConfig(
            vocab_size=tokenizer.vocab_size,
            embedding_dim=8,
            state_dim=12,
            expert_count=4,
            active_expert_count=1,
            route_candidate_count=4,
        )
    )
    brain.install_language_model(
        model,
        tokenizer,
        evaluation_report=evaluate_language_model(model, split.eval),
    )
    proposal = brain.propose_language_structure(
        routing_evidence={
            "surface": "marulho_routed_language_experts.v1",
            "total_columns": 4,
            "active_columns": 1,
            "active_expert_ids": [0],
            "stale_expert_ids": [3],
            "low_activation_expert_ids": [3],
            "expert_utilities": [0.7, 0.4, 0.3, 0.0],
            "candidate_rows_scored": 40,
            "runs_all_columns": False,
        },
        config=LanguageStructuralPlasticityConfig(
            min_expert_count=2,
            max_deep_sleep_experts=1,
            deep_sleep_utility_threshold=0.10,
        ),
        mutation_kind="deep_sleep",
    )
    transaction = brain.apply_language_structure(
        proposal,
        eval_batches=split.eval,
        checkpoint_path=tmp_path / "brain-language-deep-sleep-baseline.pt",
        operator_approved=True,
        config=LanguageStructuralPlasticityConfig(
            min_expert_count=2,
            max_deep_sleep_experts=1,
            max_eval_loss_delta=10.0,
        ),
    )
    status = brain.status()
    saved = brain.save(tmp_path / "brain-language-deep-sleep.pt")
    restored = MarulhoBrain.load(saved["path"])

    assert proposal["proposal"]["proposal_kind"] == "expert_deep_sleep"
    assert proposal["mutates_runtime_state"] is False
    assert transaction["surface"] == "marulho_brain_language_structural_transaction.v1"
    assert transaction["report"]["applied"] is True
    assert transaction["report"]["mutation"]["target_expert_count"] == 4
    assert transaction["report"]["mutation"]["deep_sleep_expert_ids"] == [3]
    assert transaction["report"]["mutation"]["sleeping_expert_ids_after"] == [3]
    assert transaction["report"]["mutation"]["awake_expert_count_after"] == 3
    assert transaction["report"]["promotion_gate"][
        "eligible_for_reviewed_deep_sleep_promotion"
    ] is True
    assert transaction["report"]["checkpoint"]["checkpoint_restore_verified"] is True
    assert transaction["trace"]["event"] == "language_structure"
    assert status["language_model"]["last_structural_transaction"]["mutation"][
        "proposal_kind"
    ] == "expert_deep_sleep"
    restored_status = restored.status()
    assert restored_status["language_model"]["last_structural_transaction"]["mutation"][
        "sleeping_expert_ids_after"
    ] == [3]


def test_marulho_brain_language_route_bank_transaction_is_checkpointed(
    tmp_path: Path,
) -> None:
    brain = MarulhoBrain.fresh(_tiny_config())
    tokenizer = ByteLevelLanguageTokenizer()
    split = build_language_model_splits(
        ["structural route bank expansion widens bounded candidate scoring. " * 6],
        tokenizer,
        sequence_length=10,
        eval_fraction=0.25,
    )
    model = MarulhoLanguageModel(
        LanguageModelConfig(
            vocab_size=tokenizer.vocab_size,
            embedding_dim=8,
            state_dim=12,
            expert_count=5,
            active_expert_count=1,
            route_candidate_count=2,
        )
    )
    brain.install_language_model(
        model,
        tokenizer,
        evaluation_report=evaluate_language_model(model, split.eval),
    )
    proposal = brain.propose_language_structure(
        routing_evidence={
            "surface": "marulho_routed_language_experts.v1",
            "total_columns": 5,
            "active_columns": 2,
            "route_candidate_count": 2,
            "output_candidate_count": 1,
            "candidate_rows_scored": 40,
            "runs_all_columns": False,
            "route_bank_pressure": True,
        },
        config=LanguageStructuralPlasticityConfig(
            route_saturation_threshold=0.5,
            max_route_candidate_growth=2,
        ),
        mutation_kind="route_bank_expansion",
    )
    transaction = brain.apply_language_structure(
        proposal,
        eval_batches=split.eval,
        checkpoint_path=tmp_path / "brain-language-route-bank-baseline.pt",
        operator_approved=True,
        config=LanguageStructuralPlasticityConfig(
            max_route_candidate_growth=2,
            max_eval_loss_delta=100.0,
        ),
    )
    status = brain.status()
    saved = brain.save(tmp_path / "brain-language-route-bank.pt")
    restored = MarulhoBrain.load(saved["path"])
    last_transaction = status["language_model"]["last_structural_transaction"]

    assert proposal["proposal"]["proposal_kind"] == "route_bank_expansion"
    assert proposal["mutates_runtime_state"] is False
    assert transaction["surface"] == "marulho_brain_language_structural_transaction.v1"
    assert transaction["report"]["applied"] is True
    assert transaction["report"]["mutation"]["target_expert_count"] == 5
    assert transaction["report"]["mutation"]["source_route_candidate_count"] == 2
    assert transaction["report"]["mutation"]["target_route_candidate_count"] == 4
    assert transaction["report"]["promotion_gate"][
        "eligible_for_reviewed_route_bank_expansion_promotion"
    ] is True
    assert transaction["report"]["checkpoint"]["checkpoint_restore_verified"] is True
    assert transaction["trace"]["event"] == "language_structure"
    assert last_transaction["mutation"]["proposal_kind"] == "route_bank_expansion"
    restored_status = restored.status()
    assert restored_status["language_model"]["last_structural_transaction"]["mutation"][
        "target_route_candidate_count"
    ] == 4


def test_marulho_brain_language_synapse_bundle_transaction_is_checkpointed(
    tmp_path: Path,
) -> None:
    brain = MarulhoBrain.fresh(_tiny_config())
    tokenizer = ByteLevelLanguageTokenizer()
    split = build_language_model_splits(
        ["structural synapse bundle growth widens expert hidden capacity. " * 6],
        tokenizer,
        sequence_length=10,
        eval_fraction=0.25,
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
    proposal = brain.propose_language_structure(
        routing_evidence={
            "surface": "marulho_routed_language_experts.v1",
            "total_columns": 3,
            "active_columns": 1,
            "synapse_bundle_pressure": True,
            "high_surprise_expert_ids": [1],
            "candidate_rows_scored": 30,
            "runs_all_columns": False,
        },
        config=LanguageStructuralPlasticityConfig(
            max_synapse_bundle_hidden_growth=4,
        ),
        mutation_kind="synapse_bundle",
    )
    transaction = brain.apply_language_structure(
        proposal,
        eval_batches=split.eval,
        checkpoint_path=tmp_path / "brain-language-synapse-bundle-baseline.pt",
        operator_approved=True,
        config=LanguageStructuralPlasticityConfig(
            max_synapse_bundle_hidden_growth=4,
            max_eval_loss_delta=100.0,
        ),
    )
    status = brain.status()
    last_transaction = status["language_model"]["last_structural_transaction"]

    assert proposal["proposal"]["proposal_kind"] == "synapse_bundle_growth"
    assert proposal["mutates_runtime_state"] is False
    assert transaction["surface"] == "marulho_brain_language_structural_transaction.v1"
    assert transaction["report"]["applied"] is True
    assert transaction["report"]["mutation"]["target_expert_count"] == 3
    assert transaction["report"]["mutation"]["source_expert_hidden_dim"] == 16
    assert transaction["report"]["mutation"]["target_expert_hidden_dim"] == 20
    assert transaction["report"]["mutation"]["synapse_bundle_hidden_growth"] == 4
    assert transaction["report"]["promotion_gate"][
        "eligible_for_reviewed_synapse_bundle_promotion"
    ] is True
    assert transaction["report"]["checkpoint"]["checkpoint_restore_verified"] is True
    assert transaction["trace"]["event"] == "language_structure"
    assert last_transaction["mutation"]["proposal_kind"] == "synapse_bundle_growth"


def test_marulho_brain_language_memory_slot_transaction_is_checkpointed(
    tmp_path: Path,
) -> None:
    brain = MarulhoBrain.fresh(_tiny_config())
    tokenizer = ByteLevelLanguageTokenizer()
    split = build_language_model_splits(
        ["structural memory slot expansion keeps bounded retrieval evidence. " * 6],
        tokenizer,
        sequence_length=10,
        eval_fraction=0.25,
    )
    model = MarulhoLanguageModel(
        LanguageModelConfig(
            vocab_size=tokenizer.vocab_size,
            embedding_dim=8,
            state_dim=12,
            expert_count=3,
            active_expert_count=1,
            route_candidate_count=2,
            memory_slot_count=0,
            memory_slot_candidate_count=0,
            active_memory_slot_count=1,
        )
    )
    brain.install_language_model(
        model,
        tokenizer,
        evaluation_report=evaluate_language_model(model, split.eval),
    )
    proposal = brain.propose_language_structure(
        routing_evidence={
            "surface": "marulho_language_memory_slots.v1",
            "memory_slot_pressure": True,
            "novel_concept_cluster": True,
            "candidate_rows_scored": 30,
            "runs_all_columns": False,
        },
        config=LanguageStructuralPlasticityConfig(
            max_memory_slot_growth=4,
            max_memory_slot_candidate_count=2,
        ),
        mutation_kind="memory_slot",
    )
    transaction = brain.apply_language_structure(
        proposal,
        eval_batches=split.eval,
        checkpoint_path=tmp_path / "brain-language-memory-slot-baseline.pt",
        operator_approved=True,
        config=LanguageStructuralPlasticityConfig(
            max_memory_slot_growth=4,
            max_memory_slot_candidate_count=2,
            max_eval_loss_delta=100.0,
        ),
    )
    status = brain.status()
    last_transaction = status["language_model"]["last_structural_transaction"]

    assert proposal["proposal"]["proposal_kind"] == "memory_slot_expansion"
    assert proposal["proposal"]["target_memory_slot_count"] == 4
    assert proposal["proposal"]["target_memory_slot_candidate_count"] == 2
    assert proposal["proposal"]["avoids_all_slot_scan"] is True
    assert proposal["mutates_runtime_state"] is False
    assert transaction["surface"] == "marulho_brain_language_structural_transaction.v1"
    assert transaction["report"]["applied"] is True
    assert transaction["report"]["mutation"]["source_memory_slot_count"] == 0
    assert transaction["report"]["mutation"]["target_memory_slot_count"] == 4
    assert transaction["report"]["mutation"]["target_memory_slot_candidate_count"] == 2
    assert transaction["report"]["mutation"]["memory_slot_count_delta"] == 4
    assert transaction["report"]["promotion_gate"][
        "eligible_for_reviewed_memory_slot_expansion_promotion"
    ] is True
    assert transaction["report"]["checkpoint"]["checkpoint_restore_verified"] is True
    assert transaction["trace"]["event"] == "language_structure"
    assert last_transaction["mutation"]["proposal_kind"] == "memory_slot_expansion"


def test_marulho_brain_language_checkpoint_evolution_keeps_parent_installed(tmp_path: Path) -> None:
    brain = MarulhoBrain.fresh(_tiny_config())
    tokenizer = ByteLevelLanguageTokenizer()
    parent_split = build_language_model_splits(
        ["parent brain checkpoint lineage stays installed. " * 6],
        tokenizer,
        sequence_length=10,
        eval_fraction=0.25,
    )
    child_split = build_language_model_splits(
        ["child fork trains separately before promotion review. " * 6],
        tokenizer,
        sequence_length=10,
        eval_fraction=0.25,
    )
    model = MarulhoLanguageModel(
        LanguageModelConfig(
            vocab_size=tokenizer.vocab_size,
            embedding_dim=8,
            state_dim=12,
        )
    )
    brain.install_language_model(
        model,
        tokenizer,
        evaluation_report=evaluate_language_model(model, parent_split.eval),
    )

    evolution = brain.evolve_language_checkpoint(
        eval_batches=parent_split.eval,
        child_train_batches=child_split.train[:2],
        child_new_eval_batches=child_split.train[:2],
        replay_batches=parent_split.train[:1],
        checkpoint_dir=tmp_path / "language-evolution",
        config=LanguageCheckpointEvolutionConfig(
            max_child_loss_delta=100.0,
            max_old_domain_forgetting=100.0,
            require_child_learning=False,
            allow_structural_growth=False,
        ),
        learning_config=LanguageContinualLearningConfig(
            learning_rate=2e-2,
            max_steps=2,
            forgetting_tolerance=100.0,
            replay_retention_tolerance=100.0,
            rollback_on_forgetting=False,
        ),
    )
    status = brain.status()
    saved = brain.save(tmp_path / "brain-language-evolution.pt")
    restored = MarulhoBrain.load(saved["path"])
    restored_status = restored.status()

    assert evolution["surface"] == "marulho_brain_language_checkpoint_evolution.v1"
    assert evolution["report"]["mutates_parent_runtime"] is False
    assert evolution["report"]["promotion_gate"]["parent_runtime_unchanged"] is True
    assert evolution["trace"]["event"] == "language_checkpoint_evolution"
    assert status["active_language_path"] == "marulho_lm_head"
    assert status["language_model"]["checkpoint_evolution_count"] == 1
    assert status["language_model"]["last_checkpoint_evolution"]["surface"] == (
        "marulho_language_checkpoint_evolution.v1"
    )
    assert restored_status["language_model"]["checkpoint_evolution_count"] == 1
    assert restored_status["language_model"]["last_checkpoint_evolution"]["promotion_gate"][
        "rollback_to_parent_verified"
    ] is True


def test_marulho_brain_replay_and_growth_reports_are_local() -> None:
    brain = _trained_brain()

    replay = brain.replay(window="micro", cycles=1)
    growth = brain.grow_prune(budget="small")

    assert replay["surface"] == "marulho_brain_replay.v1"
    assert replay["replay_updates"] >= 0
    assert growth["surface"] == "marulho_brain_growth_prune.v1"
    assert growth["growth_events"] >= 0
    assert growth["prune_events"] == 0
    assert growth["trace"]["retired_brain_surfaces"]["thought_loop_used"] is False


def test_marulho_brain_start_stop_are_brain_owned() -> None:
    brain = MarulhoBrain.fresh(_tiny_config())
    feed = brain.feed("loop owned by marulho brain.", source="loop-test")
    assert feed["accepted_tokens"] > 0

    start = brain.start(tick_tokens=4, interval_seconds=0.01, source="loop-test")
    assert start["surface"] == "marulho_brain_loop_start.v1"
    assert start["started"] is True
    assert start["loop"]["owner"] == "MarulhoBrain"
    assert start["loop"]["legacy_terminus_runtime_control"] is False

    deadline = time.time() + 1.0
    while time.time() < deadline and brain.status()["loop"]["tick_count"] == 0:
        time.sleep(0.01)

    stop = brain.stop(timeout_seconds=1.0)
    assert stop["surface"] == "marulho_brain_loop_stop.v1"
    assert stop["stopped"] is True
    assert stop["loop"]["running"] is False
    assert stop["loop"]["legacy_terminus_runtime_control"] is False
    assert brain.status()["loop"]["owner"] == "MarulhoBrain"


def test_brain_service_contract(tmp_path: Path) -> None:
    brain = MarulhoBrain.fresh(_tiny_config())
    checkpoint_path = tmp_path / "service-brain.pt"
    reports_dir = tmp_path / "reports" / "language_benchmark_suite"
    reports_dir.mkdir(parents=True)
    (reports_dir / "language-suite.json").write_text(
        json.dumps(
            {
                "artifact_kind": "marulho_language_runtime_benchmark_suite",
                "surface": "marulho_language_runtime_benchmark_suite.v1",
                "external_llm_used": False,
                "promotion_gate": {
                    "status": "blocked_missing_required_evidence",
                    "promotes_runtime_claim": False,
                    "missing_required_category_names": ["grounding_support"],
                },
            }
        ),
        encoding="utf-8",
    )
    save_trainer_checkpoint(
        checkpoint_path,
        brain.trainer,
        metadata={"brain_state": brain.export_state()},
    )
    app = create_app(
        checkpoint_path=checkpoint_path,
        trace_dir=tmp_path / "traces",
        env_root=tmp_path,
    )
    assert app.state.marulho_manager.__class__.__name__ == "MarulhoBrainServiceManager"
    assert not hasattr(app.state.marulho_manager, "_status_read_model")
    assert not hasattr(app.state.marulho_manager, "_runtime_control")

    with TestClient(app) as client:
        feed = client.post(
            "/brain/feed",
            json={"text": "service brain learns local output.", "source": "test"},
        )
        assert feed.status_code == 200
        accepted = feed.json()["accepted_tokens"]

        tick = client.post("/brain/tick", json={"tokens": accepted, "source": "test"})
        assert tick.status_code == 200
        assert tick.json()["trained_tokens"] == accepted

        generate = client.post("/brain/generate", json={"prompt": "ser", "max_tokens": 12})
        assert generate.status_code == 200
        assert generate.json()["external_llm_used"] is False
        assert generate.json()["emitted_tokens"] > 0

        status = client.get("/brain/status")
        assert status.status_code == 200
        assert status.json()["surface"] == "marulho_brain_runtime.v1"
        assert status.json()["readout"]["observed_transition_count"] > 0

        traces = client.get("/brain/traces?limit=4")
        assert traces.status_code == 200
        assert traces.json()["surface"] == "marulho_brain_trace_history.v1"

        before_reports_status = client.get("/brain/status").json()["token_count"]
        reports = client.get("/brain/evidence/reports?limit=4")
        after_reports_status = client.get("/brain/status").json()["token_count"]
        assert reports.status_code == 200
        assert reports.json()["surface"] == "marulho_evidence_report_inventory.v1"
        assert reports.json()["reports_not_run_by_service"] is True
        assert reports.json()["mutates_runtime_state"] is False
        assert reports.json()["reports"][0]["artifact_kind"] == (
            "marulho_language_runtime_benchmark_suite"
        )
        assert reports.json()["reports"][0]["promotion_status"] == (
            "blocked_missing_required_evidence"
        )
        assert before_reports_status == after_reports_status

        checkpoints = client.get("/brain/checkpoints")
        assert checkpoints.status_code == 200
        assert any(item["path"].endswith("service-brain.pt") for item in checkpoints.json()["checkpoints"])

        start = client.post(
            "/brain/start",
            json={"tick_tokens": 4, "interval_seconds": 0.01, "source": "test"},
        )
        assert start.status_code == 200
        assert start.json()["surface"] == "marulho_brain_loop_start.v1"
        assert "control" not in start.json()
        assert start.json()["loop"]["owner"] == "MarulhoBrain"

        stop = client.post("/brain/stop", json={"timeout_seconds": 1.0})
        assert stop.status_code == 200
        assert stop.json()["surface"] == "marulho_brain_loop_stop.v1"
        assert "control" not in stop.json()
        assert stop.json()["loop"]["legacy_terminus_runtime_control"] is False

        saved_path = tmp_path / "service-brain-saved.pt"
        saved = client.post("/brain/checkpoint/save", json={"path": str(saved_path)})
        assert saved.status_code == 200
        assert saved.json()["surface"] == "marulho_brain_checkpoint_save.v1"
        assert Path(saved.json()["path"]).is_file()

        restored = client.post("/brain/checkpoint/restore", json={"path": str(saved_path)})
        assert restored.status_code == 200
        assert restored.json()["restore"]["surface"] == "marulho_brain_checkpoint_restore.v1"
        assert restored.json()["brain"]["surface"] == "marulho_brain_runtime.v1"

        openapi_paths = set(client.get("/openapi.json").json()["paths"])
        assert openapi_paths == {
            "/health",
            "/",
            "/brain/status",
            "/brain/checkpoints",
            "/brain/traces",
            "/brain/evidence/reports",
            "/brain/feed",
            "/brain/tick",
            "/brain/generate",
            "/brain/replay",
            "/brain/grow-prune",
            "/brain/checkpoint/save",
            "/brain/checkpoint/restore",
            "/brain/start",
            "/brain/stop",
            "/brain/stream/status",
        }
        for legacy_path in (
            "/status",
            "/feed",
            "/query",
            "/respond",
            "/checkpoint/save",
            "/terminus",
            "/terminus/snn-language-readiness",
            "/stream/status",
        ):
            assert client.get(legacy_path).status_code == 404


def test_deleted_legacy_service_surfaces_are_not_importable() -> None:
    for module_name in (
        "marulho.service.manager",
        "marulho.service.brain_runtime",
        "marulho.service.runtime_control",
        "marulho.service.runtime_facade",
        "marulho.service.status_read_model",
    ):
        assert importlib.util.find_spec(module_name) is None
