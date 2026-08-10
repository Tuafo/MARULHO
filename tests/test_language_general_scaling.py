from __future__ import annotations

import pytest

from marulho.evaluation.language_general_scaling import (
    ADVANCE_DECISION,
    INVALID_DECISION,
    STOP_DECISION,
    GeneralScalingConfig,
    V31_STAGE,
    V32_STAGE,
    V34_STAGE,
    V35_STAGE,
    V35R_STAGE,
    _schedule_uniqueness,
    _source_coverage_audit,
    _split_coverage_audit,
    _validate_locked_training_manifest,
    _validate_baseline,
    build_model,
    scaling_decision,
)


def _row(loss: float, *, gradients: bool = True) -> dict:
    return {
        "all_parameters_received_final_gradient": gradients,
        "heldout": {"heldout_loss": loss},
    }


def test_v31_keeps_the_selected_v30_model_shape() -> None:
    model = build_model(vocab_size=8192, config=GeneralScalingConfig())
    assert sum(parameter.numel() for parameter in model.parameters()) == 20_976_128
    assert model.config.transformer_context_length == 72
    assert model.config.active_language_path == "marulho_transformer_v31_general72"


def test_v32_changes_stage_identity_without_changing_parameter_shape() -> None:
    model = build_model(
        vocab_size=8192,
        config=GeneralScalingConfig(),
        stage=V32_STAGE,
    )
    assert sum(parameter.numel() for parameter in model.parameters()) == 20_976_128
    assert model.config.active_language_path == "marulho_transformer_v32_general72"


def test_v34_is_a_preregistered_100m_capacity_jump_not_an_initial_state_match() -> None:
    config = GeneralScalingConfig(
        width=V34_STAGE.model_width,
        layers=V34_STAGE.model_layers,
        heads=V34_STAGE.model_heads,
    )
    model = build_model(vocab_size=8192, config=config, stage=V34_STAGE)
    assert sum(parameter.numel() for parameter in model.parameters()) == 100_679_424
    assert model.config.transformer_context_length == 72
    assert model.config.active_language_path == "marulho_transformer_v34_general72_100m"
    assert V34_STAGE.require_initial_state_match is False
    assert V34_STAGE.minimum_loss_gain == 0.20


def test_v35_continues_the_100m_checkpoint_on_new_tokens() -> None:
    config = GeneralScalingConfig(
        width=V35_STAGE.model_width,
        layers=V35_STAGE.model_layers,
        heads=V35_STAGE.model_heads,
    )
    model = build_model(vocab_size=8192, config=config, stage=V35_STAGE)
    assert sum(parameter.numel() for parameter in model.parameters()) == 100_679_424
    assert V35_STAGE.initialization_mode == "baseline_checkpoint"
    assert V35_STAGE.parent_processed_tokens == 67_110_912
    assert V35_STAGE.token_budget == 134_219_520
    assert V35_STAGE.parent_processed_tokens + V35_STAGE.token_budget == 201_330_432
    assert V35_STAGE.learning_rate == 3.0e-4
    assert V35_STAGE.minimum_loss_gain == 0.15
    assert set(dict(V35_STAGE.required_training_sources)) == {
        "fineweb-edu-train-75k-shard0-20260710.txt",
        "cosmopedia-v2-train-150k-shard1-20260710.txt",
        "cosmopedia-v2-train-75k-shard3-20260710.txt",
    }
    assert all(
        len(sha256) == 64
        for sha256 in dict(V35_STAGE.required_training_sources).values()
    )


def test_v35_baseline_contract_accepts_only_the_v34_survivor() -> None:
    report = {
        "artifact_kind": "marulho_general_scaling",
        "decision": "save_v34_capacity_scaling_100m_for_unseen_generation",
        "candidate": {"heldout": {"heldout_loss": 3.39018}},
        "schedule": {"processed_tokens": 67_110_912},
    }
    assert _validate_baseline(report, stage=V35_STAGE) == (3.39018, None)


def test_v35_baseline_contract_rejects_the_wrong_parent_token_count() -> None:
    report = {
        "artifact_kind": "marulho_general_scaling",
        "decision": "save_v34_capacity_scaling_100m_for_unseen_generation",
        "candidate": {"heldout": {"heldout_loss": 3.39018}},
        "schedule": {"processed_tokens": 67_110_911},
    }
    with pytest.raises(ValueError, match="baseline report token count is invalid"):
        _validate_baseline(report, stage=V35_STAGE)


def test_v35r_repairs_only_the_two_batch_manifest_mismatch() -> None:
    config = GeneralScalingConfig(
        token_budget=V35R_STAGE.token_budget,
        sample_bytes_per_train_source=V35R_STAGE.train_sample_bytes,
        minimum_loss_gain=V35R_STAGE.minimum_loss_gain,
        width=V35R_STAGE.model_width,
        layers=V35R_STAGE.model_layers,
        heads=V35R_STAGE.model_heads,
        learning_rate=V35R_STAGE.learning_rate,
    )
    model = build_model(vocab_size=8192, config=config, stage=V35R_STAGE)
    assert sum(parameter.numel() for parameter in model.parameters()) == 100_679_424
    assert V35R_STAGE.required_prepared_general_batch_counts == (
        19_419,
        19_419,
        19_419,
    )
    assert V35R_STAGE.token_budget // (72 * 32) == 58_257
    assert V35R_STAGE.token_budget == 134_224_128
    assert V35R_STAGE.parent_processed_tokens + V35R_STAGE.token_budget == 201_335_040
    assert V35R_STAGE.minimum_loss_gain == V35_STAGE.minimum_loss_gain
    assert V35R_STAGE.learning_rate == V35_STAGE.learning_rate
    assert V35R_STAGE.required_training_sources == V35_STAGE.required_training_sources
    assert V35R_STAGE.lock_training_manifest is True
    _validate_locked_training_manifest(config, stage=V35R_STAGE)


def test_v35r_rejects_a_post_hoc_manifest_override() -> None:
    config = GeneralScalingConfig(
        token_budget=V35R_STAGE.token_budget - 2_304,
        sample_bytes_per_train_source=V35R_STAGE.train_sample_bytes,
        minimum_loss_gain=V35R_STAGE.minimum_loss_gain,
        width=V35R_STAGE.model_width,
        layers=V35R_STAGE.model_layers,
        heads=V35R_STAGE.model_heads,
        learning_rate=V35R_STAGE.learning_rate,
    )
    with pytest.raises(ValueError, match="token budget"):
        _validate_locked_training_manifest(config, stage=V35R_STAGE)


def test_v31_decision_requires_loss_unique_data_gradients_and_fidelity() -> None:
    config = GeneralScalingConfig()
    winning = _row(3.84)
    assert scaling_decision(
        winning,
        baseline_loss=4.00,
        config=config,
        unique_schedule_passed=True,
        checkpoint_fidelity_passed=True,
    ) == ADVANCE_DECISION
    assert scaling_decision(
        _row(3.86),
        baseline_loss=4.00,
        config=config,
        unique_schedule_passed=True,
        checkpoint_fidelity_passed=True,
    ) == STOP_DECISION
    assert scaling_decision(
        _row(3.84, gradients=False),
        baseline_loss=4.00,
        config=config,
        unique_schedule_passed=True,
        checkpoint_fidelity_passed=True,
    ) == INVALID_DECISION
    assert scaling_decision(
        winning,
        baseline_loss=4.00,
        config=config,
        unique_schedule_passed=False,
        checkpoint_fidelity_passed=True,
    ) == INVALID_DECISION
    assert scaling_decision(
        winning,
        baseline_loss=4.00,
        config=config,
        unique_schedule_passed=True,
        checkpoint_fidelity_passed=False,
    ) == INVALID_DECISION


def test_v31_schedule_audit_rejects_repeated_source_batches() -> None:
    unique = _schedule_uniqueness(
        (("general_0", 2), ("general_1", 1), ("general_0", 0))
    )
    repeated = _schedule_uniqueness(
        (("general_0", 2), ("general_1", 1), ("general_0", 2))
    )
    assert unique["every_scheduled_source_index_unique"] is True
    assert repeated["every_scheduled_source_index_unique"] is False


def test_v31_source_coverage_requires_budget_and_full_file_span() -> None:
    good = {
        "path": "train.txt",
        "source_size_bytes": 1_000,
        "selected_size_bytes": 505,
        "ranges": [{"start": 0, "end": 250}, {"start": 750, "end": 1_000}],
    }
    narrow = {
        **good,
        "ranges": [{"start": 0, "end": 250}, {"start": 250, "end": 505}],
    }
    assert _source_coverage_audit(
        [good],
        requested_bytes_per_source=500,
        requested_range_count=2,
    )["all_sources_stratified_and_budget_filled"] is True
    assert _source_coverage_audit(
        [narrow],
        requested_bytes_per_source=500,
        requested_range_count=2,
    )["all_sources_stratified_and_budget_filled"] is False
    full = {
        "path": "full.txt",
        "source_size_bytes": 1_000,
        "selected_size_bytes": 1_000,
        "ranges": [{"start": 0, "end": 1_000}],
    }
    assert _source_coverage_audit(
        [full],
        requested_bytes_per_source=2_000,
        requested_range_count=16,
    )["all_sources_stratified_and_budget_filled"] is True


def test_v31_split_coverage_requires_stratified_full_span_windows() -> None:
    report = {
        "window_selection": "stratified",
        "train_window_selection": {
            "source_window_count": 20,
            "selected_window_count": 8,
            "spans_full_source_window": True,
        },
    }
    assert _split_coverage_audit(
        [report],
        prepared_batch_counts=[2],
        batch_size=4,
    )["all_prepared_windows_stratified_across_sources"] is True
    report["train_window_selection"]["spans_full_source_window"] = False
    assert _split_coverage_audit(
        [report],
        prepared_batch_counts=[2],
        batch_size=4,
    )["all_prepared_windows_stratified_across_sources"] is False


def test_v32_baseline_contract_reads_v31_scaling_evidence() -> None:
    report = {
        "artifact_kind": "marulho_general_scaling",
        "decision": "save_v31_general_scaling_67m_for_unseen_generation",
        "candidate": {"heldout": {"heldout_loss": 3.6291}},
        "initial_state": {"sha256": "initial"},
    }
    assert _validate_baseline(report, stage=V32_STAGE) == (3.6291, "initial")
    report["decision"] = "wrong"
    try:
        _validate_baseline(report, stage=V32_STAGE)
    except ValueError as exc:
        assert "baseline decision" in str(exc)
    else:
        raise AssertionError("invalid V32 baseline decision was accepted")


def test_v31_baseline_contract_still_reads_v30_evidence() -> None:
    report = {
        "artifact_kind": "marulho_general_context_falsification",
        "decision": "save_v30_general_context_candidate_for_unseen_generation",
        "selection": {"selected_arm": "general72"},
        "arms": {
            "general72": {
                "common_context_heldout": {"heldout_loss": 4.0093}
            }
        },
        "matched_truth": {
            "initial_state_hashes": {"general72": "initial"}
        },
    }
    assert _validate_baseline(report, stage=V31_STAGE) == (4.0093, "initial")


def test_v32_decision_uses_its_own_durable_gain_and_labels() -> None:
    config = GeneralScalingConfig(minimum_loss_gain=0.20)
    assert scaling_decision(
        _row(3.40),
        baseline_loss=3.6291,
        config=config,
        unique_schedule_passed=True,
        checkpoint_fidelity_passed=True,
        stage=V32_STAGE,
    ) == V32_STAGE.advance_decision
    assert scaling_decision(
        _row(3.45),
        baseline_loss=3.6291,
        config=config,
        unique_schedule_passed=True,
        checkpoint_fidelity_passed=True,
        stage=V32_STAGE,
    ) == V32_STAGE.stop_decision
