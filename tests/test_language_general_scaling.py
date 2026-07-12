from __future__ import annotations

from marulho.evaluation.language_general_scaling import (
    ADVANCE_DECISION,
    INVALID_DECISION,
    STOP_DECISION,
    GeneralScalingConfig,
    _schedule_uniqueness,
    _source_coverage_audit,
    _split_coverage_audit,
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
