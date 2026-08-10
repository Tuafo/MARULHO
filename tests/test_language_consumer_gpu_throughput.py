from __future__ import annotations

from marulho.evaluation.language_consumer_gpu_throughput import (
    ADVANCE_DECISION,
    ARMS,
    BASELINE_ARM,
    ConsumerGpuThroughputConfig,
    INVALID_DECISION,
    RETAIN_DECISION,
    per_head_optimizer_decision,
    select_throughput_arm,
)


def _row(loss: float, tokens_per_second: float, *, gradients: bool = True):
    return {
        "heldout": {"heldout_loss": loss},
        "training": {"tokens_per_second": tokens_per_second},
        "all_parameters_received_final_gradient": gradients,
    }


def _arms():
    return {arm.name: _row(3.10, 10_000.0) for arm in ARMS}


def test_large_batch_selection_requires_quality_and_speed_together() -> None:
    arms = _arms()
    arms["batch256_whole_qkv_lr8p5e4"] = _row(3.109, 19_000.0)
    arms["batch256_per_head_lr8p5e4"] = _row(3.105, 20_000.0)
    selected, decision = select_throughput_arm(
        arms, config=ConsumerGpuThroughputConfig()
    )
    assert selected == "batch256_per_head_lr8p5e4"
    assert decision == ADVANCE_DECISION


def test_large_batch_selection_rejects_fast_quality_regression() -> None:
    arms = _arms()
    for name in arms:
        if name != BASELINE_ARM:
            arms[name] = _row(3.12, 30_000.0)
    selected, decision = select_throughput_arm(
        arms, config=ConsumerGpuThroughputConfig()
    )
    assert selected is None
    assert decision == RETAIN_DECISION


def test_large_batch_selection_invalidates_incomplete_gradients() -> None:
    arms = _arms()
    arms["batch256_whole_qkv_lr8p5e4"] = _row(
        3.10, 20_000.0, gradients=False
    )
    selected, decision = select_throughput_arm(
        arms, config=ConsumerGpuThroughputConfig()
    )
    assert selected is None
    assert decision == INVALID_DECISION


def test_per_head_decision_is_separate_from_large_batch_gate() -> None:
    arms = _arms()
    arms["batch32_per_head_lr3e4"] = _row(3.104, 10_400.0)
    assert (
        per_head_optimizer_decision(arms, config=ConsumerGpuThroughputConfig())
        == "adopt_per_head_qkv_muon"
    )
