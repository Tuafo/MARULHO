from marulho.training.cognitive_boundary_controller import (
    CognitiveBoundaryController,
)


def _plan(
    controller: CognitiveBoundaryController,
    *,
    start_token: int,
    token_count: int = 8,
    telemetry_interval: int = 64,
    slow_memory_archive_interval: int = 256,
    hnsw_buffer_pending: bool = False,
):
    return controller.plan(
        start_token=start_token,
        token_count=token_count,
        telemetry_interval=telemetry_interval,
        slow_memory_archive_interval=slow_memory_archive_interval,
        drift_floor_window_tokens=10_000,
        hnsw_flush_interval=16,
        hnsw_buffer_pending=hnsw_buffer_pending,
        deep_sleep_interval_tokens=10_000,
        last_deep_sleep_token=0,
        pending_emergency_deep_sleep=False,
        emergency_deep_sleep_cooldown_tokens=1_000,
        micro_sleep_interval_tokens=10_000,
        last_micro_sleep_token=0,
    )


def test_observation_boundaries_do_not_interrupt_device_execution() -> None:
    controller = CognitiveBoundaryController()

    plan = _plan(
        controller,
        start_token=49,
        telemetry_interval=8,
    )

    assert plan.device_continuous is True
    assert plan.fallback_reason is None
    assert plan.drift_refresh_due is True
    assert plan.telemetry_observation_due is True
    report = controller.report()
    assert report["device_continuous_count"] == 1
    assert report["exploration_execution_gate"] is False
    assert report["telemetry_execution_gate"] is False
    assert report["drift_refresh_execution_gate"] is False


def test_drift_floor_window_closes_after_device_execution() -> None:
    controller = CognitiveBoundaryController()

    plan = controller.plan(
        start_token=56,
        token_count=8,
        telemetry_interval=64,
        slow_memory_archive_interval=256,
        drift_floor_window_tokens=64,
        hnsw_flush_interval=16,
        hnsw_buffer_pending=False,
        deep_sleep_interval_tokens=10_000,
        last_deep_sleep_token=0,
        pending_emergency_deep_sleep=False,
        emergency_deep_sleep_cooldown_tokens=1_000,
        micro_sleep_interval_tokens=10_000,
        last_micro_sleep_token=0,
    )

    assert plan.device_continuous is True
    assert plan.drift_floor_close_due is True


def test_slow_memory_boundary_remains_a_real_fallback() -> None:
    controller = CognitiveBoundaryController()

    plan = _plan(
        controller,
        start_token=248,
        slow_memory_archive_interval=256,
    )

    assert plan.device_continuous is False
    assert plan.fallback_reason == "slow_memory_boundary"
    assert controller.report()["fallback_count"] == 1


def test_pending_routing_index_flush_remains_a_real_fallback() -> None:
    controller = CognitiveBoundaryController()

    plan = _plan(
        controller,
        start_token=16,
        hnsw_buffer_pending=True,
    )

    assert plan.fallback_reason == "routing_index_flush_boundary"
