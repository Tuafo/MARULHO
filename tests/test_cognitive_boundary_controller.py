from marulho.training.cognitive_boundary_controller import (
    CognitiveBoundaryController,
    CognitiveBoundaryPlan,
    DRIFT_REFRESH_INTERVAL_TOKENS,
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


def test_classify_previews_boundaries_without_runtime_truth_mutation() -> None:
    controller = CognitiveBoundaryController()

    preview = controller.classify(
        start_token=16,
        token_count=8,
        telemetry_interval=64,
        slow_memory_archive_interval=256,
        drift_floor_window_tokens=10_000,
        hnsw_flush_interval=16,
        hnsw_buffer_pending=True,
        deep_sleep_interval_tokens=10_000,
        last_deep_sleep_token=0,
        pending_emergency_deep_sleep=False,
        emergency_deep_sleep_cooldown_tokens=1_000,
        micro_sleep_interval_tokens=10_000,
        last_micro_sleep_token=0,
    )

    assert preview.fallback_reason == "routing_index_flush_boundary"
    report = controller.report()
    assert report["plan_count"] == 0
    assert report["fallback_count"] == 0
    assert report["last_fallback_reason"] is None

    actual = _plan(
        controller,
        start_token=16,
        hnsw_buffer_pending=True,
    )
    report = controller.report()
    assert actual.fallback_reason == "routing_index_flush_boundary"
    assert report["plan_count"] == 1
    assert report["fallback_count"] == 1
    assert report["last_fallback_reason"] == "routing_index_flush_boundary"


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


def test_slow_memory_cadence_defers_after_device_execution() -> None:
    controller = CognitiveBoundaryController()

    plan = _plan(
        controller,
        start_token=248,
        slow_memory_archive_interval=256,
    )

    assert plan.device_continuous is True
    assert plan.fallback_reason is None
    assert plan.slow_memory_cadence_due is True
    assert controller.report()["fallback_count"] == 0
    controller.record_slow_memory_cadence_deferred(token=256)
    report = controller.report()
    assert report["slow_memory_cadence_deferred_count"] == 1
    assert report["last_slow_memory_cadence_token"] == 256
    assert report["slow_memory_cadence_execution_gate"] is False


def test_pending_routing_index_flush_remains_a_real_fallback() -> None:
    controller = CognitiveBoundaryController()

    plan = _plan(
        controller,
        start_token=16,
        hnsw_buffer_pending=True,
    )

    assert plan.fallback_reason == "routing_index_flush_boundary"


def _reference_loop_plan(
    *,
    start_token: int,
    token_count: int,
    telemetry_interval: int,
    slow_memory_archive_interval: int,
    drift_floor_window_tokens: int,
    hnsw_flush_interval: int,
    hnsw_buffer_pending: bool,
    deep_sleep_interval_tokens: int,
    last_deep_sleep_token: int,
    pending_emergency_deep_sleep: bool,
    emergency_deep_sleep_cooldown_tokens: int,
    micro_sleep_interval_tokens: int,
    last_micro_sleep_token: int,
) -> CognitiveBoundaryPlan:
    drift_refresh_due = False
    drift_floor_close_due = False
    telemetry_observation_due = False
    slow_memory_cadence_due = False
    fallback_reason = None
    end_token = int(start_token) + int(token_count)
    for current in range(int(start_token), end_token):
        next_token = current + 1
        drift_refresh_due = (
            drift_refresh_due
            or current % DRIFT_REFRESH_INTERVAL_TOKENS == 0
        )
        telemetry_observation_due = (
            telemetry_observation_due
            or current % max(1, int(telemetry_interval)) == 0
        )
        drift_floor_close_due = (
            drift_floor_close_due
            or next_token % max(1, int(drift_floor_window_tokens)) == 0
        )
        slow_memory_cadence_due = (
            slow_memory_cadence_due
            or next_token % max(1, int(slow_memory_archive_interval)) == 0
        )
        if (
            current % max(1, int(hnsw_flush_interval)) == 0
            and hnsw_buffer_pending
        ):
            fallback_reason = "routing_index_flush_boundary"
            break
        deep_due = (
            current >= int(deep_sleep_interval_tokens)
            and current - int(last_deep_sleep_token)
            >= int(deep_sleep_interval_tokens)
        )
        emergency_due = (
            bool(pending_emergency_deep_sleep)
            and current - int(last_deep_sleep_token)
            >= int(emergency_deep_sleep_cooldown_tokens)
        )
        micro_due = (
            current >= int(micro_sleep_interval_tokens)
            and current - int(last_micro_sleep_token)
            >= int(micro_sleep_interval_tokens)
        )
        if deep_due or emergency_due or micro_due:
            fallback_reason = "sleep_boundary"
            break
    return CognitiveBoundaryPlan(
        fallback_reason=fallback_reason,
        drift_refresh_due=drift_refresh_due,
        drift_floor_close_due=drift_floor_close_due,
        telemetry_observation_due=telemetry_observation_due,
        slow_memory_cadence_due=slow_memory_cadence_due,
    )


def test_range_arithmetic_classifier_matches_loop_semantics() -> None:
    starts = [0, 1, 15, 16, 31, 49, 56, 63, 248, 255, 9998, 10_001]
    token_counts = [0, 1, 7, 8, 16, 33]
    for start_token in starts:
        for token_count in token_counts:
            for hnsw_pending in (False, True):
                for pending_emergency in (False, True):
                    kwargs = {
                        "start_token": start_token,
                        "token_count": token_count,
                        "telemetry_interval": 8,
                        "slow_memory_archive_interval": 32,
                        "drift_floor_window_tokens": 64,
                        "hnsw_flush_interval": 16,
                        "hnsw_buffer_pending": hnsw_pending,
                        "deep_sleep_interval_tokens": 10_000,
                        "last_deep_sleep_token": 0,
                        "pending_emergency_deep_sleep": pending_emergency,
                        "emergency_deep_sleep_cooldown_tokens": 12,
                        "micro_sleep_interval_tokens": 9,
                        "last_micro_sleep_token": 0,
                    }
                    assert CognitiveBoundaryController.classify(**kwargs) == (
                        _reference_loop_plan(**kwargs)
                    )


def test_boundary_report_exposes_range_arithmetic_mode() -> None:
    controller = CognitiveBoundaryController()

    report = controller.report()

    assert report["classification_mode"] == "range_arithmetic"
