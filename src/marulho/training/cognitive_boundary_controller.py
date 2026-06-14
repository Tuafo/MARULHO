from __future__ import annotations

from dataclasses import dataclass
from typing import Any


DRIFT_REFRESH_INTERVAL_TOKENS = 50


def _first_multiple_at_or_after(value: int, interval: int) -> int:
    interval = max(1, int(interval))
    value = int(value)
    remainder = value % interval
    if remainder == 0:
        return value
    return value + (interval - remainder)


def _has_multiple_in_range(start: int, stop_exclusive: int, interval: int) -> bool:
    if int(stop_exclusive) <= int(start):
        return False
    first = _first_multiple_at_or_after(int(start), int(interval))
    return first < int(stop_exclusive)


@dataclass(frozen=True)
class CognitiveBoundaryPlan:
    fallback_reason: str | None
    drift_refresh_due: bool
    drift_floor_close_due: bool
    telemetry_observation_due: bool
    slow_memory_cadence_due: bool

    @property
    def device_continuous(self) -> bool:
        return self.fallback_reason is None


class CognitiveBoundaryController:
    """Classify hot-path boundaries without performing cognition."""

    def __init__(self) -> None:
        self.plan_count = 0
        self.device_continuous_count = 0
        self.fallback_count = 0
        self.drift_refresh_count = 0
        self.drift_refresh_sync_free_count = 0
        self.drift_refresh_global_count = 0
        self.drift_floor_close_count = 0
        self.telemetry_observation_deferred_count = 0
        self.slow_memory_cadence_deferred_count = 0
        self.last_fallback_reason: str | None = None
        self.last_drift_refresh_token: int | None = None
        self.last_drift_floor_close_token: int | None = None
        self.last_telemetry_observation_token: int | None = None
        self.last_slow_memory_cadence_token: int | None = None

    def plan(
        self,
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
        self.plan_count += 1
        plan = self.classify(
            start_token=start_token,
            token_count=token_count,
            telemetry_interval=telemetry_interval,
            slow_memory_archive_interval=slow_memory_archive_interval,
            drift_floor_window_tokens=drift_floor_window_tokens,
            hnsw_flush_interval=hnsw_flush_interval,
            hnsw_buffer_pending=hnsw_buffer_pending,
            deep_sleep_interval_tokens=deep_sleep_interval_tokens,
            last_deep_sleep_token=last_deep_sleep_token,
            pending_emergency_deep_sleep=pending_emergency_deep_sleep,
            emergency_deep_sleep_cooldown_tokens=(
                emergency_deep_sleep_cooldown_tokens
            ),
            micro_sleep_interval_tokens=micro_sleep_interval_tokens,
            last_micro_sleep_token=last_micro_sleep_token,
        )
        if plan.fallback_reason is None:
            self.device_continuous_count += 1
        else:
            self.fallback_count += 1
            self.last_fallback_reason = plan.fallback_reason
        return plan

    @staticmethod
    def classify(
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
        fallback_reason: str | None = None
        start = int(start_token)
        end_token = start + int(token_count)
        fallback_token: int | None = None

        if end_token > start and bool(hnsw_buffer_pending):
            first_routing = _first_multiple_at_or_after(
                start,
                max(1, int(hnsw_flush_interval)),
            )
            if first_routing < end_token:
                fallback_token = first_routing
                fallback_reason = "routing_index_flush_boundary"

        if end_token > start:
            sleep_candidates = [
                max(
                    int(deep_sleep_interval_tokens),
                    int(last_deep_sleep_token)
                    + int(deep_sleep_interval_tokens),
                ),
                max(
                    int(micro_sleep_interval_tokens),
                    int(last_micro_sleep_token)
                    + int(micro_sleep_interval_tokens),
                ),
            ]
            if bool(pending_emergency_deep_sleep):
                sleep_candidates.append(
                    int(last_deep_sleep_token)
                    + int(emergency_deep_sleep_cooldown_tokens)
                )
            first_sleep = max(start, min(sleep_candidates))
            if first_sleep < end_token and (
                fallback_token is None or first_sleep < fallback_token
            ):
                fallback_token = first_sleep
                fallback_reason = "sleep_boundary"

        scan_stop = (
            int(fallback_token) + 1
            if fallback_token is not None
            else end_token
        )
        drift_refresh_due = _has_multiple_in_range(
            start,
            scan_stop,
            DRIFT_REFRESH_INTERVAL_TOKENS,
        )
        telemetry_observation_due = _has_multiple_in_range(
            start,
            scan_stop,
            max(1, int(telemetry_interval)),
        )
        drift_floor_close_due = _has_multiple_in_range(
            start + 1,
            scan_stop + 1,
            max(1, int(drift_floor_window_tokens)),
        )
        slow_memory_cadence_due = _has_multiple_in_range(
            start + 1,
            scan_stop + 1,
            max(1, int(slow_memory_archive_interval)),
        )

        return CognitiveBoundaryPlan(
            fallback_reason=fallback_reason,
            drift_refresh_due=drift_refresh_due,
            drift_floor_close_due=drift_floor_close_due,
            telemetry_observation_due=telemetry_observation_due,
            slow_memory_cadence_due=slow_memory_cadence_due,
        )

    def record_drift_refresh(
        self,
        *,
        token: int,
        sync_free: bool = False,
        global_drift: bool = False,
    ) -> None:
        self.drift_refresh_count += 1
        self.drift_refresh_sync_free_count += int(bool(sync_free))
        self.drift_refresh_global_count += int(bool(global_drift))
        self.last_drift_refresh_token = int(token)

    def record_drift_floor_close(self, *, token: int) -> None:
        self.drift_floor_close_count += 1
        self.last_drift_floor_close_token = int(token)

    def record_telemetry_deferred(self, *, token: int) -> None:
        self.telemetry_observation_deferred_count += 1
        self.last_telemetry_observation_token = int(token)

    def record_slow_memory_cadence_deferred(self, *, token: int) -> None:
        self.slow_memory_cadence_deferred_count += 1
        self.last_slow_memory_cadence_token = int(token)

    def report(self) -> dict[str, Any]:
        return {
            "surface": "device_owned_cognitive_boundary_controller.v1",
            "plan_count": int(self.plan_count),
            "device_continuous_count": int(self.device_continuous_count),
            "fallback_count": int(self.fallback_count),
            "drift_refresh_interval_tokens": DRIFT_REFRESH_INTERVAL_TOKENS,
            "drift_refresh_count": int(self.drift_refresh_count),
            "drift_refresh_sync_free_count": int(
                self.drift_refresh_sync_free_count
            ),
            "drift_refresh_global_count": int(
                self.drift_refresh_global_count
            ),
            "drift_floor_close_count": int(self.drift_floor_close_count),
            "telemetry_observation_deferred_count": int(
                self.telemetry_observation_deferred_count
            ),
            "slow_memory_cadence_deferred_count": int(
                self.slow_memory_cadence_deferred_count
            ),
            "last_fallback_reason": self.last_fallback_reason,
            "last_drift_refresh_token": self.last_drift_refresh_token,
            "last_drift_floor_close_token": self.last_drift_floor_close_token,
            "last_telemetry_observation_token": (
                self.last_telemetry_observation_token
            ),
            "last_slow_memory_cadence_token": (
                self.last_slow_memory_cadence_token
            ),
            "exploration_execution_gate": False,
            "telemetry_execution_gate": False,
            "drift_refresh_execution_gate": False,
            "drift_refresh_requires_host_truth": False,
            "slow_memory_cadence_execution_gate": False,
            "cpu_maintenance_after_device_burst": True,
            "classification_mode": "range_arithmetic",
        }
