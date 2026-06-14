from __future__ import annotations

from dataclasses import dataclass
from typing import Any


DRIFT_REFRESH_INTERVAL_TOKENS = 50


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
        drift_refresh_due = False
        drift_floor_close_due = False
        telemetry_observation_due = False
        slow_memory_cadence_due = False
        fallback_reason: str | None = None
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

        if fallback_reason is None:
            self.device_continuous_count += 1
        else:
            self.fallback_count += 1
            self.last_fallback_reason = fallback_reason
        return CognitiveBoundaryPlan(
            fallback_reason=fallback_reason,
            drift_refresh_due=drift_refresh_due,
            drift_floor_close_due=drift_floor_close_due,
            telemetry_observation_due=telemetry_observation_due,
            slow_memory_cadence_due=slow_memory_cadence_due,
        )

    def record_drift_refresh(self, *, token: int) -> None:
        self.drift_refresh_count += 1
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
            "slow_memory_cadence_execution_gate": False,
            "cpu_maintenance_after_device_burst": True,
        }
