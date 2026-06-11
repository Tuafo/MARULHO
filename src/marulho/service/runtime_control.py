from __future__ import annotations

from datetime import datetime, timezone
import logging as _logging
from threading import Event, Thread
import time
from typing import Any

from marulho.config.model_config import MarulhoConfig
from marulho.service.runtime_prewarm import RuntimePrewarmer
from marulho.service.terminus_presets import TERMINUS_QUICK_START_PRESETS
from marulho.training.model import MarulhoModel
from marulho.training.trainer import MarulhoTrainer

DEFAULT_BRAIN_TICK_TOKENS = 512
DEFAULT_BRAIN_SLEEP_INTERVAL_SECONDS = 0.01
DEFAULT_BRAIN_STOP_TIMEOUT_SECONDS = 15.0

_terminus_runtime_logger = _logging.getLogger(__name__ + ".terminus_runtime")

def _build_runtime_control_initial_state() -> dict[str, Any]:
    active_execution_idle_event = Event()
    active_execution_idle_event.set()
    return {
        "_active_execution_idle_event": active_execution_idle_event,
        "_active_execution_requests": 0,
        "_brain_stop_event": None,
        "_brain_stop_requested_at": None,
        "_brain_stop_requested_perf": None,
        "_brain_stop_requested_reason": None,
        "_brain_stop_timed_out": False,
        "_brain_thread": None,
        "_brain_running": False,
        "_brain_running_since": None,
        "_brain_tick_in_progress_started_at": None,
        "_brain_tick_in_progress_started_perf": None,
        "_brain_tick_in_progress_phase": None,
        "_brain_tick_in_progress_source_name": None,
        "_brain_tick_in_progress_target_tokens": None,
        "_brain_last_stop_duration_ms": None,
        "_ingestion_configured_at": None,
        "_ingestion_configured_perf": None,
        "_ingestion_prewarm_budget_exhausted": False,
        "_ingestion_prewarm_completed_at": None,
        "_ingestion_prewarm_last_duration_ms": None,
        "_ingestion_prewarm_last_error": None,
        "_ingestion_prewarm_last_trigger": None,
        "_ingestion_prewarm_run_count": 0,
        "_ingestion_prewarm_running": False,
        "_ingestion_prewarm_started_at": None,
        "_ingestion_prewarm_started_perf": None,
        "_ingestion_prewarm_stop_event": None,
        "_ingestion_prewarm_thread": None,
        "_ingestion_startup_warm_latency_ms": None,
        "_ingestion_warm_ready_at": None,
        "_remote_warm_promotion_last_trigger": None,
        "_remote_warm_promotion_running": False,
        "_remote_warm_promotion_stop_event": None,
        "_remote_warm_promotion_thread": None,
        "_sensory_configured_at": None,
        "_sensory_configured_perf": None,
        "_sensory_prewarm_budget_exhausted": False,
        "_sensory_startup_warm_latency_ms": None,
        "_sensory_warm_ready_at": None,
    }


RUNTIME_CONTROL_STATE_FIELDS = frozenset(_build_runtime_control_initial_state())


class RuntimeControl(RuntimePrewarmer):
    """Terminus configure/start/stop/tick runtime control helpers."""

    def __init__(self, dependencies: Any | None = None) -> None:
        object.__setattr__(self, "_dependencies", dependencies)
        for field_name, initial_value in _build_runtime_control_initial_state().items():
            object.__setattr__(self, field_name, initial_value)

    @property
    def dependencies(self) -> Any:
        return object.__getattribute__(self, "_dependencies")

    def configure_terminus(
        self,
        *,
        source_bank: list[dict[str, Any]],
        tick_tokens: int = DEFAULT_BRAIN_TICK_TOKENS,
        sleep_interval_seconds: float = DEFAULT_BRAIN_SLEEP_INTERVAL_SECONDS,
        repeat_sources: bool = True,
        autonomy: dict[str, Any] | None = None,
        sensory: dict[str, Any] | None = None,
        ingestion: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        thread = self._request_brain_stop()
        self._join_brain_thread(thread)
        prewarm_thread = self._request_ingestion_prewarm_stop()
        self._join_ingestion_prewarm_thread(prewarm_thread)
        promotion_thread = self._request_remote_warm_promotion_stop()
        self._join_remote_warm_promotion_thread(promotion_thread)
        with self._lock:
            self._brain_config = self._runtime_config._normalize_brain_config(
                {
                    "source_bank": source_bank,
                    "tick_tokens": tick_tokens,
                    "sleep_interval_seconds": sleep_interval_seconds,
                    "repeat_sources": repeat_sources,
                    "autonomy": autonomy,
                    "sensory": sensory,
                    "ingestion": ingestion,
                }
            )
            self._brain_source_utility = {}
            self._brain_last_error = None
            self._last_real_sensory_episode_time = 0.0
            self._last_real_sensory_episode_token_count = int(self._trainer.token_count)
            self._real_sensory_last_error = None
            self._record_brain_event_locked({
                "type": "configured",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source_names": [str(item.get("name", "")) for item in self._brain_config.get("source_bank", [])],
            })
            self._brain_last_acquisition_summary = None
            self._brain_last_acquisition_token_count = int(self._trainer.token_count)
            self._rebuild_brain_sources_locked()
            self._start_ingestion_prewarm_locked(trigger="configure")
            self._start_remote_warm_promotion_locked(trigger="configure")
            self._runtime_state.mark_mutated()
            return {
                "terminus_runtime": self._brain_runtime_snapshot_locked(),
                **self._runtime_state.mutation_summary(),
                "token_count": int(self._trainer.token_count),
            }

    def _brain_runtime_active_locked(self) -> bool:
        thread = self._brain_thread
        if thread is None:
            return False
        if thread.is_alive():
            return True
        self._finalize_brain_stop_locked(thread)
        return False

    def _assert_manual_tick_allowed_locked(self) -> None:
        if self._brain_runtime_active_locked():
            raise ValueError(
                "Cannot tick Terminus manually while the background runtime is active. Stop the runtime first."
            )

    def start_terminus(self) -> dict[str, Any]:
        with self._lock:
            if not self._brain_config.get("source_bank"):
                raise ValueError("Terminus runtime has no configured source_bank")
            if self._brain_runtime_active_locked():
                return {
                    "terminus_runtime": self._brain_runtime_snapshot_locked(),
                    **self._runtime_state.mutation_summary(),
                    "token_count": int(self._trainer.token_count),
                }
            self._brain_stop_event = Event()
            self._start_ingestion_prewarm_locked(trigger="start")
            self._start_remote_warm_promotion_locked(trigger="start")
            self._brain_thread = Thread(target=self._brain_loop, name="marulho-brain-loop", daemon=True)
            self._brain_running = True
            self._brain_running_since = datetime.now(timezone.utc).isoformat()
            self._brain_last_error = None
            self._record_brain_event_locked({
                "type": "started",
                "timestamp": self._brain_running_since,
            })
            self._brain_thread.start()

            result = {
                "terminus_runtime": self._brain_runtime_snapshot_locked(),
                **self._runtime_state.mutation_summary(),
                "token_count": int(self._trainer.token_count),
            }
        return result

    def stop_terminus(self) -> dict[str, Any]:
        thread = self._request_brain_stop(reason="manual")
        self._join_brain_thread(thread)
        prewarm_thread = self._request_ingestion_prewarm_stop()
        self._join_ingestion_prewarm_thread(prewarm_thread)

        with self._lock:
            return {
                "terminus_runtime": self._brain_runtime_snapshot_locked(),
                **self._runtime_state.mutation_summary(),
                "token_count": int(self._trainer.token_count),
            }

    def quick_start_terminus(self, *, preset: str = "curriculum") -> dict[str, Any]:
        """Configure and start Terminus in one atomic call using a named preset.

        If the preset includes ``model_overrides`` that differ from the current
        model (e.g. different n_columns or binding_mode), the model is rebuilt
        from scratch with the new config before starting.
        """
        if preset not in TERMINUS_QUICK_START_PRESETS:
            raise ValueError(f"Unknown preset '{preset}'. Available: {', '.join(sorted(TERMINUS_QUICK_START_PRESETS))}")
        with self._lock:
            if self._brain_runtime_active_locked():
                return {
                    "terminus_runtime": self._brain_runtime_snapshot_locked(),
                    **self._runtime_state.mutation_summary(),
                    "token_count": int(self._trainer.token_count),
                    "already_running": True,
                }
        config = TERMINUS_QUICK_START_PRESETS[preset]
        overrides = config.get("model_overrides")
        if overrides:
            current_cfg = self._trainer.config
            needs_rebuild = any(
                getattr(current_cfg, k, None) != v for k, v in overrides.items()
            )
            if needs_rebuild:
                cfg_dict = {
                    field_name: getattr(current_cfg, field_name)
                    for field_name, field_obj in current_cfg.__dataclass_fields__.items()
                    if field_obj.init
                }
                cfg_dict.update(overrides)
                new_cfg = MarulhoConfig(**cfg_dict)
                new_model = MarulhoModel(new_cfg)
                self._trainer = MarulhoTrainer(new_model, new_cfg)
                self._encoder = self._trainer.encoder
                self._refresh_root_captures_locked()
        self.configure_terminus(
            source_bank=config["source_bank"],
            tick_tokens=config["tick_tokens"],
            sleep_interval_seconds=config["sleep_interval_seconds"],
            repeat_sources=config["repeat_sources"],
            autonomy=config.get("autonomy"),
            sensory=config.get("sensory"),
        )
        result = self.start_terminus()
        result["already_running"] = False
        result["preset_applied"] = preset
        return result

    @staticmethod
    def quick_start_presets() -> list[dict[str, Any]]:
        """Return available quick-start presets for the UI/API.

        The preset surface is intentionally narrow: only the current supported
        Terminus runtime path is exposed.
        """
        presets = [
            {
                "id": key,
                "label": val["label"],
                "description": val["description"],
                "source_count": len(val["source_bank"]),
                "default": bool(val.get("default", False)),
                "legacy": bool(val.get("legacy", False)),
            }
            for key, val in TERMINUS_QUICK_START_PRESETS.items()
        ]
        presets.sort(key=lambda item: (not item["default"], item["legacy"], item["label"]))
        return presets

    def terminus_tick(self, *, steps: int = 1) -> dict[str, Any]:
        tick_summaries: list[dict[str, Any]] = []
        step_count = max(1, int(steps))

        with self._lock:
            self._assert_manual_tick_allowed_locked()

        self._request_active_execution()
        try:
            with self._brain_execution_lock:
                with self._lock:
                    self._assert_manual_tick_allowed_locked()

                for _ in range(step_count):
                    summary = self._run_brain_tick_once(
                        stop_event=None,
                        sub_batch_size=1,
                        yield_seconds=0.0,
                    )
                    if summary is None:
                        break
                    summary["developmental_autonomy"] = (
                        self.dependencies._run_developmental_autonomy_after_tick(
                            tick_summary=summary
                        )
                    )
                    tick_summaries.append(summary)
                    if not bool(summary.get("did_work", False)):
                        break
        finally:
            self._release_active_execution()

        with self._lock:
            return {
                "terminus_runtime": self._brain_runtime_snapshot_locked(),
                "tick_summaries": tick_summaries,
                **self._runtime_state.mutation_summary(),
                "token_count": int(self._trainer.token_count),
            }

    def _request_brain_stop(self, *, reason: str | None = None) -> Thread | None:
        with self._lock:
            return self._request_brain_stop_locked(reason=reason)

    def _finalize_brain_stop_locked(self, thread: Thread | None) -> None:
        active_thread = self._brain_thread
        if thread is not None and active_thread is not None and active_thread is not thread and active_thread.is_alive():
            return
        elapsed_ms = None
        if self._brain_stop_requested_perf is not None:
            elapsed_ms = (time.perf_counter() - self._brain_stop_requested_perf) * 1000.0
        self._brain_running = False
        self._brain_running_since = None
        self._brain_thread = None
        self._brain_stop_event = None
        event_reason = self._brain_stop_requested_reason
        timed_out = bool(self._brain_stop_timed_out)
        self._brain_stop_requested_perf = None
        self._brain_stop_requested_at = None
        self._brain_stop_requested_reason = None
        self._brain_stop_timed_out = False
        self._active_execution_requests = 0
        self._active_execution_idle_event.set()
        self._brain_last_stop_duration_ms = elapsed_ms
        if event_reason is not None:
            self._record_brain_event_locked(
                {
                    "type": "stopped_after_timeout" if timed_out else "stopped",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "reason": event_reason,
                    "stop_duration_ms": None if elapsed_ms is None else float(elapsed_ms),
                }
            )

    def _begin_brain_tick_progress_locked(
        self,
        *,
        phase: str,
        source_name: str | None = None,
        target_tokens: int | None = None,
    ) -> None:
        if self._brain_tick_in_progress_started_perf is None:
            self._brain_tick_in_progress_started_at = datetime.now(timezone.utc).isoformat()
            self._brain_tick_in_progress_started_perf = time.perf_counter()
        self._brain_tick_in_progress_phase = str(phase)
        self._brain_tick_in_progress_source_name = source_name
        self._brain_tick_in_progress_target_tokens = None if target_tokens is None else int(target_tokens)

    def _clear_brain_tick_progress_locked(self) -> None:
        self._brain_tick_in_progress_started_at = None
        self._brain_tick_in_progress_started_perf = None
        self._brain_tick_in_progress_phase = None
        self._brain_tick_in_progress_source_name = None
        self._brain_tick_in_progress_target_tokens = None

    def _brain_execution_snapshot_locked(self) -> dict[str, Any]:
        started_perf = self._brain_tick_in_progress_started_perf
        elapsed_ms = None
        if started_perf is not None:
            elapsed_ms = float((time.perf_counter() - started_perf) * 1000.0)
        return {
            "active_execution_requests": int(self._active_execution_requests),
            "idle": bool(self._active_execution_idle_event.is_set()),
            "tick_in_progress": started_perf is not None,
            "tick_started_at": self._brain_tick_in_progress_started_at,
            "tick_elapsed_ms": elapsed_ms,
            "tick_phase": self._brain_tick_in_progress_phase,
            "tick_source_name": self._brain_tick_in_progress_source_name,
            "tick_target_tokens": self._brain_tick_in_progress_target_tokens,
        }

    def _join_brain_thread(
        self,
        thread: Thread | None,
        *,
        timeout: float = DEFAULT_BRAIN_STOP_TIMEOUT_SECONDS,
        raise_on_timeout: bool = True,
    ) -> bool:
        if thread is None:
            with self._lock:
                self._finalize_brain_stop_locked(thread)
            return True
        thread.join(timeout=timeout)
        if not thread.is_alive():
            with self._lock:
                self._finalize_brain_stop_locked(thread)
            return True

        message = (
            f"Terminus runtime did not stop within {timeout:.1f}s. "
            f"Reason={self._brain_stop_requested_reason or 'unknown'}"
        )
        with self._lock:
            self._brain_stop_timed_out = True
            self._brain_last_stop_duration_ms = float(timeout * 1000.0)
            self._brain_last_error = message
            self._record_brain_event_locked(
                {
                    "type": "stop_timeout",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "reason": self._brain_stop_requested_reason,
                    "timeout_seconds": float(timeout),
                    "thread_alive": True,
                }
            )
        if raise_on_timeout:
            raise RuntimeError(message)
        _terminus_runtime_logger.warning(message)
        return False

    def _request_brain_stop_locked(self, *, reason: str | None = None) -> Thread | None:
        thread = self._brain_thread if self._brain_thread is not None and self._brain_thread.is_alive() else None
        stop_event = self._brain_stop_event
        if stop_event is not None:
            stop_event.set()
        self._brain_running = False
        if thread is not None:
            self._brain_stop_requested_at = datetime.now(timezone.utc).isoformat()
            self._brain_stop_requested_reason = reason
            self._brain_stop_requested_perf = time.perf_counter()
            self._brain_stop_timed_out = False
            if reason is not None:
                self._record_brain_event_locked(
                    {
                        "type": "stop_requested",
                        "timestamp": self._brain_stop_requested_at,
                        "reason": reason,
                    }
                )
        else:
            self._finalize_brain_stop_locked(thread)
        return thread

    def _brain_loop(self) -> None:
        _SUB_BATCH = 1  # max tokens trained per lock acquisition
        _YIELD_SECONDS = 0.005  # yield between token steps for SSE/API/stop responsiveness
        while True:
            with self._lock:
                stop_event = self._brain_stop_event
                sleep_interval = float(self._brain_config.get("sleep_interval_seconds", DEFAULT_BRAIN_SLEEP_INTERVAL_SECONDS))
            if stop_event is None or stop_event.is_set():
                break
            try:
                self._request_active_execution()
                try:
                    with self._brain_execution_lock:
                        result = self._run_brain_tick_once(
                            stop_event=stop_event,
                            sub_batch_size=_SUB_BATCH,
                            yield_seconds=_YIELD_SECONDS,
                        )
                finally:
                    self._release_active_execution()
                if result is None:
                    break
                if isinstance(result, dict):
                    result["developmental_autonomy"] = (
                        self.dependencies._run_developmental_autonomy_after_tick(
                            tick_summary=result
                        )
                    )
                did_work = result.get("did_work", False) if isinstance(result, dict) else False
                actual_sleep = max(0.001, sleep_interval * 0.1) if did_work else max(0.05, sleep_interval)
            except Exception as exc:
                with self._lock:
                    self._brain_last_error = str(exc)
                    self._record_brain_event_locked({
                        "type": "error",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "message": str(exc),
                    })
                    self._request_brain_stop_locked(reason="error")
                break
            time.sleep(actual_sleep)

    def _run_brain_tick_once(
        self,
        *,
        stop_event: Event | None,
        sub_batch_size: int,
        yield_seconds: float,
    ) -> dict[str, Any] | None:
        tick_started = time.perf_counter()
        try:
            with self._lock:
                if stop_event is not None and stop_event.is_set():
                    return None
                self._begin_brain_tick_progress_locked(phase="select_source")
                if not self._brain_source_runtimes:
                    return self._brain_tick_idle_locked(tick_started)
                self._brain_stream_epoch += 1
                runtimes = list(self._brain_source_runtimes)
                src_index = self._brain_source_index
                tick_tokens = int(self._brain_config.get("tick_tokens", DEFAULT_BRAIN_TICK_TOKENS))
                repeat = bool(self._brain_config.get("repeat_sources", True))
                ingestion = self._brain_config.get("ingestion") or {}
                queue_target_tokens = int(
                    tick_tokens
                    if not bool(ingestion.get("enabled", True))
                    else ingestion.get("queue_target_tokens", tick_tokens)
                )
                encoder_ref = self._encoder
                window_size = self._trainer.config.window_size

            if len(runtimes) <= 1:
                ordered_indices = [0]
            else:
                ordered_indices: list[int]
                with self._lock:
                    ordered_indices, _background_focus_terms, _background_focus_pressure = self._ordered_brain_runtime_indices_locked(
                        start_index=src_index,
                    )

            with self._lock:
                first_source_name = None
                if ordered_indices and 0 <= int(ordered_indices[0]) < len(runtimes):
                    first_source_name = str(runtimes[int(ordered_indices[0])].name)
                self._begin_brain_tick_progress_locked(
                    phase="collect_source_queue",
                    source_name=first_source_name,
                    target_tokens=tick_tokens,
                )

            chunk, collect_meta = self._collect_chunk_unlocked(
                runtimes,
                ordered_indices,
                tick_tokens,
                queue_target_tokens,
                repeat,
                encoder_ref,
                window_size,
                stop_event,
            )

            if stop_event is not None and stop_event.is_set():
                return None

            if chunk is None:
                with self._lock:
                    self._begin_brain_tick_progress_locked(phase="idle_no_chunk")
                    return self._brain_tick_idle_locked(tick_started, source_meta=collect_meta)

            with self._lock:
                source_name = None if collect_meta is None else str(collect_meta["runtime"].name)
                self._begin_brain_tick_progress_locked(
                    phase="train_sub_batches",
                    source_name=source_name,
                    target_tokens=len(chunk),
                )
                self._commit_collected_runtime_locked(collect_meta)
                if collect_meta is not None:
                    self._update_brain_runtime_cache_locked(collect_meta["runtime"], served_examples=chunk)

            background_memory_metadata = None if collect_meta is None else self._brain_source_memory_metadata(collect_meta["runtime"])
            total_trained, last_metrics, evidence_windows = self._train_chunk_in_sub_batches(
                chunk,
                stop_event=stop_event,
                sub_batch_size=sub_batch_size,
                yield_seconds=yield_seconds,
                memory_metadata=background_memory_metadata,
            )
            source_info = {
                "runtime": collect_meta["runtime"],
                "idx": collect_meta["idx"],
                "source_count": collect_meta["source_count"],
            } if collect_meta else None
            with self._lock:
                self._begin_brain_tick_progress_locked(
                    phase="finalize_tick",
                    source_name=None if collect_meta is None else str(collect_meta["runtime"].name),
                    target_tokens=total_trained,
                )
                return self._finalize_tick_locked(
                    tick_started,
                    source_info,
                    total_trained,
                    last_metrics,
                    evidence_windows,
                )
        finally:
            with self._lock:
                self._clear_brain_tick_progress_locked()


def _install_dependency_forwarders(cls: type, names: tuple[str, ...]) -> None:
    for raw_name in names:
        name = str(raw_name)
        if not name or hasattr(cls, name):
            continue

        def _get(self: RuntimeControl, *, _name: str = name) -> Any:
            dependencies = object.__getattribute__(self, "_dependencies")
            if dependencies is None:
                raise AttributeError(_name)
            return getattr(dependencies, _name)

        def _set(self: RuntimeControl, value: Any, *, _name: str = name) -> None:
            dependencies = object.__getattribute__(self, "_dependencies")
            if dependencies is None:
                object.__setattr__(self, _name, value)
                return
            setattr(dependencies, _name, value)

        setattr(cls, name, property(_get, _set))


_install_dependency_forwarders(RuntimeControl, (
    "_brain_config",
    "_brain_execution_lock",
    "_brain_last_acquisition_summary",
    "_brain_last_acquisition_token_count",
    "_brain_last_error",
    "_brain_runtime_snapshot_locked",
    "_brain_source_index",
    "_brain_source_memory_metadata",
    "_brain_source_runtimes",
    "_brain_source_utility",
    "_brain_stream_epoch",
    "_brain_tick_idle_locked",
    "_collect_chunk_unlocked",
    "_commit_collected_runtime_locked",
    "_commit_prefetched_sensory_runtime_locked",
    "_encoder",
    "_finalize_tick_locked",
    "_interrupt_brain_sources_locked",
    "_interrupt_sensory_sources_locked",
    "_ingestion_full_queue_source_count_locked",
    "_ingestion_ready_source_count_locked",
    "_join_ingestion_prewarm_thread",
    "_join_remote_warm_promotion_thread",
    "_last_real_sensory_episode_time",
    "_last_real_sensory_episode_token_count",
    "_lock",
    "_next_stream_item",
    "_ordered_brain_runtime_indices_locked",
    "_prefetch_sensory_queues_unlocked",
    "_prefetch_source_queues_unlocked",
    "_real_sensory_last_error",
    "_rebuild_brain_sources_locked",
    "_record_brain_event_locked",
    "_refresh_root_captures_locked",
    "_release_active_execution",
    "_request_active_execution",
    "_request_ingestion_prewarm_stop",
    "_request_remote_warm_promotion_stop",
    "_runtime_config",
    "_runtime_state",
    "_sensory_full_queue_source_count_locked",
    "_sensory_queue_target_items_locked",
    "_sensory_ready_source_count_locked",
    "_sensory_source_runtimes",
    "_sensory_spec_uses_live_remote",
    "_sensory_stream_epoch",
    "_source_spec_uses_live_remote",
    "_start_ingestion_prewarm_locked",
    "_start_remote_warm_promotion_locked",
    "_stream_supports_ready_reads",
    "_train_chunk_in_sub_batches",
    "_trainer",
    "_update_brain_runtime_cache_locked",
    "_update_sensory_runtime_cache_locked",
))


