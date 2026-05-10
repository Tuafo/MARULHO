"""Remote warmup and ingestion prewarm helpers for Terminus.

This mixin owns background queue warming for live remote text/sensory sources.
It fills runtime buffers only; it does not approve replay datasets, train
adapters, promote contradicted content, or execute digital actions.
"""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from datetime import datetime, timezone
from queue import Empty, Queue
from threading import Event, Thread
import sys
import time
from typing import Any, Mapping, Sequence, cast

import torch

from hecsn.data.corpus_loader import load_hf_first_rows
from hecsn.data.pattern_loader import labeled_pattern_stream
from hecsn.service.runtime_sources import _BrainSourceRuntime, _SensorySourceRuntime
from hecsn.service.terminus_sensory import (
    SensoryEpisode,
    bootstrap_sensory_episode_from_row,
    sensory_bootstrap_columns,
)

DEFAULT_BRAIN_TICK_TOKENS = 512
DEFAULT_REMOTE_PREWARM_GRACE_SECONDS = 0.25
DEFAULT_REMOTE_PREWARM_POLL_SECONDS = 0.05
DEFAULT_REMOTE_PROMOTION_BOOTSTRAP_GRACE_SECONDS = 0.3
DEFAULT_REMOTE_BOOTSTRAP_ROWS = 2
DEFAULT_REMOTE_BOOTSTRAP_BUDGET_SECONDS = 3.0


def _manager_symbol(name: str, fallback: Any) -> Any:
    manager_module = sys.modules.get("hecsn.service.manager")
    if manager_module is None:
        return fallback
    return getattr(manager_module, name, fallback)


def _remote_bootstrap_budget_seconds() -> float:
    return float(_manager_symbol("DEFAULT_REMOTE_BOOTSTRAP_BUDGET_SECONDS", DEFAULT_REMOTE_BOOTSTRAP_BUDGET_SECONDS))


def _remote_prewarm_grace_seconds() -> float:
    return float(_manager_symbol("DEFAULT_REMOTE_PREWARM_GRACE_SECONDS", DEFAULT_REMOTE_PREWARM_GRACE_SECONDS))


def _remote_prewarm_poll_seconds() -> float:
    return float(_manager_symbol("DEFAULT_REMOTE_PREWARM_POLL_SECONDS", DEFAULT_REMOTE_PREWARM_POLL_SECONDS))


def _remote_promotion_bootstrap_grace_seconds() -> float:
    return float(
        _manager_symbol(
            "DEFAULT_REMOTE_PROMOTION_BOOTSTRAP_GRACE_SECONDS",
            DEFAULT_REMOTE_PROMOTION_BOOTSTRAP_GRACE_SECONDS,
        )
    )


def _source_stream_builder(owner: Any) -> Any:
    manager = getattr(owner, "_manager", None)
    if manager is not None:
        builder = getattr(manager, "_build_source_stream_from_spec", None)
        if builder is not None:
            return builder
    return type(owner)._build_source_stream_from_spec


class _TimedCallFailure:
    def __init__(self, error: BaseException) -> None:
        self.error = error


class RuntimePrewarmMixin:
    def _request_active_execution_locked(self) -> None:
        self._active_execution_requests += 1
        if self._active_execution_requests > 0:
            self._active_execution_idle_event.clear()

    def _release_active_execution_locked(self) -> None:
        self._active_execution_requests = max(0, int(self._active_execution_requests) - 1)
        if self._active_execution_requests <= 0:
            self._active_execution_idle_event.set()

    def _request_active_execution(self) -> None:
        with self._lock:
            self._request_active_execution_locked()

    def _release_active_execution(self) -> None:
        with self._lock:
            self._release_active_execution_locked()

    def _wait_for_remote_prewarm_clearance(
        self,
        stop_event: Event | None,
        *,
        remote_text_target: bool,
        remote_sensory_target: bool,
    ) -> bool:
        if not (remote_text_target or remote_sensory_target):
            return True

        grace_seconds = max(0.0, float(_remote_prewarm_grace_seconds()))
        if grace_seconds > 0.0:
            deadline = time.perf_counter() + grace_seconds
            while True:
                if stop_event is not None and stop_event.is_set():
                    return False
                remaining = deadline - time.perf_counter()
                if remaining <= 0.0:
                    break
                time.sleep(min(float(_remote_prewarm_poll_seconds()), remaining))

        wait_started_perf: float | None = None
        wait_started_at: str | None = None
        while True:
            if stop_event is not None and stop_event.is_set():
                return False
            with self._lock:
                active_requested = bool(self._active_execution_requests > 0)
                trigger = self._ingestion_prewarm_last_trigger
            if not active_requested:
                if wait_started_perf is not None:
                    waited_ms = float((time.perf_counter() - wait_started_perf) * 1000.0)
                    with self._lock:
                        self._record_brain_event_locked(
                            {
                                "type": "ingestion_prewarm_active_execution_cleared",
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                                "trigger": trigger,
                                "wait_started_at": wait_started_at,
                                "wait_duration_ms": waited_ms,
                            }
                        )
                return True
            if wait_started_perf is None:
                wait_started_perf = time.perf_counter()
                wait_started_at = datetime.now(timezone.utc).isoformat()
                with self._lock:
                    self._record_brain_event_locked(
                        {
                            "type": "ingestion_prewarm_waiting_for_active_execution",
                            "timestamp": wait_started_at,
                            "trigger": trigger,
                            "remote_text_prewarm": bool(remote_text_target),
                            "remote_sensory_prewarm": bool(remote_sensory_target),
                            "grace_seconds": float(grace_seconds),
                        }
                    )
            self._active_execution_idle_event.wait(timeout=float(_remote_prewarm_poll_seconds()))

    def _remote_warm_promotion_text_needed_locked(self) -> bool:
        ingestion = self._brain_config.get("ingestion") or {}
        if not bool(ingestion.get("enabled", True)) or bool(ingestion.get("prewarm_on_startup", False)):
            return False
        target_tokens = max(1, int(ingestion.get("queue_target_tokens", self._brain_config.get("tick_tokens", DEFAULT_BRAIN_TICK_TOKENS))))
        return any(
            self._source_spec_uses_live_remote(runtime.spec)
            and self._stream_supports_ready_reads(runtime.stream)
            and len(runtime.buffered_patterns) < target_tokens
            and not runtime.exhausted
            for runtime in self._brain_source_runtimes
        )

    def _remote_warm_promotion_sensory_needed_locked(self) -> bool:
        sensory = self._brain_config.get("sensory") or {}
        if not bool(sensory.get("enabled", False)) or bool(sensory.get("prewarm_on_startup", False)):
            return False
        target_items = self._sensory_queue_target_items_locked()
        return any(
            self._sensory_spec_uses_live_remote(runtime.spec)
            and self._stream_supports_ready_reads(runtime.stream)
            and len(runtime.buffered_episodes) < target_items
            and not runtime.exhausted
            for runtime in self._sensory_source_runtimes
        )

    def _request_remote_warm_promotion_stop(self) -> Thread | None:
        with self._lock:
            thread = (
                self._remote_warm_promotion_thread
                if self._remote_warm_promotion_thread is not None and self._remote_warm_promotion_thread.is_alive()
                else None
            )
            stop_event = self._remote_warm_promotion_stop_event
            if stop_event is not None:
                stop_event.set()
            self._remote_warm_promotion_running = False
            return thread

    def _join_remote_warm_promotion_thread(self, thread: Thread | None, *, timeout: float = 5.0) -> bool:
        if thread is None:
            with self._lock:
                if self._remote_warm_promotion_thread is not None and not self._remote_warm_promotion_thread.is_alive():
                    self._remote_warm_promotion_thread = None
                    self._remote_warm_promotion_stop_event = None
            return True
        thread.join(timeout=timeout)
        with self._lock:
            if self._remote_warm_promotion_thread is thread and not thread.is_alive():
                self._remote_warm_promotion_thread = None
                self._remote_warm_promotion_stop_event = None
        return not thread.is_alive()

    def _record_remote_warm_promotion_completed_locked(self) -> None:
        last_event = self._runtime_state.last_event
        if not isinstance(last_event, Mapping):
            last_event = {}
        if str(last_event.get("type", "")) == "remote_warm_promotion_completed":
            return
        self._record_brain_event_locked(
            {
                "type": "remote_warm_promotion_completed",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "trigger": self._remote_warm_promotion_last_trigger,
                "ready_source_count": int(self._ingestion_ready_source_count_locked()),
                "sensory_ready_source_count": int(self._sensory_ready_source_count_locked()),
            }
        )

    def _start_remote_warm_promotion_locked(self, *, trigger: str) -> bool:
        text_needed = self._remote_warm_promotion_text_needed_locked()
        sensory_needed = self._remote_warm_promotion_sensory_needed_locked()
        if not (text_needed or sensory_needed):
            return False
        thread = self._remote_warm_promotion_thread
        if thread is not None and thread.is_alive():
            return False
        self._remote_warm_promotion_stop_event = Event()
        self._remote_warm_promotion_running = True
        self._remote_warm_promotion_last_trigger = trigger
        self._record_brain_event_locked(
            {
                "type": "remote_warm_promotion_started",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "trigger": trigger,
                "text_sources": int(
                    sum(
                        1
                        for runtime in self._brain_source_runtimes
                        if self._source_spec_uses_live_remote(runtime.spec) and not runtime.exhausted
                    )
                ),
                "sensory_sources": int(
                    sum(
                        1
                        for runtime in self._sensory_source_runtimes
                        if self._sensory_spec_uses_live_remote(runtime.spec) and not runtime.exhausted
                    )
                ),
            }
        )
        thread = Thread(target=self._remote_warm_promotion_loop, name="hecsn-remote-warm-promotion", daemon=True)
        self._remote_warm_promotion_thread = thread
        thread.start()
        return True

    @staticmethod
    def _remaining_budget_seconds(deadline_perf: float | None) -> float | None:
        if deadline_perf is None:
            return None
        return max(0.0, float(deadline_perf - time.perf_counter()))

    @staticmethod
    def _run_budgeted_call(func: Any, /, *args: Any, wait_seconds: float | None = None, **kwargs: Any) -> tuple[bool, Any]:
        if wait_seconds is None:
            return True, func(*args, **kwargs)
        budget = max(0.0, float(wait_seconds))
        if budget <= 0.0:
            return False, None
        payloads: Queue[object] = Queue(maxsize=1)

        def _runner() -> None:
            try:
                payload: object = func(*args, **kwargs)
            except BaseException as exc:  # pragma: no cover - background guard
                payload = _TimedCallFailure(exc)
            try:
                payloads.put_nowait(payload)
            except Exception:
                pass

        Thread(target=_runner, name="hecsn-budgeted-call", daemon=True).start()
        try:
            payload = payloads.get(timeout=budget)
        except Empty:
            return False, None
        if isinstance(payload, _TimedCallFailure):
            raise payload.error
        return True, payload

    def _remote_text_bootstrap_candidates_locked(self) -> list[tuple[_BrainSourceRuntime, dict[str, Any], int]]:
        ingestion = self._brain_config.get("ingestion") or {}
        target_tokens = max(1, int(ingestion.get("queue_target_tokens", self._brain_config.get("tick_tokens", DEFAULT_BRAIN_TICK_TOKENS))))
        candidates: list[tuple[_BrainSourceRuntime, dict[str, Any], int]] = []
        for runtime in self._brain_source_runtimes:
            if (
                runtime.bootstrap_attempted
                or not self._source_spec_uses_live_remote(runtime.spec)
                or str(runtime.spec.get("source_type", "auto")).strip().lower() != "hf"
                or len(runtime.buffered_patterns) > 0
                or runtime.exhausted
            ):
                continue
            runtime.bootstrap_attempted = True
            candidates.append((runtime, deepcopy(runtime.spec), int(target_tokens)))
        return candidates

    def _fetch_remote_text_bootstrap_rows(
        self,
        spec: Mapping[str, Any],
        *,
        deadline_perf: float | None = None,
    ) -> tuple[list[str], bool]:
        remaining = self._remaining_budget_seconds(deadline_perf)
        if remaining is not None and remaining <= 0.0:
            return [], True
        call_budget = float(_remote_bootstrap_budget_seconds() if remaining is None else min(_remote_bootstrap_budget_seconds(), remaining))
        try:
            completed, rows = self._run_budgeted_call(
                cast(Any, _manager_symbol("load_hf_first_rows", load_hf_first_rows)),
                str(spec.get("source", "")),
                wait_seconds=call_budget,
                hf_config=cast(str | None, spec.get("hf_config")),
                split="train",
                columns=[str(spec.get("text_field", "text") or "text")],
                max_rows=DEFAULT_REMOTE_BOOTSTRAP_ROWS,
                timeout_seconds=call_budget,
            )
        except Exception:
            return [], False
        if not completed:
            return [], True
        text_field = str(spec.get("text_field", "text") or "text")
        texts: list[str] = []
        for row in list(rows or []):
            if not isinstance(row, Mapping):
                continue
            text = str(row.get(text_field, "")).strip()
            if text:
                texts.append(text)
        return texts, False

    def _apply_remote_text_bootstrap_locked(
        self,
        runtime: _BrainSourceRuntime,
        texts: Sequence[str],
        *,
        target_tokens: int,
    ) -> int:
        if len(runtime.buffered_patterns) > 0 or runtime.exhausted:
            return 0
        examples: list[tuple[str, torch.Tensor]] = []
        for text in texts:
            for raw_window, pattern in labeled_pattern_stream(
                text,
                self._encoder,
                self._trainer.config.window_size,
                learn_chunking=True,
            ):
                examples.append((raw_window, pattern))
                if len(examples) >= target_tokens:
                    break
            if len(examples) >= target_tokens:
                break
        if not examples:
            return 0
        runtime.buffered_patterns.extend(examples)
        now = datetime.now(timezone.utc).isoformat()
        self._commit_collected_runtime_locked(
            {
                "runtime": runtime,
                "cycles": runtime.cycles_completed,
                "exhausted": runtime.exhausted,
                "new_stream": None,
                "served_tokens": 0,
                "queue_hit": False,
                "prefetch_tokens": int(len(examples)),
                "prefetch_duration_ms": 0.0,
                "prefetch_at": now,
                "prefetch_error": None,
                "warm_trigger": "remote_bootstrap",
            }
        )
        self._update_brain_runtime_cache_locked(runtime)
        self._record_brain_event_locked(
            {
                "type": "remote_text_bootstrap_applied",
                "timestamp": now,
                "source_name": runtime.name,
                "token_count": int(len(examples)),
            }
        )
        self._runtime_state.mark_mutated()
        return int(len(examples))

    def _remote_sensory_bootstrap_candidates_locked(self) -> list[tuple[_SensorySourceRuntime, dict[str, Any], int, int, torch.device]]:
        target_items = self._sensory_queue_target_items_locked()
        visual_dim = int(getattr(self._trainer.config, "cross_modal_dim_visual", 64))
        audio_dim = int(getattr(self._trainer.config, "cross_modal_dim_audio", 64))
        device = self._trainer.model.device
        candidates: list[tuple[_SensorySourceRuntime, dict[str, Any], int, int, torch.device]] = []
        for runtime in self._sensory_source_runtimes:
            if (
                runtime.bootstrap_attempted
                or not self._sensory_spec_uses_live_remote(runtime.spec)
                or len(runtime.buffered_episodes) > 0
                or runtime.exhausted
            ):
                continue
            runtime.bootstrap_attempted = True
            candidates.append((runtime, deepcopy(runtime.spec), int(visual_dim), int(audio_dim), device))
        return candidates

    def _fetch_remote_sensory_bootstrap_episodes(
        self,
        spec: Mapping[str, Any],
        *,
        visual_dim: int,
        audio_dim: int,
        device: torch.device,
        deadline_perf: float | None = None,
    ) -> tuple[list[SensoryEpisode], bool]:
        remaining = self._remaining_budget_seconds(deadline_perf)
        if remaining is not None and remaining <= 0.0:
            return [], True
        call_budget = float(_remote_bootstrap_budget_seconds() if remaining is None else min(_remote_bootstrap_budget_seconds(), remaining))
        try:
            completed, rows = self._run_budgeted_call(
                cast(Any, _manager_symbol("load_hf_first_rows", load_hf_first_rows)),
                str(spec.get("source", "")),
                wait_seconds=call_budget,
                hf_config=cast(str | None, spec.get("hf_config")) or "default",
                split=str(spec.get("split", "train") or "train"),
                columns=cast(Any, _manager_symbol("sensory_bootstrap_columns", sensory_bootstrap_columns))(spec),
                max_rows=DEFAULT_REMOTE_BOOTSTRAP_ROWS,
                timeout_seconds=call_budget,
            )
        except Exception:
            return [], False
        if not completed:
            return [], True
        episodes: list[SensoryEpisode] = []
        for row in list(rows or []):
            if not isinstance(row, Mapping):
                continue
            remaining = self._remaining_budget_seconds(deadline_perf)
            if remaining is not None and remaining <= 0.0:
                return episodes, True
            build_budget = float(_remote_bootstrap_budget_seconds() if remaining is None else min(_remote_bootstrap_budget_seconds(), remaining))
            try:
                completed, episode = self._run_budgeted_call(
                    cast(Any, _manager_symbol("bootstrap_sensory_episode_from_row", bootstrap_sensory_episode_from_row)),
                    spec,
                    row,
                    wait_seconds=build_budget,
                    visual_dim=visual_dim,
                    audio_dim=audio_dim,
                    device=device,
                    timeout_seconds=build_budget,
                )
            except Exception:
                continue
            if not completed:
                return episodes, True
            if episode is not None:
                episodes.append(episode)
        return episodes, False

    def _apply_remote_sensory_bootstrap_locked(
        self,
        runtime: _SensorySourceRuntime,
        episodes: Sequence[SensoryEpisode],
        *,
        target_items: int,
    ) -> int:
        if len(runtime.buffered_episodes) > 0 or runtime.exhausted:
            return 0
        applied = list(episodes[: max(1, int(target_items))])
        if not applied:
            return 0
        runtime.buffered_episodes.extend(applied)
        now = datetime.now(timezone.utc).isoformat()
        self._commit_prefetched_sensory_runtime_locked(
            {
                "runtime": runtime,
                "cycles": runtime.cycles_completed,
                "exhausted": runtime.exhausted,
                "new_stream": None,
                "served_items": 0,
                "queue_hit": False,
                "prefetch_items": int(len(applied)),
                "prefetch_duration_ms": 0.0,
                "prefetch_at": now,
                "prefetch_error": None,
                "warm_trigger": "remote_bootstrap",
            }
        )
        self._update_sensory_runtime_cache_locked(runtime)
        self._record_brain_event_locked(
            {
                "type": "remote_sensory_bootstrap_applied",
                "timestamp": now,
                "source_name": runtime.name,
                "item_count": int(len(applied)),
            }
        )
        self._runtime_state.mark_mutated()
        return int(len(applied))

    def _promote_ready_remote_brain_items_locked(self) -> int:
        ingestion = self._brain_config.get("ingestion") or {}
        target_tokens = max(1, int(ingestion.get("queue_target_tokens", self._brain_config.get("tick_tokens", DEFAULT_BRAIN_TICK_TOKENS))))
        repeat = bool(self._brain_config.get("repeat_sources", True))
        promoted_total = 0
        for runtime in self._brain_source_runtimes:
            if (
                not self._source_spec_uses_live_remote(runtime.spec)
                or not self._stream_supports_ready_reads(runtime.stream)
                or len(runtime.buffered_patterns) >= target_tokens
                or runtime.exhausted
            ):
                continue
            cycles = runtime.cycles_completed
            exhausted = runtime.exhausted
            new_stream = None
            prefetch_error: str | None = None
            promoted = 0
            started = time.perf_counter()
            while len(runtime.buffered_patterns) < target_tokens and not exhausted:
                try:
                    runtime.buffered_patterns.append(self._next_stream_item(runtime.stream, timeout=0.0))
                    promoted += 1
                except TimeoutError:
                    break
                except StopIteration:
                    if repeat:
                        cycles += 1
                        rebuilt = _source_stream_builder(self)(
                            runtime.spec,
                            self._encoder,
                            self._trainer.config.window_size,
                        )
                        runtime.stream = rebuilt
                        new_stream = rebuilt
                        exhausted = False
                        try:
                            runtime.buffered_patterns.append(self._next_stream_item(runtime.stream, timeout=0.0))
                            promoted += 1
                        except TimeoutError:
                            break
                        except StopIteration:
                            exhausted = True
                            break
                    else:
                        exhausted = True
                        break
                except Exception as exc:
                    prefetch_error = str(exc)
                    break
            if promoted > 0 or new_stream is not None or exhausted != runtime.exhausted or prefetch_error is not None:
                duration_ms = float((time.perf_counter() - started) * 1000.0)
                self._commit_collected_runtime_locked(
                    {
                        "runtime": runtime,
                        "cycles": cycles,
                        "exhausted": exhausted,
                        "new_stream": new_stream,
                        "served_tokens": 0,
                        "queue_hit": False,
                        "prefetch_tokens": int(promoted),
                        "prefetch_duration_ms": duration_ms if promoted > 0 or prefetch_error is not None else None,
                        "prefetch_at": datetime.now(timezone.utc).isoformat() if promoted > 0 or prefetch_error is not None else None,
                        "prefetch_error": prefetch_error,
                        "warm_trigger": "remote_promotion",
                    }
                )
                self._runtime_state.mark_mutated()
            promoted_total += promoted
        return promoted_total

    def _promote_ready_remote_sensory_items_locked(self) -> int:
        target_items = self._sensory_queue_target_items_locked()
        repeat_sources = bool((self._brain_config.get("sensory") or {}).get("repeat_sources", True))
        promoted_total = 0
        for runtime in self._sensory_source_runtimes:
            if (
                not self._sensory_spec_uses_live_remote(runtime.spec)
                or not self._stream_supports_ready_reads(runtime.stream)
                or len(runtime.buffered_episodes) >= target_items
                or runtime.exhausted
            ):
                continue
            cycles = runtime.cycles_completed
            exhausted = runtime.exhausted
            new_stream = None
            prefetch_error: str | None = None
            promoted = 0
            started = time.perf_counter()
            while len(runtime.buffered_episodes) < target_items and not exhausted:
                try:
                    runtime.buffered_episodes.append(self._next_stream_item(runtime.stream, timeout=0.0))
                    promoted += 1
                except TimeoutError:
                    break
                except StopIteration:
                    if repeat_sources:
                        cycles += 1
                        rebuilt = type(self)._build_sensory_stream_from_spec(
                            runtime.spec,
                            visual_dim=int(getattr(self._trainer.config, "cross_modal_dim_visual", 64)),
                            audio_dim=int(getattr(self._trainer.config, "cross_modal_dim_audio", 64)),
                            device=self._trainer.model.device,
                        )
                        runtime.stream = rebuilt
                        new_stream = rebuilt
                        exhausted = False
                        try:
                            runtime.buffered_episodes.append(self._next_stream_item(runtime.stream, timeout=0.0))
                            promoted += 1
                        except TimeoutError:
                            break
                        except StopIteration:
                            exhausted = True
                            break
                    else:
                        exhausted = True
                        break
                except Exception as exc:
                    prefetch_error = str(exc)
                    break
            if promoted > 0 or new_stream is not None or exhausted != runtime.exhausted or prefetch_error is not None:
                duration_ms = float((time.perf_counter() - started) * 1000.0)
                self._commit_prefetched_sensory_runtime_locked(
                    {
                        "runtime": runtime,
                        "cycles": cycles,
                        "exhausted": exhausted,
                        "new_stream": new_stream,
                        "served_items": 0,
                        "queue_hit": False,
                        "prefetch_items": int(promoted),
                        "prefetch_duration_ms": duration_ms if promoted > 0 or prefetch_error is not None else None,
                        "prefetch_at": datetime.now(timezone.utc).isoformat() if promoted > 0 or prefetch_error is not None else None,
                        "prefetch_error": prefetch_error,
                        "warm_trigger": "remote_promotion",
                    }
                )
                self._runtime_state.mark_mutated()
            promoted_total += promoted
        return promoted_total

    def _remote_warm_promotion_loop(self) -> None:
        while True:
            stop_requested = False
            completed = False
            promoted_text = 0
            promoted_sensory = 0
            initial_ready_text = 0
            initial_ready_sensory = 0
            text_bootstrap_candidates: list[tuple[_BrainSourceRuntime, dict[str, Any], int]] = []
            sensory_bootstrap_candidates: list[tuple[_SensorySourceRuntime, dict[str, Any], int, int, torch.device]] = []
            with self._lock:
                stop_event = self._remote_warm_promotion_stop_event
                stop_requested = bool(stop_event is not None and stop_event.is_set())
                if not stop_requested:
                    initial_ready_text = self._promote_ready_remote_brain_items_locked()
                    initial_ready_sensory = self._promote_ready_remote_sensory_items_locked()
                    if not self._remote_warm_promotion_text_needed_locked() and not self._remote_warm_promotion_sensory_needed_locked():
                        self._record_remote_warm_promotion_completed_locked()
                        self._remote_warm_promotion_running = False
                        self._remote_warm_promotion_thread = None
                        self._remote_warm_promotion_stop_event = None
                        return

            wait_deadline = None
            if not stop_requested and initial_ready_text <= 0 and initial_ready_sensory <= 0:
                wait_deadline = time.perf_counter() + float(
                    min(_remote_promotion_bootstrap_grace_seconds(), _remote_bootstrap_budget_seconds())
                )
            while not stop_requested and wait_deadline is not None and time.perf_counter() < wait_deadline:
                time.sleep(float(_remote_prewarm_poll_seconds()))
                with self._lock:
                    stop_event = self._remote_warm_promotion_stop_event
                    stop_requested = bool(stop_event is not None and stop_event.is_set())
                    if stop_requested:
                        break
                    initial_ready_text += self._promote_ready_remote_brain_items_locked()
                    initial_ready_sensory += self._promote_ready_remote_sensory_items_locked()
                    if initial_ready_text > 0 or initial_ready_sensory > 0:
                        break
                    if not self._remote_warm_promotion_text_needed_locked() and not self._remote_warm_promotion_sensory_needed_locked():
                        self._record_remote_warm_promotion_completed_locked()
                        self._remote_warm_promotion_running = False
                        self._remote_warm_promotion_thread = None
                        self._remote_warm_promotion_stop_event = None
                        return

            with self._lock:
                stop_event = self._remote_warm_promotion_stop_event
                stop_requested = stop_requested or bool(stop_event is not None and stop_event.is_set())
                if not stop_requested:
                    initial_ready_text += self._promote_ready_remote_brain_items_locked()
                    initial_ready_sensory += self._promote_ready_remote_sensory_items_locked()
                    if not self._remote_warm_promotion_text_needed_locked() and not self._remote_warm_promotion_sensory_needed_locked():
                        self._record_remote_warm_promotion_completed_locked()
                        self._remote_warm_promotion_running = False
                        self._remote_warm_promotion_thread = None
                        self._remote_warm_promotion_stop_event = None
                        return
                    text_bootstrap_candidates = self._remote_text_bootstrap_candidates_locked()
                    sensory_bootstrap_candidates = self._remote_sensory_bootstrap_candidates_locked()

            text_bootstrap_promoted = 0
            for runtime, spec, target_tokens in text_bootstrap_candidates:
                deadline_perf = time.perf_counter() + float(_remote_bootstrap_budget_seconds())
                texts, bootstrap_timed_out = self._fetch_remote_text_bootstrap_rows(spec, deadline_perf=deadline_perf)
                if stop_requested:
                    break
                with self._lock:
                    current_stop_event = self._remote_warm_promotion_stop_event
                    if current_stop_event is not None and current_stop_event.is_set():
                        stop_requested = True
                        break
                    if any(current is runtime for current in self._brain_source_runtimes):
                        if bootstrap_timed_out:
                            self._record_brain_event_locked(
                                {
                                    "type": "remote_text_bootstrap_timed_out",
                                    "timestamp": datetime.now(timezone.utc).isoformat(),
                                    "source_name": runtime.name,
                                    "budget_seconds": float(_remote_bootstrap_budget_seconds()),
                                }
                            )
                        text_bootstrap_promoted += self._apply_remote_text_bootstrap_locked(
                            runtime,
                            texts,
                            target_tokens=target_tokens,
                        )

            sensory_bootstrap_promoted = 0
            for runtime, spec, visual_dim, audio_dim, device in sensory_bootstrap_candidates:
                deadline_perf = time.perf_counter() + float(_remote_bootstrap_budget_seconds())
                episodes, bootstrap_timed_out = self._fetch_remote_sensory_bootstrap_episodes(
                    spec,
                    visual_dim=visual_dim,
                    audio_dim=audio_dim,
                    device=device,
                    deadline_perf=deadline_perf,
                )
                if stop_requested:
                    break
                with self._lock:
                    current_stop_event = self._remote_warm_promotion_stop_event
                    if current_stop_event is not None and current_stop_event.is_set():
                        stop_requested = True
                        break
                    if any(current is runtime for current in self._sensory_source_runtimes):
                        if bootstrap_timed_out:
                            self._record_brain_event_locked(
                                {
                                    "type": "remote_sensory_bootstrap_timed_out",
                                    "timestamp": datetime.now(timezone.utc).isoformat(),
                                    "source_name": runtime.name,
                                    "budget_seconds": float(_remote_bootstrap_budget_seconds()),
                                    "item_count": int(len(episodes)),
                                }
                            )
                        sensory_bootstrap_promoted += self._apply_remote_sensory_bootstrap_locked(
                            runtime,
                            episodes,
                            target_items=self._sensory_queue_target_items_locked(),
                        )

            with self._lock:
                stop_event = self._remote_warm_promotion_stop_event
                stop_requested = stop_requested or bool(stop_event is not None and stop_event.is_set())
                if not stop_requested:
                    promoted_text = initial_ready_text + self._promote_ready_remote_brain_items_locked() + text_bootstrap_promoted
                    promoted_sensory = initial_ready_sensory + self._promote_ready_remote_sensory_items_locked() + sensory_bootstrap_promoted
                    completed = not self._remote_warm_promotion_text_needed_locked() and not self._remote_warm_promotion_sensory_needed_locked()
                    if completed:
                        self._record_remote_warm_promotion_completed_locked()
                if stop_requested or completed:
                    self._remote_warm_promotion_running = False
                    self._remote_warm_promotion_thread = None
                    self._remote_warm_promotion_stop_event = None
                    return
            if promoted_text <= 0 and promoted_sensory <= 0:
                time.sleep(float(_remote_prewarm_poll_seconds()))

    def _request_ingestion_prewarm_stop(self) -> Thread | None:
        with self._lock:
            thread = self._ingestion_prewarm_thread if self._ingestion_prewarm_thread is not None and self._ingestion_prewarm_thread.is_alive() else None
            stop_event = self._ingestion_prewarm_stop_event
            if stop_event is not None:
                stop_event.set()
            self._ingestion_prewarm_running = False
            return thread

    def _join_ingestion_prewarm_thread(self, thread: Thread | None, *, timeout: float = 5.0) -> bool:
        if thread is None:
            with self._lock:
                if self._ingestion_prewarm_thread is not None and not self._ingestion_prewarm_thread.is_alive():
                    self._ingestion_prewarm_thread = None
                    self._ingestion_prewarm_stop_event = None
            return True
        thread.join(timeout=timeout)
        with self._lock:
            if self._ingestion_prewarm_thread is thread and not thread.is_alive():
                self._ingestion_prewarm_thread = None
                self._ingestion_prewarm_stop_event = None
        return not thread.is_alive()

    def _start_ingestion_prewarm_locked(self, *, trigger: str) -> bool:
        ingestion = self._brain_config.get("ingestion") or {}
        sensory = self._brain_config.get("sensory") or {}
        text_target = (
            bool(self._brain_config.get("source_bank"))
            and bool(ingestion.get("enabled", True))
            and bool(ingestion.get("prewarm_on_startup", False))
            and self._ingestion_warm_ready_at is None
        )
        sensory_target = (
            bool(sensory)
            and bool(sensory.get("enabled", False))
            and bool(sensory.get("source_bank"))
            and bool(sensory.get("prewarm_on_startup", False))
            and self._sensory_warm_ready_at is None
        )
        if not (text_target or sensory_target):
            return False
        thread = self._ingestion_prewarm_thread
        if thread is not None and thread.is_alive():
            return False
        self._ingestion_prewarm_stop_event = Event()
        self._ingestion_prewarm_running = True
        self._ingestion_prewarm_started_at = datetime.now(timezone.utc).isoformat()
        self._ingestion_prewarm_started_perf = time.perf_counter()
        self._ingestion_prewarm_completed_at = None
        self._ingestion_prewarm_last_duration_ms = None
        self._ingestion_prewarm_last_error = None
        self._ingestion_prewarm_last_trigger = trigger
        self._ingestion_prewarm_budget_exhausted = False
        self._sensory_prewarm_budget_exhausted = False
        self._ingestion_prewarm_run_count += 1
        self._record_brain_event_locked(
            {
                "type": "ingestion_prewarm_started",
                "timestamp": self._ingestion_prewarm_started_at,
                "trigger": trigger,
                "text_prewarm": bool(text_target),
                "sensory_prewarm": bool(sensory_target),
                "queue_target_tokens": int(ingestion.get("queue_target_tokens", DEFAULT_BRAIN_TICK_TOKENS)),
                "prewarm_max_seconds": float(ingestion.get("prewarm_max_seconds", 5.0)),
                "sensory_queue_target_items": int(self._sensory_queue_target_items_locked()) if sensory_target else 0,
                "sensory_prewarm_max_seconds": float(sensory.get("prewarm_max_seconds", 5.0)) if sensory_target else 0.0,
            }
        )
        thread = Thread(target=self._ingestion_prewarm_loop, name="hecsn-ingestion-prewarm", daemon=True)
        self._ingestion_prewarm_thread = thread
        thread.start()
        return True

    def _apply_detached_brain_prewarm_locked(
        self,
        detached_runtimes: Sequence[_BrainSourceRuntime],
        prefetched: Sequence[dict[str, Any]],
        *,
        expected_epoch: int,
    ) -> bool:
        if expected_epoch != self._brain_stream_epoch:
            return False
        if len(detached_runtimes) > len(self._brain_source_runtimes):
            return False
        for idx, detached in enumerate(detached_runtimes[: len(prefetched)]):
            active = self._brain_source_runtimes[idx]
            active.buffered_patterns = deque(detached.buffered_patterns)
            self._commit_collected_runtime_locked(
                {
                    **dict(prefetched[idx]),
                    "runtime": active,
                    "cycles": detached.cycles_completed,
                    "exhausted": detached.exhausted,
                    "new_stream": detached.stream,
                    "served_tokens": 0,
                    "queue_hit": False,
                }
            )
        return True

    def _apply_detached_sensory_prewarm_locked(
        self,
        detached_runtimes: Sequence[_SensorySourceRuntime],
        prefetched: Sequence[dict[str, Any]],
        *,
        expected_epoch: int,
    ) -> bool:
        if expected_epoch != self._sensory_stream_epoch:
            return False
        if len(detached_runtimes) > len(self._sensory_source_runtimes):
            return False
        for idx, detached in enumerate(detached_runtimes[: len(prefetched)]):
            active = self._sensory_source_runtimes[idx]
            active.buffered_episodes = list(detached.buffered_episodes)
            self._commit_prefetched_sensory_runtime_locked(
                {
                    **dict(prefetched[idx]),
                    "runtime": active,
                    "cycles": detached.cycles_completed,
                    "exhausted": detached.exhausted,
                    "new_stream": detached.stream,
                    "served_items": 0,
                    "queue_hit": False,
                }
            )
        return True

    def _ingestion_prewarm_loop(self) -> None:
        with self._lock:
            stop_event = self._ingestion_prewarm_stop_event
            brain_epoch = self._brain_stream_epoch
            sensory_epoch = self._sensory_stream_epoch
            brain_specs = [deepcopy(runtime.spec) for runtime in self._brain_source_runtimes]
            repeat = bool(self._brain_config.get("repeat_sources", True))
            tick_tokens = int(self._brain_config.get("tick_tokens", DEFAULT_BRAIN_TICK_TOKENS))
            ingestion = self._brain_config.get("ingestion") or {}
            queue_target_tokens = int(ingestion.get("queue_target_tokens", tick_tokens))
            ingestion_budget_seconds = float(ingestion.get("prewarm_max_seconds", 5.0))
            encoder_ref = self._encoder
            window_size = self._trainer.config.window_size
            sensory = self._brain_config.get("sensory") or {}
            sensory_specs = [deepcopy(runtime.spec) for runtime in self._sensory_source_runtimes]
            sensory_repeat = bool(sensory.get("repeat_sources", True))
            sensory_queue_target_items = self._sensory_queue_target_items_locked()
            sensory_budget_seconds = float(sensory.get("prewarm_max_seconds", 5.0))
            visual_dim = int(getattr(self._trainer.config, "cross_modal_dim_visual", 64))
            audio_dim = int(getattr(self._trainer.config, "cross_modal_dim_audio", 64))
            device = self._trainer.model.device
            text_target = bool(ingestion.get("enabled", True)) and bool(ingestion.get("prewarm_on_startup", False))
            sensory_target = bool(sensory.get("enabled", False)) and bool(sensory.get("prewarm_on_startup", False))
            remote_text_target = bool(text_target and any(self._source_spec_uses_live_remote(spec) for spec in brain_specs))
            remote_sensory_target = bool(sensory_target and any(self._sensory_spec_uses_live_remote(spec) for spec in sensory_specs))
            text_processed_at_start = int(sum(int(runtime.tokens_processed) for runtime in self._brain_source_runtimes))
            sensory_processed_at_start = int(sum(int(runtime.episodes_processed) for runtime in self._sensory_source_runtimes))
            text_ready_at_start = int(self._ingestion_ready_source_count_locked())
            text_full_at_start = int(self._ingestion_full_queue_source_count_locked())
            sensory_ready_at_start = int(self._sensory_ready_source_count_locked())
            sensory_full_at_start = int(self._sensory_full_queue_source_count_locked())
            if remote_text_target and (text_processed_at_start > 0 or text_ready_at_start > 0):
                self._record_brain_event_locked(
                    {
                        "type": "ingestion_prewarm_skipped_after_active_execution",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "trigger": self._ingestion_prewarm_last_trigger,
                        "reason": "active_runtime_already_progressed_before_prewarm_start",
                        "ready_source_count": text_ready_at_start,
                        "full_queue_source_count": text_full_at_start,
                    }
                )
                text_target = False
                remote_text_target = False
            if remote_sensory_target and (sensory_processed_at_start > 0 or sensory_ready_at_start > 0):
                self._record_brain_event_locked(
                    {
                        "type": "sensory_prewarm_skipped_after_active_execution",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "trigger": self._ingestion_prewarm_last_trigger,
                        "reason": "active_runtime_already_progressed_before_prewarm_start",
                        "ready_source_count": sensory_ready_at_start,
                        "full_queue_source_count": sensory_full_at_start,
                    }
                )
                sensory_target = False
                remote_sensory_target = False

        if not self._wait_for_remote_prewarm_clearance(
            stop_event,
            remote_text_target=remote_text_target,
            remote_sensory_target=remote_sensory_target,
        ):
            with self._lock:
                self._record_brain_event_locked(
                    {
                        "type": "ingestion_prewarm_discarded",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "trigger": self._ingestion_prewarm_last_trigger,
                        "reason": "stop_requested",
                    }
                )
            text_target = False
            sensory_target = False

        with self._lock:
            text_processed_now = int(sum(int(runtime.tokens_processed) for runtime in self._brain_source_runtimes))
            sensory_processed_now = int(sum(int(runtime.episodes_processed) for runtime in self._sensory_source_runtimes))
            text_ready_now = int(self._ingestion_ready_source_count_locked())
            text_full_now = int(self._ingestion_full_queue_source_count_locked())
            sensory_ready_now = int(self._sensory_ready_source_count_locked())
            sensory_full_now = int(self._sensory_full_queue_source_count_locked())
            text_progressed_after_start = bool(
                text_processed_now > text_processed_at_start
                or text_ready_now > text_ready_at_start
                or text_full_now > text_full_at_start
            )
            sensory_progressed_after_start = bool(
                sensory_processed_now > sensory_processed_at_start
                or sensory_ready_now > sensory_ready_at_start
                or sensory_full_now > sensory_full_at_start
            )
            if text_target and text_progressed_after_start:
                self._record_brain_event_locked(
                    {
                        "type": "ingestion_prewarm_skipped_after_active_execution",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "trigger": self._ingestion_prewarm_last_trigger,
                        "reason": "active_runtime_progressed_while_prewarm_waited",
                        "ready_source_count": text_ready_now,
                        "full_queue_source_count": text_full_now,
                    }
                )
                text_target = False
            if sensory_target and sensory_progressed_after_start:
                self._record_brain_event_locked(
                    {
                        "type": "sensory_prewarm_skipped_after_active_execution",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "trigger": self._ingestion_prewarm_last_trigger,
                        "reason": "active_runtime_progressed_while_prewarm_waited",
                        "ready_source_count": sensory_ready_now,
                        "full_queue_source_count": sensory_full_now,
                    }
                )
                sensory_target = False

        detached_brain_runtimes = [
            _BrainSourceRuntime(
                spec=spec,
                stream=_source_stream_builder(self)(
                    spec,
                    encoder_ref,
                    window_size,
                ),
            )
            for spec in brain_specs
        ] if text_target else []
        detached_sensory_runtimes = [
            _SensorySourceRuntime(
                spec=spec,
                stream=type(self)._build_sensory_stream_from_spec(
                    spec,
                    visual_dim=visual_dim,
                    audio_dim=audio_dim,
                    device=device,
                ),
            )
            for spec in sensory_specs
        ] if sensory_target else []

        prefetched: list[dict[str, Any]] = []
        sensory_prefetched: list[dict[str, Any]] = []
        error: str | None = None
        applied_brain = False
        applied_sensory = False
        try:
            if text_target:
                prefetched = self._prefetch_source_queues_unlocked(
                    detached_brain_runtimes,
                    queue_target_tokens,
                    repeat,
                    encoder_ref,
                    window_size,
                    stop_event,
                    warm_trigger="prewarm",
                    deadline_perf=(None if ingestion_budget_seconds <= 0.0 else time.perf_counter() + ingestion_budget_seconds),
                )
            if sensory_target:
                sensory_prefetched = self._prefetch_sensory_queues_unlocked(
                    detached_sensory_runtimes,
                    sensory_queue_target_items,
                    sensory_repeat,
                    visual_dim,
                    audio_dim,
                    device,
                    stop_event,
                    warm_trigger="prewarm",
                    deadline_perf=(None if sensory_budget_seconds <= 0.0 else time.perf_counter() + sensory_budget_seconds),
                )
            with self._lock:
                self._ingestion_prewarm_budget_exhausted = any(bool(meta.get("budget_exhausted", False)) for meta in prefetched)
                self._sensory_prewarm_budget_exhausted = any(bool(meta.get("budget_exhausted", False)) for meta in sensory_prefetched)
                if text_target and prefetched:
                    applied_brain = self._apply_detached_brain_prewarm_locked(
                        detached_brain_runtimes,
                        prefetched,
                        expected_epoch=brain_epoch,
                    )
                if sensory_target and sensory_prefetched:
                    applied_sensory = self._apply_detached_sensory_prewarm_locked(
                        detached_sensory_runtimes,
                        sensory_prefetched,
                        expected_epoch=sensory_epoch,
                    )
                if text_target and prefetched and not applied_brain:
                    self._record_brain_event_locked(
                        {
                            "type": "ingestion_prewarm_discarded",
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "trigger": self._ingestion_prewarm_last_trigger,
                            "reason": "runtime_progressed",
                        }
                    )
                if sensory_target and sensory_prefetched and not applied_sensory:
                    self._record_brain_event_locked(
                        {
                            "type": "sensory_prewarm_discarded",
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "trigger": self._ingestion_prewarm_last_trigger,
                            "reason": "runtime_progressed",
                        }
                    )
                if self._ingestion_prewarm_budget_exhausted:
                    self._record_brain_event_locked(
                        {
                            "type": "ingestion_prewarm_budget_exhausted",
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "trigger": self._ingestion_prewarm_last_trigger,
                            "prewarm_max_seconds": ingestion_budget_seconds,
                            "ready_source_count": int(self._ingestion_ready_source_count_locked()),
                            "full_queue_source_count": int(self._ingestion_full_queue_source_count_locked()),
                        }
                    )
                if self._sensory_prewarm_budget_exhausted:
                    self._record_brain_event_locked(
                        {
                            "type": "sensory_prewarm_budget_exhausted",
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "trigger": self._ingestion_prewarm_last_trigger,
                            "prewarm_max_seconds": sensory_budget_seconds,
                            "ready_source_count": int(self._sensory_ready_source_count_locked()),
                            "full_queue_source_count": int(self._sensory_full_queue_source_count_locked()),
                        }
                    )
        except Exception as exc:
            error = str(exc)

        with self._lock:
            completed_at = datetime.now(timezone.utc).isoformat()
            self._ingestion_prewarm_completed_at = completed_at
            if self._ingestion_prewarm_started_perf is not None:
                self._ingestion_prewarm_last_duration_ms = float(
                    (time.perf_counter() - self._ingestion_prewarm_started_perf) * 1000.0
                )
            if error is not None:
                self._ingestion_prewarm_last_error = error
                self._record_brain_event_locked(
                    {
                        "type": "ingestion_prewarm_error",
                        "timestamp": completed_at,
                        "trigger": self._ingestion_prewarm_last_trigger,
                        "message": error,
                    }
                )
            else:
                self._record_brain_event_locked(
                    {
                        "type": "ingestion_prewarm_completed",
                        "timestamp": completed_at,
                        "trigger": self._ingestion_prewarm_last_trigger,
                        "applied_text_results": bool(applied_brain),
                        "applied_sensory_results": bool(applied_sensory),
                        "prefetch_events": int(sum(runtime.prefetch_events for runtime in self._brain_source_runtimes)),
                        "ready_source_count": int(self._ingestion_ready_source_count_locked()),
                        "sensory_prefetch_events": int(sum(runtime.prefetch_events for runtime in self._sensory_source_runtimes)),
                        "sensory_ready_source_count": int(self._sensory_ready_source_count_locked()),
                        "startup_warm_latency_ms": self._ingestion_startup_warm_latency_ms,
                        "sensory_startup_warm_latency_ms": self._sensory_startup_warm_latency_ms,
                    }
                )
            self._ingestion_prewarm_running = False
            self._ingestion_prewarm_thread = None
            self._ingestion_prewarm_stop_event = None
