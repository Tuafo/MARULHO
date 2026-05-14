"""Brain runtime orchestration helpers for Terminus.

This runtime owns source rebuilding, tick collection/training, grounded source
observation injection, source utility state/mutation, autonomy scheduling,
animation snapshots, and runtime status snapshots. Delayed-consequence
learning remains isolated in its own tracker.
"""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Event
import re
import sys
import time
from typing import Any, Callable, Iterator, Mapping, Sequence, cast

import torch

from hecsn.data.corpus_loader import StreamingCorpusLoader
from hecsn.data.pattern_loader import labeled_pattern_stream
from hecsn.semantics.grounding_text import salient_query_terms
from hecsn.service.runtime_sources import RuntimeSources, SourceType, _BrainSourceRuntime, _SensorySourceRuntime
from hecsn.training.autonomy_acquisition_runner import run_live_acquisition

DEFAULT_BRAIN_TICK_TOKENS = 512
DEFAULT_BRAIN_SLEEP_INTERVAL_SECONDS = 0.01
DEFAULT_AUTONOMY_TRIGGER_INTERVAL_TOKENS = 4096
DEFAULT_BRAIN_STOP_TIMEOUT_SECONDS = 15.0
DEFAULT_REMOTE_ACTIVE_FETCH_WAIT_SECONDS = 0.25
_BACKGROUND_SOURCE_UTILITY_INT_FIELDS = (
    "attempts",
    "selections",
    "tokens_trained_total",
)
_BACKGROUND_SOURCE_UTILITY_FLOAT_FIELDS = (
    "utility_ema",
    "semantic_alignment_ema",
    "grounding_signal_ema",
    "focus_overlap_ema",
    "grounded_outcome_ema",
    "grounded_family_summary_ema",
    "delayed_consequence_ema",
    "contradiction_decay_ema",
)
_BACKGROUND_SOURCE_UTILITY_DEFAULTS: dict[str, Any] = {
    "attempts": 0,
    "selections": 0,
    "tokens_trained_total": 0,
    "utility_ema": 0.0,
    "semantic_alignment_ema": 0.0,
    "grounding_signal_ema": 0.0,
    "focus_overlap_ema": 0.0,
    "grounded_outcome_ema": 0.0,
    "grounded_family_summary_ema": 0.0,
    "delayed_consequence_ema": 0.0,
    "contradiction_decay_ema": 0.0,
    "last_selected_at": "",
}


BRAIN_RUNTIME_STATE_FIELDS = frozenset({
    "_brain_source_runtimes",
    "_sensory_source_runtimes",
    "_brain_source_index",
    "_sensory_source_index",
    "_brain_tick_count",
    "_brain_background_tokens",
    "_brain_autonomy_tokens",
    "_brain_source_utility",
    "_brain_last_error",
    "_brain_last_acquisition_summary",
    "_brain_last_acquisition_token_count",
    "_brain_last_tick_completed_at",
    "_brain_last_tick_duration_ms",
    "_brain_last_tick_token_delta",
    "_brain_last_work_at",
    "_last_real_sensory_episode_time",
    "_last_real_sensory_episode_token_count",
    "_real_sensory_last_error",
    "_last_sensory_focus_terms",
    "_sensory_preview_history",
    "_real_sensory_episodes_completed",
    "_real_visual_accepted",
    "_real_audio_accepted",
    "_brain_stream_epoch",
    "_sensory_stream_epoch",
    "_brain_skip_next_autonomy_for_grounded_query",
})


@dataclass(frozen=True)
class BrainRuntimeDependencies:
    """Explicit production adapters for the Brain Runtime module."""

    lock: Any
    trainer: Any
    encoder: Any
    runtime_state: Any
    brain_config: Callable[[], dict[str, Any]]
    runtime_control: Callable[[], Any]
    runtime_sources: Callable[[], Any]
    delayed_consequence: Callable[[], Any]
    autonomy_planner: Callable[[], Any]
    source_focus: Callable[[], Any]
    interaction_pipeline: Callable[[], Any]
    action_executor: Callable[[], Any]
    replay_controller: Callable[[], Any]
    cortex_controller: Callable[[], Any]
    concept_store: Callable[[], Any]
    geometric_curiosity: Callable[[], Any]
    runtime_environment_summary: Callable[[], dict[str, Any]]
    huggingface_runtime_summary_locked: Callable[[], dict[str, Any]]
    ingestion_runtime_summary_locked: Callable[[], dict[str, Any]]
    multimodal_runtime_summary_locked: Callable[[], dict[str, Any]]
    sensory_runtime_summary_locked: Callable[[dict[str, Any]], dict[str, Any]]
    living_loop_snapshot_locked: Callable[..., dict[str, Any]]
    maybe_mark_ingestion_warm_locked: Callable[..., None]
    maybe_mark_sensory_warm_locked: Callable[..., None]
    observe_runtime_concepts_locked: Callable[..., dict[str, Any] | None]
    runtime_concept_callback_locked: Callable[..., Any]
    run_real_sensory_episode_locked: Callable[[], dict[str, Any] | None]
    record_brain_event_locked: Callable[[dict[str, Any]], None]
    build_brain_source_stream_locked: Callable[[dict[str, Any]], Iterator[tuple[str, Any]]]
    build_sensory_stream_locked: Callable[[dict[str, Any]], Iterator[Any]]


def _manager_symbol(name: str, fallback: Any) -> Any:
    manager_module = sys.modules.get("hecsn.service.manager")
    if manager_module is None:
        return fallback
    return getattr(manager_module, name, fallback)


def _remote_active_fetch_wait_seconds() -> float:
    return float(_manager_symbol("DEFAULT_REMOTE_ACTIVE_FETCH_WAIT_SECONDS", DEFAULT_REMOTE_ACTIVE_FETCH_WAIT_SECONDS))


def _run_live_acquisition_func() -> Any:
    return _manager_symbol("run_live_acquisition", run_live_acquisition)


def _source_stream_builder(owner: Any) -> Any:
    return type(owner)._build_source_stream_from_spec


class BrainRuntime:
    def __init__(self, dependencies: BrainRuntimeDependencies) -> None:
        self._deps = dependencies
        self._lock = dependencies.lock
        self._trainer = dependencies.trainer
        self._encoder = dependencies.encoder
        self._runtime_state = dependencies.runtime_state
        self._runtime_environment_summary = dependencies.runtime_environment_summary
        self._huggingface_runtime_summary_locked = dependencies.huggingface_runtime_summary_locked
        self._ingestion_runtime_summary_locked = dependencies.ingestion_runtime_summary_locked
        self._multimodal_runtime_summary_locked = dependencies.multimodal_runtime_summary_locked
        self._sensory_runtime_summary_locked = dependencies.sensory_runtime_summary_locked
        self._living_loop_snapshot_locked = dependencies.living_loop_snapshot_locked
        self._maybe_mark_ingestion_warm_locked = dependencies.maybe_mark_ingestion_warm_locked
        self._maybe_mark_sensory_warm_locked = dependencies.maybe_mark_sensory_warm_locked
        self._observe_runtime_concepts_locked = dependencies.observe_runtime_concepts_locked
        self._runtime_concept_callback_locked = dependencies.runtime_concept_callback_locked
        self._run_real_sensory_episode_locked = dependencies.run_real_sensory_episode_locked
        self._record_brain_event_locked = dependencies.record_brain_event_locked
        self._build_brain_source_stream_locked = dependencies.build_brain_source_stream_locked
        self._build_sensory_stream_locked = dependencies.build_sensory_stream_locked
        trainer = dependencies.trainer
        token_count = int(getattr(trainer, "token_count", 0) or 0)
        self._brain_source_runtimes: list[_BrainSourceRuntime] = []
        self._sensory_source_runtimes: list[_SensorySourceRuntime] = []
        self._brain_source_index = 0
        self._sensory_source_index = 0
        self._brain_tick_count = 0
        self._brain_background_tokens = 0
        self._brain_autonomy_tokens = 0
        self._brain_source_utility: dict[str, dict[str, Any]] = {}
        self._brain_last_error = None
        self._brain_last_acquisition_summary = None
        self._brain_last_acquisition_token_count = token_count
        self._brain_last_tick_completed_at = None
        self._brain_last_tick_duration_ms = None
        self._brain_last_tick_token_delta = 0
        self._brain_last_work_at = None
        self._last_real_sensory_episode_time = 0.0
        self._last_real_sensory_episode_token_count = token_count
        self._real_sensory_last_error = None
        self._last_sensory_focus_terms: tuple[str, ...] = ()
        self._sensory_preview_history: deque[dict[str, Any]] = deque(maxlen=8)
        self._real_sensory_episodes_completed = 0
        self._real_visual_accepted = 0
        self._real_audio_accepted = 0
        self._brain_stream_epoch = 0
        self._sensory_stream_epoch = 0
        self._brain_skip_next_autonomy_for_grounded_query = False

    def restore_runtime_state(self, state: Mapping[str, Any]) -> None:
        self._brain_source_utility = self._normalize_background_source_utility_state(
            state.get("background_source_utility")
        )
        self._brain_last_error = None
        self._brain_last_acquisition_summary = None
        trainer = getattr(self, "_trainer", None)
        self._brain_last_acquisition_token_count = int(getattr(trainer, "token_count", 0) or 0)

    def _ordered_brain_runtime_indices_locked(
        self,
        *,
        start_index: int,
        excluded_indices: set[int] | None = None,
    ) -> tuple[list[int], list[str], float]:
        excluded = excluded_indices or set()
        autonomy_planner = self._autonomy_planner
        focus_plan = autonomy_planner._autonomy_focus_plan_locked()
        focus_terms = self._background_focus_terms_locked(focus_plan=focus_plan)
        focus_pressure, _focus_pressure_details = autonomy_planner._autonomy_focus_pressure_locked(focus_plan)
        tick_tokens = int(self._brain_config.get("tick_tokens", DEFAULT_BRAIN_TICK_TOKENS))
        source_count = max(1, len(self._brain_source_runtimes))
        ranked: list[tuple[int, float, float, float, float, float, int, str]] = []
        for idx, runtime in enumerate(self._brain_source_runtimes):
            if idx in excluded or runtime.exhausted:
                continue
            score, semantic_match, fairness, readiness, effective_utility = self._brain_source_selection_score_locked(
                runtime,
                focus_terms=focus_terms,
                focus_pressure=focus_pressure,
                tick_tokens=tick_tokens,
            )
            cyclic_distance = (idx - start_index) % source_count
            ranked.append(
                (
                    idx,
                    float(score),
                    float(semantic_match),
                    float(effective_utility),
                    float(fairness),
                    float(readiness),
                    int(cyclic_distance),
                    str(runtime.name),
                )
            )
        ranked.sort(
            key=lambda item: (
                -float(item[1]),
                -float(item[2]),
                -float(item[3]),
                -float(item[4]),
                -float(item[5]),
                int(item[6]),
                item[7],
            )
        )
        return [int(item[0]) for item in ranked], focus_terms, float(focus_pressure)

    def _rebuild_brain_sources_locked(self) -> None:
        self._close_brain_sources_locked()
        self._close_sensory_sources_locked()
        self._brain_source_runtimes = [
            _BrainSourceRuntime(spec=deepcopy(spec), stream=self._build_brain_source_stream_locked(spec))
            for spec in self._brain_config.get("source_bank", [])
        ]
        for runtime in self._brain_source_runtimes:
            self._background_source_utility_entry_locked(runtime)
        sensory_config = self._brain_config.get("sensory") or {}
        self._sensory_source_runtimes = [
            _SensorySourceRuntime(spec=deepcopy(spec), stream=self._build_sensory_stream_locked(spec))
            for spec in sensory_config.get("source_bank", [])
        ]
        self._brain_source_index = 0
        self._sensory_source_index = 0
        self._brain_tick_count = 0
        self._brain_background_tokens = 0
        self._brain_autonomy_tokens = 0
        self._brain_source_utility = {
            name: value
            for name, value in self._brain_source_utility.items()
            if any(str(spec.get("name", "")).strip() == name for spec in self._brain_config.get("source_bank", []))
        }
        self._brain_last_tick_completed_at = None
        self._brain_last_tick_duration_ms = None
        self._brain_last_tick_token_delta = 0
        self._brain_last_work_at = None
        self._brain_stop_requested_at = None
        self._brain_stop_requested_reason = None
        self._brain_stop_requested_perf = None
        self._brain_stop_timed_out = False
        self._brain_last_stop_duration_ms = None
        self._real_sensory_episodes_completed = 0
        self._real_visual_accepted = 0
        self._real_audio_accepted = 0
        self._last_real_sensory_episode_time = 0.0
        self._last_real_sensory_episode_token_count = int(self._trainer.token_count)
        self._real_sensory_last_error = None
        self._last_sensory_focus_terms = ()
        self._sensory_preview_history.clear()
        self._ingestion_configured_at = (
            datetime.now(timezone.utc).isoformat() if self._brain_config.get("source_bank") else None
        )
        self._ingestion_configured_perf = time.perf_counter() if self._brain_config.get("source_bank") else None
        self._ingestion_prewarm_started_at = None
        self._ingestion_prewarm_started_perf = None
        self._ingestion_prewarm_completed_at = None
        self._ingestion_prewarm_last_duration_ms = None
        self._ingestion_prewarm_last_error = None
        self._ingestion_prewarm_run_count = 0
        self._ingestion_prewarm_last_trigger = None
        self._ingestion_prewarm_budget_exhausted = False
        self._ingestion_prewarm_running = False
        self._ingestion_prewarm_thread = None
        self._ingestion_prewarm_stop_event = None
        self._ingestion_warm_ready_at = None
        self._ingestion_startup_warm_latency_ms = None
        self._remote_warm_promotion_thread = None
        self._remote_warm_promotion_stop_event = None
        self._remote_warm_promotion_running = False
        self._remote_warm_promotion_last_trigger = None
        self._active_execution_requests = 0
        self._active_execution_idle_event.set()
        self._brain_stream_epoch += 1
        self._sensory_configured_at = (
            datetime.now(timezone.utc).isoformat() if self._sensory_source_runtimes else None
        )
        self._sensory_configured_perf = time.perf_counter() if self._sensory_source_runtimes else None
        self._sensory_prewarm_budget_exhausted = False
        self._sensory_warm_ready_at = None
        self._sensory_startup_warm_latency_ms = None
        self._sensory_stream_epoch += 1

        restored_text_sources = 0
        restored_text_tokens = 0
        for runtime in self._brain_source_runtimes:
            restored = self._restore_brain_runtime_cache_locked(runtime)
            if restored > 0:
                restored_text_sources += 1
                restored_text_tokens += restored
        if restored_text_sources > 0:
            self._maybe_mark_ingestion_warm_locked(trigger="cache_restore")
            self._record_brain_event_locked(
                {
                    "type": "ingestion_cache_restored",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "source_count": int(restored_text_sources),
                    "token_count": int(restored_text_tokens),
                }
            )

        restored_sensory_sources = 0
        restored_sensory_items = 0
        for runtime in self._sensory_source_runtimes:
            restored = self._restore_sensory_runtime_cache_locked(runtime)
            if restored > 0:
                restored_sensory_sources += 1
                restored_sensory_items += restored
        if restored_sensory_sources > 0:
            self._maybe_mark_sensory_warm_locked(trigger="cache_restore")
            self._record_brain_event_locked(
                {
                    "type": "sensory_cache_restored",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "source_count": int(restored_sensory_sources),
                    "item_count": int(restored_sensory_items),
                }
            )

    def _commit_collected_runtime_locked(self, collect_meta: dict[str, Any] | None) -> None:
        if collect_meta is None:
            return
        runtime = collect_meta["runtime"]
        runtime.cycles_completed = collect_meta["cycles"]
        runtime.exhausted = collect_meta["exhausted"]
        if collect_meta.get("new_stream") is not None:
            runtime.stream = collect_meta["new_stream"]
        runtime.last_buffer_tokens_served = int(collect_meta.get("served_tokens", 0) or 0)
        if bool(collect_meta.get("queue_hit", False)):
            runtime.queue_hits += 1
        prefetch_tokens = int(collect_meta.get("prefetch_tokens", 0) or 0)
        if prefetch_tokens > 0:
            runtime.prefetch_events += 1
            runtime.prefetched_tokens += prefetch_tokens
            runtime.last_prefetch_token_count = prefetch_tokens
            runtime.last_prefetch_at = collect_meta.get("prefetch_at")
            runtime.last_prefetch_duration_ms = collect_meta.get("prefetch_duration_ms")
        prefetch_error = collect_meta.get("prefetch_error")
        runtime.last_prefetch_error = None if prefetch_error in (None, "") else str(prefetch_error)
        self._update_brain_runtime_cache_locked(runtime)
        self._maybe_mark_ingestion_warm_locked(trigger=str(collect_meta.get("warm_trigger", "tick") or "tick"))

    def _train_chunk_in_sub_batches(
        self,
        chunk: list[tuple[str, "torch.Tensor"]],
        *,
        stop_event: Event | None,
        sub_batch_size: int,
        yield_seconds: float,
        memory_metadata: Mapping[str, Any] | None = None,
    ) -> tuple[int, Any, list[str]]:
        total_trained = 0
        last_metrics = None
        batch_size = max(1, int(sub_batch_size))
        pause_seconds = max(0.0, float(yield_seconds))
        evidence_windows: deque[str] = deque(maxlen=128)
        for i in range(0, len(chunk), batch_size):
            if stop_event is not None and stop_event.is_set():
                break
            sub = chunk[i : i + batch_size]
            with self._lock:
                for raw_window, pattern in sub:
                    last_metrics = self._trainer.train_step(
                        pattern,
                        raw_window=raw_window,
                        memory_metadata=memory_metadata,
                    )
                    raw_text = str(raw_window)
                    if raw_text:
                        evidence_windows.append(raw_text)
                if sub:
                    self._observe_runtime_concepts_locked(raw_window=sub[-1][0], metrics=last_metrics)
                total_trained += len(sub)
                self._runtime_state.mark_mutated()
            if pause_seconds > 0.0:
                time.sleep(pause_seconds)
        return total_trained, last_metrics, list(evidence_windows)

    def _prefetch_runtime_queue_unlocked(
        self,
        runtime: _BrainSourceRuntime,
        target_tokens: int,
        repeat: bool,
        encoder_ref: Any,
        window_size: int,
        stop_event: Event | None,
        *,
        warm_trigger: str,
        deadline_perf: float | None = None,
    ) -> dict[str, Any] | None:
        cycles = runtime.cycles_completed
        exhausted = runtime.exhausted
        new_stream = None
        prefetch_tokens = 0
        prefetch_duration_ms: float | None = None
        prefetch_at: str | None = None
        prefetch_error: str | None = None
        budget_exhausted = False
        if len(runtime.buffered_patterns) < target_tokens and not exhausted:
            started = time.perf_counter()
            try:
                while len(runtime.buffered_patterns) < target_tokens:
                    if stop_event is not None and stop_event.is_set():
                        return None
                    wait_timeout = None
                    if deadline_perf is not None:
                        remaining = deadline_perf - time.perf_counter()
                        if remaining <= 0.0:
                            budget_exhausted = True
                            break
                        wait_timeout = remaining
                    try:
                        runtime.buffered_patterns.append(self._next_stream_item(runtime.stream, timeout=wait_timeout))
                        prefetch_tokens += 1
                    except TimeoutError:
                        budget_exhausted = True
                        break
                    except StopIteration:
                        if repeat:
                            cycles += 1
                            rebuilt = _source_stream_builder(self)(
                                runtime.spec,
                                encoder_ref,
                                window_size,
                            )
                            runtime.stream = rebuilt
                            new_stream = rebuilt
                            exhausted = False
                            try:
                                runtime.buffered_patterns.append(self._next_stream_item(runtime.stream, timeout=wait_timeout))
                                prefetch_tokens += 1
                            except TimeoutError:
                                budget_exhausted = True
                                break
                            except StopIteration:
                                exhausted = True
                                break
                        else:
                            exhausted = True
                            break
                    if deadline_perf is not None and time.perf_counter() >= deadline_perf:
                        budget_exhausted = True
                        break
            except Exception as exc:
                if stop_event is not None and stop_event.is_set():
                    return None
                prefetch_error = str(exc)
            if prefetch_tokens > 0 or prefetch_error is not None:
                prefetch_duration_ms = float((time.perf_counter() - started) * 1000.0)
                prefetch_at = datetime.now(timezone.utc).isoformat()
        return {
            "runtime": runtime,
            "cycles": cycles,
            "exhausted": exhausted,
            "new_stream": new_stream,
            "prefetch_tokens": int(prefetch_tokens),
            "prefetch_duration_ms": prefetch_duration_ms,
            "prefetch_at": prefetch_at,
            "prefetch_error": prefetch_error,
            "budget_exhausted": bool(budget_exhausted),
            "warm_trigger": warm_trigger,
        }

    def _prefetch_source_queues_unlocked(
        self,
        runtimes: Sequence[_BrainSourceRuntime],
        target_tokens: int,
        repeat: bool,
        encoder_ref: Any,
        window_size: int,
        stop_event: Event | None,
        *,
        warm_trigger: str,
        deadline_perf: float | None = None,
    ) -> list[dict[str, Any]]:
        prefetched: list[dict[str, Any]] = []
        for runtime in runtimes:
            if stop_event is not None and stop_event.is_set():
                break
            meta = self._prefetch_runtime_queue_unlocked(
                runtime,
                target_tokens,
                repeat,
                encoder_ref,
                window_size,
                stop_event,
                warm_trigger=warm_trigger,
                deadline_perf=deadline_perf,
            )
            if meta is not None:
                prefetched.append(meta)
        return prefetched

    def _collect_chunk_unlocked(
        self,
        runtimes: list,
        ordered_indices: Sequence[int],
        tick_tokens: int,
        queue_target_tokens: int,
        repeat: bool,
        encoder_ref: Any,
        window_size: int,
        stop_event: Event | None,
    ) -> tuple[list[tuple[str, "torch.Tensor"]] | None, dict[str, Any] | None]:
        """Collect tokens from source queues WITHOUT holding self._lock.

        Remote I/O happens while filling the per-source warm queue. Deliberation
        then consumes from the in-memory queue so later ticks are less exposed to
        remote startup or transient stalls.
        """
        source_count = len(runtimes)
        target_tokens = max(int(tick_tokens), int(queue_target_tokens))
        last_meta: dict[str, Any] | None = None
        if not ordered_indices:
            ordered_indices = list(range(source_count))
        for rank, idx in enumerate(list(ordered_indices)[:source_count]):
            if stop_event is not None and stop_event.is_set():
                return None, None
            runtime = runtimes[idx]
            buffer_before = len(runtime.buffered_patterns)
            fill_target = target_tokens if buffer_before < tick_tokens else buffer_before
            deadline_perf = None
            if buffer_before < tick_tokens and self._source_spec_uses_live_remote(runtime.spec):
                deadline_perf = time.perf_counter() + float(_remote_active_fetch_wait_seconds())
            meta = self._prefetch_runtime_queue_unlocked(
                runtime,
                fill_target,
                repeat,
                encoder_ref,
                window_size,
                stop_event,
                warm_trigger="tick",
                deadline_perf=deadline_perf,
            )
            if meta is None:
                return None, None
            meta.update({"idx": idx, "source_count": source_count, "selection_rank": int(rank)})
            last_meta = meta
            if not runtime.buffered_patterns:
                continue

            served_tokens = min(int(tick_tokens), len(runtime.buffered_patterns))
            queue_hit = buffer_before >= tick_tokens and int(meta.get("prefetch_tokens", 0) or 0) == 0
            chunk = [runtime.buffered_patterns.popleft() for _ in range(served_tokens)]
            if not chunk:
                continue
            meta.update(
                {
                    "served_tokens": int(served_tokens),
                    "queue_hit": bool(queue_hit),
                }
            )
            return chunk, meta
        return None, last_meta

    @staticmethod
    def _build_source_stream_from_spec(
        spec: dict[str, Any],
        encoder: Any,
        window_size: int,
    ) -> Iterator[tuple[str, "torch.Tensor"]]:
        """Build a pattern stream without needing self._lock."""
        source_type = str(spec.get("source_type", "auto"))
        source_type = cast(SourceType, source_type)
        loader = StreamingCorpusLoader(
            source=str(spec.get("source", "")),
            source_type=source_type,
            text_field=str(spec.get("text_field", "text")),
            hf_config=spec.get("hf_config"),
        )
        stream = labeled_pattern_stream(
            loader.char_stream(),
            encoder,
            window_size,
            learn_chunking=True,
        )
        return cast(Iterator[tuple[str, torch.Tensor]], RuntimeSources._wrap_remote_stream(spec, stream, is_sensory=False))

    @staticmethod
    def _source_text_overlap(left: str, right: str) -> float:
        left_words = {word for word in re.findall(r"[a-zA-Z][a-zA-Z'-]+", left.lower()) if len(word) >= 4}
        right_words = {word for word in re.findall(r"[a-zA-Z][a-zA-Z'-]+", right.lower()) if len(word) >= 4}
        if not left_words or not right_words:
            return 0.0
        return len(left_words & right_words) / max(1.0, min(float(len(left_words)), float(len(right_words))))

    @classmethod
    def _grounded_source_sentences(
        cls,
        raw_windows: Sequence[str],
        *,
        max_sentences: int = 3,
    ) -> list[str]:
        raw_text_windows = [str(raw) for raw in raw_windows if str(raw)]
        if not raw_text_windows:
            return []

        reconstructed_raw = raw_text_windows[0]
        for window in raw_text_windows[1:]:
            max_overlap = min(len(reconstructed_raw), len(window))
            overlap = 0
            for size in range(max_overlap, 0, -1):
                if reconstructed_raw.endswith(window[:size]):
                    overlap = size
                    break
            reconstructed_raw += window[overlap:]
        reconstructed = " ".join(reconstructed_raw.split()).strip()

        normalized_windows = [" ".join(window.split()).strip() for window in raw_text_windows if " ".join(window.split()).strip()]
        selected: list[str] = []
        candidate_windows = [reconstructed, *list(reversed(normalized_windows[-24:]))]
        for window in candidate_windows:
            fragments = [fragment.strip(" ,;:") for fragment in re.split(r"(?<=[.!?])\s+", window) if fragment.strip()]
            if not fragments:
                fragments = [window]
            for fragment in fragments:
                cleaned = " ".join(fragment.split()).strip()
                words = re.findall(r"[A-Za-z][A-Za-z'-]+", cleaned)
                if len(words) < 3 or len(cleaned) < 24:
                    continue
                if any(cls._source_text_overlap(cleaned, existing) >= 0.82 for existing in selected):
                    continue
                selected.append(cleaned)
                if len(selected) >= max_sentences:
                    return selected[:max_sentences]
        if selected:
            return selected[:max_sentences]
        return [reconstructed[:320]]

    @staticmethod
    def _dedupe_grounded_topics(
        topics: Sequence[str],
        *,
        limit: int = 6,
    ) -> list[str]:
        seen: set[str] = set()
        deduped: list[str] = []
        for topic in topics:
            cleaned = " ".join(str(topic).split()).strip()
            lowered = cleaned.lower()
            if not cleaned or lowered in seen:
                continue
            seen.add(lowered)
            deduped.append(cleaned)
            if len(deduped) >= max(1, int(limit)):
                break
        return deduped

    @staticmethod
    def _grounded_observation_metadata(
        *,
        observation_kind: str,
        source_name: str,
        source_type: str,
        salience: float,
        grounding_signal: float,
        evidence_unit_count: int,
        modality: str,
        focus_terms: Sequence[str] = (),
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "grounded": True,
            "observation_kind": str(observation_kind).strip().lower(),
            "source_name": str(source_name).strip(),
            "source_type": str(source_type).strip(),
            "salience": float(max(0.0, min(1.0, salience))),
            "grounding_signal": float(max(0.0, min(1.0, grounding_signal))),
            "evidence_unit_count": int(max(1, evidence_unit_count)),
            "modality": str(modality).strip().lower() or "text",
            "focus_terms": [
                " ".join(str(term).split()).strip()
                for term in list(focus_terms)[:6]
                if " ".join(str(term).split()).strip()
            ],
        }
        if extra:
            metadata.update({str(key): deepcopy(value) for key, value in dict(extra).items()})
        return metadata

    def _inject_source_observation_locked(
        self,
        *,
        runtime: _BrainSourceRuntime,
        evidence_windows: Sequence[str],
        total_trained: int,
        last_metrics: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if self._thought_loop_actual is None:
            return {"content": "", "topics": [], "salience": 0.0}
        sentences = self._grounded_source_sentences(evidence_windows)
        excerpt = " ".join(sentences).strip()
        if not excerpt:
            return {"content": "", "topics": [], "salience": 0.0}

        concept_snapshot = self._concept_store.snapshot(limit=5)
        recent_concepts = [
            str(concept.get("label", "")).strip()
            for concept in concept_snapshot.get("top_concepts", [])[:5]
            if isinstance(concept, dict) and str(concept.get("label", "")).strip()
        ]
        topics: list[str] = []
        topics.extend(salient_query_terms(excerpt)[:6])
        topics.extend(recent_concepts[:3])
        deduped_topics = self._dedupe_grounded_topics(topics, limit=6)

        pred_error = 0.0
        surprise = 0.0
        if isinstance(last_metrics, dict):
            pred_error = max(0.0, min(1.0, float(last_metrics.get("pred_error", 0.0) or 0.0)))
            surprise = max(0.0, min(1.0, float(last_metrics.get("surprise", 0.0) or 0.0)))
        salience = max(
            0.35,
            min(
                0.95,
                0.55
                + 0.20 * pred_error
                + 0.10 * surprise
                + 0.04 * min(1.0, float(len(sentences)) / 2.0),
            ),
        )
        grounding_signal = max(
            0.35,
            min(
                1.0,
                0.52
                + 0.28 * pred_error
                + 0.14 * surprise
                + 0.06 * min(1.0, float(len(sentences)) / 2.0),
            ),
        )
        metadata = self._grounded_observation_metadata(
            observation_kind="source",
            source_name=runtime.name,
            source_type=runtime.source_type,
            salience=salience,
            grounding_signal=grounding_signal,
            evidence_unit_count=int(len(evidence_windows)),
            modality="text",
            focus_terms=deduped_topics[:4],
            extra={
                "evidence_window_count": int(len(evidence_windows)),
            },
        )
        self._thought_loop_actual.inject_observation(
            content=excerpt,
            topics=deduped_topics,
            salience=salience,
            metadata=metadata,
        )
        return {
            "content": excerpt,
            "topics": deduped_topics,
            "salience": float(salience),
            "grounding_signal": float(grounding_signal),
            "metadata": metadata,
        }

    @staticmethod
    def _normalize_background_source_utility_state(value: Any) -> dict[str, dict[str, Any]]:
        if not isinstance(value, Mapping):
            return {}

        def _safe_int(raw_value: Any) -> int:
            try:
                return max(0, int(raw_value))
            except (TypeError, ValueError):
                return 0

        def _safe_float(raw_value: Any) -> float:
            try:
                return max(0.0, min(1.0, float(raw_value)))
            except (TypeError, ValueError):
                return 0.0

        normalized: dict[str, dict[str, Any]] = {}
        for raw_name, raw_entry in value.items():
            name = " ".join(str(raw_name).split()).strip()
            if not name or not isinstance(raw_entry, Mapping):
                continue
            entry = dict(_BACKGROUND_SOURCE_UTILITY_DEFAULTS)
            for field in _BACKGROUND_SOURCE_UTILITY_INT_FIELDS:
                entry[field] = _safe_int(raw_entry.get(field, 0))
            for field in _BACKGROUND_SOURCE_UTILITY_FLOAT_FIELDS:
                entry[field] = _safe_float(raw_entry.get(field, 0.0))
            entry["last_selected_at"] = " ".join(str(raw_entry.get("last_selected_at", "")).split()).strip()
            normalized[name] = entry
        return normalized

    def _background_source_utility_entry_locked(self, runtime: _BrainSourceRuntime) -> dict[str, Any]:
        name = str(runtime.name).strip()
        entry = self._brain_source_utility.setdefault(name, dict(_BACKGROUND_SOURCE_UTILITY_DEFAULTS))
        for key, value in _BACKGROUND_SOURCE_UTILITY_DEFAULTS.items():
            entry.setdefault(key, value)
        return entry

    def _background_source_utility_metrics_locked(self, runtime: _BrainSourceRuntime) -> dict[str, float]:
        entry = self._background_source_utility_entry_locked(runtime)
        return {key: float(entry.get(key, 0.0) or 0.0) for key in _BACKGROUND_SOURCE_UTILITY_FLOAT_FIELDS}

    def _update_background_source_utility_locked(
        self,
        *,
        runtime: _BrainSourceRuntime,
        grounded_observation: Mapping[str, Any] | None,
        total_trained: int,
    ) -> None:
        entry = self._background_source_utility_entry_locked(runtime)
        autonomy_planner = self._autonomy_planner
        focus_plan = autonomy_planner._autonomy_focus_plan_locked()
        focus_terms = self._background_focus_terms_locked(focus_plan=focus_plan)
        semantic_alignment = max(0.0, min(1.0, float(runtime.last_semantic_match)))
        grounding_signal = 0.0
        if isinstance(grounded_observation, Mapping):
            grounding_signal = max(0.0, min(1.0, float(grounded_observation.get("grounding_signal", 0.0) or 0.0)))
        focus_overlap = 0.0
        background_focus_overlap = getattr(self, "_background_focus_overlap_locked", None)
        if callable(background_focus_overlap):
            focus_overlap_raw = background_focus_overlap(focus_terms, grounded_observation)
            focus_overlap = max(0.0, min(1.0, float(cast(Any, focus_overlap_raw))))
        token_fraction = min(
            1.0,
            float(max(0, int(total_trained))) / float(max(1, int(self._brain_config.get("tick_tokens", DEFAULT_BRAIN_TICK_TOKENS)))),
        )
        utility_sample = max(
            0.0,
            min(
                1.0,
                0.50 * semantic_alignment
                + 0.20 * grounding_signal
                + 0.20 * focus_overlap
                + 0.10 * token_fraction,
            ),
        )

        entry["attempts"] = int(entry.get("attempts", 0)) + 1
        entry["selections"] = int(entry.get("selections", 0)) + 1
        entry["tokens_trained_total"] = int(entry.get("tokens_trained_total", 0)) + max(0, int(total_trained))
        alpha = 0.30
        for key, sample in (
            ("utility_ema", utility_sample),
            ("semantic_alignment_ema", semantic_alignment),
            ("grounding_signal_ema", grounding_signal),
            ("focus_overlap_ema", focus_overlap),
        ):
            previous = max(0.0, min(1.0, float(entry.get(key, 0.0) or 0.0)))
            entry[key] = float(sample if int(entry["selections"]) <= 1 else (1.0 - alpha) * previous + alpha * float(sample))
        entry["last_selected_at"] = datetime.now(timezone.utc).isoformat()
        self._runtime_state.mark_mutated()

    def _finalize_tick_locked(
        self,
        tick_started: float,
        source_info: dict[str, Any] | None,
        total_trained: int,
        last_metrics: Any,
        evidence_windows: Sequence[str],
    ) -> dict[str, Any]:
        """Update counters after training, run multimodal + autonomy. Under lock."""
        token_count_before = int(self._trainer.token_count) - total_trained
        token_count_after = int(self._trainer.token_count)

        # Update source runtime counters
        source_summary: dict[str, Any]
        if source_info is not None and total_trained > 0:
            runtime = source_info["runtime"]
            idx = source_info["idx"]
            source_count = source_info["source_count"]
            runtime.tokens_processed += total_trained
            runtime.tick_visits += 1
            runtime.last_tokens_trained = int(total_trained)
            runtime.last_activity_at = datetime.now(timezone.utc).isoformat()
            self._brain_background_tokens += total_trained
            self._brain_tick_count += 1
            self._brain_source_index = (idx + 1) % source_count
            self._runtime_state.mark_mutated()
            source_summary = {
                "did_work": True,
                "source_name": runtime.name,
                "source_type": runtime.source_type,
                "source_index": int(idx),
                "tokens_trained": int(total_trained),
                "cycles_completed": int(runtime.cycles_completed),
                "exhausted": bool(runtime.exhausted),
                "buffered_tokens_remaining": int(len(runtime.buffered_patterns)),
                "prefetch_events": int(runtime.prefetch_events),
                "queue_hits": int(runtime.queue_hits),
                "last_metrics": last_metrics,
            }
        else:
            source_summary = {"did_work": False, "reason": "no_tokens"}

        autonomy_summary = self._run_brain_autonomy_locked()
        cortex_work = bool(source_summary.get("did_work")) or autonomy_summary is not None

        if cortex_work and self._thought_loop_actual is not None:
            try:
                surprise = self._trainer.model.surprise
                self._thought_loop_actual.inject_surprise(
                    dopamine=float(surprise.dopamine),
                    serotonin=float(surprise.serotonin),
                    norepinephrine=float(surprise.norepinephrine),
                    acetylcholine=float(surprise.acetylcholine),
                )
            except Exception:
                pass

        if source_info is not None and total_trained > 0:
            try:
                source_runtime = cast(_BrainSourceRuntime, source_info["runtime"])
                grounded_observation = self._inject_source_observation_locked(
                    runtime=source_runtime,
                    evidence_windows=evidence_windows,
                    total_trained=total_trained,
                    last_metrics=cast(dict[str, Any] | None, last_metrics),
                )
                if grounded_observation.get("content"):
                    source_summary["grounded_observation"] = grounded_observation
                self._update_background_source_utility_locked(
                    runtime=source_runtime,
                    grounded_observation=cast(Mapping[str, Any] | None, grounded_observation),
                    total_trained=total_trained,
                )
            except Exception:
                pass

        sensory_summary = self._run_real_sensory_episode_locked()
        token_count_after = int(self._trainer.token_count)
        multimodal_summary = self._multimodal_runtime_summary_locked() if sensory_summary is not None else None
        did_work = bool(source_summary.get("did_work")) or autonomy_summary is not None or sensory_summary is not None

        completed_at = datetime.now(timezone.utc).isoformat()
        token_delta = int(token_count_after - token_count_before)
        summary = {
            "type": "tick",
            "did_work": did_work,
            "timestamp": completed_at,
            "source": source_summary,
            "multimodal": multimodal_summary,
            "autonomy": autonomy_summary,
            "tick_duration_ms": float((time.perf_counter() - tick_started) * 1000.0),
            "token_delta": int(token_delta),
        }
        self._brain_last_tick_completed_at = completed_at
        self._brain_last_tick_duration_ms = float(summary["tick_duration_ms"])
        self._brain_last_tick_token_delta = int(token_delta)
        if did_work:
            self._brain_last_work_at = completed_at
        self._record_brain_event_locked(summary)
        return summary

    def _brain_tick_idle_locked(self, tick_started: float, source_meta: dict[str, Any] | None = None) -> dict[str, Any]:
        """Handle a tick where no source tokens were available."""
        if not self._brain_config.get("source_bank"):
            summary = {
                "type": "tick",
                "did_work": False,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "reason": "unconfigured",
            }
            self._brain_last_tick_completed_at = str(summary["timestamp"])
            self._brain_last_tick_duration_ms = float((time.perf_counter() - tick_started) * 1000.0)
            self._brain_last_tick_token_delta = 0
            self._record_brain_event_locked(summary)
            return summary

        source_summary: dict[str, Any] = {"did_work": False, "reason": "sources_exhausted"}
        if source_meta is not None:
            runtime = source_meta.get("runtime")
            if runtime is not None:
                source_summary.update(
                    {
                        "source_name": getattr(runtime, "name", "source"),
                        "source_type": getattr(runtime, "source_type", "auto"),
                        "source_index": int(source_meta.get("idx", 0) or 0),
                    }
                )
            if bool(source_meta.get("budget_exhausted", False)):
                source_summary["reason"] = "warming_remote_source"
                self._start_remote_warm_promotion_locked(trigger="tick")

        autonomy_summary = self._run_brain_autonomy_locked()
        sensory_summary = self._run_real_sensory_episode_locked()
        multimodal_summary = self._multimodal_runtime_summary_locked() if sensory_summary is not None else None
        did_work = autonomy_summary is not None or sensory_summary is not None
        completed_at = datetime.now(timezone.utc).isoformat()
        summary = {
            "type": "tick",
            "did_work": did_work,
            "timestamp": completed_at,
            "source": source_summary,
            "multimodal": multimodal_summary,
            "autonomy": autonomy_summary,
            "tick_duration_ms": float((time.perf_counter() - tick_started) * 1000.0),
            "token_delta": int(
                (0 if autonomy_summary is None else int(autonomy_summary.get("tokens_trained_total", 0) or 0))
                + (0 if sensory_summary is None else int(sensory_summary.get("steps_trained", 0) or 0))
            ),
        }
        self._brain_last_tick_completed_at = completed_at
        self._brain_last_tick_duration_ms = float(summary["tick_duration_ms"])
        self._brain_last_tick_token_delta = int(summary["token_delta"])
        if did_work:
            self._brain_last_work_at = completed_at
        self._record_brain_event_locked(summary)
        return summary

    def _run_brain_autonomy_locked(self) -> dict[str, Any] | None:
        autonomy = self._brain_config.get("autonomy")
        if not autonomy or not bool(autonomy.get("enabled", False)):
            return None
        token_delta = int(self._trainer.token_count) - int(self._brain_last_acquisition_token_count)
        autonomy_planner = self._autonomy_planner
        focus_plan = autonomy_planner._autonomy_focus_plan_locked()
        adaptive_learning = autonomy_planner._adaptive_autonomy_settings_locked(autonomy, focus_plan)
        trigger_interval = int(
            adaptive_learning.get("effective_trigger_interval_tokens", autonomy.get("trigger_interval_tokens", DEFAULT_AUTONOMY_TRIGGER_INTERVAL_TOKENS))
        )

        # Curiosity-based trigger: allow early acquisition when gap score exceeds threshold
        curiosity_gap_threshold = float(autonomy.get("curiosity_gap_threshold", 0.0))
        curiosity_cooldown = int(autonomy.get("curiosity_cooldown_tokens", max(1, trigger_interval // 2)))
        curiosity_triggered = False
        trigger_reason = "interval"

        if curiosity_gap_threshold > 0.0 and token_delta >= curiosity_cooldown:
            abstraction = getattr(self._trainer.model, "abstraction_layer", None)
            if abstraction is not None:
                gaps = abstraction.curiosity_gaps(top_n=1)
                max_gap = float(gaps[0]["gap_score"]) if gaps else 0.0
                if max_gap >= curiosity_gap_threshold:
                    curiosity_triggered = True
                    trigger_reason = "curiosity_gap"

        if not curiosity_triggered and token_delta < trigger_interval:
            return None
        if self._interaction_pipeline.consume_skip_next_autonomy_for_grounded_query():
            self._brain_last_acquisition_summary = None
            return None
        candidate_specs = autonomy_planner._autonomy_candidate_specs_locked(
            candidate_bank=list(autonomy.get("candidate_bank", [])),
            focus_plan=focus_plan,
        )
        shortlist_size, shortlist_gap_weight, shortlist_affinity_weight = autonomy_planner._autonomy_shortlist_settings_locked(
            candidate_bank=candidate_specs,
            config=autonomy,
            focus_plan=focus_plan,
        )
        curriculum_before = deepcopy(autonomy.get("provider_curriculum"))
        result = _run_live_acquisition_func()(
            trainer=self._trainer,
            encoder=self._encoder,
            candidate_bank_specs=candidate_specs,
            candidate_train_tokens=int(autonomy.get("candidate_train_tokens", 768)),
            probe_tokens=int(autonomy.get("probe_tokens", 96)),
            acquisition_tokens=int(adaptive_learning.get("effective_acquisition_tokens", autonomy.get("acquisition_tokens", 512))),
            acquisition_slots=int(adaptive_learning.get("effective_acquisition_slots", autonomy.get("acquisition_slots", 1))),
            gap_exploration_bonus=float(autonomy.get("gap_exploration_bonus", 0.03)),
            gap_ambiguity_weight=float(autonomy.get("gap_ambiguity_weight", 0.4)),
            gap_switch_weight=float(autonomy.get("gap_switch_weight", 0.2)),
            gap_margin_reference=float(autonomy.get("gap_margin_reference", 0.12)),
            coverage_balance_penalty=float(autonomy.get("coverage_balance_penalty", 0.2)),
            gap_focus_margin=float(autonomy.get("gap_focus_margin", 0.05)),
            policy_name=str(autonomy.get("policy", "active")),
            scout_commit_tokens=int(autonomy.get("scout_commit_tokens", 0)),
            scout_top_k=int(autonomy.get("scout_top_k", 1)),
            semantic_shortlist_size=shortlist_size,
            semantic_shortlist_gap_weight=shortlist_gap_weight,
            semantic_shortlist_affinity_weight=shortlist_affinity_weight,
            semantic_plan=focus_plan,
            on_train_step=self._runtime_concept_callback_locked(),
        )
        autonomy_planner._update_provider_curriculum_locked(
            autonomy=autonomy,
            result=result,
            candidate_specs=candidate_specs,
            focus_plan=focus_plan,
        )
        tokens_trained_total = int(result.get("tokens_trained_total", 0) or 0)
        self._brain_last_acquisition_token_count = int(self._trainer.token_count)
        self._brain_autonomy_tokens += tokens_trained_total
        if curriculum_before != autonomy.get("provider_curriculum"):
            self._runtime_state.mark_mutated()
        if tokens_trained_total > 0:
            self._runtime_state.mark_mutated()
        summary = {
            "executed_at": datetime.now(timezone.utc).isoformat(),
            "trigger_reason": trigger_reason,
            "policy": str(result.get("policy", autonomy.get("policy", "active"))),
            "tokens_trained_total": int(tokens_trained_total),
            "acquired_sources": list(result.get("acquired_sources", [])),
            "stopped_early": bool(result.get("stopped_early", False)),
            "final_mean_candidate_gap": result.get("final_mean_candidate_gap"),
            "final_max_candidate_gap": result.get("final_max_candidate_gap"),
            "stop_reason": result.get("stop_reason"),
            "focus_plan": deepcopy(result.get("semantic_plan")),
            "recent_query_gap_count": int(len(self._interaction_pipeline.recent_query_gaps())),
            "adaptive_learning": deepcopy(adaptive_learning),
            "provider_curriculum": deepcopy(autonomy_planner._provider_curriculum_snapshot_locked(autonomy, focus_plan)),
        }
        self._brain_last_acquisition_summary = summary
        return summary

    def _animation_snapshot_locked(self) -> dict[str, Any]:
        """Lightweight snapshot for UI animation: active column, spike counts, layer state."""
        model = self._trainer.model
        competitive = model.competitive
        n_columns = int(competitive.n_columns)
        winner = self._trainer.last_winner
        activations = competitive.thresholds.detach().cpu().tolist()
        spike_counts = competitive.spike_counts.detach().cpu().tolist() if hasattr(competitive, "spike_counts") else [0] * n_columns
        cross_modal_state = None
        if model.cross_modal is not None:
            cross_modal_state = {
                "visual_confidence": float(model.cross_modal.visual_confidence.mean().item()),
                "audio_confidence": float(model.cross_modal.audio_confidence.mean().item()),
            }
        context_tau = None
        if model.context_layer is not None and hasattr(model.context_layer, "log_tau"):
            context_tau = getattr(torch, "exp")(model.context_layer.log_tau).detach().cpu().tolist()

        # Binding layer summary
        binding_state = None
        binding = getattr(model, "binding", None)
        if binding is not None:
            binding_state = {
                "n_binding_neurons": int(binding.n_binding),
                "mean_weight": float(binding.W.detach().abs().mean().item()),
            }

        # Abstraction layer summary
        abstraction_state = None
        abstraction = getattr(model, "abstraction", None)
        if abstraction is not None:
            abstraction_state = {
                "curiosity": float(abstraction.curiosity.item()) if hasattr(abstraction, "curiosity") else 0.0,
                "n_abstract": int(abstraction.n_abstract) if hasattr(abstraction, "n_abstract") else 0,
            }

        # STDP layer summary
        stdp_state = None
        stdp = getattr(model, "stdp", None)
        if stdp is not None:
            stdp_state = {
                "mean_weight": float(stdp.weights.detach().abs().mean().item()) if hasattr(stdp, "weights") else 0.0,
            }

        return {
            "n_columns": n_columns,
            "winner_id": None if winner is None else int(winner),
            "activations": activations,
            "spike_counts": spike_counts,
            "cross_modal": cross_modal_state,
            "context_tau": context_tau,
            "binding": binding_state,
            "abstraction": abstraction_state,
            "stdp": stdp_state,
            "memory_fill": float(model.memory_store.summary_stats().get("fill_fraction", 0.0)),
        }

    def _brain_runtime_snapshot_locked(self, *, include_replay_dataset_summary: bool = True) -> dict[str, Any]:
        self._remerge_converged_delayed_consequence_families_locked()
        self._split_divergent_delayed_consequence_families_locked()
        self._compact_delayed_consequence_records_locked()
        self._cool_delayed_consequence_records_locked()
        autonomy = self._brain_config.get("autonomy")
        exhausted_source_count = sum(1 for runtime in self._brain_source_runtimes if runtime.exhausted)
        background_focus_plan: Mapping[str, Any] | None = None
        background_focus_terms: list[str] = []
        background_focus_pressure = 0.0
        next_source_name = None
        background_selection_order: list[str] = []
        if len(self._brain_source_runtimes) == 1:
            next_source_name = self._brain_source_runtimes[0].name
            background_selection_order = [self._brain_source_runtimes[0].name]
        elif len(self._brain_source_runtimes) > 1:
            autonomy_planner = self._autonomy_planner
            background_focus_plan = autonomy_planner._autonomy_focus_plan_locked()
            background_focus_terms = self._background_focus_terms_locked(focus_plan=background_focus_plan)
            background_focus_pressure, _background_focus_pressure_details = autonomy_planner._autonomy_focus_pressure_locked(background_focus_plan)
            ordered_indices, _focus_terms, _focus_pressure = self._ordered_brain_runtime_indices_locked(
                start_index=self._brain_source_index,
            )
            if ordered_indices:
                next_source_name = self._brain_source_runtimes[ordered_indices[0]].name
                background_selection_order = [
                    self._brain_source_runtimes[idx].name
                    for idx in ordered_indices
                    if 0 <= idx < len(self._brain_source_runtimes)
                ]
        autonomy_tokens_until_trigger = None
        autonomy_trigger_ready = None
        autonomy_candidate_names = None
        autonomy_focus_plan = background_focus_plan
        autonomy_provider_curriculum = None
        autonomy_adaptive_learning = None
        sensory = self._brain_config.get("sensory")
        sensory_tokens_until_trigger = None
        sensory_trigger_ready = None
        if autonomy is not None:
            autonomy_planner = self._autonomy_planner
            if autonomy_focus_plan is None:
                autonomy_focus_plan = autonomy_planner._autonomy_focus_plan_locked()
            autonomy_provider_curriculum = autonomy_planner._provider_curriculum_snapshot_locked(autonomy, autonomy_focus_plan)
            autonomy_adaptive_learning = autonomy_planner._adaptive_autonomy_settings_locked(autonomy, autonomy_focus_plan)
            trigger_interval = int(
                autonomy_adaptive_learning.get(
                    "effective_trigger_interval_tokens",
                    autonomy.get("trigger_interval_tokens", DEFAULT_AUTONOMY_TRIGGER_INTERVAL_TOKENS),
                )
            )
            token_delta = int(self._trainer.token_count) - int(self._brain_last_acquisition_token_count)
            autonomy_tokens_until_trigger = int(max(0, trigger_interval - token_delta))
            autonomy_trigger_ready = bool(token_delta >= trigger_interval)
            autonomy_candidate_names = [
                str(item.get("name", "candidate"))
                for item in list(autonomy.get("candidate_bank", []))
            ]
        if sensory is not None:
            sensory_trigger_interval = int(sensory.get("episode_interval_tokens", 2048))
            sensory_token_delta = int(self._trainer.token_count) - int(self._last_real_sensory_episode_token_count)
            sensory_tokens_until_trigger = int(max(0, sensory_trigger_interval - sensory_token_delta))
            sensory_trigger_ready = bool(sensory_token_delta >= sensory_trigger_interval)
        if self._remote_warm_promotion_running and not self._remote_warm_promotion_text_needed_locked() and not self._remote_warm_promotion_sensory_needed_locked():
            self._record_remote_warm_promotion_completed_locked()
        thread_alive = self._brain_runtime_active_locked()
        runtime_state_snapshot = self._runtime_state.snapshot()
        total_text_learning_tokens = int(self._brain_background_tokens + self._brain_autonomy_tokens)
        autonomy_share_of_text_learning = float(
            0.0
            if total_text_learning_tokens <= 0
            else float(self._brain_autonomy_tokens) / float(total_text_learning_tokens)
        )
        background_share_of_text_learning = float(
            0.0
            if total_text_learning_tokens <= 0
            else max(0.0, 1.0 - autonomy_share_of_text_learning)
        )
        cortex_snapshot = self._thought_loop_actual.snapshot() if self._thought_loop_actual is not None else self._cortex_unavailable_snapshot()
        living_loop_snapshot = self._living_loop_snapshot_locked(
            cortex_snapshot=cortex_snapshot,
            include_replay_dataset_summary=include_replay_dataset_summary,
        )
        return {
            "configured": bool(self._brain_config.get("source_bank")),
            "running": bool(thread_alive),
            "running_since": self._brain_running_since,
            "shutdown": {
                "stop_requested": self._brain_stop_requested_at is not None,
                "stop_requested_at": self._brain_stop_requested_at,
                "stop_reason": self._brain_stop_requested_reason,
                "stop_timed_out": bool(self._brain_stop_timed_out),
                "last_stop_duration_ms": self._brain_last_stop_duration_ms,
                "join_timeout_seconds": float(DEFAULT_BRAIN_STOP_TIMEOUT_SECONDS),
                "thread_alive": bool(thread_alive),
            },
            "environment": self._runtime_environment_summary(),
            "action_loop": self._action_loop_summary_locked(),
            "tick_tokens": int(self._brain_config.get("tick_tokens", DEFAULT_BRAIN_TICK_TOKENS)),
            "sleep_interval_seconds": float(
                self._brain_config.get("sleep_interval_seconds", DEFAULT_BRAIN_SLEEP_INTERVAL_SECONDS)
            ),
            "repeat_sources": bool(self._brain_config.get("repeat_sources", True)),
            "source_count": int(len(self._brain_source_runtimes)),
            "exhausted_source_count": int(exhausted_source_count),
            "next_source_name": next_source_name,
            "background_tokens_processed": int(self._brain_background_tokens),
            "tick_count": int(self._brain_tick_count),
            "last_tick_completed_at": self._brain_last_tick_completed_at,
            "last_tick_duration_ms": self._brain_last_tick_duration_ms,
            "last_tick_token_delta": int(self._brain_last_tick_token_delta),
            "tokens_per_second": float(
                (self._brain_last_tick_token_delta / (self._brain_last_tick_duration_ms / 1000.0))
                if self._brain_last_tick_duration_ms and self._brain_last_tick_duration_ms > 0
                else 0.0
            ),
            "last_work_at": self._brain_last_work_at,
            "last_error": self._brain_last_error,
            "last_event": runtime_state_snapshot["last_event"],
            "recent_events": runtime_state_snapshot["recent_events"],
            "source_bank": deepcopy(self._brain_config.get("source_bank", [])),
            "ingestion": self._ingestion_runtime_summary_locked(),
            "background_source_routing": {
                "mode": "focus_aware_allocation",
                "utility_mode": "provenance_grounded_family_summary_lineage_reconvergent_divergence_split_trajectory_sensitive_compacted_age_sensitive_consequence_calibration",
                "evidence_provenance_credit": True,
                "delayed_consequence_tracking": self._delayed_consequence_summary_locked(limit=4),
                "focus_terms": list(background_focus_terms),
                "focus_pressure": float(background_focus_pressure),
                "selection_order": list(background_selection_order),
            },
            "text_learning_balance": {
                "background_tokens_processed": int(self._brain_background_tokens),
                "autonomy_tokens_processed": int(self._brain_autonomy_tokens),
                "total_text_learning_tokens": int(total_text_learning_tokens),
                "autonomy_share_of_text_learning": float(autonomy_share_of_text_learning),
                "background_share_of_text_learning": float(background_share_of_text_learning),
            },
            "source_progress": [
                {
                    **self._background_source_utility_metrics_locked(runtime),
                    "name": runtime.name,
                    "source_type": runtime.source_type,
                    "tokens_processed": int(runtime.tokens_processed),
                    "tick_visits": int(runtime.tick_visits),
                    "last_tokens_trained": int(runtime.last_tokens_trained),
                    "last_activity_at": runtime.last_activity_at,
                    "cycles_completed": int(runtime.cycles_completed),
                    "exhausted": bool(runtime.exhausted),
                    "buffered_tokens": int(len(runtime.buffered_patterns)),
                    "buffer_fill_fraction": float(
                        0.0
                        if int((self._brain_config.get("ingestion") or {}).get("queue_target_tokens", 0) or 0) <= 0
                        else float(len(runtime.buffered_patterns))
                        / float(int((self._brain_config.get("ingestion") or {}).get("queue_target_tokens", 0) or 1))
                    ),
                    "prefetch_events": int(runtime.prefetch_events),
                    "prefetched_tokens": int(runtime.prefetched_tokens),
                    "last_prefetch_token_count": int(runtime.last_prefetch_token_count),
                    "last_prefetch_at": runtime.last_prefetch_at,
                    "last_prefetch_duration_ms": runtime.last_prefetch_duration_ms,
                    "last_prefetch_error": runtime.last_prefetch_error,
                    "queue_hits": int(runtime.queue_hits),
                    "last_buffer_tokens_served": int(runtime.last_buffer_tokens_served),
                    "last_semantic_match": float(runtime.last_semantic_match),
                    "last_selection_score": float(runtime.last_selection_score),
                    "last_fairness_score": float(runtime.last_fairness_score),
                    "last_buffer_readiness": float(runtime.last_buffer_readiness),
                    "last_utility_score": float(runtime.last_utility_score),
                    "share_of_background_tokens": float(
                        0.0
                        if self._brain_background_tokens <= 0
                        else float(runtime.tokens_processed) / float(self._brain_background_tokens)
                    ),
                }
                for runtime in self._brain_source_runtimes
            ],
            "huggingface": self._huggingface_runtime_summary_locked(),
            "sensory": None
            if sensory is None
            else (
                lambda snapshot: {
                    **snapshot,
                    "tokens_until_trigger": sensory_tokens_until_trigger,
                    "trigger_ready": sensory_trigger_ready,
                }
            )(self._sensory_runtime_summary_locked(sensory)),
            "autonomy": None
            if autonomy is None
            else {
                "enabled": bool(autonomy.get("enabled", False)),
                "policy": str(autonomy.get("policy", "active")),
                "candidate_count": int(len(autonomy.get("candidate_bank", []))),
                "candidate_bank": deepcopy(list(autonomy.get("candidate_bank", []))),
                "candidate_names": autonomy_candidate_names,
                "trigger_interval_tokens": int(
                    autonomy.get("trigger_interval_tokens", DEFAULT_AUTONOMY_TRIGGER_INTERVAL_TOKENS)
                ),
                "tokens_processed": int(self._brain_autonomy_tokens),
                "share_of_text_learning_tokens": float(autonomy_share_of_text_learning),
                "tokens_until_trigger": autonomy_tokens_until_trigger,
                "trigger_ready": autonomy_trigger_ready,
                "recent_query_gaps": self._interaction_pipeline.recent_query_gaps(),
                "focus_plan": deepcopy(autonomy_focus_plan),
                "adaptive_learning": deepcopy(autonomy_adaptive_learning),
                "provider_curriculum": deepcopy(autonomy_provider_curriculum),
                "delayed_consequence_tracking": self._delayed_consequence_summary_locked(limit=4),
                "last_acquisition_token_count": int(self._brain_last_acquisition_token_count),
                "last_acquisition_summary": deepcopy(self._brain_last_acquisition_summary),
                "geometric_curiosity": deepcopy(self._geometric_curiosity.summary()),
            },
            "multimodal": self._multimodal_runtime_summary_locked(),
            "living_loop": living_loop_snapshot,
            "cortex": cortex_snapshot,
        }

    def _brain_persisted_state_locked(self) -> dict[str, Any]:
        runtime_state_snapshot = self._runtime_state.snapshot()
        return {
            "source_bank": deepcopy(self._brain_config.get("source_bank", [])),
            "tick_tokens": int(self._brain_config.get("tick_tokens", DEFAULT_BRAIN_TICK_TOKENS)),
            "sleep_interval_seconds": float(
                self._brain_config.get("sleep_interval_seconds", DEFAULT_BRAIN_SLEEP_INTERVAL_SECONDS)
            ),
            "repeat_sources": bool(self._brain_config.get("repeat_sources", True)),
            "autonomy": deepcopy(self._brain_config.get("autonomy")),
            "sensory": deepcopy(self._brain_config.get("sensory")),
            "ingestion": deepcopy(self._brain_config.get("ingestion")),
            "background_source_utility": deepcopy(self._brain_source_utility),
            "delayed_consequence_records": [deepcopy(item) for item in list(self._delayed_consequence_records)],
            "delayed_consequence_cooled_total": int(self._delayed_consequence_cooled_total),
            "delayed_consequence_retired_total": int(self._delayed_consequence_retired_total),
            "delayed_consequence_compacted_total": int(self._delayed_consequence_compacted_total),
            "delayed_consequence_split_total": int(self._delayed_consequence_split_total),
            "delayed_consequence_remerged_total": int(self._delayed_consequence_remerged_total),
            "recent_query_gaps": self._interaction_pipeline.recent_query_gaps(),
            "action_history": [deepcopy(item) for item in list(self._action_history)],
            "runtime_episode_traces": self._interaction_pipeline.runtime_episode_traces(),
            "replay_sample_history": [deepcopy(item) for item in list(self._replay_sample_history)],
            "last_event": runtime_state_snapshot["last_event"],
            "recent_events": runtime_state_snapshot["recent_events"],
            "geometric_curiosity": self._geometric_curiosity.state_dict(),
        }


def _install_dependency_property(name: str, provider_name: str) -> None:
    def _provider(self: BrainRuntime) -> Any:
        provider = getattr(self._deps, provider_name)
        return provider()

    def _get(self: BrainRuntime) -> Any:
        return getattr(_provider(self), name)

    def _set(self: BrainRuntime, value: Any) -> None:
        setattr(_provider(self), name, value)

    setattr(BrainRuntime, name, property(_get, _set))


def _install_dependency_object_property(name: str, provider_name: str) -> None:
    def _get(self: BrainRuntime) -> Any:
        return getattr(self._deps, provider_name)()

    setattr(BrainRuntime, name, property(_get))


def _install_dependency_alias_property(name: str, provider_name: str, target_name: str) -> None:
    def _provider(self: BrainRuntime) -> Any:
        provider = getattr(self._deps, provider_name)
        return provider()

    def _get(self: BrainRuntime) -> Any:
        return getattr(_provider(self), target_name)

    def _set(self: BrainRuntime, value: Any) -> None:
        setattr(_provider(self), target_name, value)

    setattr(BrainRuntime, name, property(_get, _set))


for _name in (
    "_active_execution_idle_event",
    "_active_execution_requests",
    "_brain_last_stop_duration_ms",
    "_brain_running_since",
    "_brain_runtime_active_locked",
    "_brain_stop_requested_at",
    "_brain_stop_requested_perf",
    "_brain_stop_requested_reason",
    "_brain_stop_timed_out",
    "_ingestion_configured_at",
    "_ingestion_configured_perf",
    "_ingestion_prewarm_budget_exhausted",
    "_ingestion_prewarm_completed_at",
    "_ingestion_prewarm_last_duration_ms",
    "_ingestion_prewarm_last_error",
    "_ingestion_prewarm_last_trigger",
    "_ingestion_prewarm_run_count",
    "_ingestion_prewarm_running",
    "_ingestion_prewarm_started_at",
    "_ingestion_prewarm_started_perf",
    "_ingestion_prewarm_stop_event",
    "_ingestion_prewarm_thread",
    "_ingestion_startup_warm_latency_ms",
    "_ingestion_warm_ready_at",
    "_record_remote_warm_promotion_completed_locked",
    "_remote_warm_promotion_last_trigger",
    "_remote_warm_promotion_running",
    "_remote_warm_promotion_sensory_needed_locked",
    "_remote_warm_promotion_stop_event",
    "_remote_warm_promotion_text_needed_locked",
    "_remote_warm_promotion_thread",
    "_sensory_configured_at",
    "_sensory_configured_perf",
    "_sensory_prewarm_budget_exhausted",
    "_sensory_startup_warm_latency_ms",
    "_sensory_warm_ready_at",
    "_start_remote_warm_promotion_locked",
):
    _install_dependency_property(_name, "runtime_control")

for _name in (
    "_close_brain_sources_locked",
    "_close_sensory_sources_locked",
    "_next_stream_item",
    "_restore_brain_runtime_cache_locked",
    "_restore_sensory_runtime_cache_locked",
    "_source_spec_uses_live_remote",
    "_update_brain_runtime_cache_locked",
):
    _install_dependency_property(_name, "runtime_sources")

for _name in (
    "_compact_delayed_consequence_records_locked",
    "_cool_delayed_consequence_records_locked",
    "_delayed_consequence_compacted_total",
    "_delayed_consequence_cooled_total",
    "_delayed_consequence_records",
    "_delayed_consequence_remerged_total",
    "_delayed_consequence_retired_total",
    "_delayed_consequence_split_total",
    "_delayed_consequence_summary_locked",
    "_remerge_converged_delayed_consequence_families_locked",
    "_split_divergent_delayed_consequence_families_locked",
):
    _install_dependency_property(_name, "delayed_consequence")

for _name in (
    "_background_focus_terms_locked",
    "_brain_source_selection_score_locked",
):
    _install_dependency_property(_name, "source_focus")

_install_dependency_object_property("_autonomy_planner", "autonomy_planner")
_install_dependency_object_property("_brain_config", "brain_config")
_install_dependency_object_property("_interaction_pipeline", "interaction_pipeline")
_install_dependency_alias_property("_action_history", "action_executor", "history")
_install_dependency_property("_action_loop_summary_locked", "action_executor")
_install_dependency_alias_property("_replay_sample_history", "replay_controller", "history")

for _name in (
    "_cortex_unavailable_snapshot",
    "_thought_loop_actual",
):
    _install_dependency_property(_name, "cortex_controller")

_install_dependency_object_property("_concept_store", "concept_store")
_install_dependency_object_property("_geometric_curiosity", "geometric_curiosity")


BrainRuntimeMixin = BrainRuntime
