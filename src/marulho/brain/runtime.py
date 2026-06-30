from __future__ import annotations

from collections import deque
from copy import deepcopy
from pathlib import Path
from threading import Event, RLock, Thread, current_thread
import time
from typing import Any, Mapping, Sequence

import torch

from marulho.brain.checkpoints import (
    load_brain_trainer_checkpoint,
    save_brain_trainer_checkpoint,
)
from marulho.brain.generation import LocalTransitionReadout
from marulho.brain.sources import BrainPattern, BrainSourceBuffer
from marulho.brain.trace import BrainTrace
from marulho.config.model_config import MarulhoConfig
from marulho.training.model import MarulhoModel
from marulho.training.trainer import MarulhoTrainer


DEFAULT_BRAIN_TICK_TOKENS = 128
DEFAULT_BRAIN_QUANTUM_TOKENS = 16


class MarulhoBrain:
    """Main checkpoint-backed MARULHO brain loop.

    The brain owns source buffering, tick/learn orchestration, local readout,
    replay, compact growth/prune hooks, trace telemetry, and save/restore.
    CUDA/native graph execution remains inside MarulhoTrainer.
    """

    surface = "marulho_brain_runtime.v1"

    def __init__(
        self,
        trainer: MarulhoTrainer,
        *,
        metadata: Mapping[str, Any] | None = None,
        checkpoint_path: str | Path | None = None,
        trace_limit: int = 64,
        source_buffer_limit: int = 8192,
    ) -> None:
        self.trainer = trainer
        self.encoder = trainer.encoder
        self.metadata: dict[str, Any] = dict(metadata or {})
        self.checkpoint_path = None if checkpoint_path is None else Path(checkpoint_path)
        self._source_buffer = BrainSourceBuffer(max_items=source_buffer_limit)
        self._trace_history: deque[dict[str, Any]] = deque(maxlen=max(1, int(trace_limit)))
        self._readout = LocalTransitionReadout()
        self._step = 0
        self._last_state_key: int | None = None
        self._last_source: str | None = None
        self._last_generation: dict[str, Any] = self._empty_generation()
        self._lifecycle_lock = RLock()
        self._loop_stop = Event()
        self._loop_thread: Thread | None = None
        self._loop_started_at: float | None = None
        self._loop_tick_count = 0
        self._loop_last_error: str | None = None
        self._loop_tick_tokens = DEFAULT_BRAIN_TICK_TOKENS
        self._loop_quantum_tokens = DEFAULT_BRAIN_QUANTUM_TOKENS
        self._loop_interval_seconds = 0.25
        self._loop_allow_sleep_maintenance = False
        self._loop_source: str | None = None
        self._restore_brain_state(self.metadata.get("brain_state"))

    @classmethod
    def fresh(
        cls,
        config: MarulhoConfig,
        *,
        metadata: Mapping[str, Any] | None = None,
        checkpoint_path: str | Path | None = None,
    ) -> "MarulhoBrain":
        model = MarulhoModel(config)
        trainer = MarulhoTrainer(model, config)
        return cls(trainer, metadata=metadata, checkpoint_path=checkpoint_path)

    @classmethod
    def load(
        cls,
        checkpoint_path: str | Path,
        *,
        trace_limit: int = 64,
    ) -> "MarulhoBrain":
        trainer, metadata = load_brain_trainer_checkpoint(checkpoint_path)
        return cls(
            trainer,
            metadata=metadata,
            checkpoint_path=checkpoint_path,
            trace_limit=trace_limit,
        )

    @classmethod
    def from_trainer(
        cls,
        trainer: MarulhoTrainer,
        *,
        metadata: Mapping[str, Any] | None = None,
        checkpoint_path: str | Path | None = None,
    ) -> "MarulhoBrain":
        return cls(trainer, metadata=metadata, checkpoint_path=checkpoint_path)

    def rebind_runtime(
        self,
        trainer: MarulhoTrainer,
        *,
        metadata: Mapping[str, Any] | None = None,
        checkpoint_path: str | Path | None = None,
    ) -> None:
        self.trainer = trainer
        self.encoder = trainer.encoder
        self.metadata = dict(metadata or {})
        self.checkpoint_path = None if checkpoint_path is None else Path(checkpoint_path)
        self._restore_brain_state(self.metadata.get("brain_state"))

    def feed(self, text: str, *, source: str = "operator", learn: bool = False) -> dict[str, Any]:
        patterns = self._patterns_from_text(text, source=source, learn=True)
        added = self._source_buffer.extend(patterns)
        self._last_source = str(source)
        result = {
            "surface": "marulho_brain_feed.v1",
            "accepted_tokens": int(added),
            "queued_tokens": len(self._source_buffer),
            "source": str(source),
            "learned_immediately": bool(learn),
            "readout": self._readout_summary(),
        }
        if learn and added:
            result["tick"] = self.tick(tokens=added, source=source)
        return result

    def tick(
        self,
        *,
        tokens: int = DEFAULT_BRAIN_TICK_TOKENS,
        quantum_tokens: int = DEFAULT_BRAIN_QUANTUM_TOKENS,
        source: str | None = None,
        allow_sleep_maintenance: bool = False,
    ) -> dict[str, Any]:
        limit = max(0, int(tokens))
        quantum_tokens = max(1, int(quantum_tokens))
        batch = self._source_buffer.pop_batch(limit)
        before = self._sample_text(max_tokens=32)
        started = time.perf_counter()
        trained = 0
        result: dict[str, Any] = {
            "surface": "marulho_brain_tick.v1",
            "requested_tokens": limit,
            "trained_tokens": 0,
            "queued_tokens": len(self._source_buffer),
        }
        if batch:
            raw_windows = [item.raw_window for item in batch]
            patterns = [item.pattern for item in batch]
            train_report = self.trainer.train_text_sequence(
                patterns,
                raw_windows=raw_windows,
                memory_metadata={"brain_source": source or self._last_source or "unknown"},
                quantum_tokens=quantum_tokens,
                metric_indices=(),
                allow_sleep_maintenance=bool(allow_sleep_maintenance),
            )
            trained = int(train_report.get("trained", len(patterns)) or 0)
            state_keys = self._state_keys_for_patterns(batch)
            self._record_sequence_transitions(state_keys, raw_windows)
            if state_keys:
                self._last_state_key = int(state_keys[-1])
            result["trainer"] = self._compact_train_report(train_report)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        after = self._sample_text(max_tokens=32)
        trace = self._append_trace(
            BrainTrace(
                step=self._step + 1,
                event="tick",
                device=self._device_string(),
                token_count=int(self.trainer.token_count),
                queued_tokens=len(self._source_buffer),
                tick_tokens=len(batch),
                trained_tokens=trained,
                elapsed_ms=elapsed_ms,
                throughput_tokens_per_sec=(
                    1000.0 * trained / elapsed_ms if elapsed_ms > 0.0 else 0.0
                ),
                executor=self._executor_name(),
                route_vote_mode=str(self.trainer.config.predictive_route_vote_mode),
                cuda_available=bool(torch.cuda.is_available()),
                generation_before=str(before.get("text", "")),
                generation_after=str(after.get("text", "")),
                checkpoint_path=self._checkpoint_path_string(),
                source=source or self._last_source,
            )
        )
        result.update(
            {
                "trained_tokens": trained,
                "quantum_tokens": int(quantum_tokens),
                "queued_tokens": len(self._source_buffer),
                "generation_before": before,
                "generation_after": after,
                "trace": trace,
            }
        )
        return result

    def generate(self, prompt: str | None = None, *, max_tokens: int = 64) -> dict[str, Any]:
        start_key = self._state_key_for_prompt(prompt)
        generation = self._readout.generate(start_key, max_tokens=max_tokens)
        generation.update(
            {
                "prompt": prompt,
                "device": self._device_string(),
                "token_count": int(self.trainer.token_count),
                "checkpoint_path": self._checkpoint_path_string(),
                "not_external_llm": True,
                "not_thought_loop": True,
                "not_cortex": True,
            }
        )
        self._last_generation = dict(generation)
        self._append_trace(
            BrainTrace(
                step=self._step + 1,
                event="generate",
                device=self._device_string(),
                token_count=int(self.trainer.token_count),
                queued_tokens=len(self._source_buffer),
                executor=self._executor_name(),
                route_vote_mode=str(self.trainer.config.predictive_route_vote_mode),
                cuda_available=bool(torch.cuda.is_available()),
                generation_after=str(generation.get("text", "")),
                checkpoint_path=self._checkpoint_path_string(),
                source=self._last_source,
            )
        )
        return generation

    def replay(self, *, window: str = "recent_surprise", cycles: int = 1) -> dict[str, Any]:
        before = self._sample_text(max_tokens=32)
        updates = int(
            self.trainer.run_sleep_maintenance(
                mode="deep" if window != "micro" else "micro",
                cycles=max(1, int(cycles)),
            )
        )
        after = self._sample_text(max_tokens=32)
        trace = self._append_trace(
            BrainTrace(
                step=self._step + 1,
                event="replay",
                device=self._device_string(),
                token_count=int(self.trainer.token_count),
                queued_tokens=len(self._source_buffer),
                executor=self._executor_name(),
                route_vote_mode=str(self.trainer.config.predictive_route_vote_mode),
                cuda_available=bool(torch.cuda.is_available()),
                generation_before=str(before.get("text", "")),
                generation_after=str(after.get("text", "")),
                replay_updates=updates,
                checkpoint_path=self._checkpoint_path_string(),
                source=self._last_source,
            )
        )
        return {
            "surface": "marulho_brain_replay.v1",
            "window": str(window),
            "cycles": max(1, int(cycles)),
            "replay_updates": updates,
            "generation_before": before,
            "generation_after": after,
            "trace": trace,
        }

    def grow_prune(self, *, budget: str = "small") -> dict[str, Any]:
        window_tokens = 64 if budget == "small" else 256
        strength = 0.20 if budget == "small" else 0.35
        before_anchors = len(getattr(self.trainer, "column_anchors", {}) or {})
        captured = int(
            self.trainer.capture_recent_memory_anchors(
                window_tokens=window_tokens,
                strength=strength,
            )
        )
        after_anchors = len(getattr(self.trainer, "column_anchors", {}) or {})
        growth_events = max(0, after_anchors - before_anchors, captured)
        trace = self._append_trace(
            BrainTrace(
                step=self._step + 1,
                event="grow_prune",
                device=self._device_string(),
                token_count=int(self.trainer.token_count),
                queued_tokens=len(self._source_buffer),
                executor=self._executor_name(),
                route_vote_mode=str(self.trainer.config.predictive_route_vote_mode),
                cuda_available=bool(torch.cuda.is_available()),
                growth_events=growth_events,
                prune_events=0,
                checkpoint_path=self._checkpoint_path_string(),
                source=self._last_source,
                note="bounded recent memory anchor capture; pruning remains trainer-owned",
            )
        )
        return {
            "surface": "marulho_brain_growth_prune.v1",
            "budget": str(budget),
            "window_tokens": int(window_tokens),
            "strength": float(strength),
            "growth_events": int(growth_events),
            "prune_events": 0,
            "anchors_before": int(before_anchors),
            "anchors_after": int(after_anchors),
            "state_changed": bool(growth_events),
            "trace": trace,
        }

    def save(self, checkpoint_path: str | Path | None = None) -> dict[str, Any]:
        target = Path(checkpoint_path) if checkpoint_path is not None else self.checkpoint_path
        if target is None:
            raise ValueError("checkpoint_path is required for MarulhoBrain.save")
        metadata = self.export_metadata()
        saved_path = save_brain_trainer_checkpoint(target, self.trainer, metadata=metadata)
        self.checkpoint_path = Path(saved_path)
        trace = self._append_trace(
            BrainTrace(
                step=self._step + 1,
                event="save",
                device=self._device_string(),
                token_count=int(self.trainer.token_count),
                queued_tokens=len(self._source_buffer),
                executor=self._executor_name(),
                route_vote_mode=str(self.trainer.config.predictive_route_vote_mode),
                cuda_available=bool(torch.cuda.is_available()),
                checkpoint_path=str(saved_path),
                source=self._last_source,
            )
        )
        return {
            "surface": "marulho_brain_checkpoint_save.v1",
            "path": str(saved_path),
            "token_count": int(self.trainer.token_count),
            "trace": trace,
        }

    def start(
        self,
        *,
        tick_tokens: int = DEFAULT_BRAIN_TICK_TOKENS,
        quantum_tokens: int = DEFAULT_BRAIN_QUANTUM_TOKENS,
        interval_seconds: float = 0.25,
        source: str | None = None,
        allow_sleep_maintenance: bool = False,
    ) -> dict[str, Any]:
        tick_tokens = max(1, int(tick_tokens))
        quantum_tokens = max(1, int(quantum_tokens))
        interval_seconds = max(0.01, float(interval_seconds))
        with self._lifecycle_lock:
            running_thread = self._loop_thread
            if running_thread is not None and running_thread.is_alive():
                return {
                    "surface": "marulho_brain_loop_start.v1",
                    "started": False,
                    "already_running": True,
                    "loop": self._loop_status_locked(),
                    "brain": self.status(),
                }
            self._loop_stop.clear()
            self._loop_tick_tokens = tick_tokens
            self._loop_quantum_tokens = quantum_tokens
            self._loop_interval_seconds = interval_seconds
            self._loop_allow_sleep_maintenance = bool(allow_sleep_maintenance)
            self._loop_source = None if source is None else str(source)
            self._loop_started_at = time.time()
            self._loop_last_error = None
            thread = Thread(
                target=self._run_loop,
                name="MarulhoBrainLoop",
                daemon=True,
            )
            self._loop_thread = thread
            thread.start()
            trace = self._append_trace(
                BrainTrace(
                    step=self._step + 1,
                    event="start",
                    device=self._device_string(),
                    token_count=int(self.trainer.token_count),
                    queued_tokens=len(self._source_buffer),
                    executor=self._executor_name(),
                    route_vote_mode=str(self.trainer.config.predictive_route_vote_mode),
                    cuda_available=bool(torch.cuda.is_available()),
                    checkpoint_path=self._checkpoint_path_string(),
                    source=self._loop_source or self._last_source,
                    note="brain-owned background loop started",
                )
            )
            return {
                "surface": "marulho_brain_loop_start.v1",
                "started": True,
                "already_running": False,
                "loop": self._loop_status_locked(),
                "trace": trace,
                "brain": self.status(),
            }

    def stop(self, *, timeout_seconds: float = 2.0) -> dict[str, Any]:
        timeout_seconds = max(0.0, float(timeout_seconds))
        with self._lifecycle_lock:
            thread = self._loop_thread
            was_running = thread is not None and thread.is_alive()
            self._loop_stop.set()
        if thread is not None and thread is not current_thread():
            thread.join(timeout=timeout_seconds)
        with self._lifecycle_lock:
            still_running = thread is not None and thread.is_alive()
            if not still_running:
                self._loop_thread = None
            trace = self._append_trace(
                BrainTrace(
                    step=self._step + 1,
                    event="stop",
                    device=self._device_string(),
                    token_count=int(self.trainer.token_count),
                    queued_tokens=len(self._source_buffer),
                    executor=self._executor_name(),
                    route_vote_mode=str(self.trainer.config.predictive_route_vote_mode),
                    cuda_available=bool(torch.cuda.is_available()),
                    checkpoint_path=self._checkpoint_path_string(),
                    source=self._loop_source or self._last_source,
                    note="brain-owned background loop stop requested",
                )
            )
            return {
                "surface": "marulho_brain_loop_stop.v1",
                "was_running": bool(was_running),
                "stopped": not still_running,
                "loop": self._loop_status_locked(),
                "trace": trace,
                "brain": self.status(),
            }

    def status(self) -> dict[str, Any]:
        return {
            "surface": self.surface,
            "checkpoint_path": self._checkpoint_path_string(),
            "device": self._device_string(),
            "token_count": int(self.trainer.token_count),
            "last_winner": self.trainer.last_winner,
            "queued_tokens": len(self._source_buffer),
            "source_buffer": self._source_buffer.snapshot(),
            "executor": self._executor_name(),
            "route_vote_mode": str(self.trainer.config.predictive_route_vote_mode),
            "cuda_available": bool(torch.cuda.is_available()),
            "readout": self._readout_summary(),
            "last_generation": dict(self._last_generation),
            "last_trace": self.trace(),
            "trace_history_size": len(self._trace_history),
            "loop": self._loop_status(),
            "retired_brain_surfaces": {
                "external_llm_used": False,
                "thought_loop_used": False,
                "cortex_used": False,
            },
        }

    def trace(self) -> dict[str, Any]:
        if self._trace_history:
            return dict(self._trace_history[-1])
        return BrainTrace(
            step=int(self._step),
            event="init",
            device=self._device_string(),
            token_count=int(self.trainer.token_count),
            queued_tokens=len(self._source_buffer),
            executor=self._executor_name(),
            route_vote_mode=str(self.trainer.config.predictive_route_vote_mode),
            cuda_available=bool(torch.cuda.is_available()),
            checkpoint_path=self._checkpoint_path_string(),
            source=self._last_source,
        ).to_dict()

    def trace_history(self, *, limit: int = 20) -> list[dict[str, Any]]:
        limit = max(0, int(limit))
        if limit == 0:
            return []
        return [dict(item) for item in list(self._trace_history)[-limit:]]

    def export_metadata(self) -> dict[str, Any]:
        metadata = deepcopy(self.metadata)
        metadata["brain_state"] = self.export_state()
        return metadata

    def export_state(self) -> dict[str, Any]:
        return {
            "surface": self.surface,
            "step": int(self._step),
            "last_state_key": self._last_state_key,
            "last_source": self._last_source,
            "readout": self._readout.to_state(),
            "trace_history": self.trace_history(limit=self._trace_history.maxlen or 64),
            "source_buffer": self._source_buffer.snapshot(),
        }

    def _restore_brain_state(self, state: Any) -> None:
        if not isinstance(state, Mapping):
            return
        self._step = int(state.get("step", self._step) or 0)
        last_state_key = state.get("last_state_key")
        self._last_state_key = None if last_state_key is None else int(last_state_key)
        last_source = state.get("last_source")
        self._last_source = None if last_source is None else str(last_source)
        self._readout = LocalTransitionReadout.from_state(
            state.get("readout") if isinstance(state.get("readout"), Mapping) else None
        )
        self._trace_history.clear()
        for raw in list(state.get("trace_history") or [])[-(self._trace_history.maxlen or 64) :]:
            if isinstance(raw, Mapping):
                self._trace_history.append(dict(raw))

    def _run_loop(self) -> None:
        while not self._loop_stop.is_set():
            try:
                queued = len(self._source_buffer)
                if queued > 0:
                    self.tick(
                        tokens=min(self._loop_tick_tokens, queued),
                        quantum_tokens=self._loop_quantum_tokens,
                        source=self._loop_source or self._last_source,
                        allow_sleep_maintenance=self._loop_allow_sleep_maintenance,
                    )
                    with self._lifecycle_lock:
                        self._loop_tick_count += 1
                else:
                    self._loop_stop.wait(self._loop_interval_seconds)
            except Exception as exc:  # pragma: no cover - defensive runtime guard
                with self._lifecycle_lock:
                    self._loop_last_error = f"{type(exc).__name__}: {exc}"
                self._append_trace(
                    BrainTrace(
                        step=self._step + 1,
                        event="loop_error",
                        device=self._device_string(),
                        token_count=int(self.trainer.token_count),
                        queued_tokens=len(self._source_buffer),
                        executor=self._executor_name(),
                        route_vote_mode=str(self.trainer.config.predictive_route_vote_mode),
                        cuda_available=bool(torch.cuda.is_available()),
                        checkpoint_path=self._checkpoint_path_string(),
                        source=self._loop_source or self._last_source,
                        note=self._loop_last_error or "",
                    )
                )
                break
        with self._lifecycle_lock:
            if self._loop_thread is current_thread():
                self._loop_thread = None

    def _loop_status(self) -> dict[str, Any]:
        with self._lifecycle_lock:
            return self._loop_status_locked()

    def _loop_status_locked(self) -> dict[str, Any]:
        thread = self._loop_thread
        running = thread is not None and thread.is_alive()
        started_at = self._loop_started_at
        return {
            "surface": "marulho_brain_loop_state.v1",
            "running": bool(running),
            "owner": "MarulhoBrain",
            "tick_tokens": int(self._loop_tick_tokens),
            "quantum_tokens": int(self._loop_quantum_tokens),
            "interval_seconds": float(self._loop_interval_seconds),
            "allow_sleep_maintenance": bool(self._loop_allow_sleep_maintenance),
            "source": self._loop_source,
            "tick_count": int(self._loop_tick_count),
            "started_at": started_at,
            "uptime_seconds": (
                max(0.0, time.time() - float(started_at))
                if running and started_at is not None
                else 0.0
            ),
            "last_error": self._loop_last_error,
            "legacy_terminus_runtime_control": False,
        }


    def _patterns_from_text(
        self,
        text: str,
        *,
        source: str,
        learn: bool,
    ) -> list[BrainPattern]:
        return [
            BrainPattern(raw_window=raw_window, pattern=pattern, source=str(source))
            for raw_window, pattern in self.encoder.iter_char_patterns(
                str(text or ""),
                self.trainer.config.window_size,
                learn=bool(learn),
            )
        ]

    def _state_keys_for_patterns(self, patterns: Sequence[BrainPattern]) -> list[int]:
        keys: list[int] = []
        for item in patterns:
            try:
                keys.append(int(self.trainer.winner_for_pattern(item.pattern)))
            except Exception:
                if self.trainer.last_winner is not None:
                    keys.append(int(self.trainer.last_winner))
        return keys

    def _state_key_for_prompt(self, prompt: str | None) -> int | None:
        if prompt:
            patterns = self._patterns_from_text(prompt, source="prompt", learn=False)
            keys = self._state_keys_for_patterns(patterns)
            if keys:
                return int(keys[-1])
        if self._last_state_key is not None:
            return int(self._last_state_key)
        if self.trainer.last_winner is not None:
            return int(self.trainer.last_winner)
        return None

    def _record_sequence_transitions(
        self,
        state_keys: Sequence[int],
        raw_windows: Sequence[str],
    ) -> int:
        return self._readout.observe_sequence(state_keys, raw_windows)

    def _sample_text(self, *, max_tokens: int) -> dict[str, Any]:
        start_key = self._last_state_key
        if start_key is None and self.trainer.last_winner is not None:
            start_key = int(self.trainer.last_winner)
        return self._readout.generate(start_key, max_tokens=max_tokens)

    def _append_trace(self, trace: BrainTrace) -> dict[str, Any]:
        payload = trace.to_dict()
        self._step = max(self._step + 1, int(payload["step"]))
        payload["step"] = int(self._step)
        self._trace_history.append(payload)
        return dict(payload)

    def _readout_summary(self) -> dict[str, Any]:
        return {
            "surface": self._readout.surface,
            "observed_transition_count": self._readout.observed_transition_count,
            "transition_state_count": self._readout.state_count,
            "owned_by_marulho": True,
            "external_dependency": False,
        }

    def _compact_train_report(self, report: Mapping[str, Any]) -> dict[str, Any]:
        keys = (
            "trained",
            "quantum_count",
            "stopped",
            "sleep_maintenance_allowed",
            "fallback_train_step_count",
            "fallback_sleep_maintenance_deferred_count",
        )
        compact = {key: report.get(key) for key in keys if key in report}
        last_metrics = report.get("last_metrics")
        if isinstance(last_metrics, Mapping):
            compact["last_metrics"] = {
                key: last_metrics.get(key)
                for key in (
                    "winner",
                    "recon_error",
                    "sleep_type",
                    "sleep_replay_updates",
                    "selected_candidate_count",
                )
                if key in last_metrics
            }
        return compact

    def _executor_name(self) -> str:
        return str(getattr(self.trainer.config, "cuda_graph_sequence_executor", ""))

    def _device_string(self) -> str:
        return str(getattr(self.trainer.model, "device", "unknown"))

    def _checkpoint_path_string(self) -> str | None:
        return None if self.checkpoint_path is None else str(self.checkpoint_path)

    def _empty_generation(self) -> dict[str, Any]:
        return {
            "surface": LocalTransitionReadout.surface,
            "text": "",
            "available": False,
            "owned_by_marulho": True,
            "external_dependency": False,
            "external_llm_used": False,
            "thought_loop_used": False,
            "cortex_used": False,
        }
