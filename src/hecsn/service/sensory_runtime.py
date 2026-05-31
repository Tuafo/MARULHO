"""Sensory runtime execution helpers for Terminus.

This component owns multimodal sensory selection, prefetching, observation
injection, and sensory preview recording. It is live-runtime support only; it
does not package replay datasets or train adapters.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from threading import Event
import time
from typing import Any, Mapping, Sequence
from uuid import uuid4

import torch

from hecsn.semantics.grounding_text import salient_query_terms
from hecsn.service.operator_interaction import OperatorInteractionRuntime
from hecsn.service.runtime_sources import RuntimeSources, _SensorySourceRuntime
from hecsn.service.terminus_sensory import SensoryEpisode

DEFAULT_REMOTE_ACTIVE_FETCH_WAIT_SECONDS = 0.25


def _remote_active_fetch_wait_seconds() -> float:
    return float(DEFAULT_REMOTE_ACTIVE_FETCH_WAIT_SECONDS)


class SensoryRuntimeCore:
    def _cross_modal_confidence_means_locked(self) -> tuple[float, float]:
        cross_modal = getattr(self._trainer.model, "cross_modal", None)
        if cross_modal is None:
            return 0.0, 0.0
        try:
            visual_conf = float(cross_modal.visual_confidence.mean().item())
        except Exception:
            visual_conf = 0.0
        try:
            audio_conf = float(cross_modal.audio_confidence.mean().item())
        except Exception:
            audio_conf = 0.0
        return max(0.0, min(1.0, visual_conf)), max(0.0, min(1.0, audio_conf))

    @staticmethod
    def _sensory_runtime_modalities(adapter: str) -> tuple[bool, bool]:
        cleaned = str(adapter).strip().lower()
        if cleaned == "s1_mmalign":
            return True, False
        if cleaned == "audiocaps":
            return False, True
        return False, False

    def _sensory_focus_terms_locked(self, limit: int = 12) -> list[str]:
        phrases: list[str] = []
        recent_query_gaps = self._interaction_pipeline.recent_query_gaps()
        if recent_query_gaps:
            recent_gap = recent_query_gaps[0]
            phrases.append(str(recent_gap.get("query_text", "")))
            phrases.extend(str(term) for term in list(recent_gap.get("unsupported_terms") or [])[:4])
            phrases.extend(
                str(item.get("term", ""))
                for item in list(recent_gap.get("gap_terms") or [])[:4]
                if isinstance(item, dict)
            )
        if not phrases:
            phrases.extend(self._focus_gap_terms_locked(limit=max(4, limit // 2)))

        ordered: list[str] = []
        seen: set[str] = set()
        for phrase in phrases:
            for term in salient_query_terms(str(phrase)):
                cleaned = " ".join(str(term).split()).strip().lower()
                if len(cleaned) < 4 or cleaned in seen:
                    continue
                seen.add(cleaned)
                ordered.append(cleaned)
                if len(ordered) >= max(1, limit):
                    return ordered
        return ordered

    @staticmethod
    def _sensory_source_topic_terms(runtime: _SensorySourceRuntime) -> set[str]:
        terms: set[str] = set()
        for raw in list(runtime.spec.get("topic_terms") or []):
            for term in salient_query_terms(str(raw)):
                cleaned = " ".join(str(term).split()).strip().lower()
                if len(cleaned) >= 4:
                    terms.add(cleaned)
        metadata = runtime.spec.get("metadata")
        if isinstance(metadata, dict):
            for key in ("role", "label"):
                for term in salient_query_terms(str(metadata.get(key, ""))):
                    cleaned = " ".join(str(term).split()).strip().lower()
                    if len(cleaned) >= 4:
                        terms.add(cleaned)
        return terms

    @staticmethod
    def _sensory_episode_terms(episode: SensoryEpisode) -> set[str]:
        terms: set[str] = set()
        text_parts = [str(episode.text)]
        metadata = episode.metadata if isinstance(episode.metadata, Mapping) else {}
        for key in ("title", "caption", "categories", "summary", "label", "observation"):
            value = metadata.get(key)
            if isinstance(value, str):
                text_parts.append(value)
            elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                text_parts.extend(str(item) for item in list(value) if str(item).strip())
        for chunk in text_parts:
            for term in salient_query_terms(str(chunk)):
                cleaned = " ".join(str(term).split()).strip().lower()
                if len(cleaned) >= 4:
                    terms.add(cleaned)
        return terms

    def _sensory_episode_semantic_match_locked(
        self,
        episode: SensoryEpisode,
        focus_terms: Sequence[str] | None = None,
    ) -> float:
        normalized_focus = [
            " ".join(str(term).split()).strip().lower()
            for term in list(focus_terms or self._sensory_focus_terms_locked())
            if " ".join(str(term).split()).strip()
        ]
        episode_terms = self._sensory_episode_terms(episode)
        if not normalized_focus or not episode_terms:
            return 0.0
        focus_set = set(normalized_focus)
        overlap = len(focus_set & episode_terms) / max(1.0, min(float(len(focus_set)), float(len(episode_terms))))
        head_hits = sum(1 for term in normalized_focus[:3] if term in episode_terms)
        head_bonus = min(1.0, 0.5 * head_hits)
        combined_text = " ".join(
            part
            for part in [
                str(episode.text),
                *(str(value) for value in list((episode.metadata or {}).values()) if isinstance(value, str)),
            ]
            if part
        ).lower()
        phrase_hits = sum(1 for term in normalized_focus[:4] if term and term in combined_text)
        phrase_bonus = min(1.0, 0.34 * phrase_hits)
        return max(0.0, min(1.0, 0.55 * overlap + 0.30 * head_bonus + 0.15 * phrase_bonus))

    def _sensory_semantic_match_locked(
        self,
        runtime: _SensorySourceRuntime,
        focus_terms: Sequence[str] | None = None,
    ) -> float:
        normalized_focus = [
            " ".join(str(term).split()).strip().lower()
            for term in list(focus_terms or self._sensory_focus_terms_locked())
            if " ".join(str(term).split()).strip()
        ]
        source_terms = self._sensory_source_topic_terms(runtime)
        if not normalized_focus or not source_terms:
            return 0.0
        focus_set = set(normalized_focus)
        overlap = len(focus_set & source_terms) / max(1.0, min(float(len(focus_set)), float(len(source_terms))))
        head_hits = sum(1 for term in normalized_focus[:3] if term in source_terms)
        head_bonus = min(1.0, 0.5 * head_hits)
        return max(0.0, min(1.0, 0.65 * overlap + 0.35 * head_bonus))

    def _sensory_selection_score_locked(
        self,
        runtime: _SensorySourceRuntime,
        *,
        focus_terms: Sequence[str],
    ) -> tuple[float, float, float]:
        semantic_match = self._sensory_semantic_match_locked(runtime, focus_terms)
        modality_need = self._sensory_modality_need_locked(runtime.adapter)
        source_count = max(1, len(self._sensory_source_runtimes))
        min_episodes = min((rt.episodes_processed for rt in self._sensory_source_runtimes), default=0)
        fairness = max(
            0.0,
            min(
                1.0,
                1.0 - max(0, runtime.episodes_processed - min_episodes) / float(source_count + 1),
            ),
        )
        freshness = 1.0 if runtime.last_activity_at is None else 0.0
        score = 0.46 * semantic_match + 0.34 * modality_need + 0.12 * fairness + 0.08 * freshness
        runtime.last_semantic_match = semantic_match
        runtime.last_modality_need = modality_need
        runtime.last_selection_score = score
        return score, semantic_match, modality_need

    def _select_sensory_runtime_locked(
        self,
        excluded_indices: set[int] | None = None,
    ) -> tuple[int, _SensorySourceRuntime, float, float, float] | None:
        excluded = excluded_indices or set()
        focus_terms = self._sensory_focus_terms_locked()
        self._last_sensory_focus_terms = tuple(focus_terms)
        best: tuple[int, _SensorySourceRuntime, float, float, float] | None = None
        for idx, runtime in enumerate(self._sensory_source_runtimes):
            if idx in excluded or runtime.exhausted:
                continue
            score, semantic_match, modality_need = self._sensory_selection_score_locked(
                runtime,
                focus_terms=focus_terms,
            )
            if best is None or score > best[4] + 1e-6:
                best = (idx, runtime, semantic_match, modality_need, score)
                continue
            if best is not None and abs(score - best[4]) <= 1e-6 and runtime.episodes_processed < best[1].episodes_processed:
                best = (idx, runtime, semantic_match, modality_need, score)
        return best

    def _sensory_item_retrieval_config_locked(self) -> tuple[int, float]:
        sensory = self._brain_config.get("sensory") or {}
        lookahead = max(1, int(sensory.get("item_retrieval_lookahead", 6)))
        semantic_weight = max(0.0, min(1.0, float(sensory.get("item_retrieval_semantic_weight", 0.72))))
        return lookahead, semantic_weight

    def _prefetch_sensory_runtime_unlocked(
        self,
        runtime: _SensorySourceRuntime,
        target_items: int,
        repeat_sources: bool,
        visual_dim: int,
        audio_dim: int,
        device: torch.device,
        stop_event: Event | None,
        *,
        warm_trigger: str,
        deadline_perf: float | None = None,
    ) -> dict[str, Any] | None:
        cycles = runtime.cycles_completed
        exhausted = runtime.exhausted
        new_stream = None
        prefetched_items = 0
        prefetch_duration_ms: float | None = None
        prefetch_at: str | None = None
        prefetch_error: str | None = None
        budget_exhausted = False
        if len(runtime.buffered_episodes) < target_items and not exhausted:
            started = time.perf_counter()
            try:
                while len(runtime.buffered_episodes) < target_items:
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
                        runtime.exhausted = False
                        runtime.buffered_episodes.append(self._next_stream_item(runtime.stream, timeout=wait_timeout))
                        prefetched_items += 1
                    except TimeoutError:
                        budget_exhausted = True
                        break
                    except StopIteration:
                        if not repeat_sources:
                            exhausted = True
                            runtime.exhausted = True
                            break
                        cycles += 1
                        rebuilt = RuntimeSources._build_sensory_stream_from_spec(
                            runtime.spec,
                            visual_dim=visual_dim,
                            audio_dim=audio_dim,
                            device=device,
                        )
                        runtime.stream = rebuilt
                        new_stream = rebuilt
                        runtime.exhausted = False
                        try:
                            runtime.buffered_episodes.append(self._next_stream_item(runtime.stream, timeout=wait_timeout))
                            prefetched_items += 1
                        except TimeoutError:
                            budget_exhausted = True
                            break
                        except StopIteration:
                            exhausted = True
                            runtime.exhausted = True
                            break
                    if deadline_perf is not None and time.perf_counter() >= deadline_perf:
                        budget_exhausted = True
                        break
            except Exception as exc:
                if stop_event is not None and stop_event.is_set():
                    return None
                exhausted = True
                runtime.exhausted = True
                prefetch_error = str(exc)
            if prefetched_items > 0 or prefetch_error is not None:
                prefetch_duration_ms = float((time.perf_counter() - started) * 1000.0)
                prefetch_at = datetime.now(timezone.utc).isoformat()
        return {
            "runtime": runtime,
            "cycles": cycles,
            "exhausted": exhausted,
            "new_stream": new_stream,
            "prefetch_items": int(prefetched_items),
            "prefetch_duration_ms": prefetch_duration_ms,
            "prefetch_at": prefetch_at,
            "prefetch_error": prefetch_error,
            "budget_exhausted": bool(budget_exhausted),
            "warm_trigger": warm_trigger,
        }

    def _prefetch_sensory_queues_unlocked(
        self,
        runtimes: Sequence[_SensorySourceRuntime],
        target_items: int,
        repeat_sources: bool,
        visual_dim: int,
        audio_dim: int,
        device: torch.device,
        stop_event: Event | None,
        *,
        warm_trigger: str,
        deadline_perf: float | None = None,
    ) -> list[dict[str, Any]]:
        prefetched: list[dict[str, Any]] = []
        for runtime in runtimes:
            if stop_event is not None and stop_event.is_set():
                break
            meta = self._prefetch_sensory_runtime_unlocked(
                runtime,
                target_items,
                repeat_sources,
                visual_dim,
                audio_dim,
                device,
                stop_event,
                warm_trigger=warm_trigger,
                deadline_perf=deadline_perf,
            )
            if meta is not None:
                prefetched.append(meta)
        return prefetched

    def _commit_prefetched_sensory_runtime_locked(self, meta: dict[str, Any] | None) -> None:
        if meta is None:
            return
        runtime = meta["runtime"]
        runtime.cycles_completed = int(meta.get("cycles", runtime.cycles_completed))
        runtime.exhausted = bool(meta.get("exhausted", runtime.exhausted))
        if meta.get("new_stream") is not None:
            runtime.stream = meta["new_stream"]
        runtime.last_buffer_episodes_served = int(meta.get("served_items", 0) or 0)
        if bool(meta.get("queue_hit", False)):
            runtime.queue_hits += 1
        prefetched_items = int(meta.get("prefetch_items", 0) or 0)
        if prefetched_items > 0:
            runtime.prefetch_events += 1
            runtime.prefetched_episodes += prefetched_items
            runtime.last_prefetch_episode_count = prefetched_items
            runtime.last_prefetch_at = meta.get("prefetch_at")
            runtime.last_prefetch_duration_ms = meta.get("prefetch_duration_ms")
        prefetch_error = meta.get("prefetch_error")
        runtime.last_prefetch_error = None if prefetch_error in (None, "") else str(prefetch_error)
        if runtime.last_prefetch_error:
            self._real_sensory_last_error = runtime.last_prefetch_error
        self._update_sensory_runtime_cache_locked(runtime)
        self._maybe_mark_sensory_warm_locked(trigger=str(meta.get("warm_trigger", "sensory") or "sensory"))

    def _next_sensory_episode_locked(
        self,
        runtime: _SensorySourceRuntime,
        *,
        repeat_sources: bool,
        focus_terms: Sequence[str],
    ) -> SensoryEpisode | None:
        lookahead, semantic_weight = self._sensory_item_retrieval_config_locked()
        queue_target_items = self._sensory_queue_target_items_locked()
        visual_dim = int(getattr(self._trainer.config, "cross_modal_dim_visual", 64))
        audio_dim = int(getattr(self._trainer.config, "cross_modal_dim_audio", 64))
        buffer_before = len(runtime.buffered_episodes)
        fill_target = max(lookahead, queue_target_items) if buffer_before <= 0 else buffer_before
        deadline_perf = None
        if buffer_before <= 0 and self._sensory_spec_uses_live_remote(runtime.spec):
            deadline_perf = time.perf_counter() + float(_remote_active_fetch_wait_seconds())
        meta = self._prefetch_sensory_runtime_unlocked(
            runtime,
            fill_target,
            repeat_sources,
            visual_dim,
            audio_dim,
            self._trainer.model.device,
            None,
            warm_trigger="sensory_tick",
            deadline_perf=deadline_perf,
        )
        self._commit_prefetched_sensory_runtime_locked(meta)
        if not runtime.buffered_episodes:
            if meta is not None and bool(meta.get("budget_exhausted", False)):
                self._start_remote_warm_promotion_locked(trigger="sensory_tick")
            runtime.last_item_semantic_match = 0.0
            runtime.last_item_candidates_considered = 0
            runtime.last_item_retrieval_lookahead = lookahead
            return None

        considered = min(len(runtime.buffered_episodes), lookahead)
        best_index = 0
        best_match = self._sensory_episode_semantic_match_locked(runtime.buffered_episodes[0], focus_terms)
        best_score = semantic_weight * best_match + (1.0 - semantic_weight)
        if considered > 1:
            denom = max(1, considered - 1)
            for idx, episode in enumerate(runtime.buffered_episodes[:considered]):
                item_match = self._sensory_episode_semantic_match_locked(episode, focus_terms)
                recency = 1.0 - (idx / float(denom))
                score = semantic_weight * item_match + (1.0 - semantic_weight) * recency
                if score > best_score + 1e-6:
                    best_index = idx
                    best_match = item_match
                    best_score = score
                    continue
                if abs(score - best_score) <= 1e-6 and item_match > best_match + 1e-6:
                    best_index = idx
                    best_match = item_match
                    best_score = score

        runtime.last_item_semantic_match = float(max(0.0, min(1.0, best_match)))
        runtime.last_item_candidates_considered = int(considered)
        runtime.last_item_retrieval_lookahead = int(lookahead)
        runtime.last_buffer_episodes_served = 1
        if buffer_before > 0 and int(meta.get("prefetch_items", 0) or 0) == 0:
            runtime.queue_hits += 1
        return runtime.buffered_episodes.pop(best_index)

    def _sensory_modality_need_locked(self, adapter: str) -> float:
        sensory = self._brain_config.get("sensory") or {}
        target_confidence = float(sensory.get("modality_target_confidence", 0.70))
        visual_conf, audio_conf = self._cross_modal_confidence_means_locked()
        use_visual, use_audio = self._sensory_runtime_modalities(adapter)
        confs: list[float] = []
        if use_visual:
            confs.append(visual_conf)
        if use_audio:
            confs.append(audio_conf)
        if not confs:
            return 0.0
        mean_conf = sum(confs) / float(len(confs))
        if mean_conf >= target_confidence:
            return 0.0
        return max(0.0, min(1.0, (target_confidence - mean_conf) / max(0.1, target_confidence)))

    def _sensory_window_budget_locked(
        self,
        runtime: _SensorySourceRuntime,
        *,
        semantic_match: float | None = None,
        modality_need: float | None = None,
    ) -> int:
        sensory = self._brain_config.get("sensory") or {}
        base_windows = max(1, int(sensory.get("base_windows_per_item", 4)))
        max_windows = max(base_windows, int(sensory.get("max_windows_per_item", 10)))
        confidence_gain = max(0.0, float(sensory.get("confidence_window_gain", 3.0)))
        semantic_gain = max(0.0, float(sensory.get("semantic_window_gain", 3.0)))
        need = runtime.last_modality_need if modality_need is None else max(0.0, min(1.0, float(modality_need)))
        semantic = runtime.last_semantic_match if semantic_match is None else max(0.0, min(1.0, float(semantic_match)))
        bonus = int(round(confidence_gain * need + semantic_gain * semantic))
        return max(base_windows, min(max_windows, base_windows + bonus))

    def _inject_sensory_observation_locked(
        self,
        *,
        runtime: _SensorySourceRuntime,
        episode: SensoryEpisode,
        last_metrics: dict[str, Any] | None,
        semantic_match: float | None = None,
        evidence_unit_count: int | None = None,
    ) -> dict[str, Any]:
        text = " ".join(str(episode.text).split()).strip()
        if not text:
            return {"topics": [], "salience": 0.0}
        sensory = self._brain_config.get("sensory") or {}
        base_salience = float(sensory.get("observation_salience", 0.82))
        modality_need = self._sensory_modality_need_locked(runtime.adapter)
        semantic_score = runtime.last_semantic_match if semantic_match is None else max(0.0, min(1.0, float(semantic_match)))
        accepted_bonus = 0.0
        if isinstance(last_metrics, dict):
            if last_metrics.get("cross_modal_visual_accepted"):
                accepted_bonus += 0.04
            if last_metrics.get("cross_modal_audio_accepted"):
                accepted_bonus += 0.04
        salience = max(
            0.25,
            min(
                0.98,
                base_salience
                + 0.10 * modality_need
                + 0.08 * semantic_score
                + accepted_bonus,
            ),
        )
        topics: list[str] = []
        if runtime.adapter == "s1_mmalign":
            title = " ".join(str(episode.metadata.get("title", "")).split()).strip()
            categories = " ".join(str(episode.metadata.get("categories", "")).split()).strip()
            if title:
                topics.extend(salient_query_terms(title)[:2])
            if categories:
                topics.extend(salient_query_terms(categories)[:2])
        focus_terms = list(self._last_sensory_focus_terms)[:4]
        topics.extend(focus_terms[:2])
        topics.extend(salient_query_terms(text)[:4])
        deduped_topics = self._dedupe_grounded_topics(topics, limit=6)
        modality = "image" if episode.visual_spikes is not None and episode.audio_spikes is None else (
            "audio" if episode.audio_spikes is not None and episode.visual_spikes is None else "multisensory"
        )
        normalized_units = max(1, int(evidence_unit_count or 1))
        grounding_signal = max(
            0.35,
            min(
                1.0,
                0.48 * semantic_score
                + 0.22 * modality_need
                + 0.20 * salience
                + 0.10 * min(1.0, float(normalized_units) / 4.0),
            ),
        )
        metadata = self._grounded_observation_metadata(
            observation_kind="sensory",
            source_name=runtime.name,
            source_type="sensory",
            salience=salience,
            grounding_signal=grounding_signal,
            evidence_unit_count=normalized_units,
            modality=modality,
            focus_terms=deduped_topics[:4],
            extra={
                "adapter": runtime.adapter,
                "device": str(episode.metadata.get("device", "")),
                "encoder": deepcopy(episode.metadata.get("encoder")),
                "semantic_match": float(semantic_score),
                "item_semantic_match": float(runtime.last_item_semantic_match),
                "observation_sink": "subcortex_grounded_sensory_observation",
            },
        )
        return {
            "observation_sink": "subcortex_grounded_sensory_observation",
            "observation_kind": "sensory",
            "source_name": runtime.name,
            "source_type": "sensory",
            "adapter": runtime.adapter,
            "device": str(episode.metadata.get("device", "")),
            "encoder": deepcopy(episode.metadata.get("encoder")),
            "modality": modality,
            "content": text,
            "topics": deduped_topics,
            "salience": salience,
            "grounding_signal": grounding_signal,
            "evidence_unit_count": normalized_units,
            "semantic_match": float(semantic_score),
            "item_semantic_match": float(runtime.last_item_semantic_match),
            "focus_terms": focus_terms,
            "metadata": metadata,
        }

    def _record_sensory_preview_locked(
        self,
        *,
        runtime: _SensorySourceRuntime,
        episode: SensoryEpisode,
        text: str,
        topics: Sequence[str],
        semantic_match: float,
        item_semantic_match: float,
        modality_need: float,
        selection_score: float,
        window_budget: int,
    ) -> None:
        if episode.visual_preview is None and episode.audio_preview is None:
            return
        metadata = {
            key: deepcopy(value)
            for key, value in (episode.metadata or {}).items()
            if key not in {"bytes", "raw_bytes"}
        }
        entry = {
            "preview_id": str(uuid4()),
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "source_name": runtime.name,
            "adapter": runtime.adapter,
            "text": text[:480],
            "semantic_match": float(max(0.0, min(1.0, semantic_match))),
            "modality_need": float(max(0.0, min(1.0, modality_need))),
            "item_semantic_match": float(max(0.0, min(1.0, item_semantic_match))),
            "item_candidates_considered": int(max(0, runtime.last_item_candidates_considered)),
            "item_retrieval_lookahead": int(max(1, runtime.last_item_retrieval_lookahead or 1)),
            "selection_score": float(max(0.0, min(1.0, selection_score))),
            "window_budget": int(max(0, window_budget)),
            "topics": list(topics)[:8],
            "focus_terms": list(self._last_sensory_focus_terms)[:8],
            "metadata": metadata,
            "visual": deepcopy(episode.visual_preview),
            "audio": deepcopy(episode.audio_preview),
        }
        self._sensory_preview_history.appendleft(entry)

    def _run_real_sensory_episode_locked(self) -> dict[str, Any] | None:
        sensory = self._brain_config.get("sensory")
        if (
            not sensory
            or not sensory.get("enabled")
            or not getattr(self._trainer.config, "enable_cross_modal", False)
            or not self._sensory_source_runtimes
        ):
            return None

        self._request_active_execution_locked()
        try:
            current_tokens = int(self._trainer.token_count)
            trigger_interval = int(sensory.get("episode_interval_tokens", 2048))
            cooldown = float(sensory.get("cooldown_seconds", 10.0))
            now = time.time()
            if current_tokens - self._last_real_sensory_episode_token_count < trigger_interval:
                return None
            if (now - self._last_real_sensory_episode_time) < cooldown:
                return None

            items_per_episode = int(sensory.get("items_per_episode", 2))
            repeat_sources = bool(sensory.get("repeat_sources", True))
            source_count = len(self._sensory_source_runtimes)
            if source_count <= 0:
                return None
            self._sensory_stream_epoch += 1

            episodes_run = 0
            steps_trained = 0
            last_metrics: dict[str, Any] | None = None
            used_sources: list[dict[str, Any]] = []
            self._real_sensory_last_error = None

            selected_indices: set[int] = set()
            max_items = min(items_per_episode, source_count)
            for _ in range(max_items):
                selection = self._select_sensory_runtime_locked(selected_indices)
                if selection is None:
                    break
                idx, runtime, semantic_match, modality_need, selection_score = selection
                selected_indices.add(idx)
                focus_terms = list(self._last_sensory_focus_terms)
                episode = self._next_sensory_episode_locked(
                    runtime,
                    repeat_sources=repeat_sources,
                    focus_terms=focus_terms,
                )
                if episode is None:
                    continue
                runtime.exhausted = False
                self._sensory_source_index = (idx + 1) % source_count
                text = " ".join(str(episode.text).split()).strip()
                if not text:
                    continue
                if episode.visual_spikes is None and episode.audio_spikes is None:
                    continue

                effective_semantic_match = max(float(semantic_match), float(runtime.last_item_semantic_match))
                window_budget = self._sensory_window_budget_locked(
                    runtime,
                    semantic_match=effective_semantic_match,
                    modality_need=modality_need,
                )
                item_steps = 0
                last_raw_window = text
                for raw_window, pattern in self._encoder.iter_char_patterns(text, self._trainer.config.window_size):
                    last_raw_window = raw_window
                    last_metrics = self._trainer.train_step(
                        pattern,
                        raw_window=raw_window,
                        visual_spikes=episode.visual_spikes,
                        audio_spikes=episode.audio_spikes,
                    )
                    item_steps += 1
                    steps_trained += 1
                    if last_metrics:
                        if last_metrics.get("cross_modal_visual_accepted"):
                            self._real_visual_accepted += 1
                        if last_metrics.get("cross_modal_audio_accepted"):
                            self._real_audio_accepted += 1
                    if item_steps >= window_budget:
                        break

                if item_steps <= 0:
                    continue

                runtime.episodes_processed += 1
                runtime.last_activity_at = datetime.now(timezone.utc).isoformat()
                runtime.last_text = text[:160]
                OperatorInteractionRuntime._observe_runtime_concepts_locked(
                    self,
                    raw_window=last_raw_window,
                    metrics=last_metrics,
                )
                runtime.last_window_budget = int(window_budget)
                observation = self._inject_sensory_observation_locked(
                    runtime=runtime,
                    episode=episode,
                    last_metrics=last_metrics,
                    semantic_match=semantic_match,
                    evidence_unit_count=window_budget,
                )
                self._record_sensory_preview_locked(
                    runtime=runtime,
                    episode=episode,
                    text=text,
                    topics=list(observation.get("topics") or []),
                    semantic_match=semantic_match,
                    item_semantic_match=runtime.last_item_semantic_match,
                    modality_need=modality_need,
                    selection_score=selection_score,
                    window_budget=window_budget,
                )
                used_sources.append(
                    {
                        "name": runtime.name,
                        "adapter": runtime.adapter,
                        "steps_trained": int(item_steps),
                        "window_budget": int(window_budget),
                        "semantic_match": float(semantic_match),
                        "item_semantic_match": float(runtime.last_item_semantic_match),
                        "item_candidates_considered": int(runtime.last_item_candidates_considered),
                        "modality_need": float(modality_need),
                        "selection_score": float(selection_score),
                        "has_visual": bool(episode.visual_spikes is not None),
                        "has_audio": bool(episode.audio_spikes is not None),
                        "grounded_observation": observation,
                    }
                )
                self._update_sensory_runtime_cache_locked(runtime, served_episodes=[episode])
                episodes_run += 1

            if episodes_run <= 0:
                return None

            self._real_sensory_episodes_completed += episodes_run
            self._last_real_sensory_episode_time = now
            self._last_real_sensory_episode_token_count = int(self._trainer.token_count)
            self._runtime_state.mark_mutated()
            return {
                "type": "real_sensory_episode",
                "episodes_completed": int(self._real_sensory_episodes_completed),
                "episode_count": int(episodes_run),
                "steps_trained": int(steps_trained),
                "sources": used_sources,
                "last_metrics": last_metrics,
            }
        finally:
            self._release_active_execution_locked()
