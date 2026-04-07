from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, List, Optional, Sequence

import torch


class DualMemoryStore:
    """Reservoir slow buffer with simplified tag-and-PRP consolidation dynamics.

    The maintained runtime still uses a tractable phenomenological STC model
    rather than a molecular simulation, but replay/consolidation is now driven
    by explicit per-memory tag and PRP state instead of a tag-only proxy.
    """

    def __init__(
        self,
        capacity: int,
        ema_alpha: float = 0.01,
        slow_mean_decay: float = 0.9999,
        capture_tag_decay: float = 0.985,
        capture_release: float = 0.70,
        consolidation_rate: float = 1.0,
        functional_minute: int = 500,
        tag_duration_weak: float = 30.0,
        tag_duration_strong: float = 120.0,
        prp_tau_weak: float = 60.0,
        prp_tau_strong: float = 240.0,
        prp_synthesis_rate: float = 0.18,
        prp_capture_threshold: float = 0.15,
        prp_consumption: float = 0.50,
        strong_event_threshold: float = 0.60,
    ) -> None:
        self.capacity = int(capacity)
        self.ema_alpha = float(ema_alpha)
        self.slow_mean_decay = float(slow_mean_decay)
        self.capture_tag_decay = float(capture_tag_decay)
        self.capture_release = float(capture_release)
        self.consolidation_rate = float(consolidation_rate)

        self.functional_minute = int(functional_minute)
        self.tag_duration_weak = float(tag_duration_weak)
        self.tag_duration_strong = float(tag_duration_strong)
        self.prp_tau_weak = float(prp_tau_weak)
        self.prp_tau_strong = float(prp_tau_strong)
        self.prp_synthesis_rate = float(prp_synthesis_rate)
        self.prp_capture_threshold = float(prp_capture_threshold)
        self.prp_consumption = float(prp_consumption)
        self.strong_event_threshold = float(strong_event_threshold)

        self.slow_buffer: List[torch.Tensor] = []
        self.slow_input_patterns: List[Optional[torch.Tensor]] = []
        self.slow_routing_keys: List[Optional[torch.Tensor]] = []
        self.slow_raw_windows: List[Optional[str]] = []
        self.slow_texts: List[Optional[str]] = []
        self.slow_bucket_ids: List[Optional[int]] = []
        self.slow_importance: List[float] = []
        self.slow_capture_tag: List[float] = []
        self.slow_tag_is_strong: List[bool] = []
        self.slow_local_prp: List[float] = []
        self.slow_last_capture_token: List[int] = []
        self.slow_consolidation_level: List[float] = []
        self.slow_consolidation_events: List[int] = []
        self.slow_entry_timestamps: List[int] = []
        self.slow_last_replay_token: List[int] = []
        self.slow_replay_count: List[int] = []
        self.fast_ema: Optional[torch.Tensor] = None
        self.local_fast_ema: dict[int, torch.Tensor] = {}
        self.local_slow_mean: dict[int, torch.Tensor] = {}
        self.local_weight_sums: dict[int, float] = defaultdict(float)
        self.local_mean_tokens: dict[int, int] = {}

        self.global_prp_pool = 0.0
        self.bucket_prp_pool: dict[int, float] = defaultdict(float)
        self._state_token = 0

        self.n_seen = 0
        self._slow_mean: Optional[torch.Tensor] = None
        self._slow_weight_sum = 0.0
        self._slow_mean_token: Optional[int] = None

    def reset(self) -> None:
        self.slow_buffer = []
        self.slow_input_patterns = []
        self.slow_routing_keys = []
        self.slow_raw_windows = []
        self.slow_texts = []
        self.slow_bucket_ids = []
        self.slow_importance = []
        self.slow_capture_tag = []
        self.slow_tag_is_strong = []
        self.slow_local_prp = []
        self.slow_last_capture_token = []
        self.slow_consolidation_level = []
        self.slow_consolidation_events = []
        self.slow_entry_timestamps = []
        self.slow_last_replay_token = []
        self.slow_replay_count = []
        self.fast_ema = None
        self.local_fast_ema = {}
        self.local_slow_mean = {}
        self.local_weight_sums = defaultdict(float)
        self.local_mean_tokens = {}
        self.global_prp_pool = 0.0
        self.bucket_prp_pool = defaultdict(float)
        self._state_token = 0
        self._slow_mean = None
        self._slow_weight_sum = 0.0
        self._slow_mean_token = None
        self.n_seen = 0

    def _tag_tau_tokens(self, strong: bool) -> float:
        duration = self.tag_duration_strong if strong else self.tag_duration_weak
        return max(1.0, float(self.functional_minute) * duration)

    def _prp_tau_tokens(self, strong: bool) -> float:
        duration = self.prp_tau_strong if strong else self.prp_tau_weak
        return max(1.0, float(self.functional_minute) * duration)

    def _advance_state(self, token_marker: int) -> None:
        target = int(token_marker)
        delta = max(0, target - int(self._state_token))
        if delta <= 0:
            self._state_token = target
            return

        for idx in range(len(self.slow_buffer)):
            strong = bool(self.slow_tag_is_strong[idx]) if idx < len(self.slow_tag_is_strong) else False
            tag_tau = self._tag_tau_tokens(strong)
            prp_tau = self._prp_tau_tokens(strong)
            minute_delta = float(delta) / max(1.0, float(self.functional_minute))
            tag_decay = math.exp(-float(delta) / tag_tau)
            if self.capture_tag_decay < 1.0:
                tag_decay *= self.capture_tag_decay ** minute_delta
            prp_decay = math.exp(-float(delta) / prp_tau)
            self.slow_capture_tag[idx] = float(max(0.0, self.slow_capture_tag[idx] * tag_decay))
            self.slow_local_prp[idx] = float(max(0.0, self.slow_local_prp[idx] * prp_decay))

        global_decay = math.exp(-float(delta) / self._prp_tau_tokens(True))
        self.global_prp_pool = float(max(0.0, self.global_prp_pool * global_decay))
        for bucket in list(self.bucket_prp_pool.keys()):
            decayed = float(max(0.0, self.bucket_prp_pool[bucket] * global_decay))
            if decayed <= 1e-8:
                del self.bucket_prp_pool[bucket]
            else:
                self.bucket_prp_pool[bucket] = decayed

        self._state_token = target

    def _advance_slow_mean_time(self, token_marker: int) -> None:
        if self._slow_mean_token is None:
            self._slow_mean_token = int(token_marker)
            return

        delta = max(0, int(token_marker) - self._slow_mean_token)
        if delta > 0 and self._slow_weight_sum > 0.0:
            self._slow_weight_sum *= self.slow_mean_decay ** delta
        self._slow_mean_token = int(token_marker)

    def _append_to_slow_mean(self, x: torch.Tensor) -> None:
        if self._slow_mean is None or self._slow_weight_sum <= 0.0:
            self._slow_mean = x.clone()
            self._slow_weight_sum = 1.0
            return

        new_weight_sum = self._slow_weight_sum + 1.0
        numerator = self._slow_mean * self._slow_weight_sum + x
        self._slow_mean = numerator / new_weight_sum
        self._slow_weight_sum = new_weight_sum

    def _replace_in_slow_mean(self, old: torch.Tensor, old_timestamp: int, x: torch.Tensor, token_marker: int) -> None:
        if self._slow_mean is None or self._slow_weight_sum <= 0.0:
            self._slow_mean = x.clone()
            self._slow_weight_sum = 1.0
            return

        old_weight = self.slow_mean_decay ** max(0, int(token_marker) - int(old_timestamp))
        numerator = self._slow_mean * self._slow_weight_sum - old * old_weight + x
        new_weight_sum = max(1e-8, self._slow_weight_sum - old_weight + 1.0)
        self._slow_mean = numerator / new_weight_sum
        self._slow_weight_sum = new_weight_sum

    def _update_local_bucket(self, x: torch.Tensor, bucket_id: int, token_marker: int) -> None:
        bucket = int(bucket_id)
        if bucket not in self.local_fast_ema:
            self.local_fast_ema[bucket] = x.clone()
            self.local_slow_mean[bucket] = x.clone()
            self.local_weight_sums[bucket] = 1.0
            self.local_mean_tokens[bucket] = int(token_marker)
            return

        self.local_fast_ema[bucket] = self.ema_alpha * x + (1.0 - self.ema_alpha) * self.local_fast_ema[bucket]
        weight_sum = float(self.local_weight_sums.get(bucket, 0.0))
        last_token = int(self.local_mean_tokens.get(bucket, token_marker))
        delta = max(0, int(token_marker) - last_token)
        if delta > 0 and weight_sum > 0.0:
            weight_sum *= self.slow_mean_decay ** delta

        if weight_sum <= 0.0:
            self.local_slow_mean[bucket] = x.clone()
            self.local_weight_sums[bucket] = 1.0
        else:
            new_weight_sum = weight_sum + 1.0
            numerator = self.local_slow_mean[bucket] * weight_sum + x
            self.local_slow_mean[bucket] = numerator / new_weight_sum
            self.local_weight_sums[bucket] = new_weight_sum
        self.local_mean_tokens[bucket] = int(token_marker)

    def _is_strong_event(self, strength: float, importance: float) -> bool:
        return float(max(strength, importance)) >= self.strong_event_threshold

    def _inject_prp(self, *, bucket_id: Optional[int], strength: float, importance: float, sleep_boost: float = 1.0) -> float:
        event_strength = float(max(0.0, max(strength, importance)))
        if event_strength <= 0.0:
            return 0.0

        intensity = max(0.0, event_strength - 0.25 * self.strong_event_threshold)
        synthesized = float(self.prp_synthesis_rate * intensity * max(0.25, sleep_boost))
        if synthesized <= 0.0:
            return 0.0

        self.global_prp_pool += 0.65 * synthesized
        if bucket_id is not None:
            self.bucket_prp_pool[int(bucket_id)] += 0.35 * synthesized
        return synthesized

    def _pool_share(self, idx: int) -> tuple[float, float]:
        bucket_id = self.slow_bucket_ids[idx] if idx < len(self.slow_bucket_ids) else None
        bucket_share = 0.35 * float(self.bucket_prp_pool.get(int(bucket_id), 0.0)) if bucket_id is not None else 0.0
        global_share = 0.15 * float(self.global_prp_pool)
        return bucket_share, global_share

    def _available_prp(self, idx: int) -> float:
        local_prp = float(self.slow_local_prp[idx]) if idx < len(self.slow_local_prp) else 0.0
        bucket_share, global_share = self._pool_share(idx)
        return float(max(0.0, local_prp + bucket_share + global_share))

    def _consume_pools(self, idx: int, amount: float) -> float:
        required = float(max(0.0, amount))
        if required <= 0.0:
            return 0.0

        consumed = 0.0
        bucket_id = self.slow_bucket_ids[idx] if idx < len(self.slow_bucket_ids) else None
        if bucket_id is not None:
            bucket = int(bucket_id)
            bucket_take = min(float(self.bucket_prp_pool.get(bucket, 0.0)), required * 0.70)
            if bucket_take > 0.0:
                self.bucket_prp_pool[bucket] = float(max(0.0, self.bucket_prp_pool[bucket] - bucket_take))
                consumed += bucket_take

        remaining = max(0.0, required - consumed)
        global_take = min(float(self.global_prp_pool), remaining)
        if global_take > 0.0:
            self.global_prp_pool = float(max(0.0, self.global_prp_pool - global_take))
            consumed += global_take
        return consumed

    def _effective_capture_strength(self, idx: int, current_token: int) -> float:
        if idx < 0 or idx >= len(self.slow_capture_tag):
            return 0.0
        self._advance_state(current_token)
        return float(max(0.0, self.slow_capture_tag[idx]) * max(0.0, self._available_prp(idx)))

    def _effective_capture_tensor(self, current_token: int) -> torch.Tensor:
        self._advance_state(current_token)
        return torch.tensor(
            [self._effective_capture_strength(idx, current_token) for idx in range(len(self.slow_buffer))],
            dtype=torch.float32,
        )

    def _tag_strength_tensor(self, current_token: int) -> torch.Tensor:
        self._advance_state(current_token)
        return torch.tensor(self.slow_capture_tag, dtype=torch.float32)

    def _prp_tensor(self, current_token: int) -> torch.Tensor:
        self._advance_state(current_token)
        return torch.tensor([self._available_prp(idx) for idx in range(len(self.slow_buffer))], dtype=torch.float32)

    def _store_slot(
        self,
        index: int,
        *,
        assembly: torch.Tensor,
        stored_input: Optional[torch.Tensor],
        stored_routing: Optional[torch.Tensor],
        stored_window: Optional[str],
        stored_text: Optional[str],
        bucket_id: Optional[int],
        importance: float,
        capture_value: float,
        token_marker: int,
    ) -> None:
        strong_event = self._is_strong_event(capture_value, importance)
        injected_prp = self._inject_prp(bucket_id=bucket_id, strength=capture_value, importance=importance)
        local_prp = 0.20 * injected_prp if strong_event else 0.0
        tag_value = float(max(0.0, capture_value if capture_value > 0.0 else importance))

        self.slow_buffer[index] = assembly
        self.slow_input_patterns[index] = stored_input
        self.slow_routing_keys[index] = stored_routing
        self.slow_raw_windows[index] = stored_window
        self.slow_texts[index] = stored_text
        self.slow_bucket_ids[index] = int(bucket_id) if bucket_id is not None else None
        self.slow_importance[index] = float(max(1e-6, importance))
        self.slow_capture_tag[index] = tag_value
        self.slow_tag_is_strong[index] = bool(strong_event)
        self.slow_local_prp[index] = float(max(0.0, local_prp))
        self.slow_last_capture_token[index] = int(token_marker)
        self.slow_consolidation_level[index] = 0.0
        self.slow_consolidation_events[index] = 0
        self.slow_entry_timestamps[index] = int(token_marker)
        self.slow_last_replay_token[index] = int(token_marker)
        self.slow_replay_count[index] = 0

    def update(
        self,
        assembly: torch.Tensor,
        importance: float = 1.0,
        token_count: Optional[int] = None,
        bucket_id: Optional[int] = None,
        input_pattern: Optional[torch.Tensor] = None,
        routing_key: Optional[torch.Tensor] = None,
        raw_window: Optional[str] = None,
        text: Optional[str] = None,
        tag_strength: float = 0.0,
        capture_tag: float | None = None,
    ) -> int | None:
        x = assembly.detach().clone().cpu()
        stored_input = input_pattern.detach().clone().cpu() if input_pattern is not None else None
        stored_routing = routing_key.detach().clone().cpu() if routing_key is not None else None
        stored_window = None if raw_window is None else str(raw_window)
        stored_text = None if text is None else str(text)
        token_marker = int(self.n_seen if token_count is None else token_count)
        capture_value = float(max(0.0, tag_strength if capture_tag is None else capture_tag))

        self._advance_state(token_marker)

        if self.fast_ema is None:
            self.fast_ema = x.clone()
        else:
            self.fast_ema = self.ema_alpha * x + (1.0 - self.ema_alpha) * self.fast_ema

        self.n_seen += 1
        self._advance_slow_mean_time(token_marker)
        if bucket_id is not None:
            self._update_local_bucket(x, int(bucket_id), token_marker)

        if len(self.slow_buffer) < self.capacity:
            self.slow_buffer.append(x)
            self.slow_input_patterns.append(stored_input)
            self.slow_routing_keys.append(stored_routing)
            self.slow_raw_windows.append(stored_window)
            self.slow_texts.append(stored_text)
            self.slow_bucket_ids.append(int(bucket_id) if bucket_id is not None else None)
            self.slow_importance.append(float(max(1e-6, importance)))
            self.slow_capture_tag.append(0.0)
            self.slow_tag_is_strong.append(False)
            self.slow_local_prp.append(0.0)
            self.slow_last_capture_token.append(token_marker)
            self.slow_consolidation_level.append(0.0)
            self.slow_consolidation_events.append(0)
            self.slow_entry_timestamps.append(token_marker)
            self.slow_last_replay_token.append(token_marker)
            self.slow_replay_count.append(0)
            self._store_slot(
                len(self.slow_buffer) - 1,
                assembly=x,
                stored_input=stored_input,
                stored_routing=stored_routing,
                stored_window=stored_window,
                stored_text=stored_text,
                bucket_id=bucket_id,
                importance=importance,
                capture_value=capture_value,
                token_marker=token_marker,
            )
            self._append_to_slow_mean(x)
            return len(self.slow_buffer) - 1

        j = int(torch.randint(0, self.n_seen, (1,)).item())
        if j < self.capacity:
            old = self.slow_buffer[j]
            old_timestamp = self.slow_entry_timestamps[j]
            self._store_slot(
                j,
                assembly=x,
                stored_input=stored_input,
                stored_routing=stored_routing,
                stored_window=stored_window,
                stored_text=stored_text,
                bucket_id=bucket_id,
                importance=importance,
                capture_value=capture_value,
                token_marker=token_marker,
            )
            self._replace_in_slow_mean(old, old_timestamp, x, token_marker)
            return j
        return None

    def replay_scores(self, current_token: int) -> torch.Tensor:
        if not self.slow_buffer:
            return torch.zeros(0, dtype=torch.float32)

        self._advance_state(current_token)
        scores: list[float] = []
        for idx in range(len(self.slow_buffer)):
            importance = float(max(1e-6, self.slow_importance[idx]))
            replay_age = max(0, int(current_token) - int(self.slow_last_replay_token[idx]))
            spacing = math.log1p(float(replay_age))
            tag_strength = float(max(0.0, self.slow_capture_tag[idx]))
            prp_level = float(max(0.0, self._available_prp(idx)))
            capture_strength = float(max(0.0, tag_strength * prp_level))
            consolidation = float(max(0.0, min(1.0, self.slow_consolidation_level[idx])))
            replay_count = max(0, int(self.slow_replay_count[idx]))
            unmet_capture = max(0.0, capture_strength - consolidation)
            frontier = (
                2.00 * unmet_capture
                + 0.75 * tag_strength
                + 0.35 * prp_level
                + 0.50 * max(0.0, 1.0 - consolidation)
            )
            score = importance * (1.0 + spacing) * (1.0 + frontier) / (1.0 + 0.35 * float(replay_count))
            scores.append(float(score))
        return torch.tensor(scores, dtype=torch.float32)

    def tag_recent_entries(
        self,
        *,
        current_token: int,
        window_tokens: int,
        strength: float,
    ) -> int:
        if window_tokens <= 0 or strength <= 0.0:
            return 0

        self._advance_state(current_token)
        floor_token = max(0, int(current_token) - int(window_tokens))
        tagged = 0
        for idx, token_marker in enumerate(self.slow_entry_timestamps):
            if int(token_marker) < floor_token:
                continue

            tag_strength = float(max(0.0, strength))
            importance = float(self.slow_importance[idx]) if idx < len(self.slow_importance) else 0.0
            strong_event = self._is_strong_event(tag_strength, importance)
            self.slow_capture_tag[idx] = float(max(self.slow_capture_tag[idx], tag_strength))
            self.slow_tag_is_strong[idx] = bool(self.slow_tag_is_strong[idx] or strong_event)
            self.slow_last_capture_token[idx] = int(current_token)
            injected = self._inject_prp(
                bucket_id=self.slow_bucket_ids[idx] if idx < len(self.slow_bucket_ids) else None,
                strength=tag_strength,
                importance=importance,
            )
            if injected > 0.0:
                local_share = 0.20 if strong_event else 0.08
                self.slow_local_prp[idx] = float(max(self.slow_local_prp[idx], local_share * injected))
            tagged += 1
        return tagged

    def replay_entry(self, index: int, current_token: Optional[int] = None) -> dict[str, Any]:
        idx = int(index)
        if idx < 0 or idx >= len(self.slow_buffer):
            raise IndexError(f"Memory index out of range: {index}")

        token_marker = self._state_token if current_token is None else int(current_token)
        self._advance_state(token_marker)
        tag_strength = float(max(0.0, self.slow_capture_tag[idx]))
        prp_level = float(max(0.0, self._available_prp(idx)))
        capture_strength = float(max(0.0, tag_strength * prp_level))
        consolidation = float(max(0.0, min(1.0, self.slow_consolidation_level[idx])))
        input_pattern = self.slow_input_patterns[idx]
        routing_key = self.slow_routing_keys[idx]
        return {
            "assembly": self.slow_buffer[idx].detach().clone(),
            "input_pattern": input_pattern.detach().clone() if isinstance(input_pattern, torch.Tensor) else None,
            "routing_key": routing_key.detach().clone() if isinstance(routing_key, torch.Tensor) else None,
            "raw_window": self.slow_raw_windows[idx],
            "text": self.slow_texts[idx],
            "bucket_id": self.slow_bucket_ids[idx],
            "importance": float(self.slow_importance[idx]),
            "tag_strength": tag_strength,
            "capture_tag": tag_strength,
            "stored_capture_tag": tag_strength,
            "prp_level": prp_level,
            "capture_strength": capture_strength,
            "consolidation_level": consolidation,
            "consolidation_gap": float(max(0.0, 1.0 - consolidation)),
            "consolidation_events": int(self.slow_consolidation_events[idx]),
            "replay_count": int(self.slow_replay_count[idx]),
            "age_tokens": int(max(0, token_marker - int(self.slow_entry_timestamps[idx]))),
            "last_replay_token": int(self.slow_last_replay_token[idx]),
            "tag_is_strong": bool(self.slow_tag_is_strong[idx]),
        }

    def sample_replay_indices(
        self,
        *,
        n: int,
        current_token: int,
        candidate_pool: Optional[int] = None,
        strategy: str = "priority",
    ) -> list[int]:
        if n <= 0 or not self.slow_buffer:
            return []

        scores = self.replay_scores(current_token)
        if int(scores.numel()) <= 0:
            return []

        count = len(self.slow_buffer)
        if strategy == "random":
            perm = torch.randperm(count)
            return [int(idx) for idx in perm[: min(count, int(n))].tolist()]

        top_k = min(count, max(int(n), int(candidate_pool) if candidate_pool is not None else int(n)))
        top_values, top_indices = torch.topk(scores, k=top_k)
        if top_k <= int(n):
            return [int(idx) for idx in top_indices.tolist()]

        weights = torch.clamp(top_values, min=1e-8)
        weights = weights / (weights.sum() + 1e-8)
        draw = torch.multinomial(weights, num_samples=min(int(n), top_k), replacement=False)
        chosen = [int(top_indices[int(local_idx)].item()) for local_idx in draw.tolist()]
        chosen.sort(key=lambda item: float(scores[item].item()), reverse=True)
        return chosen

    def consolidate_replay(
        self,
        indices: Sequence[int],
        *,
        current_token: int,
        blend: float,
        protein_synthesis_level: float = 1.0,
    ) -> None:
        if not indices:
            return

        self._advance_state(current_token)
        replay_blend = float(max(0.0, blend))
        synthesis_level = float(max(0.0, protein_synthesis_level))
        replay_vectors: list[torch.Tensor] = []
        replay_buckets: dict[int, list[torch.Tensor]] = defaultdict(list)

        for raw_idx in indices:
            idx = int(raw_idx)
            if idx < 0 or idx >= len(self.slow_buffer):
                continue

            bucket_id = self.slow_bucket_ids[idx] if idx < len(self.slow_bucket_ids) else None
            tag_strength = float(max(0.0, self.slow_capture_tag[idx]))
            importance = float(max(1e-6, self.slow_importance[idx]))
            if synthesis_level > 0.0:
                injected = self._inject_prp(
                    bucket_id=bucket_id,
                    strength=tag_strength,
                    importance=importance,
                    sleep_boost=synthesis_level,
                )
                if injected > 0.0:
                    self.slow_local_prp[idx] = float(self.slow_local_prp[idx] + 0.10 * injected)

            recruit_target = max(0.0, self.prp_capture_threshold - self.slow_local_prp[idx]) + 0.25 * tag_strength
            recruited = self._consume_pools(idx, recruit_target * max(0.5, synthesis_level))
            if recruited > 0.0:
                self.slow_local_prp[idx] = float(self.slow_local_prp[idx] + recruited)

            prp_level = float(max(0.0, self._available_prp(idx)))
            capture_strength = float(max(0.0, tag_strength * prp_level))
            consolidation = float(max(0.0, min(1.0, self.slow_consolidation_level[idx])))
            capture_drive = max(0.0, capture_strength - self.prp_capture_threshold) + 0.25 * min(capture_strength, self.prp_capture_threshold)
            delta = replay_blend * self.consolidation_rate * capture_drive * max(0.0, 1.0 - consolidation)

            if delta > 0.0:
                self.slow_consolidation_level[idx] = float(min(1.0, consolidation + delta))
                self.slow_consolidation_events[idx] += 1
                release_factor = 1.0 - (1.0 - self.capture_release) * max(0.10, min(1.0, replay_blend))
                self.slow_capture_tag[idx] = float(max(0.0, self.slow_capture_tag[idx] * release_factor))

            local_prp_spend = min(
                float(self.slow_local_prp[idx]),
                max(0.0, self.prp_consumption * delta + 0.10 * capture_strength),
            )
            if local_prp_spend > 0.0:
                self.slow_local_prp[idx] = float(max(0.0, self.slow_local_prp[idx] - local_prp_spend))

            self.slow_last_replay_token[idx] = int(current_token)
            self.slow_replay_count[idx] += 1
            replay_vectors.append(self.slow_buffer[idx].detach().clone())
            if bucket_id is not None:
                replay_buckets[int(bucket_id)].append(self.slow_buffer[idx].detach().clone())

        if replay_vectors:
            replay_centroid = torch.stack(replay_vectors, dim=0).mean(dim=0)
            if self.fast_ema is None:
                self.fast_ema = replay_centroid.clone()
            else:
                self.fast_ema = (1.0 - replay_blend) * self.fast_ema + replay_blend * replay_centroid

        for bucket_id, bucket_vectors in replay_buckets.items():
            if not bucket_vectors:
                continue
            centroid = torch.stack(bucket_vectors, dim=0).mean(dim=0)
            if bucket_id not in self.local_fast_ema:
                self.local_fast_ema[bucket_id] = centroid.clone()
            else:
                self.local_fast_ema[bucket_id] = (1.0 - replay_blend) * self.local_fast_ema[bucket_id] + replay_blend * centroid

    def compute_drift(self, bucket_id: Optional[int] = None) -> float:
        if bucket_id is not None:
            bucket = int(bucket_id)
            fast = self.local_fast_ema.get(bucket)
            slow = self.local_slow_mean.get(bucket)
            if isinstance(fast, torch.Tensor) and isinstance(slow, torch.Tensor):
                denom = float(torch.norm(fast).item() + torch.norm(slow).item())
                if denom <= 1e-8:
                    return 0.0
                return float(torch.norm(fast - slow).item() / denom)

        if self.fast_ema is None or self._slow_mean is None:
            return 0.0
        denom = float(torch.norm(self.fast_ema).item() + torch.norm(self._slow_mean).item())
        if denom <= 1e-8:
            return 0.0
        return float(torch.norm(self.fast_ema - self._slow_mean).item() / denom)

    def summary_stats(self, current_token: Optional[int] = None) -> dict[str, Any]:
        token_marker = self._state_token if current_token is None else int(current_token)
        self._advance_state(token_marker)
        prp_levels = [float(self._available_prp(idx)) for idx in range(len(self.slow_buffer))]
        capture_levels = [
            float(max(0.0, self.slow_capture_tag[idx]) * max(0.0, prp_levels[idx]))
            for idx in range(len(self.slow_buffer))
        ]
        size = len(self.slow_buffer)
        return {
            "capacity": int(self.capacity),
            "size": int(size),
            "fill_fraction": float(size / max(1, self.capacity)),
            "n_seen": int(self.n_seen),
            "mean_importance": float(sum(self.slow_importance) / max(1, len(self.slow_importance))),
            "mean_capture_tag": float(sum(self.slow_capture_tag) / max(1, len(self.slow_capture_tag))),
            "mean_prp_level": float(sum(prp_levels) / max(1, len(prp_levels))),
            "mean_capture_strength": float(sum(capture_levels) / max(1, len(capture_levels))),
            "max_capture_strength": float(max(capture_levels, default=0.0)),
            "mean_consolidation_level": float(sum(self.slow_consolidation_level) / max(1, len(self.slow_consolidation_level))),
            "mean_replay_count": float(sum(self.slow_replay_count) / max(1, len(self.slow_replay_count))),
            "strong_tag_fraction": float(sum(1 for value in self.slow_tag_is_strong if value) / max(1, len(self.slow_tag_is_strong))),
            "global_prp_pool": float(self.global_prp_pool),
            "active_prp_buckets": int(len(self.bucket_prp_pool)),
            "drift": float(self.compute_drift()),
        }

    def snapshot(self) -> dict[str, Any]:
        return {
            "capacity": int(self.capacity),
            "ema_alpha": float(self.ema_alpha),
            "slow_mean_decay": float(self.slow_mean_decay),
            "capture_tag_decay": float(self.capture_tag_decay),
            "capture_release": float(self.capture_release),
            "consolidation_rate": float(self.consolidation_rate),
            "functional_minute": int(self.functional_minute),
            "tag_duration_weak": float(self.tag_duration_weak),
            "tag_duration_strong": float(self.tag_duration_strong),
            "prp_tau_weak": float(self.prp_tau_weak),
            "prp_tau_strong": float(self.prp_tau_strong),
            "prp_synthesis_rate": float(self.prp_synthesis_rate),
            "prp_capture_threshold": float(self.prp_capture_threshold),
            "prp_consumption": float(self.prp_consumption),
            "strong_event_threshold": float(self.strong_event_threshold),
            "slow_buffer": [value.detach().clone().cpu() for value in self.slow_buffer],
            "slow_input_patterns": [
                value.detach().clone().cpu() if isinstance(value, torch.Tensor) else None
                for value in self.slow_input_patterns
            ],
            "slow_routing_keys": [
                value.detach().clone().cpu() if isinstance(value, torch.Tensor) else None
                for value in self.slow_routing_keys
            ],
            "slow_raw_windows": list(self.slow_raw_windows),
            "slow_texts": list(self.slow_texts),
            "slow_bucket_ids": list(self.slow_bucket_ids),
            "slow_importance": list(self.slow_importance),
            "slow_capture_tag": list(self.slow_capture_tag),
            "slow_tag_is_strong": list(self.slow_tag_is_strong),
            "slow_local_prp": list(self.slow_local_prp),
            "slow_last_capture_token": list(self.slow_last_capture_token),
            "slow_consolidation_level": list(self.slow_consolidation_level),
            "slow_consolidation_events": list(self.slow_consolidation_events),
            "slow_entry_timestamps": list(self.slow_entry_timestamps),
            "slow_last_replay_token": list(self.slow_last_replay_token),
            "slow_replay_count": list(self.slow_replay_count),
            "fast_ema": None if self.fast_ema is None else self.fast_ema.detach().clone().cpu(),
            "local_fast_ema": {
                int(key): value.detach().clone().cpu()
                for key, value in self.local_fast_ema.items()
            },
            "local_slow_mean": {
                int(key): value.detach().clone().cpu()
                for key, value in self.local_slow_mean.items()
            },
            "local_weight_sums": {int(key): float(value) for key, value in self.local_weight_sums.items()},
            "local_mean_tokens": {int(key): int(value) for key, value in self.local_mean_tokens.items()},
            "global_prp_pool": float(self.global_prp_pool),
            "bucket_prp_pool": {int(key): float(value) for key, value in self.bucket_prp_pool.items()},
            "state_token": int(self._state_token),
            "n_seen": int(self.n_seen),
            "slow_mean": None if self._slow_mean is None else self._slow_mean.detach().clone().cpu(),
            "slow_weight_sum": float(self._slow_weight_sum),
            "slow_mean_token": None if self._slow_mean_token is None else int(self._slow_mean_token),
        }

    def restore(self, snapshot: dict[str, Any]) -> None:
        self.capacity = int(snapshot.get("capacity", self.capacity))
        self.ema_alpha = float(snapshot.get("ema_alpha", self.ema_alpha))
        self.slow_mean_decay = float(snapshot.get("slow_mean_decay", self.slow_mean_decay))
        self.capture_tag_decay = float(snapshot.get("capture_tag_decay", self.capture_tag_decay))
        self.capture_release = float(snapshot.get("capture_release", self.capture_release))
        self.consolidation_rate = float(snapshot.get("consolidation_rate", self.consolidation_rate))
        self.functional_minute = int(snapshot.get("functional_minute", self.functional_minute))
        self.tag_duration_weak = float(snapshot.get("tag_duration_weak", self.tag_duration_weak))
        self.tag_duration_strong = float(snapshot.get("tag_duration_strong", self.tag_duration_strong))
        self.prp_tau_weak = float(snapshot.get("prp_tau_weak", self.prp_tau_weak))
        self.prp_tau_strong = float(snapshot.get("prp_tau_strong", self.prp_tau_strong))
        self.prp_synthesis_rate = float(snapshot.get("prp_synthesis_rate", self.prp_synthesis_rate))
        self.prp_capture_threshold = float(snapshot.get("prp_capture_threshold", self.prp_capture_threshold))
        self.prp_consumption = float(snapshot.get("prp_consumption", self.prp_consumption))
        self.strong_event_threshold = float(snapshot.get("strong_event_threshold", self.strong_event_threshold))

        self.reset()

        def _clone_optional_list(values: Any) -> list[Optional[torch.Tensor]]:
            return [
                value.detach().clone().cpu() if isinstance(value, torch.Tensor) else None
                for value in list(values or [])
            ]

        def _pad(values: Any, fill: Any, size: int) -> list[Any]:
            items = list(values or [])
            if len(items) < size:
                items.extend([fill] * (size - len(items)))
            return items[:size]

        self.slow_buffer = [value.detach().clone().cpu() for value in list(snapshot.get("slow_buffer", []))]
        size = len(self.slow_buffer)
        self.slow_input_patterns = _clone_optional_list(snapshot.get("slow_input_patterns"))
        if len(self.slow_input_patterns) < size:
            self.slow_input_patterns.extend([None] * (size - len(self.slow_input_patterns)))
        self.slow_input_patterns = self.slow_input_patterns[:size]
        self.slow_routing_keys = _clone_optional_list(snapshot.get("slow_routing_keys"))
        if len(self.slow_routing_keys) < size:
            self.slow_routing_keys.extend([None] * (size - len(self.slow_routing_keys)))
        self.slow_routing_keys = self.slow_routing_keys[:size]
        self.slow_raw_windows = _pad(snapshot.get("slow_raw_windows"), None, size)
        self.slow_texts = _pad(snapshot.get("slow_texts"), None, size)
        self.slow_bucket_ids = [None if value is None else int(value) for value in _pad(snapshot.get("slow_bucket_ids"), None, size)]
        self.slow_importance = [float(value) for value in _pad(snapshot.get("slow_importance"), 1.0, size)]

        self.slow_capture_tag = [float(value) for value in _pad(snapshot.get("slow_capture_tag"), 0.0, size)]
        strong_flags = snapshot.get("slow_tag_is_strong")
        if strong_flags is None:
            self.slow_tag_is_strong = [bool(value >= self.strong_event_threshold) for value in self.slow_capture_tag]
        else:
            self.slow_tag_is_strong = [bool(value) for value in _pad(strong_flags, False, size)]
        self.slow_local_prp = [float(value) for value in _pad(snapshot.get("slow_local_prp"), 0.0, size)]
        timestamps = [int(value) for value in _pad(snapshot.get("slow_entry_timestamps"), 0, size)]
        self.slow_entry_timestamps = timestamps
        self.slow_last_capture_token = _pad(snapshot.get("slow_last_capture_token"), None, size)
        self.slow_last_capture_token = [
            timestamps[idx] if value is None else int(value)
            for idx, value in enumerate(self.slow_last_capture_token)
        ]
        self.slow_consolidation_level = [float(value) for value in _pad(snapshot.get("slow_consolidation_level"), 0.0, size)]
        self.slow_consolidation_events = [int(value) for value in _pad(snapshot.get("slow_consolidation_events"), 0, size)]
        self.slow_last_replay_token = _pad(snapshot.get("slow_last_replay_token"), None, size)
        self.slow_last_replay_token = [
            timestamps[idx] if value is None else int(value)
            for idx, value in enumerate(self.slow_last_replay_token)
        ]
        self.slow_replay_count = [int(value) for value in _pad(snapshot.get("slow_replay_count"), 0, size)]

        fast_ema = snapshot.get("fast_ema")
        self.fast_ema = fast_ema.detach().clone().cpu() if isinstance(fast_ema, torch.Tensor) else None
        self.local_fast_ema = {
            int(key): value.detach().clone().cpu()
            for key, value in dict(snapshot.get("local_fast_ema", {})).items()
            if isinstance(value, torch.Tensor)
        }
        self.local_slow_mean = {
            int(key): value.detach().clone().cpu()
            for key, value in dict(snapshot.get("local_slow_mean", {})).items()
            if isinstance(value, torch.Tensor)
        }
        self.local_weight_sums = defaultdict(
            float,
            {int(key): float(value) for key, value in dict(snapshot.get("local_weight_sums", {})).items()},
        )
        self.local_mean_tokens = {int(key): int(value) for key, value in dict(snapshot.get("local_mean_tokens", {})).items()}

        self.global_prp_pool = float(snapshot.get("global_prp_pool", 0.0))
        self.bucket_prp_pool = defaultdict(
            float,
            {int(key): float(value) for key, value in dict(snapshot.get("bucket_prp_pool", {})).items()},
        )
        self._state_token = int(snapshot.get("state_token", max(self.slow_entry_timestamps, default=0)))
        self.n_seen = int(snapshot.get("n_seen", size))
        slow_mean = snapshot.get("slow_mean")
        self._slow_mean = slow_mean.detach().clone().cpu() if isinstance(slow_mean, torch.Tensor) else None
        self._slow_weight_sum = float(snapshot.get("slow_weight_sum", 0.0))
        slow_mean_token = snapshot.get("slow_mean_token")
        self._slow_mean_token = None if slow_mean_token is None else int(slow_mean_token)

        if self._slow_mean is None and self.slow_buffer:
            self._slow_mean = torch.stack(self.slow_buffer, dim=0).mean(dim=0)
            self._slow_weight_sum = float(len(self.slow_buffer))
        if self._slow_mean_token is None and self.slow_entry_timestamps:
            self._slow_mean_token = int(max(self.slow_entry_timestamps))
