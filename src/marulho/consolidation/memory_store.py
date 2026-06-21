from __future__ import annotations

from array import array
from bisect import insort
import math
import time
from collections import defaultdict
from typing import Any, List, Mapping, NamedTuple, Optional, Sequence

import numpy as np
import torch


_BUCKET_CANDIDATE_SOURCE_WINDOW_POLICY = (
    "tail_indexed_bucket_round_robin_no_full_bucket_materialization"
)


class _BucketCandidateWindow(NamedTuple):
    normalized_bucket_ids: list[int] | None
    candidate_indices: list[int]
    available_count: int
    available_count_is_lower_bound: bool
    source_entry_read_count: int
    source_entry_read_budget: int
    source_entry_read_budget_exhausted: bool
    source_materialized_entry_count: int
    source_materialization_count: int
    source_full_bucket_scan: bool
    candidate_window_limit: int


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

        self._cached_summary: Optional[dict[str, Any]] = None
        self._cached_summary_token: int = -1
        self._bucket_consolidation_cpu: Optional[torch.Tensor] = None
        self._bucket_consolidation_weighted_sum: Optional[torch.Tensor] = None
        self._bucket_consolidation_weight_sum: Optional[torch.Tensor] = None
        self._bucket_consolidation_devices: dict[str, torch.Tensor] = {}
        self._bucket_consolidation_cache_generation = 0
        self.bucket_consolidation_cache_rebuild_count = 0
        self.bucket_consolidation_cache_rebuild_scan_entry_count = 0
        self.bucket_consolidation_level_cache_lookup_count = 0
        self.bucket_consolidation_level_cache_miss_count = 0
        self.last_bucket_consolidation_level_report = {
            "surface": "bucket_consolidation_level_cache_lookup.v1",
            "status": "not_run",
            "bucket_id": None,
            "level": 0.0,
            "cache_hit": False,
            "full_memory_scan": False,
            "scan_entry_count": 0,
            "cache_generation": 0,
        }
        self._bucket_entry_indices: defaultdict[int, list[int]] = defaultdict(list)
        self._recent_entry_indices: list[tuple[int, int]] = []

        self.slow_buffer: List[torch.Tensor] = []
        self.slow_input_patterns: List[Optional[torch.Tensor]] = []
        self.slow_routing_keys: List[Optional[torch.Tensor]] = []
        self.slow_raw_windows: List[Optional[str]] = []
        self.slow_texts: List[Optional[str]] = []
        self.slow_metadata: List[Optional[dict[str, Any]]] = []
        self.slow_bucket_ids: List[Optional[int]] = []
        self.slow_importance: List[float] = []
        self.slow_capture_tag = array("d")
        self.slow_tag_is_strong = array("b")
        self.slow_local_prp = array("d")
        self.slow_last_capture_token: List[int] = []
        self.slow_consolidation_level: List[float] = []
        self.slow_consolidation_events: List[int] = []
        self.slow_entry_timestamps = array("q")
        self.slow_last_replay_token: List[int] = []
        self.slow_replay_count: List[int] = []
        # Awake ripple priority strength (0=no tag, 0.5-1.0=3-5x replay boost).
        self.slow_ripple_strength = array("d")
        self.fast_ema: Optional[torch.Tensor] = None
        self.local_fast_ema: dict[int, torch.Tensor] = {}
        self.local_slow_mean: dict[int, torch.Tensor] = {}
        self.local_weight_sums: dict[int, float] = defaultdict(float)
        self.local_mean_tokens: dict[int, int] = {}

        self.global_prp_pool = 0.0
        self.bucket_prp_pool: dict[int, float] = defaultdict(float)
        self._state_token = 0
        self.update_calls = 0
        self.admission_count = 0
        self.reservoir_rejection_count = 0
        self.optional_payload_copy_count = 0
        self.optional_payload_copy_avoidance_count = 0
        self.ripple_scalar_scan_count = 0
        self.ripple_vector_scan_count = 0
        self.ripple_awake_bucket_scan_count = 0
        self.ripple_awake_bucket_candidate_count = 0
        self.last_ripple_awake_bucket_count = 0
        self.last_ripple_awake_candidate_count = 0
        self.last_ripple_scan_mode = "not_run"
        self.last_awake_ripple_tag_report = self._empty_awake_ripple_tag_report()
        self.last_replay_selection_report = self._empty_replay_selection_report()
        self.last_replay_recall_report = self._empty_replay_recall_report()
        self.last_sfa_sample_report = self._empty_sfa_sample_report()
        self.last_replay_query_collection_report = (
            self._empty_replay_query_collection_report()
        )
        self.last_query_memory_match_report = self._empty_query_memory_match_report()
        self.last_bank_memory_match_report = self._empty_bank_memory_match_report()
        self.last_runtime_concept_memory_lookup_report = (
            self._empty_runtime_concept_memory_lookup_report()
        )
        self.last_frontier_gap_collection_report = (
            self._empty_frontier_gap_collection_report()
        )
        self.last_recent_memory_window_report = (
            self._empty_recent_memory_window_report()
        )
        self.last_recent_memory_tag_report = self._empty_recent_memory_tag_report()
        self.last_anchor_capture_report = self._empty_anchor_capture_report()

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
        self.slow_metadata = []
        self.slow_bucket_ids = []
        self.slow_importance = []
        self.slow_capture_tag = array("d")
        self.slow_tag_is_strong = array("b")
        self.slow_local_prp = array("d")
        self.slow_last_capture_token = []
        self.slow_consolidation_level = []
        self.slow_consolidation_events = []
        self.slow_entry_timestamps = array("q")
        self.slow_last_replay_token = []
        self.slow_replay_count = []
        self.slow_ripple_strength = array("d")
        self.fast_ema = None
        self.local_fast_ema = {}
        self.local_slow_mean = {}
        self.local_weight_sums = defaultdict(float)
        self.local_mean_tokens = {}
        self.global_prp_pool = 0.0
        self.bucket_prp_pool = defaultdict(float)
        self._state_token = 0
        self.update_calls = 0
        self.admission_count = 0
        self.reservoir_rejection_count = 0
        self.optional_payload_copy_count = 0
        self.optional_payload_copy_avoidance_count = 0
        self.ripple_scalar_scan_count = 0
        self.ripple_vector_scan_count = 0
        self.ripple_awake_bucket_scan_count = 0
        self.ripple_awake_bucket_candidate_count = 0
        self.last_ripple_awake_bucket_count = 0
        self.last_ripple_awake_candidate_count = 0
        self.last_ripple_scan_mode = "not_run"
        self.last_awake_ripple_tag_report = self._empty_awake_ripple_tag_report()
        self.last_replay_selection_report = self._empty_replay_selection_report()
        self.last_replay_recall_report = self._empty_replay_recall_report()
        self.last_sfa_sample_report = self._empty_sfa_sample_report()
        self.last_replay_query_collection_report = (
            self._empty_replay_query_collection_report()
        )
        self.last_query_memory_match_report = self._empty_query_memory_match_report()
        self.last_bank_memory_match_report = self._empty_bank_memory_match_report()
        self.last_runtime_concept_memory_lookup_report = (
            self._empty_runtime_concept_memory_lookup_report()
        )
        self.last_frontier_gap_collection_report = (
            self._empty_frontier_gap_collection_report()
        )
        self.last_recent_memory_window_report = (
            self._empty_recent_memory_window_report()
        )
        self.last_recent_memory_tag_report = self._empty_recent_memory_tag_report()
        self.last_anchor_capture_report = self._empty_anchor_capture_report()
        self._slow_mean = None
        self._slow_weight_sum = 0.0
        self._slow_mean_token = None
        self.n_seen = 0
        self._cached_summary = None
        self._cached_summary_token = -1
        self._bucket_entry_indices = defaultdict(list)
        self._recent_entry_indices = []
        self._invalidate_bucket_consolidation_cache()

    def _invalidate_summary_cache(self) -> None:
        self._cached_summary = None
        self._cached_summary_token = -1

    def _last_report_summary_fields(self) -> dict[str, Any]:
        return {
            "last_replay_selection_report": dict(
                self.last_replay_selection_report
            ),
            "last_replay_recall_report": dict(
                self.last_replay_recall_report
            ),
            "last_sfa_sample_report": dict(
                self.last_sfa_sample_report
            ),
            "last_replay_query_collection_report": dict(
                self.last_replay_query_collection_report
            ),
            "last_query_memory_match_report": dict(
                self.last_query_memory_match_report
            ),
            "last_bank_memory_match_report": dict(
                self.last_bank_memory_match_report
            ),
            "last_runtime_concept_memory_lookup_report": dict(
                self.last_runtime_concept_memory_lookup_report
            ),
            "last_frontier_gap_collection_report": dict(
                self.last_frontier_gap_collection_report
            ),
            "last_awake_ripple_tag_report": dict(
                self.last_awake_ripple_tag_report
            ),
            "last_recent_memory_window_report": dict(
                self.last_recent_memory_window_report
            ),
            "last_recent_memory_tag_report": dict(
                self.last_recent_memory_tag_report
            ),
            "last_anchor_capture_report": dict(
                self.last_anchor_capture_report
            ),
        }

    def _cached_summary_float(self, key: str, default: float = 0.0) -> float:
        cached = self._cached_summary if isinstance(self._cached_summary, Mapping) else {}
        try:
            return float(cached.get(key, default) or default)
        except (TypeError, ValueError):
            return float(default)

    def live_summary_stats(self, current_token: Optional[int] = None) -> dict[str, Any]:
        """Bounded read-only projection for trainer/service/status hot surfaces."""

        token_marker = self._state_token if current_token is None else int(current_token)
        size = len(self.slow_buffer)
        cached_available = isinstance(self._cached_summary, Mapping)
        result = {
            "capacity": int(self.capacity),
            "size": int(size),
            "fill_fraction": float(size / max(1, self.capacity)),
            "fill_ratio": float(size / max(1, self.capacity)),
            "n_seen": int(self.n_seen),
            "total_stored": int(self.admission_count),
            "total_evicted": int(self.reservoir_rejection_count),
            "mean_importance": self._cached_summary_float("mean_importance"),
            "mean_capture_tag": self._cached_summary_float("mean_capture_tag"),
            "mean_prp_level": self._cached_summary_float("mean_prp_level"),
            "mean_capture_strength": self._cached_summary_float(
                "mean_capture_strength"
            ),
            "max_capture_strength": self._cached_summary_float(
                "max_capture_strength"
            ),
            "mean_consolidation_level": self._cached_summary_float(
                "mean_consolidation_level"
            ),
            "mean_confidence": self._cached_summary_float(
                "mean_capture_strength"
            ),
            "mean_fragility": self._cached_summary_float("mean_fragility"),
            "max_fragility": self._cached_summary_float("max_fragility"),
            "mean_replay_count": self._cached_summary_float("mean_replay_count"),
            "strong_tag_fraction": self._cached_summary_float(
                "strong_tag_fraction"
            ),
            "mean_ripple_strength": self._cached_summary_float(
                "mean_ripple_strength"
            ),
            "max_ripple_strength": self._cached_summary_float(
                "max_ripple_strength"
            ),
            "global_prp_pool": float(self.global_prp_pool),
            "active_prp_buckets": int(len(self.bucket_prp_pool)),
            "fast_ema_norm": float(torch.norm(self.fast_ema).item())
            if isinstance(self.fast_ema, torch.Tensor)
            else 0.0,
            "slow_mean_norm": float(torch.norm(self._slow_mean).item())
            if isinstance(self._slow_mean, torch.Tensor)
            else 0.0,
            "drift": float(self.compute_drift()),
            "ripple_scalar_scan_count": int(self.ripple_scalar_scan_count),
            "ripple_vector_scan_count": int(self.ripple_vector_scan_count),
            "ripple_awake_bucket_scan_count": int(
                self.ripple_awake_bucket_scan_count
            ),
            "ripple_awake_bucket_candidate_count": int(
                self.ripple_awake_bucket_candidate_count
            ),
            "last_ripple_awake_bucket_count": int(
                self.last_ripple_awake_bucket_count
            ),
            "last_ripple_awake_candidate_count": int(
                self.last_ripple_awake_candidate_count
            ),
            "last_ripple_scan_mode": str(self.last_ripple_scan_mode),
            "summary_surface": "bounded_memory_summary_projection.v1",
            "summary_full_memory_scan": False,
            "summary_scan_entry_count": 0,
            "summary_token_marker": int(token_marker),
            "summary_state_token": int(self._state_token),
            "summary_cached_full_available": bool(cached_available),
            "summary_mean_fields_source": (
                "cached_full_summary" if cached_available else "zero_until_full_summary"
            ),
            "summary_projection_read_only": True,
            "summary_projection_reason": (
                "trainer_service_status_hot_path_must_not_advance_or_scan_slow_memory"
            ),
            "provenance_distribution": {},
        }
        result.update(self._last_report_summary_fields())
        return result

    def _remove_bucket_entry_index(
        self,
        bucket_id: Optional[int],
        index: int,
    ) -> None:
        if bucket_id is None:
            return
        bucket = int(bucket_id)
        entries = self._bucket_entry_indices.get(bucket)
        if entries is None:
            return
        target = int(index)
        entries[:] = [int(item) for item in entries if int(item) != target]
        if not entries:
            self._bucket_entry_indices.pop(bucket, None)

    def _add_bucket_entry_index(
        self,
        bucket_id: Optional[int],
        index: int,
    ) -> None:
        if bucket_id is None:
            return
        bucket = int(bucket_id)
        self._remove_bucket_entry_index(bucket, int(index))
        self._bucket_entry_indices[bucket].append(int(index))

    def _remove_recent_entry_index(self, index: int) -> None:
        target = int(index)
        self._recent_entry_indices = [
            (int(token), int(item))
            for token, item in self._recent_entry_indices
            if int(item) != target
        ]

    def _add_recent_entry_index(self, index: int) -> None:
        idx = int(index)
        if idx < 0 or idx >= len(self.slow_entry_timestamps):
            return
        token_marker = int(self.slow_entry_timestamps[idx])
        self._remove_recent_entry_index(idx)
        insort(self._recent_entry_indices, (token_marker, idx))

    def _rebuild_bucket_entry_index(self) -> None:
        self._bucket_entry_indices = defaultdict(list)
        self._recent_entry_indices = []
        ordered_indices = sorted(
            range(len(self.slow_bucket_ids)),
            key=lambda idx: (
                int(self.slow_entry_timestamps[idx])
                if idx < len(self.slow_entry_timestamps)
                else 0,
                int(idx),
            ),
        )
        for index in ordered_indices:
            bucket_id = self.slow_bucket_ids[index]
            self._add_bucket_entry_index(bucket_id, int(index))
            self._add_recent_entry_index(int(index))

    def _recent_indices_for_window(
        self,
        *,
        current_token: int,
        window_tokens: int,
        max_entries: int,
        require_bucket: bool = False,
    ) -> tuple[list[int], int, bool, int]:
        floor_token = max(0, int(current_token) - int(window_tokens))
        limit = max(0, int(max_entries))
        if limit <= 0 or window_tokens <= 0:
            return [], 0, False, floor_token

        indices: list[int] = []
        observed_available = 0
        truncated = False
        size = len(self.slow_buffer)
        for token_marker, raw_index in reversed(self._recent_entry_indices):
            token = int(token_marker)
            if token < floor_token:
                break
            idx = int(raw_index)
            if idx < 0 or idx >= size:
                continue
            if require_bucket and (
                idx >= len(self.slow_bucket_ids) or self.slow_bucket_ids[idx] is None
            ):
                continue
            observed_available += 1
            if len(indices) >= limit:
                truncated = True
                break
            indices.append(idx)
        return indices, observed_available, truncated, floor_token

    def _invalidate_bucket_consolidation_cache(self) -> None:
        self._bucket_consolidation_cache_generation += 1
        self._bucket_consolidation_cpu = None
        self._bucket_consolidation_weighted_sum = None
        self._bucket_consolidation_weight_sum = None
        self._bucket_consolidation_devices = {}

    def _ensure_bucket_consolidation_cache_size(self, n_buckets: int) -> None:
        size = max(0, int(n_buckets))
        current = 0 if self._bucket_consolidation_cpu is None else int(
            self._bucket_consolidation_cpu.numel()
        )
        if current >= size:
            return
        next_cpu = torch.zeros(size, dtype=torch.float32)
        next_weighted = torch.zeros(size, dtype=torch.float32)
        next_weight = torch.zeros(size, dtype=torch.float32)
        if self._bucket_consolidation_cpu is not None and current > 0:
            next_cpu[:current].copy_(self._bucket_consolidation_cpu)
        if self._bucket_consolidation_weighted_sum is not None and current > 0:
            next_weighted[:current].copy_(self._bucket_consolidation_weighted_sum)
        if self._bucket_consolidation_weight_sum is not None and current > 0:
            next_weight[:current].copy_(self._bucket_consolidation_weight_sum)
        self._bucket_consolidation_cpu = next_cpu
        self._bucket_consolidation_weighted_sum = next_weighted
        self._bucket_consolidation_weight_sum = next_weight
        next_devices: dict[str, torch.Tensor] = {"cpu": next_cpu}
        for key, cached in list(self._bucket_consolidation_devices.items()):
            if key == "cpu":
                continue
            next_cached = torch.zeros(size, dtype=torch.float32, device=cached.device)
            if current > 0:
                next_cached[:current].copy_(cached[:current])
            next_devices[key] = next_cached
        self._bucket_consolidation_devices = next_devices
        self._bucket_consolidation_cache_generation += 1

    def _rebuild_bucket_consolidation_cache(
        self,
        *,
        n_buckets: int | None = None,
        reason: str = "explicit_rebuild",
    ) -> None:
        if n_buckets is None:
            observed = [
                int(bucket_id)
                for bucket_id in self.slow_bucket_ids
                if bucket_id is not None and int(bucket_id) >= 0
            ]
            size = 0 if not observed else max(observed) + 1
        else:
            size = max(0, int(n_buckets))
        weighted_sum = torch.zeros(size, dtype=torch.float32)
        weight_sum = torch.zeros(size, dtype=torch.float32)
        scan_count = 0
        for bucket_id, importance, consolidation in zip(
            self.slow_bucket_ids,
            self.slow_importance,
            self.slow_consolidation_level,
        ):
            scan_count += 1
            if bucket_id is None:
                continue
            bucket = int(bucket_id)
            if 0 <= bucket < size:
                weight = max(1e-6, float(importance))
                weighted_sum[bucket] += (
                    weight * max(0.0, min(1.0, float(consolidation)))
                )
                weight_sum[bucket] += weight
        snapshot = weighted_sum / weight_sum.clamp(min=1e-8)
        snapshot = torch.where(
            weight_sum > 0.0,
            snapshot,
            torch.zeros_like(snapshot),
        )
        self._bucket_consolidation_cpu = snapshot
        self._bucket_consolidation_weighted_sum = weighted_sum
        self._bucket_consolidation_weight_sum = weight_sum
        self._bucket_consolidation_devices = {"cpu": snapshot}
        self.bucket_consolidation_cache_rebuild_count += 1
        self.bucket_consolidation_cache_rebuild_scan_entry_count += int(scan_count)
        self.last_bucket_consolidation_level_report = {
            "surface": "bucket_consolidation_level_cache_lookup.v1",
            "status": f"cache_rebuilt_{reason}",
            "bucket_id": None,
            "level": 0.0,
            "cache_hit": False,
            "full_memory_scan": True,
            "scan_entry_count": int(scan_count),
            "cache_generation": int(self._bucket_consolidation_cache_generation),
        }

    @property
    def bucket_consolidation_cache_generation(self) -> int:
        """Monotonic identity for graph-safe bucket-consolidation cache reuse."""

        return int(self._bucket_consolidation_cache_generation)

    @staticmethod
    def _empty_bucket_candidate_source_fields() -> dict[str, Any]:
        return {
            "candidate_source_window_policy": "not_run",
            "candidate_source_bucket_count": 0,
            "candidate_source_entry_available_count": 0,
            "candidate_source_entry_available_count_is_lower_bound": False,
            "candidate_source_entry_read_count": 0,
            "candidate_source_entry_read_budget": 0,
            "candidate_source_entry_read_budget_exhausted": False,
            "candidate_source_materialized_entry_count": 0,
            "candidate_source_materialization_count": 0,
            "candidate_source_full_bucket_scan": False,
            "candidate_source_full_bucket_materialization": False,
            "candidate_source_device": "cpu",
        }

    @staticmethod
    def _bucket_candidate_source_fields(
        window: _BucketCandidateWindow,
    ) -> dict[str, Any]:
        return {
            "candidate_source_window_policy": (
                _BUCKET_CANDIDATE_SOURCE_WINDOW_POLICY
                if window.normalized_bucket_ids is not None
                else "bucket_scope_required_no_source_window"
            ),
            "candidate_source_bucket_count": int(
                len(window.normalized_bucket_ids or [])
            ),
            "candidate_source_entry_available_count": int(window.available_count),
            "candidate_source_entry_available_count_is_lower_bound": bool(
                window.available_count_is_lower_bound
            ),
            "candidate_source_entry_read_count": int(
                window.source_entry_read_count
            ),
            "candidate_source_entry_read_budget": int(
                window.source_entry_read_budget
            ),
            "candidate_source_entry_read_budget_exhausted": bool(
                window.source_entry_read_budget_exhausted
            ),
            "candidate_source_materialized_entry_count": int(
                window.source_materialized_entry_count
            ),
            "candidate_source_materialization_count": int(
                window.source_materialization_count
            ),
            "candidate_source_full_bucket_scan": bool(
                window.source_full_bucket_scan
            ),
            "candidate_source_full_bucket_materialization": bool(
                window.source_materialized_entry_count > 0
                or window.source_materialization_count > 0
            ),
            "candidate_source_device": "cpu",
        }

    @staticmethod
    def _empty_replay_selection_report() -> dict[str, Any]:
        return {
            "surface": "bounded_replay_window_selection.v1",
            "status": "not_run",
            "scope": "sleep_slow_path",
            "strategy": "not_run",
            "runs_live_tick": False,
            "records_replay_artifact": False,
            "raw_text_payload_loaded": False,
            "language_reasoning": False,
            "mutates_runtime_state": False,
            "applies_plasticity": False,
            "score_device": "cpu",
            "archival_storage_device": "cpu",
            "selected_indices": [],
            "selected_count": 0,
            "candidate_window_limit": 0,
            "candidate_window_policy": "not_run",
            "candidate_index_available_count": 0,
            "score_count": 0,
            **DualMemoryStore._empty_bucket_candidate_source_fields(),
            "global_score_scan": False,
            "global_candidate_scan": False,
            "diagnostic_global_score_scan": False,
            "diagnostic_global_candidate_scan": False,
            "bounded_by_bucket_index": False,
            "fallback_reason": "not_run",
        }

    @staticmethod
    def _empty_replay_recall_report() -> dict[str, Any]:
        return {
            "surface": "bounded_replay_window_recall.v1",
            "status": "not_run",
            "scope": "replay_recall_slow_path",
            "runs_live_tick": False,
            "raw_text_payload_loaded": False,
            "language_reasoning": False,
            "mutates_runtime_state": False,
            "applies_plasticity": False,
            "score_device": "cpu",
            "archival_storage_device": "cpu",
            "candidate_scope": "not_run",
            "selected_indices": [],
            "selected_count": 0,
            "routing_key_count": 0,
            "input_pattern_count": 0,
            "best_distance": None,
            "best_input_distance": None,
            "recalled_distance": None,
            "fallback_reason": "not_run",
        }

    @staticmethod
    def _empty_sfa_sample_report() -> dict[str, Any]:
        return {
            "surface": "bounded_sfa_sample.v1",
            "status": "not_run",
            "scope": "sfa_correction_slow_path",
            "memory_size": 0,
            "requested_count": 0,
            "candidate_scope": "not_run",
            "candidate_index_count": 0,
            "candidate_indices": [],
            "sample_indices": [],
            "sample_count": 0,
            "sample_device": "cpu",
            "archival_storage_device": "cpu",
            "global_candidate_scan": False,
            "diagnostic_global_candidate_scan": False,
            "runs_live_tick": False,
            "runs_every_token": False,
            "raw_text_payload_loaded": False,
            "language_reasoning": False,
            "mutates_runtime_state": False,
            "applies_plasticity": False,
            "fallback_reason": "not_run",
        }

    @staticmethod
    def _empty_replay_query_collection_report() -> dict[str, Any]:
        return {
            "surface": "bounded_replay_query_collection.v1",
            "status": "not_run",
            "scope": "replay_query_collection_slow_path",
            "memory_size": 0,
            "requested_count": 0,
            "candidate_window_limit": 0,
            "candidate_window_policy": "not_run",
            "candidate_scope": "not_run",
            "candidate_bucket_ids": [],
            "candidate_bucket_count": 0,
            "candidate_index_available_count": 0,
            "candidate_index_count": 0,
            "query_indices": [],
            "query_count": 0,
            "score_count": 0,
            **DualMemoryStore._empty_bucket_candidate_source_fields(),
            "global_score_scan": False,
            "global_candidate_scan": False,
            "runs_live_tick": False,
            "mutates_runtime_state": False,
            "applies_plasticity": False,
            "archival_storage_device": "cpu",
            "fallback_reason": "not_run",
        }

    @staticmethod
    def _empty_query_memory_match_report() -> dict[str, Any]:
        return {
            "surface": "bounded_query_memory_match_candidates.v1",
            "status": "not_run",
            "scope": "query_memory_match_slow_path",
            "memory_size": 0,
            "requested_count": 0,
            "candidate_window_limit": 0,
            "candidate_window_policy": "not_run",
            "candidate_scope": "not_run",
            "candidate_bucket_ids": [],
            "candidate_bucket_count": 0,
            "candidate_index_available_count": 0,
            "candidate_index_count": 0,
            "match_indices": [],
            "score_count": 0,
            **DualMemoryStore._empty_bucket_candidate_source_fields(),
            "global_score_scan": False,
            "global_candidate_scan": False,
            "runs_live_tick": False,
            "mutates_runtime_state": False,
            "applies_plasticity": False,
            "archival_storage_device": "cpu",
            "fallback_reason": "not_run",
        }

    @staticmethod
    def _empty_bank_memory_match_report() -> dict[str, Any]:
        return {
            "surface": "bounded_source_bank_memory_match.v1",
            "status": "not_run",
            "scope": "source_bank_semantic_recall_slow_path",
            "bank_name": "",
            "memory_size": 0,
            "requested_probe_count": 0,
            "probe_count": 0,
            "probe_indices": [],
            "memories_per_probe": 0,
            "max_matches": 0,
            "candidate_surface": "bounded_query_memory_match.v1",
            "candidate_window_policy": "not_run",
            "candidate_scope": "not_run",
            "candidate_bucket_ids": [],
            "candidate_bucket_count": 0,
            "candidate_index_available_count": 0,
            "candidate_index_count": 0,
            "unique_candidate_index_count": 0,
            "similarity_score_count": 0,
            "replay_priority_score_count": 0,
            "match_indices": [],
            "result_count": 0,
            "returned_count": 0,
            "raw_text_payload_loaded": False,
            "raw_text_payload_count": 0,
            "raw_text_payload_cache_hits": 0,
            "raw_text_payload_policy": "shared_returned_similarity_matches_only",
            "global_score_scan": False,
            "global_candidate_scan": False,
            "runs_live_tick": False,
            "runs_every_token": False,
            "mutates_runtime_state": False,
            "applies_plasticity": False,
            "language_reasoning": False,
            "score_device": "cpu",
            "archival_storage_device": "cpu",
            "quality_metric": "semantic_grounding_gap_inputs",
            "latency_ms": 0.0,
            "fallback_reason": "not_run",
            "selection_budget": {
                "memory_budget_entries": 0,
                "probe_budget": 0,
                "per_probe_return_budget": 0,
                "returned_match_limit": 0,
                "raw_text_payload_policy": "shared_returned_similarity_matches_only",
            },
            "probe_reports": [],
        }

    @staticmethod
    def _empty_runtime_concept_memory_lookup_report() -> dict[str, Any]:
        return {
            "surface": "bounded_runtime_concept_memory_lookup.v1",
            "status": "not_run",
            "scope": "cadenced_runtime_concept_observation",
            "memory_size": 0,
            "input_observation_count": 0,
            "processed_observation_count": 0,
            "truncated_observation_count": 0,
            "max_observation_count": 0,
            "candidate_window_limit": 0,
            "candidate_window_policy": "not_run",
            "candidate_scope": "not_run",
            "candidate_index_count": 0,
            "unique_candidate_index_count": 0,
            "match_indices": [],
            "match_count": 0,
            "unique_match_index_count": 0,
            "raw_text_payload_loaded": False,
            "raw_text_payload_count": 0,
            "raw_text_payload_cache_hits": 0,
            "invalid_observation_count": 0,
            "invalid_memory_index_count": 0,
            "out_of_bounds_index_count": 0,
            "missing_routing_key_count": 0,
            "empty_text_count": 0,
            "score_count": 0,
            "global_score_scan": False,
            "global_candidate_scan": False,
            "runs_live_tick": True,
            "runs_every_token": False,
            "cadenced_observation": True,
            "mutates_runtime_state": False,
            "applies_plasticity": False,
            "language_reasoning": False,
            "score_device": "cpu",
            "archival_storage_device": "cpu",
            "quality_metric": "runtime_concept_observation_match_parity",
            "latency_ms": 0.0,
            "fallback_reason": "not_run",
            "selection_budget": {
                "memory_budget_entries": 0,
                "candidate_window_entries": 0,
                "returned_match_limit": 0,
                "raw_text_payload_policy": "cached_explicit_index_payloads_only",
            },
        }

    @staticmethod
    def _empty_frontier_gap_collection_report() -> dict[str, Any]:
        return {
            "surface": "bounded_frontier_gap_candidates.v1",
            "status": "not_run",
            "scope": "frontier_gap_planner_slow_path",
            "memory_size": 0,
            "current_token": 0,
            "requested_count": 0,
            "candidate_window_limit": 0,
            "candidate_window_policy": "not_run",
            "candidate_scope": "not_run",
            "candidate_bucket_ids": [],
            "candidate_bucket_count": 0,
            "candidate_index_available_count": 0,
            "candidate_index_available_count_is_lower_bound": False,
            "candidate_index_count": 0,
            "candidate_indices": [],
            **DualMemoryStore._empty_bucket_candidate_source_fields(),
            "global_score_scan": False,
            "global_candidate_scan": False,
            "runs_live_tick": False,
            "raw_text_payload_loaded": False,
            "language_reasoning": False,
            "mutates_runtime_state": False,
            "applies_plasticity": False,
            "archival_storage_device": "cpu",
            "fallback_reason": "not_run",
        }

    @staticmethod
    def _empty_recent_memory_window_report() -> dict[str, Any]:
        return {
            "surface": "bounded_recent_memory_window.v1",
            "status": "not_run",
            "scope": "recent_memory_slow_path",
            "memory_size": 0,
            "current_token": 0,
            "window_tokens": 0,
            "floor_token": 0,
            "requested_count": 0,
            "candidate_window_limit": 0,
            "candidate_window_policy": "not_run",
            "candidate_scope": "not_run",
            "candidate_index_available_count": 0,
            "candidate_index_available_count_is_lower_bound": False,
            "candidate_index_count": 0,
            "candidate_indices": [],
            "requires_bucket": False,
            "global_score_scan": False,
            "global_candidate_scan": False,
            "runs_live_tick": False,
            "mutates_runtime_state": False,
            "applies_plasticity": False,
            "archival_storage_device": "cpu",
            "fallback_reason": "not_run",
        }

    @staticmethod
    def _empty_recent_memory_tag_report() -> dict[str, Any]:
        return {
            "surface": "bounded_recent_memory_tag.v1",
            "status": "not_run",
            "scope": "recent_memory_tagging_slow_path",
            "memory_size": 0,
            "current_token": 0,
            "window_tokens": 0,
            "candidate_window_limit": 0,
            "candidate_index_count": 0,
            "tagged_count": 0,
            "strength": 0.0,
            "global_score_scan": False,
            "global_candidate_scan": False,
            "runs_live_tick": False,
            "mutates_runtime_state": False,
            "applies_plasticity": False,
            "archival_storage_device": "cpu",
            "fallback_reason": "not_run",
        }

    @staticmethod
    def _empty_anchor_capture_report() -> dict[str, Any]:
        return {
            "surface": "bounded_recent_anchor_capture.v1",
            "status": "not_run",
            "scope": "recent_anchor_capture_slow_path",
            "memory_size": 0,
            "current_token": 0,
            "window_tokens": 0,
            "candidate_window_limit": 0,
            "candidate_index_count": 0,
            "captured_entry_count": 0,
            "captured_anchor_count": 0,
            "candidate_bucket_ids": [],
            "strength": 0.0,
            "global_score_scan": False,
            "global_candidate_scan": False,
            "runs_live_tick": False,
            "mutates_runtime_state": False,
            "applies_plasticity": False,
            "archival_storage_device": "cpu",
            "fallback_reason": "not_run",
        }

    @staticmethod
    def _empty_awake_ripple_tag_report() -> dict[str, Any]:
        return {
            "surface": "bounded_awake_ripple_tag.v1",
            "status": "not_run",
            "scope": "awake_ripple_tagging_cadenced_path",
            "memory_size": 0,
            "current_token": 0,
            "window_tokens": 0,
            "floor_token": 0,
            "da_level": 0.0,
            "da_threshold": 0.0,
            "candidate_window_limit": 0,
            "candidate_window_policy": "not_run",
            "candidate_scope": "not_run",
            "candidate_bucket_ids": [],
            "candidate_bucket_count": 0,
            "candidate_index_available_count": 0,
            "candidate_index_count": 0,
            "candidate_indices": [],
            "tagged_count": 0,
            "scan_mode": "not_run",
            **DualMemoryStore._empty_bucket_candidate_source_fields(),
            "global_candidate_scan": False,
            "diagnostic_global_candidate_scan": False,
            "runs_live_tick": True,
            "runs_every_token": False,
            "mutates_runtime_state": False,
            "applies_plasticity": False,
            "archival_storage_device": "cpu",
            "latency_ms": 0.0,
            "fallback_reason": "not_run",
        }

    def _adjust_bucket_consolidation_cache(
        self,
        bucket_id: Optional[int],
        *,
        importance: float,
        consolidation: float,
        sign: float,
    ) -> None:
        if (
            bucket_id is None
        ):
            return
        bucket = int(bucket_id)
        if bucket < 0:
            return
        self._ensure_bucket_consolidation_cache_size(bucket + 1)
        if (
            self._bucket_consolidation_cpu is None
            or self._bucket_consolidation_weighted_sum is None
            or self._bucket_consolidation_weight_sum is None
            or bucket >= int(self._bucket_consolidation_cpu.numel())
        ):
            return
        weight = max(1e-6, float(importance))
        level = max(0.0, min(1.0, float(consolidation)))
        self._bucket_consolidation_weighted_sum[bucket] += (
            float(sign) * weight * level
        )
        self._bucket_consolidation_weight_sum[bucket] += float(sign) * weight
        total_weight = max(
            0.0,
            float(self._bucket_consolidation_weight_sum[bucket].item()),
        )
        value = (
            0.0
            if total_weight <= 1e-8
            else max(
                0.0,
                float(self._bucket_consolidation_weighted_sum[bucket].item()),
            )
            / total_weight
        )
        self._bucket_consolidation_cpu[bucket] = value
        for key, cached in self._bucket_consolidation_devices.items():
            if key != "cpu":
                cached[bucket] = value

    def cached_bucket_consolidation_tensor(
        self,
        n_buckets: int,
        *,
        device: torch.device,
    ) -> torch.Tensor | None:
        size = max(0, int(n_buckets))
        if self._bucket_consolidation_cpu is None:
            return None
        if int(self._bucket_consolidation_cpu.numel()) < size:
            self._ensure_bucket_consolidation_cache_size(size)
        if (
            self._bucket_consolidation_cpu is None
            or int(self._bucket_consolidation_cpu.numel()) != size
        ):
            return None
        key = str(device)
        cached = self._bucket_consolidation_devices.get(key)
        if cached is None:
            cached = self._bucket_consolidation_cpu.to(device)
            self._bucket_consolidation_devices[key] = cached
        return cached

    def bucket_consolidation_tensor(
        self,
        n_buckets: int,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        """Return importance-weighted consolidation per bucket on the compute device."""

        size = max(0, int(n_buckets))
        if (
            self._bucket_consolidation_cpu is None
            or int(self._bucket_consolidation_cpu.numel()) != size
        ):
            self._rebuild_bucket_consolidation_cache(
                n_buckets=size,
                reason="explicit_tensor_request",
            )

        key = str(device)
        cached = self._bucket_consolidation_devices.get(key)
        if cached is None:
            cached = self._bucket_consolidation_cpu.to(device)
            self._bucket_consolidation_devices[key] = cached
        return cached

    @staticmethod
    def _tensor_device_counts(values: Sequence[torch.Tensor | None]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for value in values:
            if isinstance(value, torch.Tensor):
                device = str(value.device)
                counts[device] = counts.get(device, 0) + 1
        return counts

    def device_report(self) -> dict[str, Any]:
        """Report archival memory placement without implying CUDA execution."""
        slow_devices = self._tensor_device_counts(self.slow_buffer)
        input_devices = self._tensor_device_counts(self.slow_input_patterns)
        routing_devices = self._tensor_device_counts(self.slow_routing_keys)
        local_fast_devices = self._tensor_device_counts(list(self.local_fast_ema.values()))
        local_slow_devices = self._tensor_device_counts(list(self.local_slow_mean.values()))
        fast_ema_device = str(self.fast_ema.device) if isinstance(self.fast_ema, torch.Tensor) else None
        slow_mean_device = str(self._slow_mean.device) if isinstance(self._slow_mean, torch.Tensor) else None
        return {
            "storage_role": "archival_replay_ledger",
            "cuda_first_compute_boundary": "replay tensors move to model device when consumed",
            "expected_storage_device": "cpu",
            "slow_buffer_devices": slow_devices,
            "slow_input_pattern_devices": input_devices,
            "slow_routing_key_devices": routing_devices,
            "fast_ema_device": fast_ema_device,
            "slow_mean_device": slow_mean_device,
            "local_fast_ema_devices": local_fast_devices,
            "local_slow_mean_devices": local_slow_devices,
            "bucket_consolidation_cache_devices": sorted(
                self._bucket_consolidation_devices
            ),
            "bucket_consolidation_cache_entries": (
                0
                if self._bucket_consolidation_cpu is None
                else int(self._bucket_consolidation_cpu.numel())
            ),
            "bucket_consolidation_cache_generation": int(
                self._bucket_consolidation_cache_generation
            ),
            "bucket_consolidation_cache_rebuild_count": int(
                self.bucket_consolidation_cache_rebuild_count
            ),
            "bucket_consolidation_cache_rebuild_scan_entry_count": int(
                self.bucket_consolidation_cache_rebuild_scan_entry_count
            ),
            "bucket_consolidation_level_cache_lookup_count": int(
                self.bucket_consolidation_level_cache_lookup_count
            ),
            "bucket_consolidation_level_cache_miss_count": int(
                self.bucket_consolidation_level_cache_miss_count
            ),
            "last_bucket_consolidation_level_report": dict(
                self.last_bucket_consolidation_level_report
            ),
            "stc_state_storage": "zero_copy_array_buffer",
            "stc_decay_zero_copy": True,
            "stc_state_bytes": int(
                self.slow_capture_tag.buffer_info()[1]
                * self.slow_capture_tag.itemsize
                + self.slow_local_prp.buffer_info()[1]
                * self.slow_local_prp.itemsize
                + self.slow_tag_is_strong.buffer_info()[1]
                * self.slow_tag_is_strong.itemsize
            ),
            "hot_path": {
                "update_calls": int(self.update_calls),
                "admission_count": int(self.admission_count),
                "reservoir_rejection_count": int(self.reservoir_rejection_count),
                "optional_payload_copy_count": int(self.optional_payload_copy_count),
                "optional_payload_copy_avoidance_count": int(
                    self.optional_payload_copy_avoidance_count
                ),
                "ripple_scalar_scan_count": int(self.ripple_scalar_scan_count),
                "ripple_vector_scan_count": int(self.ripple_vector_scan_count),
                "ripple_awake_bucket_scan_count": int(
                    self.ripple_awake_bucket_scan_count
                ),
                "ripple_awake_bucket_candidate_count": int(
                    self.ripple_awake_bucket_candidate_count
                ),
                "last_ripple_awake_bucket_count": int(
                    self.last_ripple_awake_bucket_count
                ),
                "last_ripple_awake_candidate_count": int(
                    self.last_ripple_awake_candidate_count
                ),
                "last_ripple_scan_mode": str(self.last_ripple_scan_mode),
            },
            "last_awake_ripple_tag_report": dict(
                self.last_awake_ripple_tag_report
            ),
            "last_replay_selection_report": dict(
                self.last_replay_selection_report
            ),
            "last_replay_recall_report": dict(
                self.last_replay_recall_report
            ),
            "last_sfa_sample_report": dict(
                self.last_sfa_sample_report
            ),
            "last_replay_query_collection_report": dict(
                self.last_replay_query_collection_report
            ),
            "last_query_memory_match_report": dict(
                self.last_query_memory_match_report
            ),
            "last_bank_memory_match_report": dict(
                self.last_bank_memory_match_report
            ),
            "last_runtime_concept_memory_lookup_report": dict(
                self.last_runtime_concept_memory_lookup_report
            ),
            "last_frontier_gap_collection_report": dict(
                self.last_frontier_gap_collection_report
            ),
            "last_recent_memory_window_report": dict(
                self.last_recent_memory_window_report
            ),
            "last_recent_memory_tag_report": dict(
                self.last_recent_memory_tag_report
            ),
            "last_anchor_capture_report": dict(
                self.last_anchor_capture_report
            ),
            "all_archival_tensors_cpu": all(
                device == "cpu"
                for counts in (
                    slow_devices,
                    input_devices,
                    routing_devices,
                    local_fast_devices,
                    local_slow_devices,
                )
                for device in counts
            )
            and (fast_ema_device in (None, "cpu"))
            and (slow_mean_device in (None, "cpu")),
        }

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

        size = len(self.slow_buffer)
        if size > 0:
            minute_delta = float(delta) / max(1.0, float(self.functional_minute))
            tag_tau_weak = self._tag_tau_tokens(False)
            tag_tau_strong = self._tag_tau_tokens(True)
            prp_tau_weak = self._prp_tau_tokens(False)
            prp_tau_strong = self._prp_tau_tokens(True)

            tag_decay_weak = math.exp(-float(delta) / tag_tau_weak)
            tag_decay_strong = math.exp(-float(delta) / tag_tau_strong)
            prp_decay_weak = math.exp(-float(delta) / prp_tau_weak)
            prp_decay_strong = math.exp(-float(delta) / prp_tau_strong)

            if self.capture_tag_decay < 1.0:
                extra = self.capture_tag_decay ** minute_delta
                tag_decay_weak *= extra
                tag_decay_strong *= extra

            tags = np.frombuffer(
                self.slow_capture_tag,
                dtype=np.float64,
                count=size,
            )
            prps = np.frombuffer(
                self.slow_local_prp,
                dtype=np.float64,
                count=size,
            )
            strong = np.frombuffer(
                self.slow_tag_is_strong,
                dtype=np.int8,
                count=size,
            )
            tags *= np.where(strong, tag_decay_strong, tag_decay_weak)
            prps *= np.where(strong, prp_decay_strong, prp_decay_weak)
            np.maximum(tags, 0.0, out=tags)
            np.maximum(prps, 0.0, out=prps)

        global_decay = math.exp(-float(delta) / self._prp_tau_tokens(True))
        self.global_prp_pool = float(max(0.0, self.global_prp_pool * global_decay))
        for bucket in list(self.bucket_prp_pool.keys()):
            decayed = self.bucket_prp_pool[bucket] * global_decay
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

    def fragility_score(self, idx: int, current_token: int) -> float:
        if idx < 0 or idx >= len(self.slow_buffer):
            return 0.0

        self._advance_state(current_token)
        consolidation = float(max(0.0, min(1.0, self.slow_consolidation_level[idx])))
        importance = float(max(1e-6, self.slow_importance[idx]))
        replay_age = max(0, int(current_token) - int(self.slow_last_replay_token[idx]))
        replay_count = max(0, int(self.slow_replay_count[idx]))
        tag_strength = float(max(0.0, self.slow_capture_tag[idx]))
        capture_strength = float(max(0.0, tag_strength * self._available_prp(idx)))

        age_pressure = 1.0 + math.log1p(float(replay_age) / max(1.0, float(self.functional_minute)))
        access_penalty = 1.0 / (1.0 + 0.5 * float(replay_count))
        stability_gap = max(0.0, 1.0 - consolidation)
        capture_gap = max(0.0, 1.0 - capture_strength)
        importance_scale = 0.5 + min(1.0, importance)
        return float(stability_gap * age_pressure * access_penalty * importance_scale * (0.5 + capture_gap))

    def bucket_consolidation_level(self, bucket_id: Optional[int]) -> float:
        if bucket_id is None or not self.slow_buffer:
            self.last_bucket_consolidation_level_report = {
                "surface": "bucket_consolidation_level_cache_lookup.v1",
                "status": "empty_or_missing_bucket",
                "bucket_id": None if bucket_id is None else int(bucket_id),
                "level": 0.0,
                "cache_hit": False,
                "full_memory_scan": False,
                "scan_entry_count": 0,
                "cache_generation": int(self._bucket_consolidation_cache_generation),
            }
            return 0.0

        bucket = int(bucket_id)
        self.bucket_consolidation_level_cache_lookup_count += 1
        cache = self._bucket_consolidation_cpu
        if cache is None or bucket < 0 or bucket >= int(cache.numel()):
            self.bucket_consolidation_level_cache_miss_count += 1
            self.last_bucket_consolidation_level_report = {
                "surface": "bucket_consolidation_level_cache_lookup.v1",
                "status": "cache_missing_no_scan",
                "bucket_id": bucket,
                "level": 0.0,
                "cache_hit": False,
                "full_memory_scan": False,
                "scan_entry_count": 0,
                "cache_generation": int(self._bucket_consolidation_cache_generation),
            }
            return 0.0
        value = float(cache[bucket].item())
        self.last_bucket_consolidation_level_report = {
            "surface": "bucket_consolidation_level_cache_lookup.v1",
            "status": "cache_hit",
            "bucket_id": bucket,
            "level": value,
            "cache_hit": True,
            "full_memory_scan": False,
            "scan_entry_count": 0,
            "cache_generation": int(self._bucket_consolidation_cache_generation),
        }
        return value

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

    def _store_slot(
        self,
        index: int,
        *,
        assembly: torch.Tensor,
        stored_input: Optional[torch.Tensor],
        stored_routing: Optional[torch.Tensor],
        stored_window: Optional[str],
        stored_text: Optional[str],
        stored_metadata: Optional[dict[str, Any]],
        bucket_id: Optional[int],
        importance: float,
        capture_value: float,
        token_marker: int,
    ) -> None:
        old_bucket_id = self.slow_bucket_ids[index]
        old_importance = self.slow_importance[index]
        old_consolidation = self.slow_consolidation_level[index]
        new_bucket_id = int(bucket_id) if bucket_id is not None else None
        self._adjust_bucket_consolidation_cache(
            old_bucket_id,
            importance=old_importance,
            consolidation=old_consolidation,
            sign=-1.0,
        )
        self._remove_bucket_entry_index(old_bucket_id, index)
        self._remove_recent_entry_index(index)
        strong_event = self._is_strong_event(capture_value, importance)
        injected_prp = self._inject_prp(bucket_id=bucket_id, strength=capture_value, importance=importance)
        local_prp = 0.20 * injected_prp if strong_event else 0.0
        tag_value = float(max(0.0, capture_value if capture_value > 0.0 else importance))

        self.slow_buffer[index] = assembly
        self.slow_input_patterns[index] = stored_input
        self.slow_routing_keys[index] = stored_routing
        self.slow_raw_windows[index] = stored_window
        self.slow_texts[index] = stored_text
        self.slow_metadata[index] = None if stored_metadata is None else {str(key): value for key, value in dict(stored_metadata).items()}
        self.slow_bucket_ids[index] = new_bucket_id
        self.slow_importance[index] = float(max(1e-6, importance))
        self.slow_capture_tag[index] = tag_value
        self.slow_tag_is_strong[index] = bool(strong_event)
        self.slow_local_prp[index] = float(max(0.0, local_prp))
        self.slow_last_capture_token[index] = int(token_marker)
        self.slow_consolidation_level[index] = 0.0
        self._adjust_bucket_consolidation_cache(
            bucket_id,
            importance=importance,
            consolidation=0.0,
            sign=1.0,
        )
        self.slow_consolidation_events[index] = 0
        self.slow_entry_timestamps[index] = int(token_marker)
        self.slow_last_replay_token[index] = int(token_marker)
        self.slow_replay_count[index] = 0
        self.slow_ripple_strength[index] = 0.0
        self._add_bucket_entry_index(new_bucket_id, index)
        self._add_recent_entry_index(index)
        self._invalidate_summary_cache()

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
        metadata: Mapping[str, Any] | None = None,
        tag_strength: float = 0.0,
        capture_tag: float | None = None,
    ) -> int | None:
        self.update_calls += 1
        x = assembly.detach().clone().cpu()
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
            admission_index = len(self.slow_buffer)
        else:
            candidate_index = int(torch.randint(0, self.n_seen, (1,)).item())
            if candidate_index >= self.capacity:
                self.reservoir_rejection_count += 1
                self.optional_payload_copy_avoidance_count += int(
                    input_pattern is not None
                ) + int(routing_key is not None)
                return None
            admission_index = candidate_index

        stored_input = (
            input_pattern.detach().clone().cpu()
        ) if input_pattern is not None else None
        stored_routing = (
            routing_key.detach().clone().cpu()
        ) if routing_key is not None else None
        self.optional_payload_copy_count += int(stored_input is not None) + int(
            stored_routing is not None
        )
        stored_window = None if raw_window is None else str(raw_window)
        stored_text = None if text is None else str(text)
        stored_metadata = None if metadata is None else {
            str(key): value for key, value in dict(metadata).items()
        }
        self.admission_count += 1

        if admission_index == len(self.slow_buffer):
            self.slow_buffer.append(x)
            self.slow_input_patterns.append(stored_input)
            self.slow_routing_keys.append(stored_routing)
            self.slow_raw_windows.append(stored_window)
            self.slow_texts.append(stored_text)
            self.slow_metadata.append(None if stored_metadata is None else {str(key): value for key, value in dict(stored_metadata).items()})
            self.slow_bucket_ids.append(None)
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
            self.slow_ripple_strength.append(0.0)
            self._store_slot(
                len(self.slow_buffer) - 1,
                assembly=x,
                stored_input=stored_input,
                stored_routing=stored_routing,
                stored_window=stored_window,
                stored_text=stored_text,
                stored_metadata=stored_metadata,
                bucket_id=bucket_id,
                importance=importance,
                capture_value=capture_value,
                token_marker=token_marker,
            )
            self._append_to_slow_mean(x)
            return len(self.slow_buffer) - 1

        old = self.slow_buffer[admission_index]
        old_timestamp = self.slow_entry_timestamps[admission_index]
        self._store_slot(
            admission_index,
            assembly=x,
            stored_input=stored_input,
            stored_routing=stored_routing,
            stored_window=stored_window,
            stored_text=stored_text,
            stored_metadata=stored_metadata,
            bucket_id=bucket_id,
            importance=importance,
            capture_value=capture_value,
            token_marker=token_marker,
        )
        self._replace_in_slow_mean(old, old_timestamp, x, token_marker)
        return admission_index

    @staticmethod
    def _clip_ripple_strength(value: float) -> float:
        return float(max(0.0, min(1.0, value)))

    @classmethod
    def _ripple_priority_multiplier(cls, strength: float) -> float:
        clipped = cls._clip_ripple_strength(strength)
        if clipped <= 0.0:
            return 1.0
        # Research-facing retune: ripple tags now map onto a 3-5x replay boost
        # instead of a flat 3x multiplier.
        return float(3.0 + 2.0 * max(0.0, clipped - 0.5) / 0.5)

    @staticmethod
    def _normalise_awake_bucket_ids(
        awake_bucket_ids: Sequence[int] | torch.Tensor | None,
    ) -> list[int] | None:
        if awake_bucket_ids is None:
            return None
        if isinstance(awake_bucket_ids, torch.Tensor):
            raw_values = awake_bucket_ids.detach().flatten().cpu().tolist()
        else:
            raw_values = list(awake_bucket_ids)
        bucket_ids: list[int] = []
        seen: set[int] = set()
        for raw in raw_values:
            if raw is None:
                continue
            if isinstance(raw, torch.Tensor):
                if int(raw.numel()) != 1:
                    continue
                raw = raw.detach().cpu().item()
            try:
                bucket_id = int(raw)
            except (TypeError, ValueError, OverflowError):
                continue
            if bucket_id < 0 or bucket_id in seen:
                continue
            seen.add(bucket_id)
            bucket_ids.append(bucket_id)
        return bucket_ids

    def _ripple_tag_indices(
        self,
        indices: Sequence[int],
        *,
        floor_token: int,
        window_span: float,
        da_scale: float,
        size: int,
    ) -> int:
        tagged = 0
        touched = 0
        for raw_index in indices:
            idx = int(raw_index)
            if idx < 0 or idx >= size:
                continue
            entry_token = int(self.slow_entry_timestamps[idx])
            if entry_token < floor_token:
                continue
            recency_scale = max(
                0.0,
                min(
                    1.0,
                    (float(entry_token) - float(floor_token))
                    / window_span,
                ),
            )
            ripple_strength = self._clip_ripple_strength(
                0.5 + 0.30 * da_scale + 0.20 * recency_scale
            )
            was_untagged = float(self.slow_ripple_strength[idx]) <= 0.0
            self.slow_ripple_strength[idx] = float(
                max(self.slow_ripple_strength[idx], ripple_strength)
            )
            self.slow_capture_tag[idx] = float(
                min(
                    1.0,
                    self.slow_capture_tag[idx]
                    + 0.10
                    + 0.25 * ripple_strength,
                )
            )
            tagged += int(was_untagged)
            touched += 1
        if touched:
            self._invalidate_summary_cache()
        return tagged

    def _replay_priority_score(self, idx: int, current_token: int) -> float:
        if idx < 0 or idx >= len(self.slow_buffer):
            return 0.0
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
        score = (
            importance
            * (1.0 + spacing)
            * (1.0 + frontier)
            / (1.0 + 0.35 * float(replay_count))
        )
        ripple_strength = (
            0.0
            if idx >= len(self.slow_ripple_strength)
            else float(self.slow_ripple_strength[idx])
        )
        score *= self._ripple_priority_multiplier(ripple_strength)
        return float(score)

    def replay_scores_for_indices(
        self,
        indices: Sequence[int] | torch.Tensor,
        current_token: int,
    ) -> dict[int, float]:
        if not self.slow_buffer:
            return {}

        self._advance_state(current_token)
        scores: dict[int, float] = {}
        size = len(self.slow_buffer)
        raw_indices = (
            indices.detach().cpu().flatten().tolist()
            if isinstance(indices, torch.Tensor)
            else list(indices)
        )
        for raw_index in raw_indices:
            index = int(raw_index)
            if index in scores or index < 0 or index >= size:
                continue
            scores[index] = float(self._replay_priority_score(index, current_token))
        return scores

    def ripple_tag_awake(
        self,
        *,
        current_token: int,
        window_tokens: int,
        da_level: float,
        da_threshold: float = 0.7,
        awake_bucket_ids: Sequence[int] | torch.Tensor | None = None,
        max_candidate_entries: int = 256,
    ) -> int:
        """Awake ripple tagging (Yang & Buzsaki 2024, Science).

        When dopamine (DA) exceeds threshold during wakefulness,
        mark recent memories with a ripple tag. These get 3-5x
        replay priority during subsequent sleep consolidation.

        Args:
            current_token: Current training token/timestep.
            window_tokens: How far back to look for recent entries.
            da_level: Current dopamine level from SurpriseMonitor.
            da_threshold: DA threshold to trigger ripple tagging.
            awake_bucket_ids: Optional scheduler-owned column/bucket ids.
                When supplied, only entries attached to those awake buckets are
                considered. An empty list is an explicit no-awake-bucket result,
                not a request to fall back to the global scan.

        Returns:
            Number of entries ripple-tagged.
        """
        if da_level < da_threshold or window_tokens <= 0:
            self.last_awake_ripple_tag_report = {
                **self._empty_awake_ripple_tag_report(),
                "status": "skipped",
                "memory_size": int(len(self.slow_buffer)),
                "current_token": int(current_token),
                "window_tokens": int(window_tokens),
                "da_level": float(da_level),
                "da_threshold": float(da_threshold),
                "fallback_reason": (
                    "dopamine_below_threshold"
                    if da_level < da_threshold
                    else "empty_recent_window"
                ),
            }
            self._invalidate_summary_cache()
            return 0

        started = time.perf_counter()
        self._advance_state(current_token)
        floor_token = max(0, int(current_token) - int(window_tokens))
        window_span = max(1.0, float(window_tokens))
        da_scale = max(0.0, min(1.0, (float(da_level) - float(da_threshold)) / max(1e-6, 1.0 - float(da_threshold))))
        candidate_limit = max(0, int(max_candidate_entries))
        size = min(
            len(self.slow_entry_timestamps),
            len(self.slow_ripple_strength),
            len(self.slow_capture_tag),
        )
        if size <= 0:
            self.last_awake_ripple_tag_report = {
                **self._empty_awake_ripple_tag_report(),
                "status": "empty",
                "memory_size": int(len(self.slow_buffer)),
                "current_token": int(current_token),
                "window_tokens": int(window_tokens),
                "floor_token": int(floor_token),
                "da_level": float(da_level),
                "da_threshold": float(da_threshold),
                "candidate_window_limit": int(candidate_limit),
                "latency_ms": float((time.perf_counter() - started) * 1000.0),
                "fallback_reason": "empty_memory",
            }
            self._invalidate_summary_cache()
            return 0
        bucket_ids = self._normalise_awake_bucket_ids(awake_bucket_ids)
        if bucket_ids is not None:
            self.last_ripple_scan_mode = "awake_bucket_index"
            self.ripple_awake_bucket_scan_count += 1
            self.last_ripple_awake_bucket_count = int(len(bucket_ids))
            candidate_window = self._candidate_indices_for_bucket_ids(
                bucket_ids,
                max_candidates=candidate_limit,
            )
            bounded_indices = [
                int(index) for index in candidate_window.candidate_indices
            ]
            self.last_ripple_awake_candidate_count = int(len(bounded_indices))
            self.ripple_awake_bucket_candidate_count += int(
                len(bounded_indices)
            )
            tagged = 0
            if bounded_indices:
                tagged = self._ripple_tag_indices(
                    bounded_indices,
                    floor_token=floor_token,
                    window_span=window_span,
                    da_scale=da_scale,
                    size=size,
                )
            self.last_awake_ripple_tag_report = {
                "surface": "bounded_awake_ripple_tag.v1",
                "status": "tagged" if tagged else "empty",
                "scope": "awake_ripple_tagging_cadenced_path",
                "memory_size": int(len(self.slow_buffer)),
                "current_token": int(current_token),
                "window_tokens": int(window_tokens),
                "floor_token": int(floor_token),
                "da_level": float(da_level),
                "da_threshold": float(da_threshold),
                "candidate_window_limit": int(candidate_limit),
                "candidate_window_policy": "recent_bucket_round_robin_candidate_pool",
                "candidate_scope": "awake_bucket_index_candidate_window",
                "candidate_bucket_ids": [int(bucket_id) for bucket_id in bucket_ids],
                "candidate_bucket_count": int(len(bucket_ids)),
                "candidate_index_available_count": int(
                    candidate_window.available_count
                ),
                "candidate_index_count": int(len(bounded_indices)),
                "candidate_indices": [int(index) for index in bounded_indices],
                "tagged_count": int(tagged),
                "scan_mode": str(self.last_ripple_scan_mode),
                **self._bucket_candidate_source_fields(candidate_window),
                "global_candidate_scan": False,
                "diagnostic_global_candidate_scan": False,
                "runs_live_tick": True,
                "runs_every_token": False,
                "mutates_runtime_state": bool(tagged),
                "applies_plasticity": bool(tagged),
                "archival_storage_device": "cpu",
                "latency_ms": float((time.perf_counter() - started) * 1000.0),
                "fallback_reason": None
                if tagged
                else "empty_awake_bucket_candidate_window",
                "selection_budget": {
                    "memory_budget_entries": int(len(self.slow_buffer)),
                    "candidate_window_entries": int(candidate_limit),
                    "candidate_bucket_entries_available": int(
                        candidate_window.available_count
                    ),
                    "candidate_source_read_budget_entries": int(
                        candidate_window.source_entry_read_budget
                    ),
                },
            }
            self._invalidate_summary_cache()
            return int(tagged)

        self.last_ripple_scan_mode = "awake_bucket_scope_required"
        self.last_ripple_awake_bucket_count = 0
        self.last_ripple_awake_candidate_count = 0
        self.last_awake_ripple_tag_report = {
            "surface": "bounded_awake_ripple_tag.v1",
            "status": "empty",
            "scope": "awake_ripple_tagging_cadenced_path",
            "memory_size": int(len(self.slow_buffer)),
            "current_token": int(current_token),
            "window_tokens": int(window_tokens),
            "floor_token": int(floor_token),
            "da_level": float(da_level),
            "da_threshold": float(da_threshold),
            "candidate_window_limit": int(candidate_limit),
            "candidate_window_policy": "awake_bucket_scope_required_no_global_fallback",
            "candidate_scope": "awake_bucket_scope_required",
            "candidate_bucket_ids": [],
            "candidate_bucket_count": 0,
            "candidate_index_available_count": 0,
            "candidate_index_count": 0,
            "candidate_indices": [],
            "tagged_count": 0,
            "scan_mode": str(self.last_ripple_scan_mode),
            **self._empty_bucket_candidate_source_fields(),
            "global_candidate_scan": False,
            "diagnostic_global_candidate_scan": False,
            "runs_live_tick": True,
            "runs_every_token": False,
            "mutates_runtime_state": False,
            "applies_plasticity": False,
            "archival_storage_device": "cpu",
            "latency_ms": float((time.perf_counter() - started) * 1000.0),
            "fallback_reason": "awake_bucket_scope_required_for_ripple_tagging",
            "selection_budget": {
                "memory_budget_entries": int(len(self.slow_buffer)),
                "candidate_window_entries": int(candidate_limit),
            },
        }
        self._invalidate_summary_cache()
        return 0

    @property
    def ripple_tagged_count(self) -> int:
        """Number of currently ripple-tagged memories."""
        return sum(1 for value in self.slow_ripple_strength if float(value) > 0.0)

    def collect_recent_entry_indices(
        self,
        *,
        current_token: int,
        window_tokens: int,
        max_entries: int = 256,
        require_bucket: bool = False,
        scope: str = "recent_memory_slow_path",
    ) -> dict[str, Any]:
        started = time.perf_counter()
        requested = max(0, int(max_entries))
        indices, observed_available, truncated, floor_token = (
            self._recent_indices_for_window(
                current_token=int(current_token),
                window_tokens=int(window_tokens),
                max_entries=requested,
                require_bucket=bool(require_bucket),
            )
        )
        fallback_reason: str | None = None
        if int(window_tokens) <= 0:
            fallback_reason = "empty_recent_window"
        elif requested <= 0:
            fallback_reason = "empty_recent_window_budget"
        elif not indices:
            fallback_reason = "no_recent_entries_in_bounded_window"
        report = {
            "surface": "bounded_recent_memory_window.v1",
            "status": "collected" if indices else "empty",
            "scope": str(scope),
            "memory_size": int(len(self.slow_buffer)),
            "current_token": int(current_token),
            "window_tokens": int(window_tokens),
            "floor_token": int(floor_token),
            "requested_count": int(requested),
            "candidate_window_limit": int(requested),
            "candidate_window_policy": "recent_entry_index_reverse_window",
            "candidate_scope": (
                "bucketed_recent_entry_index_window"
                if require_bucket
                else "recent_entry_index_window"
            ),
            "candidate_index_available_count": int(observed_available),
            "candidate_index_available_count_is_lower_bound": bool(truncated),
            "candidate_index_count": int(len(indices)),
            "candidate_indices": [int(index) for index in indices],
            "requires_bucket": bool(require_bucket),
            "global_score_scan": False,
            "global_candidate_scan": False,
            "runs_live_tick": False,
            "mutates_runtime_state": False,
            "applies_plasticity": False,
            "archival_storage_device": "cpu",
            "latency_ms": float((time.perf_counter() - started) * 1000.0),
            "fallback_reason": fallback_reason,
            "selection_budget": {
                "memory_budget_entries": int(len(self.slow_buffer)),
                "candidate_window_entries": int(requested),
            },
        }
        self.last_recent_memory_window_report = report
        self._invalidate_summary_cache()
        return report

    def tag_recent_entries(
        self,
        *,
        current_token: int,
        window_tokens: int,
        strength: float,
        max_recent_entries: int = 256,
    ) -> int:
        if window_tokens <= 0 or strength <= 0.0:
            self.last_recent_memory_tag_report = {
                **self._empty_recent_memory_tag_report(),
                "status": "empty",
                "current_token": int(current_token),
                "window_tokens": int(window_tokens),
                "strength": float(strength),
                "fallback_reason": (
                    "empty_recent_window"
                    if window_tokens <= 0
                    else "non_positive_tag_strength"
                ),
            }
            self._invalidate_summary_cache()
            return 0

        self._advance_state(current_token)
        window_report = self.collect_recent_entry_indices(
            current_token=int(current_token),
            window_tokens=int(window_tokens),
            max_entries=int(max_recent_entries),
            require_bucket=False,
            scope="recent_memory_tagging_slow_path",
        )
        tagged = 0
        for idx in window_report.get("candidate_indices", []):
            idx = int(idx)
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
        self.last_recent_memory_tag_report = {
            **dict(window_report),
            "surface": "bounded_recent_memory_tag.v1",
            "status": "tagged" if tagged else "empty",
            "scope": "recent_memory_tagging_slow_path",
            "tagged_count": int(tagged),
            "strength": float(strength),
            "mutates_runtime_state": bool(tagged),
            "applies_plasticity": bool(tagged),
            "fallback_reason": None
            if tagged
            else window_report.get("fallback_reason", "no_recent_entries_tagged"),
        }
        self._invalidate_summary_cache()
        return tagged

    def replay_entry(
        self,
        index: int,
        current_token: Optional[int] = None,
        *,
        include_text_payload: bool = False,
    ) -> dict[str, Any]:
        """Return replay tensors; raw text payloads require an explicit opt-in."""

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
        entry = {
            "assembly": self.slow_buffer[idx].detach().clone(),
            "input_pattern": input_pattern.detach().clone() if isinstance(input_pattern, torch.Tensor) else None,
            "routing_key": routing_key.detach().clone() if isinstance(routing_key, torch.Tensor) else None,
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
            "tokens_since_last_replay": int(max(0, token_marker - int(self.slow_last_replay_token[idx]))),
            "fragility": float(self.fragility_score(idx, token_marker)),
            "age_tokens": int(max(0, token_marker - int(self.slow_entry_timestamps[idx]))),
            "last_replay_token": int(self.slow_last_replay_token[idx]),
            "tag_is_strong": bool(self.slow_tag_is_strong[idx]),
            "ripple_strength": float(self.slow_ripple_strength[idx]) if idx < len(self.slow_ripple_strength) else 0.0,
            "ripple_priority_multiplier": float(
                self._ripple_priority_multiplier(self.slow_ripple_strength[idx])
                if idx < len(self.slow_ripple_strength)
                else 1.0
            ),
        }
        if include_text_payload:
            entry.update(
                {
                    "raw_window": self.slow_raw_windows[idx],
                    "text": self.slow_texts[idx],
                    "metadata": None
                    if self.slow_metadata[idx] is None
                    else dict(self.slow_metadata[idx]),
                }
            )
        else:
            entry.update({"raw_window": None, "text": None, "metadata": None})
        return entry

    def _score_replay_index(
        self,
        idx: int,
        *,
        current_token: int,
        strategy: str,
    ) -> float:
        if idx < 0 or idx >= len(self.slow_buffer):
            return 0.0
        if strategy == "maintenance":
            consolidation = float(
                max(0.0, min(1.0, self.slow_consolidation_level[idx]))
            )
            if consolidation >= 0.8:
                return 0.0
            importance = float(max(1e-6, self.slow_importance[idx]))
            tag_strength = float(max(0.0, self.slow_capture_tag[idx]))
            fragility = self.fragility_score(idx, current_token)
            return float(importance * fragility * (1.0 + 0.5 * tag_strength))
        if strategy in {"priority", "consolidation"}:
            consolidation = float(
                max(0.0, min(1.0, self.slow_consolidation_level[idx]))
            )
            tag_strength = float(max(0.0, self.slow_capture_tag[idx]))
            prp_level = float(max(0.0, self._available_prp(idx)))
            capture_strength = float(max(0.0, tag_strength * prp_level))
            if (
                consolidation >= 0.8
                or capture_strength <= self.prp_capture_threshold
            ):
                return 0.0
            importance = float(max(1e-6, self.slow_importance[idx]))
            consolidation_gap = max(0.0, 0.8 - consolidation)
            return float(importance * capture_strength * (1.0 + consolidation_gap))
        if strategy == "repair":
            importance = float(max(1e-6, self.slow_importance[idx]))
            consolidation = float(
                max(0.0, min(1.0, self.slow_consolidation_level[idx]))
            )
            replay_age = max(
                0,
                int(current_token) - int(self.slow_last_replay_token[idx]),
            )
            return float(
                importance
                * (0.5 + consolidation)
                * (
                    1.0
                    + math.log1p(
                        float(replay_age) / max(1.0, float(self.functional_minute))
                    )
                )
            )
        raise ValueError(f"Unknown replay sampling strategy: {strategy}")

    def _candidate_indices_for_bucket_ids(
        self,
        bucket_ids: Sequence[int] | torch.Tensor | None,
        *,
        max_candidates: int | None = None,
    ) -> _BucketCandidateWindow:
        normalized = self._normalise_awake_bucket_ids(bucket_ids)
        if normalized is None:
            return _BucketCandidateWindow(
                normalized_bucket_ids=None,
                candidate_indices=[],
                available_count=0,
                available_count_is_lower_bound=False,
                source_entry_read_count=0,
                source_entry_read_budget=0,
                source_entry_read_budget_exhausted=False,
                source_materialized_entry_count=0,
                source_materialization_count=0,
                source_full_bucket_scan=False,
                candidate_window_limit=0,
            )
        size = len(self.slow_buffer)
        available_count = int(
            sum(
                len(self._bucket_entry_indices.get(int(bucket_id), ()))
                for bucket_id in normalized
            )
        )
        limit = None if max_candidates is None else max(0, int(max_candidates))
        if limit == 0:
            return _BucketCandidateWindow(
                normalized_bucket_ids=normalized,
                candidate_indices=[],
                available_count=available_count,
                available_count_is_lower_bound=False,
                source_entry_read_count=0,
                source_entry_read_budget=0,
                source_entry_read_budget_exhausted=False,
                source_materialized_entry_count=0,
                source_materialization_count=0,
                source_full_bucket_scan=False,
                candidate_window_limit=0,
            )

        per_bucket_entries = [
            self._bucket_entry_indices.get(int(bucket_id), ())
            for bucket_id in normalized
        ]
        cursors = [len(bucket_entries) - 1 for bucket_entries in per_bucket_entries]
        bounded: list[int] = []
        seen: set[int] = set()
        source_entry_read_count = 0
        source_entry_read_budget = (
            available_count
            if limit is None
            else int(max(0, limit) * max(1, len(per_bucket_entries)))
        )
        source_entry_read_budget_exhausted = False
        while True:
            progressed = False
            for bucket_pos, bucket_entries in enumerate(per_bucket_entries):
                cursor = int(cursors[bucket_pos])
                while cursor >= 0:
                    if source_entry_read_count >= source_entry_read_budget:
                        source_entry_read_budget_exhausted = True
                        break
                    raw_index = int(bucket_entries[cursor])
                    cursor -= 1
                    source_entry_read_count += 1
                    if raw_index in seen or raw_index < 0 or raw_index >= size:
                        continue
                    seen.add(raw_index)
                    bounded.append(raw_index)
                    progressed = True
                    break
                cursors[bucket_pos] = cursor
                if source_entry_read_budget_exhausted:
                    break
                if limit is not None and len(bounded) >= limit:
                    return _BucketCandidateWindow(
                        normalized_bucket_ids=normalized,
                        candidate_indices=bounded,
                        available_count=available_count,
                        available_count_is_lower_bound=False,
                        source_entry_read_count=source_entry_read_count,
                        source_entry_read_budget=source_entry_read_budget,
                        source_entry_read_budget_exhausted=False,
                        source_materialized_entry_count=0,
                        source_materialization_count=0,
                        source_full_bucket_scan=False,
                        candidate_window_limit=int(limit),
                    )
            if not progressed:
                break
        return _BucketCandidateWindow(
            normalized_bucket_ids=normalized,
            candidate_indices=bounded,
            available_count=available_count,
            available_count_is_lower_bound=False,
            source_entry_read_count=source_entry_read_count,
            source_entry_read_budget=source_entry_read_budget,
            source_entry_read_budget_exhausted=source_entry_read_budget_exhausted,
            source_materialized_entry_count=0,
            source_materialization_count=0,
            source_full_bucket_scan=bool(limit is None),
            candidate_window_limit=int(len(bounded) if limit is None else limit),
        )

    @staticmethod
    def _selection_score_summary(scores: Sequence[float]) -> dict[str, float]:
        if not scores:
            return {
                "selected_score_min": 0.0,
                "selected_score_max": 0.0,
                "selected_score_mean": 0.0,
            }
        return {
            "selected_score_min": float(min(scores)),
            "selected_score_max": float(max(scores)),
            "selected_score_mean": float(sum(scores) / len(scores)),
        }

    def select_replay_window(
        self,
        *,
        n: int,
        current_token: int,
        candidate_pool: Optional[int] = None,
        strategy: str = "priority",
        candidate_bucket_ids: Sequence[int] | torch.Tensor | None = None,
        scope: str = "sleep_slow_path",
    ) -> dict[str, Any]:
        """Select a bounded replay window and record the selection evidence.

        Candidate bucket ids make selection score only indexed memory entries
        attached to those buckets. Unscoped full-memory scoring is retired from
        the runtime API; legacy comparisons live in evaluation harnesses only.
        """

        started = time.perf_counter()
        requested = max(0, int(n))
        token_marker = int(current_token)
        count = len(self.slow_buffer)
        candidate_window_limit = max(
            requested,
            int(candidate_pool) if candidate_pool is not None else requested,
        )
        candidate_window = self._candidate_indices_for_bucket_ids(
            candidate_bucket_ids,
            max_candidates=candidate_window_limit,
        )
        normalized_buckets = candidate_window.normalized_bucket_ids
        bucket_candidates = candidate_window.candidate_indices
        bucket_scoped = normalized_buckets is not None
        selected: list[int] = []
        selected_scores: list[float] = []
        score_count = 0
        candidate_count = count
        pool_limit = 0
        fallback_reason: str | None = None

        if requested <= 0 or count <= 0:
            fallback_reason = "empty_request_or_memory"
        elif strategy == "random" and bucket_scoped:
            candidate_indices = bucket_candidates
            candidate_count = len(candidate_indices)
            score_count = 0
            pool_limit = min(candidate_count, requested)
            if candidate_count > 0:
                perm = torch.randperm(candidate_count)
                selected = [
                    int(candidate_indices[int(local_idx)])
                    for local_idx in perm[:pool_limit].tolist()
                ]
        elif bucket_scoped:
            candidate_count = len(bucket_candidates)
            if candidate_count <= 0:
                fallback_reason = "empty_bucket_index_candidate_window"
            else:
                self._advance_state(token_marker)
                score_count = candidate_count
                scored = [
                    (
                        int(idx),
                        self._score_replay_index(
                            int(idx),
                            current_token=token_marker,
                            strategy=strategy,
                        ),
                    )
                    for idx in bucket_candidates
                ]
                scored.sort(key=lambda item: (-item[1], item[0]))
                pool_limit = min(len(scored), candidate_window_limit)
                window = scored[:pool_limit]
                selected_pairs = window[: min(requested, len(window))]
                selected = [idx for idx, _ in selected_pairs]
                selected_scores = [float(score) for _, score in selected_pairs]
                if not selected:
                    fallback_reason = "empty_bucket_candidate_pool"
        else:
            candidate_count = 0
            score_count = 0
            fallback_reason = "candidate_bucket_scope_required_for_replay_window"

        latency_ms = (time.perf_counter() - started) * 1000.0
        report = {
            "surface": "bounded_replay_window_selection.v1",
            "status": "selected" if selected else "empty",
            "scope": str(scope),
            "strategy": str(strategy),
            "current_token": token_marker,
            "memory_size": int(count),
            "requested_count": int(requested),
            "candidate_pool_limit": int(
                candidate_pool if candidate_pool is not None else requested
            ),
            "candidate_pool_sampled_count": int(pool_limit),
            "candidate_window_limit": int(candidate_window_limit),
            "candidate_window_policy": (
                "recent_bucket_round_robin_candidate_pool"
                if bucket_scoped
                else "bucket_scope_required_no_global_fallback"
            ),
            "candidate_scope": (
                "bucket_indexed_candidate_window"
                if bucket_scoped
                else "bucket_index_scope_required"
            ),
            "candidate_bucket_ids": list(normalized_buckets or []),
            "candidate_bucket_count": int(len(normalized_buckets or [])),
            "candidate_index_available_count": int(
                candidate_window.available_count
                if bucket_scoped
                else 0
            ),
            "candidate_index_count": int(candidate_count),
            "score_count": int(score_count),
            "selected_indices": selected,
            "selected_count": int(len(selected)),
            **self._bucket_candidate_source_fields(candidate_window),
            "score_device": "cpu",
            "archival_storage_device": "cpu",
            "bounded_by_bucket_index": bool(bucket_scoped),
            "global_score_scan": False,
            "global_candidate_scan": False,
            "diagnostic_global_score_scan": False,
            "diagnostic_global_candidate_scan": False,
            "runs_live_tick": False,
            "records_replay_artifact": False,
            "raw_text_payload_loaded": False,
            "language_reasoning": False,
            "mutates_runtime_state": False,
            "applies_plasticity": False,
            "latency_ms": float(latency_ms),
            "fallback_reason": fallback_reason,
            "selection_budget": {
                "memory_budget_entries": int(count),
                "score_budget_entries": int(score_count),
                "selected_budget_entries": int(requested),
                "candidate_window_entries": int(candidate_window_limit),
                "candidate_pool_entries": int(
                    candidate_pool if candidate_pool is not None else requested
                ),
                "candidate_source_read_budget_entries": int(
                    candidate_window.source_entry_read_budget
                ),
            },
            **self._selection_score_summary(selected_scores),
        }
        self.last_replay_selection_report = report
        self._invalidate_summary_cache()
        return report

    @staticmethod
    def _normalise_cpu_routing_key(value: torch.Tensor) -> torch.Tensor:
        key = value.detach().clone().cpu().float().clamp(min=1e-6)
        if key.dim() != 1:
            key = key.flatten()
        return key / key.norm().clamp(min=1e-8)

    def recall_replay_window(
        self,
        *,
        query_routing_key: torch.Tensor,
        query_input_pattern: torch.Tensor | None = None,
        current_token: int,
        candidate_bucket_ids: Sequence[int] | torch.Tensor | None,
        max_candidates: int,
        strategy: str = "repair",
        temperature: float = 32.0,
        scope: str = "replay_recall_slow_path",
    ) -> dict[str, Any]:
        """Run bounded associative recall over selected replay routing keys.

        This is a slow-path local memory operator.  It never scans memory unless
        candidate bucket ids are supplied, and it never mutates runtime state.
        """

        started = time.perf_counter()
        scoped_bucket_ids = [] if candidate_bucket_ids is None else candidate_bucket_ids
        selection_report = self.select_replay_window(
            n=max(0, int(max_candidates)),
            current_token=int(current_token),
            candidate_pool=max(0, int(max_candidates)),
            strategy=strategy,
            candidate_bucket_ids=scoped_bucket_ids,
            scope=scope,
        )
        selected_indices = [
            int(index)
            for index in selection_report.get("selected_indices", [])
        ]
        selection_blocked_reason: str | None = None
        if (
            strategy != "repair"
            and float(selection_report.get("selected_score_max", 0.0) or 0.0) <= 0.0
        ):
            selected_indices = []
            selection_blocked_reason = "no_positive_recall_pressure"
        query = self._normalise_cpu_routing_key(query_routing_key)
        keys: list[torch.Tensor] = []
        key_indices: list[int] = []
        input_patterns: list[torch.Tensor] = []
        input_indices: list[int] = []
        fallback_reason: str | None = None
        for index in selected_indices:
            if index < 0 or index >= len(self.slow_routing_keys):
                continue
            routing_key = self.slow_routing_keys[index]
            if not isinstance(routing_key, torch.Tensor):
                routing_key = None
            if isinstance(routing_key, torch.Tensor) and int(routing_key.numel()) == int(query.numel()):
                keys.append(self._normalise_cpu_routing_key(routing_key))
                key_indices.append(index)
            if query_input_pattern is not None and index < len(self.slow_input_patterns):
                input_pattern = self.slow_input_patterns[index]
                if (
                    isinstance(input_pattern, torch.Tensor)
                    and int(input_pattern.numel()) == int(query_input_pattern.numel())
                ):
                    input_patterns.append(
                        self._normalise_cpu_routing_key(input_pattern)
                    )
                    input_indices.append(index)

        best_index: int | None = None
        best_similarity: float | None = None
        best_distance: float | None = None
        recalled_similarity: float | None = None
        recalled_distance: float | None = None
        recalled_key_norm = 0.0
        attention_entropy = 0.0
        best_input_index: int | None = None
        best_input_similarity: float | None = None
        best_input_distance: float | None = None
        if not keys:
            fallback_reason = (
                selection_blocked_reason
                or str(selection_report.get("fallback_reason") or "no_routing_keys")
            )
        else:
            matrix = torch.stack(keys, dim=0)
            similarities = torch.mv(matrix, query)
            best_local = int(torch.argmax(similarities).item())
            best_index = int(key_indices[best_local])
            best_similarity = float(similarities[best_local].item())
            best_distance = max(0.0, 1.0 - best_similarity)
            weights = torch.softmax(
                similarities * max(1e-6, float(temperature)),
                dim=0,
            )
            recalled = torch.mv(matrix.t(), weights)
            recalled_key_norm = float(recalled.norm().item())
            recalled = recalled.clamp(min=1e-6)
            recalled = recalled / recalled.norm().clamp(min=1e-8)
            recalled_similarity = float(torch.dot(recalled, query).item())
            recalled_distance = max(0.0, 1.0 - recalled_similarity)
            attention_entropy = float(
                -(weights * torch.log(weights.clamp(min=1e-12))).sum().item()
            )
        if query_input_pattern is not None and input_patterns:
            query_input = self._normalise_cpu_routing_key(query_input_pattern)
            input_matrix = torch.stack(input_patterns, dim=0)
            input_similarities = torch.mv(input_matrix, query_input)
            input_best_local = int(torch.argmax(input_similarities).item())
            best_input_index = int(input_indices[input_best_local])
            best_input_similarity = float(input_similarities[input_best_local].item())
            best_input_distance = max(0.0, 1.0 - best_input_similarity)

        latency_ms = (time.perf_counter() - started) * 1000.0
        report = {
            "surface": "bounded_replay_window_recall.v1",
            "status": "recalled" if keys else "empty",
            "scope": str(scope),
            "strategy": str(strategy),
            "current_token": int(current_token),
            "candidate_scope": selection_report.get("candidate_scope"),
            "candidate_bucket_ids": list(
                selection_report.get("candidate_bucket_ids", [])
            ),
            "candidate_bucket_count": int(
                selection_report.get("candidate_bucket_count", 0) or 0
            ),
            "candidate_index_count": int(
                selection_report.get("candidate_index_count", 0) or 0
            ),
            "selected_indices": selected_indices,
            "selected_count": int(len(selected_indices)),
            "routing_key_indices": key_indices,
            "routing_key_count": int(len(keys)),
            "input_pattern_indices": input_indices,
            "input_pattern_count": int(len(input_patterns)),
            "best_index": best_index,
            "best_similarity": best_similarity,
            "best_distance": best_distance,
            "best_input_index": best_input_index,
            "best_input_similarity": best_input_similarity,
            "best_input_distance": best_input_distance,
            "recalled_similarity": recalled_similarity,
            "recalled_distance": recalled_distance,
            "recalled_key_norm": recalled_key_norm,
            "attention_entropy": attention_entropy,
            "temperature": float(temperature),
            "selection_report": dict(selection_report),
            "score_device": "cpu",
            "archival_storage_device": "cpu",
            "runs_live_tick": False,
            "raw_text_payload_loaded": False,
            "language_reasoning": False,
            "mutates_runtime_state": False,
            "applies_plasticity": False,
            "latency_ms": float(latency_ms),
            "fallback_reason": fallback_reason,
        }
        self.last_replay_recall_report = report
        self._invalidate_summary_cache()
        return report

    def collect_replay_query_indices(
        self,
        *,
        candidate_bucket_ids: Sequence[int] | torch.Tensor | None,
        max_queries: int,
        require_input_pattern: bool = True,
        scope: str = "replay_query_collection_slow_path",
    ) -> dict[str, Any]:
        """Collect bounded replay-query indices from bucket-indexed memory."""

        started = time.perf_counter()
        requested = max(0, int(max_queries))
        scoped_bucket_ids = [] if candidate_bucket_ids is None else candidate_bucket_ids
        candidate_window = self._candidate_indices_for_bucket_ids(
            scoped_bucket_ids,
            max_candidates=requested,
        )
        normalized_buckets = candidate_window.normalized_bucket_ids
        candidate_indices = candidate_window.candidate_indices
        query_indices: list[int] = []
        skipped_missing_input_pattern = 0
        fallback_reason: str | None = None
        if requested <= 0:
            fallback_reason = "empty_query_request"
        elif normalized_buckets is not None and not normalized_buckets:
            fallback_reason = "empty_anchor_bucket_scope"
        elif not candidate_indices:
            fallback_reason = "empty_bucket_index_candidate_window"
        else:
            for index in candidate_indices:
                if require_input_pattern:
                    pattern = (
                        self.slow_input_patterns[index]
                        if 0 <= int(index) < len(self.slow_input_patterns)
                        else None
                    )
                    if not isinstance(pattern, torch.Tensor) or int(pattern.numel()) <= 0:
                        skipped_missing_input_pattern += 1
                        continue
                query_indices.append(int(index))
                if len(query_indices) >= requested:
                    break
            if not query_indices:
                fallback_reason = "no_replay_query_payloads"

        latency_ms = (time.perf_counter() - started) * 1000.0
        report = {
            "surface": "bounded_replay_query_collection.v1",
            "status": "collected" if query_indices else "empty",
            "scope": str(scope),
            "memory_size": int(len(self.slow_buffer)),
            "requested_count": int(requested),
            "candidate_window_limit": int(requested),
            "candidate_window_policy": (
                "recent_bucket_round_robin_candidate_pool"
                if normalized_buckets is not None
                else "unscoped_query_collection_retired"
            ),
            "candidate_scope": (
                "bucket_indexed_candidate_window"
                if normalized_buckets is not None
                else "unscoped_query_collection_retired"
            ),
            "candidate_bucket_ids": list(normalized_buckets or []),
            "candidate_bucket_count": int(len(normalized_buckets or [])),
            "candidate_index_available_count": int(candidate_window.available_count),
            "candidate_index_count": int(len(candidate_indices)),
            "query_indices": query_indices,
            "query_count": int(len(query_indices)),
            "skipped_missing_input_pattern_count": int(skipped_missing_input_pattern),
            "score_count": 0,
            **self._bucket_candidate_source_fields(candidate_window),
            "global_score_scan": False,
            "global_candidate_scan": False,
            "runs_live_tick": False,
            "mutates_runtime_state": False,
            "applies_plasticity": False,
            "archival_storage_device": "cpu",
            "latency_ms": float(latency_ms),
            "fallback_reason": fallback_reason,
            "selection_budget": {
                "memory_budget_entries": int(len(self.slow_buffer)),
                "candidate_window_entries": int(requested),
                "query_budget_entries": int(requested),
                "candidate_source_read_budget_entries": int(
                    candidate_window.source_entry_read_budget
                ),
            },
        }
        self.last_replay_query_collection_report = report
        self._invalidate_summary_cache()
        return report

    def collect_query_memory_match_indices(
        self,
        *,
        candidate_bucket_ids: Sequence[int] | torch.Tensor | None,
        max_candidates: int,
        scope: str = "query_memory_match_slow_path",
    ) -> dict[str, Any]:
        """Collect bounded memory indices for explicit query/readout matching."""

        started = time.perf_counter()
        requested = max(0, int(max_candidates))
        scoped_bucket_ids = [] if candidate_bucket_ids is None else candidate_bucket_ids
        candidate_window = self._candidate_indices_for_bucket_ids(
            scoped_bucket_ids,
            max_candidates=requested,
        )
        normalized_buckets = candidate_window.normalized_bucket_ids
        candidate_indices = candidate_window.candidate_indices
        fallback_reason: str | None = None
        if requested <= 0:
            fallback_reason = "empty_query_match_request"
        elif normalized_buckets is not None and not normalized_buckets:
            fallback_reason = "empty_query_candidate_bucket_scope"
        elif not candidate_indices:
            fallback_reason = "empty_query_candidate_window"

        latency_ms = (time.perf_counter() - started) * 1000.0
        report = {
            "surface": "bounded_query_memory_match_candidates.v1",
            "status": "collected" if candidate_indices else "empty",
            "scope": str(scope),
            "memory_size": int(len(self.slow_buffer)),
            "requested_count": int(requested),
            "candidate_window_limit": int(requested),
            "candidate_window_policy": (
                "recent_bucket_round_robin_candidate_pool"
                if normalized_buckets is not None
                else "unscoped_query_memory_match_retired"
            ),
            "candidate_scope": (
                "bucket_indexed_candidate_window"
                if normalized_buckets is not None
                else "unscoped_query_memory_match_retired"
            ),
            "candidate_bucket_ids": list(normalized_buckets or []),
            "candidate_bucket_count": int(len(normalized_buckets or [])),
            "candidate_index_available_count": int(candidate_window.available_count),
            "candidate_index_count": int(len(candidate_indices)),
            "match_indices": [int(index) for index in candidate_indices],
            "score_count": 0,
            **self._bucket_candidate_source_fields(candidate_window),
            "global_score_scan": False,
            "global_candidate_scan": False,
            "runs_live_tick": False,
            "mutates_runtime_state": False,
            "applies_plasticity": False,
            "archival_storage_device": "cpu",
            "latency_ms": float(latency_ms),
            "fallback_reason": fallback_reason,
            "selection_budget": {
                "memory_budget_entries": int(len(self.slow_buffer)),
                "candidate_window_entries": int(requested),
                "candidate_source_read_budget_entries": int(
                    candidate_window.source_entry_read_budget
                ),
            },
        }
        self.last_query_memory_match_report = report
        self._invalidate_summary_cache()
        return report

    def record_bank_memory_match_report(self, report: Mapping[str, Any]) -> dict[str, Any]:
        stored_report = dict(report)
        self.last_bank_memory_match_report = stored_report
        self._invalidate_summary_cache()
        return stored_report

    def resolve_runtime_concept_memory_matches(
        self,
        *,
        observations: Sequence[tuple[str | None, Mapping[str, Any] | None]],
        max_observations: int = 64,
        scope: str = "cadenced_runtime_concept_observation",
    ) -> dict[str, Any]:
        """Resolve runtime concept evidence from explicit train-step indices.

        The runtime callback may run during feed/source observation, so the
        archive boundary has to be stronger than "small in practice": callers
        provide already-selected `memory_index` evidence and this method direct
        indexes only those entries. It never searches or scores the archive.
        """

        started = time.perf_counter()
        observation_count = len(observations)
        limit = max(0, int(max_observations))
        processed = min(observation_count, limit)
        result_slots: list[int | None] = [None] * observation_count
        matches: list[dict[str, Any]] = []
        source_pairs: list[tuple[str, str]] = []
        candidate_indices: list[int] = []
        raw_payload_cache: dict[int, tuple[str, str, float, float, float] | None] = {}

        invalid_observation_count = 0
        invalid_memory_index_count = 0
        out_of_bounds_index_count = 0
        missing_routing_key_count = 0
        empty_text_count = 0
        raw_text_payload_count = 0
        raw_text_payload_cache_hits = 0

        routing_size = len(self.slow_routing_keys)
        for observation_index in range(processed):
            raw_window, metrics = observations[observation_index]
            if not isinstance(metrics, Mapping):
                invalid_observation_count += 1
                continue
            try:
                idx = int(metrics.get("memory_index"))
            except (TypeError, ValueError):
                invalid_memory_index_count += 1
                continue
            if idx < 0 or idx >= routing_size:
                out_of_bounds_index_count += 1
                continue

            candidate_indices.append(idx)
            cached_payload = raw_payload_cache.get(idx, None)
            cache_contains_index = idx in raw_payload_cache
            if cache_contains_index:
                raw_text_payload_cache_hits += 1
            else:
                routing_key = self.slow_routing_keys[idx]
                if not isinstance(routing_key, torch.Tensor):
                    missing_routing_key_count += 1
                    raw_payload_cache[idx] = None
                    continue

                source_text = ""
                if idx < len(self.slow_texts) and self.slow_texts[idx] is not None:
                    source_text = str(self.slow_texts[idx])
                elif idx < len(self.slow_raw_windows) and self.slow_raw_windows[idx] is not None:
                    source_text = str(self.slow_raw_windows[idx])
                elif raw_window is not None:
                    source_text = str(raw_window)
                source_text = " ".join(source_text.split()).strip()
                if not source_text or not any(char.isalnum() for char in source_text):
                    empty_text_count += 1
                    raw_payload_cache[idx] = None
                    continue

                raw_match = (
                    str(self.slow_raw_windows[idx])
                    if idx < len(self.slow_raw_windows)
                    and self.slow_raw_windows[idx] is not None
                    else source_text
                )
                payload = (
                    source_text,
                    raw_match,
                    float(self.slow_importance[idx])
                    if idx < len(self.slow_importance)
                    else 1.0,
                    float(self.slow_capture_tag[idx])
                    if idx < len(self.slow_capture_tag)
                    else 0.0,
                    float(self.slow_consolidation_level[idx])
                    if idx < len(self.slow_consolidation_level)
                    else 0.0,
                )
                raw_payload_cache[idx] = payload
                raw_text_payload_count += 1
                cached_payload = payload

            if cached_payload is None:
                continue
            source_text, raw_match, importance, capture_tag, consolidation_level = cached_payload
            matches.append(
                {
                    "memory_index": idx,
                    "text": source_text,
                    "raw_window": raw_match,
                    "similarity": 1.0,
                    "importance": importance,
                    "capture_tag": capture_tag,
                    "consolidation_level": consolidation_level,
                }
            )
            source_pairs.append((source_text, raw_match))
            result_slots[observation_index] = len(matches) - 1

        fallback_reason: str | None = None
        if limit <= 0:
            fallback_reason = "empty_runtime_concept_observation_budget"
        elif observation_count <= 0:
            fallback_reason = "empty_runtime_concept_observation_batch"
        elif not matches:
            fallback_reason = "no_valid_runtime_concept_memory_matches"

        latency_ms = (time.perf_counter() - started) * 1000.0
        report = {
            "surface": "bounded_runtime_concept_memory_lookup.v1",
            "status": "matched" if matches else "empty",
            "scope": str(scope),
            "memory_size": int(len(self.slow_buffer)),
            "input_observation_count": int(observation_count),
            "processed_observation_count": int(processed),
            "truncated_observation_count": int(max(0, observation_count - processed)),
            "max_observation_count": int(limit),
            "candidate_window_limit": int(limit),
            "candidate_window_policy": "explicit_train_step_memory_indices_only",
            "candidate_scope": "train_step_memory_index_evidence",
            "candidate_index_count": int(len(candidate_indices)),
            "unique_candidate_index_count": int(len({int(index) for index in candidate_indices})),
            "match_indices": [int(match["memory_index"]) for match in matches],
            "match_count": int(len(matches)),
            "unique_match_index_count": int(
                len({int(match["memory_index"]) for match in matches})
            ),
            "raw_text_payload_loaded": bool(raw_text_payload_count > 0),
            "raw_text_payload_count": int(raw_text_payload_count),
            "raw_text_payload_cache_hits": int(raw_text_payload_cache_hits),
            "invalid_observation_count": int(invalid_observation_count),
            "invalid_memory_index_count": int(invalid_memory_index_count),
            "out_of_bounds_index_count": int(out_of_bounds_index_count),
            "missing_routing_key_count": int(missing_routing_key_count),
            "empty_text_count": int(empty_text_count),
            "score_count": 0,
            "global_score_scan": False,
            "global_candidate_scan": False,
            "runs_live_tick": True,
            "runs_every_token": False,
            "cadenced_observation": True,
            "mutates_runtime_state": False,
            "applies_plasticity": False,
            "language_reasoning": False,
            "score_device": "cpu",
            "archival_storage_device": "cpu",
            "quality_metric": "runtime_concept_observation_match_parity",
            "latency_ms": float(latency_ms),
            "fallback_reason": fallback_reason,
            "selection_budget": {
                "memory_budget_entries": int(len(self.slow_buffer)),
                "candidate_window_entries": int(limit),
                "returned_match_limit": int(limit),
                "raw_text_payload_policy": "cached_explicit_index_payloads_only",
            },
        }
        self.last_runtime_concept_memory_lookup_report = report
        self._invalidate_summary_cache()
        return {
            "matches": matches,
            "result_slots": result_slots,
            "source_pairs": source_pairs,
            "report": report,
        }

    def collect_frontier_gap_indices(
        self,
        *,
        current_token: int,
        max_candidates: int,
        candidate_bucket_ids: Sequence[int] | torch.Tensor | None = None,
        scope: str = "frontier_gap_planner_slow_path",
    ) -> dict[str, Any]:
        """Collect a bounded frontier-planning candidate window.

        Unprompted frontier planning uses the recency index. Bucket-scoped
        callers may pass explicit buckets. Neither path scans the archive.
        """

        started = time.perf_counter()
        requested = max(0, int(max_candidates))
        token_marker = int(current_token)
        normalized_buckets: list[int] | None = None
        candidate_indices: list[int] = []
        available_count = 0
        available_is_lower_bound = False
        floor_token = 0
        fallback_reason: str | None = None
        candidate_window: _BucketCandidateWindow | None = None

        if requested <= 0:
            fallback_reason = "empty_frontier_candidate_request"
        elif candidate_bucket_ids is not None:
            candidate_window = self._candidate_indices_for_bucket_ids(
                candidate_bucket_ids,
                max_candidates=requested,
            )
            normalized_buckets = candidate_window.normalized_bucket_ids
            candidate_indices = candidate_window.candidate_indices
            available_count = candidate_window.available_count
            if normalized_buckets is not None and not normalized_buckets:
                fallback_reason = "empty_frontier_candidate_bucket_scope"
            elif not candidate_indices:
                fallback_reason = "empty_frontier_bucket_candidate_window"
        else:
            window_tokens = max(1, token_marker + 1)
            (
                candidate_indices,
                available_count,
                available_is_lower_bound,
                floor_token,
            ) = self._recent_indices_for_window(
                current_token=token_marker,
                window_tokens=window_tokens,
                max_entries=requested,
                require_bucket=False,
            )
            if not candidate_indices:
                fallback_reason = "empty_frontier_recent_candidate_window"

        latency_ms = (time.perf_counter() - started) * 1000.0
        if candidate_window is not None:
            source_fields = self._bucket_candidate_source_fields(candidate_window)
            source_read_budget = int(candidate_window.source_entry_read_budget)
        else:
            source_fields = self._empty_bucket_candidate_source_fields()
            source_read_budget = 0
        report = {
            "surface": "bounded_frontier_gap_candidates.v1",
            "status": "collected" if candidate_indices else "empty",
            "scope": str(scope),
            "memory_size": int(len(self.slow_buffer)),
            "current_token": token_marker,
            "floor_token": int(floor_token),
            "requested_count": int(requested),
            "candidate_window_limit": int(requested),
            "candidate_window_policy": (
                "recent_bucket_round_robin_candidate_pool"
                if normalized_buckets is not None
                else "recent_entry_index_candidate_window"
            ),
            "candidate_scope": (
                "bucket_indexed_candidate_window"
                if normalized_buckets is not None
                else "recent_entry_index_candidate_window"
            ),
            "candidate_bucket_ids": list(normalized_buckets or []),
            "candidate_bucket_count": int(len(normalized_buckets or [])),
            "candidate_index_available_count": int(available_count),
            "candidate_index_available_count_is_lower_bound": bool(
                available_is_lower_bound
            ),
            "candidate_index_count": int(len(candidate_indices)),
            "candidate_indices": [int(index) for index in candidate_indices],
            **source_fields,
            "global_score_scan": False,
            "global_candidate_scan": False,
            "runs_live_tick": False,
            "raw_text_payload_loaded": False,
            "language_reasoning": False,
            "mutates_runtime_state": False,
            "applies_plasticity": False,
            "archival_storage_device": "cpu",
            "latency_ms": float(latency_ms),
            "fallback_reason": fallback_reason,
            "selection_budget": {
                "memory_budget_entries": int(len(self.slow_buffer)),
                "candidate_window_entries": int(requested),
                "candidate_source_read_budget_entries": source_read_budget,
            },
        }
        self.last_frontier_gap_collection_report = report
        self._invalidate_summary_cache()
        return report

    def sample_for_sfa_with_report(
        self,
        n: int = 100,
        *,
        candidate_indices: Sequence[int] | None = None,
        scope: str = "sfa_correction_slow_path",
    ) -> tuple[list[torch.Tensor], dict[str, Any]]:
        """Sample assembly vectors for SFA correction from a bounded window."""

        started = time.perf_counter()
        count = len(self.slow_buffer)
        requested = max(0, int(n))
        fallback_reason: str | None = None
        candidate_scope = "selected_replay_window"
        candidates: list[int] = []
        invalid_candidate_count = 0
        duplicate_candidate_count = 0
        if count == 0 or requested <= 0:
            fallback_reason = "empty_request_or_memory"
        if fallback_reason is None and candidate_indices is None:
            candidate_scope = "selected_replay_window_required"
            fallback_reason = "candidate_indices_required"
        elif fallback_reason is None:
            seen: set[int] = set()
            for raw_index in candidate_indices:
                index = int(raw_index)
                if index < 0 or index >= count:
                    invalid_candidate_count += 1
                    continue
                if index in seen:
                    duplicate_candidate_count += 1
                    continue
                seen.add(index)
                candidates.append(index)
        if not candidates:
            fallback_reason = fallback_reason or "empty_candidate_window"
        k = min(len(candidates), requested)
        sample_indices: list[int] = []
        samples: list[torch.Tensor] = []
        if k > 0:
            perm = torch.randperm(len(candidates))[:k]
            sample_indices = [int(candidates[int(i)]) for i in perm.tolist()]
            samples = [
                self.slow_buffer[index].detach().clone()
                for index in sample_indices
            ]
        latency_ms = (time.perf_counter() - started) * 1000.0
        report = {
            "surface": "bounded_sfa_sample.v1",
            "status": "selected" if samples else "empty",
            "scope": str(scope),
            "memory_size": int(count),
            "requested_count": int(requested),
            "candidate_scope": candidate_scope,
            "candidate_window_policy": (
                "explicit_selected_replay_indices"
                if candidate_indices is not None
                else "selected_replay_window_required_no_global_fallback"
            ),
            "candidate_index_count": int(len(candidates)),
            "candidate_indices": [int(index) for index in candidates],
            "invalid_candidate_index_count": int(invalid_candidate_count),
            "duplicate_candidate_index_count": int(duplicate_candidate_count),
            "sample_indices": [int(index) for index in sample_indices],
            "sample_count": int(len(samples)),
            "sample_device": "cpu",
            "archival_storage_device": "cpu",
            "global_candidate_scan": False,
            "diagnostic_global_candidate_scan": False,
            "runs_live_tick": False,
            "runs_every_token": False,
            "raw_text_payload_loaded": False,
            "language_reasoning": False,
            "mutates_runtime_state": False,
            "applies_plasticity": False,
            "latency_ms": float(latency_ms),
            "fallback_reason": fallback_reason,
            "selection_budget": {
                "memory_budget_entries": int(count),
                "candidate_window_entries": int(len(candidates)),
                "sample_budget_entries": int(requested),
            },
        }
        self.last_sfa_sample_report = report
        self._invalidate_summary_cache()
        return samples, report

    def refresh_maintenance(
        self,
        indices: Sequence[int],
        *,
        current_token: int,
        tag_refresh: float = 0.05,
    ) -> None:
        if not indices:
            return

        self._advance_state(current_token)
        refresh = float(max(0.0, tag_refresh))
        for raw_idx in indices:
            idx = int(raw_idx)
            if idx < 0 or idx >= len(self.slow_buffer):
                continue
            consolidation = float(max(0.0, min(1.0, self.slow_consolidation_level[idx])))
            if consolidation >= 0.8:
                continue
            if refresh > 0.0:
                self.slow_capture_tag[idx] = float(min(1.0, self.slow_capture_tag[idx] + refresh))
                importance = float(max(1e-6, self.slow_importance[idx]))
                injected = self._inject_prp(
                    bucket_id=self.slow_bucket_ids[idx] if idx < len(self.slow_bucket_ids) else None,
                    strength=self.slow_capture_tag[idx],
                    importance=importance,
                    sleep_boost=0.5,
                )
                if injected > 0.0:
                    self.slow_local_prp[idx] = float(self.slow_local_prp[idx] + 0.10 * injected)
            self.slow_last_replay_token[idx] = int(current_token)
            self.slow_replay_count[idx] += 1

    def mark_repair_replay(
        self,
        indices: Sequence[int],
        *,
        current_token: int,
    ) -> None:
        if not indices:
            return

        self._advance_state(current_token)
        for raw_idx in indices:
            idx = int(raw_idx)
            if idx < 0 or idx >= len(self.slow_buffer):
                continue
            self.slow_last_replay_token[idx] = int(current_token)
            self.slow_replay_count[idx] += 1

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
        if self._bucket_consolidation_cpu is None and self.slow_buffer:
            self._rebuild_bucket_consolidation_cache(
                reason="selected_replay_window"
            )
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
            consolidation = float(max(0.0, min(1.0, self.slow_consolidation_level[idx])))
            if consolidation >= 0.8:
                self.slow_last_replay_token[idx] = int(current_token)
                self.slow_replay_count[idx] += 1
                continue
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
            capture_drive = max(0.0, capture_strength - self.prp_capture_threshold) + 0.25 * min(capture_strength, self.prp_capture_threshold)
            delta = replay_blend * self.consolidation_rate * capture_drive * max(0.0, 1.0 - consolidation)

            if delta > 0.0:
                next_consolidation = float(min(1.0, consolidation + delta))
                self._adjust_bucket_consolidation_cache(
                    bucket_id,
                    importance=importance,
                    consolidation=consolidation,
                    sign=-1.0,
                )
                self.slow_consolidation_level[idx] = next_consolidation
                self._adjust_bucket_consolidation_cache(
                    bucket_id,
                    importance=importance,
                    consolidation=next_consolidation,
                    sign=1.0,
                )
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

    def summary_stats(
        self,
        current_token: Optional[int] = None,
        *,
        force: bool = False,
        cache_interval: int = 50,
    ) -> dict[str, Any]:
        token_marker = self._state_token if current_token is None else int(current_token)
        if (
            not force
            and self._cached_summary is not None
            and abs(token_marker - self._cached_summary_token) < cache_interval
        ):
            return self._cached_summary

        self._advance_state(token_marker)
        size = len(self.slow_buffer)
        if size == 0:
            result = {
                "capacity": int(self.capacity), "size": 0,
                "fill_fraction": 0.0, "n_seen": int(self.n_seen),
                "mean_importance": 0.0, "mean_capture_tag": 0.0,
                "mean_prp_level": 0.0, "mean_capture_strength": 0.0,
                "max_capture_strength": 0.0, "mean_consolidation_level": 0.0,
                "mean_fragility": 0.0, "max_fragility": 0.0,
                "mean_replay_count": 0.0, "strong_tag_fraction": 0.0,
                "mean_ripple_strength": 0.0, "max_ripple_strength": 0.0,
                "global_prp_pool": float(self.global_prp_pool),
                "active_prp_buckets": int(len(self.bucket_prp_pool)),
                "fast_ema_norm": 0.0, "slow_mean_norm": 0.0,
                "drift": float(self.compute_drift()),
                "ripple_scalar_scan_count": int(self.ripple_scalar_scan_count),
                "ripple_vector_scan_count": int(self.ripple_vector_scan_count),
                "ripple_awake_bucket_scan_count": int(
                    self.ripple_awake_bucket_scan_count
                ),
                "ripple_awake_bucket_candidate_count": int(
                    self.ripple_awake_bucket_candidate_count
                ),
                "last_ripple_awake_bucket_count": int(
                    self.last_ripple_awake_bucket_count
                ),
                "last_ripple_awake_candidate_count": int(
                    self.last_ripple_awake_candidate_count
                ),
                "last_ripple_scan_mode": str(self.last_ripple_scan_mode),
                "last_replay_selection_report": dict(
                    self.last_replay_selection_report
                ),
                "last_replay_recall_report": dict(
                    self.last_replay_recall_report
                ),
                "last_sfa_sample_report": dict(
                    self.last_sfa_sample_report
                ),
                "last_replay_query_collection_report": dict(
                    self.last_replay_query_collection_report
                ),
                "last_query_memory_match_report": dict(
                    self.last_query_memory_match_report
                ),
                "last_bank_memory_match_report": dict(
                    self.last_bank_memory_match_report
                ),
                "last_runtime_concept_memory_lookup_report": dict(
                    self.last_runtime_concept_memory_lookup_report
                ),
                "last_frontier_gap_collection_report": dict(
                    self.last_frontier_gap_collection_report
                ),
                "last_awake_ripple_tag_report": dict(
                    self.last_awake_ripple_tag_report
                ),
                "last_recent_memory_window_report": dict(
                    self.last_recent_memory_window_report
                ),
                "last_recent_memory_tag_report": dict(
                    self.last_recent_memory_tag_report
                ),
                "last_anchor_capture_report": dict(
                    self.last_anchor_capture_report
                ),
                "summary_surface": "full_memory_summary.v1",
                "summary_full_memory_scan": False,
                "summary_scan_entry_count": 0,
                "summary_token_marker": int(token_marker),
                "summary_state_token": int(self._state_token),
                "summary_projection_read_only": False,
            }
            self._cached_summary = result
            self._cached_summary_token = token_marker
            return result

        # Vectorized batch computation
        local_prp_t = torch.tensor(self.slow_local_prp[:size], dtype=torch.float32)
        bucket_shares = torch.zeros(size, dtype=torch.float32)
        for idx in range(size):
            bs, _ = self._pool_share(idx)
            bucket_shares[idx] = bs
        global_share = 0.15 * float(self.global_prp_pool)
        prp_levels = (local_prp_t + bucket_shares + global_share).clamp(min=0.0)

        consol_t = torch.tensor(self.slow_consolidation_level[:size], dtype=torch.float32).clamp(0.0, 1.0)
        importance_t = torch.tensor(self.slow_importance[:size], dtype=torch.float32).clamp(min=1e-6)
        replay_age_t = torch.tensor(
            [max(0, token_marker - self.slow_last_replay_token[i]) for i in range(size)],
            dtype=torch.float32,
        )
        replay_count_t = torch.tensor(self.slow_replay_count[:size], dtype=torch.float32).clamp(min=0)
        tag_t = torch.tensor(self.slow_capture_tag[:size], dtype=torch.float32).clamp(min=0.0)
        ripple_t = torch.tensor(self.slow_ripple_strength[:size], dtype=torch.float32).clamp(0.0, 1.0)
        capture_levels = tag_t * prp_levels

        fm = max(1.0, float(self.functional_minute))
        age_pressure = 1.0 + torch.log1p(replay_age_t / fm)
        access_penalty = 1.0 / (1.0 + 0.5 * replay_count_t)
        stability_gap = (1.0 - consol_t).clamp(min=0.0)
        capture_gap = (1.0 - capture_levels).clamp(min=0.0)
        importance_scale = 0.5 + importance_t.clamp(max=1.0)
        fragility_levels = stability_gap * age_pressure * access_penalty * importance_scale * (0.5 + capture_gap)

        n_strong = sum(1 for v in self.slow_tag_is_strong[:size] if v)
        result = {
            "capacity": int(self.capacity),
            "size": int(size),
            "fill_fraction": float(size / max(1, self.capacity)),
            "n_seen": int(self.n_seen),
            "mean_importance": float(importance_t.mean().item()),
            "mean_capture_tag": float(tag_t.mean().item()),
            "mean_prp_level": float(prp_levels.mean().item()),
            "mean_capture_strength": float(capture_levels.mean().item()),
            "max_capture_strength": float(capture_levels.max().item()),
            "mean_consolidation_level": float(consol_t.mean().item()),
            "mean_fragility": float(fragility_levels.mean().item()),
            "max_fragility": float(fragility_levels.max().item()),
            "mean_replay_count": float(replay_count_t.mean().item()),
            "strong_tag_fraction": float(n_strong / max(1, size)),
            "mean_ripple_strength": float(ripple_t.mean().item()),
            "max_ripple_strength": float(ripple_t.max().item()),
            "ripple_scalar_scan_count": int(self.ripple_scalar_scan_count),
            "ripple_vector_scan_count": int(self.ripple_vector_scan_count),
            "ripple_awake_bucket_scan_count": int(
                self.ripple_awake_bucket_scan_count
            ),
            "ripple_awake_bucket_candidate_count": int(
                self.ripple_awake_bucket_candidate_count
            ),
            "last_ripple_awake_bucket_count": int(
                self.last_ripple_awake_bucket_count
            ),
            "last_ripple_awake_candidate_count": int(
                self.last_ripple_awake_candidate_count
            ),
            "last_ripple_scan_mode": str(self.last_ripple_scan_mode),
            "last_replay_selection_report": dict(
                self.last_replay_selection_report
            ),
            "last_replay_recall_report": dict(
                self.last_replay_recall_report
            ),
            "last_sfa_sample_report": dict(
                self.last_sfa_sample_report
            ),
            "last_replay_query_collection_report": dict(
                self.last_replay_query_collection_report
            ),
            "last_query_memory_match_report": dict(
                self.last_query_memory_match_report
            ),
            "last_bank_memory_match_report": dict(
                self.last_bank_memory_match_report
            ),
            "last_runtime_concept_memory_lookup_report": dict(
                self.last_runtime_concept_memory_lookup_report
            ),
            "last_frontier_gap_collection_report": dict(
                self.last_frontier_gap_collection_report
            ),
            "last_awake_ripple_tag_report": dict(
                self.last_awake_ripple_tag_report
            ),
            "last_recent_memory_window_report": dict(
                self.last_recent_memory_window_report
            ),
            "last_recent_memory_tag_report": dict(
                self.last_recent_memory_tag_report
            ),
            "last_anchor_capture_report": dict(
                self.last_anchor_capture_report
            ),
            "global_prp_pool": float(self.global_prp_pool),
            "active_prp_buckets": int(len(self.bucket_prp_pool)),
            "fast_ema_norm": float(torch.norm(self.fast_ema).item()) if isinstance(self.fast_ema, torch.Tensor) else 0.0,
            "slow_mean_norm": float(torch.norm(self._slow_mean).item()) if isinstance(self._slow_mean, torch.Tensor) else 0.0,
            "drift": float(self.compute_drift()),
            "summary_surface": "full_memory_summary.v1",
            "summary_full_memory_scan": True,
            "summary_scan_entry_count": int(size),
            "summary_token_marker": int(token_marker),
            "summary_state_token": int(self._state_token),
            "summary_projection_read_only": False,
        }
        self._cached_summary = result
        self._cached_summary_token = token_marker
        return result

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
            "slow_metadata": [None if item is None else dict(item) for item in self.slow_metadata],
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
            "slow_ripple_strength": list(self.slow_ripple_strength),
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
            "update_calls": int(self.update_calls),
            "admission_count": int(self.admission_count),
            "reservoir_rejection_count": int(self.reservoir_rejection_count),
            "optional_payload_copy_count": int(self.optional_payload_copy_count),
            "optional_payload_copy_avoidance_count": int(
                self.optional_payload_copy_avoidance_count
            ),
            "ripple_scalar_scan_count": int(self.ripple_scalar_scan_count),
            "ripple_vector_scan_count": int(self.ripple_vector_scan_count),
            "ripple_awake_bucket_scan_count": int(
                self.ripple_awake_bucket_scan_count
            ),
            "ripple_awake_bucket_candidate_count": int(
                self.ripple_awake_bucket_candidate_count
            ),
            "last_ripple_awake_bucket_count": int(
                self.last_ripple_awake_bucket_count
            ),
            "last_ripple_awake_candidate_count": int(
                self.last_ripple_awake_candidate_count
            ),
            "last_ripple_scan_mode": str(self.last_ripple_scan_mode),
            "last_replay_selection_report": dict(
                self.last_replay_selection_report
            ),
            "last_replay_recall_report": dict(
                self.last_replay_recall_report
            ),
            "last_sfa_sample_report": dict(
                self.last_sfa_sample_report
            ),
            "last_replay_query_collection_report": dict(
                self.last_replay_query_collection_report
            ),
            "last_query_memory_match_report": dict(
                self.last_query_memory_match_report
            ),
            "last_bank_memory_match_report": dict(
                self.last_bank_memory_match_report
            ),
            "last_runtime_concept_memory_lookup_report": dict(
                self.last_runtime_concept_memory_lookup_report
            ),
            "last_frontier_gap_collection_report": dict(
                self.last_frontier_gap_collection_report
            ),
            "last_awake_ripple_tag_report": dict(
                self.last_awake_ripple_tag_report
            ),
            "last_recent_memory_window_report": dict(
                self.last_recent_memory_window_report
            ),
            "last_recent_memory_tag_report": dict(
                self.last_recent_memory_tag_report
            ),
            "last_anchor_capture_report": dict(
                self.last_anchor_capture_report
            ),
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
        raw_metadata = _pad(snapshot.get("slow_metadata"), None, size)
        self.slow_metadata = [
            None if not isinstance(item, Mapping) else {str(key): value for key, value in dict(item).items()}
            for item in raw_metadata
        ]
        self.slow_bucket_ids = [None if value is None else int(value) for value in _pad(snapshot.get("slow_bucket_ids"), None, size)]
        self.slow_importance = [float(value) for value in _pad(snapshot.get("slow_importance"), 1.0, size)]

        self.slow_capture_tag = array(
            "d",
            (float(value) for value in _pad(snapshot.get("slow_capture_tag"), 0.0, size)),
        )
        strong_flags = snapshot.get("slow_tag_is_strong")
        if strong_flags is None:
            self.slow_tag_is_strong = array(
                "b",
                (
                    bool(value >= self.strong_event_threshold)
                    for value in self.slow_capture_tag
                ),
            )
        else:
            self.slow_tag_is_strong = array(
                "b",
                (bool(value) for value in _pad(strong_flags, False, size)),
            )
        self.slow_local_prp = array(
            "d",
            (float(value) for value in _pad(snapshot.get("slow_local_prp"), 0.0, size)),
        )
        timestamps = [int(value) for value in _pad(snapshot.get("slow_entry_timestamps"), 0, size)]
        self.slow_entry_timestamps = array("q", timestamps)
        self.slow_last_capture_token = _pad(snapshot.get("slow_last_capture_token"), None, size)
        self.slow_last_capture_token = [
            timestamps[idx] if value is None else int(value)
            for idx, value in enumerate(self.slow_last_capture_token)
        ]
        self.slow_consolidation_level = [float(value) for value in _pad(snapshot.get("slow_consolidation_level"), 0.0, size)]
        self._invalidate_bucket_consolidation_cache()
        self._rebuild_bucket_consolidation_cache(reason="load_state")
        self.slow_consolidation_events = [int(value) for value in _pad(snapshot.get("slow_consolidation_events"), 0, size)]
        self.slow_last_replay_token = _pad(snapshot.get("slow_last_replay_token"), None, size)
        self.slow_last_replay_token = [
            timestamps[idx] if value is None else int(value)
            for idx, value in enumerate(self.slow_last_replay_token)
        ]
        self.slow_replay_count = [int(value) for value in _pad(snapshot.get("slow_replay_count"), 0, size)]
        self.slow_ripple_strength = array(
            "d",
            [
                float(value)
                for value in _pad(
                    snapshot.get("slow_ripple_strength"),
                    0.0,
                    size,
                )
            ],
        )

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
        self._state_token = int(
            snapshot.get("state_token", max(self.slow_entry_timestamps, default=0))
        )
        self.n_seen = int(snapshot.get("n_seen", size))
        self.update_calls = int(snapshot.get("update_calls", self.n_seen))
        self.admission_count = int(snapshot.get("admission_count", size))
        self.reservoir_rejection_count = int(
            snapshot.get(
                "reservoir_rejection_count",
                max(0, self.update_calls - self.admission_count),
            )
        )
        self.optional_payload_copy_count = int(
            snapshot.get("optional_payload_copy_count", 0)
        )
        self.optional_payload_copy_avoidance_count = int(
            snapshot.get("optional_payload_copy_avoidance_count", 0)
        )
        self.ripple_scalar_scan_count = int(
            snapshot.get("ripple_scalar_scan_count", 0)
        )
        self.ripple_vector_scan_count = int(
            snapshot.get("ripple_vector_scan_count", 0)
        )
        self.ripple_awake_bucket_scan_count = int(
            snapshot.get("ripple_awake_bucket_scan_count", 0)
        )
        self.ripple_awake_bucket_candidate_count = int(
            snapshot.get("ripple_awake_bucket_candidate_count", 0)
        )
        self.last_ripple_awake_bucket_count = int(
            snapshot.get("last_ripple_awake_bucket_count", 0)
        )
        self.last_ripple_awake_candidate_count = int(
            snapshot.get("last_ripple_awake_candidate_count", 0)
        )
        self.last_ripple_scan_mode = str(
            snapshot.get("last_ripple_scan_mode", "not_run")
        )
        raw_replay_selection = snapshot.get("last_replay_selection_report")
        self.last_replay_selection_report = (
            dict(raw_replay_selection)
            if isinstance(raw_replay_selection, Mapping)
            else self._empty_replay_selection_report()
        )
        raw_replay_recall = snapshot.get("last_replay_recall_report")
        self.last_replay_recall_report = (
            dict(raw_replay_recall)
            if isinstance(raw_replay_recall, Mapping)
            else self._empty_replay_recall_report()
        )
        raw_sfa_sample = snapshot.get("last_sfa_sample_report")
        self.last_sfa_sample_report = (
            dict(raw_sfa_sample)
            if isinstance(raw_sfa_sample, Mapping)
            else self._empty_sfa_sample_report()
        )
        raw_replay_query_collection = snapshot.get(
            "last_replay_query_collection_report"
        )
        self.last_replay_query_collection_report = (
            dict(raw_replay_query_collection)
            if isinstance(raw_replay_query_collection, Mapping)
            else self._empty_replay_query_collection_report()
        )
        raw_query_memory_match = snapshot.get("last_query_memory_match_report")
        self.last_query_memory_match_report = (
            dict(raw_query_memory_match)
            if isinstance(raw_query_memory_match, Mapping)
            else self._empty_query_memory_match_report()
        )
        raw_bank_memory_match = snapshot.get("last_bank_memory_match_report")
        self.last_bank_memory_match_report = (
            dict(raw_bank_memory_match)
            if isinstance(raw_bank_memory_match, Mapping)
            else self._empty_bank_memory_match_report()
        )
        raw_runtime_concept_memory_lookup = snapshot.get(
            "last_runtime_concept_memory_lookup_report"
        )
        self.last_runtime_concept_memory_lookup_report = (
            dict(raw_runtime_concept_memory_lookup)
            if isinstance(raw_runtime_concept_memory_lookup, Mapping)
            else self._empty_runtime_concept_memory_lookup_report()
        )
        raw_frontier_gap = snapshot.get("last_frontier_gap_collection_report")
        self.last_frontier_gap_collection_report = (
            dict(raw_frontier_gap)
            if isinstance(raw_frontier_gap, Mapping)
            else self._empty_frontier_gap_collection_report()
        )
        raw_awake_ripple_tag = snapshot.get("last_awake_ripple_tag_report")
        self.last_awake_ripple_tag_report = (
            dict(raw_awake_ripple_tag)
            if isinstance(raw_awake_ripple_tag, Mapping)
            else self._empty_awake_ripple_tag_report()
        )
        raw_recent_memory_window = snapshot.get("last_recent_memory_window_report")
        self.last_recent_memory_window_report = (
            dict(raw_recent_memory_window)
            if isinstance(raw_recent_memory_window, Mapping)
            else self._empty_recent_memory_window_report()
        )
        raw_recent_memory_tag = snapshot.get("last_recent_memory_tag_report")
        self.last_recent_memory_tag_report = (
            dict(raw_recent_memory_tag)
            if isinstance(raw_recent_memory_tag, Mapping)
            else self._empty_recent_memory_tag_report()
        )
        raw_anchor_capture = snapshot.get("last_anchor_capture_report")
        self.last_anchor_capture_report = (
            dict(raw_anchor_capture)
            if isinstance(raw_anchor_capture, Mapping)
            else self._empty_anchor_capture_report()
        )
        self._rebuild_bucket_entry_index()
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
