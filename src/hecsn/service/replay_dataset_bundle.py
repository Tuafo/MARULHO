from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from typing import Any, Mapping, Sequence, cast

REPLAY_DATASET_SCHEMA_VERSION = 1
REPLAY_DATASET_BUNDLE_TRAINING_ROLE = "replay_dataset_bundle_preview_only_not_training_operator_approved"
DEFAULT_REPLAY_DATASET_DECONTAMINATION_TERMS: tuple[str, ...] = (
    "arc_agi",
    "arc-agi",
    "arc agi",
    "benchmark",
    "heldout",
    "held-out",
    "evaluation_fixture",
)
DEFAULT_REPLAY_DATASET_EXPORT_LIMIT = 20
MAX_REPLAY_DATASET_EXPORT_LIMIT = 50
DEFAULT_REPLAY_DATASET_BUNDLE_RETENTION_DAYS = 3650
_RUNTIME_TRACE_EXPORT_UNSAFE_KEYS = {
    "checkpoint_path",
    "env",
    "env_root",
    "path",
    "raw_environment",
    "root_path",
    "runtime_env",
    "trace_path",
    "workspace_root",
}


class ReplayDatasetBundleMixin:
    """Operator-gated replay dataset packaging helpers.

    This mixin keeps dataset packaging separate from the main runtime manager.
    It builds preview/export artifacts only; it does not train adapters, mutate
    memory, promote facts, execute actions, start sleep, or call tools.
    """

    def replay_dataset_bundle(
        self,
        *,
        operator_id: str,
        confirmation: bool,
        operator_note: str | None = None,
        limit: int = DEFAULT_REPLAY_DATASET_EXPORT_LIMIT,
        endpoint: str | None = None,
        holdout_fraction: float = 0.2,
        eval_fraction: float = 0.2,
        seed: int | None = None,
        retention_days: int = DEFAULT_REPLAY_DATASET_BUNDLE_RETENTION_DAYS,
        decontamination_terms: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            return self._replay_dataset_bundle_payload_locked(
                operator_id=operator_id,
                confirmation=confirmation,
                operator_note=operator_note,
                limit=limit,
                endpoint=endpoint,
                holdout_fraction=holdout_fraction,
                eval_fraction=eval_fraction,
                seed=seed,
                retention_days=retention_days,
                decontamination_terms=decontamination_terms,
            )

    @staticmethod
    def _replay_dataset_bundle_canonical_json(value: Any) -> str:
        return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), default=str)

    @classmethod
    def _replay_dataset_bundle_hash(cls, value: Any) -> str:
        return hashlib.sha256(cls._replay_dataset_bundle_canonical_json(value).encode("utf-8")).hexdigest()

    @staticmethod
    def _replay_dataset_bundle_fraction(value: float, *, name: str) -> float:
        try:
            fraction = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a number.") from exc
        if fraction < 0.0 or fraction > 0.8:
            raise ValueError(f"{name} must be between 0.0 and 0.8.")
        return fraction

    def _replay_dataset_bundle_terms(self, values: Sequence[str] | None) -> list[str]:
        terms: list[str] = []
        for raw in (*DEFAULT_REPLAY_DATASET_DECONTAMINATION_TERMS, *(values or ())):
            term = self._normalize_feedback_text(raw, max_chars=80).lower()
            if term and term not in terms:
                terms.append(term)
        return terms

    @staticmethod
    def _replay_dataset_bundle_timestamp(value: Any) -> datetime | None:
        text = " ".join(str(value or "").split()).strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            timestamp = datetime.fromisoformat(text)
        except ValueError:
            return None
        if timestamp.tzinfo is None:
            return timestamp.replace(tzinfo=timezone.utc)
        return timestamp.astimezone(timezone.utc)

    def _replay_dataset_bundle_item_fingerprint(self, item: Mapping[str, Any]) -> str:
        return self._replay_dataset_bundle_hash(
            {
                "endpoint": item.get("endpoint"),
                "operation": item.get("operation"),
                "verification_label": item.get("verification_label"),
                "provenance_label": item.get("provenance_label"),
                "example_type": item.get("example_type"),
                "sft_example": item.get("sft_example"),
                "preference_pair": item.get("preference_pair"),
            }
        )

    def _replay_dataset_bundle_exclusion_reasons(
        self,
        item: Mapping[str, Any],
        *,
        fingerprint: str,
        seen_fingerprints: set[str],
        now: datetime,
        retention_days: int,
        decontamination_terms: Sequence[str],
    ) -> list[str]:
        reasons: list[str] = []
        example_type = self._normalize_feedback_text(item.get("example_type", ""), max_chars=80).lower()
        provenance = self._normalize_feedback_text(item.get("provenance_label", ""), max_chars=80).lower()
        if example_type == "excluded_preview_context" or (
            not bool(item.get("has_positive_example")) and not bool(item.get("has_negative_example"))
        ):
            reasons.append("no_positive_or_negative_training_signal")
        if provenance in {"dreamed", "synthetic", "dreamed_synthetic", "simulated"}:
            reasons.append("synthetic_or_dreamed_content_blocked")

        canonical = self._replay_dataset_bundle_canonical_json(item).lower()
        matched_terms = [term for term in decontamination_terms if term and term in canonical]
        if matched_terms:
            reasons.append("decontamination_blocked:" + ",".join(matched_terms[:6]))

        if retention_days > 0:
            timestamp = self._replay_dataset_bundle_timestamp(item.get("timestamp"))
            if timestamp is not None and now - timestamp > timedelta(days=retention_days):
                reasons.append("retention_window_expired")

        if not reasons and fingerprint in seen_fingerprints:
            reasons.append("duplicate_item_fingerprint")
        return reasons

    def _replay_dataset_bundle_filter_items(
        self,
        items: Sequence[Any],
        *,
        now: datetime,
        retention_days: int,
        decontamination_terms: Sequence[str],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        packaged: list[dict[str, Any]] = []
        excluded: list[dict[str, Any]] = []
        seen_fingerprints: set[str] = set()
        for raw in items:
            if not isinstance(raw, Mapping):
                continue
            item = dict(raw)
            fingerprint = self._replay_dataset_bundle_item_fingerprint(item)
            reasons = self._replay_dataset_bundle_exclusion_reasons(
                item,
                fingerprint=fingerprint,
                seen_fingerprints=seen_fingerprints,
                now=now,
                retention_days=retention_days,
                decontamination_terms=decontamination_terms,
            )
            if reasons:
                excluded.append(
                    {
                        "source_item_id": item.get("item_id"),
                        "target_type": item.get("target_type"),
                        "target_id": item.get("target_id"),
                        "endpoint": item.get("endpoint"),
                        "operation": item.get("operation"),
                        "verification_label": item.get("verification_label"),
                        "provenance_label": item.get("provenance_label"),
                        "example_type": item.get("example_type"),
                        "dedupe_fingerprint": fingerprint,
                        "excluded_reasons": reasons,
                    }
                )
                continue

            seen_fingerprints.add(fingerprint)
            packaged.append(
                {
                    "package_item_id": f"bundle-item-{fingerprint[:16]}",
                    "dedupe_fingerprint": fingerprint,
                    "source_item_id": item.get("item_id"),
                    "dataset_role": "replay_dataset_bundle_item_preview",
                    "training_role": REPLAY_DATASET_BUNDLE_TRAINING_ROLE,
                    "target_type": item.get("target_type"),
                    "target_id": item.get("target_id"),
                    "trace_id": item.get("trace_id"),
                    "endpoint": item.get("endpoint"),
                    "operation": item.get("operation"),
                    "timestamp": item.get("timestamp"),
                    "verification_label": item.get("verification_label"),
                    "provenance_label": item.get("provenance_label"),
                    "example_type": item.get("example_type"),
                    "has_positive_example": bool(item.get("has_positive_example")),
                    "has_negative_example": bool(item.get("has_negative_example")),
                    "sft_example": item.get("sft_example"),
                    "preference_pair": item.get("preference_pair"),
                    "feedback_summary": item.get("feedback_summary"),
                    "replay_plan_context": item.get("replay_plan_context"),
                    "replay_sample_linkage": item.get("replay_sample_linkage"),
                    "source_item": item,
                    "safety_flags": {
                        "preview_only": True,
                        "export_only": True,
                        "training_started": False,
                        "sleep_started": False,
                        "memory_verification_promoted": False,
                        "feedback_posted": False,
                        "digital_action_executed": False,
                        "external_calls_made": False,
                        "memory_mutated": False,
                        "not_promoted": True,
                        "eligible_for_training": False,
                        "requires_separate_training_approval": True,
                    },
                }
            )
        return packaged, excluded

    def _replay_dataset_bundle_split_items(
        self,
        items: Sequence[Mapping[str, Any]],
        *,
        holdout_fraction: float,
        eval_fraction: float,
        seed: int | None,
    ) -> dict[str, list[dict[str, Any]]]:
        splits: dict[str, list[dict[str, Any]]] = {"train": [], "holdout": [], "eval": []}
        ranked = sorted(
            [dict(item) for item in items],
            key=lambda item: self._replay_dataset_bundle_hash(
                {
                    "seed": seed,
                    "fingerprint": item.get("dedupe_fingerprint"),
                    "target_id": item.get("target_id"),
                }
            ),
        )
        total = len(ranked)
        if total == 0:
            return splits

        eval_count = int(math.floor(total * eval_fraction))
        holdout_count = int(math.floor(total * holdout_fraction))
        if total >= 3 and eval_fraction > 0.0:
            eval_count = max(1, eval_count)
        if total >= 3 and holdout_fraction > 0.0:
            holdout_count = max(1, holdout_count)
        while eval_count + holdout_count >= total:
            if holdout_count >= eval_count and holdout_count > 0:
                holdout_count -= 1
            elif eval_count > 0:
                eval_count -= 1
            else:
                break

        for index, item in enumerate(ranked):
            if index < eval_count:
                split = "eval"
            elif index < eval_count + holdout_count:
                split = "holdout"
            else:
                split = "train"
            with_split = dict(item)
            with_split["split"] = split
            splits[split].append(with_split)
        return splits

    @staticmethod
    def _replay_dataset_bundle_split_summary(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        example_type_counts: Counter[str] = Counter()
        provenance_counts: Counter[str] = Counter()
        positive_count = 0
        negative_count = 0
        for item in items:
            example_type_counts[str(item.get("example_type", "unknown") or "unknown")] += 1
            provenance_counts[str(item.get("provenance_label", "unknown") or "unknown")] += 1
            if bool(item.get("has_positive_example")):
                positive_count += 1
            if bool(item.get("has_negative_example")):
                negative_count += 1
        return {
            "count": len(items),
            "positive_count": positive_count,
            "negative_count": negative_count,
            "example_type_counts": dict(example_type_counts),
            "provenance_counts": dict(provenance_counts),
        }

    def _replay_dataset_bundle_payload_locked(
        self,
        *,
        operator_id: str,
        confirmation: bool,
        operator_note: str | None,
        limit: int,
        endpoint: str | None,
        holdout_fraction: float,
        eval_fraction: float,
        seed: int | None,
        retention_days: int,
        decontamination_terms: Sequence[str] | None,
    ) -> dict[str, Any]:
        normalized_operator_id = self._normalize_feedback_text(operator_id, max_chars=160)
        if not normalized_operator_id:
            raise ValueError("Replay dataset bundle operator_id is required.")
        if not bool(confirmation):
            raise ValueError("Replay dataset bundle confirmation=true is required for operator-approved packaging.")
        normalized_note = self._normalize_feedback_text(operator_note or "", max_chars=2000)
        holdout = self._replay_dataset_bundle_fraction(holdout_fraction, name="holdout_fraction")
        eval_split = self._replay_dataset_bundle_fraction(eval_fraction, name="eval_fraction")
        if holdout + eval_split >= 1.0:
            raise ValueError("holdout_fraction + eval_fraction must leave at least one training split.")
        try:
            retention_window_days = int(retention_days)
        except (TypeError, ValueError) as exc:
            raise ValueError("retention_days must be an integer.") from exc
        if retention_window_days < 0:
            raise ValueError("retention_days must be non-negative.")

        count = min(MAX_REPLAY_DATASET_EXPORT_LIMIT, max(1, int(limit)))
        created_at = datetime.now(timezone.utc).isoformat()
        now = self._replay_dataset_bundle_timestamp(created_at) or datetime.now(timezone.utc)
        before = self._replay_sample_state_counts_locked()
        preview = self._replay_dataset_preview_payload_locked(
            limit=count,
            endpoint=endpoint,
            created_at=created_at,
        )
        source_items = [dict(item) for item in preview.get("items", []) if isinstance(item, Mapping)]
        normalized_terms = self._replay_dataset_bundle_terms(decontamination_terms)
        packaged_items, excluded_items = self._replay_dataset_bundle_filter_items(
            source_items,
            now=now,
            retention_days=retention_window_days,
            decontamination_terms=normalized_terms,
        )
        splits = self._replay_dataset_bundle_split_items(
            packaged_items,
            holdout_fraction=holdout,
            eval_fraction=eval_split,
            seed=seed,
        )
        split_summaries = {
            name: self._replay_dataset_bundle_split_summary(items)
            for name, items in splits.items()
        }
        all_packaged = [item for items in splits.values() for item in items]
        source_preview_summary = self._replay_dataset_summary_from_payload(preview)
        source_preview_hash = self._replay_dataset_bundle_hash(
            [item.get("dedupe_fingerprint") or self._replay_dataset_bundle_item_fingerprint(item) for item in source_items]
        )
        bundle_core = {
            "schema_version": REPLAY_DATASET_SCHEMA_VERSION,
            "source_preview_hash": source_preview_hash,
            "operator_id": normalized_operator_id,
            "holdout_fraction": holdout,
            "eval_fraction": eval_split,
            "seed": seed,
            "retention_days": retention_window_days,
            "items": [
                {
                    "split": item.get("split"),
                    "dedupe_fingerprint": item.get("dedupe_fingerprint"),
                    "target_type": item.get("target_type"),
                    "target_id": item.get("target_id"),
                    "example_type": item.get("example_type"),
                }
                for item in all_packaged
            ],
            "excluded": [
                {
                    "dedupe_fingerprint": item.get("dedupe_fingerprint"),
                    "target_id": item.get("target_id"),
                    "excluded_reasons": item.get("excluded_reasons"),
                }
                for item in excluded_items
            ],
        }
        bundle_hash = self._replay_dataset_bundle_hash(bundle_core)
        after = self._replay_sample_state_counts_locked()
        safety_flags = self._replay_dataset_safety_flags(before=before, after=after)
        safety_flags.update(
            {
                "operator_approved": True,
                "bundle_packaged": True,
                "bundle_written": False,
                "requires_separate_training_approval": True,
                "eligible_for_training": False,
            }
        )
        payload = {
            "schema_version": REPLAY_DATASET_SCHEMA_VERSION,
            "export_kind": "terminus_replay_dataset_bundle_preview",
            "training_role": REPLAY_DATASET_BUNDLE_TRAINING_ROLE,
            "description": (
                "Operator-approved replay dataset bundle preview with deterministic deduplication, "
                "decontamination checks, retention filtering, and train/holdout/eval splits. This "
                "artifact does not train adapters, mutate memory, promote facts, post feedback, "
                "execute actions, start sleep, or make external calls."
            ),
            "created_at": created_at,
            "endpoint": "/terminus/replay-dataset/bundle",
            "source_endpoint": "/terminus/replay-dataset/preview",
            "limit": count,
            "max_limit": MAX_REPLAY_DATASET_EXPORT_LIMIT,
            "filter_endpoint": self._normalize_runtime_trace_export_filter(endpoint),
            "bundle_id": f"terminus-replay-dataset-bundle-v{REPLAY_DATASET_SCHEMA_VERSION}-{bundle_hash[:12]}",
            "bundle_version": f"v{REPLAY_DATASET_SCHEMA_VERSION}.{bundle_hash[:12]}",
            "bundle_hash": bundle_hash,
            "source_preview_hash": source_preview_hash,
            "operator_approval": {
                "required": True,
                "approved": True,
                "confirmation": True,
                "operator_id": normalized_operator_id,
                "operator_note": normalized_note,
                "approved_at": created_at,
                "scope": "package_preview_export_only",
            },
            "packaging_policy": {
                "deduplication": "sha256_canonical_sft_preference_context",
                "decontamination": {
                    "enabled": True,
                    "blocked_terms": list(normalized_terms),
                    "blocked_sources": ["ARC", "benchmarks", "heldout_evaluation_data", "evaluation_fixtures"],
                },
                "retention": {
                    "enabled": retention_window_days > 0,
                    "retention_days": retention_window_days,
                    "timestamp_field": "timestamp",
                },
                "split_policy": {
                    "deterministic_hash_seed": seed,
                    "holdout_fraction": holdout,
                    "eval_fraction": eval_split,
                    "train_fraction_floor": max(0.0, 1.0 - holdout - eval_split),
                },
                "provenance_rules": {
                    "verified_or_corrected_positive_required_for_sft": True,
                    "contradicted_or_failed_outputs_are_negative_lessons": True,
                    "dreamed_synthetic_simulated_blocked": True,
                    "contradicted_content_not_promoted_as_fact": True,
                },
            },
            "source_count": len(source_items),
            "count": len(all_packaged),
            "excluded_count": len(excluded_items),
            "positive_count": sum(1 for item in all_packaged if bool(item.get("has_positive_example"))),
            "negative_count": sum(1 for item in all_packaged if bool(item.get("has_negative_example"))),
            "preference_pair_count": sum(1 for item in all_packaged if item.get("preference_pair") is not None),
            "sft_count": sum(1 for item in all_packaged if item.get("sft_example") is not None),
            "negative_only_count": sum(1 for item in all_packaged if item.get("sft_example") is None and bool(item.get("has_negative_example"))),
            "split_counts": {name: len(items) for name, items in splits.items()},
            "split_summaries": split_summaries,
            "source_preview_summary": source_preview_summary,
            "manifest": {
                "schema_version": REPLAY_DATASET_SCHEMA_VERSION,
                "bundle_hash": bundle_hash,
                "source_preview_hash": source_preview_hash,
                "item_hashes": [str(item.get("dedupe_fingerprint")) for item in all_packaged],
                "excluded_hashes": [str(item.get("dedupe_fingerprint")) for item in excluded_items],
                "artifact_role": "preview_export_only_not_training",
            },
            "splits": splits,
            "excluded_items": excluded_items,
            "safety_flags": safety_flags,
            "before": before,
            "after": after,
            "excluded_fields": sorted(_RUNTIME_TRACE_EXPORT_UNSAFE_KEYS),
        }
        if not all_packaged:
            payload["empty_reason"] = "no_items_survived_bundle_packaging_gate"
        return cast(dict[str, Any], self._runtime_trace_export_safe_value(payload))

    def _replay_dataset_safety_flags(self, *, before: Mapping[str, int], after: Mapping[str, int]) -> dict[str, Any]:
        return {
            "preview_only": True,
            "export_only": True,
            "training_started": False,
            "sleep_started": False,
            "memory_verification_promoted": False,
            "feedback_posted": False,
            "digital_action_executed": False,
            "external_calls_made": False,
            "memory_mutated": False,
            "state_revision_mutated": int(after.get("state_revision", 0)) != int(before.get("state_revision", 0)),
            "token_count_mutated": int(after.get("token_count", 0)) != int(before.get("token_count", 0)),
            "action_history_mutated": int(after.get("action_history_count", 0)) != int(before.get("action_history_count", 0)),
            "feedback_mutated": int(after.get("feedback_count", 0)) != int(before.get("feedback_count", 0)),
            "not_promoted": True,
            "eligible_for_training": False,
        }
