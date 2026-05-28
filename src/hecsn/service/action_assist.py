"""Query action-assist helpers for Terminus.

This mixin ranks audited action history and can inject verified action evidence
into query results. It keeps action support separate from replay/dataset policy
and never bypasses the execution gates in ActionExecutor.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import re
from typing import Any, Mapping, Sequence, cast
from urllib.parse import urlparse

from hecsn.semantics.grounding_text import match_terms, salient_query_terms


class ActionAssistMixin:
    def _normalize_action_record(self, item: Any) -> dict[str, Any] | None:
        if not isinstance(item, Mapping):
            return None
        action_id = " ".join(str(item.get("action_id", "")).split()).strip()
        action_type = " ".join(str(item.get("action_type", item.get("type", ""))).split()).strip().lower()
        if not action_id or not action_type:
            return None
        verification = item.get("verification") if isinstance(item.get("verification"), Mapping) else {}
        topics = [
            " ".join(str(value).split()).strip().lower()
            for value in list(item.get("topics") or [])
            if " ".join(str(value).split()).strip()
        ]
        try:
            feedback_count = max(0, int(verification.get("feedback_count", 0) or 0))
        except (TypeError, ValueError):
            feedback_count = 0
        return {
            "action_id": action_id,
            "action_type": action_type,
            "inputs": deepcopy(dict(item.get("inputs") or {})),
            "predicted_outcome": " ".join(str(item.get("predicted_outcome", "")).split()).strip(),
            "actual_outcome": " ".join(str(item.get("actual_outcome", "")).split()).strip(),
            "verification": {
                "status": " ".join(str(verification.get("status", "unknown")).split()).strip().lower() or "unknown",
                "success": bool(verification.get("success", False)),
                "confidence": float(verification.get("confidence", 0.0) or 0.0),
                "contradiction": bool(verification.get("contradiction", False)),
                "summary": " ".join(str(verification.get("summary", "")).split()).strip(),
                "evidence": [deepcopy(dict(raw)) for raw in list(verification.get("evidence") or []) if isinstance(raw, Mapping)],
                "provenance": self._normalize_feedback_text(verification.get("provenance", ""), max_chars=32),
                "last_feedback_id": self._normalize_feedback_text(verification.get("last_feedback_id", ""), max_chars=80),
                "last_feedback_at": self._normalize_feedback_text(verification.get("last_feedback_at", ""), max_chars=80),
                "feedback_count": feedback_count,
            },
            "feedback": self._normalize_runtime_feedback_entries(item.get("feedback", [])),
            "feedback_status": self._normalize_feedback_text(item.get("feedback_status", ""), max_chars=32),
            "feedback_provenance": self._normalize_feedback_text(item.get("feedback_provenance", ""), max_chars=32),
            "provenance": self._normalize_feedback_text(item.get("provenance", ""), max_chars=32),
            "corrected_output": self._runtime_trace_export_safe_value(item.get("corrected_output")) if item.get("corrected_output") is not None else None,
            "topics": topics[:8],
            "recorded_at": str(item.get("recorded_at") or datetime.now(timezone.utc).isoformat()),
            "episode_text": " ".join(str(item.get("episode_text", "")).split()).strip(),
            "trigger_reason": " ".join(str(item.get("trigger_reason", "operator")).split()).strip().lower() or "operator",
            "trigger_query_text": " ".join(str(item.get("trigger_query_text", "")).split()).strip(),
        }

    @classmethod
    def _action_request_has_body(cls, inputs: Mapping[str, Any]) -> bool:
        if not isinstance(inputs, Mapping):
            return False
        if "json_body" not in inputs:
            return False
        body = inputs.get("json_body")
        if body is None:
            return False
        if isinstance(body, str):
            return bool(cls._normalize_action_text(body))
        if isinstance(body, Mapping):
            return bool(dict(body))
        if isinstance(body, Sequence) and not isinstance(body, (str, bytes, bytearray)):
            return bool(list(body))
        return True

    @classmethod
    def _api_request_record_matches_explicit_url(cls, record: Mapping[str, Any], explicit_url: str) -> bool:
        if str(record.get("action_type", "")) != "api_request":
            return False
        inputs = record.get("inputs") if isinstance(record.get("inputs"), Mapping) else {}
        if cls._normalize_action_text(inputs.get("url", "")) != explicit_url:
            return False
        method = cls._normalize_action_text(inputs.get("method", "GET")).upper() or "GET"
        if method != "GET":
            return False
        return not cls._action_request_has_body(inputs)

    @classmethod
    def _action_query_terms(cls, query_text: str) -> tuple[str, ...]:
        normalized = cls._normalize_action_text(query_text).lower()
        if not normalized:
            return ()
        terms = [term.lower() for term in salient_query_terms(normalized) if term]
        if not terms:
            terms = [
                token.lower()
                for token in re.findall(r"[a-zA-Z0-9_./:-]+", normalized)
                if len(token) >= 2
            ]
        deduped: list[str] = []
        seen: set[str] = set()
        for term in terms:
            compact = cls._normalize_action_text(term).lower()
            if not compact or compact in seen:
                continue
            deduped.append(compact)
            seen.add(compact)
        return tuple(deduped[:8])

    @classmethod
    def _action_focus_query_text(cls, query_text: str) -> str:
        normalized = cls._normalize_action_text(query_text)
        if not normalized:
            return ""
        stripped = re.sub(r"https?://[^\s'\")\]>]+", " ", normalized, flags=re.IGNORECASE)
        stripped = re.sub(
            r"(?:[A-Za-z0-9_.-]+[\\/])*[A-Za-z0-9_.-]+\.(?:py|md|txt|json|yaml|yml|toml|csv|ts|tsx|js|jsx|html|css|scss|ini|cfg|log|rst)",
            " ",
            stripped,
            flags=re.IGNORECASE,
        )
        focused_terms = cls._action_query_terms(stripped)
        if focused_terms:
            return " ".join(focused_terms[:6])
        fallback_terms = cls._action_query_terms(normalized)
        if fallback_terms:
            return " ".join(fallback_terms[:6])
        return normalized

    def _query_workspace_path_candidate_locked(self, query_text: str) -> str:
        normalized = self._normalize_action_text(query_text)
        if not normalized:
            return ""
        candidates = re.findall(
            r"(?:[A-Za-z0-9_.-]+[\\/])*[A-Za-z0-9_.-]+\.(?:py|md|txt|json|yaml|yml|toml|csv|ts|tsx|js|jsx|html|css|scss|ini|cfg|log|rst)",
            normalized,
            flags=re.IGNORECASE,
        )
        for raw in candidates:
            cleaned = raw.strip("`'\".,;:!?()[]{} ").replace("\\", "/")
            if not cleaned:
                continue
            candidate = Path(cleaned)
            resolved = candidate if candidate.is_absolute() else (self._action_root / candidate)
            try:
                resolved = resolved.resolve()
            except Exception:
                continue
            if resolved != self._action_root and self._action_root not in resolved.parents:
                continue
            if not resolved.exists() or not resolved.is_file():
                continue
            try:
                return str(resolved.relative_to(self._action_root)).replace("\\", "/")
            except Exception:
                return str(resolved)
        return ""

    @classmethod
    def _query_web_url_candidate(cls, query_text: str) -> str:
        normalized = cls._normalize_action_text(query_text)
        if not normalized:
            return ""
        matches = re.findall(r"https?://[^\s'\")\]>]+", normalized, flags=re.IGNORECASE)
        for raw in matches:
            cleaned = raw.strip("`'\".,;:!?()[]{} ")
            if cleaned:
                return cleaned
        return ""

    @classmethod
    def _query_api_url_candidate(cls, query_text: str) -> str:
        candidate = cls._query_web_url_candidate(query_text)
        if not candidate:
            return ""
        lowered = cls._normalize_action_text(query_text).lower()
        parsed = urlparse(candidate)
        path = (parsed.path or "").lower()
        if path.endswith(".json") or "/api/" in path or any(token in lowered for token in (" api ", " json ", " endpoint ")):
            return candidate
        return ""

    def _action_record_relevance_score_locked(self, record: Mapping[str, Any], query_text: str) -> float:
        normalized_query = self._normalize_action_text(query_text).lower()
        if not normalized_query:
            return 0.0
        explicit_api_url = self._query_api_url_candidate(query_text).lower()
        explicit_url = self._query_web_url_candidate(query_text).lower()
        record_url = self._normalize_action_text((record.get("inputs") or {}).get("url", "")).lower()
        if explicit_api_url and explicit_api_url == record_url:
            if str(record.get("action_type", "")) != "api_request":
                return 0.0
            if self._api_request_record_matches_explicit_url(record, explicit_api_url):
                return 1.0
        if explicit_url and explicit_url == record_url:
            return 1.0
        trigger_query = self._normalize_action_text(record.get("trigger_query_text", "")).lower()
        record_query = self._normalize_action_text((record.get("inputs") or {}).get("query_text", "")).lower()
        if normalized_query and normalized_query in {trigger_query, record_query}:
            return 1.0
        query_terms = set(self._action_query_terms(normalized_query))
        if not query_terms:
            return 0.0
        record_terms: set[str] = set(
            self._normalize_action_text(term).lower()
            for term in list(record.get("topics") or [])
            if self._normalize_action_text(term)
        )
        record_terms.update(self._action_query_terms(record_query))
        record_terms.update(self._action_query_terms(str((record.get("inputs") or {}).get("path", ""))))
        record_terms.update(self._action_query_terms(str((record.get("inputs") or {}).get("url", ""))))
        verification = record.get("verification") if isinstance(record.get("verification"), Mapping) else {}
        for raw_item in list(verification.get("evidence") or []):
            if not isinstance(raw_item, Mapping):
                continue
            record_terms.update(
                self._normalize_action_text(term).lower()
                for term in list(raw_item.get("matched_terms") or [])
                if self._normalize_action_text(term)
            )
            record_terms.update(self._action_query_terms(str(raw_item.get("snippet", ""))))
        if not record_terms:
            record_terms.update(self._action_query_terms(str(record.get("actual_outcome", ""))))
        overlap = len(query_terms & record_terms)
        if overlap <= 0:
            return 0.0
        return float(overlap) / float(max(1, len(query_terms)))

    def _recent_relevant_action_records_locked(
        self,
        query_text: str,
        *,
        statuses: Sequence[str] | None = None,
        limit: int = 4,
    ) -> list[dict[str, Any]]:
        allowed = {
            self._normalize_action_text(status).lower()
            for status in list(statuses or [])
            if self._normalize_action_text(status)
        }
        ranked: list[tuple[float, dict[str, Any]]] = []
        for record in self._action_history:
            verification = record.get("verification") if isinstance(record.get("verification"), Mapping) else {}
            status = self._normalize_action_text(verification.get("status", "")).lower()
            if allowed and status not in allowed:
                continue
            score = self._action_record_relevance_score_locked(record, query_text)
            if score < 0.34:
                continue
            ranked.append((score, deepcopy(record)))
        ranked.sort(
            key=lambda item: (
                float(item[0]),
                str(item[1].get("recorded_at", "")),
            ),
            reverse=True,
        )
        return [record for _, record in ranked[: max(1, int(limit))]]

    def _action_record_to_response_episodes_locked(
        self,
        record: Mapping[str, Any],
        *,
        query_text: str,
        limit: int = 3,
    ) -> list[dict[str, Any]]:
        verification = record.get("verification") if isinstance(record.get("verification"), Mapping) else {}
        if not bool(verification.get("success", False)):
            return []
        query_terms = list(self._action_query_terms(query_text))
        evidence_items = [
            dict(raw)
            for raw in list(verification.get("evidence") or [])
            if isinstance(raw, Mapping)
        ]
        action_seed = int(hashlib.sha256(str(record.get("action_id", "")).encode("utf-8")).hexdigest()[:8], 16)
        episodes: list[dict[str, Any]] = []
        for idx, evidence in enumerate(evidence_items[: max(1, int(limit))]):
            snippet = self._normalize_action_text(evidence.get("snippet", ""))
            if not snippet:
                continue
            matching = tuple(match_terms(query_terms, snippet))
            overlap_ratio = float(len(matching)) / float(max(1, len(query_terms)))
            exact_query = bool(evidence.get("exact_query", False))
            similarity = max(0.46, 0.56 + 0.34 * overlap_ratio + (0.10 if exact_query else 0.0))
            memory_index = -1 * int(action_seed + idx + 1)
            episodes.append(
                {
                    "text": snippet,
                    "raw_window": snippet,
                    "memory_index": memory_index,
                    "memory_indices": [memory_index],
                    "similarity": float(min(0.99, similarity)),
                    "importance": float(verification.get("confidence", 0.0) or 0.0),
                    "age_tokens": 0,
                    "match_count": 1,
                    "query_overlap": int(len(matching)),
                    "focus_overlap": 0,
                    "memory_focus_priority": 0.0,
                    "complete_sentence": int(snippet.endswith((".", "!", "?"))),
                    "clipped_overlap": 0,
                    "expansion_chars": 0,
                    "action_origin": str(record.get("action_id", "")),
                    "action_type": str(record.get("action_type", "")),
                    "source_path": self._normalize_action_text(evidence.get("path", "")),
                    "line_number": int(evidence.get("line_number", 0) or 0),
                }
            )
        if episodes:
            return episodes
        summary = self._normalize_action_text(record.get("actual_outcome", ""))
        if not summary:
            return []
        matching = tuple(match_terms(query_terms, summary))
        overlap_ratio = float(len(matching)) / float(max(1, len(query_terms)))
        memory_index = -1 * int(action_seed + 999)
        return [
            {
                "text": summary,
                "raw_window": summary,
                "memory_index": memory_index,
                "memory_indices": [memory_index],
                "similarity": float(min(0.95, 0.48 + 0.32 * overlap_ratio)),
                "importance": float(verification.get("confidence", 0.0) or 0.0),
                "age_tokens": 0,
                "match_count": 1,
                "query_overlap": int(len(matching)),
                "focus_overlap": 0,
                "memory_focus_priority": 0.0,
                "complete_sentence": int(summary.endswith((".", "!", "?"))),
                "clipped_overlap": 0,
                "expansion_chars": 0,
                "action_origin": str(record.get("action_id", "")),
                "action_type": str(record.get("action_type", "")),
            }
        ]

    def _augment_query_result_with_action_records_locked(
        self,
        query_result: dict[str, Any],
        *,
        query_text: str,
        records: Sequence[Mapping[str, Any]],
    ) -> int:
        query_summary = query_result.get("query_summary")
        if not isinstance(query_summary, dict):
            return 0
        injected: list[dict[str, Any]] = []
        seen_texts: set[str] = set()
        for record in records:
            for episode in self._action_record_to_response_episodes_locked(record, query_text=query_text):
                text_key = self._normalize_action_text(episode.get("text", "")).lower()
                if not text_key or text_key in seen_texts:
                    continue
                injected.append(episode)
                seen_texts.add(text_key)
        existing = [
            deepcopy(item)
            for item in list(query_summary.get("memory_episodes") or [])
            if isinstance(item, Mapping)
        ]
        for item in existing:
            text_key = self._normalize_action_text(item.get("text", item.get("raw_window", ""))).lower()
            if text_key:
                seen_texts.add(text_key)
        if injected:
            query_summary["memory_episodes"] = injected + [
                item
                for item in existing
                if self._normalize_action_text(item.get("text", item.get("raw_window", ""))).lower() not in {
                    self._normalize_action_text(injected_item.get("text", "")).lower()
                    for injected_item in injected
                }
            ]
        return int(len(injected))

    def _contradicted_action_note_locked(self, record: Mapping[str, Any]) -> str:
        actual = self._normalize_action_text(record.get("actual_outcome", ""))
        if actual:
            return f" I checked the workspace and observed: {actual}"
        return " I checked the workspace and found no additional grounded evidence there."

    def _should_auto_execute_action_locked(
        self,
        *,
        query_text: str,
        query_result: dict[str, Any],
        response: Mapping[str, Any],
    ) -> bool:
        if not self._normalize_action_text(query_text):
            return False
        gap_plan = query_result.get("gap_plan") if isinstance(query_result.get("gap_plan"), Mapping) else {}
        meaningful_gap = bool(
            gap_plan.get("unsupported_terms")
            or gap_plan.get("gap_terms")
            or gap_plan.get("weak_concepts")
            or float(gap_plan.get("grounded_fraction", 0.0) or 0.0) < 0.999
        )
        if not meaningful_gap:
            return False
        response_mode = self._normalize_action_text(response.get("response_mode", "")).lower()
        if response_mode == "insufficient_evidence":
            return True
        unsupported_terms = list(response.get("unsupported_terms") or gap_plan.get("unsupported_terms") or [])
        evidence_coverage = float(response.get("evidence_coverage", 0.0) or 0.0)
        return bool(unsupported_terms) and evidence_coverage < 0.85

    def _maybe_auto_action_assist_locked(
        self,
        *,
        query_text: str,
        query_result: dict[str, Any],
        response: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        explicit_api_url = self._query_api_url_candidate(query_text)
        explicit_url = self._query_web_url_candidate(query_text) if not explicit_api_url else ""
        explicit_path = self._query_workspace_path_candidate_locked(query_text) if not (explicit_api_url or explicit_url) else ""
        verified_records = self._recent_relevant_action_records_locked(query_text, statuses=("verified",), limit=2)
        if explicit_api_url:
            verified_records = [
                record
                for record in verified_records
                if self._api_request_record_matches_explicit_url(record, explicit_api_url)
            ]
        elif explicit_url:
            verified_records = [
                record
                for record in verified_records
                if str(record.get("action_type", "")) == "web_fetch"
                and self._normalize_action_text((record.get("inputs") or {}).get("url", "")) == explicit_url
            ]
        elif explicit_path:
            verified_records = [
                record
                for record in verified_records
                if str(record.get("action_type", "")) == "workspace_read"
                and self._normalize_action_text((record.get("inputs") or {}).get("path", "")) == explicit_path
            ]
        if verified_records:
            injected = self._augment_query_result_with_action_records_locked(
                query_result,
                query_text=query_text,
                records=verified_records,
            )
            return {
                "triggered": True,
                "executed": False,
                "reused_recent_action": True,
                "reason": "recent_verified_action",
                "used_in_response": bool(injected > 0),
                "result": deepcopy(verified_records[0]),
                "result_count": int(len(verified_records)),
                "response_episode_count": int(injected),
            }

        contradicted_records = self._recent_relevant_action_records_locked(query_text, statuses=("contradicted",), limit=1)
        if explicit_api_url:
            contradicted_records = [
                record
                for record in contradicted_records
                if self._api_request_record_matches_explicit_url(record, explicit_api_url)
            ]
        elif explicit_url:
            contradicted_records = [
                record
                for record in contradicted_records
                if str(record.get("action_type", "")) == "web_fetch"
                and self._normalize_action_text((record.get("inputs") or {}).get("url", "")) == explicit_url
            ]
        elif explicit_path:
            contradicted_records = [
                record
                for record in contradicted_records
                if str(record.get("action_type", "")) == "workspace_read"
                and self._normalize_action_text((record.get("inputs") or {}).get("path", "")) == explicit_path
            ]
        if not self._should_auto_execute_action_locked(query_text=query_text, query_result=query_result, response=response):
            response_mode = self._normalize_action_text(response.get("response_mode", "")).lower()
            unsupported_terms = list(response.get("unsupported_terms") or [])
            if contradicted_records and (response_mode == "insufficient_evidence" or unsupported_terms):
                return {
                    "triggered": True,
                    "executed": False,
                    "reused_recent_action": True,
                    "reason": "recent_contradicted_action",
                    "used_in_response": False,
                    "result": deepcopy(contradicted_records[0]),
                    "result_count": 1,
                    "response_episode_count": 0,
                    "response_note": self._contradicted_action_note_locked(contradicted_records[0]),
                }
            return None

        if contradicted_records:
            return {
                "triggered": True,
                "executed": False,
                "reused_recent_action": True,
                "reason": "recent_contradicted_action",
                "used_in_response": False,
                "result": deepcopy(contradicted_records[0]),
                "result_count": 1,
                "response_episode_count": 0,
                "response_note": self._contradicted_action_note_locked(contradicted_records[0]),
            }

        gap_plan = query_result.get("gap_plan") if isinstance(query_result.get("gap_plan"), Mapping) else {}
        retrieval_queries = [
            self._normalize_action_text(value)
            for value in list(gap_plan.get("retrieval_queries") or [])
            if self._normalize_action_text(value)
        ]
        search_query = retrieval_queries[0] if retrieval_queries else self._normalize_action_text(query_text)
        if explicit_api_url:
            focused_query = self._action_focus_query_text(query_text)
            action_result = self.execute_digital_action(
                {
                    "action_type": "api_request",
                    "url": explicit_api_url,
                    "query_text": focused_query,
                    "predicted_outcome": f"I expect requesting structured JSON from {explicit_api_url} to provide grounded evidence relevant to: {self._normalize_action_text(query_text)}.",
                },
                trigger_reason="query_gap_auto_api_request",
                trigger_query_text=query_text,
            )
            assist_reason = "query_gap_auto_api_request"
        elif explicit_url:
            focused_query = self._action_focus_query_text(query_text)
            action_result = self.execute_digital_action(
                {
                    "action_type": "web_fetch",
                    "url": explicit_url,
                    "query_text": focused_query,
                    "predicted_outcome": f"I expect fetching {explicit_url} to provide grounded evidence relevant to: {self._normalize_action_text(query_text)}.",
                },
                trigger_reason="query_gap_auto_fetch",
                trigger_query_text=query_text,
            )
            assist_reason = "query_gap_auto_fetch"
        elif explicit_path:
            focused_query = self._action_focus_query_text(query_text)
            action_result = self.execute_digital_action(
                {
                    "action_type": "workspace_read",
                    "path": explicit_path,
                    "query_text": focused_query,
                    "predicted_outcome": f"I expect reading {explicit_path} to provide grounded evidence relevant to: {self._normalize_action_text(query_text)}.",
                },
                trigger_reason="query_gap_auto_read",
                trigger_query_text=query_text,
            )
            assist_reason = "query_gap_auto_read"
        else:
            action_result = self.execute_digital_action(
                {
                    "action_type": "workspace_search",
                    "query_text": search_query,
                    "predicted_outcome": f"I expect workspace search to find grounded evidence relevant to: {self._normalize_action_text(query_text)}.",
                },
                trigger_reason="query_gap_auto_search",
                trigger_query_text=query_text,
            )
            assist_reason = "query_gap_auto_search"
        if not bool(action_result.get("accepted", False)):
            return {
                "triggered": True,
                "executed": False,
                "reused_recent_action": False,
                "reason": "auto_action_execution_failed",
                "used_in_response": False,
                "error": self._normalize_action_text(action_result.get("reason", "execution_failed")),
            }
        record = cast(dict[str, Any], action_result.get("result") or {})
        verification = record.get("verification") if isinstance(record.get("verification"), Mapping) else {}
        injected = 0
        if bool(verification.get("success", False)):
            injected = self._augment_query_result_with_action_records_locked(
                query_result,
                query_text=query_text,
                records=[record],
            )
        assist = {
            "triggered": True,
            "executed": True,
            "reused_recent_action": False,
            "reason": assist_reason,
            "used_in_response": bool(injected > 0),
            "result": deepcopy(record),
            "result_count": 1,
            "response_episode_count": int(injected),
        }
        if bool(verification.get("contradiction", False)):
            assist["response_note"] = self._contradicted_action_note_locked(record)
        return assist
