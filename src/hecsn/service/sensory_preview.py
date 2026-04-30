from __future__ import annotations

import base64
from copy import deepcopy
from typing import Any, cast


class SensoryPreviewMixin:
    """Recent sensory preview payload helpers for the API/UI."""

    @staticmethod
    def _sensory_media_payload(media: dict[str, Any] | None) -> dict[str, Any] | None:
        if not isinstance(media, dict):
            return None
        raw_bytes = media.get("bytes")
        if not isinstance(raw_bytes, (bytes, bytearray)):
            return None
        mime_type = str(media.get("mime_type", "application/octet-stream"))
        data_url = f"data:{mime_type};base64,{base64.b64encode(bytes(raw_bytes)).decode('ascii')}"
        payload = {
            key: deepcopy(value)
            for key, value in media.items()
            if key != "bytes"
        }
        payload["byte_size"] = len(raw_bytes)
        payload["data_url"] = data_url
        return payload

    def sensory_previews(self, limit: int = 6) -> dict[str, Any]:
        acquired = self._lock.acquire(timeout=0.15)
        if not acquired:
            self._lock.acquire()
        try:
            previews = []
            for item in list(self._sensory_preview_history)[: max(1, int(limit))]:
                previews.append(
                    {
                        "preview_id": str(item.get("preview_id", "")),
                        "captured_at": str(item.get("captured_at", "")),
                        "source_name": str(item.get("source_name", "")),
                        "adapter": str(item.get("adapter", "")),
                        "text": str(item.get("text", "")),
                        "semantic_match": float(item.get("semantic_match", 0.0) or 0.0),
                        "modality_need": float(item.get("modality_need", 0.0) or 0.0),
                        "item_semantic_match": float(item.get("item_semantic_match", 0.0) or 0.0),
                        "item_candidates_considered": int(item.get("item_candidates_considered", 0) or 0),
                        "item_retrieval_lookahead": int(item.get("item_retrieval_lookahead", 1) or 1),
                        "selection_score": float(item.get("selection_score", 0.0) or 0.0),
                        "window_budget": int(item.get("window_budget", 0) or 0),
                        "topics": [str(topic) for topic in list(item.get("topics") or [])],
                        "focus_terms": [str(term) for term in list(item.get("focus_terms") or [])],
                        "metadata": deepcopy(item.get("metadata") or {}),
                        "visual": self._sensory_media_payload(cast(dict[str, Any] | None, item.get("visual"))),
                        "audio": self._sensory_media_payload(cast(dict[str, Any] | None, item.get("audio"))),
                    }
                )
            return {
                "count": int(len(self._sensory_preview_history)),
                "latest_preview_id": None if not self._sensory_preview_history else str(self._sensory_preview_history[0].get("preview_id", "")),
                "previews": previews,
            }
        finally:
            self._lock.release()

