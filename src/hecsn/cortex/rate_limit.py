"""Shared NVIDIA NIM rate limiting utilities.

Both chat completions and embeddings count against the same per-key NIM budget.
This module centralises the limiter so cortex reasoning and memory embedding do
not accidentally oversubscribe the 40 RPM free-tier cap.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time

logger = logging.getLogger(__name__)

DEFAULT_MAX_RPM = 20


class SharedRateLimiter:
    """Global rate limiter shared across all NIM clients using the same API key.

    NVIDIA NIM free tier allows 40 requests/minute per API key across endpoints.
    Terminus defaults to 20 RPM for a conservative safety margin.
    """

    _instances: dict[str, "SharedRateLimiter"] = {}
    _lock = threading.Lock()

    def __init__(self, max_rpm: int = DEFAULT_MAX_RPM) -> None:
        self._max_rpm = max_rpm
        self._min_interval = 60.0 / max_rpm
        self._last_call = 0.0
        self._cooldown_until = 0.0
        self._call_lock = threading.Lock()
        self._recent_calls: list[float] = []

    @classmethod
    def for_key(cls, api_key: str, max_rpm: int = DEFAULT_MAX_RPM) -> "SharedRateLimiter":
        """Get or create a limiter bucket for a specific API key."""
        bucket = hashlib.sha256(api_key.encode("utf-8")).hexdigest() if api_key else "anonymous"
        with cls._lock:
            if bucket not in cls._instances:
                cls._instances[bucket] = cls(max_rpm=max_rpm)
            else:
                existing = cls._instances[bucket]
                if max_rpm < existing._max_rpm:
                    existing._max_rpm = max_rpm
                    existing._min_interval = 60.0 / max_rpm
            return cls._instances[bucket]

    def wait(self) -> None:
        """Block until another request can be issued safely."""
        with self._call_lock:
            now = time.time()

            if now < self._cooldown_until:
                wait_time = self._cooldown_until - now
                logger.debug("Rate limiter: honoring cooldown for %.1fs", wait_time)
                time.sleep(wait_time)
                now = time.time()

            self._recent_calls = [t for t in self._recent_calls if now - t < 60.0]

            if len(self._recent_calls) >= self._max_rpm:
                oldest = self._recent_calls[0]
                wait_time = 60.0 - (now - oldest) + 0.1
                if wait_time > 0:
                    logger.debug("Rate limiter: waiting %.1fs (budget exhausted)", wait_time)
                    time.sleep(wait_time)
                    now = time.time()
                    self._recent_calls = [t for t in self._recent_calls if now - t < 60.0]
            else:
                elapsed = now - self._last_call
                if elapsed < self._min_interval:
                    time.sleep(self._min_interval - elapsed)
                    now = time.time()

            self._last_call = now
            self._recent_calls.append(now)

    def backoff(self, cooldown_s: float) -> None:
        """Apply a shared cooldown after a 429 so all same-key clients slow down."""
        if cooldown_s <= 0:
            return
        with self._call_lock:
            self._cooldown_until = max(self._cooldown_until, time.time() + cooldown_s)
