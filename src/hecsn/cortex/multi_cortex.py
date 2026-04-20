"""MultiCortex -- configurable multi-backend cortex for Terminus.

Supports NVIDIA NIM cloud models exclusively:
- Fast model (nemotron-nano-8b) for routine THINK/REFLECT
- Deep model (qwen3-next-80b) for DREAM/ANSWER

No local Ollama fallback. If NVIDIA_API_KEY is not set,
create_cortex_from_env() returns MockCortex for testing.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import httpx

from hecsn.cortex.core import (
    ContextPacket,
    CorticalCore,
    MockCortex,
    ThoughtResult,
)

logger = logging.getLogger(__name__)

# Default NVIDIA NIM model assignments (based on 2026-04 frontier research)
DEFAULT_NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_FAST_MODEL = "nvidia/llama-3.1-nemotron-nano-8b-v1"
DEFAULT_DEEP_MODEL = "qwen/qwen3-next-80b-a3b-thinking"


class NIMCortex(CorticalCore):
    """NVIDIA NIM cortex -- cloud-based frontier models via OpenAI-compatible API.

    Uses chat/completions endpoint which is standard across NIM models.
    No local fallback. If NIM fails, returns graceful error ThoughtResult.
    """

    def __init__(
        self,
        model: str = DEFAULT_FAST_MODEL,
        api_key: str | None = None,
        base_url: str = DEFAULT_NIM_BASE_URL,
        timeout_seconds: float = 60.0,
        temperature: float = 0.7,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout_seconds
        self.temperature = temperature
        self._api_key = api_key or os.environ.get("NVIDIA_API_KEY", "")
        self._client = httpx.Client(
            timeout=httpx.Timeout(timeout_seconds),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        self._generation_count = 0

    def generate(self, context: ContextPacket) -> ThoughtResult:
        """Generate a thought via NVIDIA NIM chat/completions API."""
        from hecsn.cortex.prompts import MODE_PROMPTS

        system_prompt = MODE_PROMPTS.get(context.mode.value, MODE_PROMPTS["think"])
        user_prompt = context.to_user_prompt()

        t0 = time.perf_counter()
        try:
            raw = self._call_nim_chat(system_prompt, user_prompt, context.max_response_tokens)
        except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError) as exc:
            logger.warning("NIM inference failed: %s", exc)
            return ThoughtResult(
                raw_text=str(exc),
                thought="[nim unavailable]",
                confidence=0.0,
                latency_ms=(time.perf_counter() - t0) * 1000,
                parse_success=False,
            )
        latency_ms = (time.perf_counter() - t0) * 1000

        self._generation_count += 1
        result = ThoughtResult.from_json(raw, latency_ms=latency_ms)
        logger.debug(
            "NIM gen #%d (%s): %.0fms, parse=%s",
            self._generation_count, self.model, latency_ms, result.parse_success,
        )
        return result

    def _call_nim_chat(self, system: str, user: str, max_tokens: int) -> str:
        """HTTP POST to NVIDIA NIM chat/completions endpoint."""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
            "temperature": self.temperature,
        }
        resp = self._client.post(
            f"{self.base_url}/chat/completions",
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()
        # OpenAI-compatible format
        choices = data.get("choices", [])
        if choices:
            return choices[0].get("message", {}).get("content", "")
        return ""

    def is_available(self) -> bool:
        """Check if NIM API is reachable."""
        if not self._api_key:
            return False
        try:
            resp = self._client.get(
                f"{self.base_url}/models",
                timeout=httpx.Timeout(5.0),
            )
            return resp.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException):
            return False

    @property
    def generation_count(self) -> int:
        return self._generation_count

    def close(self) -> None:
        self._client.close()


class MultiCortex(CorticalCore):
    """Tiered cortex -- fast model for routine, deep model for complex reasoning.

    Routes THINK/REFLECT to the fast model (low latency),
    DREAM/ANSWER to the deep model (high quality).
    """

    def __init__(
        self,
        fast_cortex: CorticalCore,
        deep_cortex: CorticalCore | None = None,
    ) -> None:
        self._fast = fast_cortex
        self._deep = deep_cortex or fast_cortex
        self._generation_count = 0
        self._temperature_override: float | None = None

    @property
    def temperature(self) -> float:
        if self._temperature_override is not None:
            return self._temperature_override
        return getattr(self._fast, "temperature", 0.7)

    @temperature.setter
    def temperature(self, value: float) -> None:
        self._temperature_override = value
        for cortex in (self._fast, self._deep):
            if hasattr(cortex, "temperature"):
                cortex.temperature = value

    @property
    def model(self) -> str:
        return f"multi({getattr(self._fast, 'model', 'fast')},{getattr(self._deep, 'model', 'deep')})"

    def generate(self, context: ContextPacket) -> ThoughtResult:
        """Route to appropriate cortex based on mode."""
        from hecsn.cortex.core import ThinkingMode

        # Use deep model for dream/answer, fast for think/reflect
        if context.mode in (ThinkingMode.DREAM, ThinkingMode.ANSWER):
            cortex = self._deep
        else:
            cortex = self._fast

        result = cortex.generate(context)
        self._generation_count += 1

        # If preferred cortex failed, try the other
        if not result.parse_success and cortex != self._fast:
            logger.info("Deep cortex failed, falling back to fast cortex")
            result = self._fast.generate(context)

        return result

    def is_available(self) -> bool:
        return self._fast.is_available() or self._deep.is_available()

    @property
    def generation_count(self) -> int:
        return self._generation_count

    def close(self) -> None:
        self._fast.close()
        if self._deep is not self._fast:
            self._deep.close()


def create_cortex_from_env() -> CorticalCore:
    """Factory: create cortex from environment config.

    Priority:
    1. MultiCortex (NIM fast + NIM deep) -- requires NVIDIA_API_KEY
    2. MockCortex if key not set (for testing only)

    Never launches Ollama or any local process.
    """
    nim_key = os.environ.get("NVIDIA_API_KEY", "")
    nim_fast_model = os.environ.get("NIM_FAST_MODEL", DEFAULT_FAST_MODEL)
    nim_deep_model = os.environ.get("NIM_DEEP_MODEL", DEFAULT_DEEP_MODEL)

    if nim_key:
        try:
            nim_fast = NIMCortex(
                model=nim_fast_model,
                api_key=nim_key,
                temperature=0.7,
            )
            nim_deep = NIMCortex(
                model=nim_deep_model,
                api_key=nim_key,
                temperature=0.8,
            )
            if nim_fast.is_available():
                logger.info("MultiCortex: NIM fast=%s, NIM deep=%s",
                            nim_fast_model, nim_deep_model)
                return MultiCortex(fast_cortex=nim_fast, deep_cortex=nim_deep)
            else:
                logger.warning("NIM API unreachable -- using MockCortex")
                return MockCortex()
        except Exception as exc:
            logger.warning("NIM init failed: %s -- using MockCortex", exc)
            return MockCortex()

    # No API key set
    logger.warning("NVIDIA_API_KEY not set -- using MockCortex (set key in .env)")
    return MockCortex()


def create_embedder_from_env():
    """Factory: create the best available embedder from environment config.

    Returns NIMEmbedder if NVIDIA_API_KEY is set, else SimpleEmbedder.
    """
    from hecsn.cortex.episodic_memory import NIMEmbedder, SimpleEmbedder

    nim_key = os.environ.get("NVIDIA_API_KEY", "")
    if nim_key:
        try:
            embedder = NIMEmbedder(api_key=nim_key)
            logger.info("Embedder: NIM (%s)", embedder.model)
            return embedder
        except Exception as exc:
            logger.info("NIM embedder failed: %s, using SimpleEmbedder", exc)

    return SimpleEmbedder(dim=128)
