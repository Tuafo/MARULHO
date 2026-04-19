"""MultiCortex — configurable multi-backend cortex for Terminus.

Supports multiple LLM backends:
- Ollama (local, e.g. Gemma 4)
- NVIDIA NIM (cloud, OpenAI-compatible API)
- FakeCortex (deterministic testing)

The cortex selection is configurable per-mode:
- Fast model for routine THINK/REFLECT
- Deep model for DREAM/ANSWER
- Local fallback when cloud unavailable

NVIDIA NIM provides free-tier access to frontier models
(meta/llama-3.3-70b-instruct, deepseek-ai/deepseek-r1,
qwen/qwen2.5-72b-instruct, etc.) with 40 req/min.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Optional

import httpx

from hecsn.cortex.core import (
    ContextPacket,
    CorticalCore,
    FakeCortex,
    ThoughtResult,
)

logger = logging.getLogger(__name__)

# Default NVIDIA NIM model assignments (based on 2026-04 frontier research)
DEFAULT_NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_FAST_MODEL = "nvidia/llama-3.1-nemotron-nano-8b-v1"
DEFAULT_DEEP_MODEL = "qwen/qwen3-next-80b-a3b-thinking"


class NIMCortex(CorticalCore):
    """NVIDIA NIM cortex — cloud-based frontier models via OpenAI-compatible API.

    Uses chat/completions endpoint which is standard across NIM models.
    Falls back to Ollama on connection failure.
    """

    def __init__(
        self,
        model: str = DEFAULT_FAST_MODEL,
        api_key: str | None = None,
        base_url: str = DEFAULT_NIM_BASE_URL,
        timeout_seconds: float = 60.0,
        temperature: float = 0.7,
        fallback_cortex: CorticalCore | None = None,
    ) -> None:
        # Store config (don't call super().__init__ — different API)
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout_seconds
        self.temperature = temperature
        self._api_key = api_key or os.environ.get("NVIDIA_API_KEY", "")
        self._fallback = fallback_cortex
        self._client = httpx.Client(
            timeout=httpx.Timeout(timeout_seconds),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        self._generation_count = 0
        self._fallback_count = 0

    def generate(self, context: ContextPacket) -> ThoughtResult:
        """Generate a thought via NVIDIA NIM chat/completions API."""
        from hecsn.cortex.prompts import MODE_PROMPTS

        system_prompt = MODE_PROMPTS.get(context.mode.value, MODE_PROMPTS["think"])
        user_prompt = context.to_user_prompt()

        t0 = time.perf_counter()
        try:
            raw = self._call_nim_chat(system_prompt, user_prompt, context.max_response_tokens)
        except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError) as exc:
            logger.warning("NIM inference failed (%s), trying fallback", exc)
            if self._fallback is not None:
                result = self._fallback.generate(context)
                self._fallback_count += 1
                return result
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
            "NIM gen #%d: %.0fms, parse=%s",
            self._generation_count, latency_ms, result.parse_success,
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

    @property
    def fallback_count(self) -> int:
        return self._fallback_count

    def close(self) -> None:
        self._client.close()
        if self._fallback:
            self._fallback.close()


class MultiCortex(CorticalCore):
    """Tiered cortex — fast model for routine, deep model for complex reasoning.

    Routes THINK/REFLECT to the fast model (low latency),
    DREAM/ANSWER to the deep model (high quality).
    Falls back to local Ollama if cloud unavailable.
    """

    def __init__(
        self,
        fast_cortex: CorticalCore,
        deep_cortex: CorticalCore | None = None,
    ) -> None:
        self._fast = fast_cortex
        self._deep = deep_cortex or fast_cortex
        self._generation_count = 0

    def generate(self, context: ContextPacket) -> ThoughtResult:
        """Route to appropriate cortex based on mode."""
        from hecsn.cortex.core import ThinkingMode

        # Use deep model for dream/answer, fast for think/reflect
        if context.mode in (ThinkingMode.DREAM, ThinkingMode.ANSWER):
            cortex = self._deep
        else:
            cortex = self._fast

        # Try preferred cortex, fall back to the other
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
    """Factory: create the best available cortex from environment config.

    Priority:
    1. MultiCortex (NIM fast + NIM deep) if NVIDIA_API_KEY set
    2. MultiCortex (NIM fast + Ollama deep) if NIM available
    3. CorticalCore (Ollama only) as fallback
    4. FakeCortex if nothing available
    """
    nim_key = os.environ.get("NVIDIA_API_KEY", "")
    ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    ollama_model = os.environ.get("OLLAMA_MODEL", "gemma4:e4b")

    nim_fast_model = os.environ.get("NIM_FAST_MODEL", DEFAULT_FAST_MODEL)
    nim_deep_model = os.environ.get("NIM_DEEP_MODEL", DEFAULT_DEEP_MODEL)

    # Try NIM + Ollama combo
    if nim_key:
        try:
            ollama_cortex = CorticalCore(
                model=ollama_model,
                base_url=ollama_url,
            )
            nim_fast = NIMCortex(
                model=nim_fast_model,
                api_key=nim_key,
                fallback_cortex=ollama_cortex,
                temperature=0.7,
            )
            nim_deep = NIMCortex(
                model=nim_deep_model,
                api_key=nim_key,
                fallback_cortex=ollama_cortex,
                temperature=0.8,
            )
            if nim_fast.is_available():
                logger.info("MultiCortex: NIM fast=%s, NIM deep=%s, Ollama fallback",
                            nim_fast_model, nim_deep_model)
                return MultiCortex(fast_cortex=nim_fast, deep_cortex=nim_deep)
            else:
                logger.info("NIM unavailable, falling back to Ollama-only")
                if ollama_cortex.is_available():
                    return ollama_cortex
        except Exception as exc:
            logger.info("NIM init failed: %s, trying Ollama", exc)

    # Ollama-only
    try:
        cortex = CorticalCore(model=ollama_model, base_url=ollama_url)
        if cortex.is_available():
            logger.info("Cortex: Ollama %s", ollama_model)
            return cortex
    except Exception:
        pass

    # Nothing available
    logger.warning("No LLM backend available — using FakeCortex")
    return FakeCortex()


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
