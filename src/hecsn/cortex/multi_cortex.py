"""MultiCortex -- NVIDIA NIM cortex routing for Terminus.

Supports NVIDIA NIM cloud models exclusively:
- Fast model (nemotron-nano-8b) for routine THINK/REFLECT
- Deep model (llama-3.3-70b) for DREAM/ANSWER

No local Ollama fallback and no silent runtime downgrade.
`create_cortex_from_env()` raises if NVIDIA NIM is unavailable.
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
    ThoughtResult,
)
from hecsn.cortex.rate_limit import DEFAULT_MAX_RPM, SharedRateLimiter

logger = logging.getLogger(__name__)

# Default NVIDIA NIM model assignments (based on 2026-04 frontier research)
DEFAULT_NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_FAST_MODEL = "nvidia/llama-3.1-nemotron-nano-8b-v1"
DEFAULT_DEEP_MODEL = "meta/llama-3.3-70b-instruct"


class NIMCortex(CorticalCore):
    """NVIDIA NIM cortex -- cloud-based frontier models via OpenAI-compatible API.

    Uses chat/completions endpoint which is standard across NIM models.
    No local fallback. If NIM fails, returns graceful error ThoughtResult.
    Rate limiting is shared across all instances using the same API key.
    """

    def __init__(
        self,
        model: str = DEFAULT_FAST_MODEL,
        api_key: str | None = None,
        base_url: str = DEFAULT_NIM_BASE_URL,
        timeout_seconds: float = 60.0,
        temperature: float = 0.7,
        max_rpm: int = DEFAULT_MAX_RPM,
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
        self.backend_kind = "external_llm"
        self.llm_backed = True
        self.external_service = "nvidia_nim"
        self._generation_count = 0
        self._rate_limiter = SharedRateLimiter.for_key(self._api_key, max_rpm=max_rpm)
        self._consecutive_failures = 0
        self._last_success_time = time.time()
        self._max_retries = 1  # Only 1 retry on 429 to conserve rate budget

    def generate(self, context: ContextPacket) -> ThoughtResult:
        """Generate a thought via NVIDIA NIM chat/completions API."""
        from hecsn.cortex.prompts import MODE_PROMPTS, PHASE_PROMPTS

        # Use phase-specific prompt for deliberation chains, otherwise mode prompt
        if context.deliberation_phase and context.deliberation_phase in PHASE_PROMPTS:
            system_prompt = PHASE_PROMPTS[context.deliberation_phase]
        else:
            system_prompt = MODE_PROMPTS.get(context.mode.value, MODE_PROMPTS["think"])
        user_prompt = context.to_user_prompt()

        t0 = time.perf_counter()
        for attempt in range(1 + self._max_retries):
            try:
                raw = self._call_nim_chat(system_prompt, user_prompt, context.max_response_tokens)
                self._consecutive_failures = 0
                self._last_success_time = time.time()
                break
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 429:
                    wait = self._retry_after_seconds(exc.response) or (6.0 * (attempt + 1))
                    self._rate_limiter.backoff(wait)
                    if attempt < self._max_retries:
                        logger.info("NIM 429, backing off %.1fs (attempt %d)", wait, attempt + 1)
                        continue
                logger.warning("NIM inference failed: %s", exc)
                self._consecutive_failures += 1
                return ThoughtResult(
                    raw_text=str(exc),
                    thought="[nim unavailable]",
                    confidence=0.0,
                    latency_ms=(time.perf_counter() - t0) * 1000,
                    parse_success=False,
                )
            except (httpx.ConnectError, httpx.TimeoutException) as exc:
                logger.warning("NIM inference failed: %s", exc)
                self._consecutive_failures += 1
                return ThoughtResult(
                    raw_text=str(exc),
                    thought="[nim unavailable]",
                    confidence=0.0,
                    latency_ms=(time.perf_counter() - t0) * 1000,
                    parse_success=False,
                )
        else:
            # All retries exhausted
            return ThoughtResult(
                raw_text="rate limited after retries",
                thought="[nim rate limited]",
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

    @staticmethod
    def _retry_after_seconds(response: httpx.Response) -> float | None:
        """Parse Retry-After as seconds when provided by the server."""
        value = response.headers.get("retry-after")
        if value is None:
            return None
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        return parsed if parsed > 0 else None

    def _call_nim_chat(self, system: str, user: str, max_tokens: int) -> str:
        """HTTP POST to NVIDIA NIM chat/completions endpoint.

        Rate limiting is handled by the shared SharedRateLimiter which
        tracks all calls across all NIMCortex instances using the same key.
        """
        self._rate_limiter.wait()

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


def cortex_ok(cortex: CorticalCore) -> bool:
    """Check if a cortex is healthy enough to use.

    Returns False if the cortex has had 2+ consecutive failures,
    indicating it's rate-limited or down. Resets after 120s.
    """
    failures = getattr(cortex, "_consecutive_failures", 0)
    if failures < 2:
        return True
    # Allow retry after 120 seconds
    last_success = getattr(cortex, "_last_success_time", 0.0)
    if time.time() - last_success > 120.0:
        cortex._consecutive_failures = 0  # type: ignore[attr-defined]
        return True
    return False


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
        self.backend_kind = "tiered_cortex"
        self.llm_backed = any(
            bool(getattr(cortex, "llm_backed", False))
            for cortex in (self._fast, self._deep)
        )
        services = {
            str(service)
            for service in (
                getattr(self._fast, "external_service", None),
                getattr(self._deep, "external_service", None),
            )
            if service
        }
        self.external_service = ",".join(sorted(services)) if services else None
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
        """Route to appropriate cortex based on mode and phase.

        Budget-aware routing (40 RPM NIM cap):
        - Chain phases (question/reason/synthesize/dream_compose/dream_test):
          always fast model. These are structured continuation/validation calls
          where prompt obedience and budget efficiency matter more than raw size.
        - Generic dream/answer: deep model if healthy, else fast
        - Think/reflect: fast model
        """
        from hecsn.cortex.core import ThinkingMode

        # Chain continuation phases: always fast (budget preservation)
        if context.deliberation_phase in ("question", "reason", "synthesize", "dream_compose", "dream_test"):
            use_deep = False
        else:
            # Use deep model for dream/answer, fast for think/reflect
            # BUT skip deep if it's been failing
            use_deep = (
                context.mode in (ThinkingMode.DREAM, ThinkingMode.ANSWER)
                and cortex_ok(self._deep)
            )

        cortex = self._deep if use_deep else self._fast

        result = cortex.generate(context)
        self._generation_count += 1

        # If deep cortex failed, try fast (only costs 1 more call)
        if not result.parse_success and cortex is not self._fast:
            logger.info("Deep cortex failed, falling back to fast cortex")
            result = self._fast.generate(context)

        return result

    def is_available(self) -> bool:
        return self._fast.is_available() or self._deep.is_available()

    @property
    def generation_count(self) -> int:
        return self._generation_count

    def backend_report(self) -> dict[str, Any]:
        return {
            "implementation": type(self).__name__,
            "model": self.model,
            "backend_kind": self.backend_kind,
            "llm_backed": self.llm_backed,
            "external_service": self.external_service,
            "replaceable": True,
            "retention_gate": "runtime_evidence",
            "available": None,
            "generation_count": int(self.generation_count),
            "fast": self._fast.backend_report(),
            "deep": self._deep.backend_report(),
        }

    def close(self) -> None:
        self._fast.close()
        if self._deep is not self._fast:
            self._deep.close()


def create_cortex_from_env() -> CorticalCore:
    """Factory: create a strict NVIDIA NIM cortex from environment config.

    Requires NVIDIA_API_KEY in environment. Raises RuntimeError if
    the key is missing or NIM is unreachable.
    """
    nim_key = os.environ.get("NVIDIA_API_KEY", "")
    nim_fast_model = os.environ.get("NIM_FAST_MODEL", DEFAULT_FAST_MODEL)
    nim_deep_model = os.environ.get("NIM_DEEP_MODEL", DEFAULT_DEEP_MODEL)
    try:
        nim_max_rpm = int(os.environ.get("NIM_MAX_RPM", str(DEFAULT_MAX_RPM)))
    except ValueError:
        nim_max_rpm = DEFAULT_MAX_RPM

    if not nim_key:
        raise RuntimeError(
            "NVIDIA_API_KEY not set. Add it to .env or set the environment variable. "
            "Get a free key at https://build.nvidia.com/"
        )

    nim_fast = NIMCortex(
        model=nim_fast_model,
        api_key=nim_key,
        temperature=0.7,
        max_rpm=nim_max_rpm,
    )
    nim_deep = NIMCortex(
        model=nim_deep_model,
        api_key=nim_key,
        temperature=0.8,
        max_rpm=nim_max_rpm,
    )

    # Retry health check up to 3 times (NIM can be briefly unreachable)
    import time as _time
    for attempt in range(3):
        if nim_fast.is_available():
            logger.info(
                "MultiCortex: NIM fast=%s, NIM deep=%s, budget=%drpm",
                nim_fast_model,
                nim_deep_model,
                nim_max_rpm,
            )
            return MultiCortex(fast_cortex=nim_fast, deep_cortex=nim_deep)
        if attempt < 2:
            logger.warning("NIM health check failed (attempt %d/3), retrying in 5s...", attempt + 1)
            _time.sleep(5)

    raise RuntimeError(
        f"NIM API unreachable after 3 attempts (model={nim_fast_model}). "
        "Check your NVIDIA_API_KEY and network connection."
    )


def create_embedder_from_env(*, allow_fallback: bool = False):
    """Factory: create an embedder from environment config.

    By default this is strict for runtime use: if NVIDIA_API_KEY is missing or
    the NIM embedder cannot be initialised, raise RuntimeError instead of
    silently downgrading to SimpleEmbedder. Tests and offline tools can opt into
    fallback mode with ``allow_fallback=True``.
    """
    from hecsn.cortex.episodic_memory import NIMEmbedder, SimpleEmbedder

    nim_key = os.environ.get("NVIDIA_API_KEY", "")
    try:
        nim_max_rpm = int(os.environ.get("NIM_MAX_RPM", str(DEFAULT_MAX_RPM)))
    except ValueError:
        nim_max_rpm = DEFAULT_MAX_RPM

    if not nim_key:
        if allow_fallback:
            logger.info("Embedder: SimpleEmbedder (no NVIDIA_API_KEY)")
            return SimpleEmbedder(dim=128)
        raise RuntimeError(
            "NVIDIA_API_KEY not set. Cortex embeddings require NIM; "
            "set the key or call create_embedder_from_env(allow_fallback=True)."
        )

    try:
        embedder = NIMEmbedder(api_key=nim_key, max_rpm=nim_max_rpm, allow_fallback=allow_fallback)
        logger.info(
            "Embedder: NIM (%s, budget=%drpm%s)",
            embedder.model,
            nim_max_rpm,
            ", fallback enabled" if allow_fallback else "",
        )
        return embedder
    except Exception as exc:
        if allow_fallback:
            logger.info("NIM embedder failed: %s, using SimpleEmbedder", exc)
            return SimpleEmbedder(dim=128)
        raise RuntimeError(f"NIM embedder unavailable: {exc}") from exc
