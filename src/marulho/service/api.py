from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .brain_manager import MarulhoBrainServiceManager
from .api_schemas import (
    CheckpointListResponse,
    CheckpointRecord,
    CheckpointRestoreRequest,
    CheckpointSaveRequest,
)


DEFAULT_WEB_DIST_DIR = Path("MARULHO_UI") / "dist"


def _cors_origins() -> list[str]:
    """Read allowed CORS origins from MARULHO_CORS_ORIGINS env var (comma-separated).

    Falls back to the canonical set of local dev origins when not set.
    """
    env_val = os.environ.get("MARULHO_CORS_ORIGINS", "").strip()
    if env_val:
        return [o.strip() for o in env_val.split(",") if o.strip()]
    return [
        "http://127.0.0.1:5173",
        "http://localhost:5173",
        "http://127.0.0.1:4173",
        "http://localhost:4173",
        "http://127.0.0.1:3000",
        "http://localhost:3000",
    ]


def create_app(
    checkpoint_path: str | Path,
    trace_history_limit: int = 200,
    trace_dir: str | Path | None = None,
    web_dist_dir: str | Path | None = None,
    env_root: str | Path | None = None,
) -> FastAPI:
    manager = MarulhoBrainServiceManager(
        checkpoint_path=checkpoint_path,
        trace_history_limit=trace_history_limit,
        trace_dir=trace_dir,
        env_root=env_root,
    )
    app = FastAPI(
        title="MARULHO Local Service",
        version="0.1.0",
        description="Thin service adapter over the checkpoint-backed MarulhoBrain runtime.",
    )
    app.state.marulho_manager = manager
    runtime = manager.runtime_facade
    app.state.marulho_runtime = runtime
    app.router.on_shutdown.append(manager.close)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    dist_dir = Path(web_dist_dir) if web_dist_dir is not None else DEFAULT_WEB_DIST_DIR
    app.state.web_dist_dir = dist_dir
    if dist_dir.exists():
        app.mount("/app", StaticFiles(directory=dist_dir, html=True), name="app")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/", response_class=HTMLResponse)
    def root() -> str:
        app_hint = '<p>Built frontend available at <a href="/app">/app</a>.</p>' if dist_dir.exists() else ""
        return (
            "<html><body style=\"font-family:Segoe UI, sans-serif; padding: 24px;\">"
            "<h1>MARULHO Local Service</h1>"
            f"{app_hint}"
            "<p>Interactive API docs are available at <a href=\"/docs\">/docs</a>.</p>"
            "</body></html>"
        )

    @app.get("/brain/status")
    def brain_status() -> dict[str, Any]:
        return runtime.brain_status()

    @app.get("/brain/checkpoints", response_model=CheckpointListResponse)
    def brain_checkpoints() -> CheckpointListResponse:
        return CheckpointListResponse(checkpoints=[CheckpointRecord(**item) for item in runtime.checkpoint_list()])

    @app.get("/brain/traces")
    def brain_traces(limit: int = Query(20, ge=1, le=200)) -> dict[str, Any]:
        return {
            "surface": "marulho_brain_trace_history.v1",
            "traces": runtime.brain_traces(limit=limit),
        }

    @app.post("/brain/feed")
    def brain_feed(request: dict[str, Any]) -> dict[str, Any]:
        text = str(request.get("text") or "")
        if not text:
            raise HTTPException(status_code=400, detail="text is required")
        return runtime.brain_feed(
            text=text,
            source=str(request.get("source") or "operator"),
            learn=bool(request.get("learn", False)),
        )

    @app.post("/brain/tick")
    def brain_tick(request: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = dict(request or {})
        return runtime.brain_tick(
            tokens=int(payload.get("tokens", payload.get("steps", 128)) or 128),
            quantum_tokens=int(payload.get("quantum_tokens", 16) or 16),
            source=payload.get("source"),
            allow_sleep_maintenance=bool(payload.get("allow_sleep_maintenance", False)),
        )

    @app.post("/brain/generate")
    def brain_generate(request: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = dict(request or {})
        prompt = payload.get("prompt")
        return runtime.brain_generate(
            prompt=None if prompt is None else str(prompt),
            max_tokens=int(payload.get("max_tokens", 64) or 64),
        )

    @app.post("/brain/replay")
    def brain_replay(request: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = dict(request or {})
        return runtime.brain_replay(
            window=str(payload.get("window") or "recent_surprise"),
            cycles=int(payload.get("cycles", 1) or 1),
        )

    @app.post("/brain/grow-prune")
    def brain_grow_prune(request: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = dict(request or {})
        return runtime.brain_grow_prune(budget=str(payload.get("budget") or "small"))

    @app.post("/brain/checkpoint/save")
    def brain_checkpoint_save(request: CheckpointSaveRequest) -> dict[str, Any]:
        return runtime.save_brain_checkpoint(request.path)

    @app.post("/brain/checkpoint/restore")
    def brain_checkpoint_restore(request: CheckpointRestoreRequest) -> dict[str, Any]:
        restore = runtime.restore_checkpoint(request.path)
        return {
            "surface": "marulho_brain_checkpoint_restore.v1",
            "restore": restore,
            "brain": runtime.brain_status(),
        }

    @app.post("/brain/start")
    def brain_start(request: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = dict(request or {})
        return runtime.brain_start(
            tick_tokens=int(payload.get("tick_tokens", payload.get("tokens", 128)) or 128),
            quantum_tokens=int(payload.get("quantum_tokens", 16) or 16),
            interval_seconds=float(payload.get("interval_seconds", 0.25) or 0.25),
            source=None if payload.get("source") is None else str(payload.get("source")),
            allow_sleep_maintenance=bool(payload.get("allow_sleep_maintenance", False)),
        )

    @app.post("/brain/stop")
    def brain_stop(request: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = dict(request or {})
        return runtime.brain_stop(
            timeout_seconds=float(payload.get("timeout_seconds", 2.0) or 2.0),
        )

    @app.get("/brain/stream/status")
    async def brain_status_stream(interval: float = Query(1.0, ge=0.25, le=30.0)) -> StreamingResponse:
        async def events() -> AsyncIterator[str]:
            while True:
                yield f"data: {json.dumps(runtime.brain_status())}\n\n"
                await asyncio.sleep(interval)

        return StreamingResponse(events(), media_type="text/event-stream")

    return app
