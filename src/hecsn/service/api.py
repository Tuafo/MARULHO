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

from .manager import HECSNServiceManager
from .schemas import (
    CheckpointActionResponse,
    CheckpointListResponse,
    CheckpointRecord,
    CheckpointRestoreRequest,
    CheckpointSaveRequest,
    FeedRequest,
    FeedResponse,
    QueryRequest,
    QueryResponse,
    RespondRequest,
    ResponseBundle,
    StatusResponse,
    TerminusConfigureRequest,
    TerminusRuntimeResponse,
    TerminusTickRequest,
    TraceHistoryResponse,
)


def _model_to_dict(model: object) -> dict:
    if hasattr(model, "model_dump"):
        return getattr(model, "model_dump")()
    return getattr(model, "dict")()


def _cors_origins() -> list[str]:
    """Read allowed CORS origins from HECSN_CORS_ORIGINS env var (comma-separated).

    Falls back to the canonical set of local dev origins when not set.
    """
    env_val = os.environ.get("HECSN_CORS_ORIGINS", "").strip()
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
) -> FastAPI:
    manager = HECSNServiceManager(
        checkpoint_path=checkpoint_path,
        trace_history_limit=trace_history_limit,
        trace_dir=trace_dir,
    )
    app = FastAPI(
        title="HECSN Local Service",
        version="0.1.0",
        description="Strict-evidence local service for querying and steering a checkpoint-backed HECSN Terminus runtime.",
    )
    app.state.hecsn_manager = manager
    app.router.on_shutdown.append(manager.close)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    dist_dir = Path(web_dist_dir) if web_dist_dir is not None else (Path("web") / "dist")
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
            "<h1>HECSN Local Service</h1>"
            f"{app_hint}"
            "<p>Interactive API docs are available at <a href=\"/docs\">/docs</a>.</p>"
            "</body></html>"
        )

    @app.get("/status", response_model=StatusResponse)
    def status() -> StatusResponse:
        return StatusResponse(**manager.status())

    @app.get("/checkpoints", response_model=CheckpointListResponse)
    def checkpoints() -> CheckpointListResponse:
        return CheckpointListResponse(checkpoints=[CheckpointRecord(**item) for item in manager.checkpoint_list()])

    @app.post("/checkpoint/save", response_model=CheckpointActionResponse)
    def save_checkpoint(request: CheckpointSaveRequest) -> CheckpointActionResponse:
        return CheckpointActionResponse(**manager.save_checkpoint(request.path))

    @app.post("/checkpoint/restore", response_model=CheckpointActionResponse)
    def restore_checkpoint(request: CheckpointRestoreRequest) -> CheckpointActionResponse:
        return CheckpointActionResponse(**manager.restore_checkpoint(request.path))

    @app.post("/feed", response_model=FeedResponse)
    def feed(request: FeedRequest) -> FeedResponse:
        return FeedResponse(**manager.feed(text=request.text))

    @app.post("/query", response_model=QueryResponse)
    def query(request: QueryRequest) -> QueryResponse:
        result = manager.query(
            query_text=request.query_text,
            context_text=request.context_text,
            top_k_candidates=request.top_k_candidates,
            top_k_memories=request.top_k_memories,
            top_chars=request.top_chars,
        )
        return QueryResponse(
            query_summary=result.get("query_summary") or {},
            concept_summary=result.get("concept_summary") or {},
            gap_plan=result.get("gap_plan") or {},
            service_state=result.get("service_state") or {},
        )

    @app.post("/respond", response_model=ResponseBundle)
    def respond(request: RespondRequest) -> ResponseBundle:
        try:
            return ResponseBundle(
                **manager.respond(
                    query_text=request.query_text,
                    context_text=request.context_text,
                    top_k_candidates=request.top_k_candidates,
                    top_k_memories=request.top_k_memories,
                    top_chars=request.top_chars,
                    max_evidence_items=request.max_evidence_items,
                    learn_mode=request.learn_mode,
                )
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/terminus", response_model=TerminusRuntimeResponse)
    def terminus_status() -> TerminusRuntimeResponse:
        return TerminusRuntimeResponse(**manager.terminus_status())

    @app.post("/terminus/configure", response_model=TerminusRuntimeResponse)
    def terminus_configure(request: TerminusConfigureRequest) -> TerminusRuntimeResponse:
        try:
            return TerminusRuntimeResponse(
                **manager.configure_terminus(
                    source_bank=[_model_to_dict(item) for item in request.source_bank],
                    tick_tokens=request.tick_tokens,
                    sleep_interval_seconds=request.sleep_interval_seconds,
                    repeat_sources=request.repeat_sources,
                    autonomy=None if request.autonomy is None else _model_to_dict(request.autonomy),
                )
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/terminus/start", response_model=TerminusRuntimeResponse)
    def terminus_start() -> TerminusRuntimeResponse:
        try:
            return TerminusRuntimeResponse(**manager.start_terminus())
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/terminus/stop", response_model=TerminusRuntimeResponse)
    def terminus_stop() -> TerminusRuntimeResponse:
        return TerminusRuntimeResponse(**manager.stop_terminus())

    @app.post("/terminus/tick", response_model=TerminusRuntimeResponse)
    def terminus_tick(request: TerminusTickRequest) -> TerminusRuntimeResponse:
        try:
            return TerminusRuntimeResponse(**manager.terminus_tick(steps=request.steps))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/terminus/quick-start")
    def terminus_quick_start(preset: str = Query("wikipedia")) -> dict[str, Any]:
        try:
            return manager.quick_start_terminus(preset=preset)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/terminus/presets")
    def terminus_presets() -> list[dict[str, Any]]:
        return HECSNServiceManager.quick_start_presets()

    @app.get("/traces", response_model=TraceHistoryResponse)
    def traces(limit: int = Query(20, ge=1, le=200)) -> TraceHistoryResponse:
        return TraceHistoryResponse(traces=manager.recent_traces(limit=limit))

    @app.get("/architecture")
    def architecture() -> dict[str, Any]:
        return manager.architecture_summary()

    @app.post("/grounding-probe/run")
    def grounding_probe_run() -> dict[str, Any]:
        return manager.run_grounding_probe()

    @app.get("/stream/status")
    async def stream_status(interval: float = Query(1.0, ge=0.25, le=10.0)) -> StreamingResponse:
        async def event_stream() -> AsyncIterator[str]:
            last_payload = ""
            heartbeat_counter = 0
            while True:
                payload = json.dumps(manager.telemetry_snapshot())
                if payload != last_payload:
                    yield f"event: status\ndata: {payload}\n\n"
                    last_payload = payload
                else:
                    # Send a heartbeat comment every ~15 s to keep proxies alive.
                    heartbeat_counter += 1
                    if heartbeat_counter * interval >= 15.0:
                        yield ": heartbeat\n\n"
                        heartbeat_counter = 0
                await asyncio.sleep(interval)

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get("/datasets")
    async def datasets():
        """List available multimodal datasets and their health status."""
        cwd = Path.cwd()
        result = []

        # N-MNIST
        nmnist_base = cwd / "N-MNIST"
        nmnist_train = nmnist_base / "Train"
        nmnist_files = 0
        if nmnist_train.exists():
            for digit_dir in nmnist_train.iterdir():
                if digit_dir.is_dir():
                    nmnist_files += sum(1 for _ in digit_dir.glob("*.bin"))
        result.append({
            "name": "N-MNIST",
            "type": "visual",
            "path": str(nmnist_base),
            "exists": nmnist_train.exists(),
            "file_count": nmnist_files,
            "description": "Neuromorphic MNIST (34x34 DVS events)",
        })

        # FSDD
        fsdd_base = cwd / "free-spoken-digit-dataset-master" / "recordings"
        fsdd_files = 0
        if fsdd_base.exists():
            fsdd_files = sum(1 for _ in fsdd_base.glob("*.wav"))
        result.append({
            "name": "FSDD",
            "type": "audio",
            "path": str(fsdd_base),
            "exists": fsdd_base.exists(),
            "file_count": fsdd_files,
            "description": "Free Spoken Digit Dataset (8kHz WAV)",
        })

        # Wikitext-103 (HuggingFace streaming)
        result.append({
            "name": "wikitext-103",
            "type": "text",
            "path": "wikitext/wikitext-103-raw-v1 (HuggingFace)",
            "exists": True,
            "file_count": None,
            "description": "Wikitext-103 via HuggingFace streaming",
        })

        return {"datasets": result}

    return app
