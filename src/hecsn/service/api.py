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

from .manager import HECSNServiceManager, PUBLIC_ACQUISITION_PRESETS
from .schemas import (
    AcquisitionActionResponse,
    AcquisitionPresetListResponse,
    AcquisitionRunRequest,
    BenchmarkReportsResponse,
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
    TraceHistoryResponse,
)
from .benchmark_reports import load_benchmark_reports


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
        description="Strict-evidence local service for querying, feeding, and inspecting a checkpoint-backed HECSN runtime.",
    )
    app.state.hecsn_manager = manager
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

    @app.get("/acquisition/presets", response_model=AcquisitionPresetListResponse)
    def acquisition_presets() -> AcquisitionPresetListResponse:
        return AcquisitionPresetListResponse(presets=list(PUBLIC_ACQUISITION_PRESETS))

    @app.post("/acquisition/run", response_model=AcquisitionActionResponse)
    def acquisition_run(request: AcquisitionRunRequest) -> AcquisitionActionResponse:
        try:
            return AcquisitionActionResponse(
                **manager.acquire(
                    preset=request.preset,
                    policy=request.policy,
                    acquisition_slots=request.acquisition_slots,
                    acquisition_tokens=request.acquisition_tokens,
                    save_checkpoint_path=request.save_checkpoint_path,
                )
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/traces", response_model=TraceHistoryResponse)
    def traces(limit: int = Query(20, ge=1, le=200)) -> TraceHistoryResponse:
        return TraceHistoryResponse(traces=manager.recent_traces(limit=limit))

    @app.get("/reports/benchmarks", response_model=BenchmarkReportsResponse)
    def benchmark_reports(max_points: int = Query(120, ge=20, le=400)) -> BenchmarkReportsResponse:
        return BenchmarkReportsResponse(**load_benchmark_reports(max_points=max_points))

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

    return app
