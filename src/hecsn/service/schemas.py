from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class FeedRequest(BaseModel):
    text: str = Field(..., min_length=1)


class QueryRequest(BaseModel):
    query_text: str = Field(..., min_length=1)
    context_text: str | None = None
    top_k_candidates: int = Field(5, ge=1, le=64)
    top_k_memories: int = Field(5, ge=1, le=64)
    top_chars: int = Field(6, ge=1, le=32)


class RespondRequest(QueryRequest):
    learn_mode: Literal["none", "user_only", "user_and_selected_evidence"] = "user_and_selected_evidence"
    max_evidence_items: int = Field(3, ge=1, le=8)


class AcquisitionRunRequest(BaseModel):
    preset: str = Field(..., min_length=1)
    policy: Literal["active", "round_robin"] = "active"
    acquisition_slots: int | None = Field(None, ge=1, le=16)
    acquisition_tokens: int | None = Field(None, ge=1, le=20000)
    save_checkpoint_path: str | None = None


class CheckpointSaveRequest(BaseModel):
    path: str | None = None


class CheckpointRestoreRequest(BaseModel):
    path: str = Field(..., min_length=1)


class CheckpointRecord(BaseModel):
    path: str
    name: str
    size_bytes: int
    modified_at: str


class CheckpointListResponse(BaseModel):
    checkpoints: list[CheckpointRecord]


class AcquisitionPresetListResponse(BaseModel):
    presets: list[str]


class CheckpointActionResponse(BaseModel):
    path: str
    dirty_state: bool
    state_revision: int
    token_count: int


class FeedResponse(BaseModel):
    feed_summary: dict[str, Any]
    dirty_state: bool
    state_revision: int


class StatusResponse(BaseModel):
    checkpoint_path: str
    dirty_state: bool
    state_revision: int
    token_count: int
    last_winner: int | None
    context_supported: bool
    context_state_norm: float
    trace_history_size: int
    trace_storage_dir: str
    last_trace_id: str | None
    last_trace_created_at: str | None
    checkpoint_metadata: dict[str, Any]
    runtime_scope: dict[str, Any]
    memory_store: dict[str, Any]
    concept_store: dict[str, Any]


class ResponseBundle(BaseModel):
    trace_id: str
    trace_path: str
    created_at: str
    query_result: dict[str, Any]
    response: dict[str, Any]
    learning: dict[str, Any] | None
    dirty_state: bool
    state_revision: int


class TraceHistoryResponse(BaseModel):
    traces: list[dict[str, Any]]


class AcquisitionActionResponse(BaseModel):
    trace_id: str
    trace_path: str
    created_at: str
    preset: str
    policy: str
    acquisition_result: dict[str, Any]
    checkpoint_save: dict[str, Any] | None
    dirty_state: bool
    state_revision: int
    token_count: int


class QueryResponse(BaseModel):
    query_summary: dict[str, Any]
    concept_summary: dict[str, Any]
    gap_plan: dict[str, Any]
    service_state: dict[str, Any]


class BenchmarkReportsResponse(BaseModel):
    reports_root: str
    benchmarks: list[dict[str, Any]]
