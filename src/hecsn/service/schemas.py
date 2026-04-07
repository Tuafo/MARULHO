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


class TerminusSourceSpec(BaseModel):
    name: str = Field(..., min_length=1)
    source: str = Field(..., min_length=1)
    source_type: Literal["auto", "file", "hf", "web"] = "auto"
    text_field: str = Field("text", min_length=1)
    hf_config: str | None = None
    metadata: dict[str, Any] | None = None


class TerminusCatalogEntrySpec(BaseModel):
    name: str = Field(..., min_length=1)
    source: str = Field(..., min_length=1)
    source_type: Literal["auto", "hf", "web"] = "auto"
    summary: str | None = None
    title: str | None = None
    description: str | None = None
    text_field: str = Field("text", min_length=1)
    hf_config: str | None = None
    query_text: str | None = None
    provider: str | None = None
    tags: list[str] = Field(default_factory=list)
    terms: list[str] = Field(default_factory=list)
    catalog_priority: float | None = None
    prior_weight: float | None = None


class TerminusCatalogSpec(BaseModel):
    name: str | None = None
    catalog_mode: Literal["semantic_registry", "live_remote_search"]
    catalog_entries: list[TerminusCatalogEntrySpec] = Field(default_factory=list)
    catalog_limit: int = Field(8, ge=1, le=128)
    catalog_probe_pool_limit: int | None = Field(None, ge=1, le=256)
    catalog_focus_text: str | None = None
    catalog_focus_terms: list[str] = Field(default_factory=list)
    catalog_diversity_weight: float = 0.20
    catalog_semantic_weight: float = 1.0
    catalog_prior_weight: float = 1.0
    catalog_exclude_sources: list[str] = Field(default_factory=list)
    catalog_exclude_names: list[str] = Field(default_factory=list)
    catalog_providers: list[str] = Field(default_factory=list)
    catalog_queries_per_provider: int = Field(2, ge=1, le=16)
    catalog_provider_result_limit: int = Field(4, ge=1, le=32)
    catalog_provider_timeout_seconds: float = Field(15.0, ge=1.0, le=60.0)


TerminusCandidateSpec = TerminusSourceSpec | TerminusCatalogSpec


class TerminusAutonomyConfig(BaseModel):
    enabled: bool = True
    policy: Literal["active", "round_robin"] = "active"
    candidate_bank: list[TerminusCandidateSpec] = Field(default_factory=list)
    trigger_interval_tokens: int = Field(4096, ge=1, le=200000)
    candidate_train_tokens: int = Field(768, ge=1, le=20000)
    probe_tokens: int = Field(96, ge=1, le=20000)
    acquisition_tokens: int = Field(512, ge=1, le=20000)
    acquisition_slots: int = Field(1, ge=1, le=16)
    gap_exploration_bonus: float = 0.03
    gap_ambiguity_weight: float = 0.4
    gap_switch_weight: float = 0.2
    gap_margin_reference: float = 0.12
    coverage_balance_penalty: float = 0.2
    gap_focus_margin: float = 0.05
    scout_commit_tokens: int = Field(0, ge=0, le=20000)
    scout_top_k: int = Field(1, ge=1, le=16)
    semantic_shortlist_size: int = Field(0, ge=0, le=32)
    semantic_shortlist_gap_weight: float = 0.5
    semantic_shortlist_affinity_weight: float = 0.5


class TerminusConfigureRequest(BaseModel):
    source_bank: list[TerminusSourceSpec] = Field(..., min_length=1)
    tick_tokens: int = Field(128, ge=1, le=20000)
    sleep_interval_seconds: float = Field(0.25, ge=0.01, le=60.0)
    repeat_sources: bool = True
    autonomy: TerminusAutonomyConfig | None = None


class TerminusTickRequest(BaseModel):
    steps: int = Field(1, ge=1, le=128)


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
    terminus_runtime: dict[str, Any]


class TerminusRuntimeResponse(BaseModel):
    terminus_runtime: dict[str, Any]
    tick_summaries: list[dict[str, Any]] | None = None
    dirty_state: bool
    state_revision: int
    token_count: int


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


class QueryResponse(BaseModel):
    query_summary: dict[str, Any]
    concept_summary: dict[str, Any]
    gap_plan: dict[str, Any]
    service_state: dict[str, Any]
