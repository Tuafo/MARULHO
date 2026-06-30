from __future__ import annotations

from pydantic import BaseModel, Field


class CheckpointSaveRequest(BaseModel):
    path: str | None = None


class CheckpointRestoreRequest(BaseModel):
    path: str = Field(..., min_length=1)


class CheckpointRecord(BaseModel):
    path: str
    name: str
    size_bytes: int
    modified_at: str | None = None


class CheckpointListResponse(BaseModel):
    checkpoints: list[CheckpointRecord]
