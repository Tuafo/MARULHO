from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from marulho.training.checkpointing import load_trainer_checkpoint, save_trainer_checkpoint
from marulho.training.trainer import MarulhoTrainer


def load_brain_trainer_checkpoint(path: str | Path) -> tuple[MarulhoTrainer, dict[str, Any]]:
    return load_trainer_checkpoint(path)


def save_brain_trainer_checkpoint(
    path: str | Path,
    trainer: MarulhoTrainer,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    return save_trainer_checkpoint(path, trainer, metadata=metadata)
