from __future__ import annotations

from collections import deque
from dataclasses import asdict
import os
from pathlib import Path
from uuid import uuid4
from typing import Any

import numpy as np
import torch

from marulho.config.model_config import MarulhoConfig
from marulho.training.model import MarulhoModel
from marulho.training.trainer import MarulhoTrainer


def _clone_optional_tensor(value: Any) -> torch.Tensor | None:
    return value.detach().clone().cpu() if isinstance(value, torch.Tensor) else None


def _surprise_snapshot(trainer: MarulhoTrainer) -> dict[str, Any]:
    layers: dict[str, dict[str, Any]] = {}
    for name, state in trainer.model.surprise.layers.items():
        layers[name] = {
            "errors": list(state["errors"]),
            "precision": float(state["precision"]),
        }
    return {
        "layers": layers,
        "predicted_error": float(trainer.model.surprise.predicted_error),
        "dopamine": float(trainer.model.surprise.dopamine),
        "serotonin": float(trainer.model.surprise.serotonin),
        "acetylcholine": float(trainer.model.surprise.acetylcholine),
        "norepinephrine": float(trainer.model.surprise.norepinephrine),
    }


def _restore_surprise(trainer: MarulhoTrainer, snapshot: dict[str, Any]) -> None:
    for name, state in snapshot.get("layers", {}).items():
        if name not in trainer.model.surprise.layers:
            trainer.model.surprise.layers[name] = {"errors": deque(maxlen=100), "precision": 1.0}
        buf = trainer.model.surprise.layers[name]["errors"]
        if isinstance(buf, deque):
            buf.clear()
            for value in state.get("errors", []):
                buf.append(float(value))
        trainer.model.surprise.layers[name]["precision"] = float(state.get("precision", 1.0))

    trainer.model.surprise.predicted_error = float(snapshot.get("predicted_error", 0.5))
    trainer.model.surprise.dopamine = float(snapshot.get("dopamine", 0.5))
    trainer.model.surprise.serotonin = float(snapshot.get("serotonin", 0.5))
    trainer.model.surprise.acetylcholine = float(snapshot.get("acetylcholine", 0.5))
    trainer.model.surprise.norepinephrine = float(snapshot.get("norepinephrine", 0.5))


def _model_snapshot(trainer: MarulhoTrainer) -> dict[str, Any]:
    competitive = trainer.model.competitive
    return {
        "competitive": competitive.state_dict(),
        "W_assembly_project": trainer.model.W_assembly_project.detach().clone().cpu(),
        "predictive": trainer.model.predictive.state_dict(),
        "surprise": _surprise_snapshot(trainer),
        "context_layer": None if trainer.model.context_layer is None else trainer.model.context_layer.state_dict(),
        "abstraction_layer": None if trainer.model.abstraction_layer is None else trainer.model.abstraction_layer.state_dict(),
        "binding_layer": None if trainer.model.binding_layer is None else trainer.model.binding_layer.state_dict(),
        "cross_modal": None if trainer.model.cross_modal is None else trainer.model.cross_modal.state_dict(),
        "memory_store": trainer.model.memory_store.snapshot(),
        "bootstrap": {
            "W": trainer.bootstrap.W.detach().clone().cpu(),
            "lr": float(trainer.bootstrap.lr),
            "prev_pattern": _clone_optional_tensor(trainer.bootstrap.prev_pattern),
        },
    }


def _restore_model(trainer: MarulhoTrainer, snapshot: dict[str, Any]) -> None:
    competitive = trainer.model.competitive
    competitive.load_state_dict(snapshot["competitive"])

    trainer.model.W_assembly_project = snapshot["W_assembly_project"].to(trainer.model.device)
    if snapshot.get("predictive") is not None:
        trainer.model.predictive.load_state_dict(snapshot["predictive"])
    _restore_surprise(trainer, snapshot["surprise"])

    if trainer.model.context_layer is not None and snapshot.get("context_layer") is not None:
        trainer.model.context_layer.load_state_dict(snapshot["context_layer"])

    if trainer.model.abstraction_layer is not None and snapshot.get("abstraction_layer") is not None:
        trainer.model.abstraction_layer.load_state_dict(snapshot["abstraction_layer"])

    if trainer.model.binding_layer is not None and snapshot.get("binding_layer") is not None:
        trainer.model.binding_layer.load_state_dict(snapshot["binding_layer"])

    if trainer.model.cross_modal is not None and snapshot.get("cross_modal") is not None:
        trainer.model.cross_modal.load_state_dict(snapshot["cross_modal"])

    trainer.model.memory_store.restore(snapshot["memory_store"])

    trainer.bootstrap.W = snapshot["bootstrap"]["W"].to(trainer.model.device)
    trainer.bootstrap.lr = float(snapshot["bootstrap"]["lr"])
    prev_pattern = snapshot["bootstrap"].get("prev_pattern")
    trainer.bootstrap.prev_pattern = None if prev_pattern is None else prev_pattern.to(trainer.model.device)

    all_ids = np.arange(trainer.config.n_columns, dtype=np.int64)
    trainer.model.hnsw_index.add(trainer.model.competitive.prototypes.detach(), all_ids)
    trainer.model.hnsw_index.rebuild()


def save_trainer_checkpoint(path: str | Path, trainer: MarulhoTrainer, metadata: dict[str, Any] | None = None) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    config_snapshot = asdict(trainer.config)
    config_snapshot.pop("input_dim", None)
    payload = {
        "config": config_snapshot,
        "model": _model_snapshot(trainer),
        "encoder": trainer.encoder.state_dict(),
        "trainer": {
            "token_count": int(trainer.token_count),
            "is_bootstrap": bool(trainer.is_bootstrap),
            "sleep_events": int(trainer.sleep_events),
            "micro_sleep_events": int(trainer.micro_sleep_events),
            "deep_sleep_events": int(trainer.deep_sleep_events),
            "last_micro_sleep_token": int(trainer.last_micro_sleep_token),
            "last_deep_sleep_token": int(trainer.last_deep_sleep_token),
            "current_window_min_drift": float(trainer.current_window_min_drift),
            "previous_window_min_drift": None if trainer.previous_window_min_drift is None else float(trainer.previous_window_min_drift),
            "recent_drifts": list(trainer.recent_drifts),
            "current_rolling_drift_floor": None if trainer.current_rolling_drift_floor is None else float(trainer.current_rolling_drift_floor),
            "previous_rolling_drift_floor": None if trainer.previous_rolling_drift_floor is None else float(trainer.previous_rolling_drift_floor),
            "last_floor_check_token": int(trainer.last_floor_check_token),
            "memory_warm_started": bool(trainer.memory_warm_started),
            "last_winner": None if trainer.last_winner is None else int(trainer.last_winner),
            "pending_emergency_deep_sleep": bool(trainer.pending_emergency_deep_sleep),
            "last_network_reset_token": int(trainer.last_network_reset_token),
            "developmental_stage": int(trainer.developmental_stage),
            "stage2_bootstrap_budget": int(trainer._stage2_bootstrap_budget),
            "stage2_bootstrap_used_visual": int(trainer._stage2_bootstrap_used_visual),
            "stage2_bootstrap_used_audio": int(trainer._stage2_bootstrap_used_audio),
            "column_anchors": {
                int(key): {
                    "prototype": value["prototype"].detach().clone().cpu(),
                    "input_weights": value["input_weights"].detach().clone().cpu(),
                    "strength": float(value["strength"]),
                }
                for key, value in trainer.column_anchors.items()
                if isinstance(value.get("prototype"), torch.Tensor) and isinstance(value.get("input_weights"), torch.Tensor)
            },
        },
        "metadata": dict(metadata or {}),
    }
    temporary_path = output_path.with_name(f".{output_path.name}.{uuid4().hex}.tmp")
    try:
        with temporary_path.open("wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, output_path)
        _sync_parent_directory(output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return output_path


def _sync_parent_directory(path: Path) -> None:
    try:
        descriptor = os.open(str(path.parent), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        return
    finally:
        os.close(descriptor)


def _checkpoint_load_device() -> torch.device:
    env = os.environ.get("MARULHO_DEVICE")
    if env:
        return torch.device(env)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_trainer_checkpoint(path: str | Path) -> tuple[MarulhoTrainer, dict[str, Any]]:
    checkpoint_path = Path(path)
    payload = torch.load(checkpoint_path, map_location=_checkpoint_load_device())
    cfg = MarulhoConfig(**payload["config"])
    model = MarulhoModel(cfg)
    trainer = MarulhoTrainer(model, cfg)
    _restore_model(trainer, payload["model"])
    encoder_snapshot = payload.get("encoder")
    if isinstance(encoder_snapshot, dict):
        trainer.encoder.load_state_dict(encoder_snapshot)

    trainer_snapshot = payload.get("trainer", {})
    trainer.token_count = int(trainer_snapshot.get("token_count", 0))
    trainer.is_bootstrap = bool(trainer_snapshot.get("is_bootstrap", True))
    trainer.sleep_events = int(trainer_snapshot.get("sleep_events", 0))
    trainer.micro_sleep_events = int(trainer_snapshot.get("micro_sleep_events", 0))
    trainer.deep_sleep_events = int(trainer_snapshot.get("deep_sleep_events", 0))
    trainer.last_micro_sleep_token = int(trainer_snapshot.get("last_micro_sleep_token", -10**9))
    trainer.last_deep_sleep_token = int(trainer_snapshot.get("last_deep_sleep_token", -10**9))
    trainer.current_window_min_drift = float(trainer_snapshot.get("current_window_min_drift", float("inf")))
    previous_window_min_drift = trainer_snapshot.get("previous_window_min_drift")
    trainer.previous_window_min_drift = None if previous_window_min_drift is None else float(previous_window_min_drift)
    trainer.recent_drifts = deque(
        [float(value) for value in trainer_snapshot.get("recent_drifts", [])],
        maxlen=trainer.config.drift_floor_history_tokens,
    )
    current_rolling = trainer_snapshot.get("current_rolling_drift_floor")
    previous_rolling = trainer_snapshot.get("previous_rolling_drift_floor")
    trainer.current_rolling_drift_floor = None if current_rolling is None else float(current_rolling)
    trainer.previous_rolling_drift_floor = None if previous_rolling is None else float(previous_rolling)
    trainer.last_floor_check_token = int(trainer_snapshot.get("last_floor_check_token", -10**9))
    trainer.memory_warm_started = bool(trainer_snapshot.get("memory_warm_started", True))
    last_winner = trainer_snapshot.get("last_winner")
    trainer.last_winner = None if last_winner is None else int(last_winner)
    trainer.pending_emergency_deep_sleep = bool(trainer_snapshot.get("pending_emergency_deep_sleep", False))
    trainer.last_network_reset_token = int(trainer_snapshot.get("last_network_reset_token", -10**9))
    trainer.developmental_stage = int(trainer_snapshot.get("developmental_stage", 1))
    trainer._stage2_bootstrap_budget = int(trainer_snapshot.get("stage2_bootstrap_budget", 50))
    trainer._stage2_bootstrap_used_visual = int(trainer_snapshot.get("stage2_bootstrap_used_visual", 0))
    trainer._stage2_bootstrap_used_audio = int(trainer_snapshot.get("stage2_bootstrap_used_audio", 0))
    trainer.column_anchors = {
        int(key): {
            "prototype": value["prototype"].detach().clone().to(trainer.model.device),
            "input_weights": value["input_weights"].detach().clone().to(trainer.model.device),
            "strength": float(value["strength"]),
        }
        for key, value in dict(trainer_snapshot.get("column_anchors", {})).items()
        if isinstance(value.get("prototype"), torch.Tensor) and isinstance(value.get("input_weights"), torch.Tensor)
    }
    return trainer, dict(payload.get("metadata", {}))
