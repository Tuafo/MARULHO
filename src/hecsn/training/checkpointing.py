from __future__ import annotations

from collections import deque
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from hecsn.config.model_config import HECSNConfig
from hecsn.training.trainer import HECSNModelLite, HECSNTrainer


def _clone_optional_tensor(value: Any) -> torch.Tensor | None:
    return value.detach().clone().cpu() if isinstance(value, torch.Tensor) else None


def _surprise_snapshot(trainer: HECSNTrainer) -> dict[str, Any]:
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
        "acetylcholine": float(trainer.model.surprise.acetylcholine),
        "norepinephrine": float(trainer.model.surprise.norepinephrine),
    }


def _restore_surprise(trainer: HECSNTrainer, snapshot: dict[str, Any]) -> None:
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
    trainer.model.surprise.acetylcholine = float(snapshot.get("acetylcholine", 0.5))
    trainer.model.surprise.norepinephrine = float(snapshot.get("norepinephrine", 0.5))


def _model_snapshot(trainer: HECSNTrainer) -> dict[str, Any]:
    competitive = trainer.model.competitive
    return {
        "competitive": {
            "W_project": competitive.W_project.detach().clone().cpu(),
            "input_weights": competitive.input_weights.detach().clone().cpu(),
            "prototypes": competitive.prototypes.detach().clone().cpu(),
            "prototype_velocity": competitive.prototype_velocity.detach().clone().cpu(),
            "last_input_pattern": _clone_optional_tensor(competitive.last_input_pattern),
            "last_projected_input": _clone_optional_tensor(competitive.last_projected_input),
            "thresholds": competitive.thresholds.detach().clone().cpu(),
            "target_firing_rate": float(competitive.target_firing_rate),
            "win_rate_ema": competitive.win_rate_ema.detach().clone().cpu(),
            "steps_since_win": competitive.steps_since_win.detach().clone().cpu(),
            "update_count": int(competitive.update_count),
            "last_revived_indices": competitive.last_revived_indices.detach().clone().cpu(),
            "local_plasticity": None if competitive.local_plasticity is None else competitive.local_plasticity.state_dict(),
        },
        "W_assembly_project": trainer.model.W_assembly_project.detach().clone().cpu(),
        "surprise": _surprise_snapshot(trainer),
        "context_layer": None if trainer.model.context_layer is None else trainer.model.context_layer.state_dict(),
        "binding_layer": None if trainer.model.binding_layer is None else trainer.model.binding_layer.state_dict(),
        "memory_store": trainer.model.memory_store.snapshot(),
        "bootstrap": {
            "W": trainer.bootstrap.W.detach().clone().cpu(),
            "lr": float(trainer.bootstrap.lr),
            "prev_pattern": _clone_optional_tensor(trainer.bootstrap.prev_pattern),
        },
    }


def _restore_model(trainer: HECSNTrainer, snapshot: dict[str, Any]) -> None:
    competitive = trainer.model.competitive
    competitive_snapshot = snapshot["competitive"]
    competitive.W_project = competitive_snapshot["W_project"].to(trainer.model.device)
    competitive.input_weights = competitive_snapshot["input_weights"].to(trainer.model.device)
    competitive.prototypes = competitive_snapshot["prototypes"].to(trainer.model.device)
    competitive.prototype_velocity = competitive_snapshot["prototype_velocity"].to(trainer.model.device)
    last_input_pattern = competitive_snapshot.get("last_input_pattern")
    competitive.last_input_pattern = None if last_input_pattern is None else last_input_pattern.to(trainer.model.device)
    last_projected_input = competitive_snapshot.get("last_projected_input")
    competitive.last_projected_input = None if last_projected_input is None else last_projected_input.to(trainer.model.device)
    competitive.thresholds = competitive_snapshot["thresholds"].to(trainer.model.device)
    competitive.target_firing_rate = float(competitive_snapshot["target_firing_rate"])
    competitive.win_rate_ema = competitive_snapshot["win_rate_ema"].to(trainer.model.device)
    competitive.steps_since_win = competitive_snapshot["steps_since_win"].to(trainer.model.device)
    competitive.update_count = int(competitive_snapshot["update_count"])
    competitive.last_revived_indices = competitive_snapshot.get("last_revived_indices", torch.empty(0, dtype=torch.long)).to(trainer.model.device)
    if competitive.local_plasticity is not None and competitive_snapshot.get("local_plasticity") is not None:
        competitive.local_plasticity.load_state_dict(competitive_snapshot["local_plasticity"])

    trainer.model.W_assembly_project = snapshot["W_assembly_project"].to(trainer.model.device)
    _restore_surprise(trainer, snapshot["surprise"])

    if trainer.model.context_layer is not None and snapshot.get("context_layer") is not None:
        trainer.model.context_layer.load_state_dict(snapshot["context_layer"])

    if trainer.model.binding_layer is not None and snapshot.get("binding_layer") is not None:
        trainer.model.binding_layer.load_state_dict(snapshot["binding_layer"])

    trainer.model.memory_store.restore(snapshot["memory_store"])

    trainer.bootstrap.W = snapshot["bootstrap"]["W"].to(trainer.model.device)
    trainer.bootstrap.lr = float(snapshot["bootstrap"]["lr"])
    prev_pattern = snapshot["bootstrap"].get("prev_pattern")
    trainer.bootstrap.prev_pattern = None if prev_pattern is None else prev_pattern.to(trainer.model.device)

    all_ids = np.arange(trainer.config.n_columns, dtype=np.int64)
    trainer.model.hnsw_index.add(trainer.model.competitive.prototypes.detach(), all_ids)
    trainer.model.hnsw_index.rebuild()


def save_trainer_checkpoint(path: str | Path, trainer: HECSNTrainer, metadata: dict[str, Any] | None = None) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    config_snapshot = asdict(trainer.config)
    config_snapshot.pop("input_dim", None)
    payload = {
        "config": config_snapshot,
        "model": _model_snapshot(trainer),
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
    torch.save(payload, output_path)
    return output_path


def load_trainer_checkpoint(path: str | Path) -> tuple[HECSNTrainer, dict[str, Any]]:
    checkpoint_path = Path(path)
    payload = torch.load(checkpoint_path, map_location="cpu")
    cfg = HECSNConfig(**payload["config"])
    model = HECSNModelLite(cfg)
    trainer = HECSNTrainer(model, cfg)
    _restore_model(trainer, payload["model"])

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
