"""Atomic, resume-exact continuation snapshots for MARULHO language training."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping

import torch

from marulho.training.language_model import (
    CHECKPOINT_SURFACE,
    MarulhoLanguageModel,
    language_model_state_sha256,
    load_language_model_checkpoint,
    save_language_model_checkpoint,
)


TRAINING_CONTINUATION_SURFACE = "marulho_language_training_continuation.v1"
OptimizerBuilder = Callable[[MarulhoLanguageModel], Any]


def tree_to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: tree_to_cpu(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(tree_to_cpu(item) for item in value)
    if isinstance(value, list):
        return [tree_to_cpu(item) for item in value]
    return value


def _update_tree_digest(digest: Any, value: Any) -> None:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        digest.update(b"tensor\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        return
    if isinstance(value, Mapping):
        digest.update(b"mapping\0")
        for key in sorted(value, key=lambda item: (type(item).__name__, repr(item))):
            _update_tree_digest(digest, key)
            _update_tree_digest(digest, value[key])
        return
    if isinstance(value, tuple):
        digest.update(b"tuple\0")
        for item in value:
            _update_tree_digest(digest, item)
        return
    if isinstance(value, list):
        digest.update(b"list\0")
        for item in value:
            _update_tree_digest(digest, item)
        return
    digest.update(type(value).__name__.encode("ascii"))
    digest.update(b"\0")
    digest.update(
        json.dumps(value, sort_keys=True, allow_nan=False).encode("utf-8")
        if value is not None
        else b"null"
    )
    digest.update(b"\0")


def tree_sha256(value: Any) -> str:
    digest = hashlib.sha256()
    _update_tree_digest(digest, value)
    return digest.hexdigest()


def _model_state_mapping_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in state.items():
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def capture_rng_state() -> dict[str, Any]:
    return {
        "cpu": torch.get_rng_state().cpu().clone(),
        "cuda_all": [state.cpu().clone() for state in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else [],
    }


def restore_rng_state(snapshot: Mapping[str, Any]) -> None:
    cpu = snapshot.get("cpu")
    if not isinstance(cpu, torch.Tensor):
        raise ValueError("Language training snapshot lacks CPU RNG state")
    torch.set_rng_state(cpu.cpu())
    cuda_all = snapshot.get("cuda_all", [])
    if cuda_all:
        if not torch.cuda.is_available():
            raise RuntimeError("Language training snapshot requires CUDA RNG state")
        torch.cuda.set_rng_state_all([state.cpu() for state in cuda_all])


def save_language_training_snapshot(
    path: str | Path,
    model: MarulhoLanguageModel,
    tokenizer: Any,
    optimizer: torch.optim.Optimizer,
    *,
    completed_steps: int,
    schedule_sha256: str,
    next_schedule_offset: int,
    training_state: Mapping[str, Any],
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if completed_steps < 0 or next_schedule_offset < 0:
        raise ValueError("Language training continuation counters must be nonnegative")
    floating_dtypes = {
        parameter.dtype for parameter in model.parameters() if parameter.is_floating_point()
    }
    if floating_dtypes != {torch.bfloat16}:
        raise ValueError(f"Training snapshot requires BF16 model state: {floating_dtypes}")
    optimizer_state = tree_to_cpu(optimizer.state_dict())
    rng_state = capture_rng_state()
    continuation = {
        "surface": TRAINING_CONTINUATION_SURFACE,
        "completed_steps": int(completed_steps),
        "schedule_sha256": str(schedule_sha256),
        "next_schedule_offset": int(next_schedule_offset),
        "model_dtype": "torch.bfloat16",
        "model_state_sha256": language_model_state_sha256(model),
        "optimizer_state": optimizer_state,
        "optimizer_state_sha256": tree_sha256(optimizer_state),
        "rng_state": rng_state,
        "rng_state_sha256": tree_sha256(rng_state),
        "training_state": tree_to_cpu(dict(training_state)),
    }
    snapshot_metadata = dict(metadata or {})
    snapshot_metadata.update(
        {
            "decision": "save_language_training_continuation_snapshot",
            "external_llm_used": False,
            "optimizer_state_saved": True,
            "training_continuation": continuation,
        }
    )
    output = save_language_model_checkpoint(path, model, tokenizer, snapshot_metadata)
    raw = torch.load(output, map_location="cpu", weights_only=False)
    raw_continuation = raw.get("metadata", {}).get("training_continuation", {})
    verification = {
        "checkpoint_surface_exact": raw.get("surface") == CHECKPOINT_SURFACE,
        "continuation_surface_exact": raw_continuation.get("surface")
        == TRAINING_CONTINUATION_SURFACE,
        "completed_steps_exact": raw_continuation.get("completed_steps")
        == int(completed_steps),
        "next_schedule_offset_exact": raw_continuation.get("next_schedule_offset")
        == int(next_schedule_offset),
        "schedule_exact": raw_continuation.get("schedule_sha256")
        == str(schedule_sha256),
        "raw_model_state_exact": _model_state_mapping_sha256(raw["model_state"])
        == continuation["model_state_sha256"],
        "optimizer_state_exact": tree_sha256(raw_continuation["optimizer_state"])
        == continuation["optimizer_state_sha256"],
        "rng_state_exact": tree_sha256(raw_continuation["rng_state"])
        == continuation["rng_state_sha256"],
        "external_llm_absent": raw.get("external_llm_used") is False
        and raw.get("metadata", {}).get("external_llm_used") is False,
    }
    verification["passed"] = all(verification.values())
    if not verification["passed"]:
        Path(output).unlink(missing_ok=True)
        raise RuntimeError(f"Language training snapshot verification failed: {verification}")
    return {
        "path": str(output),
        "size_bytes": Path(output).stat().st_size,
        "completed_steps": int(completed_steps),
        "next_schedule_offset": int(next_schedule_offset),
        "model_state_sha256": continuation["model_state_sha256"],
        "optimizer_state_sha256": continuation["optimizer_state_sha256"],
        "rng_state_sha256": continuation["rng_state_sha256"],
        "verification": verification,
    }


def load_language_training_snapshot(
    path: str | Path,
    *,
    optimizer_builder: OptimizerBuilder,
    device: torch.device,
    expected_schedule_sha256: str,
    restore_rng: bool = True,
) -> tuple[
    MarulhoLanguageModel,
    Any,
    torch.optim.Optimizer,
    dict[str, Any],
    dict[str, Any],
]:
    model, tokenizer, metadata = load_language_model_checkpoint(path, map_location="cpu")
    continuation = dict(metadata.get("training_continuation") or {})
    if continuation.get("surface") != TRAINING_CONTINUATION_SURFACE:
        raise ValueError("Checkpoint is not a MARULHO language training continuation")
    if continuation.get("schedule_sha256") != expected_schedule_sha256:
        raise ValueError("Language training continuation schedule changed")
    if continuation.get("model_dtype") != "torch.bfloat16":
        raise ValueError("Language training continuation is not BF16")
    model = model.to(device=device, dtype=torch.bfloat16)
    built = optimizer_builder(model)
    optimizer = built[0] if isinstance(built, tuple) else built
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("optimizer_builder did not return a torch optimizer")
    optimizer.load_state_dict(continuation["optimizer_state"])
    model_hash = language_model_state_sha256(model)
    optimizer_hash = tree_sha256(optimizer.state_dict())
    rng_hash = tree_sha256(continuation["rng_state"])
    audit = {
        "model_state_exact": model_hash == continuation.get("model_state_sha256"),
        "optimizer_state_exact": optimizer_hash
        == continuation.get("optimizer_state_sha256"),
        "rng_state_exact": rng_hash == continuation.get("rng_state_sha256"),
        "completed_steps_nonnegative": int(continuation.get("completed_steps", -1))
        >= 0,
        "next_schedule_offset_nonnegative": int(
            continuation.get("next_schedule_offset", -1)
        )
        >= 0,
        "external_llm_absent": metadata.get("external_llm_used") is False,
    }
    audit["passed"] = all(audit.values())
    if not audit["passed"]:
        raise RuntimeError(f"Language training continuation load failed: {audit}")
    if restore_rng:
        restore_rng_state(continuation["rng_state"])
    return model, tokenizer, optimizer, continuation, audit
