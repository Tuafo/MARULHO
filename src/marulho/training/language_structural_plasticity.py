from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence
from uuid import uuid4

import torch

from marulho.training.language_model import (
    LanguageBatch,
    LanguageModelConfig,
    MarulhoLanguageModel,
    evaluate_language_model,
)


@dataclass(frozen=True)
class LanguageStructuralPlasticityConfig:
    max_added_experts: int = 2
    max_pruned_experts: int = 1
    route_saturation_threshold: float = 0.75
    prune_utility_threshold: float = 0.05
    max_eval_loss_delta: float = 0.05
    min_expert_count: int = 1
    require_operator_approval: bool = True


def _json_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _model_state_hash(model: MarulhoLanguageModel) -> str:
    digest = hashlib.sha256()
    for key, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(str(key).encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _save_baseline_checkpoint(
    path: str | Path,
    model: MarulhoLanguageModel,
    *,
    proposal: Mapping[str, Any],
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "artifact_kind": "marulho_language_structural_plasticity_baseline_checkpoint",
        "surface": "marulho_language_structural_plasticity_baseline_checkpoint.v1",
        "active_language_path": model.config.active_language_path,
        "config": asdict(model.config),
        "model_state": {
            key: value.detach().cpu()
            for key, value in model.state_dict().items()
        },
        "proposal_hash": proposal.get("proposal_hash"),
        "owned_by_marulho": True,
        "external_llm_used": False,
    }
    temporary_path = output_path.with_name(f".{output_path.name}.{uuid4().hex}.tmp")
    try:
        with temporary_path.open("wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return output_path


def _load_checkpoint_hash(path: str | Path) -> str:
    payload = torch.load(Path(path), map_location="cpu")
    model_state = payload["model_state"]
    digest = hashlib.sha256()
    for key, tensor in sorted(model_state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(str(key).encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _clone_expanded_model(
    model: MarulhoLanguageModel,
    *,
    target_expert_count: int,
) -> MarulhoLanguageModel:
    old_expert_count = max(0, int(model.config.expert_count))
    new_config = replace(
        model.config,
        expert_count=max(int(target_expert_count), old_expert_count),
        active_expert_count=max(1, int(model.config.active_expert_count)),
        route_candidate_count=max(
            int(model.config.route_candidate_count),
            max(1, int(model.config.active_expert_count)),
        ),
    )
    expanded = MarulhoLanguageModel(new_config).to(model.device)
    expanded_state = expanded.state_dict()
    source_state = model.state_dict()
    for key, source_value in source_state.items():
        if key in expanded_state and expanded_state[key].shape == source_value.shape:
            expanded_state[key] = source_value.detach().clone().to(expanded_state[key].device)
    expanded.load_state_dict(expanded_state)
    with torch.no_grad():
        if expanded.routed_experts.enabled and old_expert_count < expanded.routed_experts.expert_count:
            expanded.routed_experts.route_keys[old_expert_count:].zero_()
            expanded.routed_experts.route_bias[old_expert_count:].fill_(-10.0)
            for expert in expanded.routed_experts.experts[old_expert_count:]:
                for parameter in expert.parameters():
                    parameter.zero_()
    expanded.train(model.training)
    return expanded


def _clone_pruned_model(
    model: MarulhoLanguageModel,
    *,
    retained_expert_ids: Sequence[int],
) -> MarulhoLanguageModel:
    retained_ids = tuple(int(value) for value in retained_expert_ids)
    if not retained_ids:
        raise ValueError("retained_expert_ids must not be empty")
    old_expert_count = max(0, int(model.config.expert_count))
    if any(value < 0 or value >= old_expert_count for value in retained_ids):
        raise ValueError("retained_expert_ids contains an out-of-range expert id")
    if len(set(retained_ids)) != len(retained_ids):
        raise ValueError("retained_expert_ids must be unique")
    target_expert_count = len(retained_ids)
    new_config = replace(
        model.config,
        expert_count=target_expert_count,
        active_expert_count=min(
            max(1, int(model.config.active_expert_count)),
            target_expert_count,
        ),
        route_candidate_count=min(
            max(1, int(model.config.route_candidate_count)),
            target_expert_count,
        ),
    )
    pruned = MarulhoLanguageModel(new_config).to(model.device)
    pruned_state = pruned.state_dict()
    source_state = model.state_dict()
    for key, source_value in source_state.items():
        if key in pruned_state and pruned_state[key].shape == source_value.shape:
            pruned_state[key] = source_value.detach().clone().to(pruned_state[key].device)
    pruned.load_state_dict(pruned_state)
    retained_tensor = torch.tensor(retained_ids, dtype=torch.long, device=model.device)
    with torch.no_grad():
        if pruned.routed_experts.enabled and model.routed_experts.enabled:
            pruned.routed_experts.route_keys.copy_(
                model.routed_experts.route_keys.index_select(0, retained_tensor).to(
                    device=pruned.routed_experts.route_keys.device,
                    dtype=pruned.routed_experts.route_keys.dtype,
                )
            )
            pruned.routed_experts.route_bias.copy_(
                model.routed_experts.route_bias.index_select(0, retained_tensor).to(
                    device=pruned.routed_experts.route_bias.device,
                    dtype=pruned.routed_experts.route_bias.dtype,
                )
            )
            for new_expert_id, old_expert_id in enumerate(retained_ids):
                for target_parameter, source_parameter in zip(
                    pruned.routed_experts.experts[new_expert_id].parameters(),
                    model.routed_experts.experts[old_expert_id].parameters(),
                    strict=True,
                ):
                    target_parameter.copy_(
                        source_parameter.detach().to(
                            device=target_parameter.device,
                            dtype=target_parameter.dtype,
                        )
                    )
    pruned.train(model.training)
    return pruned


def _normalise_expert_ids(value: Any) -> tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, Mapping):
        iterable = value.keys()
    elif isinstance(value, (str, bytes)):
        return ()
    else:
        try:
            iterable = list(value)
        except TypeError:
            return ()
    result: list[int] = []
    for item in iterable:
        try:
            result.append(int(item))
        except (TypeError, ValueError):
            continue
    return tuple(result)


def _low_utility_ids_from_evidence(
    value: Any,
    *,
    threshold: float,
) -> tuple[int, ...]:
    result: list[int] = []
    if isinstance(value, Mapping):
        for raw_id, raw_utility in value.items():
            try:
                if float(raw_utility) <= float(threshold):
                    result.append(int(raw_id))
            except (TypeError, ValueError):
                continue
    elif not isinstance(value, (str, bytes)):
        try:
            for expert_id, raw_utility in enumerate(value):
                if float(raw_utility) <= float(threshold):
                    result.append(int(expert_id))
        except TypeError:
            pass
    return tuple(result)


def build_language_structural_plasticity_proposal(
    model: MarulhoLanguageModel,
    *,
    routing_evidence: Mapping[str, Any],
    learning_evidence: Mapping[str, Any] | None = None,
    config: LanguageStructuralPlasticityConfig | None = None,
) -> dict[str, Any]:
    cfg = config or LanguageStructuralPlasticityConfig()
    route = dict(routing_evidence or {})
    learning = dict(learning_evidence or {})
    total_columns = int(route.get("total_columns") or model.config.expert_count or 0)
    active_columns = int(route.get("active_columns") or 0)
    saturation = active_columns / max(1, total_columns)
    expert_count = max(total_columns, int(model.config.expert_count), int(cfg.min_expert_count))
    added_experts = max(1, min(int(cfg.max_added_experts), max(1, expert_count // 2)))
    target_expert_count = expert_count + added_experts
    route_pressure = saturation >= float(cfg.route_saturation_threshold)
    loss_pressure = float(learning.get("new_domain_loss_delta", 0.0) or 0.0) <= 0.0
    ready = model.config.expert_count > 0 and (route_pressure or loss_pressure)
    status = "ready_for_operator_review" if ready else "collect_more_growth_pressure"
    proposal_body = {
        "proposal_kind": "expert_spawn",
        "source_expert_count": int(model.config.expert_count),
        "target_expert_count": int(target_expert_count),
        "added_expert_count": int(added_experts),
        "route_saturation": float(saturation),
        "route_pressure": bool(route_pressure),
        "loss_pressure": bool(loss_pressure),
        "routing_evidence_hash": _json_hash(route),
        "learning_evidence_hash": _json_hash(learning) if learning else None,
        "config": asdict(cfg),
    }
    proposal_hash = _json_hash(proposal_body)
    return {
        "artifact_kind": "marulho_language_structural_plasticity_proposal",
        "surface": "marulho_language_structural_plasticity_proposal.v1",
        "owned_by_marulho": True,
        "external_llm_used": False,
        "loads_external_checkpoint": False,
        "mutates_runtime_state": False,
        "proposal_hash": proposal_hash,
        "status": status,
        "proposal": proposal_body,
        "routing_evidence": route,
        "learning_evidence": learning,
        "promotion_gate": {
            "status": status,
            "eligible_for_checkpointed_transaction": bool(ready),
            "requires_operator_approval": bool(cfg.require_operator_approval),
            "checkpoint_required": True,
            "rollback_required": True,
            "route_pressure": bool(route_pressure),
            "loss_pressure": bool(loss_pressure),
        },
    }


def build_language_structural_prune_proposal(
    model: MarulhoLanguageModel,
    *,
    routing_evidence: Mapping[str, Any],
    learning_evidence: Mapping[str, Any] | None = None,
    config: LanguageStructuralPlasticityConfig | None = None,
) -> dict[str, Any]:
    cfg = config or LanguageStructuralPlasticityConfig()
    route = dict(routing_evidence or {})
    learning = dict(learning_evidence or {})
    expert_count = max(0, int(model.config.expert_count))
    active_requirement = max(1, int(model.config.active_expert_count), int(cfg.min_expert_count))
    active_ids = set(_normalise_expert_ids(route.get("active_expert_ids")))
    explicit_low_utility_ids = set(_normalise_expert_ids(route.get("low_utility_expert_ids")))
    explicit_inactive_ids = set(_normalise_expert_ids(route.get("inactive_expert_ids")))
    utility_ids = set(
        _low_utility_ids_from_evidence(
            route.get("expert_utilities") or route.get("utility_by_expert"),
            threshold=float(cfg.prune_utility_threshold),
        )
    )
    valid_pressure_ids = {
        value
        for value in explicit_low_utility_ids | explicit_inactive_ids | utility_ids
        if 0 <= value < expert_count and value not in active_ids
    }
    prune_budget = max(0, expert_count - active_requirement)
    pruned_ids = tuple(
        sorted(valid_pressure_ids)[: min(int(cfg.max_pruned_experts), prune_budget)]
    )
    retained_ids = tuple(value for value in range(expert_count) if value not in set(pruned_ids))
    ready = expert_count > active_requirement and bool(pruned_ids)
    status = "ready_for_operator_review" if ready else "collect_more_prune_pressure"
    proposal_body = {
        "proposal_kind": "expert_prune",
        "source_expert_count": int(expert_count),
        "target_expert_count": int(len(retained_ids)),
        "pruned_expert_ids": list(pruned_ids),
        "retained_expert_ids": list(retained_ids),
        "active_expert_count_floor": int(active_requirement),
        "prune_pressure": bool(pruned_ids),
        "explicit_inactive_expert_ids": sorted(explicit_inactive_ids),
        "explicit_low_utility_expert_ids": sorted(explicit_low_utility_ids),
        "utility_threshold": float(cfg.prune_utility_threshold),
        "routing_evidence_hash": _json_hash(route),
        "learning_evidence_hash": _json_hash(learning) if learning else None,
        "config": asdict(cfg),
    }
    proposal_hash = _json_hash(proposal_body)
    return {
        "artifact_kind": "marulho_language_structural_plasticity_proposal",
        "surface": "marulho_language_structural_plasticity_proposal.v1",
        "owned_by_marulho": True,
        "external_llm_used": False,
        "loads_external_checkpoint": False,
        "mutates_runtime_state": False,
        "proposal_hash": proposal_hash,
        "status": status,
        "proposal": proposal_body,
        "routing_evidence": route,
        "learning_evidence": learning,
        "promotion_gate": {
            "status": status,
            "eligible_for_checkpointed_transaction": bool(ready),
            "requires_operator_approval": bool(cfg.require_operator_approval),
            "checkpoint_required": True,
            "rollback_required": True,
            "prune_pressure": bool(pruned_ids),
            "min_expert_count_preserved": len(retained_ids) >= int(cfg.min_expert_count),
            "active_expert_count_preserved": len(retained_ids)
            >= int(model.config.active_expert_count),
        },
    }


def apply_language_structural_plasticity_transaction(
    model: MarulhoLanguageModel,
    proposal: Mapping[str, Any],
    *,
    eval_batches: Sequence[LanguageBatch],
    checkpoint_path: str | Path,
    operator_approved: bool,
    config: LanguageStructuralPlasticityConfig | None = None,
) -> tuple[MarulhoLanguageModel, dict[str, Any]]:
    if not eval_batches:
        raise ValueError("eval_batches must not be empty")
    cfg = config or LanguageStructuralPlasticityConfig()
    gate = proposal.get("promotion_gate") if isinstance(proposal.get("promotion_gate"), Mapping) else {}
    body = proposal.get("proposal") if isinstance(proposal.get("proposal"), Mapping) else {}
    approved = bool(operator_approved) or not bool(cfg.require_operator_approval)
    if not bool(gate.get("eligible_for_checkpointed_transaction")) or not approved:
        return model, {
            "artifact_kind": "marulho_language_structural_plasticity_transaction",
            "surface": "marulho_language_structural_plasticity_transaction.v1",
            "applied": False,
            "status": "rejected_operator_or_gate_not_ready",
            "owned_by_marulho": True,
            "external_llm_used": False,
            "mutates_runtime_state": False,
            "operator_approved": bool(operator_approved),
            "proposal_hash": proposal.get("proposal_hash"),
        }

    baseline_hash = _model_state_hash(model)
    saved_checkpoint = _save_baseline_checkpoint(
        checkpoint_path,
        model,
        proposal=proposal,
    )
    checkpoint_restore_hash = _load_checkpoint_hash(saved_checkpoint)
    baseline_eval = evaluate_language_model(model, eval_batches)
    proposal_kind = str(body.get("proposal_kind") or "")
    if proposal_kind == "expert_prune":
        candidate = _clone_pruned_model(
            model,
            retained_expert_ids=[
                int(value) for value in list(body.get("retained_expert_ids") or [])
            ],
        )
    else:
        target_expert_count = int(body.get("target_expert_count") or model.config.expert_count)
        candidate = _clone_expanded_model(model, target_expert_count=target_expert_count)
    candidate_hash = _model_state_hash(candidate)
    candidate_eval = evaluate_language_model(candidate, eval_batches)
    loss_delta = float(candidate_eval["heldout_loss"]) - float(baseline_eval["heldout_loss"])
    accepted = loss_delta <= float(cfg.max_eval_loss_delta)
    final_model = candidate if accepted else model
    final_hash = _model_state_hash(final_model)
    rollback_verified = (
        accepted
        or _model_state_hash(model) == baseline_hash == checkpoint_restore_hash
    )
    status = "applied_structural_mutation" if accepted else "rolled_back_candidate_regression"
    return final_model, {
        "artifact_kind": "marulho_language_structural_plasticity_transaction",
        "surface": "marulho_language_structural_plasticity_transaction.v1",
        "owned_by_marulho": True,
        "external_llm_used": False,
        "loads_external_checkpoint": False,
        "active_language_path": model.config.active_language_path,
        "applied": bool(accepted),
        "mutates_runtime_state": bool(accepted),
        "operator_approved": bool(operator_approved),
        "status": status,
        "proposal_hash": proposal.get("proposal_hash"),
        "checkpoint": {
            "path": str(saved_checkpoint),
            "baseline_state_hash": baseline_hash,
            "checkpoint_restore_hash": checkpoint_restore_hash,
            "checkpoint_restore_verified": checkpoint_restore_hash == baseline_hash,
        },
        "mutation": {
            "proposal_kind": body.get("proposal_kind"),
            "source_expert_count": int(model.config.expert_count),
            "target_expert_count": int(final_model.config.expert_count),
            "candidate_target_expert_count": int(candidate.config.expert_count),
            "added_expert_count": int(
                max(0, candidate.config.expert_count - model.config.expert_count)
            ),
            "pruned_expert_count": int(
                max(0, model.config.expert_count - candidate.config.expert_count)
            ),
            "pruned_expert_ids": [
                int(value) for value in list(body.get("pruned_expert_ids") or [])
            ],
            "retained_expert_ids": [
                int(value) for value in list(body.get("retained_expert_ids") or [])
            ],
        },
        "evaluation": {
            "baseline": baseline_eval,
            "candidate": candidate_eval,
            "heldout_loss_delta": float(loss_delta),
            "max_eval_loss_delta": float(cfg.max_eval_loss_delta),
        },
        "rollback_evidence": {
            "candidate_state_hash": candidate_hash,
            "final_state_hash": final_hash,
            "rollback_required": not bool(accepted),
            "rollback_verified": bool(rollback_verified),
        },
        "promotion_gate": {
            "status": status,
            "eligible_for_reviewed_growth_promotion": bool(
                accepted and proposal_kind == "expert_spawn"
            ),
            "eligible_for_reviewed_prune_promotion": bool(
                accepted and proposal_kind == "expert_prune"
            ),
            "eligible_for_reviewed_structural_promotion": bool(accepted),
            "checkpoint_backed": True,
            "operator_approved": bool(operator_approved),
            "heldout_non_regression": bool(accepted),
            "rollback_verified": bool(rollback_verified),
        },
    }
