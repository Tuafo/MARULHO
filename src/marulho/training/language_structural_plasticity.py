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
    max_split_experts: int = 1
    max_synapse_bundle_hidden_growth: int = 16
    max_memory_slot_growth: int = 4
    max_memory_slot_count: int = 0
    max_memory_slot_candidate_count: int = 4
    max_pruned_experts: int = 1
    max_merged_expert_pairs: int = 1
    max_deep_sleep_experts: int = 1
    max_retired_experts: int = 1
    max_route_candidate_growth: int = 2
    max_route_candidate_count: int = 0
    route_saturation_threshold: float = 0.75
    split_load_threshold: float = 0.80
    prune_utility_threshold: float = 0.05
    merge_similarity_threshold: float = 0.95
    deep_sleep_utility_threshold: float = 0.10
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
    digest.update(
        json.dumps(
            asdict(model.config),
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    )
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
    digest.update(
        json.dumps(
            payload.get("config") or {},
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    )
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
            expanded.routed_experts.sleeping_expert_mask[:old_expert_count].copy_(
                model.routed_experts.sleeping_expert_mask[:old_expert_count].to(
                    device=expanded.routed_experts.sleeping_expert_mask.device
                )
            )
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
            pruned.routed_experts.sleeping_expert_mask.copy_(
                model.routed_experts.sleeping_expert_mask.index_select(
                    0,
                    retained_tensor,
                ).to(device=pruned.routed_experts.sleeping_expert_mask.device)
            )
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


def _clone_merged_model(
    model: MarulhoLanguageModel,
    *,
    merged_expert_groups: Sequence[Sequence[int]],
) -> MarulhoLanguageModel:
    groups = tuple(tuple(int(value) for value in group) for group in merged_expert_groups)
    if not groups:
        raise ValueError("merged_expert_groups must not be empty")
    old_expert_count = max(0, int(model.config.expert_count))
    seen: set[int] = set()
    for group in groups:
        if len(group) < 2:
            raise ValueError("each merged expert group must contain at least two experts")
        if any(value < 0 or value >= old_expert_count for value in group):
            raise ValueError("merged_expert_groups contains an out-of-range expert id")
        if seen.intersection(group):
            raise ValueError("merged expert groups must be disjoint")
        seen.update(group)
    merge_by_primary = {group[0]: group for group in groups}
    removed_ids = {value for group in groups for value in group[1:]}
    retained_ids = tuple(value for value in range(old_expert_count) if value not in removed_ids)
    target_expert_count = len(retained_ids)
    if target_expert_count <= 0:
        raise ValueError("expert merge must retain at least one expert")
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
    merged = MarulhoLanguageModel(new_config).to(model.device)
    merged_state = merged.state_dict()
    source_state = model.state_dict()
    for key, source_value in source_state.items():
        if key in merged_state and merged_state[key].shape == source_value.shape:
            merged_state[key] = source_value.detach().clone().to(merged_state[key].device)
    merged.load_state_dict(merged_state)
    with torch.no_grad():
        if merged.routed_experts.enabled and model.routed_experts.enabled:
            for new_expert_id, old_expert_id in enumerate(retained_ids):
                group = merge_by_primary.get(old_expert_id, (old_expert_id,))
                group_tensor = torch.tensor(group, dtype=torch.long, device=model.device)
                merged.routed_experts.sleeping_expert_mask[new_expert_id].copy_(
                    model.routed_experts.sleeping_expert_mask.index_select(
                        0,
                        group_tensor.to(model.routed_experts.sleeping_expert_mask.device),
                    ).all().to(
                        device=merged.routed_experts.sleeping_expert_mask.device
                    )
                )
                merged.routed_experts.route_keys[new_expert_id].copy_(
                    model.routed_experts.route_keys.index_select(0, group_tensor)
                    .mean(dim=0)
                    .to(
                        device=merged.routed_experts.route_keys.device,
                        dtype=merged.routed_experts.route_keys.dtype,
                    )
                )
                merged.routed_experts.route_bias[new_expert_id].copy_(
                    model.routed_experts.route_bias.index_select(0, group_tensor)
                    .mean(dim=0)
                    .to(
                        device=merged.routed_experts.route_bias.device,
                        dtype=merged.routed_experts.route_bias.dtype,
                    )
                )
                source_expert_parameters = [
                    list(model.routed_experts.experts[source_id].parameters())
                    for source_id in group
                ]
                for parameter_index, target_parameter in enumerate(
                    merged.routed_experts.experts[new_expert_id].parameters()
                ):
                    averaged = torch.stack(
                        [
                            source_parameters[parameter_index].detach().to(
                                device=target_parameter.device,
                                dtype=target_parameter.dtype,
                            )
                            for source_parameters in source_expert_parameters
                        ],
                        dim=0,
                    ).mean(dim=0)
                    target_parameter.copy_(averaged)
    merged.train(model.training)
    return merged


def _clone_deep_sleep_model(
    model: MarulhoLanguageModel,
    *,
    deep_sleep_expert_ids: Sequence[int],
) -> MarulhoLanguageModel:
    sleep_ids = tuple(int(value) for value in deep_sleep_expert_ids)
    if not sleep_ids:
        raise ValueError("deep_sleep_expert_ids must not be empty")
    old_expert_count = max(0, int(model.config.expert_count))
    if any(value < 0 or value >= old_expert_count for value in sleep_ids):
        raise ValueError("deep_sleep_expert_ids contains an out-of-range expert id")
    if len(set(sleep_ids)) != len(sleep_ids):
        raise ValueError("deep_sleep_expert_ids must be unique")
    candidate = MarulhoLanguageModel(model.config).to(model.device)
    candidate_state = candidate.state_dict()
    source_state = model.state_dict()
    for key, source_value in source_state.items():
        if key in candidate_state and candidate_state[key].shape == source_value.shape:
            candidate_state[key] = source_value.detach().clone().to(
                candidate_state[key].device
            )
    candidate.load_state_dict(candidate_state)
    with torch.no_grad():
        if candidate.routed_experts.enabled:
            sleep_tensor = torch.tensor(
                sleep_ids,
                dtype=torch.long,
                device=candidate.routed_experts.sleeping_expert_mask.device,
            )
            candidate.routed_experts.sleeping_expert_mask[sleep_tensor] = True
    candidate.train(model.training)
    return candidate


def _clone_split_model(
    model: MarulhoLanguageModel,
    *,
    split_expert_ids: Sequence[int],
) -> MarulhoLanguageModel:
    parent_ids = tuple(int(value) for value in split_expert_ids)
    if not parent_ids:
        raise ValueError("split_expert_ids must not be empty")
    old_expert_count = max(0, int(model.config.expert_count))
    if any(value < 0 or value >= old_expert_count for value in parent_ids):
        raise ValueError("split_expert_ids contains an out-of-range expert id")
    if len(set(parent_ids)) != len(parent_ids):
        raise ValueError("split_expert_ids must be unique")
    target_expert_count = old_expert_count + len(parent_ids)
    new_config = replace(
        model.config,
        expert_count=target_expert_count,
        active_expert_count=max(1, int(model.config.active_expert_count)),
        route_candidate_count=max(
            max(1, int(model.config.route_candidate_count)),
            max(1, int(model.config.active_expert_count)),
        ),
    )
    split_model = MarulhoLanguageModel(new_config).to(model.device)
    split_state = split_model.state_dict()
    source_state = model.state_dict()
    for key, source_value in source_state.items():
        if key in split_state and split_state[key].shape == source_value.shape:
            split_state[key] = source_value.detach().clone().to(split_state[key].device)
    split_model.load_state_dict(split_state)
    with torch.no_grad():
        if split_model.routed_experts.enabled and model.routed_experts.enabled:
            split_model.routed_experts.sleeping_expert_mask[:old_expert_count].copy_(
                model.routed_experts.sleeping_expert_mask[:old_expert_count].to(
                    device=split_model.routed_experts.sleeping_expert_mask.device
                )
            )
            for child_offset, parent_id in enumerate(parent_ids):
                child_id = old_expert_count + child_offset
                split_model.routed_experts.sleeping_expert_mask[child_id] = False
                split_model.routed_experts.route_keys[child_id].copy_(
                    model.routed_experts.route_keys[parent_id].to(
                        device=split_model.routed_experts.route_keys.device,
                        dtype=split_model.routed_experts.route_keys.dtype,
                    )
                )
                if int(split_model.routed_experts.route_keys.shape[1]) > 0:
                    perturb_index = child_offset % int(
                        split_model.routed_experts.route_keys.shape[1]
                    )
                    split_model.routed_experts.route_keys[
                        child_id,
                        perturb_index,
                    ].add_(1e-4)
                split_model.routed_experts.route_bias[child_id].copy_(
                    model.routed_experts.route_bias[parent_id].to(
                        device=split_model.routed_experts.route_bias.device,
                        dtype=split_model.routed_experts.route_bias.dtype,
                    )
                )
                for target_parameter, source_parameter in zip(
                    split_model.routed_experts.experts[child_id].parameters(),
                    model.routed_experts.experts[parent_id].parameters(),
                    strict=True,
                ):
                    target_parameter.copy_(
                        source_parameter.detach().to(
                            device=target_parameter.device,
                            dtype=target_parameter.dtype,
                        )
                    )
    split_model.train(model.training)
    return split_model


def _clone_synapse_bundle_expanded_model(
    model: MarulhoLanguageModel,
    *,
    target_expert_hidden_dim: int,
) -> MarulhoLanguageModel:
    if not model.routed_experts.enabled:
        raise ValueError("synapse bundle growth requires routed experts")
    old_hidden_dim = int(model.routed_experts.expert_hidden_dim)
    target_hidden_dim = max(int(target_expert_hidden_dim), old_hidden_dim)
    if target_hidden_dim <= old_hidden_dim:
        raise ValueError("target_expert_hidden_dim must grow expert hidden capacity")
    new_config = replace(model.config, expert_hidden_dim=target_hidden_dim)
    expanded = MarulhoLanguageModel(new_config).to(model.device)
    expanded_state = expanded.state_dict()
    source_state = model.state_dict()
    for key, source_value in source_state.items():
        if key in expanded_state and expanded_state[key].shape == source_value.shape:
            expanded_state[key] = source_value.detach().clone().to(expanded_state[key].device)
    expanded.load_state_dict(expanded_state)
    with torch.no_grad():
        expanded.routed_experts.sleeping_expert_mask.copy_(
            model.routed_experts.sleeping_expert_mask.to(
                device=expanded.routed_experts.sleeping_expert_mask.device
            )
        )
        expanded.routed_experts.route_keys.copy_(
            model.routed_experts.route_keys.to(
                device=expanded.routed_experts.route_keys.device,
                dtype=expanded.routed_experts.route_keys.dtype,
            )
        )
        expanded.routed_experts.route_bias.copy_(
            model.routed_experts.route_bias.to(
                device=expanded.routed_experts.route_bias.device,
                dtype=expanded.routed_experts.route_bias.dtype,
            )
        )
        for expert_id in range(int(model.routed_experts.expert_count)):
            source_expert = model.routed_experts.experts[expert_id]
            target_expert = expanded.routed_experts.experts[expert_id]
            target_expert[0].weight[:old_hidden_dim].copy_(
                source_expert[0].weight.detach().to(
                    device=target_expert[0].weight.device,
                    dtype=target_expert[0].weight.dtype,
                )
            )
            target_expert[0].bias[:old_hidden_dim].copy_(
                source_expert[0].bias.detach().to(
                    device=target_expert[0].bias.device,
                    dtype=target_expert[0].bias.dtype,
                )
            )
            target_expert[0].weight[old_hidden_dim:].zero_()
            target_expert[0].bias[old_hidden_dim:].zero_()
            target_expert[2].weight[:, :old_hidden_dim].copy_(
                source_expert[2].weight.detach().to(
                    device=target_expert[2].weight.device,
                    dtype=target_expert[2].weight.dtype,
                )
            )
            target_expert[2].weight[:, old_hidden_dim:].zero_()
            target_expert[2].bias.copy_(
                source_expert[2].bias.detach().to(
                    device=target_expert[2].bias.device,
                    dtype=target_expert[2].bias.dtype,
                )
            )
    expanded.train(model.training)
    return expanded


def _clone_memory_slot_expanded_model(
    model: MarulhoLanguageModel,
    *,
    target_memory_slot_count: int,
    target_memory_slot_candidate_count: int,
    target_active_memory_slot_count: int,
) -> MarulhoLanguageModel:
    source_slot_count = max(0, int(model.config.memory_slot_count))
    target_slot_count = max(int(target_memory_slot_count), source_slot_count)
    if target_slot_count <= source_slot_count:
        raise ValueError("target_memory_slot_count must grow memory slots")
    target_candidate_count = min(
        max(1, int(target_memory_slot_candidate_count)),
        target_slot_count,
    )
    target_active_count = min(
        max(1, int(target_active_memory_slot_count)),
        target_candidate_count,
    )
    new_config = replace(
        model.config,
        memory_slot_count=target_slot_count,
        memory_slot_candidate_count=target_candidate_count,
        active_memory_slot_count=target_active_count,
    )
    expanded = MarulhoLanguageModel(new_config).to(model.device)
    expanded_state = expanded.state_dict()
    source_state = model.state_dict()
    for key, source_value in source_state.items():
        if key in expanded_state and expanded_state[key].shape == source_value.shape:
            expanded_state[key] = source_value.detach().clone().to(expanded_state[key].device)
    expanded.load_state_dict(expanded_state)
    with torch.no_grad():
        if source_slot_count > 0 and model.memory_slots is not None:
            assert expanded.memory_slots is not None
            expanded.memory_slots[:source_slot_count].copy_(
                model.memory_slots.detach().to(
                    device=expanded.memory_slots.device,
                    dtype=expanded.memory_slots.dtype,
                )
            )
            if model.memory_slot_gate is not None and bool(
                model.memory_slot_gate.detach().abs().max().item() > 0.0
            ):
                expanded.memory_slots[source_slot_count:].zero_()
        if source_slot_count <= 0 and expanded.memory_slot_gate is not None:
            expanded.memory_slot_gate.zero_()
    expanded.train(model.training)
    return expanded


def _route_bank_candidate_ceiling(model: MarulhoLanguageModel) -> int:
    if not model.routed_experts.enabled:
        return 0
    expert_count = max(0, int(model.config.expert_count))
    awake_count = max(
        0,
        expert_count - len(model.routed_experts.sleeping_expert_ids()),
    )
    if awake_count <= 0:
        return 0
    if awake_count >= expert_count:
        return max(0, expert_count - 1)
    return awake_count


def _clone_route_bank_expanded_model(
    model: MarulhoLanguageModel,
    *,
    target_route_candidate_count: int,
) -> MarulhoLanguageModel:
    source_candidate_count = max(0, int(model.config.route_candidate_count))
    candidate_ceiling = _route_bank_candidate_ceiling(model)
    target_candidate_count = min(
        max(0, int(target_route_candidate_count)),
        candidate_ceiling,
    )
    if source_candidate_count <= 0:
        raise ValueError("route bank expansion requires a bounded source candidate count")
    if target_candidate_count <= source_candidate_count:
        raise ValueError("target_route_candidate_count must increase the bounded route bank")
    new_config = replace(
        model.config,
        route_candidate_count=target_candidate_count,
    )
    candidate = MarulhoLanguageModel(new_config).to(model.device)
    candidate_state = candidate.state_dict()
    source_state = model.state_dict()
    for key, source_value in source_state.items():
        if key in candidate_state and candidate_state[key].shape == source_value.shape:
            candidate_state[key] = source_value.detach().clone().to(
                candidate_state[key].device
            )
    candidate.load_state_dict(candidate_state)
    candidate.train(model.training)
    return candidate


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


def _normalise_expert_pairs(value: Any) -> tuple[tuple[int, int], ...]:
    if value is None or isinstance(value, (str, bytes)):
        return ()
    raw_pairs: list[Any]
    if isinstance(value, Mapping):
        raw_pairs = list(value.keys())
    else:
        try:
            raw_pairs = list(value)
        except TypeError:
            return ()
    pairs: list[tuple[int, int]] = []
    for raw_pair in raw_pairs:
        pair_value = raw_pair
        if isinstance(raw_pair, str):
            for delimiter in (",", ":", "|", "-"):
                if delimiter in raw_pair:
                    pair_value = raw_pair.split(delimiter, 1)
                    break
        try:
            left, right = list(pair_value)[:2]
            first = int(left)
            second = int(right)
        except (TypeError, ValueError):
            continue
        if first == second:
            continue
        pairs.append((min(first, second), max(first, second)))
    return tuple(dict.fromkeys(pairs))


def _similar_expert_pairs_from_evidence(
    value: Any,
    *,
    threshold: float,
) -> tuple[tuple[int, int], ...]:
    result: list[tuple[int, int]] = []
    if isinstance(value, Mapping):
        iterable = value.items()
    elif isinstance(value, (str, bytes)):
        return ()
    else:
        try:
            iterable = list(value)
        except TypeError:
            return ()
    for item in iterable:
        raw_pair: Any
        raw_similarity: Any
        if isinstance(value, Mapping):
            raw_pair, raw_similarity = item
        elif isinstance(item, Mapping):
            raw_pair = (
                item.get("expert_ids")
                or item.get("pair")
                or item.get("experts")
            )
            raw_similarity = item.get("similarity")
        else:
            try:
                raw_pair = list(item)[:2]
                raw_similarity = list(item)[2]
            except (TypeError, IndexError):
                continue
        try:
            similarity = float(raw_similarity)
        except (TypeError, ValueError):
            continue
        if similarity < float(threshold):
            continue
        result.extend(_normalise_expert_pairs([raw_pair]))
    return tuple(dict.fromkeys(result))


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


def _high_load_ids_from_evidence(
    value: Any,
    *,
    threshold: float,
) -> tuple[int, ...]:
    result: list[int] = []
    if isinstance(value, Mapping):
        for raw_id, raw_load in value.items():
            try:
                if float(raw_load) >= float(threshold):
                    result.append(int(raw_id))
            except (TypeError, ValueError):
                continue
    elif not isinstance(value, (str, bytes)):
        try:
            for expert_id, raw_load in enumerate(value):
                if float(raw_load) >= float(threshold):
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


def build_language_structural_column_split_proposal(
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
    sleeping_ids = set(
        model.routed_experts.sleeping_expert_ids()
        if model.routed_experts.enabled
        else []
    )
    explicit_split_ids = set(
        _normalise_expert_ids(
            route.get("split_candidate_expert_ids")
            or route.get("column_split_candidate_ids")
            or route.get("overloaded_expert_ids")
            or route.get("specialist_overload_expert_ids")
        )
    )
    high_surprise_ids = set(
        _normalise_expert_ids(
            route.get("high_surprise_expert_ids")
            or learning.get("high_surprise_expert_ids")
            or learning.get("prediction_failure_expert_ids")
        )
    )
    high_load_ids = set(
        _high_load_ids_from_evidence(
            route.get("expert_loads")
            or route.get("load_by_expert")
            or route.get("expert_activation_fraction"),
            threshold=float(cfg.split_load_threshold),
        )
    )
    pressure_ids = {
        value
        for value in explicit_split_ids | high_surprise_ids | high_load_ids
        if 0 <= value < expert_count and value not in sleeping_ids
    }
    split_budget = max(
        0,
        min(int(cfg.max_split_experts), int(cfg.max_added_experts)),
    )
    split_ids = tuple(sorted(pressure_ids)[:split_budget])
    child_ids = tuple(expert_count + offset for offset in range(len(split_ids)))
    ready = expert_count > 0 and bool(split_ids)
    status = "ready_for_operator_review" if ready else "collect_more_split_pressure"
    proposal_body = {
        "proposal_kind": "column_split",
        "source_expert_count": int(expert_count),
        "target_expert_count": int(expert_count + len(split_ids)),
        "split_expert_ids": list(split_ids),
        "child_expert_ids": list(child_ids),
        "parent_child_expert_pairs": [
            [int(parent_id), int(child_id)]
            for parent_id, child_id in zip(split_ids, child_ids, strict=True)
        ],
        "added_expert_count": int(len(split_ids)),
        "split_pressure": bool(split_ids),
        "explicit_split_expert_ids": sorted(explicit_split_ids),
        "high_surprise_expert_ids": sorted(high_surprise_ids),
        "high_load_expert_ids": sorted(high_load_ids),
        "sleeping_expert_ids_excluded": sorted(sleeping_ids),
        "split_load_threshold": float(cfg.split_load_threshold),
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
            "split_pressure": bool(split_ids),
            "bounded_added_expert_count": len(split_ids)
            <= max(0, min(int(cfg.max_split_experts), int(cfg.max_added_experts))),
            "parent_experts_checkpointed_before_split": True,
            "child_experts_initialized_from_parent": bool(split_ids),
        },
    }


def build_language_structural_synapse_bundle_proposal(
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
    source_hidden_dim = int(model.routed_experts.expert_hidden_dim) if model.routed_experts.enabled else 0
    growth = min(
        max(0, int(cfg.max_synapse_bundle_hidden_growth)),
        max(0, max(1, source_hidden_dim) // 2),
    )
    target_hidden_dim = source_hidden_dim + growth
    explicit_ids = set(
        _normalise_expert_ids(
            route.get("synapse_bundle_candidate_expert_ids")
            or route.get("new_synapse_bundle_expert_ids")
            or route.get("high_surprise_expert_ids")
            or learning.get("high_surprise_expert_ids")
        )
    )
    replay_conflict = bool(
        route.get("replay_conflict")
        or learning.get("replay_conflict")
        or learning.get("replay_conflict_detected")
    )
    low_confidence = bool(
        route.get("low_confidence_high_uncertainty")
        or learning.get("low_confidence_high_uncertainty")
        or learning.get("low_confidence_with_high_uncertainty")
    )
    pressure = bool(
        route.get("synapse_bundle_pressure")
        or learning.get("synapse_bundle_pressure")
        or explicit_ids
        or replay_conflict
        or low_confidence
    )
    affected_ids = tuple(range(expert_count))
    ready = (
        expert_count > 0
        and source_hidden_dim > 0
        and target_hidden_dim > source_hidden_dim
        and bool(pressure)
    )
    status = "ready_for_operator_review" if ready else "collect_more_synapse_bundle_pressure"
    proposal_body = {
        "proposal_kind": "synapse_bundle_growth",
        "source_expert_count": int(expert_count),
        "target_expert_count": int(expert_count),
        "source_expert_hidden_dim": int(source_hidden_dim),
        "target_expert_hidden_dim": int(target_hidden_dim),
        "added_hidden_units_per_expert": int(max(0, target_hidden_dim - source_hidden_dim)),
        "affected_expert_ids": list(affected_ids),
        "explicit_candidate_expert_ids": sorted(explicit_ids),
        "synapse_bundle_pressure": bool(pressure),
        "replay_conflict": bool(replay_conflict),
        "low_confidence_high_uncertainty": bool(low_confidence),
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
            "synapse_bundle_pressure": bool(pressure),
            "bounded_hidden_growth": bool(
                target_hidden_dim - source_hidden_dim
                <= int(cfg.max_synapse_bundle_hidden_growth)
            ),
            "existing_synapses_preserved": True,
            "new_bundle_initially_neutral": True,
        },
    }


def build_language_structural_memory_slot_expansion_proposal(
    model: MarulhoLanguageModel,
    *,
    routing_evidence: Mapping[str, Any],
    learning_evidence: Mapping[str, Any] | None = None,
    config: LanguageStructuralPlasticityConfig | None = None,
) -> dict[str, Any]:
    cfg = config or LanguageStructuralPlasticityConfig()
    route = dict(routing_evidence or {})
    learning = dict(learning_evidence or {})
    source_slot_count = max(0, int(model.config.memory_slot_count))
    slot_ceiling = int(cfg.max_memory_slot_count)
    if slot_ceiling <= 0:
        slot_ceiling = source_slot_count + max(0, int(cfg.max_memory_slot_growth))
    growth = min(
        max(0, int(cfg.max_memory_slot_growth)),
        max(0, slot_ceiling - source_slot_count),
    )
    target_slot_count = source_slot_count + growth
    max_candidate_count = max(1, int(cfg.max_memory_slot_candidate_count))
    bounded_candidate_ceiling = (
        max(1, target_slot_count - 1)
        if target_slot_count > 1
        else target_slot_count
    )
    target_candidate_count = min(max_candidate_count, bounded_candidate_ceiling)
    target_active_count = min(
        max(1, int(model.config.active_memory_slot_count)),
        max(1, target_candidate_count),
    )
    novel_cluster = bool(
        route.get("novel_concept_cluster")
        or learning.get("novel_concept_cluster")
        or learning.get("novel_concept_cluster_detected")
    )
    replay_conflict = bool(
        route.get("replay_conflict")
        or learning.get("replay_conflict")
        or learning.get("replay_conflict_detected")
    )
    high_surprise = bool(
        route.get("high_surprise")
        or learning.get("high_surprise")
        or learning.get("repeated_high_surprise")
    )
    pressure = bool(
        route.get("memory_slot_pressure")
        or learning.get("memory_slot_pressure")
        or novel_cluster
        or replay_conflict
        or high_surprise
    )
    avoids_all_slot_scan = bool(
        target_slot_count <= 1 or target_candidate_count < target_slot_count
    )
    ready = (
        target_slot_count > source_slot_count
        and bool(pressure)
        and bool(avoids_all_slot_scan)
    )
    status = "ready_for_operator_review" if ready else "collect_more_memory_slot_pressure"
    proposal_body = {
        "proposal_kind": "memory_slot_expansion",
        "source_expert_count": int(model.config.expert_count),
        "target_expert_count": int(model.config.expert_count),
        "source_memory_slot_count": int(source_slot_count),
        "target_memory_slot_count": int(target_slot_count),
        "added_memory_slot_count": int(max(0, target_slot_count - source_slot_count)),
        "target_memory_slot_candidate_count": int(target_candidate_count),
        "target_active_memory_slot_count": int(target_active_count),
        "memory_slot_pressure": bool(pressure),
        "novel_concept_cluster": bool(novel_cluster),
        "replay_conflict": bool(replay_conflict),
        "high_surprise": bool(high_surprise),
        "avoids_all_slot_scan": bool(avoids_all_slot_scan),
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
            "memory_slot_pressure": bool(pressure),
            "bounded_slot_growth": bool(
                target_slot_count - source_slot_count
                <= int(cfg.max_memory_slot_growth)
            ),
            "avoids_all_slot_scan": bool(avoids_all_slot_scan),
            "new_slots_initially_neutral": True,
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


def build_language_structural_retire_proposal(
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
    explicit_retire_ids = set(
        _normalise_expert_ids(
            route.get("retire_candidate_expert_ids")
            or route.get("terminal_retire_expert_ids")
            or route.get("stale_expert_ids")
        )
    )
    harmful_ids = set(
        _normalise_expert_ids(
            route.get("harmful_interference_expert_ids")
            or learning.get("harmful_interference_expert_ids")
        )
    )
    dead_spike_ids = set(_normalise_expert_ids(route.get("dead_spike_expert_ids")))
    high_cost_low_contribution_ids = set(
        _normalise_expert_ids(route.get("high_cost_low_contribution_expert_ids"))
    )
    utility_ids = set(
        _low_utility_ids_from_evidence(
            route.get("expert_utilities") or route.get("utility_by_expert"),
            threshold=float(cfg.prune_utility_threshold),
        )
    )
    pressure_ids = {
        value
        for value in (
            explicit_retire_ids
            | harmful_ids
            | dead_spike_ids
            | high_cost_low_contribution_ids
            | utility_ids
        )
        if 0 <= value < expert_count and value not in active_ids
    }
    retire_budget = max(0, expert_count - active_requirement)
    retired_ids = tuple(
        sorted(pressure_ids)[: min(int(cfg.max_retired_experts), retire_budget)]
    )
    retained_ids = tuple(value for value in range(expert_count) if value not in set(retired_ids))
    ready = expert_count > active_requirement and bool(retired_ids)
    status = "ready_for_operator_review" if ready else "collect_more_retire_pressure"
    proposal_body = {
        "proposal_kind": "expert_retire",
        "source_expert_count": int(expert_count),
        "target_expert_count": int(len(retained_ids)),
        "retired_expert_ids": list(retired_ids),
        "retained_expert_ids": list(retained_ids),
        "active_expert_count_floor": int(active_requirement),
        "retire_pressure": bool(retired_ids),
        "explicit_retire_expert_ids": sorted(explicit_retire_ids),
        "harmful_interference_expert_ids": sorted(harmful_ids),
        "dead_spike_expert_ids": sorted(dead_spike_ids),
        "high_cost_low_contribution_expert_ids": sorted(
            high_cost_low_contribution_ids
        ),
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
            "retire_pressure": bool(retired_ids),
            "min_expert_count_preserved": len(retained_ids) >= int(cfg.min_expert_count),
            "active_expert_count_preserved": len(retained_ids)
            >= int(model.config.active_expert_count),
            "terminal_retirement_reviewable": bool(retired_ids),
        },
    }


def build_language_structural_merge_proposal(
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
    duplicate_pairs = set(
        _normalise_expert_pairs(
            route.get("duplicate_expert_pairs")
            or route.get("merge_candidate_pairs")
            or route.get("duplicate_function_pairs")
        )
    )
    similar_pairs = set(
        _similar_expert_pairs_from_evidence(
            route.get("expert_similarity_pairs")
            or route.get("expert_pair_similarities"),
            threshold=float(cfg.merge_similarity_threshold),
        )
    )
    pressure_pairs = sorted(duplicate_pairs | similar_pairs)
    selected_groups: list[tuple[int, int]] = []
    used_ids: set[int] = set()
    merge_budget = max(0, expert_count - active_requirement)
    max_pairs = min(int(cfg.max_merged_expert_pairs), merge_budget)
    for first, second in pressure_pairs:
        if len(selected_groups) >= max_pairs:
            break
        if not (0 <= first < expert_count and 0 <= second < expert_count):
            continue
        if first in used_ids or second in used_ids:
            continue
        selected_groups.append((first, second))
        used_ids.update((first, second))
    removed_ids = tuple(group[1] for group in selected_groups)
    retained_ids = tuple(value for value in range(expert_count) if value not in set(removed_ids))
    ready = expert_count > active_requirement and bool(selected_groups)
    status = "ready_for_operator_review" if ready else "collect_more_merge_pressure"
    proposal_body = {
        "proposal_kind": "expert_merge",
        "source_expert_count": int(expert_count),
        "target_expert_count": int(len(retained_ids)),
        "merged_expert_groups": [list(group) for group in selected_groups],
        "removed_expert_ids": list(removed_ids),
        "retained_expert_ids": list(retained_ids),
        "active_expert_count_floor": int(active_requirement),
        "merge_pressure": bool(selected_groups),
        "duplicate_expert_pairs": [list(pair) for pair in sorted(duplicate_pairs)],
        "similar_expert_pairs": [list(pair) for pair in sorted(similar_pairs)],
        "merge_similarity_threshold": float(cfg.merge_similarity_threshold),
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
            "merge_pressure": bool(selected_groups),
            "min_expert_count_preserved": len(retained_ids) >= int(cfg.min_expert_count),
            "active_expert_count_preserved": len(retained_ids)
            >= int(model.config.active_expert_count),
        },
    }


def build_language_structural_deep_sleep_proposal(
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
    current_sleeping_ids = set(
        model.routed_experts.sleeping_expert_ids()
        if model.routed_experts.enabled
        else []
    )
    explicit_sleep_ids = set(
        _normalise_expert_ids(
            route.get("sleep_candidate_expert_ids")
            or route.get("stale_expert_ids")
            or route.get("deep_sleep_candidate_expert_ids")
        )
    )
    low_activation_ids = set(_normalise_expert_ids(route.get("low_activation_expert_ids")))
    high_cost_low_contribution_ids = set(
        _normalise_expert_ids(route.get("high_cost_low_contribution_expert_ids"))
    )
    dead_spike_ids = set(_normalise_expert_ids(route.get("dead_spike_expert_ids")))
    utility_ids = set(
        _low_utility_ids_from_evidence(
            route.get("expert_utilities") or route.get("utility_by_expert"),
            threshold=float(cfg.deep_sleep_utility_threshold),
        )
    )
    pressure_ids = {
        value
        for value in (
            explicit_sleep_ids
            | low_activation_ids
            | high_cost_low_contribution_ids
            | dead_spike_ids
            | utility_ids
        )
        if 0 <= value < expert_count
        and value not in active_ids
        and value not in current_sleeping_ids
    }
    awake_count_before = max(0, expert_count - len(current_sleeping_ids))
    sleep_budget = max(0, awake_count_before - active_requirement)
    selected_sleep_ids = tuple(
        sorted(pressure_ids)[: min(int(cfg.max_deep_sleep_experts), sleep_budget)]
    )
    sleeping_after = sorted(current_sleeping_ids | set(selected_sleep_ids))
    awake_after = max(0, expert_count - len(sleeping_after))
    ready = expert_count > active_requirement and bool(selected_sleep_ids)
    status = "ready_for_operator_review" if ready else "collect_more_sleep_pressure"
    proposal_body = {
        "proposal_kind": "expert_deep_sleep",
        "source_expert_count": int(expert_count),
        "target_expert_count": int(expert_count),
        "deep_sleep_expert_ids": list(selected_sleep_ids),
        "existing_sleeping_expert_ids": sorted(current_sleeping_ids),
        "sleeping_expert_ids_after": list(sleeping_after),
        "awake_expert_count_before": int(awake_count_before),
        "awake_expert_count_after": int(awake_after),
        "active_expert_count_floor": int(active_requirement),
        "sleep_pressure": bool(selected_sleep_ids),
        "explicit_sleep_candidate_expert_ids": sorted(explicit_sleep_ids),
        "low_activation_expert_ids": sorted(low_activation_ids),
        "high_cost_low_contribution_expert_ids": sorted(
            high_cost_low_contribution_ids
        ),
        "dead_spike_expert_ids": sorted(dead_spike_ids),
        "utility_threshold": float(cfg.deep_sleep_utility_threshold),
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
            "sleep_pressure": bool(selected_sleep_ids),
            "min_expert_count_preserved": expert_count >= int(cfg.min_expert_count),
            "active_expert_count_preserved": awake_after
            >= int(model.config.active_expert_count),
            "deep_sleep_reduces_awake_candidates": bool(
                awake_after < awake_count_before
            ),
        },
    }


def build_language_structural_route_bank_expansion_proposal(
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
    configured_candidate_count = max(0, int(model.config.route_candidate_count))
    source_candidate_count = int(
        route.get("route_candidate_count") or configured_candidate_count or 0
    )
    bounded_candidate_ceiling = _route_bank_candidate_ceiling(model)
    configured_ceiling = int(cfg.max_route_candidate_count)
    if configured_ceiling > 0:
        bounded_candidate_ceiling = min(bounded_candidate_ceiling, configured_ceiling)
    bounded_bank_enabled = (
        configured_candidate_count > 0
        and source_candidate_count > 0
        and source_candidate_count < bounded_candidate_ceiling
    )
    active_columns = int(route.get("active_columns") or 0)
    route_bank_saturation = active_columns / max(1, int(source_candidate_count))
    explicit_saturation = route.get("route_bank_saturation")
    if explicit_saturation is None:
        explicit_saturation = route.get("route_saturation")
    if explicit_saturation is not None:
        try:
            route_bank_saturation = float(explicit_saturation)
        except (TypeError, ValueError):
            pass
    route_pressure = (
        route_bank_saturation >= float(cfg.route_saturation_threshold)
        or bool(route.get("route_bank_pressure"))
        or bool(route.get("route_saturation_pressure"))
    )
    candidate_growth = min(
        max(0, int(cfg.max_route_candidate_growth)),
        max(0, bounded_candidate_ceiling - source_candidate_count),
    )
    target_candidate_count = source_candidate_count + candidate_growth
    ready = (
        expert_count > 0
        and bool(bounded_bank_enabled)
        and bool(route_pressure)
        and target_candidate_count > source_candidate_count
    )
    status = "ready_for_operator_review" if ready else "collect_more_route_bank_pressure"
    proposal_body = {
        "proposal_kind": "route_bank_expansion",
        "source_expert_count": int(expert_count),
        "target_expert_count": int(expert_count),
        "source_route_candidate_count": int(source_candidate_count),
        "target_route_candidate_count": int(target_candidate_count),
        "added_route_candidate_count": int(
            max(0, target_candidate_count - source_candidate_count)
        ),
        "configured_route_candidate_count": int(configured_candidate_count),
        "bounded_route_candidate_ceiling": int(bounded_candidate_ceiling),
        "route_bank_saturation": float(route_bank_saturation),
        "route_pressure": bool(route_pressure),
        "bounded_bank_enabled": bool(bounded_bank_enabled),
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
            "route_bank_pressure": bool(route_pressure),
            "bounded_bank_enabled": bool(bounded_bank_enabled),
            "bounded_route_candidate_ceiling_preserved": bool(
                target_candidate_count <= bounded_candidate_ceiling
            ),
            "no_parameter_topology_change": True,
            "avoids_all_column_route_scan": bool(
                target_candidate_count < expert_count
                or len(model.routed_experts.sleeping_expert_ids()) > 0
            ),
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
    elif proposal_kind == "expert_retire":
        candidate = _clone_pruned_model(
            model,
            retained_expert_ids=[
                int(value) for value in list(body.get("retained_expert_ids") or [])
            ],
        )
    elif proposal_kind == "column_split":
        candidate = _clone_split_model(
            model,
            split_expert_ids=[
                int(value) for value in list(body.get("split_expert_ids") or [])
            ],
        )
    elif proposal_kind == "expert_merge":
        candidate = _clone_merged_model(
            model,
            merged_expert_groups=[
                [int(value) for value in list(group)]
                for group in list(body.get("merged_expert_groups") or [])
            ],
        )
    elif proposal_kind == "expert_deep_sleep":
        candidate = _clone_deep_sleep_model(
            model,
            deep_sleep_expert_ids=[
                int(value) for value in list(body.get("deep_sleep_expert_ids") or [])
            ],
        )
    elif proposal_kind == "route_bank_expansion":
        candidate = _clone_route_bank_expanded_model(
            model,
            target_route_candidate_count=int(
                body.get("target_route_candidate_count")
                or model.config.route_candidate_count
            ),
        )
    elif proposal_kind == "synapse_bundle_growth":
        candidate = _clone_synapse_bundle_expanded_model(
            model,
            target_expert_hidden_dim=int(
                body.get("target_expert_hidden_dim")
                or model.config.expert_hidden_dim
            ),
        )
    elif proposal_kind == "memory_slot_expansion":
        candidate = _clone_memory_slot_expanded_model(
            model,
            target_memory_slot_count=int(
                body.get("target_memory_slot_count")
                or model.config.memory_slot_count
            ),
            target_memory_slot_candidate_count=int(
                body.get("target_memory_slot_candidate_count")
                or model.config.memory_slot_candidate_count
            ),
            target_active_memory_slot_count=int(
                body.get("target_active_memory_slot_count")
                or model.config.active_memory_slot_count
            ),
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
            "source_route_candidate_count": int(model.config.route_candidate_count),
            "target_route_candidate_count": int(
                final_model.config.route_candidate_count
            ),
            "candidate_target_route_candidate_count": int(
                candidate.config.route_candidate_count
            ),
            "source_expert_hidden_dim": int(model.config.expert_hidden_dim),
            "target_expert_hidden_dim": int(final_model.config.expert_hidden_dim),
            "candidate_target_expert_hidden_dim": int(
                candidate.config.expert_hidden_dim
            ),
            "synapse_bundle_hidden_growth": int(
                max(
                    0,
                    candidate.config.expert_hidden_dim - model.config.expert_hidden_dim,
                )
                if proposal_kind == "synapse_bundle_growth"
                else 0
            ),
            "source_memory_slot_count": int(model.config.memory_slot_count),
            "target_memory_slot_count": int(final_model.config.memory_slot_count),
            "candidate_target_memory_slot_count": int(
                candidate.config.memory_slot_count
            ),
            "source_memory_slot_candidate_count": int(
                model.config.memory_slot_candidate_count
            ),
            "target_memory_slot_candidate_count": int(
                final_model.config.memory_slot_candidate_count
            ),
            "candidate_target_memory_slot_candidate_count": int(
                candidate.config.memory_slot_candidate_count
            ),
            "source_active_memory_slot_count": int(
                model.config.active_memory_slot_count
            ),
            "target_active_memory_slot_count": int(
                final_model.config.active_memory_slot_count
            ),
            "candidate_target_active_memory_slot_count": int(
                candidate.config.active_memory_slot_count
            ),
            "memory_slot_count_delta": int(
                max(
                    0,
                    candidate.config.memory_slot_count - model.config.memory_slot_count,
                )
                if proposal_kind == "memory_slot_expansion"
                else 0
            ),
            "route_bank_candidate_count_delta": int(
                max(
                    0,
                    candidate.config.route_candidate_count
                    - model.config.route_candidate_count,
                )
                if proposal_kind == "route_bank_expansion"
                else 0
            ),
            "added_expert_count": int(
                max(0, candidate.config.expert_count - model.config.expert_count)
                if proposal_kind in {"expert_spawn", "column_split"}
                else 0
            ),
            "split_expert_count": int(
                len(list(body.get("split_expert_ids") or []))
                if proposal_kind == "column_split"
                else 0
            ),
            "split_expert_ids": [
                int(value) for value in list(body.get("split_expert_ids") or [])
            ],
            "split_child_expert_ids": [
                int(value) for value in list(body.get("child_expert_ids") or [])
            ],
            "parent_child_expert_pairs": [
                [int(value) for value in list(pair)]
                for pair in list(body.get("parent_child_expert_pairs") or [])
            ],
            "pruned_expert_count": int(
                max(0, model.config.expert_count - candidate.config.expert_count)
                if proposal_kind == "expert_prune"
                else 0
            ),
            "retired_expert_count": int(
                max(0, model.config.expert_count - candidate.config.expert_count)
                if proposal_kind == "expert_retire"
                else 0
            ),
            "retired_expert_ids": [
                int(value) for value in list(body.get("retired_expert_ids") or [])
            ],
            "merged_expert_group_count": int(
                len(list(body.get("merged_expert_groups") or []))
                if proposal_kind == "expert_merge"
                else 0
            ),
            "structural_reduction_count": int(
                max(0, model.config.expert_count - candidate.config.expert_count)
            ),
            "deep_sleep_expert_count": int(
                len(list(body.get("deep_sleep_expert_ids") or []))
                if proposal_kind == "expert_deep_sleep"
                else 0
            ),
            "deep_sleep_expert_ids": [
                int(value) for value in list(body.get("deep_sleep_expert_ids") or [])
            ],
            "sleeping_expert_ids_after": [
                int(value)
                for value in (
                    candidate.routed_experts.sleeping_expert_ids()
                    if candidate.routed_experts.enabled
                    else []
                )
            ],
            "awake_expert_count_after": int(
                len(
                    candidate.routed_experts.awake_expert_ids()
                    if candidate.routed_experts.enabled
                    else []
                )
            ),
            "pruned_expert_ids": [
                int(value) for value in list(body.get("pruned_expert_ids") or [])
            ],
            "removed_expert_ids": [
                int(value) for value in list(body.get("removed_expert_ids") or [])
            ],
            "retained_expert_ids": [
                int(value) for value in list(body.get("retained_expert_ids") or [])
            ],
            "merged_expert_groups": [
                [int(value) for value in list(group)]
                for group in list(body.get("merged_expert_groups") or [])
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
            "eligible_for_reviewed_column_split_promotion": bool(
                accepted and proposal_kind == "column_split"
            ),
            "eligible_for_reviewed_prune_promotion": bool(
                accepted and proposal_kind == "expert_prune"
            ),
            "eligible_for_reviewed_retire_promotion": bool(
                accepted and proposal_kind == "expert_retire"
            ),
            "eligible_for_reviewed_merge_promotion": bool(
                accepted and proposal_kind == "expert_merge"
            ),
            "eligible_for_reviewed_deep_sleep_promotion": bool(
                accepted and proposal_kind == "expert_deep_sleep"
            ),
            "eligible_for_reviewed_route_bank_expansion_promotion": bool(
                accepted and proposal_kind == "route_bank_expansion"
            ),
            "eligible_for_reviewed_synapse_bundle_promotion": bool(
                accepted and proposal_kind == "synapse_bundle_growth"
            ),
            "eligible_for_reviewed_memory_slot_expansion_promotion": bool(
                accepted and proposal_kind == "memory_slot_expansion"
            ),
            "eligible_for_reviewed_structural_promotion": bool(accepted),
            "checkpoint_backed": True,
            "operator_approved": bool(operator_approved),
            "heldout_non_regression": bool(accepted),
            "rollback_verified": bool(rollback_verified),
        },
    }
