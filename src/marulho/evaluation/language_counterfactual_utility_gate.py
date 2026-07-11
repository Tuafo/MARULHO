"""Train a causal utility gate from frozen V11 counterfactual route labels."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence
from uuid import uuid4

import torch
from torch import nn
import torch.nn.functional as F

from marulho.evaluation.language_hashed_micro_expert_counterfactual import (
    CounterfactualRouteAuditConfig,
    _precision_context,
    _source_batches,
)
from marulho.evaluation.language_matched_support import parameter_sha256, sha256_file
from marulho.reporting.readme_reports import write_json_report_with_readme
from marulho.training.language_hashed_micro_experts import (
    MarulhoHashedMicroExpertLanguageModel,
    load_hashed_micro_expert_checkpoint,
)
from marulho.training.language_model import LanguageBatch


SURFACE = "marulho_counterfactual_utility_gate_training.v1"
ARTIFACT_KIND = "marulho_counterfactual_utility_gate_training"
GATE_CHECKPOINT_SURFACE = "marulho_counterfactual_utility_gate_checkpoint.v1"
PROMOTE_DECISION = "promote_v12_gate_for_integrated_falsification"


@dataclass(frozen=True)
class CounterfactualUtilityGateConfig:
    sequence_length: int = 72
    batch_size: int = 144
    train_batches_per_source: int = 64
    eval_batches_per_source: int = 16
    train_sample_bytes_per_source: int = 64 * 1024 * 1024
    eval_sample_bytes_per_source: int = 32 * 1024 * 1024
    sample_range_count: int = 16
    alternative_seed_offsets: tuple[int, ...] = (4093, 8191, 12289, 16381)
    precision: str = "bfloat16"
    hidden_width: int = 64
    epochs: int = 40
    gate_batch_size: int = 512
    learning_rate: float = 2.0e-3
    weight_decay: float = 1.0e-4
    gain_clip: float = 2.0
    opportunity_weight: float = 2.0
    route_thresholds: tuple[float, ...] = (0.0, 0.01, 0.05, 0.10)
    minimum_eval_loss_improvement_per_source: float = 0.02
    minimum_alternative_selection_fraction: float = 0.05
    seed: int = 2031


class CounterfactualUtilityGate(nn.Module):
    def __init__(
        self,
        *,
        feature_mean: torch.Tensor,
        feature_scale: torch.Tensor,
        alternative_count: int,
        kind: str,
        hidden_width: int,
    ) -> None:
        super().__init__()
        if kind not in {"linear", "mlp"}:
            raise ValueError("Gate kind must be linear or mlp")
        self.kind = str(kind)
        self.alternative_count = int(alternative_count)
        self.hidden_width = int(hidden_width)
        self.register_buffer("feature_mean", feature_mean.detach().float().clone())
        self.register_buffer("feature_scale", feature_scale.detach().float().clone())
        width = int(feature_mean.numel())
        self.network: nn.Module
        if kind == "linear":
            self.network = nn.Linear(width, self.alternative_count)
        else:
            self.network = nn.Sequential(
                nn.Linear(width, self.hidden_width, bias=False),
                nn.SiLU(),
                nn.Linear(self.hidden_width, self.alternative_count),
            )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        normalized = (features.float() - self.feature_mean) / self.feature_scale
        return self.network(normalized)


def utility_gate_decision(
    source_improvements: Sequence[float],
    *,
    alternative_selection_fraction: float,
    parameters_unchanged: bool,
    minimum_source_improvement: float = 0.02,
    minimum_alternative_selection_fraction: float = 0.05,
) -> str:
    if not parameters_unchanged:
        return "invalid_v12_gate_training_mutated_parent"
    if not source_improvements:
        return "incomplete_v12_gate_evaluation"
    if (
        min(float(value) for value in source_improvements)
        >= float(minimum_source_improvement)
        and float(alternative_selection_fraction)
        >= float(minimum_alternative_selection_fraction)
    ):
        return PROMOTE_DECISION
    if max(float(value) for value in source_improvements) > 0.0:
        return "redesign_v12_gate_not_general_across_sources"
    return "retire_v12_gate_cannot_predict_counterfactual_utility"


def _collect_examples(
    model: MarulhoHashedMicroExpertLanguageModel,
    batches: Sequence[LanguageBatch],
    *,
    alternative_seed_offsets: Sequence[int],
    precision: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not batches:
        raise ValueError("Utility-gate collection requires batches")
    layer = model.state_block.expert_layer
    features: list[torch.Tensor] = []
    loss_rows: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for batch in batches:
            input_ids = batch.input_ids.to(device=model.device, dtype=torch.long)
            targets = batch.target_ids[:, -1].to(
                device=model.device,
                dtype=torch.long,
            )
            captured: list[torch.Tensor] = []

            def capture_feature(_module, _inputs, output) -> None:
                captured.append(output[:, -1].detach().float().cpu())

            handle = layer.mlp_norm.register_forward_hook(capture_feature)
            try:
                with _precision_context(model.device, precision):
                    baseline_logits = model(
                        input_ids,
                        collect_telemetry=False,
                    )["logits"][:, -1]
            finally:
                handle.remove()
            if len(captured) != 1:
                raise RuntimeError("Utility-gate feature hook did not fire exactly once")
            baseline_loss = F.cross_entropy(
                baseline_logits.float(),
                targets,
                reduction="none",
            )
            baseline_routes = layer.hash_routes(input_ids)
            candidate_losses = [baseline_loss]
            for offset in alternative_seed_offsets:
                forced_routes = baseline_routes.clone()
                forced_routes[:, -1] = layer.hash_routes(
                    input_ids[:, -1:],
                    hash_seed=layer.hash_seed + int(offset),
                )[:, 0]
                with _precision_context(model.device, precision):
                    alternative_logits = model.forward_with_forced_expert_ids(
                        input_ids,
                        forced_routes,
                    )["logits"][:, -1]
                candidate_losses.append(
                    F.cross_entropy(
                        alternative_logits.float(),
                        targets,
                        reduction="none",
                    )
                )
            features.append(captured[0])
            loss_rows.append(torch.stack(candidate_losses, dim=1).detach().cpu())
    return torch.cat(features).float(), torch.cat(loss_rows).float()


def _gate_metrics(
    gate: CounterfactualUtilityGate,
    features: torch.Tensor,
    candidate_losses: torch.Tensor,
    *,
    threshold: float,
    device: torch.device,
) -> dict[str, Any]:
    gate.eval()
    with torch.no_grad():
        predicted_gain = gate(features.to(device)).detach().cpu()
    best_predicted_gain, alternative_index = predicted_gain.max(dim=1)
    choose_alternative = best_predicted_gain > float(threshold)
    selected_index = torch.where(
        choose_alternative,
        alternative_index + 1,
        torch.zeros_like(alternative_index),
    )
    realized = candidate_losses.gather(1, selected_index.unsqueeze(1))[:, 0]
    baseline = candidate_losses[:, 0]
    oracle, oracle_index = candidate_losses.min(dim=1)
    selection_counts = torch.bincount(
        selected_index,
        minlength=int(candidate_losses.shape[1]),
    )
    return {
        "token_count": int(features.shape[0]),
        "threshold": float(threshold),
        "baseline_mean_loss": float(baseline.mean().item()),
        "realized_mean_loss": float(realized.mean().item()),
        "realized_loss_improvement": float(
            baseline.mean().item() - realized.mean().item()
        ),
        "oracle_mean_loss": float(oracle.mean().item()),
        "oracle_loss_improvement": float(
            baseline.mean().item() - oracle.mean().item()
        ),
        "alternative_selection_fraction": float(
            choose_alternative.float().mean().item()
        ),
        "positive_realized_improvement_fraction": float(
            (realized < baseline).float().mean().item()
        ),
        "negative_realized_improvement_fraction": float(
            (realized > baseline).float().mean().item()
        ),
        "mean_predicted_best_gain": float(best_predicted_gain.mean().item()),
        "route_selection_counts": [
            int(value) for value in selection_counts.tolist()
        ],
        "oracle_selection_accuracy": float(
            (selected_index == oracle_index).float().mean().item()
        ),
        "prediction_uses_targets": False,
        "targets_metrics_only": True,
    }


def _train_gate(
    *,
    kind: str,
    train_features: torch.Tensor,
    train_losses: torch.Tensor,
    config: CounterfactualUtilityGateConfig,
    device: torch.device,
) -> tuple[CounterfactualUtilityGate, dict[str, Any]]:
    feature_mean = train_features.mean(dim=0)
    feature_scale = train_features.std(dim=0).clamp_min(1.0e-4)
    torch.manual_seed(int(config.seed) + (0 if kind == "linear" else 1))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(config.seed) + (0 if kind == "linear" else 1))
    gate = CounterfactualUtilityGate(
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        alternative_count=int(train_losses.shape[1]) - 1,
        kind=kind,
        hidden_width=int(config.hidden_width),
    ).to(device)
    optimizer = torch.optim.AdamW(
        gate.parameters(),
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )
    features = train_features.to(device)
    target_gain = (
        train_losses[:, :1] - train_losses[:, 1:]
    ).clamp(-float(config.gain_clip), float(config.gain_clip)).to(device)
    opportunity = target_gain.clamp_min(0.0).max(dim=1).values
    example_weight = 1.0 + float(config.opportunity_weight) * opportunity.clamp_max(1.0)
    generator = torch.Generator(device="cpu").manual_seed(int(config.seed))
    trace: list[dict[str, float | int]] = []
    count = int(features.shape[0])
    for epoch in range(int(config.epochs)):
        order = torch.randperm(count, generator=generator)
        epoch_loss = 0.0
        seen = 0
        gate.train()
        for start in range(0, count, int(config.gate_batch_size)):
            indices = order[start : start + int(config.gate_batch_size)].to(device)
            prediction = gate(features.index_select(0, indices))
            element_loss = F.smooth_l1_loss(
                prediction,
                target_gain.index_select(0, indices),
                reduction="none",
            ).mean(dim=1)
            loss = (
                element_loss * example_weight.index_select(0, indices)
            ).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            batch_count = int(indices.numel())
            epoch_loss += float(loss.detach().cpu().item()) * batch_count
            seen += batch_count
        if epoch in {0, int(config.epochs) // 2, int(config.epochs) - 1}:
            trace.append(
                {
                    "epoch": epoch + 1,
                    "weighted_huber_loss": epoch_loss / max(1, seen),
                }
            )
    threshold_rows = [
        _gate_metrics(
            gate,
            train_features,
            train_losses,
            threshold=float(threshold),
            device=device,
        )
        for threshold in config.route_thresholds
    ]
    selected = min(
        threshold_rows,
        key=lambda row: float(row["realized_mean_loss"]),
    )
    return gate, {
        "kind": kind,
        "parameter_count": sum(parameter.numel() for parameter in gate.parameters()),
        "training_trace": trace,
        "threshold_search_training_only": threshold_rows,
        "selected_threshold": float(selected["threshold"]),
        "selected_training_metrics": selected,
    }


def _save_gate_checkpoint(
    path: Path,
    *,
    gate: CounterfactualUtilityGate,
    gate_record: Mapping[str, Any],
    parent_checkpoint: Path,
    parent_sha256: str,
    tokenizer_hash: str,
    config: CounterfactualUtilityGateConfig,
    evaluation: Mapping[str, Any],
) -> Path:
    payload = {
        "surface": GATE_CHECKPOINT_SURFACE,
        "owned_by_marulho": True,
        "external_llm_used": False,
        "parent_checkpoint": str(parent_checkpoint),
        "parent_checkpoint_sha256": parent_sha256,
        "tokenizer_hash": tokenizer_hash,
        "alternative_seed_offsets": list(config.alternative_seed_offsets),
        "gate_kind": gate.kind,
        "hidden_width": gate.hidden_width,
        "alternative_count": gate.alternative_count,
        "selected_threshold": float(gate_record["selected_threshold"]),
        "state_dict": gate.state_dict(),
        "configuration": asdict(config),
        "evaluation": dict(evaluation),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def run_counterfactual_utility_gate_training(
    *,
    checkpoint_path: str | Path,
    train_corpus_paths: Sequence[str | Path],
    eval_corpus_paths: Sequence[str | Path],
    output_path: str | Path,
    gate_checkpoint_output_path: str | Path | None = None,
    config: CounterfactualUtilityGateConfig = CounterfactualUtilityGateConfig(),
    device: str = "auto",
) -> dict[str, Any]:
    if len(train_corpus_paths) < 1 or len(eval_corpus_paths) < 1:
        raise ValueError("Utility-gate training requires train and eval corpora")
    resolved = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if str(device) == "auto"
        else torch.device(device)
    )
    checkpoint = Path(checkpoint_path)
    model, tokenizer, metadata = load_hashed_micro_expert_checkpoint(
        checkpoint,
        map_location="cpu",
    )
    if int(metadata.get("processed_tokens") or 0) < 251_000_000:
        raise ValueError("Utility gate requires the 251M V11 checkpoint")
    model = model.to(resolved).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    parent_hash_before = parameter_sha256(model)
    parent_sha256 = sha256_file(checkpoint)
    started = time.perf_counter()

    train_features_parts: list[torch.Tensor] = []
    train_losses_parts: list[torch.Tensor] = []
    train_sources: list[dict[str, Any]] = []
    train_data_config = CounterfactualRouteAuditConfig(
        sequence_length=config.sequence_length,
        batch_size=config.batch_size,
        eval_batches_per_source=config.train_batches_per_source,
        sample_bytes_per_source=config.train_sample_bytes_per_source,
        sample_range_count=config.sample_range_count,
        alternative_seed_offsets=config.alternative_seed_offsets,
        precision=config.precision,
    )
    for path in train_corpus_paths:
        batches, selection = _source_batches(path, tokenizer, config=train_data_config)
        features, losses = _collect_examples(
            model,
            batches,
            alternative_seed_offsets=config.alternative_seed_offsets,
            precision=config.precision,
        )
        train_features_parts.append(features)
        train_losses_parts.append(losses)
        train_sources.append(
            {
                "path": str(path),
                "selection": selection,
                "example_count": int(features.shape[0]),
            }
        )
    train_features = torch.cat(train_features_parts)
    train_losses = torch.cat(train_losses_parts)

    eval_sets: list[dict[str, Any]] = []
    eval_data_config = CounterfactualRouteAuditConfig(
        sequence_length=config.sequence_length,
        batch_size=config.batch_size,
        eval_batches_per_source=config.eval_batches_per_source,
        sample_bytes_per_source=config.eval_sample_bytes_per_source,
        sample_range_count=config.sample_range_count,
        alternative_seed_offsets=config.alternative_seed_offsets,
        precision=config.precision,
    )
    for path in eval_corpus_paths:
        batches, selection = _source_batches(path, tokenizer, config=eval_data_config)
        features, losses = _collect_examples(
            model,
            batches,
            alternative_seed_offsets=config.alternative_seed_offsets,
            precision=config.precision,
        )
        eval_sets.append(
            {
                "path": str(path),
                "selection": selection,
                "features": features,
                "losses": losses,
            }
        )

    gate_rows: list[dict[str, Any]] = []
    trained_gates: dict[str, CounterfactualUtilityGate] = {}
    for kind in ("linear", "mlp"):
        gate, training_record = _train_gate(
            kind=kind,
            train_features=train_features,
            train_losses=train_losses,
            config=config,
            device=resolved,
        )
        threshold = float(training_record["selected_threshold"])
        evaluations = [
            {
                "path": row["path"],
                "selection": row["selection"],
                "metrics": _gate_metrics(
                    gate,
                    row["features"],
                    row["losses"],
                    threshold=threshold,
                    device=resolved,
                ),
            }
            for row in eval_sets
        ]
        combined_features = torch.cat([row["features"] for row in eval_sets])
        combined_losses = torch.cat([row["losses"] for row in eval_sets])
        combined = _gate_metrics(
            gate,
            combined_features,
            combined_losses,
            threshold=threshold,
            device=resolved,
        )
        gate_rows.append(
            {
                **training_record,
                "evaluations": evaluations,
                "combined_evaluation": combined,
            }
        )
        trained_gates[kind] = gate

    selected_gate_record = min(
        gate_rows,
        key=lambda row: float(row["combined_evaluation"]["realized_mean_loss"]),
    )
    selected_kind = str(selected_gate_record["kind"])
    parent_hash_after = parameter_sha256(model)
    source_improvements = [
        float(row["metrics"]["realized_loss_improvement"])
        for row in selected_gate_record["evaluations"]
    ]
    decision = utility_gate_decision(
        source_improvements,
        alternative_selection_fraction=float(
            selected_gate_record["combined_evaluation"][
                "alternative_selection_fraction"
            ]
        ),
        parameters_unchanged=parent_hash_before == parent_hash_after,
        minimum_source_improvement=(
            config.minimum_eval_loss_improvement_per_source
        ),
        minimum_alternative_selection_fraction=(
            config.minimum_alternative_selection_fraction
        ),
    )
    gate_checkpoint_record: dict[str, Any] | None = None
    if gate_checkpoint_output_path is not None:
        gate_output = Path(gate_checkpoint_output_path)
        if decision == PROMOTE_DECISION:
            saved = _save_gate_checkpoint(
                gate_output,
                gate=trained_gates[selected_kind],
                gate_record=selected_gate_record,
                parent_checkpoint=checkpoint,
                parent_sha256=parent_sha256,
                tokenizer_hash=tokenizer.vocabulary_hash(),
                config=config,
                evaluation={
                    "decision": decision,
                    "source_improvements": source_improvements,
                    "combined": selected_gate_record["combined_evaluation"],
                },
            )
            gate_checkpoint_record = {
                "path": str(saved),
                "sha256": sha256_file(saved),
                "size_bytes": int(saved.stat().st_size),
                "integrated_model_checkpoint": False,
            }
        elif gate_output.exists():
            gate_output.unlink()

    report = {
        "artifact_kind": ARTIFACT_KIND,
        "surface": SURFACE,
        "owned_by_marulho": True,
        "external_llm_used": False,
        "configuration": asdict(config),
        "parent": {
            "path": str(checkpoint),
            "sha256": parent_sha256,
            "processed_tokens": int(metadata["processed_tokens"]),
            "parameter_sha256_before": parent_hash_before,
            "parameter_sha256_after": parent_hash_after,
            "parameters_unchanged": parent_hash_before == parent_hash_after,
            "all_parameters_frozen": all(
                not parameter.requires_grad for parameter in model.parameters()
            ),
        },
        "training_data": {
            "sources": train_sources,
            "example_count": int(train_features.shape[0]),
            "labels": "detached_exact_counterfactual_loss_improvement",
            "eval_sources_excluded": True,
        },
        "gates": gate_rows,
        "selected_gate_kind": selected_kind,
        "selected_gate": selected_gate_record,
        "decision": decision,
        "decision_contract": {
            "minimum_eval_loss_improvement_per_source": (
                config.minimum_eval_loss_improvement_per_source
            ),
            "minimum_alternative_selection_fraction": (
                config.minimum_alternative_selection_fraction
            ),
            "oracle_not_available_to_gate": True,
            "integrated_falsification_required": True,
        },
        "gate_checkpoint": gate_checkpoint_record,
        "anti_cheat": {
            "gate_inputs": "causal_pre_expert_hidden_state_only",
            "training_targets_use_next_token_labels": True,
            "evaluation_route_selection_uses_targets": False,
            "evaluation_targets_metrics_only": True,
            "parent_model_or_experts_updated": False,
            "gate_checkpoint_is_full_model": False,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json_report_with_readme(
        output_path,
        report,
        title="MARULHO V12 Counterfactual Utility Gate",
    )
    print(
        f"[v12-utility-gate] decision {decision}; selected {selected_kind}, "
        f"source gains {source_improvements}",
        flush=True,
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-corpus", action="append", type=Path, required=True)
    parser.add_argument("--eval-corpus", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gate-checkpoint-output", type=Path)
    parser.add_argument("--train-batches-per-source", type=int, default=64)
    parser.add_argument("--eval-batches-per-source", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    run_counterfactual_utility_gate_training(
        checkpoint_path=args.checkpoint,
        train_corpus_paths=tuple(args.train_corpus),
        eval_corpus_paths=tuple(args.eval_corpus),
        output_path=args.output,
        gate_checkpoint_output_path=args.gate_checkpoint_output,
        config=CounterfactualUtilityGateConfig(
            train_batches_per_source=int(args.train_batches_per_source),
            eval_batches_per_source=int(args.eval_batches_per_source),
            epochs=int(args.epochs),
        ),
        device=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
