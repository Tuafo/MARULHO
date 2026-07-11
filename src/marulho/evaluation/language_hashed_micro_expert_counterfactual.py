"""Read-only counterfactual route-regret audit for the V11 checkpoint."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
import time
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from marulho.evaluation.language_matched_support import (
    full_sized_batches,
    parameter_sha256,
    sample_corpus_ranges,
    sha256_file,
)
from marulho.reporting.readme_reports import write_json_report_with_readme
from marulho.training.language_hashed_micro_experts import (
    MarulhoHashedMicroExpertLanguageModel,
    load_hashed_micro_expert_checkpoint,
)
from marulho.training.language_model import (
    LanguageBatch,
    build_language_model_splits,
)


SURFACE = "marulho_hashed_micro_expert_counterfactual_route_audit.v1"
ARTIFACT_KIND = "marulho_hashed_micro_expert_counterfactual_route_audit"
TRAIN_GATE_DECISION = "train_v12_counterfactual_gate"


@dataclass(frozen=True)
class CounterfactualRouteAuditConfig:
    sequence_length: int = 72
    batch_size: int = 144
    eval_batches_per_source: int = 16
    sample_bytes_per_source: int = 32 * 1024 * 1024
    sample_range_count: int = 16
    alternative_seed_offsets: tuple[int, ...] = (4093, 8191, 12289, 16381)
    precision: str = "bfloat16"
    minimum_mean_oracle_loss_improvement: float = 0.02
    minimum_fraction_regret_005: float = 0.10
    forced_baseline_logit_tolerance: float = 1.0e-5


def counterfactual_route_decision(
    *,
    mean_oracle_loss_improvement: float,
    fraction_regret_005: float,
    parameters_unchanged: bool,
    forced_baseline_max_logit_delta: float,
    forced_baseline_logit_tolerance: float = 1.0e-5,
    minimum_mean_oracle_loss_improvement: float = 0.02,
    minimum_fraction_regret_005: float = 0.10,
) -> str:
    if not parameters_unchanged:
        return "invalid_counterfactual_audit_mutated_model"
    if float(forced_baseline_max_logit_delta) > float(
        forced_baseline_logit_tolerance
    ):
        return "invalid_counterfactual_audit_forced_route_mismatch"
    if (
        float(mean_oracle_loss_improvement)
        >= float(minimum_mean_oracle_loss_improvement)
        and float(fraction_regret_005) >= float(minimum_fraction_regret_005)
    ):
        return TRAIN_GATE_DECISION
    if float(mean_oracle_loss_improvement) > 0.0:
        return "redesign_v12_route_bank_no_broad_opportunity"
    return "retain_v11_no_counterfactual_route_opportunity"


def _quantiles(values: torch.Tensor) -> dict[str, float]:
    runtime = values.detach().float().cpu()
    return {
        "minimum": float(runtime.min().item()),
        "p10": float(torch.quantile(runtime, 0.10).item()),
        "median": float(torch.quantile(runtime, 0.50).item()),
        "p90": float(torch.quantile(runtime, 0.90).item()),
        "p99": float(torch.quantile(runtime, 0.99).item()),
        "maximum": float(runtime.max().item()),
        "mean": float(runtime.mean().item()),
    }


def _precision_context(device: torch.device, precision: str):
    if device.type != "cuda" or str(precision) == "float32":
        return nullcontext()
    if str(precision) != "bfloat16":
        raise ValueError("Counterfactual audit precision must be float32 or bfloat16")
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def _audit_batches(
    model: MarulhoHashedMicroExpertLanguageModel,
    batches: Sequence[LanguageBatch],
    *,
    alternative_seed_offsets: Sequence[int],
    precision: str,
) -> dict[str, Any]:
    if not batches:
        raise ValueError("Counterfactual audit requires at least one batch")
    if not alternative_seed_offsets:
        raise ValueError("Counterfactual audit requires alternative route seeds")
    layer = model.state_block.expert_layer
    offsets = tuple(int(value) for value in alternative_seed_offsets)
    if len(set(value % layer.expert_pool_size for value in offsets)) != len(offsets):
        raise ValueError("Alternative seed offsets must be distinct modulo pool size")
    if any(value % layer.expert_pool_size == 0 for value in offsets):
        raise ValueError("Alternative seed offsets must differ from baseline")

    before_hash = parameter_sha256(model)
    model.eval()
    baseline_parts: list[torch.Tensor] = []
    candidate_parts: list[torch.Tensor] = []
    route_parts: list[torch.Tensor] = []
    forced_baseline_max_logit_delta = 0.0
    evaluated_tokens = 0
    with torch.no_grad():
        for batch_index, batch in enumerate(batches):
            input_ids = batch.input_ids.to(device=model.device, dtype=torch.long)
            targets = batch.target_ids[:, -1].to(
                device=model.device,
                dtype=torch.long,
            )
            baseline_routes = layer.hash_routes(input_ids)
            with _precision_context(model.device, precision):
                baseline_logits = model(
                    input_ids,
                    collect_telemetry=False,
                )["logits"][:, -1]
            baseline_loss = F.cross_entropy(
                baseline_logits.float(),
                targets,
                reduction="none",
            )
            if batch_index == 0:
                with _precision_context(model.device, precision):
                    forced_baseline = model.forward_with_forced_expert_ids(
                        input_ids,
                        baseline_routes,
                    )["logits"][:, -1]
                forced_baseline_max_logit_delta = float(
                    (forced_baseline.float() - baseline_logits.float())
                    .abs()
                    .max()
                    .item()
                )

            losses = [baseline_loss]
            final_routes = [baseline_routes[:, -1].detach().cpu()]
            for offset in offsets:
                forced_routes = baseline_routes.clone()
                alternative_final = layer.hash_routes(
                    input_ids[:, -1:],
                    hash_seed=layer.hash_seed + offset,
                )[:, 0]
                forced_routes[:, -1] = alternative_final
                with _precision_context(model.device, precision):
                    alternative_logits = model.forward_with_forced_expert_ids(
                        input_ids,
                        forced_routes,
                    )["logits"][:, -1]
                losses.append(
                    F.cross_entropy(
                        alternative_logits.float(),
                        targets,
                        reduction="none",
                    )
                )
                final_routes.append(alternative_final.detach().cpu())
            baseline_parts.append(baseline_loss.detach().cpu())
            candidate_parts.append(torch.stack(losses, dim=1).detach().cpu())
            route_parts.append(torch.stack(final_routes, dim=1))
            evaluated_tokens += int(targets.numel())

    after_hash = parameter_sha256(model)
    baseline = torch.cat(baseline_parts).float()
    candidate_losses = torch.cat(candidate_parts).float()
    routes = torch.cat(route_parts).long()
    best_loss, best_index = candidate_losses.min(dim=1)
    regret = baseline - best_loss
    improved = regret > 0.0
    fragile_threshold = float(torch.quantile(baseline, 0.50).item())
    fragile = baseline >= fragile_threshold
    confident = ~fragile
    flattened_routes = routes.reshape(routes.shape[0], routes.shape[1], -1)
    sorted_routes = flattened_routes.sort(dim=-1).values
    duplicate_count = (
        sorted_routes[..., 1:] == sorted_routes[..., :-1]
    ).sum(dim=-1)
    route_policy_names = (
        "v11_token_hash",
        *(f"hash_seed_offset_{offset}" for offset in offsets),
    )
    fixed_policy_rows = [
        {
            "name": name,
            "mean_loss": float(candidate_losses[:, index].mean().item()),
            "loss_delta_vs_v11": float(
                candidate_losses[:, index].mean().item() - baseline.mean().item()
            ),
        }
        for index, name in enumerate(route_policy_names)
    ]
    selection_counts = torch.bincount(
        best_index,
        minlength=len(route_policy_names),
    )
    unique_experts = torch.unique(routes)

    def subset_row(mask: torch.Tensor) -> dict[str, Any]:
        if not bool(mask.any()):
            return {"token_count": 0}
        subset_regret = regret[mask]
        return {
            "token_count": int(mask.sum().item()),
            "baseline_mean_loss": float(baseline[mask].mean().item()),
            "mean_oracle_loss_improvement": float(subset_regret.mean().item()),
            "fraction_any_improvement": float((subset_regret > 0.0).float().mean()),
            "fraction_regret_005": float((subset_regret >= 0.05).float().mean()),
        }

    return {
        "surface": "marulho_counterfactual_route_source_audit.v1",
        "evaluated_token_count": evaluated_tokens,
        "sequence_length": int(batches[0].input_ids.shape[1]),
        "batch_count": len(batches),
        "batch_size": int(batches[0].input_ids.shape[0]),
        "baseline_mean_loss": float(baseline.mean().item()),
        "oracle_mean_loss": float(best_loss.mean().item()),
        "mean_oracle_loss_improvement": float(regret.mean().item()),
        "fraction_any_improvement": float(improved.float().mean().item()),
        "fraction_regret_001": float((regret >= 0.01).float().mean().item()),
        "fraction_regret_005": float((regret >= 0.05).float().mean().item()),
        "fraction_regret_010": float((regret >= 0.10).float().mean().item()),
        "regret_quantiles": _quantiles(regret),
        "baseline_loss_quantiles": _quantiles(baseline),
        "confidence_split": {
            "median_baseline_loss_threshold": fragile_threshold,
            "confident": subset_row(confident),
            "fragile": subset_row(fragile),
        },
        "fixed_route_policies": fixed_policy_rows,
        "oracle_route_selection": [
            {
                "name": name,
                "selected_token_count": int(selection_counts[index].item()),
                "selected_token_fraction": float(
                    selection_counts[index].item() / max(1, evaluated_tokens)
                ),
            }
            for index, name in enumerate(route_policy_names)
        ],
        "routing": {
            "route_policy_names": list(route_policy_names),
            "active_experts_per_token_per_policy": int(
                layer.routing_heads * layer.experts_per_head
            ),
            "equal_active_compute": True,
            "unique_experts_across_policies": int(unique_experts.numel()),
            "expert_pool_size": int(layer.expert_pool_size),
            "expert_pool_usage_fraction": float(
                unique_experts.numel() / layer.expert_pool_size
            ),
            "mean_duplicate_experts_within_policy_per_token": float(
                duplicate_count.float().mean().item()
            ),
            "alternative_seed_offsets": list(offsets),
        },
        "forced_baseline_max_logit_delta": forced_baseline_max_logit_delta,
        "parameter_sha256_before": before_hash,
        "parameter_sha256_after": after_hash,
        "parameters_unchanged": before_hash == after_hash,
        "anti_cheat": {
            "prediction_routes_use_targets": False,
            "targets_score_completed_routes_only": True,
            "oracle_selection_uses_labels": True,
            "oracle_selection_promotable": False,
            "model_or_router_updates": False,
            "external_llm_used": False,
        },
    }


def _source_batches(
    path: str | Path,
    tokenizer,
    *,
    config: CounterfactualRouteAuditConfig,
) -> tuple[tuple[LanguageBatch, ...], dict[str, Any]]:
    text, selection = sample_corpus_ranges(
        path,
        byte_budget=int(config.sample_bytes_per_source),
        range_count=int(config.sample_range_count),
    )
    split = build_language_model_splits(
        [text],
        tokenizer,
        sequence_length=int(config.sequence_length),
        eval_fraction=0.50,
        stride=int(config.sequence_length),
        batch_size=int(config.batch_size),
        max_train_batches=1,
        max_eval_batches=int(config.eval_batches_per_source),
        window_selection="stratified",
    )
    batches = full_sized_batches(split.eval, batch_size=int(config.batch_size))
    return batches, {
        **selection,
        "evaluated_batch_count": len(batches),
        "evaluated_token_count": sum(
            int(batch.target_ids[:, -1].numel()) for batch in batches
        ),
        "only_final_token_route_changed_per_sequence": True,
    }


def run_counterfactual_route_audit(
    *,
    checkpoint_path: str | Path,
    eval_corpus_paths: Sequence[str | Path],
    output_path: str | Path,
    config: CounterfactualRouteAuditConfig = CounterfactualRouteAuditConfig(),
    device: str = "auto",
) -> dict[str, Any]:
    if len(eval_corpus_paths) < 1:
        raise ValueError("At least one evaluation corpus is required")
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
    if model.hashed_config.mode != "token_hash":
        raise ValueError("Counterfactual audit requires V11 token_hash mode")
    if int(metadata.get("processed_tokens") or 0) < 251_000_000:
        raise ValueError("Counterfactual audit requires the 251M V11 checkpoint")
    if metadata.get("external_llm_used") is not False:
        raise ValueError("Counterfactual audit checkpoint must be MARULHO-owned")
    model = model.to(resolved).eval()
    started = time.perf_counter()
    source_rows: list[dict[str, Any]] = []
    for path in eval_corpus_paths:
        batches, selection = _source_batches(path, tokenizer, config=config)
        audit = _audit_batches(
            model,
            batches,
            alternative_seed_offsets=config.alternative_seed_offsets,
            precision=config.precision,
        )
        source_rows.append(
            {
                "path": str(path),
                "selection": selection,
                "audit": audit,
            }
        )

    token_count = sum(row["audit"]["evaluated_token_count"] for row in source_rows)
    mean_improvement = sum(
        row["audit"]["mean_oracle_loss_improvement"]
        * row["audit"]["evaluated_token_count"]
        for row in source_rows
    ) / max(1, token_count)
    fraction_regret_005 = sum(
        row["audit"]["fraction_regret_005"]
        * row["audit"]["evaluated_token_count"]
        for row in source_rows
    ) / max(1, token_count)
    max_parity_delta = max(
        row["audit"]["forced_baseline_max_logit_delta"] for row in source_rows
    )
    parameters_unchanged = all(
        row["audit"]["parameters_unchanged"] for row in source_rows
    )
    decision = counterfactual_route_decision(
        mean_oracle_loss_improvement=mean_improvement,
        fraction_regret_005=fraction_regret_005,
        parameters_unchanged=parameters_unchanged,
        forced_baseline_max_logit_delta=max_parity_delta,
        forced_baseline_logit_tolerance=config.forced_baseline_logit_tolerance,
        minimum_mean_oracle_loss_improvement=(
            config.minimum_mean_oracle_loss_improvement
        ),
        minimum_fraction_regret_005=config.minimum_fraction_regret_005,
    )
    report = {
        "artifact_kind": ARTIFACT_KIND,
        "surface": SURFACE,
        "owned_by_marulho": True,
        "external_llm_used": False,
        "configuration": asdict(config),
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": sha256_file(checkpoint),
            "processed_tokens": int(metadata["processed_tokens"]),
            "decision": metadata["decision"],
            "tokenizer_hash": tokenizer.vocabulary_hash(),
        },
        "sources": source_rows,
        "combined": {
            "evaluated_token_count": token_count,
            "mean_oracle_loss_improvement": mean_improvement,
            "fraction_regret_005": fraction_regret_005,
            "forced_baseline_max_logit_delta": max_parity_delta,
            "parameters_unchanged": parameters_unchanged,
        },
        "decision": decision,
        "decision_contract": {
            "required_mean_oracle_loss_improvement": (
                config.minimum_mean_oracle_loss_improvement
            ),
            "required_fraction_regret_005": config.minimum_fraction_regret_005,
            "oracle_gain_only_admits_gate_training": True,
            "oracle_gain_does_not_promote_model": True,
        },
        "anti_cheat": {
            "labels_metrics_only": True,
            "prediction_routes_use_targets": False,
            "oracle_route_not_available_at_inference": True,
            "model_mutation": False,
            "checkpoint_written": False,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json_report_with_readme(
        output_path,
        report,
        title="MARULHO V11 Counterfactual Route-Regret Audit",
    )
    print(
        f"[v11-counterfactual] decision {decision}; mean oracle gain "
        f"{mean_improvement:.4f}, regret>=0.05 {fraction_regret_005:.3f}",
        flush=True,
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--eval-corpus", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--eval-batches-per-source", type=int, default=16)
    parser.add_argument("--sample-mib-per-source", type=int, default=32)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    run_counterfactual_route_audit(
        checkpoint_path=args.checkpoint,
        eval_corpus_paths=tuple(args.eval_corpus),
        output_path=args.output,
        config=CounterfactualRouteAuditConfig(
            eval_batches_per_source=int(args.eval_batches_per_source),
            sample_bytes_per_source=int(args.sample_mib_per_source) * 1024 * 1024,
        ),
        device=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
