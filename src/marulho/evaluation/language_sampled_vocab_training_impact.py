"""Measure full-model sampled-vocab training impact for the MARULHO LM head."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
import time
from typing import Any, Mapping

import torch

from marulho.core.language_sampled_vocab_ce_triton import (
    language_sampled_vocab_ce_triton_stats,
    language_sampled_vocab_ce_triton_stats_delta,
)
from marulho.data.language_tokenizer import ByteLevelLanguageTokenizer
from marulho.evaluation.language_training_experiment import DEFAULT_CORPUS
from marulho.reporting.readme_reports import write_json_report_with_readme
from marulho.training.language_model import (
    LanguageBatch,
    LanguageModelConfig,
    MarulhoLanguageModel,
    build_language_model_splits,
)


SURFACE = "marulho_language_sampled_vocab_training_impact.v1"
ARTIFACT_KIND = "marulho_language_sampled_vocab_training_impact"


@dataclass(frozen=True)
class SampledVocabTrainingImpactConfig:
    vocab_size: int = 524288
    sampled_vocab_size: int = 1024
    embedding_dim: int = 64
    state_dim: int = 128
    expert_count: int = 16
    active_expert_count: int = 4
    route_candidate_count: int = 8
    expert_hidden_dim: int = 192
    adaptive_timestep_budget: int = 1
    sequence_length: int = 64
    batch_size: int = 4
    warmup_steps: int = 1
    repeats: int = 3
    learning_rate: float = 1e-3
    max_grad_norm: float = 1.0
    run_dense_baseline: bool = True
    device: str = "auto"
    seed: int = 20260704


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA device was requested but torch.cuda.is_available() is false")
    return resolved


def _sync_if_cuda(device: torch.device) -> bool:
    if device.type != "cuda":
        return False
    torch.cuda.synchronize(device)
    return True


def _cuda_memory(device: torch.device) -> dict[str, Any]:
    if device.type != "cuda":
        return {
            "cuda_available": bool(torch.cuda.is_available()),
            "device": str(device),
            "allocated_mib": 0.0,
            "reserved_mib": 0.0,
            "peak_allocated_mib": 0.0,
            "peak_reserved_mib": 0.0,
        }
    return {
        "cuda_available": bool(torch.cuda.is_available()),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "allocated_mib": float(torch.cuda.memory_allocated(device) / (1024.0 * 1024.0)),
        "reserved_mib": float(torch.cuda.memory_reserved(device) / (1024.0 * 1024.0)),
        "peak_allocated_mib": float(
            torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
        ),
        "peak_reserved_mib": float(
            torch.cuda.max_memory_reserved(device) / (1024.0 * 1024.0)
        ),
    }


def _model_config(
    config: SampledVocabTrainingImpactConfig,
    *,
    sampled_vocab_size: int,
    sparse_vocab_gradients: bool = False,
) -> LanguageModelConfig:
    return LanguageModelConfig(
        vocab_size=int(config.vocab_size),
        embedding_dim=int(config.embedding_dim),
        state_dim=int(config.state_dim),
        adaptive_timestep_budget=int(config.adaptive_timestep_budget),
        expert_count=int(config.expert_count),
        active_expert_count=int(config.active_expert_count),
        route_candidate_count=int(config.route_candidate_count),
        expert_hidden_dim=int(config.expert_hidden_dim),
        sampled_vocab_size=int(sampled_vocab_size),
        sampled_vocab_sparse_lm_head_gradient=bool(
            sparse_vocab_gradients and sampled_vocab_size > 0
        ),
        sparse_token_embedding_gradients=bool(
            sparse_vocab_gradients and sampled_vocab_size > 0
        ),
    )


def _build_batch(
    tokenizer: ByteLevelLanguageTokenizer,
    config: SampledVocabTrainingImpactConfig,
    *,
    device: torch.device,
) -> tuple[LanguageBatch, dict[str, Any]]:
    split = build_language_model_splits(
        [DEFAULT_CORPUS],
        tokenizer,
        sequence_length=int(config.sequence_length),
        eval_fraction=0.20,
        stride=int(config.sequence_length),
        batch_size=int(config.batch_size),
        device=device,
    )
    return split.train[0], split.report


def _clone_base_state(
    config: SampledVocabTrainingImpactConfig,
) -> dict[str, torch.Tensor]:
    torch.manual_seed(int(config.seed))
    base_model = MarulhoLanguageModel(_model_config(config, sampled_vocab_size=0))
    return {
        key: value.detach().clone()
        for key, value in base_model.state_dict().items()
    }


def _optimizer_policy(
    model: MarulhoLanguageModel,
    *,
    config: SampledVocabTrainingImpactConfig,
    sparse_vocab_optimizer: bool,
) -> tuple[list[torch.optim.Optimizer], str]:
    if not bool(sparse_vocab_optimizer):
        return [
            torch.optim.AdamW(model.parameters(), lr=float(config.learning_rate))
        ], "AdamW_all_parameters"

    sparse_names = {"token_embedding.weight", "lm_head.weight"}
    sparse_params: list[torch.nn.Parameter] = []
    dense_params: list[torch.nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name in sparse_names:
            sparse_params.append(parameter)
        else:
            dense_params.append(parameter)
    optimizers: list[torch.optim.Optimizer] = []
    if dense_params:
        optimizers.append(torch.optim.AdamW(dense_params, lr=float(config.learning_rate)))
    if sparse_params:
        optimizers.append(torch.optim.SparseAdam(sparse_params, lr=float(config.learning_rate)))
    return optimizers, "AdamW_dense_core_plus_SparseAdam_vocab_rows"


def _all_parameters(model: MarulhoLanguageModel) -> list[torch.nn.Parameter]:
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def _clip_grad_norm_sparse_aware(
    parameters: list[torch.nn.Parameter],
    *,
    max_norm: float,
    device: torch.device,
) -> torch.Tensor:
    total_sq = torch.zeros((), device=device, dtype=torch.float32)
    for parameter in parameters:
        grad = parameter.grad
        if grad is None:
            continue
        values = grad.coalesce().values() if grad.is_sparse else grad
        total_sq = total_sq + values.detach().float().pow(2).sum()
    total_norm = torch.sqrt(total_sq)
    limit = float(max_norm)
    if limit > 0.0:
        clip_coef = torch.clamp(
            torch.tensor(limit, device=device, dtype=torch.float32)
            / (total_norm + 1e-6),
            max=1.0,
        )
        for parameter in parameters:
            grad = parameter.grad
            if grad is None:
                continue
            if grad.is_sparse:
                grad = grad.coalesce()
                grad.values().mul_(clip_coef)
                parameter.grad = grad
            else:
                grad.mul_(clip_coef)
    return total_norm


def _run_training_steps(
    model: MarulhoLanguageModel,
    optimizers: list[torch.optim.Optimizer],
    batch: LanguageBatch,
    *,
    config: SampledVocabTrainingImpactConfig,
    step_count: int,
) -> tuple[int, torch.Tensor | None, torch.Tensor | None, Mapping[str, Any]]:
    token_count = 0
    last_loss: torch.Tensor | None = None
    last_grad_norm: torch.Tensor | None = None
    last_result: Mapping[str, Any] = {}
    assume_no_sleeping = bool(model.routed_experts.enabled)
    parameters = _all_parameters(model)
    for _step in range(max(0, int(step_count))):
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        result = model.next_token_loss(
            batch.input_ids.to(model.device),
            batch.target_ids.to(model.device),
            collect_telemetry=False,
            assume_no_sleeping_experts=assume_no_sleeping,
        )
        loss = result["loss"]
        loss.backward()
        grad_norm = _clip_grad_norm_sparse_aware(
            parameters,
            max_norm=float(config.max_grad_norm),
            device=model.device,
        )
        for optimizer in optimizers:
            optimizer.step()
        token_count += int(batch.target_ids.numel())
        last_loss = loss.detach()
        last_grad_norm = (
            grad_norm.detach()
            if isinstance(grad_norm, torch.Tensor)
            else torch.tensor(float(grad_norm), device=model.device)
        )
        last_result = result
    return token_count, last_loss, last_grad_norm, last_result


def _tensor_scalar(value: torch.Tensor | None) -> float | None:
    if value is None:
        return None
    return float(value.detach().cpu().item())


def _run_arm(
    name: str,
    *,
    sampled_vocab_size: int,
    base_state: Mapping[str, torch.Tensor],
    batch: LanguageBatch,
    config: SampledVocabTrainingImpactConfig,
    device: torch.device,
) -> dict[str, Any]:
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    model_config = _model_config(config, sampled_vocab_size=sampled_vocab_size)
    model: MarulhoLanguageModel | None = None
    optimizers: list[torch.optim.Optimizer] = []
    try:
        sparse_vocab_optimizer = sampled_vocab_size > 0
        model_config = _model_config(
            config,
            sampled_vocab_size=sampled_vocab_size,
            sparse_vocab_gradients=sparse_vocab_optimizer,
        )
        model = MarulhoLanguageModel(model_config).to(device)
        model.load_state_dict(dict(base_state))
        model.train()
        optimizers, optimizer_policy = _optimizer_policy(
            model,
            config=config,
            sparse_vocab_optimizer=sparse_vocab_optimizer,
        )
        warmup_tokens, _warmup_loss, _warmup_grad, _warmup_result = _run_training_steps(
            model,
            optimizers,
            batch,
            config=config,
            step_count=int(config.warmup_steps),
        )
        sampled_vocab_ce_stats_before = language_sampled_vocab_ce_triton_stats()
        cuda_synchronized_before_timing_start = _sync_if_cuda(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        token_count, last_loss, last_grad_norm, last_result = _run_training_steps(
            model,
            optimizers,
            batch,
            config=config,
            step_count=int(config.repeats),
        )
        cuda_synchronized_before_timing_stop = _sync_if_cuda(device)
        sampled_vocab_ce_stats_delta = language_sampled_vocab_ce_triton_stats_delta(
            sampled_vocab_ce_stats_before,
            language_sampled_vocab_ce_triton_stats(),
        )
        elapsed = max(0.0, time.perf_counter() - started)
        memory = _cuda_memory(device)
        loss_evidence = dict(last_result.get("loss_evidence", {}))
        return {
            "surface": "marulho_language_sampled_vocab_training_arm.v1",
            "name": name,
            "success": True,
            "failure_reason": None,
            "model_config": asdict(model_config),
            "optimizer_policy": optimizer_policy,
            "warmup_steps": int(config.warmup_steps),
            "warmup_tokens": int(warmup_tokens),
            "measured_steps": int(config.repeats),
            "token_count": int(token_count),
            "tokens_per_optimizer_step": int(batch.target_ids.numel()),
            "elapsed_seconds": elapsed,
            "tokens_per_second": (
                float(token_count) / elapsed if elapsed > 0.0 else 0.0
            ),
            "loss": _tensor_scalar(last_loss),
            "gradient_norm": _tensor_scalar(last_grad_norm),
            "loss_kind": str(last_result.get("loss_kind")),
            "loss_evidence": loss_evidence,
            "sampled_vocab_ce_triton_stats_delta": sampled_vocab_ce_stats_delta,
            "full_vocab_logits_materialized": bool(
                loss_evidence.get("full_vocab_logits_materialized", True)
            ),
            "sampled_vocab_training": bool(
                loss_evidence.get("sampled_vocab_training", False)
            ),
            "cuda_synchronized_before_timing_start": bool(
                cuda_synchronized_before_timing_start
            ),
            "cuda_synchronized_before_timing_stop": bool(
                cuda_synchronized_before_timing_stop
            ),
            "device": str(device),
            "cuda_memory": memory,
        }
    except RuntimeError as exc:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return {
            "surface": "marulho_language_sampled_vocab_training_arm.v1",
            "name": name,
            "success": False,
            "failure_reason": f"{type(exc).__name__}: {exc}",
            "model_config": asdict(model_config),
            "optimizer_policy": (
                "AdamW_dense_core_plus_SparseAdam_vocab_rows"
                if sampled_vocab_size > 0
                else "AdamW_all_parameters"
            ),
            "warmup_steps": int(config.warmup_steps),
            "measured_steps": int(config.repeats),
            "token_count": 0,
            "elapsed_seconds": 0.0,
            "tokens_per_second": 0.0,
            "loss": None,
            "gradient_norm": None,
            "loss_kind": None,
            "loss_evidence": {},
            "sampled_vocab_ce_triton_stats_delta": {},
            "full_vocab_logits_materialized": sampled_vocab_size <= 0,
            "sampled_vocab_training": sampled_vocab_size > 0,
            "device": str(device),
            "cuda_memory": _cuda_memory(device),
        }
    finally:
        del optimizers
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _comparison(
    dense: dict[str, Any] | None,
    sampled: dict[str, Any],
) -> dict[str, Any]:
    dense_success = bool(dense and dense.get("success"))
    sampled_success = bool(sampled.get("success"))
    dense_tps = float(dense.get("tokens_per_second", 0.0)) if dense else 0.0
    sampled_tps = float(sampled.get("tokens_per_second", 0.0) or 0.0)
    speedup = sampled_tps / dense_tps if dense_success and dense_tps > 0.0 else None
    dense_peak = (
        float(dense.get("cuda_memory", {}).get("peak_allocated_mib", 0.0))
        if dense
        else 0.0
    )
    sampled_peak = float(
        sampled.get("cuda_memory", {}).get("peak_allocated_mib", 0.0) or 0.0
    )
    peak_ratio = sampled_peak / dense_peak if dense_success and dense_peak > 0.0 else None
    if sampled_success and dense is None:
        scalability_evidence = "sampled_measured_dense_not_run"
    elif sampled_success and not dense_success:
        scalability_evidence = "sampled_succeeded_dense_failed"
    elif sampled_success and dense_success:
        scalability_evidence = "sampled_vs_dense_measured"
    else:
        scalability_evidence = "sampled_failed"
    return {
        "surface": "marulho_language_sampled_vocab_training_comparison.v1",
        "sampled_training_success": sampled_success,
        "dense_baseline_success": dense_success,
        "dense_failure_reason": dense.get("failure_reason") if dense else "not_run",
        "sampled_tokens_per_second": sampled_tps,
        "dense_tokens_per_second": dense_tps,
        "sampled_vs_dense_tokens_per_second_ratio": speedup,
        "sampled_peak_cuda_allocated_mib": sampled_peak,
        "dense_peak_cuda_allocated_mib": dense_peak,
        "sampled_vs_dense_peak_memory_ratio": peak_ratio,
        "scalability_evidence": scalability_evidence,
    }


def run_language_sampled_vocab_training_impact(
    *,
    output_path: str | Path,
    config: SampledVocabTrainingImpactConfig | None = None,
) -> dict[str, Any]:
    cfg = config or SampledVocabTrainingImpactConfig()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = ByteLevelLanguageTokenizer()
    if int(cfg.vocab_size) < int(tokenizer.vocab_size):
        raise ValueError("vocab_size must be at least the tokenizer vocab size")
    if int(cfg.sampled_vocab_size) <= 0:
        raise ValueError("sampled_vocab_size must be positive")
    if int(cfg.sampled_vocab_size) >= int(cfg.vocab_size):
        raise ValueError("sampled_vocab_size must be smaller than vocab_size")
    device = _resolve_device(str(cfg.device))
    batch, split_report = _build_batch(tokenizer, cfg, device=device)
    base_state = _clone_base_state(cfg)
    dense_report: dict[str, Any] | None = None
    if bool(cfg.run_dense_baseline):
        dense_report = _run_arm(
            "dense_full_vocab_cross_entropy",
            sampled_vocab_size=0,
            base_state=base_state,
            batch=batch,
            config=cfg,
            device=device,
        )
    sampled_report = _run_arm(
        "sampled_adaptive_vocab_cross_entropy",
        sampled_vocab_size=int(cfg.sampled_vocab_size),
        base_state=base_state,
        batch=batch,
        config=cfg,
        device=device,
    )
    comparison = _comparison(dense_report, sampled_report)
    report = {
        "artifact_kind": ARTIFACT_KIND,
        "surface": SURFACE,
        "output_path": str(output),
        "owned_by_marulho": True,
        "external_llm_used": False,
        "loads_external_checkpoint": False,
        "active_language_path": "marulho_lm_head",
        "config": asdict(cfg),
        "tokenizer": tokenizer.state_dict(),
        "model_vocab_size": int(cfg.vocab_size),
        "tokenizer_vocab_size": int(tokenizer.vocab_size),
        "padded_vocab_rows": int(cfg.vocab_size) - int(tokenizer.vocab_size),
        "batch": {
            "sequence_length": int(cfg.sequence_length),
            "batch_size": int(cfg.batch_size),
            "tokens_per_optimizer_step": int(batch.target_ids.numel()),
            "input_device": str(batch.input_ids.device),
            "target_device": str(batch.target_ids.device),
        },
        "split": split_report,
        "arms": {
            "dense_full_vocab": dense_report,
            "sampled_adaptive_vocab": sampled_report,
        },
        "comparison": comparison,
        "review": {
            "complete_training_step_impact": True,
            "includes_backward": bool(sampled_report.get("gradient_norm") is not None),
            "includes_optimizer_step": bool(sampled_report.get("success")),
            "not_kernel_microbench_only": True,
            "sampled_loss_avoids_full_vocab_logits": bool(
                sampled_report.get("success")
                and not sampled_report.get("full_vocab_logits_materialized", True)
            ),
            "dense_baseline_materializes_full_vocab_logits": bool(
                dense_report and dense_report.get("full_vocab_logits_materialized")
            ),
            "padded_vocab_generation_policy_reviewed": False,
            "promotes_runtime_claim": False,
            "promotes_generation_quality_claim": False,
            "next_experiment": (
                "profile batch16 optimizer/state-block cost and keep forced Triton "
                "sampled-vocab CE training disabled until complete-runtime evidence wins"
            ),
        },
    }
    write_json_report_with_readme(output, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vocab-size", type=int, default=524288)
    parser.add_argument("--sampled-vocab-size", type=int, default=1024)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--state-dim", type=int, default=128)
    parser.add_argument("--expert-count", type=int, default=16)
    parser.add_argument("--active-expert-count", type=int, default=4)
    parser.add_argument("--route-candidate-count", type=int, default=8)
    parser.add_argument("--expert-hidden-dim", type=int, default=192)
    parser.add_argument("--adaptive-timestep-budget", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--skip-dense-baseline", action="store_true")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    config = SampledVocabTrainingImpactConfig(
        vocab_size=args.vocab_size,
        sampled_vocab_size=args.sampled_vocab_size,
        embedding_dim=args.embedding_dim,
        state_dim=args.state_dim,
        expert_count=args.expert_count,
        active_expert_count=args.active_expert_count,
        route_candidate_count=args.route_candidate_count,
        expert_hidden_dim=args.expert_hidden_dim,
        adaptive_timestep_budget=args.adaptive_timestep_budget,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        warmup_steps=args.warmup_steps,
        repeats=args.repeats,
        learning_rate=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        run_dense_baseline=not bool(args.skip_dense_baseline),
        device=args.device,
    )
    report = run_language_sampled_vocab_training_impact(
        output_path=args.output,
        config=config,
    )
    comparison = report["comparison"]
    print(
        "wrote "
        f"{args.output} sampled_tps={comparison['sampled_tokens_per_second']:.3f} "
        f"dense_tps={comparison['dense_tokens_per_second']:.3f} "
        f"ratio={comparison['sampled_vs_dense_tokens_per_second_ratio']}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
