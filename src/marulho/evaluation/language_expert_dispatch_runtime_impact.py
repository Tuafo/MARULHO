"""Measure routed LM forward impact from block-sparse expert dispatch Triton."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import os
from pathlib import Path
import time
from typing import Any, Iterator, Mapping

import torch

from marulho.core.language_expert_dispatch_triton import (
    language_expert_dispatch_triton_stats,
    language_expert_dispatch_triton_stats_delta,
    reset_language_expert_dispatch_triton_stats,
)
from marulho.core.language_route_topk_triton import (
    language_route_topk_triton_stats,
    language_route_topk_triton_stats_delta,
    reset_language_route_topk_triton_stats,
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


SURFACE = "marulho_language_expert_dispatch_runtime_impact.v1"
ARTIFACT_KIND = "marulho_language_expert_dispatch_runtime_impact"
_EXPERT_DISPATCH_MIN_TOKENS_ENV = (
    "MARULHO_LANGUAGE_EXPERT_DISPATCH_TRITON_MIN_TOKENS"
)
_ROUTE_TOPK_MIN_ROWS_ENV = "MARULHO_LANGUAGE_ROUTE_TOPK_TRITON_MIN_ROWS"


@dataclass(frozen=True)
class ExpertDispatchRuntimeImpactConfig:
    vocab_size: int = 524288
    embedding_dim: int = 64
    state_dim: int = 128
    expert_count: int = 16
    active_expert_count: int = 4
    route_candidate_count: int = 8
    expert_hidden_dim: int = 192
    adaptive_timestep_budget: int = 1
    sequence_length: int = 64
    batch_size: int = 16
    warmup_steps: int = 5
    repeats: int = 50
    fallback_min_tokens: int = 1_000_000_000
    triton_min_tokens: int = 1
    route_topk_min_rows: int = 1
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


@contextmanager
def _env_int(name: str, value: int) -> Iterator[None]:
    previous = os.environ.get(name)
    os.environ[name] = str(max(1, int(value)))
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


def _model_config(
    config: ExpertDispatchRuntimeImpactConfig,
    tokenizer: ByteLevelLanguageTokenizer,
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
        generation_vocab_size=int(tokenizer.vocab_size),
    )


def _build_batch(
    tokenizer: ByteLevelLanguageTokenizer,
    config: ExpertDispatchRuntimeImpactConfig,
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
    config: ExpertDispatchRuntimeImpactConfig,
    tokenizer: ByteLevelLanguageTokenizer,
) -> dict[str, torch.Tensor]:
    torch.manual_seed(int(config.seed))
    base_model = MarulhoLanguageModel(_model_config(config, tokenizer))
    return {
        key: value.detach().clone()
        for key, value in base_model.state_dict().items()
    }


def _forward_once(
    model: MarulhoLanguageModel,
    batch: LanguageBatch,
    *,
    collect_telemetry: bool,
) -> Mapping[str, Any]:
    return model.forward(
        batch.input_ids.to(model.device),
        collect_telemetry=collect_telemetry,
        assume_no_sleeping_experts=bool(model.routed_experts.enabled),
        decode_vocab_only=True,
    )


def _zero_expert_stats(after: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "triton_available": after.get("triton_available"),
        "triton_forward_calls": 0,
        "triton_forward_elements": 0,
        "torch_fallback_calls": 0,
        "torch_fallback_elements": 0,
        "triton_failure_count": 0,
    }


def _zero_route_topk_stats(after: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "triton_available": after.get("triton_available"),
        "triton_forward_calls": 0,
        "triton_forward_elements": 0,
        "torch_fallback_calls": 0,
        "torch_fallback_elements": 0,
        "triton_failure_count": 0,
    }


def _run_arm(
    name: str,
    *,
    expert_dispatch_min_tokens: int,
    base_state: Mapping[str, torch.Tensor],
    batch: LanguageBatch,
    config: ExpertDispatchRuntimeImpactConfig,
    tokenizer: ByteLevelLanguageTokenizer,
    device: torch.device,
) -> tuple[dict[str, Any], torch.Tensor | None]:
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    model_config = _model_config(config, tokenizer)
    model: MarulhoLanguageModel | None = None
    try:
        model = MarulhoLanguageModel(model_config).to(device)
        model.load_state_dict(dict(base_state))
        model.eval()
        with (
            _env_int(
                _EXPERT_DISPATCH_MIN_TOKENS_ENV,
                int(expert_dispatch_min_tokens),
            ),
            _env_int(_ROUTE_TOPK_MIN_ROWS_ENV, int(config.route_topk_min_rows)),
            torch.inference_mode(),
        ):
            telemetry_result = _forward_once(
                model,
                batch,
                collect_telemetry=True,
            )
            for _ in range(max(0, int(config.warmup_steps))):
                _forward_once(model, batch, collect_telemetry=False)
            cuda_synchronized_before_timing_start = _sync_if_cuda(device)
            reset_language_expert_dispatch_triton_stats()
            reset_language_route_topk_triton_stats()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            last_result: Mapping[str, Any] | None = None
            for _ in range(max(1, int(config.repeats))):
                last_result = _forward_once(model, batch, collect_telemetry=False)
            cuda_synchronized_before_timing_stop = _sync_if_cuda(device)
            elapsed = max(0.0, time.perf_counter() - started)
            expert_stats_after = language_expert_dispatch_triton_stats()
            route_topk_stats_after = language_route_topk_triton_stats()
        token_count = int(batch.input_ids.numel()) * max(1, int(config.repeats))
        logits = (
            None
            if last_result is None
            else last_result["logits"].detach().float().cpu()
        )
        routing = (
            telemetry_result.get("telemetry", {}).get("routing", {})
            if isinstance(telemetry_result.get("telemetry"), Mapping)
            else {}
        )
        expert_stats_delta = language_expert_dispatch_triton_stats_delta(
            _zero_expert_stats(expert_stats_after),
            expert_stats_after,
        )
        route_topk_stats_delta = language_route_topk_triton_stats_delta(
            _zero_route_topk_stats(route_topk_stats_after),
            route_topk_stats_after,
        )
        return {
            "surface": "marulho_language_expert_dispatch_runtime_arm.v1",
            "name": name,
            "success": True,
            "failure_reason": None,
            "expert_dispatch_min_tokens": int(expert_dispatch_min_tokens),
            "route_topk_min_rows": int(config.route_topk_min_rows),
            "model_config": asdict(model_config),
            "warmup_steps": int(config.warmup_steps),
            "measured_steps": int(config.repeats),
            "tokens_per_forward": int(batch.input_ids.numel()),
            "token_count": int(token_count),
            "elapsed_seconds": elapsed,
            "tokens_per_second": (
                float(token_count) / elapsed if elapsed > 0.0 else 0.0
            ),
            "route_selection_backend": str(
                routing.get("route_selection_backend", "unknown")
            ),
            "expert_dispatch_backend": str(
                routing.get("expert_dispatch_backend", "unknown")
            ),
            "route_candidate_count": int(routing.get("route_candidate_count", 0) or 0),
            "active_expert_count_per_token": int(
                routing.get("active_expert_count_per_token", 0) or 0
            ),
            "candidate_rows_scored": int(routing.get("candidate_rows_scored", 0) or 0),
            "active_parameters_per_token": int(
                routing.get("active_parameters_per_token", 0) or 0
            ),
            "runs_all_columns": bool(routing.get("runs_all_columns", False)),
            "fallback_reason": routing.get("fallback_reason"),
            "expert_dispatch_triton_stats_delta": expert_stats_delta,
            "route_topk_triton_stats_delta": route_topk_stats_delta,
            "cuda_synchronized_before_timing_start": bool(
                cuda_synchronized_before_timing_start
            ),
            "cuda_synchronized_before_timing_stop": bool(
                cuda_synchronized_before_timing_stop
            ),
            "device": str(device),
            "cuda_memory": _cuda_memory(device),
        }, logits
    except RuntimeError as exc:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return {
            "surface": "marulho_language_expert_dispatch_runtime_arm.v1",
            "name": name,
            "success": False,
            "failure_reason": f"{type(exc).__name__}: {exc}",
            "expert_dispatch_min_tokens": int(expert_dispatch_min_tokens),
            "route_topk_min_rows": int(config.route_topk_min_rows),
            "model_config": asdict(model_config),
            "warmup_steps": int(config.warmup_steps),
            "measured_steps": int(config.repeats),
            "tokens_per_forward": int(batch.input_ids.numel()),
            "token_count": 0,
            "elapsed_seconds": 0.0,
            "tokens_per_second": 0.0,
            "route_selection_backend": "failed",
            "expert_dispatch_backend": "failed",
            "route_candidate_count": 0,
            "active_expert_count_per_token": 0,
            "candidate_rows_scored": 0,
            "active_parameters_per_token": 0,
            "runs_all_columns": False,
            "fallback_reason": f"{type(exc).__name__}: {exc}",
            "expert_dispatch_triton_stats_delta": {},
            "route_topk_triton_stats_delta": {},
            "device": str(device),
            "cuda_memory": _cuda_memory(device),
        }, None
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _comparison(
    fallback: dict[str, Any],
    triton_arm: dict[str, Any],
    *,
    fallback_logits: torch.Tensor | None,
    triton_logits: torch.Tensor | None,
) -> dict[str, Any]:
    fallback_success = bool(fallback.get("success"))
    triton_success = bool(triton_arm.get("success"))
    fallback_tps = float(fallback.get("tokens_per_second", 0.0) or 0.0)
    triton_tps = float(triton_arm.get("tokens_per_second", 0.0) or 0.0)
    speedup = (
        triton_tps / fallback_tps
        if fallback_success and triton_success and fallback_tps > 0.0
        else None
    )
    max_abs_error: float | None = None
    max_rel_error: float | None = None
    parity_passed = False
    if fallback_logits is not None and triton_logits is not None:
        diff = (triton_logits - fallback_logits).abs()
        max_abs_error = float(diff.max().item())
        denominator = fallback_logits.abs().clamp_min(1e-8)
        max_rel_error = float((diff / denominator).max().item())
        parity_passed = bool(max_abs_error <= 1e-4 or max_rel_error <= 1e-4)
    triton_stats = triton_arm.get("expert_dispatch_triton_stats_delta", {})
    fallback_stats = fallback.get("expert_dispatch_triton_stats_delta", {})
    triton_kernel_used = bool(triton_stats.get("triton_kernel_used", False))
    fallback_used_torch = int(fallback_stats.get("torch_fallback_calls", 0) or 0) > 0
    route_topk_stats = triton_arm.get("route_topk_triton_stats_delta", {})
    route_topk_triton_used = bool(route_topk_stats.get("triton_kernel_used", False))
    if triton_success and fallback_success and triton_kernel_used and fallback_used_torch:
        evidence_status = "measured_triton_vs_torch_expert_dispatch_forward"
    elif triton_success and fallback_success:
        evidence_status = "measured_without_expert_dispatch_triton_use"
    else:
        evidence_status = "runtime_impact_measurement_failed"
    return {
        "surface": "marulho_language_expert_dispatch_runtime_comparison.v1",
        "fallback_success": fallback_success,
        "triton_success": triton_success,
        "fallback_tokens_per_second": fallback_tps,
        "triton_tokens_per_second": triton_tps,
        "triton_vs_fallback_tokens_per_second_ratio": speedup,
        "fallback_expert_dispatch_torch_used": fallback_used_torch,
        "triton_expert_dispatch_kernel_used": triton_kernel_used,
        "route_topk_held_constant_triton_used": route_topk_triton_used,
        "max_abs_logit_error": max_abs_error,
        "max_rel_logit_error": max_rel_error,
        "parity_passed": parity_passed,
        "evidence_status": evidence_status,
    }


def run_language_expert_dispatch_runtime_impact(
    *,
    output_path: str | Path,
    config: ExpertDispatchRuntimeImpactConfig | None = None,
) -> dict[str, Any]:
    cfg = config or ExpertDispatchRuntimeImpactConfig()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = ByteLevelLanguageTokenizer()
    if int(cfg.vocab_size) < int(tokenizer.vocab_size):
        raise ValueError("vocab_size must be at least the tokenizer vocab size")
    device = _resolve_device(str(cfg.device))
    batch, split_report = _build_batch(tokenizer, cfg, device=device)
    base_state = _clone_base_state(cfg, tokenizer)
    fallback_report, fallback_logits = _run_arm(
        "torch_expert_dispatch_policy_fallback",
        expert_dispatch_min_tokens=int(cfg.fallback_min_tokens),
        base_state=base_state,
        batch=batch,
        config=cfg,
        tokenizer=tokenizer,
        device=device,
    )
    triton_report, triton_logits = _run_arm(
        "triton_expert_dispatch_enabled",
        expert_dispatch_min_tokens=int(cfg.triton_min_tokens),
        base_state=base_state,
        batch=batch,
        config=cfg,
        tokenizer=tokenizer,
        device=device,
    )
    comparison = _comparison(
        fallback_report,
        triton_report,
        fallback_logits=fallback_logits,
        triton_logits=triton_logits,
    )
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
        "generation_vocab_size": int(tokenizer.vocab_size),
        "padded_vocab_rows": int(cfg.vocab_size) - int(tokenizer.vocab_size),
        "batch": {
            "sequence_length": int(cfg.sequence_length),
            "batch_size": int(cfg.batch_size),
            "tokens_per_forward": int(batch.input_ids.numel()),
            "input_device": str(batch.input_ids.device),
        },
        "split": split_report,
        "arms": {
            "torch_expert_dispatch_policy_fallback": fallback_report,
            "triton_expert_dispatch_enabled": triton_report,
        },
        "comparison": comparison,
        "review": {
            "complete_forward_runtime_impact": True,
            "includes_embedding_state_block_routing_dispatch_and_decode_head": True,
            "not_kernel_microbench_only": True,
            "mutates_model_state": False,
            "gradient_training_unchanged": True,
            "one_token_streaming_policy_unchanged": True,
            "route_topk_policy_held_constant": True,
            "promotes_hot_path": False,
            "promotes_runtime_claim": False,
            "next_experiment": (
                "keep expert-dispatch Triton behind row-count policy until repeated "
                "full-forward and sustained generation evidence beat the torch "
                "selected-expert path"
            ),
        },
        "promotion_gate": {
            "runtime_impact_available": bool(
                comparison["fallback_success"] and comparison["triton_success"]
            ),
            "expert_dispatch_triton_used": bool(
                comparison["triton_expert_dispatch_kernel_used"]
            ),
            "route_topk_held_constant": True,
            "parity_passed": bool(comparison["parity_passed"]),
            "complete_runtime_impact_available": bool(
                comparison["fallback_success"] and comparison["triton_success"]
            ),
            "promotes_hot_path": False,
            "promotes_runtime_claim": False,
        },
    }
    write_json_report_with_readme(output, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vocab-size", type=int, default=524288)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--state-dim", type=int, default=128)
    parser.add_argument("--expert-count", type=int, default=16)
    parser.add_argument("--active-expert-count", type=int, default=4)
    parser.add_argument("--route-candidate-count", type=int, default=8)
    parser.add_argument("--expert-hidden-dim", type=int, default=192)
    parser.add_argument("--adaptive-timestep-budget", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--fallback-min-tokens", type=int, default=1_000_000_000)
    parser.add_argument("--triton-min-tokens", type=int, default=1)
    parser.add_argument("--route-topk-min-rows", type=int, default=1)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    config = ExpertDispatchRuntimeImpactConfig(
        vocab_size=args.vocab_size,
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
        fallback_min_tokens=args.fallback_min_tokens,
        triton_min_tokens=args.triton_min_tokens,
        route_topk_min_rows=args.route_topk_min_rows,
        device=args.device,
    )
    report = run_language_expert_dispatch_runtime_impact(
        output_path=args.output,
        config=config,
    )
    comparison = report["comparison"]
    print(
        "wrote "
        f"{args.output} triton_tps={comparison['triton_tokens_per_second']:.3f} "
        f"fallback_tps={comparison['fallback_tokens_per_second']:.3f} "
        f"ratio={comparison['triton_vs_fallback_tokens_per_second_ratio']}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
