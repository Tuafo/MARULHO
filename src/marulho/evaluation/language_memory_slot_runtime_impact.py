"""Measure LM forward impact from bounded language memory-slot retrieval."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
import time
from typing import Any, Mapping

import torch

from marulho.data.language_tokenizer import ByteLevelLanguageTokenizer
from marulho.evaluation.language_training_experiment import DEFAULT_CORPUS
from marulho.reporting.readme_reports import write_json_report_with_readme
from marulho.training.language_model import (
    LanguageBatch,
    LanguageModelConfig,
    MarulhoLanguageModel,
    build_language_model_splits,
)


SURFACE = "marulho_language_memory_slot_runtime_impact.v1"
ARTIFACT_KIND = "marulho_language_memory_slot_runtime_impact"


@dataclass(frozen=True)
class MemorySlotRuntimeImpactConfig:
    vocab_size: int = 524288
    embedding_dim: int = 64
    state_dim: int = 128
    expert_count: int = 16
    active_expert_count: int = 4
    route_candidate_count: int = 8
    expert_hidden_dim: int = 192
    adaptive_timestep_budget: int = 1
    memory_slot_count: int = 1024
    bounded_memory_slot_candidate_count: int = 8
    active_memory_slot_count: int = 2
    sequence_length: int = 64
    batch_size: int = 16
    warmup_steps: int = 5
    repeats: int = 50
    device: str = "auto"
    seed: int = 20260705


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
    config: MemorySlotRuntimeImpactConfig,
    tokenizer: ByteLevelLanguageTokenizer,
    *,
    memory_slot_count: int,
    memory_slot_candidate_count: int,
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
        memory_slot_count=max(0, int(memory_slot_count)),
        memory_slot_candidate_count=max(0, int(memory_slot_candidate_count)),
        active_memory_slot_count=max(1, int(config.active_memory_slot_count)),
        generation_vocab_size=int(tokenizer.vocab_size),
    )


def _build_batch(
    tokenizer: ByteLevelLanguageTokenizer,
    config: MemorySlotRuntimeImpactConfig,
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


def _clone_control_state(
    config: MemorySlotRuntimeImpactConfig,
    tokenizer: ByteLevelLanguageTokenizer,
) -> dict[str, torch.Tensor]:
    torch.manual_seed(int(config.seed))
    base_model = MarulhoLanguageModel(
        _model_config(
            config,
            tokenizer,
            memory_slot_count=0,
            memory_slot_candidate_count=0,
        )
    )
    return {
        key: value.detach().clone()
        for key, value in base_model.state_dict().items()
    }


def _load_matching_state(
    model: MarulhoLanguageModel,
    base_state: Mapping[str, torch.Tensor],
) -> None:
    target_state = model.state_dict()
    for key, source_value in base_state.items():
        if key in target_state and target_state[key].shape == source_value.shape:
            target_state[key] = source_value.detach().clone().to(target_state[key].device)
    model.load_state_dict(target_state)


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


def _run_arm(
    name: str,
    *,
    memory_slot_count: int,
    memory_slot_candidate_count: int,
    base_state: Mapping[str, torch.Tensor],
    batch: LanguageBatch,
    config: MemorySlotRuntimeImpactConfig,
    tokenizer: ByteLevelLanguageTokenizer,
    device: torch.device,
) -> tuple[dict[str, Any], torch.Tensor | None]:
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    model_config = _model_config(
        config,
        tokenizer,
        memory_slot_count=memory_slot_count,
        memory_slot_candidate_count=memory_slot_candidate_count,
    )
    model: MarulhoLanguageModel | None = None
    try:
        model = MarulhoLanguageModel(model_config).to(device)
        _load_matching_state(model, base_state)
        model.eval()
        with torch.inference_mode():
            telemetry_result = _forward_once(
                model,
                batch,
                collect_telemetry=True,
            )
            for _ in range(max(0, int(config.warmup_steps))):
                _forward_once(model, batch, collect_telemetry=False)
            cuda_synchronized_before_timing_start = _sync_if_cuda(device)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            last_result: Mapping[str, Any] | None = None
            for _ in range(max(1, int(config.repeats))):
                last_result = _forward_once(model, batch, collect_telemetry=False)
            cuda_synchronized_before_timing_stop = _sync_if_cuda(device)
            elapsed = max(0.0, time.perf_counter() - started)
        token_count = int(batch.input_ids.numel()) * max(1, int(config.repeats))
        logits = (
            None
            if last_result is None
            else last_result["logits"].detach().float().cpu()
        )
        telemetry = (
            telemetry_result.get("telemetry", {})
            if isinstance(telemetry_result.get("telemetry"), Mapping)
            else {}
        )
        memory = telemetry.get("memory") if isinstance(telemetry.get("memory"), Mapping) else {}
        routing = telemetry.get("routing") if isinstance(telemetry.get("routing"), Mapping) else {}
        memory_slot_nonzero_count = (
            0
            if model.memory_slots is None
            else int(torch.count_nonzero(model.memory_slots.detach()).item())
        )
        memory_slot_gate_initial_value = (
            None
            if model.memory_slot_gate is None
            else float(model.memory_slot_gate.detach().item())
        )
        return {
            "surface": "marulho_language_memory_slot_runtime_arm.v1",
            "name": name,
            "success": True,
            "failure_reason": None,
            "model_config": asdict(model_config),
            "warmup_steps": int(config.warmup_steps),
            "measured_steps": int(config.repeats),
            "tokens_per_forward": int(batch.input_ids.numel()),
            "token_count": int(token_count),
            "elapsed_seconds": elapsed,
            "tokens_per_second": (
                float(token_count) / elapsed if elapsed > 0.0 else 0.0
            ),
            "memory_enabled": bool(memory.get("enabled", False)),
            "total_slots": int(memory.get("total_slots", 0) or 0),
            "candidate_slot_count": int(memory.get("candidate_slot_count", 0) or 0),
            "active_slots_per_token": int(
                memory.get("active_slots_per_token", 0) or 0
            ),
            "candidate_slots_scored": int(
                memory.get("candidate_slots_scored", 0) or 0
            ),
            "runs_all_slots": bool(memory.get("runs_all_slots", False)),
            "memory_fallback_reason": memory.get("fallback_reason"),
            "candidate_id_source": memory.get("candidate_id_source"),
            "memory_gate_readback": bool(memory.get("memory_gate_readback", False)),
            "memory_device": memory.get("memory_device"),
            "memory_active_parameters_per_token": int(
                memory.get("active_parameters_per_token", 0) or 0
            ),
            "memory_slot_nonzero_count": int(memory_slot_nonzero_count),
            "memory_slot_gate_initial_value": memory_slot_gate_initial_value,
            "memory_slot_trainable_neutral_initialization": bool(
                memory_slot_nonzero_count > 0
                and memory_slot_gate_initial_value == 0.0
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
            "route_candidate_rows_scored": int(
                routing.get("candidate_rows_scored", 0) or 0
            ),
            "runs_all_columns": bool(routing.get("runs_all_columns", False)),
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
            "surface": "marulho_language_memory_slot_runtime_arm.v1",
            "name": name,
            "success": False,
            "failure_reason": f"{type(exc).__name__}: {exc}",
            "model_config": asdict(model_config),
            "warmup_steps": int(config.warmup_steps),
            "measured_steps": int(config.repeats),
            "tokens_per_forward": int(batch.input_ids.numel()),
            "token_count": 0,
            "elapsed_seconds": 0.0,
            "tokens_per_second": 0.0,
            "memory_enabled": False,
            "total_slots": 0,
            "candidate_slot_count": 0,
            "active_slots_per_token": 0,
            "candidate_slots_scored": 0,
            "runs_all_slots": False,
            "memory_fallback_reason": f"{type(exc).__name__}: {exc}",
            "candidate_id_source": None,
            "memory_gate_readback": False,
            "memory_device": str(device),
            "memory_active_parameters_per_token": 0,
            "memory_slot_nonzero_count": 0,
            "memory_slot_gate_initial_value": None,
            "memory_slot_trainable_neutral_initialization": False,
            "route_selection_backend": "failed",
            "expert_dispatch_backend": "failed",
            "route_candidate_count": 0,
            "active_expert_count_per_token": 0,
            "route_candidate_rows_scored": 0,
            "runs_all_columns": False,
            "device": str(device),
            "cuda_memory": _cuda_memory(device),
        }, None
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _logit_parity(
    reference_logits: torch.Tensor | None,
    candidate_logits: torch.Tensor | None,
) -> dict[str, Any]:
    if reference_logits is None or candidate_logits is None:
        return {
            "available": False,
            "max_abs_error": None,
            "max_rel_error": None,
            "passed": False,
        }
    diff = (candidate_logits - reference_logits).abs()
    max_abs_error = float(diff.max().item())
    denominator = reference_logits.abs().clamp_min(1e-8)
    max_rel_error = float((diff / denominator).max().item())
    return {
        "available": True,
        "max_abs_error": max_abs_error,
        "max_rel_error": max_rel_error,
        "passed": bool(max_abs_error <= 1e-6 or max_rel_error <= 1e-6),
    }


def _ratio(numerator: float, denominator: float) -> float | None:
    if denominator <= 0.0:
        return None
    return numerator / denominator


def _comparison(
    control: dict[str, Any],
    bounded: dict[str, Any],
    all_slot: dict[str, Any],
    *,
    control_logits: torch.Tensor | None,
    bounded_logits: torch.Tensor | None,
    all_slot_logits: torch.Tensor | None,
) -> dict[str, Any]:
    control_success = bool(control.get("success"))
    bounded_success = bool(bounded.get("success"))
    all_slot_success = bool(all_slot.get("success"))
    control_tps = float(control.get("tokens_per_second", 0.0) or 0.0)
    bounded_tps = float(bounded.get("tokens_per_second", 0.0) or 0.0)
    all_slot_tps = float(all_slot.get("tokens_per_second", 0.0) or 0.0)
    bounded_parity = _logit_parity(control_logits, bounded_logits)
    all_slot_parity = _logit_parity(control_logits, all_slot_logits)
    bounded_avoids_all_slot_scan = bool(
        bounded_success
        and bounded.get("memory_enabled")
        and int(bounded.get("candidate_slot_count", 0) or 0)
        < int(bounded.get("total_slots", 0) or 0)
        and not bool(bounded.get("runs_all_slots", False))
    )
    all_slot_contrast_available = bool(
        all_slot_success
        and all_slot.get("memory_enabled")
        and bool(all_slot.get("runs_all_slots", False))
    )
    trainable_neutral_initialization = bool(
        bounded.get("memory_slot_trainable_neutral_initialization", False)
    )
    if control_success and bounded_success and bounded_avoids_all_slot_scan:
        evidence_status = "measured_bounded_memory_slot_forward_impact"
    elif control_success and bounded_success:
        evidence_status = "measured_memory_slot_forward_without_bounded_retrieval"
    else:
        evidence_status = "memory_slot_runtime_impact_measurement_failed"
    return {
        "surface": "marulho_language_memory_slot_runtime_comparison.v1",
        "control_success": control_success,
        "bounded_success": bounded_success,
        "all_slot_success": all_slot_success,
        "control_tokens_per_second": control_tps,
        "bounded_tokens_per_second": bounded_tps,
        "all_slot_tokens_per_second": all_slot_tps,
        "bounded_vs_control_tokens_per_second_ratio": _ratio(
            bounded_tps,
            control_tps,
        ),
        "all_slot_vs_bounded_tokens_per_second_ratio": _ratio(
            all_slot_tps,
            bounded_tps,
        ),
        "bounded_memory_enabled": bool(bounded.get("memory_enabled", False)),
        "bounded_avoids_all_slot_scan": bounded_avoids_all_slot_scan,
        "all_slot_scan_contrast_available": all_slot_contrast_available,
        "bounded_memory_slot_nonzero_count": int(
            bounded.get("memory_slot_nonzero_count", 0) or 0
        ),
        "bounded_memory_slot_gate_initial_value": bounded.get(
            "memory_slot_gate_initial_value"
        ),
        "bounded_trainable_neutral_initialization": trainable_neutral_initialization,
        "bounded_candidate_slots_scored_per_forward": int(
            bounded.get("candidate_slots_scored", 0) or 0
        ),
        "all_slot_candidate_slots_scored_per_forward": int(
            all_slot.get("candidate_slots_scored", 0) or 0
        ),
        "bounded_neutral_initialization_parity": bounded_parity,
        "all_slot_neutral_initialization_parity": all_slot_parity,
        "memory_gate_readback": bool(
            bounded.get("memory_gate_readback", False)
            or all_slot.get("memory_gate_readback", False)
        ),
        "evidence_status": evidence_status,
    }


def run_language_memory_slot_runtime_impact(
    *,
    output_path: str | Path,
    config: MemorySlotRuntimeImpactConfig | None = None,
) -> dict[str, Any]:
    cfg = config or MemorySlotRuntimeImpactConfig()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = ByteLevelLanguageTokenizer()
    if int(cfg.vocab_size) < int(tokenizer.vocab_size):
        raise ValueError("vocab_size must be at least the tokenizer vocab size")
    if int(cfg.memory_slot_count) <= 1:
        raise ValueError("memory_slot_count must be greater than one")
    if int(cfg.bounded_memory_slot_candidate_count) >= int(cfg.memory_slot_count):
        raise ValueError(
            "bounded_memory_slot_candidate_count must be smaller than memory_slot_count"
        )
    device = _resolve_device(str(cfg.device))
    batch, split_report = _build_batch(tokenizer, cfg, device=device)
    base_state = _clone_control_state(cfg, tokenizer)
    control_report, control_logits = _run_arm(
        "memory_slots_disabled_control",
        memory_slot_count=0,
        memory_slot_candidate_count=0,
        base_state=base_state,
        batch=batch,
        config=cfg,
        tokenizer=tokenizer,
        device=device,
    )
    bounded_report, bounded_logits = _run_arm(
        "bounded_memory_slots_enabled",
        memory_slot_count=int(cfg.memory_slot_count),
        memory_slot_candidate_count=int(cfg.bounded_memory_slot_candidate_count),
        base_state=base_state,
        batch=batch,
        config=cfg,
        tokenizer=tokenizer,
        device=device,
    )
    all_slot_report, all_slot_logits = _run_arm(
        "all_slot_memory_scan_contrast",
        memory_slot_count=int(cfg.memory_slot_count),
        memory_slot_candidate_count=int(cfg.memory_slot_count),
        base_state=base_state,
        batch=batch,
        config=cfg,
        tokenizer=tokenizer,
        device=device,
    )
    comparison = _comparison(
        control_report,
        bounded_report,
        all_slot_report,
        control_logits=control_logits,
        bounded_logits=bounded_logits,
        all_slot_logits=all_slot_logits,
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
            "memory_slots_disabled_control": control_report,
            "bounded_memory_slots_enabled": bounded_report,
            "all_slot_memory_scan_contrast": all_slot_report,
        },
        "comparison": comparison,
        "review": {
            "complete_forward_runtime_impact": True,
            "includes_embedding_state_memory_routing_dispatch_and_decode_head": True,
            "not_kernel_microbench_only": True,
            "mutates_model_state": False,
            "neutral_memory_gate_keeps_initial_logits_unchanged": bool(
                comparison["bounded_neutral_initialization_parity"]["passed"]
            ),
            "memory_slots_nonzero_with_zero_gate": bool(
                comparison["bounded_trainable_neutral_initialization"]
            ),
            "gradient_training_unchanged": True,
            "one_token_streaming_policy_unchanged": True,
            "promotes_hot_path": False,
            "promotes_runtime_claim": False,
            "next_experiment": (
                "profile bounded memory slots in gradient training and sustained "
                "generation before allowing memory growth to promote a hot path"
            ),
        },
        "promotion_gate": {
            "runtime_impact_available": bool(
                comparison["control_success"] and comparison["bounded_success"]
            ),
            "bounded_memory_slots_enabled": bool(
                comparison["bounded_memory_enabled"]
            ),
            "bounded_avoids_all_slot_scan": bool(
                comparison["bounded_avoids_all_slot_scan"]
            ),
            "all_slot_scan_contrast_available": bool(
                comparison["all_slot_scan_contrast_available"]
            ),
            "neutral_initialization_parity": bool(
                comparison["bounded_neutral_initialization_parity"]["passed"]
            ),
            "trainable_neutral_initialization": bool(
                comparison["bounded_trainable_neutral_initialization"]
            ),
            "complete_runtime_impact_available": bool(
                comparison["control_success"] and comparison["bounded_success"]
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
    parser.add_argument("--memory-slot-count", type=int, default=1024)
    parser.add_argument("--bounded-memory-slot-candidate-count", type=int, default=8)
    parser.add_argument("--active-memory-slot-count", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    config = MemorySlotRuntimeImpactConfig(
        vocab_size=args.vocab_size,
        embedding_dim=args.embedding_dim,
        state_dim=args.state_dim,
        expert_count=args.expert_count,
        active_expert_count=args.active_expert_count,
        route_candidate_count=args.route_candidate_count,
        expert_hidden_dim=args.expert_hidden_dim,
        adaptive_timestep_budget=args.adaptive_timestep_budget,
        memory_slot_count=args.memory_slot_count,
        bounded_memory_slot_candidate_count=args.bounded_memory_slot_candidate_count,
        active_memory_slot_count=args.active_memory_slot_count,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        warmup_steps=args.warmup_steps,
        repeats=args.repeats,
        device=args.device,
    )
    report = run_language_memory_slot_runtime_impact(
        output_path=args.output,
        config=config,
    )
    comparison = report["comparison"]
    print(
        "wrote "
        f"{args.output} bounded_tps={comparison['bounded_tokens_per_second']:.3f} "
        f"control_tps={comparison['control_tokens_per_second']:.3f} "
        f"ratio={comparison['bounded_vs_control_tokens_per_second_ratio']}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
