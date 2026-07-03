"""Sustained evidence runner for the checkpointed MARULHO LM head."""

from __future__ import annotations

import argparse
from collections import deque
from collections.abc import Callable
from pathlib import Path
import time
from typing import Any, Mapping

import torch

from marulho.data.language_tokenizer import ByteLevelLanguageTokenizer
from marulho.evaluation.continuous_runtime_stress_benchmark import (
    _collect_velocity_environment_snapshot,
    _summarize_velocity_environment,
)
from marulho.reporting.readme_reports import write_json_report_with_readme
from marulho.training.language_model import (
    MarulhoLanguageModel,
    load_language_model_checkpoint,
)


SURFACE = "marulho_language_sustained_runtime_evidence.v1"
ARTIFACT_KIND = "marulho_language_sustained_runtime_evidence"


def _environment_snapshot(*, collect: bool) -> dict[str, Any]:
    if not bool(collect):
        return {
            "available": False,
            "sample_source": "disabled",
            "reason": "environment_collection_disabled",
        }
    try:
        return _collect_velocity_environment_snapshot()
    except Exception as exc:  # pragma: no cover - defensive evidence guard
        return {
            "available": False,
            "sample_source": "velocity_environment",
            "error": f"{type(exc).__name__}: {exc}",
        }


def _environment_summary(
    before: Mapping[str, Any] | None,
    after: Mapping[str, Any] | None,
) -> dict[str, Any]:
    try:
        return _summarize_velocity_environment(before, after)
    except Exception as exc:  # pragma: no cover - defensive evidence guard
        return {
            "surface": "velocity_environment.v1",
            "not_hot_path": True,
            "contention": {"verdict": "unknown"},
            "error": f"{type(exc).__name__}: {exc}",
        }


def _status_for_run(
    *,
    success: bool,
    timeout: bool,
    manual_stop: bool,
    interrupted: bool,
    exception: BaseException | None,
) -> str:
    if interrupted:
        return "interrupt"
    if exception is not None:
        return "exception"
    if success:
        return "final"
    if timeout:
        return "timeout"
    if manual_stop:
        return "partial"
    return "partial"


def _routing_value(
    routing: Mapping[str, Any],
    key: str,
    default: int | float | bool | str | None = None,
) -> Any:
    value = routing.get(key, default)
    return default if value is None else value


def _last_language_trace(
    *,
    token_delta: int,
    target_tokens: int,
    active_language_path: str,
    telemetry: Mapping[str, Any],
    last_token_id: int | None,
) -> dict[str, Any]:
    routing = telemetry.get("routing") if isinstance(telemetry.get("routing"), Mapping) else {}
    return {
        "surface": "marulho_language_sustained_trace.v1",
        "event": "language_lm_head_stream",
        "token_delta": int(token_delta),
        "target_tokens": int(target_tokens),
        "last_token_id": last_token_id,
        "active_language_path": active_language_path,
        "external_llm_used": False,
        "device": str(telemetry.get("device") or routing.get("route_device") or "unknown"),
        "spike_rate": float(telemetry.get("spike_rate", 0.0) or 0.0),
        "active_columns": int(_routing_value(routing, "active_columns", 0) or 0),
        "total_columns": int(_routing_value(routing, "total_columns", 0) or 0),
        "active_parameters_per_token": int(
            _routing_value(routing, "active_parameters_per_token", 0) or 0
        ),
        "runs_all_columns": bool(_routing_value(routing, "runs_all_columns", False)),
        "fallback_reason": routing.get("fallback_reason"),
    }


def _new_execution_evidence(model: MarulhoLanguageModel) -> dict[str, Any]:
    backend = "torch_eager_cuda" if model.device.type == "cuda" else "torch_eager_cpu"
    return {
        "surface": "marulho_language_sustained_execution_evidence.v1",
        "mode": "torch_eager_step",
        "backend": backend,
        "cuda_graph_burst_available": bool(
            model.device.type == "cuda"
            and torch.cuda.is_available()
            and hasattr(torch.cuda, "CUDAGraph")
        ),
        "cuda_graph_burst_used": False,
        "cuda_graph_burst_tokens": 0,
        "cuda_graph_burst_replay_count": 0,
        "cuda_graph_language_token_count": 0,
        "cuda_graph_setup_seconds": 0.0,
        "cuda_graph_failure_count": 0,
        "cuda_graph_failure_reason": None,
        "pytorch_eager_language_token_count": 0,
        "pytorch_eager_tail_token_count": 0,
    }


def _try_capture_cuda_graph_burst(
    *,
    model: MarulhoLanguageModel,
    next_logits: torch.Tensor,
    state: Mapping[str, torch.Tensor],
    burst_tokens: int,
    assume_no_sleeping_experts: bool,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    evidence = _new_execution_evidence(model)
    if model.device.type != "cuda" or not torch.cuda.is_available():
        evidence["cuda_graph_failure_reason"] = "cuda_unavailable"
        return None, evidence
    if not hasattr(torch.cuda, "CUDAGraph"):
        evidence["cuda_graph_failure_reason"] = "torch_cuda_graph_unavailable"
        return None, evidence
    burst_tokens = max(1, int(burst_tokens))
    if burst_tokens <= 1:
        evidence["cuda_graph_failure_reason"] = "burst_tokens_not_greater_than_one"
        return None, evidence

    setup_started = time.perf_counter()
    graph: torch.cuda.CUDAGraph | None = None
    try:
        token_buffer = torch.argmax(next_logits, dim=-1, keepdim=True).detach().clone()
        state_buffers = {
            key: value.detach().clone()
            for key, value in state.items()
        }
        reset_token = token_buffer.detach().clone()
        reset_state = {
            key: value.detach().clone()
            for key, value in state_buffers.items()
        }
        tail_buffer = torch.empty(
            (burst_tokens,),
            device=model.device,
            dtype=torch.long,
        )

        warmup_stream = torch.cuda.Stream(device=model.device)
        warmup_stream.wait_stream(torch.cuda.current_stream(model.device))
        with torch.cuda.stream(warmup_stream):
            for _ in range(2):
                model.forward_step(
                    token_buffer,
                    state_buffers,
                    collect_telemetry=False,
                    assume_no_sleeping_experts=assume_no_sleeping_experts,
                )
        torch.cuda.current_stream(model.device).wait_stream(warmup_stream)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            current_token = token_buffer
            current_state: Mapping[str, torch.Tensor] = state_buffers
            for step in range(burst_tokens):
                tail_buffer[step].copy_(current_token.reshape(-1)[0])
                step_result = model.forward_step(
                    current_token,
                    current_state,
                    collect_telemetry=False,
                    assume_no_sleeping_experts=assume_no_sleeping_experts,
                )
                current_state = step_result["state"]
                current_token = torch.argmax(
                    step_result["logits"][:, -1, :],
                    dim=-1,
                    keepdim=True,
                )
            token_buffer.copy_(current_token)
            for key in state_buffers:
                state_buffers[key].copy_(current_state[key])

        token_buffer.copy_(reset_token)
        for key in state_buffers:
            state_buffers[key].copy_(reset_state[key])
        torch.cuda.synchronize(model.device)
        evidence.update(
            {
                "mode": "torch_cuda_graph_burst",
                "backend": "torch_cuda_graph_burst",
                "cuda_graph_burst_used": True,
                "cuda_graph_burst_tokens": int(burst_tokens),
                "cuda_graph_setup_seconds": time.perf_counter() - setup_started,
            }
        )
        return (
            {
                "graph": graph,
                "token_buffer": token_buffer,
                "state_buffers": state_buffers,
                "tail_buffer": tail_buffer,
                "burst_tokens": int(burst_tokens),
            },
            evidence,
        )
    except Exception as exc:  # pragma: no cover - hardware/runtime dependent
        if graph is not None:
            del graph
        evidence.update(
            {
                "cuda_graph_failure_count": 1,
                "cuda_graph_failure_reason": f"{type(exc).__name__}: {exc}",
                "cuda_graph_setup_seconds": time.perf_counter() - setup_started,
            }
        )
        return None, evidence


def _report_payload(
    *,
    output_path: Path,
    checkpoint_path: str | Path | None,
    target_tokens: int,
    token_delta: int,
    prompt_token_count: int,
    tick_tokens: int,
    quantum_tokens: int,
    timeout_seconds: float,
    elapsed_seconds: float,
    success: bool,
    failure_reason: str | None,
    manual_stop: bool,
    interrupted: bool,
    exception: BaseException | None,
    model: MarulhoLanguageModel,
    tokenizer: ByteLevelLanguageTokenizer,
    telemetry: Mapping[str, Any],
    last_token_id: int | None,
    generated_tail_ids: list[int],
    environment_before: Mapping[str, Any] | None,
    environment_after: Mapping[str, Any] | None,
    checkpoint_metadata: Mapping[str, Any] | None,
    execution_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    elapsed_seconds = max(0.0, float(elapsed_seconds))
    timeout = (
        not bool(success)
        and exception is None
        and not bool(interrupted)
        and not bool(manual_stop)
        and elapsed_seconds >= float(timeout_seconds)
    )
    report_status = _status_for_run(
        success=bool(success),
        timeout=bool(timeout),
        manual_stop=bool(manual_stop),
        interrupted=bool(interrupted),
        exception=exception,
    )
    routing = telemetry.get("routing") if isinstance(telemetry.get("routing"), Mapping) else {}
    active_language_path = str(model.config.active_language_path)
    model_device = model.device
    execution = dict(execution_evidence or _new_execution_evidence(model))
    backend = str(
        execution.get("backend")
        or ("torch_eager_cuda" if model_device.type == "cuda" else "torch_eager_cpu")
    )
    cuda_graph_tokens = int(execution.get("cuda_graph_language_token_count", 0) or 0)
    eager_tokens = int(
        execution.get("pytorch_eager_language_token_count", int(token_delta)) or 0
    )
    fallback_reason = (
        "language_lm_head_uses_cuda_graph_burst_until_triton_parity_gate"
        if cuda_graph_tokens > 0
        else "language_lm_head_uses_pytorch_eager_until_triton_parity_gate"
    )
    trace = _last_language_trace(
        token_delta=token_delta,
        target_tokens=target_tokens,
        active_language_path=active_language_path,
        telemetry={**dict(telemetry), "device": str(model_device)},
        last_token_id=last_token_id,
    )
    return {
        "artifact_kind": ARTIFACT_KIND,
        "surface": SURFACE,
        "report_status": report_status,
        "success": bool(success),
        "failure_reason": failure_reason,
        "target_tokens": int(target_tokens),
        "token_delta": int(token_delta),
        "prompt_token_count": int(prompt_token_count),
        "elapsed_seconds": float(elapsed_seconds),
        "tokens_per_second": (
            float(token_delta) / elapsed_seconds if elapsed_seconds > 0.0 else 0.0
        ),
        "checkpoint_path": None if checkpoint_path is None else str(checkpoint_path),
        "checkpoint_metadata": dict(checkpoint_metadata or {}),
        "output_path": str(output_path),
        "runtime_owner": "MarulhoLanguageModel",
        "trainer_owner": "marulho.training.language_model",
        "active_language_path": active_language_path,
        "owned_by_marulho": True,
        "external_llm_used": False,
        "thought_loop_used": False,
        "cortex_used": False,
        "loads_external_checkpoint": False,
        "tick_tokens": int(tick_tokens),
        "quantum_tokens": int(quantum_tokens),
        "tick_count": int((token_delta + max(1, tick_tokens) - 1) // max(1, tick_tokens)),
        "quantum_count": int(
            (token_delta + max(1, quantum_tokens) - 1) // max(1, quantum_tokens)
        ),
        "last_trace": trace,
        "device_backend": {
            "surface": "marulho_language_sustained_device_backend.v1",
            "device": str(model_device),
            "backend": backend,
            "torch_cuda_available": bool(torch.cuda.is_available()),
            "cuda_selected": bool(model_device.type == "cuda"),
            "cuda_graph_burst_used": bool(execution.get("cuda_graph_burst_used", False)),
            "cuda_graph_burst_tokens": int(execution.get("cuda_graph_burst_tokens", 0) or 0),
            "cuda_graph_burst_replay_count": int(
                execution.get("cuda_graph_burst_replay_count", 0) or 0
            ),
            "cuda_graph_language_token_count": cuda_graph_tokens,
            "cuda_graph_setup_seconds": float(
                execution.get("cuda_graph_setup_seconds", 0.0) or 0.0
            ),
            "triton_kernel_used": False,
            "promoted_hot_path": False,
        },
        "failure_fallback_counters": {
            "surface": "marulho_language_sustained_failure_fallback_counters.v1",
            "cuda_graph_failure_count": int(
                execution.get("cuda_graph_failure_count", 0) or 0
            ),
            "native_burst_replay_failure_count": 0,
            "native_sequence_loop_failure_count": 0,
            "torch_sequence_graph_failure_count": 0,
            "triton_kernel_failure_count": 0,
            "triton_kernel_fallback_count": int(token_delta),
            "fallback_reason": fallback_reason,
            "cuda_graph_failure_reason": execution.get("cuda_graph_failure_reason"),
        },
        "fallback_counts": {
            "pytorch_eager_language_token_count": eager_tokens,
            "torch_cuda_graph_language_token_count": cuda_graph_tokens,
            "triton_kernel_fallback_count": int(token_delta),
            "external_lm_fallback_count": 0,
        },
        "execution_evidence": execution,
        "active_columns": int(_routing_value(routing, "active_columns", 0) or 0),
        "total_columns": int(_routing_value(routing, "total_columns", 0) or 0),
        "active_parameters_per_token": int(
            _routing_value(routing, "active_parameters_per_token", 0) or 0
        ),
        "route_candidate_rows_scored": int(
            _routing_value(routing, "candidate_rows_scored", 0) or 0
        ),
        "runs_all_columns": bool(_routing_value(routing, "runs_all_columns", False)),
        "route_device": routing.get("route_device"),
        "route_latency_ms": routing.get("route_latency_ms"),
        "spike_rate": float(telemetry.get("spike_rate", 0.0) or 0.0),
        "dead_neuron_rate": float(telemetry.get("dead_neuron_rate", 0.0) or 0.0),
        "over_firing_rate": float(telemetry.get("over_firing_rate", 0.0) or 0.0),
        "replay_consolidation_events": 0,
        "growth_prune_proposals": 0,
        "generated_tail_ids": list(generated_tail_ids[-32:]),
        "tokenizer_hash": tokenizer.vocabulary_hash(),
        "vocab_size": int(tokenizer.vocab_size),
        "environment_contention": _environment_summary(
            environment_before,
            environment_after,
        ),
        "evidence_state": {
            "final": report_status == "final",
            "partial": report_status == "partial",
            "timeout": report_status == "timeout",
            "interrupt": report_status == "interrupt",
            "exception": report_status == "exception",
            "manual_stop": bool(manual_stop),
        },
        "exception": (
            None
            if exception is None
            else {
                "type": type(exception).__name__,
                "message": str(exception),
            }
        ),
        "promotion_gate": {
            "diagnostic_boundary_reached": int(token_delta) >= 8192,
            "long_run_gate_reached": int(token_delta) >= 131072,
            "house_scale_gate_reached": int(token_delta) >= 524288,
            "eligible_for_language_long_run_review": bool(success)
            and int(token_delta) >= 131072,
            "promotes_runtime_claim": False,
            "promotes_hot_path": False,
            "requires_triton_or_cuda_hot_path_evidence": True,
            "short_run_is_smoke_only": int(target_tokens) < 8192,
        },
    }


def run_language_sustained_runtime_evidence(
    model: MarulhoLanguageModel,
    tokenizer: ByteLevelLanguageTokenizer,
    *,
    output_path: str | Path,
    target_tokens: int,
    checkpoint_path: str | Path | None = None,
    checkpoint_metadata: Mapping[str, Any] | None = None,
    prompt: str = "MARULHO",
    tick_tokens: int = 128,
    quantum_tokens: int = 16,
    timeout_seconds: float = 60.0,
    stop_on_eos: bool = False,
    should_stop: Callable[[], bool] | None = None,
    collect_environment: bool = True,
) -> dict[str, Any]:
    """Stream the LM head and always write a final or partial evidence report."""

    if int(target_tokens) <= 0:
        raise ValueError("target_tokens must be positive")
    if int(tick_tokens) <= 0:
        raise ValueError("tick_tokens must be positive")
    if int(quantum_tokens) <= 0:
        raise ValueError("quantum_tokens must be positive")
    output = Path(output_path)
    environment_before = _environment_snapshot(collect=collect_environment)
    started = time.perf_counter()
    deadline = started + max(0.0, float(timeout_seconds))
    environment_after: Mapping[str, Any] | None = None
    was_training = bool(model.training)
    token_delta = 0
    prompt_token_count = 0
    last_token_id: int | None = None
    generated_tail_ids: list[int] = []
    tail_token_tensors: deque[torch.Tensor] = deque(maxlen=32)
    last_token_tensor: torch.Tensor | None = None
    telemetry: Mapping[str, Any] = {}
    success = False
    failure_reason: str | None = None
    manual_stop = False
    interrupted = False
    exception: BaseException | None = None
    finished = started
    execution_evidence = _new_execution_evidence(model)
    graph_tail_tensor: torch.Tensor | None = None

    try:
        model.eval()
        prompt_ids = tokenizer.encode(str(prompt or ""), add_bos=True, add_eos=False)
        if not prompt_ids:
            prompt_ids = [tokenizer.bos_id]
        prompt_token_count = len(prompt_ids)
        assume_no_sleeping = (
            model.routed_experts.enabled
            and not bool(
                model.routed_experts.sleeping_expert_mask.detach().any().cpu().item()
            )
        )
        with torch.no_grad():
            generated = torch.tensor(
                [prompt_ids],
                dtype=torch.long,
                device=model.device,
            )
            result = model(
                generated,
                collect_telemetry=True,
                assume_no_sleeping_experts=assume_no_sleeping,
            )
            state = result["state"]
            telemetry = dict(result["telemetry"])
            next_logits = result["logits"][:, -1, :]
            graph_runner: dict[str, Any] | None = None
            graph_generated_tokens = 0
            graph_replay_count = 0
            eager_generated_tokens = 0
            current_next_id: torch.Tensor | None = None
            if should_stop is None and not bool(stop_on_eos):
                graph_runner, execution_evidence = _try_capture_cuda_graph_burst(
                    model=model,
                    next_logits=next_logits,
                    state=state,
                    burst_tokens=min(
                        max(2, int(quantum_tokens)),
                        int(target_tokens),
                    ),
                    assume_no_sleeping_experts=assume_no_sleeping,
                )
            else:
                execution_evidence["cuda_graph_failure_reason"] = (
                    "per_token_stop_or_eos_required"
                )

            if graph_runner is not None:
                burst_tokens = int(graph_runner["burst_tokens"])
                while token_delta + burst_tokens <= int(target_tokens):
                    if time.perf_counter() >= deadline:
                        failure_reason = "target_tokens_not_reached_before_timeout"
                        break
                    graph_runner["graph"].replay()
                    token_delta += burst_tokens
                    graph_generated_tokens += burst_tokens
                    graph_replay_count += 1
                if graph_generated_tokens > 0:
                    graph_tail_tensor = graph_runner["tail_buffer"].detach().clone()
                    last_token_tensor = graph_tail_tensor[-1:].reshape(-1)
                state = graph_runner["state_buffers"]
                current_next_id = graph_runner["token_buffer"]

            while token_delta < int(target_tokens):
                if should_stop is not None and bool(should_stop()):
                    manual_stop = True
                    failure_reason = "manual_stop"
                    break
                if time.perf_counter() >= deadline:
                    failure_reason = "target_tokens_not_reached_before_timeout"
                    break
                next_id = (
                    current_next_id
                    if current_next_id is not None
                    else torch.argmax(next_logits, dim=-1, keepdim=True)
                )
                last_token_tensor = next_id.detach().reshape(-1)[:1]
                tail_token_tensors.append(last_token_tensor)
                token_delta += 1
                eager_generated_tokens += 1
                collect_step_telemetry = (
                    token_delta >= int(target_tokens)
                    or token_delta % max(1, int(tick_tokens)) == 0
                )
                result = model.forward_step(
                    next_id,
                    state,
                    collect_telemetry=collect_step_telemetry,
                    assume_no_sleeping_experts=assume_no_sleeping,
                )
                state = result["state"]
                if collect_step_telemetry:
                    telemetry = dict(result["telemetry"])
                next_logits = result["logits"][:, -1, :]
                if current_next_id is not None:
                    current_next_id = torch.argmax(next_logits, dim=-1, keepdim=True)
                if bool(stop_on_eos):
                    last_token_id = int(last_token_tensor.detach().cpu().item())
                    if last_token_id == int(tokenizer.eos_id):
                        failure_reason = "eos_before_target_tokens"
                        break
            execution_evidence["cuda_graph_burst_replay_count"] = int(graph_replay_count)
            execution_evidence["cuda_graph_language_token_count"] = int(
                graph_generated_tokens
            )
            execution_evidence["pytorch_eager_tail_token_count"] = int(
                eager_generated_tokens
            )
            execution_evidence["pytorch_eager_language_token_count"] = int(
                eager_generated_tokens if graph_generated_tokens > 0 else token_delta
            )
            success = token_delta >= int(target_tokens)
            if success:
                failure_reason = None
            elif failure_reason is None:
                failure_reason = "target_tokens_not_reached"
    except KeyboardInterrupt as exc:  # pragma: no cover - hard to trigger under pytest
        interrupted = True
        exception = exc
        failure_reason = "keyboard_interrupt_manual_stop"
    except Exception as exc:
        exception = exc
        failure_reason = f"exception:{type(exc).__name__}"
    finally:
        finished = time.perf_counter()
        if was_training:
            model.train()
        else:
            model.eval()
        environment_after = _environment_snapshot(collect=collect_environment)

    if last_token_tensor is not None and last_token_id is None:
        last_token_id = int(last_token_tensor.detach().cpu().item())
    tail_values: list[int] = []
    if graph_tail_tensor is not None:
        tail_values.extend(
            int(value)
            for value in graph_tail_tensor.detach().cpu().reshape(-1).tolist()
        )
    if tail_token_tensors:
        tail_values.extend(
            int(value)
            for value in torch.cat([item.reshape(1) for item in tail_token_tensors])
            .detach()
            .cpu()
            .tolist()
        )
    if tail_values:
        generated_tail_ids = tail_values[-32:]

    report = _report_payload(
        output_path=output,
        checkpoint_path=checkpoint_path,
        target_tokens=int(target_tokens),
        token_delta=int(token_delta),
        prompt_token_count=int(prompt_token_count),
        tick_tokens=int(tick_tokens),
        quantum_tokens=int(quantum_tokens),
        timeout_seconds=float(timeout_seconds),
        elapsed_seconds=max(0.0, finished - started),
        success=bool(success),
        failure_reason=failure_reason,
        manual_stop=bool(manual_stop),
        interrupted=bool(interrupted),
        exception=exception,
        model=model,
        tokenizer=tokenizer,
        telemetry=telemetry,
        last_token_id=last_token_id,
        generated_tail_ids=generated_tail_ids,
        environment_before=environment_before,
        environment_after=environment_after,
        checkpoint_metadata=checkpoint_metadata,
        execution_evidence=execution_evidence,
    )
    write_json_report_with_readme(output, report)
    return report


def run_language_sustained_runtime_evidence_from_checkpoint(
    checkpoint_path: str | Path,
    *,
    output_path: str | Path,
    target_tokens: int,
    prompt: str = "MARULHO",
    tick_tokens: int = 128,
    quantum_tokens: int = 16,
    timeout_seconds: float = 60.0,
    stop_on_eos: bool = False,
    collect_environment: bool = True,
    map_location: str | torch.device | None = None,
) -> dict[str, Any]:
    model, tokenizer, metadata = load_language_model_checkpoint(
        checkpoint_path,
        map_location=map_location,
    )
    if map_location is not None:
        target_device = torch.device(map_location)
        if target_device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA map_location requested but CUDA is unavailable")
        model.to(target_device)
    return run_language_sustained_runtime_evidence(
        model,
        tokenizer,
        output_path=output_path,
        target_tokens=target_tokens,
        checkpoint_path=checkpoint_path,
        checkpoint_metadata=metadata,
        prompt=prompt,
        tick_tokens=tick_tokens,
        quantum_tokens=quantum_tokens,
        timeout_seconds=timeout_seconds,
        stop_on_eos=stop_on_eos,
        collect_environment=collect_environment,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-tokens", type=int, default=8192)
    parser.add_argument("--prompt", default="MARULHO")
    parser.add_argument("--tick-tokens", type=int, default=128)
    parser.add_argument("--quantum-tokens", type=int, default=16)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--stop-on-eos", action="store_true")
    parser.add_argument("--no-environment-snapshot", action="store_true")
    parser.add_argument("--map-location", default=None)
    args = parser.parse_args()
    report = run_language_sustained_runtime_evidence_from_checkpoint(
        args.checkpoint,
        output_path=args.output,
        target_tokens=args.target_tokens,
        prompt=args.prompt,
        tick_tokens=args.tick_tokens,
        quantum_tokens=args.quantum_tokens,
        timeout_seconds=args.timeout_seconds,
        stop_on_eos=bool(args.stop_on_eos),
        collect_environment=not bool(args.no_environment_snapshot),
        map_location=args.map_location,
    )
    return 0 if bool(report.get("success")) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
