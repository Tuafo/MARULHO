"""A/B maintained continuous MarulhoBrain execution quanta on one checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time
from typing import Any

from marulho.brain import MarulhoBrain
from marulho.reporting.readme_reports import write_json_report_with_readme


SOURCE_TEXT_UNIT = (
    "Adaptive memory plasticity stabilizes sparse spike routing. "
    "Grounded local observations support prediction error and replay readiness. "
)
DEFAULT_SOURCE_TEXT = SOURCE_TEXT_UNIT * 8


def source_text_for_target(target_tokens: int, *, slack_tokens: int = 64) -> str:
    desired_tokens = max(32, int(target_tokens) + max(0, int(slack_tokens)))
    repeat_count = max(1, (desired_tokens // len(SOURCE_TEXT_UNIT)) + 1)
    return (SOURCE_TEXT_UNIT * repeat_count)[:desired_tokens]


def _run_arm(
    *,
    checkpoint: Path,
    source_path: Path,
    arm_root: Path,
    quantum_tokens: int,
    yield_seconds: float,
    target_tokens: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    brain = MarulhoBrain.load(checkpoint)
    try:
        source_text = source_path.read_text(encoding="utf-8")
        feed = brain.feed(
            source_text,
            source="continuous_quantum_source",
            learn=False,
        )
        if int(feed.get("accepted_tokens", 0) or 0) <= 0:
            return {
                "success": False,
                "failure_reason": "source_feed_empty",
                "execution_schedule": {
                    "owner": "MarulhoBrain",
                    "quantum_tokens": int(quantum_tokens),
                    "yield_seconds": float(yield_seconds),
                },
                "warm_ingestion": {
                    "surface": "marulho_brain_source_buffer.v1",
                    "accepted_tokens": 0,
                    "full_warm_ready": False,
                },
            }

        start_token = int(brain.trainer.token_count)
        brain.start(
            tick_tokens=128,
            quantum_tokens=int(quantum_tokens),
            interval_seconds=max(
                0.01,
                float(yield_seconds) if yield_seconds > 0.0 else 0.01,
            ),
            source="continuous_quantum_source",
        )
        started = time.perf_counter()
        deadline = started + max(0.1, float(timeout_seconds))
        current_token = start_token
        while time.perf_counter() < deadline:
            current_token = int(brain.trainer.token_count)
            if current_token - start_token >= int(target_tokens):
                break
            time.sleep(0.001)
        elapsed_seconds = time.perf_counter() - started
        stop_started = time.perf_counter()
        stop = brain.stop(timeout_seconds=max(1.0, float(timeout_seconds)))
        stop_latency_ms = (time.perf_counter() - stop_started) * 1000.0

        final_token = int(brain.trainer.token_count)
        final_status = brain.status()
        transition_report = brain.trainer.column_transition_runtime_report()
        device_report = brain.trainer.config.device_report()
        token_delta = max(0, final_token - start_token)
        return {
            "success": bool(token_delta >= int(target_tokens)),
            "failure_reason": (
                None
                if token_delta >= int(target_tokens)
                else "target_tokens_not_reached_before_timeout"
            ),
            "target_tokens": int(target_tokens),
            "token_delta": int(token_delta),
            "elapsed_seconds": float(elapsed_seconds),
            "tokens_per_second": float(
                token_delta / max(elapsed_seconds, 1e-9)
            ),
            "stop_latency_ms": float(stop_latency_ms),
            "shutdown": {
                "surface": "marulho_brain_loop_stop.v1",
                "stopped": bool(stop.get("stopped")),
            },
            "execution_schedule": {
                "surface": "marulho_brain_loop_state.v1",
                "owner": "MarulhoBrain",
                "tick_tokens": 128,
                "quantum_tokens": int(quantum_tokens),
                "yield_seconds": float(yield_seconds),
            },
            "last_tick_duration_ms": final_status.get("last_trace", {}).get(
                "elapsed_ms"
            ),
            "last_tick_token_delta": final_status.get("last_trace", {}).get(
                "trained_tokens"
            ),
            "last_tick_stage_timings_ms": {},
            "warm_ingestion": {
                "surface": "marulho_brain_source_buffer.v1",
                "accepted_tokens": int(feed.get("accepted_tokens", 0) or 0),
                "full_warm_ready": True,
            },
            "runtime_device": device_report,
            "column_transition_runtime": transition_report,
        }
    finally:
        brain.stop(timeout_seconds=1.0)


def run_continuous_runtime_quantum_ab(
    checkpoint: Path,
    *,
    output_path: Path,
    target_tokens: int = 256,
    timeout_seconds: float = 30.0,
    baseline_quantum_tokens: int = 8,
    candidate_quantum_tokens: int = 16,
) -> dict[str, Any]:
    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")
    if baseline_quantum_tokens <= 0:
        raise ValueError("baseline_quantum_tokens must be positive")
    if candidate_quantum_tokens <= 0:
        raise ValueError("candidate_quantum_tokens must be positive")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arm_specs = (
        ("baseline_quantum_a", int(baseline_quantum_tokens), 0.0),
        ("candidate_quantum_a", int(candidate_quantum_tokens), 0.0),
        ("candidate_quantum_b", int(candidate_quantum_tokens), 0.0),
        ("baseline_quantum_b", int(baseline_quantum_tokens), 0.0),
    )
    arms: list[dict[str, Any]] = []
    for name, quantum_tokens, yield_seconds in arm_specs:
        source_path = output_path.parent / f"{name}-continuous-quantum-source.txt"
        source_path.write_text(source_text_for_target(target_tokens), encoding="utf-8")
        arm = _run_arm(
            checkpoint=checkpoint,
            source_path=source_path,
            arm_root=output_path.parent / name,
            quantum_tokens=quantum_tokens,
            yield_seconds=yield_seconds,
            target_tokens=target_tokens,
            timeout_seconds=timeout_seconds,
        )
        arm["name"] = name
        arms.append(arm)

    baseline_quantum_values = [
        float(arm["tokens_per_second"])
        for arm in arms
        if str(arm["name"]).startswith("baseline_quantum") and arm.get("success")
    ]
    candidate_quantum_values = [
        float(arm["tokens_per_second"])
        for arm in arms
        if str(arm["name"]).startswith("candidate_quantum") and arm.get("success")
    ]
    baseline_quantum_mean = (
        float(statistics.fmean(baseline_quantum_values))
        if baseline_quantum_values
        else 0.0
    )
    candidate_quantum_mean = (
        float(statistics.fmean(candidate_quantum_values))
        if candidate_quantum_values
        else 0.0
    )
    report = {
        "surface": "continuous_runtime_quantum_ab.v2",
        "checkpoint": str(checkpoint),
        "scope": (
            "background_marulho_brain_loop_with_prefed_local_source_and_"
            "sequential_train_step"
        ),
        "claim_boundary": (
            "compares brain-loop quantum scheduling; neural token order, "
            "trainer math, source window, checkpoint, and CUDA executor are unchanged"
        ),
        "target_tokens_per_arm": int(target_tokens),
        "timeout_seconds_per_arm": float(timeout_seconds),
        "baseline_quantum_tokens": int(baseline_quantum_tokens),
        "candidate_quantum_tokens": int(candidate_quantum_tokens),
        "arms": arms,
        "baseline_quantum_mean_tokens_per_second": baseline_quantum_mean,
        "candidate_quantum_mean_tokens_per_second": candidate_quantum_mean,
        "candidate_over_baseline_quantum_speedup": (
            candidate_quantum_mean / max(baseline_quantum_mean, 1e-9)
        ),
        "success": bool(
            len(baseline_quantum_values) == 2
            and len(candidate_quantum_values) == 2
            and candidate_quantum_mean >= baseline_quantum_mean
        ),
    }
    write_json_report_with_readme(output_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-tokens", type=int, default=256)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--baseline-quantum-tokens", type=int, default=8)
    parser.add_argument("--candidate-quantum-tokens", type=int, default=16)
    args = parser.parse_args()
    report = run_continuous_runtime_quantum_ab(
        args.checkpoint,
        output_path=args.output,
        target_tokens=args.target_tokens,
        timeout_seconds=args.timeout_seconds,
        baseline_quantum_tokens=args.baseline_quantum_tokens,
        candidate_quantum_tokens=args.candidate_quantum_tokens,
    )
    print(json.dumps(report, indent=2))
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
