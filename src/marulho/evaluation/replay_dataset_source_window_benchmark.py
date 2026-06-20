from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import tempfile
import time
import tracemalloc
from typing import Any, Mapping, Sequence

from marulho.config.model_config import MarulhoConfig
from marulho.service.manager import MarulhoServiceManager
from marulho.service.runtime_evidence import MAX_REPLAY_DATASET_EXPORT_LIMIT
from marulho.training.checkpointing import save_trainer_checkpoint
from marulho.training.model import MarulhoModel
from marulho.training.trainer import MarulhoTrainer


def _process_rss_mib() -> float | None:
    try:
        import psutil  # type: ignore
    except Exception:
        return None
    try:
        return float(psutil.Process().memory_info().rss) / (1024.0 * 1024.0)
    except Exception:
        return None


def _cuda_report() -> dict[str, Any]:
    try:
        import torch
    except Exception:
        return {"torch_available": False, "cuda_available": False, "gpu_used": False}
    available = bool(torch.cuda.is_available())
    report: dict[str, Any] = {
        "torch_available": True,
        "cuda_available": available,
        "gpu_used": False,
    }
    if available:
        try:
            report["device_name"] = torch.cuda.get_device_name(0)
            report["memory_allocated_mib"] = round(
                float(torch.cuda.memory_allocated(0)) / (1024.0 * 1024.0),
                3,
            )
            report["memory_reserved_mib"] = round(
                float(torch.cuda.memory_reserved(0)) / (1024.0 * 1024.0),
                3,
            )
        except Exception as exc:
            report["cuda_query_error"] = str(exc)
    return report


def _stats(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean_ms": None, "median_ms": None, "max_ms": None}
    return {
        "count": int(len(values)),
        "mean_ms": round(float(statistics.fmean(values)), 6),
        "median_ms": round(float(statistics.median(values)), 6),
        "max_ms": round(float(max(values)), 6),
        "samples_ms": [round(float(value), 6) for value in values],
    }


def _runtime_trace_payload(index: int) -> dict[str, Any]:
    minute = (index // 60) % 60
    second = index % 60
    return {
        "episode_id": f"episode-{index:04d}",
        "trace_id": f"trace-{index:04d}",
        "operation": "respond",
        "status": "succeeded",
        "created_at": f"2026-06-20T00:{minute:02d}:{second:02d}+00:00",
        "completed_at": f"2026-06-20T00:{minute:02d}:{(second + 1) % 60:02d}+00:00",
        "latency_ms": 1.0,
        "request": {"query_text": f"bounded replay question {index}", "top_k_memories": 4},
        "prediction": {"proposed_answer": f"bounded replay answer {index}"},
        "action": {"action_type": "respond"},
        "actual_output": {"response_text": f"bounded replay answer {index}"},
        "verification": {"status": "verified", "success": True, "confidence": 0.9},
        "feedback": [
            {
                "feedback_id": f"feedback-{index:04d}",
                "created_at": f"2026-06-20T01:{minute:02d}:{second:02d}+00:00",
                "target_type": "runtime_episode",
                "target_id": f"episode-{index:04d}",
                "verdict": "verified",
                "applied_status": "verified",
                "confidence": 0.9,
                "summary": "Synthetic evaluator verified the bounded replay fixture.",
            }
        ],
        "corrected_output": {"response_text": f"bounded replay answer {index}"},
        "provenance": "verified",
    }


def _replay_sample_payload(
    index: int,
    *,
    trace_count: int,
    selected_candidates_per_sample: int,
) -> dict[str, Any]:
    target_index = index % max(1, int(trace_count))
    selected_candidates = [
        {
            "candidate_id": f"candidate-{index:04d}-{candidate:02d}",
            "target_type": "runtime_episode",
            "target_id": f"episode-{target_index:04d}",
        }
        for candidate in range(max(0, int(selected_candidates_per_sample)))
    ]
    return {
        "schema_version": 1,
        "replay_sample_id": f"sample-{index:04d}",
        "execution_id": f"execute-{index:04d}",
        "created_at": f"2026-06-20T02:{(index // 60) % 60:02d}:{index % 60:02d}+00:00",
        "mode": "execute",
        "status": "recorded",
        "operator_id": "benchmark",
        "selected_candidates": selected_candidates,
        "safety_flags": {
            "training_started": False,
            "sleep_started": False,
            "memory_mutated": False,
            "external_calls_made": False,
        },
    }


def _build_checkpoint(
    root: Path,
    *,
    trace_count: int,
    sample_count: int,
    selected_candidates_per_sample: int,
) -> Path:
    cfg = MarulhoConfig(
        n_columns=4,
        column_latent_dim=8,
        bootstrap_tokens=0,
        memory_capacity=64,
        eta_competitive=0.05,
        eta_decay=0.0,
        input_weight_blend=0.0,
        enable_context_layer=True,
        enable_binding_layer=True,
    )
    model = MarulhoModel(cfg)
    trainer = MarulhoTrainer(model, cfg)
    metadata = {
        "benchmark": "replay_dataset_source_window",
        "service_state": {
            "terminus_runtime": {
                "runtime_episode_traces": [
                    _runtime_trace_payload(index) for index in range(max(0, int(trace_count)))
                ],
                "replay_sample_history": [
                    _replay_sample_payload(
                        index,
                        trace_count=max(1, int(trace_count)),
                        selected_candidates_per_sample=selected_candidates_per_sample,
                    )
                    for index in range(max(0, int(sample_count)))
                ],
            }
        },
    }
    return save_trainer_checkpoint(root / "initial.pt", trainer, metadata=metadata)


def _diagnostic_full_retained_preview(
    traces: Sequence[Mapping[str, Any]],
    replay_samples: Sequence[Mapping[str, Any]],
    *,
    limit: int,
    endpoint: str | None,
) -> dict[str, Any]:
    normalized_endpoint = None
    if endpoint is not None:
        normalized_endpoint = " ".join(str(endpoint).split()).strip().lower().lstrip("/")

    links: dict[tuple[str, str], bool] = {}
    selected_candidate_count = 0
    for record in replay_samples:
        candidates = record.get("selected_candidates")
        if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
            continue
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                continue
            selected_candidate_count += 1
            target_type = str(candidate.get("target_type", "") or "").strip()
            target_id = str(candidate.get("target_id", "") or "").strip()
            if target_type and target_id:
                links[(target_type, target_id)] = True

    selected_ids: list[str] = []
    linked_count = 0
    for trace in traces:
        operation = str(trace.get("operation", "") or "unknown").strip().lower() or "unknown"
        if normalized_endpoint is not None and normalized_endpoint not in {
            operation,
            f"/{operation}".lower(),
        }:
            continue
        episode_id = str(trace.get("episode_id", "") or "")
        selected_ids.append(episode_id)
        if links.get(("runtime_episode", episode_id)):
            linked_count += 1
        if len(selected_ids) >= max(1, int(limit)):
            break

    return {
        "surface": "diagnostic_full_retained_replay_dataset_preview.v1",
        "trace_records_scanned": int(len(traces)),
        "replay_sample_records_scanned": int(len(replay_samples)),
        "selected_candidate_records_scanned": int(selected_candidate_count),
        "selected_target_ids": selected_ids,
        "linked_selected_count": int(linked_count),
    }


def _mean(values: Sequence[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    limit = min(MAX_REPLAY_DATASET_EXPORT_LIMIT, max(1, int(args.limit)))
    trace_count = max(1, int(args.trace_count))
    sample_count = max(1, int(args.sample_count))
    selected_candidates_per_sample = max(1, int(args.selected_candidates_per_sample))
    runs = max(1, int(args.runs))
    endpoint = args.endpoint

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        checkpoint = _build_checkpoint(
            root,
            trace_count=trace_count,
            sample_count=sample_count,
            selected_candidates_per_sample=selected_candidates_per_sample,
        )
        manager = MarulhoServiceManager(
            checkpoint,
            trace_history_limit=limit,
            trace_dir=root / "traces",
            env_root=root / "env",
        )
        try:
            with manager._lock:
                retained_traces = manager._interaction_pipeline.runtime_episode_traces()
                retained_replay_samples = [deepcopy(item) for item in list(manager._replay_sample_history)]

            diagnostic_latencies: list[float] = []
            diagnostic: dict[str, Any] = {}
            for _ in range(runs):
                started = time.perf_counter()
                diagnostic = _diagnostic_full_retained_preview(
                    retained_traces,
                    retained_replay_samples,
                    limit=limit,
                    endpoint=endpoint,
                )
                diagnostic_latencies.append((time.perf_counter() - started) * 1000.0)

            rss_before = _process_rss_mib()
            bounded_latencies: list[float] = []
            last_dataset: dict[str, Any] = {}
            tracemalloc.start()
            for _ in range(runs):
                started = time.perf_counter()
                last_dataset = manager.runtime_facade.replay_dataset_preview(
                    limit=limit,
                    endpoint=endpoint,
                )
                bounded_latencies.append((time.perf_counter() - started) * 1000.0)
            traced_current, traced_peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            rss_after = _process_rss_mib()
        finally:
            manager.close()

    source_window = dict(last_dataset.get("source_window") or {})
    link_window = dict(source_window.get("replay_sample_link_source_window") or {})
    bounded_selected_ids = [
        str(item.get("target_id", "") or "")
        for item in list(last_dataset.get("items") or [])
        if isinstance(item, Mapping)
    ]
    diagnostic_selected_ids = list(diagnostic.get("selected_target_ids") or [])
    bounded_linked_count = sum(
        1
        for item in list(last_dataset.get("items") or [])
        if isinstance(item, Mapping)
        and isinstance(item.get("replay_sample_linkage"), Mapping)
        and bool(item["replay_sample_linkage"].get("selected"))
    )
    trace_reduction = float(diagnostic.get("trace_records_scanned", 0) or 0) / max(
        1.0,
        float(source_window.get("source_window_count", 0) or 0),
    )
    replay_sample_reduction = float(diagnostic.get("replay_sample_records_scanned", 0) or 0) / max(
        1.0,
        float(link_window.get("source_window_count", 0) or 0),
    )
    selected_candidate_reduction = float(
        diagnostic.get("selected_candidate_records_scanned", 0) or 0
    ) / max(1.0, float(link_window.get("selected_candidate_window_count", 0) or 0))

    quality = {
        "metric": "bounded_preview_matches_full_retained_history_for_returned_window",
        "diagnostic_selected_target_ids": diagnostic_selected_ids,
        "bounded_selected_target_ids": bounded_selected_ids,
        "selected_target_ids_match": diagnostic_selected_ids == bounded_selected_ids,
        "diagnostic_linked_selected_count": int(diagnostic.get("linked_selected_count", 0) or 0),
        "bounded_linked_selected_count": int(bounded_linked_count),
        "link_coverage_matches": int(diagnostic.get("linked_selected_count", 0) or 0)
        == int(bounded_linked_count),
    }
    pass_checks = {
        "dataset_surface": last_dataset.get("export_kind") == "terminus_replay_dataset_preview",
        "source_window_surface": source_window.get("surface")
        == "bounded_replay_dataset_preview_source_window.v1",
        "link_window_surface": link_window.get("surface")
        == "bounded_replay_dataset_sample_link_source_window.v1",
        "selected_target_ids_match": bool(quality["selected_target_ids_match"]),
        "link_coverage_matches": bool(quality["link_coverage_matches"]),
        "source_window_bounded": int(source_window.get("source_window_count", 0) or 0)
        <= int(source_window.get("source_window_limit", 0) or 0),
        "replay_sample_window_bounded": int(link_window.get("source_window_count", 0) or 0)
        <= int(link_window.get("source_window_limit", 0) or 0),
        "trace_work_reduced": trace_reduction >= 1.0,
        "replay_sample_work_reduced": replay_sample_reduction >= 2.0,
        "selected_candidate_work_reduced": selected_candidate_reduction >= 2.0,
        "runs_live_tick_false": source_window.get("runs_live_tick") is False
        and link_window.get("runs_live_tick") is False,
        "runs_every_token_false": source_window.get("runs_every_token") is False
        and link_window.get("runs_every_token") is False,
        "no_training_or_plasticity": source_window.get("trains_adapter") is False
        and link_window.get("trains_adapter") is False
        and source_window.get("applies_plasticity") is False
        and link_window.get("applies_plasticity") is False,
        "archival_metadata_cpu": source_window.get("archival_storage_device") == "cpu"
        and link_window.get("archival_storage_device") == "cpu"
        and source_window.get("gpu_resident_archival_metadata") is False
        and link_window.get("gpu_resident_archival_metadata") is False,
    }

    return {
        "surface": "bounded_replay_dataset_source_window_benchmark.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pass": all(pass_checks.values()),
        "pass_checks": pass_checks,
        "input": {
            "trace_count": int(trace_count),
            "sample_count": int(sample_count),
            "selected_candidates_per_sample": int(selected_candidates_per_sample),
            "limit": int(limit),
            "endpoint": endpoint,
            "runs": int(runs),
        },
        "quality": quality,
        "latency_ms": {
            "diagnostic_full_retained_preview": _stats(diagnostic_latencies),
            "bounded_source_window_preview": _stats(bounded_latencies),
            "bounded_mean_cost_ms": round(_mean(bounded_latencies), 6),
            "diagnostic_mean_ms": round(_mean(diagnostic_latencies), 6),
        },
        "work_reduction": {
            "trace_record_reduction": round(float(trace_reduction), 6),
            "replay_sample_record_reduction": round(float(replay_sample_reduction), 6),
            "selected_candidate_record_reduction": round(float(selected_candidate_reduction), 6),
        },
        "diagnostic": diagnostic,
        "dataset_summary": {
            "count": int(last_dataset.get("count", 0) or 0),
            "positive_count": int(last_dataset.get("positive_count", 0) or 0),
            "negative_count": int(last_dataset.get("negative_count", 0) or 0),
            "safety_flags": dict(last_dataset.get("safety_flags") or {}),
        },
        "source_window": source_window,
        "resource_behavior": {
            "process_rss_before_mib": None if rss_before is None else round(float(rss_before), 3),
            "process_rss_after_mib": None if rss_after is None else round(float(rss_after), 3),
            "process_rss_delta_mib": None
            if rss_before is None or rss_after is None
            else round(float(rss_after - rss_before), 3),
            "python_tracemalloc_current_mib": round(float(traced_current) / (1024.0 * 1024.0), 3),
            "python_tracemalloc_peak_mib": round(float(traced_peak) / (1024.0 * 1024.0), 3),
            "cuda": _cuda_report(),
        },
        "device_placement": {
            "archival_storage_device": "cpu",
            "source_window_selection_device": "cpu",
            "active_replay_computation_device": "cpu",
            "gpu_used": False,
            "gpu_resident_archival_metadata": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark bounded replay dataset source windows.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace-count", type=int, default=64)
    parser.add_argument("--sample-count", type=int, default=256)
    parser.add_argument("--selected-candidates-per-sample", type=int, default=16)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--endpoint", type=str, default="respond")
    parser.add_argument("--runs", type=int, default=7)
    args = parser.parse_args()

    report = run_benchmark(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
