from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REPORTS_ROOT = REPO_ROOT / "reports"
MAINTAINED_REPRESENTATION = "order_weighted_ascii"
MECHANISM_REPORT_PATTERNS = (
    "phase7_mechanism_validation_baseline_*",
    "phase7_mechanism_validation_smoke*",
    "refactor_mechanism_validation_smoke",
    "refactor_stage0_smoke",
)
EMERGENCE_REPORT_PATTERNS = (
    "phase7_emergence_evaluation_*",
    "refactor_emergence_evaluation_smoke",
)
REPRESENTATION_REPORT_PATTERNS = (
    "phase7_representation_baseline_*",
    "phase7_representation_smoke*",
    "refactor_representation_smoke",
)
CONTEXTUAL_ROUTING_REPORT_PATTERNS = (
    "phase7_contextual_routing_smoke*",
    "refactor_contextual_routing_smoke",
    "refactor_phase3_smoke",
)
HIERARCHICAL_SCALE_REPORT_PATTERNS = (
    "phase7_routing_scale_baseline_*",
    "phase7_routing_scale_smoke*",
    "refactor_hierarchical_scale_smoke",
    "refactor_phase4_smoke",
)
KNOWLEDGE_GAP_REPORT_PATTERNS = (
    "phase6_autonomy_baseline_feedback_tiebreak_*",
    "phase6_autonomy_baseline_feedback_*",
    "phase6_autonomy_baseline_rerun_*",
    "refactor_autonomy_smoke",
)
ACQUISITION_REPORT_PATTERNS = (
    "refactor_autonomy_acquisition_hf_allocation_rng_*",
    "refactor_autonomy_acquisition_open_web_scout_projected_*",
    "refactor_autonomy_acquisition_open_web_scout_confirm_*",
)
BENCHMARK_CATALOG: tuple[tuple[str, str, str, Any], ...] = (
    (
        "mechanism_validation",
        "Mechanism validation",
        "Competitive self-organization, surprise stabilization, and early predictive error trends.",
        "_load_mechanism_validation",
    ),
    (
        "emergence_evaluation",
        "Emergence evaluation",
        "Label-free temporal coherence, compositionality, novelty coverage, grounding, and forgetting diagnostics.",
        "_load_emergence_evaluation",
    ),
    (
        "representation_benchmark",
        "Representation benchmark",
        "Comparison of maintained and ablation input representations under the shared routing contract.",
        "_load_representation_benchmark",
    ),
    (
        "memory_consolidation",
        "Memory consolidation",
        "Replay, tagging, PRP recruitment, and post-interference retention.",
        "_load_memory_consolidation",
    ),
    (
        "contextual_routing",
        "Contextual routing",
        "Multiscale context, binding, and explicit polysemy separation diagnostics.",
        "_load_contextual_routing",
    ),
    (
        "hierarchical_scale",
        "Hierarchical scale",
        "Sharded routing quality, latency, throughput, and scale balance.",
        "_load_hierarchical_scale",
    ),
    (
        "knowledge_gap_seeking",
        "Knowledge-gap seeking",
        "Closed-loop gap reduction under active information-seeking versus baseline selection.",
        "_load_knowledge_gap_seeking",
    ),
    (
        "source_acquisition",
        "Source acquisition",
        "Maintained HF candidate-bank allocation benchmark contrasting active acquisition against round-robin.",
        "_load_source_acquisition",
    ),
)


def load_benchmark_reports(*, reports_root: str | Path | None = None, max_points: int = 120) -> dict[str, Any]:
    root = Path(reports_root) if reports_root is not None else DEFAULT_REPORTS_ROOT
    benchmarks: list[dict[str, Any]] = []

    for benchmark_id, label, description, loader_name in BENCHMARK_CATALOG:
        loader = globals()[loader_name]
        payload = loader(root, max_points=max_points)
        if payload is not None:
            payload["benchmark_id"] = benchmark_id
            payload["label"] = label
            payload["description"] = description
            benchmarks.append(payload)

    return {
        "reports_root": _repo_relative(root),
        "benchmarks": benchmarks,
    }


def _load_mechanism_validation(root: Path, *, max_points: int) -> dict[str, Any] | None:
    summary_path = _latest_matching_summary(root, MECHANISM_REPORT_PATTERNS)
    metrics_path = None if summary_path is None else summary_path.parent / "metrics.csv"
    if summary_path is None or metrics_path is None:
        return None
    summary = _load_json(summary_path)
    metrics_rows = _load_csv_rows(metrics_path)
    if summary is None or not metrics_rows:
        return None

    series = _sample_points(
        [
            {
                "token": _int_value(row, "token"),
                "drift": _float_value(row, "drift"),
                "surprise": _float_value(row, "surprise"),
                "recon_error": _float_value(row, "recon_error"),
                "pred_error": _float_value(row, "pred_error"),
                "drift_floor": _float_value(row, "drift_floor"),
                "sleep_events_total": _int_value(row, "sleep_events_total"),
            }
            for row in metrics_rows
        ],
        max_points=max_points,
    )

    return {
        "summary_path": _repo_relative(summary_path),
        "metrics_path": _repo_relative(metrics_path),
        "summary": {
            "drift_mean": _float_value(summary, "drift_mean"),
            "surprise_mean": _float_value(summary, "surprise_mean"),
            "sparsity_mean": _float_value(summary, "sparsity_mean"),
            "recon_error_mean": _float_value(summary, "recon_error_mean"),
            "winner_entropy_bits": _float_value(summary, "winner_entropy_bits"),
            "winner_max_share": _float_value(summary, "winner_max_share"),
        },
        "series": series,
    }


def _load_emergence_evaluation(root: Path, *, max_points: int) -> dict[str, Any] | None:
    del max_points

    summary_path = _latest_matching_summary(root, EMERGENCE_REPORT_PATTERNS)
    if summary_path is None:
        return None
    summary = _load_json(summary_path)
    if summary is None:
        return None

    metrics = summary.get("metrics", {})
    novelty = summary.get("novelty_coverage", {})
    gate = summary.get("emergence_evaluation_gate", {})
    meaning_grounding = summary.get("meaning_grounding", {})
    return {
        "summary_path": _repo_relative(summary_path),
        "summary": {
            "temporal_coherence_mean": _float_value(metrics, "temporal_coherence_mean"),
            "grounded_query_accuracy": _float_value(metrics, "grounded_query_accuracy"),
            "compositional_query_accuracy": _float_value(metrics, "compositional_query_accuracy"),
            "phase_a_interference_retention": _float_value(metrics, "phase_a_interference_retention"),
            "phase_a_final_retention": _float_value(metrics, "phase_a_final_retention"),
            "supported_topic_coverage": _float_value(metrics, "supported_topic_coverage"),
            "answerability_growth_mean": _float_value(metrics, "answerability_growth_mean"),
            "meaning_grounding_gate_pass": bool(meaning_grounding.get("mixed_world", {}).get("gate_pass", False)),
            "gate_pass": bool(gate.get("pass", False)),
        },
        "coverage_summary": {
            "supportedTopicCoverage": _float_value(novelty, "supported_topic_coverage"),
            "answerabilityGrowthMean": _float_value(novelty, "answerability_growth_mean"),
            "revisitRetentionRate": _float_value(novelty, "revisit_retention_rate"),
            "uniqueAcquiredSourceCount": _int_value(novelty, "unique_acquired_source_count"),
        },
    }


def _load_representation_benchmark(root: Path, *, max_points: int) -> dict[str, Any] | None:
    del max_points

    summary_path = _latest_matching_summary(root, REPRESENTATION_REPORT_PATTERNS)
    if summary_path is None:
        return None
    summary = _load_json(summary_path)
    if summary is None:
        return None

    results = summary.get("results", [])
    if not results:
        return None

    comparison_series = []
    best_entry: dict[str, Any] | None = None
    maintained_entry: dict[str, Any] | None = None
    for result in results:
        representation = str(result.get("representation", "representation"))
        hecsn = result.get("hecsn_competitive_only", {})
        baseline = result.get("online_kmeans", {})
        completion = result.get("completion_coherence", {})
        entry = {
            "representation": representation,
            "label": _representation_label(representation),
            "featureDim": _int_value(result, "feature_dim"),
            "hecsnSilhouette": _float_value(hecsn, "silhouette"),
            "baselineSilhouette": _float_value(baseline, "silhouette"),
            "hecsnDaviesBouldin": _float_value(hecsn, "davies_bouldin"),
            "baselineDaviesBouldin": _float_value(baseline, "davies_bouldin"),
            "completionMargin": _float_value(completion, "mean_margin"),
            "completionSuccessRate": _float_value(completion, "success_rate"),
        }
        comparison_series.append(entry)

        if best_entry is None or entry["hecsnSilhouette"] > best_entry["hecsnSilhouette"]:
            best_entry = entry
        if representation == MAINTAINED_REPRESENTATION:
            maintained_entry = entry

    data_setup = summary.get("data_setup", {})
    benchmark_setup = summary.get("benchmark_setup", {})
    return {
        "summary_path": _repo_relative(summary_path),
        "summary": {
            "source": str(data_setup.get("source", "")),
            "source_type": str(data_setup.get("source_type", "")),
            "train_windows": _int_value(data_setup, "train_windows"),
            "eval_windows": _int_value(data_setup, "eval_windows"),
            "probe_samples": _int_value(benchmark_setup, "probe_samples"),
            "maintained_default": MAINTAINED_REPRESENTATION,
            "maintained_default_label": _representation_label(MAINTAINED_REPRESENTATION),
            "maintained_hecsn_silhouette": None if maintained_entry is None else maintained_entry["hecsnSilhouette"],
            "maintained_baseline_silhouette": None if maintained_entry is None else maintained_entry["baselineSilhouette"],
            "best_representation": None if best_entry is None else best_entry["representation"],
            "best_representation_label": None if best_entry is None else best_entry["label"],
            "best_hecsn_silhouette": None if best_entry is None else best_entry["hecsnSilhouette"],
        },
        "comparison_series": comparison_series,
    }


def _load_memory_consolidation(root: Path, *, max_points: int) -> dict[str, Any] | None:
    summary_path = _first_existing_path(
        root / "phase7_fragility_consolidation_smoke" / "summary.json",
        root / "refactor_memory_consolidation_smoke" / "summary.json",
        root / "refactor_phase2_smoke" / "summary.json",
    )
    if summary_path is None:
        return None
    summary = _load_json(summary_path)
    if summary is None:
        return None

    metrics = summary.get("metrics", {})
    gate = summary.get("memory_consolidation_gate", summary.get("phase2_gate", {}))
    return {
        "summary_path": _repo_relative(summary_path),
        "summary": {
            "task_a_recovery_delta": _float_value(metrics, "task_a_recovery_delta"),
            "task_a_overlap_after_consolidation": _float_value(metrics, "task_a_overlap_after_consolidation"),
            "task_b_overlap_after_consolidation": _float_value(metrics, "task_b_overlap_after_consolidation"),
            "task_a_relative_degradation_after_consolidation": _float_value(metrics, "task_a_relative_degradation_after_consolidation"),
            "mean_capture_tag_before": _float_value(summary.get("consolidation", {}), "mean_capture_tag_before"),
            "mean_capture_tag_after": _float_value(summary.get("consolidation", {}), "mean_capture_tag_after"),
            "mean_prp_level_before": _float_value(summary.get("consolidation", {}), "mean_prp_level_before"),
            "mean_prp_level_after": _float_value(summary.get("consolidation", {}), "mean_prp_level_after"),
            "mean_capture_strength_before": _float_value(summary.get("consolidation", {}), "mean_capture_strength_before"),
            "mean_capture_strength_after": _float_value(summary.get("consolidation", {}), "mean_capture_strength_after"),
            "mean_consolidation_level_before": _float_value(summary.get("consolidation", {}), "mean_consolidation_level_before"),
            "mean_consolidation_level_after": _float_value(summary.get("consolidation", {}), "mean_consolidation_level_after"),
            "gate_pass": bool(gate.get("pass", False)),
        },
        "reconstruction_series": [
            {
                "step": "After task A",
                "taskA": _float_value(metrics, "task_a_recon_after_a"),
                "taskB": None,
            },
            {
                "step": "Before task B",
                "taskA": None,
                "taskB": _float_value(metrics, "task_b_recon_before_b"),
            },
            {
                "step": "After task B",
                "taskA": _float_value(metrics, "task_a_recon_after_b"),
                "taskB": _float_value(metrics, "task_b_recon_after_b"),
            },
            {
                "step": "After consolidation",
                "taskA": _float_value(metrics, "task_a_recon_after_consolidation"),
                "taskB": _float_value(metrics, "task_b_recon_after_consolidation"),
            },
        ],
        "overlap_series": [
            {
                "step": "After task B",
                "taskAOverlap": _float_value(metrics, "task_a_overlap_after_b"),
                "taskBOverlap": _float_value(metrics, "task_b_overlap_after_b"),
            },
            {
                "step": "After consolidation",
                "taskAOverlap": _float_value(metrics, "task_a_overlap_after_consolidation"),
                "taskBOverlap": _float_value(metrics, "task_b_overlap_after_consolidation"),
            },
        ],
    }


def _load_contextual_routing(root: Path, *, max_points: int) -> dict[str, Any] | None:
    summary_path = _latest_matching_summary(root, CONTEXTUAL_ROUTING_REPORT_PATTERNS)
    if summary_path is None:
        return None
    summary = _load_json(summary_path)
    if summary is None:
        return None

    diagnostics = summary.get("training_diagnostics", {})
    metrics = summary.get("contextual_routing_metrics", summary.get("phase3_metrics", {}))
    gate = summary.get("contextual_routing_gate", summary.get("phase3_gate", {}))
    return {
        "summary_path": _repo_relative(summary_path),
        "summary": {
            "task_a_recon_error": _float_value(metrics, "task_a_recon_error"),
            "task_b_recon_error": _float_value(metrics, "task_b_recon_error"),
            "context_state_separation": _float_value(metrics, "context_state_separation"),
            "probe_winner_switch_rate": _float_value(metrics, "probe_winner_switch_rate"),
            "probe_mean_assembly_distance": _float_value(metrics, "probe_mean_assembly_distance"),
            "bank_polysemy_accuracy": _float_value(metrics, "bank_polysemy_accuracy"),
            "bank_polysemy_signature_margin": _float_value(metrics, "bank_polysemy_signature_margin"),
            "bank_polysemy_winner_sequence_difference_rate": _float_value(metrics, "bank_polysemy_winner_sequence_difference_rate"),
            "gate_pass": bool(gate.get("pass", False)),
        },
        "metric_series": [
            {
                "label": "Context separation",
                "value": _float_value(metrics, "context_state_separation"),
            },
            {
                "label": "Winner switch rate",
                "value": _float_value(metrics, "probe_winner_switch_rate"),
            },
            {
                "label": "Assembly distance",
                "value": _float_value(metrics, "probe_mean_assembly_distance"),
            },
            {
                "label": "B3 bank accuracy",
                "value": _float_value(metrics, "bank_polysemy_accuracy"),
            },
            {
                "label": "B3 signature margin",
                "value": _float_value(metrics, "bank_polysemy_signature_margin"),
            },
            {
                "label": "B3 winner-sequence diff",
                "value": _float_value(metrics, "bank_polysemy_winner_sequence_difference_rate"),
            },
        ],
        "regulator_series": [
            {
                "label": "Dopamine",
                "value": _float_value(diagnostics, "mean_dopamine"),
            },
            {
                "label": "Serotonin",
                "value": _float_value(diagnostics, "mean_serotonin"),
            },
            {
                "label": "Acetylcholine",
                "value": _float_value(diagnostics, "mean_acetylcholine"),
            },
            {
                "label": "Norepinephrine",
                "value": _float_value(diagnostics, "mean_norepinephrine"),
            },
            {
                "label": "Context gain",
                "value": _float_value(diagnostics, "mean_context_gain"),
            },
        ],
    }


def _load_hierarchical_scale(root: Path, *, max_points: int) -> dict[str, Any] | None:
    summary_path = _latest_matching_summary(root, HIERARCHICAL_SCALE_REPORT_PATTERNS)
    if summary_path is None:
        return None
    summary = _load_json(summary_path)
    if summary is None:
        return None

    routing = summary.get("routing_metrics", {})
    training = summary.get("training_diagnostics", {})
    sharding = summary.get("sharding", {})
    gate = summary.get("hierarchical_scale_gate", summary.get("phase4_gate", {}))
    shard_sizes = sharding.get("index_shard_sizes", [])
    primary_counts = sharding.get("primary_query_shard_counts", [])
    winner_counts = sharding.get("winner_shard_counts", [])
    shard_series = []
    for index, size in enumerate(shard_sizes):
        shard_series.append(
            {
                "shard": f"Shard {index}",
                "size": int(size),
                "primaryQueries": int(primary_counts[index]) if index < len(primary_counts) else 0,
                "winnerCount": int(winner_counts[index]) if index < len(winner_counts) else 0,
            }
        )

    return {
        "summary_path": _repo_relative(summary_path),
        "summary": {
            "eval_recon_error": _float_value(training, "eval_recon_error"),
            "recall_at_k": _float_value(routing, "recall_at_k"),
            "top1_recall": _float_value(routing, "top1_recall"),
            "mean_latency_ms": _float_value(routing, "mean_latency_ms"),
            "p95_latency_ms": _float_value(routing, "p95_latency_ms"),
            "throughput_chars_per_sec": _float_value(training, "throughput_chars_per_sec"),
            "gate_pass": bool(gate.get("pass", False)),
        },
        "routing_series": [
            {
                "label": "Recall@k",
                "value": _float_value(routing, "recall_at_k"),
            },
            {
                "label": "Top-1 recall",
                "value": _float_value(routing, "top1_recall"),
            },
            {
                "label": "Shard balance",
                "value": _float_value(sharding, "index_shard_balance_ratio"),
            },
            {
                "label": "Winner coverage",
                "value": _float_value(sharding, "winner_shard_coverage"),
            },
        ],
        "latency_series": [
            {
                "label": "Mean latency",
                "value": _float_value(routing, "mean_latency_ms"),
            },
            {
                "label": "P95 latency",
                "value": _float_value(routing, "p95_latency_ms"),
            },
        ],
        "shard_series": shard_series,
    }


def _load_knowledge_gap_seeking(root: Path, *, max_points: int) -> dict[str, Any] | None:
    summary_path = _latest_matching_summary(root, KNOWLEDGE_GAP_REPORT_PATTERNS)
    if summary_path is None:
        return None
    summary = _load_json(summary_path)
    if summary is None:
        return None

    active = summary.get("policy_results", {}).get("active", {})
    round_robin = summary.get("policy_results", {}).get("round_robin", {})
    sources = sorted(
        set(active.get("final_gap_by_source", {}).keys()) | set(round_robin.get("final_gap_by_source", {}).keys())
    )

    gap_by_source = [
        {
            "source": source,
            "activeGap": _float_value(active.get("final_gap_by_source", {}), source),
            "roundRobinGap": _float_value(round_robin.get("final_gap_by_source", {}), source),
            "activeDiagnosticGap": _float_value(active.get("final_diagnostic_gap_score_by_source", {}), source),
            "roundRobinDiagnosticGap": _float_value(round_robin.get("final_diagnostic_gap_score_by_source", {}), source),
            "activeInfoGain": _float_value(active.get("final_info_gain_by_source", {}), source),
            "roundRobinInfoGain": _float_value(round_robin.get("final_info_gain_by_source", {}), source),
            "activeVisits": _int_value(active.get("source_visit_counts", {}), source),
            "roundRobinVisits": _int_value(round_robin.get("source_visit_counts", {}), source),
        }
        for source in sources
    ]

    episode_history = _sample_points(
        [
            {
                "episode": int(item.get("episode_index", index + 1)),
                "gapReduction": _float_value(item, "selected_gap_reduction"),
                "gapScoreReduction": _float_value(item, "selected_gap_score_reduction"),
            }
            for index, item in enumerate(active.get("episode_history", []))
            if item.get("phase") == "seek"
        ],
        max_points=max_points,
    )

    return {
        "summary_path": _repo_relative(summary_path),
        "summary": {
            "active_mean_gap": _float_value(active, "final_mean_gap"),
            "round_robin_mean_gap": _float_value(round_robin, "final_mean_gap"),
            "active_mean_gap_score": _float_value(active, "final_mean_gap_score"),
            "round_robin_mean_gap_score": _float_value(round_robin, "final_mean_gap_score"),
            "active_mean_info_gain": _float_value(active, "final_mean_info_gain"),
            "round_robin_mean_info_gain": _float_value(round_robin, "final_mean_info_gain"),
            "active_mean_selected_gap_reduction": _float_value(active, "mean_selected_gap_reduction"),
        },
        "gap_by_source": gap_by_source,
        "episode_history": episode_history,
    }


def _load_source_acquisition(root: Path, *, max_points: int) -> dict[str, Any] | None:
    summary_path = _latest_matching_summary(root, ACQUISITION_REPORT_PATTERNS)
    if summary_path is None:
        return None

    summary = _load_json(summary_path)
    if summary is None:
        return None

    benchmark_id = summary_path.parent.name
    active = summary.get("policy_results", {}).get("active", {})
    round_robin = summary.get("policy_results", {}).get("round_robin", {})
    scout_commit = summary.get("policy_results", {}).get("scout_commit", {})
    runtime_scope = summary.get("runtime_scope", {})
    sources = sorted(
        set(active.get("final_candidate_gap_by_source", {}).keys())
        | set(round_robin.get("final_candidate_gap_by_source", {}).keys())
        | set(scout_commit.get("final_candidate_gap_by_source", {}).keys())
    )

    gap_by_source = [
        {
            "source": source,
            "activeGap": _float_value(active.get("final_candidate_gap_by_source", {}), source),
            "roundRobinGap": _float_value(round_robin.get("final_candidate_gap_by_source", {}), source),
            "activeDiagnosticGap": _float_value(active.get("final_candidate_diagnostic_gap_by_source", {}), source),
            "roundRobinDiagnosticGap": _float_value(round_robin.get("final_candidate_diagnostic_gap_by_source", {}), source),
            "activeInfoGain": _float_value(active.get("final_candidate_info_gain_by_source", {}), source),
            "roundRobinInfoGain": _float_value(round_robin.get("final_candidate_info_gain_by_source", {}), source),
            **(
                {
                    "scoutGap": _float_value(scout_commit.get("final_candidate_gap_by_source", {}), source),
                    "scoutDiagnosticGap": _float_value(scout_commit.get("final_candidate_diagnostic_gap_by_source", {}), source),
                    "scoutInfoGain": _float_value(scout_commit.get("final_candidate_info_gain_by_source", {}), source),
                }
                if scout_commit
                else {}
            ),
        }
        for source in sources
    ]

    history_policy = "scout_commit" if scout_commit else "active"
    history_source = scout_commit if scout_commit else active
    lookahead_history = []
    selected_projected_mean_gaps = []
    selected_projected_max_gaps = []
    lookahead_advantages = []

    for index, item in enumerate(scout_commit.get("acquisition_history", [])):
        slot = int(item.get("slot", index + 1))
        selected_source = str(item.get("selected_source", "candidate"))
        slot_selected_mean = None
        slot_rejected_means = []

        for action in item.get("scout_actions", []):
            projected_mean = _optional_float_value(action, "projected_final_mean_candidate_gap")
            projected_max = _optional_float_value(action, "projected_final_max_candidate_gap")
            if projected_mean is None and projected_max is None:
                continue

            source = str(action.get("source", "candidate"))
            is_selected = source == selected_source
            lookahead_history.append(
                {
                    "slot": slot,
                    "source": source,
                    "slotCandidate": f"S{slot} {source}{' *' if is_selected else ''}",
                    "projectedMeanCandidateGap": projected_mean,
                    "projectedMaxCandidateGap": projected_max,
                    "projectedCommitTokens": _int_value(action, "projected_commit_tokens"),
                    "selected": is_selected,
                }
            )

            if projected_mean is not None:
                if is_selected:
                    slot_selected_mean = projected_mean
                    selected_projected_mean_gaps.append(projected_mean)
                else:
                    slot_rejected_means.append(projected_mean)

            if is_selected and projected_max is not None:
                selected_projected_max_gaps.append(projected_max)

        if slot_selected_mean is not None and slot_rejected_means:
            lookahead_advantages.append(min(slot_rejected_means) - slot_selected_mean)

    selection_history = _sample_points(
        [
            {
                "slot": int(item.get("slot", index + 1)),
                "selectedSource": str(item.get("selected_source", "candidate")),
                "gapReduction": _float_value(item, "selected_gap_reduction"),
                "diagnosticGapReduction": _float_value(item, "selected_diagnostic_gap_reduction"),
            }
            for index, item in enumerate(history_source.get("acquisition_history", []))
        ],
        max_points=max_points,
    )

    summary_payload = {
        "active_mean_candidate_gap": _float_value(active, "final_mean_candidate_gap"),
        "round_robin_mean_candidate_gap": _float_value(round_robin, "final_mean_candidate_gap"),
        "active_mean_candidate_diagnostic_gap": _float_value(active, "final_mean_candidate_diagnostic_gap"),
        "round_robin_mean_candidate_diagnostic_gap": _float_value(round_robin, "final_mean_candidate_diagnostic_gap"),
        "active_mean_candidate_info_gain": _float_value(active, "final_mean_candidate_info_gain"),
        "round_robin_mean_candidate_info_gain": _float_value(round_robin, "final_mean_candidate_info_gain"),
    }
    if scout_commit:
        summary_payload.update(
            {
                "scout_mean_candidate_gap": _float_value(scout_commit, "final_mean_candidate_gap"),
                "scout_mean_candidate_diagnostic_gap": _float_value(scout_commit, "final_mean_candidate_diagnostic_gap"),
                "scout_mean_candidate_info_gain": _float_value(scout_commit, "final_mean_candidate_info_gain"),
            }
        )
    if selected_projected_mean_gaps:
        summary_payload["scout_projected_mean_candidate_gap"] = _mean_value(selected_projected_mean_gaps)
    if selected_projected_max_gaps:
        summary_payload["scout_projected_max_candidate_gap"] = _mean_value(selected_projected_max_gaps)
    if lookahead_advantages:
        summary_payload["scout_mean_lookahead_advantage"] = _mean_value(lookahead_advantages)

    payload = {
        "summary_path": _repo_relative(summary_path),
        "artifact_id": benchmark_id,
        "artifact_label": _acquisition_artifact_label(benchmark_id),
        "summary": summary_payload,
        "gap_by_source": gap_by_source,
        "selection_history": selection_history,
        "history_policy": history_policy,
        "scout_policy": "isolated_lookahead" if lookahead_history else ("scout_commit" if scout_commit else "active"),
        "runtime_scope": {
            "input_representation": runtime_scope.get("input_representation"),
            "supports_contextual_routing": bool(runtime_scope.get("supports_contextual_routing", False)),
            "supports_binding_conjunction_memory": bool(
                runtime_scope.get("supports_binding_conjunction_memory")
                or runtime_scope.get("supports_binding_coincidence", False)
            ),
        },
        "active_acquired_sources": [str(item) for item in active.get("acquired_sources", [])],
        "round_robin_acquired_sources": [str(item) for item in round_robin.get("acquired_sources", [])],
    }
    if scout_commit:
        payload["scout_acquired_sources"] = [str(item) for item in scout_commit.get("acquired_sources", [])]
    if lookahead_history:
        payload["scout_lookahead_history"] = lookahead_history
    return payload


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []

    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _sample_points(points: list[dict[str, Any]], *, max_points: int) -> list[dict[str, Any]]:
    if len(points) <= max_points:
        return points

    if max_points <= 1:
        return [points[-1]]

    step = (len(points) - 1) / float(max_points - 1)
    sampled = []
    for index in range(max_points):
        sampled.append(points[round(index * step)])
    return sampled


def _float_value(payload: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = payload.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int_value(payload: dict[str, Any], key: str, default: int = 0) -> int:
    value = payload.get(key, default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _optional_float_value(payload: dict[str, Any], key: str) -> float | None:
    if key not in payload:
        return None
    try:
        return float(payload.get(key))
    except (TypeError, ValueError):
        return None


def _mean_value(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / float(len(values))


def _latest_matching_summary(root: Path, patterns: tuple[str, ...]) -> Path | None:
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(root.glob(f"{pattern}/summary.json"))
    if not matches:
        return None
    return max(matches, key=lambda path: (path.stat().st_mtime, path.parent.name))


def _first_existing_path(*paths: Path) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _acquisition_artifact_label(artifact_id: str) -> str:
    if artifact_id.startswith("refactor_autonomy_acquisition_open_web_scout_projected"):
        return "Open-web scout projected frontier"
    if artifact_id.startswith("refactor_autonomy_acquisition_open_web_scout_confirm"):
        return "Open-web scout confirmation rerun"
    if artifact_id.startswith("refactor_autonomy_acquisition_hf_allocation_rng"):
        return "HF projected active allocation"
    return artifact_id.replace("_", " ")


def _representation_label(representation: str) -> str:
    labels = {
        "order_weighted_ascii": "Order-weighted ASCII",
        "unigram_ascii": "Unigram ASCII",
        "hashed_ngram": "Hashed n-gram",
    }
    return labels.get(representation, representation.replace("_", " "))


def _repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")
