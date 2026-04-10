from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from hecsn.data.rtf_encoder import RTFEncoder
from hecsn.evaluation.grounding_probe import (
    GROUNDING_PROBE_TRIPLES_50,
    CONCRETE_TRIPLES,
    evaluate_grounding_probe,
)
from hecsn.reporting.io import write_json_file
from hecsn.training.meaning_grounding_runner import meaning_grounding_benchmark_config
from hecsn.training.meaning_grounding_runner import meaning_grounding_scenario_payload
from hecsn.training.meaning_grounding_runner import run_meaning_grounding_benchmark
from hecsn.training.behavioral_metrics import compositionality_score
from hecsn.training.behavioral_metrics import grounding_probe as grounding_probe_score
from hecsn.training.behavioral_metrics import novelty_coverage_curve
from hecsn.training.query_runner import feed_text, text_pattern_stream
from hecsn.training.runner_utils import set_seed
from hecsn.training.trainer import HECSNModelLite, HECSNTrainer


REPO_ROOT = Path(__file__).resolve().parents[3]
REPORTS_ROOT = REPO_ROOT / "reports"
MEMORY_RETENTION_SUMMARY = REPORTS_ROOT / "phase6_memory_baseline_replay_fix_20260407" / "summary.json"
LONG_HORIZON_SUMMARY = REPORTS_ROOT / "phase6_long_horizon_autonomy_20260408" / "summary.json"
MECHANISM_REPORT_PATTERNS = (
    "phase7_emergence_evaluation_protocol_*",
)
REPRESENTATION_REPORT_PATTERNS = (
    "phase7_emergence_evaluation_protocol_*",
)
HIERARCHICAL_SCALE_REPORT_PATTERNS = (
    "phase7_emergence_evaluation_protocol_*",
)
MAINTAINED_REPRESENTATION = "order_weighted_ascii"
COMPOSITIONALITY_TEST_PAIRS = (
    ("cats", "mice"),
    ("dogs", "strangers"),
    ("octopuses", "jars"),
    ("rainbows", "water droplets"),
    ("libraries", "books"),
    ("volcanoes", "lava"),
    ("mercury", "sun"),
)
GROUNDING_PROBE_TRIPLES = GROUNDING_PROBE_TRIPLES_50
NOVELTY_CHECKPOINT_FRACTIONS = (0.10, 0.25, 0.50, 0.75, 1.0)
NOVELTY_SHIFT_THRESHOLD_MIN = 1e-4
NOVELTY_SHIFT_CALIBRATION_FRACTION = 0.50
NOVELTY_SHIFT_CALIBRATION_QUANTILE = 0.90


def _load_json_summary(path: Path, *, required: bool = False) -> dict[str, Any]:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required maintained summary is missing: {path}")
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Summary payload must be an object: {path}")
    return payload


def _repo_relative(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _latest_matching_summary(root: Path, patterns: Sequence[str]) -> Path | None:
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(root.glob(f"{pattern}/summary.json"))
    if not matches:
        return None
    return max(matches, key=lambda path: (path.stat().st_mtime, path.parent.name))


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_optional_latest_summary(patterns: Sequence[str]) -> tuple[str, dict[str, Any]] | None:
    summary_path = _latest_matching_summary(REPORTS_ROOT, patterns)
    if summary_path is None:
        return None
    return _repo_relative(summary_path), _load_json_summary(summary_path)


def _pass_rate(cases: Sequence[dict[str, Any]]) -> float:
    if not cases:
        return 0.0
    passed = sum(1 for case in cases if bool(case.get("pass")))
    return float(passed) / float(len(cases))


def _query_cases(summary: dict[str, Any], *, benchmark: str) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for item in summary.get("queries") or []:
        if not isinstance(item, dict):
            continue
        response = item.get("response") or {}
        cases.append(
            {
                "benchmark": benchmark,
                "name": str(item.get("name", "")),
                "query": str(item.get("query", "")),
                "pass": bool(item.get("pass")),
                "response_mode": str(response.get("response_mode", "")),
            }
        )
    return cases


def _temporal_coherence_values(summary: dict[str, Any]) -> list[float]:
    values: list[float] = []
    concept_snapshot = summary.get("concept_snapshot") or {}
    for concept in concept_snapshot.get("top_concepts") or []:
        if not isinstance(concept, dict):
            continue
        temporal = concept.get("temporal_coherence")
        if temporal is None:
            continue
        values.append(float(temporal))
    return values


def _representation_result(summary: dict[str, Any], representation: str) -> dict[str, Any] | None:
    for item in summary.get("results") or []:
        if isinstance(item, dict) and str(item.get("representation")) == representation:
            return item
    return None


def _representation_baseline(summary_ref: tuple[str, dict[str, Any]] | None) -> dict[str, Any] | None:
    if summary_ref is None:
        return None
    summary_path, summary = summary_ref
    maintained = _representation_result(summary, MAINTAINED_REPRESENTATION) or {}
    hecsn = maintained.get("hecsn_competitive_only") or {}
    baseline = maintained.get("online_kmeans") or {}
    completion = maintained.get("completion_coherence") or {}
    cluster_count = int(hecsn.get("cluster_count_used", 0))
    clustering_status = str(hecsn.get("clustering_status", ""))
    collapse = cluster_count <= 1 or clustering_status.startswith("skipped: single cluster")
    return {
        "summary_path": summary_path,
        "maintained_representation": MAINTAINED_REPRESENTATION,
        "completion_coherence_mean_margin": _safe_float(completion.get("mean_margin")),
        "hecsn_cluster_count_used": cluster_count,
        "hecsn_winner_entropy_bits": _safe_float(hecsn.get("winner_entropy_bits")),
        "hecsn_silhouette": _safe_float(hecsn.get("silhouette")),
        "hecsn_clustering_status": clustering_status,
        "online_kmeans_cluster_count_used": int(baseline.get("cluster_count_used", 0)),
        "online_kmeans_silhouette": _safe_float(baseline.get("silhouette")),
        "online_kmeans_clustering_status": str(baseline.get("clustering_status", "")),
        "representation_collapse_detected": bool(collapse),
    }


def _mechanism_baseline(summary_ref: tuple[str, dict[str, Any]] | None) -> dict[str, Any] | None:
    if summary_ref is None:
        return None
    summary_path, summary = summary_ref
    eval_metrics = summary.get("eval_metrics") or {}
    behavioral_metrics = summary.get("behavioral_metrics") or {}
    character_ngram = behavioral_metrics.get("character_ngram_recovery") or {}
    char_ngram_baseline = character_ngram.get("baseline") or {}
    distributional = behavioral_metrics.get("distributional_clustering") or {}
    gate = summary.get("mechanism_validation_gate") or {}
    return {
        "summary_path": summary_path,
        "mechanism_gate_pass": bool(gate.get("pass", False)),
        "ablation_gain": _safe_float(eval_metrics.get("ablation_gain")),
        "trained_eval_recon_error": _safe_float(eval_metrics.get("trained_eval_recon_error")),
        "random_assignment_eval_recon_error": _safe_float(eval_metrics.get("random_assignment_eval_recon_error")),
        "routing_key_clustering_status": str(eval_metrics.get("routing_key_clustering_status", "")),
        "distributional_clustering_silhouette": _safe_float(distributional.get("silhouette")),
        "distributional_clustering_pass": bool(distributional.get("pass", False)),
        "character_ngram_recovery_pass": bool(character_ngram.get("pass", False)),
        "character_ngram_recovery_mean_margin": _safe_float(character_ngram.get("mean_margin")),
        "character_ngram_baseline_success_rate": _safe_float(char_ngram_baseline.get("success_rate")),
        "character_ngram_baseline_mean_margin": _safe_float(char_ngram_baseline.get("mean_margin")),
    }


def _routing_scale_context(summary_ref: tuple[str, dict[str, Any]] | None) -> dict[str, Any] | None:
    if summary_ref is None:
        return None
    summary_path, summary = summary_ref
    routing_metrics = summary.get("routing_metrics") or {}
    integrity_metrics = summary.get("index_integrity") or {}
    memory_budget = summary.get("memory_budget_estimate") or {}
    gate = summary.get("hierarchical_scale_gate") or {}
    if not gate and not routing_metrics:
        return None
    return {
        "summary_path": summary_path,
        "gate_pass": bool(gate.get("pass", False)),
        "estimated_neurons": int(memory_budget.get("estimated_neurons", 0)),
        "mean_latency_ms": _safe_float(routing_metrics.get("mean_latency_ms")),
        "p95_latency_ms": _safe_float(routing_metrics.get("p95_latency_ms")),
        "unreachable_fraction": _safe_float(integrity_metrics.get("unreachable_fraction")),
    }


def _last_pattern(text: str, encoder: RTFEncoder, window_size: int) -> torch.Tensor:
    examples = list(text_pattern_stream(text, encoder, window_size))
    if not examples:
        raise ValueError(f"Text produced no probe pattern: {text!r}")
    return examples[-1][1]


def _normalize_vector(value: torch.Tensor) -> torch.Tensor:
    vector = value.detach().cpu().float()
    norm = float(torch.norm(vector).item())
    if norm <= 0.0:
        return vector
    return vector / norm


def _cosine_similarity(left: torch.Tensor, right: torch.Tensor) -> float:
    left_vec = _normalize_vector(left)
    right_vec = _normalize_vector(right)
    if left_vec.numel() == 0 or right_vec.numel() == 0:
        return 0.0
    return float(F.cosine_similarity(left_vec.unsqueeze(0), right_vec.unsqueeze(0), dim=1).item())


def _probe_feed_corpus() -> str:
    return "\n".join(
        [
            str(meaning_grounding_scenario_payload("simple_animals")["feed_text"]),
            str(meaning_grounding_scenario_payload("mixed_world")["feed_text"]),
        ]
    )


def _build_direct_probe_world(seed: int) -> tuple[HECSNTrainer, RTFEncoder]:
    set_seed(seed)
    cfg = meaning_grounding_benchmark_config()
    trainer = HECSNTrainer(HECSNModelLite(cfg), cfg)
    encoder = RTFEncoder.from_config(cfg)
    feed_text(trainer, encoder, _probe_feed_corpus())
    trainer.model.hnsw_index.rebuild()
    return trainer, encoder


def _probe_representation(trainer: HECSNTrainer, encoder: RTFEncoder, text: str) -> tuple[int, torch.Tensor]:
    pattern = _last_pattern(text, encoder, trainer.config.window_size)
    winner = trainer.winner_for_pattern(pattern)
    representation = trainer.model.routing_key_from_pattern(pattern)
    return int(winner), _normalize_vector(representation)


def _direct_compositionality_probe(trainer: HECSNTrainer, encoder: RTFEncoder) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    scores: list[float] = []
    winners: set[int] = set()
    vector_triples: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for chunk_a, chunk_b in COMPOSITIONALITY_TEST_PAIRS:
        winner_a, rep_a = _probe_representation(trainer, encoder, chunk_a)
        winner_b, rep_b = _probe_representation(trainer, encoder, chunk_b)
        winner_ab, rep_ab = _probe_representation(trainer, encoder, f"{chunk_a} {chunk_b}")
        winners.update((int(winner_a), int(winner_b), int(winner_ab)))
        vector_triples.append((rep_a, rep_b, rep_ab))
        expected = _normalize_vector(rep_a + rep_b)
        score = _cosine_similarity(rep_ab, expected)
        scores.append(score)
        cases.append(
            {
                "chunk_a": chunk_a,
                "chunk_b": chunk_b,
                "chunk_ab": f"{chunk_a} {chunk_b}",
                "winner_a": int(winner_a),
                "winner_b": int(winner_b),
                "winner_ab": int(winner_ab),
                "routing_key_between_score": float(score),
            }
        )
    metric = compositionality_score(vector_triples)
    mean_score = 0.0 if metric["mean_score"] is None else float(metric["mean_score"])
    threshold_min = 0.60
    unique_winner_count = int(len(winners))
    winner_collapse_detected = bool(unique_winner_count <= 1)
    return {
        "metric_name": "routing_key_between_score",
        "sample_count": int(len(cases)),
        "score": float(mean_score),
        "threshold_min": float(threshold_min),
        "unique_winner_count": unique_winner_count,
        "winner_collapse_detected": winner_collapse_detected,
        "pass": bool(mean_score >= threshold_min and not winner_collapse_detected),
        "cases": cases,
    }


def _direct_grounding_probe(trainer: HECSNTrainer, encoder: RTFEncoder) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    winners: set[int] = set()
    vector_triples: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    concrete_limit = len(CONCRETE_TRIPLES)
    concrete_correct = 0
    abstract_correct = 0
    concrete_margins: list[float] = []
    abstract_margins: list[float] = []
    for idx, (anchor, positive, negative) in enumerate(GROUNDING_PROBE_TRIPLES):
        winner_anchor, rep_anchor = _probe_representation(trainer, encoder, anchor)
        winner_positive, rep_positive = _probe_representation(trainer, encoder, positive)
        winner_negative, rep_negative = _probe_representation(trainer, encoder, negative)
        winners.update((int(winner_anchor), int(winner_positive), int(winner_negative)))
        vector_triples.append((rep_anchor, rep_positive, rep_negative))
        positive_similarity = _cosine_similarity(rep_anchor, rep_positive)
        negative_similarity = _cosine_similarity(rep_anchor, rep_negative)
        margin = float(positive_similarity - negative_similarity)
        case_pass = bool(positive_similarity > negative_similarity)
        is_concrete = idx < concrete_limit
        category = "concrete" if is_concrete else "abstract"
        cases.append(
            {
                "anchor": anchor,
                "positive": positive,
                "negative": negative,
                "category": category,
                "winner_anchor": int(winner_anchor),
                "winner_positive": int(winner_positive),
                "winner_negative": int(winner_negative),
                "positive_similarity": float(positive_similarity),
                "negative_similarity": float(negative_similarity),
                "margin": float(margin),
                "pass": case_pass,
            }
        )
        if is_concrete:
            concrete_margins.append(margin)
            if case_pass:
                concrete_correct += 1
        else:
            abstract_margins.append(margin)
            if case_pass:
                abstract_correct += 1
    metric = grounding_probe_score(vector_triples)
    accuracy = 0.0 if metric["accuracy"] is None else float(metric["accuracy"])
    mean_margin = 0.0 if metric["mean_margin"] is None else float(metric["mean_margin"])
    threshold_min = 0.50
    v4_target_threshold = 0.65
    unique_winner_count = int(len(winners))
    winner_collapse_detected = bool(unique_winner_count <= 1)
    n_concrete = max(len(concrete_margins), 1)
    n_abstract = max(len(abstract_margins), 1)
    concrete_accuracy = concrete_correct / n_concrete
    abstract_accuracy = abstract_correct / n_abstract
    concreteness_gap = concrete_accuracy - abstract_accuracy
    return {
        "metric_name": "semantic_triple_accuracy",
        "sample_count": int(len(cases)),
        "accuracy": float(accuracy),
        "mean_margin": float(mean_margin),
        "threshold_min": float(threshold_min),
        "v4_target_threshold": float(v4_target_threshold),
        "unique_winner_count": unique_winner_count,
        "winner_collapse_detected": winner_collapse_detected,
        "pass": bool(accuracy >= threshold_min),
        "concrete_accuracy": float(concrete_accuracy),
        "abstract_accuracy": float(abstract_accuracy),
        "concreteness_gap": float(concreteness_gap),
        "concreteness_gap_pass": bool(concreteness_gap > 0.10),
        "concrete_count": int(n_concrete),
        "abstract_count": int(n_abstract),
        "concrete_mean_margin": float(sum(concrete_margins) / n_concrete),
        "abstract_mean_margin": float(sum(abstract_margins) / n_abstract),
        "cases": cases,
    }


def _checkpoint_targets(total_steps: int) -> list[int]:
    if total_steps <= 0:
        return []
    checkpoints = sorted(
        {
            max(1, min(total_steps, int(round(float(total_steps) * fraction))))
            for fraction in NOVELTY_CHECKPOINT_FRACTIONS
        }
    )
    if checkpoints[-1] != total_steps:
        checkpoints.append(total_steps)
    return checkpoints


def _calibrated_novelty_shift_threshold(shifts: Sequence[float]) -> float:
    if not shifts:
        return float(NOVELTY_SHIFT_THRESHOLD_MIN)
    calibration_steps = max(1, int(len(shifts) * NOVELTY_SHIFT_CALIBRATION_FRACTION))
    calibration = [float(shift) for shift in shifts[:calibration_steps] if float(shift) > 0.0]
    if not calibration:
        return float(NOVELTY_SHIFT_THRESHOLD_MIN)
    threshold = torch.quantile(
        torch.tensor(calibration, dtype=torch.float32),
        float(NOVELTY_SHIFT_CALIBRATION_QUANTILE),
    ).item()
    return float(max(NOVELTY_SHIFT_THRESHOLD_MIN, threshold))


def _direct_novelty_coverage_probe(seed: int) -> dict[str, Any]:
    set_seed(seed)
    cfg = meaning_grounding_benchmark_config()
    trainer = HECSNTrainer(HECSNModelLite(cfg), cfg)
    encoder = RTFEncoder.from_config(cfg)
    trainer.encoder = encoder
    segments = encoder.segment_text(_probe_feed_corpus(), learn=True)
    checkpoint_targets = _checkpoint_targets(len(segments))
    if not segments:
        return {
            "metric_name": "novelty_rate_by_checkpoint",
            "stream_unit": "learned_chunk",
            "sample_count": 0,
            "segment_count": 0,
            "checkpoints": [],
            "terminal_novelty_rate": 0.0,
            "healthy_range": [0.05, 0.40],
            "saturation_detected": True,
            "instability_detected": False,
            "winner_collapse_detected": True,
            "unique_winner_count": 0,
            "prototype_shift_threshold": float(NOVELTY_SHIFT_THRESHOLD_MIN),
            "prototype_shift_threshold_source": "first_half_nonzero_q90",
            "pass": False,
        }

    winners: list[int] = []
    prototype_shifts: list[float] = []
    for segment in segments:
        pattern = _last_pattern(segment, encoder, cfg.window_size)
        pre_winner = trainer.winner_for_pattern(pattern)
        before = trainer.model.competitive.prototypes[int(pre_winner)].detach().cpu().clone()
        metrics = trainer.train_step(pattern, raw_window=segment)
        winner = int(metrics["winner"])
        after = trainer.model.competitive.prototypes[winner].detach().cpu().clone()
        winners.append(winner)
        prototype_shifts.append(max(0.0, 1.0 - _cosine_similarity(before, after)))

    prototype_shift_threshold = _calibrated_novelty_shift_threshold(prototype_shifts)
    seen_winners: set[int] = set()
    novelty_events: list[bool] = []
    interval_new_winner = 0
    interval_shift = 0
    checkpoint_set = set(checkpoint_targets)
    interval_new_winner_counts: list[int] = []
    interval_shift_counts: list[int] = []
    checkpoint_unique_winner_counts: list[int] = []
    for chunk_index, (winner, prototype_shift) in enumerate(zip(winners, prototype_shifts), start=1):
        new_winner = winner not in seen_winners
        significant_shift = prototype_shift >= prototype_shift_threshold
        novelty_events.append(bool(new_winner or significant_shift))
        interval_new_winner += int(new_winner)
        interval_shift += int(significant_shift)
        seen_winners.add(winner)
        if chunk_index not in checkpoint_set:
            continue
        interval_new_winner_counts.append(interval_new_winner)
        interval_shift_counts.append(interval_shift)
        checkpoint_unique_winner_counts.append(int(len(seen_winners)))
        interval_new_winner = 0
        interval_shift = 0
    curve = novelty_coverage_curve(novelty_events, checkpoint_targets)
    checkpoints: list[dict[str, Any]] = []
    for index, row in enumerate(curve["novelty_rate_by_checkpoint"]):
        interval_length = int(row["window_size"])
        checkpoints.append(
            {
                "stream_unit": "learned_chunk",
                "token_count": int(row["token_end"]),
                "chunk_count": int(row["token_end"]),
                "interval_start_token": int(row["token_start"]),
                "interval_end_token": int(row["token_end"]),
                "interval_start_chunk": int(row["token_start"]),
                "interval_end_chunk": int(row["token_end"]),
                "novelty_rate": float(row["novelty_rate"]),
                "new_winner_fraction": (
                    0.0
                    if interval_length <= 0
                    else float(interval_new_winner_counts[index]) / float(interval_length)
                ),
                "prototype_shift_fraction": (
                    0.0
                    if interval_length <= 0
                    else float(interval_shift_counts[index]) / float(interval_length)
                ),
                "unique_winner_count": int(checkpoint_unique_winner_counts[index]),
            }
        )

    terminal_rate = 0.0 if curve["final_novelty_rate"] is None else float(curve["final_novelty_rate"])
    healthy_low = float(curve["healthy_range"]["min"])
    healthy_high = float(curve["healthy_range"]["max"])
    saturation_detected = bool(curve["saturation_detected"])
    instability_detected = bool(curve["instability_detected"])
    winner_collapse_detected = bool(len(seen_winners) <= 1)
    healthy_terminal_range = bool(curve["healthy_final_range"])
    return {
        "metric_name": "novelty_rate_by_checkpoint",
        "stream_unit": "learned_chunk",
        "sample_count": int(len(checkpoints)),
        "segment_count": int(len(segments)),
        "checkpoints": checkpoints,
        "terminal_novelty_rate": float(terminal_rate),
        "healthy_range": [float(healthy_low), float(healthy_high)],
        "saturation_detected": saturation_detected,
        "instability_detected": instability_detected,
        "winner_collapse_detected": winner_collapse_detected,
        "unique_winner_count": int(len(seen_winners)),
        "prototype_shift_threshold": float(prototype_shift_threshold),
        "prototype_shift_threshold_source": "first_half_nonzero_q90",
        "pass": bool(
            healthy_terminal_range
            and not saturation_detected
            and not instability_detected
            and not winner_collapse_detected
        ),
    }


def run_emergence_evaluation_benchmark(
    *,
    output_dir: Path,
    seed: int = 7,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    simple_summary = run_meaning_grounding_benchmark(
        output_dir=output_dir / "meaning_grounding_simple_animals",
        scenario="simple_animals",
        seed=seed,
    )
    mixed_summary = run_meaning_grounding_benchmark(
        output_dir=output_dir / "meaning_grounding_mixed_world",
        scenario="mixed_world",
        seed=seed,
    )
    memory_summary = _load_json_summary(MEMORY_RETENTION_SUMMARY)
    long_horizon_summary = _load_json_summary(LONG_HORIZON_SUMMARY)
    _has_memory_report = bool(memory_summary)
    _has_long_horizon_report = bool(long_horizon_summary)
    representation_baseline = _representation_baseline(_load_optional_latest_summary(REPRESENTATION_REPORT_PATTERNS))
    mechanism_baseline = _mechanism_baseline(_load_optional_latest_summary(MECHANISM_REPORT_PATTERNS))
    routing_scale_context = _routing_scale_context(_load_optional_latest_summary(HIERARCHICAL_SCALE_REPORT_PATTERNS))
    probe_trainer, probe_encoder = _build_direct_probe_world(seed)
    direct_compositionality = _direct_compositionality_probe(probe_trainer, probe_encoder)
    direct_grounding = _direct_grounding_probe(probe_trainer, probe_encoder)
    direct_novelty_coverage = _direct_novelty_coverage_probe(seed)

    all_query_cases = [
        *_query_cases(simple_summary, benchmark="simple_animals"),
        *_query_cases(mixed_summary, benchmark="mixed_world"),
    ]
    grounded_cases = [case for case in all_query_cases if case["response_mode"] != "insufficient_evidence"]
    unsupported_cases = [case for case in all_query_cases if case["response_mode"] == "insufficient_evidence"]
    compositional_cases = [case for case in all_query_cases if "composition" in case["name"]]

    temporal_values = [*_temporal_coherence_values(simple_summary), *_temporal_coherence_values(mixed_summary)]
    temporal_coherence_mean = (
        0.0 if not temporal_values else float(sum(temporal_values) / len(temporal_values))
    )
    grounded_query_accuracy = _pass_rate(grounded_cases)
    unsupported_query_accuracy = _pass_rate(unsupported_cases)
    compositional_query_accuracy = _pass_rate(compositional_cases)

    memory_metrics = memory_summary.get("metrics") or {}
    memory_gate = memory_summary.get("memory_consolidation_gate") or {}
    long_metrics = long_horizon_summary.get("metrics") or {}

    phase_a_interference_retention = float(memory_metrics.get("task_a_overlap_after_b", 0.0))
    phase_a_final_retention = float(memory_metrics.get("task_a_overlap_after_consolidation", 0.0))
    supported_topic_coverage = float(long_metrics.get("supported_topic_coverage", 0.0))
    answerability_growth_mean = float(long_metrics.get("answerability_growth_mean", 0.0))
    revisit_retention_rate = float(long_metrics.get("revisit_retention_rate", 0.0))
    concept_stability_mean = float(long_metrics.get("concept_stability_mean", 0.0))
    unique_acquired_source_count = int(long_metrics.get("unique_acquired_source_count", 0))

    novelty_coverage = {
        "summary_path": _repo_relative(LONG_HORIZON_SUMMARY),
        "supported_topic_coverage": float(supported_topic_coverage),
        "answerability_growth_mean": float(answerability_growth_mean),
        "revisit_retention_rate": float(revisit_retention_rate),
        "concept_stability_mean": float(concept_stability_mean),
        "unique_acquired_source_count": int(unique_acquired_source_count),
        "healthy_coverage": bool(
            not _has_long_horizon_report or (
                supported_topic_coverage >= 1.0
                and answerability_growth_mean >= 0.25
                and revisit_retention_rate >= 1.0
                and unique_acquired_source_count >= 1
            )
        ),
    }

    thresholds = {
        "temporal_coherence_mean_min": 0.95,
        "grounded_query_accuracy_min": 0.85,
        "compositional_query_accuracy_min": 0.60,
        "phase_a_interference_retention_min": 0.95,
        "phase_a_final_retention_min": 0.95,
        "supported_topic_coverage_min": 1.0,
        "answerability_growth_mean_min": 0.25,
        "revisit_retention_rate_min": 1.0,
        "unique_acquired_source_count_min": 1,
        "routing_key_between_score_min": float(direct_compositionality["threshold_min"]),
        "semantic_triple_accuracy_min": float(direct_grounding["threshold_min"]),
    }
    meaning_grounding_pass = bool(
        simple_summary["meaning_grounding_gate"]["pass"] and mixed_summary["meaning_grounding_gate"]["pass"]
    )
    gate_pass = bool(
        meaning_grounding_pass
        and (not _has_memory_report or bool(memory_gate.get("pass", False)))
        and temporal_coherence_mean >= thresholds["temporal_coherence_mean_min"]
        and grounded_query_accuracy >= thresholds["grounded_query_accuracy_min"]
        and compositional_query_accuracy >= thresholds["compositional_query_accuracy_min"]
        and (not _has_memory_report or phase_a_interference_retention >= thresholds["phase_a_interference_retention_min"])
        and (not _has_memory_report or phase_a_final_retention >= thresholds["phase_a_final_retention_min"])
        and (not _has_long_horizon_report or supported_topic_coverage >= thresholds["supported_topic_coverage_min"])
        and (not _has_long_horizon_report or answerability_growth_mean >= thresholds["answerability_growth_mean_min"])
        and (not _has_long_horizon_report or revisit_retention_rate >= thresholds["revisit_retention_rate_min"])
        and (not _has_long_horizon_report or unique_acquired_source_count >= thresholds["unique_acquired_source_count_min"])
    )

    structural_temporal_pass = bool(temporal_coherence_mean >= thresholds["temporal_coherence_mean_min"])
    forgetting_pass = bool(
        (not _has_memory_report or phase_a_interference_retention >= thresholds["phase_a_interference_retention_min"])
        and (not _has_memory_report or phase_a_final_retention >= thresholds["phase_a_final_retention_min"])
        and (not _has_memory_report or bool(memory_gate.get("pass", False)))
    )
    feedback_label_free_levels = {
        "structural_coherence": {
            "status": "direct_with_sanity_checks",
            "direct_metric_available": True,
            "temporal_coherence_mean": float(temporal_coherence_mean),
            "temporal_threshold_min": thresholds["temporal_coherence_mean_min"],
            "temporal_pass": structural_temporal_pass,
            "representation_collapse_detected": (
                None
                if representation_baseline is None
                else bool(representation_baseline["representation_collapse_detected"])
            ),
            "mechanism_distributional_clustering_pass": (
                None
                if mechanism_baseline is None
                else bool(mechanism_baseline["distributional_clustering_pass"])
            ),
            "pass": structural_temporal_pass,
        },
        "compositionality": {
            "status": "direct_with_proxy_support",
            "direct_metric_available": True,
            "direct_metric_name": str(direct_compositionality["metric_name"]),
            "direct_score": float(direct_compositionality["score"]),
            "direct_sample_count": int(direct_compositionality["sample_count"]),
            "direct_threshold_min": float(direct_compositionality["threshold_min"]),
            "winner_collapse_detected": bool(direct_compositionality["winner_collapse_detected"]),
            "unique_winner_count": int(direct_compositionality["unique_winner_count"]),
            "proxy_metric_name": "grounded_composition_query_accuracy",
            "proxy_accuracy": float(compositional_query_accuracy),
            "proxy_sample_count": int(len(compositional_cases)),
            "proxy_threshold_min": thresholds["compositional_query_accuracy_min"],
            "proxy_pass": bool(compositional_query_accuracy >= thresholds["compositional_query_accuracy_min"]),
            "pass": bool(direct_compositionality["pass"]),
            "blocking_reason": (
                None
                if bool(direct_compositionality["pass"])
                else (
                    "Direct routing-key compositionality is degenerate because the current maintained probe corpus collapses every tested chunk to the same winner column."
                    if bool(direct_compositionality["winner_collapse_detected"])
                    else "Direct routing-key compositionality remains below the feedback threshold on the current maintained probe corpus."
                )
            ),
        },
        "novelty_coverage": {
            "status": "direct_with_proxy_support",
            "direct_metric_available": True,
            "direct_metric_name": str(direct_novelty_coverage["metric_name"]),
            "direct_sample_count": int(direct_novelty_coverage["sample_count"]),
            "direct_terminal_novelty_rate": float(direct_novelty_coverage["terminal_novelty_rate"]),
            "direct_healthy_range": list(direct_novelty_coverage["healthy_range"]),
            "direct_saturation_detected": bool(direct_novelty_coverage["saturation_detected"]),
            "direct_instability_detected": bool(direct_novelty_coverage["instability_detected"]),
            "winner_collapse_detected": bool(direct_novelty_coverage["winner_collapse_detected"]),
            "unique_winner_count": int(direct_novelty_coverage["unique_winner_count"]),
            "proxy_supported_topic_coverage": float(supported_topic_coverage),
            "proxy_answerability_growth_mean": float(answerability_growth_mean),
            "proxy_revisit_retention_rate": float(revisit_retention_rate),
            "proxy_unique_acquired_source_count": int(unique_acquired_source_count),
            "proxy_healthy_coverage": bool(novelty_coverage["healthy_coverage"]),
            "feedback_target_name": "novelty_rate_by_checkpoint",
            "feedback_healthy_range": list(direct_novelty_coverage["healthy_range"]),
            "pass": bool(direct_novelty_coverage["pass"]),
            "blocking_reason": (
                None
                if bool(direct_novelty_coverage["pass"])
                else (
                    "Direct novelty-rate coverage is degenerate because the current maintained probe corpus collapses every tested chunk to the same winner column."
                    if bool(direct_novelty_coverage["winner_collapse_detected"])
                    else (
                        "Direct novelty-rate coverage saturates near zero on the current maintained probe corpus."
                        if bool(direct_novelty_coverage["saturation_detected"])
                        else (
                            "Direct novelty-rate coverage remains unstable above the healthy terminal range on the current maintained probe corpus."
                            if bool(direct_novelty_coverage["instability_detected"])
                            else "Direct novelty-rate coverage remains outside the healthy terminal range on the current maintained probe corpus."
                        )
                    )
                )
            ),
        },
        "grounding_probe": {
            "status": "direct_with_proxy_support",
            "direct_metric_available": True,
            "direct_metric_name": str(direct_grounding["metric_name"]),
            "direct_accuracy": float(direct_grounding["accuracy"]),
            "direct_mean_margin": float(direct_grounding["mean_margin"]),
            "direct_sample_count": int(direct_grounding["sample_count"]),
            "direct_threshold_min": float(direct_grounding["threshold_min"]),
            "winner_collapse_detected": bool(direct_grounding["winner_collapse_detected"]),
            "unique_winner_count": int(direct_grounding["unique_winner_count"]),
            "proxy_metric_name": "grounded_query_accuracy",
            "proxy_accuracy": float(grounded_query_accuracy),
            "proxy_sample_count": int(len(grounded_cases)),
            "proxy_threshold_min": thresholds["grounded_query_accuracy_min"],
            "proxy_pass": bool(grounded_query_accuracy >= thresholds["grounded_query_accuracy_min"]),
            "feedback_target_name": "semantic_triple_accuracy",
            "representation_collapse_detected": (
                None
                if representation_baseline is None
                else bool(representation_baseline["representation_collapse_detected"])
            ),
            "mechanism_distributional_clustering_pass": (
                None
                if mechanism_baseline is None
                else bool(mechanism_baseline["distributional_clustering_pass"])
            ),
            "pass": bool(direct_grounding["pass"]),
            "blocking_reason": (
                None
                if bool(direct_grounding["pass"])
                else (
                    "Direct semantic-triple accuracy is degenerate because the current maintained probe corpus collapses every tested term to the same winner column."
                    if bool(direct_grounding["winner_collapse_detected"])
                    else "Direct semantic-triple accuracy remains below the feedback threshold on the current maintained probe corpus."
                )
            ),
        },
        "forgetting": {
            "status": "direct",
            "direct_metric_available": True,
            "phase_a_interference_retention": float(phase_a_interference_retention),
            "phase_a_final_retention": float(phase_a_final_retention),
            "thresholds": {
                "phase_a_interference_retention_min": thresholds["phase_a_interference_retention_min"],
                "phase_a_final_retention_min": thresholds["phase_a_final_retention_min"],
            },
            "memory_gate_pass": bool(memory_gate.get("pass", False)),
            "pass": forgetting_pass,
        },
    }
    feedback_direct_levels_ready = {
        name: bool(level["direct_metric_available"])
        for name, level in feedback_label_free_levels.items()
    }
    feedback_blocking_reasons = [
        str(level["blocking_reason"])
        for level in feedback_label_free_levels.values()
        if isinstance(level, dict) and level.get("blocking_reason")
    ]
    feedback_gate_pass = bool(
        all(feedback_direct_levels_ready.values())
        and all(bool(level["pass"]) for level in feedback_label_free_levels.values())
    )

    summary = {
        "benchmark": "emergence_evaluation",
        "scenario": "maintained_proxy_aggregate",
        "seed": int(seed),
        "runtime_scope": {
            "mode": "maintained_proxy_aggregate",
            "note": "This Stage-0 report aggregates the maintained grounded-response, retention, long-horizon autonomy, baseline-comparison surfaces, a direct routing-key compositionality score, a direct semantic-triple grounding probe, and a direct learned-chunk novelty-rate curve.",
        },
        "metrics": {
            "temporal_coherence_mean": float(temporal_coherence_mean),
            "grounded_query_accuracy": float(grounded_query_accuracy),
            "unsupported_query_accuracy": float(unsupported_query_accuracy),
            "compositional_query_accuracy": float(compositional_query_accuracy),
            "phase_a_interference_retention": float(phase_a_interference_retention),
            "phase_a_final_retention": float(phase_a_final_retention),
            "supported_topic_coverage": float(supported_topic_coverage),
            "answerability_growth_mean": float(answerability_growth_mean),
            "revisit_retention_rate": float(revisit_retention_rate),
            "concept_stability_mean": float(concept_stability_mean),
            "unique_acquired_source_count": int(unique_acquired_source_count),
        },
        "meaning_grounding": {
            "simple_animals": {
                "summary_path": str(Path("meaning_grounding_simple_animals") / "summary.json"),
                "gate_pass": bool(simple_summary["meaning_grounding_gate"]["pass"]),
                "concept_separation_pass": bool(simple_summary["concept_separation_gate"]["pass"]),
            },
            "mixed_world": {
                "summary_path": str(Path("meaning_grounding_mixed_world") / "summary.json"),
                "gate_pass": bool(mixed_summary["meaning_grounding_gate"]["pass"]),
                "concept_separation_pass": bool(mixed_summary["concept_separation_gate"]["pass"]),
                "concept_count": int(mixed_summary["concept_snapshot"]["concept_count"]),
                "growth_ready": bool((mixed_summary["concept_snapshot"].get("growth") or {}).get("growth_ready", False)),
            },
            "temporal_coherence_mean": float(temporal_coherence_mean),
            "grounded_query_accuracy": float(grounded_query_accuracy),
            "unsupported_query_accuracy": float(unsupported_query_accuracy),
        },
        "compositionality": {
            "sample_count": int(len(compositional_cases)),
            "accuracy": float(compositional_query_accuracy),
            "cases": compositional_cases,
            "direct_metric_name": str(direct_compositionality["metric_name"]),
            "direct_score": float(direct_compositionality["score"]),
            "direct_sample_count": int(direct_compositionality["sample_count"]),
            "direct_unique_winner_count": int(direct_compositionality["unique_winner_count"]),
            "direct_winner_collapse_detected": bool(direct_compositionality["winner_collapse_detected"]),
            "direct_cases": direct_compositionality["cases"],
        },
        "grounding_probe": {
            "sample_count": int(len(grounded_cases)),
            "accuracy": float(grounded_query_accuracy),
            "unsupported_accuracy": float(unsupported_query_accuracy),
            "cases": grounded_cases,
            "direct_metric_name": str(direct_grounding["metric_name"]),
            "semantic_triple_accuracy": float(direct_grounding["accuracy"]),
            "semantic_triple_mean_margin": float(direct_grounding["mean_margin"]),
            "direct_sample_count": int(direct_grounding["sample_count"]),
            "direct_unique_winner_count": int(direct_grounding["unique_winner_count"]),
            "direct_winner_collapse_detected": bool(direct_grounding["winner_collapse_detected"]),
            "concrete_accuracy": float(direct_grounding["concrete_accuracy"]),
            "abstract_accuracy": float(direct_grounding["abstract_accuracy"]),
            "concreteness_gap": float(direct_grounding["concreteness_gap"]),
            "concreteness_gap_pass": bool(direct_grounding["concreteness_gap_pass"]),
            "concrete_count": int(direct_grounding["concrete_count"]),
            "abstract_count": int(direct_grounding["abstract_count"]),
            "concrete_mean_margin": float(direct_grounding["concrete_mean_margin"]),
            "abstract_mean_margin": float(direct_grounding["abstract_mean_margin"]),
            "semantic_triples": direct_grounding["cases"],
        },
        "forgetting": {
            "summary_path": _repo_relative(MEMORY_RETENTION_SUMMARY),
            "memory_gate_pass": bool(memory_gate.get("pass", False)),
            "phase_a_interference_retention": float(phase_a_interference_retention),
            "phase_a_final_retention": float(phase_a_final_retention),
            "task_a_relative_degradation_after_consolidation": float(
                memory_metrics.get("task_a_relative_degradation_after_consolidation", 0.0)
            ),
        },
        "novelty_coverage": novelty_coverage,
        "direct_novelty_coverage": {
            "metric_name": str(direct_novelty_coverage["metric_name"]),
            "stream_unit": str(direct_novelty_coverage["stream_unit"]),
            "sample_count": int(direct_novelty_coverage["sample_count"]),
            "segment_count": int(direct_novelty_coverage["segment_count"]),
            "terminal_novelty_rate": float(direct_novelty_coverage["terminal_novelty_rate"]),
            "healthy_range": list(direct_novelty_coverage["healthy_range"]),
            "saturation_detected": bool(direct_novelty_coverage["saturation_detected"]),
            "instability_detected": bool(direct_novelty_coverage["instability_detected"]),
            "winner_collapse_detected": bool(direct_novelty_coverage["winner_collapse_detected"]),
            "unique_winner_count": int(direct_novelty_coverage["unique_winner_count"]),
            "prototype_shift_threshold": float(direct_novelty_coverage["prototype_shift_threshold"]),
            "prototype_shift_threshold_source": str(direct_novelty_coverage["prototype_shift_threshold_source"]),
            "checkpoints": direct_novelty_coverage["checkpoints"],
        },
        "feedback_label_free_levels": feedback_label_free_levels,
        "baseline_comparison": {
            "representation": representation_baseline,
            "mechanism": mechanism_baseline,
        },
        "supporting_scaffolds": {
            "routing_scale": routing_scale_context,
        },
        "feedback_emergence_gate": {
            "pass": bool(feedback_gate_pass),
            "direct_levels_ready": feedback_direct_levels_ready,
            "proxy_aggregate_pass": bool(gate_pass),
            "blocking_reasons": feedback_blocking_reasons,
        },
        "emergence_evaluation_gate": {
            "pass": bool(gate_pass),
            "thresholds": thresholds,
            "meaning_grounding_pass": bool(meaning_grounding_pass),
            "memory_gate_pass": bool(memory_gate.get("pass", False)),
            "long_horizon_coverage_pass": bool(novelty_coverage["healthy_coverage"]),
        },
    }
    write_json_file(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the maintained Stage-0 emergence evaluation aggregate.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where summary.json should be written.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for deterministic local benchmarks.",
    )
    args = parser.parse_args()
    summary = run_emergence_evaluation_benchmark(
        output_dir=args.output_dir,
        seed=int(args.seed),
    )
    print(
        f"[emergence_evaluation] scenario={summary['scenario']} "
        f"proxy_pass={summary['emergence_evaluation_gate']['pass']} "
        f"feedback_ready={summary['feedback_emergence_gate']['pass']} "
        f"temporal_coherence={summary['metrics']['temporal_coherence_mean']:.3f} "
        f"grounded_query_accuracy={summary['metrics']['grounded_query_accuracy']:.3f} "
        f"topic_coverage={summary['metrics']['supported_topic_coverage']:.3f}"
    )


if __name__ == "__main__":
    main()
