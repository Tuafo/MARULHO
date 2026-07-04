"""Repeatable continual-learning report runner for the MARULHO LM head."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Sequence

import torch

from marulho.data.language_tokenizer import ByteLevelLanguageTokenizer
from marulho.evaluation.language_training_experiment import (
    _apply_cuda_math_policy,
    _resolve_device,
    _restore_cuda_math_policy,
)
from marulho.reporting.readme_reports import write_json_report_with_readme
from marulho.training.language_continual_learning import (
    LanguageContinualLearningConfig,
    run_language_continual_learning_window,
)
from marulho.training.language_model import (
    LanguageModelConfig,
    MarulhoLanguageModel,
    build_language_model_splits,
)


SURFACE = "marulho_language_continual_learning_experiment.v1"
ARTIFACT_KIND = "marulho_language_continual_learning_window"

DEFAULT_OLD_CORPUS = (
    "Old replay domain preserves runtime truth, checkpoint rollback evidence, "
    "bounded specialist routing, and retained source-window language. "
    "The old domain repeats audit terms so heldout replay retention has enough "
    "tokens to measure forgetting under sampled vocabulary training. "
) * 320

DEFAULT_NEW_CORPUS = (
    "New continual domain updates MARULHO-owned language weights with sparse "
    "sampled vocabulary rows, replay protection, deferred metric readbacks, "
    "phase timings, and GPU-first throughput evidence. "
    "The new domain repeats adaptation terms so online learning can improve "
    "heldout loss without hiding forgetting or replay cost. "
) * 320


@dataclass(frozen=True)
class LanguageContinualLearningExperimentConfig:
    model_vocab_size: int = 0
    sampled_vocab_size: int = 0
    sparse_vocab_optimizer: bool = True
    embedding_dim: int = 32
    state_dim: int = 64
    expert_count: int = 8
    active_expert_count: int = 2
    route_candidate_count: int = 4
    expert_hidden_dim: int = 96
    recurrent_gradient_horizon: int = 0
    sequence_length: int = 32
    stride: int = 16
    batch_size: int = 8
    eval_fraction: float = 0.2
    max_old_eval_batches: int = 0
    max_new_eval_batches: int = 0
    max_new_batches: int = 4
    max_replay_batches: int = 4
    learning_rate: float = 2e-3
    max_steps: int = 2
    replay_loss_weight: float = 0.25
    max_grad_norm: float = 1.0
    gradient_clip_interval: int = 1
    forgetting_tolerance: float = 100.0
    replay_retention_tolerance: float = 100.0
    rollback_on_forgetting: bool = False
    collect_training_telemetry: bool = False
    cuda_allow_tf32: bool = True
    cuda_float32_matmul_precision: str = "high"
    seed: int = 20260704
    device: str = "auto"


def _read_text(path: str | Path | None, *, default: str) -> tuple[str, str]:
    if path is None:
        return default, "default_inline"
    resolved = Path(path)
    text = resolved.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"Language continual corpus is empty: {resolved}")
    return text, str(resolved)


def _model_config(
    tokenizer: ByteLevelLanguageTokenizer,
    config: LanguageContinualLearningExperimentConfig,
) -> LanguageModelConfig:
    model_vocab_size = (
        int(config.model_vocab_size)
        if int(config.model_vocab_size) > 0
        else int(tokenizer.vocab_size)
    )
    if model_vocab_size < int(tokenizer.vocab_size):
        raise ValueError("model_vocab_size must be at least tokenizer vocab size")
    sampled_vocab_size = max(0, int(config.sampled_vocab_size))
    if sampled_vocab_size >= model_vocab_size:
        raise ValueError("sampled_vocab_size must be smaller than model_vocab_size")
    sparse_vocab_gradients = bool(
        config.sparse_vocab_optimizer and sampled_vocab_size > 0
    )
    return LanguageModelConfig(
        vocab_size=model_vocab_size,
        embedding_dim=int(config.embedding_dim),
        state_dim=int(config.state_dim),
        expert_count=int(config.expert_count),
        active_expert_count=int(config.active_expert_count),
        route_candidate_count=int(config.route_candidate_count),
        expert_hidden_dim=int(config.expert_hidden_dim),
        sampled_vocab_size=sampled_vocab_size,
        sampled_vocab_sparse_lm_head_gradient=sparse_vocab_gradients,
        sparse_token_embedding_gradients=sparse_vocab_gradients,
        generation_vocab_size=(
            int(tokenizer.vocab_size)
            if model_vocab_size > int(tokenizer.vocab_size)
            else 0
        ),
        recurrent_gradient_horizon=max(0, int(config.recurrent_gradient_horizon)),
    )


def _trim(batches: Sequence[Any], limit: int) -> tuple[Any, ...]:
    if int(limit) <= 0:
        return tuple(batches)
    return tuple(batches[: int(limit)])


def _load_report(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    resolved = Path(path)
    if not resolved.exists():
        return None
    return json.loads(resolved.read_text(encoding="utf-8"))


def _percent_delta(current: float, baseline: float) -> float | None:
    if baseline == 0.0:
        return None
    return ((current - baseline) / baseline) * 100.0


def _phase(report: dict[str, Any], key: str) -> float:
    evidence = report.get("learning_evidence") or {}
    timings = evidence.get("window_phase_timings") or {}
    return float(timings.get(key, 0.0) or 0.0)


def _same_shape_comparison(
    report: dict[str, Any],
    *,
    comparison_report_path: str | Path | None,
    original_baseline_report_path: str | Path | None,
    precompute_report_path: str | Path | None,
    deferred_metric_report_path: str | Path | None,
) -> dict[str, Any]:
    evidence = report.get("learning_evidence") or {}
    current_update_tps = float(evidence.get("tokens_per_second", 0.0) or 0.0)
    current_total_tps = float(
        evidence.get("total_window_tokens_per_second", 0.0) or 0.0
    )
    comparison = _load_report(comparison_report_path)
    original = _load_report(original_baseline_report_path)
    precompute = _load_report(precompute_report_path)
    deferred_metric = _load_report(deferred_metric_report_path)

    payload: dict[str, Any] = {
        "surface": "marulho_language_continual_learning_same_shape_comparison.v1",
        "comparison_report": (
            str(comparison_report_path) if comparison_report_path is not None else None
        ),
        "original_baseline_report": (
            str(original_baseline_report_path)
            if original_baseline_report_path is not None
            else None
        ),
        "precompute_report": (
            str(precompute_report_path) if precompute_report_path is not None else None
        ),
        "deferred_metric_report": (
            str(deferred_metric_report_path)
            if deferred_metric_report_path is not None
            else None
        ),
        "current_update_tokens_per_second": current_update_tps,
        "current_total_window_tokens_per_second": current_total_tps,
        "current_eval_metric_readback_mode": report["old_domain_before"].get(
            "metric_readback_mode"
        ),
        "current_eval_per_batch_metric_cpu_sync": report["old_domain_before"].get(
            "per_batch_metric_cpu_sync"
        ),
        "current_pre_update_evaluation_seconds": _phase(
            report,
            "pre_update_evaluation_seconds",
        ),
        "current_post_update_evaluation_seconds": _phase(
            report,
            "post_update_evaluation_seconds",
        ),
        "same_model_vocab_size": True,
        "same_sampled_vocab_size": True,
        "same_update_token_count": True,
        "same_old_eval_batch_count": True,
        "same_new_eval_batch_count": True,
        "notes": (
            "Same sampled/padded continual-learning shape when model vocab, "
            "sampled vocab, update batch count, replay batch count, max steps, "
            "and heldout eval batch counts match. Corpus text may differ across "
            "retained artifacts, so this is throughput and sync-boundary evidence, "
            "not a quality comparison."
        ),
    }
    if comparison is not None:
        comparison_evidence = comparison.get("learning_evidence") or {}
        comparison_update_tps = float(
            comparison_evidence.get("tokens_per_second", 0.0) or 0.0
        )
        comparison_total_tps = float(
            comparison_evidence.get("total_window_tokens_per_second", 0.0) or 0.0
        )
        comparison_old_eval_batch_count = int(
            comparison["old_domain_before"].get("eval_batch_count", 0) or 0
        )
        comparison_new_eval_batch_count = int(
            comparison["new_domain_before"].get("eval_batch_count", 0) or 0
        )
        current_old_eval_batch_count = int(
            report["old_domain_before"].get("eval_batch_count", 0) or 0
        )
        current_new_eval_batch_count = int(
            report["new_domain_before"].get("eval_batch_count", 0) or 0
        )
        payload.update(
            {
                "comparison_update_tokens_per_second": comparison_update_tps,
                "comparison_total_window_tokens_per_second": comparison_total_tps,
                "delta_vs_comparison_update_tokens_per_second": (
                    current_update_tps - comparison_update_tps
                ),
                "delta_vs_comparison_update_percent": _percent_delta(
                    current_update_tps,
                    comparison_update_tps,
                ),
                "delta_vs_comparison_total_window_tokens_per_second": (
                    current_total_tps - comparison_total_tps
                ),
                "delta_vs_comparison_total_window_percent": _percent_delta(
                    current_total_tps,
                    comparison_total_tps,
                ),
                "comparison_pre_update_evaluation_seconds": _phase(
                    comparison,
                    "pre_update_evaluation_seconds",
                ),
                "comparison_post_update_evaluation_seconds": _phase(
                    comparison,
                    "post_update_evaluation_seconds",
                ),
                "comparison_old_eval_batch_count": comparison_old_eval_batch_count,
                "comparison_new_eval_batch_count": comparison_new_eval_batch_count,
                "current_old_eval_batch_count": current_old_eval_batch_count,
                "current_new_eval_batch_count": current_new_eval_batch_count,
                "same_model_vocab_size": int(report.get("model_vocab_size", 0) or 0)
                == int(comparison.get("model_vocab_size", 0) or 0),
                "same_sampled_vocab_size": int(report.get("sampled_vocab_size", 0) or 0)
                == int(comparison.get("sampled_vocab_size", 0) or 0),
                "same_update_token_count": int(
                    evidence.get("update_token_count", 0) or 0
                )
                == int(comparison_evidence.get("update_token_count", 0) or 0),
                "same_old_eval_batch_count": current_old_eval_batch_count
                == comparison_old_eval_batch_count,
                "same_new_eval_batch_count": current_new_eval_batch_count
                == comparison_new_eval_batch_count,
                "delta_pre_update_evaluation_seconds": _phase(
                    report,
                    "pre_update_evaluation_seconds",
                )
                - _phase(comparison, "pre_update_evaluation_seconds"),
                "delta_post_update_evaluation_seconds": _phase(
                    report,
                    "post_update_evaluation_seconds",
                )
                - _phase(comparison, "post_update_evaluation_seconds"),
            }
        )
    if original is not None:
        original_tps = float(
            (original.get("learning_evidence") or {}).get("tokens_per_second", 0.0)
            or 0.0
        )
        payload["original_baseline_tokens_per_second"] = original_tps
        payload["delta_vs_original_baseline_percent"] = _percent_delta(
            current_update_tps,
            original_tps,
        )
    if precompute is not None:
        precompute_tps = float(
            (precompute.get("learning_evidence") or {}).get("tokens_per_second", 0.0)
            or 0.0
        )
        payload["precompute_tokens_per_second"] = precompute_tps
        payload["delta_vs_precompute_percent"] = _percent_delta(
            current_update_tps,
            precompute_tps,
        )
    if deferred_metric is not None:
        deferred_metric_tps = float(
            (deferred_metric.get("learning_evidence") or {}).get(
                "tokens_per_second",
                0.0,
            )
            or 0.0
        )
        payload["deferred_metric_tokens_per_second"] = deferred_metric_tps
        payload["delta_vs_deferred_metric_percent"] = _percent_delta(
            current_update_tps,
            deferred_metric_tps,
        )
    return payload


def run_language_continual_learning_experiment(
    *,
    output_path: str | Path,
    old_corpus_path: str | Path | None = None,
    new_corpus_path: str | Path | None = None,
    comparison_report_path: str | Path | None = None,
    original_baseline_report_path: str | Path | None = None,
    precompute_report_path: str | Path | None = None,
    deferred_metric_report_path: str | Path | None = None,
    config: LanguageContinualLearningExperimentConfig | None = None,
) -> dict[str, Any]:
    cfg = config or LanguageContinualLearningExperimentConfig()
    device = _resolve_device(cfg.device)
    torch.manual_seed(int(cfg.seed))
    cuda_math_policy = _apply_cuda_math_policy(device, cfg)
    try:
        tokenizer = ByteLevelLanguageTokenizer()
        old_text, old_source = _read_text(old_corpus_path, default=DEFAULT_OLD_CORPUS)
        new_text, new_source = _read_text(new_corpus_path, default=DEFAULT_NEW_CORPUS)
        model = MarulhoLanguageModel(_model_config(tokenizer, cfg)).to(device)
        old_split = build_language_model_splits(
            [old_text],
            tokenizer,
            sequence_length=int(cfg.sequence_length),
            eval_fraction=float(cfg.eval_fraction),
            stride=int(cfg.stride),
            batch_size=int(cfg.batch_size),
            device=device,
        )
        new_split = build_language_model_splits(
            [new_text],
            tokenizer,
            sequence_length=int(cfg.sequence_length),
            eval_fraction=float(cfg.eval_fraction),
            stride=int(cfg.stride),
            batch_size=int(cfg.batch_size),
            device=device,
        )
        learning_config = LanguageContinualLearningConfig(
            learning_rate=float(cfg.learning_rate),
            max_steps=int(cfg.max_steps),
            replay_loss_weight=float(cfg.replay_loss_weight),
            forgetting_tolerance=float(cfg.forgetting_tolerance),
            replay_retention_tolerance=float(cfg.replay_retention_tolerance),
            rollback_on_forgetting=bool(cfg.rollback_on_forgetting),
            sparse_vocab_optimizer=bool(cfg.sparse_vocab_optimizer),
            max_grad_norm=float(cfg.max_grad_norm),
            gradient_clip_interval=max(0, int(cfg.gradient_clip_interval)),
            collect_training_telemetry=bool(cfg.collect_training_telemetry),
        )
        used_new_batches = _trim(new_split.train, int(cfg.max_new_batches))
        used_replay_batches = _trim(old_split.train, int(cfg.max_replay_batches))
        old_eval_batches = _trim(old_split.eval, int(cfg.max_old_eval_batches))
        new_eval_batches = _trim(new_split.eval, int(cfg.max_new_eval_batches))
        report = dict(
            run_language_continual_learning_window(
                model,
                new_batches=used_new_batches,
                old_eval_batches=old_eval_batches,
                new_eval_batches=new_eval_batches,
                replay_batches=used_replay_batches,
                config=learning_config,
            )
        )
        report.update(
            {
                "surface": "marulho_language_continual_learning_window.v1",
                "output_path": str(Path(output_path)),
                "cuda_math_policy": cuda_math_policy,
                "experiment_surface": SURFACE,
                "experiment_config": asdict(cfg),
                "continual_learning_config": asdict(learning_config),
                "model_config": asdict(model.config),
                "split": {
                    "old": old_split.report,
                    "new": new_split.report,
                    "used_new_train_batches": len(used_new_batches),
                    "used_replay_batches": len(used_replay_batches),
                    "used_old_eval_batches": len(old_eval_batches),
                    "used_new_eval_batches": len(new_eval_batches),
                },
                "corpus": {
                    "old_source": old_source,
                    "new_source": new_source,
                    "old_character_count": len(old_text),
                    "new_character_count": len(new_text),
                },
            }
        )
        report["baseline_comparison"] = _same_shape_comparison(
            report,
            comparison_report_path=comparison_report_path,
            original_baseline_report_path=original_baseline_report_path,
            precompute_report_path=precompute_report_path,
            deferred_metric_report_path=deferred_metric_report_path,
        )
        report["experiment_review"] = {
            "fast_mutable_experiment": True,
            "records_actual_continual_learning": bool(
                report["learning_evidence"]["update_token_count"] > 0
            ),
            "records_forgetting": "old_domain_forgetting" in report["learning_evidence"],
            "records_replay_retention": (
                "general_replay_retention_delta" in report["learning_evidence"]
            ),
            "records_sampled_vocab_training": bool(
                report["learning_evidence"].get("sampled_vocab_training", False)
            ),
            "records_sampled_vocab_precompute": bool(
                report["learning_evidence"]["sampled_vocab_precompute"]["new_batches"][
                    "enabled"
                ]
            ),
            "records_eval_sampled_vocab_precompute": bool(
                report["learning_evidence"]["sampled_vocab_precompute"][
                    "old_eval_batches"
                ]["enabled"]
            ),
            "records_deferred_metric_readback": (
                report["learning_evidence"].get("metric_readback_mode")
                == "deferred_gpu_scalar_aggregation"
            ),
            "records_eval_metric_readback": (
                report["old_domain_before"].get("metric_readback_mode")
                == "deferred_gpu_scalar_aggregation"
            ),
            "records_window_phase_timings": bool(
                report["learning_evidence"].get("window_phase_timings")
            ),
            "records_sparse_optimizer_policy": (
                report["learning_evidence"].get("optimizer_policy")
                == "AdamW_dense_core_plus_SparseAdam_vocab_rows"
            ),
            "promotes_runtime_claim": False,
            "promotes_generation_quality_claim": False,
        }
        write_json_report_with_readme(output_path, report)
        return report
    finally:
        before_policy = cuda_math_policy.get("before")
        if isinstance(before_policy, dict):
            _restore_cuda_math_policy(before_policy)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--old-corpus", type=Path, default=None)
    parser.add_argument("--new-corpus", type=Path, default=None)
    parser.add_argument("--comparison-report", type=Path, default=None)
    parser.add_argument("--original-baseline-report", type=Path, default=None)
    parser.add_argument("--precompute-report", type=Path, default=None)
    parser.add_argument("--deferred-metric-report", type=Path, default=None)
    parser.add_argument("--model-vocab-size", type=int, default=0)
    parser.add_argument("--sampled-vocab-size", type=int, default=0)
    parser.add_argument("--disable-sparse-vocab-optimizer", action="store_true")
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--state-dim", type=int, default=64)
    parser.add_argument("--expert-count", type=int, default=8)
    parser.add_argument("--active-expert-count", type=int, default=2)
    parser.add_argument("--route-candidate-count", type=int, default=4)
    parser.add_argument("--expert-hidden-dim", type=int, default=96)
    parser.add_argument("--recurrent-gradient-horizon", type=int, default=0)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-fraction", type=float, default=0.2)
    parser.add_argument("--max-old-eval-batches", type=int, default=0)
    parser.add_argument("--max-new-eval-batches", type=int, default=0)
    parser.add_argument("--max-new-batches", type=int, default=4)
    parser.add_argument("--max-replay-batches", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--max-steps", type=int, default=2)
    parser.add_argument("--replay-loss-weight", type=float, default=0.25)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--gradient-clip-interval", type=int, default=1)
    parser.add_argument("--forgetting-tolerance", type=float, default=100.0)
    parser.add_argument("--replay-retention-tolerance", type=float, default=100.0)
    parser.add_argument("--rollback-on-forgetting", action="store_true")
    parser.add_argument("--collect-training-telemetry", action="store_true")
    parser.add_argument("--disable-cuda-tf32", action="store_true")
    parser.add_argument(
        "--cuda-float32-matmul-precision",
        choices=("highest", "high", "medium"),
        default="high",
    )
    parser.add_argument("--seed", type=int, default=20260704)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    config = LanguageContinualLearningExperimentConfig(
        model_vocab_size=args.model_vocab_size,
        sampled_vocab_size=args.sampled_vocab_size,
        sparse_vocab_optimizer=not bool(args.disable_sparse_vocab_optimizer),
        embedding_dim=args.embedding_dim,
        state_dim=args.state_dim,
        expert_count=args.expert_count,
        active_expert_count=args.active_expert_count,
        route_candidate_count=args.route_candidate_count,
        expert_hidden_dim=args.expert_hidden_dim,
        recurrent_gradient_horizon=max(0, int(args.recurrent_gradient_horizon)),
        sequence_length=args.sequence_length,
        stride=args.stride,
        batch_size=args.batch_size,
        eval_fraction=args.eval_fraction,
        max_old_eval_batches=args.max_old_eval_batches,
        max_new_eval_batches=args.max_new_eval_batches,
        max_new_batches=args.max_new_batches,
        max_replay_batches=args.max_replay_batches,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        replay_loss_weight=args.replay_loss_weight,
        max_grad_norm=args.max_grad_norm,
        gradient_clip_interval=max(0, int(args.gradient_clip_interval)),
        forgetting_tolerance=args.forgetting_tolerance,
        replay_retention_tolerance=args.replay_retention_tolerance,
        rollback_on_forgetting=bool(args.rollback_on_forgetting),
        collect_training_telemetry=bool(args.collect_training_telemetry),
        cuda_allow_tf32=not bool(args.disable_cuda_tf32),
        cuda_float32_matmul_precision=args.cuda_float32_matmul_precision,
        seed=args.seed,
        device=args.device,
    )
    report = run_language_continual_learning_experiment(
        output_path=args.output,
        old_corpus_path=args.old_corpus,
        new_corpus_path=args.new_corpus,
        comparison_report_path=args.comparison_report,
        original_baseline_report_path=args.original_baseline_report,
        precompute_report_path=args.precompute_report,
        deferred_metric_report_path=args.deferred_metric_report,
        config=config,
    )
    return 0 if report["status"] == "accepted_online_update" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
