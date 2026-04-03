from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence, cast

import numpy as np
import torch

from hecsn.config.model_config import HECSNConfig
from hecsn.config.presets import get_representation_preset, representation_preset_names
from hecsn.data.corpus_loader import SourceType
from hecsn.data.pattern_loader import load_train_eval_windows
from hecsn.data.rtf_encoder import RTFEncoder, RepresentationMode
from hecsn.reporting.io import write_json_file
from hecsn.training.baselines import OnlineKMeans
from hecsn.training.behavioral_metrics import (
    ascii_codes,
    clustering_metrics,
    mean,
    order_sensitivity,
    vector_completion_coherence,
)
from hecsn.training.runner_utils import set_seed
from hecsn.training.trainer import HECSNModelLite, HECSNTrainer


def _feature_windows(encoder: RTFEncoder, windows: Iterable[str]) -> list[torch.Tensor]:
    return [encoder.feature_vector(ascii_codes(window)) for window in windows]


def _label_entropy_bits(labels: Sequence[int]) -> float:
    if not labels:
        return float("nan")
    counts = Counter(labels)
    total = float(sum(counts.values()))
    entropy = 0.0
    for count in counts.values():
        prob = float(count) / total
        entropy -= prob * float(np.log2(max(prob, 1e-12)))
    return float(entropy)


def _run_online_kmeans(
    train_features: Sequence[torch.Tensor],
    eval_features: Sequence[torch.Tensor],
    n_clusters: int,
) -> dict[str, Any]:
    if not train_features or not eval_features:
        raise ValueError("Online baseline requires non-empty train and eval features")

    cluster_count = max(1, min(int(n_clusters), len(train_features)))
    baseline = OnlineKMeans(n_clusters=cluster_count, feature_dim=int(train_features[0].numel()))
    baseline.fit(train_features)
    labels = baseline.predict_many(eval_features)
    silhouette, davies_bouldin, status = clustering_metrics(eval_features, labels)
    return {
        "n_clusters": int(cluster_count),
        "cluster_count_used": int(len(set(labels))),
        "winner_entropy_bits": _label_entropy_bits(labels),
        "mean_assignment_distance": baseline.mean_assignment_distance(eval_features),
        "silhouette": silhouette,
        "davies_bouldin": davies_bouldin,
        "clustering_status": status,
    }


def _mean_reconstruction_error(trainer: HECSNTrainer, patterns: Sequence[torch.Tensor]) -> float:
    values = [trainer.reconstruction_error(pattern) for pattern in patterns]
    return mean(values)


def _run_hecsn_competitive_only(
    representation: RepresentationMode,
    train_windows: Sequence[str],
    eval_windows: Sequence[str],
    window_size: int,
    n_columns: int,
    column_latent_dim: int,
    memory_capacity: int,
    seed: int,
    hashed_ngram_dim: int,
    hashed_ngram_min_n: int,
    hashed_ngram_max_n: int,
) -> dict[str, Any]:
    cfg = HECSNConfig(
        window_size=window_size,
        n_columns=n_columns,
        column_latent_dim=column_latent_dim,
        memory_capacity=memory_capacity,
        bootstrap_tokens=0,
        input_representation=representation,
        hashed_ngram_dim=hashed_ngram_dim,
        hashed_ngram_min_n=hashed_ngram_min_n,
        hashed_ngram_max_n=hashed_ngram_max_n,
        slow_memory_start_tokens=max(1, len(train_windows) + len(eval_windows) + 1),
        micro_sleep_interval_tokens=max(10_000, len(train_windows) + len(eval_windows) + 1),
        deep_sleep_interval_tokens=max(20_000, 2 * (len(train_windows) + len(eval_windows) + 1)),
    )
    encoder = RTFEncoder.from_config(cfg)
    train_patterns = _feature_windows(encoder, train_windows)
    eval_patterns = _feature_windows(encoder, eval_windows)

    set_seed(seed)
    model = HECSNModelLite(cfg)
    trainer = HECSNTrainer(model, cfg)
    for raw_window, pattern in zip(train_windows, train_patterns):
        trainer.train_step(pattern, raw_window=raw_window)

    labels = [trainer.winner_for_pattern(pattern) for pattern in eval_patterns]
    routing_keys = [trainer.routing_key_for_pattern(pattern).detach().cpu() for pattern in eval_patterns]
    silhouette, davies_bouldin, status = clustering_metrics(routing_keys, labels)
    return {
        "n_columns": int(n_columns),
        "column_latent_dim": int(column_latent_dim),
        "cluster_count_used": int(len(set(labels))),
        "winner_entropy_bits": _label_entropy_bits(labels),
        "mean_reconstruction_error": _mean_reconstruction_error(trainer, eval_patterns),
        "silhouette": silhouette,
        "davies_bouldin": davies_bouldin,
        "clustering_status": status,
        "sleep_events": int(trainer.sleep_events),
        "memory_warm_started": bool(trainer.memory_warm_started),
    }


def run_representation_benchmark(
    source: str,
    source_type: SourceType,
    hf_config: Optional[str],
    text_field: str,
    train_tokens: int,
    eval_tokens: int,
    window_size: int,
    representations: Sequence[RepresentationMode],
    hashed_ngram_dim: int,
    hashed_ngram_min_n: int,
    hashed_ngram_max_n: int,
    n_columns: int,
    column_latent_dim: int,
    memory_capacity: int,
    baseline_clusters: int,
    probe_samples: int,
    output_dir: Path,
    seed: int,
) -> None:
    train_windows, eval_windows = load_train_eval_windows(
        source=source,
        source_type=source_type,
        hf_config=hf_config,
        text_field=text_field,
        window_size=window_size,
        train_tokens=train_tokens,
        eval_tokens=eval_tokens,
    )
    if not train_windows:
        raise ValueError("Representation benchmark produced no training windows")
    if not eval_windows:
        raise ValueError("Representation benchmark produced no evaluation windows")

    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for representation in representations:
        encoder = RTFEncoder(
            window_size=window_size,
            representation=representation,
            hashed_ngram_dim=hashed_ngram_dim,
            hashed_ngram_min_n=hashed_ngram_min_n,
            hashed_ngram_max_n=hashed_ngram_max_n,
        )
        train_features = _feature_windows(encoder, train_windows)
        eval_features = _feature_windows(encoder, eval_windows)
        feature_for_window = lambda window: encoder.feature_vector(ascii_codes(window))
        results.append(
            {
                "representation": representation,
                "feature_dim": int(encoder.output_dim),
                "order_sensitivity": order_sensitivity(feature_for_window, eval_windows, probe_samples),
                "completion_coherence": vector_completion_coherence(feature_for_window, eval_windows, probe_samples),
                "online_kmeans": _run_online_kmeans(train_features, eval_features, baseline_clusters),
                "hecsn_competitive_only": _run_hecsn_competitive_only(
                    representation=representation,
                    train_windows=train_windows,
                    eval_windows=eval_windows,
                    window_size=window_size,
                    n_columns=n_columns,
                    column_latent_dim=column_latent_dim,
                    memory_capacity=memory_capacity,
                    seed=seed,
                    hashed_ngram_dim=hashed_ngram_dim,
                    hashed_ngram_min_n=hashed_ngram_min_n,
                    hashed_ngram_max_n=hashed_ngram_max_n,
                ),
            }
        )

    summary = {
        "protocol": "representation_benchmark",
        "data_setup": {
            "source": source,
            "source_type": source_type,
            "hf_config": hf_config,
            "text_field": text_field,
            "window_size": int(window_size),
            "train_tokens": int(train_tokens),
            "eval_tokens": int(eval_tokens),
            "train_windows": int(len(train_windows)),
            "eval_windows": int(len(eval_windows)),
        },
        "benchmark_setup": {
            "representations": list(representations),
            "hashed_ngram_dim": int(hashed_ngram_dim),
            "hashed_ngram_min_n": int(hashed_ngram_min_n),
            "hashed_ngram_max_n": int(hashed_ngram_max_n),
            "n_columns": int(n_columns),
            "column_latent_dim": int(column_latent_dim),
            "memory_capacity": int(memory_capacity),
            "baseline_clusters": int(baseline_clusters),
            "probe_samples": int(probe_samples),
            "seed": int(seed),
        },
        "results": results,
    }
    write_json_file(output_dir / "summary.json", summary)

    print("HECSN representation benchmark")
    print(f"output_dir={output_dir}")
    print(f"summary_json={output_dir / 'summary.json'}")
    for result in results:
        hecsn = result["hecsn_competitive_only"]
        baseline = result["online_kmeans"]
        print(
            f"representation={result['representation']} "
            f"hecsn_silhouette={hecsn['silhouette']} "
            f"baseline_silhouette={baseline['silhouette']}"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    preset_parser = argparse.ArgumentParser(add_help=False)
    preset_parser.add_argument("--preset", choices=representation_preset_names(), default=None)
    preset_args, _ = preset_parser.parse_known_args()
    preset_defaults = get_representation_preset(preset_args.preset)

    parser = argparse.ArgumentParser(description="Benchmark HECSN input representations against a simple online baseline")
    parser.add_argument("--preset", choices=representation_preset_names(), default=preset_args.preset)
    parser.add_argument("--source", type=str, default=preset_defaults.get("source"), help="File path or HuggingFace dataset id")
    parser.add_argument("--source-type", choices=["auto", "file", "hf", "web"], default=preset_defaults.get("source_type", "auto"))
    parser.add_argument("--hf-config", type=str, default=preset_defaults.get("hf_config"))
    parser.add_argument("--text-field", type=str, default=preset_defaults.get("text_field", "text"))
    parser.add_argument("--train-tokens", type=int, default=preset_defaults.get("train_tokens", 4000))
    parser.add_argument("--eval-tokens", type=int, default=preset_defaults.get("eval_tokens", 1000))
    parser.add_argument("--window-size", type=int, default=preset_defaults.get("window_size", 10))
    parser.add_argument(
        "--representations",
        nargs="+",
        default=preset_defaults.get("representations", ["order_weighted_ascii", "unigram_ascii", "hashed_ngram"]),
    )
    parser.add_argument("--hashed-ngram-dim", type=int, default=preset_defaults.get("hashed_ngram_dim", 2048))
    parser.add_argument("--hashed-ngram-min-n", type=int, default=preset_defaults.get("hashed_ngram_min_n", 2))
    parser.add_argument("--hashed-ngram-max-n", type=int, default=preset_defaults.get("hashed_ngram_max_n", 3))
    parser.add_argument("--n-columns", type=int, default=preset_defaults.get("n_columns", 64))
    parser.add_argument("--column-latent-dim", type=int, default=preset_defaults.get("column_latent_dim", 256))
    parser.add_argument("--memory-capacity", type=int, default=preset_defaults.get("memory_capacity", 256))
    parser.add_argument("--baseline-clusters", type=int, default=preset_defaults.get("baseline_clusters", 32))
    parser.add_argument("--probe-samples", type=int, default=preset_defaults.get("probe_samples", 128))
    parser.add_argument("--seed", type=int, default=preset_defaults.get("seed", 7))
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    output_dir = args.output_dir
    if output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = args.preset if args.preset is not None else "representation_benchmark"
        output_dir = Path("reports") / f"{suffix}_{stamp}"

    run_representation_benchmark(
        source=args.source,
        source_type=cast(SourceType, args.source_type),
        hf_config=args.hf_config,
        text_field=args.text_field,
        train_tokens=args.train_tokens,
        eval_tokens=args.eval_tokens,
        window_size=args.window_size,
        representations=[cast(RepresentationMode, value) for value in args.representations],
        hashed_ngram_dim=args.hashed_ngram_dim,
        hashed_ngram_min_n=args.hashed_ngram_min_n,
        hashed_ngram_max_n=args.hashed_ngram_max_n,
        n_columns=args.n_columns,
        column_latent_dim=args.column_latent_dim,
        memory_capacity=args.memory_capacity,
        baseline_clusters=args.baseline_clusters,
        probe_samples=args.probe_samples,
        output_dir=output_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()