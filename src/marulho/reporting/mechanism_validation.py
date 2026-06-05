from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def write_mechanism_validation_metrics_csv(path: Path, metrics_rows: list[dict[str, Any]]) -> None:
    fields = [
        "token",
        "drift",
        "surprise",
        "sparsity",
        "winner",
        "pred_error",
        "recon_error",
        "sleep_triggered",
        "sleep_type",
        "sleep_replay_updates",
        "sleep_events_total",
        "micro_sleep_events_total",
        "deep_sleep_events_total",
        "deep_sleep_emergency",
        "drift_floor",
        "drift_floor_rising",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in metrics_rows:
            writer.writerow(
                {
                    "token": row.get("token", ""),
                    "drift": row.get("drift", ""),
                    "surprise": row.get("surprise", ""),
                    "sparsity": row.get("sparsity", ""),
                    "winner": row.get("winner", ""),
                    "pred_error": row.get("pred_error", ""),
                    "recon_error": row.get("recon_error", ""),
                    "sleep_triggered": row.get("sleep_triggered", ""),
                    "sleep_type": row.get("sleep_type", ""),
                    "sleep_replay_updates": row.get("sleep_replay_updates", ""),
                    "sleep_events_total": row.get("sleep_events_total", ""),
                    "micro_sleep_events_total": row.get("micro_sleep_events_total", ""),
                    "deep_sleep_events_total": row.get("deep_sleep_events_total", ""),
                    "deep_sleep_emergency": row.get("deep_sleep_emergency", ""),
                    "drift_floor": row.get("drift_floor", ""),
                    "drift_floor_rising": row.get("drift_floor_rising", ""),
                }
            )


def _status_style(is_good: bool) -> tuple[str, str]:
    if is_good:
        return "GOOD", "#2e7d32"
    return "BAD", "#c62828"


def _plot_mechanism_validation_scorecard(output_dir: Path, summary: dict[str, Any]) -> None:
    gate = summary.get("mechanism_validation_gate", {})
    thresholds = gate.get("thresholds", {})

    checks = [
        (
            "Cluster separation",
            bool(gate.get("gate_clustering", False)),
            f"GOOD if silhouette >= {thresholds.get('silhouette_min')} or DBI <= {thresholds.get('davies_bouldin_max')}",
        ),
        (
            "Recon trend",
            bool(gate.get("gate_reconstruction_trend", False)),
            f"GOOD if slope < {thresholds.get('recon_slope_max')}",
        ),
        (
            "Ablation superiority",
            bool(gate.get("gate_ablation_superiority", False)),
            "GOOD if trained model beats random assignment baseline",
        ),
        (
            "Winner entropy",
            bool(gate.get("gate_winner_entropy", False)),
            f"GOOD if entropy >= {thresholds.get('winner_entropy_min_bits')} bits",
        ),
    ]

    labels = [check[0] for check in checks]
    status_vals = [1 if check[1] else 0 for check in checks]
    colors = ["#2e7d32" if check[1] else "#c62828" for check in checks]

    fig, ax = plt.subplots(figsize=(11, 5), dpi=120)
    y = np.arange(len(labels))
    bars = ax.barh(y, status_vals, color=colors, alpha=0.9)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 1.25)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["BAD", "GOOD"])
    ax.set_xlabel("Status")
    ax.set_title("Mechanism-Validation Scorecard")

    for idx, bar in enumerate(bars):
        state, state_color = _status_style(status_vals[idx] == 1)
        ax.text(
            1.02,
            bar.get_y() + bar.get_height() / 2,
            state,
            va="center",
            ha="left",
            color=state_color,
            fontsize=10,
            fontweight="bold",
        )
        ax.text(
            0.02,
            bar.get_y() + bar.get_height() / 2,
            checks[idx][2],
            va="center",
            ha="left",
            color="#333333",
            fontsize=8,
        )

    overall = bool(gate.get("pass", False))
    overall_text, overall_color = _status_style(overall)
    fig.suptitle(
        f"Overall Mechanism-Validation Result: {overall_text}",
        color=overall_color,
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(output_dir / "mechanism_validation_scorecard.png")
    plt.close(fig)


def plot_mechanism_validation_artifacts(output_dir: Path, metrics_rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    if not metrics_rows:
        return

    tokens = [int(row["token"]) for row in metrics_rows]
    drifts = [float(row["drift"]) for row in metrics_rows]
    surprises = [float(row["surprise"]) for row in metrics_rows]
    recon_errors = [float(row["recon_error"]) for row in metrics_rows]
    winners = [int(row["winner"]) for row in metrics_rows]

    gate = summary.get("mechanism_validation_gate", {})
    thresholds = gate.get("thresholds", {})
    drift_good = float(summary.get("drift_slope", 0.0)) <= 0.0
    recon_good = bool(gate.get("gate_reconstruction_trend", False))
    entropy_good = bool(gate.get("gate_winner_entropy", False))
    clustering_good = bool(gate.get("gate_clustering", False))
    benchmark_pass = bool(gate.get("pass", False))
    winner_entropy = float(summary.get("winner_entropy_bits", 0.0))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=120)

    axes[0, 0].plot(tokens, drifts, color="#3366cc", linewidth=1.5)
    axes[0, 0].set_title("Drift Over Tokens")
    axes[0, 0].set_xlabel("Token")
    axes[0, 0].set_ylabel("Drift")
    drift_label, drift_color = _status_style(drift_good)
    axes[0, 0].text(
        0.02,
        0.94,
        f"{drift_label}: slope <= 0 means memory drift not exploding",
        transform=axes[0, 0].transAxes,
        va="top",
        color=drift_color,
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85},
    )

    axes[0, 1].plot(tokens, surprises, color="#dc3912", linewidth=1.5)
    axes[0, 1].set_title("Surprise Over Tokens")
    axes[0, 1].set_xlabel("Token")
    axes[0, 1].set_ylabel("Surprise")
    surprise_abs_mean = float(np.mean(np.abs(np.asarray(surprises, dtype=np.float32))))
    surprise_good = surprise_abs_mean <= 0.25
    surprise_label, surprise_color = _status_style(surprise_good)
    axes[0, 1].text(
        0.02,
        0.94,
        f"{surprise_label}: lower sustained surprise is better (mean|s|={surprise_abs_mean:.3f})",
        transform=axes[0, 1].transAxes,
        va="top",
        color=surprise_color,
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85},
    )

    axes[1, 0].plot(tokens, recon_errors, color="#109618", linewidth=1.5)
    axes[1, 0].set_title("Reconstruction Error")
    axes[1, 0].set_xlabel("Token")
    axes[1, 0].set_ylabel("Recon Error")
    recon_label, recon_color = _status_style(recon_good)
    axes[1, 0].text(
        0.02,
        0.94,
        (
            f"{recon_label}: slope < {thresholds.get('recon_slope_max')} means"
            " representation is improving"
        ),
        transform=axes[1, 0].transAxes,
        va="top",
        color=recon_color,
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85},
    )

    axes[1, 1].hist(winners, bins=max(5, len(set(winners))), color="#ff9900", alpha=0.85)
    axes[1, 1].set_title("Winner Distribution")
    axes[1, 1].set_xlabel("Winner Column")
    axes[1, 1].set_ylabel("Count")
    entropy_label, entropy_color = _status_style(entropy_good)
    axes[1, 1].text(
        0.02,
        0.94,
        (
            f"{entropy_label}: entropy >= {thresholds.get('winner_entropy_min_bits')} bits"
            f" (observed={winner_entropy:.3f})"
        ),
        transform=axes[1, 1].transAxes,
        va="top",
        color=entropy_color,
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85},
    )
    cluster_label, cluster_color = _status_style(clustering_good)
    axes[1, 1].text(
        0.02,
        0.80,
        f"{cluster_label}: cluster separation gate",
        transform=axes[1, 1].transAxes,
        va="top",
        color=cluster_color,
        fontsize=8,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85},
    )

    overall_text, overall_color = _status_style(benchmark_pass)
    fig.suptitle(
        f"Mechanism-Validation Diagnostics: {overall_text}",
        color=overall_color,
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(output_dir / "mechanism_validation_metrics.png")
    plt.close(fig)

    pred_rows = [row for row in metrics_rows if row.get("pred_error") is not None]
    if pred_rows:
        p_tokens = [int(row["token"]) for row in pred_rows]
        p_errors = [float(row["pred_error"]) for row in pred_rows]

        fig2, ax = plt.subplots(figsize=(10, 4), dpi=120)
        ax.plot(p_tokens, p_errors, color="#990099", linewidth=1.5)
        ax.set_title("Bootstrap Prediction Error")
        ax.set_xlabel("Token")
        ax.set_ylabel("KL Error")
        ax.text(
            0.02,
            0.94,
            "GOOD: downward trend means early predictive coding is stabilizing",
            transform=ax.transAxes,
            va="top",
            color="#2e7d32",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85},
        )
        fig2.tight_layout()
        fig2.savefig(output_dir / "bootstrap_prediction_error.png")
        plt.close(fig2)

    _plot_mechanism_validation_scorecard(output_dir, summary)
