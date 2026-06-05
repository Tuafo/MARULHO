from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def plot_memory_consolidation_summary(output_dir: Path, summary: dict[str, Any]) -> None:
    metrics = summary["metrics"]
    memory_before = summary["memory_stats_before_consolidation"]
    memory_after = summary["memory_stats_after_consolidation"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=120)

    stages = ["After A", "After B", "After Consolidation"]
    task_a_values = [
        float(metrics["task_a_recon_after_a"]),
        float(metrics["task_a_recon_after_b"]),
        float(metrics["task_a_recon_after_consolidation"]),
    ]
    task_b_values = [
        float(metrics["task_b_recon_before_b"]),
        float(metrics["task_b_recon_after_b"]),
        float(metrics["task_b_recon_after_consolidation"]),
    ]
    axes[0].plot(stages, task_a_values, marker="o", linewidth=2.0, color="#1b5e20", label="Task A")
    axes[0].plot(stages, task_b_values, marker="o", linewidth=2.0, color="#0d47a1", label="Task B")
    axes[0].set_title("Reconstruction Error by Milestone")
    axes[0].set_ylabel("Lower is better")
    axes[0].legend()
    axes[0].grid(alpha=0.2)

    delta_labels = ["A forgetting", "A recovery", "B consolidation shift"]
    delta_values = [
        float(metrics["task_a_forgetting_delta"]),
        float(metrics["task_a_recovery_delta"]),
        float(metrics["task_b_consolidation_shift"]),
    ]
    delta_colors = ["#c62828", "#2e7d32", "#2e7d32" if delta_values[2] <= 0.0 else "#c62828"]
    axes[1].bar(delta_labels, delta_values, color=delta_colors, alpha=0.85)
    axes[1].axhline(0.0, color="#555555", linewidth=1.0)
    axes[1].set_title("Forgetting and Recovery Deltas")
    axes[1].set_ylabel("Delta in reconstruction error")
    axes[1].grid(axis="y", alpha=0.2)

    memory_labels = ["Mean replay", "Fast EMA norm", "Slow mean norm"]
    before_values = [
        float(memory_before["mean_replay_count"]),
        float(memory_before["fast_ema_norm"]),
        float(memory_before["slow_mean_norm"]),
    ]
    after_values = [
        float(memory_after["mean_replay_count"]),
        float(memory_after["fast_ema_norm"]),
        float(memory_after["slow_mean_norm"]),
    ]
    x = np.arange(len(memory_labels))
    width = 0.36
    axes[2].bar(x - width / 2, before_values, width=width, color="#757575", label="Before consolidation")
    axes[2].bar(x + width / 2, after_values, width=width, color="#ff8f00", label="After consolidation")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(memory_labels)
    axes[2].set_title("Memory State Before vs After")
    axes[2].legend()
    axes[2].grid(axis="y", alpha=0.2)

    fig.suptitle("MARULHO Memory-Consolidation Diagnostics", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "memory_consolidation_diagnostics.png")
    plt.close(fig)


def plot_contextual_routing_summary(output_dir: Path, summary: dict[str, Any], metrics_rows: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=120)

    if metrics_rows:
        sample_step = max(1, len(metrics_rows) // 300)
        sampled = metrics_rows[::sample_step]
        tokens = [int(row["token"]) for row in sampled]
        dopamine = [float(row["dopamine"]) for row in sampled]
        serotonin = [float(row.get("serotonin", float("nan"))) for row in sampled]
        acetylcholine = [float(row["acetylcholine"]) for row in sampled]
        norepinephrine = [float(row["norepinephrine"]) for row in sampled]
        axes[0].plot(tokens, dopamine, color="#1565c0", linewidth=1.8, label="Dopamine")
        axes[0].plot(tokens, serotonin, color="#6a1b9a", linewidth=1.8, label="Serotonin")
        axes[0].plot(tokens, acetylcholine, color="#2e7d32", linewidth=1.8, label="Acetylcholine")
        axes[0].plot(tokens, norepinephrine, color="#ef6c00", linewidth=1.8, label="Norepinephrine")
        axes[0].set_title("Neuromodulator Traces")
        axes[0].set_xlabel("Token")
        axes[0].set_ylabel("Level")
        axes[0].grid(alpha=0.2)
        axes[0].legend()

    contextual_metrics = summary["contextual_routing_metrics"]
    bar_labels = [
        "Context separation",
        "Winner switch rate",
        "Probe distance",
        "B3 accuracy",
        "B3 margin",
    ]
    bar_values = [
        float(contextual_metrics["context_state_separation"]),
        float(contextual_metrics["probe_winner_switch_rate"]),
        float(contextual_metrics["probe_mean_assembly_distance"]),
        float(contextual_metrics["bank_polysemy_accuracy"]),
        float(contextual_metrics["bank_polysemy_signature_margin"]),
    ]
    axes[1].bar(bar_labels, bar_values, color=["#6a1b9a", "#00897b", "#c62828"], alpha=0.85)
    axes[1].set_title("Contextual Routing Diagnostics")
    axes[1].grid(axis="y", alpha=0.2)

    recon_labels = ["Task A recon", "Task B recon", "Binding", "Context gain"]
    recon_values = [
        float(contextual_metrics["task_a_recon_error"]),
        float(contextual_metrics["task_b_recon_error"]),
        float(summary["training_diagnostics"]["mean_binding_strength"]),
        float(summary["training_diagnostics"]["mean_context_gain"]),
    ]
    axes[2].bar(recon_labels, recon_values, color=["#1b5e20", "#0d47a1", "#8e24aa", "#546e7a"], alpha=0.85)
    axes[2].set_title("Runtime Summary")
    axes[2].grid(axis="y", alpha=0.2)

    fig.suptitle("MARULHO Contextual-Routing Diagnostics", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "contextual_routing_diagnostics.png")
    plt.close(fig)


def plot_hierarchical_scale_summary(output_dir: Path, summary: dict[str, Any], metrics_samples: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), dpi=120)

    if metrics_samples:
        tokens = [int(row["token"]) for row in metrics_samples]
        recon = [float(row["recon_error"]) for row in metrics_samples]
        drift = [float(row["drift"]) for row in metrics_samples]
        axes[0].plot(tokens, recon, color="#0d47a1", linewidth=2.0, label="Recon error")
        axes[0].plot(tokens, drift, color="#c62828", linewidth=1.8, label="Drift")
        axes[0].set_title("Training Trace")
        axes[0].set_xlabel("Token")
        axes[0].grid(alpha=0.2)
        axes[0].legend()

    routing_metrics = summary["routing_metrics"]
    integrity_metrics = summary["index_integrity"]
    bar_labels = ["Recall@k", "Top-1", "Self recall"]
    bar_values = [
        float(routing_metrics["recall_at_k"]),
        float(routing_metrics["top1_recall"]),
        float(integrity_metrics["self_recall"]),
    ]
    axes[1].bar(bar_labels, bar_values, color=["#2e7d32", "#00897b", "#6a1b9a"], alpha=0.85)
    axes[1].axhline(0.95, color="#555555", linewidth=1.0, linestyle="--")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_title(
        f"Routing Quality (mean latency={float(routing_metrics['mean_latency_ms']):.3f} ms)"
    )
    axes[1].grid(axis="y", alpha=0.2)

    sharding = summary["sharding"]
    shard_labels = [f"S{idx}" for idx in range(len(sharding["index_shard_sizes"]))]
    x = np.arange(len(shard_labels))
    width = 0.38
    axes[2].bar(x - width / 2, sharding["index_shard_sizes"], width=width, color="#1565c0", label="Index size")
    axes[2].bar(x + width / 2, sharding["primary_query_shard_counts"], width=width, color="#ef6c00", label="Top shard hits")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(shard_labels)
    axes[2].set_title("Shard Distribution")
    axes[2].grid(axis="y", alpha=0.2)
    axes[2].legend()

    gate_text = "PASS" if summary["hierarchical_scale_gate"]["pass"] else "FAIL"
    gate_color = "#2e7d32" if summary["hierarchical_scale_gate"]["pass"] else "#c62828"
    fig.suptitle(
        f"MARULHO Hierarchical-Scale Diagnostics: {gate_text}",
        fontsize=13,
        fontweight="bold",
        color=gate_color,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "hierarchical_scale_diagnostics.png")
    plt.close(fig)
