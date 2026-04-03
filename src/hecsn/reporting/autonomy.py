from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def plot_autonomy_summary(output_dir: Path, summary: dict[str, Any]) -> None:
    comparison = summary["comparison"]
    active_history = summary["policy_results"]["active"]["episode_history"]
    rr_history = summary["policy_results"]["round_robin"]["episode_history"]
    source_names = summary["source_names"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=120)

    def plot_policy(ax: Any, rows: list[dict[str, Any]], title: str) -> None:
        seek_rows = [row for row in rows if row["phase"] == "seek"]
        xs = list(range(1, len(seek_rows) + 1))
        ys = [float(row["selected_gap_before"]) for row in seek_rows]
        ax.plot(xs, ys, marker="o", linewidth=1.8, color="#1565c0")
        for idx, row in enumerate(seek_rows, start=1):
            ax.annotate(
                str(row["selected_source"]),
                (idx, float(row["selected_gap_before"])),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=8,
            )
        ax.set_title(title)
        ax.set_xlabel("Seek episode")
        ax.set_ylabel("Selected source gap")
        ax.grid(alpha=0.2)

    plot_policy(axes[0], active_history, "Active Seeking")
    plot_policy(axes[1], rr_history, "Round Robin")

    x = np.arange(len(source_names))
    width = 0.35
    active_final = [float(summary["policy_results"]["active"]["final_gap_by_source"][name]) for name in source_names]
    rr_final = [float(summary["policy_results"]["round_robin"]["final_gap_by_source"][name]) for name in source_names]
    axes[2].bar(x - width / 2, active_final, width=width, color="#2e7d32", label="Active")
    axes[2].bar(x + width / 2, rr_final, width=width, color="#ef6c00", label="Round robin")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(source_names)
    axes[2].set_title("Final Gap by Source")
    axes[2].grid(axis="y", alpha=0.2)
    axes[2].legend()

    fig.suptitle(
        f"Autonomy Benchmark: active max-gap delta {comparison['active_minus_round_robin_max_gap']:.4f}",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(output_dir / "autonomy_diagnostics.png")
    plt.close(fig)


def plot_acquisition_summary(output_dir: Path, summary: dict[str, Any]) -> None:
    comparison = summary["comparison"]
    candidate_names = summary["candidate_names"]
    scout_title = "Scout and Commit"
    if "scout_commit" in summary["policy_results"] and int(summary["policy_results"]["scout_commit"].get("semantic_shortlist_size", 0)) > 0:
        scout_title = "Semantic Shortlist + Scout"
    policy_specs = [
        ("active", "Active Acquisition"),
        ("round_robin", "Round Robin Acquisition"),
    ]
    if "scout_commit" in summary["policy_results"]:
        policy_specs.insert(1, ("scout_commit", scout_title))

    fig, axes = plt.subplots(1, len(policy_specs) + 1, figsize=(5.0 * (len(policy_specs) + 1), 4.5), dpi=120)
    axes = np.atleast_1d(axes)

    def plot_history(ax: Any, rows: list[dict[str, Any]], title: str) -> None:
        xs = list(range(1, len(rows) + 1))
        ys = [float(row["selected_gap_before"]) for row in rows]
        ax.plot(xs, ys, marker="o", linewidth=1.8, color="#1565c0")
        for idx, row in enumerate(rows, start=1):
            ax.annotate(
                str(row["selected_source"]),
                (idx, float(row["selected_gap_before"])),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=8,
            )
        ax.set_title(title)
        ax.set_xlabel("Acquisition slot")
        ax.set_ylabel("Selected candidate gap")
        ax.grid(alpha=0.2)

    for axis, (policy_name, title) in zip(axes[:-1], policy_specs):
        plot_history(axis, summary["policy_results"][policy_name]["acquisition_history"], title)

    x = np.arange(len(candidate_names))
    bar_axis = axes[-1]
    width = 0.8 / max(1, len(policy_specs))
    colors = {
        "active": "#2e7d32",
        "scout_commit": "#6a1b9a",
        "round_robin": "#ef6c00",
    }
    labels = {
        "active": "Active",
        "scout_commit": "Scout+commit",
        "round_robin": "Round robin",
    }
    offsets = np.linspace(-(len(policy_specs) - 1) / 2, (len(policy_specs) - 1) / 2, len(policy_specs)) * width
    for offset, (policy_name, _) in zip(offsets, policy_specs):
        values = [float(summary["policy_results"][policy_name]["final_candidate_gap_by_source"][name]) for name in candidate_names]
        bar_axis.bar(x + offset, values, width=width, color=colors[policy_name], label=labels[policy_name])
    bar_axis.set_xticks(x)
    bar_axis.set_xticklabels(candidate_names)
    bar_axis.set_title("Final Candidate Gap")
    bar_axis.grid(axis="y", alpha=0.2)
    bar_axis.legend()

    if "scout_commit_minus_round_robin_max_candidate_gap" in comparison:
        title = f"Acquisition Benchmark: scout+commit max-gap delta {comparison['scout_commit_minus_round_robin_max_candidate_gap']:.4f}"
    else:
        title = f"Acquisition Benchmark: active max-gap delta {comparison['active_minus_round_robin_max_candidate_gap']:.4f}"
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "autonomy_acquisition_diagnostics.png")
    plt.close(fig)