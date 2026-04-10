from __future__ import annotations

import argparse
from pathlib import Path
import tempfile
from typing import Any
from unittest.mock import patch

import torch

from hecsn.config.model_config import HECSNConfig
from hecsn.reporting.io import write_json_file
from hecsn.service.manager import AUTO_REMOTE_QUERY_BUDGET_MAX, HECSNServiceManager
from hecsn.training.checkpointing import save_trainer_checkpoint
from hecsn.training.trainer import HECSNModelLite, HECSNTrainer


def run_self_expanded_curriculum_benchmark(*, output_dir: Path, seed: int = 29) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(int(seed))

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        cfg = HECSNConfig(
            n_columns=8,
            column_latent_dim=16,
            bootstrap_tokens=0,
            memory_capacity=64,
            eta_competitive=0.05,
            eta_decay=0.0,
            input_weight_blend=0.0,
            enable_context_layer=True,
            enable_binding_layer=True,
            enable_abstraction_layer=True,
        )
        trainer = HECSNTrainer(HECSNModelLite(cfg), cfg)
        checkpoint_path = save_trainer_checkpoint(
            root / "initial_curriculum.pt",
            trainer,
            metadata={"benchmark": "self_expanded_curriculum_smoke"},
        )
        manager = HECSNServiceManager(
            checkpoint_path,
            trace_dir=root / "traces",
        )
        observed_path = root / "observed.txt"
        observed_path.write_text("neutral background signal " * 24, encoding="utf-8")
        try:
            manager.configure_terminus(
                source_bank=[
                    {
                        "name": "observed_source",
                        "source": str(observed_path),
                        "source_type": "file",
                    }
                ],
                tick_tokens=12,
                sleep_interval_seconds=0.01,
                repeat_sources=True,
                autonomy={
                    "enabled": True,
                    "policy": "active",
                    "candidate_bank": [
                        {
                            "name": "live_remote_pool",
                            "catalog_mode": "live_remote_search",
                            "catalog_providers": ["wikipedia", "openalex"],
                            "catalog_queries_per_provider": 2,
                            "catalog_provider_result_limit": 4,
                            "catalog_limit": 4,
                        }
                    ],
                    "trigger_interval_tokens": 1,
                },
            )
            manager.feed(text=("river stream water current bank loan credit " * 12).strip())
            first_runtime = manager.terminus_status()["terminus_runtime"]
            focus_plan = dict(first_runtime["autonomy"]["focus_plan"] or {})
            retrieval_queries = [str(item).strip().lower() for item in list(focus_plan.get("retrieval_queries") or []) if str(item).strip()]
            selected_query = retrieval_queries[0] if retrieval_queries else "bank credit loan current water stream"

            with patch(
                "hecsn.service.manager.run_live_acquisition",
                side_effect=[
                    {
                        "policy": "active",
                        "tokens_trained_total": 32,
                        "acquired_sources": ["wikipedia_gap_source"],
                        "semantic_plan": focus_plan,
                        "acquisition_history": [
                            {
                                "selected_source": "wikipedia_gap_source",
                                "selected_provider": "wikipedia",
                                "selected_query_text": selected_query,
                                "selected_semantic_relevance": 0.91,
                                "selected_gap_reduction": 0.22,
                                "selected_diagnostic_gap_reduction": 0.31,
                                "tokens_trained": 32,
                                "selected_metadata": {
                                    "provider": "wikipedia",
                                    "query_text": selected_query,
                                    "semantic_relevance": 0.91,
                                    "catalog_terms": ["river current", "bank finance"],
                                },
                                "candidate_snapshot": {
                                    "wikipedia_gap_source": {
                                        "semantic_answerability": 0.22,
                                        "concept_uncertainty": 0.72,
                                        "concept_support": 0.18,
                                        "semantic_weak_concept_pressure": 0.76,
                                    }
                                },
                                "selected_semantic_answerability_after": 0.64,
                                "selected_concept_uncertainty_after": 0.28,
                                "selected_concept_support_after": 0.58,
                                "selected_weak_concept_pressure_after": 0.18,
                            }
                        ],
                    },
                    {
                        "policy": "active",
                        "tokens_trained_total": 0,
                        "acquired_sources": [],
                        "semantic_plan": focus_plan,
                        "acquisition_history": [],
                    },
                ],
            ) as mocked_acquire:
                manager.terminus_tick()
                manager.terminus_tick()

            first_kwargs = mocked_acquire.call_args_list[0].kwargs
            second_kwargs = mocked_acquire.call_args_list[1].kwargs
            runtime = manager.terminus_status()["terminus_runtime"]
            provider_curriculum = runtime["autonomy"]["provider_curriculum"] or {}
            ranked_providers = list(provider_curriculum.get("ranked_providers") or [])
            top_provider = dict(ranked_providers[0] if ranked_providers else {})
            first_spec = dict(first_kwargs["candidate_bank_specs"][0])
            second_spec = dict(second_kwargs["candidate_bank_specs"][0])
        finally:
            manager.close()

    summary = {
        "benchmark": "self_expanded_curriculum_smoke",
        "seed": int(seed),
        "focus_plan": focus_plan,
        "first_spec": {
            "catalog_providers": list(first_spec.get("catalog_providers") or []),
            "catalog_queries_per_provider": int(first_spec.get("catalog_queries_per_provider", 0)),
            "catalog_query_family_budget_bonus": int(first_spec.get("catalog_query_family_budget_bonus", 0)),
        },
        "second_spec": {
            "catalog_providers": list(second_spec.get("catalog_providers") or []),
            "catalog_queries_per_provider": int(second_spec.get("catalog_queries_per_provider", 0)),
            "catalog_provider_query_families": dict(second_spec.get("catalog_provider_query_families") or {}),
            "catalog_query_family_budget_bonus": int(second_spec.get("catalog_query_family_budget_bonus", 0)),
        },
        "provider_curriculum": {
            "focus_terms": list(provider_curriculum.get("focus_terms") or []),
            "top_provider": {
                "provider": str(top_provider.get("provider", "")),
                "query_family_strength": float(top_provider.get("query_family_strength", 0.0)),
                "query_family_focus_score": float(top_provider.get("query_family_focus_score", 0.0)),
                "query_family_query_bonus": int(top_provider.get("query_family_query_bonus", 0)),
                "matched_query_families": list(top_provider.get("matched_query_families") or []),
                "query_families": dict(top_provider.get("query_families") or {}),
            },
        },
    }
    second_provider_query_families = dict(summary["second_spec"]["catalog_provider_query_families"] or {})
    summary["self_expanded_curriculum_gate"] = {
        "pass": bool(
            list(focus_plan.get("geometric_gaps") or [])
            and second_spec.get("catalog_providers", [None])[0] == "wikipedia"
            and int(second_spec.get("catalog_queries_per_provider", 0))
            >= int(first_spec.get("catalog_queries_per_provider", 0))
            and int(second_spec.get("catalog_query_family_budget_bonus", 0)) >= 1
            and bool(second_provider_query_families.get("wikipedia"))
            and selected_query in [str(item).strip().lower() for item in second_provider_query_families.get("wikipedia", [])]
            and float(top_provider.get("query_family_strength", 0.0)) > 0.0
        ),
        "thresholds": {
            "geometric_gaps": True,
            "query_budget_nondecrease": True,
            "query_budget_cap": int(AUTO_REMOTE_QUERY_BUDGET_MAX),
            "query_family_budget_bonus_gte": 1,
            "top_provider_query_family_strength_gt": 0.0,
        },
    }
    write_json_file(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the self-expanded curriculum smoke.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports") / "phase7_self_expanded_curriculum_smoke",
        help="Directory where the benchmark summary will be written.",
    )
    parser.add_argument("--seed", type=int, default=29, help="Deterministic seed for the benchmark.")
    args = parser.parse_args()

    summary = run_self_expanded_curriculum_benchmark(output_dir=args.output_dir, seed=args.seed)
    print(f"[self_expanded_curriculum_smoke] pass={summary['self_expanded_curriculum_gate']['pass']}")


if __name__ == "__main__":
    main()
