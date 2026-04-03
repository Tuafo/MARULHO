from __future__ import annotations

import random
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from hecsn.training.autonomy_acquisition_runner import (
    acquisition_gate_from_comparison,
    consume_previewed_chunk,
    execute_acquisition_policy,
    evaluate_projected_candidates,
    preview_source_chunk,
    run_live_acquisition,
    semantic_shortlist,
    select_projected_commit_target,
    select_scout_commit_target,
)
from hecsn.training.autonomy_runner import (
    ProbeGapMetrics,
    SourceBank,
    autonomy_gate_from_comparison,
    probe_diagnostics,
    select_active_source,
)


def make_bank(name: str, visits: int) -> SourceBank:
    return SourceBank(
        name=name,
        source=name,
        source_type="test",
        hf_config=None,
        text_field="text",
        probe_patterns=[],
        probe_raw_windows=[],
        train_patterns=[],
        train_raw_windows=[],
        cursor=0,
        visits=visits,
    )


def make_gap(recon_error: float, diagnostic_gap_score: float, *, info_gain_score: float | None = None) -> ProbeGapMetrics:
    return {
        "recon_error": recon_error,
        "winner_entropy_bits": 0.0,
        "winner_switch_rate": 0.0,
        "mean_top1_margin": 0.0,
        "ambiguity": 0.0,
        "exploration_bonus": 0.0,
        "gap_score": recon_error,
        "diagnostic_gap_score": diagnostic_gap_score,
        "concept_novelty": 0.0,
        "concept_uncertainty": 0.0,
        "concept_support": 0.0,
        "info_gain_score": diagnostic_gap_score if info_gain_score is None else info_gain_score,
    }


def make_chunk_bank(name: str) -> SourceBank:
    return SourceBank(
        name=name,
        source=name,
        source_type="test",
        hf_config=None,
        text_field="text",
        probe_patterns=[],
        probe_raw_windows=[],
        train_patterns=[
            torch.tensor([1.0, 0.0]),
            torch.tensor([0.0, 1.0]),
            torch.tensor([1.0, 1.0]),
        ],
        train_raw_windows=[f"{name}-a", f"{name}-b", f"{name}-c"],
        cursor=0,
        visits=0,
    )


def make_window_bank(name: str, windows: list[str]) -> SourceBank:
    return SourceBank(
        name=name,
        source=name,
        source_type="test",
        hf_config=None,
        text_field="text",
        probe_patterns=[],
        probe_raw_windows=[],
        train_patterns=[],
        train_raw_windows=windows,
        cursor=0,
        visits=0,
    )


class AutonomySelectionTests(unittest.TestCase):
    def test_autonomy_gate_scales_for_small_gap_regime(self) -> None:
        gate = autonomy_gate_from_comparison(
            {
                "round_robin_final_mean_gap": 4.0e-5,
                "round_robin_final_max_gap": 5.0e-5,
                "active_minus_round_robin_mean_gap": -2.0e-6,
                "active_minus_round_robin_max_gap": -2.0e-6,
            }
        )

        self.assertTrue(gate["pass"])
        self.assertAlmostEqual(gate["thresholds"]["active_minus_round_robin_mean_gap_max"], -1.2e-6, places=10)
        self.assertAlmostEqual(gate["thresholds"]["active_minus_round_robin_max_gap_max"], -1.5e-6, places=10)

    def test_autonomy_gate_preserves_legacy_cap_for_large_gaps(self) -> None:
        gate = autonomy_gate_from_comparison(
            {
                "round_robin_final_mean_gap": 0.36,
                "round_robin_final_max_gap": 0.39,
                "active_minus_round_robin_mean_gap": -0.011,
                "active_minus_round_robin_max_gap": -0.012,
            }
        )

        self.assertTrue(gate["pass"])
        self.assertEqual(gate["thresholds"]["active_minus_round_robin_mean_gap_max"], -0.01)
        self.assertEqual(gate["thresholds"]["active_minus_round_robin_max_gap_max"], -0.01)

    def test_acquisition_gate_scales_for_small_gap_regime(self) -> None:
        gate = acquisition_gate_from_comparison(
            {
                "round_robin_final_mean_candidate_gap": 4.0e-7,
                "round_robin_final_max_candidate_gap": 5.0e-7,
                "active_minus_round_robin_mean_candidate_gap": -2.0e-8,
                "active_minus_round_robin_max_candidate_gap": -2.0e-8,
            }
        )

        self.assertTrue(gate["pass"])
        self.assertAlmostEqual(gate["thresholds"]["active_minus_round_robin_mean_candidate_gap_max"], -1.2e-8, places=12)
        self.assertAlmostEqual(gate["thresholds"]["active_minus_round_robin_max_candidate_gap_max"], -1.5e-8, places=12)

    def test_probe_diagnostics_clamps_tiny_negative_reconstruction_error(self) -> None:
        routing_key = torch.tensor([1.0, 1.0], dtype=torch.float32)
        normalized = torch.nn.functional.normalize(routing_key, dim=0)
        trainer = SimpleNamespace(
            config=SimpleNamespace(n_columns=1),
            model=SimpleNamespace(
                device=torch.device("cpu"),
                competitive=SimpleNamespace(
                    prototypes=torch.stack([normalized * 1.000001]),
                ),
            ),
            routing_key_for_pattern=lambda pattern: pattern,
        )

        diagnostics = probe_diagnostics(trainer, [normalized])

        self.assertEqual(diagnostics["recon_error"], 0.0)

    def test_select_active_source_keeps_clear_leader_despite_visit_penalty(self) -> None:
        available = [
            make_bank("leader", visits=5),
            make_bank("other", visits=1),
            make_bank("third", visits=1),
        ]
        snapshot = {
            "leader": make_gap(0.45, 0.60),
            "other": make_gap(0.44, 0.55),
            "third": make_gap(0.43, 0.52),
        }

        selected, scores = select_active_source(
            available,
            snapshot,
            coverage_balance_penalty=0.05,
            gap_focus_margin=0.02,
        )

        self.assertEqual(selected.name, "leader")
        self.assertEqual(scores["leader"], 0.60)

    def test_select_active_source_balances_near_ties(self) -> None:
        available = [
            make_bank("overused", visits=4),
            make_bank("fresh", visits=1),
        ]
        snapshot = {
            "overused": make_gap(0.45, 0.51),
            "fresh": make_gap(0.44, 0.505),
        }

        selected, _ = select_active_source(
            available,
            snapshot,
            coverage_balance_penalty=0.02,
            gap_focus_margin=0.02,
        )

        self.assertEqual(selected.name, "fresh")

    def test_select_active_source_uses_diagnostic_gap_score_not_info_gain(self) -> None:
        # "better" has a clearly higher diagnostic_gap_score but a *lower* info_gain_score.
        # "worse" has a higher info_gain_score but a lower diagnostic_gap_score.
        # selection_metric() must use diagnostic_gap_score, so "better" must win.
        available = [
            make_bank("better", visits=0),
            make_bank("worse", visits=0),
        ]
        snapshot = {
            "better": make_gap(0.45, 0.70, info_gain_score=0.50),
            "worse":  make_gap(0.44, 0.55, info_gain_score=0.90),
        }

        selected, scores = select_active_source(
            available,
            snapshot,
            coverage_balance_penalty=0.0,
            gap_focus_margin=0.02,
        )

        self.assertEqual(selected.name, "better")
        # scores dict contains diagnostic_gap_score values, not info_gain_score values
        self.assertEqual(scores["better"], 0.70)
        self.assertEqual(scores["worse"], 0.55)

    def test_preview_source_chunk_does_not_advance_candidate_state(self) -> None:
        bank = make_chunk_bank("candidate")

        preview = preview_source_chunk(bank, 2)

        self.assertEqual([raw for raw, _ in preview], ["candidate-a", "candidate-b"])
        self.assertEqual(bank.cursor, 0)
        self.assertEqual(bank.visits, 0)

    def test_preview_source_chunk_offset_keeps_state_and_reads_later_windows(self) -> None:
        bank = make_chunk_bank("candidate")

        preview = preview_source_chunk(bank, 1, offset_tokens=1)

        self.assertEqual([raw for raw, _ in preview], ["candidate-b"])
        self.assertEqual(bank.cursor, 0)
        self.assertEqual(bank.visits, 0)

    def test_consume_previewed_chunk_only_commits_selected_bank(self) -> None:
        selected = make_chunk_bank("selected")
        rejected = make_chunk_bank("rejected")

        selected_preview = preview_source_chunk(selected, 2)
        _ = preview_source_chunk(rejected, 2)
        committed = consume_previewed_chunk(selected, selected_preview)

        self.assertEqual([raw for raw, _ in committed], ["selected-a", "selected-b"])
        self.assertEqual(selected.cursor, 2)
        self.assertEqual(selected.visits, 1)
        self.assertEqual(rejected.cursor, 0)
        self.assertEqual(rejected.visits, 0)

    def test_semantic_shortlist_disabled_uses_active_ranking(self) -> None:
        available = [
            make_bank("yelp", visits=0),
            make_bank("dbpedia", visits=0),
            make_bank("reviews", visits=0),
        ]
        snapshot = {
            "yelp": make_gap(0.44, 0.44),
            "dbpedia": make_gap(0.43, 0.43),
            "reviews": make_gap(0.45, 0.49),
        }

        ranked, scores = semantic_shortlist(
            trainer=None,
            available=available,
            gap_snapshot=snapshot,
            shortlist_size=0,
            gap_weight=0.5,
            affinity_weight=0.5,
            coverage_balance_penalty=0.02,
            gap_focus_margin=0.02,
        )

        self.assertEqual([bank.name for bank in ranked], ["reviews", "yelp", "dbpedia"])
        self.assertEqual(scores["reviews"], 0.49)

    def test_semantic_shortlist_disabled_keeps_balance_for_near_ties(self) -> None:
        available = [
            make_bank("overused", visits=4),
            make_bank("fresh", visits=1),
        ]
        snapshot = {
            "overused": make_gap(0.45, 0.51),
            "fresh": make_gap(0.44, 0.505),
        }

        ranked, _ = semantic_shortlist(
            trainer=None,
            available=available,
            gap_snapshot=snapshot,
            shortlist_size=0,
            gap_weight=0.5,
            affinity_weight=0.5,
            coverage_balance_penalty=0.02,
            gap_focus_margin=0.02,
        )

        self.assertEqual(ranked[0].name, "fresh")

    def test_semantic_shortlist_can_use_frontier_terms_for_semantic_relevance(self) -> None:
        available = [
            make_window_bank("finance", ["credit loan deposit bank", "interest credit score"]),
            make_window_bank("river", ["river shore current water", "mud reeds stream"]),
        ]
        snapshot = {
            "finance": make_gap(0.44, 0.50),
            "river": make_gap(0.45, 0.52),
        }
        fake_trainer = object()

        with patch(
            "hecsn.training.autonomy_acquisition_runner.frontier_semantic_plan",
            return_value={
                "gap_terms": [{"term": "credit", "weight": 2.0}, {"term": "loan", "weight": 1.0}],
                "unsupported_terms": ["credit", "loan"],
                "retrieval_queries": ["credit loan"],
                "follow_up_questions": ["What grounded evidence links credit and loan in current frontier memory?"],
            },
        ), patch(
            "hecsn.training.autonomy_acquisition_runner.current_context_signature",
            return_value=None,
        ), patch(
            "hecsn.training.autonomy_acquisition_runner.candidate_semantic_signature",
            return_value=None,
        ):
            ranked, scores = semantic_shortlist(
                trainer=fake_trainer,
                available=available,
                gap_snapshot=snapshot,
                shortlist_size=1,
                gap_weight=0.2,
                affinity_weight=0.8,
                coverage_balance_penalty=0.02,
                gap_focus_margin=0.02,
            )

        self.assertEqual(ranked[0].name, "finance")
        self.assertGreater(scores["finance"], scores["river"])

    def test_select_projected_commit_target_prefers_frontier_reduction_over_current_gain(self) -> None:
        candidates = [
            make_bank("dbpedia", visits=0),
            make_bank("reviews", visits=0),
        ]
        snapshot = {
            "dbpedia": make_gap(0.30, 0.51, info_gain_score=0.91),
            "reviews": make_gap(0.28, 0.48, info_gain_score=0.74),
        }
        projected_rows = [
            {
                "source": "dbpedia",
                "gap_reduction": -0.03,
                "diagnostic_gap_reduction": -0.04,
                "projected_final_mean_candidate_gap": 0.33,
                "projected_final_max_candidate_gap": 0.36,
                "projected_final_mean_candidate_diagnostic_gap": 0.42,
                "projected_final_max_candidate_diagnostic_gap": 0.45,
            },
            {
                "source": "reviews",
                "gap_reduction": 0.02,
                "diagnostic_gap_reduction": 0.03,
                "projected_final_mean_candidate_gap": 0.24,
                "projected_final_max_candidate_gap": 0.28,
                "projected_final_mean_candidate_diagnostic_gap": 0.31,
                "projected_final_max_candidate_diagnostic_gap": 0.35,
            },
        ]

        selected, scores = select_projected_commit_target(
            candidates=candidates,
            projected_rows=projected_rows,
            snapshot=snapshot,
            coverage_balance_penalty=0.02,
            gap_focus_margin=0.02,
        )

        self.assertEqual(selected.name, "reviews")
        self.assertGreater(scores["reviews"], scores["dbpedia"])

    def test_evaluate_projected_candidates_restores_rng_for_each_trial(self) -> None:
        candidates = [
            make_chunk_bank("dbpedia"),
            make_chunk_bank("reviews"),
        ]
        preview_chunks = {
            bank.name: preview_source_chunk(bank, 1)
            for bank in candidates
        }
        snapshot = {
            bank.name: make_gap(0.30, 0.40)
            for bank in candidates
        }
        expected_projected_value = random.Random(1234).random()
        random.seed(1234)

        with patch("hecsn.training.autonomy_acquisition_runner.project_candidate_frontier") as mocked_projection:
            def fake_projection(*args, **kwargs):
                return {
                    "source": args[1].name,
                    "projected_final_mean_candidate_gap": random.random(),
                }

            mocked_projection.side_effect = fake_projection
            projected_rows = evaluate_projected_candidates(
                trainer=None,
                available=candidates,
                preview_chunks=preview_chunks,
                projection_candidates=candidates,
                gap_exploration_bonus=0.0,
                gap_ambiguity_weight=0.0,
                gap_switch_weight=0.0,
                gap_margin_reference=0.0,
                snapshot=snapshot,
            )

        self.assertEqual(len(projected_rows), 2)
        self.assertAlmostEqual(projected_rows[0]["projected_final_mean_candidate_gap"], expected_projected_value)
        self.assertAlmostEqual(projected_rows[1]["projected_final_mean_candidate_gap"], expected_projected_value)
        self.assertAlmostEqual(random.random(), expected_projected_value)

    def test_select_scout_commit_target_prefers_lower_projected_frontier(self) -> None:
        shortlist = [
            make_bank("reviews", visits=1),
            make_bank("yelp", visits=1),
            make_bank("dbpedia", visits=0),
        ]
        snapshot = {
            "reviews": make_gap(0.31, 0.379),
            "yelp": make_gap(0.33, 0.397),
            "dbpedia": make_gap(0.29, 0.371),
        }
        scout_rows = [
            {
                "source": "reviews",
                "diagnostic_gap_reduction": 0.069,
                "projected_final_mean_candidate_gap": 0.312,
                "projected_final_max_candidate_gap": 0.344,
            },
            {
                "source": "yelp",
                "diagnostic_gap_reduction": -0.029,
                "projected_final_mean_candidate_gap": 0.281,
                "projected_final_max_candidate_gap": 0.331,
            },
        ]

        selected, scores = select_scout_commit_target(
            shortlist=shortlist,
            scout_rows=scout_rows,
            snapshot=snapshot,
            coverage_balance_penalty=0.02,
            gap_focus_margin=0.02,
            commit_tokens=1500,
            scout_commit_tokens=250,
        )

        self.assertEqual(selected.name, "yelp")
        self.assertNotIn("dbpedia", scores)

    def test_select_scout_commit_target_breaks_projection_ties_with_max_gap(self) -> None:
        shortlist = [
            make_bank("yelp", visits=0),
            make_bank("dbpedia", visits=1),
            make_bank("reviews", visits=1),
        ]
        snapshot = {
            "yelp": make_gap(0.37, 0.464),
            "dbpedia": make_gap(0.34, 0.433),
            "reviews": make_gap(0.33, 0.415),
        }
        scout_rows = [
            {
                "source": "dbpedia",
                "diagnostic_gap_reduction": -0.007,
                "projected_final_mean_candidate_gap": 0.301,
                "projected_final_max_candidate_gap": 0.391,
            },
            {
                "source": "reviews",
                "diagnostic_gap_reduction": 0.025,
                "projected_final_mean_candidate_gap": 0.301,
                "projected_final_max_candidate_gap": 0.348,
            },
        ]

        selected, scores = select_scout_commit_target(
            shortlist=shortlist,
            scout_rows=scout_rows,
            snapshot=snapshot,
            coverage_balance_penalty=0.02,
            gap_focus_margin=0.02,
            commit_tokens=1500,
            scout_commit_tokens=250,
        )

        self.assertEqual(selected.name, "reviews")
        self.assertNotIn("yelp", scores)

    def test_execute_acquisition_policy_active_uses_semantic_shortlist_when_enabled(self) -> None:
        trainer = SimpleNamespace(
            token_count=0,
            model=SimpleNamespace(runtime_scope_report=lambda: {"mode": "test"}),
        )
        candidates = [make_chunk_bank("alpha"), make_chunk_bank("beta")]
        candidates[0].metadata = {"semantic_relevance": 0.1, "query_text": "alpha"}
        candidates[1].metadata = {"semantic_relevance": 0.9, "query_text": "beta"}
        snapshot = {
            "alpha": make_gap(0.30, 0.40),
            "beta": make_gap(0.31, 0.41),
        }
        selected_candidates: list[str] = []

        def fake_select_projected_active_source(_trainer, available, *_args, **_kwargs):
            selected_candidates[:] = [bank.name for bank in available]
            return available[0], {available[0].name: 1.0}, [], {}

        with patch(
            "hecsn.training.autonomy_acquisition_runner.candidate_gap_snapshot",
            return_value=snapshot,
        ), patch(
            "hecsn.training.autonomy_acquisition_runner.semantic_shortlist",
            return_value=([candidates[1]], {"alpha": 0.1, "beta": 0.9}),
        ), patch(
            "hecsn.training.autonomy_acquisition_runner.select_projected_active_source",
            side_effect=fake_select_projected_active_source,
        ), patch(
            "hecsn.training.autonomy_acquisition_runner.train_source_chunk",
            return_value=1,
        ), patch(
            "hecsn.training.autonomy_acquisition_runner.probe_gap",
            return_value=make_gap(0.20, 0.25),
        ):
            result = execute_acquisition_policy(
                trainer=trainer,
                candidate_state=candidates,
                candidate_bank_specs=[],
                encoder=object(),
                candidate_train_tokens=32,
                probe_tokens=8,
                metrics_rows=[],
                policy_name="active",
                acquisition_tokens=1,
                acquisition_slots=1,
                gap_exploration_bonus=0.0,
                gap_ambiguity_weight=0.0,
                gap_switch_weight=0.0,
                gap_margin_reference=0.0,
                coverage_balance_penalty=0.0,
                gap_focus_margin=0.0,
                acquisition_phase="acquisition",
                semantic_shortlist_size=1,
                semantic_shortlist_gap_weight=0.35,
                semantic_shortlist_affinity_weight=0.65,
            )

        self.assertEqual(selected_candidates, ["beta"])
        self.assertEqual(result["acquired_sources"], ["beta"])

    def test_execute_acquisition_policy_active_skips_semantic_shortlist_without_semantic_signal(self) -> None:
        trainer = SimpleNamespace(
            token_count=0,
            model=SimpleNamespace(runtime_scope_report=lambda: {"mode": "test"}),
        )
        candidates = [make_chunk_bank("alpha"), make_chunk_bank("beta")]
        candidates[0].metadata = {"semantic_relevance": 0.0, "query_text": "None"}
        candidates[1].metadata = {"semantic_relevance": 0.0, "query_text": "None"}
        snapshot = {
            "alpha": make_gap(0.30, 0.40),
            "beta": make_gap(0.31, 0.41),
        }
        selected_candidates: list[str] = []

        def fake_select_projected_active_source(_trainer, available, *_args, **_kwargs):
            selected_candidates[:] = [bank.name for bank in available]
            return available[0], {available[0].name: 1.0}, [], {}

        with patch(
            "hecsn.training.autonomy_acquisition_runner.candidate_gap_snapshot",
            return_value=snapshot,
        ), patch(
            "hecsn.training.autonomy_acquisition_runner.semantic_shortlist",
        ) as shortlist_mock, patch(
            "hecsn.training.autonomy_acquisition_runner.select_projected_active_source",
            side_effect=fake_select_projected_active_source,
        ), patch(
            "hecsn.training.autonomy_acquisition_runner.train_source_chunk",
            return_value=1,
        ), patch(
            "hecsn.training.autonomy_acquisition_runner.probe_gap",
            return_value=make_gap(0.20, 0.25),
        ):
            result = execute_acquisition_policy(
                trainer=trainer,
                candidate_state=candidates,
                candidate_bank_specs=[],
                encoder=object(),
                candidate_train_tokens=32,
                probe_tokens=8,
                metrics_rows=[],
                policy_name="active",
                acquisition_tokens=1,
                acquisition_slots=1,
                gap_exploration_bonus=0.0,
                gap_ambiguity_weight=0.0,
                gap_switch_weight=0.0,
                gap_margin_reference=0.0,
                coverage_balance_penalty=0.0,
                gap_focus_margin=0.0,
                acquisition_phase="acquisition",
                semantic_shortlist_size=1,
                semantic_shortlist_gap_weight=0.35,
                semantic_shortlist_affinity_weight=0.65,
            )

        shortlist_mock.assert_not_called()
        self.assertEqual(selected_candidates, ["alpha", "beta"])
        self.assertEqual(result["acquired_sources"], ["alpha"])

    def test_run_live_acquisition_uses_frontier_plan_for_catalog_discovery(self) -> None:
        fake_plan = {
            "gap_terms": [{"term": "plasticity", "weight": 2.0}],
            "unsupported_terms": ["plasticity"],
            "retrieval_queries": ["synaptic plasticity memory"],
            "follow_up_questions": ["What grounded evidence would stabilize plasticity in memory?"],
        }
        trainer = SimpleNamespace(config=SimpleNamespace(window_size=10))

        with patch(
            "hecsn.training.autonomy_acquisition_runner.frontier_semantic_plan",
            return_value=fake_plan,
        ), patch(
            "hecsn.training.autonomy_acquisition_runner.load_source_banks",
            return_value=[make_bank("candidate", visits=0)],
        ) as mocked_load, patch(
            "hecsn.training.autonomy_acquisition_runner.execute_acquisition_policy",
            return_value={"policy": "active"},
        ):
            result = run_live_acquisition(
                trainer=trainer,
                encoder=object(),
                candidate_bank_specs=[
                    {
                        "catalog_mode": "semantic_registry",
                        "catalog_entries": [
                            {
                                "name": "candidate",
                                "source": "https://example.com/candidate",
                                "source_type": "web",
                                "summary": "synaptic plasticity and consolidation",
                            }
                        ],
                    }
                ],
                candidate_train_tokens=32,
                probe_tokens=8,
                acquisition_tokens=16,
                acquisition_slots=1,
                gap_exploration_bonus=0.0,
                gap_ambiguity_weight=0.0,
                gap_switch_weight=0.0,
                gap_margin_reference=0.0,
                coverage_balance_penalty=0.0,
                gap_focus_margin=0.0,
            )

        self.assertEqual(mocked_load.call_args.kwargs["semantic_plan"], fake_plan)
        self.assertEqual(result["candidate_discovery_plan"], fake_plan)

    def test_execute_acquisition_policy_refreshes_dynamic_catalog_per_slot(self) -> None:
        trainer = SimpleNamespace(
            token_count=0,
            model=SimpleNamespace(runtime_scope_report=lambda: {"mode": "test"}),
        )
        exclusion_history: list[list[str]] = []

        def fake_refresh(**kwargs):
            excluded = list(kwargs.get("excluded_names") or [])
            exclusion_history.append(excluded)
            if excluded == []:
                return [make_chunk_bank("alpha"), make_chunk_bank("beta")], {"retrieval_queries": ["alpha beta"]}
            if excluded == ["alpha"]:
                return [make_chunk_bank("beta")], {"retrieval_queries": ["beta"]}
            return [], {"retrieval_queries": []}

        def fake_snapshot(_trainer, available, *_args, **_kwargs):
            return {
                bank.name: make_gap(0.30 + 0.01 * idx, 0.40 + 0.01 * idx)
                for idx, bank in enumerate(available)
            }

        with patch(
            "hecsn.training.autonomy_acquisition_runner.refresh_candidate_catalog_state",
            side_effect=fake_refresh,
        ), patch(
            "hecsn.training.autonomy_acquisition_runner.candidate_gap_snapshot",
            side_effect=fake_snapshot,
        ), patch(
            "hecsn.training.autonomy_acquisition_runner.train_source_chunk",
            return_value=1,
        ), patch(
            "hecsn.training.autonomy_acquisition_runner.probe_gap",
            return_value=make_gap(0.20, 0.25),
        ):
            result = execute_acquisition_policy(
                trainer=trainer,
                candidate_state=[],
                candidate_bank_specs=[{"catalog_mode": "semantic_registry", "catalog_entries": []}],
                encoder=object(),
                candidate_train_tokens=32,
                probe_tokens=8,
                metrics_rows=[],
                policy_name="round_robin",
                acquisition_tokens=1,
                acquisition_slots=2,
                gap_exploration_bonus=0.0,
                gap_ambiguity_weight=0.0,
                gap_switch_weight=0.0,
                gap_margin_reference=0.0,
                coverage_balance_penalty=0.0,
                gap_focus_margin=0.0,
                acquisition_phase="acquisition",
            )

        self.assertEqual(result["acquired_sources"], ["alpha", "beta"])
        self.assertEqual(exclusion_history, [[], ["alpha"], ["alpha", "beta"]])
        self.assertEqual(len(result["candidate_discovery_history"]), 2)


if __name__ == "__main__":
    unittest.main()
