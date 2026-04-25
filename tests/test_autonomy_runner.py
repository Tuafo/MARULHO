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
    refresh_candidate_catalog_state,
    run_live_acquisition,
    semantic_shortlist,
    select_projected_commit_target,
    select_scout_commit_target,
    train_source_chunk,
)
from hecsn.training.autonomy_runner import (
    ProbeGapMetrics,
    SourceBank,
    _catalog_metadata_prefix_text,
    autonomy_gate_from_comparison,
    load_source_banks,
    probe_diagnostics,
    select_active_source,
    update_source_feedback,
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

    def test_select_active_source_uses_empirical_feedback_to_escape_stale_leader(self) -> None:
        available = [
            make_bank("stale", visits=4),
            make_bank("fresh", visits=1),
        ]
        snapshot = {
            "stale": make_gap(0.45, 0.41),
            "fresh": make_gap(0.44, 0.39),
        }
        feedback_state: dict[str, dict[str, float]] = {}
        update_source_feedback(
            feedback_state,
            source_name="stale",
            gap_before=0.40,
            gap_after=0.395,
            gap_score_before=0.41,
            gap_score_after=0.405,
        )

        selected, scores = select_active_source(
            available,
            snapshot,
            coverage_balance_penalty=0.02,
            gap_focus_margin=0.02,
            feedback_state=feedback_state,
        )

        self.assertEqual(selected.name, "fresh")
        self.assertGreater(scores["fresh"], scores["stale"])

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

    def test_semantic_shortlist_can_use_external_semantic_plan_for_semantic_relevance(self) -> None:
        available = [
            make_window_bank("submarine", ["submarine buoyancy ballast pressure", "depth control ballast tank"]),
            make_window_bank("garden", ["garden tomato soil sunlight", "rain compost seedlings"]),
        ]
        snapshot = {
            "submarine": make_gap(0.44, 0.50),
            "garden": make_gap(0.45, 0.52),
        }
        external_plan = {
            "planner_mode": "recent_query_gap_focus",
            "gap_terms": [{"term": "submarine", "weight": 2.0}, {"term": "ballast", "weight": 1.0}],
            "unsupported_terms": ["submarine", "ballast"],
            "retrieval_queries": ["submarine ballast buoyancy"],
            "follow_up_questions": ["What grounded evidence explains submarine ballast control?"],
        }
        fake_trainer = object()

        with patch(
            "hecsn.training.autonomy_acquisition_runner.frontier_semantic_plan",
            return_value={},
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
                semantic_plan=external_plan,
            )

        self.assertEqual(ranked[0].name, "submarine")
        self.assertGreater(scores["submarine"], scores["garden"])

    def test_semantic_shortlist_keeps_external_focus_ahead_of_frontier_noise(self) -> None:
        available = [
            make_window_bank("submarine", ["submarine buoyancy ballast pressure", "depth control ballast tank"]),
            make_window_bank("garden", ["garden tomato soil sunlight", "rain compost seedlings"]),
        ]
        snapshot = {
            "submarine": make_gap(0.44, 0.50),
            "garden": make_gap(0.45, 0.52),
        }
        external_plan = {
            "planner_mode": "recent_query_gap_focus",
            "gap_terms": [{"term": "submarine", "weight": 2.0}, {"term": "ballast", "weight": 1.0}],
            "unsupported_terms": ["submarine", "ballast"],
            "retrieval_queries": ["submarine ballast buoyancy"],
            "follow_up_questions": ["What grounded evidence explains submarine ballast control?"],
        }
        noisy_frontier_plan = {
            "planner_mode": "frontier_semantic_plan",
            "gap_terms": [{"term": "garden", "weight": 3.0}, {"term": "tomato", "weight": 2.0}],
            "unsupported_terms": ["garden", "tomato"],
            "retrieval_queries": ["garden tomato soil"],
            "follow_up_questions": ["What stable evidence is still missing for garden tomato soil?"],
        }
        fake_trainer = object()

        with patch(
            "hecsn.training.autonomy_acquisition_runner.frontier_semantic_plan",
            return_value=noisy_frontier_plan,
        ), patch(
            "hecsn.training.autonomy_acquisition_runner.current_context_signature",
            return_value=torch.tensor([1.0, 0.0], dtype=torch.float32),
        ), patch(
            "hecsn.training.autonomy_acquisition_runner.candidate_semantic_signature",
            side_effect=lambda _trainer, bank: torch.tensor(
                [1.0, 0.0] if bank.name == "garden" else [0.0, 1.0],
                dtype=torch.float32,
            ),
        ):
            ranked, scores = semantic_shortlist(
                trainer=fake_trainer,
                available=available,
                gap_snapshot=snapshot,
                shortlist_size=1,
                gap_weight=0.0,
                affinity_weight=1.0,
                coverage_balance_penalty=0.02,
                gap_focus_margin=0.02,
                semantic_plan=external_plan,
            )

        self.assertEqual(ranked[0].name, "submarine")
        self.assertGreater(scores["submarine"], scores["garden"])

    def test_semantic_shortlist_treats_follow_up_only_plan_as_explicit_focus(self) -> None:
        available = [
            make_window_bank("submarine", ["submarine buoyancy ballast pressure", "depth control ballast tank"]),
            make_window_bank("garden", ["garden tomato soil sunlight", "rain compost seedlings"]),
        ]
        snapshot = {
            "submarine": make_gap(0.44, 0.50),
            "garden": make_gap(0.45, 0.52),
        }
        external_plan = {
            "planner_mode": "recent_query_gap_focus",
            "follow_up_questions": ["What grounded evidence explains submarine ballast control?"],
        }
        noisy_frontier_plan = {
            "planner_mode": "frontier_semantic_plan",
            "gap_terms": [{"term": "garden", "weight": 3.0}, {"term": "tomato", "weight": 2.0}],
            "unsupported_terms": ["garden", "tomato"],
            "retrieval_queries": ["garden tomato soil"],
            "follow_up_questions": ["What stable evidence is still missing for garden tomato soil?"],
        }
        fake_trainer = object()

        with patch(
            "hecsn.training.autonomy_acquisition_runner.frontier_semantic_plan",
            return_value=noisy_frontier_plan,
        ), patch(
            "hecsn.training.autonomy_acquisition_runner.current_context_signature",
            return_value=torch.tensor([1.0, 0.0], dtype=torch.float32),
        ), patch(
            "hecsn.training.autonomy_acquisition_runner.candidate_semantic_signature",
            side_effect=lambda _trainer, bank: torch.tensor(
                [1.0, 0.0] if bank.name == "garden" else [0.0, 1.0],
                dtype=torch.float32,
            ),
        ):
            ranked, scores = semantic_shortlist(
                trainer=fake_trainer,
                available=available,
                gap_snapshot=snapshot,
                shortlist_size=1,
                gap_weight=0.0,
                affinity_weight=1.0,
                coverage_balance_penalty=0.02,
                gap_focus_margin=0.02,
                semantic_plan=external_plan,
            )

        self.assertEqual(ranked[0].name, "submarine")
        self.assertGreater(scores["submarine"], scores["garden"])

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

    def test_select_projected_commit_target_prefers_semantically_aligned_candidate_on_near_tie(self) -> None:
        candidates = [
            make_bank("submarine", visits=0),
            make_bank("garden", visits=0),
        ]
        snapshot = {
            "submarine": {
                **make_gap(0.10463128332048655, 0.5231153989707431),
                "frontier_semantic_relevance": 0.35886147778190935,
                "semantic_action_score": 0.8934959494808362,
            },
            "garden": {
                **make_gap(0.10462089255452156, 0.5229352653523287),
                "frontier_semantic_relevance": 0.004831481588629439,
                "semantic_action_score": 0.8414926786415609,
            },
        }
        projected_rows = [
            {
                "source": "submarine",
                "gap_reduction": 0.04371026158332825,
                "diagnostic_gap_reduction": 0.05447149305000254,
                "projected_final_mean_candidate_gap": 0.06093825865536928,
                "projected_final_max_candidate_gap": 0.060954395681619644,
                "projected_final_mean_candidate_diagnostic_gap": 0.4753205345423164,
                "projected_final_max_candidate_diagnostic_gap": 0.47758205823600297,
            },
            {
                "source": "garden",
                "gap_reduction": 0.044204539619386196,
                "diagnostic_gap_reduction": 0.05064562332833156,
                "projected_final_mean_candidate_gap": 0.060408932622522116,
                "projected_final_max_candidate_gap": 0.06043654680252075,
                "projected_final_mean_candidate_diagnostic_gap": 0.4788445137231193,
                "projected_final_max_candidate_diagnostic_gap": 0.48112050225337355,
            },
        ]

        selected, _scores = select_projected_commit_target(
            candidates=candidates,
            projected_rows=projected_rows,
            snapshot=snapshot,
            coverage_balance_penalty=0.02,
            gap_focus_margin=0.02,
        )

        self.assertEqual(selected.name, "submarine")

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

    def test_execute_acquisition_policy_active_stops_when_best_projected_frontier_is_non_positive(self) -> None:
        trainer = SimpleNamespace(
            token_count=0,
            model=SimpleNamespace(runtime_scope_report=lambda: {"mode": "test"}),
        )
        candidates = [make_chunk_bank("alpha"), make_chunk_bank("beta")]
        snapshot = {
            "alpha": make_gap(0.30, 0.40),
            "beta": make_gap(0.32, 0.42),
        }
        selection_call_count = 0

        def fake_select_projected_active_source(_trainer, available, *_args, **_kwargs):
            nonlocal selection_call_count
            selection_call_count += 1
            if selection_call_count == 1:
                return (
                    available[0],
                    {available[0].name: 0.05},
                    [
                        {
                            "source": available[0].name,
                            "projected_final_mean_candidate_gap": 0.25,
                            "projected_final_max_candidate_gap": 0.30,
                            "projected_final_mean_candidate_diagnostic_gap": 0.31,
                            "projected_final_max_candidate_diagnostic_gap": 0.35,
                        }
                    ],
                    {},
                )
            return (
                available[1],
                {available[1].name: -0.01},
                [
                    {
                        "source": available[1].name,
                        "projected_final_mean_candidate_gap": 0.31,
                        "projected_final_max_candidate_gap": 0.33,
                        "projected_final_mean_candidate_diagnostic_gap": 0.41,
                        "projected_final_max_candidate_diagnostic_gap": 0.43,
                    }
                ],
                {},
            )

        with patch(
            "hecsn.training.autonomy_acquisition_runner.candidate_gap_snapshot",
            return_value=snapshot,
        ), patch(
            "hecsn.training.autonomy_acquisition_runner.select_projected_active_source",
            side_effect=fake_select_projected_active_source,
        ), patch(
            "hecsn.training.autonomy_acquisition_runner.train_source_chunk",
            return_value=1,
        ) as train_mock, patch(
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
                acquisition_slots=2,
                gap_exploration_bonus=0.0,
                gap_ambiguity_weight=0.0,
                gap_switch_weight=0.0,
                gap_margin_reference=0.0,
                coverage_balance_penalty=0.0,
                gap_focus_margin=0.0,
                acquisition_phase="acquisition",
            )

        self.assertEqual(selection_call_count, 2)
        self.assertEqual(train_mock.call_count, 1)
        self.assertEqual(result["acquired_sources"], ["alpha"])
        self.assertEqual(len(result["acquisition_history"]), 1)
        self.assertTrue(result["stopped_early"])
        self.assertEqual(len(result["stop_decisions"]), 1)
        self.assertEqual(result["stop_decisions"][0]["selected_source"], "beta")
        self.assertEqual(result["stop_decisions"][0]["reason"], "best_projected_frontier_non_positive")

    def test_execute_acquisition_policy_active_commits_semantically_focused_candidate_despite_small_negative_projection(self) -> None:
        trainer = SimpleNamespace(
            token_count=0,
            model=SimpleNamespace(runtime_scope_report=lambda: {"mode": "test"}),
        )
        candidates = [make_chunk_bank("submarine"), make_chunk_bank("garden")]
        snapshot = {
            "submarine": {
                **make_gap(0.30, 0.42),
                "frontier_semantic_relevance": 0.85,
                "semantic_action_score": 0.44,
            },
            "garden": {
                **make_gap(0.32, 0.39),
                "frontier_semantic_relevance": 0.05,
                "semantic_action_score": 0.39,
            },
        }

        def fake_select_projected_active_source(_trainer, available, *_args, **_kwargs):
            return (
                available[0],
                {available[0].name: -0.008},
                [
                    {
                        "source": available[0].name,
                        "projected_final_mean_candidate_gap": 0.317,
                        "projected_final_max_candidate_gap": 0.328,
                        "projected_final_mean_candidate_diagnostic_gap": 0.43,
                        "projected_final_max_candidate_diagnostic_gap": 0.45,
                    }
                ],
                {available[0].name: [("submarine buoyancy", torch.tensor([1.0, 0.0]))]},
            )

        with patch(
            "hecsn.training.autonomy_acquisition_runner.candidate_gap_snapshot",
            return_value=snapshot,
        ), patch(
            "hecsn.training.autonomy_acquisition_runner.select_projected_active_source",
            side_effect=fake_select_projected_active_source,
        ), patch(
            "hecsn.training.autonomy_acquisition_runner.consume_previewed_chunk",
            return_value=[("submarine buoyancy", torch.tensor([1.0, 0.0]))],
        ), patch(
            "hecsn.training.autonomy_acquisition_runner.replay_source_chunk",
            return_value=1,
        ) as train_mock, patch(
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
            )

        self.assertEqual(train_mock.call_count, 1)
        self.assertEqual(result["acquired_sources"], ["submarine"])
        self.assertFalse(result["stopped_early"])
        self.assertEqual(result["stop_decisions"], [])

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

    def test_run_live_acquisition_keeps_external_explicit_focus_for_catalog_discovery(self) -> None:
        frontier_plan = {
            "planner_mode": "frontier_gap_planner",
            "gap_terms": [{"term": "plasticity", "weight": 2.0}],
            "unsupported_terms": ["plasticity"],
            "retrieval_queries": ["synaptic plasticity memory"],
            "follow_up_questions": ["What grounded evidence would stabilize plasticity in memory?"],
        }
        external_plan = {
            "planner_mode": "recent_query_gap_focus",
            "gap_terms": [{"term": "submarine", "weight": 3.0}],
            "unsupported_terms": ["submarine"],
            "retrieval_queries": ["submarine ballast buoyancy"],
            "follow_up_questions": ["What grounded evidence explains submarine ballast control?"],
        }
        trainer = SimpleNamespace(config=SimpleNamespace(window_size=10))

        with patch(
            "hecsn.training.autonomy_acquisition_runner.frontier_semantic_plan",
            return_value=frontier_plan,
        ), patch(
            "hecsn.training.autonomy_acquisition_runner.load_source_banks",
            return_value=[make_bank("candidate", visits=0)],
        ) as mocked_load, patch(
            "hecsn.training.autonomy_acquisition_runner.execute_acquisition_policy",
            return_value={"policy": "active"},
        ) as mocked_execute:
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
                                "summary": "submarine ballast and plasticity",
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
                semantic_plan=external_plan,
            )

        discovery_plan = mocked_load.call_args.kwargs["semantic_plan"]
        self.assertEqual(discovery_plan, external_plan)
        self.assertEqual(mocked_execute.call_args.kwargs["semantic_plan"], external_plan)
        self.assertEqual(result["candidate_discovery_plan"], external_plan)
        self.assertEqual(result["semantic_plan"], external_plan)

    def test_run_live_acquisition_keeps_external_weak_concept_focus_for_catalog_discovery(self) -> None:
        frontier_plan = {
            "planner_mode": "frontier_gap_planner",
            "gap_terms": [{"term": "plasticity", "weight": 2.0}],
            "unsupported_terms": ["plasticity"],
            "retrieval_queries": ["synaptic plasticity memory"],
            "follow_up_questions": ["What grounded evidence would stabilize plasticity in memory?"],
        }
        external_plan = {
            "planner_mode": "recent_query_gap_focus",
            "weak_concepts": [
                {
                    "label": "buoyancy control",
                    "weakness": 0.7,
                    "uncertainty": 0.6,
                    "drift": 0.2,
                    "top_terms": ["submarine", "ballast", "buoyancy"],
                    "match_count": 1,
                }
            ],
        }
        trainer = SimpleNamespace(config=SimpleNamespace(window_size=10))

        with patch(
            "hecsn.training.autonomy_acquisition_runner.frontier_semantic_plan",
            return_value=frontier_plan,
        ), patch(
            "hecsn.training.autonomy_acquisition_runner.load_source_banks",
            return_value=[make_bank("candidate", visits=0)],
        ) as mocked_load, patch(
            "hecsn.training.autonomy_acquisition_runner.execute_acquisition_policy",
            return_value={"policy": "active"},
        ) as mocked_execute:
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
                                "summary": "submarine ballast and buoyancy",
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
                semantic_plan=external_plan,
            )

        discovery_plan = mocked_load.call_args.kwargs["semantic_plan"]
        self.assertEqual(discovery_plan, external_plan)
        self.assertEqual(mocked_execute.call_args.kwargs["semantic_plan"], external_plan)
        self.assertEqual(result["candidate_discovery_plan"], external_plan)
        self.assertEqual(result["semantic_plan"], external_plan)

    def test_train_source_chunk_calls_runtime_step_callback(self) -> None:
        bank = make_chunk_bank("candidate")
        observed_metadata: list[dict[str, object] | None] = []

        def fake_train_step(
            pattern: torch.Tensor,
            raw_window: str | None = None,
            memory_metadata: dict[str, object] | None = None,
        ) -> dict[str, int]:
            observed_metadata.append(memory_metadata)
            return {"winner": int(torch.sum(pattern).item())}

        trainer = SimpleNamespace(train_step=fake_train_step)
        metrics_rows: list[dict[str, object]] = []
        observed_steps: list[tuple[str, int, str]] = []

        tokens = train_source_chunk(
            trainer,
            bank,
            2,
            "probe",
            metrics_rows,
            on_train_step=lambda raw_window, row: observed_steps.append(
                (raw_window, int(row["winner"]), str(row["phase"]))
            ),
        )

        self.assertEqual(tokens, 2)
        self.assertEqual(len(metrics_rows), 2)
        self.assertEqual(
            observed_steps,
            [
                ("candidate-a", 1, "probe"),
                ("candidate-b", 1, "probe"),
            ],
        )
        self.assertEqual(
            observed_metadata,
            [
                {
                    "observation_kind": "source",
                    "source_name": "candidate",
                    "source_type": "test",
                    "source": "candidate",
                },
                {
                    "observation_kind": "source",
                    "source_name": "candidate",
                    "source_type": "test",
                    "source": "candidate",
                },
            ],
        )

    def test_run_live_acquisition_passes_runtime_step_callback_into_execute_policy(self) -> None:
        fake_plan = {"unsupported_terms": ["plasticity"]}
        trainer = SimpleNamespace(config=SimpleNamespace(window_size=10))
        callback = object()

        with patch(
            "hecsn.training.autonomy_acquisition_runner.frontier_semantic_plan",
            return_value=fake_plan,
        ), patch(
            "hecsn.training.autonomy_acquisition_runner.load_source_banks",
            return_value=[make_bank("candidate", visits=0)],
        ), patch(
            "hecsn.training.autonomy_acquisition_runner.execute_acquisition_policy",
            return_value={"policy": "active"},
        ) as mocked_execute:
            run_live_acquisition(
                trainer=trainer,
                encoder=object(),
                candidate_bank_specs=[{"name": "candidate", "source": "candidate", "source_type": "test"}],
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
                on_train_step=callback,
            )

        self.assertIs(mocked_execute.call_args.kwargs["on_train_step"], callback)

    def test_load_source_banks_refreshes_loaded_catalog_semantic_relevance(self) -> None:
        plan = {
            "gap_terms": [{"term": "wall", "weight": 1.0}, {"term": "street", "weight": 1.0}],
            "unsupported_terms": ["wall", "street"],
            "retrieval_queries": ["wall street"],
            "follow_up_questions": ["What grounded evidence links wall and street in current frontier memory?"],
        }
        probe_patterns = [torch.tensor([1.0, 0.0])]
        train_patterns = [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])]

        with patch(
            "hecsn.training.autonomy_runner.load_probe_train_examples",
            return_value=(
                probe_patterns,
                ["Wall Street trading"],
                train_patterns,
                ["Wall Street trading", "market close"],
            ),
        ):
            banks = load_source_banks(
                [
                    {
                        "name": "catalog_candidate",
                        "source": "catalog_candidate",
                        "source_type": "hf",
                        "text_field": "text",
                        "metadata": {
                            "catalog_mode": "semantic_registry",
                            "semantic_relevance": 0.0,
                            "provider": "None",
                            "query_text": "None",
                        },
                    }
                ],
                encoder=object(),
                window_size=8,
                probe_tokens=1,
                source_train_tokens=2,
                semantic_plan=plan,
            )

        self.assertEqual(len(banks), 1)
        self.assertGreater(float(banks[0].metadata["semantic_relevance"]), 0.0)
        self.assertEqual(banks[0].metadata["provider"], "")
        self.assertEqual(banks[0].metadata["query_text"], "")

    def test_catalog_metadata_prefix_prioritizes_best_focus_fragments(self) -> None:
        prefix = _catalog_metadata_prefix_text(
            {
                "metadata": {
                    "catalog_title": "Ballast tank",
                    "catalog_summary": (
                        "A ballast tank is a compartment within a boat, ship or other floating structure that holds water, "
                        "which is used as ballast to provide hydrostatic stability for a vessel, to reduce or control buoyancy, "
                        "as in a submarine, and to correct trim."
                    ),
                    "query_text": "submarine buoyancy ballast",
                }
            }
        )

        self.assertIn("Ballast tank:", prefix)
        self.assertIn("Terms:", prefix)
        self.assertIn("ballast", prefix)
        self.assertIn("buoyancy", prefix)
        self.assertIn("submarine", prefix)
        self.assertIn("to reduce or control buoyancy, as in a submarine", prefix)
        self.assertIn("A ballast tank is a compartment within a boat", prefix)
        self.assertLess(prefix.index("Ballast tank:"), prefix.index("Terms:"))

    def test_catalog_metadata_prefix_uses_content_preview_when_available(self) -> None:
        prefix = _catalog_metadata_prefix_text(
            {
                "metadata": {
                    "catalog_title": "Ballast tank",
                    "catalog_summary": "Ballast tanks are compartments used for vessel stability and trim.",
                    "catalog_content_preview": "Ballast tanks reduce submarine buoyancy and support underwater trim control.",
                    "query_text": "submarine buoyancy ballast",
                }
            }
        )

        self.assertIn("Terms:", prefix)
        self.assertIn("reduce submarine buoyancy", prefix)
        self.assertLess(prefix.index("Ballast tanks reduce submarine buoyancy"), prefix.index("Terms:"))
        self.assertLess(
            prefix.index("Ballast tanks reduce submarine buoyancy"),
            prefix.index("Ballast tanks are compartments used for vessel stability and trim."),
        )

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

    def test_refresh_candidate_catalog_state_probe_first_scouts_pool_before_full_load(self) -> None:
        trainer = SimpleNamespace(config=SimpleNamespace(window_size=8))
        scout_alpha = make_chunk_bank("alpha")
        scout_alpha.metadata = {"semantic_relevance": 0.10, "combined_score": 0.15}
        scout_beta = make_chunk_bank("beta")
        scout_beta.metadata = {"semantic_relevance": 0.30, "combined_score": 0.35}
        scout_gamma = make_chunk_bank("gamma")
        scout_gamma.metadata = {"semantic_relevance": 0.20, "combined_score": 0.25}
        load_calls: list[dict[str, object]] = []
        fake_plan = {"retrieval_queries": ["probe-first scout"]}

        def fake_load(
            source_bank_specs,
            encoder,
            window_size,
            probe_tokens,
            source_train_tokens,
            *,
            semantic_plan=None,
            metadata_prefilter=False,
        ):
            load_calls.append(
                {
                    "source_bank_specs": source_bank_specs,
                    "probe_tokens": probe_tokens,
                    "source_train_tokens": source_train_tokens,
                    "semantic_plan": semantic_plan,
                    "metadata_prefilter": metadata_prefilter,
                }
            )
            if metadata_prefilter:
                return [scout_alpha, scout_beta, scout_gamma]
            return [make_chunk_bank(str(spec["name"])) for spec in source_bank_specs]

        with patch(
            "hecsn.training.autonomy_acquisition_runner.frontier_semantic_plan",
            return_value=fake_plan,
        ), patch(
            "hecsn.training.autonomy_acquisition_runner.load_source_banks",
            side_effect=fake_load,
        ), patch(
            "hecsn.training.autonomy_acquisition_runner.candidate_gap_snapshot",
            return_value={
                "alpha": {
                    **make_gap(0.31, 0.60),
                    "semantic_action_score": 0.95,
                    "frontier_semantic_relevance": 0.99,
                },
                "beta": {
                    **make_gap(0.29, 0.70),
                    "semantic_action_score": 0.20,
                    "frontier_semantic_relevance": 0.10,
                },
                "gamma": {
                    **make_gap(0.30, 0.60),
                    "semantic_action_score": 0.10,
                    "frontier_semantic_relevance": 0.40,
                },
            },
        ):
            candidates, plan = refresh_candidate_catalog_state(
                trainer=trainer,
                encoder=object(),
                candidate_bank_specs=[
                    {
                        "catalog_mode": "semantic_registry",
                        "catalog_limit": 2,
                        "catalog_probe_pool_limit": 3,
                        "catalog_probe_tokens": 3,
                        "catalog_scout_tokens": 5,
                        "catalog_entries": [],
                    }
                ],
                candidate_train_tokens=32,
                probe_tokens=8,
                gap_exploration_bonus=0.0,
                gap_ambiguity_weight=0.0,
                gap_switch_weight=0.0,
                gap_margin_reference=0.0,
            )

        self.assertEqual(plan, fake_plan)
        self.assertEqual([bank.name for bank in candidates], ["alpha", "beta"])
        self.assertEqual(load_calls[0]["probe_tokens"], 3)
        self.assertEqual(load_calls[0]["source_train_tokens"], 5)
        self.assertTrue(bool(load_calls[0]["metadata_prefilter"]))
        self.assertEqual(load_calls[0]["semantic_plan"], fake_plan)
        self.assertEqual(load_calls[1]["probe_tokens"], 8)
        self.assertEqual(load_calls[1]["source_train_tokens"], 32)
        self.assertFalse(bool(load_calls[1]["metadata_prefilter"]))
        finalist_specs = list(load_calls[1]["source_bank_specs"])
        self.assertEqual([str(spec["name"]) for spec in finalist_specs], ["alpha", "beta"])
        self.assertEqual(
            [float(spec["metadata"]["catalog_probe_selection_score"]) for spec in finalist_specs],
            [0.95, 0.20],
        )
        self.assertEqual(
            [float(spec["metadata"]["catalog_probe_diagnostic_gap_score"]) for spec in finalist_specs],
            [0.60, 0.70],
        )
        self.assertEqual(
            [float(spec["metadata"]["catalog_probe_frontier_semantic_relevance"]) for spec in finalist_specs],
            [0.99, 0.10],
        )
        self.assertTrue(all("catalog_mode" not in spec for spec in finalist_specs))

    def test_refresh_candidate_catalog_state_probe_first_finalists_shift_holdout_on_selection_score(self) -> None:
        trainer = SimpleNamespace(config=SimpleNamespace(window_size=8))
        scout_reviews = make_chunk_bank("reviews")
        scout_reviews.metadata = {"semantic_relevance": 0.30, "combined_score": 0.35}
        scout_dbpedia = make_chunk_bank("dbpedia")
        scout_dbpedia.metadata = {"semantic_relevance": 0.25, "combined_score": 0.30}
        scout_yelp = make_chunk_bank("yelp")
        scout_yelp.metadata = {"semantic_relevance": 0.20, "combined_score": 0.25}
        scout_amazon = make_chunk_bank("amazon")
        scout_amazon.metadata = {"semantic_relevance": 0.22, "combined_score": 0.27}
        load_calls: list[dict[str, object]] = []
        fake_plan = {"retrieval_queries": ["probe-first scout"]}

        def fake_load(
            source_bank_specs,
            encoder,
            window_size,
            probe_tokens,
            source_train_tokens,
            *,
            semantic_plan=None,
            metadata_prefilter=False,
        ):
            load_calls.append(
                {
                    "source_bank_specs": source_bank_specs,
                    "probe_tokens": probe_tokens,
                    "source_train_tokens": source_train_tokens,
                    "semantic_plan": semantic_plan,
                    "metadata_prefilter": metadata_prefilter,
                }
            )
            if metadata_prefilter:
                return [scout_reviews, scout_dbpedia, scout_yelp, scout_amazon]
            return [make_chunk_bank(str(spec["name"])) for spec in source_bank_specs]

        with patch(
            "hecsn.training.autonomy_acquisition_runner.frontier_semantic_plan",
            return_value=fake_plan,
        ), patch(
            "hecsn.training.autonomy_acquisition_runner.load_source_banks",
            side_effect=fake_load,
        ), patch(
            "hecsn.training.autonomy_acquisition_runner.candidate_gap_snapshot",
            return_value={
                "reviews": {
                    **make_gap(0.31, 0.720),
                    "semantic_action_score": 0.720,
                    "frontier_semantic_relevance": 0.20,
                },
                "dbpedia": {
                    **make_gap(0.30, 0.706),
                    "semantic_action_score": 0.706,
                    "frontier_semantic_relevance": 0.18,
                },
                "yelp": {
                    **make_gap(0.29, 0.705),
                    "semantic_action_score": 0.705,
                    "frontier_semantic_relevance": 0.17,
                },
                "amazon": {
                    **make_gap(0.28, 0.704),
                    "semantic_action_score": 0.711,
                    "frontier_semantic_relevance": 0.35,
                },
            },
        ):
            candidates, plan = refresh_candidate_catalog_state(
                trainer=trainer,
                encoder=object(),
                candidate_bank_specs=[
                    {
                        "catalog_mode": "semantic_registry",
                        "catalog_limit": 3,
                        "catalog_probe_pool_limit": 4,
                        "catalog_probe_tokens": 3,
                        "catalog_scout_tokens": 5,
                        "catalog_entries": [],
                    }
                ],
                candidate_train_tokens=32,
                probe_tokens=8,
                gap_exploration_bonus=0.0,
                gap_ambiguity_weight=0.0,
                gap_switch_weight=0.0,
                gap_margin_reference=0.0,
            )

        self.assertEqual(plan, fake_plan)
        self.assertEqual([bank.name for bank in candidates], ["reviews", "amazon", "dbpedia"])
        finalist_specs = list(load_calls[1]["source_bank_specs"])
        finalist_names = [str(spec["name"]) for spec in finalist_specs]
        self.assertEqual(finalist_names, ["reviews", "amazon", "dbpedia"])
        self.assertNotIn("yelp", finalist_names)
        self.assertEqual(
            [float(spec["metadata"]["catalog_probe_selection_score"]) for spec in finalist_specs],
            [0.720, 0.711, 0.706],
        )
        self.assertEqual(
            [float(spec["metadata"]["catalog_probe_diagnostic_gap_score"]) for spec in finalist_specs],
            [0.720, 0.704, 0.706],
        )


if __name__ == "__main__":
    unittest.main()
