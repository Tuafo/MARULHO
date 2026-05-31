from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from hecsn.config.model_config import HECSNConfig
from hecsn.data.rtf_encoder import RTFEncoder
from hecsn.gap_planner import plan_query_gaps
from hecsn.interaction import EvidenceResponder
from hecsn.semantics import ConceptStore
from hecsn.training.meaning_grounding_runner import run_meaning_grounding_benchmark
from hecsn.training.query_runner import build_memory_episodes, build_query_result
from hecsn.training.runner_utils import set_seed
from hecsn.training.model import HECSNModel
from hecsn.training.trainer import HECSNTrainer


class MeaningGroundingTests(unittest.TestCase):
    def test_build_memory_episodes_uses_abstraction_focus_to_resegment_priority_evidence(self) -> None:
        memory_matches = [
            {
                "memory_index": 0,
                "text": "The submarine crew rotates watches during patrols. Ballast water controls buoyancy and depth.",
                "raw_window": "submarine crew rotates watches ballast water controls buoyancy depth",
                "similarity": 0.40,
                "importance": 1.0,
            },
            {
                "memory_index": 1,
                "text": "The submarine crew rotates watches during patrols.",
                "raw_window": "submarine crew rotates watches during patrols",
                "similarity": 0.45,
                "importance": 1.0,
            },
        ]

        baseline = build_memory_episodes(
            memory_matches,
            top_k=2,
            query_terms=["submarine", "depth"],
        )
        focused = build_memory_episodes(
            memory_matches,
            top_k=2,
            query_terms=["submarine", "depth"],
            focus_terms=["ballast", "buoyancy", "depth"],
            memory_priority={"0": 1.0},
        )

        self.assertEqual(
            baseline[0]["text"].lower(),
            "the submarine crew rotates watches during patrols.",
        )
        self.assertIn("ballast water controls buoyancy and depth", focused[0]["text"].lower())
        self.assertEqual(focused[0]["memory_index"], 0)

    def test_query_result_exposes_episode_evidence_and_grounded_quote(self) -> None:
        set_seed(7)
        cfg = HECSNConfig(
            n_columns=16,
            column_latent_dim=24,
            bootstrap_tokens=0,
            memory_capacity=96,
            eta_competitive=0.05,
            eta_decay=0.0,
            input_weight_blend=0.0,
            enable_context_layer=True,
            enable_binding_layer=True,
        )
        trainer = HECSNTrainer(HECSNModel(cfg), cfg)
        encoder = RTFEncoder.from_config(cfg)

        corpus = "\n".join(
            [
                "a cat purrs when it feels safe.",
                "cats rest indoors.",
                "cats chase mice at night.",
                "a dog guards the house and barks at strangers.",
            ]
        )
        build_query_result(
            trainer=trainer,
            checkpoint=Path("test://meaning-grounding"),
            metadata={},
            encoder=encoder,
            query_text_resolved=None,
            feed_text_resolved="\n".join([corpus] * 12),
            context_text=None,
            top_k_candidates=4,
            top_k_memories=6,
            top_chars=8,
            compare_context_a=None,
            compare_context_b=None,
        )
        query_result = build_query_result(
            trainer=trainer,
            checkpoint=Path("test://meaning-grounding"),
            metadata={},
            encoder=encoder,
            query_text_resolved="What purrs when it feels safe?",
            feed_text_resolved=None,
            context_text=None,
            top_k_candidates=4,
            top_k_memories=6,
            top_chars=8,
            compare_context_a=None,
            compare_context_b=None,
        )

        query_summary = dict(query_result["query_summary"])
        self.assertTrue(query_summary["memory_episodes"])
        top_episode = query_summary["memory_episodes"][0]
        self.assertIn("cat purrs when it feels safe", top_episode["text"].lower())
        self.assertGreater(len(top_episode["text"]), len(top_episode.get("raw_window") or ""))

        concept_summary = ConceptStore().observe(
            query_text="What purrs when it feels safe?",
            memory_matches=query_summary["memory_matches"],
            memory_episodes=query_summary["memory_episodes"],
            memory_store=trainer.model.memory_store,
        )
        response = EvidenceResponder(min_similarity=0.0, min_token_coverage=0.25).build_response(
            "What purrs when it feels safe?",
            query_summary,
            concept_summary=concept_summary,
        )
        self.assertIn(response["response_mode"], {"quote", "stitch"})
        self.assertIn("cat", response["response_text"].lower())
        self.assertFalse(response["selected_evidence"][0]["fragmentary"])
        self.assertTrue(any(term in {"cat", "cats"} for term in concept_summary["concepts"][0]["top_terms"]))
        self.assertFalse(any(term in {"dog", "dogs"} for term in concept_summary["concepts"][0]["top_terms"]))

        dog_query_result = build_query_result(
            trainer=trainer,
            checkpoint=Path("test://meaning-grounding"),
            metadata={},
            encoder=encoder,
            query_text_resolved="What guards the house and barks at strangers?",
            feed_text_resolved=None,
            context_text=None,
            top_k_candidates=4,
            top_k_memories=6,
            top_chars=8,
            compare_context_a=None,
            compare_context_b=None,
        )
        dog_summary = dict(dog_query_result["query_summary"])
        dog_concepts = ConceptStore().observe(
            query_text="What guards the house and barks at strangers?",
            memory_matches=dog_summary["memory_matches"],
            memory_episodes=dog_summary["memory_episodes"],
            memory_store=trainer.model.memory_store,
        )
        self.assertTrue(any(term in {"dog", "dogs"} for term in dog_concepts["concepts"][0]["top_terms"]))
        self.assertFalse(any(term in {"cat", "cats"} for term in dog_concepts["concepts"][0]["top_terms"]))

        compositional_query_result = build_query_result(
            trainer=trainer,
            checkpoint=Path("test://meaning-grounding"),
            metadata={},
            encoder=encoder,
            query_text_resolved="Where do cats rest and what do they chase at night?",
            feed_text_resolved=None,
            context_text=None,
            top_k_candidates=4,
            top_k_memories=8,
            top_chars=8,
            compare_context_a=None,
            compare_context_b=None,
        )
        compositional_summary = dict(compositional_query_result["query_summary"])
        compositional_concepts = ConceptStore().observe(
            query_text="Where do cats rest and what do they chase at night?",
            memory_matches=compositional_summary["memory_matches"],
            memory_episodes=compositional_summary["memory_episodes"],
            memory_store=trainer.model.memory_store,
        )
        compositional_response = EvidenceResponder(min_similarity=0.0, min_token_coverage=0.25).build_response(
            "Where do cats rest and what do they chase at night?",
            compositional_summary,
            concept_summary=compositional_concepts,
        )
        self.assertEqual(compositional_response["response_mode"], "grounded_synthesis")
        self.assertIn("indoors", compositional_response["response_text"].lower())
        self.assertIn("mice", compositional_response["response_text"].lower())
        self.assertGreaterEqual(len(compositional_response["selected_evidence"]), 2)
        self.assertTrue(all(not item["fragmentary"] for item in compositional_response["selected_evidence"]))
        self.assertTrue(any(term in {"cat", "cats"} for term in compositional_concepts["concepts"][0]["top_terms"]))
        self.assertFalse(any(term in {"dog", "dogs"} for term in compositional_concepts["concepts"][0]["top_terms"]))

        gap_plan = plan_query_gaps(
            query_text="What purrs when it feels safe?",
            query_summary=query_summary,
            concept_summary=concept_summary,
        )
        self.assertIn("safe", gap_plan["query_terms"])

    def test_meaning_grounding_benchmark_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            summary = run_meaning_grounding_benchmark(output_dir=Path(tmpdir))
        self.assertTrue(summary["meaning_grounding_gate"]["pass"])

    def test_mixed_world_benchmark_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            summary = run_meaning_grounding_benchmark(output_dir=Path(tmpdir), scenario="mixed_world")
        self.assertTrue(summary["meaning_grounding_gate"]["pass"])

        octopus_case = next(item for item in summary["queries"] if item["name"] == "octopus-tools")
        self.assertEqual(octopus_case["response"]["response_mode"], "quote")
        self.assertNotIn("opens", octopus_case["response"]["unsupported_terms"])
        self.assertNotIn("solves", octopus_case["response"]["unsupported_terms"])

        composition_case = next(item for item in summary["queries"] if item["name"] == "planet-volcano-composition")
        self.assertIn(composition_case["response"]["response_mode"], ("grounded_synthesis", "quote"))
        self.assertIn("mercury", composition_case["response"]["response_text"].lower())


if __name__ == "__main__":
    unittest.main()
