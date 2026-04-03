from __future__ import annotations

import unittest

from hecsn.interaction import EvidenceResponder


class EvidenceResponderTests(unittest.TestCase):
    def test_concept_grounding_spreads_evidence_across_top_concepts(self) -> None:
        responder = EvidenceResponder(min_similarity=0.0, min_token_coverage=0.25)

        query_summary = {
            "memory_matches": [
                {
                    "memory_index": 0,
                    "raw_window": "river bank water current",
                    "similarity": 0.82,
                    "importance": 1.0,
                    "age_tokens": 3,
                },
                {
                    "memory_index": 1,
                    "raw_window": "river bank mud shore",
                    "similarity": 0.81,
                    "importance": 1.0,
                    "age_tokens": 2,
                },
                {
                    "memory_index": 2,
                    "raw_window": "bank account credit deposit",
                    "similarity": 0.78,
                    "importance": 1.0,
                    "age_tokens": 1,
                },
            ],
            "native_decode": {"available": False},
        }
        concept_summary = {
            "concepts": [
                {
                    "label": "river / bank",
                    "score": 1.35,
                    "match_count": 2,
                    "memory_indices": [0, 1],
                    "top_terms": ["river", "water", "shore"],
                },
                {
                    "label": "bank / account",
                    "score": 1.28,
                    "match_count": 1,
                    "memory_indices": [2],
                    "top_terms": ["account", "credit", "deposit"],
                },
            ]
        }

        response = responder.build_response(
            "bank river account",
            query_summary,
            concept_summary=concept_summary,
            max_evidence_items=2,
        )

        self.assertEqual(response["response_mode"], "stitch")
        self.assertEqual(
            [item["memory_index"] for item in response["selected_evidence"]],
            [0, 2],
        )
        self.assertEqual(
            [item["primary_concept"] for item in response["selected_evidence"]],
            ["river / bank", "bank / account"],
        )
        self.assertEqual(
            [item["label"] for item in response["concept_grounding"]["selected_concepts"]],
            ["river / bank", "bank / account"],
        )
        self.assertAlmostEqual(response["concept_grounding"]["query_concept_coverage"], 1.0)


if __name__ == "__main__":
    unittest.main()