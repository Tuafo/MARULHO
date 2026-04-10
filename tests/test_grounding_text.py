from __future__ import annotations

import unittest

from hecsn.semantics.grounding_text import match_terms
from hecsn.semantics.grounding_text import query_focused_clauses
from hecsn.semantics.grounding_text import query_focused_text
from hecsn.semantics.grounding_text import salient_query_terms
from hecsn.semantics.grounding_text import stream_matching_units


class GroundingTextTests(unittest.TestCase):
    def test_salient_query_terms_split_soft_camel_case_boundaries(self) -> None:
        self.assertEqual(
            salient_query_terms("submarineBallastControl"),
            ["submarine", "ballast", "control"],
        )

    def test_match_terms_matches_boundary_free_compound_against_spaced_text(self) -> None:
        matches = match_terms(
            ["submarineballastcontrol"],
            "submarine ballast control keeps buoyancy stable",
        )

        self.assertEqual(matches, ["submarineballastcontrol"])
        self.assertIn(
            "submarineballastcontrol",
            stream_matching_units("submarine ballast control keeps buoyancy stable"),
        )

    def test_query_focused_clauses_extracts_dense_focus_chunk_from_unpunctuated_window(self) -> None:
        clauses = query_focused_clauses(
            "submarine crew rotates watches ballast water controls buoyancy depth",
            ["submarine", "ballast", "buoyancy", "depth"],
        )

        lowered = [clause.lower() for clause in clauses]
        self.assertTrue(any("ballast water controls buoyancy depth" in clause for clause in lowered))
        self.assertFalse(
            any(
                clause == "submarine crew rotates watches ballast water controls buoyancy depth"
                for clause in lowered
            )
        )

    def test_query_focused_clauses_keeps_multiple_chunks_when_terms_are_distributed(self) -> None:
        clauses = query_focused_clauses(
            "cats rest indoors. cats chase mice at night.",
            ["cats", "rest", "mice", "night"],
        )

        lowered = [clause.lower() for clause in clauses]
        self.assertTrue(any("cats rest indoors" in clause for clause in lowered))
        self.assertTrue(any("cats chase mice at night" in clause for clause in lowered))

    def test_query_focused_text_preserves_single_sentence_context_for_concept_building(self) -> None:
        focused_text = query_focused_text(
            "a cat purrs when it feels safe.",
            ["purrs", "feels", "safe"],
        )

        self.assertEqual(focused_text.lower(), "a cat purrs when it feels safe.")


if __name__ == "__main__":
    unittest.main()
