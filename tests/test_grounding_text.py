from __future__ import annotations

import unittest

from hecsn.semantics.grounding_text import match_terms
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


if __name__ == "__main__":
    unittest.main()
