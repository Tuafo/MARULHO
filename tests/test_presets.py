from __future__ import annotations

import unittest

from hecsn.config.presets import (
    autonomy_acquisition_preset_names,
    autonomy_preset_names,
    get_autonomy_acquisition_preset,
    get_autonomy_preset,
)


class AutonomyPresetTests(unittest.TestCase):
    def test_public_autonomy_presets_match_maintained_hf_surface(self) -> None:
        self.assertEqual(
            autonomy_preset_names(),
            [
                "autonomy_hf_baseline",
                "autonomy_hf_smoke",
            ],
        )

    def test_autonomy_presets_use_explicit_maintained_source_bank(self) -> None:
        for preset_name in ("autonomy_hf_smoke", "autonomy_hf_baseline"):
            with self.subTest(preset_name=preset_name):
                preset = get_autonomy_preset(preset_name)
                self.assertEqual(
                    [entry["name"] for entry in preset["source_bank"]],
                    ["news", "wiki", "reviews"],
                )
                self.assertTrue(all("catalog_mode" not in entry for entry in preset["source_bank"]))

    def test_removed_open_web_autonomy_preset_is_rejected(self) -> None:
        with self.assertRaises(KeyError):
            get_autonomy_preset("autonomy_open_web_smoke")


class AcquisitionPresetTests(unittest.TestCase):
    def test_public_acquisition_presets_match_maintained_hf_surface(self) -> None:
        self.assertEqual(
            autonomy_acquisition_preset_names(),
            [
                "autonomy_acquisition_hf_allocation",
                "autonomy_acquisition_hf_baseline",
                "autonomy_acquisition_hf_catalog",
            ],
        )

    def test_removed_scout_and_open_web_preset_names_are_rejected(self) -> None:
        for preset_name in (
            "autonomy_acquisition_hf_scout",
            "autonomy_acquisition_hf_scout_exploratory",
            "autonomy_acquisition_hf_catalog_semantic",
            "autonomy_acquisition_open_web_smoke",
            "autonomy_acquisition_open_web_scout",
        ):
            with self.subTest(preset_name=preset_name):
                with self.assertRaises(KeyError):
                    get_autonomy_acquisition_preset(preset_name)

    def test_hf_allocation_preset_remains_semantic_registry_backed(self) -> None:
        preset = get_autonomy_acquisition_preset("autonomy_acquisition_hf_allocation")

        self.assertEqual(len(preset["candidate_bank"]), 1)
        catalog_spec = preset["candidate_bank"][0]
        self.assertEqual(catalog_spec["catalog_mode"], "semantic_registry")
        self.assertEqual({entry["name"] for entry in catalog_spec["catalog_entries"]}, {"dbpedia", "reviews", "yelp"})

    def test_catalog_presets_use_semantic_registry_not_local_files(self) -> None:
        preset = get_autonomy_acquisition_preset("autonomy_acquisition_hf_catalog")

        self.assertEqual(len(preset["candidate_bank"]), 1)
        catalog_spec = preset["candidate_bank"][0]
        self.assertEqual(catalog_spec["catalog_mode"], "semantic_registry")
        self.assertTrue(all(entry["source_type"] != "file" for entry in catalog_spec["catalog_entries"]))
        for key in ("scout_commit_tokens", "scout_top_k"):
            self.assertNotIn(key, preset)
        self.assertEqual(preset["semantic_shortlist_size"], 2)
        self.assertEqual(preset["semantic_shortlist_gap_weight"], 0.35)
        self.assertEqual(preset["semantic_shortlist_affinity_weight"], 0.65)


if __name__ == "__main__":
    unittest.main()
