from __future__ import annotations

from marulho.service.terminus_hf_sources import current_runtime_autonomy_config, current_runtime_source_bank
from marulho.service.terminus_presets import TERMINUS_QUICK_START_PRESETS


class TestRuntimeAutonomyCurriculum:
    def test_runtime_source_bank_exposes_focus_routing_terms(self):
        source_bank = current_runtime_source_bank()

        assert len(source_bank) >= 3
        assert all(source.get("topic_terms") for source in source_bank)
        assert any(source["name"] == "open_textbooks" for source in source_bank)
        assert not any(source["name"] == "wikipedia_en" for source in source_bank)
        assert any(source["name"] == "s2orc_arxiv_abstracts" for source in source_bank)
        assert any(source["name"] == "fineweb_edu" for source in source_bank)

    def test_runtime_autonomy_config_uses_real_source_semantic_registry(self):
        config = current_runtime_autonomy_config()

        assert config["enabled"] is True
        assert config["policy"] == "active"
        assert len(config["candidate_bank"]) == 1
        registry = config["candidate_bank"][0]
        assert registry["catalog_mode"] == "semantic_registry"
        entries = registry["catalog_entries"]
        assert len(entries) >= 3
        names = [entry["name"] for entry in entries]
        assert "fineweb_edu" in names
        assert "open_textbooks" in names
        assert "wikipedia_en" not in names
        assert "s2orc_arxiv_abstracts" in names
        assert all(entry["source_type"] == "hf" for entry in entries)
        assert all(entry.get("summary") for entry in entries)

    def test_curriculum_quick_start_preset_uses_autonomy_not_text_curriculum(self):
        preset = TERMINUS_QUICK_START_PRESETS["curriculum"]

        assert "autonomy" in preset
        assert "curriculum" not in preset
        assert preset["autonomy"]["enabled"] is True
        assert preset["autonomy"]["candidate_bank"][0]["catalog_mode"] == "semantic_registry"
        assert preset["model_overrides"]["memory_capacity"] == 1000
