from __future__ import annotations

import unittest
from unittest.mock import patch

from hecsn.data.source_catalog import (
    discover_remote_search_source_specs,
    expand_source_bank_specs,
    select_catalog_source_specs,
)


class SourceCatalogTests(unittest.TestCase):
    def test_select_catalog_source_specs_prefers_semantic_match_with_diversity(self) -> None:
        spec = {
            "catalog_mode": "semantic_registry",
            "catalog_limit": 2,
            "catalog_focus_text": "predictive coding memory consolidation plasticity",
            "catalog_diversity_weight": 0.80,
            "catalog_entries": [
                {
                    "name": "predictive",
                    "source": "https://example.com/predictive",
                    "source_type": "web",
                    "summary": "predictive coding hierarchy inference prediction error",
                    "catalog_priority": 0.60,
                },
                {
                    "name": "plasticity",
                    "source": "https://example.com/plasticity",
                    "source_type": "web",
                    "summary": "synaptic plasticity memory consolidation replay stability",
                    "catalog_priority": 0.55,
                },
                {
                    "name": "predictive_dup",
                    "source": "https://example.com/predictive-dup",
                    "source_type": "web",
                    "summary": "predictive coding hierarchy prediction error generative model",
                    "catalog_priority": 0.58,
                },
            ],
        }

        selected = select_catalog_source_specs(spec)

        self.assertEqual([item["name"] for item in selected], ["predictive", "plasticity"])
        self.assertGreater(
            selected[0]["metadata"]["combined_score"],
            0.0,
        )

    def test_select_catalog_source_specs_rejects_local_file_sources(self) -> None:
        spec = {
            "catalog_mode": "semantic_registry",
            "catalog_entries": [
                {
                    "name": "local",
                    "source": "notes.txt",
                    "source_type": "file",
                    "summary": "should not be allowed",
                }
            ],
        }

        with self.assertRaises(ValueError):
            select_catalog_source_specs(spec)

    def test_expand_source_bank_specs_passes_through_explicit_sources(self) -> None:
        specs = [
            {"name": "news", "source": "ag_news", "source_type": "hf", "text_field": "text"},
        ]

        expanded = expand_source_bank_specs(specs)

        self.assertEqual(expanded, specs)

    def test_discover_remote_search_source_specs_uses_plan_queries(self) -> None:
        queries: list[str] = []

        def fake_search(provider: str, query: str, *, result_limit: int, timeout_seconds: float) -> list[dict[str, object]]:
            queries.append(f"{provider}:{query}")
            return [
                {
                    "name": f"{provider}_result",
                    "source": f"https://example.com/{provider}/{query.replace(' ', '_')}",
                    "source_type": "web",
                    "summary": f"{query} spiking plasticity memory",
                    "query_text": query,
                    "catalog_priority": 0.6,
                    "provider": provider,
                }
            ]

        with patch("hecsn.data.source_catalog._search_remote_provider", side_effect=fake_search):
            selected = discover_remote_search_source_specs(
                {
                    "catalog_mode": "live_remote_search",
                    "catalog_providers": ["wikipedia", "arxiv"],
                    "catalog_limit": 2,
                    "catalog_queries_per_provider": 2,
                    "catalog_provider_result_limit": 1,
                },
                semantic_plan={
                    "retrieval_queries": ["spiking plasticity", "memory consolidation"],
                    "gap_terms": [{"term": "plasticity", "weight": 2.0}],
                    "unsupported_terms": ["plasticity"],
                },
            )

        self.assertEqual(len(selected), 2)
        self.assertIn("wikipedia:spiking plasticity", queries)
        self.assertIn("arxiv:memory consolidation", queries)


if __name__ == "__main__":
    unittest.main()
