from __future__ import annotations

import unittest
from unittest.mock import patch

from marulho.data.source_catalog import (
    discover_remote_search_source_specs,
    expand_source_bank_specs,
    select_catalog_source_specs,
)
from marulho.gap_planner import plan_query_gaps


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

    def test_select_catalog_source_specs_normalizes_missing_optional_metadata_fields(self) -> None:
        selected = select_catalog_source_specs(
            {
                "catalog_mode": "semantic_registry",
                "catalog_limit": 1,
                "catalog_focus_text": "wall street",
                "catalog_entries": [
                    {
                        "name": "candidate",
                        "source": "https://example.com/candidate",
                        "source_type": "web",
                        "summary": "generic catalog entry",
                    }
                ],
            }
        )

        metadata = selected[0]["metadata"]
        self.assertEqual(metadata["provider"], "")
        self.assertEqual(metadata["query_text"], "")

    def test_select_catalog_source_specs_matches_boundary_free_focus_text(self) -> None:
        selected = select_catalog_source_specs(
            {
                "catalog_mode": "semantic_registry",
                "catalog_limit": 1,
                "catalog_focus_text": "submarineballastcontrol",
                "catalog_entries": [
                    {
                        "name": "submarine",
                        "source": "https://example.com/submarine",
                        "source_type": "web",
                        "summary": "submarine ballast control regulates buoyancy underwater",
                        "catalog_priority": 0.35,
                    },
                    {
                        "name": "garden",
                        "source": "https://example.com/garden",
                        "source_type": "web",
                        "summary": "garden tomatoes soil moisture sunlight watering",
                        "catalog_priority": 0.55,
                    },
                ],
            }
        )

        self.assertEqual([item["name"] for item in selected], ["submarine"])
        self.assertGreater(float(selected[0]["metadata"]["semantic_relevance"]), 0.0)

    def test_expand_source_bank_specs_passes_through_explicit_sources(self) -> None:
        specs = [
            {"name": "news", "source": "ag_news", "source_type": "hf", "text_field": "text"},
        ]

        expanded = expand_source_bank_specs(specs)

        self.assertEqual(expanded, specs)

    def test_expand_source_bank_specs_metadata_prefilter_uses_probe_pool_limit(self) -> None:
        spec = {
            "catalog_mode": "semantic_registry",
            "catalog_limit": 1,
            "catalog_probe_pool_limit": 2,
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

        finalists = expand_source_bank_specs([spec])
        probe_pool = expand_source_bank_specs([spec], metadata_prefilter=True)

        self.assertEqual([item["name"] for item in finalists], ["predictive"])
        self.assertEqual([item["name"] for item in probe_pool], ["predictive", "plasticity"])

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

        with patch("marulho.data.source_catalog._search_remote_provider", side_effect=fake_search):
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

    def test_discover_remote_search_source_specs_emits_learned_provider_query_families(self) -> None:
        queries: list[str] = []

        def fake_search(provider: str, query: str, *, result_limit: int, timeout_seconds: float) -> list[dict[str, object]]:
            queries.append(f"{provider}:{query}")
            return [
                {
                    "name": f"{provider}_{query.replace(' ', '_')}",
                    "source": f"https://example.com/{provider}/{query.replace(' ', '_')}",
                    "source_type": "web",
                    "summary": f"{query} submarine ballast trim stability",
                    "query_text": query,
                    "catalog_priority": 0.6,
                    "provider": provider,
                }
            ]

        with patch("marulho.data.source_catalog._search_remote_provider", side_effect=fake_search):
            selected = discover_remote_search_source_specs(
                {
                    "catalog_mode": "live_remote_search",
                    "catalog_providers": ["wikipedia"],
                    "catalog_limit": 2,
                    "catalog_queries_per_provider": 2,
                    "catalog_provider_result_limit": 1,
                    "catalog_provider_query_families": {
                        "wikipedia": ["ballast trim stability"],
                    },
                },
                semantic_plan={
                    "planner_mode": "geometric_abstraction_gap_focus",
                    "retrieval_queries": ["submarine buoyancy ballast"],
                    "geometric_gaps": [
                        {
                            "concept_index": 0,
                            "gap_score": 0.4,
                            "top_terms": ["submarine", "ballast", "trim"],
                        }
                    ],
                },
            )

        self.assertEqual(
            queries,
            [
                "wikipedia:submarine buoyancy ballast",
                "wikipedia:ballast trim stability",
            ],
        )
        self.assertEqual(len(selected), 2)

    def test_discover_remote_search_source_specs_can_use_follow_up_questions_as_queries(self) -> None:
        queries: list[str] = []

        def fake_search(provider: str, query: str, *, result_limit: int, timeout_seconds: float) -> list[dict[str, object]]:
            queries.append(query)
            return [
                {
                    "name": "submarine_source",
                    "source": "https://example.com/submarine",
                    "source_type": "web",
                    "summary": "submarine buoyancy ballast pressure trim tanks",
                    "query_text": query,
                    "catalog_priority": 0.6,
                    "provider": provider,
                }
            ]

        with patch("marulho.data.source_catalog._search_remote_provider", side_effect=fake_search):
            selected = discover_remote_search_source_specs(
                {
                    "catalog_mode": "live_remote_search",
                    "catalog_providers": ["wikipedia"],
                    "catalog_limit": 1,
                    "catalog_queries_per_provider": 1,
                    "catalog_provider_result_limit": 1,
                },
                semantic_plan={
                    "planner_mode": "recent_query_gap_focus",
                    "follow_up_questions": ["What grounded evidence explains submarine ballast control?"],
                },
            )

        self.assertEqual(queries, ["submarine ballast control"])
        self.assertEqual([item["name"] for item in selected], ["submarine_source"])
        self.assertEqual(selected[0]["metadata"]["query_text"], "submarine ballast control")

    def test_discover_remote_search_source_specs_can_use_weak_concepts_as_queries(self) -> None:
        queries: list[str] = []

        def fake_search(provider: str, query: str, *, result_limit: int, timeout_seconds: float) -> list[dict[str, object]]:
            queries.append(query)
            return [
                {
                    "name": "submarine_source",
                    "source": "https://example.com/submarine",
                    "source_type": "web",
                    "summary": "submarine buoyancy ballast pressure trim tanks",
                    "query_text": query,
                    "catalog_priority": 0.6,
                    "provider": provider,
                }
            ]

        with patch("marulho.data.source_catalog._search_remote_provider", side_effect=fake_search):
            selected = discover_remote_search_source_specs(
                {
                    "catalog_mode": "live_remote_search",
                    "catalog_providers": ["wikipedia"],
                    "catalog_limit": 1,
                    "catalog_queries_per_provider": 1,
                    "catalog_provider_result_limit": 1,
                },
                semantic_plan={
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
                },
            )

        self.assertEqual(queries, ["submarine ballast buoyancy"])
        self.assertEqual([item["name"] for item in selected], ["submarine_source"])
        self.assertEqual(selected[0]["metadata"]["query_text"], "submarine ballast buoyancy")

    def test_discover_remote_search_source_specs_supports_openalex_provider(self) -> None:
        with patch(
            "marulho.data.source_catalog._http_get_json",
            return_value={
                "results": [
                    {
                        "display_name": "Submarine buoyancy control",
                        "id": "https://openalex.org/W123",
                        "primary_location": {
                            "landing_page_url": "https://example.com/openalex/submarine",
                        },
                        "primary_topic": {
                            "display_name": "Marine engineering systems",
                            "subfield": {"display_name": "Marine engineering"},
                            "field": {"display_name": "Engineering"},
                            "domain": {"display_name": "Physical sciences"},
                        },
                        "topics": [
                            {
                                "display_name": "Ballast control",
                                "subfield": {"display_name": "Naval architecture"},
                                "field": {"display_name": "Engineering"},
                                "domain": {"display_name": "Physical sciences"},
                            }
                        ],
                        "keywords": [
                            {"display_name": "Submarine"},
                            {"display_name": "Ballast tank"},
                        ],
                        "concepts": [
                            {"display_name": "Buoyancy"},
                            {"display_name": "Trim control"},
                        ],
                        "abstract_inverted_index": {
                            "Submarine": [0],
                            "buoyancy": [1],
                            "control": [2],
                            "uses": [3],
                            "ballast": [4],
                            "tanks": [5],
                        },
                    }
                ]
            },
        ):
            selected = discover_remote_search_source_specs(
                {
                    "catalog_mode": "live_remote_search",
                    "catalog_providers": ["openalex"],
                    "catalog_limit": 1,
                    "catalog_queries_per_provider": 1,
                    "catalog_provider_result_limit": 1,
                },
                semantic_plan={
                    "retrieval_queries": ["submarine buoyancy ballast"],
                    "gap_terms": [{"term": "submarine", "weight": 2.0}],
                    "unsupported_terms": ["submarine"],
                },
            )

        self.assertEqual([item["name"] for item in selected], ["submarine_buoyancy_control"])
        self.assertEqual(selected[0]["source"], "https://example.com/openalex/submarine")
        self.assertEqual(selected[0]["metadata"]["provider"], "openalex")
        self.assertIn("ballast", selected[0]["metadata"]["catalog_summary"].lower())
        catalog_terms = [term.lower() for term in selected[0]["metadata"]["catalog_terms"]]
        self.assertIn("marine engineering systems", catalog_terms)
        self.assertIn("ballast tank", catalog_terms)

    def test_discover_remote_search_source_specs_extracts_arxiv_topic_terms(self) -> None:
        payload = """<?xml version='1.0' encoding='UTF-8'?>
<feed xmlns:arxiv='http://arxiv.org/schemas/atom' xmlns='http://www.w3.org/2005/Atom'>
  <entry>
    <id>http://arxiv.org/abs/1234.5678v1</id>
    <title>Solution of Supplee's submarine paradox</title>
    <summary>Relativistic analysis of submarine buoyancy and ballast behavior.</summary>
    <category term='physics.class-ph' />
    <category term='gr-qc' />
    <arxiv:primary_category term='physics.class-ph' />
    <arxiv:comment>Keywords: Supplee's submarine paradox, Archimedes principle, Lorentz force</arxiv:comment>
  </entry>
</feed>
"""
        with patch("marulho.data.source_catalog._http_get_text", return_value=payload):
            selected = discover_remote_search_source_specs(
                {
                    "catalog_mode": "live_remote_search",
                    "catalog_providers": ["arxiv"],
                    "catalog_limit": 1,
                    "catalog_queries_per_provider": 1,
                    "catalog_provider_result_limit": 1,
                },
                semantic_plan={
                    "retrieval_queries": ["submarine buoyancy ballast"],
                    "gap_terms": [{"term": "submarine", "weight": 2.0}],
                    "unsupported_terms": ["submarine", "buoyancy", "ballast"],
                },
            )

        catalog_terms = [term.lower() for term in selected[0]["metadata"]["catalog_terms"]]
        self.assertIn("physics class ph", catalog_terms)
        self.assertIn("gr qc", catalog_terms)
        self.assertIn("archimedes principle", catalog_terms)

    def test_discover_remote_search_source_specs_uses_openalex_api_fallback_for_closed_doi(self) -> None:
        with patch(
            "marulho.data.source_catalog._http_get_json",
            return_value={
                "results": [
                    {
                        "display_name": "Closed-access submarine paper",
                        "id": "https://openalex.org/W999",
                        "primary_location": {
                            "landing_page_url": "https://doi.org/10.1000/example",
                        },
                        "open_access": {
                            "oa_url": None,
                        },
                        "abstract_inverted_index": {
                            "Submarine": [0],
                            "ballast": [1],
                        },
                    }
                ]
            },
        ):
            selected = discover_remote_search_source_specs(
                {
                    "catalog_mode": "live_remote_search",
                    "catalog_providers": ["openalex"],
                    "catalog_limit": 1,
                    "catalog_queries_per_provider": 1,
                    "catalog_provider_result_limit": 1,
                },
                semantic_plan={
                    "retrieval_queries": ["submarine ballast"],
                    "gap_terms": [{"term": "submarine", "weight": 2.0}],
                    "unsupported_terms": ["submarine", "ballast"],
                },
            )

        self.assertEqual(selected[0]["source"], "https://api.openalex.org/works/W999")

    def test_discover_remote_search_source_specs_uses_wikipedia_extracts_for_content_ranking(self) -> None:
        search_payload = {
            "query": {
                "search": [
                    {
                        "title": "Submarine overview alpha",
                        "pageid": 10,
                        "snippet": "A submarine overview article.",
                    },
                    {
                        "title": "Submarine overview beta",
                        "pageid": 20,
                        "snippet": "A submarine overview article.",
                    },
                ]
            }
        }
        extract_payload = {
            "query": {
                "pages": {
                    "10": {
                        "pageid": 10,
                        "extract": "Submarine cables carry internet traffic across oceans.",
                    },
                    "20": {
                        "pageid": 20,
                        "extract": "Submarine buoyancy is controlled by ballast tanks and trim systems.",
                    },
                }
            }
        }
        with patch(
            "marulho.data.source_catalog._http_get_json",
            side_effect=[search_payload, extract_payload],
        ):
            selected = discover_remote_search_source_specs(
                {
                    "catalog_mode": "live_remote_search",
                    "catalog_providers": ["wikipedia"],
                    "catalog_limit": 1,
                    "catalog_prior_weight": 0.0,
                    "catalog_queries_per_provider": 1,
                    "catalog_provider_result_limit": 2,
                },
                semantic_plan={
                    "retrieval_queries": ["submarine ballast buoyancy"],
                    "gap_terms": [{"term": "submarine", "weight": 2.0}],
                    "unsupported_terms": ["submarine", "ballast", "buoyancy"],
                },
            )

        self.assertEqual([item["name"] for item in selected], ["submarine_overview_beta"])
        self.assertIn("ballast tanks", selected[0]["metadata"]["catalog_summary"].lower())
        self.assertEqual(selected[0]["metadata"]["provider"], "wikipedia")

    def test_discover_remote_search_source_specs_can_probe_cross_provider_page_content_for_ranking(self) -> None:
        def fake_search(provider: str, query: str, *, result_limit: int, timeout_seconds: float) -> list[dict[str, object]]:
            if provider == "wikipedia":
                return [
                    {
                        "name": "cable_source",
                        "source": "https://example.com/wiki/cable",
                        "source_type": "web",
                        "summary": "Submarine infrastructure and communications systems.",
                        "query_text": query,
                        "catalog_priority": 0.65,
                        "provider": provider,
                    }
                ][:result_limit]
            return [
                {
                    "name": "ballast_paper",
                    "source": "https://example.com/openalex/ballast",
                    "source_type": "web",
                    "summary": "Marine systems analysis of vessel trim and stability.",
                    "query_text": query,
                    "catalog_priority": 0.35,
                    "provider": provider,
                }
            ][:result_limit]

        def fake_content(source: str, *, timeout_seconds: float) -> str:
            if source.endswith("/ballast"):
                return "Ballast tanks reduce submarine buoyancy and support trim control underwater."
            return "Submarine cables carry internet traffic between continents."

        with patch("marulho.data.source_catalog._search_remote_provider", side_effect=fake_search):
            with patch("marulho.data.source_catalog._fetch_remote_content_text_cached", side_effect=fake_content):
                selected = discover_remote_search_source_specs(
                    {
                        "catalog_mode": "live_remote_search",
                        "catalog_providers": ["wikipedia", "openalex"],
                        "catalog_limit": 1,
                        "catalog_probe_pool_limit": 2,
                        "catalog_prior_weight": 0.4,
                        "catalog_queries_per_provider": 1,
                        "catalog_provider_result_limit": 1,
                    },
                    semantic_plan={
                        "retrieval_queries": ["submarine buoyancy ballast"],
                        "gap_terms": [
                            {"term": "submarine", "weight": 2.0},
                            {"term": "buoyancy", "weight": 1.8},
                            {"term": "ballast", "weight": 1.7},
                        ],
                        "unsupported_terms": ["submarine", "buoyancy", "ballast"],
                    },
                )

        self.assertEqual([item["name"] for item in selected], ["ballast_paper"])
        self.assertIn("ballast tanks", selected[0]["metadata"]["catalog_summary"].lower())
        self.assertIn("submarine", selected[0]["metadata"]["catalog_content_preview"].lower())
        self.assertTrue(selected[0]["metadata"]["catalog_content_preview_preferred"])

    def test_discover_remote_search_source_specs_ranks_unsegmented_character_stream_query(self) -> None:
        def fake_search(provider: str, query: str, *, result_limit: int, timeout_seconds: float) -> list[dict[str, object]]:
            return [
                {
                    "name": "submarine_source",
                    "source": "https://example.com/submarine",
                    "source_type": "web",
                    "summary": "submarines regulate buoyancy with ballast tanks underwater",
                    "query_text": query,
                    "catalog_priority": 0.45,
                    "provider": provider,
                },
                {
                    "name": "garden_source",
                    "source": "https://example.com/garden",
                    "source_type": "web",
                    "summary": "garden tomatoes require soil sunlight and watering",
                    "query_text": query,
                    "catalog_priority": 0.65,
                    "provider": provider,
                },
            ][:result_limit]

        with patch("marulho.data.source_catalog._search_remote_provider", side_effect=fake_search):
            selected = discover_remote_search_source_specs(
                {
                    "catalog_mode": "live_remote_search",
                    "catalog_providers": ["wikipedia"],
                    "catalog_limit": 1,
                    "catalog_queries_per_provider": 1,
                    "catalog_provider_result_limit": 2,
                },
                semantic_plan={
                    "retrieval_queries": ["submarineballast"],
                    "gap_terms": [{"term": "submarineballast", "weight": 2.0}],
                    "unsupported_terms": ["submarineballast"],
                },
            )

        self.assertEqual([item["name"] for item in selected], ["submarine_source"])

    def test_discover_remote_search_source_specs_uses_chunked_boundary_free_query(self) -> None:
        seen_queries: list[str] = []

        def fake_search(provider: str, query: str, *, result_limit: int, timeout_seconds: float) -> list[dict[str, object]]:
            seen_queries.append(query)
            return [
                {
                    "name": "submarine_source",
                    "source": "https://example.com/submarine",
                    "source_type": "web",
                    "summary": "submarine ballast control regulates buoyancy underwater",
                    "query_text": query,
                    "catalog_priority": 0.45,
                    "provider": provider,
                }
            ][:result_limit]

        semantic_plan = plan_query_gaps(
            query_text="submarineballastcontrol",
            query_summary={
                "memory_matches": [
                    {
                        "raw_window": "submarine ballast control regulates buoyancy underwater",
                        "similarity": 0.91,
                    }
                ]
            },
            concept_summary={"concepts": []},
        )

        with patch("marulho.data.source_catalog._search_remote_provider", side_effect=fake_search):
            selected = discover_remote_search_source_specs(
                {
                    "catalog_mode": "live_remote_search",
                    "catalog_providers": ["wikipedia"],
                    "catalog_limit": 1,
                    "catalog_queries_per_provider": 1,
                    "catalog_provider_result_limit": 1,
                },
                semantic_plan=semantic_plan,
            )

        self.assertEqual([item["name"] for item in selected], ["submarine_source"])
        self.assertEqual(len(seen_queries), 1)
        self.assertTrue(seen_queries[0].startswith("submarine ballast control"))

    def test_discover_remote_search_source_specs_ranks_semantic_match_above_distractors(self) -> None:
        def fake_search(provider: str, query: str, *, result_limit: int, timeout_seconds: float) -> list[dict[str, object]]:
            return [
                {
                    "name": "submarine_source",
                    "source": "https://example.com/submarine",
                    "source_type": "web",
                    "summary": "submarine buoyancy ballast pressure trim tanks",
                    "query_text": query,
                    "catalog_priority": 0.2,
                    "provider": provider,
                },
                {
                    "name": "garden_source",
                    "source": "https://example.com/garden",
                    "source_type": "web",
                    "summary": "garden tomatoes soil sunlight watering",
                    "query_text": query,
                    "catalog_priority": 0.45,
                    "provider": provider,
                },
                {
                    "name": "astronomy_source",
                    "source": "https://example.com/astronomy",
                    "source_type": "web",
                    "summary": "astronomy planets observatory telescope orbit",
                    "query_text": query,
                    "catalog_priority": 0.35,
                    "provider": provider,
                },
            ][:result_limit]

        with patch("marulho.data.source_catalog._search_remote_provider", side_effect=fake_search):
            selected = discover_remote_search_source_specs(
                {
                    "catalog_mode": "live_remote_search",
                    "catalog_providers": ["wikipedia"],
                    "catalog_limit": 2,
                    "catalog_queries_per_provider": 1,
                    "catalog_provider_result_limit": 3,
                },
                semantic_plan={
                    "retrieval_queries": ["submarine buoyancy ballast"],
                    "gap_terms": [
                        {"term": "submarine", "weight": 2.0},
                        {"term": "buoyancy", "weight": 1.8},
                        {"term": "ballast", "weight": 1.7},
                    ],
                    "unsupported_terms": ["submarine", "buoyancy", "ballast"],
                },
            )

        self.assertEqual([item["name"] for item in selected], ["submarine_source", "garden_source"])
        self.assertEqual(selected[0]["metadata"]["provider"], "wikipedia")
        self.assertEqual(selected[0]["metadata"]["query_text"], "submarine buoyancy ballast")
        self.assertGreater(
            float(selected[0]["metadata"]["semantic_relevance"]),
            float(selected[1]["metadata"]["semantic_relevance"]),
        )

    def test_discover_remote_search_source_specs_can_bias_ranking_by_provider_curriculum(self) -> None:
        def fake_search(provider: str, query: str, *, result_limit: int, timeout_seconds: float) -> list[dict[str, object]]:
            return [
                {
                    "name": f"{provider}_submarine_source",
                    "source": f"https://example.com/{provider}/submarine",
                    "source_type": "web",
                    "summary": "submarine ballast buoyancy pressure trim tanks",
                    "query_text": query,
                    "catalog_priority": 0.30,
                    "provider": provider,
                }
            ][:result_limit]

        with patch("marulho.data.source_catalog._search_remote_provider", side_effect=fake_search):
            selected = discover_remote_search_source_specs(
                {
                    "catalog_mode": "live_remote_search",
                    "catalog_providers": ["arxiv", "wikipedia"],
                    "catalog_provider_priority_map": {"wikipedia": 0.8, "arxiv": 0.1},
                    "catalog_provider_priority_weight": 0.50,
                    "catalog_limit": 1,
                    "catalog_queries_per_provider": 1,
                    "catalog_provider_result_limit": 1,
                },
                semantic_plan={
                    "retrieval_queries": ["submarine buoyancy ballast"],
                    "gap_terms": [{"term": "submarine", "weight": 2.0}],
                    "unsupported_terms": ["submarine"],
                },
            )

        self.assertEqual([item["name"] for item in selected], ["wikipedia_submarine_source"])
        self.assertEqual(selected[0]["metadata"]["provider"], "wikipedia")
        self.assertGreater(float(selected[0]["metadata"]["provider_priority"]), 0.0)

    def test_discover_remote_search_source_specs_can_expand_provider_queries_from_topic_terms(self) -> None:
        queries: list[str] = []

        def fake_search(provider: str, query: str, *, result_limit: int, timeout_seconds: float) -> list[dict[str, object]]:
            queries.append(f"{provider}:{query}")
            return [
                {
                    "name": "provider_topic_source",
                    "source": f"https://example.com/{provider}/topic",
                    "source_type": "web",
                    "summary": "submarine buoyancy ballast marine engineering",
                    "query_text": query,
                    "catalog_priority": 0.9,
                    "provider": provider,
                }
            ]

        with patch("marulho.data.source_catalog._search_remote_provider", side_effect=fake_search):
            discover_remote_search_source_specs(
                {
                    "catalog_mode": "live_remote_search",
                    "catalog_providers": ["openalex"],
                    "catalog_provider_topic_terms": {
                        "openalex": ["marine engineering", "ballast tank"],
                    },
                    "catalog_limit": 1,
                    "catalog_queries_per_provider": 2,
                    "catalog_provider_result_limit": 1,
                },
                semantic_plan={
                    "retrieval_queries": ["submarine buoyancy ballast"],
                    "gap_terms": [{"term": "submarine", "weight": 2.0}],
                    "unsupported_terms": ["submarine", "buoyancy", "ballast"],
                },
            )

        self.assertEqual(
            queries,
            [
                "openalex:submarine buoyancy ballast",
                "openalex:submarine buoyancy ballast marine engineering ballast tank",
            ],
        )

    def test_discover_remote_search_source_specs_reuses_cached_provider_results(self) -> None:
        search_calls = 0

        def fake_search(provider: str, query: str, *, result_limit: int, timeout_seconds: float) -> list[dict[str, object]]:
            nonlocal search_calls
            search_calls += 1
            return [
                {
                    "name": "submarine_source",
                    "source": "https://example.com/submarine",
                    "source_type": "web",
                    "summary": "submarine buoyancy ballast pressure trim tanks",
                    "query_text": query,
                    "catalog_priority": 0.6,
                    "provider": provider,
                }
            ]

        spec = {
            "catalog_mode": "live_remote_search",
            "catalog_providers": ["wikipedia"],
            "catalog_limit": 1,
            "catalog_queries_per_provider": 1,
            "catalog_provider_result_limit": 1,
        }
        semantic_plan = {
            "retrieval_queries": ["submarine buoyancy ballast"],
            "gap_terms": [{"term": "submarine", "weight": 2.0}],
            "unsupported_terms": ["submarine"],
        }

        with patch("marulho.data.source_catalog._search_remote_provider", side_effect=fake_search):
            first = discover_remote_search_source_specs(spec, semantic_plan=semantic_plan)
            second = discover_remote_search_source_specs(spec, semantic_plan=semantic_plan)

        self.assertEqual(search_calls, 1)
        self.assertEqual([item["name"] for item in first], ["submarine_source"])
        self.assertEqual([item["name"] for item in second], ["submarine_source"])

    def test_discover_remote_search_source_specs_merges_duplicate_sources_across_queries(self) -> None:
        def fake_search(provider: str, query: str, *, result_limit: int, timeout_seconds: float) -> list[dict[str, object]]:
            if query == "submarine vehicle":
                return [
                    {
                        "name": "submarine_source",
                        "source": "https://example.com/submarine",
                        "source_type": "web",
                        "summary": "submarine vehicle history naval engineering",
                        "query_text": query,
                        "catalog_priority": 0.10,
                        "provider": provider,
                    },
                    {
                        "name": "garden_source",
                        "source": "https://example.com/garden",
                        "source_type": "web",
                        "summary": "garden tomatoes soil sunlight watering",
                        "query_text": query,
                        "catalog_priority": 0.55,
                        "provider": provider,
                    },
                ][:result_limit]
            return [
                {
                    "name": "submarine_source",
                    "source": "https://example.com/submarine",
                    "source_type": "web",
                    "summary": "ballast tanks control submarine buoyancy and trim underwater",
                    "query_text": query,
                    "catalog_priority": 0.90,
                    "provider": provider,
                }
            ][:result_limit]

        with patch("marulho.data.source_catalog._search_remote_provider", side_effect=fake_search):
            selected = discover_remote_search_source_specs(
                {
                    "catalog_mode": "live_remote_search",
                    "catalog_providers": ["wikipedia"],
                    "catalog_limit": 1,
                    "catalog_queries_per_provider": 2,
                    "catalog_provider_result_limit": 2,
                },
                semantic_plan={
                    "retrieval_queries": ["submarine vehicle", "submarine buoyancy ballast"],
                    "gap_terms": [
                        {"term": "submarine", "weight": 2.0},
                        {"term": "buoyancy", "weight": 1.8},
                        {"term": "ballast", "weight": 1.7},
                    ],
                    "unsupported_terms": ["submarine", "buoyancy", "ballast"],
                },
            )

        self.assertEqual([item["name"] for item in selected], ["submarine_source"])
        self.assertEqual(
            selected[0]["metadata"]["catalog_summary"],
            "ballast tanks control submarine buoyancy and trim underwater",
        )
        self.assertEqual(selected[0]["metadata"]["duplicate_count"], 2)
        self.assertEqual(selected[0]["metadata"]["providers"], ["wikipedia"])
        self.assertEqual(
            selected[0]["metadata"]["query_texts"],
            ["submarine vehicle", "submarine buoyancy ballast"],
        )

    def test_discover_remote_search_source_specs_briefly_caches_provider_failures(self) -> None:
        search_calls = 0

        def fake_search(provider: str, query: str, *, result_limit: int, timeout_seconds: float) -> list[dict[str, object]]:
            nonlocal search_calls
            search_calls += 1
            raise TimeoutError("provider timeout")

        spec = {
            "catalog_mode": "live_remote_search",
            "catalog_providers": ["wikipedia"],
            "catalog_limit": 1,
            "catalog_queries_per_provider": 1,
            "catalog_provider_result_limit": 1,
        }
        semantic_plan = {
            "retrieval_queries": ["submarine buoyancy ballast"],
            "gap_terms": [{"term": "submarine", "weight": 2.0}],
            "unsupported_terms": ["submarine"],
        }

        with patch("marulho.data.source_catalog._search_remote_provider", side_effect=fake_search):
            with self.assertRaisesRegex(ValueError, "provider timeout"):
                discover_remote_search_source_specs(spec, semantic_plan=semantic_plan)
            with self.assertRaisesRegex(ValueError, "provider timeout"):
                discover_remote_search_source_specs(spec, semantic_plan=semantic_plan)

        self.assertEqual(search_calls, 1)


if __name__ == "__main__":
    unittest.main()
