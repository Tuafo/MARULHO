from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from marulho.config.model_config import MarulhoConfig
from marulho.service.runtime_config import RuntimeConfig


class _RuntimeConfigPriorityFake:
    def provider_topic_family_priority(self, family_entry):
        return float(family_entry.get("commits", 0)) + 0.5 * float(family_entry.get("successes", 0))

    def provider_query_family_priority(self, family_entry):
        return float(family_entry.get("commits", 0)) + 0.5 * float(family_entry.get("successes", 0))


def _runtime_config() -> RuntimeConfig:
    fake = _RuntimeConfigPriorityFake()
    return RuntimeConfig(
        provider_query_family_priority=fake.provider_query_family_priority,
        provider_topic_family_priority=fake.provider_topic_family_priority,
    )


class RuntimeConfigSeamTests(unittest.TestCase):
    def test_normalize_brain_config_defaults_to_measured_source_tick_window(self) -> None:
        module = _runtime_config()

        normalized = module._normalize_brain_config(None)

        self.assertEqual(normalized["tick_tokens"], 128)
        self.assertEqual(normalized["ingestion"]["queue_target_tokens"], 256)

    def test_normalize_brain_config_preserves_operator_shape(self) -> None:
        module = _runtime_config()

        normalized = module._normalize_brain_config(
            {
                "source_bank": [
                    {
                        "source": "notes.txt",
                        "topic_terms": ["Cats", "  mice  "],
                        "metadata": {"label": "cats and mice"},
                    }
                ],
                "tick_tokens": 128,
                "sleep_interval_seconds": 0.02,
                "repeat_sources": False,
                "autonomy": {
                    "enabled": True,
                    "candidate_bank": [
                        {
                            "catalog_mode": "live_remote_search",
                            "catalog_providers": ["Wikipedia"],
                            "catalog_queries_per_provider": 2,
                            "catalog_provider_result_limit": 4,
                        }
                    ],
                    "provider_curriculum": {
                        "wikipedia": {
                            "topic_terms": {"cats": 1, "dogs": 0.5},
                            "topic_families": {
                                "biology": {
                                    "commits": 2,
                                    "successes": 1,
                                    "semantic_relevance_ema": 0.8,
                                }
                            },
                        }
                    },
                },
                "ingestion": {"prewarm_on_startup": True},
            }
        )

        self.assertEqual(normalized["source_bank"][0]["source"], "notes.txt")
        self.assertEqual(normalized["source_bank"][0]["name"], "source_1")
        self.assertEqual(normalized["tick_tokens"], 128)
        self.assertFalse(normalized["repeat_sources"])
        self.assertEqual(normalized["autonomy"]["enabled"], True)
        self.assertEqual(normalized["autonomy"]["candidate_bank"][0]["catalog_mode"], "live_remote_search")
        self.assertEqual(normalized["autonomy"]["provider_curriculum"]["wikipedia"]["topic_terms"]["cat"], 1.0)
        self.assertGreaterEqual(normalized["ingestion"]["queue_target_tokens"], 128)
        self.assertTrue(normalized["ingestion"]["prewarm_on_startup"])

    def test_normalize_brain_config_rejects_non_object(self) -> None:
        module = _runtime_config()
        with self.assertRaises(ValueError):
            module._normalize_brain_config("not a config")

    def test_model_config_device_report_exposes_cpu_fallback_evidence(self) -> None:
        cfg = MarulhoConfig(device="auto")

        with patch.dict("os.environ", {"MARULHO_DEVICE": "cpu"}):
            report = cfg.device_report()

        self.assertEqual(report["requested_device"], "auto")
        self.assertEqual(report["env_device"], "cpu")
        self.assertEqual(report["resolved_device"], "cpu")
        self.assertFalse(report["cuda_selected"])

    def test_model_config_device_report_exposes_explicit_cuda_selection_without_gpu(self) -> None:
        cfg = MarulhoConfig(device="cuda")

        report = cfg.device_report()

        self.assertEqual(report["requested_device"], "cuda")
        self.assertEqual(report["resolved_device"], "cuda")
        self.assertEqual(report["cuda_selected"], True)
        self.assertEqual(report["cuda_available"], torch.cuda.is_available())
