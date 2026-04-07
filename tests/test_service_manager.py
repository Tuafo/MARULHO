from __future__ import annotations

from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from hecsn.config.model_config import HECSNConfig
from hecsn.service.manager import HECSNServiceManager
from hecsn.training.checkpointing import save_trainer_checkpoint
from hecsn.training.trainer import HECSNModelLite, HECSNTrainer


def _build_manager(root: Path, *, test_case: str) -> HECSNServiceManager:
    cfg = HECSNConfig(
        n_columns=4,
        column_latent_dim=8,
        bootstrap_tokens=0,
        memory_capacity=64,
        eta_competitive=0.05,
        eta_decay=0.0,
        input_weight_blend=0.0,
        enable_context_layer=True,
        enable_binding_layer=True,
    )
    model = HECSNModelLite(cfg)
    trainer = HECSNTrainer(model, cfg)
    checkpoint_path = save_trainer_checkpoint(
        root / "initial.pt",
        trainer,
        metadata={"test_case": test_case},
    )
    return HECSNServiceManager(
        checkpoint_path,
        trace_dir=root / "traces",
    )


class ServiceManagerCheckpointTests(unittest.TestCase):
    def test_save_restore_round_trips_concept_store_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="service_manager_checkpoint_roundtrip")
            try:
                manager.feed(text="river bank water current\nmoney bank credit loan\nriver reeds current bank\n")
                fed = manager.status()["concept_store"]
                self.assertGreater(int(fed["concept_count"]), 0)
                self.assertGreater(int(fed["observations"]), 0)
                river_query = manager.query(query_text="river bank current", top_k_memories=6)
                manager.query(query_text="money bank loan", top_k_memories=6)

                self.assertIn("gap_plan", river_query)
                self.assertEqual(river_query["gap_plan"]["planner_mode"], "semantic_gap_planner")

                before = manager.status()["concept_store"]
                self.assertGreater(int(before["concept_count"]), 0)
                self.assertGreater(int(before["observations"]), 0)

                saved = manager.save_checkpoint(str(root / "service.pt"))
                restored = HECSNServiceManager(
                    saved["path"],
                    trace_dir=root / "restored_traces",
                )
                try:
                    after_status = restored.status()
                    after = after_status["concept_store"]
                    metadata = after_status["checkpoint_metadata"]

                    self.assertEqual(int(after["concept_count"]), int(before["concept_count"]))
                    self.assertEqual(int(after["observations"]), int(before["observations"]))
                    self.assertEqual(
                        sorted(entry["concept_id"] for entry in after.get("top_concepts", [])),
                        sorted(entry["concept_id"] for entry in before.get("top_concepts", [])),
                    )
                    self.assertEqual(
                        metadata["service_state"]["concept_store"]["concept_mode"],
                        "slow_feature_concept_memory",
                    )
                finally:
                    restored.close()
            finally:
                manager.close()


class ServiceManagerTerminusRuntimeTests(unittest.TestCase):
    def test_terminus_tick_trains_from_configured_file_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="service_manager_terminus_tick")
            source_path = root / "terminus_source.txt"
            source_path.write_text("adaptive memory plasticity signal " * 32, encoding="utf-8")
            try:
                configured = manager.configure_terminus(
                    source_bank=[
                        {
                            "name": "local_terminus_source",
                            "source": str(source_path),
                            "source_type": "file",
                        }
                    ],
                    tick_tokens=24,
                    sleep_interval_seconds=0.01,
                    repeat_sources=False,
                )
                before_tokens = configured["token_count"]
                ticked = manager.terminus_tick(steps=2)
                runtime = ticked["terminus_runtime"]

                self.assertTrue(runtime["configured"])
                self.assertFalse(runtime["running"])
                self.assertGreater(ticked["token_count"], before_tokens)
                self.assertGreater(runtime["background_tokens_processed"], 0)
                self.assertEqual(runtime["source_count"], 1)
                self.assertEqual(runtime["exhausted_source_count"], 0)
                self.assertEqual(runtime["next_source_name"], "local_terminus_source")
                self.assertIsNotNone(runtime["last_tick_completed_at"])
                self.assertGreater(float(runtime["last_tick_duration_ms"]), 0.0)
                self.assertGreater(int(runtime["last_tick_token_delta"]), 0)
                self.assertTrue(any(event.get("type") == "tick" for event in runtime["recent_events"]))
                self.assertEqual(runtime["last_event"]["type"], "tick")
                self.assertEqual(runtime["last_event"]["source"]["source_name"], "local_terminus_source")
                self.assertEqual(runtime["source_progress"][0]["name"], "local_terminus_source")
                self.assertGreater(runtime["source_progress"][0]["tokens_processed"], 0)
                self.assertGreater(runtime["source_progress"][0]["tick_visits"], 0)
                self.assertGreater(runtime["source_progress"][0]["last_tokens_trained"], 0)
                self.assertIsNotNone(runtime["source_progress"][0]["last_activity_at"])
                self.assertAlmostEqual(runtime["source_progress"][0]["share_of_background_tokens"], 1.0, places=6)
                concept_store = manager.status()["concept_store"]
                self.assertGreater(int(concept_store["concept_count"]), 0)
                self.assertGreater(int(concept_store["observations"]), 0)
                top_terms = {
                    term
                    for concept in concept_store["top_concepts"]
                    for term in concept.get("top_terms", [])
                }
                self.assertIn("plasticity", top_terms)
            finally:
                manager.close()

    def test_terminus_runtime_reports_autonomy_trigger_progress(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="service_manager_terminus_observability")
            source_path = root / "terminus_source.txt"
            source_path.write_text("active source seeking telemetry " * 24, encoding="utf-8")
            try:
                manager.configure_terminus(
                    source_bank=[
                        {
                            "name": "observed_source",
                            "source": str(source_path),
                            "source_type": "file",
                        }
                    ],
                    tick_tokens=12,
                    sleep_interval_seconds=0.01,
                    repeat_sources=True,
                    autonomy={
                        "enabled": True,
                        "policy": "active",
                        "candidate_bank": [
                            {
                                "name": "candidate_source",
                                "source": str(source_path),
                                "source_type": "file",
                            }
                        ],
                        "trigger_interval_tokens": 50,
                    },
                )
                manager.terminus_tick(steps=2)
                runtime = manager.terminus_status()["terminus_runtime"]

                self.assertEqual(runtime["autonomy"]["candidate_count"], 1)
                self.assertEqual(runtime["autonomy"]["candidate_bank"][0]["name"], "candidate_source")
                self.assertEqual(runtime["autonomy"]["candidate_names"], ["candidate_source"])
                self.assertFalse(runtime["autonomy"]["trigger_ready"])
                self.assertLess(runtime["autonomy"]["tokens_until_trigger"], 50)
                self.assertIsNone(runtime["autonomy"]["last_acquisition_summary"])
            finally:
                manager.close()

    def test_terminus_runtime_persists_recent_query_gap_focus(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="service_manager_recent_query_gap_focus")
            source_path = root / "terminus_source.txt"
            source_path.write_text("neutral background signal " * 24, encoding="utf-8")
            try:
                manager.configure_terminus(
                    source_bank=[
                        {
                            "name": "observed_source",
                            "source": str(source_path),
                            "source_type": "file",
                        }
                    ],
                    tick_tokens=12,
                    sleep_interval_seconds=0.01,
                    repeat_sources=True,
                    autonomy={
                        "enabled": True,
                        "policy": "active",
                        "candidate_bank": [
                            {
                                "name": "candidate_source",
                                "source": str(source_path),
                                "source_type": "file",
                            }
                        ],
                        "trigger_interval_tokens": 50,
                    },
                )
                manager.query(query_text="submarine buoyancy ballast", top_k_memories=6)
                runtime = manager.terminus_status()["terminus_runtime"]

                self.assertEqual(runtime["autonomy"]["recent_query_gaps"][0]["query_text"], "submarine buoyancy ballast")
                self.assertIn("submarine", runtime["autonomy"]["focus_plan"]["unsupported_terms"])

                saved = manager.save_checkpoint(str(root / "terminus_focus.pt"))
                restored = HECSNServiceManager(saved["path"], trace_dir=root / "restored_traces")
                try:
                    restored_runtime = restored.terminus_status()["terminus_runtime"]

                    self.assertEqual(
                        restored_runtime["autonomy"]["recent_query_gaps"][0]["query_text"],
                        "submarine buoyancy ballast",
                    )
                    self.assertIn("submarine", restored_runtime["autonomy"]["focus_plan"]["unsupported_terms"])
                finally:
                    restored.close()
            finally:
                manager.close()

    def test_terminus_autonomy_passes_recent_query_focus_into_acquisition(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="service_manager_autonomy_query_focus_bridge")
            source_path = root / "terminus_source.txt"
            related_path = root / "submarine_source.txt"
            unrelated_path = root / "garden_source.txt"
            source_path.write_text("neutral background signal " * 24, encoding="utf-8")
            related_path.write_text("submarine buoyancy ballast pressure " * 24, encoding="utf-8")
            unrelated_path.write_text("garden tomato soil sunlight " * 24, encoding="utf-8")
            try:
                manager.configure_terminus(
                    source_bank=[
                        {
                            "name": "observed_source",
                            "source": str(source_path),
                            "source_type": "file",
                        }
                    ],
                    tick_tokens=12,
                    sleep_interval_seconds=0.01,
                    repeat_sources=True,
                    autonomy={
                        "enabled": True,
                        "policy": "active",
                        "candidate_bank": [
                            {
                                "name": "submarine_source",
                                "source": str(related_path),
                                "source_type": "file",
                            },
                            {
                                "name": "garden_source",
                                "source": str(unrelated_path),
                                "source_type": "file",
                            },
                        ],
                        "trigger_interval_tokens": 1,
                        "semantic_shortlist_size": 1,
                        "semantic_shortlist_gap_weight": 0.0,
                        "semantic_shortlist_affinity_weight": 1.0,
                    },
                )
                manager.query(query_text="submarine buoyancy ballast", top_k_memories=6)
                with patch(
                    "hecsn.service.manager.run_live_acquisition",
                    return_value={
                        "policy": "active",
                        "tokens_trained_total": 0,
                        "acquired_sources": [],
                        "semantic_plan": {
                            "unsupported_terms": ["submarine", "buoyancy", "ballast"],
                        },
                    },
                ) as mocked_acquire:
                    manager.terminus_tick()

                kwargs = mocked_acquire.call_args.kwargs
                self.assertIn("submarine", kwargs["semantic_plan"]["unsupported_terms"])
                self.assertIn("submarine", kwargs["candidate_bank_specs"][0]["metadata"]["query_text"].lower())
                self.assertIn("ballast", kwargs["candidate_bank_specs"][1]["metadata"]["query_text"].lower())
            finally:
                manager.close()

    def test_terminus_autonomy_auto_enables_focus_shortlist_for_broader_candidate_bank(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="service_manager_focus_shortlist_auto")
            source_path = root / "terminus_source.txt"
            submarine_path = root / "submarine_source.txt"
            garden_path = root / "garden_source.txt"
            astronomy_path = root / "astronomy_source.txt"
            cooking_path = root / "cooking_source.txt"
            source_path.write_text("neutral background signal " * 24, encoding="utf-8")
            submarine_path.write_text("submarine buoyancy ballast pressure " * 24, encoding="utf-8")
            garden_path.write_text("garden tomato soil sunlight " * 24, encoding="utf-8")
            astronomy_path.write_text("planet orbit telescope observatory " * 24, encoding="utf-8")
            cooking_path.write_text("kitchen simmer recipe broth " * 24, encoding="utf-8")
            try:
                manager.configure_terminus(
                    source_bank=[
                        {
                            "name": "observed_source",
                            "source": str(source_path),
                            "source_type": "file",
                        }
                    ],
                    tick_tokens=12,
                    sleep_interval_seconds=0.01,
                    repeat_sources=True,
                    autonomy={
                        "enabled": True,
                        "policy": "active",
                        "candidate_bank": [
                            {
                                "name": "submarine_source",
                                "source": str(submarine_path),
                                "source_type": "file",
                            },
                            {
                                "name": "garden_source",
                                "source": str(garden_path),
                                "source_type": "file",
                            },
                            {
                                "name": "astronomy_source",
                                "source": str(astronomy_path),
                                "source_type": "file",
                            },
                            {
                                "name": "cooking_source",
                                "source": str(cooking_path),
                                "source_type": "file",
                            },
                        ],
                        "trigger_interval_tokens": 1,
                    },
                )
                manager.query(query_text="submarine buoyancy ballast", top_k_memories=6)
                with patch(
                    "hecsn.service.manager.run_live_acquisition",
                    return_value={
                        "policy": "active",
                        "tokens_trained_total": 0,
                        "acquired_sources": [],
                        "semantic_plan": {
                            "unsupported_terms": ["submarine", "buoyancy", "ballast"],
                        },
                    },
                ) as mocked_acquire:
                    manager.terminus_tick()

                kwargs = mocked_acquire.call_args.kwargs
                self.assertEqual(kwargs["semantic_shortlist_size"], 2)
                self.assertAlmostEqual(kwargs["semantic_shortlist_gap_weight"], 0.2)
                self.assertAlmostEqual(kwargs["semantic_shortlist_affinity_weight"], 0.8)
            finally:
                manager.close()

    def test_terminus_autonomy_preserves_registry_candidate_bank_and_shortlists_estimated_pool(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="service_manager_catalog_candidate_bank")
            source_path = root / "terminus_source.txt"
            source_path.write_text("neutral background signal " * 24, encoding="utf-8")
            try:
                manager.configure_terminus(
                    source_bank=[
                        {
                            "name": "observed_source",
                            "source": str(source_path),
                            "source_type": "file",
                        }
                    ],
                    tick_tokens=12,
                    sleep_interval_seconds=0.01,
                    repeat_sources=True,
                    autonomy={
                        "enabled": True,
                        "policy": "active",
                        "candidate_bank": [
                            {
                                "name": "registry_pool",
                                "catalog_mode": "semantic_registry",
                                "catalog_limit": 4,
                                "catalog_entries": [
                                    {
                                        "name": "submarine_source",
                                        "source": "https://example.test/submarine",
                                        "source_type": "web",
                                        "summary": "submarine buoyancy ballast pressure",
                                    },
                                    {
                                        "name": "garden_source",
                                        "source": "https://example.test/garden",
                                        "source_type": "web",
                                        "summary": "garden tomato soil sunlight",
                                    },
                                    {
                                        "name": "astronomy_source",
                                        "source": "https://example.test/astronomy",
                                        "source_type": "web",
                                        "summary": "planet orbit telescope observatory",
                                    },
                                    {
                                        "name": "cooking_source",
                                        "source": "https://example.test/cooking",
                                        "source_type": "web",
                                        "summary": "kitchen simmer recipe broth",
                                    },
                                ],
                            }
                        ],
                        "trigger_interval_tokens": 1,
                    },
                )
                manager.query(query_text="submarine buoyancy ballast", top_k_memories=6)
                with patch(
                    "hecsn.service.manager.run_live_acquisition",
                    return_value={
                        "policy": "active",
                        "tokens_trained_total": 0,
                        "acquired_sources": [],
                        "semantic_plan": {
                            "unsupported_terms": ["submarine", "buoyancy", "ballast"],
                        },
                    },
                ) as mocked_acquire:
                    manager.terminus_tick()

                runtime = manager.terminus_status()["terminus_runtime"]
                kwargs = mocked_acquire.call_args.kwargs
                self.assertEqual(runtime["autonomy"]["candidate_bank"][0]["catalog_mode"], "semantic_registry")
                self.assertEqual(len(runtime["autonomy"]["candidate_bank"][0]["catalog_entries"]), 4)
                self.assertEqual(kwargs["candidate_bank_specs"][0]["catalog_mode"], "semantic_registry")
                self.assertIn("submarine", kwargs["candidate_bank_specs"][0]["catalog_focus_text"].lower())
                self.assertEqual(kwargs["semantic_shortlist_size"], 2)
                self.assertAlmostEqual(kwargs["semantic_shortlist_gap_weight"], 0.2)
                self.assertAlmostEqual(kwargs["semantic_shortlist_affinity_weight"], 0.8)
            finally:
                manager.close()

    def test_terminus_autonomy_preserves_live_remote_search_candidate_bank_and_shortlists_estimated_pool(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="service_manager_live_remote_candidate_bank")
            source_path = root / "terminus_source.txt"
            source_path.write_text("neutral background signal " * 24, encoding="utf-8")
            try:
                manager.configure_terminus(
                    source_bank=[
                        {
                            "name": "observed_source",
                            "source": str(source_path),
                            "source_type": "file",
                        }
                    ],
                    tick_tokens=12,
                    sleep_interval_seconds=0.01,
                    repeat_sources=True,
                    autonomy={
                        "enabled": True,
                        "policy": "active",
                        "candidate_bank": [
                            {
                                "name": "live_remote_pool",
                                "catalog_mode": "live_remote_search",
                                "catalog_providers": ["wikipedia", "arxiv"],
                                "catalog_queries_per_provider": 2,
                                "catalog_provider_result_limit": 4,
                                "catalog_limit": 4,
                            }
                        ],
                        "trigger_interval_tokens": 1,
                    },
                )
                manager.query(query_text="submarine buoyancy ballast", top_k_memories=6)
                with patch(
                    "hecsn.service.manager.run_live_acquisition",
                    return_value={
                        "policy": "active",
                        "tokens_trained_total": 0,
                        "acquired_sources": [],
                        "semantic_plan": {
                            "unsupported_terms": ["submarine", "buoyancy", "ballast"],
                        },
                    },
                ) as mocked_acquire:
                    manager.terminus_tick()

                runtime = manager.terminus_status()["terminus_runtime"]
                kwargs = mocked_acquire.call_args.kwargs
                self.assertEqual(runtime["autonomy"]["candidate_bank"][0]["catalog_mode"], "live_remote_search")
                self.assertEqual(runtime["autonomy"]["candidate_bank"][0]["catalog_providers"], ["wikipedia", "arxiv"])
                self.assertEqual(kwargs["candidate_bank_specs"][0]["catalog_mode"], "live_remote_search")
                self.assertEqual(kwargs["candidate_bank_specs"][0]["catalog_providers"], ["wikipedia", "arxiv"])
                self.assertEqual(kwargs["candidate_bank_specs"][0]["catalog_queries_per_provider"], 2)
                self.assertEqual(kwargs["candidate_bank_specs"][0]["catalog_provider_result_limit"], 4)
                self.assertIn("submarine", kwargs["candidate_bank_specs"][0]["catalog_focus_text"].lower())
                self.assertEqual(kwargs["semantic_shortlist_size"], 3)
                self.assertAlmostEqual(kwargs["semantic_shortlist_gap_weight"], 0.2)
                self.assertAlmostEqual(kwargs["semantic_shortlist_affinity_weight"], 0.8)
            finally:
                manager.close()

    def test_terminus_autonomy_defaults_to_live_remote_search_candidate_bank_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="service_manager_default_live_remote_candidate_bank")
            source_path = root / "terminus_source.txt"
            source_path.write_text("neutral background signal " * 24, encoding="utf-8")
            try:
                manager.configure_terminus(
                    source_bank=[
                        {
                            "name": "observed_source",
                            "source": str(source_path),
                            "source_type": "file",
                        }
                    ],
                    tick_tokens=12,
                    sleep_interval_seconds=0.01,
                    repeat_sources=True,
                    autonomy={
                        "enabled": True,
                        "policy": "active",
                        "trigger_interval_tokens": 1,
                    },
                )
                manager.query(query_text="submarine buoyancy ballast", top_k_memories=6)
                with patch(
                    "hecsn.service.manager.run_live_acquisition",
                    return_value={
                        "policy": "active",
                        "tokens_trained_total": 0,
                        "acquired_sources": [],
                        "semantic_plan": {
                            "unsupported_terms": ["submarine", "buoyancy", "ballast"],
                        },
                    },
                ) as mocked_acquire:
                    manager.terminus_tick()

                runtime = manager.terminus_status()["terminus_runtime"]
                kwargs = mocked_acquire.call_args.kwargs
                self.assertEqual(runtime["autonomy"]["candidate_count"], 1)
                self.assertEqual(runtime["autonomy"]["candidate_bank"][0]["catalog_mode"], "live_remote_search")
                self.assertEqual(runtime["autonomy"]["candidate_bank"][0]["catalog_providers"], ["wikipedia", "arxiv"])
                self.assertEqual(kwargs["candidate_bank_specs"][0]["catalog_mode"], "live_remote_search")
                self.assertEqual(kwargs["candidate_bank_specs"][0]["catalog_providers"], ["wikipedia", "arxiv"])
                self.assertEqual(kwargs["candidate_bank_specs"][0]["catalog_queries_per_provider"], 1)
                self.assertEqual(kwargs["candidate_bank_specs"][0]["catalog_provider_result_limit"], 4)
                self.assertIn("submarine", kwargs["candidate_bank_specs"][0]["catalog_focus_text"].lower())
                self.assertEqual(kwargs["semantic_shortlist_size"], 1)
                self.assertAlmostEqual(kwargs["semantic_shortlist_gap_weight"], 0.0)
                self.assertAlmostEqual(kwargs["semantic_shortlist_affinity_weight"], 1.0)
            finally:
                manager.close()

    def test_save_restore_round_trips_terminus_runtime_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="service_manager_terminus_checkpoint")
            source_path = root / "terminus_source.txt"
            source_path.write_text("hebbian memory consolidation " * 24, encoding="utf-8")
            try:
                manager.configure_terminus(
                    source_bank=[
                        {
                            "name": "checkpoint_source",
                            "source": str(source_path),
                            "source_type": "file",
                        }
                    ],
                    tick_tokens=12,
                    sleep_interval_seconds=0.01,
                    repeat_sources=True,
                    autonomy={
                        "enabled": True,
                        "policy": "active",
                        "candidate_bank": [
                            {
                                "name": "candidate_memory",
                                "source": str(source_path),
                                "source_type": "file",
                            }
                        ],
                        "trigger_interval_tokens": 100,
                    },
                )
                saved = manager.save_checkpoint(str(root / "terminus_service.pt"))
                restored = HECSNServiceManager(saved["path"], trace_dir=root / "restored_traces")
                try:
                    terminus_runtime = restored.status()["terminus_runtime"]

                    self.assertTrue(terminus_runtime["configured"])
                    self.assertEqual(terminus_runtime["source_bank"][0]["name"], "checkpoint_source")
                    self.assertEqual(terminus_runtime["autonomy"]["candidate_count"], 1)
                finally:
                    restored.close()
            finally:
                manager.close()

    def test_save_restore_round_trips_catalog_candidate_bank_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="service_manager_catalog_checkpoint")
            source_path = root / "terminus_source.txt"
            source_path.write_text("hebbian memory consolidation " * 24, encoding="utf-8")
            try:
                manager.configure_terminus(
                    source_bank=[
                        {
                            "name": "checkpoint_source",
                            "source": str(source_path),
                            "source_type": "file",
                        }
                    ],
                    tick_tokens=12,
                    sleep_interval_seconds=0.01,
                    repeat_sources=True,
                    autonomy={
                        "enabled": True,
                        "policy": "active",
                        "candidate_bank": [
                            {
                                "name": "registry_pool",
                                "catalog_mode": "semantic_registry",
                                "catalog_limit": 2,
                                "catalog_entries": [
                                    {
                                        "name": "submarine_source",
                                        "source": "https://example.test/submarine",
                                        "source_type": "web",
                                        "summary": "submarine buoyancy ballast pressure",
                                    },
                                    {
                                        "name": "garden_source",
                                        "source": "https://example.test/garden",
                                        "source_type": "web",
                                        "summary": "garden tomato soil sunlight",
                                    },
                                ],
                            }
                        ],
                        "trigger_interval_tokens": 100,
                    },
                )
                saved = manager.save_checkpoint(str(root / "terminus_catalog_service.pt"))
                restored = HECSNServiceManager(saved["path"], trace_dir=root / "restored_traces")
                try:
                    terminus_runtime = restored.status()["terminus_runtime"]

                    self.assertTrue(terminus_runtime["configured"])
                    self.assertEqual(terminus_runtime["source_bank"][0]["name"], "checkpoint_source")
                    self.assertEqual(terminus_runtime["autonomy"]["candidate_bank"][0]["catalog_mode"], "semantic_registry")
                    self.assertEqual(len(terminus_runtime["autonomy"]["candidate_bank"][0]["catalog_entries"]), 2)
                finally:
                    restored.close()
            finally:
                manager.close()

    def test_start_and_stop_terminus_loop_updates_running_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="service_manager_terminus_loop")
            source_path = root / "terminus_source.txt"
            source_path.write_text("unsupervised knowledge accumulation " * 64, encoding="utf-8")
            try:
                manager.configure_terminus(
                    source_bank=[
                        {
                            "name": "loop_source",
                            "source": str(source_path),
                            "source_type": "file",
                        }
                    ],
                    tick_tokens=16,
                    sleep_interval_seconds=0.01,
                    repeat_sources=True,
                )

                started = manager.start_terminus()
                self.assertTrue(started["terminus_runtime"]["running"])
                self.assertIsNotNone(started["terminus_runtime"]["running_since"])
                time.sleep(0.05)
                stopped = manager.stop_terminus()

                self.assertFalse(stopped["terminus_runtime"]["running"])
                self.assertIsNone(stopped["terminus_runtime"]["running_since"])
                self.assertGreaterEqual(stopped["terminus_runtime"]["background_tokens_processed"], 16)
                self.assertEqual(stopped["terminus_runtime"]["recent_events"][0]["type"], "stopped")
            finally:
                manager.close()


if __name__ == "__main__":
    unittest.main()
