from __future__ import annotations

from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from hecsn.config.model_config import HECSNConfig
from hecsn.service.manager import HECSNServiceManager
from hecsn.training.checkpointing import save_trainer_checkpoint
from hecsn.training.trainer import HECSNModel, HECSNTrainer


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
    model = HECSNModel(cfg)
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

                before_status = manager.status()
                before = before_status["concept_store"]
                before_serotonin = float(before_status["serotonin"])
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
                    self.assertIn("serotonin", after_status)
                    self.assertAlmostEqual(float(after_status["serotonin"]), before_serotonin, places=6)
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

    def test_terminus_focus_plan_preserves_recent_weak_concepts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="service_manager_recent_weak_concepts")
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
                with manager._lock:
                    manager._record_recent_query_gap_locked(
                        query_text="submarine buoyancy ballast",
                        source="query",
                        gap_plan={
                            "unsupported_terms": ["submarine"],
                            "gap_terms": [{"term": "submarine", "weight": 2.0}],
                            "retrieval_queries": ["submarine buoyancy ballast"],
                            "follow_up_questions": ["What grounded evidence explains submarine ballast control?"],
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
                            "grounded_fraction": 0.0,
                        },
                    )
                runtime = manager.terminus_status()["terminus_runtime"]

                self.assertEqual(
                    runtime["autonomy"]["recent_query_gaps"][0]["weak_concepts"][0]["label"],
                    "buoyancy control",
                )
                self.assertEqual(
                    runtime["autonomy"]["focus_plan"]["weak_concepts"][0]["label"],
                    "buoyancy control",
                )
                self.assertEqual(
                    runtime["autonomy"]["focus_plan"]["weak_concepts"][0]["top_terms"],
                    ["submarine", "ballast", "buoyancy"],
                )
            finally:
                manager.close()

    def test_fully_grounded_query_does_not_persist_recent_gap_focus(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="service_manager_fully_grounded_query_gap")
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
                        "trigger_interval_tokens": 50,
                    },
                )
                with manager._lock:
                    manager._record_recent_query_gap_locked(
                        query_text="what corrects submarine trim",
                        source="query",
                        gap_plan={
                            "unsupported_terms": [],
                            "gap_terms": [{"term": "submarine", "weight": 1.0}],
                            "retrieval_queries": ["submarine ballast trim"],
                            "follow_up_questions": ["What grounded evidence would reduce drift for buoyancy control?"],
                            "weak_concepts": [
                                {
                                    "label": "buoyancy control",
                                    "weakness": 0.7,
                                    "uncertainty": 0.4,
                                    "drift": 0.3,
                                    "top_terms": ["submarine", "ballast", "trim"],
                                    "match_count": 2,
                                }
                            ],
                            "grounded_fraction": 1.0,
                        },
                    )
                runtime = manager.terminus_status()["terminus_runtime"]

                self.assertEqual(runtime["autonomy"]["recent_query_gaps"], [])
                self.assertIsNone(runtime["autonomy"]["focus_plan"])
            finally:
                manager.close()

    def test_fully_grounded_query_skips_next_autonomy_cycle(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="service_manager_grounded_query_autonomy_skip")
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
                with manager._lock:
                    manager._record_recent_query_gap_locked(
                        query_text="what corrects submarine trim",
                        source="query",
                        gap_plan={
                            "unsupported_terms": [],
                            "gap_terms": [{"term": "submarine", "weight": 1.0}],
                            "retrieval_queries": ["submarine ballast trim"],
                            "follow_up_questions": ["What grounded evidence would reduce drift for buoyancy control?"],
                            "weak_concepts": [
                                {
                                    "label": "buoyancy control",
                                    "weakness": 0.7,
                                    "uncertainty": 0.4,
                                    "drift": 0.3,
                                    "top_terms": ["submarine", "ballast", "trim"],
                                    "match_count": 2,
                                }
                            ],
                            "grounded_fraction": 1.0,
                        },
                    )
                with patch("hecsn.service.manager.run_live_acquisition") as mocked_acquire:
                    manager.terminus_tick()

                runtime = manager.terminus_status()["terminus_runtime"]
                mocked_acquire.assert_not_called()
                self.assertIsNone(runtime["autonomy"]["last_acquisition_summary"])
            finally:
                manager.close()

    def test_terminus_autonomy_uses_concept_store_focus_without_recent_query_gap(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="service_manager_concept_store_focus")
            source_path = root / "terminus_source.txt"
            candidate_path = root / "candidate_source.txt"
            source_path.write_text("neutral background signal " * 24, encoding="utf-8")
            candidate_path.write_text("submarine ballast buoyancy pressure " * 24, encoding="utf-8")
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
                                "source": str(candidate_path),
                                "source_type": "file",
                            }
                        ],
                        "trigger_interval_tokens": 1,
                    },
                )
                manager.feed(text=("submarine ballast buoyancy pressure control " * 16).strip())
                concept_store = manager.status()["concept_store"]
                runtime = manager.terminus_status()["terminus_runtime"]

                self.assertGreater(concept_store["growth"]["expansion_events"], 0)
                self.assertGreater(concept_store["abstraction"]["requested_output_dim"], 8)
                self.assertEqual(runtime["autonomy"]["recent_query_gaps"], [])
                self.assertIsNotNone(runtime["autonomy"]["focus_plan"])
                self.assertEqual(runtime["autonomy"]["focus_plan"]["planner_mode"], "concept_store_abstraction_focus")
                self.assertIn("submarine", " ".join(runtime["autonomy"]["focus_plan"]["retrieval_queries"]).lower())
                self.assertGreater(
                    runtime["autonomy"]["focus_plan"]["structural_growth"]["expansion_events"],
                    0,
                )

                with patch(
                    "hecsn.service.manager.run_live_acquisition",
                    return_value={
                        "policy": "active",
                        "tokens_trained_total": 0,
                        "acquired_sources": [],
                        "semantic_plan": runtime["autonomy"]["focus_plan"],
                    },
                ) as mocked_acquire:
                    manager.terminus_tick()

                kwargs = mocked_acquire.call_args.kwargs
                self.assertEqual(kwargs["semantic_plan"]["planner_mode"], "concept_store_abstraction_focus")
                self.assertIn("submarine", " ".join(kwargs["semantic_plan"]["retrieval_queries"]).lower())
                self.assertIn("submarine", kwargs["candidate_bank_specs"][0]["metadata"]["query_text"].lower())
            finally:
                manager.close()

    def test_terminus_autonomy_surfaces_geometric_curiosity_focus_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
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
                enable_abstraction_layer=True,
            )
            trainer = HECSNTrainer(HECSNModel(cfg), cfg)
            checkpoint_path = save_trainer_checkpoint(
                root / "initial_abstraction.pt",
                trainer,
                metadata={"test_case": "service_manager_geometric_curiosity_focus"},
            )
            manager = HECSNServiceManager(
                checkpoint_path,
                trace_dir=root / "traces",
            )
            source_path = root / "terminus_source.txt"
            candidate_path = root / "candidate_source.txt"
            source_path.write_text("neutral background signal " * 24, encoding="utf-8")
            candidate_path.write_text("river stream water current bank " * 24, encoding="utf-8")
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
                                "source": str(candidate_path),
                                "source_type": "file",
                            }
                        ],
                        "trigger_interval_tokens": 1,
                    },
                )
                manager.feed(text=("river stream water current bank " * 12).strip())
                runtime = manager.terminus_status()["terminus_runtime"]
                autonomy = runtime["autonomy"]

                self.assertTrue(autonomy["geometric_curiosity"]["enabled"])
                self.assertTrue(autonomy["geometric_curiosity"]["has_focus_plan"])
                self.assertIsNotNone(autonomy["focus_plan"])
                self.assertIn("geometric_gaps", autonomy["focus_plan"])
                self.assertTrue(autonomy["focus_plan"]["retrieval_queries"])
            finally:
                manager.close()

    def test_terminus_live_remote_search_learns_geometric_query_families(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
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
                enable_abstraction_layer=True,
            )
            trainer = HECSNTrainer(HECSNModel(cfg), cfg)
            checkpoint_path = save_trainer_checkpoint(
                root / "initial_geometric_curriculum.pt",
                trainer,
                metadata={"test_case": "service_manager_geometric_query_families"},
            )
            manager = HECSNServiceManager(
                checkpoint_path,
                trace_dir=root / "traces",
            )
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
                                "catalog_providers": ["wikipedia", "openalex"],
                                "catalog_queries_per_provider": 2,
                                "catalog_provider_result_limit": 4,
                                "catalog_limit": 4,
                            }
                        ],
                        "trigger_interval_tokens": 1,
                    },
                )
                manager.feed(text=("river stream water current bank loan credit " * 12).strip())
                focus_plan = manager.terminus_status()["terminus_runtime"]["autonomy"]["focus_plan"]
                self.assertTrue(focus_plan["geometric_gaps"])
                selected_query = str(focus_plan["retrieval_queries"][0]).strip().lower()

                with patch(
                    "hecsn.service.manager.run_live_acquisition",
                    side_effect=[
                        {
                            "policy": "active",
                            "tokens_trained_total": 32,
                            "acquired_sources": ["wikipedia_gap_source"],
                            "semantic_plan": focus_plan,
                            "acquisition_history": [
                                {
                                    "selected_source": "wikipedia_gap_source",
                                    "selected_provider": "wikipedia",
                                    "selected_query_text": selected_query,
                                    "selected_semantic_relevance": 0.91,
                                    "selected_gap_reduction": 0.22,
                                    "selected_diagnostic_gap_reduction": 0.31,
                                    "tokens_trained": 32,
                                    "selected_metadata": {
                                        "provider": "wikipedia",
                                        "query_text": selected_query,
                                        "semantic_relevance": 0.91,
                                        "catalog_terms": ["river current", "bank finance"],
                                    },
                                    "candidate_snapshot": {
                                        "wikipedia_gap_source": {
                                            "semantic_answerability": 0.22,
                                            "concept_uncertainty": 0.72,
                                            "concept_support": 0.18,
                                            "semantic_weak_concept_pressure": 0.76,
                                        }
                                    },
                                    "selected_semantic_answerability_after": 0.64,
                                    "selected_concept_uncertainty_after": 0.28,
                                    "selected_concept_support_after": 0.58,
                                    "selected_weak_concept_pressure_after": 0.18,
                                }
                            ],
                        },
                        {
                            "policy": "active",
                            "tokens_trained_total": 0,
                            "acquired_sources": [],
                            "semantic_plan": focus_plan,
                            "acquisition_history": [],
                        },
                    ],
                ) as mocked_acquire:
                    manager.terminus_tick()
                    manager.terminus_tick()

                first_kwargs = mocked_acquire.call_args_list[0].kwargs
                second_kwargs = mocked_acquire.call_args_list[1].kwargs
                spec = second_kwargs["candidate_bank_specs"][0]
                runtime = manager.terminus_status()["terminus_runtime"]
                provider_curriculum = runtime["autonomy"]["provider_curriculum"]

                self.assertEqual(spec["catalog_providers"][0], "wikipedia")
                self.assertGreaterEqual(
                    int(spec["catalog_queries_per_provider"]),
                    int(first_kwargs["candidate_bank_specs"][0]["catalog_queries_per_provider"]),
                )
                self.assertEqual(int(spec["catalog_query_family_budget_bonus"]), 1)
                self.assertIn("catalog_provider_query_families", spec)
                self.assertIn(selected_query, spec["catalog_provider_query_families"]["wikipedia"])
                self.assertEqual(provider_curriculum["ranked_providers"][0]["provider"], "wikipedia")
                self.assertGreater(
                    float(provider_curriculum["ranked_providers"][0]["query_family_strength"]),
                    0.0,
                )
                self.assertGreater(
                    float(provider_curriculum["ranked_providers"][0]["query_family_focus_score"]),
                    0.0,
                )
                self.assertEqual(
                    int(provider_curriculum["ranked_providers"][0]["query_family_query_bonus"]),
                    1,
                )
                self.assertIn(
                    selected_query,
                    provider_curriculum["ranked_providers"][0]["matched_query_families"],
                )
                self.assertEqual(
                    int(provider_curriculum["ranked_providers"][0]["query_families"][selected_query]["commits"]),
                    1,
                )
            finally:
                manager.close()

    def test_query_passes_query_conditioned_concept_focus_into_retrieval(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="service_manager_query_conditioned_abstraction")
            try:
                manager.feed(text=("submarine ballast buoyancy pressure control " * 16).strip())
                captured: dict[str, object] = {}

                def _fake_build_query_result(**kwargs: object) -> dict[str, object]:
                    captured.update(kwargs)
                    return {
                        "checkpoint": "test://service-manager",
                        "checkpoint_metadata": {},
                        "config": {},
                        "feed_summary": None,
                        "context_summary": None,
                        "context_comparison": None,
                        "query_summary": {
                            "query_text": "submarine control depth",
                            "memory_matches": [],
                            "memory_episodes": [],
                            "native_decode": {"available": False},
                        },
                    }

                with patch("hecsn.service.manager.build_query_result", side_effect=_fake_build_query_result):
                    result = manager.query(query_text="submarine control depth", top_k_memories=4)

                self.assertIn("ballast", " ".join(captured["retrieval_focus_terms"]).lower())
                self.assertTrue(captured["memory_priority"])
                self.assertEqual(
                    result["query_summary"]["abstraction_focus"]["planner_mode"],
                    "concept_store_abstraction_focus",
                )
                self.assertIn(
                    "submarine",
                    " ".join(result["query_summary"]["abstraction_focus"]["retrieval_queries"]).lower(),
                )
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
                self.assertEqual(
                    runtime["autonomy"]["candidate_bank"][0]["catalog_providers"],
                    ["wikipedia", "arxiv"],
                )
                self.assertEqual(kwargs["candidate_bank_specs"][0]["catalog_mode"], "live_remote_search")
                self.assertEqual(
                    kwargs["candidate_bank_specs"][0]["catalog_providers"],
                    ["wikipedia", "arxiv"],
                )
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
                self.assertEqual(
                    runtime["autonomy"]["candidate_bank"][0]["catalog_providers"],
                    ["wikipedia", "arxiv", "openalex"],
                )
                self.assertEqual(kwargs["candidate_bank_specs"][0]["catalog_mode"], "live_remote_search")
                self.assertEqual(
                    kwargs["candidate_bank_specs"][0]["catalog_providers"],
                    ["wikipedia", "arxiv", "openalex"],
                )
                self.assertEqual(kwargs["candidate_bank_specs"][0]["catalog_queries_per_provider"], 2)
                self.assertEqual(kwargs["candidate_bank_specs"][0]["catalog_provider_result_limit"], 4)
                self.assertIn("submarine", kwargs["candidate_bank_specs"][0]["catalog_focus_text"].lower())
                self.assertEqual(kwargs["semantic_shortlist_size"], 1)
                self.assertAlmostEqual(kwargs["semantic_shortlist_gap_weight"], 0.0)
                self.assertAlmostEqual(kwargs["semantic_shortlist_affinity_weight"], 1.0)
            finally:
                manager.close()

    def test_terminus_default_live_remote_search_grows_query_budget_for_multiple_weak_concepts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="service_manager_default_live_remote_query_growth")
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
                with manager._lock:
                    manager._record_recent_query_gap_locked(
                        query_text="submarine buoyancy ballast",
                        source="query",
                        gap_plan={
                            "unsupported_terms": ["submarine"],
                            "gap_terms": [{"term": "submarine", "weight": 2.0}],
                            "retrieval_queries": [],
                            "follow_up_questions": [],
                            "weak_concepts": [
                                {
                                    "label": "garden soil",
                                    "weakness": 0.9,
                                    "uncertainty": 0.5,
                                    "drift": 0.1,
                                    "top_terms": ["garden", "tomato", "soil"],
                                    "match_count": 1,
                                },
                                {
                                    "label": "buoyancy control",
                                    "weakness": 0.7,
                                    "uncertainty": 0.6,
                                    "drift": 0.2,
                                    "top_terms": ["submarine", "ballast", "buoyancy"],
                                    "match_count": 1,
                                },
                            ],
                            "grounded_fraction": 0.0,
                        },
                    )
                with patch(
                    "hecsn.service.manager.run_live_acquisition",
                    return_value={
                        "policy": "active",
                        "tokens_trained_total": 0,
                        "acquired_sources": [],
                        "semantic_plan": {
                            "unsupported_terms": ["submarine"],
                        },
                    },
                ) as mocked_acquire:
                    manager.terminus_tick()

                kwargs = mocked_acquire.call_args.kwargs
                self.assertEqual(kwargs["candidate_bank_specs"][0]["catalog_mode"], "live_remote_search")
                self.assertEqual(kwargs["candidate_bank_specs"][0]["catalog_queries_per_provider"], 3)
                self.assertEqual(kwargs["candidate_bank_specs"][0]["catalog_provider_result_limit"], 4)
            finally:
                manager.close()

    def test_terminus_live_remote_search_learns_provider_curriculum(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="service_manager_provider_curriculum")
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
                                "catalog_providers": ["arxiv", "wikipedia"],
                                "catalog_queries_per_provider": 2,
                                "catalog_provider_result_limit": 4,
                                "catalog_limit": 4,
                            }
                        ],
                        "trigger_interval_tokens": 1,
                    },
                )
                manager.query(query_text="submarine buoyancy ballast", top_k_memories=6)
                manager._brain_recent_query_gaps[0]["weak_concepts"] = [
                    {
                        "label": "buoyancy control",
                        "weakness": 0.9,
                        "uncertainty": 0.8,
                        "drift": 0.1,
                        "top_terms": ["submarine", "ballast", "buoyancy"],
                        "match_count": 1,
                    }
                ]
                with patch(
                    "hecsn.service.manager.run_live_acquisition",
                    side_effect=[
                        {
                            "policy": "active",
                            "tokens_trained_total": 32,
                            "acquired_sources": ["wikipedia_submarine_source"],
                            "semantic_plan": {
                                "unsupported_terms": ["submarine", "buoyancy", "ballast"],
                            },
                            "acquisition_history": [
                                {
                                    "selected_source": "wikipedia_submarine_source",
                                    "selected_provider": "wikipedia",
                                    "selected_query_text": "submarine buoyancy ballast",
                                    "selected_semantic_relevance": 0.9,
                                    "selected_gap_reduction": 0.25,
                                    "selected_diagnostic_gap_reduction": 0.35,
                                    "tokens_trained": 32,
                                     "selected_metadata": {
                                         "provider": "wikipedia",
                                         "query_text": "submarine buoyancy ballast",
                                         "semantic_relevance": 0.9,
                                         "catalog_terms": ["marine engineering", "ballast tank"],
                                      },
                                     "candidate_snapshot": {
                                         "wikipedia_submarine_source": {
                                             "semantic_answerability": 0.20,
                                             "concept_uncertainty": 0.70,
                                             "concept_support": 0.15,
                                             "semantic_weak_concept_pressure": 0.80,
                                         }
                                    },
                                     "selected_semantic_answerability_after": 0.65,
                                     "selected_concept_uncertainty_after": 0.25,
                                     "selected_concept_support_after": 0.60,
                                     "selected_weak_concept_pressure_after": 0.20,
                                   }
                               ],
                           },
                        {
                            "policy": "active",
                            "tokens_trained_total": 24,
                            "acquired_sources": ["wikipedia_submarine_follow_up"],
                            "semantic_plan": {
                                "unsupported_terms": ["submarine", "buoyancy", "ballast"],
                            },
                            "acquisition_history": [
                                {
                                    "selected_source": "wikipedia_submarine_follow_up",
                                    "selected_provider": "wikipedia",
                                    "selected_query_text": "submarine buoyancy ballast trim",
                                    "selected_semantic_relevance": 0.92,
                                    "selected_gap_reduction": 0.18,
                                    "selected_diagnostic_gap_reduction": 0.21,
                                    "tokens_trained": 24,
                                    "selected_metadata": {
                                        "provider": "wikipedia",
                                        "query_text": "submarine buoyancy ballast trim",
                                        "semantic_relevance": 0.92,
                                        "catalog_terms": ["marine engineering", "ballast tank", "trim control"],
                                    },
                                    "candidate_snapshot": {
                                        "wikipedia_submarine_follow_up": {
                                            "semantic_answerability": 0.45,
                                            "concept_uncertainty": 0.42,
                                            "concept_support": 0.32,
                                            "semantic_weak_concept_pressure": 0.46,
                                        }
                                    },
                                    "selected_semantic_answerability_after": 0.79,
                                    "selected_concept_uncertainty_after": 0.18,
                                    "selected_concept_support_after": 0.74,
                                    "selected_weak_concept_pressure_after": 0.10,
                                }
                            ],
                        },
                        {
                            "policy": "active",
                            "tokens_trained_total": 0,
                            "acquired_sources": [],
                            "semantic_plan": {
                                "unsupported_terms": ["submarine", "buoyancy", "ballast"],
                            },
                            "acquisition_history": [],
                        },
                    ],
                ) as mocked_acquire:
                    manager.terminus_tick()
                    manager.terminus_tick()
                    manager.terminus_tick()

                runtime = manager.terminus_status()["terminus_runtime"]
                first_kwargs = mocked_acquire.call_args_list[0].kwargs
                second_kwargs = mocked_acquire.call_args_list[1].kwargs
                third_kwargs = mocked_acquire.call_args_list[2].kwargs
                self.assertEqual(
                    first_kwargs["candidate_bank_specs"][0]["catalog_providers"],
                    ["arxiv", "wikipedia"],
                )
                self.assertEqual(
                    second_kwargs["candidate_bank_specs"][0]["catalog_providers"][0],
                    "wikipedia",
                )
                self.assertGreater(
                    float(second_kwargs["candidate_bank_specs"][0]["catalog_provider_priority_map"]["wikipedia"]),
                    float(second_kwargs["candidate_bank_specs"][0]["catalog_provider_priority_map"]["arxiv"]),
                )
                self.assertEqual(
                    second_kwargs["candidate_bank_specs"][0]["catalog_queries_per_provider"],
                    3,
                )
                self.assertEqual(
                    third_kwargs["candidate_bank_specs"][0]["catalog_queries_per_provider"],
                    4,
                )
                provider_curriculum = runtime["autonomy"]["provider_curriculum"]
                self.assertIsNotNone(provider_curriculum)
                self.assertEqual(provider_curriculum["ranked_providers"][0]["provider"], "wikipedia")
                self.assertEqual(provider_curriculum["ranked_providers"][0]["successes"], 2)
                self.assertIn("submarine", provider_curriculum["focus_terms"])
                self.assertIn(
                    "marine engineering",
                    provider_curriculum["ranked_providers"][0]["topic_terms"],
                )
                self.assertGreater(
                    float(provider_curriculum["ranked_providers"][0]["answerability_gain_ema"]),
                    0.0,
                )
                self.assertGreater(
                    float(provider_curriculum["ranked_providers"][0]["uncertainty_reduction_ema"]),
                    0.0,
                )
                self.assertGreater(
                    float(provider_curriculum["ranked_providers"][0]["weak_concept_stabilization_ema"]),
                    0.0,
                )
                self.assertGreater(
                    float(provider_curriculum["ranked_providers"][0]["topic_family_strength"]),
                    0.0,
                )
                self.assertEqual(
                    int(provider_curriculum["ranked_providers"][0]["topic_family_query_bonus"]),
                    1,
                )
                self.assertIn(
                    "submarine",
                    provider_curriculum["ranked_providers"][0]["matched_topic_families"],
                )
                self.assertEqual(
                    int(provider_curriculum["ranked_providers"][0]["topic_families"]["submarine"]["commits"]),
                    2,
                )
                self.assertEqual(
                    second_kwargs["candidate_bank_specs"][0]["catalog_provider_topic_terms"]["wikipedia"][0],
                    "submarine",
                )
                self.assertIn(
                    "marine engineering",
                    second_kwargs["candidate_bank_specs"][0]["catalog_provider_topic_terms"]["wikipedia"],
                )
                self.assertEqual(
                    int(third_kwargs["candidate_bank_specs"][0]["catalog_topic_family_budget_bonus"]),
                    1,
                )
            finally:
                manager.close()

    def test_terminus_live_remote_search_avoids_off_topic_provider_term_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="service_manager_provider_topic_filter")
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
                                "catalog_providers": ["wikipedia", "openalex"],
                                "catalog_queries_per_provider": 2,
                                "catalog_provider_result_limit": 4,
                                "catalog_limit": 4,
                            }
                        ],
                        "trigger_interval_tokens": 1,
                    },
                )
                manager._brain_config["autonomy"]["provider_curriculum"] = {
                    "wikipedia": {
                        "attempts": 2,
                        "commits": 2,
                        "successes": 2,
                        "diagnostic_gain_ema": 0.2,
                        "semantic_relevance_ema": 0.9,
                        "answerability_gain_ema": 0.4,
                        "uncertainty_reduction_ema": 0.3,
                        "weak_concept_stabilization_ema": 0.2,
                        "topic_terms": {"submarine": 1.0, "buoyancy": 0.5},
                        "topic_families": {
                            "submarine": {
                                "commits": 2,
                                "successes": 2,
                                "semantic_relevance_ema": 0.9,
                                "answerability_gain_ema": 0.4,
                                "uncertainty_reduction_ema": 0.3,
                                "weak_concept_stabilization_ema": 0.2,
                            }
                        },
                    },
                    "openalex": {
                        "attempts": 2,
                        "commits": 2,
                        "successes": 2,
                        "diagnostic_gain_ema": 0.25,
                        "semantic_relevance_ema": 0.92,
                        "answerability_gain_ema": 0.45,
                        "uncertainty_reduction_ema": 0.35,
                        "weak_concept_stabilization_ema": 0.25,
                        "topic_terms": {"octopus": 1.0, "jars": 0.5},
                        "topic_families": {
                            "octopus": {
                                "commits": 2,
                                "successes": 2,
                                "semantic_relevance_ema": 0.92,
                                "answerability_gain_ema": 0.45,
                                "uncertainty_reduction_ema": 0.35,
                                "weak_concept_stabilization_ema": 0.25,
                            }
                        },
                    },
                }
                manager.query(query_text="What opens jars and solves puzzles?", top_k_memories=6)
                with patch(
                    "hecsn.service.manager.run_live_acquisition",
                    return_value={
                        "policy": "active",
                        "tokens_trained_total": 0,
                        "acquired_sources": [],
                        "semantic_plan": {
                            "unsupported_terms": ["opens", "jars", "solves", "puzzles"],
                        },
                        "acquisition_history": [],
                    },
                ) as mocked_acquire:
                    manager.terminus_tick()

                spec = mocked_acquire.call_args.kwargs["candidate_bank_specs"][0]
                self.assertEqual(spec["catalog_providers"][0], "openalex")
                self.assertNotIn("catalog_provider_topic_terms", spec)
            finally:
                manager.close()

    def test_terminus_live_remote_search_prefers_matched_topic_family_on_revisit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="service_manager_provider_revisit_alignment")
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
                                "catalog_providers": ["wikipedia", "openalex"],
                                "catalog_queries_per_provider": 2,
                                "catalog_provider_result_limit": 4,
                                "catalog_limit": 4,
                            }
                        ],
                        "trigger_interval_tokens": 1,
                    },
                )
                manager._brain_config["autonomy"]["provider_curriculum"] = {
                    "wikipedia": {
                        "attempts": 2,
                        "commits": 1,
                        "successes": 1,
                        "diagnostic_gain_ema": 0.18,
                        "semantic_relevance_ema": 0.70,
                        "answerability_gain_ema": 0.18,
                        "uncertainty_reduction_ema": 0.12,
                        "weak_concept_stabilization_ema": 0.08,
                        "topic_terms": {"submarine": 1.0, "ballast": 0.8, "buoyancy": 0.6},
                        "topic_families": {
                            "submarine": {
                                "commits": 1,
                                "successes": 1,
                                "semantic_relevance_ema": 0.70,
                                "answerability_gain_ema": 0.18,
                                "uncertainty_reduction_ema": 0.12,
                                "weak_concept_stabilization_ema": 0.08,
                            }
                        },
                    },
                    "openalex": {
                        "attempts": 2,
                        "commits": 1,
                        "successes": 1,
                        "diagnostic_gain_ema": 0.28,
                        "semantic_relevance_ema": 0.92,
                        "answerability_gain_ema": 0.42,
                        "uncertainty_reduction_ema": 0.30,
                        "weak_concept_stabilization_ema": 0.18,
                        "topic_terms": {"octopus": 1.0, "jars": 0.7},
                        "topic_families": {
                            "octopus": {
                                "commits": 1,
                                "successes": 1,
                                "semantic_relevance_ema": 0.92,
                                "answerability_gain_ema": 0.42,
                                "uncertainty_reduction_ema": 0.30,
                                "weak_concept_stabilization_ema": 0.18,
                            }
                        },
                    },
                }
                manager._brain_recent_query_gaps.appendleft(
                    manager._normalize_recent_query_gap(
                        {
                            "source": "query",
                            "query_text": "What corrects submarine trim?",
                            "unsupported_terms": ["corrects", "trim"],
                            "gap_terms": [
                                {"term": "corrects", "weight": 2.0},
                                {"term": "trim", "weight": 2.0},
                            ],
                            "retrieval_queries": ["corrects trim"],
                            "follow_up_questions": [
                                "What grounded evidence is still missing for corrects?",
                                "What grounded evidence is still missing for trim?",
                            ],
                            "weak_concepts": [
                                {
                                    "label": "terms / octopuses",
                                    "weakness": 0.55,
                                    "uncertainty": 0.54,
                                    "drift": 0.0,
                                    "top_terms": ["terms", "octopuses", "open"],
                                    "match_count": 1,
                                }
                            ],
                            "grounded_fraction": 0.3333333333333333,
                        }
                    )
                )
                with patch(
                    "hecsn.service.manager.run_live_acquisition",
                    return_value={
                        "policy": "active",
                        "tokens_trained_total": 0,
                        "acquired_sources": [],
                        "semantic_plan": {
                            "unsupported_terms": ["corrects", "submarine", "trim"],
                        },
                        "acquisition_history": [],
                    },
                ) as mocked_acquire:
                    manager.terminus_tick()

                spec = mocked_acquire.call_args.kwargs["candidate_bank_specs"][0]
                runtime = manager.terminus_status()["terminus_runtime"]
                provider_curriculum = runtime["autonomy"]["provider_curriculum"]

                self.assertEqual(spec["catalog_providers"][0], "wikipedia")
                self.assertGreater(
                    float(spec["catalog_provider_priority_map"]["wikipedia"]),
                    float(spec["catalog_provider_priority_map"]["openalex"]),
                )
                self.assertIn("submarine", str(spec["catalog_focus_text"]))
                self.assertIn("catalog_provider_topic_terms", spec)
                self.assertEqual(spec["catalog_provider_topic_terms"]["wikipedia"][0], "submarine")
                self.assertNotIn("openalex", spec["catalog_provider_topic_terms"])
                self.assertEqual(provider_curriculum["ranked_providers"][0]["provider"], "wikipedia")
                self.assertGreater(
                    float(provider_curriculum["ranked_providers"][0]["topic_family_focus_score"]),
                    0.0,
                )
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


class CortexIntegrationTests(unittest.TestCase):
    """Test cortex/ThoughtLoop integration with service manager."""

    def test_cortex_methods_available_without_ollama(self) -> None:
        """Cortex methods return graceful fallbacks when Ollama is unavailable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="cortex_no_ollama")
            try:
                # cortex_ask returns unavailable
                result = manager.cortex_ask("hello")
                self.assertIn("accepted", result)

                # cortex_thoughts returns disabled
                thoughts = manager.cortex_thoughts()
                self.assertIn("thoughts", thoughts)

                # cortex_snapshot returns disabled
                snap = manager.cortex_snapshot()
                self.assertIn("enabled", snap)
            finally:
                manager.close()

    def test_runtime_snapshot_includes_cortex(self) -> None:
        """The terminus runtime snapshot includes a 'cortex' key."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="cortex_snapshot_key")
            try:
                status = manager.status()
                runtime = status["terminus_runtime"]
                self.assertIn("cortex", runtime)
                self.assertIn("enabled", runtime["cortex"])
            finally:
                manager.close()

    def test_cortex_with_fake_cortex(self) -> None:
        """Wire a FakeCortex into the manager to test the full integration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="cortex_fake_integration")
            try:
                from hecsn.cortex.core import FakeCortex
                from hecsn.cortex.thought_loop import ThoughtLoop

                cortex = FakeCortex()
                thought_loop = ThoughtLoop(cortex=cortex)
                manager._thought_loop = thought_loop
                manager._cortex_available = True

                # Ask should be accepted now
                result = manager.cortex_ask("What is the meaning of life?")
                self.assertTrue(result["accepted"])
                self.assertEqual(result["query"], "What is the meaning of life?")

                # Step the thought loop manually
                thought_loop.step()

                # Thoughts should now be available
                thoughts = manager.cortex_thoughts()
                self.assertTrue(thoughts["enabled"])
                self.assertGreater(thoughts["thoughts_generated"], 0)
                self.assertGreater(len(thoughts["thoughts"]), 0)

                # Snapshot should include drives
                snap = manager.cortex_snapshot()
                self.assertTrue(snap["enabled"])
                self.assertIn("drives", snap)
                self.assertIn("recent_thoughts", snap)

                # Runtime snapshot should include cortex
                status = manager.status()
                cortex_data = status["terminus_runtime"]["cortex"]
                self.assertTrue(cortex_data["enabled"])
                self.assertGreater(cortex_data["thoughts_generated"], 0)
            finally:
                manager.close()

    def test_telemetry_includes_cortex(self) -> None:
        """Telemetry snapshot includes cortex key from runtime."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = _build_manager(root, test_case="cortex_telemetry")
            try:
                telemetry = manager.telemetry_snapshot()
                runtime = telemetry["terminus_runtime"]
                self.assertIn("cortex", runtime)
            finally:
                manager.close()


if __name__ == "__main__":
    unittest.main()
