from __future__ import annotations

from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import socket
import tempfile
import threading
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from hecsn.config.model_config import HECSNConfig
from hecsn.service.api import create_app
from hecsn.training.runner_utils import set_seed
from hecsn.training.checkpointing import save_trainer_checkpoint
from hecsn.training.trainer import HECSNModelLite, HECSNTrainer


def _build_checkpoint(root: Path, *, test_case: str) -> Path:
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
    return save_trainer_checkpoint(
        root / "initial.pt",
        trainer,
        metadata={"test_case": test_case},
    )


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class _SilentSimpleHTTPRequestHandler(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:  # pragma: no cover - suppress test noise
        return None


class ServiceApiTerminusRuntimeTests(unittest.TestCase):
    def test_terminus_configure_and_tick_endpoint_train_from_file_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_path = root / "terminus_source.txt"
            source_path.write_text("character stream learning " * 32, encoding="utf-8")
            app = create_app(_build_checkpoint(root, test_case="service_api_terminus_tick"), trace_dir=root / "traces")
            with TestClient(app) as client:
                configure_response = client.post(
                    "/terminus/configure",
                    json={
                        "source_bank": [
                            {
                                "name": "api_terminus_source",
                                "source": str(source_path),
                                "source_type": "file",
                            }
                        ],
                        "tick_tokens": 20,
                        "sleep_interval_seconds": 0.01,
                        "repeat_sources": False,
                    },
                )
                tick_response = client.post("/terminus/tick", json={"steps": 2})
                status_response = client.get("/status")

            self.assertEqual(configure_response.status_code, 200)
            self.assertEqual(tick_response.status_code, 200)
            self.assertEqual(status_response.status_code, 200)
            self.assertTrue(configure_response.json()["terminus_runtime"]["configured"])
            self.assertGreater(tick_response.json()["token_count"], configure_response.json()["token_count"])
            self.assertEqual(status_response.json()["terminus_runtime"]["source_bank"][0]["name"], "api_terminus_source")
            self.assertGreater(tick_response.json()["terminus_runtime"]["last_tick_token_delta"], 0)
            self.assertTrue(
                any(event.get("type") == "tick" for event in status_response.json()["terminus_runtime"]["recent_events"])
            )
            self.assertGreater(
                status_response.json()["terminus_runtime"]["source_progress"][0]["tick_visits"],
                0,
            )

    def test_terminus_tick_then_respond_returns_grounded_synthesis(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            set_seed(7)
            root = Path(tmpdir)
            source_path = root / "terminus_grounding.txt"
            source_path.write_text(
                (
                    "\n".join(
                        [
                            "cats rest indoors.",
                            "cats chase mice at night.",
                        ]
                    )
                    + "\n"
                )
                * 24,
                encoding="utf-8",
            )
            app = create_app(_build_checkpoint(root, test_case="service_api_terminus_grounded_synthesis"), trace_dir=root / "traces")
            with TestClient(app) as client:
                configure_response = client.post(
                    "/terminus/configure",
                    json={
                        "source_bank": [
                            {
                                "name": "terminus_grounding_source",
                                "source": str(source_path),
                                "source_type": "file",
                            }
                        ],
                        "tick_tokens": 48,
                        "sleep_interval_seconds": 0.01,
                        "repeat_sources": True,
                    },
                )
                tick_response = client.post("/terminus/tick", json={"steps": 6})
                respond_response = client.post(
                    "/respond",
                    json={
                        "query_text": "Where do cats rest and what do they chase at night?",
                        "max_evidence_items": 3,
                        "learn_mode": "none",
                    },
                )

            self.assertEqual(configure_response.status_code, 200)
            self.assertEqual(tick_response.status_code, 200)
            self.assertEqual(respond_response.status_code, 200)
            body = respond_response.json()
            self.assertIn(body["response"]["response_mode"], {"quote", "grounded_synthesis", "stitch"})
            self.assertIn("indoors", body["response"]["response_text"].lower())
            self.assertIn("mice", body["response"]["response_text"].lower())

    def test_terminus_tick_then_respond_handles_unsegmented_character_stream_query(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            set_seed(7)
            root = Path(tmpdir)
            source_path = root / "terminus_unsegmented_query.txt"
            source_path.write_text(
                (
                    "\n".join(
                        [
                            "submarines regulate buoyancy with ballast tanks.",
                            "ballast water shifts pressure and buoyancy inside a submarine.",
                        ]
                    )
                    + "\n"
                )
                * 24,
                encoding="utf-8",
            )
            app = create_app(
                _build_checkpoint(root, test_case="service_api_unsegmented_character_stream_query"),
                trace_dir=root / "traces",
            )
            with TestClient(app) as client:
                configure_response = client.post(
                    "/terminus/configure",
                    json={
                        "source_bank": [
                            {
                                "name": "terminus_unsegmented_query_source",
                                "source": str(source_path),
                                "source_type": "file",
                            }
                        ],
                        "tick_tokens": 48,
                        "sleep_interval_seconds": 0.01,
                        "repeat_sources": True,
                    },
                )
                tick_response = client.post("/terminus/tick", json={"steps": 6})
                respond_response = client.post(
                    "/respond",
                    json={
                        "query_text": "submarineballast",
                        "max_evidence_items": 3,
                        "learn_mode": "none",
                    },
                )

            self.assertEqual(configure_response.status_code, 200)
            self.assertEqual(tick_response.status_code, 200)
            self.assertEqual(respond_response.status_code, 200)
            body = respond_response.json()
            self.assertIn(body["response"]["response_mode"], {"quote", "grounded_synthesis", "stitch"})
            self.assertTrue(
                any(term in body["response"]["response_text"].lower() for term in ("submarine", "ballast", "buoyancy"))
            )
            self.assertEqual(body["response"]["unsupported_terms"], [])

    def test_terminus_tick_then_respond_handles_mixed_world_grounding(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            set_seed(7)
            root = Path(tmpdir)
            source_path = root / "terminus_mixed_world.txt"
            source_path.write_text(
                (
                    "\n".join(
                        [
                            "mercury is the closest planet to the sun.",
                            "volcanoes release ash and lava during eruptions.",
                            "octopuses solve puzzles and open jars.",
                            "rainbows form when sunlight passes through water droplets.",
                            "libraries lend books and provide quiet reading rooms.",
                            "moss grows on damp forest stones.",
                        ]
                    )
                    + "\n"
                )
                * 24,
                encoding="utf-8",
            )
            app = create_app(_build_checkpoint(root, test_case="service_api_terminus_mixed_world"), trace_dir=root / "traces")
            with TestClient(app) as client:
                configure_response = client.post(
                    "/terminus/configure",
                    json={
                        "source_bank": [
                            {
                                "name": "terminus_mixed_world_source",
                                "source": str(source_path),
                                "source_type": "file",
                            }
                        ],
                        "tick_tokens": 64,
                        "sleep_interval_seconds": 0.01,
                        "repeat_sources": True,
                    },
                )
                tick_response = client.post("/terminus/tick", json={"steps": 10})
                respond_response = client.post(
                    "/respond",
                    json={
                        "query_text": "What is closest to the sun and what do volcanoes release?",
                        "max_evidence_items": 3,
                        "learn_mode": "none",
                    },
                )

            self.assertEqual(configure_response.status_code, 200)
            self.assertEqual(tick_response.status_code, 200)
            self.assertEqual(respond_response.status_code, 200)
            body = respond_response.json()
            self.assertEqual(body["response"]["response_mode"], "grounded_synthesis")
            self.assertIn("mercury", body["response"]["response_text"].lower())
            self.assertTrue(
                any(term in body["response"]["response_text"].lower() for term in ("ash", "lava"))
            )

    def test_terminus_configure_accepts_registry_backed_candidate_bank(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_path = root / "terminus_source.txt"
            source_path.write_text("neutral background signal " * 24, encoding="utf-8")
            app = create_app(_build_checkpoint(root, test_case="service_api_catalog_candidate_bank"), trace_dir=root / "traces")
            with TestClient(app) as client:
                configure_response = client.post(
                    "/terminus/configure",
                    json={
                        "source_bank": [
                            {
                                "name": "api_terminus_source",
                                "source": str(source_path),
                                "source_type": "file",
                            }
                        ],
                        "tick_tokens": 20,
                        "sleep_interval_seconds": 0.01,
                        "repeat_sources": False,
                        "autonomy": {
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
                            "trigger_interval_tokens": 200,
                        },
                    },
                )
                status_response = client.get("/status")

            self.assertEqual(configure_response.status_code, 200)
            self.assertEqual(status_response.status_code, 200)
            autonomy = status_response.json()["terminus_runtime"]["autonomy"]
            self.assertEqual(autonomy["candidate_bank"][0]["catalog_mode"], "semantic_registry")
            self.assertEqual(len(autonomy["candidate_bank"][0]["catalog_entries"]), 2)
            self.assertEqual(autonomy["candidate_names"], ["registry_pool"])

    def test_terminus_tick_commits_live_remote_search_candidate_from_recent_query_focus(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            background_path = root / "terminus_background.txt"
            background_path.write_text("neutral background signal " * 40, encoding="utf-8")
            submarine_path = root / "submarine.txt"
            submarine_path.write_text(
                ("submarines regulate buoyancy with ballast tanks. ballast water shifts pressure and buoyancy inside a submarine. " * 24),
                encoding="utf-8",
            )
            garden_path = root / "garden.txt"
            garden_path.write_text("garden tomatoes need soil sunlight and watering. " * 24, encoding="utf-8")
            astronomy_path = root / "astronomy.txt"
            astronomy_path.write_text("astronomy studies planets observatories and telescope images. " * 24, encoding="utf-8")

            content_port = _free_port()
            content_server = ThreadingHTTPServer(
                ("127.0.0.1", content_port),
                partial(_SilentSimpleHTTPRequestHandler, directory=str(root)),
            )
            content_thread = threading.Thread(target=content_server.serve_forever, daemon=True)
            content_thread.start()

            app = create_app(_build_checkpoint(root, test_case="service_api_live_remote_search_commit"), trace_dir=root / "traces")

            def fake_search(provider: str, query: str, *, result_limit: int, timeout_seconds: float) -> list[dict[str, object]]:
                return [
                    {
                        "name": "submarine_source",
                        "source": f"http://127.0.0.1:{content_port}/submarine.txt",
                        "source_type": "web",
                        "summary": "submarine buoyancy ballast pressure trim tanks",
                        "query_text": query,
                        "catalog_priority": 0.9,
                        "provider": provider,
                    },
                    {
                        "name": "garden_source",
                        "source": f"http://127.0.0.1:{content_port}/garden.txt",
                        "source_type": "web",
                        "summary": "garden tomatoes soil sunlight watering",
                        "query_text": query,
                        "catalog_priority": 0.5,
                        "provider": provider,
                    },
                    {
                        "name": "astronomy_source",
                        "source": f"http://127.0.0.1:{content_port}/astronomy.txt",
                        "source_type": "web",
                        "summary": "astronomy planets observatory telescope orbit",
                        "query_text": query,
                        "catalog_priority": 0.4,
                        "provider": provider,
                    },
                ][:result_limit]

            try:
                with patch("hecsn.data.source_catalog._search_remote_provider", side_effect=fake_search):
                    with TestClient(app) as client:
                        configure_response = client.post(
                            "/terminus/configure",
                            json={
                                "source_bank": [
                                    {
                                        "name": "api_terminus_source",
                                        "source": str(background_path),
                                        "source_type": "file",
                                    }
                                ],
                                "tick_tokens": 24,
                                "sleep_interval_seconds": 0.01,
                                "repeat_sources": True,
                                "autonomy": {
                                    "enabled": True,
                                    "policy": "active",
                                    "candidate_bank": [
                                        {
                                            "name": "live_remote_pool",
                                            "catalog_mode": "live_remote_search",
                                            "catalog_providers": ["wikipedia"],
                                            "catalog_queries_per_provider": 1,
                                            "catalog_provider_result_limit": 3,
                                            "catalog_limit": 3,
                                            "catalog_probe_pool_limit": 3,
                                        }
                                    ],
                                    "trigger_interval_tokens": 1,
                                    "candidate_train_tokens": 96,
                                    "probe_tokens": 48,
                                    "acquisition_tokens": 128,
                                    "acquisition_slots": 1,
                                    "semantic_shortlist_size": 1,
                                    "semantic_shortlist_gap_weight": 0.0,
                                    "semantic_shortlist_affinity_weight": 1.0,
                                },
                            },
                        )
                        query_response = client.post(
                            "/query",
                            json={
                                "query_text": "submarine buoyancy ballast",
                                "top_k_memories": 6,
                            },
                        )
                        tick_response = client.post("/terminus/tick", json={"steps": 1})

                self.assertEqual(configure_response.status_code, 200)
                self.assertEqual(query_response.status_code, 200)
                self.assertEqual(tick_response.status_code, 200)

                query_body = query_response.json()
                tick_body = tick_response.json()
                autonomy = tick_body["terminus_runtime"]["autonomy"]
                acquisition = autonomy["last_acquisition_summary"]

                self.assertEqual(autonomy["candidate_bank"][0]["catalog_mode"], "live_remote_search")
                self.assertIn("submarine", query_body["gap_plan"]["unsupported_terms"])
                self.assertIn("submarine", autonomy["focus_plan"]["unsupported_terms"])
                self.assertEqual(acquisition["acquired_sources"], ["submarine_source"])
                self.assertGreater(acquisition["tokens_trained_total"], 0)
                self.assertFalse(acquisition["stopped_early"])
            finally:
                content_server.shutdown()
                content_server.server_close()

    def test_terminus_default_autonomy_remote_search_recovers_grounded_answer_from_recent_query_focus(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            set_seed(7)
            root = Path(tmpdir)
            background_path = root / "terminus_background.txt"
            background_path.write_text("neutral background signal " * 40, encoding="utf-8")
            submarine_path = root / "submarine.txt"
            submarine_path.write_text(
                ("submarines regulate buoyancy with ballast tanks. ballast water shifts pressure and buoyancy inside a submarine. " * 24),
                encoding="utf-8",
            )
            garden_path = root / "garden.txt"
            garden_path.write_text("garden tomatoes need soil sunlight and watering. " * 24, encoding="utf-8")
            astronomy_path = root / "astronomy.txt"
            astronomy_path.write_text("astronomy studies planets observatories and telescope images. " * 24, encoding="utf-8")

            content_port = _free_port()
            content_server = ThreadingHTTPServer(
                ("127.0.0.1", content_port),
                partial(_SilentSimpleHTTPRequestHandler, directory=str(root)),
            )
            content_thread = threading.Thread(target=content_server.serve_forever, daemon=True)
            content_thread.start()

            app = create_app(
                _build_checkpoint(root, test_case="service_api_default_live_remote_answer_recovery"),
                trace_dir=root / "traces",
            )

            def fake_search(provider: str, query: str, *, result_limit: int, timeout_seconds: float) -> list[dict[str, object]]:
                if provider != "wikipedia":
                    return []
                return [
                    {
                        "name": "submarine_source",
                        "source": f"http://127.0.0.1:{content_port}/submarine.txt",
                        "source_type": "web",
                        "summary": "submarine buoyancy ballast pressure trim tanks",
                        "query_text": query,
                        "catalog_priority": 0.9,
                        "provider": provider,
                    },
                    {
                        "name": "garden_source",
                        "source": f"http://127.0.0.1:{content_port}/garden.txt",
                        "source_type": "web",
                        "summary": "garden tomatoes soil sunlight watering",
                        "query_text": query,
                        "catalog_priority": 0.5,
                        "provider": provider,
                    },
                    {
                        "name": "astronomy_source",
                        "source": f"http://127.0.0.1:{content_port}/astronomy.txt",
                        "source_type": "web",
                        "summary": "astronomy planets observatory telescope orbit",
                        "query_text": query,
                        "catalog_priority": 0.4,
                        "provider": provider,
                    },
                ][:result_limit]

            try:
                with patch("hecsn.data.source_catalog._search_remote_provider", side_effect=fake_search):
                    with TestClient(app) as client:
                        configure_response = client.post(
                            "/terminus/configure",
                            json={
                                "source_bank": [
                                    {
                                        "name": "api_terminus_source",
                                        "source": str(background_path),
                                        "source_type": "file",
                                    }
                                ],
                                "tick_tokens": 24,
                                "sleep_interval_seconds": 0.01,
                                "repeat_sources": True,
                                "autonomy": {
                                    "enabled": True,
                                    "policy": "active",
                                    "trigger_interval_tokens": 1,
                                    "candidate_train_tokens": 96,
                                    "probe_tokens": 48,
                                    "acquisition_tokens": 128,
                                    "acquisition_slots": 1,
                                },
                            },
                        )
                        query_response = client.post(
                            "/query",
                            json={
                                "query_text": "submarine buoyancy ballast",
                                "top_k_memories": 6,
                            },
                        )
                        tick_response = client.post("/terminus/tick", json={"steps": 1})
                        respond_response = client.post(
                            "/respond",
                            json={
                                "query_text": "submarine buoyancy ballast",
                                "top_k_memories": 6,
                                "max_evidence_items": 3,
                                "learn_mode": "none",
                            },
                        )

                self.assertEqual(configure_response.status_code, 200)
                self.assertEqual(query_response.status_code, 200)
                self.assertEqual(tick_response.status_code, 200)
                self.assertEqual(respond_response.status_code, 200)

                query_body = query_response.json()
                tick_body = tick_response.json()
                respond_body = respond_response.json()
                autonomy = tick_body["terminus_runtime"]["autonomy"]
                acquisition = autonomy["last_acquisition_summary"]
                response = respond_body["response"]

                self.assertEqual(autonomy["candidate_bank"][0]["catalog_mode"], "live_remote_search")
                self.assertEqual(
                    autonomy["candidate_bank"][0]["catalog_providers"],
                    ["wikipedia", "arxiv", "openalex"],
                )
                self.assertIn("submarine", query_body["gap_plan"]["unsupported_terms"])
                self.assertIn("submarine", autonomy["focus_plan"]["unsupported_terms"])
                self.assertEqual(acquisition["acquired_sources"], ["submarine_source"])
                self.assertGreater(acquisition["tokens_trained_total"], 0)
                self.assertFalse(acquisition["stopped_early"])
                self.assertIn(response["response_mode"], {"quote", "grounded_synthesis", "stitch"})
                self.assertIn("submarine", response["response_text"].lower())
                self.assertTrue(
                    any(term in response["response_text"].lower() for term in ("buoyancy", "ballast"))
                )
                self.assertEqual(response["unsupported_terms"], [])
            finally:
                content_server.shutdown()
                content_server.server_close()

    def test_terminus_default_autonomy_remote_search_uses_catalog_summary_for_first_tick_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            set_seed(7)
            root = Path(tmpdir)
            background_path = root / "terminus_background.txt"
            background_path.write_text("neutral background signal " * 40, encoding="utf-8")
            submarine_path = root / "submarine_delayed.txt"
            submarine_path.write_text(
                (
                    "submarines travel underwater for naval operations and long endurance patrols near ocean fleets. "
                    * 10
                )
                + (
                    "submarines regulate buoyancy with ballast tanks. ballast water shifts pressure and buoyancy inside a submarine. "
                    * 16
                ),
                encoding="utf-8",
            )
            garden_path = root / "garden.txt"
            garden_path.write_text("garden tomatoes need soil sunlight and watering. " * 24, encoding="utf-8")
            astronomy_path = root / "astronomy.txt"
            astronomy_path.write_text("astronomy studies planets observatories and telescope images. " * 24, encoding="utf-8")

            content_port = _free_port()
            content_server = ThreadingHTTPServer(
                ("127.0.0.1", content_port),
                partial(_SilentSimpleHTTPRequestHandler, directory=str(root)),
            )
            content_thread = threading.Thread(target=content_server.serve_forever, daemon=True)
            content_thread.start()

            app = create_app(
                _build_checkpoint(root, test_case="service_api_default_live_remote_catalog_summary_recovery"),
                trace_dir=root / "traces",
            )

            def fake_search(provider: str, query: str, *, result_limit: int, timeout_seconds: float) -> list[dict[str, object]]:
                if provider != "wikipedia":
                    return []
                return [
                    {
                        "name": "submarine_source",
                        "source": f"http://127.0.0.1:{content_port}/submarine_delayed.txt",
                        "source_type": "web",
                        "summary": (
                            "Submarines regulate buoyancy with ballast tanks. "
                            "Ballast water shifts pressure and buoyancy inside a submarine."
                        ),
                        "terms": ["marine engineering", "ballast tank"],
                        "query_text": query,
                        "catalog_priority": 0.9,
                        "provider": provider,
                    },
                    {
                        "name": "garden_source",
                        "source": f"http://127.0.0.1:{content_port}/garden.txt",
                        "source_type": "web",
                        "summary": "garden tomatoes soil sunlight watering",
                        "query_text": query,
                        "catalog_priority": 0.5,
                        "provider": provider,
                    },
                    {
                        "name": "astronomy_source",
                        "source": f"http://127.0.0.1:{content_port}/astronomy.txt",
                        "source_type": "web",
                        "summary": "astronomy planets observatory telescope orbit",
                        "query_text": query,
                        "catalog_priority": 0.4,
                        "provider": provider,
                    },
                ][:result_limit]

            try:
                with patch("hecsn.data.source_catalog._search_remote_provider", side_effect=fake_search):
                    with TestClient(app) as client:
                        client.post(
                            "/terminus/configure",
                            json={
                                "source_bank": [
                                    {
                                        "name": "api_terminus_source",
                                        "source": str(background_path),
                                        "source_type": "file",
                                    }
                                ],
                                "tick_tokens": 24,
                                "sleep_interval_seconds": 0.01,
                                "repeat_sources": True,
                                "autonomy": {
                                    "enabled": True,
                                    "policy": "active",
                                    "trigger_interval_tokens": 1,
                                    "candidate_train_tokens": 96,
                                    "probe_tokens": 48,
                                    "acquisition_tokens": 128,
                                    "acquisition_slots": 1,
                                },
                            },
                        )
                        client.post(
                            "/query",
                            json={
                                "query_text": "submarine buoyancy ballast",
                                "top_k_memories": 6,
                            },
                        )
                        tick_response = client.post("/terminus/tick", json={"steps": 1})
                        respond_response = client.post(
                            "/respond",
                            json={
                                "query_text": "submarine buoyancy ballast",
                                "top_k_memories": 6,
                                "max_evidence_items": 3,
                                "learn_mode": "none",
                            },
                        )

                self.assertEqual(tick_response.status_code, 200)
                self.assertEqual(respond_response.status_code, 200)

                tick_body = tick_response.json()
                respond_body = respond_response.json()
                acquisition = tick_body["terminus_runtime"]["autonomy"]["last_acquisition_summary"]
                response = respond_body["response"]

                self.assertEqual(acquisition["acquired_sources"], ["submarine_source"])
                self.assertGreater(acquisition["tokens_trained_total"], 0)
                self.assertIn(response["response_mode"], {"quote", "grounded_synthesis", "stitch"})
                self.assertIn("submarine", response["response_text"].lower())
                self.assertTrue(
                    any(term in response["response_text"].lower() for term in ("buoyancy", "ballast"))
                )
                self.assertEqual(response["unsupported_terms"], [])
            finally:
                content_server.shutdown()
                content_server.server_close()

    def test_terminus_default_autonomy_remote_search_focuses_late_summary_fragments_for_first_tick_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            set_seed(7)
            root = Path(tmpdir)
            background_path = root / "terminus_background.txt"
            background_path.write_text("neutral background signal " * 40, encoding="utf-8")
            ballast_path = root / "ballast_delayed.txt"
            ballast_path.write_text(
                (
                    "ballast tanks are mechanical compartments for marine systems and vessel trim adjustments. "
                    * 10
                )
                + (
                    "ballast water reduces buoyancy in a submarine and supports underwater trim control. "
                    * 16
                ),
                encoding="utf-8",
            )
            garden_path = root / "garden.txt"
            garden_path.write_text("garden tomatoes need soil sunlight and watering. " * 24, encoding="utf-8")
            astronomy_path = root / "astronomy.txt"
            astronomy_path.write_text("astronomy studies planets observatories and telescope images. " * 24, encoding="utf-8")

            content_port = _free_port()
            content_server = ThreadingHTTPServer(
                ("127.0.0.1", content_port),
                partial(_SilentSimpleHTTPRequestHandler, directory=str(root)),
            )
            content_thread = threading.Thread(target=content_server.serve_forever, daemon=True)
            content_thread.start()

            app = create_app(
                _build_checkpoint(root, test_case="service_api_default_live_remote_focus_fragment_recovery"),
                trace_dir=root / "traces",
            )

            def fake_search(provider: str, query: str, *, result_limit: int, timeout_seconds: float) -> list[dict[str, object]]:
                if provider != "wikipedia":
                    return []
                return [
                    {
                        "name": "ballast_tank_source",
                        "source": f"http://127.0.0.1:{content_port}/ballast_delayed.txt",
                        "source_type": "web",
                        "summary": (
                            "A ballast tank is a compartment within a boat, ship, or floating structure that holds water, "
                            "ballast water reduces buoyancy in a submarine, and helps correct trim."
                        ),
                        "query_text": query,
                        "catalog_priority": 0.9,
                        "provider": provider,
                    },
                    {
                        "name": "garden_source",
                        "source": f"http://127.0.0.1:{content_port}/garden.txt",
                        "source_type": "web",
                        "summary": "garden tomatoes soil sunlight watering",
                        "query_text": query,
                        "catalog_priority": 0.5,
                        "provider": provider,
                    },
                    {
                        "name": "astronomy_source",
                        "source": f"http://127.0.0.1:{content_port}/astronomy.txt",
                        "source_type": "web",
                        "summary": "astronomy planets observatory telescope orbit",
                        "query_text": query,
                        "catalog_priority": 0.4,
                        "provider": provider,
                    },
                ][:result_limit]

            try:
                with patch("hecsn.data.source_catalog._search_remote_provider", side_effect=fake_search):
                    with TestClient(app) as client:
                        client.post(
                            "/terminus/configure",
                            json={
                                "source_bank": [
                                    {
                                        "name": "api_terminus_source",
                                        "source": str(background_path),
                                        "source_type": "file",
                                    }
                                ],
                                "tick_tokens": 24,
                                "sleep_interval_seconds": 0.01,
                                "repeat_sources": True,
                                "autonomy": {
                                    "enabled": True,
                                    "policy": "active",
                                    "trigger_interval_tokens": 1,
                                    "candidate_train_tokens": 96,
                                    "probe_tokens": 48,
                                    "acquisition_tokens": 128,
                                    "acquisition_slots": 1,
                                },
                            },
                        )
                        client.post(
                            "/query",
                            json={
                                "query_text": "submarine buoyancy ballast",
                                "top_k_memories": 6,
                            },
                        )
                        tick_response = client.post("/terminus/tick", json={"steps": 1})
                        respond_response = client.post(
                            "/respond",
                            json={
                                "query_text": "submarine buoyancy ballast",
                                "top_k_memories": 6,
                                "max_evidence_items": 3,
                                "learn_mode": "none",
                            },
                        )

                self.assertEqual(tick_response.status_code, 200)
                self.assertEqual(respond_response.status_code, 200)

                tick_body = tick_response.json()
                respond_body = respond_response.json()
                acquisition = tick_body["terminus_runtime"]["autonomy"]["last_acquisition_summary"]
                response = respond_body["response"]

                self.assertEqual(acquisition["acquired_sources"], ["ballast_tank_source"])
                self.assertGreater(acquisition["tokens_trained_total"], 0)
                self.assertIn(response["response_mode"], {"quote", "grounded_synthesis", "stitch"})
                self.assertIn("ballast", response["response_text"].lower())
                self.assertTrue(
                    any(term in response["response_text"].lower() for term in ("buoyancy", "submarine"))
                )
                self.assertEqual(response["unsupported_terms"], [])
            finally:
                content_server.shutdown()
                content_server.server_close()

    def test_terminus_live_remote_search_probes_cross_provider_page_content_for_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            set_seed(7)
            root = Path(tmpdir)
            background_path = root / "terminus_background.txt"
            background_path.write_text("neutral background signal " * 40, encoding="utf-8")
            ballast_paper_path = root / "ballast_paper.txt"
            ballast_paper_path.write_text(
                (
                    "marine engineering reports compare underwater trim and ballast systems. "
                    "ballast tanks reduce submarine buoyancy and support underwater trim control. "
                )
                * 18,
                encoding="utf-8",
            )
            cable_path = root / "submarine_cable.txt"
            cable_path.write_text(
                "submarine cables carry internet traffic between continents and coastal landing stations. " * 24,
                encoding="utf-8",
            )

            content_port = _free_port()
            content_server = ThreadingHTTPServer(
                ("127.0.0.1", content_port),
                partial(_SilentSimpleHTTPRequestHandler, directory=str(root)),
            )
            content_thread = threading.Thread(target=content_server.serve_forever, daemon=True)
            content_thread.start()

            app = create_app(
                _build_checkpoint(root, test_case="service_api_live_remote_cross_provider_content_probe"),
                trace_dir=root / "traces",
            )

            def fake_search(provider: str, query: str, *, result_limit: int, timeout_seconds: float) -> list[dict[str, object]]:
                if provider == "wikipedia":
                    return [
                        {
                            "name": "cable_source",
                            "source": f"http://127.0.0.1:{content_port}/submarine_cable.txt",
                            "source_type": "web",
                            "summary": "Submarine infrastructure and communications systems.",
                            "query_text": query,
                            "catalog_priority": 0.70,
                            "provider": provider,
                        }
                    ][:result_limit]
                if provider == "openalex":
                    return [
                        {
                            "name": "ballast_paper_source",
                            "source": f"http://127.0.0.1:{content_port}/ballast_paper.txt",
                            "source_type": "web",
                            "summary": "Marine systems analysis of vessel stability and trim.",
                            "terms": ["marine engineering"],
                            "query_text": query,
                            "catalog_priority": 0.35,
                            "provider": provider,
                        }
                    ][:result_limit]
                return []

            try:
                with patch("hecsn.data.source_catalog._search_remote_provider", side_effect=fake_search):
                    with TestClient(app) as client:
                        client.post(
                            "/terminus/configure",
                            json={
                                "source_bank": [
                                    {
                                        "name": "api_terminus_source",
                                        "source": str(background_path),
                                        "source_type": "file",
                                    }
                                ],
                                "tick_tokens": 24,
                                "sleep_interval_seconds": 0.01,
                                "repeat_sources": True,
                                "autonomy": {
                                    "enabled": True,
                                    "policy": "active",
                                    "candidate_bank": [
                                        {
                                            "name": "live_remote_pool",
                                            "catalog_mode": "live_remote_search",
                                            "catalog_providers": ["wikipedia", "openalex"],
                                            "catalog_limit": 1,
                                            "catalog_probe_pool_limit": 2,
                                            "catalog_prior_weight": 0.4,
                                            "catalog_queries_per_provider": 1,
                                            "catalog_provider_result_limit": 1,
                                        }
                                    ],
                                    "trigger_interval_tokens": 1,
                                    "candidate_train_tokens": 96,
                                    "probe_tokens": 48,
                                    "acquisition_tokens": 128,
                                    "acquisition_slots": 1,
                                },
                            },
                        )
                        client.post(
                            "/query",
                            json={
                                "query_text": "submarine buoyancy ballast",
                                "top_k_memories": 6,
                            },
                        )
                        tick_response = client.post("/terminus/tick", json={"steps": 1})
                        respond_response = client.post(
                            "/respond",
                            json={
                                "query_text": "submarine buoyancy ballast",
                                "top_k_memories": 6,
                                "max_evidence_items": 3,
                                "learn_mode": "none",
                            },
                        )

                self.assertEqual(tick_response.status_code, 200)
                self.assertEqual(respond_response.status_code, 200)

                tick_body = tick_response.json()
                respond_body = respond_response.json()
                acquisition = tick_body["terminus_runtime"]["autonomy"]["last_acquisition_summary"]
                response = respond_body["response"]

                self.assertEqual(acquisition["acquired_sources"], ["ballast_paper_source"])
                self.assertGreater(acquisition["tokens_trained_total"], 0)
                self.assertIn(response["response_mode"], {"quote", "grounded_synthesis", "stitch"})
                self.assertIn("submarine", response["response_text"].lower())
                self.assertTrue(
                    any(term in response["response_text"].lower() for term in ("buoyancy", "ballast"))
                )
                self.assertEqual(response["unsupported_terms"], [])
            finally:
                content_server.shutdown()
                content_server.server_close()

    def test_terminus_default_autonomy_remote_search_can_recover_via_follow_up_probe(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            set_seed(7)
            root = Path(tmpdir)
            background_path = root / "terminus_background.txt"
            background_path.write_text("neutral background signal " * 40, encoding="utf-8")
            submarine_path = root / "submarine.txt"
            submarine_path.write_text(
                ("submarines regulate buoyancy with ballast tanks. ballast water shifts pressure and buoyancy inside a submarine. " * 24),
                encoding="utf-8",
            )

            content_port = _free_port()
            content_server = ThreadingHTTPServer(
                ("127.0.0.1", content_port),
                partial(_SilentSimpleHTTPRequestHandler, directory=str(root)),
            )
            content_thread = threading.Thread(target=content_server.serve_forever, daemon=True)
            content_thread.start()

            app = create_app(
                _build_checkpoint(root, test_case="service_api_default_live_remote_follow_up_probe"),
                trace_dir=root / "traces",
            )
            provider_queries: list[str] = []

            def fake_search(provider: str, query: str, *, result_limit: int, timeout_seconds: float) -> list[dict[str, object]]:
                provider_queries.append(f"{provider}:{query}")
                if provider != "wikipedia":
                    return []
                if query.strip().lower() != "submarine":
                    return []
                return [
                    {
                        "name": "submarine_source",
                        "source": f"http://127.0.0.1:{content_port}/submarine.txt",
                        "source_type": "web",
                        "summary": "submarine buoyancy ballast pressure trim tanks",
                        "query_text": query,
                        "catalog_priority": 0.9,
                        "provider": provider,
                    }
                ][:result_limit]

            try:
                with patch("hecsn.data.source_catalog._search_remote_provider", side_effect=fake_search):
                    with TestClient(app) as client:
                        configure_response = client.post(
                            "/terminus/configure",
                            json={
                                "source_bank": [
                                    {
                                        "name": "api_terminus_source",
                                        "source": str(background_path),
                                        "source_type": "file",
                                    }
                                ],
                                "tick_tokens": 24,
                                "sleep_interval_seconds": 0.01,
                                "repeat_sources": True,
                                "autonomy": {
                                    "enabled": True,
                                    "policy": "active",
                                    "trigger_interval_tokens": 1,
                                    "candidate_train_tokens": 96,
                                    "probe_tokens": 48,
                                    "acquisition_tokens": 128,
                                    "acquisition_slots": 1,
                                },
                            },
                        )
                        query_response = client.post(
                            "/query",
                            json={
                                "query_text": "submarine buoyancy ballast",
                                "top_k_memories": 6,
                            },
                        )
                        tick_response = client.post("/terminus/tick", json={"steps": 1})
                        respond_response = client.post(
                            "/respond",
                            json={
                                "query_text": "submarine buoyancy ballast",
                                "top_k_memories": 6,
                                "max_evidence_items": 3,
                                "learn_mode": "none",
                            },
                        )

                self.assertEqual(configure_response.status_code, 200)
                self.assertEqual(query_response.status_code, 200)
                self.assertEqual(tick_response.status_code, 200)
                self.assertEqual(respond_response.status_code, 200)

                tick_body = tick_response.json()
                respond_body = respond_response.json()
                autonomy = tick_body["terminus_runtime"]["autonomy"]
                acquisition = autonomy["last_acquisition_summary"]
                response = respond_body["response"]

                self.assertEqual(autonomy["candidate_bank"][0]["catalog_mode"], "live_remote_search")
                self.assertEqual(
                    autonomy["candidate_bank"][0]["catalog_providers"],
                    ["wikipedia", "arxiv", "openalex"],
                )
                self.assertEqual(autonomy["candidate_bank"][0]["catalog_queries_per_provider"], 2)
                self.assertIn("wikipedia:submarine buoyancy ballast", provider_queries)
                self.assertIn("wikipedia:submarine", provider_queries)
                self.assertEqual(acquisition["acquired_sources"], ["submarine_source"])
                self.assertGreater(acquisition["tokens_trained_total"], 0)
                self.assertFalse(acquisition["stopped_early"])
                self.assertIn(response["response_mode"], {"quote", "grounded_synthesis", "stitch"})
                self.assertIn("submarine", response["response_text"].lower())
                self.assertTrue(
                    any(term in response["response_text"].lower() for term in ("buoyancy", "ballast"))
                )
                self.assertEqual(response["unsupported_terms"], [])
            finally:
                content_server.shutdown()
                content_server.server_close()

    def test_terminus_default_autonomy_remote_search_grows_query_budget_from_weak_concepts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            set_seed(7)
            root = Path(tmpdir)
            background_path = root / "terminus_background.txt"
            background_path.write_text("neutral background signal " * 40, encoding="utf-8")
            submarine_path = root / "submarine.txt"
            submarine_path.write_text(
                ("submarines regulate buoyancy with ballast tanks. ballast water shifts pressure and buoyancy inside a submarine. " * 24),
                encoding="utf-8",
            )

            content_port = _free_port()
            content_server = ThreadingHTTPServer(
                ("127.0.0.1", content_port),
                partial(_SilentSimpleHTTPRequestHandler, directory=str(root)),
            )
            content_thread = threading.Thread(target=content_server.serve_forever, daemon=True)
            content_thread.start()

            app = create_app(
                _build_checkpoint(root, test_case="service_api_default_live_remote_query_growth"),
                trace_dir=root / "traces",
            )
            provider_queries: list[str] = []

            def fake_search(provider: str, query: str, *, result_limit: int, timeout_seconds: float) -> list[dict[str, object]]:
                provider_queries.append(f"{provider}:{query}")
                if provider != "wikipedia":
                    return []
                if query.strip().lower() != "submarine ballast buoyancy":
                    return []
                return [
                    {
                        "name": "submarine_source",
                        "source": f"http://127.0.0.1:{content_port}/submarine.txt",
                        "source_type": "web",
                        "summary": "submarine buoyancy ballast pressure trim tanks",
                        "query_text": query,
                        "catalog_priority": 0.9,
                        "provider": provider,
                    }
                ][:result_limit]

            try:
                with patch("hecsn.data.source_catalog._search_remote_provider", side_effect=fake_search):
                    with TestClient(app) as client:
                        configure_response = client.post(
                            "/terminus/configure",
                            json={
                                "source_bank": [
                                    {
                                        "name": "api_terminus_source",
                                        "source": str(background_path),
                                        "source_type": "file",
                                    }
                                ],
                                "tick_tokens": 24,
                                "sleep_interval_seconds": 0.01,
                                "repeat_sources": True,
                                "autonomy": {
                                    "enabled": True,
                                    "policy": "active",
                                    "trigger_interval_tokens": 1,
                                    "candidate_train_tokens": 96,
                                    "probe_tokens": 48,
                                    "acquisition_tokens": 128,
                                    "acquisition_slots": 1,
                                },
                            },
                        )
                        with app.state.hecsn_manager._lock:
                            app.state.hecsn_manager._record_recent_query_gap_locked(
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
                        tick_response = client.post("/terminus/tick", json={"steps": 1})
                        respond_response = client.post(
                            "/respond",
                            json={
                                "query_text": "submarine buoyancy ballast",
                                "top_k_memories": 6,
                                "max_evidence_items": 3,
                                "learn_mode": "none",
                            },
                        )

                self.assertEqual(configure_response.status_code, 200)
                self.assertEqual(tick_response.status_code, 200)
                self.assertEqual(respond_response.status_code, 200)

                tick_body = tick_response.json()
                respond_body = respond_response.json()
                autonomy = tick_body["terminus_runtime"]["autonomy"]
                acquisition = autonomy["last_acquisition_summary"]
                response = respond_body["response"]

                self.assertEqual(autonomy["candidate_bank"][0]["catalog_mode"], "live_remote_search")
                self.assertEqual(autonomy["candidate_bank"][0]["catalog_queries_per_provider"], 2)
                self.assertEqual(
                    autonomy["focus_plan"]["weak_concepts"][0]["top_terms"],
                    ["garden", "tomato", "soil"],
                )
                self.assertIn("wikipedia:submarine", provider_queries)
                self.assertIn("wikipedia:garden tomato soil", provider_queries)
                self.assertIn("wikipedia:submarine ballast buoyancy", provider_queries)
                self.assertEqual(acquisition["acquired_sources"], ["submarine_source"])
                self.assertGreater(acquisition["tokens_trained_total"], 0)
                self.assertFalse(acquisition["stopped_early"])
                self.assertIn(response["response_mode"], {"quote", "grounded_synthesis", "stitch"})
                self.assertIn("submarine", response["response_text"].lower())
                self.assertTrue(
                    any(term in response["response_text"].lower() for term in ("buoyancy", "ballast"))
                )
                self.assertEqual(response["unsupported_terms"], [])
            finally:
                content_server.shutdown()
                content_server.server_close()

    def test_terminus_start_rejects_missing_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            app = create_app(_build_checkpoint(root, test_case="service_api_terminus_start_guard"), trace_dir=root / "traces")
            with TestClient(app) as client:
                response = client.post("/terminus/start")

            self.assertEqual(response.status_code, 422)
            self.assertIn("source_bank", response.text)

    def test_legacy_brain_report_and_preset_endpoints_are_not_exposed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            app = create_app(_build_checkpoint(root, test_case="service_api_legacy_surface_removed"), trace_dir=root / "traces")
            with TestClient(app) as client:
                brain_response = client.get("/brain")
                presets_response = client.get("/acquisition/presets")
                reports_response = client.get("/reports/benchmarks")

            self.assertEqual(brain_response.status_code, 404)
            self.assertEqual(presets_response.status_code, 404)
            self.assertEqual(reports_response.status_code, 404)

    def test_architecture_endpoint_returns_layer_topology(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            app = create_app(_build_checkpoint(root, test_case="service_api_architecture"), trace_dir=root / "traces")
            with TestClient(app) as client:
                resp = client.get("/architecture")

            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertEqual(data["model_name"], "HECSNModelLite")
            self.assertEqual(data["version"], "v4")
            layers = data["layers"]
            self.assertIsInstance(layers, list)
            self.assertGreater(len(layers), 0)
            layer_ids = [l["id"] for l in layers]
            self.assertIn("input_encoding", layer_ids)
            self.assertIn("competitive_routing", layer_ids)
            self.assertIn("memory_consolidation", layer_ids)
            for layer in layers:
                self.assertIn("id", layer)
                self.assertIn("name", layer)
                self.assertIn("enabled", layer)
                self.assertIn("type", layer)
                self.assertIn("params", layer)
            enabled_layers = [l for l in layers if l["enabled"]]
            self.assertGreaterEqual(len(enabled_layers), 3)

    def test_grounding_probe_endpoint_returns_accuracy(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            app = create_app(_build_checkpoint(root, test_case="service_api_grounding_probe"), trace_dir=root / "traces")
            with TestClient(app) as client:
                resp = client.post("/grounding-probe/run")

            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertIn("total_accuracy", data)
            self.assertIn("concrete_accuracy", data)
            self.assertIn("abstract_accuracy", data)
            self.assertIn("concreteness_gap", data)
            self.assertIsInstance(data["total_accuracy"], float)
            self.assertGreaterEqual(data["total_accuracy"], 0.0)
            self.assertLessEqual(data["total_accuracy"], 1.0)

    def test_telemetry_snapshot_includes_animation_data(self) -> None:
        """telemetry_snapshot includes animation sub-object for SSE consumers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            from hecsn.service.manager import HECSNServiceManager
            mgr = HECSNServiceManager(
                _build_checkpoint(root, test_case="service_api_animation"),
                trace_dir=root / "traces",
            )
            snapshot = mgr.telemetry_snapshot()
            mgr.close()

        self.assertIn("animation", snapshot)
        anim = snapshot["animation"]
        self.assertIn("n_columns", anim)
        self.assertIn("activations", anim)
        self.assertIn("spike_counts", anim)
        self.assertIn("memory_fill", anim)
        self.assertEqual(len(anim["activations"]), anim["n_columns"])


if __name__ == "__main__":
    unittest.main()
