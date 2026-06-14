from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
from functools import partial
from http.server import BaseHTTPRequestHandler, SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import socket
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

from fastapi.testclient import TestClient

from marulho.config.model_config import MarulhoConfig
from marulho.semantics import (
    build_snn_language_transition_memory_prediction_evaluation,
    build_spike_language_decoder_probe,
    predict_spike_language_sequence,
)
from marulho.service.api import (
    DEFAULT_WEB_DIST_DIR,
    _internal_snn_language_payload,
    _public_snn_language_payload,
    create_app,
)
from marulho.service.server import build_arg_parser
from marulho.training.runner_utils import set_seed
from marulho.training.checkpointing import load_trainer_checkpoint, save_trainer_checkpoint
from marulho.training.model import MarulhoModel
from marulho.training.trainer import MarulhoTrainer


def _sha256_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _build_checkpoint(root: Path, *, test_case: str) -> Path:
    cfg = MarulhoConfig(
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
    model = MarulhoModel(cfg)
    trainer = MarulhoTrainer(model, cfg)
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


class _EchoJsonApiHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:  # pragma: no cover - suppress test noise
        return None

    def do_POST(self) -> None:  # noqa: N802
        content_length = int(self.headers.get("Content-Length", "0") or 0)
        payload = self.rfile.read(content_length) if content_length > 0 else b""
        parsed_body = json.loads(payload.decode("utf-8") or "null")
        parsed_url = urlparse(self.path)
        response = {
            "method": "POST",
            "path": parsed_url.path,
            "query": {key: values[0] if len(values) == 1 else values for key, values in parse_qs(parsed_url.query).items()},
            "payload": parsed_body,
        }
        encoded = json.dumps(response).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


class ServiceApiTerminusRuntimeTests(unittest.TestCase):
    def test_checkpoint_save_maps_busy_runtime_to_conflict(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            app = create_app(
                _build_checkpoint(root, test_case="service_api_checkpoint_busy"),
                trace_dir=root / "traces",
            )
            try:
                with patch.object(
                    app.state.marulho_runtime,
                    "save_checkpoint",
                    side_effect=TimeoutError("Terminus is running; stop first."),
                ):
                    with TestClient(app) as client:
                        response = client.post(
                            "/checkpoint/save",
                            json={"path": str(root / "busy.pt")},
                        )
            finally:
                app.state.marulho_manager.close()

        self.assertEqual(response.status_code, 409)
        self.assertIn("stop first", response.json()["detail"])

    def test_snn_language_public_payload_uses_readout_vocabulary(self) -> None:
        internal_payload = {
            "artifact_kind": "terminus_snn_language_autonomous_snn_language_thought_surface_design",
            "surface": "snn_language_autonomous_snn_language_thought_surface_design.v1",
            "autonomous_snn_language_thought_surface_design": {
                "thought_role": "inner_speech_candidate",
                "binding_mode": "hash_bound_inner_language",
                "max_thought_fragments": 1,
                "thought_surface_hash": "0" * 64,
            },
            "promotion_gate": {
                "eligible_for_autonomous_snn_language_thought_surface_preflight": True,
                "next_gate": "autonomous_snn_language_thought_surface_preflight",
            },
        }

        public_payload = _public_snn_language_payload(internal_payload)
        public_text = json.dumps(public_payload, sort_keys=True)

        self.assertNotIn("autonomous_snn_language_thought", public_text)
        self.assertNotIn("snn_language_autonomous_snn_language_thought", public_text)
        self.assertNotIn("thought_", public_text)
        self.assertIn("snn_language_readout_surface_design", public_text)
        self.assertIn("bounded_readout_candidate", public_text)
        self.assertIn("hash_bound_readout_language", public_text)
        internalized_payload = _internal_snn_language_payload(public_payload)
        self.assertEqual(
            internalized_payload["surface"],
            "snn_language_autonomous_snn_language_thought_surface_design.v1",
        )
        self.assertIn(
            "autonomous_snn_language_thought_surface_design",
            internalized_payload,
        )
        self.assertEqual(
            internalized_payload["autonomous_snn_language_thought_surface_design"]["thought_role"],
            "inner_speech_candidate",
        )

    def test_app_creation_health_status_do_not_eagerly_initialize_cortex(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            app = create_app(_build_checkpoint(root, test_case="service_api_cortex_lazy_startup"), trace_dir=root / "traces")
            with TestClient(app) as client:
                health_response = client.get("/health")
                status_response = client.get("/status")
            app.state.marulho_manager.close()

            self.assertEqual(health_response.status_code, 200)
            self.assertEqual(status_response.status_code, 200)
            terminus_runtime = status_response.json()["terminus_runtime"]
            self.assertNotIn("cortex", terminus_runtime)
            self.assertNotIn("retired_runtime_path", terminus_runtime)
            self.assertNotIn("retired_runtime_dependency", terminus_runtime["living_loop"]["subcortex_sleep_pressure"])

    def test_snn_language_developmental_canonical_routes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            app = create_app(
                _build_checkpoint(
                    root,
                    test_case="service_api_snn_language_developmental_routes",
                ),
                trace_dir=root / "traces",
            )
            with TestClient(app) as client:
                status_response = client.get("/status")
                expected_revision = status_response.json()["state_revision"]
                design_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/snn-language-capacity-mutation-design",
                    json={
                        "snn_language_structural_plasticity_event_review": {
                            "surface": (
                                "snn_language_autonomous_snn_language_thought_"
                                "structural_plasticity_event_review.v1"
                            ),
                            "accepted": False,
                            "ready": False,
                            "promotion_gate": {
                                "eligible_for_autonomous_snn_language_thought_"
                                "capacity_mutation_design": False
                            },
                        },
                        "capacity_policy": {
                            "mutation_scope": "language_readout_capacity"
                        },
                    },
                )
                preflight_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/snn-language-capacity-mutation-preflight",
                    json={
                        "snn_language_capacity_mutation_design": (
                            design_response.json()
                        ),
                        "expected_state_revision": expected_revision,
                        "checkpoint_transaction": {
                            "checkpoint_path": (
                                "memory://language-capacity-preflight"
                            ),
                            "snapshot_id": "language-capacity-snapshot",
                        },
                        "device_evidence": {
                            "device": "cpu",
                            "cuda_available": False,
                        },
                        "executor_capabilities": {
                            "snn_language_capacity_mutation_executor": True
                        },
                    },
                )
                executor_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/snn-language-capacity-mutation-executor",
                    json={
                        "snn_language_capacity_mutation_preflight": (
                            preflight_response.json()
                        ),
                        "expected_state_revision": expected_revision,
                        "requested_device": "cpu",
                    },
                )
                review_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/snn-language-capacity-mutation-event-review",
                    json={
                        "snn_language_capacity_mutation_executor": (
                            executor_response.json()
                        ),
                        "expected_state_revision": expected_revision,
                    },
                )
                newborn_design_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/snn-language-newborn-neuron-integration-design",
                    json={
                        "snn_language_capacity_mutation_event_review": (
                            review_response.json()
                        ),
                        "integration_policy": {
                            "max_newborn_neurons": 1,
                            "max_seed_synapses_per_newborn": 1,
                        },
                    },
                )
                newborn_preflight_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/snn-language-newborn-neuron-integration-preflight",
                    json={
                        "snn_language_newborn_neuron_integration_design": (
                            newborn_design_response.json()
                        ),
                        "expected_state_revision": expected_revision,
                        "live_spike_evidence": {
                            "surface": "snn_language_live_spike_population_evidence.v1",
                            "state_revision": expected_revision,
                            "observation_window_id": "canonical-newborn-route-window",
                            "observation_window_hash": "0" * 64,
                            "device": "cpu",
                            "tensor_is_cuda": False,
                            "candidate_observations": [],
                        },
                        "checkpoint_transaction": {},
                        "executor_capabilities": {},
                    },
                )
                newborn_executor_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/snn-language-newborn-neuron-integration-executor",
                    json={
                        "snn_language_newborn_neuron_integration_preflight": (
                            newborn_preflight_response.json()
                        ),
                        "expected_state_revision": expected_revision,
                    },
                )
                newborn_review_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/snn-language-newborn-neuron-integration-event-review",
                    json={
                        "snn_language_newborn_neuron_integration_executor": (
                            newborn_executor_response.json()
                        ),
                        "expected_state_revision": expected_revision,
                    },
                )
                critical_design_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/snn-language-newborn-neuron-critical-period-learning-design",
                    json={
                        "snn_language_newborn_neuron_integration_event_review": (
                            newborn_review_response.json()
                        )
                    },
                )
                critical_preflight_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/snn-language-newborn-neuron-critical-period-learning-preflight",
                    json={
                        "snn_language_newborn_neuron_critical_period_learning_design": (
                            critical_design_response.json()
                        ),
                        "expected_state_revision": expected_revision,
                        "critical_period_activity_evidence": {
                            "surface": "snn_language_newborn_critical_period_activity.v1",
                            "state_revision": expected_revision,
                            "observation_window_id": "canonical-critical-period-route-window",
                            "observation_window_hash": "0" * 64,
                            "device": "cpu",
                            "tensor_is_cuda": False,
                            "candidate_observations": [],
                        },
                        "checkpoint_transaction": {},
                        "executor_capabilities": {},
                    },
                )
                critical_executor_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/snn-language-newborn-neuron-critical-period-learning-executor",
                    json={
                        "snn_language_newborn_neuron_critical_period_learning_preflight": (
                            critical_preflight_response.json()
                        ),
                        "expected_state_revision": expected_revision,
                    },
                )
                critical_review_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/snn-language-newborn-neuron-critical-period-learning-event-review",
                    json={
                        "snn_language_newborn_neuron_critical_period_learning_executor": (
                            critical_executor_response.json()
                        ),
                        "expected_state_revision": expected_revision,
                    },
                )
                critical_continuation_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/snn-language-newborn-neuron-critical-period-learning-continuation-design",
                    json={
                        "snn_language_newborn_neuron_critical_period_learning_event_review": (
                            critical_review_response.json()
                        )
                    },
                )
                maturation_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/snn-language-newborn-neuron-maturation-outcome-review",
                    json={
                        "snn_language_newborn_neuron_critical_period_learning_event_review": (
                            critical_review_response.json()
                        )
                    },
                )
                pruning_design_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/snn-language-newborn-synapse-pruning-design",
                    json={
                        "snn_language_newborn_neuron_maturation_outcome_review": (
                            maturation_response.json()
                        )
                    },
                )
                pruning_preflight_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/snn-language-newborn-synapse-pruning-preflight",
                    json={
                        "snn_language_newborn_synapse_pruning_design": (
                            pruning_design_response.json()
                        ),
                        "expected_state_revision": expected_revision,
                        "checkpoint_transaction": {},
                        "executor_capabilities": {},
                    },
                )
                pruning_executor_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/snn-language-newborn-synapse-pruning-executor",
                    json={
                        "snn_language_newborn_synapse_pruning_preflight": (
                            pruning_preflight_response.json()
                        ),
                        "expected_state_revision": expected_revision,
                    },
                )
            app.state.marulho_manager.close()

            self.assertEqual(design_response.status_code, 200)
            self.assertEqual(preflight_response.status_code, 200)
            self.assertEqual(executor_response.status_code, 200)
            self.assertEqual(review_response.status_code, 200)
            self.assertEqual(newborn_design_response.status_code, 200)
            self.assertEqual(newborn_preflight_response.status_code, 200)
            self.assertEqual(newborn_executor_response.status_code, 200)
            self.assertEqual(newborn_review_response.status_code, 200)
            self.assertEqual(critical_design_response.status_code, 200)
            self.assertEqual(critical_preflight_response.status_code, 200)
            self.assertEqual(critical_executor_response.status_code, 200)
            self.assertEqual(critical_review_response.status_code, 200)
            self.assertEqual(critical_continuation_response.status_code, 200)
            self.assertEqual(maturation_response.status_code, 200)
            self.assertEqual(pruning_design_response.status_code, 200)
            self.assertEqual(pruning_preflight_response.status_code, 200)
            self.assertEqual(pruning_executor_response.status_code, 200)
            self.assertEqual(
                design_response.json()["surface"],
                "snn_language_readout_capacity_mutation_design.v1",
            )
            self.assertEqual(
                preflight_response.json()["surface"],
                "snn_language_readout_capacity_mutation_preflight.v1",
            )
            self.assertEqual(
                executor_response.json()["surface"],
                "snn_language_readout_capacity_mutation_executor.v1",
            )
            self.assertEqual(
                review_response.json()["surface"],
                "snn_language_readout_capacity_mutation_event_review.v1",
            )
            self.assertEqual(
                newborn_design_response.json()["surface"],
                "snn_language_readout_newborn_neuron_integration_design.v1",
            )
            self.assertEqual(
                newborn_preflight_response.json()["surface"],
                "snn_language_readout_newborn_neuron_integration_preflight.v1",
            )
            self.assertEqual(
                newborn_executor_response.json()["surface"],
                "snn_language_readout_newborn_neuron_integration_executor.v1",
            )
            self.assertEqual(
                newborn_review_response.json()["surface"],
                "snn_language_readout_newborn_neuron_integration_event_review.v1",
            )
            self.assertEqual(
                critical_design_response.json()["surface"],
                "snn_language_readout_newborn_neuron_critical_period_learning_design.v1",
            )
            self.assertEqual(
                critical_preflight_response.json()["surface"],
                "snn_language_readout_newborn_neuron_critical_period_learning_preflight.v1",
            )
            self.assertEqual(
                critical_executor_response.json()["surface"],
                "snn_language_readout_newborn_neuron_critical_period_learning_executor.v1",
            )
            self.assertEqual(
                critical_review_response.json()["surface"],
                "snn_language_readout_newborn_neuron_critical_period_learning_event_review.v1",
            )
            self.assertEqual(
                critical_continuation_response.json()["surface"],
                "snn_language_readout_newborn_neuron_critical_period_learning_design.v1",
            )
            self.assertEqual(
                maturation_response.json()["surface"],
                "snn_language_readout_newborn_neuron_maturation_outcome_review.v1",
            )
            self.assertEqual(
                pruning_design_response.json()["surface"],
                "snn_language_readout_newborn_synapse_pruning_design.v1",
            )
            self.assertEqual(
                pruning_preflight_response.json()["surface"],
                "snn_language_readout_newborn_synapse_pruning_preflight.v1",
            )
            self.assertEqual(
                pruning_executor_response.json()["surface"],
                "snn_language_readout_newborn_synapse_pruning_executor.v1",
            )
            public_payloads = [
                design_response.json(),
                preflight_response.json(),
                executor_response.json(),
                review_response.json(),
                newborn_design_response.json(),
                newborn_preflight_response.json(),
                newborn_executor_response.json(),
                newborn_review_response.json(),
                critical_design_response.json(),
                critical_preflight_response.json(),
                critical_executor_response.json(),
                critical_review_response.json(),
                critical_continuation_response.json(),
                maturation_response.json(),
                pruning_design_response.json(),
                pruning_preflight_response.json(),
                pruning_executor_response.json(),
            ]
            public_text = json.dumps(public_payloads, sort_keys=True)
            self.assertNotIn("autonomous_snn_language_thought", public_text)
            self.assertNotIn("snn_language_autonomous_snn_language_thought", public_text)
            self.assertNotIn("thought_", public_text)
            self.assertFalse(design_response.json()["mutates_runtime_state"])
            self.assertFalse(preflight_response.json()["mutates_runtime_state"])
            self.assertFalse(executor_response.json()["mutates_runtime_state"])
            self.assertFalse(review_response.json()["mutates_runtime_state"])
            self.assertFalse(newborn_design_response.json()["mutates_runtime_state"])
            self.assertFalse(newborn_preflight_response.json()["mutates_runtime_state"])
            self.assertFalse(newborn_review_response.json()["mutates_runtime_state"])
            self.assertFalse(critical_design_response.json()["mutates_runtime_state"])
            self.assertFalse(critical_preflight_response.json()["mutates_runtime_state"])
            self.assertFalse(critical_review_response.json()["mutates_runtime_state"])
            self.assertFalse(critical_continuation_response.json()["mutates_runtime_state"])
            self.assertFalse(maturation_response.json()["mutates_runtime_state"])
            self.assertFalse(pruning_design_response.json()["mutates_runtime_state"])
            self.assertFalse(pruning_preflight_response.json()["mutates_runtime_state"])

    def test_status_and_terminus_endpoints_expose_runtime_truth_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            app = create_app(_build_checkpoint(root, test_case="service_api_runtime_truth"), trace_dir=root / "traces")
            with TestClient(app) as client:
                status_response = client.get("/status")
                terminus_response = client.get("/terminus")
                capacity_expansion_response = client.post(
                    "/terminus/snn-language-sequence/capacity-expansion-design",
                    json={
                        "capacity_pressure": status_response.json()["runtime_truth"][
                            "evidence"
                        ]["snn_language_capacity_pressure"],
                        "device_evidence": {"device": "cpu", "source": "service_api"},
                        "rollback_policy": {
                            "available": False,
                            "snapshot_id": "service-api",
                        },
                    },
                )
                capacity_preflight_response = client.post(
                    "/terminus/snn-language-sequence/capacity-expansion-preflight",
                    json={
                        "capacity_expansion_design": capacity_expansion_response.json(),
                        "expected_state_revision": status_response.json()[
                            "state_revision"
                        ],
                        "checkpoint_transaction": {
                            "checkpoint_path": str(root / "capacity.pt"),
                            "snapshot_id": "service-api",
                            "pre_expansion_checkpoint_saved": True,
                            "pre_expansion_checkpoint_restore_verified": True,
                        },
                        "device_evidence": {"device": "cpu", "source": "service_api"},
                    },
                )
                capacity_compatibility_response = client.post(
                    "/terminus/snn-language-sequence/capacity-resize-compatibility-audit",
                    json={
                        "capacity_expansion_preflight": capacity_preflight_response.json(),
                        "language_capacity_state": {
                            "surface": "snn_language_capacity_state.v1",
                            "language_neuron_count": 64,
                            "sparse_edge_budget": 256,
                            "outgoing_fanout_budget": 16,
                            "capacity_expansion_count": 0,
                        },
                    },
                )
                dense_readout_resize_plan_response = client.post(
                    "/terminus/snn-language-sequence/dense-readout-resize-plan",
                    json={
                        "capacity_pressure": status_response.json()["runtime_truth"][
                            "evidence"
                        ]["snn_language_capacity_pressure"],
                        "fixed_boundaries": status_response.json()["runtime_truth"][
                            "evidence"
                        ]["snn_language_capacity_fixed_boundaries"],
                    },
                )
                dense_readout_resize_preflight_response = client.post(
                    "/terminus/snn-language-sequence/dense-readout-resize-preflight",
                    json={
                        "dense_readout_resize_plan": dense_readout_resize_plan_response.json(),
                        "expected_state_revision": status_response.json()[
                            "state_revision"
                        ],
                        "checkpoint_transaction": {
                            "checkpoint_path": str(root / "dense-readout.pt"),
                            "snapshot_id": "service-api-dense-readout",
                            "pre_resize_checkpoint_saved": True,
                            "pre_resize_checkpoint_restore_verified": True,
                        },
                        "device_evidence": {
                            "device": "cuda:0",
                            "source": "service_api",
                            "requested_cuda_honored": True,
                        },
                    },
                )
                dense_readout_resize_transaction_response = client.post(
                    "/terminus/snn-language-sequence/dense-readout-resize-transaction-proposal",
                    json={
                        "dense_readout_resize_preflight": dense_readout_resize_preflight_response.json(),
                        "expected_state_revision": status_response.json()[
                            "state_revision"
                        ],
                        "operator_id": "service-api",
                        "confirmation": True,
                    },
                )
                dense_readout_resize_readiness_response = client.post(
                    "/terminus/snn-language-sequence/dense-readout-resize-executor-readiness-audit",
                    json={
                        "dense_readout_resize_transaction_proposal": (
                            dense_readout_resize_transaction_response.json()
                        )
                    },
                )
                dense_readout_layout_migration_response = client.post(
                    "/terminus/snn-language-sequence/dense-readout-layout-migration",
                    json={
                        "dense_readout_resize_transaction_proposal": (
                            dense_readout_resize_transaction_response.json()
                        ),
                        "dense_readout_resize_executor_readiness_audit": (
                            dense_readout_resize_readiness_response.json()
                        ),
                        "expected_state_revision": status_response.json()[
                            "state_revision"
                        ],
                        "operator_id": "service-api",
                        "confirmation": True,
                        "checkpoint_path": str(root / "dense-layout.pt"),
                    },
                )
                dense_readout_tensor_materialization_readiness_response = client.post(
                    "/terminus/snn-language-sequence/dense-readout-tensor-materialization-readiness",
                    json={
                        "dense_readout_layout_migration": (
                            dense_readout_layout_migration_response.json()
                        )
                    },
                )
                dense_readout_tensor_materialization_response = client.post(
                    "/terminus/snn-language-sequence/dense-readout-tensor-materialization",
                    json={
                        "dense_readout_tensor_materialization_readiness": (
                            dense_readout_tensor_materialization_readiness_response.json()
                        ),
                        "expected_state_revision": status_response.json()[
                            "state_revision"
                        ],
                        "operator_id": "service-api",
                        "confirmation": True,
                        "checkpoint_path": str(root / "dense-tensor.pt"),
                        "requested_device": "cpu",
                    },
                )
                dense_readout_training_readiness_response = client.post(
                    "/terminus/snn-language-sequence/dense-readout-training-readiness",
                    json={
                        "dense_readout_tensor_integrity": status_response.json()[
                            "runtime_truth"
                        ]["evidence"]["snn_language_dense_readout_tensor_integrity"],
                    },
                )
                dense_readout_training_loop_design_response = client.post(
                    "/terminus/snn-language-sequence/dense-readout-training-loop-design",
                    json={
                        "dense_readout_training_readiness": (
                            dense_readout_training_readiness_response.json()
                        ),
                        "training_plan": {
                            "training_transition_count": 2,
                            "validation_transition_count": 1,
                        },
                        "device_evidence": {"device": "cpu", "source": "service_api"},
                        "rollback_policy": {
                            "checkpoint_available": True,
                            "restore_endpoint_available": True,
                        },
                    },
                )
                dense_readout_training_loop_preflight_response = client.post(
                    "/terminus/snn-language-sequence/dense-readout-training-loop-preflight",
                    json={
                        "dense_readout_training_loop_design": (
                            dense_readout_training_loop_design_response.json()
                        ),
                        "expected_state_revision": status_response.json()[
                            "state_revision"
                        ],
                        "checkpoint_path": str(root / "dense-training.pt"),
                        "executor_capabilities": {
                            "checkpoint_writer_available": True,
                            "bounded_delta_application_available": True,
                        },
                    },
                )
                dense_readout_training_response = client.post(
                    "/terminus/snn-language-sequence/dense-readout-training",
                    json={
                        "dense_readout_training_loop_preflight": (
                            dense_readout_training_loop_preflight_response.json()
                        ),
                        "training_transitions": [
                            {
                                "transition_id": "service-api-training",
                                "pre_indices": [1],
                                "post_indices": [2],
                            }
                        ],
                        "expected_state_revision": status_response.json()[
                            "state_revision"
                        ],
                        "operator_id": "service-api",
                        "confirmation": True,
                        "checkpoint_path": str(root / "dense-training.pt"),
                    },
                )
                dense_readout_post_training_evaluation_response = client.post(
                    "/terminus/snn-language-sequence/dense-readout-post-training-evaluation",
                    json={
                        "dense_readout_training": (
                            dense_readout_training_response.json()
                        ),
                        "dense_readout_tensor_integrity": status_response.json()[
                            "runtime_truth"
                        ]["evidence"]["snn_language_dense_readout_tensor_integrity"],
                    },
                )
                dense_readout_decoder_probe_design_response = client.post(
                    "/terminus/snn-language-sequence/dense-readout-decoder-probe-design",
                    json={
                        "dense_readout_post_training_evaluation": (
                            dense_readout_post_training_evaluation_response.json()
                        ),
                        "readout_slots": [
                            {
                                "label": "prediction error",
                                "pressure_band": "high",
                                "grounded": True,
                            }
                        ],
                        "device_evidence": {"device": "cpu", "source": "service_api"},
                    },
                )
                dense_readout_decoder_probe_preflight_response = client.post(
                    "/terminus/snn-language-sequence/dense-readout-decoder-probe-preflight",
                    json={
                        "dense_readout_decoder_probe_design": (
                            dense_readout_decoder_probe_design_response.json()
                        ),
                        "expected_state_revision": status_response.json()[
                            "state_revision"
                        ],
                        "device_evidence": {"device": "cpu", "source": "service_api"},
                    },
                )
                dense_readout_decoder_probe_response = client.post(
                    "/terminus/snn-language-sequence/dense-readout-decoder-probe",
                    json={
                        "dense_readout_decoder_probe_preflight": (
                            dense_readout_decoder_probe_preflight_response.json()
                        ),
                        "max_candidate_labels": 4,
                    },
                )
                dense_readout_label_candidate_review_response = client.post(
                    "/terminus/snn-language-sequence/dense-readout-label-candidate-review",
                    json={
                        "dense_readout_decoder_probe_execution": (
                            dense_readout_decoder_probe_response.json()
                        ),
                        "operator_id": "service-api",
                        "confirmation": True,
                        "review_note": "service API candidate review",
                    },
                )
                dense_readout_label_candidate_record_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/record-dense-label-candidate-review",
                    json={
                        "dense_readout_label_candidate_review": (
                            dense_readout_label_candidate_review_response.json()
                        ),
                        "expected_state_revision": status_response.json()[
                            "state_revision"
                        ],
                        "operator_id": "service-api",
                        "confirmation": True,
                    },
                )
                dense_label_candidate_history_response = client.get(
                    "/terminus/snn-language-sequence/readout-ledger/dense-label-candidate-history",
                    params={"limit": 4},
                )
                dense_label_candidate_calibration_policy_response = client.get(
                    "/terminus/snn-language-sequence/readout-ledger/dense-label-candidate-calibration-policy",
                    params={"limit": 4},
                )
                dense_label_candidate_calibration_design_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/dense-label-candidate-calibration-evaluation-design",
                    json={
                        "dense_label_candidate_calibration_policy": (
                            dense_label_candidate_calibration_policy_response.json()
                        ),
                        "heldout_label_evidence": {
                            "labels": ["prediction error"],
                            "target_hash": status_response.json()["runtime_truth"][
                                "evidence"
                            ]["snn_language_capacity_pressure"]["surface"],
                        },
                        "device_evidence": {"device": "cpu", "source": "service_api"},
                    },
                )
                dense_label_candidate_calibration_preflight_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/dense-label-candidate-calibration-evaluation-preflight",
                    json={
                        "dense_label_candidate_calibration_evaluation_design": (
                            dense_label_candidate_calibration_design_response.json()
                        ),
                        "expected_state_revision": status_response.json()[
                            "state_revision"
                        ],
                        "device_evidence": {"device": "cpu", "source": "service_api"},
                        "executor_capabilities": {
                            "calibration_evaluation_executor": False
                        },
                    },
                )
                dense_label_candidate_calibration_evaluation_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/dense-label-candidate-calibration-evaluation",
                    json={
                        "dense_label_candidate_calibration_evaluation_preflight": (
                            dense_label_candidate_calibration_preflight_response.json()
                        ),
                        "heldout_label_evidence": {"labels": ["prediction error"]},
                        "bin_count": 5,
                    },
                )
                dense_label_candidate_calibration_review_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/dense-label-candidate-calibration-evaluation-review",
                    json={
                        "dense_label_candidate_calibration_evaluation": (
                            dense_label_candidate_calibration_evaluation_response.json()
                        ),
                        "expected_state_revision": status_response.json()[
                            "state_revision"
                        ],
                        "operator_id": "service-api",
                        "confirmation": True,
                    },
                )
                dense_label_candidate_calibration_update_design_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/dense-label-candidate-calibration-update-design",
                    json={
                        "dense_label_candidate_calibration_evaluation_review": (
                            dense_label_candidate_calibration_review_response.json()
                        ),
                        "update_policy": {"method": "bounded_temperature_scaling"},
                        "rollback_policy": {
                            "available": True,
                            "snapshot_id": "service-api-calibration",
                        },
                        "device_evidence": {"device": "cpu", "source": "service_api"},
                    },
                )
                dense_label_candidate_calibration_update_preflight_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/dense-label-candidate-calibration-update-preflight",
                    json={
                        "dense_label_candidate_calibration_update_design": (
                            dense_label_candidate_calibration_update_design_response.json()
                        ),
                        "expected_state_revision": status_response.json()[
                            "state_revision"
                        ],
                        "checkpoint_path": "checkpoints/service-api-calibration.json",
                        "device_evidence": {"device": "cpu", "source": "service_api"},
                        "executor_capabilities": {
                            "calibration_update_executor": False
                        },
                    },
                )
                dense_label_candidate_calibration_update_application_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/dense-label-candidate-calibration-update-application",
                    json={
                        "dense_label_candidate_calibration_update_preflight": (
                            dense_label_candidate_calibration_update_preflight_response.json()
                        ),
                        "expected_state_revision": status_response.json()[
                            "state_revision"
                        ],
                        "operator_id": "service-api",
                        "confirmation": True,
                    },
                )
                dense_label_candidate_calibration_update_application_review_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/dense-label-candidate-calibration-update-application-review",
                    json={
                        "dense_label_candidate_calibration_update_application": (
                            dense_label_candidate_calibration_update_application_response.json()
                        ),
                        "expected_state_revision": status_response.json()[
                            "state_revision"
                        ],
                    },
                )
                dense_label_candidate_post_calibration_observation_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/dense-label-candidate-post-calibration-observation-window",
                    json={
                        "dense_label_candidate_calibration_update_application_review": (
                            dense_label_candidate_calibration_update_application_review_response.json()
                        ),
                        "observation_evidence": {
                            "samples": [
                                {
                                    "sample_hash": "a" * 64,
                                    "label_hash": "b" * 64,
                                    "pre_calibration_confidence": 0.8,
                                    "calibrated_confidence": 0.75,
                                    "correct": True,
                                }
                            ]
                        },
                        "expected_state_revision": status_response.json()[
                            "state_revision"
                        ],
                        "window_policy": {"min_samples": 3},
                    },
                )
                dense_label_candidate_post_calibration_operator_review_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/dense-label-candidate-post-calibration-operator-review",
                    json={
                        "dense_label_candidate_post_calibration_observation_window": (
                            dense_label_candidate_post_calibration_observation_response.json()
                        ),
                        "expected_state_revision": status_response.json()[
                            "state_revision"
                        ],
                        "operator_id": "service-api",
                        "confirmation": True,
                    },
                )
                calibrated_dense_label_confidence_use_design_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/calibrated-dense-label-confidence-use-design",
                    json={
                        "dense_label_candidate_post_calibration_operator_review": (
                            dense_label_candidate_post_calibration_operator_review_response.json()
                        ),
                        "confidence_use_policy": {
                            "use_mode": "threshold_and_abstain",
                            "min_confidence_threshold": 0.6,
                        },
                        "device_evidence": {"device": "cpu", "source": "service_api"},
                    },
                )
                calibrated_dense_label_confidence_use_preflight_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/calibrated-dense-label-confidence-use-preflight",
                    json={
                        "dense_label_confidence_use_design": (
                            calibrated_dense_label_confidence_use_design_response.json()
                        ),
                        "expected_state_revision": status_response.json()[
                            "state_revision"
                        ],
                        "candidate_evidence": {
                            "candidates": [
                                {
                                    "dense_label_candidate_evidence_hash": "c" * 64,
                                    "label_hash": "d" * 64,
                                    "calibrated_confidence": 0.7,
                                    "pre_calibration_confidence": 0.8,
                                }
                            ]
                        },
                        "device_evidence": {"device": "cpu", "source": "service_api"},
                        "executor_capabilities": {
                            "calibrated_confidence_use_executor": True
                        },
                    },
                )
                calibrated_dense_label_confidence_use_executor_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/calibrated-dense-label-confidence-use-executor",
                    json={
                        "calibrated_dense_label_confidence_use_preflight": (
                            calibrated_dense_label_confidence_use_preflight_response.json()
                        ),
                        "expected_state_revision": status_response.json()[
                            "state_revision"
                        ],
                        "candidate_evidence": {
                            "candidates": [
                                {
                                    "dense_label_candidate_evidence_hash": "c" * 64,
                                    "label_hash": "d" * 64,
                                    "calibrated_confidence": 0.7,
                                    "pre_calibration_confidence": 0.8,
                                }
                            ]
                        },
                        "execution_policy": {"max_selected_candidates": 1},
                    },
                )
                calibrated_dense_label_confidence_operator_display_review_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/calibrated-dense-label-confidence-operator-display-review",
                    json={
                        "calibrated_dense_label_confidence_use_executor": (
                            calibrated_dense_label_confidence_use_executor_response.json()
                        ),
                        "expected_state_revision": status_response.json()[
                            "state_revision"
                        ],
                        "operator_id": "service-api",
                        "confirmation": True,
                    },
                )
                calibrated_dense_label_confidence_internal_stability_review_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/calibrated-dense-label-confidence-internal-stability-review",
                    json={
                        "calibrated_dense_label_confidence_use_executor": (
                            calibrated_dense_label_confidence_use_executor_response.json()
                        ),
                        "expected_state_revision": status_response.json()[
                            "state_revision"
                        ],
                        "stability_evidence": {
                            "cycles": [
                                {
                                    "selected_candidate_hashes": ["c" * 64],
                                    "selected_confidence": 0.7,
                                }
                            ]
                        },
                        "review_policy": {"min_cycles": 3},
                    },
                )
                calibrated_dense_label_confidence_autonomous_replay_design_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/calibrated-dense-label-confidence-autonomous-replay-review-design",
                    json={
                        "calibrated_dense_label_confidence_internal_stability_review": (
                            calibrated_dense_label_confidence_internal_stability_review_response.json()
                        ),
                        "replay_policy": {"max_replay_cycles": 4},
                        "device_evidence": {"device": "cpu", "source": "service_api"},
                    },
                )
                calibrated_dense_label_confidence_autonomous_replay_preflight_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/calibrated-dense-label-confidence-autonomous-replay-review-preflight",
                    json={
                        "calibrated_dense_label_confidence_autonomous_replay_review_design": (
                            calibrated_dense_label_confidence_autonomous_replay_design_response.json()
                        ),
                        "expected_state_revision": status_response.json()[
                            "state_revision"
                        ],
                        "device_evidence": {"device": "cpu", "source": "service_api"},
                        "executor_capabilities": {
                            "autonomous_confidence_replay_review_executor": False
                        },
                    },
                )
                calibrated_dense_label_confidence_autonomous_replay_executor_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/calibrated-dense-label-confidence-autonomous-replay-review-executor",
                    json={
                        "calibrated_dense_label_confidence_autonomous_replay_review_preflight": (
                            calibrated_dense_label_confidence_autonomous_replay_preflight_response.json()
                        ),
                        "expected_state_revision": status_response.json()[
                            "state_revision"
                        ],
                        "replay_cycle_evidence": {
                            "cycles": [
                                {
                                    "cycle_index": 0,
                                    "selected_candidate_hashes": ["c" * 64],
                                    "selected_confidence": 0.7,
                                }
                            ]
                        },
                    },
                )
                calibrated_dense_label_confidence_autonomous_recalibration_design_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/calibrated-dense-label-confidence-autonomous-recalibration-design",
                    json={
                        "calibrated_dense_label_confidence_autonomous_replay_review_executor": (
                            calibrated_dense_label_confidence_autonomous_replay_executor_response.json()
                        ),
                        "recalibration_policy": {
                            "method": "bounded_temperature_scaling",
                            "max_temperature_delta": 0.05,
                            "max_confidence_rescale_delta": 0.05,
                        },
                        "rollback_policy": {
                            "can_restore_previous_calibration": True
                        },
                        "device_evidence": {"device": "cpu", "source": "service_api"},
                    },
                )
                calibrated_dense_label_confidence_autonomous_recalibration_preflight_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/calibrated-dense-label-confidence-autonomous-recalibration-preflight",
                    json={
                        "calibrated_dense_label_confidence_autonomous_recalibration_design": (
                            calibrated_dense_label_confidence_autonomous_recalibration_design_response.json()
                        ),
                        "expected_state_revision": status_response.json()[
                            "state_revision"
                        ],
                        "device_evidence": {"device": "cpu", "source": "service_api"},
                        "executor_capabilities": {
                            "autonomous_confidence_recalibration_executor": False
                        },
                    },
                )
                calibrated_dense_label_confidence_autonomous_recalibration_executor_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/calibrated-dense-label-confidence-autonomous-recalibration-executor",
                    json={
                        "calibrated_dense_label_confidence_autonomous_recalibration_preflight": (
                            calibrated_dense_label_confidence_autonomous_recalibration_preflight_response.json()
                        ),
                        "expected_state_revision": status_response.json()[
                            "state_revision"
                        ],
                    },
                )
                calibrated_dense_label_confidence_autonomous_recalibration_application_review_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/calibrated-dense-label-confidence-autonomous-recalibration-application-review",
                    json={
                        "calibrated_dense_label_confidence_autonomous_recalibration_executor": (
                            calibrated_dense_label_confidence_autonomous_recalibration_executor_response.json()
                        ),
                        "expected_state_revision": status_response.json()[
                            "state_revision"
                        ],
                        "review_policy": {
                            "max_temperature_delta": 0.05,
                            "max_confidence_rescale_delta": 0.05,
                        },
                    },
                )
                calibrated_dense_label_confidence_autonomous_post_calibration_observation_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/calibrated-dense-label-confidence-autonomous-post-calibration-observation-window",
                    json={
                        "calibrated_dense_label_confidence_autonomous_recalibration_application_review": (
                            calibrated_dense_label_confidence_autonomous_recalibration_application_review_response.json()
                        ),
                        "observation_evidence": {
                            "samples": [
                                {
                                    "sample_hash": "s" * 64,
                                    "label_hash": "l" * 64,
                                    "pre_calibration_confidence": 0.8,
                                    "calibrated_confidence": 0.7,
                                    "correct": True,
                                }
                            ]
                        },
                        "expected_state_revision": status_response.json()[
                            "state_revision"
                        ],
                        "window_policy": {"min_samples": 3},
                    },
                )
                calibrated_dense_label_confidence_autonomous_post_calibration_stability_review_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/calibrated-dense-label-confidence-autonomous-post-calibration-stability-review",
                    json={
                        "calibrated_dense_label_confidence_autonomous_post_calibration_observation_window": (
                            calibrated_dense_label_confidence_autonomous_post_calibration_observation_response.json()
                        ),
                        "expected_state_revision": status_response.json()[
                            "state_revision"
                        ],
                        "stability_policy": {"min_samples": 3},
                    },
                )
                calibrated_dense_label_confidence_autonomous_use_design_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/calibrated-dense-label-confidence-autonomous-use-design",
                    json={
                        "calibrated_dense_label_confidence_autonomous_post_calibration_stability_review": (
                            calibrated_dense_label_confidence_autonomous_post_calibration_stability_review_response.json()
                        ),
                        "confidence_use_policy": {
                            "use_mode": "threshold_and_abstain",
                            "min_confidence_threshold": 0.6,
                            "max_candidates": 4,
                        },
                        "device_evidence": {"device": "cpu", "source": "service_api"},
                    },
                )
                calibrated_dense_label_confidence_autonomous_use_preflight_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/calibrated-dense-label-confidence-autonomous-use-preflight",
                    json={
                        "calibrated_dense_label_confidence_autonomous_use_design": (
                            calibrated_dense_label_confidence_autonomous_use_design_response.json()
                        ),
                        "expected_state_revision": status_response.json()[
                            "state_revision"
                        ],
                        "candidate_evidence": {
                            "candidates": [
                                {
                                    "dense_label_candidate_evidence_hash": "c" * 64,
                                    "label_hash": "l" * 64,
                                    "calibrated_confidence": 0.7,
                                    "pre_calibration_confidence": 0.8,
                                }
                            ]
                        },
                        "device_evidence": {"device": "cpu", "source": "service_api"},
                        "executor_capabilities": {
                            "autonomous_calibrated_confidence_use_executor": False
                        },
                    },
                )
                calibrated_dense_label_confidence_autonomous_use_executor_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/calibrated-dense-label-confidence-autonomous-use-executor",
                    json={
                        "calibrated_dense_label_confidence_autonomous_use_preflight": (
                            calibrated_dense_label_confidence_autonomous_use_preflight_response.json()
                        ),
                        "expected_state_revision": status_response.json()[
                            "state_revision"
                        ],
                        "candidate_evidence": {
                            "candidates": [
                                {
                                    "dense_label_candidate_evidence_hash": "c" * 64,
                                    "label_hash": "l" * 64,
                                    "calibrated_confidence": 0.7,
                                    "pre_calibration_confidence": 0.8,
                                }
                            ]
                        },
                        "execution_policy": {"max_selected_candidates": 1},
                    },
                )
                calibrated_dense_label_confidence_autonomous_use_event_review_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/calibrated-dense-label-confidence-autonomous-use-event-review",
                    json={
                        "calibrated_dense_label_confidence_autonomous_use_executor": (
                            calibrated_dense_label_confidence_autonomous_use_executor_response.json()
                        ),
                        "expected_state_revision": status_response.json()[
                            "state_revision"
                        ],
                        "review_policy": {
                            "min_selected_candidates": 1,
                            "max_selected_candidates": 2,
                        },
                    },
                )
                autonomous_hash_readout_binding_design_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/autonomous-hash-readout-binding-design",
                    json={
                        "calibrated_dense_label_confidence_autonomous_use_event_review": (
                            calibrated_dense_label_confidence_autonomous_use_event_review_response.json()
                        ),
                        "readout_vocabulary_slots": [
                            {
                                "label": "service concept",
                                "pressure_band": "medium",
                                "grounded": True,
                                "slot_id": "slot-service-concept",
                            }
                        ],
                        "binding_policy": {"max_bindings": 1},
                        "device_evidence": {"device": "cpu", "source": "service_api"},
                    },
                )
                autonomous_hash_readout_binding_preflight_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/autonomous-hash-readout-binding-preflight",
                    json={
                        "autonomous_hash_readout_binding_design": (
                            autonomous_hash_readout_binding_design_response.json()
                        ),
                        "expected_state_revision": status_response.json()[
                            "state_revision"
                        ],
                        "device_evidence": {"device": "cpu", "source": "service_api"},
                        "executor_capabilities": {
                            "autonomous_hash_readout_binding_executor": False
                        },
                    },
                )
                autonomous_hash_readout_binding_executor_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/autonomous-hash-readout-binding-executor",
                    json={
                        "autonomous_hash_readout_binding_preflight": (
                            autonomous_hash_readout_binding_preflight_response.json()
                        ),
                        "expected_state_revision": status_response.json()[
                            "state_revision"
                        ],
                        "execution_policy": {"max_commit_bindings": 1},
                    },
                )
                autonomous_hash_readout_binding_event_review_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/autonomous-hash-readout-binding-event-review",
                    json={
                        "autonomous_hash_readout_binding_executor": (
                            autonomous_hash_readout_binding_executor_response.json()
                        ),
                        "expected_state_revision": status_response.json()[
                            "state_revision"
                        ],
                        "review_policy": {"min_bindings": 1, "max_bindings": 2},
                    },
                )
                autonomous_bound_readout_observation_design_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/autonomous-bound-readout-observation-design",
                    json={
                        "autonomous_hash_readout_binding_event_review": (
                            autonomous_hash_readout_binding_event_review_response.json()
                        ),
                        "observation_policy": {
                            "observation_cycles": 4,
                            "min_activation_sparsity": 0.5,
                            "max_slot_drift": 0.15,
                            "min_binding_reactivation": 0.5,
                        },
                        "device_evidence": {"device": "cpu", "source": "service_api"},
                    },
                )
                autonomous_bound_readout_observation_preflight_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/autonomous-bound-readout-observation-preflight",
                    json={
                        "autonomous_bound_readout_observation_design": (
                            autonomous_bound_readout_observation_design_response.json()
                        ),
                        "expected_state_revision": status_response.json()[
                            "state_revision"
                        ],
                        "device_evidence": {"device": "cpu", "source": "service_api"},
                        "executor_capabilities": {
                            "autonomous_bound_readout_observation_executor": False
                        },
                    },
                )
                autonomous_bound_readout_observation_executor_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/autonomous-bound-readout-observation-executor",
                    json={
                        "autonomous_bound_readout_observation_preflight": (
                            autonomous_bound_readout_observation_preflight_response.json()
                        ),
                        "expected_state_revision": status_response.json()[
                            "state_revision"
                        ],
                        "observation_evidence": {"samples": []},
                        "execution_policy": {"max_samples": 4},
                    },
                )
                autonomous_bound_readout_observation_event_review_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/autonomous-bound-readout-observation-event-review",
                    json={
                        "autonomous_bound_readout_observation_executor": (
                            autonomous_bound_readout_observation_executor_response.json()
                        ),
                        "expected_state_revision": status_response.json()[
                            "state_revision"
                        ],
                        "review_policy": {
                            "min_samples": 4,
                            "max_samples": 4,
                            "min_activation_sparsity": 0.5,
                            "max_slot_drift": 0.15,
                            "min_binding_reactivation": 0.5,
                        },
                    },
                )
                autonomous_readout_training_window_design_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/autonomous-readout-training-window-design",
                    json={
                        "autonomous_bound_readout_observation_event_review": (
                            autonomous_bound_readout_observation_event_review_response.json()
                        ),
                        "training_policy": {
                            "training_window_steps": 4,
                            "truncated_bptt_steps": 4,
                            "micro_batch_size": 1,
                            "max_learning_rate": 0.0003,
                            "learning_rule": "surrogate_gradient",
                            "use_spike_compression": True,
                            "use_gradient_checkpointing": True,
                        },
                        "device_evidence": {
                            "device": "cpu",
                            "source": "service_api",
                        },
                    },
                )
                autonomous_readout_training_window_preflight_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/autonomous-readout-training-window-preflight",
                    json={
                        "autonomous_readout_training_window_design": (
                            autonomous_readout_training_window_design_response.json()
                        ),
                        "expected_state_revision": status_response.json()[
                            "state_revision"
                        ],
                        "device_evidence": {
                            "device": "cpu",
                            "source": "service_api",
                        },
                        "executor_capabilities": {
                            "autonomous_readout_training_window_executor": False
                        },
                    },
                )
                autonomous_readout_training_window_executor_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/autonomous-readout-training-window-executor",
                    json={
                        "autonomous_readout_training_window_preflight": (
                            autonomous_readout_training_window_preflight_response.json()
                        ),
                        "expected_state_revision": status_response.json()[
                            "state_revision"
                        ],
                        "training_evidence": {
                            "sample_hashes": [],
                            "runtime_weights_updated": False,
                        },
                        "execution_policy": {
                            "max_loss_increase": 0.02,
                            "max_gradient_norm": 10.0,
                            "max_weight_delta": 0.05,
                            "min_spike_sparsity": 0.5,
                        },
                    },
                )
                autonomous_readout_training_window_event_review_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/autonomous-readout-training-window-event-review",
                    json={
                        "autonomous_readout_training_window_executor": (
                            autonomous_readout_training_window_executor_response.json()
                        ),
                        "expected_state_revision": status_response.json()[
                            "state_revision"
                        ],
                        "review_policy": {
                            "max_loss_increase": 0.02,
                            "max_gradient_norm": 10.0,
                            "max_weight_delta": 0.05,
                            "min_spike_sparsity": 0.5,
                        },
                    },
                )
                autonomous_decoder_probe_design_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/autonomous-decoder-probe-design",
                    json={
                        "autonomous_readout_training_window_event_review": (
                            autonomous_readout_training_window_event_review_response.json()
                        ),
                        "probe_policy": {
                            "probe_mode": "hash_rank_probe",
                            "max_probe_steps": 4,
                            "top_k": 1,
                            "min_spike_sparsity": 0.5,
                            "max_slot_drift": 0.2,
                        },
                        "device_evidence": {
                            "device": "cpu",
                            "source": "service_api",
                        },
                    },
                )
                autonomous_decoder_probe_preflight_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/autonomous-decoder-probe-preflight",
                    json={
                        "autonomous_decoder_probe_design": (
                            autonomous_decoder_probe_design_response.json()
                        ),
                        "expected_state_revision": status_response.json()[
                            "state_revision"
                        ],
                        "device_evidence": {
                            "device": "cpu",
                            "source": "service_api",
                        },
                        "executor_capabilities": {
                            "autonomous_decoder_probe_executor": False
                        },
                    },
                )
                autonomous_decoder_probe_executor_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/autonomous-decoder-probe-executor",
                    json={
                        "autonomous_decoder_probe_preflight": (
                            autonomous_decoder_probe_preflight_response.json()
                        ),
                        "expected_state_revision": status_response.json()[
                            "state_revision"
                        ],
                        "probe_evidence": {
                            "probe_results": [],
                            "checkpoint_written": False,
                        },
                        "execution_policy": {"min_top_score": 0.5},
                    },
                )
                autonomous_decoder_probe_event_review_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/autonomous-decoder-probe-event-review",
                    json={
                        "autonomous_decoder_probe_executor": (
                            autonomous_decoder_probe_executor_response.json()
                        ),
                        "expected_state_revision": status_response.json()[
                            "state_revision"
                        ],
                        "review_policy": {
                            "min_top_score": 0.5,
                            "min_spike_sparsity": 0.5,
                            "max_slot_drift": 0.2,
                        },
                    },
                )
                autonomous_language_output_design_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/autonomous-language-output-design",
                    json={
                        "autonomous_decoder_probe_event_review": (
                            autonomous_decoder_probe_event_review_response.json()
                        ),
                        "output_policy": {
                            "output_mode": "token_hash_sequence",
                            "max_output_tokens": 3,
                            "min_top_score": 0.5,
                            "min_spike_sparsity": 0.5,
                            "max_slot_drift": 0.2,
                        },
                        "device_evidence": {
                            "device": "cpu",
                            "source": "service_api",
                        },
                    },
                )
                autonomous_language_output_preflight_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/autonomous-language-output-preflight",
                    json={
                        "autonomous_language_output_design": (
                            autonomous_language_output_design_response.json()
                        ),
                        "expected_state_revision": status_response.json()[
                            "state_revision"
                        ],
                        "device_evidence": {
                            "device": "cpu",
                            "source": "service_api",
                        },
                        "executor_capabilities": {
                            "autonomous_language_output_executor": False
                        },
                    },
                )
                autonomous_language_output_executor_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/autonomous-language-output-executor",
                    json={
                        "autonomous_language_output_preflight": (
                            autonomous_language_output_preflight_response.json()
                        ),
                        "expected_state_revision": status_response.json()[
                            "state_revision"
                        ],
                        "output_evidence": {
                            "output_slot_results": [],
                            "checkpoint_written": False,
                        },
                        "execution_policy": {
                            "min_confidence_score": 0.5,
                            "min_spike_sparsity": 0.5,
                            "max_slot_drift": 0.2,
                        },
                    },
                )
                autonomous_language_output_event_review_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/autonomous-language-output-event-review",
                    json={
                        "autonomous_language_output_executor": (
                            autonomous_language_output_executor_response.json()
                        ),
                        "expected_state_revision": status_response.json()[
                            "state_revision"
                        ],
                        "review_policy": {
                            "min_confidence_score": 0.5,
                            "min_spike_sparsity": 0.5,
                            "max_slot_drift": 0.2,
                        },
                    },
                )
                autonomous_decoded_output_design_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/autonomous-decoded-output-design",
                    json={
                        "autonomous_language_output_event_review": (
                            autonomous_language_output_event_review_response.json()
                        ),
                        "vocabulary_binding": {
                            "token_candidate_hashes": [],
                            "token_vocabulary_hash": "",
                            "tokenizer_hash": "",
                            "decode_constraint_hash": "",
                        },
                        "decode_policy": {
                            "decode_mode": "constrained_token_hash_map",
                            "max_decoded_tokens": 3,
                            "min_confidence_score": 0.5,
                            "min_spike_sparsity": 0.5,
                            "max_slot_drift": 0.2,
                        },
                        "device_evidence": {
                            "device": "cpu",
                            "source": "service_api",
                        },
                    },
                )
                autonomous_decoded_output_preflight_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/autonomous-decoded-output-preflight",
                    json={
                        "autonomous_decoded_output_design": (
                            autonomous_decoded_output_design_response.json()
                        ),
                        "expected_state_revision": status_response.json()[
                            "state_revision"
                        ],
                        "device_evidence": {
                            "device": "cpu",
                            "source": "service_api",
                        },
                        "executor_capabilities": {
                            "autonomous_decoded_output_executor": False
                        },
                    },
                )
                autonomous_decoded_output_executor_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/autonomous-decoded-output-executor",
                    json={
                        "autonomous_decoded_output_preflight": (
                            autonomous_decoded_output_preflight_response.json()
                        ),
                        "expected_state_revision": status_response.json()[
                            "state_revision"
                        ],
                        "decode_evidence": {
                            "decoded_token_results": [],
                            "checkpoint_written": False,
                        },
                        "execution_policy": {
                            "min_confidence_score": 0.5,
                            "min_spike_sparsity": 0.5,
                            "max_slot_drift": 0.2,
                        },
                    },
                )
                autonomous_decoded_output_event_review_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/autonomous-decoded-output-event-review",
                    json={
                        "autonomous_decoded_output_executor": (
                            autonomous_decoded_output_executor_response.json()
                        ),
                        "expected_state_revision": status_response.json()[
                            "state_revision"
                        ],
                        "review_policy": {
                            "min_confidence_score": 0.5,
                            "min_spike_sparsity": 0.5,
                            "max_slot_drift": 0.2,
                        },
                    },
                )
                autonomous_bounded_text_emission_design_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/autonomous-bounded-text-emission-design",
                    json={
                        "autonomous_decoded_output_event_review": (
                            autonomous_decoded_output_event_review_response.json()
                        ),
                        "text_surface_binding": {
                            "text_fragment_hashes": [],
                            "text_surface_schema_hash": "",
                            "text_normalizer_hash": "",
                            "semantic_constraint_hash": "",
                        },
                        "emission_policy": {
                            "emission_mode": "bounded_text_hash_sequence",
                            "max_text_fragments": 3,
                            "min_confidence_score": 0.5,
                            "min_spike_sparsity": 0.5,
                            "max_slot_drift": 0.2,
                        },
                        "device_evidence": {
                            "device": "cpu",
                            "source": "service_api",
                        },
                    },
                )
                autonomous_bounded_text_emission_preflight_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/autonomous-bounded-text-emission-preflight",
                    json={
                        "autonomous_bounded_text_emission_design": (
                            autonomous_bounded_text_emission_design_response.json()
                        ),
                        "expected_state_revision": status_response.json()[
                            "state_revision"
                        ],
                        "device_evidence": {
                            "device": "cpu",
                            "source": "service_api",
                        },
                        "executor_capabilities": {
                            "autonomous_bounded_text_emission_executor": False,
                        },
                    },
                )
                autonomous_bounded_text_emission_executor_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/autonomous-bounded-text-emission-executor",
                    json={
                        "autonomous_bounded_text_emission_preflight": (
                            autonomous_bounded_text_emission_preflight_response.json()
                        ),
                        "expected_state_revision": status_response.json()[
                            "state_revision"
                        ],
                        "emission_evidence": {
                            "text_emission_results": [],
                            "checkpoint_written": False,
                        },
                        "execution_policy": {
                            "min_confidence_score": 0.5,
                            "min_spike_sparsity": 0.5,
                            "max_slot_drift": 0.2,
                        },
                    },
                )
                autonomous_bounded_text_emission_event_review_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/autonomous-bounded-text-emission-event-review",
                    json={
                        "autonomous_bounded_text_emission_executor": (
                            autonomous_bounded_text_emission_executor_response.json()
                        ),
                        "expected_state_revision": status_response.json()[
                            "state_revision"
                        ],
                        "review_policy": {
                            "min_text_fragments": 1,
                            "max_text_fragments": 3,
                            "min_confidence_score": 0.5,
                            "min_spike_sparsity": 0.5,
                            "max_slot_drift": 0.2,
                        },
                    },
                )
                autonomous_text_surface_sequence_review_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/autonomous-text-surface-sequence-review",
                    json={
                        "autonomous_bounded_text_emission_event_review": (
                            autonomous_bounded_text_emission_event_review_response.json()
                        ),
                        "sequence_policy": {
                            "sequence_mode": "bounded_hash_fragment_sequence",
                            "min_text_fragments": 1,
                            "max_text_fragments": 3,
                            "min_confidence_score": 0.5,
                            "min_spike_sparsity": 0.5,
                            "max_slot_drift": 0.2,
                        },
                    },
                )
                autonomous_text_surface_commit_design_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/autonomous-text-surface-commit-design",
                    json={
                        "autonomous_text_surface_sequence_review": (
                            autonomous_text_surface_sequence_review_response.json()
                        ),
                        "commit_policy": {
                            "commit_scope": "hash_surface_state",
                            "retention_class": "ephemeral_hash_surface",
                            "min_text_fragments": 1,
                            "max_text_fragments": 3,
                        },
                    },
                )
                autonomous_text_surface_commit_preflight_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/autonomous-text-surface-commit-preflight",
                    json={
                        "autonomous_text_surface_commit_design": (
                            autonomous_text_surface_commit_design_response.json()
                        ),
                        "expected_state_revision": status_response.json()[
                            "state_revision"
                        ],
                        "device_evidence": {
                            "device": "cpu",
                            "source": "service_api",
                        },
                        "executor_capabilities": {
                            "autonomous_text_surface_commit_executor": False,
                        },
                    },
                )
                autonomous_text_surface_commit_executor_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/autonomous-text-surface-commit-executor",
                    json={
                        "autonomous_text_surface_commit_preflight": (
                            autonomous_text_surface_commit_preflight_response.json()
                        ),
                        "expected_state_revision": status_response.json()[
                            "state_revision"
                        ],
                        "commit_evidence": {
                            "checkpoint_written": False,
                        },
                        "execution_policy": {
                            "max_text_fragments": 3,
                        },
                    },
                )
                autonomous_text_surface_commit_event_review_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/autonomous-text-surface-commit-event-review",
                    json={
                        "autonomous_text_surface_commit_executor": (
                            autonomous_text_surface_commit_executor_response.json()
                        ),
                        "expected_state_revision": status_response.json()[
                            "state_revision"
                        ],
                        "review_policy": {
                            "max_text_fragments": 3,
                        },
                    },
                )
                autonomous_text_surface_materialization_design_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/autonomous-text-surface-materialization-design",
                    json={
                        "autonomous_text_surface_commit_event_review": (
                            autonomous_text_surface_commit_event_review_response.json()
                        ),
                        "materialization_policy": {
                            "max_text_fragments": 3,
                            "max_surface_chars": 256,
                        },
                    },
                )
                autonomous_text_surface_materialization_preflight_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/autonomous-text-surface-materialization-preflight",
                    json={
                        "autonomous_text_surface_materialization_design": (
                            autonomous_text_surface_materialization_design_response.json()
                        ),
                        "expected_state_revision": status_response.json()[
                            "state_revision"
                        ],
                        "device_evidence": {
                            "device": "cpu",
                            "source": "service_api",
                        },
                        "executor_capabilities": {
                            "autonomous_text_surface_materialization_executor": False,
                        },
                    },
                )
                autonomous_text_surface_materialization_executor_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/autonomous-text-surface-materialization-executor",
                    json={
                        "autonomous_text_surface_materialization_preflight": (
                            autonomous_text_surface_materialization_preflight_response.json()
                        ),
                        "expected_state_revision": status_response.json()[
                            "state_revision"
                        ],
                        "materialization_evidence": {
                            "text_fragments": [
                                "spike trace stable",
                                "hash surface committed",
                                "bounded output ready",
                            ],
                            "checkpoint_written": False,
                        },
                        "execution_policy": {
                            "max_text_fragments": 3,
                            "max_surface_chars": 256,
                        },
                    },
                )
                autonomous_text_surface_materialization_event_review_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/autonomous-text-surface-materialization-event-review",
                    json={
                        "autonomous_text_surface_materialization_executor": (
                            autonomous_text_surface_materialization_executor_response.json()
                        ),
                        "expected_state_revision": status_response.json()[
                            "state_revision"
                        ],
                        "review_policy": {
                            "max_text_fragments": 3,
                            "max_surface_chars": 256,
                        },
                    },
                )
                autonomous_bounded_language_surface_review_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/autonomous-bounded-language-surface-review",
                    json={
                        "autonomous_text_surface_materialization_event_review": (
                            autonomous_text_surface_materialization_event_review_response.json()
                        ),
                        "language_surface_policy": {
                            "max_text_fragments": 3,
                            "max_surface_chars": 256,
                        },
                    },
                )
                autonomous_bounded_language_surface_commit_design_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/autonomous-bounded-language-surface-commit-design",
                    json={
                        "autonomous_bounded_language_surface_review": (
                            autonomous_bounded_language_surface_review_response.json()
                        ),
                        "commit_policy": {
                            "commit_scope": "bounded_language_surface",
                            "retention_class": "ephemeral_language_surface",
                            "max_surface_chars": 256,
                        },
                    },
                )
                autonomous_bounded_language_surface_commit_preflight_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/autonomous-bounded-language-surface-commit-preflight",
                    json={
                        "autonomous_bounded_language_surface_commit_design": (
                            autonomous_bounded_language_surface_commit_design_response.json()
                        ),
                        "expected_state_revision": status_response.json()[
                            "state_revision"
                        ],
                        "device_evidence": {"device": "cpu", "source": "api-test"},
                        "executor_capabilities": {
                            "autonomous_bounded_language_surface_commit_executor": False
                        },
                    },
                )
                autonomous_bounded_language_surface_commit_executor_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/autonomous-bounded-language-surface-commit-executor",
                    json={
                        "autonomous_bounded_language_surface_commit_preflight": (
                            autonomous_bounded_language_surface_commit_preflight_response.json()
                        ),
                        "expected_state_revision": status_response.json()[
                            "state_revision"
                        ],
                        "commit_evidence": {"checkpoint_written": False},
                        "execution_policy": {
                            "max_text_fragments": 3,
                            "max_surface_chars": 256,
                        },
                    },
                )
                autonomous_bounded_language_surface_commit_event_review_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/autonomous-bounded-language-surface-commit-event-review",
                    json={
                        "autonomous_bounded_language_surface_commit_executor": (
                            autonomous_bounded_language_surface_commit_executor_response.json()
                        ),
                        "expected_state_revision": status_response.json()[
                            "state_revision"
                        ],
                        "review_policy": {
                            "max_text_fragments": 3,
                            "max_surface_chars": 256,
                        },
                    },
                )
                autonomous_bounded_language_surface_use_review_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/autonomous-bounded-language-surface-use-review",
                    json={
                        "autonomous_bounded_language_surface_commit_event_review": (
                            autonomous_bounded_language_surface_commit_event_review_response.json()
                        ),
                        "use_policy": {
                            "language_use_scope": "bounded_language_evidence",
                            "max_surface_chars": 256,
                        },
                    },
                )
                autonomous_bounded_language_surface_use_preflight_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/autonomous-bounded-language-surface-use-preflight",
                    json={
                        "autonomous_bounded_language_surface_use_review": (
                            autonomous_bounded_language_surface_use_review_response.json()
                        ),
                        "expected_state_revision": status_response.json()[
                            "state_revision"
                        ],
                        "device_evidence": {"device": "cpu", "source": "api-test"},
                        "executor_capabilities": {
                            "autonomous_bounded_language_surface_use_executor": False
                        },
                    },
                )
                autonomous_bounded_language_surface_use_executor_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/autonomous-bounded-language-surface-use-executor",
                    json={
                        "autonomous_bounded_language_surface_use_preflight": (
                            autonomous_bounded_language_surface_use_preflight_response.json()
                        ),
                        "expected_state_revision": status_response.json()[
                            "state_revision"
                        ],
                        "use_evidence": {
                            "use_mode": "bounded_language_evidence_observation",
                            "checkpoint_written": False,
                        },
                        "execution_policy": {
                            "max_text_fragments": 3,
                            "max_surface_chars": 256,
                        },
                    },
                )
                autonomous_bounded_language_surface_use_event_review_response = (
                    client.post(
                        "/terminus/snn-language-sequence/readout-ledger/autonomous-bounded-language-surface-use-event-review",
                        json={
                            "autonomous_bounded_language_surface_use_executor": (
                                autonomous_bounded_language_surface_use_executor_response.json()
                            ),
                            "expected_state_revision": status_response.json()[
                                "state_revision"
                            ],
                            "review_policy": {
                                "max_text_fragments": 3,
                                "max_surface_chars": 256,
                            },
                        },
                    )
                )
                autonomous_snn_language_generation_design_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/autonomous-snn-language-generation-design",
                    json={
                        "autonomous_bounded_language_surface_use_event_review": (
                            autonomous_bounded_language_surface_use_event_review_response.json()
                        ),
                        "generation_policy": {
                            "generation_mode": "snn_bounded_next_token_projection",
                            "decoding_strategy": "spike_sparse_top_k",
                            "max_new_tokens": 16,
                            "max_generated_fragments": 2,
                            "target_device": "cpu",
                            "requires_cuda": False,
                        },
                    },
                )
                autonomous_snn_language_generation_preflight_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/autonomous-snn-language-generation-preflight",
                    json={
                        "autonomous_snn_language_generation_design": (
                            autonomous_snn_language_generation_design_response.json()
                        ),
                        "expected_state_revision": status_response.json()[
                            "state_revision"
                        ],
                        "device_evidence": {"device": "cpu", "source": "api-test"},
                        "executor_capabilities": {
                            "autonomous_snn_language_generation_executor": False
                        },
                    },
                )
                autonomous_snn_language_generation_executor_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/autonomous-snn-language-generation-executor",
                    json={
                        "autonomous_snn_language_generation_preflight": (
                            autonomous_snn_language_generation_preflight_response.json()
                        ),
                        "expected_state_revision": status_response.json()[
                            "state_revision"
                        ],
                        "generation_evidence": {
                            "generated_token_hashes": [
                                _sha256_json({"api_generated_token": 0})
                            ],
                            "spike_projection_hashes": [
                                _sha256_json({"api_spike_projection": 0})
                            ],
                            "active_neuron_hashes": [
                                _sha256_json({"api_active_neurons": [1, 2]})
                            ],
                            "membrane_state_hashes": [
                                _sha256_json({"api_membrane_state": 0})
                            ],
                            "output_fragment_hashes": [
                                _sha256_json({"api_output_fragment": 0})
                            ],
                            "checkpoint_written": False,
                        },
                        "execution_policy": {"max_new_tokens": 1},
                    },
                )
                autonomous_snn_language_generation_event_review_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/autonomous-snn-language-generation-event-review",
                    json={
                        "autonomous_snn_language_generation_executor": (
                            autonomous_snn_language_generation_executor_response.json()
                        ),
                        "expected_state_revision": status_response.json()[
                            "state_revision"
                        ],
                        "review_policy": {"max_generated_tokens": 1},
                    },
                )
                autonomous_snn_language_decoding_design_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/autonomous-snn-language-decoding-design",
                    json={
                        "autonomous_snn_language_generation_event_review": (
                            autonomous_snn_language_generation_event_review_response.json()
                        ),
                        "decoding_policy": {
                            "decoding_mode": "bounded_hash_token_projection",
                            "materialization_target": "bounded_text_surface",
                            "max_decoded_tokens": 1,
                            "max_decoded_fragments": 1,
                            "max_surface_chars": 256,
                        },
                    },
                )
                autonomous_snn_language_decoding_preflight_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/autonomous-snn-language-decoding-preflight",
                    json={
                        "autonomous_snn_language_decoding_design": (
                            autonomous_snn_language_decoding_design_response.json()
                        ),
                        "expected_state_revision": status_response.json()[
                            "state_revision"
                        ],
                        "device_evidence": {
                            "device": "cuda:0",
                            "cuda_available": True,
                        },
                        "decoder_capabilities": {
                            "autonomous_snn_language_decoding_executor": True
                        },
                    },
                )
                autonomous_snn_language_decoding_executor_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/autonomous-snn-language-decoding-executor",
                    json={
                        "autonomous_snn_language_decoding_preflight": (
                            autonomous_snn_language_decoding_preflight_response.json()
                        ),
                        "expected_state_revision": status_response.json()[
                            "state_revision"
                        ],
                        "decoding_evidence": {
                            "decoded_token_hashes": [
                                _sha256_json({"api_generated_token": 0})
                            ],
                            "decoded_text_fragments": ["api spike"],
                            "rendered_text": "api spike",
                            "schema_valid": True,
                            "text_normalized": True,
                            "semantic_constraint_valid": True,
                            "checkpoint_written": False,
                        },
                    },
                )
                autonomous_snn_language_decoding_event_review_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/autonomous-snn-language-decoding-event-review",
                    json={
                        "autonomous_snn_language_decoding_executor": (
                            autonomous_snn_language_decoding_executor_response.json()
                        ),
                        "expected_state_revision": status_response.json()[
                            "state_revision"
                        ],
                        "review_policy": {
                            "max_decoded_tokens": 1,
                            "max_decoded_fragments": 1,
                            "max_surface_chars": 256,
                        },
                    },
                )
                snn_language_readout_surface_design_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/snn-language-surface-design",
                    json={
                        "autonomous_snn_language_decoding_event_review": (
                            autonomous_snn_language_decoding_event_review_response.json()
                        ),
                        "surface_policy": {
                            "thought_role": "inner_speech_candidate",
                            "binding_mode": "hash_bound_inner_language",
                            "max_thought_fragments": 1,
                            "max_surface_chars": 256,
                            "max_association_edges": 4,
                        },
                    },
                )
                snn_language_readout_surface_preflight_response = (
                    client.post(
                        "/terminus/snn-language-sequence/readout-ledger/snn-language-surface-preflight",
                        json={
                            "snn_language_surface_design": (
                                snn_language_readout_surface_design_response.json()
                            ),
                            "expected_state_revision": status_response.json()[
                                "state_revision"
                            ],
                            "device_evidence": {
                                "device": "cuda:0",
                                "cuda_available": True,
                            },
                            "executor_capabilities": {
                                "snn_language_readout_surface_executor": True
                            },
                        },
                    )
                )
                snn_language_readout_surface_executor_response = (
                    client.post(
                        "/terminus/snn-language-sequence/readout-ledger/snn-language-surface-executor",
                        json={
                            "snn_language_surface_preflight": (
                                snn_language_readout_surface_preflight_response.json()
                            ),
                            "expected_state_revision": status_response.json()[
                                "state_revision"
                            ],
                        },
                    )
                )
                snn_language_readout_surface_event_review_response = (
                    client.post(
                        "/terminus/snn-language-sequence/readout-ledger/snn-language-surface-event-review",
                        json={
                            "snn_language_surface_executor": (
                                snn_language_readout_surface_executor_response.json()
                            ),
                            "expected_state_revision": status_response.json()[
                                "state_revision"
                            ],
                            "review_policy": {
                                "max_thought_fragments": 1,
                                "max_surface_chars": 256,
                                "max_association_edges": 4,
                            },
                        },
                    )
                )
                snn_language_readout_memory_design_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/snn-language-memory-design",
                    json={
                        "snn_language_surface_event_review": (
                            snn_language_readout_surface_event_review_response.json()
                        ),
                        "memory_policy": {
                            "memory_scope": "working_trace",
                            "consolidation_route": "deferred_local_trace",
                            "max_trace_fragments": 1,
                            "max_trace_chars": 256,
                            "max_local_learning_targets": 4,
                        },
                    },
                )
                snn_language_readout_memory_preflight_response = (
                    client.post(
                        "/terminus/snn-language-sequence/readout-ledger/snn-language-memory-preflight",
                        json={
                            "snn_language_memory_design": (
                                snn_language_readout_memory_design_response.json()
                            ),
                            "expected_state_revision": status_response.json()[
                                "state_revision"
                            ],
                            "device_evidence": {
                                "device": "cuda:0",
                                "cuda_available": True,
                            },
                            "executor_capabilities": {
                                "snn_language_readout_memory_executor": True
                            },
                        },
                    )
                )
                snn_language_readout_memory_executor_response = (
                    client.post(
                        "/terminus/snn-language-sequence/readout-ledger/snn-language-memory-executor",
                        json={
                            "snn_language_memory_preflight": (
                                snn_language_readout_memory_preflight_response.json()
                            ),
                            "expected_state_revision": status_response.json()[
                                "state_revision"
                            ],
                        },
                    )
                )
                snn_language_readout_memory_event_review_response = (
                    client.post(
                        "/terminus/snn-language-sequence/readout-ledger/snn-language-memory-event-review",
                        json={
                            "snn_language_memory_executor": (
                                snn_language_readout_memory_executor_response.json()
                            ),
                            "expected_state_revision": status_response.json()[
                                "state_revision"
                            ],
                            "review_policy": {
                                "max_trace_fragments": 1,
                                "max_trace_chars": 256,
                                "max_local_learning_targets": 4,
                            },
                        },
                    )
                )
                snn_language_readout_consolidation_design_response = (
                    client.post(
                        "/terminus/snn-language-sequence/readout-ledger/snn-language-consolidation-design",
                        json={
                            "snn_language_memory_event_review": (
                                snn_language_readout_memory_event_review_response.json()
                            ),
                            "consolidation_policy": {
                                "consolidation_scope": "local_trace_reinforcement",
                                "consolidation_route": "deferred_local_trace",
                                "learning_rate": 0.02,
                                "max_weight_delta": 0.04,
                                "homeostatic_decay": 0.01,
                                "max_candidate_updates": 4,
                            },
                        },
                    )
                )
                snn_language_readout_consolidation_preflight_response = (
                    client.post(
                        "/terminus/snn-language-sequence/readout-ledger/snn-language-consolidation-preflight",
                        json={
                            "snn_language_consolidation_design": (
                                snn_language_readout_consolidation_design_response.json()
                            ),
                            "expected_state_revision": status_response.json()[
                                "state_revision"
                            ],
                            "device_evidence": {
                                "device": "cuda:0",
                                "cuda_available": True,
                            },
                            "executor_capabilities": {
                                "snn_language_readout_consolidation_executor": True
                            },
                        },
                    )
                )
                snn_language_readout_consolidation_executor_response = (
                    client.post(
                        "/terminus/snn-language-sequence/readout-ledger/snn-language-consolidation-executor",
                        json={
                            "snn_language_consolidation_preflight": (
                                snn_language_readout_consolidation_preflight_response.json()
                            ),
                            "expected_state_revision": status_response.json()[
                                "state_revision"
                            ],
                        },
                    )
                )
                snn_language_readout_consolidation_event_review_response = (
                    client.post(
                        "/terminus/snn-language-sequence/readout-ledger/snn-language-consolidation-event-review",
                        json={
                            "snn_language_consolidation_executor": (
                                snn_language_readout_consolidation_executor_response.json()
                            ),
                            "expected_state_revision": status_response.json()[
                                "state_revision"
                            ],
                            "review_policy": {
                                "max_candidate_updates": 4,
                                "max_learning_rate": 0.02,
                                "max_weight_delta": 0.04,
                                "max_homeostatic_decay": 0.01,
                            },
                        },
                    )
                )
                snn_language_readout_structural_plasticity_design_response = (
                    client.post(
                        "/terminus/snn-language-sequence/readout-ledger/snn-language-structural-plasticity-design",
                        json={
                            "snn_language_consolidation_event_review": (
                                snn_language_readout_consolidation_event_review_response.json()
                            ),
                            "structural_policy": {
                                "structural_scope": "thought_trace_sparse_capacity",
                                "structural_route": "reviewed_consolidation_to_growth_prune",
                                "max_growth_candidates": 4,
                                "max_prune_candidates": 2,
                                "max_new_neurons": 2,
                                "max_new_synapses": 4,
                                "max_prune_synapses": 2,
                            },
                        },
                    )
                )
                snn_language_readout_structural_plasticity_preflight_response = (
                    client.post(
                        "/terminus/snn-language-sequence/readout-ledger/snn-language-structural-plasticity-preflight",
                        json={
                            "snn_language_structural_plasticity_design": (
                                snn_language_readout_structural_plasticity_design_response.json()
                            ),
                            "expected_state_revision": status_response.json()[
                                "state_revision"
                            ],
                            "device_evidence": {
                                "device": "cuda:0",
                                "cuda_available": True,
                            },
                            "executor_capabilities": {
                                "snn_language_readout_structural_plasticity_executor": True
                            },
                        },
                    )
                )
                snn_language_readout_structural_plasticity_executor_response = (
                    client.post(
                        "/terminus/snn-language-sequence/readout-ledger/snn-language-structural-plasticity-executor",
                        json={
                            "snn_language_structural_plasticity_preflight": (
                                snn_language_readout_structural_plasticity_preflight_response.json()
                            ),
                            "expected_state_revision": status_response.json()[
                                "state_revision"
                            ],
                        },
                    )
                )
                snn_language_readout_structural_plasticity_event_review_response = (
                    client.post(
                        "/terminus/snn-language-sequence/readout-ledger/snn-language-structural-plasticity-event-review",
                        json={
                            "snn_language_structural_plasticity_executor": (
                                snn_language_readout_structural_plasticity_executor_response.json()
                            ),
                            "expected_state_revision": status_response.json()[
                                "state_revision"
                            ],
                            "review_policy": {
                                "max_growth_candidates": 4,
                                "max_prune_candidates": 2,
                                "max_new_neurons": 2,
                                "max_new_synapses": 4,
                                "max_prune_synapses": 2,
                            },
                        },
                    )
                )
                snn_language_readout_capacity_mutation_design_response = (
                    client.post(
                        "/terminus/snn-language-sequence/readout-ledger/snn-language-capacity-mutation-design",
                        json={
                            "snn_language_structural_plasticity_event_review": (
                                snn_language_readout_structural_plasticity_event_review_response.json()
                            ),
                            "capacity_policy": {
                                "mutation_scope": "thought_driven_sparse_capacity",
                                "mutation_route": "reviewed_structural_plasticity_to_capacity_resize",
                                "current_neuron_capacity": 64,
                                "current_sparse_synapse_budget": 256,
                                "current_dense_rows": 64,
                                "current_dense_cols": 64,
                                "max_capacity_growth_factor": 2.0,
                            },
                        },
                    )
                )
                snn_language_readout_capacity_mutation_preflight_response = (
                    client.post(
                        "/terminus/snn-language-sequence/readout-ledger/snn-language-capacity-mutation-preflight",
                        json={
                            "snn_language_capacity_mutation_design": (
                                snn_language_readout_capacity_mutation_design_response.json()
                            ),
                            "expected_state_revision": status_response.json()[
                                "state_revision"
                            ],
                            "checkpoint_transaction": {
                                "checkpoint_path": (
                                    "memory://thought-capacity-preflight"
                                ),
                                "snapshot_id": "thought-capacity-snapshot",
                                "pre_capacity_mutation_checkpoint_saved": True,
                                "pre_capacity_mutation_checkpoint_restore_verified": True,
                            },
                            "device_evidence": {
                                "device": "cuda:0",
                                "cuda_available": True,
                                "cuda_relayout_verified": True,
                            },
                            "executor_capabilities": {
                                "snn_language_readout_capacity_mutation_executor": True
                            },
                        },
                    )
                )
                snn_language_readout_capacity_mutation_executor_response = (
                    client.post(
                        "/terminus/snn-language-sequence/readout-ledger/snn-language-capacity-mutation-executor",
                        json={
                            "snn_language_capacity_mutation_preflight": (
                                snn_language_readout_capacity_mutation_preflight_response.json()
                            ),
                            "expected_state_revision": status_response.json()[
                                "state_revision"
                            ],
                            "requested_device": "cpu",
                        },
                    )
                )
                snn_language_readout_capacity_mutation_event_review_response = (
                    client.post(
                        "/terminus/snn-language-sequence/readout-ledger/snn-language-capacity-mutation-event-review",
                        json={
                            "snn_language_capacity_mutation_executor": (
                                snn_language_readout_capacity_mutation_executor_response.json()
                            ),
                            "expected_state_revision": status_response.json()[
                                "state_revision"
                            ],
                        },
                    )
                )
                snn_language_readout_newborn_neuron_integration_design_response = (
                    client.post(
                        "/terminus/snn-language-sequence/readout-ledger/snn-language-newborn-neuron-integration-design",
                        json={
                            "snn_language_capacity_mutation_event_review": (
                                snn_language_readout_capacity_mutation_event_review_response.json()
                            ),
                            "integration_policy": {
                                "max_newborn_neurons": 2,
                                "max_seed_synapses_per_newborn": 2,
                                "critical_period_cycles": 64,
                            },
                        },
                    )
                )
                snn_language_readout_newborn_neuron_integration_preflight_response = (
                    client.post(
                        "/terminus/snn-language-sequence/readout-ledger/snn-language-newborn-neuron-integration-preflight",
                        json={
                            "snn_language_newborn_neuron_integration_design": (
                                snn_language_readout_newborn_neuron_integration_design_response.json()
                            ),
                            "expected_state_revision": status_response.json()[
                                "state_revision"
                            ],
                            "live_spike_evidence": {
                                "surface": (
                                    "snn_language_live_spike_population_evidence.v1"
                                ),
                                "state_revision": status_response.json()[
                                    "state_revision"
                                ],
                                "observation_window_id": "blocked-api-window",
                                "observation_window_hash": "0" * 64,
                                "device": "cpu",
                                "tensor_is_cuda": False,
                                "candidate_observations": [],
                            },
                            "checkpoint_transaction": {},
                            "executor_capabilities": {},
                        },
                    )
                )
                snn_language_readout_newborn_neuron_integration_executor_response = (
                    client.post(
                        "/terminus/snn-language-sequence/readout-ledger/snn-language-newborn-neuron-integration-executor",
                        json={
                            "snn_language_newborn_neuron_integration_preflight": (
                                snn_language_readout_newborn_neuron_integration_preflight_response.json()
                            ),
                            "expected_state_revision": status_response.json()[
                                "state_revision"
                            ],
                        },
                    )
                )
                snn_language_readout_newborn_neuron_integration_event_review_response = (
                    client.post(
                        "/terminus/snn-language-sequence/readout-ledger/snn-language-newborn-neuron-integration-event-review",
                        json={
                            "snn_language_newborn_neuron_integration_executor": (
                                snn_language_readout_newborn_neuron_integration_executor_response.json()
                            ),
                            "expected_state_revision": status_response.json()[
                                "state_revision"
                            ],
                        },
                    )
                )
                snn_language_readout_newborn_neuron_critical_period_learning_design_response = (
                    client.post(
                        "/terminus/snn-language-sequence/readout-ledger/snn-language-newborn-neuron-critical-period-learning-design",
                        json={
                            "snn_language_newborn_neuron_integration_event_review": (
                                snn_language_readout_newborn_neuron_integration_event_review_response.json()
                            )
                        },
                    )
                )
                snn_language_readout_newborn_neuron_critical_period_learning_preflight_response = (
                    client.post(
                        "/terminus/snn-language-sequence/readout-ledger/snn-language-newborn-neuron-critical-period-learning-preflight",
                        json={
                            "snn_language_newborn_neuron_critical_period_learning_design": (
                                snn_language_readout_newborn_neuron_critical_period_learning_design_response.json()
                            ),
                            "expected_state_revision": status_response.json()[
                                "state_revision"
                            ],
                            "critical_period_activity_evidence": {
                                "surface": "snn_language_newborn_critical_period_activity.v1",
                                "state_revision": status_response.json()[
                                    "state_revision"
                                ],
                                "observation_window_id": "blocked-critical-period-api-window",
                                "observation_window_hash": "0" * 64,
                                "device": "cpu",
                                "tensor_is_cuda": False,
                                "candidate_observations": [],
                            },
                            "checkpoint_transaction": {},
                            "executor_capabilities": {},
                        },
                    )
                )
                snn_language_readout_newborn_neuron_critical_period_learning_executor_response = (
                    client.post(
                        "/terminus/snn-language-sequence/readout-ledger/snn-language-newborn-neuron-critical-period-learning-executor",
                        json={
                            "snn_language_newborn_neuron_critical_period_learning_preflight": (
                                snn_language_readout_newborn_neuron_critical_period_learning_preflight_response.json()
                            ),
                            "expected_state_revision": status_response.json()[
                                "state_revision"
                            ],
                        },
                    )
                )
                snn_language_readout_newborn_neuron_critical_period_learning_event_review_response = (
                    client.post(
                        "/terminus/snn-language-sequence/readout-ledger/snn-language-newborn-neuron-critical-period-learning-event-review",
                        json={
                            "snn_language_newborn_neuron_critical_period_learning_executor": (
                                snn_language_readout_newborn_neuron_critical_period_learning_executor_response.json()
                            ),
                            "expected_state_revision": status_response.json()[
                                "state_revision"
                            ],
                        },
                    )
                )
                snn_language_readout_newborn_neuron_critical_period_learning_continuation_design_response = (
                    client.post(
                        "/terminus/snn-language-sequence/readout-ledger/snn-language-newborn-neuron-critical-period-learning-continuation-design",
                        json={
                            "snn_language_newborn_neuron_critical_period_learning_event_review": (
                                snn_language_readout_newborn_neuron_critical_period_learning_event_review_response.json()
                            )
                        },
                    )
                )
            app.state.marulho_manager.close()

        self.assertEqual(status_response.status_code, 200)
        self.assertEqual(terminus_response.status_code, 200)
        self.assertEqual(capacity_expansion_response.status_code, 200)
        self.assertEqual(capacity_preflight_response.status_code, 200)
        self.assertEqual(capacity_compatibility_response.status_code, 200)
        self.assertEqual(dense_readout_resize_plan_response.status_code, 200)
        self.assertEqual(dense_readout_resize_preflight_response.status_code, 200)
        self.assertEqual(dense_readout_resize_transaction_response.status_code, 200)
        self.assertEqual(dense_readout_resize_readiness_response.status_code, 200)
        self.assertEqual(dense_readout_layout_migration_response.status_code, 200)
        self.assertEqual(
            dense_readout_tensor_materialization_readiness_response.status_code,
            200,
        )
        self.assertEqual(dense_readout_tensor_materialization_response.status_code, 200)
        self.assertEqual(dense_readout_training_readiness_response.status_code, 200)
        self.assertEqual(dense_readout_training_loop_design_response.status_code, 200)
        self.assertEqual(dense_readout_training_loop_preflight_response.status_code, 200)
        self.assertEqual(dense_readout_training_response.status_code, 200)
        self.assertEqual(
            dense_readout_post_training_evaluation_response.status_code,
            200,
        )
        self.assertEqual(dense_readout_decoder_probe_design_response.status_code, 200)
        self.assertEqual(dense_readout_decoder_probe_preflight_response.status_code, 200)
        self.assertEqual(dense_readout_decoder_probe_response.status_code, 200)
        self.assertEqual(
            dense_readout_label_candidate_review_response.status_code,
            200,
        )
        self.assertEqual(
            dense_readout_label_candidate_record_response.status_code,
            200,
        )
        self.assertEqual(dense_label_candidate_history_response.status_code, 200)
        self.assertEqual(
            dense_label_candidate_calibration_policy_response.status_code,
            200,
        )
        self.assertEqual(
            dense_label_candidate_calibration_design_response.status_code,
            200,
        )
        self.assertEqual(
            dense_label_candidate_calibration_preflight_response.status_code,
            200,
        )
        self.assertEqual(
            dense_label_candidate_calibration_evaluation_response.status_code,
            200,
        )
        self.assertEqual(
            dense_label_candidate_calibration_review_response.status_code,
            200,
        )
        self.assertEqual(
            dense_label_candidate_calibration_update_design_response.status_code,
            200,
        )
        self.assertEqual(
            dense_label_candidate_calibration_update_preflight_response.status_code,
            200,
        )
        self.assertEqual(
            dense_label_candidate_calibration_update_application_response.status_code,
            200,
        )
        self.assertEqual(
            dense_label_candidate_calibration_update_application_review_response.status_code,
            200,
        )
        self.assertEqual(
            dense_label_candidate_post_calibration_observation_response.status_code,
            200,
        )
        self.assertEqual(
            dense_label_candidate_post_calibration_operator_review_response.status_code,
            200,
        )
        self.assertEqual(
            calibrated_dense_label_confidence_use_design_response.status_code,
            200,
        )
        self.assertEqual(
            calibrated_dense_label_confidence_use_preflight_response.status_code,
            200,
        )
        self.assertEqual(
            calibrated_dense_label_confidence_use_executor_response.status_code,
            200,
        )
        self.assertEqual(
            calibrated_dense_label_confidence_operator_display_review_response.status_code,
            200,
        )
        self.assertEqual(
            calibrated_dense_label_confidence_internal_stability_review_response.status_code,
            200,
        )
        self.assertEqual(
            calibrated_dense_label_confidence_autonomous_replay_design_response.status_code,
            200,
        )
        self.assertEqual(
            calibrated_dense_label_confidence_autonomous_replay_preflight_response.status_code,
            200,
        )
        self.assertEqual(
            calibrated_dense_label_confidence_autonomous_replay_executor_response.status_code,
            200,
        )
        self.assertEqual(
            calibrated_dense_label_confidence_autonomous_recalibration_design_response.status_code,
            200,
        )
        self.assertEqual(
            calibrated_dense_label_confidence_autonomous_recalibration_preflight_response.status_code,
            200,
        )
        self.assertEqual(
            calibrated_dense_label_confidence_autonomous_recalibration_executor_response.status_code,
            200,
        )
        self.assertEqual(
            calibrated_dense_label_confidence_autonomous_recalibration_application_review_response.status_code,
            200,
        )
        self.assertEqual(
            calibrated_dense_label_confidence_autonomous_post_calibration_observation_response.status_code,
            200,
        )
        self.assertEqual(
            calibrated_dense_label_confidence_autonomous_post_calibration_stability_review_response.status_code,
            200,
        )
        self.assertEqual(
            calibrated_dense_label_confidence_autonomous_use_design_response.status_code,
            200,
        )
        self.assertEqual(
            calibrated_dense_label_confidence_autonomous_use_preflight_response.status_code,
            200,
        )
        self.assertEqual(
            calibrated_dense_label_confidence_autonomous_use_executor_response.status_code,
            200,
        )
        self.assertEqual(
            calibrated_dense_label_confidence_autonomous_use_event_review_response.status_code,
            200,
        )
        self.assertEqual(
            autonomous_hash_readout_binding_design_response.status_code,
            200,
        )
        self.assertEqual(
            autonomous_hash_readout_binding_preflight_response.status_code,
            200,
        )
        self.assertEqual(
            autonomous_hash_readout_binding_executor_response.status_code,
            200,
        )
        self.assertEqual(
            autonomous_hash_readout_binding_event_review_response.status_code,
            200,
        )
        self.assertEqual(
            autonomous_bound_readout_observation_design_response.status_code,
            200,
        )
        self.assertEqual(
            autonomous_bound_readout_observation_preflight_response.status_code,
            200,
        )
        self.assertEqual(
            autonomous_bound_readout_observation_executor_response.status_code,
            200,
        )
        self.assertEqual(
            autonomous_bound_readout_observation_event_review_response.status_code,
            200,
        )
        self.assertEqual(
            autonomous_readout_training_window_design_response.status_code,
            200,
        )
        self.assertEqual(
            autonomous_readout_training_window_preflight_response.status_code,
            200,
        )
        self.assertEqual(
            autonomous_readout_training_window_executor_response.status_code,
            200,
        )
        self.assertEqual(
            autonomous_readout_training_window_event_review_response.status_code,
            200,
        )
        self.assertEqual(autonomous_decoder_probe_design_response.status_code, 200)
        self.assertEqual(autonomous_decoder_probe_preflight_response.status_code, 200)
        self.assertEqual(autonomous_decoder_probe_executor_response.status_code, 200)
        self.assertEqual(
            autonomous_decoder_probe_event_review_response.status_code, 200
        )
        self.assertEqual(autonomous_language_output_design_response.status_code, 200)
        self.assertEqual(autonomous_language_output_preflight_response.status_code, 200)
        self.assertEqual(autonomous_language_output_executor_response.status_code, 200)
        self.assertEqual(
            autonomous_language_output_event_review_response.status_code, 200
        )
        self.assertEqual(autonomous_decoded_output_design_response.status_code, 200)
        self.assertEqual(autonomous_decoded_output_preflight_response.status_code, 200)
        self.assertEqual(autonomous_decoded_output_executor_response.status_code, 200)
        self.assertEqual(
            autonomous_decoded_output_event_review_response.status_code, 200
        )
        self.assertEqual(
            autonomous_bounded_text_emission_design_response.status_code, 200
        )
        self.assertEqual(
            autonomous_bounded_text_emission_preflight_response.status_code, 200
        )
        self.assertEqual(
            autonomous_bounded_text_emission_executor_response.status_code, 200
        )
        self.assertEqual(
            autonomous_bounded_text_emission_event_review_response.status_code,
            200,
        )
        self.assertEqual(
            autonomous_text_surface_sequence_review_response.status_code,
            200,
        )
        self.assertEqual(
            autonomous_text_surface_commit_design_response.status_code,
            200,
        )
        self.assertEqual(
            autonomous_text_surface_commit_preflight_response.status_code,
            200,
        )
        self.assertEqual(
            autonomous_text_surface_commit_executor_response.status_code,
            200,
        )
        self.assertEqual(
            autonomous_text_surface_commit_event_review_response.status_code,
            200,
        )
        self.assertEqual(
            autonomous_text_surface_materialization_design_response.status_code,
            200,
        )
        self.assertEqual(
            autonomous_text_surface_materialization_preflight_response.status_code,
            200,
        )
        self.assertEqual(
            autonomous_text_surface_materialization_executor_response.status_code,
            200,
        )
        self.assertEqual(
            autonomous_text_surface_materialization_event_review_response.status_code,
            200,
        )
        self.assertEqual(
            autonomous_bounded_language_surface_review_response.status_code,
            200,
        )
        self.assertEqual(
            autonomous_bounded_language_surface_commit_design_response.status_code,
            200,
        )
        self.assertEqual(
            autonomous_bounded_language_surface_commit_preflight_response.status_code,
            200,
        )
        self.assertEqual(
            autonomous_bounded_language_surface_commit_executor_response.status_code,
            200,
        )
        self.assertEqual(
            autonomous_bounded_language_surface_commit_event_review_response.status_code,
            200,
        )
        self.assertEqual(
            autonomous_bounded_language_surface_use_review_response.status_code,
            200,
        )
        self.assertEqual(
            autonomous_bounded_language_surface_use_preflight_response.status_code,
            200,
        )
        self.assertEqual(
            autonomous_bounded_language_surface_use_executor_response.status_code,
            200,
        )
        self.assertEqual(
            autonomous_bounded_language_surface_use_event_review_response.status_code,
            200,
        )
        self.assertEqual(
            autonomous_snn_language_generation_design_response.status_code,
            200,
        )
        self.assertEqual(
            autonomous_snn_language_generation_preflight_response.status_code,
            200,
        )
        self.assertEqual(
            autonomous_snn_language_generation_executor_response.status_code,
            200,
        )
        self.assertEqual(
            autonomous_snn_language_generation_event_review_response.status_code,
            200,
        )
        self.assertEqual(
            autonomous_snn_language_decoding_design_response.status_code,
            200,
        )
        self.assertEqual(
            autonomous_snn_language_decoding_preflight_response.status_code,
            200,
        )
        self.assertEqual(
            autonomous_snn_language_decoding_executor_response.status_code,
            200,
        )
        self.assertEqual(
            autonomous_snn_language_decoding_event_review_response.status_code,
            200,
        )
        self.assertEqual(
            snn_language_readout_surface_design_response.status_code,
            200,
        )
        self.assertEqual(
            snn_language_readout_surface_preflight_response.status_code,
            200,
        )
        self.assertEqual(
            snn_language_readout_surface_executor_response.status_code,
            200,
        )
        self.assertEqual(
            snn_language_readout_surface_event_review_response.status_code,
            200,
        )
        self.assertEqual(
            snn_language_readout_memory_design_response.status_code,
            200,
        )
        self.assertEqual(
            snn_language_readout_memory_preflight_response.status_code,
            200,
        )
        self.assertEqual(
            snn_language_readout_memory_executor_response.status_code,
            200,
        )
        self.assertEqual(
            snn_language_readout_memory_event_review_response.status_code,
            200,
        )
        self.assertEqual(
            snn_language_readout_consolidation_design_response.status_code,
            200,
        )
        self.assertEqual(
            snn_language_readout_consolidation_preflight_response.status_code,
            200,
        )
        self.assertEqual(
            snn_language_readout_consolidation_executor_response.status_code,
            200,
        )
        self.assertEqual(
            snn_language_readout_consolidation_event_review_response.status_code,
            200,
        )
        self.assertEqual(
            snn_language_readout_structural_plasticity_design_response.status_code,
            200,
        )
        self.assertEqual(
            snn_language_readout_structural_plasticity_preflight_response.status_code,
            200,
        )
        self.assertEqual(
            snn_language_readout_structural_plasticity_executor_response.status_code,
            200,
        )
        self.assertEqual(
            snn_language_readout_structural_plasticity_event_review_response.status_code,
            200,
        )
        self.assertEqual(
            snn_language_readout_capacity_mutation_design_response.status_code,
            200,
        )
        self.assertEqual(
            snn_language_readout_capacity_mutation_preflight_response.status_code,
            200,
        )
        self.assertEqual(
            snn_language_readout_capacity_mutation_executor_response.status_code,
            200,
        )
        self.assertEqual(
            snn_language_readout_capacity_mutation_event_review_response.status_code,
            200,
        )
        self.assertEqual(
            snn_language_readout_newborn_neuron_integration_design_response.status_code,
            200,
        )
        self.assertEqual(
            snn_language_readout_newborn_neuron_integration_preflight_response.status_code,
            200,
        )
        self.assertEqual(
            snn_language_readout_newborn_neuron_integration_executor_response.status_code,
            200,
        )
        self.assertEqual(
            snn_language_readout_newborn_neuron_integration_event_review_response.status_code,
            200,
        )
        self.assertEqual(
            snn_language_readout_newborn_neuron_critical_period_learning_design_response.status_code,
            200,
        )
        self.assertEqual(
            snn_language_readout_newborn_neuron_critical_period_learning_preflight_response.status_code,
            200,
        )
        self.assertEqual(
            snn_language_readout_newborn_neuron_critical_period_learning_executor_response.status_code,
            200,
        )
        self.assertEqual(
            snn_language_readout_newborn_neuron_critical_period_learning_event_review_response.status_code,
            200,
        )
        self.assertEqual(
            snn_language_readout_newborn_neuron_critical_period_learning_continuation_design_response.status_code,
            200,
        )
        status_truth = status_response.json()["runtime_truth"]
        terminus_truth = terminus_response.json()["runtime_truth"]
        status_device = status_truth["evidence"]["runtime_device"]
        terminus_device = terminus_truth["evidence"]["runtime_device"]
        status_uses_cuda = str(status_device["tensor_device"]).startswith("cuda")
        terminus_uses_cuda = str(terminus_device["tensor_device"]).startswith("cuda")
        self.assertEqual(status_truth["evidence"]["device"], status_device["resolved_device"])
        self.assertEqual(status_truth["evidence"]["cuda_available"], status_device["cuda_available"])
        self.assertEqual(status_truth["evidence"]["observed_cuda_execution"], status_uses_cuda)
        self.assertEqual(status_device["observed_cuda_execution"], status_uses_cuda)
        self.assertEqual(
            status_device["claim_boundary"],
            (
                "observed_cuda_execution_only_not_cuda_speedup"
                if status_uses_cuda
                else "observed_device_placement_only_not_cuda_speedup"
            ),
        )
        self.assertEqual(terminus_truth["evidence"]["device"], terminus_device["resolved_device"])
        self.assertEqual(terminus_truth["evidence"]["observed_cuda_execution"], terminus_uses_cuda)
        capacity_expansion_design = capacity_expansion_response.json()
        capacity_preflight = capacity_preflight_response.json()
        capacity_compatibility = capacity_compatibility_response.json()
        dense_readout_resize_plan = dense_readout_resize_plan_response.json()
        dense_readout_resize_preflight = dense_readout_resize_preflight_response.json()
        dense_readout_resize_transaction = dense_readout_resize_transaction_response.json()
        dense_readout_resize_readiness = dense_readout_resize_readiness_response.json()
        dense_readout_layout_migration = dense_readout_layout_migration_response.json()
        dense_readout_tensor_materialization_readiness = (
            dense_readout_tensor_materialization_readiness_response.json()
        )
        dense_readout_tensor_materialization = (
            dense_readout_tensor_materialization_response.json()
        )
        dense_readout_training_readiness = (
            dense_readout_training_readiness_response.json()
        )
        dense_readout_training_loop_design = (
            dense_readout_training_loop_design_response.json()
        )
        dense_readout_training_loop_preflight = (
            dense_readout_training_loop_preflight_response.json()
        )
        dense_readout_training = dense_readout_training_response.json()
        dense_readout_post_training_evaluation = (
            dense_readout_post_training_evaluation_response.json()
        )
        dense_readout_decoder_probe_design = (
            dense_readout_decoder_probe_design_response.json()
        )
        dense_readout_decoder_probe_preflight = (
            dense_readout_decoder_probe_preflight_response.json()
        )
        dense_readout_decoder_probe = dense_readout_decoder_probe_response.json()
        dense_readout_label_candidate_review = (
            dense_readout_label_candidate_review_response.json()
        )
        dense_readout_label_candidate_record = (
            dense_readout_label_candidate_record_response.json()
        )
        dense_label_candidate_history = dense_label_candidate_history_response.json()
        dense_label_candidate_calibration_policy = (
            dense_label_candidate_calibration_policy_response.json()
        )
        dense_label_candidate_calibration_design = (
            dense_label_candidate_calibration_design_response.json()
        )
        dense_label_candidate_calibration_preflight = (
            dense_label_candidate_calibration_preflight_response.json()
        )
        dense_label_candidate_calibration_evaluation = (
            dense_label_candidate_calibration_evaluation_response.json()
        )
        dense_label_candidate_calibration_review = (
            dense_label_candidate_calibration_review_response.json()
        )
        dense_label_candidate_calibration_update_design = (
            dense_label_candidate_calibration_update_design_response.json()
        )
        dense_label_candidate_calibration_update_preflight = (
            dense_label_candidate_calibration_update_preflight_response.json()
        )
        dense_label_candidate_calibration_update_application = (
            dense_label_candidate_calibration_update_application_response.json()
        )
        dense_label_candidate_calibration_update_application_review = (
            dense_label_candidate_calibration_update_application_review_response.json()
        )
        dense_label_candidate_post_calibration_observation = (
            dense_label_candidate_post_calibration_observation_response.json()
        )
        dense_label_candidate_post_calibration_operator_review = (
            dense_label_candidate_post_calibration_operator_review_response.json()
        )
        calibrated_dense_label_confidence_use_design = (
            calibrated_dense_label_confidence_use_design_response.json()
        )
        calibrated_dense_label_confidence_use_preflight = (
            calibrated_dense_label_confidence_use_preflight_response.json()
        )
        calibrated_dense_label_confidence_use_executor = (
            calibrated_dense_label_confidence_use_executor_response.json()
        )
        calibrated_dense_label_confidence_operator_display_review = (
            calibrated_dense_label_confidence_operator_display_review_response.json()
        )
        calibrated_dense_label_confidence_internal_stability_review = (
            calibrated_dense_label_confidence_internal_stability_review_response.json()
        )
        calibrated_dense_label_confidence_autonomous_replay_design = (
            calibrated_dense_label_confidence_autonomous_replay_design_response.json()
        )
        calibrated_dense_label_confidence_autonomous_replay_preflight = (
            calibrated_dense_label_confidence_autonomous_replay_preflight_response.json()
        )
        calibrated_dense_label_confidence_autonomous_replay_executor = (
            calibrated_dense_label_confidence_autonomous_replay_executor_response.json()
        )
        calibrated_dense_label_confidence_autonomous_recalibration_design = (
            calibrated_dense_label_confidence_autonomous_recalibration_design_response.json()
        )
        calibrated_dense_label_confidence_autonomous_recalibration_preflight = (
            calibrated_dense_label_confidence_autonomous_recalibration_preflight_response.json()
        )
        calibrated_dense_label_confidence_autonomous_recalibration_executor = (
            calibrated_dense_label_confidence_autonomous_recalibration_executor_response.json()
        )
        calibrated_dense_label_confidence_autonomous_recalibration_application_review = (
            calibrated_dense_label_confidence_autonomous_recalibration_application_review_response.json()
        )
        calibrated_dense_label_confidence_autonomous_post_calibration_observation = (
            calibrated_dense_label_confidence_autonomous_post_calibration_observation_response.json()
        )
        calibrated_dense_label_confidence_autonomous_post_calibration_stability_review = (
            calibrated_dense_label_confidence_autonomous_post_calibration_stability_review_response.json()
        )
        calibrated_dense_label_confidence_autonomous_use_design = (
            calibrated_dense_label_confidence_autonomous_use_design_response.json()
        )
        calibrated_dense_label_confidence_autonomous_use_preflight = (
            calibrated_dense_label_confidence_autonomous_use_preflight_response.json()
        )
        calibrated_dense_label_confidence_autonomous_use_executor = (
            calibrated_dense_label_confidence_autonomous_use_executor_response.json()
        )
        calibrated_dense_label_confidence_autonomous_use_event_review = (
            calibrated_dense_label_confidence_autonomous_use_event_review_response.json()
        )
        autonomous_hash_readout_binding_design = (
            autonomous_hash_readout_binding_design_response.json()
        )
        autonomous_hash_readout_binding_preflight = (
            autonomous_hash_readout_binding_preflight_response.json()
        )
        autonomous_hash_readout_binding_executor = (
            autonomous_hash_readout_binding_executor_response.json()
        )
        autonomous_hash_readout_binding_event_review = (
            autonomous_hash_readout_binding_event_review_response.json()
        )
        autonomous_bound_readout_observation_design = (
            autonomous_bound_readout_observation_design_response.json()
        )
        autonomous_bound_readout_observation_preflight = (
            autonomous_bound_readout_observation_preflight_response.json()
        )
        autonomous_bound_readout_observation_executor = (
            autonomous_bound_readout_observation_executor_response.json()
        )
        autonomous_bound_readout_observation_event_review = (
            autonomous_bound_readout_observation_event_review_response.json()
        )
        autonomous_readout_training_window_design = (
            autonomous_readout_training_window_design_response.json()
        )
        autonomous_readout_training_window_preflight = (
            autonomous_readout_training_window_preflight_response.json()
        )
        autonomous_readout_training_window_executor = (
            autonomous_readout_training_window_executor_response.json()
        )
        autonomous_readout_training_window_event_review = (
            autonomous_readout_training_window_event_review_response.json()
        )
        autonomous_decoder_probe_design = (
            autonomous_decoder_probe_design_response.json()
        )
        autonomous_decoder_probe_preflight = (
            autonomous_decoder_probe_preflight_response.json()
        )
        autonomous_decoder_probe_executor = (
            autonomous_decoder_probe_executor_response.json()
        )
        autonomous_decoder_probe_event_review = (
            autonomous_decoder_probe_event_review_response.json()
        )
        autonomous_language_output_design = (
            autonomous_language_output_design_response.json()
        )
        autonomous_language_output_preflight = (
            autonomous_language_output_preflight_response.json()
        )
        autonomous_language_output_executor = (
            autonomous_language_output_executor_response.json()
        )
        autonomous_language_output_event_review = (
            autonomous_language_output_event_review_response.json()
        )
        autonomous_decoded_output_design = (
            autonomous_decoded_output_design_response.json()
        )
        autonomous_decoded_output_preflight = (
            autonomous_decoded_output_preflight_response.json()
        )
        autonomous_decoded_output_executor = (
            autonomous_decoded_output_executor_response.json()
        )
        autonomous_decoded_output_event_review = (
            autonomous_decoded_output_event_review_response.json()
        )
        autonomous_bounded_text_emission_design = (
            autonomous_bounded_text_emission_design_response.json()
        )
        autonomous_bounded_text_emission_preflight = (
            autonomous_bounded_text_emission_preflight_response.json()
        )
        autonomous_bounded_text_emission_executor = (
            autonomous_bounded_text_emission_executor_response.json()
        )
        autonomous_bounded_text_emission_event_review = (
            autonomous_bounded_text_emission_event_review_response.json()
        )
        autonomous_text_surface_sequence_review = (
            autonomous_text_surface_sequence_review_response.json()
        )
        autonomous_text_surface_commit_design = (
            autonomous_text_surface_commit_design_response.json()
        )
        autonomous_text_surface_commit_preflight = (
            autonomous_text_surface_commit_preflight_response.json()
        )
        autonomous_text_surface_commit_executor = (
            autonomous_text_surface_commit_executor_response.json()
        )
        autonomous_text_surface_commit_event_review = (
            autonomous_text_surface_commit_event_review_response.json()
        )
        autonomous_text_surface_materialization_design = (
            autonomous_text_surface_materialization_design_response.json()
        )
        autonomous_text_surface_materialization_preflight = (
            autonomous_text_surface_materialization_preflight_response.json()
        )
        autonomous_text_surface_materialization_executor = (
            autonomous_text_surface_materialization_executor_response.json()
        )
        autonomous_text_surface_materialization_event_review = (
            autonomous_text_surface_materialization_event_review_response.json()
        )
        autonomous_bounded_language_surface_review = (
            autonomous_bounded_language_surface_review_response.json()
        )
        autonomous_bounded_language_surface_commit_design = (
            autonomous_bounded_language_surface_commit_design_response.json()
        )
        autonomous_bounded_language_surface_commit_preflight = (
            autonomous_bounded_language_surface_commit_preflight_response.json()
        )
        autonomous_bounded_language_surface_commit_executor = (
            autonomous_bounded_language_surface_commit_executor_response.json()
        )
        autonomous_bounded_language_surface_commit_event_review = (
            autonomous_bounded_language_surface_commit_event_review_response.json()
        )
        autonomous_bounded_language_surface_use_review = (
            autonomous_bounded_language_surface_use_review_response.json()
        )
        autonomous_bounded_language_surface_use_preflight = (
            autonomous_bounded_language_surface_use_preflight_response.json()
        )
        autonomous_bounded_language_surface_use_executor = (
            autonomous_bounded_language_surface_use_executor_response.json()
        )
        autonomous_bounded_language_surface_use_event_review = (
            autonomous_bounded_language_surface_use_event_review_response.json()
        )
        autonomous_snn_language_generation_design = (
            autonomous_snn_language_generation_design_response.json()
        )
        autonomous_snn_language_generation_preflight = (
            autonomous_snn_language_generation_preflight_response.json()
        )
        autonomous_snn_language_generation_executor = (
            autonomous_snn_language_generation_executor_response.json()
        )
        autonomous_snn_language_generation_event_review = (
            autonomous_snn_language_generation_event_review_response.json()
        )
        autonomous_snn_language_decoding_design = (
            autonomous_snn_language_decoding_design_response.json()
        )
        autonomous_snn_language_decoding_preflight = (
            autonomous_snn_language_decoding_preflight_response.json()
        )
        autonomous_snn_language_decoding_executor = (
            autonomous_snn_language_decoding_executor_response.json()
        )
        autonomous_snn_language_decoding_event_review = (
            autonomous_snn_language_decoding_event_review_response.json()
        )
        snn_language_readout_surface_design = (
            snn_language_readout_surface_design_response.json()
        )
        snn_language_readout_surface_preflight = (
            snn_language_readout_surface_preflight_response.json()
        )
        snn_language_readout_surface_executor = (
            snn_language_readout_surface_executor_response.json()
        )
        snn_language_readout_surface_event_review = (
            snn_language_readout_surface_event_review_response.json()
        )
        snn_language_readout_memory_design = (
            snn_language_readout_memory_design_response.json()
        )
        snn_language_readout_memory_preflight = (
            snn_language_readout_memory_preflight_response.json()
        )
        snn_language_readout_memory_executor = (
            snn_language_readout_memory_executor_response.json()
        )
        snn_language_readout_memory_event_review = (
            snn_language_readout_memory_event_review_response.json()
        )
        snn_language_readout_consolidation_design = (
            snn_language_readout_consolidation_design_response.json()
        )
        snn_language_readout_consolidation_preflight = (
            snn_language_readout_consolidation_preflight_response.json()
        )
        snn_language_readout_consolidation_executor = (
            snn_language_readout_consolidation_executor_response.json()
        )
        snn_language_readout_consolidation_event_review = (
            snn_language_readout_consolidation_event_review_response.json()
        )
        snn_language_readout_structural_plasticity_design = (
            snn_language_readout_structural_plasticity_design_response.json()
        )
        snn_language_readout_structural_plasticity_preflight = (
            snn_language_readout_structural_plasticity_preflight_response.json()
        )
        snn_language_readout_structural_plasticity_executor = (
            snn_language_readout_structural_plasticity_executor_response.json()
        )
        snn_language_readout_structural_plasticity_event_review = (
            snn_language_readout_structural_plasticity_event_review_response.json()
        )
        snn_language_readout_capacity_mutation_design = (
            snn_language_readout_capacity_mutation_design_response.json()
        )
        snn_language_readout_capacity_mutation_preflight = (
            snn_language_readout_capacity_mutation_preflight_response.json()
        )
        snn_language_readout_capacity_mutation_executor = (
            snn_language_readout_capacity_mutation_executor_response.json()
        )
        snn_language_readout_capacity_mutation_event_review = (
            snn_language_readout_capacity_mutation_event_review_response.json()
        )
        snn_language_readout_newborn_neuron_integration_design = (
            snn_language_readout_newborn_neuron_integration_design_response.json()
        )
        snn_language_readout_newborn_neuron_integration_preflight = (
            snn_language_readout_newborn_neuron_integration_preflight_response.json()
        )
        snn_language_readout_newborn_neuron_integration_executor = (
            snn_language_readout_newborn_neuron_integration_executor_response.json()
        )
        snn_language_readout_newborn_neuron_integration_event_review = (
            snn_language_readout_newborn_neuron_integration_event_review_response.json()
        )
        snn_language_readout_newborn_neuron_critical_period_learning_design = (
            snn_language_readout_newborn_neuron_critical_period_learning_design_response.json()
        )
        snn_language_readout_newborn_neuron_critical_period_learning_preflight = (
            snn_language_readout_newborn_neuron_critical_period_learning_preflight_response.json()
        )
        snn_language_readout_newborn_neuron_critical_period_learning_executor = (
            snn_language_readout_newborn_neuron_critical_period_learning_executor_response.json()
        )
        snn_language_readout_newborn_neuron_critical_period_learning_event_review = (
            snn_language_readout_newborn_neuron_critical_period_learning_event_review_response.json()
        )
        snn_language_readout_newborn_neuron_critical_period_learning_continuation_design = (
            snn_language_readout_newborn_neuron_critical_period_learning_continuation_design_response.json()
        )
        self.assertEqual(status_truth["schema_version"], 1)
        self.assertEqual(status_truth["verdict"], "partial")
        self.assertEqual(status_truth["recommended_action"], "configure_terminus_sources")
        self.assertIn("evidence", status_truth)
        self.assertIn("memory_pressure", status_truth)
        self.assertIn("safety_flags", status_truth)
        self.assertNotIn("retired_runtime_path", status_truth)
        self.assertNotIn("retired_runtime_path", status_truth["evidence"])
        self.assertEqual(terminus_truth["verdict"], status_truth["verdict"])
        self.assertEqual(
            capacity_expansion_design["surface"],
            "snn_language_neuron_capacity_expansion_design.v1",
        )
        self.assertFalse(capacity_expansion_design["ready"])
        self.assertFalse(capacity_expansion_design["mutates_runtime_state"])
        self.assertFalse(capacity_expansion_design["writes_checkpoint"])
        self.assertFalse(capacity_expansion_design["resizes_network"])
        self.assertFalse(capacity_expansion_design["adds_neurons"])
        self.assertFalse(capacity_expansion_design["adds_layers"])
        self.assertEqual(
            capacity_preflight["surface"],
            "snn_language_neuron_capacity_expansion_preflight.v1",
        )
        self.assertFalse(capacity_preflight["ready"])
        self.assertFalse(capacity_preflight["mutates_runtime_state"])
        self.assertFalse(capacity_preflight["writes_checkpoint"])
        self.assertFalse(capacity_preflight["resizes_network"])
        self.assertFalse(capacity_preflight["adds_neurons"])
        self.assertFalse(capacity_preflight["adds_layers"])
        self.assertEqual(
            capacity_compatibility["surface"],
            "snn_language_capacity_resize_compatibility_audit.v1",
        )
        self.assertFalse(capacity_compatibility["ready"])
        self.assertFalse(capacity_compatibility["mutates_runtime_state"])
        self.assertFalse(capacity_compatibility["writes_checkpoint"])
        self.assertFalse(capacity_compatibility["resizes_network"])
        self.assertEqual(
            dense_readout_resize_plan["surface"],
            "snn_language_dense_readout_resize_plan.v1",
        )
        self.assertTrue(dense_readout_resize_plan["advisory"])
        self.assertFalse(dense_readout_resize_plan["executable"])
        self.assertFalse(dense_readout_resize_plan["mutates_runtime_state"])
        self.assertFalse(dense_readout_resize_plan["writes_checkpoint"])
        self.assertFalse(dense_readout_resize_plan["resizes_network"])
        self.assertEqual(
            dense_readout_resize_plan["target_dense_readout_shape"],
            [128, 128],
        )
        self.assertFalse(
            dense_readout_resize_plan["promotion_gate"][
                "eligible_for_dense_readout_resize_executor"
            ]
        )
        self.assertEqual(
            dense_readout_resize_preflight["surface"],
            "snn_language_dense_readout_resize_preflight.v1",
        )
        self.assertFalse(dense_readout_resize_preflight["ready"])
        self.assertFalse(dense_readout_resize_preflight["executable"])
        self.assertFalse(dense_readout_resize_preflight["mutates_runtime_state"])
        self.assertFalse(dense_readout_resize_preflight["writes_checkpoint"])
        self.assertFalse(dense_readout_resize_preflight["resizes_network"])
        self.assertEqual(
            dense_readout_resize_preflight["dense_readout_resize_plan_hash"],
            dense_readout_resize_plan["dense_readout_resize_plan_hash"],
        )
        self.assertTrue(
            dense_readout_resize_preflight["promotion_gate"]["required_evidence"][
                "cuda_relayout_evidence_available"
            ]
        )
        self.assertFalse(
            dense_readout_resize_preflight["promotion_gate"][
                "eligible_for_dense_readout_resize_executor"
            ]
        )
        self.assertEqual(
            dense_readout_resize_transaction["surface"],
            "snn_language_dense_readout_resize_transaction_proposal.v1",
        )
        self.assertFalse(dense_readout_resize_transaction["ready"])
        self.assertFalse(dense_readout_resize_transaction["executable"])
        self.assertFalse(dense_readout_resize_transaction["mutates_runtime_state"])
        self.assertFalse(dense_readout_resize_transaction["writes_checkpoint"])
        self.assertFalse(dense_readout_resize_transaction["resizes_network"])
        self.assertEqual(
            dense_readout_resize_transaction["dense_readout_resize_plan_hash"],
            dense_readout_resize_plan["dense_readout_resize_plan_hash"],
        )
        self.assertIn(
            "allocate_target_dense_readout_tensor_on_cuda",
            dense_readout_resize_transaction["transaction_recipe"]["steps"],
        )
        self.assertFalse(
            dense_readout_resize_transaction["promotion_gate"][
                "eligible_for_dense_readout_resize_executor"
            ]
        )
        self.assertEqual(
            dense_readout_resize_readiness["surface"],
            "snn_language_dense_readout_resize_executor_readiness_audit.v1",
        )
        self.assertFalse(dense_readout_resize_readiness["executable"])
        self.assertFalse(dense_readout_resize_readiness["mutates_runtime_state"])
        self.assertFalse(dense_readout_resize_readiness["writes_checkpoint"])
        self.assertFalse(dense_readout_resize_readiness["resizes_network"])
        self.assertEqual(
            dense_readout_resize_readiness["remaining_dense_boundary_count"],
            2,
        )
        self.assertTrue(
            dense_readout_resize_readiness["promotion_gate"]["required_evidence"][
                "dense_readout_layout_state_available"
            ]
        )
        self.assertTrue(
            dense_readout_resize_readiness["promotion_gate"]["required_evidence"][
                "dense_readout_tensor_owner_available"
            ]
        )
        self.assertIn(
            "dense_readout_tensor_weight_owner_available",
            dense_readout_resize_readiness["missing_executor_capabilities"],
        )
        self.assertNotIn(
            "dense_readout_tensor_owner_available",
            dense_readout_resize_readiness["missing_executor_capabilities"],
        )
        self.assertEqual(
            dense_readout_layout_migration["surface"],
            "snn_language_dense_readout_layout_migration.v1",
        )
        self.assertFalse(dense_readout_layout_migration["accepted"])
        self.assertFalse(dense_readout_layout_migration["mutates_runtime_state"])
        self.assertFalse(dense_readout_layout_migration["writes_checkpoint"])
        self.assertFalse(dense_readout_layout_migration["resizes_network"])
        self.assertFalse(
            dense_readout_layout_migration["materializes_dense_tensor_weights"]
        )
        self.assertEqual(
            dense_readout_tensor_materialization_readiness["surface"],
            "snn_language_dense_readout_tensor_materialization_readiness.v1",
        )
        self.assertFalse(dense_readout_tensor_materialization_readiness["ready"])
        self.assertFalse(dense_readout_tensor_materialization_readiness["executable"])
        self.assertFalse(
            dense_readout_tensor_materialization_readiness["mutates_runtime_state"]
        )
        self.assertFalse(dense_readout_tensor_materialization_readiness["writes_checkpoint"])
        self.assertFalse(dense_readout_tensor_materialization_readiness["resizes_network"])
        self.assertFalse(
            dense_readout_tensor_materialization_readiness[
                "materializes_dense_tensor_weights"
            ]
        )
        self.assertEqual(
            dense_readout_tensor_materialization["surface"],
            "snn_language_dense_readout_tensor_materialization.v1",
        )
        self.assertFalse(dense_readout_tensor_materialization["accepted"])
        self.assertFalse(dense_readout_tensor_materialization["mutates_runtime_state"])
        self.assertFalse(dense_readout_tensor_materialization["writes_checkpoint"])
        self.assertFalse(dense_readout_tensor_materialization["resizes_network"])
        self.assertFalse(
            dense_readout_tensor_materialization["materializes_dense_tensor_weights"]
        )
        self.assertEqual(
            dense_readout_training_readiness["surface"],
            "snn_language_dense_readout_training_readiness.v1",
        )
        self.assertFalse(dense_readout_training_readiness["ready"])
        self.assertFalse(dense_readout_training_readiness["executable"])
        self.assertFalse(dense_readout_training_readiness["mutates_runtime_state"])
        self.assertFalse(dense_readout_training_readiness["writes_checkpoint"])
        self.assertFalse(dense_readout_training_readiness["generates_text"])
        self.assertEqual(
            dense_readout_training_loop_design["surface"],
            "snn_language_dense_readout_training_loop_design.v1",
        )
        self.assertFalse(dense_readout_training_loop_design["ready"])
        self.assertFalse(dense_readout_training_loop_design["executable"])
        self.assertFalse(dense_readout_training_loop_design["trains_runtime_model"])
        self.assertFalse(dense_readout_training_loop_design["returns_trained_weights"])
        self.assertFalse(dense_readout_training_loop_design["mutates_runtime_state"])
        self.assertFalse(dense_readout_training_loop_design["writes_checkpoint"])
        self.assertFalse(dense_readout_training_loop_design["generates_text"])
        self.assertEqual(
            dense_readout_training_loop_preflight["surface"],
            "snn_language_dense_readout_training_loop_preflight.v1",
        )
        self.assertFalse(dense_readout_training_loop_preflight["ready"])
        self.assertFalse(dense_readout_training_loop_preflight["executable"])
        self.assertFalse(dense_readout_training_loop_preflight["trains_runtime_model"])
        self.assertFalse(
            dense_readout_training_loop_preflight["returns_trained_weights"]
        )
        self.assertFalse(dense_readout_training_loop_preflight["mutates_runtime_state"])
        self.assertFalse(dense_readout_training_loop_preflight["writes_checkpoint"])
        self.assertFalse(dense_readout_training_loop_preflight["generates_text"])
        self.assertEqual(
            dense_readout_training["surface"],
            "snn_language_dense_readout_training.v1",
        )
        self.assertFalse(dense_readout_training["accepted"])
        self.assertFalse(dense_readout_training["trains_runtime_model"])
        self.assertFalse(dense_readout_training["returns_trained_weights"])
        self.assertFalse(dense_readout_training["mutates_runtime_state"])
        self.assertFalse(dense_readout_training["writes_checkpoint"])
        self.assertFalse(dense_readout_training["generates_text"])
        self.assertEqual(
            dense_readout_post_training_evaluation["surface"],
            "snn_language_dense_readout_post_training_evaluation.v1",
        )
        self.assertFalse(dense_readout_post_training_evaluation["ready"])
        self.assertFalse(dense_readout_post_training_evaluation["executable"])
        self.assertFalse(
            dense_readout_post_training_evaluation["trains_runtime_model"]
        )
        self.assertFalse(
            dense_readout_post_training_evaluation["returns_trained_weights"]
        )
        self.assertFalse(
            dense_readout_post_training_evaluation["mutates_runtime_state"]
        )
        self.assertFalse(dense_readout_post_training_evaluation["writes_checkpoint"])
        self.assertFalse(dense_readout_post_training_evaluation["generates_text"])
        self.assertEqual(
            dense_readout_decoder_probe_design["surface"],
            "snn_language_dense_readout_decoder_probe_design.v1",
        )
        self.assertFalse(dense_readout_decoder_probe_design["ready"])
        self.assertFalse(dense_readout_decoder_probe_design["executable"])
        self.assertFalse(dense_readout_decoder_probe_design["mutates_runtime_state"])
        self.assertFalse(dense_readout_decoder_probe_design["writes_checkpoint"])
        self.assertFalse(dense_readout_decoder_probe_design["generates_text"])
        self.assertFalse(
            dense_readout_decoder_probe_design["freeform_language_generation"]
        )
        self.assertEqual(
            dense_readout_decoder_probe_preflight["surface"],
            "snn_language_dense_readout_decoder_probe_preflight.v1",
        )
        self.assertFalse(dense_readout_decoder_probe_preflight["ready"])
        self.assertFalse(dense_readout_decoder_probe_preflight["executable"])
        self.assertFalse(dense_readout_decoder_probe_preflight["mutates_runtime_state"])
        self.assertFalse(dense_readout_decoder_probe_preflight["writes_checkpoint"])
        self.assertFalse(dense_readout_decoder_probe_preflight["generates_text"])
        self.assertFalse(
            dense_readout_decoder_probe_preflight["freeform_language_generation"]
        )
        self.assertEqual(
            dense_readout_decoder_probe["surface"],
            "snn_language_dense_readout_decoder_probe_execution.v1",
        )
        self.assertFalse(dense_readout_decoder_probe["ready"])
        self.assertFalse(dense_readout_decoder_probe["probe_executed"])
        self.assertFalse(dense_readout_decoder_probe["executable"])
        self.assertFalse(dense_readout_decoder_probe["mutates_runtime_state"])
        self.assertFalse(dense_readout_decoder_probe["writes_checkpoint"])
        self.assertFalse(dense_readout_decoder_probe["generates_text"])
        self.assertFalse(dense_readout_decoder_probe["freeform_language_generation"])
        self.assertEqual(
            dense_readout_label_candidate_review["surface"],
            "snn_language_dense_readout_label_candidate_review.v1",
        )
        self.assertFalse(dense_readout_label_candidate_review["ready"])
        self.assertFalse(dense_readout_label_candidate_review["review_recorded"])
        self.assertFalse(dense_readout_label_candidate_review["executable"])
        self.assertFalse(
            dense_readout_label_candidate_review["mutates_runtime_state"]
        )
        self.assertFalse(dense_readout_label_candidate_review["writes_checkpoint"])
        self.assertFalse(dense_readout_label_candidate_review["generates_text"])
        self.assertFalse(
            dense_readout_label_candidate_review["freeform_language_generation"]
        )
        self.assertFalse(dense_readout_label_candidate_review["decodes_text"])
        self.assertFalse(dense_readout_label_candidate_review["applies_plasticity"])
        self.assertFalse(
            dense_readout_label_candidate_review["records_replay_artifact"]
        )
        self.assertFalse(dense_readout_label_candidate_review["promotes_facts"])
        self.assertFalse(dense_readout_label_candidate_review["executes_actions"])
        self.assertEqual(
            dense_readout_label_candidate_record["surface"],
            "snn_language_dense_readout_label_candidate_evidence_record.v1",
        )
        self.assertFalse(dense_readout_label_candidate_record["accepted"])
        self.assertFalse(dense_readout_label_candidate_record["mutates_runtime_state"])
        self.assertFalse(dense_readout_label_candidate_record["writes_checkpoint"])
        self.assertFalse(dense_readout_label_candidate_record["generates_text"])
        self.assertFalse(dense_readout_label_candidate_record["decodes_text"])
        self.assertFalse(dense_readout_label_candidate_record["applies_plasticity"])
        self.assertFalse(
            dense_readout_label_candidate_record["records_replay_artifact"]
        )
        self.assertFalse(dense_readout_label_candidate_record["promotes_facts"])
        self.assertFalse(dense_readout_label_candidate_record["executes_actions"])
        self.assertEqual(
            dense_label_candidate_history["surface"],
            "snn_language_dense_label_candidate_history.v1",
        )
        self.assertFalse(dense_label_candidate_history["executable"])
        self.assertFalse(dense_label_candidate_history["records_ledger_event"])
        self.assertFalse(dense_label_candidate_history["runs_replay"])
        self.assertFalse(dense_label_candidate_history["writes_checkpoint"])
        self.assertFalse(dense_label_candidate_history["generates_text"])
        self.assertFalse(dense_label_candidate_history["decodes_text"])
        self.assertTrue(dense_label_candidate_history["exposes_reviewed_bounded_labels"])
        self.assertFalse(dense_label_candidate_history["applies_plasticity"])
        self.assertFalse(dense_label_candidate_history["mutates_runtime_state"])
        self.assertEqual(
            dense_label_candidate_history["summary"][
                "returned_dense_label_candidate_event_count"
            ],
            0,
        )
        self.assertFalse(
            dense_label_candidate_history["promotion_gate"][
                "eligible_for_operator_dense_label_candidate_history_inspection"
            ]
        )
        self.assertEqual(
            dense_label_candidate_calibration_policy["surface"],
            "snn_language_dense_label_candidate_calibration_policy.v1",
        )
        self.assertFalse(dense_label_candidate_calibration_policy["executable"])
        self.assertFalse(
            dense_label_candidate_calibration_policy["records_ledger_event"]
        )
        self.assertFalse(dense_label_candidate_calibration_policy["runs_replay"])
        self.assertFalse(dense_label_candidate_calibration_policy["writes_checkpoint"])
        self.assertFalse(dense_label_candidate_calibration_policy["generates_text"])
        self.assertFalse(dense_label_candidate_calibration_policy["decodes_text"])
        self.assertFalse(
            dense_label_candidate_calibration_policy["trains_runtime_model"]
        )
        self.assertFalse(dense_label_candidate_calibration_policy["applies_plasticity"])
        self.assertFalse(dense_label_candidate_calibration_policy["mutates_runtime_state"])
        self.assertEqual(
            dense_label_candidate_calibration_policy["ready_candidate_count"],
            0,
        )
        self.assertFalse(
            dense_label_candidate_calibration_policy["promotion_gate"][
                "eligible_for_operator_dense_label_calibration_review"
            ]
        )
        self.assertFalse(
            dense_label_candidate_calibration_policy["promotion_gate"][
                "eligible_for_language_generation"
            ]
        )
        self.assertEqual(
            dense_label_candidate_calibration_design["surface"],
            "snn_language_dense_label_candidate_calibration_evaluation_design.v1",
        )
        self.assertFalse(dense_label_candidate_calibration_design["ready"])
        self.assertFalse(dense_label_candidate_calibration_design["executable"])
        self.assertFalse(
            dense_label_candidate_calibration_design["records_ledger_event"]
        )
        self.assertFalse(dense_label_candidate_calibration_design["runs_replay"])
        self.assertFalse(
            dense_label_candidate_calibration_design["runs_calibration_evaluation"]
        )
        self.assertFalse(dense_label_candidate_calibration_design["writes_checkpoint"])
        self.assertFalse(dense_label_candidate_calibration_design["generates_text"])
        self.assertFalse(dense_label_candidate_calibration_design["decodes_text"])
        self.assertFalse(
            dense_label_candidate_calibration_design["trains_runtime_model"]
        )
        self.assertFalse(dense_label_candidate_calibration_design["applies_plasticity"])
        self.assertFalse(dense_label_candidate_calibration_design["mutates_runtime_state"])
        self.assertFalse(
            dense_label_candidate_calibration_design["promotion_gate"][
                "eligible_for_language_generation"
            ]
        )
        self.assertEqual(
            dense_label_candidate_calibration_preflight["surface"],
            "snn_language_dense_label_candidate_calibration_evaluation_preflight.v1",
        )
        self.assertFalse(dense_label_candidate_calibration_preflight["ready"])
        self.assertFalse(dense_label_candidate_calibration_preflight["executable"])
        self.assertFalse(
            dense_label_candidate_calibration_preflight["records_ledger_event"]
        )
        self.assertFalse(dense_label_candidate_calibration_preflight["runs_replay"])
        self.assertFalse(
            dense_label_candidate_calibration_preflight[
                "runs_calibration_evaluation"
            ]
        )
        self.assertFalse(dense_label_candidate_calibration_preflight["writes_checkpoint"])
        self.assertFalse(dense_label_candidate_calibration_preflight["generates_text"])
        self.assertFalse(dense_label_candidate_calibration_preflight["decodes_text"])
        self.assertFalse(
            dense_label_candidate_calibration_preflight["trains_runtime_model"]
        )
        self.assertFalse(dense_label_candidate_calibration_preflight["applies_plasticity"])
        self.assertFalse(dense_label_candidate_calibration_preflight["mutates_runtime_state"])
        self.assertFalse(
            dense_label_candidate_calibration_preflight["promotion_gate"][
                "eligible_for_language_generation"
            ]
        )
        self.assertEqual(
            dense_label_candidate_calibration_evaluation["surface"],
            "snn_language_dense_label_candidate_calibration_evaluation.v1",
        )
        self.assertFalse(dense_label_candidate_calibration_evaluation["ready"])
        self.assertFalse(dense_label_candidate_calibration_evaluation["executable"])
        self.assertFalse(
            dense_label_candidate_calibration_evaluation["records_ledger_event"]
        )
        self.assertFalse(dense_label_candidate_calibration_evaluation["runs_replay"])
        self.assertFalse(dense_label_candidate_calibration_evaluation["writes_checkpoint"])
        self.assertFalse(dense_label_candidate_calibration_evaluation["generates_text"])
        self.assertFalse(dense_label_candidate_calibration_evaluation["decodes_text"])
        self.assertFalse(
            dense_label_candidate_calibration_evaluation["trains_runtime_model"]
        )
        self.assertFalse(dense_label_candidate_calibration_evaluation["applies_plasticity"])
        self.assertFalse(dense_label_candidate_calibration_evaluation["mutates_runtime_state"])
        self.assertEqual(dense_label_candidate_calibration_evaluation["sample_count"], 0)
        self.assertFalse(
            dense_label_candidate_calibration_evaluation["promotion_gate"][
                "eligible_for_language_generation"
            ]
        )
        self.assertEqual(
            dense_label_candidate_calibration_review["surface"],
            "snn_language_dense_label_candidate_calibration_evaluation_review.v1",
        )
        self.assertFalse(dense_label_candidate_calibration_review["ready"])
        self.assertFalse(dense_label_candidate_calibration_review["review_recorded"])
        self.assertFalse(dense_label_candidate_calibration_review["executable"])
        self.assertFalse(
            dense_label_candidate_calibration_review["records_ledger_event"]
        )
        self.assertFalse(dense_label_candidate_calibration_review["runs_replay"])
        self.assertFalse(
            dense_label_candidate_calibration_review["runs_calibration_evaluation"]
        )
        self.assertFalse(dense_label_candidate_calibration_review["writes_checkpoint"])
        self.assertFalse(dense_label_candidate_calibration_review["generates_text"])
        self.assertFalse(dense_label_candidate_calibration_review["decodes_text"])
        self.assertFalse(
            dense_label_candidate_calibration_review["trains_runtime_model"]
        )
        self.assertFalse(dense_label_candidate_calibration_review["applies_plasticity"])
        self.assertFalse(dense_label_candidate_calibration_review["mutates_runtime_state"])
        self.assertFalse(
            dense_label_candidate_calibration_review["promotion_gate"][
                "eligible_for_language_generation"
            ]
        )
        self.assertEqual(
            dense_label_candidate_calibration_update_design["surface"],
            "snn_language_dense_label_candidate_calibration_update_design.v1",
        )
        self.assertFalse(dense_label_candidate_calibration_update_design["ready"])
        self.assertFalse(dense_label_candidate_calibration_update_design["executable"])
        self.assertFalse(
            dense_label_candidate_calibration_update_design["records_ledger_event"]
        )
        self.assertFalse(dense_label_candidate_calibration_update_design["runs_replay"])
        self.assertFalse(dense_label_candidate_calibration_update_design["writes_checkpoint"])
        self.assertFalse(dense_label_candidate_calibration_update_design["generates_text"])
        self.assertFalse(dense_label_candidate_calibration_update_design["decodes_text"])
        self.assertFalse(
            dense_label_candidate_calibration_update_design["trains_runtime_model"]
        )
        self.assertFalse(dense_label_candidate_calibration_update_design["applies_plasticity"])
        self.assertFalse(dense_label_candidate_calibration_update_design["mutates_runtime_state"])
        self.assertFalse(
            dense_label_candidate_calibration_update_design["promotion_gate"][
                "eligible_for_language_generation"
            ]
        )
        self.assertEqual(
            dense_label_candidate_calibration_update_preflight["surface"],
            "snn_language_dense_label_candidate_calibration_update_preflight.v1",
        )
        self.assertFalse(dense_label_candidate_calibration_update_preflight["ready"])
        self.assertFalse(dense_label_candidate_calibration_update_preflight["executable"])
        self.assertFalse(
            dense_label_candidate_calibration_update_preflight["records_ledger_event"]
        )
        self.assertFalse(
            dense_label_candidate_calibration_update_preflight["runs_calibration_update"]
        )
        self.assertFalse(
            dense_label_candidate_calibration_update_preflight["writes_checkpoint"]
        )
        self.assertFalse(
            dense_label_candidate_calibration_update_preflight["generates_text"]
        )
        self.assertFalse(
            dense_label_candidate_calibration_update_preflight["decodes_text"]
        )
        self.assertFalse(
            dense_label_candidate_calibration_update_preflight["trains_runtime_model"]
        )
        self.assertFalse(
            dense_label_candidate_calibration_update_preflight["applies_plasticity"]
        )
        self.assertFalse(
            dense_label_candidate_calibration_update_preflight["mutates_runtime_state"]
        )
        self.assertFalse(
            dense_label_candidate_calibration_update_preflight["promotion_gate"][
                "eligible_for_dense_label_calibration_update_executor"
            ]
        )
        self.assertFalse(
            dense_label_candidate_calibration_update_preflight["promotion_gate"][
                "eligible_for_language_generation"
            ]
        )
        self.assertFalse(
            dense_label_candidate_calibration_update_preflight["promotion_gate"][
                "required_evidence"
            ]["executor_capability_available"]
        )
        self.assertEqual(
            dense_label_candidate_calibration_update_application["surface"],
            "snn_language_dense_label_candidate_calibration_update_application.v1",
        )
        self.assertFalse(dense_label_candidate_calibration_update_application["accepted"])
        self.assertFalse(
            dense_label_candidate_calibration_update_application["records_ledger_event"]
        )
        self.assertFalse(
            dense_label_candidate_calibration_update_application["runs_calibration_update"]
        )
        self.assertFalse(
            dense_label_candidate_calibration_update_application["writes_checkpoint"]
        )
        self.assertFalse(
            dense_label_candidate_calibration_update_application["generates_text"]
        )
        self.assertFalse(
            dense_label_candidate_calibration_update_application["decodes_text"]
        )
        self.assertFalse(
            dense_label_candidate_calibration_update_application["trains_runtime_model"]
        )
        self.assertFalse(
            dense_label_candidate_calibration_update_application["applies_plasticity"]
        )
        self.assertFalse(
            dense_label_candidate_calibration_update_application["mutates_runtime_state"]
        )
        self.assertFalse(
            dense_label_candidate_calibration_update_application["promotion_gate"][
                "eligible_for_dense_label_calibration_application_review"
            ]
        )
        self.assertFalse(
            dense_label_candidate_calibration_update_application["promotion_gate"][
                "eligible_for_language_generation"
            ]
        )
        self.assertEqual(
            dense_label_candidate_calibration_update_application_review["surface"],
            "snn_language_dense_label_candidate_calibration_update_application_review.v1",
        )
        self.assertFalse(dense_label_candidate_calibration_update_application_review["ready"])
        self.assertFalse(
            dense_label_candidate_calibration_update_application_review["executable"]
        )
        self.assertFalse(
            dense_label_candidate_calibration_update_application_review[
                "records_ledger_event"
            ]
        )
        self.assertFalse(
            dense_label_candidate_calibration_update_application_review[
                "runs_calibration_update"
            ]
        )
        self.assertFalse(
            dense_label_candidate_calibration_update_application_review["writes_checkpoint"]
        )
        self.assertFalse(
            dense_label_candidate_calibration_update_application_review["generates_text"]
        )
        self.assertFalse(
            dense_label_candidate_calibration_update_application_review["decodes_text"]
        )
        self.assertFalse(
            dense_label_candidate_calibration_update_application_review[
                "trains_runtime_model"
            ]
        )
        self.assertFalse(
            dense_label_candidate_calibration_update_application_review["applies_plasticity"]
        )
        self.assertFalse(
            dense_label_candidate_calibration_update_application_review[
                "mutates_runtime_state"
            ]
        )
        self.assertFalse(
            dense_label_candidate_calibration_update_application_review["promotion_gate"][
                "eligible_for_post_calibration_observation_window"
            ]
        )
        self.assertFalse(
            dense_label_candidate_calibration_update_application_review["promotion_gate"][
                "eligible_for_language_generation"
            ]
        )
        self.assertEqual(
            dense_label_candidate_post_calibration_observation["surface"],
            "snn_language_dense_label_candidate_post_calibration_observation_window.v1",
        )
        self.assertFalse(dense_label_candidate_post_calibration_observation["ready"])
        self.assertFalse(dense_label_candidate_post_calibration_observation["executable"])
        self.assertFalse(
            dense_label_candidate_post_calibration_observation["records_ledger_event"]
        )
        self.assertFalse(
            dense_label_candidate_post_calibration_observation["runs_calibration_update"]
        )
        self.assertFalse(
            dense_label_candidate_post_calibration_observation["writes_checkpoint"]
        )
        self.assertFalse(
            dense_label_candidate_post_calibration_observation["generates_text"]
        )
        self.assertFalse(
            dense_label_candidate_post_calibration_observation["decodes_text"]
        )
        self.assertFalse(
            dense_label_candidate_post_calibration_observation["trains_runtime_model"]
        )
        self.assertFalse(
            dense_label_candidate_post_calibration_observation["applies_plasticity"]
        )
        self.assertFalse(
            dense_label_candidate_post_calibration_observation["mutates_runtime_state"]
        )
        self.assertFalse(
            dense_label_candidate_post_calibration_observation["promotion_gate"][
                "eligible_for_post_calibration_operator_review"
            ]
        )
        self.assertFalse(
            dense_label_candidate_post_calibration_observation["promotion_gate"][
                "eligible_for_language_generation"
            ]
        )
        self.assertEqual(
            dense_label_candidate_post_calibration_operator_review["surface"],
            "snn_language_dense_label_candidate_post_calibration_operator_review.v1",
        )
        self.assertFalse(dense_label_candidate_post_calibration_operator_review["ready"])
        self.assertFalse(
            dense_label_candidate_post_calibration_operator_review["executable"]
        )
        self.assertFalse(
            dense_label_candidate_post_calibration_operator_review["records_ledger_event"]
        )
        self.assertFalse(
            dense_label_candidate_post_calibration_operator_review[
                "runs_calibration_update"
            ]
        )
        self.assertFalse(
            dense_label_candidate_post_calibration_operator_review["writes_checkpoint"]
        )
        self.assertFalse(
            dense_label_candidate_post_calibration_operator_review["generates_text"]
        )
        self.assertFalse(
            dense_label_candidate_post_calibration_operator_review["decodes_text"]
        )
        self.assertFalse(
            dense_label_candidate_post_calibration_operator_review["trains_runtime_model"]
        )
        self.assertFalse(
            dense_label_candidate_post_calibration_operator_review["applies_plasticity"]
        )
        self.assertFalse(
            dense_label_candidate_post_calibration_operator_review["mutates_runtime_state"]
        )
        self.assertFalse(
            dense_label_candidate_post_calibration_operator_review["promotion_gate"][
                "eligible_for_calibrated_dense_label_confidence_use_design"
            ]
        )
        self.assertFalse(
            dense_label_candidate_post_calibration_operator_review["promotion_gate"][
                "eligible_for_language_generation"
            ]
        )
        self.assertEqual(
            calibrated_dense_label_confidence_use_design["surface"],
            "snn_language_calibrated_dense_label_confidence_use_design.v1",
        )
        self.assertFalse(calibrated_dense_label_confidence_use_design["ready"])
        self.assertFalse(calibrated_dense_label_confidence_use_design["executable"])
        self.assertFalse(
            calibrated_dense_label_confidence_use_design["records_ledger_event"]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_use_design["runs_calibration_update"]
        )
        self.assertFalse(calibrated_dense_label_confidence_use_design["writes_checkpoint"])
        self.assertFalse(calibrated_dense_label_confidence_use_design["generates_text"])
        self.assertFalse(calibrated_dense_label_confidence_use_design["decodes_text"])
        self.assertFalse(
            calibrated_dense_label_confidence_use_design["trains_runtime_model"]
        )
        self.assertFalse(calibrated_dense_label_confidence_use_design["applies_plasticity"])
        self.assertFalse(calibrated_dense_label_confidence_use_design["mutates_runtime_state"])
        self.assertFalse(
            calibrated_dense_label_confidence_use_design["promotion_gate"][
                "eligible_for_calibrated_dense_label_confidence_use_preflight"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_use_design["promotion_gate"][
                "eligible_for_language_generation"
            ]
        )
        self.assertEqual(
            calibrated_dense_label_confidence_use_preflight["surface"],
            "snn_language_calibrated_dense_label_confidence_use_preflight.v1",
        )
        self.assertFalse(calibrated_dense_label_confidence_use_preflight["ready"])
        self.assertFalse(calibrated_dense_label_confidence_use_preflight["executable"])
        self.assertFalse(
            calibrated_dense_label_confidence_use_preflight["records_ledger_event"]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_use_preflight["runs_calibration_update"]
        )
        self.assertFalse(calibrated_dense_label_confidence_use_preflight["writes_checkpoint"])
        self.assertFalse(calibrated_dense_label_confidence_use_preflight["generates_text"])
        self.assertFalse(calibrated_dense_label_confidence_use_preflight["decodes_text"])
        self.assertFalse(
            calibrated_dense_label_confidence_use_preflight["trains_runtime_model"]
        )
        self.assertFalse(calibrated_dense_label_confidence_use_preflight["applies_plasticity"])
        self.assertFalse(calibrated_dense_label_confidence_use_preflight["mutates_runtime_state"])
        self.assertFalse(
            calibrated_dense_label_confidence_use_preflight["promotion_gate"][
                "eligible_for_calibrated_dense_label_confidence_use_executor"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_use_preflight["promotion_gate"][
                "eligible_for_language_generation"
            ]
        )
        self.assertEqual(
            calibrated_dense_label_confidence_use_executor["surface"],
            "snn_language_calibrated_dense_label_confidence_use_executor.v1",
        )
        self.assertFalse(calibrated_dense_label_confidence_use_executor["ready"])
        self.assertFalse(calibrated_dense_label_confidence_use_executor["executable"])
        self.assertFalse(
            calibrated_dense_label_confidence_use_executor["records_ledger_event"]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_use_executor["runs_calibration_update"]
        )
        self.assertFalse(calibrated_dense_label_confidence_use_executor["writes_checkpoint"])
        self.assertFalse(calibrated_dense_label_confidence_use_executor["generates_text"])
        self.assertFalse(calibrated_dense_label_confidence_use_executor["decodes_text"])
        self.assertFalse(
            calibrated_dense_label_confidence_use_executor["trains_runtime_model"]
        )
        self.assertFalse(calibrated_dense_label_confidence_use_executor["applies_plasticity"])
        self.assertFalse(calibrated_dense_label_confidence_use_executor["mutates_runtime_state"])
        self.assertFalse(
            calibrated_dense_label_confidence_use_executor["promotion_gate"][
                "eligible_for_operator_display_confidence_result"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_use_executor["promotion_gate"][
                "eligible_for_language_generation"
            ]
        )
        self.assertEqual(
            calibrated_dense_label_confidence_operator_display_review["surface"],
            "snn_language_calibrated_dense_label_confidence_operator_display_review.v1",
        )
        self.assertFalse(calibrated_dense_label_confidence_operator_display_review["ready"])
        self.assertFalse(
            calibrated_dense_label_confidence_operator_display_review["executable"]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_operator_display_review[
                "records_ledger_event"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_operator_display_review[
                "runs_calibration_update"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_operator_display_review["writes_checkpoint"]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_operator_display_review["generates_text"]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_operator_display_review["decodes_text"]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_operator_display_review[
                "trains_runtime_model"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_operator_display_review[
                "applies_plasticity"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_operator_display_review[
                "mutates_runtime_state"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_operator_display_review["promotion_gate"][
                "eligible_for_operator_display_only"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_operator_display_review["promotion_gate"][
                "eligible_for_language_generation"
            ]
        )
        self.assertEqual(
            calibrated_dense_label_confidence_internal_stability_review["surface"],
            "snn_language_calibrated_dense_label_confidence_internal_stability_review.v1",
        )
        self.assertFalse(calibrated_dense_label_confidence_internal_stability_review["ready"])
        self.assertFalse(
            calibrated_dense_label_confidence_internal_stability_review[
                "requires_operator_approval"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_internal_stability_review["executable"]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_internal_stability_review[
                "records_ledger_event"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_internal_stability_review[
                "runs_calibration_update"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_internal_stability_review["writes_checkpoint"]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_internal_stability_review["generates_text"]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_internal_stability_review["decodes_text"]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_internal_stability_review[
                "trains_runtime_model"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_internal_stability_review[
                "applies_plasticity"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_internal_stability_review[
                "mutates_runtime_state"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_internal_stability_review[
                "promotion_gate"
            ]["eligible_for_autonomous_confidence_replay_review"]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_internal_stability_review[
                "promotion_gate"
            ]["eligible_for_language_generation"]
        )
        self.assertEqual(
            calibrated_dense_label_confidence_autonomous_replay_design["surface"],
            "snn_language_calibrated_dense_label_confidence_autonomous_replay_review_design.v1",
        )
        self.assertFalse(calibrated_dense_label_confidence_autonomous_replay_design["ready"])
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_replay_design[
                "requires_operator_approval"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_replay_design["executable"]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_replay_design[
                "records_ledger_event"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_replay_design["runs_replay"]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_replay_design[
                "runs_calibration_update"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_replay_design["writes_checkpoint"]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_replay_design["generates_text"]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_replay_design["decodes_text"]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_replay_design[
                "trains_runtime_model"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_replay_design[
                "applies_plasticity"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_replay_design[
                "mutates_runtime_state"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_replay_design[
                "promotion_gate"
            ]["eligible_for_autonomous_confidence_replay_review_preflight"]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_replay_design[
                "promotion_gate"
            ]["eligible_for_language_generation"]
        )
        self.assertEqual(
            calibrated_dense_label_confidence_autonomous_replay_preflight["surface"],
            "snn_language_calibrated_dense_label_confidence_autonomous_replay_review_preflight.v1",
        )
        self.assertFalse(calibrated_dense_label_confidence_autonomous_replay_preflight["ready"])
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_replay_preflight[
                "requires_operator_approval"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_replay_preflight["executable"]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_replay_preflight[
                "records_ledger_event"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_replay_preflight[
                "runs_replay"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_replay_preflight[
                "runs_calibration_update"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_replay_preflight[
                "writes_checkpoint"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_replay_preflight[
                "generates_text"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_replay_preflight[
                "decodes_text"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_replay_preflight[
                "trains_runtime_model"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_replay_preflight[
                "applies_plasticity"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_replay_preflight[
                "mutates_runtime_state"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_replay_preflight[
                "promotion_gate"
            ]["eligible_for_autonomous_confidence_replay_review_executor"]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_replay_preflight[
                "promotion_gate"
            ]["eligible_for_language_generation"]
        )
        self.assertEqual(
            calibrated_dense_label_confidence_autonomous_replay_executor["surface"],
            "snn_language_calibrated_dense_label_confidence_autonomous_replay_review_executor.v1",
        )
        self.assertFalse(calibrated_dense_label_confidence_autonomous_replay_executor["ready"])
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_replay_executor[
                "requires_operator_approval"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_replay_executor["executable"]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_replay_executor[
                "records_ledger_event"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_replay_executor["runs_replay"]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_replay_executor[
                "runs_live_replay"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_replay_executor[
                "runs_calibration_update"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_replay_executor[
                "writes_checkpoint"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_replay_executor[
                "generates_text"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_replay_executor[
                "decodes_text"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_replay_executor[
                "trains_runtime_model"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_replay_executor[
                "applies_plasticity"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_replay_executor[
                "mutates_runtime_state"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_replay_executor[
                "promotion_gate"
            ]["eligible_for_autonomous_confidence_recalibration_design"]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_replay_executor[
                "promotion_gate"
            ]["eligible_for_language_generation"]
        )
        self.assertEqual(
            calibrated_dense_label_confidence_autonomous_recalibration_design[
                "surface"
            ],
            "snn_language_calibrated_dense_label_confidence_autonomous_recalibration_design.v1",
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_recalibration_design["ready"]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_recalibration_design[
                "requires_operator_approval"
            ]
        )
        self.assertTrue(
            calibrated_dense_label_confidence_autonomous_recalibration_design[
                "advisory"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_recalibration_design[
                "executable"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_recalibration_design[
                "records_ledger_event"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_recalibration_design[
                "runs_replay"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_recalibration_design[
                "runs_recalibration"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_recalibration_design[
                "runs_calibration_update"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_recalibration_design[
                "writes_checkpoint"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_recalibration_design[
                "generates_text"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_recalibration_design[
                "decodes_text"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_recalibration_design[
                "trains_runtime_model"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_recalibration_design[
                "applies_plasticity"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_recalibration_design[
                "mutates_runtime_state"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_recalibration_design[
                "promotion_gate"
            ]["eligible_for_autonomous_confidence_recalibration_preflight"]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_recalibration_design[
                "promotion_gate"
            ]["eligible_for_autonomous_confidence_recalibration_executor"]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_recalibration_design[
                "promotion_gate"
            ]["eligible_for_language_generation"]
        )
        self.assertEqual(
            calibrated_dense_label_confidence_autonomous_recalibration_preflight[
                "surface"
            ],
            "snn_language_calibrated_dense_label_confidence_autonomous_recalibration_preflight.v1",
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_recalibration_preflight["ready"]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_recalibration_preflight[
                "requires_operator_approval"
            ]
        )
        self.assertTrue(
            calibrated_dense_label_confidence_autonomous_recalibration_preflight[
                "advisory"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_recalibration_preflight[
                "executable"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_recalibration_preflight[
                "records_ledger_event"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_recalibration_preflight[
                "runs_replay"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_recalibration_preflight[
                "runs_recalibration"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_recalibration_preflight[
                "runs_calibration_update"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_recalibration_preflight[
                "writes_checkpoint"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_recalibration_preflight[
                "generates_text"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_recalibration_preflight[
                "decodes_text"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_recalibration_preflight[
                "trains_runtime_model"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_recalibration_preflight[
                "applies_plasticity"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_recalibration_preflight[
                "mutates_runtime_state"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_recalibration_preflight[
                "promotion_gate"
            ]["eligible_for_autonomous_confidence_recalibration_executor"]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_recalibration_preflight[
                "promotion_gate"
            ]["eligible_for_language_generation"]
        )
        self.assertEqual(
            calibrated_dense_label_confidence_autonomous_recalibration_executor[
                "surface"
            ],
            "snn_language_calibrated_dense_label_confidence_autonomous_recalibration_executor.v1",
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_recalibration_executor[
                "accepted"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_recalibration_executor[
                "requires_operator_approval"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_recalibration_executor[
                "records_ledger_event"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_recalibration_executor[
                "runs_replay"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_recalibration_executor[
                "runs_recalibration"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_recalibration_executor[
                "runs_calibration_update"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_recalibration_executor[
                "writes_checkpoint"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_recalibration_executor[
                "generates_text"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_recalibration_executor[
                "decodes_text"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_recalibration_executor[
                "trains_runtime_model"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_recalibration_executor[
                "applies_plasticity"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_recalibration_executor[
                "mutates_runtime_state"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_recalibration_executor[
                "promotion_gate"
            ]["eligible_for_autonomous_confidence_recalibration_application_review"]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_recalibration_executor[
                "promotion_gate"
            ]["eligible_for_language_generation"]
        )
        self.assertEqual(
            calibrated_dense_label_confidence_autonomous_recalibration_application_review[
                "surface"
            ],
            "snn_language_calibrated_dense_label_confidence_autonomous_recalibration_application_review.v1",
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_recalibration_application_review[
                "ready"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_recalibration_application_review[
                "requires_operator_approval"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_recalibration_application_review[
                "executable"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_recalibration_application_review[
                "records_ledger_event"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_recalibration_application_review[
                "runs_replay"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_recalibration_application_review[
                "runs_recalibration"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_recalibration_application_review[
                "runs_calibration_update"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_recalibration_application_review[
                "writes_checkpoint"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_recalibration_application_review[
                "generates_text"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_recalibration_application_review[
                "decodes_text"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_recalibration_application_review[
                "trains_runtime_model"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_recalibration_application_review[
                "applies_plasticity"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_recalibration_application_review[
                "mutates_runtime_state"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_recalibration_application_review[
                "promotion_gate"
            ]["eligible_for_autonomous_post_calibration_observation_window"]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_recalibration_application_review[
                "promotion_gate"
            ]["eligible_for_language_generation"]
        )
        self.assertEqual(
            calibrated_dense_label_confidence_autonomous_post_calibration_observation[
                "surface"
            ],
            "snn_language_calibrated_dense_label_confidence_autonomous_post_calibration_observation_window.v1",
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_post_calibration_observation[
                "ready"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_post_calibration_observation[
                "requires_operator_approval"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_post_calibration_observation[
                "executable"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_post_calibration_observation[
                "records_ledger_event"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_post_calibration_observation[
                "runs_replay"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_post_calibration_observation[
                "runs_recalibration"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_post_calibration_observation[
                "runs_calibration_update"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_post_calibration_observation[
                "writes_checkpoint"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_post_calibration_observation[
                "generates_text"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_post_calibration_observation[
                "decodes_text"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_post_calibration_observation[
                "trains_runtime_model"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_post_calibration_observation[
                "applies_plasticity"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_post_calibration_observation[
                "mutates_runtime_state"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_post_calibration_observation[
                "promotion_gate"
            ]["eligible_for_autonomous_post_calibration_stability_review"]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_post_calibration_observation[
                "promotion_gate"
            ]["eligible_for_language_generation"]
        )
        self.assertEqual(
            calibrated_dense_label_confidence_autonomous_post_calibration_stability_review[
                "surface"
            ],
            "snn_language_calibrated_dense_label_confidence_autonomous_post_calibration_stability_review.v1",
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_post_calibration_stability_review[
                "ready"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_post_calibration_stability_review[
                "requires_operator_approval"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_post_calibration_stability_review[
                "executable"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_post_calibration_stability_review[
                "records_ledger_event"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_post_calibration_stability_review[
                "runs_replay"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_post_calibration_stability_review[
                "runs_recalibration"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_post_calibration_stability_review[
                "runs_calibration_update"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_post_calibration_stability_review[
                "writes_checkpoint"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_post_calibration_stability_review[
                "generates_text"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_post_calibration_stability_review[
                "decodes_text"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_post_calibration_stability_review[
                "trains_runtime_model"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_post_calibration_stability_review[
                "applies_plasticity"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_post_calibration_stability_review[
                "mutates_runtime_state"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_post_calibration_stability_review[
                "promotion_gate"
            ]["eligible_for_autonomous_calibrated_confidence_use_design"]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_post_calibration_stability_review[
                "promotion_gate"
            ]["eligible_for_language_generation"]
        )
        self.assertEqual(
            calibrated_dense_label_confidence_autonomous_use_design["surface"],
            "snn_language_calibrated_dense_label_confidence_autonomous_use_design.v1",
        )
        self.assertFalse(calibrated_dense_label_confidence_autonomous_use_design["ready"])
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_use_design[
                "requires_operator_approval"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_use_design["executable"]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_use_design[
                "records_ledger_event"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_use_design["runs_replay"]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_use_design[
                "runs_recalibration"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_use_design[
                "runs_calibration_update"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_use_design["writes_checkpoint"]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_use_design["generates_text"]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_use_design["decodes_text"]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_use_design[
                "trains_runtime_model"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_use_design["applies_plasticity"]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_use_design[
                "mutates_runtime_state"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_use_design["promotion_gate"][
                "eligible_for_autonomous_calibrated_confidence_use_preflight"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_use_design["promotion_gate"][
                "eligible_for_language_generation"
            ]
        )
        self.assertEqual(
            calibrated_dense_label_confidence_autonomous_use_preflight["surface"],
            "snn_language_calibrated_dense_label_confidence_autonomous_use_preflight.v1",
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_use_preflight["ready"]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_use_preflight[
                "requires_operator_approval"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_use_preflight["executable"]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_use_preflight[
                "records_ledger_event"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_use_preflight["runs_replay"]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_use_preflight[
                "runs_recalibration"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_use_preflight[
                "runs_calibration_update"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_use_preflight[
                "writes_checkpoint"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_use_preflight["generates_text"]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_use_preflight["decodes_text"]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_use_preflight[
                "trains_runtime_model"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_use_preflight[
                "applies_plasticity"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_use_preflight[
                "mutates_runtime_state"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_use_preflight[
                "promotion_gate"
            ]["eligible_for_autonomous_calibrated_confidence_use_executor"]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_use_preflight[
                "promotion_gate"
            ]["eligible_for_language_generation"]
        )
        self.assertEqual(
            calibrated_dense_label_confidence_autonomous_use_executor["surface"],
            "snn_language_calibrated_dense_label_confidence_autonomous_use_executor.v1",
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_use_executor["accepted"]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_use_executor[
                "requires_operator_approval"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_use_executor[
                "records_ledger_event"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_use_executor[
                "mutates_runtime_state"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_use_executor["runs_replay"]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_use_executor[
                "runs_recalibration"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_use_executor[
                "runs_calibration_update"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_use_executor[
                "writes_checkpoint"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_use_executor["generates_text"]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_use_executor["decodes_text"]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_use_executor[
                "trains_runtime_model"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_use_executor[
                "applies_plasticity"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_use_executor[
                "promotion_gate"
            ]["eligible_for_language_generation"]
        )
        self.assertEqual(
            calibrated_dense_label_confidence_autonomous_use_event_review["surface"],
            "snn_language_calibrated_dense_label_confidence_autonomous_use_event_review.v1",
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_use_event_review["ready"]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_use_event_review[
                "requires_operator_approval"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_use_event_review[
                "executable"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_use_event_review[
                "records_ledger_event"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_use_event_review[
                "mutates_runtime_state"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_use_event_review[
                "runs_replay"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_use_event_review[
                "runs_recalibration"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_use_event_review[
                "runs_calibration_update"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_use_event_review[
                "writes_checkpoint"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_use_event_review[
                "generates_text"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_use_event_review[
                "decodes_text"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_use_event_review[
                "trains_runtime_model"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_use_event_review[
                "applies_plasticity"
            ]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_use_event_review[
                "promotion_gate"
            ]["eligible_for_autonomous_hash_readout_binding_design"]
        )
        self.assertFalse(
            calibrated_dense_label_confidence_autonomous_use_event_review[
                "promotion_gate"
            ]["eligible_for_language_generation"]
        )
        self.assertEqual(
            autonomous_hash_readout_binding_design["surface"],
            "snn_language_autonomous_hash_readout_binding_design.v1",
        )
        self.assertFalse(autonomous_hash_readout_binding_design["ready"])
        self.assertFalse(
            autonomous_hash_readout_binding_design["requires_operator_approval"]
        )
        self.assertFalse(autonomous_hash_readout_binding_design["executable"])
        self.assertFalse(
            autonomous_hash_readout_binding_design["records_ledger_event"]
        )
        self.assertFalse(
            autonomous_hash_readout_binding_design["mutates_runtime_state"]
        )
        self.assertFalse(autonomous_hash_readout_binding_design["runs_replay"])
        self.assertFalse(
            autonomous_hash_readout_binding_design["runs_calibration_update"]
        )
        self.assertFalse(autonomous_hash_readout_binding_design["writes_checkpoint"])
        self.assertFalse(autonomous_hash_readout_binding_design["generates_text"])
        self.assertFalse(autonomous_hash_readout_binding_design["decodes_text"])
        self.assertFalse(
            autonomous_hash_readout_binding_design["trains_runtime_model"]
        )
        self.assertFalse(autonomous_hash_readout_binding_design["applies_plasticity"])
        self.assertFalse(
            autonomous_hash_readout_binding_design["promotion_gate"][
                "eligible_for_autonomous_hash_readout_binding_preflight"
            ]
        )
        self.assertFalse(
            autonomous_hash_readout_binding_design["promotion_gate"][
                "eligible_for_language_generation"
            ]
        )
        self.assertEqual(
            autonomous_hash_readout_binding_preflight["surface"],
            "snn_language_autonomous_hash_readout_binding_preflight.v1",
        )
        self.assertFalse(autonomous_hash_readout_binding_preflight["ready"])
        self.assertFalse(
            autonomous_hash_readout_binding_preflight["requires_operator_approval"]
        )
        self.assertFalse(autonomous_hash_readout_binding_preflight["executable"])
        self.assertFalse(
            autonomous_hash_readout_binding_preflight["records_ledger_event"]
        )
        self.assertFalse(
            autonomous_hash_readout_binding_preflight["mutates_runtime_state"]
        )
        self.assertFalse(autonomous_hash_readout_binding_preflight["runs_replay"])
        self.assertFalse(
            autonomous_hash_readout_binding_preflight["runs_calibration_update"]
        )
        self.assertFalse(autonomous_hash_readout_binding_preflight["writes_checkpoint"])
        self.assertFalse(autonomous_hash_readout_binding_preflight["generates_text"])
        self.assertFalse(autonomous_hash_readout_binding_preflight["decodes_text"])
        self.assertFalse(
            autonomous_hash_readout_binding_preflight["trains_runtime_model"]
        )
        self.assertFalse(
            autonomous_hash_readout_binding_preflight["applies_plasticity"]
        )
        self.assertFalse(
            autonomous_hash_readout_binding_preflight["promotion_gate"][
                "eligible_for_autonomous_hash_readout_binding_executor"
            ]
        )
        self.assertFalse(
            autonomous_hash_readout_binding_preflight["promotion_gate"][
                "eligible_for_language_generation"
            ]
        )
        self.assertEqual(
            autonomous_hash_readout_binding_executor["surface"],
            "snn_language_autonomous_hash_readout_binding_executor.v1",
        )
        self.assertFalse(autonomous_hash_readout_binding_executor["accepted"])
        self.assertFalse(
            autonomous_hash_readout_binding_executor["requires_operator_approval"]
        )
        self.assertFalse(
            autonomous_hash_readout_binding_executor["records_ledger_event"]
        )
        self.assertFalse(
            autonomous_hash_readout_binding_executor["mutates_runtime_state"]
        )
        self.assertFalse(autonomous_hash_readout_binding_executor["runs_replay"])
        self.assertFalse(
            autonomous_hash_readout_binding_executor["runs_calibration_update"]
        )
        self.assertFalse(autonomous_hash_readout_binding_executor["writes_checkpoint"])
        self.assertFalse(autonomous_hash_readout_binding_executor["generates_text"])
        self.assertFalse(autonomous_hash_readout_binding_executor["decodes_text"])
        self.assertFalse(
            autonomous_hash_readout_binding_executor["trains_runtime_model"]
        )
        self.assertFalse(autonomous_hash_readout_binding_executor["applies_plasticity"])
        self.assertFalse(
            autonomous_hash_readout_binding_executor["promotion_gate"][
                "eligible_for_autonomous_hash_readout_binding_event_review"
            ]
        )
        self.assertFalse(
            autonomous_hash_readout_binding_executor["promotion_gate"][
                "eligible_for_language_generation"
            ]
        )
        self.assertEqual(
            autonomous_hash_readout_binding_event_review["surface"],
            "snn_language_autonomous_hash_readout_binding_event_review.v1",
        )
        self.assertFalse(autonomous_hash_readout_binding_event_review["ready"])
        self.assertFalse(
            autonomous_hash_readout_binding_event_review["requires_operator_approval"]
        )
        self.assertFalse(autonomous_hash_readout_binding_event_review["executable"])
        self.assertFalse(
            autonomous_hash_readout_binding_event_review["records_ledger_event"]
        )
        self.assertFalse(
            autonomous_hash_readout_binding_event_review["mutates_runtime_state"]
        )
        self.assertFalse(autonomous_hash_readout_binding_event_review["runs_replay"])
        self.assertFalse(
            autonomous_hash_readout_binding_event_review["runs_calibration_update"]
        )
        self.assertFalse(
            autonomous_hash_readout_binding_event_review["writes_checkpoint"]
        )
        self.assertFalse(autonomous_hash_readout_binding_event_review["generates_text"])
        self.assertFalse(autonomous_hash_readout_binding_event_review["decodes_text"])
        self.assertFalse(
            autonomous_hash_readout_binding_event_review["trains_runtime_model"]
        )
        self.assertFalse(
            autonomous_hash_readout_binding_event_review["applies_plasticity"]
        )
        self.assertFalse(
            autonomous_hash_readout_binding_event_review["promotion_gate"][
                "eligible_for_autonomous_bound_readout_observation_design"
            ]
        )
        self.assertFalse(
            autonomous_hash_readout_binding_event_review["promotion_gate"][
                "eligible_for_language_generation"
            ]
        )
        self.assertEqual(
            autonomous_bound_readout_observation_design["surface"],
            "snn_language_autonomous_bound_readout_observation_design.v1",
        )
        self.assertFalse(autonomous_bound_readout_observation_design["ready"])
        self.assertFalse(
            autonomous_bound_readout_observation_design["requires_operator_approval"]
        )
        self.assertFalse(autonomous_bound_readout_observation_design["executable"])
        self.assertFalse(
            autonomous_bound_readout_observation_design["records_ledger_event"]
        )
        self.assertFalse(
            autonomous_bound_readout_observation_design["mutates_runtime_state"]
        )
        self.assertFalse(autonomous_bound_readout_observation_design["runs_replay"])
        self.assertFalse(
            autonomous_bound_readout_observation_design["runs_calibration_update"]
        )
        self.assertFalse(
            autonomous_bound_readout_observation_design["writes_checkpoint"]
        )
        self.assertFalse(autonomous_bound_readout_observation_design["generates_text"])
        self.assertFalse(autonomous_bound_readout_observation_design["decodes_text"])
        self.assertFalse(
            autonomous_bound_readout_observation_design["trains_runtime_model"]
        )
        self.assertFalse(
            autonomous_bound_readout_observation_design["applies_plasticity"]
        )
        self.assertFalse(
            autonomous_bound_readout_observation_design["promotion_gate"][
                "eligible_for_autonomous_bound_readout_observation_preflight"
            ]
        )
        self.assertFalse(
            autonomous_bound_readout_observation_design["promotion_gate"][
                "eligible_for_language_generation"
            ]
        )
        self.assertEqual(
            autonomous_bound_readout_observation_preflight["surface"],
            "snn_language_autonomous_bound_readout_observation_preflight.v1",
        )
        self.assertFalse(autonomous_bound_readout_observation_preflight["ready"])
        self.assertFalse(
            autonomous_bound_readout_observation_preflight[
                "requires_operator_approval"
            ]
        )
        self.assertFalse(autonomous_bound_readout_observation_preflight["executable"])
        self.assertFalse(
            autonomous_bound_readout_observation_preflight["records_ledger_event"]
        )
        self.assertFalse(
            autonomous_bound_readout_observation_preflight["mutates_runtime_state"]
        )
        self.assertFalse(autonomous_bound_readout_observation_preflight["runs_replay"])
        self.assertFalse(
            autonomous_bound_readout_observation_preflight["runs_calibration_update"]
        )
        self.assertFalse(
            autonomous_bound_readout_observation_preflight["writes_checkpoint"]
        )
        self.assertFalse(
            autonomous_bound_readout_observation_preflight["generates_text"]
        )
        self.assertFalse(
            autonomous_bound_readout_observation_preflight["decodes_text"]
        )
        self.assertFalse(
            autonomous_bound_readout_observation_preflight["trains_runtime_model"]
        )
        self.assertFalse(
            autonomous_bound_readout_observation_preflight["applies_plasticity"]
        )
        self.assertFalse(
            autonomous_bound_readout_observation_preflight["promotion_gate"][
                "eligible_for_autonomous_bound_readout_observation_executor"
            ]
        )
        self.assertFalse(
            autonomous_bound_readout_observation_preflight["promotion_gate"][
                "eligible_for_language_generation"
            ]
        )
        self.assertEqual(
            autonomous_bound_readout_observation_executor["surface"],
            "snn_language_autonomous_bound_readout_observation_executor.v1",
        )
        self.assertFalse(autonomous_bound_readout_observation_executor["accepted"])
        self.assertFalse(
            autonomous_bound_readout_observation_executor[
                "requires_operator_approval"
            ]
        )
        self.assertFalse(autonomous_bound_readout_observation_executor["executable"])
        self.assertFalse(
            autonomous_bound_readout_observation_executor["records_ledger_event"]
        )
        self.assertFalse(
            autonomous_bound_readout_observation_executor["mutates_runtime_state"]
        )
        self.assertFalse(autonomous_bound_readout_observation_executor["runs_replay"])
        self.assertFalse(
            autonomous_bound_readout_observation_executor["runs_calibration_update"]
        )
        self.assertFalse(
            autonomous_bound_readout_observation_executor["writes_checkpoint"]
        )
        self.assertFalse(
            autonomous_bound_readout_observation_executor["generates_text"]
        )
        self.assertFalse(autonomous_bound_readout_observation_executor["decodes_text"])
        self.assertFalse(
            autonomous_bound_readout_observation_executor["trains_runtime_model"]
        )
        self.assertFalse(
            autonomous_bound_readout_observation_executor["applies_plasticity"]
        )
        self.assertFalse(
            autonomous_bound_readout_observation_executor["promotion_gate"][
                "eligible_for_autonomous_bound_readout_observation_event_review"
            ]
        )
        self.assertFalse(
            autonomous_bound_readout_observation_executor["promotion_gate"][
                "eligible_for_language_generation"
            ]
        )
        self.assertEqual(
            autonomous_bound_readout_observation_event_review["surface"],
            "snn_language_autonomous_bound_readout_observation_event_review.v1",
        )
        self.assertFalse(autonomous_bound_readout_observation_event_review["ready"])
        self.assertFalse(
            autonomous_bound_readout_observation_event_review[
                "requires_operator_approval"
            ]
        )
        self.assertTrue(autonomous_bound_readout_observation_event_review["advisory"])
        self.assertFalse(autonomous_bound_readout_observation_event_review["executable"])
        self.assertFalse(
            autonomous_bound_readout_observation_event_review["records_ledger_event"]
        )
        self.assertFalse(
            autonomous_bound_readout_observation_event_review["mutates_runtime_state"]
        )
        self.assertFalse(
            autonomous_bound_readout_observation_event_review["runs_replay"]
        )
        self.assertFalse(
            autonomous_bound_readout_observation_event_review[
                "runs_calibration_update"
            ]
        )
        self.assertFalse(
            autonomous_bound_readout_observation_event_review["writes_checkpoint"]
        )
        self.assertFalse(
            autonomous_bound_readout_observation_event_review["generates_text"]
        )
        self.assertFalse(
            autonomous_bound_readout_observation_event_review["decodes_text"]
        )
        self.assertFalse(
            autonomous_bound_readout_observation_event_review["trains_runtime_model"]
        )
        self.assertFalse(
            autonomous_bound_readout_observation_event_review["applies_plasticity"]
        )
        self.assertFalse(
            autonomous_bound_readout_observation_event_review["promotion_gate"][
                "eligible_for_autonomous_readout_training_window_design"
            ]
        )
        self.assertFalse(
            autonomous_bound_readout_observation_event_review["promotion_gate"][
                "eligible_for_language_generation"
            ]
        )
        self.assertEqual(
            autonomous_readout_training_window_design["surface"],
            "snn_language_autonomous_readout_training_window_design.v1",
        )
        self.assertFalse(autonomous_readout_training_window_design["ready"])
        self.assertFalse(
            autonomous_readout_training_window_design["requires_operator_approval"]
        )
        self.assertTrue(autonomous_readout_training_window_design["advisory"])
        self.assertFalse(autonomous_readout_training_window_design["executable"])
        self.assertFalse(
            autonomous_readout_training_window_design["records_ledger_event"]
        )
        self.assertFalse(
            autonomous_readout_training_window_design["mutates_runtime_state"]
        )
        self.assertFalse(autonomous_readout_training_window_design["runs_replay"])
        self.assertFalse(
            autonomous_readout_training_window_design["runs_calibration_update"]
        )
        self.assertFalse(autonomous_readout_training_window_design["writes_checkpoint"])
        self.assertFalse(autonomous_readout_training_window_design["generates_text"])
        self.assertFalse(autonomous_readout_training_window_design["decodes_text"])
        self.assertFalse(
            autonomous_readout_training_window_design["trains_runtime_model"]
        )
        self.assertFalse(autonomous_readout_training_window_design["applies_plasticity"])
        self.assertFalse(
            autonomous_readout_training_window_design["promotion_gate"][
                "eligible_for_autonomous_readout_training_window_preflight"
            ]
        )
        self.assertFalse(
            autonomous_readout_training_window_design["promotion_gate"][
                "eligible_for_autonomous_readout_training_execution"
            ]
        )
        self.assertFalse(
            autonomous_readout_training_window_design["promotion_gate"][
                "eligible_for_language_generation"
            ]
        )
        self.assertEqual(
            autonomous_readout_training_window_preflight["surface"],
            "snn_language_autonomous_readout_training_window_preflight.v1",
        )
        self.assertFalse(autonomous_readout_training_window_preflight["ready"])
        self.assertFalse(
            autonomous_readout_training_window_preflight[
                "requires_operator_approval"
            ]
        )
        self.assertTrue(autonomous_readout_training_window_preflight["advisory"])
        self.assertFalse(autonomous_readout_training_window_preflight["executable"])
        self.assertFalse(
            autonomous_readout_training_window_preflight["records_ledger_event"]
        )
        self.assertFalse(
            autonomous_readout_training_window_preflight["mutates_runtime_state"]
        )
        self.assertFalse(autonomous_readout_training_window_preflight["runs_replay"])
        self.assertFalse(
            autonomous_readout_training_window_preflight["runs_calibration_update"]
        )
        self.assertFalse(
            autonomous_readout_training_window_preflight["writes_checkpoint"]
        )
        self.assertFalse(autonomous_readout_training_window_preflight["generates_text"])
        self.assertFalse(autonomous_readout_training_window_preflight["decodes_text"])
        self.assertFalse(
            autonomous_readout_training_window_preflight["trains_runtime_model"]
        )
        self.assertFalse(
            autonomous_readout_training_window_preflight["applies_plasticity"]
        )
        self.assertFalse(
            autonomous_readout_training_window_preflight["promotion_gate"][
                "eligible_for_autonomous_readout_training_window_executor"
            ]
        )
        self.assertFalse(
            autonomous_readout_training_window_preflight["promotion_gate"][
                "eligible_for_language_generation"
            ]
        )
        self.assertEqual(
            autonomous_readout_training_window_executor["surface"],
            "snn_language_autonomous_readout_training_window_executor.v1",
        )
        self.assertFalse(autonomous_readout_training_window_executor["accepted"])
        self.assertFalse(
            autonomous_readout_training_window_executor["requires_operator_approval"]
        )
        self.assertFalse(autonomous_readout_training_window_executor["executable"])
        self.assertFalse(
            autonomous_readout_training_window_executor["records_ledger_event"]
        )
        self.assertFalse(
            autonomous_readout_training_window_executor["mutates_runtime_state"]
        )
        self.assertFalse(autonomous_readout_training_window_executor["runs_replay"])
        self.assertFalse(
            autonomous_readout_training_window_executor["runs_calibration_update"]
        )
        self.assertFalse(
            autonomous_readout_training_window_executor["writes_checkpoint"]
        )
        self.assertFalse(autonomous_readout_training_window_executor["generates_text"])
        self.assertFalse(autonomous_readout_training_window_executor["decodes_text"])
        self.assertFalse(
            autonomous_readout_training_window_executor["trains_runtime_model"]
        )
        self.assertFalse(
            autonomous_readout_training_window_executor["applies_plasticity"]
        )
        self.assertFalse(
            autonomous_readout_training_window_executor["promotion_gate"][
                "eligible_for_autonomous_readout_training_window_event_review"
            ]
        )
        self.assertFalse(
            autonomous_readout_training_window_executor["promotion_gate"][
                "eligible_for_language_generation"
            ]
        )
        self.assertEqual(
            autonomous_readout_training_window_event_review["surface"],
            "snn_language_autonomous_readout_training_window_event_review.v1",
        )
        self.assertFalse(autonomous_readout_training_window_event_review["ready"])
        self.assertFalse(
            autonomous_readout_training_window_event_review[
                "requires_operator_approval"
            ]
        )
        self.assertTrue(autonomous_readout_training_window_event_review["advisory"])
        self.assertFalse(autonomous_readout_training_window_event_review["executable"])
        self.assertFalse(
            autonomous_readout_training_window_event_review["records_ledger_event"]
        )
        self.assertFalse(
            autonomous_readout_training_window_event_review["mutates_runtime_state"]
        )
        self.assertFalse(autonomous_readout_training_window_event_review["runs_replay"])
        self.assertFalse(
            autonomous_readout_training_window_event_review[
                "runs_calibration_update"
            ]
        )
        self.assertFalse(
            autonomous_readout_training_window_event_review["writes_checkpoint"]
        )
        self.assertFalse(autonomous_readout_training_window_event_review["generates_text"])
        self.assertFalse(autonomous_readout_training_window_event_review["decodes_text"])
        self.assertFalse(
            autonomous_readout_training_window_event_review["trains_runtime_model"]
        )
        self.assertFalse(
            autonomous_readout_training_window_event_review["applies_plasticity"]
        )
        self.assertFalse(
            autonomous_readout_training_window_event_review["promotion_gate"][
                "eligible_for_autonomous_decoder_probe_design"
            ]
        )
        self.assertFalse(
            autonomous_readout_training_window_event_review["promotion_gate"][
                "eligible_for_language_generation"
            ]
        )
        self.assertEqual(
            autonomous_decoder_probe_design["surface"],
            "snn_language_autonomous_decoder_probe_design.v1",
        )
        self.assertFalse(autonomous_decoder_probe_design["ready"])
        self.assertFalse(
            autonomous_decoder_probe_design["requires_operator_approval"]
        )
        self.assertTrue(autonomous_decoder_probe_design["advisory"])
        self.assertFalse(autonomous_decoder_probe_design["executable"])
        self.assertFalse(autonomous_decoder_probe_design["records_ledger_event"])
        self.assertFalse(autonomous_decoder_probe_design["mutates_runtime_state"])
        self.assertFalse(autonomous_decoder_probe_design["runs_replay"])
        self.assertFalse(autonomous_decoder_probe_design["runs_calibration_update"])
        self.assertFalse(autonomous_decoder_probe_design["writes_checkpoint"])
        self.assertFalse(autonomous_decoder_probe_design["generates_text"])
        self.assertFalse(autonomous_decoder_probe_design["decodes_text"])
        self.assertFalse(autonomous_decoder_probe_design["trains_runtime_model"])
        self.assertFalse(autonomous_decoder_probe_design["applies_plasticity"])
        self.assertFalse(
            autonomous_decoder_probe_design["promotion_gate"][
                "eligible_for_autonomous_decoder_probe_preflight"
            ]
        )
        self.assertFalse(
            autonomous_decoder_probe_design["promotion_gate"][
                "eligible_for_language_generation"
            ]
        )
        self.assertEqual(
            autonomous_decoder_probe_preflight["surface"],
            "snn_language_autonomous_decoder_probe_preflight.v1",
        )
        self.assertFalse(autonomous_decoder_probe_preflight["ready"])
        self.assertFalse(
            autonomous_decoder_probe_preflight["requires_operator_approval"]
        )
        self.assertTrue(autonomous_decoder_probe_preflight["advisory"])
        self.assertFalse(autonomous_decoder_probe_preflight["executable"])
        self.assertFalse(autonomous_decoder_probe_preflight["records_ledger_event"])
        self.assertFalse(autonomous_decoder_probe_preflight["mutates_runtime_state"])
        self.assertFalse(autonomous_decoder_probe_preflight["runs_replay"])
        self.assertFalse(autonomous_decoder_probe_preflight["runs_calibration_update"])
        self.assertFalse(autonomous_decoder_probe_preflight["writes_checkpoint"])
        self.assertFalse(autonomous_decoder_probe_preflight["generates_text"])
        self.assertFalse(autonomous_decoder_probe_preflight["decodes_text"])
        self.assertFalse(autonomous_decoder_probe_preflight["trains_runtime_model"])
        self.assertFalse(autonomous_decoder_probe_preflight["applies_plasticity"])
        self.assertFalse(
            autonomous_decoder_probe_preflight["promotion_gate"][
                "eligible_for_autonomous_decoder_probe_executor"
            ]
        )
        self.assertFalse(
            autonomous_decoder_probe_preflight["promotion_gate"][
                "eligible_for_language_generation"
            ]
        )
        self.assertEqual(
            autonomous_decoder_probe_executor["surface"],
            "snn_language_autonomous_decoder_probe_executor.v1",
        )
        self.assertFalse(autonomous_decoder_probe_executor["accepted"])
        self.assertFalse(
            autonomous_decoder_probe_executor["requires_operator_approval"]
        )
        self.assertFalse(autonomous_decoder_probe_executor["executable"])
        self.assertFalse(autonomous_decoder_probe_executor["records_ledger_event"])
        self.assertFalse(autonomous_decoder_probe_executor["mutates_runtime_state"])
        self.assertFalse(autonomous_decoder_probe_executor["runs_replay"])
        self.assertFalse(autonomous_decoder_probe_executor["runs_calibration_update"])
        self.assertFalse(autonomous_decoder_probe_executor["writes_checkpoint"])
        self.assertFalse(autonomous_decoder_probe_executor["generates_text"])
        self.assertFalse(autonomous_decoder_probe_executor["decodes_text"])
        self.assertFalse(autonomous_decoder_probe_executor["trains_runtime_model"])
        self.assertFalse(autonomous_decoder_probe_executor["applies_plasticity"])
        self.assertFalse(
            autonomous_decoder_probe_executor["promotion_gate"][
                "eligible_for_autonomous_decoder_probe_event_review"
            ]
        )
        self.assertFalse(
            autonomous_decoder_probe_executor["promotion_gate"][
                "eligible_for_language_generation"
            ]
        )
        self.assertEqual(
            autonomous_decoder_probe_event_review["surface"],
            "snn_language_autonomous_decoder_probe_event_review.v1",
        )
        self.assertFalse(autonomous_decoder_probe_event_review["ready"])
        self.assertFalse(
            autonomous_decoder_probe_event_review["requires_operator_approval"]
        )
        self.assertTrue(autonomous_decoder_probe_event_review["advisory"])
        self.assertFalse(autonomous_decoder_probe_event_review["executable"])
        self.assertFalse(
            autonomous_decoder_probe_event_review["records_ledger_event"]
        )
        self.assertFalse(
            autonomous_decoder_probe_event_review["mutates_runtime_state"]
        )
        self.assertFalse(autonomous_decoder_probe_event_review["runs_replay"])
        self.assertFalse(
            autonomous_decoder_probe_event_review["runs_calibration_update"]
        )
        self.assertFalse(autonomous_decoder_probe_event_review["writes_checkpoint"])
        self.assertFalse(autonomous_decoder_probe_event_review["generates_text"])
        self.assertFalse(autonomous_decoder_probe_event_review["decodes_text"])
        self.assertFalse(
            autonomous_decoder_probe_event_review["trains_runtime_model"]
        )
        self.assertFalse(autonomous_decoder_probe_event_review["applies_plasticity"])
        self.assertFalse(
            autonomous_decoder_probe_event_review["promotion_gate"][
                "eligible_for_autonomous_language_output_design"
            ]
        )
        self.assertFalse(
            autonomous_decoder_probe_event_review["promotion_gate"][
                "eligible_for_language_generation"
            ]
        )
        self.assertEqual(
            autonomous_language_output_design["surface"],
            "snn_language_autonomous_language_output_design.v1",
        )
        self.assertFalse(autonomous_language_output_design["ready"])
        self.assertFalse(
            autonomous_language_output_design["requires_operator_approval"]
        )
        self.assertTrue(autonomous_language_output_design["advisory"])
        self.assertFalse(autonomous_language_output_design["executable"])
        self.assertFalse(autonomous_language_output_design["records_ledger_event"])
        self.assertFalse(autonomous_language_output_design["mutates_runtime_state"])
        self.assertFalse(autonomous_language_output_design["runs_replay"])
        self.assertFalse(autonomous_language_output_design["runs_calibration_update"])
        self.assertFalse(autonomous_language_output_design["writes_checkpoint"])
        self.assertFalse(autonomous_language_output_design["generates_text"])
        self.assertFalse(autonomous_language_output_design["decodes_text"])
        self.assertFalse(autonomous_language_output_design["trains_runtime_model"])
        self.assertFalse(autonomous_language_output_design["applies_plasticity"])
        self.assertFalse(
            autonomous_language_output_design["promotion_gate"][
                "eligible_for_autonomous_language_output_preflight"
            ]
        )
        self.assertFalse(
            autonomous_language_output_design["promotion_gate"][
                "eligible_for_language_generation"
            ]
        )
        self.assertEqual(
            autonomous_language_output_preflight["surface"],
            "snn_language_autonomous_language_output_preflight.v1",
        )
        self.assertFalse(autonomous_language_output_preflight["ready"])
        self.assertFalse(
            autonomous_language_output_preflight["requires_operator_approval"]
        )
        self.assertTrue(autonomous_language_output_preflight["advisory"])
        self.assertFalse(autonomous_language_output_preflight["executable"])
        self.assertFalse(autonomous_language_output_preflight["records_ledger_event"])
        self.assertFalse(autonomous_language_output_preflight["mutates_runtime_state"])
        self.assertFalse(autonomous_language_output_preflight["runs_replay"])
        self.assertFalse(autonomous_language_output_preflight["runs_calibration_update"])
        self.assertFalse(autonomous_language_output_preflight["writes_checkpoint"])
        self.assertFalse(autonomous_language_output_preflight["generates_text"])
        self.assertFalse(autonomous_language_output_preflight["decodes_text"])
        self.assertFalse(autonomous_language_output_preflight["trains_runtime_model"])
        self.assertFalse(autonomous_language_output_preflight["applies_plasticity"])
        self.assertFalse(
            autonomous_language_output_preflight["promotion_gate"][
                "eligible_for_autonomous_language_output_executor"
            ]
        )
        self.assertFalse(
            autonomous_language_output_preflight["promotion_gate"][
                "eligible_for_language_generation"
            ]
        )
        self.assertEqual(
            autonomous_language_output_executor["surface"],
            "snn_language_autonomous_language_output_executor.v1",
        )
        self.assertFalse(autonomous_language_output_executor["accepted"])
        self.assertFalse(
            autonomous_language_output_executor["requires_operator_approval"]
        )
        self.assertFalse(autonomous_language_output_executor["executable"])
        self.assertFalse(autonomous_language_output_executor["records_ledger_event"])
        self.assertFalse(autonomous_language_output_executor["mutates_runtime_state"])
        self.assertFalse(autonomous_language_output_executor["runs_replay"])
        self.assertFalse(autonomous_language_output_executor["runs_calibration_update"])
        self.assertFalse(autonomous_language_output_executor["writes_checkpoint"])
        self.assertFalse(autonomous_language_output_executor["generates_text"])
        self.assertFalse(autonomous_language_output_executor["decodes_text"])
        self.assertFalse(autonomous_language_output_executor["trains_runtime_model"])
        self.assertFalse(autonomous_language_output_executor["applies_plasticity"])
        self.assertFalse(
            autonomous_language_output_executor["promotion_gate"][
                "eligible_for_autonomous_language_output_event_review"
            ]
        )
        self.assertFalse(
            autonomous_language_output_executor["promotion_gate"][
                "eligible_for_language_generation"
            ]
        )
        self.assertEqual(
            autonomous_language_output_event_review["surface"],
            "snn_language_autonomous_language_output_event_review.v1",
        )
        self.assertFalse(autonomous_language_output_event_review["ready"])
        self.assertFalse(
            autonomous_language_output_event_review["requires_operator_approval"]
        )
        self.assertTrue(autonomous_language_output_event_review["advisory"])
        self.assertFalse(autonomous_language_output_event_review["executable"])
        self.assertFalse(
            autonomous_language_output_event_review["records_ledger_event"]
        )
        self.assertFalse(
            autonomous_language_output_event_review["mutates_runtime_state"]
        )
        self.assertFalse(autonomous_language_output_event_review["runs_replay"])
        self.assertFalse(
            autonomous_language_output_event_review["runs_calibration_update"]
        )
        self.assertFalse(autonomous_language_output_event_review["writes_checkpoint"])
        self.assertFalse(autonomous_language_output_event_review["generates_text"])
        self.assertFalse(autonomous_language_output_event_review["decodes_text"])
        self.assertFalse(
            autonomous_language_output_event_review["trains_runtime_model"]
        )
        self.assertFalse(autonomous_language_output_event_review["applies_plasticity"])
        self.assertFalse(
            autonomous_language_output_event_review["promotion_gate"][
                "eligible_for_autonomous_decoded_output_design"
            ]
        )
        self.assertFalse(
            autonomous_language_output_event_review["promotion_gate"][
                "eligible_for_language_generation"
            ]
        )
        self.assertEqual(
            autonomous_decoded_output_design["surface"],
            "snn_language_autonomous_decoded_output_design.v1",
        )
        self.assertFalse(autonomous_decoded_output_design["ready"])
        self.assertFalse(
            autonomous_decoded_output_design["requires_operator_approval"]
        )
        self.assertTrue(autonomous_decoded_output_design["advisory"])
        self.assertFalse(autonomous_decoded_output_design["executable"])
        self.assertFalse(autonomous_decoded_output_design["records_ledger_event"])
        self.assertFalse(autonomous_decoded_output_design["mutates_runtime_state"])
        self.assertFalse(autonomous_decoded_output_design["runs_replay"])
        self.assertFalse(autonomous_decoded_output_design["runs_calibration_update"])
        self.assertFalse(autonomous_decoded_output_design["writes_checkpoint"])
        self.assertFalse(autonomous_decoded_output_design["generates_text"])
        self.assertFalse(autonomous_decoded_output_design["decodes_text"])
        self.assertFalse(autonomous_decoded_output_design["trains_runtime_model"])
        self.assertFalse(autonomous_decoded_output_design["applies_plasticity"])
        self.assertFalse(
            autonomous_decoded_output_design["promotion_gate"][
                "eligible_for_autonomous_decoded_output_preflight"
            ]
        )
        self.assertFalse(
            autonomous_decoded_output_design["promotion_gate"][
                "eligible_for_language_generation"
            ]
        )
        self.assertEqual(
            autonomous_decoded_output_preflight["surface"],
            "snn_language_autonomous_decoded_output_preflight.v1",
        )
        self.assertFalse(autonomous_decoded_output_preflight["ready"])
        self.assertFalse(
            autonomous_decoded_output_preflight["requires_operator_approval"]
        )
        self.assertTrue(autonomous_decoded_output_preflight["advisory"])
        self.assertFalse(autonomous_decoded_output_preflight["executable"])
        self.assertFalse(autonomous_decoded_output_preflight["records_ledger_event"])
        self.assertFalse(autonomous_decoded_output_preflight["mutates_runtime_state"])
        self.assertFalse(autonomous_decoded_output_preflight["runs_replay"])
        self.assertFalse(autonomous_decoded_output_preflight["runs_calibration_update"])
        self.assertFalse(autonomous_decoded_output_preflight["writes_checkpoint"])
        self.assertFalse(autonomous_decoded_output_preflight["generates_text"])
        self.assertFalse(autonomous_decoded_output_preflight["decodes_text"])
        self.assertFalse(autonomous_decoded_output_preflight["trains_runtime_model"])
        self.assertFalse(autonomous_decoded_output_preflight["applies_plasticity"])
        self.assertFalse(
            autonomous_decoded_output_preflight["promotion_gate"][
                "eligible_for_autonomous_decoded_output_executor"
            ]
        )
        self.assertFalse(
            autonomous_decoded_output_preflight["promotion_gate"][
                "eligible_for_language_generation"
            ]
        )
        self.assertEqual(
            autonomous_decoded_output_executor["surface"],
            "snn_language_autonomous_decoded_output_executor.v1",
        )
        self.assertFalse(autonomous_decoded_output_executor["accepted"])
        self.assertFalse(
            autonomous_decoded_output_executor["requires_operator_approval"]
        )
        self.assertFalse(autonomous_decoded_output_executor["executable"])
        self.assertFalse(autonomous_decoded_output_executor["records_ledger_event"])
        self.assertFalse(autonomous_decoded_output_executor["mutates_runtime_state"])
        self.assertFalse(autonomous_decoded_output_executor["runs_replay"])
        self.assertFalse(autonomous_decoded_output_executor["runs_calibration_update"])
        self.assertFalse(autonomous_decoded_output_executor["writes_checkpoint"])
        self.assertFalse(autonomous_decoded_output_executor["generates_text"])
        self.assertFalse(autonomous_decoded_output_executor["decodes_text"])
        self.assertFalse(autonomous_decoded_output_executor["trains_runtime_model"])
        self.assertFalse(autonomous_decoded_output_executor["applies_plasticity"])
        self.assertFalse(
            autonomous_decoded_output_executor["promotion_gate"][
                "eligible_for_autonomous_decoded_output_event_review"
            ]
        )
        self.assertFalse(
            autonomous_decoded_output_executor["promotion_gate"][
                "eligible_for_language_generation"
            ]
        )
        self.assertEqual(
            autonomous_decoded_output_event_review["surface"],
            "snn_language_autonomous_decoded_output_event_review.v1",
        )
        self.assertFalse(autonomous_decoded_output_event_review["ready"])
        self.assertFalse(
            autonomous_decoded_output_event_review["requires_operator_approval"]
        )
        self.assertTrue(autonomous_decoded_output_event_review["advisory"])
        self.assertFalse(autonomous_decoded_output_event_review["executable"])
        self.assertFalse(
            autonomous_decoded_output_event_review["records_ledger_event"]
        )
        self.assertFalse(
            autonomous_decoded_output_event_review["mutates_runtime_state"]
        )
        self.assertFalse(autonomous_decoded_output_event_review["runs_replay"])
        self.assertFalse(
            autonomous_decoded_output_event_review["runs_calibration_update"]
        )
        self.assertFalse(autonomous_decoded_output_event_review["writes_checkpoint"])
        self.assertFalse(autonomous_decoded_output_event_review["generates_text"])
        self.assertFalse(autonomous_decoded_output_event_review["decodes_text"])
        self.assertFalse(
            autonomous_decoded_output_event_review["trains_runtime_model"]
        )
        self.assertFalse(autonomous_decoded_output_event_review["applies_plasticity"])
        self.assertFalse(
            autonomous_decoded_output_event_review["promotion_gate"][
                "eligible_for_autonomous_bounded_text_emission_design"
            ]
        )
        self.assertFalse(
            autonomous_decoded_output_event_review["promotion_gate"][
                "eligible_for_language_generation"
            ]
        )
        self.assertEqual(
            autonomous_bounded_text_emission_design["surface"],
            "snn_language_autonomous_bounded_text_emission_design.v1",
        )
        self.assertFalse(autonomous_bounded_text_emission_design["ready"])
        self.assertFalse(
            autonomous_bounded_text_emission_design["requires_operator_approval"]
        )
        self.assertTrue(autonomous_bounded_text_emission_design["advisory"])
        self.assertFalse(autonomous_bounded_text_emission_design["executable"])
        self.assertFalse(
            autonomous_bounded_text_emission_design["records_ledger_event"]
        )
        self.assertFalse(
            autonomous_bounded_text_emission_design["mutates_runtime_state"]
        )
        self.assertFalse(autonomous_bounded_text_emission_design["runs_replay"])
        self.assertFalse(
            autonomous_bounded_text_emission_design["runs_calibration_update"]
        )
        self.assertFalse(autonomous_bounded_text_emission_design["writes_checkpoint"])
        self.assertFalse(autonomous_bounded_text_emission_design["generates_text"])
        self.assertFalse(autonomous_bounded_text_emission_design["decodes_text"])
        self.assertFalse(
            autonomous_bounded_text_emission_design["trains_runtime_model"]
        )
        self.assertFalse(autonomous_bounded_text_emission_design["applies_plasticity"])
        self.assertFalse(
            autonomous_bounded_text_emission_design["promotion_gate"][
                "eligible_for_autonomous_bounded_text_emission_preflight"
            ]
        )
        self.assertFalse(
            autonomous_bounded_text_emission_design["promotion_gate"][
                "eligible_for_language_generation"
            ]
        )
        self.assertEqual(
            autonomous_bounded_text_emission_preflight["surface"],
            "snn_language_autonomous_bounded_text_emission_preflight.v1",
        )
        self.assertFalse(autonomous_bounded_text_emission_preflight["ready"])
        self.assertFalse(
            autonomous_bounded_text_emission_preflight["requires_operator_approval"]
        )
        self.assertTrue(autonomous_bounded_text_emission_preflight["advisory"])
        self.assertFalse(autonomous_bounded_text_emission_preflight["executable"])
        self.assertFalse(
            autonomous_bounded_text_emission_preflight["records_ledger_event"]
        )
        self.assertFalse(
            autonomous_bounded_text_emission_preflight["mutates_runtime_state"]
        )
        self.assertFalse(autonomous_bounded_text_emission_preflight["runs_replay"])
        self.assertFalse(
            autonomous_bounded_text_emission_preflight["runs_calibration_update"]
        )
        self.assertFalse(
            autonomous_bounded_text_emission_preflight["writes_checkpoint"]
        )
        self.assertFalse(autonomous_bounded_text_emission_preflight["generates_text"])
        self.assertFalse(autonomous_bounded_text_emission_preflight["decodes_text"])
        self.assertFalse(
            autonomous_bounded_text_emission_preflight["trains_runtime_model"]
        )
        self.assertFalse(
            autonomous_bounded_text_emission_preflight["applies_plasticity"]
        )
        self.assertFalse(
            autonomous_bounded_text_emission_preflight["promotion_gate"][
                "eligible_for_autonomous_bounded_text_emission_executor"
            ]
        )
        self.assertFalse(
            autonomous_bounded_text_emission_preflight["promotion_gate"][
                "eligible_for_language_generation"
            ]
        )
        self.assertEqual(
            autonomous_bounded_text_emission_executor["surface"],
            "snn_language_autonomous_bounded_text_emission_executor.v1",
        )
        self.assertFalse(autonomous_bounded_text_emission_executor["accepted"])
        self.assertFalse(autonomous_bounded_text_emission_executor["ready"])
        self.assertFalse(
            autonomous_bounded_text_emission_executor["requires_operator_approval"]
        )
        self.assertFalse(autonomous_bounded_text_emission_executor["advisory"])
        self.assertFalse(autonomous_bounded_text_emission_executor["executable"])
        self.assertFalse(
            autonomous_bounded_text_emission_executor["records_ledger_event"]
        )
        self.assertFalse(
            autonomous_bounded_text_emission_executor["mutates_runtime_state"]
        )
        self.assertFalse(autonomous_bounded_text_emission_executor["runs_replay"])
        self.assertFalse(
            autonomous_bounded_text_emission_executor["runs_calibration_update"]
        )
        self.assertFalse(
            autonomous_bounded_text_emission_executor["writes_checkpoint"]
        )
        self.assertFalse(autonomous_bounded_text_emission_executor["generates_text"])
        self.assertFalse(autonomous_bounded_text_emission_executor["decodes_text"])
        self.assertFalse(
            autonomous_bounded_text_emission_executor["trains_runtime_model"]
        )
        self.assertFalse(
            autonomous_bounded_text_emission_executor["applies_plasticity"]
        )
        self.assertFalse(
            autonomous_bounded_text_emission_executor["promotion_gate"][
                "eligible_for_autonomous_bounded_text_emission_event_review"
            ]
        )
        self.assertFalse(
            autonomous_bounded_text_emission_executor["promotion_gate"][
                "eligible_for_language_generation"
            ]
        )
        self.assertEqual(
            autonomous_bounded_text_emission_event_review["surface"],
            "snn_language_autonomous_bounded_text_emission_event_review.v1",
        )
        self.assertFalse(autonomous_bounded_text_emission_event_review["ready"])
        self.assertFalse(autonomous_bounded_text_emission_event_review["accepted"])
        self.assertFalse(
            autonomous_bounded_text_emission_event_review[
                "requires_operator_approval"
            ]
        )
        self.assertTrue(autonomous_bounded_text_emission_event_review["advisory"])
        self.assertFalse(autonomous_bounded_text_emission_event_review["executable"])
        self.assertFalse(
            autonomous_bounded_text_emission_event_review["records_ledger_event"]
        )
        self.assertFalse(
            autonomous_bounded_text_emission_event_review["mutates_runtime_state"]
        )
        self.assertFalse(autonomous_bounded_text_emission_event_review["runs_replay"])
        self.assertFalse(
            autonomous_bounded_text_emission_event_review[
                "runs_calibration_update"
            ]
        )
        self.assertFalse(
            autonomous_bounded_text_emission_event_review["writes_checkpoint"]
        )
        self.assertFalse(
            autonomous_bounded_text_emission_event_review["generates_text"]
        )
        self.assertFalse(
            autonomous_bounded_text_emission_event_review["decodes_text"]
        )
        self.assertFalse(
            autonomous_bounded_text_emission_event_review["trains_runtime_model"]
        )
        self.assertFalse(
            autonomous_bounded_text_emission_event_review["applies_plasticity"]
        )
        self.assertFalse(
            autonomous_bounded_text_emission_event_review["promotion_gate"][
                "eligible_for_autonomous_text_surface_sequence_review"
            ]
        )
        self.assertFalse(
            autonomous_bounded_text_emission_event_review["promotion_gate"][
                "eligible_for_language_generation"
            ]
        )
        self.assertEqual(
            autonomous_text_surface_sequence_review["surface"],
            "snn_language_autonomous_text_surface_sequence_review.v1",
        )
        self.assertFalse(autonomous_text_surface_sequence_review["ready"])
        self.assertFalse(autonomous_text_surface_sequence_review["accepted"])
        self.assertFalse(
            autonomous_text_surface_sequence_review["requires_operator_approval"]
        )
        self.assertTrue(autonomous_text_surface_sequence_review["advisory"])
        self.assertFalse(autonomous_text_surface_sequence_review["executable"])
        self.assertFalse(
            autonomous_text_surface_sequence_review["records_ledger_event"]
        )
        self.assertFalse(
            autonomous_text_surface_sequence_review["mutates_runtime_state"]
        )
        self.assertFalse(autonomous_text_surface_sequence_review["runs_replay"])
        self.assertFalse(
            autonomous_text_surface_sequence_review["runs_calibration_update"]
        )
        self.assertFalse(autonomous_text_surface_sequence_review["writes_checkpoint"])
        self.assertFalse(autonomous_text_surface_sequence_review["generates_text"])
        self.assertFalse(autonomous_text_surface_sequence_review["decodes_text"])
        self.assertFalse(
            autonomous_text_surface_sequence_review["trains_runtime_model"]
        )
        self.assertFalse(autonomous_text_surface_sequence_review["applies_plasticity"])
        self.assertFalse(
            autonomous_text_surface_sequence_review["promotion_gate"][
                "eligible_for_autonomous_text_surface_commit_design"
            ]
        )
        self.assertFalse(
            autonomous_text_surface_sequence_review["promotion_gate"][
                "eligible_for_language_generation"
            ]
        )
        self.assertEqual(
            autonomous_text_surface_commit_design["surface"],
            "snn_language_autonomous_text_surface_commit_design.v1",
        )
        self.assertFalse(autonomous_text_surface_commit_design["ready"])
        self.assertFalse(autonomous_text_surface_commit_design["accepted"])
        self.assertFalse(
            autonomous_text_surface_commit_design["requires_operator_approval"]
        )
        self.assertTrue(autonomous_text_surface_commit_design["advisory"])
        self.assertFalse(autonomous_text_surface_commit_design["executable"])
        self.assertFalse(
            autonomous_text_surface_commit_design["records_ledger_event"]
        )
        self.assertFalse(
            autonomous_text_surface_commit_design["mutates_runtime_state"]
        )
        self.assertFalse(autonomous_text_surface_commit_design["runs_replay"])
        self.assertFalse(
            autonomous_text_surface_commit_design["runs_calibration_update"]
        )
        self.assertFalse(autonomous_text_surface_commit_design["writes_checkpoint"])
        self.assertFalse(autonomous_text_surface_commit_design["generates_text"])
        self.assertFalse(autonomous_text_surface_commit_design["decodes_text"])
        self.assertFalse(
            autonomous_text_surface_commit_design["trains_runtime_model"]
        )
        self.assertFalse(autonomous_text_surface_commit_design["applies_plasticity"])
        self.assertFalse(
            autonomous_text_surface_commit_design["promotion_gate"][
                "eligible_for_autonomous_text_surface_commit_preflight"
            ]
        )
        self.assertFalse(
            autonomous_text_surface_commit_design["promotion_gate"][
                "eligible_for_language_generation"
            ]
        )
        self.assertEqual(
            autonomous_text_surface_commit_preflight["surface"],
            "snn_language_autonomous_text_surface_commit_preflight.v1",
        )
        self.assertFalse(autonomous_text_surface_commit_preflight["ready"])
        self.assertFalse(autonomous_text_surface_commit_preflight["accepted"])
        self.assertFalse(
            autonomous_text_surface_commit_preflight["requires_operator_approval"]
        )
        self.assertTrue(autonomous_text_surface_commit_preflight["advisory"])
        self.assertFalse(autonomous_text_surface_commit_preflight["executable"])
        self.assertFalse(
            autonomous_text_surface_commit_preflight["records_ledger_event"]
        )
        self.assertFalse(
            autonomous_text_surface_commit_preflight["mutates_runtime_state"]
        )
        self.assertFalse(autonomous_text_surface_commit_preflight["runs_replay"])
        self.assertFalse(
            autonomous_text_surface_commit_preflight["runs_calibration_update"]
        )
        self.assertFalse(autonomous_text_surface_commit_preflight["writes_checkpoint"])
        self.assertFalse(autonomous_text_surface_commit_preflight["generates_text"])
        self.assertFalse(autonomous_text_surface_commit_preflight["decodes_text"])
        self.assertFalse(
            autonomous_text_surface_commit_preflight["trains_runtime_model"]
        )
        self.assertFalse(autonomous_text_surface_commit_preflight["applies_plasticity"])
        self.assertFalse(
            autonomous_text_surface_commit_preflight["promotion_gate"][
                "eligible_for_autonomous_text_surface_commit_executor"
            ]
        )
        self.assertFalse(
            autonomous_text_surface_commit_preflight["promotion_gate"][
                "eligible_for_language_generation"
            ]
        )
        self.assertEqual(
            autonomous_text_surface_commit_executor["surface"],
            "snn_language_autonomous_text_surface_commit_executor.v1",
        )
        self.assertFalse(autonomous_text_surface_commit_executor["accepted"])
        self.assertFalse(autonomous_text_surface_commit_executor["ready"])
        self.assertFalse(
            autonomous_text_surface_commit_executor["requires_operator_approval"]
        )
        self.assertFalse(autonomous_text_surface_commit_executor["advisory"])
        self.assertFalse(autonomous_text_surface_commit_executor["executable"])
        self.assertFalse(
            autonomous_text_surface_commit_executor["records_ledger_event"]
        )
        self.assertFalse(
            autonomous_text_surface_commit_executor["mutates_runtime_state"]
        )
        self.assertFalse(autonomous_text_surface_commit_executor["runs_replay"])
        self.assertFalse(
            autonomous_text_surface_commit_executor["runs_calibration_update"]
        )
        self.assertFalse(autonomous_text_surface_commit_executor["writes_checkpoint"])
        self.assertFalse(autonomous_text_surface_commit_executor["generates_text"])
        self.assertFalse(autonomous_text_surface_commit_executor["decodes_text"])
        self.assertFalse(
            autonomous_text_surface_commit_executor["trains_runtime_model"]
        )
        self.assertFalse(autonomous_text_surface_commit_executor["applies_plasticity"])
        self.assertFalse(
            autonomous_text_surface_commit_executor["promotion_gate"][
                "eligible_for_autonomous_text_surface_commit_event_review"
            ]
        )
        self.assertFalse(
            autonomous_text_surface_commit_executor["promotion_gate"][
                "eligible_for_language_generation"
            ]
        )
        self.assertEqual(
            autonomous_text_surface_commit_event_review["surface"],
            "snn_language_autonomous_text_surface_commit_event_review.v1",
        )
        self.assertFalse(autonomous_text_surface_commit_event_review["accepted"])
        self.assertFalse(autonomous_text_surface_commit_event_review["ready"])
        self.assertFalse(
            autonomous_text_surface_commit_event_review["requires_operator_approval"]
        )
        self.assertTrue(autonomous_text_surface_commit_event_review["advisory"])
        self.assertFalse(autonomous_text_surface_commit_event_review["executable"])
        self.assertFalse(
            autonomous_text_surface_commit_event_review["records_ledger_event"]
        )
        self.assertFalse(
            autonomous_text_surface_commit_event_review["mutates_runtime_state"]
        )
        self.assertFalse(autonomous_text_surface_commit_event_review["runs_replay"])
        self.assertFalse(
            autonomous_text_surface_commit_event_review["runs_calibration_update"]
        )
        self.assertFalse(
            autonomous_text_surface_commit_event_review["writes_checkpoint"]
        )
        self.assertFalse(autonomous_text_surface_commit_event_review["generates_text"])
        self.assertFalse(autonomous_text_surface_commit_event_review["decodes_text"])
        self.assertFalse(
            autonomous_text_surface_commit_event_review["trains_runtime_model"]
        )
        self.assertFalse(
            autonomous_text_surface_commit_event_review["applies_plasticity"]
        )
        self.assertFalse(
            autonomous_text_surface_commit_event_review["promotion_gate"][
                "eligible_for_autonomous_text_surface_materialization_design"
            ]
        )
        self.assertFalse(
            autonomous_text_surface_commit_event_review["promotion_gate"][
                "eligible_for_language_generation"
            ]
        )
        self.assertEqual(
            autonomous_text_surface_materialization_design["surface"],
            "snn_language_autonomous_text_surface_materialization_design.v1",
        )
        self.assertFalse(autonomous_text_surface_materialization_design["accepted"])
        self.assertFalse(autonomous_text_surface_materialization_design["ready"])
        self.assertFalse(
            autonomous_text_surface_materialization_design[
                "requires_operator_approval"
            ]
        )
        self.assertTrue(autonomous_text_surface_materialization_design["advisory"])
        self.assertFalse(autonomous_text_surface_materialization_design["executable"])
        self.assertFalse(
            autonomous_text_surface_materialization_design["records_ledger_event"]
        )
        self.assertFalse(
            autonomous_text_surface_materialization_design["mutates_runtime_state"]
        )
        self.assertFalse(autonomous_text_surface_materialization_design["runs_replay"])
        self.assertFalse(
            autonomous_text_surface_materialization_design[
                "runs_calibration_update"
            ]
        )
        self.assertFalse(
            autonomous_text_surface_materialization_design["writes_checkpoint"]
        )
        self.assertFalse(
            autonomous_text_surface_materialization_design["generates_text"]
        )
        self.assertFalse(autonomous_text_surface_materialization_design["decodes_text"])
        self.assertFalse(
            autonomous_text_surface_materialization_design["trains_runtime_model"]
        )
        self.assertFalse(
            autonomous_text_surface_materialization_design["applies_plasticity"]
        )
        self.assertFalse(
            autonomous_text_surface_materialization_design["promotion_gate"][
                "eligible_for_autonomous_text_surface_materialization_preflight"
            ]
        )
        self.assertFalse(
            autonomous_text_surface_materialization_design["promotion_gate"][
                "eligible_for_language_generation"
            ]
        )
        self.assertEqual(
            autonomous_text_surface_materialization_preflight["surface"],
            "snn_language_autonomous_text_surface_materialization_preflight.v1",
        )
        self.assertFalse(autonomous_text_surface_materialization_preflight["accepted"])
        self.assertFalse(autonomous_text_surface_materialization_preflight["ready"])
        self.assertFalse(
            autonomous_text_surface_materialization_preflight[
                "requires_operator_approval"
            ]
        )
        self.assertTrue(autonomous_text_surface_materialization_preflight["advisory"])
        self.assertFalse(autonomous_text_surface_materialization_preflight["executable"])
        self.assertFalse(
            autonomous_text_surface_materialization_preflight["records_ledger_event"]
        )
        self.assertFalse(
            autonomous_text_surface_materialization_preflight["mutates_runtime_state"]
        )
        self.assertFalse(
            autonomous_text_surface_materialization_preflight["runs_replay"]
        )
        self.assertFalse(
            autonomous_text_surface_materialization_preflight[
                "runs_calibration_update"
            ]
        )
        self.assertFalse(
            autonomous_text_surface_materialization_preflight["writes_checkpoint"]
        )
        self.assertFalse(
            autonomous_text_surface_materialization_preflight["generates_text"]
        )
        self.assertFalse(
            autonomous_text_surface_materialization_preflight["decodes_text"]
        )
        self.assertFalse(
            autonomous_text_surface_materialization_preflight["trains_runtime_model"]
        )
        self.assertFalse(
            autonomous_text_surface_materialization_preflight["applies_plasticity"]
        )
        self.assertFalse(
            autonomous_text_surface_materialization_preflight["promotion_gate"][
                "eligible_for_autonomous_text_surface_materialization_executor"
            ]
        )
        self.assertFalse(
            autonomous_text_surface_materialization_preflight["promotion_gate"][
                "eligible_for_language_generation"
            ]
        )
        self.assertEqual(
            autonomous_text_surface_materialization_executor["surface"],
            "snn_language_autonomous_text_surface_materialization_executor.v1",
        )
        self.assertFalse(autonomous_text_surface_materialization_executor["accepted"])
        self.assertFalse(autonomous_text_surface_materialization_executor["ready"])
        self.assertFalse(
            autonomous_text_surface_materialization_executor[
                "requires_operator_approval"
            ]
        )
        self.assertFalse(autonomous_text_surface_materialization_executor["advisory"])
        self.assertFalse(autonomous_text_surface_materialization_executor["executable"])
        self.assertFalse(
            autonomous_text_surface_materialization_executor["records_ledger_event"]
        )
        self.assertFalse(
            autonomous_text_surface_materialization_executor["mutates_runtime_state"]
        )
        self.assertFalse(
            autonomous_text_surface_materialization_executor["runs_replay"]
        )
        self.assertFalse(
            autonomous_text_surface_materialization_executor[
                "runs_calibration_update"
            ]
        )
        self.assertFalse(
            autonomous_text_surface_materialization_executor["writes_checkpoint"]
        )
        self.assertFalse(
            autonomous_text_surface_materialization_executor["generates_text"]
        )
        self.assertFalse(autonomous_text_surface_materialization_executor["decodes_text"])
        self.assertFalse(
            autonomous_text_surface_materialization_executor["trains_runtime_model"]
        )
        self.assertFalse(
            autonomous_text_surface_materialization_executor["applies_plasticity"]
        )
        self.assertFalse(
            autonomous_text_surface_materialization_executor["literal_text_returned"]
        )
        self.assertIsNone(autonomous_text_surface_materialization_executor["rendered_text"])
        self.assertFalse(
            autonomous_text_surface_materialization_executor["promotion_gate"][
                "eligible_for_autonomous_text_surface_materialization_event_review"
            ]
        )
        self.assertFalse(
            autonomous_text_surface_materialization_executor["promotion_gate"][
                "eligible_for_language_generation"
            ]
        )
        self.assertEqual(
            autonomous_text_surface_materialization_event_review["surface"],
            "snn_language_autonomous_text_surface_materialization_event_review.v1",
        )
        self.assertFalse(
            autonomous_text_surface_materialization_event_review["accepted"]
        )
        self.assertFalse(autonomous_text_surface_materialization_event_review["ready"])
        self.assertFalse(
            autonomous_text_surface_materialization_event_review[
                "requires_operator_approval"
            ]
        )
        self.assertTrue(autonomous_text_surface_materialization_event_review["advisory"])
        self.assertFalse(
            autonomous_text_surface_materialization_event_review["executable"]
        )
        self.assertFalse(
            autonomous_text_surface_materialization_event_review[
                "records_ledger_event"
            ]
        )
        self.assertFalse(
            autonomous_text_surface_materialization_event_review[
                "mutates_runtime_state"
            ]
        )
        self.assertFalse(
            autonomous_text_surface_materialization_event_review["runs_replay"]
        )
        self.assertFalse(
            autonomous_text_surface_materialization_event_review[
                "runs_calibration_update"
            ]
        )
        self.assertFalse(
            autonomous_text_surface_materialization_event_review["writes_checkpoint"]
        )
        self.assertFalse(
            autonomous_text_surface_materialization_event_review["generates_text"]
        )
        self.assertFalse(
            autonomous_text_surface_materialization_event_review["decodes_text"]
        )
        self.assertFalse(
            autonomous_text_surface_materialization_event_review[
                "trains_runtime_model"
            ]
        )
        self.assertFalse(
            autonomous_text_surface_materialization_event_review["applies_plasticity"]
        )
        self.assertFalse(
            autonomous_text_surface_materialization_event_review["promotion_gate"][
                "eligible_for_autonomous_bounded_language_surface_review"
            ]
        )
        self.assertFalse(
            autonomous_text_surface_materialization_event_review["promotion_gate"][
                "eligible_for_language_generation"
            ]
        )
        self.assertEqual(
            autonomous_bounded_language_surface_review["surface"],
            "snn_language_autonomous_bounded_language_surface_review.v1",
        )
        self.assertFalse(autonomous_bounded_language_surface_review["accepted"])
        self.assertFalse(autonomous_bounded_language_surface_review["ready"])
        self.assertFalse(
            autonomous_bounded_language_surface_review["requires_operator_approval"]
        )
        self.assertTrue(autonomous_bounded_language_surface_review["advisory"])
        self.assertFalse(autonomous_bounded_language_surface_review["executable"])
        self.assertFalse(
            autonomous_bounded_language_surface_review["records_ledger_event"]
        )
        self.assertFalse(
            autonomous_bounded_language_surface_review["mutates_runtime_state"]
        )
        self.assertFalse(autonomous_bounded_language_surface_review["runs_replay"])
        self.assertFalse(
            autonomous_bounded_language_surface_review["runs_calibration_update"]
        )
        self.assertFalse(
            autonomous_bounded_language_surface_review["writes_checkpoint"]
        )
        self.assertFalse(autonomous_bounded_language_surface_review["generates_text"])
        self.assertFalse(autonomous_bounded_language_surface_review["decodes_text"])
        self.assertFalse(
            autonomous_bounded_language_surface_review["trains_runtime_model"]
        )
        self.assertFalse(
            autonomous_bounded_language_surface_review["applies_plasticity"]
        )
        self.assertIsNone(autonomous_bounded_language_surface_review["rendered_text"])
        self.assertFalse(
            autonomous_bounded_language_surface_review["promotion_gate"][
                "eligible_for_autonomous_bounded_language_surface_commit_design"
            ]
        )
        self.assertFalse(
            autonomous_bounded_language_surface_review["promotion_gate"][
                "eligible_for_language_generation"
            ]
        )
        self.assertEqual(
            autonomous_bounded_language_surface_commit_design["surface"],
            "snn_language_autonomous_bounded_language_surface_commit_design.v1",
        )
        self.assertFalse(autonomous_bounded_language_surface_commit_design["accepted"])
        self.assertFalse(autonomous_bounded_language_surface_commit_design["ready"])
        self.assertFalse(
            autonomous_bounded_language_surface_commit_design[
                "requires_operator_approval"
            ]
        )
        self.assertTrue(autonomous_bounded_language_surface_commit_design["advisory"])
        self.assertFalse(autonomous_bounded_language_surface_commit_design["executable"])
        self.assertFalse(
            autonomous_bounded_language_surface_commit_design[
                "records_ledger_event"
            ]
        )
        self.assertFalse(
            autonomous_bounded_language_surface_commit_design[
                "mutates_runtime_state"
            ]
        )
        self.assertFalse(
            autonomous_bounded_language_surface_commit_design["runs_replay"]
        )
        self.assertFalse(
            autonomous_bounded_language_surface_commit_design[
                "runs_calibration_update"
            ]
        )
        self.assertFalse(
            autonomous_bounded_language_surface_commit_design["writes_checkpoint"]
        )
        self.assertFalse(
            autonomous_bounded_language_surface_commit_design["generates_text"]
        )
        self.assertFalse(
            autonomous_bounded_language_surface_commit_design["decodes_text"]
        )
        self.assertFalse(
            autonomous_bounded_language_surface_commit_design["trains_runtime_model"]
        )
        self.assertFalse(
            autonomous_bounded_language_surface_commit_design["applies_plasticity"]
        )
        self.assertFalse(
            autonomous_bounded_language_surface_commit_design["promotion_gate"][
                "eligible_for_autonomous_bounded_language_surface_commit_preflight"
            ]
        )
        self.assertFalse(
            autonomous_bounded_language_surface_commit_design["promotion_gate"][
                "eligible_for_language_generation"
            ]
        )
        self.assertEqual(
            autonomous_bounded_language_surface_commit_preflight["surface"],
            "snn_language_autonomous_bounded_language_surface_commit_preflight.v1",
        )
        self.assertFalse(
            autonomous_bounded_language_surface_commit_preflight["accepted"]
        )
        self.assertFalse(autonomous_bounded_language_surface_commit_preflight["ready"])
        self.assertFalse(
            autonomous_bounded_language_surface_commit_preflight[
                "requires_operator_approval"
            ]
        )
        self.assertTrue(autonomous_bounded_language_surface_commit_preflight["advisory"])
        self.assertFalse(
            autonomous_bounded_language_surface_commit_preflight["executable"]
        )
        self.assertFalse(
            autonomous_bounded_language_surface_commit_preflight[
                "records_ledger_event"
            ]
        )
        self.assertFalse(
            autonomous_bounded_language_surface_commit_preflight[
                "mutates_runtime_state"
            ]
        )
        self.assertFalse(
            autonomous_bounded_language_surface_commit_preflight["runs_replay"]
        )
        self.assertFalse(
            autonomous_bounded_language_surface_commit_preflight[
                "runs_calibration_update"
            ]
        )
        self.assertFalse(
            autonomous_bounded_language_surface_commit_preflight["writes_checkpoint"]
        )
        self.assertFalse(
            autonomous_bounded_language_surface_commit_preflight["generates_text"]
        )
        self.assertFalse(
            autonomous_bounded_language_surface_commit_preflight["decodes_text"]
        )
        self.assertFalse(
            autonomous_bounded_language_surface_commit_preflight[
                "trains_runtime_model"
            ]
        )
        self.assertFalse(
            autonomous_bounded_language_surface_commit_preflight["applies_plasticity"]
        )
        self.assertFalse(
            autonomous_bounded_language_surface_commit_preflight["promotion_gate"][
                "eligible_for_autonomous_bounded_language_surface_commit_executor"
            ]
        )
        self.assertFalse(
            autonomous_bounded_language_surface_commit_preflight["promotion_gate"][
                "eligible_for_language_generation"
            ]
        )
        self.assertEqual(
            autonomous_bounded_language_surface_commit_executor["surface"],
            "snn_language_autonomous_bounded_language_surface_commit_executor.v1",
        )
        self.assertFalse(
            autonomous_bounded_language_surface_commit_executor["accepted"]
        )
        self.assertFalse(autonomous_bounded_language_surface_commit_executor["ready"])
        self.assertFalse(
            autonomous_bounded_language_surface_commit_executor[
                "requires_operator_approval"
            ]
        )
        self.assertFalse(autonomous_bounded_language_surface_commit_executor["advisory"])
        self.assertFalse(
            autonomous_bounded_language_surface_commit_executor["executable"]
        )
        self.assertFalse(
            autonomous_bounded_language_surface_commit_executor[
                "records_ledger_event"
            ]
        )
        self.assertFalse(
            autonomous_bounded_language_surface_commit_executor[
                "mutates_runtime_state"
            ]
        )
        self.assertFalse(
            autonomous_bounded_language_surface_commit_executor["runs_replay"]
        )
        self.assertFalse(
            autonomous_bounded_language_surface_commit_executor[
                "runs_calibration_update"
            ]
        )
        self.assertFalse(
            autonomous_bounded_language_surface_commit_executor["writes_checkpoint"]
        )
        self.assertFalse(
            autonomous_bounded_language_surface_commit_executor["generates_text"]
        )
        self.assertFalse(
            autonomous_bounded_language_surface_commit_executor["decodes_text"]
        )
        self.assertFalse(
            autonomous_bounded_language_surface_commit_executor[
                "trains_runtime_model"
            ]
        )
        self.assertFalse(
            autonomous_bounded_language_surface_commit_executor["applies_plasticity"]
        )
        self.assertFalse(
            autonomous_bounded_language_surface_commit_executor["promotion_gate"][
                "eligible_for_autonomous_bounded_language_surface_commit_event_review"
            ]
        )
        self.assertFalse(
            autonomous_bounded_language_surface_commit_executor["promotion_gate"][
                "eligible_for_language_generation"
            ]
        )
        self.assertEqual(
            autonomous_bounded_language_surface_commit_event_review["surface"],
            "snn_language_autonomous_bounded_language_surface_commit_event_review.v1",
        )
        self.assertFalse(
            autonomous_bounded_language_surface_commit_event_review["accepted"]
        )
        self.assertFalse(
            autonomous_bounded_language_surface_commit_event_review["ready"]
        )
        self.assertFalse(
            autonomous_bounded_language_surface_commit_event_review[
                "requires_operator_approval"
            ]
        )
        self.assertTrue(
            autonomous_bounded_language_surface_commit_event_review["advisory"]
        )
        self.assertFalse(
            autonomous_bounded_language_surface_commit_event_review["executable"]
        )
        self.assertFalse(
            autonomous_bounded_language_surface_commit_event_review[
                "records_ledger_event"
            ]
        )
        self.assertFalse(
            autonomous_bounded_language_surface_commit_event_review[
                "mutates_runtime_state"
            ]
        )
        self.assertFalse(
            autonomous_bounded_language_surface_commit_event_review["runs_replay"]
        )
        self.assertFalse(
            autonomous_bounded_language_surface_commit_event_review[
                "runs_calibration_update"
            ]
        )
        self.assertFalse(
            autonomous_bounded_language_surface_commit_event_review[
                "writes_checkpoint"
            ]
        )
        self.assertFalse(
            autonomous_bounded_language_surface_commit_event_review[
                "generates_text"
            ]
        )
        self.assertFalse(
            autonomous_bounded_language_surface_commit_event_review["decodes_text"]
        )
        self.assertFalse(
            autonomous_bounded_language_surface_commit_event_review[
                "trains_runtime_model"
            ]
        )
        self.assertFalse(
            autonomous_bounded_language_surface_commit_event_review[
                "applies_plasticity"
            ]
        )
        self.assertFalse(
            autonomous_bounded_language_surface_commit_event_review["promotion_gate"][
                "eligible_for_autonomous_bounded_language_surface_use_review"
            ]
        )
        self.assertFalse(
            autonomous_bounded_language_surface_commit_event_review["promotion_gate"][
                "eligible_for_language_generation"
            ]
        )
        self.assertEqual(
            autonomous_bounded_language_surface_use_review["surface"],
            "snn_language_autonomous_bounded_language_surface_use_review.v1",
        )
        self.assertFalse(autonomous_bounded_language_surface_use_review["accepted"])
        self.assertFalse(autonomous_bounded_language_surface_use_review["ready"])
        self.assertFalse(
            autonomous_bounded_language_surface_use_review[
                "requires_operator_approval"
            ]
        )
        self.assertTrue(autonomous_bounded_language_surface_use_review["advisory"])
        self.assertFalse(autonomous_bounded_language_surface_use_review["executable"])
        self.assertFalse(
            autonomous_bounded_language_surface_use_review["records_ledger_event"]
        )
        self.assertFalse(
            autonomous_bounded_language_surface_use_review["mutates_runtime_state"]
        )
        self.assertFalse(autonomous_bounded_language_surface_use_review["runs_replay"])
        self.assertFalse(
            autonomous_bounded_language_surface_use_review[
                "runs_calibration_update"
            ]
        )
        self.assertFalse(
            autonomous_bounded_language_surface_use_review["writes_checkpoint"]
        )
        self.assertFalse(
            autonomous_bounded_language_surface_use_review["generates_text"]
        )
        self.assertFalse(
            autonomous_bounded_language_surface_use_review["decodes_text"]
        )
        self.assertFalse(
            autonomous_bounded_language_surface_use_review["trains_runtime_model"]
        )
        self.assertFalse(
            autonomous_bounded_language_surface_use_review["applies_plasticity"]
        )
        self.assertFalse(
            autonomous_bounded_language_surface_use_review["promotion_gate"][
                "eligible_for_autonomous_bounded_language_surface_use_preflight"
            ]
        )
        self.assertFalse(
            autonomous_bounded_language_surface_use_review["promotion_gate"][
                "eligible_for_language_generation"
            ]
        )
        self.assertEqual(
            autonomous_bounded_language_surface_use_preflight["surface"],
            "snn_language_autonomous_bounded_language_surface_use_preflight.v1",
        )
        self.assertFalse(autonomous_bounded_language_surface_use_preflight["accepted"])
        self.assertFalse(autonomous_bounded_language_surface_use_preflight["ready"])
        self.assertFalse(
            autonomous_bounded_language_surface_use_preflight[
                "requires_operator_approval"
            ]
        )
        self.assertTrue(autonomous_bounded_language_surface_use_preflight["advisory"])
        self.assertFalse(
            autonomous_bounded_language_surface_use_preflight["executable"]
        )
        self.assertFalse(
            autonomous_bounded_language_surface_use_preflight["records_ledger_event"]
        )
        self.assertFalse(
            autonomous_bounded_language_surface_use_preflight["mutates_runtime_state"]
        )
        self.assertFalse(autonomous_bounded_language_surface_use_preflight["runs_replay"])
        self.assertFalse(
            autonomous_bounded_language_surface_use_preflight[
                "runs_calibration_update"
            ]
        )
        self.assertFalse(
            autonomous_bounded_language_surface_use_preflight["writes_checkpoint"]
        )
        self.assertFalse(
            autonomous_bounded_language_surface_use_preflight["generates_text"]
        )
        self.assertFalse(
            autonomous_bounded_language_surface_use_preflight["decodes_text"]
        )
        self.assertFalse(
            autonomous_bounded_language_surface_use_preflight["trains_runtime_model"]
        )
        self.assertFalse(
            autonomous_bounded_language_surface_use_preflight["applies_plasticity"]
        )
        self.assertFalse(
            autonomous_bounded_language_surface_use_preflight["promotion_gate"][
                "eligible_for_autonomous_bounded_language_surface_use_executor"
            ]
        )
        self.assertFalse(
            autonomous_bounded_language_surface_use_preflight["promotion_gate"][
                "eligible_for_language_generation"
            ]
        )
        self.assertEqual(
            autonomous_bounded_language_surface_use_executor["surface"],
            "snn_language_autonomous_bounded_language_surface_use_executor.v1",
        )
        self.assertFalse(autonomous_bounded_language_surface_use_executor["accepted"])
        self.assertFalse(autonomous_bounded_language_surface_use_executor["ready"])
        self.assertFalse(
            autonomous_bounded_language_surface_use_executor[
                "requires_operator_approval"
            ]
        )
        self.assertFalse(autonomous_bounded_language_surface_use_executor["advisory"])
        self.assertFalse(autonomous_bounded_language_surface_use_executor["executable"])
        self.assertFalse(
            autonomous_bounded_language_surface_use_executor[
                "records_ledger_event"
            ]
        )
        self.assertFalse(
            autonomous_bounded_language_surface_use_executor[
                "mutates_runtime_state"
            ]
        )
        self.assertFalse(autonomous_bounded_language_surface_use_executor["runs_replay"])
        self.assertFalse(
            autonomous_bounded_language_surface_use_executor[
                "runs_calibration_update"
            ]
        )
        self.assertFalse(
            autonomous_bounded_language_surface_use_executor["writes_checkpoint"]
        )
        self.assertFalse(
            autonomous_bounded_language_surface_use_executor["generates_text"]
        )
        self.assertFalse(
            autonomous_bounded_language_surface_use_executor["decodes_text"]
        )
        self.assertFalse(
            autonomous_bounded_language_surface_use_executor["trains_runtime_model"]
        )
        self.assertFalse(
            autonomous_bounded_language_surface_use_executor["applies_plasticity"]
        )
        self.assertFalse(
            autonomous_bounded_language_surface_use_executor["promotion_gate"][
                "eligible_for_autonomous_bounded_language_surface_use_event_review"
            ]
        )
        self.assertFalse(
            autonomous_bounded_language_surface_use_executor["promotion_gate"][
                "eligible_for_language_generation"
            ]
        )
        self.assertEqual(
            autonomous_bounded_language_surface_use_event_review["surface"],
            "snn_language_autonomous_bounded_language_surface_use_event_review.v1",
        )
        self.assertFalse(
            autonomous_bounded_language_surface_use_event_review["accepted"]
        )
        self.assertFalse(autonomous_bounded_language_surface_use_event_review["ready"])
        self.assertFalse(
            autonomous_bounded_language_surface_use_event_review[
                "requires_operator_approval"
            ]
        )
        self.assertTrue(autonomous_bounded_language_surface_use_event_review["advisory"])
        self.assertFalse(
            autonomous_bounded_language_surface_use_event_review["executable"]
        )
        self.assertFalse(
            autonomous_bounded_language_surface_use_event_review[
                "records_ledger_event"
            ]
        )
        self.assertFalse(
            autonomous_bounded_language_surface_use_event_review[
                "mutates_runtime_state"
            ]
        )
        self.assertFalse(
            autonomous_bounded_language_surface_use_event_review["runs_replay"]
        )
        self.assertFalse(
            autonomous_bounded_language_surface_use_event_review[
                "runs_calibration_update"
            ]
        )
        self.assertFalse(
            autonomous_bounded_language_surface_use_event_review[
                "writes_checkpoint"
            ]
        )
        self.assertFalse(
            autonomous_bounded_language_surface_use_event_review["generates_text"]
        )
        self.assertFalse(
            autonomous_bounded_language_surface_use_event_review["decodes_text"]
        )
        self.assertFalse(
            autonomous_bounded_language_surface_use_event_review[
                "trains_runtime_model"
            ]
        )
        self.assertFalse(
            autonomous_bounded_language_surface_use_event_review["applies_plasticity"]
        )
        self.assertFalse(
            autonomous_bounded_language_surface_use_event_review["promotion_gate"][
                "eligible_for_autonomous_snn_language_generation_design"
            ]
        )
        self.assertFalse(
            autonomous_bounded_language_surface_use_event_review["promotion_gate"][
                "eligible_for_language_generation"
            ]
        )
        self.assertEqual(
            autonomous_snn_language_generation_design["surface"],
            "snn_language_autonomous_snn_language_generation_design.v1",
        )
        self.assertFalse(autonomous_snn_language_generation_design["accepted"])
        self.assertFalse(autonomous_snn_language_generation_design["ready"])
        self.assertFalse(
            autonomous_snn_language_generation_design["requires_operator_approval"]
        )
        self.assertTrue(autonomous_snn_language_generation_design["advisory"])
        self.assertFalse(autonomous_snn_language_generation_design["executable"])
        self.assertFalse(
            autonomous_snn_language_generation_design["records_ledger_event"]
        )
        self.assertFalse(
            autonomous_snn_language_generation_design["mutates_runtime_state"]
        )
        self.assertFalse(autonomous_snn_language_generation_design["runs_replay"])
        self.assertFalse(
            autonomous_snn_language_generation_design["runs_calibration_update"]
        )
        self.assertFalse(
            autonomous_snn_language_generation_design["writes_checkpoint"]
        )
        self.assertFalse(autonomous_snn_language_generation_design["generates_text"])
        self.assertFalse(autonomous_snn_language_generation_design["decodes_text"])
        self.assertFalse(
            autonomous_snn_language_generation_design["trains_runtime_model"]
        )
        self.assertFalse(autonomous_snn_language_generation_design["applies_plasticity"])
        self.assertFalse(autonomous_snn_language_generation_design["planned_generation"])
        self.assertFalse(
            autonomous_snn_language_generation_design["promotion_gate"][
                "eligible_for_autonomous_snn_language_generation_preflight"
            ]
        )
        self.assertFalse(
            autonomous_snn_language_generation_design["promotion_gate"][
                "eligible_for_language_generation"
            ]
        )
        self.assertEqual(
            autonomous_snn_language_generation_preflight["surface"],
            "snn_language_autonomous_snn_language_generation_preflight.v1",
        )
        self.assertFalse(autonomous_snn_language_generation_preflight["accepted"])
        self.assertFalse(autonomous_snn_language_generation_preflight["ready"])
        self.assertFalse(
            autonomous_snn_language_generation_preflight["requires_operator_approval"]
        )
        self.assertTrue(autonomous_snn_language_generation_preflight["advisory"])
        self.assertFalse(autonomous_snn_language_generation_preflight["executable"])
        self.assertFalse(
            autonomous_snn_language_generation_preflight["records_ledger_event"]
        )
        self.assertFalse(
            autonomous_snn_language_generation_preflight["mutates_runtime_state"]
        )
        self.assertFalse(autonomous_snn_language_generation_preflight["runs_replay"])
        self.assertFalse(
            autonomous_snn_language_generation_preflight["runs_calibration_update"]
        )
        self.assertFalse(
            autonomous_snn_language_generation_preflight["writes_checkpoint"]
        )
        self.assertFalse(autonomous_snn_language_generation_preflight["generates_text"])
        self.assertFalse(autonomous_snn_language_generation_preflight["decodes_text"])
        self.assertFalse(
            autonomous_snn_language_generation_preflight["trains_runtime_model"]
        )
        self.assertFalse(
            autonomous_snn_language_generation_preflight["applies_plasticity"]
        )
        self.assertFalse(
            autonomous_snn_language_generation_preflight["promotion_gate"][
                "eligible_for_autonomous_snn_language_generation_executor"
            ]
        )
        self.assertFalse(
            autonomous_snn_language_generation_preflight["promotion_gate"][
                "eligible_for_language_generation"
            ]
        )
        self.assertEqual(
            autonomous_snn_language_generation_executor["surface"],
            "snn_language_autonomous_snn_language_generation_executor.v1",
        )
        self.assertFalse(autonomous_snn_language_generation_executor["accepted"])
        self.assertFalse(autonomous_snn_language_generation_executor["ready"])
        self.assertFalse(
            autonomous_snn_language_generation_executor["requires_operator_approval"]
        )
        self.assertFalse(autonomous_snn_language_generation_executor["advisory"])
        self.assertFalse(autonomous_snn_language_generation_executor["executable"])
        self.assertFalse(
            autonomous_snn_language_generation_executor["records_ledger_event"]
        )
        self.assertFalse(
            autonomous_snn_language_generation_executor["mutates_runtime_state"]
        )
        self.assertFalse(autonomous_snn_language_generation_executor["runs_replay"])
        self.assertFalse(
            autonomous_snn_language_generation_executor["runs_calibration_update"]
        )
        self.assertFalse(
            autonomous_snn_language_generation_executor["writes_checkpoint"]
        )
        self.assertFalse(autonomous_snn_language_generation_executor["generates_text"])
        self.assertFalse(autonomous_snn_language_generation_executor["decodes_text"])
        self.assertFalse(
            autonomous_snn_language_generation_executor["trains_runtime_model"]
        )
        self.assertFalse(
            autonomous_snn_language_generation_executor["applies_plasticity"]
        )
        self.assertFalse(
            autonomous_snn_language_generation_executor["generated_text_returned"]
        )
        self.assertFalse(
            autonomous_snn_language_generation_executor["promotion_gate"][
                "eligible_for_autonomous_snn_language_generation_event_review"
            ]
        )
        self.assertFalse(
            autonomous_snn_language_generation_executor["promotion_gate"][
                "eligible_for_language_generation"
            ]
        )
        self.assertEqual(
            autonomous_snn_language_generation_event_review["surface"],
            "snn_language_autonomous_snn_language_generation_event_review.v1",
        )
        self.assertFalse(autonomous_snn_language_generation_event_review["accepted"])
        self.assertFalse(autonomous_snn_language_generation_event_review["ready"])
        self.assertFalse(
            autonomous_snn_language_generation_event_review[
                "requires_operator_approval"
            ]
        )
        self.assertTrue(autonomous_snn_language_generation_event_review["advisory"])
        self.assertFalse(autonomous_snn_language_generation_event_review["executable"])
        self.assertFalse(
            autonomous_snn_language_generation_event_review["records_ledger_event"]
        )
        self.assertFalse(
            autonomous_snn_language_generation_event_review["mutates_runtime_state"]
        )
        self.assertFalse(autonomous_snn_language_generation_event_review["runs_replay"])
        self.assertFalse(
            autonomous_snn_language_generation_event_review[
                "runs_calibration_update"
            ]
        )
        self.assertFalse(
            autonomous_snn_language_generation_event_review["writes_checkpoint"]
        )
        self.assertFalse(
            autonomous_snn_language_generation_event_review["generates_text"]
        )
        self.assertFalse(
            autonomous_snn_language_generation_event_review["decodes_text"]
        )
        self.assertFalse(
            autonomous_snn_language_generation_event_review["trains_runtime_model"]
        )
        self.assertFalse(
            autonomous_snn_language_generation_event_review["applies_plasticity"]
        )
        self.assertFalse(
            autonomous_snn_language_generation_event_review["generated_text_returned"]
        )
        self.assertFalse(
            autonomous_snn_language_generation_event_review["promotion_gate"][
                "eligible_for_autonomous_snn_language_decoding_design"
            ]
        )
        self.assertFalse(
            autonomous_snn_language_generation_event_review["promotion_gate"][
                "eligible_for_language_generation"
            ]
        )
        self.assertEqual(
            autonomous_snn_language_decoding_design["surface"],
            "snn_language_autonomous_snn_language_decoding_design.v1",
        )
        self.assertFalse(autonomous_snn_language_decoding_design["accepted"])
        self.assertFalse(autonomous_snn_language_decoding_design["ready"])
        self.assertFalse(
            autonomous_snn_language_decoding_design["requires_operator_approval"]
        )
        self.assertTrue(autonomous_snn_language_decoding_design["advisory"])
        self.assertFalse(autonomous_snn_language_decoding_design["executable"])
        self.assertFalse(
            autonomous_snn_language_decoding_design["records_ledger_event"]
        )
        self.assertFalse(
            autonomous_snn_language_decoding_design["mutates_runtime_state"]
        )
        self.assertFalse(autonomous_snn_language_decoding_design["runs_replay"])
        self.assertFalse(
            autonomous_snn_language_decoding_design["runs_calibration_update"]
        )
        self.assertFalse(autonomous_snn_language_decoding_design["writes_checkpoint"])
        self.assertFalse(autonomous_snn_language_decoding_design["generates_text"])
        self.assertFalse(autonomous_snn_language_decoding_design["decodes_text"])
        self.assertFalse(
            autonomous_snn_language_decoding_design["trains_runtime_model"]
        )
        self.assertFalse(
            autonomous_snn_language_decoding_design["applies_plasticity"]
        )
        self.assertFalse(
            autonomous_snn_language_decoding_design["generated_text_returned"]
        )
        self.assertFalse(
            autonomous_snn_language_decoding_design["promotion_gate"][
                "eligible_for_autonomous_snn_language_decoding_preflight"
            ]
        )
        self.assertFalse(
            autonomous_snn_language_decoding_design["promotion_gate"][
                "eligible_for_language_generation"
            ]
        )
        self.assertEqual(
            autonomous_snn_language_decoding_preflight["surface"],
            "snn_language_autonomous_snn_language_decoding_preflight.v1",
        )
        self.assertFalse(autonomous_snn_language_decoding_preflight["accepted"])
        self.assertFalse(autonomous_snn_language_decoding_preflight["ready"])
        self.assertFalse(
            autonomous_snn_language_decoding_preflight[
                "requires_operator_approval"
            ]
        )
        self.assertTrue(autonomous_snn_language_decoding_preflight["advisory"])
        self.assertFalse(autonomous_snn_language_decoding_preflight["executable"])
        self.assertFalse(
            autonomous_snn_language_decoding_preflight["records_ledger_event"]
        )
        self.assertFalse(
            autonomous_snn_language_decoding_preflight["mutates_runtime_state"]
        )
        self.assertFalse(autonomous_snn_language_decoding_preflight["runs_replay"])
        self.assertFalse(
            autonomous_snn_language_decoding_preflight[
                "runs_calibration_update"
            ]
        )
        self.assertFalse(
            autonomous_snn_language_decoding_preflight["writes_checkpoint"]
        )
        self.assertFalse(
            autonomous_snn_language_decoding_preflight["generates_text"]
        )
        self.assertFalse(
            autonomous_snn_language_decoding_preflight["decodes_text"]
        )
        self.assertFalse(
            autonomous_snn_language_decoding_preflight["trains_runtime_model"]
        )
        self.assertFalse(
            autonomous_snn_language_decoding_preflight["applies_plasticity"]
        )
        self.assertFalse(
            autonomous_snn_language_decoding_preflight["generated_text_returned"]
        )
        self.assertFalse(
            autonomous_snn_language_decoding_preflight["promotion_gate"][
                "eligible_for_autonomous_snn_language_decoding_executor"
            ]
        )
        self.assertFalse(
            autonomous_snn_language_decoding_preflight["promotion_gate"][
                "eligible_for_language_generation"
            ]
        )
        self.assertEqual(
            autonomous_snn_language_decoding_executor["surface"],
            "snn_language_autonomous_snn_language_decoding_executor.v1",
        )
        self.assertFalse(autonomous_snn_language_decoding_executor["accepted"])
        self.assertFalse(autonomous_snn_language_decoding_executor["ready"])
        self.assertFalse(
            autonomous_snn_language_decoding_executor[
                "requires_operator_approval"
            ]
        )
        self.assertFalse(autonomous_snn_language_decoding_executor["advisory"])
        self.assertTrue(autonomous_snn_language_decoding_executor["executable"])
        self.assertFalse(
            autonomous_snn_language_decoding_executor["records_ledger_event"]
        )
        self.assertFalse(
            autonomous_snn_language_decoding_executor["mutates_runtime_state"]
        )
        self.assertFalse(autonomous_snn_language_decoding_executor["runs_replay"])
        self.assertFalse(
            autonomous_snn_language_decoding_executor["runs_calibration_update"]
        )
        self.assertFalse(
            autonomous_snn_language_decoding_executor["writes_checkpoint"]
        )
        self.assertFalse(autonomous_snn_language_decoding_executor["generates_text"])
        self.assertFalse(autonomous_snn_language_decoding_executor["decodes_text"])
        self.assertFalse(
            autonomous_snn_language_decoding_executor["trains_runtime_model"]
        )
        self.assertFalse(
            autonomous_snn_language_decoding_executor["applies_plasticity"]
        )
        self.assertFalse(
            autonomous_snn_language_decoding_executor["generated_text_returned"]
        )
        self.assertFalse(
            autonomous_snn_language_decoding_executor["promotion_gate"][
                "eligible_for_autonomous_snn_language_decoding_event_review"
            ]
        )
        self.assertFalse(
            autonomous_snn_language_decoding_executor["promotion_gate"][
                "eligible_for_language_generation"
            ]
        )
        self.assertEqual(
            autonomous_snn_language_decoding_event_review["surface"],
            "snn_language_autonomous_snn_language_decoding_event_review.v1",
        )
        self.assertFalse(autonomous_snn_language_decoding_event_review["accepted"])
        self.assertFalse(autonomous_snn_language_decoding_event_review["ready"])
        self.assertFalse(
            autonomous_snn_language_decoding_event_review[
                "requires_operator_approval"
            ]
        )
        self.assertTrue(autonomous_snn_language_decoding_event_review["advisory"])
        self.assertFalse(autonomous_snn_language_decoding_event_review["executable"])
        self.assertFalse(
            autonomous_snn_language_decoding_event_review["records_ledger_event"]
        )
        self.assertFalse(
            autonomous_snn_language_decoding_event_review["mutates_runtime_state"]
        )
        self.assertFalse(
            autonomous_snn_language_decoding_event_review["runs_replay"]
        )
        self.assertFalse(
            autonomous_snn_language_decoding_event_review[
                "runs_calibration_update"
            ]
        )
        self.assertFalse(
            autonomous_snn_language_decoding_event_review["writes_checkpoint"]
        )
        self.assertFalse(
            autonomous_snn_language_decoding_event_review["generates_text"]
        )
        self.assertFalse(
            autonomous_snn_language_decoding_event_review["decodes_text"]
        )
        self.assertFalse(
            autonomous_snn_language_decoding_event_review["trains_runtime_model"]
        )
        self.assertFalse(
            autonomous_snn_language_decoding_event_review["applies_plasticity"]
        )
        self.assertFalse(
            autonomous_snn_language_decoding_event_review["generated_text_returned"]
        )
        self.assertFalse(
            autonomous_snn_language_decoding_event_review["promotion_gate"].get(
                "eligible_for_snn_language_readout_surface_design",
                autonomous_snn_language_decoding_event_review["promotion_gate"].get(
                    "eligible_for_autonomous_snn_language_thought_surface_design"
                ),
            )
        )
        self.assertFalse(
            autonomous_snn_language_decoding_event_review["promotion_gate"][
                "eligible_for_language_generation"
            ]
        )
        self.assertEqual(
            snn_language_readout_surface_design["surface"],
            "snn_language_readout_surface_design.v1",
        )
        self.assertFalse(snn_language_readout_surface_design["accepted"])
        self.assertFalse(snn_language_readout_surface_design["ready"])
        self.assertFalse(
            snn_language_readout_surface_design[
                "requires_operator_approval"
            ]
        )
        self.assertTrue(snn_language_readout_surface_design["advisory"])
        self.assertFalse(snn_language_readout_surface_design["executable"])
        self.assertFalse(
            snn_language_readout_surface_design["records_ledger_event"]
        )
        self.assertFalse(
            snn_language_readout_surface_design["mutates_runtime_state"]
        )
        self.assertFalse(snn_language_readout_surface_design["runs_replay"])
        self.assertFalse(
            snn_language_readout_surface_design["writes_checkpoint"]
        )
        self.assertFalse(
            snn_language_readout_surface_design["trains_runtime_model"]
        )
        self.assertFalse(
            snn_language_readout_surface_design["applies_plasticity"]
        )
        self.assertFalse(
            snn_language_readout_surface_design["promotion_gate"][
                "eligible_for_snn_language_readout_surface_preflight"
            ]
        )
        self.assertFalse(
            snn_language_readout_surface_design["promotion_gate"][
                "eligible_for_cognition_substrate"
            ]
        )
        self.assertFalse(
            snn_language_readout_surface_design["promotion_gate"][
                "eligible_for_fact_promotion"
            ]
        )
        self.assertFalse(
            snn_language_readout_surface_design["promotion_gate"][
                "eligible_for_action"
            ]
        )
        self.assertEqual(
            snn_language_readout_surface_preflight["surface"],
            "snn_language_readout_surface_preflight.v1",
        )
        self.assertFalse(snn_language_readout_surface_preflight["accepted"])
        self.assertFalse(snn_language_readout_surface_preflight["ready"])
        self.assertFalse(
            snn_language_readout_surface_preflight[
                "requires_operator_approval"
            ]
        )
        self.assertTrue(snn_language_readout_surface_preflight["advisory"])
        self.assertFalse(
            snn_language_readout_surface_preflight["executable"]
        )
        self.assertFalse(
            snn_language_readout_surface_preflight["records_ledger_event"]
        )
        self.assertFalse(
            snn_language_readout_surface_preflight[
                "mutates_runtime_state"
            ]
        )
        self.assertFalse(snn_language_readout_surface_preflight["runs_replay"])
        self.assertFalse(
            snn_language_readout_surface_preflight["writes_checkpoint"]
        )
        self.assertFalse(
            snn_language_readout_surface_preflight["trains_runtime_model"]
        )
        self.assertFalse(
            snn_language_readout_surface_preflight["applies_plasticity"]
        )
        self.assertFalse(
            snn_language_readout_surface_preflight["promotion_gate"][
                "eligible_for_snn_language_readout_surface_executor"
            ]
        )
        self.assertFalse(
            snn_language_readout_surface_preflight["promotion_gate"][
                "eligible_for_cognition_substrate"
            ]
        )
        self.assertFalse(
            snn_language_readout_surface_preflight["promotion_gate"][
                "eligible_for_fact_promotion"
            ]
        )
        self.assertFalse(
            snn_language_readout_surface_preflight["promotion_gate"][
                "eligible_for_action"
            ]
        )
        self.assertEqual(
            snn_language_readout_surface_executor["surface"],
            "snn_language_readout_surface_executor.v1",
        )
        self.assertFalse(snn_language_readout_surface_executor["accepted"])
        self.assertFalse(snn_language_readout_surface_executor["ready"])
        self.assertFalse(
            snn_language_readout_surface_executor[
                "requires_operator_approval"
            ]
        )
        self.assertFalse(snn_language_readout_surface_executor["advisory"])
        self.assertTrue(snn_language_readout_surface_executor["executable"])
        self.assertFalse(
            snn_language_readout_surface_executor["records_ledger_event"]
        )
        self.assertFalse(
            snn_language_readout_surface_executor["mutates_runtime_state"]
        )
        self.assertFalse(snn_language_readout_surface_executor["runs_replay"])
        self.assertFalse(
            snn_language_readout_surface_executor["writes_checkpoint"]
        )
        self.assertFalse(
            snn_language_readout_surface_executor["trains_runtime_model"]
        )
        self.assertFalse(
            snn_language_readout_surface_executor["applies_plasticity"]
        )
        self.assertFalse(
            snn_language_readout_surface_executor["promotion_gate"][
                "eligible_for_snn_language_readout_surface_event_review"
            ]
        )
        self.assertFalse(
            snn_language_readout_surface_executor["promotion_gate"][
                "eligible_for_cognition_substrate"
            ]
        )
        self.assertFalse(
            snn_language_readout_surface_executor["promotion_gate"][
                "eligible_for_fact_promotion"
            ]
        )
        self.assertFalse(
            snn_language_readout_surface_executor["promotion_gate"][
                "eligible_for_action"
            ]
        )
        self.assertEqual(
            snn_language_readout_surface_event_review["surface"],
            "snn_language_readout_surface_event_review.v1",
        )
        self.assertFalse(
            snn_language_readout_surface_event_review["accepted"]
        )
        self.assertFalse(
            snn_language_readout_surface_event_review["ready"]
        )
        self.assertFalse(
            snn_language_readout_surface_event_review[
                "requires_operator_approval"
            ]
        )
        self.assertTrue(snn_language_readout_surface_event_review["advisory"])
        self.assertFalse(
            snn_language_readout_surface_event_review["executable"]
        )
        self.assertFalse(
            snn_language_readout_surface_event_review[
                "records_ledger_event"
            ]
        )
        self.assertFalse(
            snn_language_readout_surface_event_review[
                "mutates_runtime_state"
            ]
        )
        self.assertFalse(
            snn_language_readout_surface_event_review["runs_replay"]
        )
        self.assertFalse(
            snn_language_readout_surface_event_review["writes_checkpoint"]
        )
        self.assertFalse(
            snn_language_readout_surface_event_review[
                "trains_runtime_model"
            ]
        )
        self.assertFalse(
            snn_language_readout_surface_event_review["applies_plasticity"]
        )
        self.assertFalse(
            snn_language_readout_surface_event_review["promotion_gate"][
                "eligible_for_snn_language_readout_memory_design"
            ]
        )
        self.assertFalse(
            snn_language_readout_surface_event_review["promotion_gate"][
                "eligible_for_cognition_substrate"
            ]
        )
        self.assertFalse(
            snn_language_readout_surface_event_review["promotion_gate"][
                "eligible_for_fact_promotion"
            ]
        )
        self.assertFalse(
            snn_language_readout_surface_event_review["promotion_gate"][
                "eligible_for_action"
            ]
        )
        self.assertEqual(
            snn_language_readout_memory_design["surface"],
            "snn_language_readout_memory_design.v1",
        )
        self.assertFalse(snn_language_readout_memory_design["accepted"])
        self.assertFalse(snn_language_readout_memory_design["ready"])
        self.assertFalse(
            snn_language_readout_memory_design[
                "requires_operator_approval"
            ]
        )
        self.assertTrue(snn_language_readout_memory_design["advisory"])
        self.assertFalse(snn_language_readout_memory_design["executable"])
        self.assertFalse(
            snn_language_readout_memory_design["records_ledger_event"]
        )
        self.assertFalse(
            snn_language_readout_memory_design["mutates_runtime_state"]
        )
        self.assertFalse(snn_language_readout_memory_design["runs_replay"])
        self.assertFalse(
            snn_language_readout_memory_design["writes_checkpoint"]
        )
        self.assertFalse(
            snn_language_readout_memory_design["trains_runtime_model"]
        )
        self.assertFalse(
            snn_language_readout_memory_design["applies_plasticity"]
        )
        self.assertFalse(
            snn_language_readout_memory_design["promotion_gate"][
                "eligible_for_snn_language_readout_memory_preflight"
            ]
        )
        self.assertFalse(
            snn_language_readout_memory_design["promotion_gate"][
                "eligible_for_cognition_substrate"
            ]
        )
        self.assertFalse(
            snn_language_readout_memory_design["promotion_gate"][
                "eligible_for_fact_promotion"
            ]
        )
        self.assertFalse(
            snn_language_readout_memory_design["promotion_gate"][
                "eligible_for_action"
            ]
        )
        self.assertEqual(
            snn_language_readout_memory_preflight["surface"],
            "snn_language_readout_memory_preflight.v1",
        )
        self.assertFalse(snn_language_readout_memory_preflight["accepted"])
        self.assertFalse(snn_language_readout_memory_preflight["ready"])
        self.assertFalse(
            snn_language_readout_memory_preflight[
                "requires_operator_approval"
            ]
        )
        self.assertTrue(snn_language_readout_memory_preflight["advisory"])
        self.assertFalse(
            snn_language_readout_memory_preflight["executable"]
        )
        self.assertFalse(
            snn_language_readout_memory_preflight["records_ledger_event"]
        )
        self.assertFalse(
            snn_language_readout_memory_preflight["mutates_runtime_state"]
        )
        self.assertFalse(snn_language_readout_memory_preflight["runs_replay"])
        self.assertFalse(
            snn_language_readout_memory_preflight["writes_checkpoint"]
        )
        self.assertFalse(
            snn_language_readout_memory_preflight["trains_runtime_model"]
        )
        self.assertFalse(
            snn_language_readout_memory_preflight["applies_plasticity"]
        )
        self.assertFalse(
            snn_language_readout_memory_preflight["promotion_gate"][
                "eligible_for_snn_language_readout_memory_executor"
            ]
        )
        self.assertFalse(
            snn_language_readout_memory_preflight["promotion_gate"][
                "eligible_for_cognition_substrate"
            ]
        )
        self.assertFalse(
            snn_language_readout_memory_preflight["promotion_gate"][
                "eligible_for_fact_promotion"
            ]
        )
        self.assertFalse(
            snn_language_readout_memory_preflight["promotion_gate"][
                "eligible_for_action"
            ]
        )
        self.assertEqual(
            snn_language_readout_memory_executor["surface"],
            "snn_language_readout_memory_executor.v1",
        )
        self.assertFalse(snn_language_readout_memory_executor["accepted"])
        self.assertFalse(snn_language_readout_memory_executor["ready"])
        self.assertFalse(
            snn_language_readout_memory_executor[
                "requires_operator_approval"
            ]
        )
        self.assertFalse(snn_language_readout_memory_executor["advisory"])
        self.assertTrue(snn_language_readout_memory_executor["executable"])
        self.assertFalse(
            snn_language_readout_memory_executor["records_ledger_event"]
        )
        self.assertFalse(
            snn_language_readout_memory_executor["mutates_runtime_state"]
        )
        self.assertFalse(snn_language_readout_memory_executor["runs_replay"])
        self.assertFalse(
            snn_language_readout_memory_executor["writes_checkpoint"]
        )
        self.assertFalse(
            snn_language_readout_memory_executor["trains_runtime_model"]
        )
        self.assertFalse(
            snn_language_readout_memory_executor["applies_plasticity"]
        )
        self.assertFalse(
            snn_language_readout_memory_executor["promotion_gate"][
                "eligible_for_snn_language_readout_memory_event_review"
            ]
        )
        self.assertFalse(
            snn_language_readout_memory_executor["promotion_gate"][
                "eligible_for_cognition_substrate"
            ]
        )
        self.assertFalse(
            snn_language_readout_memory_executor["promotion_gate"][
                "eligible_for_fact_promotion"
            ]
        )
        self.assertFalse(
            snn_language_readout_memory_executor["promotion_gate"][
                "eligible_for_action"
            ]
        )
        self.assertEqual(
            snn_language_readout_memory_event_review["surface"],
            "snn_language_readout_memory_event_review.v1",
        )
        self.assertFalse(
            snn_language_readout_memory_event_review["accepted"]
        )
        self.assertFalse(
            snn_language_readout_memory_event_review["ready"]
        )
        self.assertFalse(
            snn_language_readout_memory_event_review[
                "requires_operator_approval"
            ]
        )
        self.assertTrue(snn_language_readout_memory_event_review["advisory"])
        self.assertFalse(
            snn_language_readout_memory_event_review["executable"]
        )
        self.assertFalse(
            snn_language_readout_memory_event_review[
                "records_ledger_event"
            ]
        )
        self.assertFalse(
            snn_language_readout_memory_event_review[
                "mutates_runtime_state"
            ]
        )
        self.assertFalse(
            snn_language_readout_memory_event_review["runs_replay"]
        )
        self.assertFalse(
            snn_language_readout_memory_event_review["writes_checkpoint"]
        )
        self.assertFalse(
            snn_language_readout_memory_event_review[
                "trains_runtime_model"
            ]
        )
        self.assertFalse(
            snn_language_readout_memory_event_review["applies_plasticity"]
        )
        self.assertFalse(
            snn_language_readout_memory_event_review["promotion_gate"][
                "eligible_for_snn_language_readout_consolidation_design"
            ]
        )
        self.assertFalse(
            snn_language_readout_memory_event_review["promotion_gate"][
                "eligible_for_cognition_substrate"
            ]
        )
        self.assertFalse(
            snn_language_readout_memory_event_review["promotion_gate"][
                "eligible_for_fact_promotion"
            ]
        )
        self.assertFalse(
            snn_language_readout_memory_event_review["promotion_gate"][
                "eligible_for_action"
            ]
        )
        self.assertEqual(
            snn_language_readout_consolidation_design["surface"],
            "snn_language_readout_consolidation_design.v1",
        )
        self.assertFalse(
            snn_language_readout_consolidation_design["accepted"]
        )
        self.assertFalse(
            snn_language_readout_consolidation_design["ready"]
        )
        self.assertFalse(
            snn_language_readout_consolidation_design[
                "requires_operator_approval"
            ]
        )
        self.assertTrue(
            snn_language_readout_consolidation_design["advisory"]
        )
        self.assertFalse(
            snn_language_readout_consolidation_design["executable"]
        )
        self.assertFalse(
            snn_language_readout_consolidation_design[
                "records_ledger_event"
            ]
        )
        self.assertFalse(
            snn_language_readout_consolidation_design[
                "mutates_runtime_state"
            ]
        )
        self.assertFalse(
            snn_language_readout_consolidation_design["runs_replay"]
        )
        self.assertFalse(
            snn_language_readout_consolidation_design[
                "writes_checkpoint"
            ]
        )
        self.assertFalse(
            snn_language_readout_consolidation_design[
                "trains_runtime_model"
            ]
        )
        self.assertFalse(
            snn_language_readout_consolidation_design[
                "applies_plasticity"
            ]
        )
        self.assertFalse(
            snn_language_readout_consolidation_design["promotion_gate"][
                "eligible_for_snn_language_readout_consolidation_preflight"
            ]
        )
        self.assertFalse(
            snn_language_readout_consolidation_design["promotion_gate"][
                "eligible_for_cognition_substrate"
            ]
        )
        self.assertFalse(
            snn_language_readout_consolidation_design["promotion_gate"][
                "eligible_for_fact_promotion"
            ]
        )
        self.assertFalse(
            snn_language_readout_consolidation_design["promotion_gate"][
                "eligible_for_action"
            ]
        )
        self.assertEqual(
            snn_language_readout_consolidation_preflight["surface"],
            "snn_language_readout_consolidation_preflight.v1",
        )
        self.assertFalse(
            snn_language_readout_consolidation_preflight["accepted"]
        )
        self.assertFalse(
            snn_language_readout_consolidation_preflight["ready"]
        )
        self.assertFalse(
            snn_language_readout_consolidation_preflight[
                "requires_operator_approval"
            ]
        )
        self.assertTrue(
            snn_language_readout_consolidation_preflight["advisory"]
        )
        self.assertFalse(
            snn_language_readout_consolidation_preflight["executable"]
        )
        self.assertFalse(
            snn_language_readout_consolidation_preflight[
                "records_ledger_event"
            ]
        )
        self.assertFalse(
            snn_language_readout_consolidation_preflight[
                "mutates_runtime_state"
            ]
        )
        self.assertFalse(
            snn_language_readout_consolidation_preflight["runs_replay"]
        )
        self.assertFalse(
            snn_language_readout_consolidation_preflight[
                "writes_checkpoint"
            ]
        )
        self.assertFalse(
            snn_language_readout_consolidation_preflight[
                "trains_runtime_model"
            ]
        )
        self.assertFalse(
            snn_language_readout_consolidation_preflight[
                "applies_plasticity"
            ]
        )
        self.assertFalse(
            snn_language_readout_consolidation_preflight["promotion_gate"][
                "eligible_for_snn_language_readout_consolidation_executor"
            ]
        )
        self.assertFalse(
            snn_language_readout_consolidation_preflight["promotion_gate"][
                "eligible_for_cognition_substrate"
            ]
        )
        self.assertFalse(
            snn_language_readout_consolidation_preflight["promotion_gate"][
                "eligible_for_fact_promotion"
            ]
        )
        self.assertFalse(
            snn_language_readout_consolidation_preflight["promotion_gate"][
                "eligible_for_action"
            ]
        )
        self.assertEqual(
            snn_language_readout_consolidation_executor["surface"],
            "snn_language_readout_consolidation_executor.v1",
        )
        self.assertFalse(
            snn_language_readout_consolidation_executor["accepted"]
        )
        self.assertFalse(
            snn_language_readout_consolidation_executor["ready"]
        )
        self.assertFalse(
            snn_language_readout_consolidation_executor[
                "requires_operator_approval"
            ]
        )
        self.assertFalse(
            snn_language_readout_consolidation_executor["advisory"]
        )
        self.assertTrue(
            snn_language_readout_consolidation_executor["executable"]
        )
        self.assertFalse(
            snn_language_readout_consolidation_executor[
                "records_ledger_event"
            ]
        )
        self.assertFalse(
            snn_language_readout_consolidation_executor[
                "mutates_runtime_state"
            ]
        )
        self.assertFalse(
            snn_language_readout_consolidation_executor["runs_replay"]
        )
        self.assertFalse(
            snn_language_readout_consolidation_executor[
                "writes_checkpoint"
            ]
        )
        self.assertFalse(
            snn_language_readout_consolidation_executor[
                "trains_runtime_model"
            ]
        )
        self.assertFalse(
            snn_language_readout_consolidation_executor[
                "applies_plasticity"
            ]
        )
        self.assertFalse(
            snn_language_readout_consolidation_executor["promotion_gate"][
                "eligible_for_snn_language_readout_consolidation_event_review"
            ]
        )
        self.assertFalse(
            snn_language_readout_consolidation_executor["promotion_gate"][
                "eligible_for_cognition_substrate"
            ]
        )
        self.assertFalse(
            snn_language_readout_consolidation_executor["promotion_gate"][
                "eligible_for_fact_promotion"
            ]
        )
        self.assertFalse(
            snn_language_readout_consolidation_executor["promotion_gate"][
                "eligible_for_action"
            ]
        )
        self.assertEqual(
            snn_language_readout_consolidation_event_review["surface"],
            "snn_language_readout_consolidation_event_review.v1",
        )
        self.assertFalse(
            snn_language_readout_consolidation_event_review["accepted"]
        )
        self.assertFalse(
            snn_language_readout_consolidation_event_review["ready"]
        )
        self.assertFalse(
            snn_language_readout_consolidation_event_review[
                "requires_operator_approval"
            ]
        )
        self.assertTrue(
            snn_language_readout_consolidation_event_review["advisory"]
        )
        self.assertFalse(
            snn_language_readout_consolidation_event_review["executable"]
        )
        self.assertFalse(
            snn_language_readout_consolidation_event_review[
                "records_ledger_event"
            ]
        )
        self.assertFalse(
            snn_language_readout_consolidation_event_review[
                "mutates_runtime_state"
            ]
        )
        self.assertFalse(
            snn_language_readout_consolidation_event_review["runs_replay"]
        )
        self.assertFalse(
            snn_language_readout_consolidation_event_review[
                "writes_checkpoint"
            ]
        )
        self.assertFalse(
            snn_language_readout_consolidation_event_review[
                "trains_runtime_model"
            ]
        )
        self.assertFalse(
            snn_language_readout_consolidation_event_review[
                "applies_plasticity"
            ]
        )
        self.assertFalse(
            snn_language_readout_consolidation_event_review[
                "promotion_gate"
            ][
                "eligible_for_snn_language_readout_structural_plasticity_design"
            ]
        )
        self.assertFalse(
            snn_language_readout_consolidation_event_review[
                "promotion_gate"
            ]["eligible_for_cognition_substrate"]
        )
        self.assertFalse(
            snn_language_readout_consolidation_event_review[
                "promotion_gate"
            ]["eligible_for_fact_promotion"]
        )
        self.assertFalse(
            snn_language_readout_consolidation_event_review[
                "promotion_gate"
            ]["eligible_for_action"]
        )
        self.assertEqual(
            snn_language_readout_structural_plasticity_design["surface"],
            "snn_language_readout_structural_plasticity_design.v1",
        )
        self.assertFalse(
            snn_language_readout_structural_plasticity_design["accepted"]
        )
        self.assertFalse(
            snn_language_readout_structural_plasticity_design["ready"]
        )
        self.assertFalse(
            snn_language_readout_structural_plasticity_design[
                "requires_operator_approval"
            ]
        )
        self.assertTrue(
            snn_language_readout_structural_plasticity_design["advisory"]
        )
        self.assertFalse(
            snn_language_readout_structural_plasticity_design["executable"]
        )
        self.assertFalse(
            snn_language_readout_structural_plasticity_design[
                "records_ledger_event"
            ]
        )
        self.assertFalse(
            snn_language_readout_structural_plasticity_design[
                "mutates_runtime_state"
            ]
        )
        self.assertFalse(
            snn_language_readout_structural_plasticity_design["runs_replay"]
        )
        self.assertFalse(
            snn_language_readout_structural_plasticity_design[
                "writes_checkpoint"
            ]
        )
        self.assertFalse(
            snn_language_readout_structural_plasticity_design[
                "trains_runtime_model"
            ]
        )
        self.assertFalse(
            snn_language_readout_structural_plasticity_design[
                "applies_plasticity"
            ]
        )
        self.assertFalse(
            snn_language_readout_structural_plasticity_design[
                "resizes_network"
            ]
        )
        self.assertFalse(
            snn_language_readout_structural_plasticity_design[
                "prunes_network"
            ]
        )
        self.assertFalse(
            snn_language_readout_structural_plasticity_design[
                "adds_neurons"
            ]
        )
        self.assertFalse(
            snn_language_readout_structural_plasticity_design[
                "adds_synapses"
            ]
        )
        self.assertFalse(
            snn_language_readout_structural_plasticity_design[
                "promotion_gate"
            ][
                "eligible_for_snn_language_readout_structural_plasticity_preflight"
            ]
        )
        self.assertFalse(
            snn_language_readout_structural_plasticity_design[
                "promotion_gate"
            ]["eligible_for_cognition_substrate"]
        )
        self.assertFalse(
            snn_language_readout_structural_plasticity_design[
                "promotion_gate"
            ]["eligible_for_fact_promotion"]
        )
        self.assertFalse(
            snn_language_readout_structural_plasticity_design[
                "promotion_gate"
            ]["eligible_for_action"]
        )
        self.assertEqual(
            snn_language_readout_structural_plasticity_preflight["surface"],
            "snn_language_readout_structural_plasticity_preflight.v1",
        )
        self.assertFalse(
            snn_language_readout_structural_plasticity_preflight["accepted"]
        )
        self.assertFalse(
            snn_language_readout_structural_plasticity_preflight["ready"]
        )
        self.assertFalse(
            snn_language_readout_structural_plasticity_preflight[
                "requires_operator_approval"
            ]
        )
        self.assertTrue(
            snn_language_readout_structural_plasticity_preflight["advisory"]
        )
        self.assertTrue(
            snn_language_readout_structural_plasticity_preflight["executable"]
        )
        self.assertFalse(
            snn_language_readout_structural_plasticity_preflight[
                "records_ledger_event"
            ]
        )
        self.assertFalse(
            snn_language_readout_structural_plasticity_preflight[
                "mutates_runtime_state"
            ]
        )
        self.assertFalse(
            snn_language_readout_structural_plasticity_preflight["runs_replay"]
        )
        self.assertFalse(
            snn_language_readout_structural_plasticity_preflight[
                "writes_checkpoint"
            ]
        )
        self.assertFalse(
            snn_language_readout_structural_plasticity_preflight[
                "trains_runtime_model"
            ]
        )
        self.assertFalse(
            snn_language_readout_structural_plasticity_preflight[
                "applies_plasticity"
            ]
        )
        self.assertFalse(
            snn_language_readout_structural_plasticity_preflight[
                "resizes_network"
            ]
        )
        self.assertFalse(
            snn_language_readout_structural_plasticity_preflight[
                "prunes_network"
            ]
        )
        self.assertFalse(
            snn_language_readout_structural_plasticity_preflight[
                "adds_neurons"
            ]
        )
        self.assertFalse(
            snn_language_readout_structural_plasticity_preflight[
                "adds_synapses"
            ]
        )
        self.assertFalse(
            snn_language_readout_structural_plasticity_preflight[
                "promotion_gate"
            ][
                "eligible_for_snn_language_readout_structural_plasticity_executor"
            ]
        )
        self.assertFalse(
            snn_language_readout_structural_plasticity_preflight[
                "promotion_gate"
            ]["eligible_for_cognition_substrate"]
        )
        self.assertFalse(
            snn_language_readout_structural_plasticity_preflight[
                "promotion_gate"
            ]["eligible_for_fact_promotion"]
        )
        self.assertFalse(
            snn_language_readout_structural_plasticity_preflight[
                "promotion_gate"
            ]["eligible_for_action"]
        )
        self.assertEqual(
            snn_language_readout_structural_plasticity_executor["surface"],
            "snn_language_readout_structural_plasticity_executor.v1",
        )
        self.assertFalse(
            snn_language_readout_structural_plasticity_executor["accepted"]
        )
        self.assertFalse(
            snn_language_readout_structural_plasticity_executor["ready"]
        )
        self.assertFalse(
            snn_language_readout_structural_plasticity_executor[
                "requires_operator_approval"
            ]
        )
        self.assertFalse(
            snn_language_readout_structural_plasticity_executor["advisory"]
        )
        self.assertTrue(
            snn_language_readout_structural_plasticity_executor["executable"]
        )
        self.assertFalse(
            snn_language_readout_structural_plasticity_executor[
                "records_ledger_event"
            ]
        )
        self.assertFalse(
            snn_language_readout_structural_plasticity_executor[
                "mutates_runtime_state"
            ]
        )
        self.assertFalse(
            snn_language_readout_structural_plasticity_executor["runs_replay"]
        )
        self.assertFalse(
            snn_language_readout_structural_plasticity_executor[
                "writes_checkpoint"
            ]
        )
        self.assertFalse(
            snn_language_readout_structural_plasticity_executor[
                "trains_runtime_model"
            ]
        )
        self.assertFalse(
            snn_language_readout_structural_plasticity_executor[
                "applies_plasticity"
            ]
        )
        self.assertFalse(
            snn_language_readout_structural_plasticity_executor[
                "structural_plasticity_applied"
            ]
        )
        self.assertFalse(
            snn_language_readout_structural_plasticity_executor[
                "resizes_network"
            ]
        )
        self.assertFalse(
            snn_language_readout_structural_plasticity_executor[
                "adds_neurons"
            ]
        )
        self.assertFalse(
            snn_language_readout_structural_plasticity_executor[
                "adds_synapses"
            ]
        )
        self.assertFalse(
            snn_language_readout_structural_plasticity_executor[
                "prunes_network"
            ]
        )
        self.assertFalse(
            snn_language_readout_structural_plasticity_executor[
                "promotion_gate"
            ][
                "eligible_for_snn_language_readout_structural_plasticity_event_review"
            ]
        )
        self.assertFalse(
            snn_language_readout_structural_plasticity_executor[
                "promotion_gate"
            ]["eligible_for_cognition_substrate"]
        )
        self.assertFalse(
            snn_language_readout_structural_plasticity_executor[
                "promotion_gate"
            ]["eligible_for_fact_promotion"]
        )
        self.assertFalse(
            snn_language_readout_structural_plasticity_executor[
                "promotion_gate"
            ]["eligible_for_action"]
        )
        self.assertEqual(
            snn_language_readout_structural_plasticity_event_review[
                "surface"
            ],
            "snn_language_readout_structural_plasticity_event_review.v1",
        )
        self.assertFalse(
            snn_language_readout_structural_plasticity_event_review[
                "accepted"
            ]
        )
        self.assertFalse(
            snn_language_readout_structural_plasticity_event_review[
                "ready"
            ]
        )
        self.assertFalse(
            snn_language_readout_structural_plasticity_event_review[
                "requires_operator_approval"
            ]
        )
        self.assertTrue(
            snn_language_readout_structural_plasticity_event_review[
                "advisory"
            ]
        )
        self.assertFalse(
            snn_language_readout_structural_plasticity_event_review[
                "executable"
            ]
        )
        self.assertFalse(
            snn_language_readout_structural_plasticity_event_review[
                "records_ledger_event"
            ]
        )
        self.assertFalse(
            snn_language_readout_structural_plasticity_event_review[
                "mutates_runtime_state"
            ]
        )
        self.assertFalse(
            snn_language_readout_structural_plasticity_event_review[
                "runs_replay"
            ]
        )
        self.assertFalse(
            snn_language_readout_structural_plasticity_event_review[
                "writes_checkpoint"
            ]
        )
        self.assertFalse(
            snn_language_readout_structural_plasticity_event_review[
                "trains_runtime_model"
            ]
        )
        self.assertFalse(
            snn_language_readout_structural_plasticity_event_review[
                "applies_plasticity"
            ]
        )
        self.assertFalse(
            snn_language_readout_structural_plasticity_event_review[
                "structural_plasticity_applied"
            ]
        )
        self.assertFalse(
            snn_language_readout_structural_plasticity_event_review[
                "resizes_network"
            ]
        )
        self.assertFalse(
            snn_language_readout_structural_plasticity_event_review[
                "adds_neurons"
            ]
        )
        self.assertFalse(
            snn_language_readout_structural_plasticity_event_review[
                "adds_synapses"
            ]
        )
        self.assertFalse(
            snn_language_readout_structural_plasticity_event_review[
                "prunes_network"
            ]
        )
        self.assertFalse(
            snn_language_readout_structural_plasticity_event_review[
                "promotion_gate"
            ][
                "eligible_for_snn_language_readout_capacity_mutation_design"
            ]
        )
        self.assertFalse(
            snn_language_readout_structural_plasticity_event_review[
                "promotion_gate"
            ]["eligible_for_cognition_substrate"]
        )
        self.assertFalse(
            snn_language_readout_structural_plasticity_event_review[
                "promotion_gate"
            ]["eligible_for_fact_promotion"]
        )
        self.assertFalse(
            snn_language_readout_structural_plasticity_event_review[
                "promotion_gate"
            ]["eligible_for_action"]
        )
        self.assertEqual(
            snn_language_readout_capacity_mutation_design["surface"],
            "snn_language_readout_capacity_mutation_design.v1",
        )
        self.assertFalse(
            snn_language_readout_capacity_mutation_design["accepted"]
        )
        self.assertFalse(
            snn_language_readout_capacity_mutation_design["ready"]
        )
        self.assertFalse(
            snn_language_readout_capacity_mutation_design[
                "requires_operator_approval"
            ]
        )
        self.assertTrue(snn_language_readout_capacity_mutation_design["advisory"])
        self.assertFalse(
            snn_language_readout_capacity_mutation_design["executable"]
        )
        self.assertFalse(
            snn_language_readout_capacity_mutation_design[
                "records_ledger_event"
            ]
        )
        self.assertFalse(
            snn_language_readout_capacity_mutation_design[
                "mutates_runtime_state"
            ]
        )
        self.assertFalse(
            snn_language_readout_capacity_mutation_design["runs_replay"]
        )
        self.assertFalse(
            snn_language_readout_capacity_mutation_design[
                "writes_checkpoint"
            ]
        )
        self.assertFalse(
            snn_language_readout_capacity_mutation_design[
                "trains_runtime_model"
            ]
        )
        self.assertFalse(
            snn_language_readout_capacity_mutation_design[
                "applies_plasticity"
            ]
        )
        self.assertFalse(
            snn_language_readout_capacity_mutation_design[
                "resizes_network"
            ]
        )
        self.assertFalse(
            snn_language_readout_capacity_mutation_design["adds_neurons"]
        )
        self.assertFalse(
            snn_language_readout_capacity_mutation_design["adds_synapses"]
        )
        self.assertFalse(
            snn_language_readout_capacity_mutation_design["prunes_network"]
        )
        self.assertFalse(
            snn_language_readout_capacity_mutation_design[
                "promotion_gate"
            ][
                "eligible_for_snn_language_readout_capacity_mutation_preflight"
            ]
        )
        self.assertFalse(
            snn_language_readout_capacity_mutation_design[
                "promotion_gate"
            ]["eligible_for_cognition_substrate"]
        )
        self.assertFalse(
            snn_language_readout_capacity_mutation_design[
                "promotion_gate"
            ]["eligible_for_fact_promotion"]
        )
        self.assertFalse(
            snn_language_readout_capacity_mutation_design[
                "promotion_gate"
            ]["eligible_for_action"]
        )
        self.assertEqual(
            snn_language_readout_capacity_mutation_preflight["surface"],
            "snn_language_readout_capacity_mutation_preflight.v1",
        )
        self.assertFalse(
            snn_language_readout_capacity_mutation_preflight["accepted"]
        )
        self.assertFalse(
            snn_language_readout_capacity_mutation_preflight["ready"]
        )
        self.assertFalse(
            snn_language_readout_capacity_mutation_preflight[
                "requires_operator_approval"
            ]
        )
        self.assertTrue(
            snn_language_readout_capacity_mutation_preflight["advisory"]
        )
        self.assertTrue(
            snn_language_readout_capacity_mutation_preflight["executable"]
        )
        self.assertFalse(
            snn_language_readout_capacity_mutation_preflight[
                "records_ledger_event"
            ]
        )
        self.assertFalse(
            snn_language_readout_capacity_mutation_preflight[
                "mutates_runtime_state"
            ]
        )
        self.assertFalse(
            snn_language_readout_capacity_mutation_preflight[
                "runs_replay"
            ]
        )
        self.assertFalse(
            snn_language_readout_capacity_mutation_preflight[
                "writes_checkpoint"
            ]
        )
        self.assertFalse(
            snn_language_readout_capacity_mutation_preflight[
                "trains_runtime_model"
            ]
        )
        self.assertFalse(
            snn_language_readout_capacity_mutation_preflight[
                "applies_plasticity"
            ]
        )
        self.assertFalse(
            snn_language_readout_capacity_mutation_preflight[
                "resizes_network"
            ]
        )
        self.assertFalse(
            snn_language_readout_capacity_mutation_preflight[
                "adds_neurons"
            ]
        )
        self.assertFalse(
            snn_language_readout_capacity_mutation_preflight[
                "adds_synapses"
            ]
        )
        self.assertFalse(
            snn_language_readout_capacity_mutation_preflight[
                "prunes_network"
            ]
        )
        self.assertFalse(
            snn_language_readout_capacity_mutation_preflight[
                "promotion_gate"
            ][
                "eligible_for_snn_language_readout_capacity_mutation_executor"
            ]
        )
        self.assertFalse(
            snn_language_readout_capacity_mutation_preflight[
                "promotion_gate"
            ]["eligible_for_cognition_substrate"]
        )
        self.assertFalse(
            snn_language_readout_capacity_mutation_preflight[
                "promotion_gate"
            ]["eligible_for_fact_promotion"]
        )
        self.assertFalse(
            snn_language_readout_capacity_mutation_preflight[
                "promotion_gate"
            ]["eligible_for_action"]
        )
        self.assertEqual(
            snn_language_readout_capacity_mutation_executor["surface"],
            "snn_language_readout_capacity_mutation_executor.v1",
        )
        self.assertFalse(
            snn_language_readout_capacity_mutation_executor["accepted"]
        )
        self.assertFalse(
            snn_language_readout_capacity_mutation_executor["ready"]
        )
        self.assertFalse(
            snn_language_readout_capacity_mutation_executor[
                "requires_operator_approval"
            ]
        )
        self.assertTrue(
            snn_language_readout_capacity_mutation_executor["executable"]
        )
        self.assertFalse(
            snn_language_readout_capacity_mutation_executor[
                "records_ledger_event"
            ]
        )
        self.assertFalse(
            snn_language_readout_capacity_mutation_executor[
                "mutates_runtime_state"
            ]
        )
        self.assertFalse(
            snn_language_readout_capacity_mutation_executor[
                "writes_checkpoint"
            ]
        )
        self.assertFalse(
            snn_language_readout_capacity_mutation_executor[
                "resizes_network"
            ]
        )
        self.assertFalse(
            snn_language_readout_capacity_mutation_executor[
                "adds_neurons"
            ]
        )
        self.assertFalse(
            snn_language_readout_capacity_mutation_executor[
                "adds_synapses"
            ]
        )
        self.assertFalse(
            snn_language_readout_capacity_mutation_executor[
                "prunes_network"
            ]
        )
        self.assertFalse(
            snn_language_readout_capacity_mutation_executor[
                "promotion_gate"
            ][
                "eligible_for_snn_language_readout_capacity_mutation_event_review"
            ]
        )
        self.assertEqual(
            snn_language_readout_capacity_mutation_event_review[
                "surface"
            ],
            "snn_language_readout_capacity_mutation_event_review.v1",
        )
        self.assertFalse(
            snn_language_readout_capacity_mutation_event_review[
                "accepted"
            ]
        )
        self.assertFalse(
            snn_language_readout_capacity_mutation_event_review[
                "ready"
            ]
        )
        self.assertFalse(
            snn_language_readout_capacity_mutation_event_review[
                "requires_operator_approval"
            ]
        )
        self.assertTrue(
            snn_language_readout_capacity_mutation_event_review[
                "advisory"
            ]
        )
        self.assertFalse(
            snn_language_readout_capacity_mutation_event_review[
                "executable"
            ]
        )
        self.assertFalse(
            snn_language_readout_capacity_mutation_event_review[
                "records_ledger_event"
            ]
        )
        self.assertFalse(
            snn_language_readout_capacity_mutation_event_review[
                "mutates_runtime_state"
            ]
        )
        self.assertFalse(
            snn_language_readout_capacity_mutation_event_review[
                "writes_checkpoint"
            ]
        )
        self.assertFalse(
            snn_language_readout_capacity_mutation_event_review[
                "resizes_network"
            ]
        )
        self.assertFalse(
            snn_language_readout_capacity_mutation_event_review[
                "promotion_gate"
            ][
                "eligible_for_snn_language_readout_newborn_neuron_integration_design"
            ]
        )
        self.assertEqual(
            snn_language_readout_newborn_neuron_integration_design[
                "surface"
            ],
            "snn_language_readout_newborn_neuron_integration_design.v1",
        )
        self.assertFalse(
            snn_language_readout_newborn_neuron_integration_design[
                "accepted"
            ]
        )
        self.assertFalse(
            snn_language_readout_newborn_neuron_integration_design[
                "requires_operator_approval"
            ]
        )
        self.assertFalse(
            snn_language_readout_newborn_neuron_integration_design[
                "mutates_runtime_state"
            ]
        )
        self.assertFalse(
            snn_language_readout_newborn_neuron_integration_design[
                "promotion_gate"
            ][
                "eligible_for_snn_language_readout_newborn_neuron_integration_preflight"
            ]
        )
        self.assertEqual(
            snn_language_readout_newborn_neuron_integration_preflight[
                "surface"
            ],
            "snn_language_readout_newborn_neuron_integration_preflight.v1",
        )
        self.assertFalse(
            snn_language_readout_newborn_neuron_integration_preflight[
                "accepted"
            ]
        )
        self.assertFalse(
            snn_language_readout_newborn_neuron_integration_preflight[
                "requires_operator_approval"
            ]
        )
        self.assertFalse(
            snn_language_readout_newborn_neuron_integration_preflight[
                "mutates_runtime_state"
            ]
        )
        self.assertFalse(
            snn_language_readout_newborn_neuron_integration_preflight[
                "promotion_gate"
            ][
                "eligible_for_snn_language_readout_newborn_neuron_integration_executor"
            ]
        )
        self.assertEqual(
            snn_language_readout_newborn_neuron_integration_executor[
                "surface"
            ],
            "snn_language_readout_newborn_neuron_integration_executor.v1",
        )
        self.assertFalse(
            snn_language_readout_newborn_neuron_integration_executor[
                "accepted"
            ]
        )
        self.assertEqual(
            snn_language_readout_newborn_neuron_integration_event_review[
                "surface"
            ],
            "snn_language_readout_newborn_neuron_integration_event_review.v1",
        )
        self.assertFalse(
            snn_language_readout_newborn_neuron_integration_event_review[
                "accepted"
            ]
        )
        self.assertFalse(
            snn_language_readout_newborn_neuron_integration_event_review[
                "mutates_runtime_state"
            ]
        )
        self.assertEqual(
            snn_language_readout_newborn_neuron_critical_period_learning_design[
                "surface"
            ],
            "snn_language_readout_newborn_neuron_critical_period_learning_design.v1",
        )
        self.assertFalse(
            snn_language_readout_newborn_neuron_critical_period_learning_design[
                "accepted"
            ]
        )
        self.assertEqual(
            snn_language_readout_newborn_neuron_critical_period_learning_preflight[
                "surface"
            ],
            "snn_language_readout_newborn_neuron_critical_period_learning_preflight.v1",
        )
        self.assertFalse(
            snn_language_readout_newborn_neuron_critical_period_learning_preflight[
                "accepted"
            ]
        )
        self.assertEqual(
            snn_language_readout_newborn_neuron_critical_period_learning_executor[
                "surface"
            ],
            "snn_language_readout_newborn_neuron_critical_period_learning_executor.v1",
        )
        self.assertFalse(
            snn_language_readout_newborn_neuron_critical_period_learning_executor[
                "accepted"
            ]
        )
        self.assertFalse(
            snn_language_readout_newborn_neuron_critical_period_learning_executor[
                "mutates_runtime_state"
            ]
        )
        self.assertEqual(
            snn_language_readout_newborn_neuron_critical_period_learning_event_review[
                "surface"
            ],
            "snn_language_readout_newborn_neuron_critical_period_learning_event_review.v1",
        )
        self.assertFalse(
            snn_language_readout_newborn_neuron_critical_period_learning_event_review[
                "accepted"
            ]
        )
        self.assertFalse(
            snn_language_readout_newborn_neuron_critical_period_learning_event_review[
                "mutates_runtime_state"
            ]
        )
        self.assertEqual(
            snn_language_readout_newborn_neuron_critical_period_learning_continuation_design[
                "surface"
            ],
            "snn_language_readout_newborn_neuron_critical_period_learning_design.v1",
        )
        self.assertFalse(
            snn_language_readout_newborn_neuron_critical_period_learning_continuation_design[
                "accepted"
            ]
        )
        self.assertFalse(
            snn_language_readout_newborn_neuron_integration_executor[
                "requires_operator_approval"
            ]
        )
        self.assertFalse(
            snn_language_readout_newborn_neuron_integration_executor[
                "mutates_runtime_state"
            ]
        )
        self.assertFalse(
            snn_language_readout_newborn_neuron_integration_executor[
                "adds_synapses"
            ]
        )
        self.assertFalse(
            snn_language_readout_newborn_neuron_integration_executor[
                "promotion_gate"
            ][
                "eligible_for_snn_language_readout_newborn_neuron_integration_event_review"
            ]
        )
        self.assertFalse(capacity_compatibility["adds_neurons"])
        self.assertFalse(
            capacity_compatibility["promotion_gate"][
                "eligible_for_capacity_resize_executor"
            ]
        )
        status_gate = status_truth["evidence"]["self_repair_gate"]
        terminus_gate = terminus_truth["evidence"]["self_repair_gate"]
        self.assertEqual(status_gate["artifact_kind"], "terminus_subcortical_self_repair_gate_plan")
        self.assertEqual(status_gate["surface"], "subcortical_self_repair_candidates.v1")
        self.assertTrue(status_gate["advisory"])
        self.assertFalse(status_gate["executable"])
        self.assertFalse(status_gate["eligible_for_action"])
        self.assertFalse(status_gate["eligible_for_fact_promotion"])
        self.assertFalse(status_gate["eligible_for_structural_mutation"])
        self.assertNotIn("candidates", status_gate)
        self.assertNotIn("endpoint", status_gate)

        self.assertNotIn("suggested_endpoint", status_gate)
        self.assertEqual(terminus_gate["artifact_kind"], status_gate["artifact_kind"])
        self.assertEqual(terminus_gate["surface"], status_gate["surface"])
        status_evaluation_gate = status_truth["evidence"]["self_repair_evaluation_gate"]
        terminus_evaluation_gate = terminus_truth["evidence"]["self_repair_evaluation_gate"]
        self.assertEqual(
            status_evaluation_gate["artifact_kind"],
            "terminus_subcortical_self_repair_evaluation_plan",
        )
        self.assertEqual(status_evaluation_gate["surface"], "subcortical_self_repair_evaluation.v1")
        self.assertTrue(status_evaluation_gate["advisory"])
        self.assertFalse(status_evaluation_gate["executable"])
        self.assertFalse(status_evaluation_gate["mutates_runtime_state"])
        self.assertFalse(status_evaluation_gate["eligible_for_action"])
        self.assertFalse(status_evaluation_gate["eligible_for_fact_promotion"])
        self.assertFalse(status_evaluation_gate["eligible_for_structural_mutation"])
        self.assertTrue(status_evaluation_gate["requires_isolated_replay_or_deep_sleep"])
        self.assertTrue(status_evaluation_gate["requires_runtime_truth_improvement"])
        self.assertTrue(status_evaluation_gate["requires_device_evidence"])
        self.assertIn("runtime_truth_delta", status_evaluation_gate["success_evidence"])
        self.assertNotIn("repair_surface", status_evaluation_gate)
        self.assertNotIn("evaluation_cases", status_evaluation_gate)
        self.assertNotIn("endpoint", status_evaluation_gate)
        self.assertEqual(terminus_evaluation_gate["artifact_kind"], status_evaluation_gate["artifact_kind"])
        self.assertEqual(terminus_evaluation_gate["surface"], status_evaluation_gate["surface"])
        status_structural_gate = status_truth["evidence"]["structural_plasticity_gate"]
        terminus_structural_gate = terminus_truth["evidence"]["structural_plasticity_gate"]
        self.assertEqual(
            status_structural_gate["artifact_kind"],
            "terminus_subcortical_structural_plasticity_gate_plan",
        )
        self.assertEqual(status_structural_gate["surface"], "subcortical_structural_plasticity.v1")
        self.assertTrue(status_structural_gate["advisory"])
        self.assertFalse(status_structural_gate["executable"])
        self.assertFalse(status_structural_gate["mutates_runtime_state"])
        self.assertFalse(status_structural_gate["eligible_for_action"])
        self.assertFalse(status_structural_gate["eligible_for_fact_promotion"])
        self.assertFalse(status_structural_gate["eligible_for_structural_mutation"])
        self.assertIn("eligible_for_replay_review", status_structural_gate)
        self.assertIn("requires_operator_approval", status_structural_gate)
        self.assertTrue(status_structural_gate["requires_isolated_evaluation"])
        self.assertTrue(status_structural_gate["requires_runtime_truth_improvement"])
        self.assertTrue(status_structural_gate["requires_reversible_mutation_ledger"])
        self.assertTrue(status_structural_gate["requires_device_evidence"])
        self.assertIn("rollback_policy", status_structural_gate["success_evidence"])
        self.assertIn("runtime_truth_delta", status_structural_gate["success_evidence"])
        self.assertIn("local_plasticity_report_available", status_structural_gate)
        self.assertIn("local_plasticity_homeostatic_state_available", status_structural_gate)
        self.assertIn("local_plasticity_spike_backend", status_structural_gate)
        self.assertIn("local_plasticity_rule", status_structural_gate)
        self.assertIn("local_plasticity_spike_health_risk", status_structural_gate)
        self.assertIn("local_plasticity_synaptic_validation_available", status_structural_gate)
        self.assertIn("local_plasticity_synaptic_validation_passed", status_structural_gate)
        self.assertIn("local_plasticity_synaptic_validation_failed", status_structural_gate)
        self.assertNotIn("structural_cases", status_structural_gate)
        self.assertNotIn("endpoint", status_structural_gate)
        self.assertNotIn("device_evidence", status_structural_gate)
        self.assertNotIn("local_plasticity", status_structural_gate)
        self.assertNotIn("recent_events", status_structural_gate)
        self.assertNotIn("active_growth_concepts", status_structural_gate)
        self.assertEqual(terminus_structural_gate["artifact_kind"], status_structural_gate["artifact_kind"])
        self.assertEqual(terminus_structural_gate["surface"], status_structural_gate["surface"])
        self.assertEqual(
            terminus_structural_gate["requires_runtime_truth_improvement"],
            status_structural_gate["requires_runtime_truth_improvement"],
        )
        status_language_gate = status_truth["evidence"]["snn_language_readiness_gate"]
        terminus_language_gate = terminus_truth["evidence"]["snn_language_readiness_gate"]
        self.assertEqual(status_language_gate["artifact_kind"], "terminus_snn_native_language_readiness_gate")
        self.assertEqual(status_language_gate["surface"], "snn_native_language_readiness.v1")
        self.assertTrue(status_language_gate["advisory"])
        self.assertFalse(status_language_gate["executable"])
        self.assertFalse(status_language_gate["mutates_runtime_state"])
        self.assertTrue(status_language_gate["not_cognition_substrate"])
        self.assertNotIn("retired_runtime_dependency", status_language_gate)
        self.assertFalse(status_language_gate["eligible_for_action"])
        self.assertFalse(status_language_gate["eligible_for_fact_promotion"])
        self.assertFalse(status_language_gate["eligible_for_cognition_substrate"])
        self.assertTrue(status_language_gate["requires_marulho_owned_implementation"])
        self.assertTrue(status_language_gate["marulho_spike_readout_evidence_available"])
        self.assertTrue(status_language_gate["marulho_spike_readout_grounded"])
        self.assertTrue(status_language_gate["marulho_spike_readout_non_generative"])
        self.assertIn("marulho_spike_decoder_probe_available", status_language_gate)
        self.assertIn("marulho_spike_decoder_probe_owned", status_language_gate)
        self.assertIn("marulho_spike_decoder_probe_non_generative", status_language_gate)
        self.assertIn("marulho_spike_decoder_probe_sparse", status_language_gate)
        self.assertIn("marulho_spike_decoder_probe_device_evidence_available", status_language_gate)
        self.assertIn("marulho_spike_decoder_probe_grounding_supported", status_language_gate)
        self.assertIn("marulho_spike_language_neuron_adapter_available", status_language_gate)
        self.assertIn("marulho_spike_language_neuron_adapter_owned", status_language_gate)
        self.assertIn("marulho_spike_language_neuron_adapter_sparse", status_language_gate)
        self.assertIn("marulho_spike_language_neuron_adapter_dynamic", status_language_gate)
        self.assertEqual(
            terminus_language_gate["marulho_spike_readout_evidence_available"],
            status_language_gate["marulho_spike_readout_evidence_available"],
        )
        self.assertEqual(
            terminus_language_gate["marulho_spike_readout_non_generative"],
            status_language_gate["marulho_spike_readout_non_generative"],
        )
        self.assertNotIn("research_candidates", status_language_gate)
        self.assertNotIn("endpoint", status_language_gate)
        self.assertNotIn("readiness_checks", status_language_gate)
        self.assertNotIn("current_decoder_probe_evidence", status_language_gate)
        self.assertEqual(terminus_language_gate["artifact_kind"], status_language_gate["artifact_kind"])
        self.assertEqual(terminus_language_gate["surface"], status_language_gate["surface"])
        status_plasticity_path = status_truth["evidence"]["snn_language_plasticity_path"]
        terminus_plasticity_path = terminus_truth["evidence"]["snn_language_plasticity_path"]
        self.assertEqual(
            status_plasticity_path["artifact_kind"],
            "terminus_snn_language_plasticity_path_evidence",
        )
        self.assertEqual(status_plasticity_path["surface"], "snn_language_plasticity_path_evidence.v1")
        self.assertEqual(
            status_plasticity_path["latest_gate"],
            "snn_language_plasticity_live_application_preflight.v1",
        )
        self.assertFalse(status_plasticity_path["generates_text"])
        self.assertFalse(status_plasticity_path["decodes_text"])
        self.assertFalse(status_plasticity_path["trains_runtime_model"])
        self.assertFalse(status_plasticity_path["applies_plasticity"])
        self.assertFalse(status_plasticity_path["mutates_runtime_state"])
        self.assertTrue(status_plasticity_path["requires_device_evidence"])
        self.assertTrue(status_plasticity_path["requires_runtime_truth_delta"])
        self.assertTrue(status_plasticity_path["requires_rollback_evidence"])
        self.assertTrue(status_plasticity_path["rollback_readiness"]["rollback_policy_required"])
        self.assertTrue(status_plasticity_path["rollback_readiness"]["restore_endpoint_available"])
        self.assertTrue(status_plasticity_path["rollback_readiness"]["checkpoint_metadata_available"])
        self.assertIn("checkpoint_path", status_plasticity_path["rollback_readiness"])
        self.assertIn("snn_language_plasticity_shadow_application.v1", status_plasticity_path["gates"])
        self.assertIn("snn_language_plasticity_live_application_readiness.v1", status_plasticity_path["gates"])
        self.assertIn("snn_language_plasticity_live_application_preflight.v1", status_plasticity_path["gates"])
        self.assertEqual(terminus_plasticity_path["artifact_kind"], status_plasticity_path["artifact_kind"])
        self.assertEqual(terminus_plasticity_path["surface"], status_plasticity_path["surface"])
        self.assertEqual(
            terminus_plasticity_path["rollback_readiness"]["checkpoint_path"],
            status_plasticity_path["rollback_readiness"]["checkpoint_path"],
        )
        status_rollout_binding = status_truth["evidence"]["snn_readout_rollout_server_state_binding"]
        terminus_rollout_binding = terminus_truth["evidence"]["snn_readout_rollout_server_state_binding"]
        self.assertEqual(
            status_rollout_binding["artifact_kind"],
            "terminus_snn_readout_rollout_server_state_binding_gate",
        )
        self.assertEqual(
            status_rollout_binding["surface"],
            "snn_readout_rollout_server_state_binding.v1",
        )
        self.assertTrue(status_rollout_binding["owned_by_marulho"])
        self.assertFalse(status_rollout_binding["external_dependency"])
        self.assertTrue(status_rollout_binding["advisory"])
        self.assertFalse(status_rollout_binding["executable"])
        self.assertFalse(status_rollout_binding["generates_text"])
        self.assertFalse(status_rollout_binding["decodes_text"])
        self.assertFalse(status_rollout_binding["freeform_language_generation"])
        self.assertFalse(status_rollout_binding["loads_external_checkpoint"])
        self.assertFalse(status_rollout_binding["accepts_caller_transition_memory_state"])
        self.assertTrue(status_rollout_binding["requires_server_transition_memory_state"])
        self.assertTrue(status_rollout_binding["runtime_mutation_absent"])
        self.assertTrue(status_rollout_binding["plasticity_absent"])
        self.assertTrue(status_rollout_binding["checkpoint_write_absent"])
        self.assertTrue(status_rollout_binding["rollout_execution_absent"])
        self.assertFalse(status_rollout_binding["runs_replay"])
        self.assertFalse(status_rollout_binding["records_ledger_event"])
        self.assertFalse(status_rollout_binding["calls_rollout"])
        self.assertFalse(status_rollout_binding["eligible_for_rollout_execution"])
        self.assertFalse(status_rollout_binding["eligible_for_fact_promotion"])
        self.assertFalse(status_rollout_binding["eligible_for_cognition_substrate"])
        self.assertEqual(
            status_rollout_binding["transition_memory_state_source"],
            "service.runtime_facade.snn_language_plasticity_runtime_state",
        )
        self.assertEqual(
            terminus_rollout_binding["artifact_kind"],
            status_rollout_binding["artifact_kind"],
        )
        self.assertEqual(terminus_rollout_binding["surface"], status_rollout_binding["surface"])
        self.assertEqual(
            terminus_rollout_binding["server_transition_memory_hash"],
            status_rollout_binding["server_transition_memory_hash"],
        )
        self.assertEqual(
            terminus_rollout_binding["server_transition_weight_count"],
            status_rollout_binding["server_transition_weight_count"],
        )
        for forbidden_key in (
            "rollout",
            "labels",
            "text",
            "prediction_report",
            "transition_memory_evaluation",
            "suggested_endpoint",
            "candidate",
            "transition_memory_state",
        ):
            self.assertNotIn(forbidden_key, status_rollout_binding)
        status_consolidation_path = status_truth["evidence"]["snn_readout_rollout_consolidation_path"]
        terminus_consolidation_path = terminus_truth["evidence"][
            "snn_readout_rollout_consolidation_path"
        ]
        self.assertEqual(
            status_consolidation_path["artifact_kind"],
            "terminus_snn_readout_rollout_consolidation_path_evidence",
        )
        self.assertEqual(
            status_consolidation_path["surface"],
            "snn_readout_rollout_consolidation_path_evidence.v1",
        )
        self.assertTrue(status_consolidation_path["owned_by_marulho"])
        self.assertFalse(status_consolidation_path["external_dependency"])
        self.assertTrue(status_consolidation_path["advisory"])
        self.assertFalse(status_consolidation_path["executable"])
        self.assertFalse(status_consolidation_path["executes_rehearsal"])
        self.assertFalse(status_consolidation_path["executes_consolidation"])
        self.assertFalse(status_consolidation_path["runs_live_replay"])
        self.assertFalse(status_consolidation_path["records_ledger_event"])
        self.assertFalse(status_consolidation_path["writes_checkpoint"])
        self.assertFalse(status_consolidation_path["generates_text"])
        self.assertFalse(status_consolidation_path["decodes_text"])
        self.assertFalse(status_consolidation_path["freeform_language_generation"])
        self.assertFalse(status_consolidation_path["applies_plasticity"])
        self.assertFalse(status_consolidation_path["mutates_runtime_state"])
        self.assertFalse(status_consolidation_path["eligible_for_live_replay"])
        self.assertFalse(status_consolidation_path["eligible_for_plasticity_application"])
        self.assertFalse(status_consolidation_path["eligible_for_cognition_substrate"])
        self.assertEqual(
            terminus_consolidation_path["artifact_kind"],
            status_consolidation_path["artifact_kind"],
        )
        self.assertEqual(
            terminus_consolidation_path["surface"],
            status_consolidation_path["surface"],
        )
        self.assertEqual(
            terminus_consolidation_path["rollout_event_count"],
            status_consolidation_path["rollout_event_count"],
        )
        for forbidden_key in (
            "rollout",
            "labels",
            "text",
            "prediction_report",
            "transition_memory_evaluation",
            "candidate",
            "replay_targets",
        ):
            self.assertNotIn(forbidden_key, status_consolidation_path)
        status_emission_review_history = status_truth["evidence"][
            "snn_readout_emission_review_history"
        ]
        terminus_emission_review_history = terminus_truth["evidence"][
            "snn_readout_emission_review_history"
        ]
        self.assertEqual(
            status_emission_review_history["artifact_kind"],
            "terminus_snn_readout_emission_review_history_evidence",
        )
        self.assertEqual(
            status_emission_review_history["surface"],
            "snn_readout_emission_review_history_evidence.v1",
        )
        self.assertTrue(status_emission_review_history["owned_by_marulho"])
        self.assertFalse(status_emission_review_history["external_dependency"])
        self.assertTrue(status_emission_review_history["advisory"])
        self.assertFalse(status_emission_review_history["executable"])
        self.assertFalse(status_emission_review_history["calls_endpoint"])
        self.assertFalse(status_emission_review_history["records_ledger_event"])
        self.assertFalse(status_emission_review_history["runs_replay"])
        self.assertFalse(status_emission_review_history["writes_checkpoint"])
        self.assertFalse(status_emission_review_history["generates_text"])
        self.assertFalse(status_emission_review_history["decodes_text"])
        self.assertFalse(status_emission_review_history["exposes_raw_text"])
        self.assertFalse(status_emission_review_history["freeform_language_generation"])
        self.assertFalse(status_emission_review_history["applies_plasticity"])
        self.assertFalse(status_emission_review_history["mutates_runtime_state"])
        self.assertFalse(status_emission_review_history["eligible_for_replay_memory"])
        self.assertFalse(status_emission_review_history["eligible_for_live_replay"])
        self.assertFalse(
            status_emission_review_history["eligible_for_plasticity_application"]
        )
        self.assertFalse(status_emission_review_history["eligible_for_fact_promotion"])
        self.assertFalse(status_emission_review_history["eligible_for_action"])
        self.assertEqual(
            terminus_emission_review_history["artifact_kind"],
            status_emission_review_history["artifact_kind"],
        )
        self.assertEqual(
            terminus_emission_review_history["surface"],
            status_emission_review_history["surface"],
        )
        self.assertEqual(
            terminus_emission_review_history["emission_review_event_count"],
            status_emission_review_history["emission_review_event_count"],
        )
        for forbidden_key in (
            "rollout",
            "labels",
            "text",
            "prediction_report",
            "transition_memory_evaluation",
            "candidate",
            "language_output",
            "emission_review_events",
        ):
            self.assertNotIn(forbidden_key, status_emission_review_history)
        status_emission_replay_design_path = status_truth["evidence"][
            "snn_readout_emission_replay_design_path"
        ]
        terminus_emission_replay_design_path = terminus_truth["evidence"][
            "snn_readout_emission_replay_design_path"
        ]
        self.assertEqual(
            status_emission_replay_design_path["artifact_kind"],
            "terminus_snn_readout_emission_replay_design_path_evidence",
        )
        self.assertEqual(
            status_emission_replay_design_path["surface"],
            "snn_readout_emission_replay_design_path_evidence.v1",
        )
        self.assertTrue(status_emission_replay_design_path["owned_by_marulho"])
        self.assertFalse(status_emission_replay_design_path["external_dependency"])
        self.assertTrue(status_emission_replay_design_path["advisory"])
        self.assertFalse(status_emission_replay_design_path["executable"])
        self.assertFalse(status_emission_replay_design_path["calls_endpoint"])
        self.assertFalse(status_emission_replay_design_path["records_ledger_event"])
        self.assertFalse(status_emission_replay_design_path["records_replay_context"])
        self.assertFalse(status_emission_replay_design_path["runs_replay"])
        self.assertFalse(status_emission_replay_design_path["writes_checkpoint"])
        self.assertFalse(status_emission_replay_design_path["generates_text"])
        self.assertFalse(status_emission_replay_design_path["decodes_text"])
        self.assertFalse(status_emission_replay_design_path["exposes_raw_text"])
        self.assertFalse(status_emission_replay_design_path["applies_plasticity"])
        self.assertFalse(status_emission_replay_design_path["mutates_runtime_state"])
        self.assertFalse(
            status_emission_replay_design_path[
                "eligible_for_emission_replay_evaluation_design_review"
            ]
        )
        self.assertFalse(
            status_emission_replay_design_path[
                "eligible_for_operator_replay_context_review"
            ]
        )
        self.assertFalse(
            status_emission_replay_design_path["eligible_for_replay_context_recording"]
        )
        self.assertTrue(
            status_emission_replay_design_path["requires_device_review_evidence"]
        )
        self.assertFalse(status_emission_replay_design_path["eligible_for_replay_memory"])
        self.assertFalse(status_emission_replay_design_path["eligible_for_live_replay"])
        self.assertFalse(
            status_emission_replay_design_path["eligible_for_plasticity_application"]
        )
        self.assertFalse(status_emission_replay_design_path["eligible_for_fact_promotion"])
        self.assertFalse(status_emission_replay_design_path["eligible_for_action"])
        self.assertEqual(
            terminus_emission_replay_design_path["artifact_kind"],
            status_emission_replay_design_path["artifact_kind"],
        )
        self.assertEqual(
            terminus_emission_replay_design_path["surface"],
            status_emission_replay_design_path["surface"],
        )
        self.assertEqual(
            terminus_emission_replay_design_path["design_seed_candidate_count"],
            status_emission_replay_design_path["design_seed_candidate_count"],
        )
        for forbidden_key in (
            "rollout",
            "labels",
            "text",
            "prediction_report",
            "transition_memory_evaluation",
            "candidate",
            "language_output",
            "emission_review_events",
            "selected_replay_context_seeds",
            "events",
        ):
            self.assertNotIn(forbidden_key, status_emission_replay_design_path)
        status_applied_provenance = status_truth["evidence"][
            "snn_readout_applied_synapse_provenance"
        ]
        terminus_applied_provenance = terminus_truth["evidence"][
            "snn_readout_applied_synapse_provenance"
        ]
        self.assertEqual(
            status_applied_provenance["artifact_kind"],
            "terminus_snn_readout_applied_synapse_provenance_evidence",
        )
        self.assertEqual(
            status_applied_provenance["surface"],
            "snn_readout_applied_synapse_provenance_evidence.v1",
        )
        self.assertTrue(status_applied_provenance["owned_by_marulho"])
        self.assertFalse(status_applied_provenance["external_dependency"])
        self.assertTrue(status_applied_provenance["advisory"])
        self.assertFalse(status_applied_provenance["executable"])
        self.assertFalse(status_applied_provenance["runs_audit"])
        self.assertFalse(status_applied_provenance["runs_replay"])
        self.assertFalse(status_applied_provenance["calls_endpoint"])
        self.assertFalse(status_applied_provenance["generates_text"])
        self.assertFalse(status_applied_provenance["decodes_text"])
        self.assertFalse(status_applied_provenance["freeform_language_generation"])
        self.assertFalse(status_applied_provenance["applies_plasticity"])
        self.assertFalse(status_applied_provenance["mutates_runtime_state"])
        self.assertFalse(status_applied_provenance["writes_checkpoint"])
        self.assertFalse(status_applied_provenance["restore_validation_available"])
        self.assertFalse(status_applied_provenance["restore_validation_blocks_audit"])
        status_capacity_pressure = status_truth["evidence"][
            "snn_language_capacity_pressure"
        ]
        terminus_capacity_pressure = terminus_truth["evidence"][
            "snn_language_capacity_pressure"
        ]
        self.assertEqual(
            status_capacity_pressure["artifact_kind"],
            "terminus_snn_language_capacity_pressure_evidence",
        )
        self.assertEqual(
            status_capacity_pressure["surface"],
            "snn_language_capacity_pressure_evidence.v1",
        )
        self.assertFalse(status_capacity_pressure["mutates_runtime_state"])
        self.assertFalse(status_capacity_pressure["resizes_network"])
        self.assertFalse(status_capacity_pressure["adds_neurons"])
        self.assertFalse(status_capacity_pressure["adds_layers"])
        self.assertEqual(
            terminus_capacity_pressure["current_language_neuron_count"],
            status_capacity_pressure["current_language_neuron_count"],
        )
        status_capacity_boundaries = status_truth["evidence"][
            "snn_language_capacity_fixed_boundaries"
        ]
        terminus_capacity_boundaries = terminus_truth["evidence"][
            "snn_language_capacity_fixed_boundaries"
        ]
        self.assertEqual(
            status_capacity_boundaries["surface"],
            "snn_language_capacity_fixed_boundary_evidence.v1",
        )
        self.assertFalse(status_capacity_boundaries["mutates_runtime_state"])
        self.assertFalse(status_capacity_boundaries["resizes_network"])
        self.assertTrue(
            status_capacity_boundaries[
                "capacity_resize_blocked_by_fixed_boundaries"
            ]
        )
        self.assertEqual(
            terminus_capacity_boundaries["fixed_boundary_count"],
            status_capacity_boundaries["fixed_boundary_count"],
        )
        self.assertFalse(
            status_capacity_boundaries[
                "eligible_for_capacity_resize_compatibility_audit"
            ]
        )
        self.assertEqual(
            terminus_applied_provenance["artifact_kind"],
            status_applied_provenance["artifact_kind"],
        )
        self.assertEqual(
            terminus_applied_provenance["surface"],
            status_applied_provenance["surface"],
        )
        self.assertEqual(
            terminus_applied_provenance["synapse_provenance_count"],
            status_applied_provenance["synapse_provenance_count"],
        )
        self.assertEqual(
            terminus_applied_provenance["restore_validation_blocks_audit"],
            status_applied_provenance["restore_validation_blocks_audit"],
        )
        for forbidden_key in (
            "rollout",
            "labels",
            "text",
            "prediction_report",
            "transition_memory_evaluation",
            "candidate",
            "synapse_provenance_by_key",
            "sparse_transition_weights",
        ):
            self.assertNotIn(forbidden_key, status_applied_provenance)

    def test_snn_language_readout_draft_endpoint_generates_bounded_grounded_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            app = create_app(_build_checkpoint(root, test_case="service_api_snn_readout_draft"), trace_dir=root / "traces")
            vocabulary = [
                {"label": "memory pressure", "pressure_band": "medium", "grounded": True},
                {"label": "prediction error", "pressure_band": "high", "grounded": True},
            ]
            current = [{"label": "concept focus", "pressure_band": "medium", "grounded": True}]
            current_probe = build_spike_language_decoder_probe(
                {
                    "readout_slots": current,
                    "device_evidence": {"device": "cpu", "source": "service_api_readout_draft"},
                }
            )
            target_probe = build_spike_language_decoder_probe(
                {
                    "readout_slots": [vocabulary[0]],
                    "device_evidence": {"device": "cpu", "source": "service_api_readout_draft"},
                }
            )
            current_index = current_probe["sparse_code_evidence"]["active_indices"][0]
            target_index = target_probe["sparse_code_evidence"]["active_indices"][0]
            training_batches = [
                [{"label": "prediction error", "pressure_band": "high", "grounded": True}],
                current,
            ]
            transition_weights = {f"{current_index}:{target_index}": 0.9}
            prediction_report = predict_spike_language_sequence(
                training_batches,
                current,
                {"device": "cpu", "source": "service_api_readout_draft"},
                top_k=4,
                persistent_transition_weights=transition_weights,
            )
            transition_memory_evaluation = build_snn_language_transition_memory_prediction_evaluation(
                training_batches,
                [current, [vocabulary[0]]],
                {"sparse_transition_weights": transition_weights},
                {"device": "cpu", "source": "service_api_readout_draft"},
                top_k=4,
            )
            app.state.marulho_manager._snn_language_plasticity_state[
                "sparse_transition_weights"
            ] = dict(transition_weights)
            with TestClient(app) as client:
                status_response = client.get("/status")
                response = client.post(
                    "/terminus/snn-language-sequence/readout-draft",
                    json={
                        "prediction_report": prediction_report,
                        "readout_vocabulary_slots": vocabulary,
                        "device_evidence": {"device": "cpu", "source": "service_api_readout_draft"},
                        "transition_memory_evaluation": transition_memory_evaluation,
                        "max_draft_terms": 4,
                    },
                )
                emission_response = client.post(
                    "/terminus/snn-language-sequence/readout-emission",
                    json={"readout_draft": response.json()},
                )
                emission_review_response = client.post(
                    "/terminus/snn-language-sequence/readout-emission/operator-review",
                    json={
                        "readout_emission": emission_response.json(),
                        "expected_state_revision": status_response.json()["state_revision"],
                        "operator_id": "operator-test",
                        "confirmation": True,
                    },
                )
                emission_review_history_response = client.get(
                    "/terminus/snn-language-sequence/readout-emission/operator-review/history",
                    params={"limit": 4},
                )
                blocked_emission_review_response = client.post(
                    "/terminus/snn-language-sequence/readout-emission/operator-review",
                    json={
                        "readout_emission": emission_response.json(),
                        "expected_state_revision": status_response.json()["state_revision"],
                        "operator_id": "operator-test",
                        "confirmation": False,
                    },
                )
                rollout_response = client.post(
                    "/terminus/snn-language-sequence/readout-rollout-candidate",
                    json={
                        "prediction_report": prediction_report,
                        "readout_vocabulary_slots": vocabulary,
                        "transition_memory_state": {
                            "sparse_transition_weights": {"999:998": 9.9},
                            "source": "caller_fabricated_transition_memory",
                        },
                        "device_evidence": {"device": "cpu", "source": "service_api_readout_draft"},
                        "transition_memory_evaluation": transition_memory_evaluation,
                        "rollout_steps": 3,
                        "top_k": 4,
                    },
                )
                app.state.marulho_manager._snn_language_plasticity_state[
                    "sparse_transition_weights"
                ] = {}
                rollout_replay_evaluation_response = client.post(
                    "/terminus/snn-language-sequence/readout-rollout-candidate/replay-evaluation",
                    json={
                        "readout_rollout_candidate": rollout_response.json(),
                        "candidate_limit": 4,
                        "device_evidence": {
                            "device": "cpu",
                            "source": "service_api_readout_rollout_replay",
                        },
                    },
                )
                rollout_record_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/record-rollout-replay-evaluation",
                    json={
                        "readout_rollout_replay_evaluation": rollout_replay_evaluation_response.json(),
                        "expected_state_revision": status_response.json()["state_revision"],
                        "operator_id": "operator-test",
                        "confirmation": True,
                    },
                )
                blocked_rollout_record_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/record-rollout-replay-evaluation",
                    json={
                        "readout_rollout_replay_evaluation": rollout_replay_evaluation_response.json(),
                        "expected_state_revision": status_response.json()["state_revision"],
                        "operator_id": "operator-test",
                        "confirmation": False,
                    },
                )
                rollout_rehearsal_policy_response = client.get(
                    "/terminus/snn-language-sequence/readout-ledger/rollout-rehearsal-promotion-policy",
                    params={"limit": 4},
                )
                rollout_rehearsal_evaluation_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/rollout-rehearsal-evaluation",
                    json={
                        "rollout_rehearsal_promotion_policy": rollout_rehearsal_policy_response.json(),
                        "candidate_limit": 4,
                    },
                )
                rollout_rehearsal_experiment_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/rollout-rehearsal-experiment",
                    json={
                        "rollout_rehearsal_evaluation": rollout_rehearsal_evaluation_response.json(),
                        "replay_cycles": 4,
                        "stability_floor": 0.95,
                    },
                )
                rollout_consolidation_design_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/rollout-consolidation-design",
                    json={
                        "rollout_rehearsal_experiment": rollout_rehearsal_experiment_response.json(),
                        "consolidation_policy": {
                            "learning_rate": 0.02,
                            "max_weight_delta": 0.04,
                            "homeostatic_decay": 0.01,
                            "local_only": True,
                            "normalization": True,
                        },
                        "rollback_policy": {
                            "available": True,
                            "snapshot_id": "service-api-rollout-snapshot",
                        },
                    },
                )
                rollout_consolidation_shadow_delta_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/rollout-consolidation-shadow-delta",
                    json={
                        "rollout_consolidation_design": rollout_consolidation_design_response.json(),
                        "device_evidence": {
                            "device": "cpu",
                            "source": "service-api-rollout-shadow-delta",
                        },
                    },
                )
                rollout_consolidation_shadow_application_preflight_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/rollout-consolidation-shadow-application-preflight",
                    json={
                        "rollout_consolidation_design": rollout_consolidation_design_response.json(),
                        "rollout_consolidation_shadow_delta": rollout_consolidation_shadow_delta_response.json(),
                    },
                )
                rollout_developmental_plasticity_review_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/rollout-developmental-plasticity-review",
                    json={
                        "rollout_consolidation_design": rollout_consolidation_design_response.json(),
                        "rollout_consolidation_shadow_application_preflight": (
                            rollout_consolidation_shadow_application_preflight_response.json()
                        ),
                    },
                )
                rollout_regeneration_proposal_adapter_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/rollout-regeneration-proposal-adapter",
                    json={
                        "rollout_developmental_plasticity_review": (
                            rollout_developmental_plasticity_review_response.json()
                        ),
                    },
                )
                rollout_regeneration_replay_artifact_review_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/rollout-regeneration-replay-artifact-review",
                    json={
                        "rollout_regeneration_proposal_adapter": (
                            rollout_regeneration_proposal_adapter_response.json()
                        ),
                        "snn_transition_memory_replay_artifact": {
                            "artifact_kind": "terminus_snn_transition_memory_replay_artifact",
                            "surface": "snn_transition_memory_replay_artifact.v1",
                            "available": True,
                            "ready": False,
                            "owned_by_marulho": True,
                        },
                    },
                )
                rollout_regeneration_permit_request_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/rollout-regeneration-permit-request",
                    json={
                        "rollout_regeneration_replay_artifact_review": (
                            rollout_regeneration_replay_artifact_review_response.json()
                        ),
                        "operator_id": "operator-test",
                        "confirmation": True,
                    },
                )
                rollout_regeneration_application_preflight_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/rollout-regeneration-application-preflight",
                    json={
                        "rollout_regeneration_permit_request": (
                            rollout_regeneration_permit_request_response.json()
                        ),
                        "expected_state_revision": status_response.json()["state_revision"],
                        "checkpoint_path": str(root / "rollout_regeneration_preflight.pt"),
                    },
                )
                rollout_regeneration_application_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/rollout-regeneration-application",
                    json={
                        "rollout_regeneration_application_preflight": (
                            rollout_regeneration_application_preflight_response.json()
                        ),
                        "expected_state_revision": status_response.json()["state_revision"],
                        "operator_id": "operator-test",
                        "confirmation": True,
                        "checkpoint_path": str(root / "rollout_regeneration_preflight.pt"),
                    },
                )
                pending_evaluation_response = client.post(
                    "/terminus/snn-language-sequence/readout-draft",
                    json={
                        "prediction_report": prediction_report,
                        "readout_vocabulary_slots": vocabulary,
                        "device_evidence": {"device": "cpu", "source": "service_api_readout_draft"},
                        "max_draft_terms": 4,
                    },
                )
                pending_emission_response = client.post(
                    "/terminus/snn-language-sequence/readout-emission",
                    json={"readout_draft": pending_evaluation_response.json()},
                )
                blocked_record_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/record",
                    json={
                        "readout_draft": pending_evaluation_response.json(),
                        "expected_state_revision": status_response.json()["state_revision"],
                        "operator_id": "operator-test",
                        "confirmation": True,
                    },
                )
                record_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/record",
                    json={
                        "readout_draft": response.json(),
                        "expected_state_revision": status_response.json()["state_revision"],
                        "operator_id": "operator-test",
                        "confirmation": True,
                    },
                )
                emission_replay_policy_response = client.get(
                    "/terminus/snn-language-sequence/readout-emission/operator-review/replay-evaluation-policy",
                    params={"limit": 4},
                )
                emission_replay_design_response = client.post(
                    "/terminus/snn-language-sequence/readout-emission/operator-review/replay-evaluation-design",
                    json={
                        "emission_replay_evaluation_policy": emission_replay_policy_response.json(),
                        "design_policy": {"max_candidates": 1, "min_ready_candidates": 1},
                        "device_evidence": {
                            "device": "cpu",
                            "source": "service_api_emission_replay_design",
                        },
                    },
                )
                blocked_emission_replay_context_response = client.post(
                    "/terminus/snn-language-sequence/readout-emission/operator-review/replay-context-review",
                    json={
                        "emission_replay_evaluation_design": emission_replay_design_response.json(),
                        "prediction_report": prediction_report,
                        "observed_readout_slots": [
                            {
                                "label": "novel mismatch alpha",
                                "pressure_band": "high",
                                "grounded": True,
                            }
                        ],
                        "device_evidence": {
                            "device": "cpu",
                            "source": "service_api_emission_replay_context",
                        },
                        "runtime_truth_delta": {"improved_or_stable": True},
                        "rollback_policy": {
                            "available": True,
                            "snapshot_id": "service-api-emission-replay-context",
                        },
                        "operator_id": "operator-test",
                        "confirmation": False,
                    },
                )
                emission_replay_context_response = client.post(
                    "/terminus/snn-language-sequence/readout-emission/operator-review/replay-context-review",
                    json={
                        "emission_replay_evaluation_design": emission_replay_design_response.json(),
                        "prediction_report": prediction_report,
                        "observed_readout_slots": [
                            {
                                "label": "novel mismatch alpha",
                                "pressure_band": "high",
                                "grounded": True,
                            },
                            {
                                "label": "novel mismatch beta",
                                "pressure_band": "high",
                                "grounded": True,
                            },
                            {
                                "label": "novel mismatch gamma",
                                "pressure_band": "high",
                                "grounded": True,
                            },
                        ],
                        "device_evidence": {
                            "device": "cpu",
                            "source": "service_api_emission_replay_context",
                        },
                        "runtime_truth_delta": {"improved_or_stable": True},
                        "rollback_policy": {
                            "available": True,
                            "snapshot_id": "service-api-emission-replay-context",
                        },
                        "operator_id": "operator-test",
                        "confirmation": True,
                    },
                )
                emission_replay_consolidation_priority_response = client.get(
                    "/terminus/snn-language-sequence/replay-consolidation-priority-queue",
                    params={"limit": 4},
                )
                ledger_response = client.get("/terminus/snn-language-sequence/readout-ledger")
                replay_priority_response = client.get(
                    "/terminus/snn-language-sequence/readout-ledger/replay-priority"
                )
                rehearsal_evaluation_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/rehearsal-evaluation",
                    json={
                        "replay_priority_report": replay_priority_response.json(),
                        "candidate_limit": 4,
                        "device_evidence": {"device": "cpu", "source": "service_api_readout_rehearsal"},
                    },
                )
                rehearsal_experiment_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/rehearsal-experiment",
                    json={
                        "rehearsal_evaluation": rehearsal_evaluation_response.json(),
                        "replay_cycles": 4,
                        "stability_floor": 0.85,
                    },
                )
                replay_design_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/replay-design",
                    json={
                        "rehearsal_experiment": rehearsal_experiment_response.json(),
                        "replay_policy": {
                            "max_candidates": 1,
                            "max_replay_cycles": 3,
                            "min_pressure_gain": 0.01,
                        },
                        "rollback_policy": {"available": True, "snapshot_id": "service-api-snapshot"},
                    },
                )
                replay_dry_run_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/replay-dry-run",
                    json={
                        "replay_design": replay_design_response.json(),
                        "operator_approval": True,
                        "operator_id": "operator-test",
                        "device_evidence": {"device": "cpu", "source": "service_api_readout_replay"},
                    },
                )
                readout_plasticity_preflight_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/plasticity-preflight",
                    json={
                        "readout_replay_dry_run": replay_dry_run_response.json(),
                        "plasticity_policy": {
                            "learning_rate": 0.02,
                            "max_weight_delta": 0.03,
                            "locality_radius": 8,
                            "normalization": True,
                            "local_only": True,
                        },
                        "runtime_truth_delta": {"improved_or_stable": True},
                        "rollback_policy": {"available": True, "snapshot_id": "service-api-snapshot"},
                    },
                )
                readout_plasticity_replay_bridge_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/plasticity-replay-bridge",
                    json={
                        "readout_plasticity_preflight": readout_plasticity_preflight_response.json(),
                        "runtime_truth_delta": {"improved_or_stable": True},
                        "rollback_policy": {"available": True, "snapshot_id": "service-api-snapshot"},
                    },
                )
                readout_application_design_response = client.post(
                    "/terminus/snn-language-sequence/plasticity-application-design",
                    json={
                        "replay_experiment": readout_plasticity_replay_bridge_response.json(),
                        "application_policy": {
                            "learning_rate": 0.02,
                            "max_weight_delta": 0.03,
                            "locality_radius": 8,
                            "normalization": True,
                            "local_only": True,
                        },
                        "device_evidence": {"device": "cpu", "source": "service_api_readout_bridge"},
                        "runtime_truth_delta": {"improved_or_stable": True},
                        "rollback_policy": {"available": True, "snapshot_id": "service-api-snapshot"},
                    },
                )
                readout_shadow_delta_response = client.post(
                    "/terminus/snn-language-sequence/plasticity-shadow-delta",
                    json={
                        "application_design": readout_plasticity_replay_bridge_response.json(),
                        "replay_sequences": readout_plasticity_replay_bridge_response.json()[
                            "canonical_replay_sequences"
                        ],
                        "device_evidence": {"device": "cpu", "source": "service_api_readout_shadow_delta"},
                    },
                )
                readout_shadow_application_response = client.post(
                    "/terminus/snn-language-sequence/plasticity-shadow-application",
                    json={
                        "application_design": readout_plasticity_replay_bridge_response.json(),
                        "shadow_delta": readout_shadow_delta_response.json(),
                        "device_evidence": {"device": "cpu", "source": "service_api_readout_shadow"},
                        "runtime_truth_delta": {"improved_or_stable": True},
                        "rollback_policy": {"available": True, "snapshot_id": "service-api-snapshot"},
                    },
                )
                readout_live_readiness_response = client.post(
                    "/terminus/snn-language-sequence/plasticity-live-application-readiness",
                    json={
                        "shadow_application": readout_shadow_application_response.json(),
                        "rollback_readiness": {
                            "checkpoint_available": True,
                            "checkpoint_path": "checkpoint://readout-shadow",
                            "restore_endpoint_available": True,
                        },
                        "operator_approval": {
                            "approved": True,
                            "operator_id": "operator-test",
                            "approval_id": "readout-approval-1",
                        },
                    },
                )
                readout_live_preflight_response = client.post(
                    "/terminus/snn-language-sequence/plasticity-live-application-preflight",
                    json={
                        "live_application_readiness": readout_live_readiness_response.json(),
                        "application_target": readout_plasticity_replay_bridge_response.json()[
                            "application_target_hint"
                        ],
                        "checkpoint_transaction": {
                            "pre_update_checkpoint_saved": True,
                            "checkpoint_path": str(root / "readout_pre_language_plasticity.pt"),
                            "restore_verified": True,
                            "records_shadow_delta": True,
                        },
                    },
                )
                status_before_readout_live_application_response = client.get("/status")
                blocked_readout_live_application_response = client.post(
                    "/terminus/snn-language-sequence/plasticity-live-application",
                    json={
                        "live_application_readiness": readout_live_readiness_response.json(),
                        "shadow_delta": readout_shadow_delta_response.json(),
                        "expected_state_revision": status_before_readout_live_application_response.json()[
                            "state_revision"
                        ],
                        "operator_id": "operator-test",
                        "confirmation": False,
                    },
                )
                readout_live_application_response = client.post(
                    "/terminus/snn-language-sequence/plasticity-live-application",
                    json={
                        "live_application_readiness": readout_live_readiness_response.json(),
                        "shadow_delta": readout_shadow_delta_response.json(),
                        "expected_state_revision": status_before_readout_live_application_response.json()[
                            "state_revision"
                        ],
                        "operator_id": "operator-test",
                        "confirmation": True,
                        "checkpoint_path": str(root / "readout_pre_language_plasticity.pt"),
                    },
                )
                readout_plasticity_runtime_state_response = client.get(
                    "/terminus/snn-language-sequence/plasticity-runtime-state"
                )
                runtime_default_rollout_response = client.post(
                    "/terminus/snn-language-sequence/readout-rollout-candidate",
                    json={
                        "prediction_report": prediction_report,
                        "readout_vocabulary_slots": vocabulary,
                        "device_evidence": {"device": "cpu", "source": "service_api_readout_default_rollout"},
                        "transition_memory_evaluation": transition_memory_evaluation,
                        "rollout_steps": 3,
                        "top_k": 4,
                    },
                )
                readout_synapse_audit_response = client.get(
                    "/terminus/snn-language-sequence/readout-ledger/synapse-provenance-audit"
                )
                checkpoint_save_response = client.post(
                    "/checkpoint/save",
                    json={"path": str(root / "readout_ledger.pt")},
                )
                checkpoint_restore_response = client.post(
                    "/checkpoint/restore",
                    json={"path": checkpoint_save_response.json()["path"]},
                )
                restored_readout_plasticity_runtime_state_response = client.get(
                    "/terminus/snn-language-sequence/plasticity-runtime-state"
                )
                restored_readout_synapse_audit_response = client.get(
                    "/terminus/snn-language-sequence/readout-ledger/synapse-provenance-audit"
                )
                restored_ledger_response = client.get("/terminus/snn-language-sequence/readout-ledger")
                restored_replay_priority_response = client.get(
                    "/terminus/snn-language-sequence/readout-ledger/replay-priority"
                )
            app.state.marulho_manager.close()

        self.assertEqual(status_response.status_code, 200)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(emission_response.status_code, 200)
        self.assertEqual(emission_review_response.status_code, 200)
        self.assertEqual(emission_review_history_response.status_code, 200)
        self.assertEqual(blocked_emission_review_response.status_code, 200)
        self.assertEqual(rollout_response.status_code, 200)
        self.assertEqual(rollout_replay_evaluation_response.status_code, 200)
        self.assertEqual(rollout_record_response.status_code, 200)
        self.assertEqual(blocked_rollout_record_response.status_code, 200)
        self.assertEqual(rollout_rehearsal_policy_response.status_code, 200)
        self.assertEqual(rollout_rehearsal_evaluation_response.status_code, 200)
        self.assertEqual(rollout_rehearsal_experiment_response.status_code, 200)
        self.assertEqual(rollout_consolidation_design_response.status_code, 200)
        self.assertEqual(rollout_consolidation_shadow_delta_response.status_code, 200)
        self.assertEqual(rollout_consolidation_shadow_application_preflight_response.status_code, 200)
        self.assertEqual(rollout_developmental_plasticity_review_response.status_code, 200)
        self.assertEqual(rollout_regeneration_proposal_adapter_response.status_code, 200)
        self.assertEqual(rollout_regeneration_replay_artifact_review_response.status_code, 200)
        self.assertEqual(rollout_regeneration_permit_request_response.status_code, 200)
        self.assertEqual(rollout_regeneration_application_preflight_response.status_code, 200)
        self.assertEqual(rollout_regeneration_application_response.status_code, 200)
        self.assertEqual(pending_evaluation_response.status_code, 200)
        self.assertEqual(pending_emission_response.status_code, 200)
        self.assertEqual(blocked_record_response.status_code, 200)
        self.assertEqual(record_response.status_code, 200)
        self.assertEqual(emission_replay_policy_response.status_code, 200)
        self.assertEqual(emission_replay_design_response.status_code, 200)
        self.assertEqual(blocked_emission_replay_context_response.status_code, 200)
        self.assertEqual(emission_replay_context_response.status_code, 200)
        self.assertEqual(emission_replay_consolidation_priority_response.status_code, 200)
        self.assertEqual(ledger_response.status_code, 200)
        self.assertEqual(replay_priority_response.status_code, 200)
        self.assertEqual(rehearsal_evaluation_response.status_code, 200)
        self.assertEqual(rehearsal_experiment_response.status_code, 200)
        self.assertEqual(replay_design_response.status_code, 200)
        self.assertEqual(replay_dry_run_response.status_code, 200)
        self.assertEqual(readout_plasticity_preflight_response.status_code, 200)
        self.assertEqual(readout_plasticity_replay_bridge_response.status_code, 200)
        self.assertEqual(readout_application_design_response.status_code, 200)
        self.assertEqual(readout_shadow_delta_response.status_code, 200)
        self.assertEqual(readout_shadow_application_response.status_code, 200)
        self.assertEqual(readout_live_readiness_response.status_code, 200)
        self.assertEqual(readout_live_preflight_response.status_code, 200)
        self.assertEqual(status_before_readout_live_application_response.status_code, 200)
        self.assertEqual(blocked_readout_live_application_response.status_code, 200)
        self.assertEqual(readout_live_application_response.status_code, 200)
        self.assertEqual(readout_plasticity_runtime_state_response.status_code, 200)
        self.assertEqual(runtime_default_rollout_response.status_code, 200)
        self.assertEqual(readout_synapse_audit_response.status_code, 200)
        self.assertEqual(checkpoint_save_response.status_code, 200)
        self.assertEqual(checkpoint_restore_response.status_code, 200)
        self.assertEqual(restored_readout_plasticity_runtime_state_response.status_code, 200)
        self.assertEqual(restored_readout_synapse_audit_response.status_code, 200)
        self.assertEqual(restored_ledger_response.status_code, 200)
        self.assertEqual(restored_replay_priority_response.status_code, 200)
        body = response.json()
        emission = emission_response.json()
        emission_review = emission_review_response.json()
        emission_review_history = emission_review_history_response.json()
        blocked_emission_review = blocked_emission_review_response.json()
        pending_evaluation_body = pending_evaluation_response.json()
        pending_emission = pending_emission_response.json()
        blocked_record = blocked_record_response.json()
        record = record_response.json()
        emission_replay_policy = emission_replay_policy_response.json()
        emission_replay_design = emission_replay_design_response.json()
        blocked_emission_replay_context = blocked_emission_replay_context_response.json()
        emission_replay_context = emission_replay_context_response.json()
        emission_replay_consolidation_priority = (
            emission_replay_consolidation_priority_response.json()
        )
        ledger = ledger_response.json()
        replay_priority = replay_priority_response.json()
        rehearsal_evaluation = rehearsal_evaluation_response.json()
        rehearsal_experiment = rehearsal_experiment_response.json()
        replay_design = replay_design_response.json()
        replay_dry_run = replay_dry_run_response.json()
        readout_plasticity_preflight = readout_plasticity_preflight_response.json()
        readout_plasticity_replay_bridge = readout_plasticity_replay_bridge_response.json()
        readout_application_design = readout_application_design_response.json()
        readout_shadow_delta = readout_shadow_delta_response.json()
        readout_shadow_application = readout_shadow_application_response.json()
        readout_live_readiness = readout_live_readiness_response.json()
        readout_live_preflight = readout_live_preflight_response.json()
        status_before_readout_live_application = status_before_readout_live_application_response.json()
        blocked_readout_live_application = blocked_readout_live_application_response.json()
        readout_live_application = readout_live_application_response.json()
        readout_plasticity_runtime_state = readout_plasticity_runtime_state_response.json()
        runtime_default_rollout = runtime_default_rollout_response.json()
        readout_synapse_audit = readout_synapse_audit_response.json()
        restored_readout_plasticity_runtime_state = (
            restored_readout_plasticity_runtime_state_response.json()
        )
        restored_readout_synapse_audit = restored_readout_synapse_audit_response.json()
        restored_ledger = restored_ledger_response.json()
        restored_replay_priority = restored_replay_priority_response.json()
        rollout = rollout_response.json()
        rollout_replay_evaluation = rollout_replay_evaluation_response.json()
        rollout_record = rollout_record_response.json()
        blocked_rollout_record = blocked_rollout_record_response.json()
        rollout_rehearsal_policy = rollout_rehearsal_policy_response.json()
        rollout_rehearsal_evaluation = rollout_rehearsal_evaluation_response.json()
        rollout_rehearsal_experiment = rollout_rehearsal_experiment_response.json()
        rollout_consolidation_design = rollout_consolidation_design_response.json()
        rollout_consolidation_shadow_delta = rollout_consolidation_shadow_delta_response.json()
        rollout_consolidation_shadow_application_preflight = (
            rollout_consolidation_shadow_application_preflight_response.json()
        )
        rollout_developmental_plasticity_review = (
            rollout_developmental_plasticity_review_response.json()
        )
        rollout_regeneration_proposal_adapter = (
            rollout_regeneration_proposal_adapter_response.json()
        )
        rollout_regeneration_replay_artifact_review = (
            rollout_regeneration_replay_artifact_review_response.json()
        )
        rollout_regeneration_permit_request = rollout_regeneration_permit_request_response.json()
        rollout_regeneration_application_preflight = (
            rollout_regeneration_application_preflight_response.json()
        )
        rollout_regeneration_application = rollout_regeneration_application_response.json()
        self.assertEqual(body["surface"], "snn_language_readout_draft.v1")
        self.assertTrue(body["generates_text"])
        self.assertTrue(body["decodes_text"])
        self.assertFalse(body["freeform_language_generation"])
        self.assertFalse(body["mutates_runtime_state"])
        self.assertIn("memory pressure", body["draft"]["text"])
        self.assertTrue(body["transition_memory_evaluation_evidence"]["review_ready"])
        self.assertTrue(body["promotion_gate"]["eligible_for_bounded_readout_generation"])
        self.assertFalse(body["promotion_gate"]["eligible_for_cognition_substrate"])
        self.assertEqual(emission["surface"], "snn_language_readout_emission.v1")
        self.assertTrue(emission["ready"])
        self.assertTrue(emission["generates_text"])
        self.assertTrue(emission["decodes_text"])
        self.assertFalse(emission["freeform_language_generation"])
        self.assertFalse(emission["mutates_runtime_state"])
        self.assertFalse(emission["promotes_fact"])
        self.assertFalse(emission["promotes_action"])
        self.assertFalse(emission["cognition_substrate"])
        self.assertEqual(emission["language_output"]["text"], body["draft"]["text"])
        self.assertIn("memory pressure", emission["language_output"]["text"])
        self.assertTrue(emission["promotion_gate"]["eligible_for_operator_display"])
        self.assertFalse(
            emission["promotion_gate"]["eligible_for_freeform_language_generation"]
        )
        self.assertEqual(
            emission["emission_binding"]["transition_memory_evaluation_hash"],
            transition_memory_evaluation["provenance_evidence"]["evaluation_hash"],
        )
        self.assertEqual(
            emission["emission_binding"]["trajectory_hash"],
            body["readout_trajectory_evidence"]["provenance_evidence"]["trajectory_hash"],
        )
        self.assertEqual(
            emission_review["surface"],
            "snn_language_readout_emission_review_record.v1",
        )
        self.assertTrue(emission_review["accepted"])
        self.assertTrue(emission_review["mutates_runtime_state"])
        self.assertFalse(emission_review["generates_text"])
        self.assertFalse(emission_review["decodes_text"])
        self.assertFalse(
            emission_review["promotion_gate"]["eligible_for_replay_memory"]
        )
        self.assertFalse(
            emission_review["promotion_gate"]["eligible_for_plasticity_application"]
        )
        self.assertFalse(
            emission_review["promotion_gate"]["eligible_for_fact_promotion"]
        )
        self.assertFalse(emission_review["promotion_gate"]["eligible_for_action"])
        self.assertEqual(
            emission_review["recorded_event"]["emission_hash"],
            emission["emission_hash"],
        )
        self.assertEqual(
            emission_review["ledger_summary"]["emission_review_event_count"],
            1,
        )
        self.assertEqual(
            emission_review_history["surface"],
            "snn_language_readout_emission_review_history.v1",
        )
        self.assertFalse(emission_review_history["executable"])
        self.assertFalse(emission_review_history["records_ledger_event"])
        self.assertFalse(emission_review_history["mutates_runtime_state"])
        self.assertFalse(emission_review_history["generates_text"])
        self.assertFalse(emission_review_history["decodes_text"])
        self.assertTrue(emission_review_history["exposes_reviewed_bounded_text"])
        self.assertEqual(
            emission_review_history["summary"]["returned_emission_review_event_count"],
            1,
        )
        self.assertEqual(
            emission_review_history["emission_review_events"][0]["text"],
            emission["language_output"]["text"],
        )
        self.assertEqual(
            emission_review_history["emission_review_events"][0]["labels"],
            emission["language_output"]["labels"],
        )
        self.assertFalse(
            emission_review_history["emission_review_events"][0][
                "eligible_for_replay_memory"
            ]
        )
        self.assertFalse(
            emission_review_history["emission_review_events"][0][
                "eligible_for_plasticity_application"
            ]
        )
        self.assertFalse(
            emission_review_history["emission_review_events"][0][
                "eligible_for_fact_promotion"
            ]
        )
        self.assertFalse(
            emission_review_history["emission_review_events"][0]["eligible_for_action"]
        )
        self.assertFalse(
            emission_review_history["promotion_gate"]["eligible_for_replay_memory"]
        )
        self.assertFalse(
            emission_review_history["promotion_gate"][
                "eligible_for_plasticity_application"
            ]
        )
        self.assertFalse(
            emission_review_history["promotion_gate"]["eligible_for_fact_promotion"]
        )
        self.assertFalse(emission_review_history["promotion_gate"]["eligible_for_action"])
        self.assertNotIn("events", emission_review_history)
        self.assertNotIn("rollout_events", emission_review_history)
        self.assertNotIn("replay_targets", emission_review_history)
        self.assertNotIn(
            "prediction_report",
            emission_review_history["emission_review_events"][0],
        )
        self.assertNotIn(
            "transition_memory_evaluation",
            emission_review_history["emission_review_events"][0],
        )
        self.assertEqual(
            emission_replay_policy["surface"],
            "snn_language_readout_emission_replay_evaluation_policy.v1",
        )
        self.assertFalse(emission_replay_policy["executable"])
        self.assertFalse(emission_replay_policy["records_ledger_event"])
        self.assertFalse(emission_replay_policy["runs_replay"])
        self.assertFalse(emission_replay_policy["generates_text"])
        self.assertFalse(emission_replay_policy["decodes_text"])
        self.assertFalse(emission_replay_policy["exposes_reviewed_bounded_text"])
        self.assertFalse(emission_replay_policy["mutates_runtime_state"])
        self.assertFalse(emission_replay_policy["eligible_for_replay_memory"])
        self.assertFalse(emission_replay_policy["eligible_for_live_replay"])
        self.assertFalse(emission_replay_policy["eligible_for_plasticity_application"])
        self.assertFalse(emission_replay_policy["eligible_for_fact_promotion"])
        self.assertFalse(emission_replay_policy["eligible_for_action"])
        self.assertEqual(emission_replay_policy["candidate_count"], 1)
        self.assertEqual(emission_replay_policy["ready_candidate_count"], 1)
        self.assertEqual(emission_replay_policy["unmatched_emission_review_count"], 0)
        self.assertEqual(
            emission_replay_policy["candidates"][0]["readout_evidence_hash"],
            record["recorded_event"]["readout_evidence_hash"],
        )
        self.assertEqual(
            emission_replay_policy["candidates"][0]["emission_hash"],
            emission["emission_hash"],
        )
        self.assertTrue(
            emission_replay_policy["candidates"][0][
                "eligible_for_replay_evaluation_policy_review"
            ]
        )
        self.assertFalse(
            emission_replay_policy["candidates"][0]["eligible_for_replay_memory"]
        )
        self.assertNotIn("text", emission_replay_policy["candidates"][0])
        self.assertNotIn("labels", emission_replay_policy["candidates"][0])
        self.assertNotIn("events", emission_replay_policy)
        self.assertNotIn("rollout_events", emission_replay_policy)
        self.assertEqual(
            emission_replay_design["surface"],
            "snn_language_readout_emission_replay_evaluation_design.v1",
        )
        self.assertFalse(emission_replay_design["executable"])
        self.assertFalse(emission_replay_design["records_ledger_event"])
        self.assertFalse(emission_replay_design["runs_replay"])
        self.assertFalse(emission_replay_design["generates_text"])
        self.assertFalse(emission_replay_design["decodes_text"])
        self.assertFalse(emission_replay_design["exposes_reviewed_bounded_text"])
        self.assertFalse(emission_replay_design["mutates_runtime_state"])
        self.assertFalse(emission_replay_design["eligible_for_replay_memory"])
        self.assertEqual(
            emission_replay_design["emission_replay_evaluation_design"][
                "selected_seed_count"
            ],
            1,
        )
        self.assertFalse(
            emission_replay_design["emission_replay_evaluation_design"][
                "records_replay_context"
            ]
        )
        self.assertEqual(
            emission_replay_design["selected_replay_context_seeds"][0][
                "readout_evidence_hash"
            ],
            record["recorded_event"]["readout_evidence_hash"],
        )
        self.assertTrue(
            emission_replay_design["selected_replay_context_seeds"][0][
                "internal_readout_ledger_match"
            ]
        )
        self.assertTrue(
            emission_replay_design["selected_replay_context_seeds"][0][
                "eligible_for_replay_context_review"
            ]
        )
        self.assertFalse(
            emission_replay_design["selected_replay_context_seeds"][0][
                "eligible_for_replay_memory"
            ]
        )
        self.assertNotIn("text", emission_replay_design["selected_replay_context_seeds"][0])
        self.assertNotIn("labels", emission_replay_design["selected_replay_context_seeds"][0])
        self.assertFalse(
            emission_replay_design["replay_context_review_requirements"][
                "accepts_display_text"
            ]
        )
        self.assertFalse(
            emission_replay_design["promotion_gate"]["eligible_for_replay_context_recording"]
        )
        self.assertFalse(
            emission_replay_design["promotion_gate"]["eligible_for_plasticity_application"]
        )
        self.assertEqual(
            emission_replay_design["promotion_gate"]["next_gate"],
            "/terminus/snn-language-sequence/replay-evaluation-context",
        )
        self.assertEqual(
            blocked_emission_replay_context["surface"],
            "snn_language_readout_emission_replay_context_review.v1",
        )
        self.assertFalse(blocked_emission_replay_context["accepted"])
        self.assertFalse(blocked_emission_replay_context["records_replay_context"])
        self.assertFalse(blocked_emission_replay_context["mutates_runtime_state"])
        self.assertFalse(
            blocked_emission_replay_context["promotion_gate"]["required_evidence"][
                "operator_confirmation"
            ]
        )
        self.assertEqual(
            emission_replay_context["surface"],
            "snn_language_readout_emission_replay_context_review.v1",
        )
        self.assertTrue(emission_replay_context["accepted"])
        self.assertTrue(emission_replay_context["records_replay_context"])
        self.assertFalse(emission_replay_context["records_ledger_event"])
        self.assertFalse(emission_replay_context["runs_replay"])
        self.assertFalse(emission_replay_context["writes_checkpoint"])
        self.assertFalse(emission_replay_context["generates_text"])
        self.assertFalse(emission_replay_context["decodes_text"])
        self.assertFalse(emission_replay_context["exposes_reviewed_bounded_text"])
        self.assertFalse(emission_replay_context["applies_plasticity"])
        self.assertTrue(emission_replay_context["mutates_runtime_state"])
        self.assertFalse(emission_replay_context["eligible_for_replay_memory"])
        self.assertFalse(emission_replay_context["eligible_for_live_replay"])
        self.assertFalse(emission_replay_context["eligible_for_plasticity_application"])
        self.assertFalse(emission_replay_context["eligible_for_fact_promotion"])
        self.assertFalse(emission_replay_context["eligible_for_action"])
        self.assertEqual(
            emission_replay_context["review"]["prediction_hash"],
            prediction_report["provenance_evidence"]["prediction_hash"],
        )
        self.assertEqual(
            emission_replay_context["review"]["readout_evidence_hash"],
            record["recorded_event"]["readout_evidence_hash"],
        )
        self.assertIsNotNone(
            emission_replay_context["review"]["replay_evaluation_context_id"]
        )
        self.assertIsNotNone(
            emission_replay_context["review"]["replay_evaluation_context_hash"]
        )
        self.assertIsNotNone(
            emission_replay_context["review"][
                "replay_evaluation_context_source_metadata_hash"
            ]
        )
        self.assertFalse(
            emission_replay_context["promotion_gate"][
                "eligible_for_replay_context_recording"
            ]
        )
        self.assertEqual(
            emission_replay_context["promotion_gate"]["next_gate"],
            "/terminus/snn-language-sequence/replay-consolidation-priority-queue",
        )
        recorded_context = next(
            iter(app.state.marulho_manager._replay_controller.snn_replay_evaluation_contexts)
        )
        self.assertEqual(
            recorded_context["source_metadata"]["emission_hash"],
            emission["emission_hash"],
        )
        self.assertEqual(
            recorded_context["source_metadata"]["readout_evidence_hash"],
            record["recorded_event"]["readout_evidence_hash"],
        )
        self.assertEqual(
            recorded_context["source_metadata"]["prediction_hash"],
            prediction_report["provenance_evidence"]["prediction_hash"],
        )
        self.assertEqual(
            recorded_context["source_metadata_hash"],
            emission_replay_context["review"][
                "replay_evaluation_context_source_metadata_hash"
            ],
        )
        self.assertEqual(
            emission_replay_consolidation_priority["surface"],
            "snn_replay_consolidation_priority_queue.v1",
        )
        self.assertEqual(emission_replay_consolidation_priority["candidate_count"], 1)
        emission_replay_candidate = emission_replay_consolidation_priority["candidates"][0]
        self.assertEqual(
            emission_replay_candidate["source_metadata_hash"],
            emission_replay_context["review"][
                "replay_evaluation_context_source_metadata_hash"
            ],
        )
        self.assertEqual(
            emission_replay_candidate["emission_lineage"]["emission_hash"],
            emission["emission_hash"],
        )
        self.assertEqual(
            emission_replay_candidate["emission_lineage"]["readout_evidence_hash"],
            record["recorded_event"]["readout_evidence_hash"],
        )
        self.assertEqual(
            emission_replay_candidate["emission_lineage"]["prediction_hash"],
            prediction_report["provenance_evidence"]["prediction_hash"],
        )
        self.assertNotIn("source_metadata", emission_replay_candidate)
        self.assertNotIn("operator_id", emission_replay_candidate["emission_lineage"])
        self.assertFalse(blocked_emission_review["accepted"])
        self.assertFalse(
            blocked_emission_review["promotion_gate"]["required_evidence"][
                "confirmation"
            ]
        )
        self.assertFalse(pending_emission["ready"])
        self.assertFalse(pending_emission["generates_text"])
        self.assertEqual(pending_emission["language_output"]["text"], "")
        self.assertFalse(
            pending_emission["promotion_gate"]["required_evidence"][
                "draft_bounded_generation_ready"
            ]
        )
        self.assertEqual(rollout["surface"], "snn_language_readout_rollout_candidate.v1")
        self.assertTrue(rollout["generates_text"])
        self.assertFalse(rollout["freeform_language_generation"])
        self.assertFalse(rollout["mutates_runtime_state"])
        self.assertFalse(rollout["loads_external_checkpoint"])
        self.assertFalse(rollout["promotion_gate"]["eligible_for_cognition_substrate"])
        self.assertFalse(rollout["promotion_gate"]["eligible_for_fact_promotion"])
        self.assertFalse(rollout["promotion_gate"]["eligible_for_action"])
        self.assertIn("memory pressure", rollout["rollout"]["labels"])
        self.assertEqual(
            rollout["readout_rollout_evidence"]["transition_memory_state_source"],
            "service.runtime_facade.snn_language_plasticity_runtime_state",
        )
        self.assertTrue(
            rollout["readout_rollout_evidence"]["server_transition_memory_hash_match"]
        )
        self.assertTrue(
            rollout["readout_rollout_evidence"][
                "caller_transition_memory_state_absent_or_ignored"
            ]
        )
        self.assertTrue(
            rollout["readout_rollout_evidence"][
                "prediction_transition_memory_hash_match"
            ]
        )
        self.assertTrue(
            rollout["readout_rollout_evidence"][
                "transition_memory_evaluation_hash_match"
            ]
        )
        self.assertTrue(
            rollout["readout_trajectory_evidence"]["promotion_gate"][
                "eligible_for_bounded_snn_language_readout"
            ]
        )
        self.assertEqual(
            rollout_replay_evaluation["surface"],
            "snn_language_readout_rollout_replay_evaluation.v1",
        )
        self.assertFalse(rollout_replay_evaluation["generates_text"])
        self.assertFalse(rollout_replay_evaluation["freeform_language_generation"])
        self.assertFalse(rollout_replay_evaluation["decodes_text"])
        self.assertFalse(rollout_replay_evaluation["mutates_runtime_state"])
        self.assertFalse(rollout_replay_evaluation["recorded_in_ledger"])
        self.assertFalse(rollout_replay_evaluation["eligible_for_replay_priority"])
        self.assertTrue(
            rollout_replay_evaluation["promotion_gate"][
                "eligible_for_readout_rollout_ledger_recording_review"
            ]
        )
        self.assertFalse(
            rollout_replay_evaluation["promotion_gate"]["eligible_for_replay_priority"]
        )
        self.assertGreaterEqual(
            rollout_replay_evaluation["replay_evaluation"]["target_count"],
            1,
        )
        self.assertEqual(
            rollout_replay_evaluation["provenance_evidence"][
                "server_transition_memory_hash"
            ],
            rollout["readout_rollout_evidence"]["server_transition_memory_hash"],
        )
        self.assertTrue(
            rollout_replay_evaluation["provenance_evidence"][
                "server_transition_memory_hash_match"
            ]
        )
        self.assertEqual(
            rollout_replay_evaluation["provenance_evidence"][
                "transition_memory_state_source"
            ],
            "service.runtime_facade.snn_language_plasticity_runtime_state",
        )
        self.assertEqual(
            rollout_record["surface"],
            "snn_language_readout_rollout_evidence_ledger_record.v1",
        )
        self.assertTrue(rollout_record["accepted"])
        self.assertTrue(rollout_record["mutates_runtime_state"])
        self.assertFalse(rollout_record["generates_text"])
        self.assertFalse(rollout_record["promotion_gate"]["eligible_for_replay_priority"])
        self.assertTrue(rollout_record["promotion_gate"]["eligible_for_rollout_replay_memory"])
        self.assertTrue(rollout_record["recorded_event"]["recorded_in_ledger"])
        self.assertFalse(rollout_record["recorded_event"]["eligible_for_replay_priority"])
        self.assertEqual(
            rollout_record["recorded_event"]["server_transition_memory_hash"],
            rollout["readout_rollout_evidence"]["server_transition_memory_hash"],
        )
        self.assertTrue(
            rollout_record["recorded_event"]["server_transition_memory_hash_match"]
        )
        self.assertEqual(
            rollout_record["recorded_event"]["transition_memory_state_source"],
            "service.runtime_facade.snn_language_plasticity_runtime_state",
        )
        self.assertFalse(blocked_rollout_record["accepted"])
        self.assertFalse(blocked_rollout_record["promotion_gate"]["required_evidence"]["confirmation"])
        self.assertEqual(
            rollout_rehearsal_policy["surface"],
            "snn_language_readout_rollout_rehearsal_promotion_policy.v1",
        )
        self.assertTrue(rollout_rehearsal_policy["advisory"])
        self.assertFalse(rollout_rehearsal_policy["executable"])
        self.assertFalse(rollout_rehearsal_policy["mutates_runtime_state"])
        self.assertFalse(rollout_rehearsal_policy["generates_text"])
        self.assertFalse(rollout_rehearsal_policy["applies_plasticity"])
        self.assertEqual(rollout_rehearsal_policy["candidate_count"], 1)
        self.assertEqual(
            rollout_rehearsal_policy["candidates"][0]["device_evidence"]["tensor_device"],
            "cpu",
        )
        self.assertEqual(
            rollout_rehearsal_policy["candidates"][0]["server_transition_memory_hash"],
            rollout["readout_rollout_evidence"]["server_transition_memory_hash"],
        )
        self.assertTrue(
            rollout_rehearsal_policy["candidates"][0][
                "server_transition_memory_hash_match"
            ]
        )
        self.assertTrue(
            rollout_rehearsal_policy["promotion_gate"][
                "eligible_for_operator_rollout_rehearsal_review"
            ]
        )
        self.assertFalse(rollout_rehearsal_policy["promotion_gate"]["eligible_for_replay_priority"])
        self.assertFalse(rollout_rehearsal_policy["promotion_gate"]["eligible_for_live_replay"])
        self.assertEqual(
            rollout_rehearsal_evaluation["surface"],
            "snn_language_readout_rollout_rehearsal_evaluation.v1",
        )
        self.assertFalse(rollout_rehearsal_evaluation["generates_text"])
        self.assertFalse(rollout_rehearsal_evaluation["mutates_runtime_state"])
        self.assertFalse(rollout_rehearsal_evaluation["applies_plasticity"])
        self.assertFalse(rollout_rehearsal_evaluation["returns_trained_weights"])
        self.assertEqual(rollout_rehearsal_evaluation["rehearsal_summary"]["candidate_count"], 1)
        self.assertGreater(
            rollout_rehearsal_evaluation["rehearsal_summary"]["activation_sparsity"],
            0.0,
        )
        self.assertFalse(
            rollout_rehearsal_evaluation["ephemeral_rehearsal"]["runtime_update_applied"]
        )
        self.assertFalse(
            rollout_rehearsal_evaluation["ephemeral_rehearsal"]["weights_persisted"]
        )
        self.assertFalse(
            rollout_rehearsal_evaluation["ephemeral_rehearsal"]["checkpoint_written"]
        )
        self.assertTrue(
            rollout_rehearsal_evaluation["promotion_gate"][
                "eligible_for_operator_rollout_rehearsal_review"
            ]
        )
        self.assertFalse(
            rollout_rehearsal_evaluation["promotion_gate"]["eligible_for_live_replay"]
        )
        self.assertEqual(
            rollout_rehearsal_experiment["surface"],
            "snn_language_readout_rollout_rehearsal_experiment.v1",
        )
        self.assertFalse(rollout_rehearsal_experiment["generates_text"])
        self.assertFalse(rollout_rehearsal_experiment["mutates_runtime_state"])
        self.assertFalse(rollout_rehearsal_experiment["applies_plasticity"])
        self.assertFalse(rollout_rehearsal_experiment["returns_trained_weights"])
        self.assertEqual(rollout_rehearsal_experiment["experiment_summary"]["replay_cycles"], 4)
        self.assertEqual(
            rollout_rehearsal_experiment["experiment_summary"]["minimum_cycle_stability"],
            1.0,
        )
        self.assertFalse(
            rollout_rehearsal_experiment["ephemeral_experiment"]["runtime_update_applied"]
        )
        self.assertFalse(
            rollout_rehearsal_experiment["ephemeral_experiment"]["weights_persisted"]
        )
        self.assertFalse(
            rollout_rehearsal_experiment["ephemeral_experiment"]["checkpoint_written"]
        )
        self.assertFalse(
            rollout_rehearsal_experiment["ephemeral_experiment"]["plasticity_applied"]
        )
        self.assertTrue(
            rollout_rehearsal_experiment["promotion_gate"][
                "eligible_for_operator_rollout_rehearsal_experiment_review"
            ]
        )
        self.assertFalse(
            rollout_rehearsal_experiment["promotion_gate"]["eligible_for_live_replay"]
        )
        self.assertEqual(
            rollout_consolidation_design["surface"],
            "snn_language_readout_rollout_consolidation_design.v1",
        )
        self.assertFalse(rollout_consolidation_design["generates_text"])
        self.assertFalse(rollout_consolidation_design["mutates_runtime_state"])
        self.assertFalse(rollout_consolidation_design["applies_plasticity"])
        self.assertFalse(rollout_consolidation_design["returns_trained_weights"])
        self.assertEqual(
            rollout_consolidation_design["rollout_consolidation_design"]["candidate_synapse_count"],
            0,
        )
        self.assertFalse(
            rollout_consolidation_design["rollout_consolidation_design"]["runtime_update_applied"]
        )
        self.assertFalse(
            rollout_consolidation_design["rollout_consolidation_design"]["weights_persisted"]
        )
        self.assertFalse(
            rollout_consolidation_design["rollout_consolidation_design"]["structural_write_applied"]
        )
        self.assertFalse(
            rollout_consolidation_design["promotion_gate"][
                "eligible_for_operator_rollout_consolidation_design_review"
            ]
        )
        self.assertFalse(
            rollout_consolidation_design["promotion_gate"]["required_evidence"][
                "candidate_synapses_available"
            ]
        )
        self.assertFalse(
            rollout_consolidation_design["promotion_gate"]["eligible_for_structural_write"]
        )
        self.assertFalse(
            rollout_consolidation_design["promotion_gate"]["eligible_for_plasticity_application"]
        )
        self.assertEqual(
            rollout_consolidation_shadow_delta["surface"],
            "snn_language_readout_rollout_consolidation_shadow_delta.v1",
        )
        self.assertFalse(rollout_consolidation_shadow_delta["generates_text"])
        self.assertFalse(rollout_consolidation_shadow_delta["mutates_runtime_state"])
        self.assertFalse(rollout_consolidation_shadow_delta["applies_plasticity"])
        self.assertFalse(rollout_consolidation_shadow_delta["returns_trained_weights"])
        self.assertEqual(rollout_consolidation_shadow_delta["affected_synapse_count"], 0)
        self.assertEqual(rollout_consolidation_shadow_delta["device_evidence"]["tensor_device"], "cpu")
        self.assertFalse(
            rollout_consolidation_shadow_delta["shadow_delta"]["runtime_update_applied"]
        )
        self.assertFalse(
            rollout_consolidation_shadow_delta["shadow_delta"]["weights_persisted"]
        )
        self.assertFalse(
            rollout_consolidation_shadow_delta["promotion_gate"][
                "eligible_for_operator_rollout_consolidation_shadow_review"
            ]
        )
        self.assertFalse(
            rollout_consolidation_shadow_delta["promotion_gate"]["required_evidence"][
                "candidate_synapses_available"
            ]
        )
        self.assertEqual(
            rollout_consolidation_shadow_application_preflight["surface"],
            "snn_language_readout_rollout_consolidation_shadow_application_preflight.v1",
        )
        self.assertFalse(rollout_consolidation_shadow_application_preflight["generates_text"])
        self.assertFalse(
            rollout_consolidation_shadow_application_preflight["mutates_runtime_state"]
        )
        self.assertFalse(rollout_consolidation_shadow_application_preflight["applies_plasticity"])
        self.assertEqual(
            rollout_consolidation_shadow_application_preflight["preflight_summary"][
                "affected_synapse_count"
            ],
            0,
        )
        self.assertFalse(
            rollout_consolidation_shadow_application_preflight["promotion_gate"][
                "eligible_for_operator_rollout_consolidation_shadow_application_preflight_review"
            ]
        )
        self.assertFalse(
            rollout_consolidation_shadow_application_preflight["promotion_gate"][
                "required_evidence"
            ]["shadow_synapses_available"]
        )
        self.assertFalse(
            rollout_consolidation_shadow_application_preflight["promotion_gate"][
                "eligible_for_structural_write"
            ]
        )
        self.assertEqual(
            rollout_developmental_plasticity_review["surface"],
            "snn_language_readout_rollout_developmental_plasticity_review.v1",
        )
        self.assertFalse(rollout_developmental_plasticity_review["generates_text"])
        self.assertFalse(rollout_developmental_plasticity_review["mutates_runtime_state"])
        self.assertFalse(rollout_developmental_plasticity_review["applies_plasticity"])
        self.assertEqual(
            rollout_developmental_plasticity_review["developmental_plasticity_review"][
                "growth_candidate_count"
            ],
            0,
        )
        self.assertFalse(
            rollout_developmental_plasticity_review["promotion_gate"][
                "eligible_for_operator_rollout_developmental_plasticity_review"
            ]
        )
        self.assertFalse(
            rollout_developmental_plasticity_review["promotion_gate"]["eligible_for_structural_write"]
        )
        self.assertEqual(
            rollout_regeneration_proposal_adapter["surface"],
            "snn_language_readout_rollout_regeneration_proposal_adapter.v1",
        )
        self.assertFalse(rollout_regeneration_proposal_adapter["generates_text"])
        self.assertFalse(rollout_regeneration_proposal_adapter["mutates_runtime_state"])
        self.assertFalse(rollout_regeneration_proposal_adapter["applies_plasticity"])
        self.assertFalse(rollout_regeneration_proposal_adapter["issues_regeneration_permit"])
        self.assertFalse(rollout_regeneration_proposal_adapter["executor_ready"])
        self.assertEqual(
            rollout_regeneration_proposal_adapter["regeneration_design"]["candidate_count"],
            0,
        )
        self.assertFalse(rollout_regeneration_proposal_adapter["blocked_replay_evidence"]["ready"])
        self.assertFalse(
            rollout_regeneration_proposal_adapter["promotion_gate"][
                "eligible_for_regeneration_permit_request"
            ]
        )
        self.assertFalse(
            rollout_regeneration_proposal_adapter["promotion_gate"][
                "eligible_for_regeneration_application"
            ]
        )
        self.assertEqual(
            rollout_regeneration_replay_artifact_review["surface"],
            "snn_language_readout_rollout_regeneration_replay_artifact_review.v1",
        )
        self.assertFalse(rollout_regeneration_replay_artifact_review["generates_text"])
        self.assertFalse(rollout_regeneration_replay_artifact_review["mutates_runtime_state"])
        self.assertFalse(rollout_regeneration_replay_artifact_review["applies_plasticity"])
        self.assertFalse(rollout_regeneration_replay_artifact_review["issues_regeneration_permit"])
        self.assertFalse(
            rollout_regeneration_replay_artifact_review["promotion_gate"][
                "eligible_for_regeneration_permit_request"
            ]
        )
        self.assertFalse(
            rollout_regeneration_replay_artifact_review["promotion_gate"][
                "eligible_for_regeneration_application"
            ]
        )
        self.assertEqual(
            rollout_regeneration_permit_request["surface"],
            "snn_language_readout_rollout_regeneration_permit_request.v1",
        )
        self.assertFalse(rollout_regeneration_permit_request["accepted"])
        self.assertFalse(rollout_regeneration_permit_request["issues_regeneration_permit"])
        self.assertFalse(rollout_regeneration_permit_request["applies_plasticity"])
        self.assertFalse(rollout_regeneration_permit_request["mutates_runtime_state"])
        self.assertFalse(
            rollout_regeneration_permit_request["promotion_gate"][
                "eligible_for_regeneration_application"
            ]
        )
        self.assertEqual(
            rollout_regeneration_application_preflight["surface"],
            "snn_language_readout_rollout_regeneration_application_preflight.v1",
        )
        self.assertFalse(rollout_regeneration_application_preflight["ready"])
        self.assertFalse(rollout_regeneration_application_preflight["executor_called"])
        self.assertFalse(rollout_regeneration_application_preflight["writes_checkpoint"])
        self.assertFalse(rollout_regeneration_application_preflight["applies_plasticity"])
        self.assertFalse(rollout_regeneration_application_preflight["mutates_runtime_state"])
        self.assertEqual(
            rollout_regeneration_application["surface"],
            "snn_language_readout_rollout_regeneration_application.v1",
        )
        self.assertFalse(rollout_regeneration_application["accepted"])
        self.assertFalse(rollout_regeneration_application["executor_called"])
        self.assertFalse(rollout_regeneration_application["writes_checkpoint"])
        self.assertFalse(rollout_regeneration_application["applies_plasticity"])
        self.assertFalse(rollout_regeneration_application["mutates_runtime_state"])
        self.assertEqual(runtime_default_rollout["surface"], "snn_language_readout_rollout_candidate.v1")
        self.assertFalse(runtime_default_rollout["mutates_runtime_state"])
        self.assertEqual(
            runtime_default_rollout["readout_rollout_evidence"]["persistent_transition_weight_count"],
            readout_plasticity_runtime_state["sparse_transition_weight_count"],
        )
        self.assertTrue(pending_evaluation_body["generates_text"])
        self.assertFalse(
            pending_evaluation_body["promotion_gate"]["eligible_for_bounded_readout_generation"]
        )
        self.assertEqual(
            pending_evaluation_body["promotion_gate"]["status"],
            "collect_transition_memory_prediction_evaluation",
        )
        self.assertFalse(blocked_record["accepted"])
        self.assertFalse(
            blocked_record["promotion_gate"]["required_evidence"]["bounded_readout_generation_ready"]
        )
        self.assertTrue(record["accepted"])
        self.assertTrue(record["mutates_runtime_state"])
        self.assertFalse(record["generates_text"])
        self.assertFalse(record["promotion_gate"]["eligible_for_cognition_substrate"])
        self.assertEqual(ledger["surface"], "snn_language_readout_evidence_ledger.v1")
        self.assertEqual(ledger["summary"]["event_count"], 1)
        self.assertEqual(ledger["summary"]["rollout_event_count"], 1)
        self.assertEqual(
            ledger["rollout_events"][0]["rollout_replay_evaluation_hash"],
            rollout_replay_evaluation["provenance_evidence"]["rollout_replay_evaluation_hash"],
        )
        self.assertEqual(ledger["events"][0]["prediction_hash"], body["transition_memory_evaluation_evidence"]["prediction_hash"])
        self.assertEqual(replay_priority["surface"], "snn_language_readout_replay_priority.v1")
        self.assertTrue(replay_priority["advisory"])
        self.assertFalse(replay_priority["executable"])
        self.assertFalse(replay_priority["mutates_runtime_state"])
        self.assertFalse(replay_priority["generates_text"])
        self.assertEqual(replay_priority["candidate_count"], 1)
        self.assertEqual(replay_priority["candidates"][0]["rank"], 1)
        self.assertFalse(replay_priority["candidates"][0]["eligible_for_action"])
        self.assertFalse(replay_priority["promotion_gate"]["eligible_for_live_replay"])
        self.assertEqual(
            rehearsal_evaluation["surface"],
            "snn_language_readout_rehearsal_evaluation.v1",
        )
        self.assertFalse(rehearsal_evaluation["generates_text"])
        self.assertFalse(rehearsal_evaluation["mutates_runtime_state"])
        self.assertFalse(rehearsal_evaluation["applies_plasticity"])
        self.assertEqual(rehearsal_evaluation["device_evidence"]["tensor_device"], "cpu")
        self.assertTrue(
            rehearsal_evaluation["promotion_gate"]["eligible_for_operator_rehearsal_review"]
        )
        self.assertFalse(rehearsal_evaluation["promotion_gate"]["eligible_for_live_replay"])
        self.assertEqual(
            rehearsal_experiment["surface"],
            "snn_language_readout_rehearsal_experiment.v1",
        )
        self.assertFalse(rehearsal_experiment["generates_text"])
        self.assertFalse(rehearsal_experiment["mutates_runtime_state"])
        self.assertFalse(rehearsal_experiment["applies_plasticity"])
        self.assertTrue(
            rehearsal_experiment["promotion_gate"][
                "eligible_for_operator_rehearsal_experiment_review"
            ]
        )
        self.assertFalse(rehearsal_experiment["promotion_gate"]["eligible_for_live_replay"])
        self.assertEqual(replay_design["surface"], "snn_language_readout_replay_design.v1")
        self.assertFalse(replay_design["generates_text"])
        self.assertFalse(replay_design["mutates_runtime_state"])
        self.assertFalse(replay_design["applies_plasticity"])
        self.assertEqual(replay_design["readout_replay_design"]["selected_candidate_count"], 1)
        self.assertFalse(replay_design["readout_replay_design"]["execution_allowed"])
        self.assertTrue(
            replay_design["promotion_gate"]["eligible_for_operator_replay_design_review"]
        )
        self.assertFalse(replay_design["promotion_gate"]["eligible_for_live_replay"])
        self.assertEqual(replay_dry_run["surface"], "snn_language_readout_replay_dry_run.v1")
        self.assertFalse(replay_dry_run["generates_text"])
        self.assertFalse(replay_dry_run["decodes_text"])
        self.assertFalse(replay_dry_run["mutates_runtime_state"])
        self.assertFalse(replay_dry_run["applies_plasticity"])
        self.assertFalse(replay_dry_run["returns_trained_weights"])
        self.assertEqual(replay_dry_run["device_evidence"]["tensor_device"], "cpu")
        self.assertEqual(replay_dry_run["isolated_replay_summary"]["target_count"], 1)
        self.assertFalse(replay_dry_run["ephemeral_replay"]["runtime_update_applied"])
        self.assertFalse(replay_dry_run["ephemeral_replay"]["weights_persisted"])
        self.assertFalse(replay_dry_run["ephemeral_replay"]["checkpoint_written"])
        self.assertTrue(
            replay_dry_run["promotion_gate"]["eligible_for_operator_replay_dry_run_review"]
        )
        self.assertFalse(replay_dry_run["promotion_gate"]["eligible_for_live_replay"])
        self.assertEqual(
            readout_plasticity_preflight["surface"],
            "snn_language_readout_plasticity_preflight.v1",
        )
        self.assertFalse(readout_plasticity_preflight["generates_text"])
        self.assertFalse(readout_plasticity_preflight["decodes_text"])
        self.assertFalse(readout_plasticity_preflight["mutates_runtime_state"])
        self.assertFalse(readout_plasticity_preflight["applies_plasticity"])
        self.assertFalse(readout_plasticity_preflight["returns_trained_weights"])
        self.assertGreater(
            readout_plasticity_preflight["plasticity_preflight"]["candidate_synapse_count"],
            0,
        )
        self.assertFalse(
            readout_plasticity_preflight["plasticity_preflight"]["runtime_update_applied"]
        )
        self.assertTrue(
            readout_plasticity_preflight["promotion_gate"][
                "eligible_for_operator_readout_plasticity_review"
            ]
        )
        self.assertFalse(
            readout_plasticity_preflight["promotion_gate"]["eligible_for_plasticity_application"]
        )
        self.assertEqual(
            readout_plasticity_replay_bridge["surface"],
            "snn_language_plasticity_replay_experiment.v1",
        )
        self.assertEqual(
            readout_plasticity_replay_bridge["artifact_kind"],
            "terminus_snn_language_plasticity_replay_experiment",
        )
        self.assertFalse(readout_plasticity_replay_bridge["mutates_runtime_state"])
        self.assertFalse(readout_plasticity_replay_bridge["applies_plasticity"])
        self.assertGreater(
            readout_plasticity_replay_bridge["replay_experiment"]["replay_sequence_count"],
            0,
        )
        self.assertEqual(readout_plasticity_replay_bridge["device_evidence"]["tensor_device"], "cpu")
        self.assertTrue(
            readout_plasticity_replay_bridge["device_evidence"]["device_report_available"]
        )
        self.assertEqual(
            readout_plasticity_replay_bridge["application_design"]["learning_rate"],
            readout_plasticity_preflight["plasticity_preflight"]["learning_rate"],
        )
        self.assertEqual(
            readout_plasticity_replay_bridge["application_design"]["max_weight_delta"],
            readout_plasticity_preflight["plasticity_preflight"]["max_weight_delta"],
        )
        self.assertFalse(
            readout_plasticity_replay_bridge["application_design"]["runtime_update_applied"]
        )
        self.assertTrue(
            readout_plasticity_replay_bridge["promotion_gate"][
                "eligible_for_operator_application_review"
            ]
        )
        self.assertEqual(
            readout_application_design["surface"],
            "snn_language_plasticity_application_design.v1",
        )
        self.assertFalse(readout_application_design["mutates_runtime_state"])
        self.assertFalse(readout_application_design["applies_plasticity"])
        self.assertTrue(
            readout_application_design["promotion_gate"][
                "eligible_for_operator_application_review"
            ]
        )
        self.assertEqual(readout_shadow_delta["surface"], "snn_language_plasticity_shadow_delta.v1")
        self.assertFalse(readout_shadow_delta["mutates_runtime_state"])
        self.assertFalse(readout_shadow_delta["applies_plasticity"])
        self.assertGreater(readout_shadow_delta["affected_synapse_count"], 0)
        self.assertEqual(
            readout_shadow_delta["bounded_synapses"][0]["readout_evidence_hash"],
            readout_plasticity_replay_bridge["canonical_replay_sequences"][0][
                "readout_evidence_hash"
            ],
        )
        self.assertEqual(
            readout_shadow_delta["bounded_synapses"][0]["transition_memory_evaluation_hash"],
            readout_plasticity_replay_bridge["canonical_replay_sequences"][0][
                "transition_memory_evaluation_hash"
            ],
        )
        self.assertEqual(
            readout_shadow_application["surface"],
            "snn_language_plasticity_shadow_application.v1",
        )
        self.assertFalse(readout_shadow_application["mutates_runtime_state"])
        self.assertFalse(readout_shadow_application["applies_plasticity"])
        self.assertFalse(
            readout_shadow_application["promotion_gate"]["eligible_for_plasticity_application"]
        )
        self.assertFalse(
            readout_shadow_application["promotion_gate"]["eligible_for_live_application"]
        )
        self.assertEqual(
            readout_live_readiness["surface"],
            "snn_language_plasticity_live_application_readiness.v1",
        )
        self.assertFalse(readout_live_readiness["mutates_runtime_state"])
        self.assertFalse(readout_live_readiness["applies_plasticity"])
        self.assertTrue(
            readout_live_readiness["promotion_gate"][
                "eligible_for_operator_live_application_review"
            ]
        )
        self.assertFalse(
            readout_live_readiness["promotion_gate"]["eligible_for_live_application"]
        )
        self.assertEqual(
            readout_live_preflight["surface"],
            "snn_language_plasticity_live_application_preflight.v1",
        )
        self.assertFalse(readout_live_preflight["mutates_runtime_state"])
        self.assertFalse(readout_live_preflight["applies_plasticity"])
        self.assertEqual(
            readout_live_preflight["application_target"]["target_id"],
            "marulho.snn_language.sparse_transition_weights",
        )
        self.assertTrue(
            readout_live_preflight["promotion_gate"]["eligible_for_operator_execution_review"]
        )
        self.assertFalse(readout_live_preflight["promotion_gate"]["eligible_for_live_application"])
        self.assertFalse(blocked_readout_live_application["accepted"])
        self.assertFalse(
            blocked_readout_live_application["promotion_gate"]["required_evidence"]["confirmation"]
        )
        self.assertTrue(readout_live_application["accepted"])
        self.assertTrue(readout_live_application["applies_plasticity"])
        self.assertTrue(readout_live_application["mutates_runtime_state"])
        self.assertFalse(readout_live_application["generates_text"])
        self.assertFalse(readout_live_application["decodes_text"])
        self.assertFalse(readout_live_application["loads_external_checkpoint"])
        self.assertEqual(
            readout_live_application["before"]["state_revision"],
            status_before_readout_live_application["state_revision"],
        )
        self.assertGreater(
            readout_live_application["after"]["state_revision"],
            status_before_readout_live_application["state_revision"],
        )
        self.assertGreater(
            readout_live_application["application_target"]["applied_synapse_count"],
            0,
        )
        self.assertEqual(
            readout_live_application["applied_synapses"][0]["readout_evidence_hash"],
            readout_shadow_delta["bounded_synapses"][0]["readout_evidence_hash"],
        )
        self.assertEqual(
            readout_live_application["applied_synapses"][0]["prediction_hash"],
            readout_shadow_delta["bounded_synapses"][0]["prediction_hash"],
        )
        self.assertGreater(
            readout_plasticity_runtime_state["applied_update_count"],
            0,
        )
        applied_key = (
            f"{readout_live_application['applied_synapses'][0]['pre_index']}:"
            f"{readout_live_application['applied_synapses'][0]['post_index']}"
        )
        self.assertEqual(
            readout_plasticity_runtime_state["synapse_provenance_by_key"][applied_key][
                "readout_evidence_hash"
            ],
            readout_live_application["applied_synapses"][0]["readout_evidence_hash"],
        )
        self.assertEqual(
            readout_plasticity_runtime_state["recent_live_applications"][0]["applied_synapses"][0][
                "transition_memory_evaluation_hash"
            ],
            readout_live_application["applied_synapses"][0]["transition_memory_evaluation_hash"],
        )
        self.assertEqual(
            restored_readout_plasticity_runtime_state["synapse_provenance_by_key"][applied_key][
                "readout_evidence_hash"
            ],
            readout_live_application["applied_synapses"][0]["readout_evidence_hash"],
        )
        self.assertEqual(
            readout_synapse_audit["surface"],
            "snn_language_readout_synapse_provenance_audit.v1",
        )
        self.assertTrue(
            readout_synapse_audit["promotion_gate"][
                "eligible_for_readout_synapse_audit_review"
            ]
        )
        self.assertEqual(readout_synapse_audit["audit_summary"]["orphan_weight_count"], 0)
        self.assertTrue(readout_synapse_audit["audited_synapses"][0]["ledger_evidence_present"])
        self.assertTrue(readout_synapse_audit["audited_synapses"][0]["ledger_field_match"])
        self.assertTrue(readout_synapse_audit["audited_synapses"][0]["ledger_hash_valid"])
        self.assertTrue(readout_synapse_audit["audited_synapses"][0]["canonical_synapse_key"])
        self.assertTrue(readout_synapse_audit["audited_synapses"][0]["synapse_indices_in_range"])
        self.assertTrue(readout_synapse_audit["audited_synapses"][0]["weight_finite"])
        self.assertTrue(readout_synapse_audit["audited_synapses"][0]["weight_bounded"])
        self.assertTrue(readout_synapse_audit["audited_synapses"][0]["source_indices_match_synapse"])
        self.assertTrue(
            restored_readout_synapse_audit["promotion_gate"][
                "eligible_for_readout_synapse_audit_review"
            ]
        )
        self.assertEqual(restored_ledger["summary"]["event_count"], 1)
        self.assertEqual(restored_ledger["summary"]["rollout_event_count"], 1)
        self.assertEqual(
            restored_ledger["rollout_events"][0]["rollout_replay_evaluation_hash"],
            rollout_replay_evaluation["provenance_evidence"]["rollout_replay_evaluation_hash"],
        )
        self.assertEqual(restored_replay_priority["candidate_count"], 1)

    def test_readout_synapse_audit_blocks_incomplete_checkpoint_restore_halves(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            app = create_app(
                _build_checkpoint(root, test_case="service_api_readout_synapse_restore_halves"),
                trace_dir=root / "traces",
            )
            runtime = app.state.marulho_runtime
            manager = app.state.marulho_manager
            with TestClient(app) as client:
                record = runtime.snn_language_readout_evidence_ledger_record(
                    readout_draft={
                        "surface": "snn_language_readout_draft.v1",
                        "generation_scope": "bounded_grounded_readout_label_draft",
                        "freeform_language_generation": False,
                        "mutates_runtime_state": False,
                        "draft": {"labels": ["memory pressure"], "text": "memory pressure"},
                        "sparse_decode_evidence": {
                            "candidate_matches": [
                                {"label": "memory pressure", "grounded": True}
                            ]
                        },
                        "transition_memory_evaluation_evidence": {
                            "provenance_match": True,
                            "prediction_hash": "prediction-restore-audit",
                            "transition_memory_evaluation_hash": "evaluation-restore-audit",
                            "persistent_transition_weights_hash": "weights-restore-audit",
                        },
                        "promotion_gate": {
                            "eligible_for_bounded_readout_generation": True,
                            "eligible_for_cognition_substrate": False,
                        },
                    },
                    expected_state_revision=manager._runtime_state.state_revision,
                    operator_id="operator-restore-audit",
                    confirmation=True,
                )
                evidence_hash = record["recorded_event"]["readout_evidence_hash"]
                manager._snn_language_plasticity_state = {
                    "sparse_transition_weights": {"1:2": 0.03},
                    "synapse_provenance_by_key": {
                        "1:2": {
                            "readout_evidence_hash": evidence_hash,
                            "prediction_hash": "prediction-restore-audit",
                            "transition_memory_evaluation_hash": "evaluation-restore-audit",
                            "persistent_transition_weights_hash": "weights-restore-audit",
                            "source_pre_indices": [1],
                            "source_post_indices": [2],
                            "source_active_indices": [1, 2],
                        }
                    },
                }
                complete_checkpoint = runtime.save_checkpoint(str(root / "readout-complete.pt"))
                complete_audit = client.get(
                    "/terminus/snn-language-sequence/readout-ledger/synapse-provenance-audit"
                ).json()

                trainer, metadata = load_trainer_checkpoint(complete_checkpoint["path"])
                missing_ledger_metadata = deepcopy(metadata)
                missing_ledger_service_state = dict(missing_ledger_metadata.get("service_state") or {})
                missing_ledger_service_state.pop("snn_language_readout_ledger", None)
                missing_ledger_metadata["service_state"] = missing_ledger_service_state
                missing_ledger_path = save_trainer_checkpoint(
                    root / "readout-missing-ledger.pt",
                    trainer,
                    metadata=missing_ledger_metadata,
                )
                runtime.restore_checkpoint(str(missing_ledger_path))
                missing_ledger_audit = client.get(
                    "/terminus/snn-language-sequence/readout-ledger/synapse-provenance-audit"
                ).json()

                trainer, metadata = load_trainer_checkpoint(complete_checkpoint["path"])
                missing_provenance_metadata = deepcopy(metadata)
                missing_provenance_service_state = dict(
                    missing_provenance_metadata.get("service_state") or {}
                )
                plasticity_state = dict(
                    missing_provenance_service_state.get("snn_language_plasticity") or {}
                )
                plasticity_state.pop("synapse_provenance_by_key", None)
                missing_provenance_service_state["snn_language_plasticity"] = plasticity_state
                missing_provenance_metadata["service_state"] = missing_provenance_service_state
                missing_provenance_path = save_trainer_checkpoint(
                    root / "readout-missing-provenance.pt",
                    trainer,
                    metadata=missing_provenance_metadata,
                )
                runtime.restore_checkpoint(str(missing_provenance_path))
                missing_provenance_audit = client.get(
                    "/terminus/snn-language-sequence/readout-ledger/synapse-provenance-audit"
                ).json()

                trainer, metadata = load_trainer_checkpoint(complete_checkpoint["path"])
                mismatched_restore_metadata = deepcopy(metadata)
                mismatched_restore_service_state = dict(
                    mismatched_restore_metadata.get("service_state") or {}
                )
                mismatched_restore_service_state[
                    "snn_applied_replay_lineage_checkpoint_summary"
                ] = {
                    "surface": "snn_applied_replay_lineage_checkpoint_summary.v1",
                    "applied_replay_lineage_count": 0,
                    "lineage_material_hash": "tampered",
                }
                mismatched_restore_metadata["service_state"] = (
                    mismatched_restore_service_state
                )
                mismatched_restore_path = save_trainer_checkpoint(
                    root / "readout-mismatched-restore-lineage.pt",
                    trainer,
                    metadata=mismatched_restore_metadata,
                )
                runtime.restore_checkpoint(str(mismatched_restore_path))
                mismatched_restore_audit = client.get(
                    "/terminus/snn-language-sequence/readout-ledger/synapse-provenance-audit"
                ).json()
            manager.close()

        self.assertTrue(
            complete_audit["promotion_gate"]["eligible_for_readout_synapse_audit_review"]
        )
        self.assertFalse(
            missing_ledger_audit["promotion_gate"]["eligible_for_readout_synapse_audit_review"]
        )
        self.assertFalse(
            missing_ledger_audit["promotion_gate"]["required_evidence"][
                "audited_synapses_present_in_ledger"
            ]
        )
        self.assertFalse(
            missing_provenance_audit["promotion_gate"]["eligible_for_readout_synapse_audit_review"]
        )
        self.assertFalse(
            missing_provenance_audit["promotion_gate"]["required_evidence"][
                "synapse_provenance_available"
            ]
        )
        self.assertFalse(
            missing_provenance_audit["promotion_gate"]["required_evidence"][
                "no_unprovenanced_weights"
            ]
        )
        self.assertFalse(
            mismatched_restore_audit["promotion_gate"][
                "eligible_for_readout_synapse_audit_review"
            ]
        )
        self.assertTrue(
            mismatched_restore_audit["audit_summary"]["restore_validation_available"]
        )
        self.assertTrue(
            mismatched_restore_audit["audit_summary"]["restore_validation_blocks_audit"]
        )
        self.assertFalse(
            mismatched_restore_audit["promotion_gate"]["required_evidence"][
                "applied_replay_lineage_restore_validation_not_mismatched"
            ]
        )

    def test_evaluated_transition_memory_replay_artifact_uses_internal_readout_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            app = create_app(
                _build_checkpoint(root, test_case="service_api_evaluated_snn_replay_artifact"),
                trace_dir=root / "traces",
            )
            manager = app.state.marulho_manager
            with TestClient(app) as client:
                record_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/record",
                    json={
                        "readout_draft": {
                            "surface": "snn_language_readout_draft.v1",
                            "generation_scope": "bounded_grounded_readout_label_draft",
                            "freeform_language_generation": False,
                            "mutates_runtime_state": False,
                            "draft": {"labels": ["memory pressure"], "text": "memory pressure"},
                            "sparse_decode_evidence": {
                                "candidate_matches": [
                                    {"label": "memory pressure", "grounded": True}
                                ]
                            },
                            "transition_memory_evaluation_evidence": {
                                "provenance_match": True,
                                "prediction_hash": "prediction-evaluated-artifact",
                                "transition_memory_evaluation_hash": "evaluation-evaluated-artifact",
                                "persistent_transition_weights_hash": "weights-evaluated-artifact",
                            },
                            "promotion_gate": {
                                "eligible_for_bounded_readout_generation": True,
                                "eligible_for_cognition_substrate": False,
                            },
                        },
                        "expected_state_revision": manager._runtime_state.state_revision,
                        "operator_id": "operator-evaluated-artifact",
                        "confirmation": True,
                    },
                )
                context_response = client.post(
                    "/terminus/snn-language-sequence/replay-evaluation-context",
                    json={
                        "prediction_report": {
                            "surface": "snn_language_sequence_prediction_probe.v1",
                            "available": True,
                            "prediction": {"predicted_sparse_indices": [14]},
                        },
                        "observed_readout_slots": [
                            {"label": "memory pressure", "pressure_band": "medium", "grounded": True}
                        ],
                        "runtime_truth_delta": {"improved_or_stable": True},
                        "rollback_policy": {"available": True, "snapshot_id": "pre-evaluated-context"},
                    },
                )
                status_before_priority_response = client.get("/status")
                consolidation_priority_response = client.get(
                    "/terminus/snn-language-sequence/replay-consolidation-priority-queue",
                    params={"limit": 4},
                )
                artifact_recording_policy_response = client.get(
                    "/terminus/snn-language-sequence/replay-artifact-recording-policy",
                    params={"limit": 4, "min_priority_score": 60.0},
                )
                review_ticket_response = client.post(
                    "/terminus/snn-language-sequence/replay-artifact-recording-review-ticket",
                    json={
                        "limit": 4,
                        "min_priority_score": 60.0,
                        "operator_id": "operator-evaluated-artifact",
                        "confirmation": True,
                    },
                )
                status_after_priority_response = client.get("/status")
                proposal_response = client.post(
                    "/terminus/snn-language-sequence/transition-memory-replay-artifact/proposal",
                    json={
                        "replay_evaluation_context_id": context_response.json()[
                            "replay_evaluation_context_id"
                        ],
                    },
                )
                artifact_response = client.post(
                    "/terminus/snn-language-sequence/transition-memory-replay-artifact/evaluated-record",
                    json={
                        "replay_evaluation_context_id": context_response.json()[
                            "replay_evaluation_context_id"
                        ],
                        "review_ticket_id": review_ticket_response.json()["review_ticket_id"],
                        "operator_id": "operator-evaluated-artifact",
                        "confirmation": True,
                    },
                )
                spoofed_response = client.post(
                    "/terminus/snn-language-sequence/transition-memory-replay-artifact/evaluated-record",
                    json={
                        "replay_evaluation_context_id": "fabricated-context",
                        "review_ticket_id": "fabricated-ticket",
                        "operator_id": "operator-evaluated-artifact",
                        "confirmation": True,
                    },
                )
            manager.close()

        self.assertEqual(record_response.status_code, 200)
        self.assertEqual(context_response.status_code, 200)
        self.assertEqual(consolidation_priority_response.status_code, 200)
        self.assertEqual(artifact_recording_policy_response.status_code, 200)
        self.assertEqual(review_ticket_response.status_code, 200)
        self.assertEqual(proposal_response.status_code, 200)
        self.assertEqual(artifact_response.status_code, 200)
        self.assertEqual(spoofed_response.status_code, 400)
        proposal = proposal_response.json()
        artifact = artifact_response.json()
        context = context_response.json()
        consolidation_priority = consolidation_priority_response.json()
        artifact_recording_policy = artifact_recording_policy_response.json()
        review_ticket = review_ticket_response.json()
        self.assertEqual(context["surface"], "snn_replay_evaluation_context.v1")
        self.assertEqual(
            status_before_priority_response.json()["state_revision"],
            status_after_priority_response.json()["state_revision"],
        )
        self.assertEqual(consolidation_priority["surface"], "snn_replay_consolidation_priority_queue.v1")
        self.assertTrue(consolidation_priority["advisory"])
        self.assertFalse(consolidation_priority["mutates_runtime_state"])
        self.assertFalse(consolidation_priority["eligible_for_live_replay"])
        self.assertFalse(consolidation_priority["eligible_for_artifact_recording"])
        self.assertFalse(
            consolidation_priority["promotion_gate"]["eligible_for_artifact_recording"]
        )
        self.assertFalse(
            consolidation_priority["promotion_gate"]["eligible_for_live_replay"]
        )
        self.assertEqual(consolidation_priority["candidate_count"], 1)
        self.assertEqual(
            consolidation_priority["candidates"][0]["replay_evaluation_context_id"],
            context["replay_evaluation_context_id"],
        )
        self.assertEqual(
            artifact_recording_policy["surface"],
            "snn_replay_artifact_recording_policy_proposal.v1",
        )
        self.assertTrue(artifact_recording_policy["recommended"])
        self.assertTrue(artifact_recording_policy["advisory"])
        self.assertFalse(artifact_recording_policy["mutates_runtime_state"])
        self.assertFalse(artifact_recording_policy["eligible_for_artifact_recording"])
        self.assertFalse(
            artifact_recording_policy["promotion_gate"]["eligible_for_artifact_recording"]
        )
        self.assertEqual(
            artifact_recording_policy["recommended_review"]["replay_evaluation_context_id"],
            context["replay_evaluation_context_id"],
        )
        self.assertEqual(review_ticket["surface"], "snn_replay_artifact_recording_review_ticket.v1")
        self.assertEqual(
            review_ticket["replay_evaluation_context_id"],
            context["replay_evaluation_context_id"],
        )
        self.assertEqual(proposal["surface"], "snn_transition_memory_replay_artifact_proposal.v1")
        self.assertEqual(proposal["replay_evaluation_context_id"], context["replay_evaluation_context_id"])
        self.assertTrue(proposal["promotion_gate"]["eligible_for_operator_recording_review"])
        self.assertEqual(artifact["surface"], "snn_transition_memory_replay_artifact.v1")
        self.assertTrue(artifact["internal_ledger_backed"])
        self.assertEqual(artifact["replay_evaluation_context_id"], context["replay_evaluation_context_id"])
        self.assertEqual(artifact["review_ticket_id"], review_ticket["review_ticket_id"])
        self.assertEqual(
            artifact["artifact_proposal_surface"],
            "snn_transition_memory_replay_artifact_proposal.v1",
        )

    def test_rollout_regeneration_application_uses_replay_bound_mismatch_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            app = create_app(
                _build_checkpoint(root, test_case="service_api_rollout_replay_bound_regeneration"),
                trace_dir=root / "traces",
            )
            manager = app.state.marulho_manager
            adapter_design = {
                "locality_radius": 2,
                "initial_weight": 0.1,
                "max_new_synapses": 1,
                "mismatch_score": 0.0,
                "candidate_count": 1,
                "candidate_synapses": [
                    {
                        "pre_index": 1,
                        "post_index": 3,
                        "synapse": "1:3",
                        "initial_weight": 0.1,
                        "locality_distance": 2,
                    }
                ],
            }
            adapter_review_hash = "rollout-developmental-review-service-api"
            adapter_hash = manager._snn_language_readout_ledger._sha256_json(
                {
                    "rollout_developmental_plasticity_review_hash": adapter_review_hash,
                    "regeneration_design": adapter_design,
                }
            )
            rollout_adapter = {
                "artifact_kind": "terminus_snn_language_readout_rollout_regeneration_proposal_adapter",
                "surface": "snn_language_readout_rollout_regeneration_proposal_adapter.v1",
                "available": True,
                "owned_by_marulho": True,
                "generates_text": False,
                "applies_plasticity": False,
                "mutates_runtime_state": False,
                "issues_regeneration_permit": False,
                "executor_ready": False,
                "rollout_developmental_plasticity_review_hash": adapter_review_hash,
                "rollout_regeneration_proposal_adapter_hash": adapter_hash,
                "regeneration_design": adapter_design,
                "blocked_replay_evidence": {
                    "available": False,
                    "ready": False,
                    "permit_id": None,
                },
                "executor_bypass_evidence": {
                    "replay_controller_permit_required": True,
                    "checkpoint_executor_required": True,
                    "direct_executor_submission_expected_to_block": True,
                },
                "promotion_gate": {
                    "eligible_for_operator_rollout_regeneration_adapter_review": True,
                },
            }
            with TestClient(app) as client:
                record_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/record",
                    json={
                        "readout_draft": {
                            "surface": "snn_language_readout_draft.v1",
                            "generation_scope": "bounded_grounded_readout_label_draft",
                            "freeform_language_generation": False,
                            "mutates_runtime_state": False,
                            "draft": {"labels": ["memory pressure"], "text": "memory pressure"},
                            "sparse_decode_evidence": {
                                "candidate_matches": [
                                    {"label": "memory pressure", "grounded": True}
                                ]
                            },
                            "transition_memory_evaluation_evidence": {
                                "provenance_match": True,
                                "prediction_hash": "prediction-rollout-regeneration",
                                "transition_memory_evaluation_hash": "evaluation-rollout-regeneration",
                                "persistent_transition_weights_hash": "weights-rollout-regeneration",
                            },
                            "promotion_gate": {
                                "eligible_for_bounded_readout_generation": True,
                                "eligible_for_cognition_substrate": False,
                            },
                        },
                        "expected_state_revision": manager._runtime_state.state_revision,
                        "operator_id": "operator-rollout-regeneration",
                        "confirmation": True,
                    },
                )
                context_response = client.post(
                    "/terminus/snn-language-sequence/replay-evaluation-context",
                    json={
                        "prediction_report": {
                            "surface": "snn_language_sequence_prediction_probe.v1",
                            "available": True,
                            "prediction": {"predicted_sparse_indices": [14]},
                        },
                        "observed_readout_slots": [
                            {"label": "memory pressure", "pressure_band": "medium", "grounded": True}
                        ],
                        "runtime_truth_delta": {"improved_or_stable": True},
                        "rollback_policy": {
                            "available": True,
                            "snapshot_id": "pre-rollout-regeneration-context",
                        },
                    },
                )
                review_ticket_response = client.post(
                    "/terminus/snn-language-sequence/replay-artifact-recording-review-ticket",
                    json={
                        "limit": 4,
                        "min_priority_score": 60.0,
                        "operator_id": "operator-rollout-regeneration",
                        "confirmation": True,
                    },
                )
                artifact_response = client.post(
                    "/terminus/snn-language-sequence/transition-memory-replay-artifact/evaluated-record",
                    json={
                        "replay_evaluation_context_id": context_response.json()[
                            "replay_evaluation_context_id"
                        ],
                        "review_ticket_id": review_ticket_response.json()["review_ticket_id"],
                        "operator_id": "operator-rollout-regeneration",
                        "confirmation": True,
                    },
                )
                review_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/rollout-regeneration-replay-artifact-review",
                    json={
                        "rollout_regeneration_proposal_adapter": rollout_adapter,
                        "snn_transition_memory_replay_artifact": artifact_response.json(),
                    },
                )
                permit_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/rollout-regeneration-permit-request",
                    json={
                        "rollout_regeneration_replay_artifact_review": review_response.json(),
                        "operator_id": "operator-rollout-regeneration",
                        "confirmation": True,
                    },
                )
                status_before_application_response = client.get("/status")
                preflight_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/rollout-regeneration-application-preflight",
                    json={
                        "rollout_regeneration_permit_request": permit_response.json(),
                        "expected_state_revision": status_before_application_response.json()[
                            "state_revision"
                        ],
                        "checkpoint_path": str(root / "rollout_replay_bound_regeneration.pt"),
                    },
                )
                application_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/rollout-regeneration-application",
                    json={
                        "rollout_regeneration_application_preflight": preflight_response.json(),
                        "expected_state_revision": status_before_application_response.json()[
                            "state_revision"
                        ],
                        "operator_id": "operator-rollout-regeneration",
                        "confirmation": True,
                        "checkpoint_path": str(root / "rollout_replay_bound_regeneration.pt"),
                    },
                )
                runtime_state_response = client.get(
                    "/terminus/snn-language-sequence/plasticity-runtime-state"
                )
                sleep_policy_response = client.post(
                    "/terminus/snn-language-sequence/plasticity-sleep-policy",
                    json={
                        "transition_memory_state": runtime_state_response.json(),
                        "subcortex_sleep_pressure": {
                            "pressure": 0.7,
                            "source": "service_api_rollout_regeneration",
                        },
                        "rollout_regeneration_evidence": application_response.json(),
                    },
                )
                status_before_sleep_ticket_response = client.get("/status")
                sleep_ticket_response = client.post(
                    "/terminus/snn-language-sequence/plasticity-sleep-policy/review-ticket",
                    json={
                        "sleep_policy": sleep_policy_response.json(),
                        "operator_id": "operator-sleep-policy",
                        "confirmation": True,
                    },
                )
                sleep_ticket_queue_response = client.get(
                    "/terminus/snn-language-sequence/plasticity-sleep-policy/review-tickets",
                    params={"limit": 4},
                )
                sleep_autonomy_proposal_response = client.get(
                    "/terminus/snn-language-sequence/plasticity-sleep-policy/autonomy-proposal",
                    params={"limit": 4},
                )
                sleep_scheduler_experiment_response = client.get(
                    "/terminus/snn-language-sequence/plasticity-sleep-policy/scheduler-experiment",
                    params={"limit": 4, "cycles": 3},
                )
                sleep_scheduler_design_response = client.get(
                    "/terminus/snn-language-sequence/plasticity-sleep-policy/scheduler-design",
                    params={
                        "limit": 4,
                        "cycles": 3,
                        "min_stable_cycles": 3,
                        "max_review_interval_seconds": 120.0,
                    },
                )
                sleep_scheduler_design_review_ticket_response = client.post(
                    "/terminus/snn-language-sequence/plasticity-sleep-policy/scheduler-design/review-ticket",
                    json={
                        "limit": 4,
                        "cycles": 3,
                        "min_stable_cycles": 3,
                        "max_review_interval_seconds": 120.0,
                        "expected_state_revision": sleep_ticket_response.json()[
                            "recorded_state_revision"
                        ],
                        "scheduler_design_hash": sleep_scheduler_design_response.json()[
                            "provenance_evidence"
                        ]["scheduler_design_hash"],
                        "operator_id": "operator-scheduler-design",
                        "confirmation": True,
                    },
                )
                sleep_scheduler_design_review_ticket_queue_response = client.get(
                    "/terminus/snn-language-sequence/plasticity-sleep-policy/scheduler-design/review-tickets",
                    params={"limit": 4},
                )
                sleep_scheduler_installation_autonomy_proposal_response = client.get(
                    "/terminus/snn-language-sequence/plasticity-sleep-policy/"
                    "scheduler-installation-autonomy-proposal",
                    params={"limit": 4},
                )
                sleep_scheduler_installation_preflight_response = client.get(
                    "/terminus/snn-language-sequence/plasticity-sleep-policy/"
                    "scheduler-installation-preflight",
                    params={"limit": 4},
                )
                sleep_review_scheduler_installation_response = client.post(
                    "/terminus/snn-language-sequence/plasticity-sleep-policy/"
                    "review-scheduler/install",
                    json={
                        "limit": 4,
                        "expected_state_revision": sleep_ticket_response.json()[
                            "recorded_state_revision"
                        ],
                        "scheduler_installation_preflight_hash": (
                            sleep_scheduler_installation_preflight_response.json()[
                                "provenance_evidence"
                            ]["scheduler_installation_preflight_hash"]
                        ),
                        "operator_id": "operator-review-scheduler",
                        "confirmation": True,
                    },
                )
                sleep_review_scheduler_runtime_response = client.get(
                    "/terminus/snn-language-sequence/plasticity-sleep-policy/"
                    "review-scheduler"
                )
                sleep_review_scheduler_cycle_inspection_response = client.get(
                    "/terminus/snn-language-sequence/plasticity-sleep-policy/"
                    "review-scheduler/cycle-inspection"
                )
                blocked_sleep_review_scheduler_cycle_acknowledgment_response = (
                    client.post(
                        "/terminus/snn-language-sequence/plasticity-sleep-policy/"
                        "review-scheduler/cycle-acknowledgment",
                        json={
                            "expected_state_revision": sleep_ticket_response.json()[
                                "recorded_state_revision"
                            ],
                            "scheduler_installation_id": (
                                sleep_review_scheduler_installation_response.json()[
                                    "scheduler_installation_id"
                                ]
                            ),
                            "scheduler_installation_evidence_hash": (
                                sleep_review_scheduler_installation_response.json()[
                                    "evidence_hash"
                                ]
                            ),
                            "review_ticket_id": "not-due-cycle-ticket",
                            "operator_id": "operator-review-scheduler-cycle",
                            "confirmation": True,
                        },
                    )
                )
                blocked_sleep_review_scheduler_cycle_acknowledgment_preflight_response = (
                    client.get(
                        "/terminus/snn-language-sequence/plasticity-sleep-policy/"
                        "review-scheduler/cycle-acknowledgment-preflight",
                        params={
                            "scheduler_installation_id": (
                                sleep_review_scheduler_installation_response.json()[
                                    "scheduler_installation_id"
                                ]
                            ),
                            "scheduler_installation_evidence_hash": (
                                sleep_review_scheduler_installation_response.json()[
                                    "evidence_hash"
                                ]
                            ),
                            "review_ticket_id": "not-due-cycle-ticket",
                        },
                    )
                )
                sleep_review_scheduler_cycle_autonomy_proposal_response = client.get(
                    "/terminus/snn-language-sequence/plasticity-sleep-policy/"
                    "review-scheduler/cycle-autonomy-proposal"
                )
                sleep_replay_selection_proposal_response = client.get(
                    "/terminus/snn-language-sequence/plasticity-sleep-policy/"
                    "review-scheduler/due-cycle-bounded-replay-selection-proposal",
                    params={"limit": 4, "max_candidates": 1},
                )
                blocked_sleep_replay_selection_excess_candidates_response = client.get(
                    "/terminus/snn-language-sequence/plasticity-sleep-policy/"
                    "review-scheduler/due-cycle-bounded-replay-selection-proposal",
                    params={"limit": 4, "max_candidates": 9},
                )
                due_cycle_recording_review_proposal_response = client.get(
                    "/terminus/snn-language-sequence/plasticity-sleep-policy/"
                    "review-scheduler/"
                    "due-cycle-replay-artifact-recording-review-proposal",
                    params={
                        "limit": 4,
                        "max_candidates": 1,
                        "min_priority_score": 60.0,
                    },
                )
                blocked_due_cycle_recording_review_ticket_response = client.post(
                    "/terminus/snn-language-sequence/plasticity-sleep-policy/"
                    "review-scheduler/"
                    "due-cycle-replay-artifact-recording-review-ticket",
                    json={
                        "limit": 4,
                        "max_candidates": 1,
                        "min_priority_score": 60.0,
                        "operator_id": "operator-due-cycle-review",
                        "confirmation": True,
                    },
                )
                sleep_phase_separation_response = client.get(
                    "/terminus/snn-language-sequence/plasticity-sleep-policy/"
                    "review-scheduler/sleep-phase-separation-proposal",
                    params={"limit": 4, "max_candidates": 1},
                )
                rem_homeostatic_preflight_response = client.get(
                    "/terminus/snn-language-sequence/plasticity-sleep-policy/"
                    "review-scheduler/rem-like-homeostatic-stabilization-preflight",
                    params={"limit": 4, "max_candidates": 1},
                )
                blocked_rem_homeostatic_preflight_policy_response = client.get(
                    "/terminus/snn-language-sequence/plasticity-sleep-policy/"
                    "review-scheduler/rem-like-homeostatic-stabilization-preflight",
                    params={
                        "limit": 4,
                        "max_candidates": 1,
                        "decay_factor": 1.2,
                    },
                )
                blocked_sleep_scheduler_zero_cycles_response = client.get(
                    "/terminus/snn-language-sequence/plasticity-sleep-policy/scheduler-experiment",
                    params={"limit": 4, "cycles": 0},
                )
                blocked_sleep_scheduler_excess_cycles_response = client.get(
                    "/terminus/snn-language-sequence/plasticity-sleep-policy/scheduler-experiment",
                    params={"limit": 4, "cycles": 17},
                )
                blocked_sleep_scheduler_zero_limit_response = client.get(
                    "/terminus/snn-language-sequence/plasticity-sleep-policy/scheduler-experiment",
                    params={"limit": 0, "cycles": 3},
                )
                blocked_sleep_scheduler_design_zero_stable_cycles_response = client.get(
                    "/terminus/snn-language-sequence/plasticity-sleep-policy/scheduler-design",
                    params={"limit": 4, "cycles": 3, "min_stable_cycles": 0},
                )
                blocked_sleep_scheduler_design_weak_cycles_response = client.get(
                    "/terminus/snn-language-sequence/plasticity-sleep-policy/scheduler-design",
                    params={"limit": 4, "cycles": 2, "min_stable_cycles": 3},
                )
                blocked_sleep_scheduler_design_stable_cycles_above_cycles_response = client.get(
                    "/terminus/snn-language-sequence/plasticity-sleep-policy/scheduler-design",
                    params={"limit": 4, "cycles": 3, "min_stable_cycles": 4},
                )
                blocked_sleep_scheduler_design_short_interval_response = client.get(
                    "/terminus/snn-language-sequence/plasticity-sleep-policy/scheduler-design",
                    params={
                        "limit": 4,
                        "cycles": 3,
                        "max_review_interval_seconds": 59.0,
                    },
                )
                blocked_sleep_scheduler_design_excess_interval_response = client.get(
                    "/terminus/snn-language-sequence/plasticity-sleep-policy/scheduler-design",
                    params={
                        "limit": 4,
                        "cycles": 3,
                        "max_review_interval_seconds": 3601.0,
                    },
                )
                living_loop_after_sleep_ticket_response = client.get("/terminus/living-loop")
                policy_after_sleep_ticket_response = client.get("/terminus/policy-actuator")
                terminus_after_sleep_ticket_response = client.get("/terminus")
                status_after_sleep_ticket_response = client.get("/status")
            manager.close()

        self.assertEqual(record_response.status_code, 200)
        self.assertEqual(context_response.status_code, 200)
        self.assertEqual(review_ticket_response.status_code, 200)
        self.assertEqual(artifact_response.status_code, 200)
        self.assertEqual(review_response.status_code, 200)
        self.assertEqual(permit_response.status_code, 200)
        self.assertEqual(status_before_application_response.status_code, 200)
        self.assertEqual(preflight_response.status_code, 200)
        self.assertEqual(application_response.status_code, 200)
        self.assertEqual(runtime_state_response.status_code, 200)
        self.assertEqual(sleep_policy_response.status_code, 200)
        self.assertEqual(status_before_sleep_ticket_response.status_code, 200)
        self.assertEqual(sleep_ticket_response.status_code, 200)
        self.assertEqual(sleep_ticket_queue_response.status_code, 200)
        self.assertEqual(sleep_autonomy_proposal_response.status_code, 200)
        self.assertEqual(sleep_scheduler_experiment_response.status_code, 200)
        self.assertEqual(sleep_scheduler_design_response.status_code, 200)
        self.assertEqual(sleep_scheduler_design_review_ticket_response.status_code, 200)
        self.assertEqual(
            sleep_scheduler_design_review_ticket_queue_response.status_code,
            200,
        )
        self.assertEqual(
            sleep_scheduler_installation_autonomy_proposal_response.status_code,
            200,
        )
        self.assertEqual(sleep_scheduler_installation_preflight_response.status_code, 200)
        self.assertEqual(sleep_review_scheduler_installation_response.status_code, 200)
        self.assertEqual(sleep_review_scheduler_runtime_response.status_code, 200)
        self.assertEqual(
            sleep_review_scheduler_cycle_inspection_response.status_code,
            200,
        )
        self.assertEqual(
            blocked_sleep_review_scheduler_cycle_acknowledgment_response.status_code,
            400,
        )
        self.assertEqual(
            blocked_sleep_review_scheduler_cycle_acknowledgment_preflight_response.status_code,
            200,
        )
        self.assertFalse(
            blocked_sleep_review_scheduler_cycle_acknowledgment_preflight_response.json()[
                "ready"
            ]
        )
        self.assertEqual(
            sleep_review_scheduler_cycle_autonomy_proposal_response.status_code,
            200,
        )
        self.assertEqual(sleep_replay_selection_proposal_response.status_code, 200)
        self.assertEqual(
            blocked_sleep_replay_selection_excess_candidates_response.status_code,
            422,
        )
        self.assertEqual(due_cycle_recording_review_proposal_response.status_code, 200)
        self.assertEqual(
            blocked_due_cycle_recording_review_ticket_response.status_code,
            400,
        )
        self.assertEqual(sleep_phase_separation_response.status_code, 200)
        self.assertEqual(rem_homeostatic_preflight_response.status_code, 200)
        self.assertEqual(
            blocked_rem_homeostatic_preflight_policy_response.status_code,
            422,
        )
        self.assertEqual(blocked_sleep_scheduler_zero_cycles_response.status_code, 422)
        self.assertEqual(blocked_sleep_scheduler_excess_cycles_response.status_code, 422)
        self.assertEqual(blocked_sleep_scheduler_zero_limit_response.status_code, 422)
        self.assertEqual(
            blocked_sleep_scheduler_design_zero_stable_cycles_response.status_code,
            422,
        )
        self.assertEqual(blocked_sleep_scheduler_design_weak_cycles_response.status_code, 422)
        self.assertEqual(
            blocked_sleep_scheduler_design_stable_cycles_above_cycles_response.status_code,
            422,
        )
        self.assertEqual(
            blocked_sleep_scheduler_design_short_interval_response.status_code,
            422,
        )
        self.assertEqual(
            blocked_sleep_scheduler_design_excess_interval_response.status_code,
            422,
        )
        self.assertEqual(living_loop_after_sleep_ticket_response.status_code, 200)
        self.assertEqual(policy_after_sleep_ticket_response.status_code, 200)
        self.assertEqual(terminus_after_sleep_ticket_response.status_code, 200)
        self.assertEqual(status_after_sleep_ticket_response.status_code, 200)
        artifact = artifact_response.json()
        review = review_response.json()
        permit = permit_response.json()
        preflight = preflight_response.json()
        application = application_response.json()
        runtime_state = runtime_state_response.json()
        sleep_policy = sleep_policy_response.json()
        sleep_ticket = sleep_ticket_response.json()
        sleep_ticket_queue = sleep_ticket_queue_response.json()
        sleep_autonomy_proposal = sleep_autonomy_proposal_response.json()
        sleep_scheduler_experiment = sleep_scheduler_experiment_response.json()
        sleep_scheduler_design = sleep_scheduler_design_response.json()
        sleep_scheduler_design_review_ticket = (
            sleep_scheduler_design_review_ticket_response.json()
        )
        sleep_scheduler_design_review_ticket_queue = (
            sleep_scheduler_design_review_ticket_queue_response.json()
        )
        sleep_scheduler_installation_autonomy_proposal = (
            sleep_scheduler_installation_autonomy_proposal_response.json()
        )
        sleep_scheduler_installation_preflight = (
            sleep_scheduler_installation_preflight_response.json()
        )
        sleep_review_scheduler_installation = (
            sleep_review_scheduler_installation_response.json()
        )
        sleep_review_scheduler_runtime = sleep_review_scheduler_runtime_response.json()
        sleep_review_scheduler_cycle_inspection = (
            sleep_review_scheduler_cycle_inspection_response.json()
        )
        sleep_review_scheduler_cycle_autonomy_proposal = (
            sleep_review_scheduler_cycle_autonomy_proposal_response.json()
        )
        sleep_replay_selection_proposal = sleep_replay_selection_proposal_response.json()
        due_cycle_recording_review_proposal = (
            due_cycle_recording_review_proposal_response.json()
        )
        sleep_phase_separation = sleep_phase_separation_response.json()
        rem_homeostatic_preflight = rem_homeostatic_preflight_response.json()
        living_loop_after_sleep_ticket = living_loop_after_sleep_ticket_response.json()
        policy_after_sleep_ticket = policy_after_sleep_ticket_response.json()
        terminus_after_sleep_ticket = terminus_after_sleep_ticket_response.json()
        status_after_sleep_ticket = status_after_sleep_ticket_response.json()
        self.assertEqual(artifact["surface"], "snn_transition_memory_replay_artifact.v1")
        self.assertGreaterEqual(artifact["mismatch_score"], 0.66)
        self.assertEqual(review["surface"], "snn_language_readout_rollout_regeneration_replay_artifact_review.v1")
        self.assertEqual(review["regeneration_design"]["mismatch_score"], artifact["mismatch_score"])
        self.assertTrue(review["promotion_gate"]["required_evidence"]["replay_mismatch_score_high"])
        self.assertTrue(review["promotion_gate"]["eligible_for_regeneration_permit_request"])
        self.assertTrue(permit["accepted"])
        self.assertEqual(permit["regeneration_design"]["mismatch_score"], artifact["mismatch_score"])
        self.assertTrue(preflight["ready"])
        self.assertTrue(preflight["regeneration_proposal"]["ready"])
        self.assertTrue(application["accepted"])
        self.assertTrue(application["executor_called"])
        self.assertTrue(application["writes_checkpoint"])
        self.assertTrue(application["applies_plasticity"])
        self.assertTrue(application["mutates_runtime_state"])
        self.assertEqual(
            application["executor_result"]["surface"],
            "snn_language_transition_memory_regeneration.v1",
        )
        self.assertEqual(runtime_state["regeneration_count"], 1)
        self.assertEqual(runtime_state["regenerated_synapse_count_total"], 1)
        self.assertIn("1:3", runtime_state["sparse_transition_weights"])
        self.assertEqual(
            sleep_policy["recommendation"]["action"],
            "review_transition_memory_homeostatic_maintenance",
        )
        self.assertIn(
            "post_growth_homeostatic_maintenance_due",
            sleep_policy["recommendation"]["reason_codes"],
        )
        self.assertFalse(sleep_policy["mutates_runtime_state"])
        self.assertEqual(sleep_ticket["surface"], "snn_sleep_plasticity_review_ticket.v1")
        self.assertEqual(
            sleep_ticket["recommended_action"],
            sleep_policy["recommendation"]["action"],
        )
        self.assertEqual(
            sleep_ticket["suggested_endpoint"],
            sleep_policy["recommendation"]["suggested_endpoint"],
        )
        self.assertFalse(sleep_ticket["executable"])
        self.assertFalse(sleep_ticket["mutates_runtime_state"])
        self.assertFalse(sleep_ticket["applies_plasticity"])
        self.assertFalse(sleep_ticket["writes_checkpoint"])
        self.assertEqual(
            sleep_ticket_queue["surface"],
            "snn_sleep_plasticity_review_ticket_queue.v1",
        )
        self.assertEqual(sleep_ticket_queue["verified_count"], 1)
        self.assertEqual(sleep_ticket_queue["stale_count"], 0)
        self.assertEqual(
            sleep_ticket_queue["latest_verified_ticket"]["review_ticket_id"],
            sleep_ticket["review_ticket_id"],
        )
        self.assertEqual(
            sleep_ticket_queue["next_gate"],
            sleep_policy["recommendation"]["suggested_endpoint"],
        )
        self.assertFalse(sleep_ticket_queue["executable"])
        self.assertFalse(sleep_ticket_queue["mutates_runtime_state"])
        self.assertEqual(
            sleep_autonomy_proposal["surface"],
            "snn_sleep_plasticity_autonomy_proposal.v1",
        )
        self.assertTrue(sleep_autonomy_proposal["ready"])
        self.assertEqual(
            sleep_autonomy_proposal["candidate"]["review_ticket_id"],
            sleep_ticket["review_ticket_id"],
        )
        self.assertEqual(
            sleep_autonomy_proposal["promotion_gate"]["next_gate"],
            sleep_policy["recommendation"]["suggested_endpoint"],
        )
        self.assertTrue(
            sleep_autonomy_proposal["promotion_gate"]["eligible_for_autonomy_planning"]
        )
        self.assertFalse(sleep_autonomy_proposal["promotion_gate"]["eligible_for_action"])
        self.assertFalse(
            sleep_autonomy_proposal["promotion_gate"]["eligible_for_structural_write"]
        )
        self.assertFalse(sleep_autonomy_proposal["executable"])
        self.assertFalse(sleep_autonomy_proposal["mutates_runtime_state"])
        self.assertFalse(sleep_autonomy_proposal["applies_plasticity"])
        self.assertEqual(
            sleep_scheduler_experiment["surface"],
            "snn_sleep_plasticity_scheduler_experiment.v1",
        )
        self.assertTrue(sleep_scheduler_experiment["ready"])
        self.assertTrue(sleep_scheduler_experiment["isolated"])
        self.assertEqual(sleep_scheduler_experiment["experiment_summary"]["cycle_count"], 3)
        self.assertEqual(
            sleep_scheduler_experiment["experiment_summary"]["stable_cycle_count"],
            3,
        )
        self.assertTrue(
            sleep_scheduler_experiment["experiment_summary"]["proposal_stable"]
        )
        self.assertEqual(
            sleep_scheduler_experiment["experiment_summary"]["review_ticket_id"],
            sleep_ticket["review_ticket_id"],
        )
        self.assertFalse(
            sleep_scheduler_experiment["device_evidence"]["tensor_execution_required"]
        )
        self.assertFalse(sleep_scheduler_experiment["device_evidence"]["cuda_applicable"])
        self.assertFalse(sleep_scheduler_experiment["external_dependency"])
        self.assertFalse(sleep_scheduler_experiment["loads_external_checkpoint"])
        self.assertFalse(sleep_scheduler_experiment["returns_trained_weights"])
        self.assertFalse(sleep_scheduler_experiment["eligible_for_plasticity"])
        self.assertTrue(
            sleep_scheduler_experiment["provenance_evidence"]["scheduler_experiment_hash"]
        )
        self.assertFalse(sleep_scheduler_experiment["executes_suggested_endpoint"])
        self.assertFalse(
            sleep_scheduler_experiment["ephemeral_experiment"]["scheduler_installed"]
        )
        self.assertFalse(
            sleep_scheduler_experiment["ephemeral_experiment"]["suggested_endpoint_called"]
        )
        self.assertTrue(
            sleep_scheduler_experiment["promotion_gate"][
                "eligible_for_operator_scheduler_design_review"
            ]
        )
        self.assertFalse(
            sleep_scheduler_experiment["promotion_gate"][
                "eligible_for_scheduler_installation"
            ]
        )
        self.assertFalse(sleep_scheduler_experiment["promotion_gate"]["eligible_for_action"])
        self.assertFalse(
            sleep_scheduler_experiment["promotion_gate"]["eligible_for_structural_write"]
        )
        self.assertEqual(
            sleep_scheduler_design["surface"],
            "snn_sleep_plasticity_scheduler_design.v1",
        )
        self.assertTrue(sleep_scheduler_design["ready"])
        self.assertTrue(sleep_scheduler_design["isolated"])
        self.assertEqual(
            sleep_scheduler_design["scheduler_design"]["scheduler_mode"],
            "operator_review_only",
        )
        self.assertEqual(
            sleep_scheduler_design["scheduler_design"]["review_ticket_id"],
            sleep_ticket["review_ticket_id"],
        )
        self.assertEqual(
            sleep_scheduler_design["scheduler_design"]["bound_state_revision"],
            sleep_ticket["recorded_state_revision"],
        )
        self.assertEqual(
            sleep_scheduler_design["scheduler_design"]["source_scheduler_experiment_hash"],
            sleep_scheduler_experiment["provenance_evidence"]["scheduler_experiment_hash"],
        )
        self.assertEqual(sleep_scheduler_design["scheduler_design"]["min_stable_cycles"], 3)
        self.assertEqual(
            sleep_scheduler_design["scheduler_design"]["max_review_interval_seconds"],
            120.0,
        )
        self.assertFalse(
            sleep_scheduler_design["scheduler_design"]["automatic_endpoint_execution"]
        )
        self.assertFalse(sleep_scheduler_design["scheduler_design"]["automatic_plasticity"])
        self.assertFalse(sleep_scheduler_design["installs_scheduler"])
        self.assertFalse(sleep_scheduler_design["executes_suggested_endpoint"])
        self.assertFalse(sleep_scheduler_design["writes_checkpoint"])
        self.assertFalse(sleep_scheduler_design["applies_plasticity"])
        self.assertFalse(sleep_scheduler_design["mutates_runtime_state"])
        self.assertFalse(sleep_scheduler_design["eligible_for_plasticity"])
        self.assertFalse(
            sleep_scheduler_design["safety_contract"]["scheduler_installation_allowed"]
        )
        self.assertFalse(
            sleep_scheduler_design["safety_contract"]["suggested_endpoint_execution_allowed"]
        )
        self.assertFalse(
            sleep_scheduler_design["safety_contract"]["runtime_mutation_allowed"]
        )
        self.assertFalse(
            sleep_scheduler_design["promotion_gate"]["eligible_for_scheduler_installation"]
        )
        self.assertEqual(
            sleep_scheduler_design_review_ticket["surface"],
            "snn_sleep_plasticity_scheduler_design_review_ticket.v1",
        )
        self.assertEqual(
            sleep_scheduler_design_review_ticket["scheduler_design_hash"],
            sleep_scheduler_design["provenance_evidence"]["scheduler_design_hash"],
        )
        self.assertFalse(sleep_scheduler_design_review_ticket["installs_scheduler"])
        self.assertFalse(
            sleep_scheduler_design_review_ticket["executes_suggested_endpoint"]
        )
        self.assertFalse(sleep_scheduler_design_review_ticket["records_replay_artifact"])
        self.assertFalse(
            sleep_scheduler_design_review_ticket["issues_regeneration_permit"]
        )
        self.assertFalse(sleep_scheduler_design_review_ticket["writes_checkpoint"])
        self.assertFalse(sleep_scheduler_design_review_ticket["applies_plasticity"])
        self.assertFalse(
            sleep_scheduler_design_review_ticket["mutates_transition_memory"]
        )
        self.assertFalse(sleep_scheduler_design_review_ticket["mutates_runtime_state"])
        self.assertEqual(
            sleep_scheduler_design_review_ticket_queue["surface"],
            "snn_sleep_plasticity_scheduler_design_review_ticket_queue.v1",
        )
        self.assertEqual(sleep_scheduler_design_review_ticket_queue["verified_count"], 1)
        self.assertEqual(
            sleep_scheduler_design_review_ticket_queue["latest_verified_ticket"][
                "scheduler_design_review_ticket_id"
            ],
            sleep_scheduler_design_review_ticket["scheduler_design_review_ticket_id"],
        )
        self.assertFalse(sleep_scheduler_design_review_ticket_queue["installs_scheduler"])
        self.assertFalse(
            sleep_scheduler_design_review_ticket_queue["executes_suggested_endpoint"]
        )
        self.assertFalse(
            sleep_scheduler_design_review_ticket_queue["mutates_runtime_state"]
        )
        self.assertEqual(
            sleep_scheduler_installation_autonomy_proposal["surface"],
            "snn_sleep_plasticity_scheduler_installation_autonomy_proposal.v1",
        )
        self.assertTrue(sleep_scheduler_installation_autonomy_proposal["ready"])
        self.assertFalse(
            sleep_scheduler_installation_autonomy_proposal["installs_scheduler"]
        )
        self.assertFalse(
            sleep_scheduler_installation_autonomy_proposal["registers_timer"]
        )
        self.assertFalse(
            sleep_scheduler_installation_autonomy_proposal["starts_background_worker"]
        )
        self.assertFalse(
            sleep_scheduler_installation_autonomy_proposal["executes_suggested_endpoint"]
        )
        self.assertEqual(
            sleep_scheduler_installation_preflight["surface"],
            "snn_sleep_plasticity_scheduler_installation_preflight.v1",
        )
        self.assertTrue(sleep_scheduler_installation_preflight["ready"])
        self.assertFalse(sleep_scheduler_installation_preflight["installs_scheduler"])
        self.assertFalse(sleep_scheduler_installation_preflight["registers_timer"])
        self.assertFalse(
            sleep_scheduler_installation_preflight["starts_background_worker"]
        )
        self.assertFalse(
            sleep_scheduler_installation_preflight["executes_suggested_endpoint"]
        )
        self.assertFalse(sleep_scheduler_installation_preflight["writes_checkpoint"])
        self.assertFalse(sleep_scheduler_installation_preflight["applies_plasticity"])
        self.assertFalse(sleep_scheduler_installation_preflight["mutates_runtime_state"])
        self.assertEqual(
            sleep_review_scheduler_installation["surface"],
            "snn_sleep_plasticity_review_scheduler_installation.v1",
        )
        self.assertTrue(sleep_review_scheduler_installation["scheduler_installed"])
        self.assertEqual(
            sleep_review_scheduler_installation["scheduler_mode"],
            "review_only",
        )
        self.assertFalse(sleep_review_scheduler_installation["registers_os_timer"])
        self.assertFalse(sleep_review_scheduler_installation["starts_background_worker"])
        self.assertFalse(
            sleep_review_scheduler_installation["executes_suggested_endpoint"]
        )
        self.assertFalse(sleep_review_scheduler_installation["writes_checkpoint"])
        self.assertFalse(sleep_review_scheduler_installation["applies_plasticity"])
        self.assertFalse(sleep_review_scheduler_installation["mutates_runtime_state"])
        self.assertEqual(
            sleep_review_scheduler_runtime["surface"],
            "snn_sleep_plasticity_review_scheduler_runtime.v1",
        )
        self.assertTrue(sleep_review_scheduler_runtime["scheduler_installed"])
        self.assertFalse(sleep_review_scheduler_runtime["review_due"])
        self.assertFalse(sleep_review_scheduler_runtime["registers_os_timer"])
        self.assertFalse(sleep_review_scheduler_runtime["starts_background_worker"])
        self.assertFalse(sleep_review_scheduler_runtime["executes_suggested_endpoint"])
        self.assertFalse(sleep_review_scheduler_runtime["applies_plasticity"])
        self.assertFalse(sleep_review_scheduler_runtime["mutates_runtime_state"])
        self.assertEqual(
            sleep_review_scheduler_cycle_inspection["surface"],
            "snn_sleep_plasticity_review_scheduler_cycle_inspection.v1",
        )
        self.assertFalse(sleep_review_scheduler_cycle_inspection["ready"])
        self.assertFalse(
            sleep_review_scheduler_cycle_inspection["cycle_inspection"]["review_due"]
        )
        self.assertFalse(
            sleep_review_scheduler_cycle_inspection["executes_suggested_endpoint"]
        )
        self.assertFalse(sleep_review_scheduler_cycle_inspection["records_replay_artifact"])
        self.assertFalse(sleep_review_scheduler_cycle_inspection["applies_plasticity"])
        self.assertFalse(sleep_review_scheduler_cycle_inspection["mutates_runtime_state"])
        self.assertEqual(
            sleep_review_scheduler_cycle_autonomy_proposal["surface"],
            "snn_sleep_plasticity_review_scheduler_cycle_autonomy_proposal.v1",
        )
        self.assertFalse(sleep_review_scheduler_cycle_autonomy_proposal["ready"])
        self.assertFalse(
            sleep_review_scheduler_cycle_autonomy_proposal[
                "executes_suggested_endpoint"
            ]
        )
        self.assertFalse(
            sleep_review_scheduler_cycle_autonomy_proposal["records_replay_artifact"]
        )
        self.assertFalse(
            sleep_review_scheduler_cycle_autonomy_proposal["applies_plasticity"]
        )
        self.assertFalse(
            sleep_review_scheduler_cycle_autonomy_proposal["mutates_runtime_state"]
        )
        self.assertEqual(
            sleep_review_scheduler_cycle_autonomy_proposal["promotion_gate"]["status"],
            "waiting_for_review_cadence",
        )
        self.assertEqual(
            sleep_replay_selection_proposal["surface"],
            "snn_due_cycle_bounded_replay_selection_proposal.v1",
        )
        self.assertFalse(sleep_replay_selection_proposal["ready"])
        self.assertEqual(
            sleep_replay_selection_proposal["selection"]["candidate_count"],
            0,
        )
        self.assertFalse(sleep_replay_selection_proposal["executes_suggested_endpoint"])
        self.assertFalse(sleep_replay_selection_proposal["records_replay_artifact"])
        self.assertFalse(sleep_replay_selection_proposal["runs_live_replay"])
        self.assertFalse(sleep_replay_selection_proposal["applies_plasticity"])
        self.assertFalse(sleep_replay_selection_proposal["mutates_runtime_state"])
        self.assertEqual(
            due_cycle_recording_review_proposal["surface"],
            "snn_due_cycle_replay_artifact_recording_review_proposal.v1",
        )
        self.assertFalse(due_cycle_recording_review_proposal["ready"])
        self.assertIsNone(
            due_cycle_recording_review_proposal["review_target"][
                "replay_evaluation_context_id"
            ]
        )
        self.assertFalse(due_cycle_recording_review_proposal["records_replay_artifact"])
        self.assertFalse(due_cycle_recording_review_proposal["runs_live_replay"])
        self.assertFalse(due_cycle_recording_review_proposal["applies_plasticity"])
        self.assertFalse(due_cycle_recording_review_proposal["mutates_runtime_state"])
        self.assertEqual(
            sleep_phase_separation["surface"],
            "snn_sleep_phase_separation_proposal.v1",
        )
        self.assertFalse(
            sleep_phase_separation["nrem_like_replay_nomination"]["ready"]
        )
        self.assertFalse(
            sleep_phase_separation["rem_like_stabilization_review"]["ready"]
        )
        self.assertFalse(sleep_phase_separation["records_replay_artifact"])
        self.assertFalse(sleep_phase_separation["runs_live_replay"])
        self.assertFalse(sleep_phase_separation["applies_plasticity"])
        self.assertFalse(sleep_phase_separation["mutates_runtime_state"])
        self.assertEqual(
            rem_homeostatic_preflight["surface"],
            "snn_rem_like_homeostatic_stabilization_preflight.v1",
        )
        self.assertFalse(rem_homeostatic_preflight["ready"])
        self.assertFalse(rem_homeostatic_preflight["executes_suggested_endpoint"])
        self.assertFalse(rem_homeostatic_preflight["writes_checkpoint"])
        self.assertFalse(rem_homeostatic_preflight["applies_plasticity"])
        self.assertFalse(rem_homeostatic_preflight["mutates_transition_memory"])
        self.assertFalse(rem_homeostatic_preflight["mutates_runtime_state"])
        terminus_review_scheduler = status_after_sleep_ticket["terminus_runtime"][
            "snn_sleep_plasticity_review_scheduler"
        ]
        self.assertTrue(terminus_review_scheduler["scheduler_installed"])
        self.assertEqual(
            terminus_review_scheduler["scheduler_installation_id"],
            sleep_review_scheduler_installation["scheduler_installation_id"],
        )
        living_loop_sleep_proposal = living_loop_after_sleep_ticket["living_loop"][
            "snn_sleep_plasticity_autonomy_proposal"
        ]
        self.assertEqual(
            living_loop_sleep_proposal["candidate"]["review_ticket_id"],
            sleep_ticket["review_ticket_id"],
        )
        self.assertFalse(living_loop_sleep_proposal["executable"])
        self.assertFalse(living_loop_sleep_proposal["mutates_runtime_state"])
        policy_sleep_proposal = policy_after_sleep_ticket[
            "snn_sleep_plasticity_autonomy_proposal"
        ]
        self.assertEqual(
            policy_sleep_proposal["candidate"]["review_ticket_id"],
            sleep_ticket["review_ticket_id"],
        )
        self.assertFalse(policy_sleep_proposal["executable"])
        self.assertFalse(policy_sleep_proposal["mutates_runtime_state"])
        self.assertEqual(
            terminus_after_sleep_ticket["snn_sleep_plasticity_autonomy_proposal"][
                "candidate"
            ]["review_ticket_id"],
            sleep_ticket["review_ticket_id"],
        )
        self.assertEqual(
            status_after_sleep_ticket["snn_sleep_plasticity_autonomy_proposal"][
                "candidate"
            ]["review_ticket_id"],
            sleep_ticket["review_ticket_id"],
        )
        status_sleep_gate = status_after_sleep_ticket["runtime_truth"]["evidence"][
            "snn_sleep_plasticity_autonomy_gate"
        ]
        self.assertEqual(status_sleep_gate["review_ticket_id"], sleep_ticket["review_ticket_id"])
        self.assertEqual(
            status_sleep_gate["next_gate"],
            sleep_policy["recommendation"]["suggested_endpoint"],
        )
        self.assertTrue(status_sleep_gate["eligible_for_autonomy_planning"])
        self.assertFalse(status_sleep_gate["eligible_for_action"])
        self.assertFalse(status_sleep_gate["eligible_for_structural_write"])
        living_loop_scheduler_installation_proposal = living_loop_after_sleep_ticket[
            "living_loop"
        ]["snn_sleep_plasticity_scheduler_installation_autonomy_proposal"]
        self.assertEqual(
            living_loop_scheduler_installation_proposal["candidate"][
                "scheduler_design_review_ticket_id"
            ],
            sleep_scheduler_design_review_ticket["scheduler_design_review_ticket_id"],
        )
        self.assertFalse(living_loop_scheduler_installation_proposal["installs_scheduler"])
        policy_scheduler_installation_proposal = policy_after_sleep_ticket[
            "snn_sleep_plasticity_scheduler_installation_autonomy_proposal"
        ]
        self.assertFalse(policy_scheduler_installation_proposal["registers_timer"])
        self.assertEqual(
            terminus_after_sleep_ticket[
                "snn_sleep_plasticity_scheduler_installation_autonomy_proposal"
            ]["candidate"]["scheduler_design_review_ticket_id"],
            sleep_scheduler_design_review_ticket["scheduler_design_review_ticket_id"],
        )
        self.assertEqual(
            status_after_sleep_ticket[
                "snn_sleep_plasticity_scheduler_installation_autonomy_proposal"
            ]["candidate"]["scheduler_design_review_ticket_id"],
            sleep_scheduler_design_review_ticket["scheduler_design_review_ticket_id"],
        )
        status_scheduler_installation_gate = status_after_sleep_ticket["runtime_truth"][
            "evidence"
        ]["snn_sleep_plasticity_scheduler_installation_autonomy_gate"]
        self.assertTrue(status_scheduler_installation_gate["ready"])
        self.assertFalse(status_scheduler_installation_gate["installs_scheduler"])
        self.assertFalse(status_scheduler_installation_gate["registers_timer"])
        self.assertFalse(status_scheduler_installation_gate["starts_background_worker"])
        self.assertFalse(
            status_scheduler_installation_gate["eligible_for_scheduler_installation"]
        )
        living_loop_replay_selection = living_loop_after_sleep_ticket["living_loop"][
            "snn_due_cycle_bounded_replay_selection_proposal"
        ]
        self.assertFalse(living_loop_replay_selection["ready"])
        self.assertFalse(living_loop_replay_selection["runs_live_replay"])
        policy_replay_selection = policy_after_sleep_ticket[
            "snn_due_cycle_bounded_replay_selection_proposal"
        ]
        self.assertFalse(policy_replay_selection["records_replay_artifact"])
        self.assertFalse(policy_replay_selection["mutates_runtime_state"])
        self.assertFalse(
            terminus_after_sleep_ticket[
                "snn_due_cycle_bounded_replay_selection_proposal"
            ]["ready"]
        )
        self.assertFalse(
            status_after_sleep_ticket[
                "snn_due_cycle_bounded_replay_selection_proposal"
            ]["ready"]
        )
        status_replay_selection_gate = status_after_sleep_ticket["runtime_truth"][
            "evidence"
        ]["snn_due_cycle_bounded_replay_selection_gate"]
        self.assertFalse(status_replay_selection_gate["ready"])
        self.assertFalse(status_replay_selection_gate["runs_live_replay"])
        self.assertFalse(status_replay_selection_gate["records_replay_artifact"])
        self.assertFalse(status_replay_selection_gate["applies_plasticity"])
        self.assertEqual(
            status_after_sleep_ticket["state_revision"],
            status_before_sleep_ticket_response.json()["state_revision"],
        )

    def test_cognitive_signal_endpoint_exposes_subcortical_language_surface(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            app = create_app(_build_checkpoint(root, test_case="service_api_cognitive_signal"), trace_dir=root / "traces")
            with TestClient(app) as client:
                signal_response = client.get("/terminus/cognitive-signal")
                language_response = client.get("/terminus/subcortical-language")
                deliberation_response = client.get("/terminus/subcortical-deliberation")
                readiness_response = client.get("/terminus/snn-language-readiness")
                evaluation_response = client.get("/terminus/snn-language-evaluation")
                heldout_response = client.post(
                    "/terminus/snn-language-evaluation/heldout",
                    json={
                        "heldout_readout_slot_batches": [
                            [
                                {
                                    "label": "prediction error",
                                    "pressure_band": "high",
                                    "grounded": True,
                                },
                                {
                                    "label": "concept focus",
                                    "pressure_band": "medium",
                                    "grounded": True,
                                },
                            ]
                        ],
                        "device_evidence": {"device": "cpu", "source": "service_api_test"},
                    },
                )
                training_readiness_response = client.post(
                    "/terminus/snn-language-training/readiness",
                    json={
                        "heldout_evaluation": heldout_response.json(),
                        "runtime_truth_delta": {"improved_or_stable": True},
                        "rollback_policy": {"available": True, "snapshot_id": "pre-language-training"},
                    },
                )
                trainer_dry_run_response = client.post(
                    "/terminus/snn-language-training/dry-run",
                    json={
                        "training_readout_slot_batches": [
                            [{"label": "prediction error", "pressure_band": "high", "grounded": True}],
                            [{"label": "concept focus", "pressure_band": "medium", "grounded": True}],
                            [{"label": "memory pressure", "pressure_band": "medium", "grounded": True}],
                        ],
                        "validation_readout_slot_batches": [
                            [{"label": "prediction error", "pressure_band": "high", "grounded": True}],
                            [{"label": "concept focus", "pressure_band": "medium", "grounded": True}],
                        ],
                        "device_evidence": {"device": "cpu", "source": "service_api_test"},
                        "learning_rate": 0.1,
                        "epochs": 2,
                    },
                )
                trainer_evaluation_response = client.post(
                    "/terminus/snn-language-training/evaluate",
                    json={
                        "dry_run_report": trainer_dry_run_response.json(),
                        "runtime_truth_delta": {"improved_or_stable": True},
                        "rollback_policy": {"available": True, "snapshot_id": "pre-trainer-eval"},
                    },
                )
                sequence_prediction_response = client.post(
                    "/terminus/snn-language-sequence/predict",
                    json={
                        "training_readout_slot_batches": [
                            [{"label": "prediction error", "pressure_band": "high", "grounded": True}],
                            [{"label": "concept focus", "pressure_band": "medium", "grounded": True}],
                            [{"label": "memory pressure", "pressure_band": "medium", "grounded": True}],
                        ],
                        "current_readout_slots": [
                            {"label": "concept focus", "pressure_band": "medium", "grounded": True}
                        ],
                        "device_evidence": {"device": "cpu", "source": "service_api_test"},
                        "top_k": 4,
                    },
                )
                sequence_mismatch_response = client.post(
                    "/terminus/snn-language-sequence/mismatch",
                    json={
                        "prediction_report": sequence_prediction_response.json(),
                        "observed_readout_slots": [
                            {"label": "memory pressure", "pressure_band": "medium", "grounded": True}
                        ],
                        "device_evidence": {"device": "cpu", "source": "service_api_test"},
                    },
                )
                plasticity_pressure_response = client.post(
                    "/terminus/snn-language-sequence/plasticity-pressure",
                    json={
                        "mismatch_report": sequence_mismatch_response.json(),
                        "runtime_truth_delta": {"improved_or_stable": True},
                        "rollback_policy": {"available": True, "snapshot_id": "pre-language-plasticity"},
                    },
                )
                plasticity_trial_response = client.post(
                    "/terminus/snn-language-sequence/plasticity-trial",
                    json={
                        "pressure_report": plasticity_pressure_response.json(),
                        "runtime_truth_delta": {"improved_or_stable": True},
                        "rollback_policy": {"available": True, "snapshot_id": "pre-language-plasticity"},
                    },
                )
                plasticity_replay_response = client.post(
                    "/terminus/snn-language-sequence/plasticity-replay-evaluation",
                    json={
                        "trial_report": plasticity_trial_response.json(),
                        "replay_window": [{"case_id": "sequence-replay-1", "grounded": True}],
                        "runtime_truth_delta": {"improved_or_stable": True},
                        "rollback_policy": {"available": True, "snapshot_id": "pre-language-plasticity"},
                    },
                )
                plasticity_replay_experiment_response = client.post(
                    "/terminus/snn-language-sequence/plasticity-replay-experiment",
                    json={
                        "replay_evaluation": plasticity_replay_response.json(),
                        "replay_sequences": [{"sequence_id": "sequence-replay-1", "grounded": True}],
                        "runtime_truth_delta": {"improved_or_stable": True},
                        "rollback_policy": {"available": True, "snapshot_id": "pre-language-plasticity"},
                    },
                )
                plasticity_application_design_response = client.post(
                    "/terminus/snn-language-sequence/plasticity-application-design",
                    json={
                        "replay_experiment": plasticity_replay_experiment_response.json(),
                        "application_policy": {
                            "learning_rate": 0.03,
                            "max_weight_delta": 0.04,
                            "locality_radius": 2,
                        },
                        "device_evidence": {"device": "cpu", "source": "service_api_test"},
                        "runtime_truth_delta": {"improved_or_stable": True},
                        "rollback_policy": {"available": True, "snapshot_id": "pre-language-plasticity"},
                    },
                )
                current_sparse_indices = sequence_prediction_response.json()["current_sparse_code"]["active_indices"]
                persistent_target_index = (int(current_sparse_indices[0]) + 1) % 64
                plasticity_shadow_delta_response = client.post(
                    "/terminus/snn-language-sequence/plasticity-shadow-delta",
                    json={
                        "application_design": plasticity_application_design_response.json(),
                        "replay_sequences": [
                            {
                                "pre_indices": [int(current_sparse_indices[0])],
                                "post_indices": [persistent_target_index],
                                "grounded": True,
                            }
                        ],
                        "device_evidence": {"device": "cpu", "source": "service_api_test"},
                    },
                )
                plasticity_shadow_application_response = client.post(
                    "/terminus/snn-language-sequence/plasticity-shadow-application",
                    json={
                        "application_design": plasticity_application_design_response.json(),
                        "shadow_delta": plasticity_shadow_delta_response.json(),
                        "device_evidence": {"device": "cpu", "source": "service_api_test"},
                        "runtime_truth_delta": {"improved_or_stable": True},
                        "rollback_policy": {"available": True, "snapshot_id": "pre-language-plasticity"},
                    },
                )
                plasticity_live_readiness_response = client.post(
                    "/terminus/snn-language-sequence/plasticity-live-application-readiness",
                    json={
                        "shadow_application": plasticity_shadow_application_response.json(),
                        "rollback_readiness": {
                            "checkpoint_available": True,
                            "checkpoint_path": "checkpoint://pre-language-plasticity",
                            "restore_endpoint_available": True,
                        },
                        "operator_approval": {
                            "approved": True,
                            "operator_id": "operator-test",
                            "approval_id": "approval-1",
                        },
                    },
                )
                plasticity_preflight_response = client.post(
                    "/terminus/snn-language-sequence/plasticity-live-application-preflight",
                    json={
                        "live_application_readiness": plasticity_live_readiness_response.json(),
                        "application_target": {
                            "available": True,
                            "target_id": "marulho.snn_language.sparse_transition_weights",
                            "owned_by_marulho": True,
                            "mutable": True,
                            "sparse": True,
                            "checkpointed": True,
                        },
                        "checkpoint_transaction": {
                            "pre_update_checkpoint_saved": True,
                            "checkpoint_path": str(root / "pre_language_plasticity.pt"),
                            "restore_verified": True,
                            "records_shadow_delta": True,
                        },
                    },
                )
                status_before_live_application_response = client.get("/status")
                blocked_live_application_response = client.post(
                    "/terminus/snn-language-sequence/plasticity-live-application",
                    json={
                        "live_application_readiness": plasticity_live_readiness_response.json(),
                        "shadow_delta": plasticity_shadow_delta_response.json(),
                        "expected_state_revision": status_before_live_application_response.json()["state_revision"],
                        "operator_id": "operator-test",
                        "confirmation": False,
                    },
                )
                status_after_blocked_live_application_response = client.get("/status")
                plasticity_live_application_response = client.post(
                    "/terminus/snn-language-sequence/plasticity-live-application",
                    json={
                        "live_application_readiness": plasticity_live_readiness_response.json(),
                        "shadow_delta": plasticity_shadow_delta_response.json(),
                        "expected_state_revision": status_before_live_application_response.json()["state_revision"],
                        "operator_id": "operator-test",
                        "confirmation": True,
                        "checkpoint_path": str(root / "pre_language_plasticity.pt"),
                    },
                )
                plasticity_runtime_state_response = client.get(
                    "/terminus/snn-language-sequence/plasticity-runtime-state"
                )
                transition_memory_sleep_policy_response = client.post(
                    "/terminus/snn-language-sequence/plasticity-sleep-policy",
                    json={
                        "subcortex_sleep_pressure": {
                            "pressure": 0.8,
                            "source": "living_loop.subcortex_sleep_pressure",
                        },
                        "replay_evidence": {
                            "available": True,
                            "ready": True,
                            "source": "replay_controller",
                        },
                    },
                )
                persistent_sequence_prediction_response = client.post(
                    "/terminus/snn-language-sequence/predict",
                    json={
                        "training_readout_slot_batches": [
                            [{"label": "prediction error", "pressure_band": "high", "grounded": True}],
                            [{"label": "concept focus", "pressure_band": "medium", "grounded": True}],
                            [{"label": "memory pressure", "pressure_band": "medium", "grounded": True}],
                        ],
                        "current_readout_slots": [
                            {"label": "concept focus", "pressure_band": "medium", "grounded": True}
                        ],
                        "device_evidence": {"device": "cpu", "source": "service_api_test"},
                        "top_k": 4,
                    },
                )
                transition_memory_prediction_evaluation_response = client.post(
                    "/terminus/snn-language-sequence/transition-memory-prediction-evaluation",
                    json={
                        "training_readout_slot_batches": [
                            [{"label": "prediction error", "pressure_band": "high", "grounded": True}],
                            [{"label": "concept focus", "pressure_band": "medium", "grounded": True}],
                            [{"label": "memory pressure", "pressure_band": "medium", "grounded": True}],
                        ],
                        "evaluation_readout_slot_batches": [
                            [{"label": "concept focus", "pressure_band": "medium", "grounded": True}],
                            [{"label": "memory pressure", "pressure_band": "medium", "grounded": True}],
                        ],
                        "device_evidence": {"device": "cpu", "source": "service_api_test"},
                        "top_k": 4,
                    },
                )
                persistent_readout_draft_response = client.post(
                    "/terminus/snn-language-sequence/readout-draft",
                    json={
                        "prediction_report": persistent_sequence_prediction_response.json(),
                        "readout_vocabulary_slots": [
                            {"label": "memory pressure", "pressure_band": "medium", "grounded": True},
                            {"label": "prediction error", "pressure_band": "high", "grounded": True},
                            {"label": "concept focus", "pressure_band": "medium", "grounded": True},
                        ],
                        "device_evidence": {"device": "cpu", "source": "service_api_test"},
                        "transition_memory_evaluation": transition_memory_prediction_evaluation_response.json(),
                        "max_draft_terms": 4,
                    },
                )
                checkpoint_save_response = client.post(
                    "/checkpoint/save",
                    json={"path": str(root / "post_language_plasticity.pt")},
                )
                checkpoint_restore_response = client.post(
                    "/checkpoint/restore",
                    json={"path": checkpoint_save_response.json()["path"]},
                )
                plasticity_runtime_state_after_restore_response = client.get(
                    "/terminus/snn-language-sequence/plasticity-runtime-state"
                )
                persistent_sequence_prediction_after_restore_response = client.post(
                    "/terminus/snn-language-sequence/predict",
                    json={
                        "training_readout_slot_batches": [
                            [{"label": "prediction error", "pressure_band": "high", "grounded": True}],
                            [{"label": "concept focus", "pressure_band": "medium", "grounded": True}],
                            [{"label": "memory pressure", "pressure_band": "medium", "grounded": True}],
                        ],
                        "current_readout_slots": [
                            {"label": "concept focus", "pressure_band": "medium", "grounded": True}
                        ],
                        "device_evidence": {"device": "cpu", "source": "service_api_test"},
                        "top_k": 4,
                    },
                )
                transition_memory_prediction_evaluation_after_restore_response = client.post(
                    "/terminus/snn-language-sequence/transition-memory-prediction-evaluation",
                    json={
                        "training_readout_slot_batches": [
                            [{"label": "prediction error", "pressure_band": "high", "grounded": True}],
                            [{"label": "concept focus", "pressure_band": "medium", "grounded": True}],
                            [{"label": "memory pressure", "pressure_band": "medium", "grounded": True}],
                        ],
                        "evaluation_readout_slot_batches": [
                            [{"label": "concept focus", "pressure_band": "medium", "grounded": True}],
                            [{"label": "memory pressure", "pressure_band": "medium", "grounded": True}],
                        ],
                        "device_evidence": {"device": "cpu", "source": "service_api_test"},
                        "top_k": 4,
                    },
                )
                status_before_homeostatic_maintenance_response = client.get("/status")
                blocked_homeostatic_maintenance_response = client.post(
                    "/terminus/snn-language-sequence/plasticity-homeostatic-maintenance",
                    json={
                        "expected_state_revision": status_before_homeostatic_maintenance_response.json()["state_revision"],
                        "operator_id": "operator-test",
                        "confirmation": False,
                    },
                )
                homeostatic_maintenance_response = client.post(
                    "/terminus/snn-language-sequence/plasticity-homeostatic-maintenance",
                    json={
                        "expected_state_revision": status_before_homeostatic_maintenance_response.json()["state_revision"],
                        "operator_id": "operator-test",
                        "confirmation": True,
                        "checkpoint_path": str(root / "pre_homeostatic_maintenance.pt"),
                        "decay_factor": 0.5,
                        "prune_below": 0.02,
                        "max_outgoing_row_mass": 1.0,
                    },
                )
                plasticity_runtime_state_after_maintenance_response = client.get(
                    "/terminus/snn-language-sequence/plasticity-runtime-state"
                )
                regeneration_context_response = client.post(
                    "/terminus/snn-language-sequence/replay-evaluation-context",
                    json={
                        "prediction_report": {
                            "surface": "snn_language_sequence_prediction_probe.v1",
                            "available": True,
                            "prediction": {"predicted_sparse_indices": [14]},
                        },
                        "observed_readout_slots": [
                            {"label": "memory pressure", "pressure_band": "medium", "grounded": True}
                        ],
                        "runtime_truth_delta": {"improved_or_stable": True},
                        "rollback_policy": {"available": True, "snapshot_id": "pre-regeneration-context"},
                    },
                )
                regeneration_mismatch = regeneration_context_response.json()["mismatch_report"]
                regeneration_design_preview_response = client.post(
                    "/terminus/snn-language-sequence/plasticity-regeneration-proposal",
                    json={"mismatch_report": regeneration_mismatch},
                )
                status_before_regeneration_readout_record_response = client.get("/status")
                regeneration_readout_record_response = client.post(
                    "/terminus/snn-language-sequence/readout-ledger/record",
                    json={
                        "readout_draft": persistent_readout_draft_response.json(),
                        "expected_state_revision": status_before_regeneration_readout_record_response.json()[
                            "state_revision"
                        ],
                        "operator_id": "operator-test",
                        "confirmation": True,
                    },
                )
                regeneration_review_ticket_response = client.post(
                    "/terminus/snn-language-sequence/replay-artifact-recording-review-ticket",
                    json={
                        "limit": 4,
                        "min_priority_score": 60.0,
                        "operator_id": "operator-test",
                        "confirmation": True,
                    },
                )
                regeneration_replay_artifact_response = client.post(
                    "/terminus/snn-language-sequence/transition-memory-replay-artifact/evaluated-record",
                    json={
                        "replay_evaluation_context_id": regeneration_context_response.json()[
                            "replay_evaluation_context_id"
                        ],
                        "review_ticket_id": regeneration_review_ticket_response.json()["review_ticket_id"],
                        "operator_id": "operator-test",
                        "confirmation": True,
                    },
                )
                regeneration_permit_response = client.post(
                    "/terminus/snn-language-sequence/plasticity-regeneration-permit",
                    json={
                        "replay_artifact_id": regeneration_replay_artifact_response.json()[
                            "replay_artifact_id"
                        ],
                        "regeneration_design": regeneration_design_preview_response.json()[
                            "regeneration_design"
                        ],
                        "operator_id": "operator-test",
                        "confirmation": True,
                    },
                )
                regeneration_proposal_response = client.post(
                    "/terminus/snn-language-sequence/plasticity-regeneration-proposal",
                    json={
                        "mismatch_report": regeneration_mismatch,
                        "replay_evidence": regeneration_permit_response.json(),
                    },
                )
                status_before_regeneration_response = client.get("/status")
                blocked_regeneration_response = client.post(
                    "/terminus/snn-language-sequence/plasticity-regeneration",
                    json={
                        "regeneration_proposal": regeneration_proposal_response.json(),
                        "expected_state_revision": status_before_regeneration_response.json()["state_revision"],
                        "operator_id": "operator-test",
                        "confirmation": False,
                    },
                )
                tampered_regeneration_proposal = deepcopy(regeneration_proposal_response.json())
                tampered_regeneration_proposal["regeneration_design"]["candidate_synapses"][0][
                    "post_index"
                ] = 4
                tampered_regeneration_response = client.post(
                    "/terminus/snn-language-sequence/plasticity-regeneration",
                    json={
                        "regeneration_proposal": tampered_regeneration_proposal,
                        "expected_state_revision": status_before_regeneration_response.json()[
                            "state_revision"
                        ],
                        "operator_id": "operator-test",
                        "confirmation": True,
                    },
                )
                regeneration_response = client.post(
                    "/terminus/snn-language-sequence/plasticity-regeneration",
                    json={
                        "regeneration_proposal": regeneration_proposal_response.json(),
                        "expected_state_revision": status_before_regeneration_response.json()["state_revision"],
                        "operator_id": "operator-test",
                        "confirmation": True,
                        "checkpoint_path": str(root / "pre_regeneration.pt"),
                    },
                )
                plasticity_runtime_state_after_regeneration_response = client.get(
                    "/terminus/snn-language-sequence/plasticity-runtime-state"
                )
                regeneration_provenance_audit_response = client.get(
                    "/terminus/snn-language-sequence/readout-ledger/synapse-provenance-audit",
                    params={"limit": 8},
                )
                status_after_regeneration_response = client.get("/status")
                stale_regeneration_response = client.post(
                    "/terminus/snn-language-sequence/plasticity-regeneration",
                    json={
                        "regeneration_proposal": regeneration_proposal_response.json(),
                        "expected_state_revision": status_after_regeneration_response.json()["state_revision"],
                        "operator_id": "operator-test",
                        "confirmation": True,
                    },
                )
                committed_regeneration_restore_response = client.post(
                    "/checkpoint/restore",
                    json={"path": regeneration_response.json()["checkpoint_transaction"]["committed_checkpoint_path"]},
                )
                plasticity_runtime_state_after_committed_regeneration_restore_response = client.get(
                    "/terminus/snn-language-sequence/plasticity-runtime-state"
                )
            app.state.marulho_manager.close()

        self.assertEqual(signal_response.status_code, 200)
        self.assertEqual(language_response.status_code, 200)
        self.assertEqual(deliberation_response.status_code, 200)
        self.assertEqual(readiness_response.status_code, 200)
        self.assertEqual(evaluation_response.status_code, 200)
        self.assertEqual(heldout_response.status_code, 200)
        self.assertEqual(training_readiness_response.status_code, 200)
        self.assertEqual(trainer_dry_run_response.status_code, 200)
        self.assertEqual(trainer_evaluation_response.status_code, 200)
        self.assertEqual(sequence_prediction_response.status_code, 200)
        self.assertEqual(sequence_mismatch_response.status_code, 200)
        self.assertEqual(plasticity_pressure_response.status_code, 200)
        self.assertEqual(plasticity_trial_response.status_code, 200)
        self.assertEqual(plasticity_replay_response.status_code, 200)
        self.assertEqual(plasticity_replay_experiment_response.status_code, 200)
        self.assertEqual(plasticity_application_design_response.status_code, 200)
        self.assertEqual(plasticity_shadow_delta_response.status_code, 200)
        self.assertEqual(plasticity_shadow_application_response.status_code, 200)
        self.assertEqual(plasticity_live_readiness_response.status_code, 200)
        self.assertEqual(plasticity_preflight_response.status_code, 200)
        self.assertEqual(status_before_live_application_response.status_code, 200)
        self.assertEqual(blocked_live_application_response.status_code, 200)
        self.assertEqual(status_after_blocked_live_application_response.status_code, 200)
        self.assertEqual(plasticity_live_application_response.status_code, 200)
        self.assertEqual(plasticity_runtime_state_response.status_code, 200)
        self.assertEqual(transition_memory_sleep_policy_response.status_code, 200)
        self.assertEqual(persistent_sequence_prediction_response.status_code, 200)
        self.assertEqual(transition_memory_prediction_evaluation_response.status_code, 200)
        self.assertEqual(persistent_readout_draft_response.status_code, 200)
        self.assertEqual(checkpoint_save_response.status_code, 200)
        self.assertEqual(checkpoint_restore_response.status_code, 200)
        self.assertEqual(plasticity_runtime_state_after_restore_response.status_code, 200)
        self.assertEqual(persistent_sequence_prediction_after_restore_response.status_code, 200)
        self.assertEqual(transition_memory_prediction_evaluation_after_restore_response.status_code, 200)
        self.assertEqual(status_before_homeostatic_maintenance_response.status_code, 200)
        self.assertEqual(blocked_homeostatic_maintenance_response.status_code, 200)
        self.assertEqual(homeostatic_maintenance_response.status_code, 200)
        self.assertEqual(plasticity_runtime_state_after_maintenance_response.status_code, 200)
        self.assertEqual(regeneration_design_preview_response.status_code, 200)
        self.assertEqual(regeneration_readout_record_response.status_code, 200)
        self.assertEqual(regeneration_replay_artifact_response.status_code, 200)
        self.assertEqual(regeneration_proposal_response.status_code, 200)
        self.assertEqual(regeneration_permit_response.status_code, 200)
        self.assertEqual(status_before_regeneration_response.status_code, 200)
        self.assertEqual(blocked_regeneration_response.status_code, 200)
        self.assertEqual(tampered_regeneration_response.status_code, 200)
        self.assertEqual(regeneration_response.status_code, 200)
        self.assertEqual(plasticity_runtime_state_after_regeneration_response.status_code, 200)
        self.assertEqual(regeneration_provenance_audit_response.status_code, 200)
        self.assertEqual(stale_regeneration_response.status_code, 200)
        self.assertEqual(committed_regeneration_restore_response.status_code, 200)
        self.assertEqual(plasticity_runtime_state_after_committed_regeneration_restore_response.status_code, 200)
        signal = signal_response.json()
        language = language_response.json()
        deliberation = deliberation_response.json()
        readiness = readiness_response.json()
        evaluation = evaluation_response.json()
        heldout = heldout_response.json()
        training_readiness = training_readiness_response.json()
        trainer_dry_run = trainer_dry_run_response.json()
        trainer_evaluation = trainer_evaluation_response.json()
        sequence_prediction = sequence_prediction_response.json()
        sequence_mismatch = sequence_mismatch_response.json()
        plasticity_pressure = plasticity_pressure_response.json()
        plasticity_trial = plasticity_trial_response.json()
        plasticity_replay = plasticity_replay_response.json()
        plasticity_replay_experiment = plasticity_replay_experiment_response.json()
        plasticity_application_design = plasticity_application_design_response.json()
        plasticity_shadow_delta = plasticity_shadow_delta_response.json()
        plasticity_shadow_application = plasticity_shadow_application_response.json()
        plasticity_live_readiness = plasticity_live_readiness_response.json()
        plasticity_preflight = plasticity_preflight_response.json()
        status_before_live_application = status_before_live_application_response.json()
        blocked_live_application = blocked_live_application_response.json()
        status_after_blocked_live_application = status_after_blocked_live_application_response.json()
        plasticity_live_application = plasticity_live_application_response.json()
        plasticity_runtime_state = plasticity_runtime_state_response.json()
        transition_memory_sleep_policy = transition_memory_sleep_policy_response.json()
        persistent_sequence_prediction = persistent_sequence_prediction_response.json()
        transition_memory_prediction_evaluation = transition_memory_prediction_evaluation_response.json()
        persistent_readout_draft = persistent_readout_draft_response.json()
        plasticity_runtime_state_after_restore = plasticity_runtime_state_after_restore_response.json()
        persistent_sequence_prediction_after_restore = persistent_sequence_prediction_after_restore_response.json()
        transition_memory_prediction_evaluation_after_restore = (
            transition_memory_prediction_evaluation_after_restore_response.json()
        )
        status_before_homeostatic_maintenance = status_before_homeostatic_maintenance_response.json()
        blocked_homeostatic_maintenance = blocked_homeostatic_maintenance_response.json()
        homeostatic_maintenance = homeostatic_maintenance_response.json()
        plasticity_runtime_state_after_maintenance = plasticity_runtime_state_after_maintenance_response.json()
        regeneration_proposal = regeneration_proposal_response.json()
        regeneration_design_preview = regeneration_design_preview_response.json()
        regeneration_replay_artifact = regeneration_replay_artifact_response.json()
        regeneration_permit = regeneration_permit_response.json()
        blocked_regeneration = blocked_regeneration_response.json()
        tampered_regeneration = tampered_regeneration_response.json()
        regeneration = regeneration_response.json()
        plasticity_runtime_state_after_regeneration = plasticity_runtime_state_after_regeneration_response.json()
        regeneration_provenance_audit = regeneration_provenance_audit_response.json()
        stale_regeneration = stale_regeneration_response.json()
        plasticity_runtime_state_after_committed_regeneration_restore = (
            plasticity_runtime_state_after_committed_regeneration_restore_response.json()
        )
        self.assertEqual(signal["subcortical_language"]["surface"], "subcortical_language.v1")
        self.assertEqual(signal["subcortical_deliberation"]["surface"], "subcortical_control_candidates.v1")
        self.assertEqual(language["surface"], "subcortical_language.v1")
        self.assertEqual(deliberation["surface"], "subcortical_control_candidates.v1")
        self.assertEqual(readiness["surface"], "snn_native_language_readiness.v1")
        self.assertEqual(readiness["artifact_kind"], "terminus_snn_native_language_readiness_gate")
        self.assertEqual(evaluation["surface"], "snn_language_adapter_evaluation.v1")
        self.assertEqual(evaluation["artifact_kind"], "terminus_snn_language_adapter_evaluation_gate")
        self.assertEqual(heldout["surface"], "snn_language_adapter_heldout_evaluation.v1")
        self.assertEqual(heldout["artifact_kind"], "terminus_snn_language_adapter_heldout_evaluation")
        self.assertEqual(training_readiness["surface"], "snn_language_training_readiness.v1")
        self.assertEqual(
            training_readiness["artifact_kind"],
            "terminus_snn_language_training_readiness_gate",
        )
        self.assertEqual(trainer_dry_run["surface"], "snn_language_trainer_dry_run.v1")
        self.assertEqual(trainer_dry_run["artifact_kind"], "terminus_snn_language_trainer_dry_run")
        self.assertEqual(trainer_evaluation["surface"], "snn_language_trainer_isolated_evaluation.v1")
        self.assertEqual(
            trainer_evaluation["artifact_kind"],
            "terminus_snn_language_trainer_isolated_evaluation",
        )
        self.assertEqual(sequence_prediction["surface"], "snn_language_sequence_prediction_probe.v1")
        self.assertEqual(
            sequence_prediction["artifact_kind"],
            "terminus_snn_language_sequence_prediction_probe",
        )
        self.assertEqual(sequence_mismatch["surface"], "snn_language_sequence_mismatch_probe.v1")
        self.assertEqual(
            sequence_mismatch["artifact_kind"],
            "terminus_snn_language_sequence_mismatch_probe",
        )
        self.assertEqual(plasticity_pressure["surface"], "snn_language_plasticity_pressure.v1")
        self.assertEqual(
            plasticity_pressure["artifact_kind"],
            "terminus_snn_language_plasticity_pressure_gate",
        )
        self.assertEqual(plasticity_trial["surface"], "snn_language_plasticity_trial.v1")
        self.assertEqual(plasticity_trial["artifact_kind"], "terminus_snn_language_plasticity_trial")
        self.assertEqual(plasticity_replay["surface"], "snn_language_plasticity_replay_evaluation.v1")
        self.assertEqual(
            plasticity_replay["artifact_kind"],
            "terminus_snn_language_plasticity_replay_evaluation",
        )
        self.assertEqual(
            plasticity_replay_experiment["surface"],
            "snn_language_plasticity_replay_experiment.v1",
        )
        self.assertEqual(
            plasticity_replay_experiment["artifact_kind"],
            "terminus_snn_language_plasticity_replay_experiment",
        )
        self.assertEqual(
            plasticity_application_design["surface"],
            "snn_language_plasticity_application_design.v1",
        )
        self.assertEqual(
            plasticity_application_design["artifact_kind"],
            "terminus_snn_language_plasticity_application_design",
        )
        self.assertEqual(plasticity_shadow_delta["surface"], "snn_language_plasticity_shadow_delta.v1")
        self.assertEqual(
            plasticity_shadow_delta["artifact_kind"],
            "terminus_snn_language_plasticity_shadow_delta",
        )
        self.assertEqual(
            plasticity_shadow_application["surface"],
            "snn_language_plasticity_shadow_application.v1",
        )
        self.assertEqual(
            plasticity_shadow_application["artifact_kind"],
            "terminus_snn_language_plasticity_shadow_application",
        )
        self.assertEqual(
            plasticity_live_readiness["surface"],
            "snn_language_plasticity_live_application_readiness.v1",
        )
        self.assertEqual(
            plasticity_live_readiness["artifact_kind"],
            "terminus_snn_language_plasticity_live_application_readiness",
        )
        self.assertEqual(
            plasticity_preflight["surface"],
            "snn_language_plasticity_live_application_preflight.v1",
        )
        self.assertEqual(
            plasticity_preflight["artifact_kind"],
            "terminus_snn_language_plasticity_live_application_preflight",
        )
        self.assertEqual(plasticity_live_application["surface"], "snn_language_plasticity_live_application.v1")
        self.assertEqual(
            plasticity_live_application["artifact_kind"],
            "terminus_snn_language_plasticity_live_application",
        )
        self.assertEqual(
            plasticity_runtime_state["surface"],
            "snn_language_plasticity_runtime_state.v1",
        )
        self.assertEqual(
            regeneration_proposal["surface"],
            "snn_language_transition_memory_regeneration_proposal.v1",
        )
        self.assertEqual(
            regeneration_permit["surface"],
            "snn_language_transition_memory_regeneration_permit.v1",
        )
        self.assertEqual(
            regeneration_replay_artifact["surface"],
            "snn_transition_memory_replay_artifact.v1",
        )
        self.assertTrue(regeneration_replay_artifact["owned_by_marulho"])
        self.assertTrue(regeneration_replay_artifact["readout_evidence_hashes"])
        self.assertEqual(
            regeneration_permit["replay_artifact_id"],
            regeneration_replay_artifact["replay_artifact_id"],
        )
        self.assertEqual(
            regeneration_permit["readout_evidence_hashes"],
            regeneration_replay_artifact["readout_evidence_hashes"],
        )
        self.assertEqual(
            regeneration_proposal["replay_evidence"]["readout_evidence_hashes"],
            regeneration_permit["readout_evidence_hashes"],
        )
        self.assertEqual(
            regeneration_permit["regeneration_design_hash"],
            regeneration_proposal["replay_evidence"]["regeneration_design_hash"],
        )
        self.assertEqual(
            regeneration_permit["regeneration_design_candidate_count"],
            regeneration_design_preview["regeneration_design"]["candidate_count"],
        )
        self.assertEqual(regeneration["surface"], "snn_language_transition_memory_regeneration.v1")
        self.assertEqual(
            regeneration["checkpoint_transaction"]["current_checkpoint_manifest"]["checkpoint_path"],
            regeneration["checkpoint_transaction"]["committed_checkpoint_path"],
        )
        self.assertFalse(regeneration_proposal["mutates_runtime_state"])
        self.assertFalse(blocked_regeneration["accepted"])
        self.assertFalse(tampered_regeneration["accepted"])
        self.assertFalse(
            tampered_regeneration["promotion_gate"]["required_evidence"][
                "replay_permit_server_verified"
            ]
        )
        self.assertTrue(regeneration["accepted"])
        self.assertEqual(plasticity_runtime_state_after_regeneration["regeneration_count"], 1)
        self.assertEqual(plasticity_runtime_state_after_regeneration["regenerated_synapse_count_total"], 1)
        regenerated_synapse_key = regeneration["regeneration"]["regenerated_synapses"][0]["synapse"]
        regenerated_provenance = plasticity_runtime_state_after_regeneration[
            "synapse_provenance_by_key"
        ][regenerated_synapse_key]
        self.assertEqual(regenerated_provenance["provenance_type"], "replay_regeneration")
        self.assertEqual(regenerated_provenance["permit_id"], regeneration_permit["permit_id"])
        self.assertEqual(
            regenerated_provenance["replay_artifact_id"],
            regeneration_replay_artifact["replay_artifact_id"],
        )
        self.assertEqual(
            regenerated_provenance["readout_evidence_hashes"],
            regeneration_permit["readout_evidence_hashes"],
        )
        self.assertEqual(
            regeneration_provenance_audit["surface"],
            "snn_language_readout_synapse_provenance_audit.v1",
        )
        regenerated_audit_row = next(
            item
            for item in regeneration_provenance_audit["audited_synapses"]
            if item["synapse_key"] == regenerated_synapse_key
        )
        self.assertEqual(regenerated_audit_row["provenance_type"], "replay_regeneration")
        self.assertTrue(regenerated_audit_row["provenance_complete"])
        self.assertTrue(regenerated_audit_row["replay_ledger_evidence_present"])
        self.assertTrue(regenerated_audit_row["ledger_hash_valid"])
        self.assertFalse(stale_regeneration["accepted"])
        self.assertFalse(stale_regeneration["promotion_gate"]["required_evidence"]["replay_permit_server_verified"])
        self.assertEqual(plasticity_runtime_state_after_committed_regeneration_restore["regeneration_count"], 1)
        self.assertEqual(
            plasticity_runtime_state_after_committed_regeneration_restore["regenerated_synapse_count_total"],
            1,
        )
        self.assertTrue(language["grounded"])
        self.assertTrue(deliberation["grounded"])
        self.assertTrue(readiness["grounded"])
        self.assertTrue(evaluation["grounded"])
        self.assertNotIn("retired_runtime_dependency", language)
        self.assertNotIn("retired_runtime_dependency", deliberation)
        self.assertNotIn("retired_runtime_dependency", readiness)
        self.assertNotIn("retired_runtime_dependency", evaluation)
        self.assertNotIn("retired_runtime_dependency", heldout)
        self.assertNotIn("retired_runtime_dependency", training_readiness)
        self.assertNotIn("retired_runtime_dependency", trainer_dry_run)
        self.assertNotIn("retired_runtime_dependency", trainer_evaluation)
        self.assertNotIn("retired_runtime_dependency", sequence_prediction)
        self.assertNotIn("retired_runtime_dependency", sequence_mismatch)
        self.assertNotIn("retired_runtime_dependency", plasticity_pressure)
        self.assertNotIn("retired_runtime_dependency", plasticity_trial)
        self.assertNotIn("retired_runtime_dependency", plasticity_replay)
        self.assertNotIn("retired_runtime_dependency", plasticity_replay_experiment)
        self.assertNotIn("retired_runtime_dependency", plasticity_application_design)
        self.assertNotIn("retired_runtime_dependency", plasticity_shadow_delta)
        self.assertNotIn("retired_runtime_dependency", plasticity_shadow_application)
        self.assertNotIn("retired_runtime_dependency", plasticity_live_readiness)
        self.assertNotIn("retired_runtime_dependency", plasticity_preflight)
        self.assertNotIn("retired_runtime_dependency", plasticity_live_application)
        self.assertFalse(readiness["executable"])
        self.assertFalse(readiness["mutates_runtime_state"])
        self.assertFalse(readiness["promotion_gate"]["eligible_for_cognition_substrate"])
        self.assertFalse(evaluation["executable"])
        self.assertFalse(evaluation["mutates_runtime_state"])
        self.assertFalse(evaluation["promotion_gate"]["eligible_for_language_generation"])
        self.assertFalse(evaluation["promotion_gate"]["eligible_for_cognition_substrate"])
        self.assertFalse(heldout["generates_text"])
        self.assertFalse(heldout["trains"])
        self.assertFalse(heldout["mutates_runtime_state"])
        self.assertFalse(training_readiness["executable"])
        self.assertFalse(training_readiness["mutates_runtime_state"])
        self.assertFalse(training_readiness["promotion_gate"]["eligible_for_training"])
        self.assertTrue(training_readiness["promotion_gate"]["eligible_for_training_loop_design"])
        self.assertFalse(trainer_dry_run["generates_text"])
        self.assertFalse(trainer_dry_run["trains_runtime_model"])
        self.assertFalse(trainer_dry_run["returns_trained_weights"])
        self.assertFalse(trainer_dry_run["mutates_runtime_state"])
        self.assertFalse(trainer_evaluation["generates_text"])
        self.assertFalse(trainer_evaluation["trains_runtime_model"])
        self.assertFalse(trainer_evaluation["promotes_runtime_trainer"])
        self.assertFalse(trainer_evaluation["mutates_runtime_state"])
        self.assertFalse(sequence_prediction["generates_text"])
        self.assertFalse(sequence_prediction["decodes_text"])
        self.assertFalse(sequence_prediction["trains_runtime_model"])
        self.assertFalse(sequence_prediction["returns_trained_weights"])
        self.assertFalse(sequence_prediction["mutates_runtime_state"])
        self.assertFalse(sequence_mismatch["generates_text"])
        self.assertFalse(sequence_mismatch["decodes_text"])
        self.assertFalse(sequence_mismatch["trains_runtime_model"])
        self.assertFalse(sequence_mismatch["mutates_runtime_state"])
        self.assertFalse(plasticity_pressure["generates_text"])
        self.assertFalse(plasticity_pressure["decodes_text"])
        self.assertFalse(plasticity_pressure["trains_runtime_model"])
        self.assertFalse(plasticity_pressure["applies_plasticity"])
        self.assertFalse(plasticity_pressure["mutates_runtime_state"])
        self.assertFalse(plasticity_trial["generates_text"])
        self.assertFalse(plasticity_trial["decodes_text"])
        self.assertFalse(plasticity_trial["trains_runtime_model"])
        self.assertFalse(plasticity_trial["applies_plasticity"])
        self.assertFalse(plasticity_trial["returns_trained_weights"])
        self.assertFalse(plasticity_trial["mutates_runtime_state"])
        self.assertFalse(plasticity_replay["generates_text"])
        self.assertFalse(plasticity_replay["decodes_text"])
        self.assertFalse(plasticity_replay["trains_runtime_model"])
        self.assertFalse(plasticity_replay["applies_plasticity"])
        self.assertFalse(plasticity_replay["mutates_runtime_state"])
        self.assertFalse(plasticity_replay["promotion_gate"]["eligible_for_plasticity_application"])
        self.assertFalse(plasticity_replay_experiment["generates_text"])
        self.assertFalse(plasticity_replay_experiment["decodes_text"])
        self.assertFalse(plasticity_replay_experiment["trains_runtime_model"])
        self.assertFalse(plasticity_replay_experiment["applies_plasticity"])
        self.assertFalse(plasticity_replay_experiment["returns_trained_weights"])
        self.assertFalse(plasticity_replay_experiment["mutates_runtime_state"])
        self.assertFalse(plasticity_replay_experiment["promotion_gate"]["eligible_for_plasticity_application"])
        self.assertFalse(plasticity_application_design["generates_text"])
        self.assertFalse(plasticity_application_design["decodes_text"])
        self.assertFalse(plasticity_application_design["trains_runtime_model"])
        self.assertFalse(plasticity_application_design["applies_plasticity"])
        self.assertFalse(plasticity_application_design["returns_trained_weights"])
        self.assertFalse(plasticity_application_design["mutates_runtime_state"])
        self.assertEqual(plasticity_application_design["device_evidence"]["tensor_device"], "cpu")
        self.assertTrue(plasticity_application_design["device_evidence"]["device_report_available"])
        self.assertFalse(plasticity_application_design["promotion_gate"]["eligible_for_plasticity_application"])
        self.assertFalse(plasticity_application_design["promotion_gate"]["eligible_for_live_application"])
        self.assertFalse(plasticity_shadow_delta["generates_text"])
        self.assertFalse(plasticity_shadow_delta["decodes_text"])
        self.assertFalse(plasticity_shadow_delta["trains_runtime_model"])
        self.assertFalse(plasticity_shadow_delta["applies_plasticity"])
        self.assertFalse(plasticity_shadow_delta["returns_trained_weights"])
        self.assertFalse(plasticity_shadow_delta["mutates_runtime_state"])
        self.assertEqual(plasticity_shadow_delta["device_evidence"]["tensor_device"], "cpu")
        self.assertTrue(plasticity_shadow_delta["device_evidence"]["device_report_available"])
        self.assertGreater(plasticity_shadow_delta["affected_synapse_count"], 0)
        self.assertFalse(plasticity_shadow_application["generates_text"])
        self.assertFalse(plasticity_shadow_application["decodes_text"])
        self.assertFalse(plasticity_shadow_application["trains_runtime_model"])
        self.assertFalse(plasticity_shadow_application["applies_plasticity"])
        self.assertFalse(plasticity_shadow_application["returns_trained_weights"])
        self.assertFalse(plasticity_shadow_application["mutates_runtime_state"])
        self.assertEqual(plasticity_shadow_application["device_evidence"]["tensor_device"], "cpu")
        self.assertTrue(plasticity_shadow_application["device_evidence"]["device_report_available"])
        self.assertFalse(plasticity_shadow_application["promotion_gate"]["eligible_for_plasticity_application"])
        self.assertFalse(plasticity_shadow_application["promotion_gate"]["eligible_for_live_application"])
        self.assertFalse(plasticity_live_readiness["generates_text"])
        self.assertFalse(plasticity_live_readiness["decodes_text"])
        self.assertFalse(plasticity_live_readiness["trains_runtime_model"])
        self.assertFalse(plasticity_live_readiness["applies_plasticity"])
        self.assertFalse(plasticity_live_readiness["returns_trained_weights"])
        self.assertFalse(plasticity_live_readiness["mutates_runtime_state"])
        self.assertFalse(plasticity_live_readiness["promotion_gate"]["eligible_for_plasticity_application"])
        self.assertFalse(plasticity_live_readiness["promotion_gate"]["eligible_for_live_application"])
        self.assertTrue(
            plasticity_live_readiness["promotion_gate"]["eligible_for_operator_live_application_review"]
        )
        self.assertFalse(plasticity_preflight["applies_plasticity"])
        self.assertFalse(plasticity_preflight["mutates_runtime_state"])
        self.assertFalse(plasticity_preflight["promotion_gate"]["eligible_for_live_application"])
        self.assertTrue(plasticity_preflight["promotion_gate"]["eligible_for_operator_execution_review"])
        self.assertFalse(blocked_live_application["accepted"])
        self.assertFalse(blocked_live_application["applies_plasticity"])
        self.assertFalse(blocked_live_application["mutates_runtime_state"])
        self.assertFalse(
            blocked_live_application["promotion_gate"]["required_evidence"]["confirmation"]
        )
        self.assertEqual(
            status_after_blocked_live_application["state_revision"],
            status_before_live_application["state_revision"],
        )
        self.assertTrue(plasticity_live_application["accepted"])
        self.assertTrue(plasticity_live_application["applies_plasticity"])
        self.assertTrue(plasticity_live_application["mutates_runtime_state"])
        self.assertFalse(plasticity_live_application["generates_text"])
        self.assertFalse(plasticity_live_application["decodes_text"])
        self.assertFalse(plasticity_live_application["loads_external_checkpoint"])
        self.assertGreater(plasticity_live_application["application_target"]["applied_synapse_count"], 0)
        self.assertEqual(
            plasticity_live_application["after"]["state_revision"],
            int(status_before_live_application["state_revision"]) + 1,
        )
        self.assertGreater(plasticity_runtime_state["sparse_transition_weight_count"], 0)
        self.assertEqual(plasticity_runtime_state["applied_update_count"], 1)
        self.assertEqual(
            transition_memory_sleep_policy["surface"],
            "snn_language_transition_memory_sleep_policy.v1",
        )
        self.assertTrue(transition_memory_sleep_policy["recommendation"]["recommended"])
        self.assertFalse(transition_memory_sleep_policy["recommendation"]["executable"])
        self.assertFalse(transition_memory_sleep_policy["mutates_runtime_state"])
        self.assertFalse(transition_memory_sleep_policy["subcortex_sleep_pressure"]["retired_runtime_dependency"])
        self.assertTrue(persistent_sequence_prediction["persistent_transition_evidence"]["influenced_prediction"])
        self.assertGreater(persistent_sequence_prediction["persistent_transition_evidence"]["support_strength"], 0.0)
        self.assertIn(persistent_target_index, persistent_sequence_prediction["prediction"]["predicted_sparse_indices"])
        self.assertEqual(
            transition_memory_prediction_evaluation["surface"],
            "snn_language_transition_memory_prediction_evaluation.v1",
        )
        self.assertFalse(transition_memory_prediction_evaluation["generates_text"])
        self.assertFalse(transition_memory_prediction_evaluation["decodes_text"])
        self.assertFalse(transition_memory_prediction_evaluation["mutates_runtime_state"])
        self.assertGreater(
            transition_memory_prediction_evaluation["evaluation_summary"]["persistent_transition_weight_count"],
            0,
        )
        self.assertTrue(persistent_readout_draft["generates_text"])
        self.assertTrue(persistent_readout_draft["transition_memory_evaluation_evidence"]["review_ready"])
        self.assertTrue(persistent_readout_draft["promotion_gate"]["eligible_for_bounded_readout_generation"])
        self.assertFalse(persistent_readout_draft["promotion_gate"]["eligible_for_cognition_substrate"])
        self.assertEqual(
            plasticity_runtime_state_after_restore["sparse_transition_weight_count"],
            plasticity_runtime_state["sparse_transition_weight_count"],
        )
        self.assertEqual(
            plasticity_runtime_state_after_restore["applied_update_count"],
            plasticity_runtime_state["applied_update_count"],
        )
        self.assertTrue(
            persistent_sequence_prediction_after_restore["persistent_transition_evidence"]["influenced_prediction"]
        )
        self.assertGreater(
            persistent_sequence_prediction_after_restore["persistent_transition_evidence"]["support_strength"],
            0.0,
        )
        self.assertIn(
            persistent_target_index,
            persistent_sequence_prediction_after_restore["prediction"]["predicted_sparse_indices"],
        )
        self.assertGreater(
            transition_memory_prediction_evaluation_after_restore["evaluation_summary"][
                "persistent_transition_weight_count"
            ],
            0,
        )
        self.assertFalse(transition_memory_prediction_evaluation_after_restore["mutates_runtime_state"])
        self.assertFalse(blocked_homeostatic_maintenance["accepted"])
        self.assertFalse(blocked_homeostatic_maintenance["mutates_runtime_state"])
        self.assertFalse(
            blocked_homeostatic_maintenance["promotion_gate"]["required_evidence"]["confirmation"]
        )
        self.assertTrue(homeostatic_maintenance["accepted"])
        self.assertTrue(homeostatic_maintenance["mutates_runtime_state"])
        self.assertTrue(homeostatic_maintenance["applies_plasticity"])
        self.assertFalse(homeostatic_maintenance["generates_text"])
        self.assertFalse(homeostatic_maintenance["decodes_text"])
        self.assertFalse(homeostatic_maintenance["loads_external_checkpoint"])
        self.assertGreater(homeostatic_maintenance["homeostatic_maintenance"]["pruned_synapse_count"], 0)
        self.assertEqual(
            homeostatic_maintenance["after"]["state_revision"],
            int(status_before_homeostatic_maintenance["state_revision"]) + 1,
        )
        self.assertEqual(plasticity_runtime_state_after_maintenance["sparse_transition_weight_count"], 0)
        self.assertEqual(plasticity_runtime_state_after_maintenance["homeostatic_maintenance_count"], 1)
        self.assertGreater(plasticity_runtime_state_after_maintenance["pruned_synapse_count_total"], 0)
        self.assertEqual(
            readiness["current_spike_readout_evidence"]["surface"],
            "subcortical_spike_readout_evidence.v1",
        )
        self.assertFalse(readiness["current_spike_readout_evidence"]["generates_text"])
        self.assertEqual(
            readiness["current_decoder_probe_evidence"]["surface"],
            "snn_language_decoder_probe_evidence.v1",
        )
        self.assertFalse(readiness["current_decoder_probe_evidence"]["generates_text"])
        self.assertFalse(readiness["current_decoder_probe_evidence"]["executable"])
        self.assertIn("tensor_device", readiness["current_decoder_probe_evidence"])
        self.assertIn("mean_sparsity", readiness["current_decoder_probe_evidence"])
        self.assertIn("grounded_slot_count", readiness["current_decoder_probe_evidence"])
        self.assertEqual(
            readiness["current_language_neuron_adapter_evidence"]["surface"],
            "snn_language_neuron_adapter_evidence.v1",
        )
        self.assertFalse(readiness["current_language_neuron_adapter_evidence"]["generates_text"])
        self.assertFalse(readiness["current_language_neuron_adapter_evidence"]["executable"])
        self.assertIn("active_spike_count", readiness["current_language_neuron_adapter_evidence"])
        self.assertIn("activation_sparsity", readiness["current_language_neuron_adapter_evidence"])
        self.assertEqual(
            [candidate["name"] for candidate in readiness["research_candidates"]],
            ["NeuronSpark", "Nord-AI"],
        )
        self.assertEqual(
            readiness["research_candidates"][0]["integration_status"],
            "reference_for_marulho_owned_reimplementation",
        )
        self.assertIn(
            "marulho_owned_language_neuron_module",
            readiness["research_candidates"][0]["required_local_evidence"],
        )
        self.assertIn("marulho_native_snn_decoder", readiness["research_candidates"][0]["required_local_evidence"])
        self.assertEqual(evaluation["evaluation_cases"][0]["target"], "spike_language_neuron_adapter")
        self.assertIn("adapter_activation_sparsity_delta", evaluation["success_evidence"])
        self.assertEqual(heldout["heldout_summary"]["case_count"], 1)
        self.assertGreater(heldout["adapter_delta"]["mean_active_spike_count"], 0.0)

    def test_subcortical_self_repair_endpoint_is_gate_artifact_not_replay_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            app = create_app(_build_checkpoint(root, test_case="service_api_self_repair_gate"), trace_dir=root / "traces")
            runtime = app.state.marulho_runtime
            with TestClient(app) as client:
                before_revision = runtime.status()["state_revision"]
                before_history = runtime.action_history()["count"]
                response = client.get("/terminus/subcortical-self-repair")
                second_response = client.get("/terminus/subcortical-self-repair")
                after_revision = runtime.status()["state_revision"]
                after_history = runtime.action_history()["count"]
            app.state.marulho_manager.close()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(second_response.status_code, 200)
        body = response.json()
        self.assertEqual(body["schema_version"], 1)
        self.assertEqual(body["artifact_kind"], "terminus_subcortical_self_repair_gate_plan")
        self.assertEqual(body["endpoint"], "/terminus/subcortical-self-repair")
        self.assertEqual(body["review_role"], "operator_replay_deep_sleep_review_only")
        self.assertEqual(body["surface"], "subcortical_self_repair_candidates.v1")
        self.assertTrue(body["advisory"])
        self.assertFalse(body["executable"])
        self.assertIn("promotion_gate", body)
        self.assertFalse(body["promotion_gate"]["eligible_for_action"])
        self.assertFalse(body["promotion_gate"]["eligible_for_structural_mutation"])
        self.assertIn(body["promotion_gate"]["next_gate"], {
            "collect_spike_window",
            "deep_sleep_or_replay_repair_gate",
            "continue_monitoring",
        })
        self.assertNotIn("candidate_id", body["candidates"][0])
        self.assertNotIn("suggested_endpoint", body["candidates"][0])
        self.assertEqual(before_revision, after_revision)
        self.assertEqual(before_history, after_history)
        self.assertEqual(second_response.json()["surface"], body["surface"])

    def test_subcortical_self_repair_evaluation_endpoint_is_read_only_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            app = create_app(
                _build_checkpoint(root, test_case="service_api_self_repair_evaluation"),
                trace_dir=root / "traces",
            )
            runtime = app.state.marulho_runtime
            with TestClient(app) as client:
                before_revision = runtime.status()["state_revision"]
                before_history = runtime.action_history()["count"]
                response = client.get("/terminus/subcortical-self-repair/evaluation")
                after_revision = runtime.status()["state_revision"]
                after_history = runtime.action_history()["count"]
            app.state.marulho_manager.close()

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["schema_version"], 1)
        self.assertEqual(body["artifact_kind"], "terminus_subcortical_self_repair_evaluation_plan")
        self.assertEqual(body["surface"], "subcortical_self_repair_evaluation.v1")
        self.assertEqual(body["endpoint"], "/terminus/subcortical-self-repair/evaluation")
        self.assertTrue(body["advisory"])
        self.assertFalse(body["executable"])
        self.assertFalse(body["mutates_runtime_state"])
        self.assertIn(
            body["evaluation_gate"]["status"],
            {"ready_for_isolated_evaluation", "blocked_missing_spike_window", "monitor_only"},
        )
        self.assertFalse(body["evaluation_gate"]["eligible_for_action"])
        self.assertFalse(body["evaluation_gate"]["eligible_for_fact_promotion"])
        self.assertFalse(body["evaluation_gate"]["eligible_for_structural_mutation"])
        self.assertIn("runtime_truth_delta", body["success_evidence"])
        self.assertNotIn("suggested_endpoint", body["evaluation_cases"][0])
        self.assertNotIn("suggested_input", body["evaluation_cases"][0])
        self.assertEqual(before_revision, after_revision)
        self.assertEqual(before_history, after_history)

    def test_subcortical_structural_plasticity_endpoint_is_read_only_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            app = create_app(
                _build_checkpoint(root, test_case="service_api_structural_plasticity"),
                trace_dir=root / "traces",
            )
            runtime = app.state.marulho_runtime
            pre_snapshot = {
                "current_state_revision": 5,
                "binding_topology": {
                    "edges_added_total": 1,
                    "edges_removed_total": 0,
                    "growth_events": 1,
                    "prune_events": 0,
                },
                "device_evidence": {
                    "binding_devices": {"binding_state_device": "cuda:0"},
                    "local_plasticity_devices": {"input_eligibility_device": "cuda:0"},
                },
                "spike_health": {
                    "silent_fraction": 0.20,
                    "saturated_fraction": 0.05,
                    "stale_fraction": 0.25,
                },
                "runtime_truth": {"verdict": "degraded"},
            }
            pre_snapshot_hash = hashlib.sha256(
                json.dumps(
                    pre_snapshot,
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                    default=str,
                ).encode("utf-8")
            ).hexdigest()
            with TestClient(app) as client:
                before_revision = runtime.status()["state_revision"]
                before_history = runtime.action_history()["count"]
                response = client.get("/terminus/subcortical-structural-plasticity")
                growth_trial_response = client.get(
                    "/terminus/subcortical-structural-plasticity/binding-growth-trial",
                    params={"max_candidates": 4, "max_total_edge_delta": 8},
                )
                evaluation_response = client.post(
                    "/terminus/subcortical-structural-plasticity/evaluate",
                    json={
                        "pre_snapshot": pre_snapshot,
                        "post_snapshot": {
                            "current_state_revision": 6,
                            "binding_topology": {
                                "edges_added_total": 2,
                                "edges_removed_total": 1,
                                "growth_events": 2,
                                "prune_events": 1,
                            },
                            "device_evidence": {
                                "binding_devices": {"binding_state_device": "cuda:0"},
                                "local_plasticity_devices": {"input_eligibility_device": "cuda:0"},
                            },
                            "spike_health": {
                                "silent_fraction": 0.10,
                                "saturated_fraction": 0.02,
                                "stale_fraction": 0.20,
                            },
                            "runtime_truth": {"verdict": "alive"},
                        },
                        "rollback_policy": {
                            "available": True,
                            "snapshot_id": "pre-structural-eval",
                            "pre_snapshot_hash": pre_snapshot_hash,
                        },
                    },
                )
                design_response = client.post(
                    "/terminus/subcortical-structural-plasticity/mutation-design",
                    json={
                        "isolated_evaluation": evaluation_response.json(),
                        "operator_id": "operator-structural-design",
                        "confirmation": True,
                        "mutation_reason": "repeated isolated prediction failure",
                    },
                )
                preflight_response = client.post(
                    "/terminus/subcortical-structural-plasticity/mutation-preflight",
                    json={
                        "structural_mutation_design": design_response.json(),
                        "expected_state_revision": before_revision,
                        "checkpoint_path": str(root / "pre_structural_mutation.pt"),
                    },
                )
                blocked_application_response = client.post(
                    "/terminus/subcortical-structural-plasticity/mutation-application",
                    json={
                        "structural_mutation_preflight": preflight_response.json(),
                        "expected_state_revision": before_revision,
                        "operator_id": "operator-structural-design",
                        "confirmation": False,
                    },
                )
                after_revision = runtime.status()["state_revision"]
                after_history = runtime.action_history()["count"]
            app.state.marulho_manager.close()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(growth_trial_response.status_code, 200)
        self.assertEqual(evaluation_response.status_code, 200)
        self.assertEqual(design_response.status_code, 200)
        self.assertEqual(preflight_response.status_code, 200)
        self.assertEqual(blocked_application_response.status_code, 200)
        body = response.json()
        growth_trial = growth_trial_response.json()
        evaluation = evaluation_response.json()
        design = design_response.json()
        preflight = preflight_response.json()
        blocked_application = blocked_application_response.json()
        self.assertEqual(body["schema_version"], 1)
        self.assertEqual(body["artifact_kind"], "terminus_subcortical_structural_plasticity_gate_plan")
        self.assertEqual(body["surface"], "subcortical_structural_plasticity.v1")
        self.assertEqual(body["endpoint"], "/terminus/subcortical-structural-plasticity")
        self.assertTrue(body["advisory"])
        self.assertFalse(body["executable"])
        self.assertFalse(body["mutates_runtime_state"])
        self.assertIn(
            body["promotion_gate"]["status"],
            {
                "ready_for_isolated_structural_evaluation",
                "insufficient_device_evidence",
                "monitor_only",
            },
        )
        self.assertFalse(body["promotion_gate"]["eligible_for_action"])
        self.assertFalse(body["promotion_gate"]["eligible_for_fact_promotion"])
        self.assertFalse(body["promotion_gate"]["eligible_for_structural_mutation"])
        self.assertIn("local_plasticity", body)
        self.assertIn("local_plasticity_report_available", body["device_evidence"])
        self.assertIn("device_evidence_report", body["success_evidence"])
        self.assertIn("local_plasticity_stability_delta", body["success_evidence"])
        self.assertNotIn("suggested_endpoint", body["structural_cases"][0])
        self.assertNotIn("suggested_input", body["structural_cases"][0])
        self.assertEqual(growth_trial["surface"], "binding_growth_trial_design.v1")
        self.assertFalse(growth_trial["executable"])
        self.assertFalse(growth_trial["mutates_runtime_state"])
        self.assertFalse(growth_trial["calls_topology_refresh"])
        self.assertEqual(
            evaluation["artifact_kind"],
            "terminus_subcortical_structural_plasticity_isolated_evaluation",
        )
        self.assertEqual(evaluation["surface"], "subcortical_structural_plasticity_isolated_evaluation.v1")
        self.assertFalse(evaluation["executable"])
        self.assertFalse(evaluation["mutates_runtime_state"])
        self.assertFalse(evaluation["promotion_gate"]["eligible_for_structural_mutation"])
        self.assertEqual(evaluation["promotion_gate"]["status"], "ready_for_operator_review")
        self.assertEqual(evaluation["structural_delta"]["edges_added_delta"], 1)
        self.assertEqual(evaluation["snapshot_binding"]["hash_algorithm"], "sha256_canonical_json")
        self.assertTrue(evaluation["snapshot_binding"]["snapshot_hashes_distinct"])
        self.assertTrue(evaluation["snapshot_binding"]["structural_delta_present"])
        self.assertEqual(evaluation["snapshot_binding"]["pre_state_revision"], 5)
        self.assertEqual(evaluation["snapshot_binding"]["post_state_revision"], 6)
        self.assertTrue(evaluation["snapshot_binding"]["state_revision_order_valid"])
        self.assertFalse(evaluation["snapshot_binding"]["raw_snapshots_exposed"])
        self.assertTrue(evaluation["rollback_evidence"]["bound_to_pre_snapshot"])
        self.assertTrue(evaluation["rollback_evidence"]["pre_snapshot_hash_match"])
        self.assertTrue(evaluation["promotion_gate"]["requires_bound_snapshot_hashes"])
        self.assertTrue(evaluation["promotion_gate"]["requires_nonzero_structural_delta"])
        self.assertTrue(evaluation["promotion_gate"]["requires_rollback_pre_snapshot_binding"])
        self.assertEqual(design["artifact_kind"], "terminus_subcortical_structural_mutation_design")
        self.assertEqual(design["surface"], "subcortical_structural_mutation_design.v1")
        self.assertFalse(design["executable"])
        self.assertFalse(design["mutates_runtime_state"])
        self.assertFalse(design["writes_checkpoint"])
        self.assertFalse(design["calls_growth_or_prune"])
        self.assertFalse(design["applies_structural_mutation"])
        self.assertTrue(design["evaluation_binding"]["rollback_bound_to_pre_snapshot"])
        self.assertEqual(
            design["promotion_gate"]["status"],
            "ready_for_structural_mutation_preflight_review",
        )
        self.assertFalse(design["promotion_gate"]["eligible_for_structural_mutation"])
        self.assertTrue(
            design["promotion_gate"]["eligible_for_structural_mutation_preflight_review"]
        )
        self.assertEqual(preflight["artifact_kind"], "terminus_subcortical_structural_mutation_preflight")
        self.assertEqual(preflight["surface"], "subcortical_structural_mutation_preflight.v1")
        self.assertFalse(preflight["executable"])
        self.assertFalse(preflight["mutates_runtime_state"])
        self.assertFalse(preflight["writes_checkpoint"])
        self.assertFalse(preflight["calls_growth_or_prune"])
        self.assertFalse(preflight["applies_structural_mutation"])
        self.assertTrue(preflight["design_binding"]["design_hash_recomputed_match"])
        self.assertTrue(
            preflight["checkpoint_transaction_requirements"]["expected_state_revision_current"]
        )
        self.assertTrue(
            preflight["checkpoint_transaction_requirements"]["checkpoint_path_available"]
        )
        self.assertEqual(
            preflight["promotion_gate"]["status"],
            "ready_for_operator_structural_mutation_execution_review",
        )
        self.assertFalse(preflight["promotion_gate"]["eligible_for_structural_mutation"])
        self.assertTrue(preflight["promotion_gate"]["eligible_for_operator_execution_review"])
        self.assertEqual(
            blocked_application["surface"],
            "subcortical_structural_mutation_application.v1",
        )
        self.assertFalse(blocked_application["accepted"])
        self.assertFalse(blocked_application["writes_checkpoint"])
        self.assertFalse(blocked_application["calls_growth_or_prune"])
        self.assertFalse(blocked_application["applies_structural_mutation"])
        self.assertFalse(
            blocked_application["promotion_gate"]["required_evidence"]["confirmation"]
        )
        self.assertEqual(before_revision, after_revision)
        self.assertEqual(before_history, after_history)

    def test_validation_report_endpoints_list_and_read_reports(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            fresh_generated_at = datetime.now(timezone.utc).isoformat()
            stale_generated_at = (datetime.now(timezone.utc) - timedelta(hours=96)).isoformat()
            report_dir = root / "reports" / "phase15"
            report_dir.mkdir(parents=True)
            report_path = report_dir / "phase15.json"
            readme_path = report_dir / "README.md"
            report_path.write_text(
                json.dumps(
                    {
                        "artifact_kind": "terminus_bounded_self_improvement_readiness",
                        "status": "ready_for_bounded_level_5_experiment",
                        "passed": True,
                        "operator_visible_report": {"summary": "ready"},
                    }
                ),
                encoding="utf-8",
            )
            regression_dir = root / "reports" / "service_benchmark_regression_gate"
            regression_dir.mkdir(parents=True)
            regression_path = regression_dir / "comparison.json"
            regression_path.write_text(
                json.dumps(
                    {
                        "artifact_kind": "marulho_service_benchmark_regression_gate",
                        "status": "passed",
                        "generated_at": fresh_generated_at,
                        "runtime_truth": {
                            "before": "alive",
                            "after": "alive",
                            "regressed": False,
                        },
                        "hot_path": {
                            "after_p95_ms": 439.258,
                            "after_total_ms": 818.798,
                            "allowed_after_p95_ms": 549.072,
                            "allowed_after_total_ms": 1023.497,
                            "regression_tolerance": 0.25,
                        },
                        "endpoint_grouping": {
                            "setup_leaked_into_hot_path": False,
                            "slow_path_leaked_into_hot_path": False,
                        },
                        "configured_source": {
                            "source_name": "benchmark_local_source",
                            "tick_tokens_processed": 24,
                        },
                        "accepted_baseline": {
                            "baseline_id": "service-benchmark-baseline:abc123",
                            "label": "configured-source-cpu",
                            "accepted_by": "operator-a",
                            "baseline_path": "reports/service_benchmark_baseline/accepted-baseline.json",
                        },
                        "checks": {
                            "runtime_truth_no_regression": True,
                            "configured_source_alive": True,
                            "hot_path_p95_no_relative_regression": True,
                        },
                    }
                ),
                encoding="utf-8",
            )
            readme_path.write_text("# Phase 15\n", encoding="utf-8")
            baseline_dir = root / "reports" / "service_benchmark_baseline"
            baseline_dir.mkdir(parents=True)
            baseline_snapshot = {
                "benchmark": "marulho_service_endpoint_latency",
                "success": True,
                "endpoint_metabolism_summary": {
                    "hot_path": {
                        "latency_ms_p95": 439.258,
                        "latency_ms_total": 818.798,
                    }
                },
            }
            baseline_hash = _sha256_json(baseline_snapshot)
            acceptance_material = {
                "baseline_id": "service-benchmark-baseline:abc123",
                "label": "configured-source-cpu",
                "accepted_by": "operator-a",
                "note": "accepted baseline for local regression checks",
                "accepted_at": fresh_generated_at,
                "source_report_sha256_canonical_json": baseline_hash,
                "source_report_generated_at": "",
                "runtime_truth_verdict": "alive",
                "hot_path_p95_ms": 439.258,
                "hot_path_total_ms": 818.798,
            }
            acceptance_hash = _sha256_json(acceptance_material)
            (baseline_dir / "accepted-baseline.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "artifact_kind": "marulho_service_benchmark_accepted_baseline",
                        "generated_at": fresh_generated_at,
                        "status": "accepted",
                        "baseline_id": "service-benchmark-baseline:abc123",
                        "label": "configured-source-cpu",
                        "operator_review": {
                            "accepted_by": "operator-a",
                            "note": "accepted baseline for local regression checks",
                            "accepted_at": fresh_generated_at,
                            "acceptance_material": acceptance_material,
                            "acceptance_hash": acceptance_hash,
                            "acceptance_hash_algorithm": "sha256_canonical_json",
                        },
                        "checks": {
                            "accepted_by_present": True,
                            "benchmark_success": True,
                            "runtime_truth_known": True,
                            "hot_path_p95_available": True,
                            "hot_path_total_available": True,
                        },
                        "source_report": {
                            "path": "reports/service_benchmark_cycle_configured/service-benchmark.json",
                            "sha256_canonical_json": baseline_hash,
                            "runtime_truth_verdict": "alive",
                            "hot_path_p95_ms": 439.258,
                            "hot_path_total_ms": 818.798,
                        },
                        "baseline_report_snapshot": baseline_snapshot,
                    }
                ),
                encoding="utf-8",
            )
            tampered_dir = root / "reports" / "service_benchmark_baseline_tampered"
            tampered_dir.mkdir(parents=True)
            (tampered_dir / "accepted-baseline.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "artifact_kind": "marulho_service_benchmark_accepted_baseline",
                        "generated_at": stale_generated_at,
                        "status": "accepted",
                        "baseline_id": "service-benchmark-baseline:tampered",
                        "label": "tampered",
                        "operator_review": {
                            "accepted_by": "operator-a",
                            "acceptance_material": {
                                **acceptance_material,
                                "baseline_id": "service-benchmark-baseline:tampered",
                            },
                            "acceptance_hash": acceptance_hash,
                            "acceptance_hash_algorithm": "sha256_canonical_json",
                        },
                        "checks": {"accepted_by_present": True},
                        "source_report": {
                            "path": "reports/service_benchmark_cycle_configured/service-benchmark.json",
                            "sha256_canonical_json": baseline_hash,
                            "runtime_truth_verdict": "alive",
                            "hot_path_p95_ms": 439.258,
                            "hot_path_total_ms": 818.798,
                        },
                        "baseline_report_snapshot": {
                            **baseline_snapshot,
                            "success": False,
                        },
                    }
                ),
                encoding="utf-8",
            )
            bundle_dir = root / "reports" / "service_benchmark_baseline_fresh_cycle"
            bundle_dir.mkdir(parents=True)
            (bundle_dir / "bundle-summary.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "artifact_kind": "marulho_service_benchmark_baseline_run_bundle",
                        "generated_at": fresh_generated_at,
                        "status": "passed",
                        "success": True,
                        "paths": {
                            "bundle_dir": "reports/service_benchmark_baseline_fresh_cycle",
                            "benchmark": "reports/service_benchmark_baseline_fresh_cycle/fresh-benchmark.json",
                            "comparison": "reports/service_benchmark_baseline_fresh_cycle/comparison.json",
                            "baseline": "reports/service_benchmark_baseline/accepted-baseline.json",
                        },
                        "accepted_baseline": {
                            "baseline_id": "service-benchmark-baseline:abc123",
                            "label": "configured-source-cpu",
                            "accepted_by": "operator-a",
                            "baseline_report_hash": baseline_hash,
                            "after_report_hash": "f" * 64,
                        },
                        "runtime_truth": {
                            "before": "alive",
                            "after": "alive",
                            "regressed": False,
                        },
                        "hot_path": {
                            "after_p95_ms": 432.406,
                            "after_total_ms": 739.666,
                            "allowed_after_p95_ms": 549.072,
                            "allowed_after_total_ms": 1023.497,
                            "regression_tolerance": 0.25,
                        },
                        "configured_source": {
                            "source_name": "benchmark_local_source",
                            "tick_tokens_processed": 24,
                            "source_count": 1,
                        },
                        "checks": {
                            "runtime_truth_no_regression": True,
                            "configured_source_alive": True,
                            "hot_path_p95_no_relative_regression": True,
                        },
                        "claim_boundary": "fresh_benchmark_plus_baseline_compare_slow_path_no_runtime_mutation_no_cuda_speedup_claim",
                    }
                ),
                encoding="utf-8",
            )
            app = create_app(
                _build_checkpoint(root, test_case="service_api_validation_reports"),
                trace_dir=root / "traces",
                env_root=root,
            )
            with TestClient(app) as client:
                list_response = client.get("/terminus/validation/reports")
                read_response = client.get("/terminus/validation/report", params={"path": "phase15/README.md"})
            app.state.marulho_manager.close()

        self.assertEqual(list_response.status_code, 200)
        listed = list_response.json()
        self.assertEqual(listed["phase_status"]["phase15"]["status"], "ready_for_bounded_level_5_experiment")
        regression_summary = next(
            item
            for item in listed["reports"]
            if item["artifact_kind"] == "marulho_service_benchmark_regression_gate"
        )
        self.assertEqual(regression_summary["status"], "passed")
        self.assertEqual(regression_summary["runtime_truth_verdict"], "alive")
        self.assertEqual(regression_summary["runtime_truth_before"], "alive")
        self.assertFalse(regression_summary["runtime_truth_regressed"])
        self.assertEqual(regression_summary["hot_path_p95_ms"], 439.258)
        self.assertEqual(regression_summary["hot_path_allowed_p95_ms"], 549.072)
        self.assertEqual(regression_summary["configured_source"], "benchmark_local_source")
        self.assertEqual(regression_summary["configured_source_tick_tokens"], 24)
        self.assertEqual(regression_summary["accepted_baseline_id"], "service-benchmark-baseline:abc123")
        self.assertEqual(regression_summary["accepted_baseline_label"], "configured-source-cpu")
        self.assertEqual(regression_summary["accepted_baseline_by"], "operator-a")
        self.assertEqual(regression_summary["evidence_freshness_status"], "fresh")
        self.assertLessEqual(regression_summary["evidence_age_hours"], 1.0)
        self.assertEqual(regression_summary["failed_checks"], [])
        baseline_summary = next(
            item
            for item in listed["reports"]
            if item["artifact_kind"] == "marulho_service_benchmark_accepted_baseline"
            and item.get("accepted_baseline_id") == "service-benchmark-baseline:abc123"
        )
        self.assertEqual(baseline_summary["status"], "accepted")
        self.assertEqual(baseline_summary["accepted_baseline_id"], "service-benchmark-baseline:abc123")
        self.assertEqual(baseline_summary["accepted_baseline_label"], "configured-source-cpu")
        self.assertEqual(baseline_summary["accepted_baseline_by"], "operator-a")
        self.assertEqual(baseline_summary["baseline_report_hash"], baseline_hash)
        self.assertEqual(baseline_summary["baseline_snapshot_hash"], baseline_hash)
        self.assertTrue(baseline_summary["baseline_hash_match"])
        self.assertEqual(baseline_summary["baseline_integrity_status"], "verified")
        self.assertEqual(baseline_summary["acceptance_hash"], acceptance_hash)
        self.assertEqual(baseline_summary["acceptance_material_hash"], acceptance_hash)
        self.assertTrue(baseline_summary["acceptance_hash_match"])
        self.assertEqual(baseline_summary["acceptance_integrity_status"], "verified")
        self.assertEqual(baseline_summary["evidence_freshness_status"], "fresh")
        self.assertLessEqual(baseline_summary["evidence_age_hours"], 1.0)
        self.assertIn("--run-against-baseline", baseline_summary["baseline_operator_action_hint"])
        self.assertEqual(len(baseline_summary["baseline_operator_action_commands"]), 1)
        self.assertIn("--run-against-baseline", baseline_summary["baseline_operator_action_commands"][0])
        self.assertEqual(baseline_summary["runtime_truth_verdict"], "alive")
        self.assertEqual(baseline_summary["hot_path_p95_ms"], 439.258)
        self.assertEqual(baseline_summary["hot_path_total_ms"], 818.798)
        self.assertEqual(baseline_summary["failed_checks"], [])
        tampered_summary = next(
            item
            for item in listed["reports"]
            if item.get("accepted_baseline_id") == "service-benchmark-baseline:tampered"
        )
        self.assertEqual(tampered_summary["baseline_integrity_status"], "failed")
        self.assertFalse(tampered_summary["baseline_hash_match"])
        self.assertEqual(tampered_summary["acceptance_integrity_status"], "failed")
        self.assertFalse(tampered_summary["acceptance_hash_match"])
        self.assertEqual(tampered_summary["evidence_freshness_status"], "stale")
        self.assertGreaterEqual(tampered_summary["evidence_age_hours"], 96.0)
        self.assertIn("baseline_snapshot_hash_match", tampered_summary["failed_checks"])
        self.assertIn("baseline_acceptance_hash_match", tampered_summary["failed_checks"])
        self.assertIn("hash mismatch", tampered_summary["baseline_operator_action_hint"])
        self.assertEqual(len(tampered_summary["baseline_operator_action_commands"]), 2)
        self.assertIn("--accept-baseline-from", tampered_summary["baseline_operator_action_commands"][1])
        bundle_summary = next(
            item
            for item in listed["reports"]
            if item["artifact_kind"] == "marulho_service_benchmark_baseline_run_bundle"
        )
        self.assertEqual(bundle_summary["status"], "passed")
        self.assertTrue(bundle_summary["success"])
        self.assertTrue(bundle_summary["passed"])
        self.assertEqual(bundle_summary["runtime_truth_verdict"], "alive")
        self.assertFalse(bundle_summary["runtime_truth_regressed"])
        self.assertEqual(bundle_summary["hot_path_p95_ms"], 432.406)
        self.assertEqual(bundle_summary["hot_path_allowed_total_ms"], 1023.497)
        self.assertEqual(bundle_summary["configured_source"], "benchmark_local_source")
        self.assertEqual(bundle_summary["configured_source_tick_tokens"], 24)
        self.assertEqual(bundle_summary["accepted_baseline_id"], "service-benchmark-baseline:abc123")
        self.assertEqual(bundle_summary["baseline_report_hash"], baseline_hash)
        self.assertEqual(bundle_summary["after_report_hash"], "f" * 64)
        self.assertEqual(bundle_summary["evidence_freshness_status"], "fresh")
        self.assertLessEqual(bundle_summary["evidence_age_hours"], 1.0)
        self.assertEqual(
            bundle_summary["fresh_benchmark_path"],
            "reports/service_benchmark_baseline_fresh_cycle/fresh-benchmark.json",
        )
        self.assertEqual(bundle_summary["failed_checks"], [])
        phase_summary = next(
            item
            for item in listed["reports"]
            if item["artifact_kind"] == "terminus_bounded_self_improvement_readiness"
        )
        self.assertEqual(phase_summary["readme_path"], "phase15/README.md")
        self.assertEqual(read_response.status_code, 200)
        self.assertEqual(read_response.json()["media_type"], "text/markdown")
        self.assertIn("Phase 15", read_response.json()["content"])

    def test_static_ui_default_points_to_built_frontend_dist(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ckpt = _build_checkpoint(root, test_case="service_api_static_default")
            app = create_app(ckpt, trace_dir=root / "traces")
            self.assertEqual(app.state.web_dist_dir, DEFAULT_WEB_DIST_DIR)
            app.state.marulho_manager.close()

        self.assertEqual(DEFAULT_WEB_DIST_DIR, Path("MARULHO_UI") / "dist")

        parser = build_arg_parser()
        args = parser.parse_args(["--checkpoint", "checkpoints\\terminus\\model.pt"])
        self.assertEqual(args.web_dist_dir, DEFAULT_WEB_DIST_DIR)

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
                        "source_concept_observation_tick_interval": 3,
                        "sleep_interval_seconds": 0.01,
                        "execution_quantum_tokens": 5,
                        "execution_yield_seconds": 0.0,
                        "repeat_sources": False,
                        "ingestion": {"queue_target_tokens": 40, "prewarm_on_startup": False, "prewarm_max_seconds": 0.2},
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
            self.assertEqual(
                status_response.json()["terminus_runtime"]["source_concept_observation_tick_interval"],
                3,
            )
            self.assertEqual(
                status_response.json()["terminus_runtime"]["execution_schedule"],
                {
                    "quantum_tokens": 5,
                    "yield_seconds": 0.0,
                    "stop_check_boundary": "between_quanta",
                    "sequential_token_training": True,
                },
            )
            self.assertEqual(status_response.json()["terminus_runtime"]["ingestion"]["queue_target_tokens"], 40)
            self.assertFalse(status_response.json()["terminus_runtime"]["ingestion"]["prewarm_on_startup"])
            self.assertAlmostEqual(float(status_response.json()["terminus_runtime"]["ingestion"]["prewarm_max_seconds"]), 0.2, places=6)
            self.assertEqual(status_response.json()["replay_dataset_summary"]["endpoint"], "/terminus/replay-dataset/preview")
            self.assertEqual(
                status_response.json()["terminus_runtime"]["living_loop"]["replay_dataset_summary"]["endpoint"],
                "/terminus/replay-dataset/preview",
            )
            self.assertGreater(tick_response.json()["terminus_runtime"]["last_tick_token_delta"], 0)
            self.assertTrue(
                any(event.get("type") == "tick" for event in status_response.json()["terminus_runtime"]["recent_events"])
            )
            self.assertGreater(
                status_response.json()["terminus_runtime"]["source_progress"][0]["tick_visits"],
                0,
            )

    def test_terminus_action_endpoint_executes_workspace_search_and_records_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "notes.md").write_text(
                "Cats rest indoors during the day.\nCats chase mice at night.\n",
                encoding="utf-8",
            )
            app = create_app(
                _build_checkpoint(root, test_case="service_api_terminus_action"),
                trace_dir=root / "traces",
                env_root=root,
            )
            with TestClient(app) as client:
                action_response = client.post(
                    "/terminus/action",
                    json={
                        "action_type": "workspace_search",
                        "query_text": "cats chase mice",
                        "predicted_outcome": "I expect to find evidence about cats chasing mice.",
                    },
                )
                history_response = client.get("/terminus/actions")

            self.assertEqual(action_response.status_code, 200)
            self.assertEqual(history_response.status_code, 200)
            action_body = action_response.json()
            history_body = history_response.json()
            self.assertTrue(action_body["accepted"])
            self.assertEqual(action_body["result"]["verification"]["status"], "verified")
            self.assertEqual(action_body["terminus_runtime"]["action_loop"]["verified_actions"], 1)
            self.assertEqual(action_body["terminus_runtime"]["action_loop"]["ledger_scope"], "subcortex_action_ledger")
            self.assertNotIn("retired_loop_sync", action_body["terminus_runtime"]["action_loop"])
            self.assertEqual(history_body["count"], 1)
            self.assertEqual(history_body["actions"][0]["action_type"], "workspace_search")
            self.assertEqual(history_body["actions"][0]["verification"]["status"], "verified")

    def test_policy_actuator_endpoint_is_advisory_and_does_not_mutate_action_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "notes.md").write_text(
                "Cats rest indoors during the day.\nCats chase mice at night.\n",
                encoding="utf-8",
            )
            app = create_app(
                _build_checkpoint(root, test_case="service_api_policy_actuator"),
                trace_dir=root / "traces",
                env_root=root,
            )
            manager = app.state.marulho_manager
            runtime = app.state.marulho_runtime
            with TestClient(app) as client:
                action_response = client.post(
                    "/terminus/action",
                    json={
                        "action_type": "workspace_search",
                        "query_text": "cats chase mice",
                        "predicted_outcome": "I expect to find evidence about cats chasing mice.",
                    },
                )
                before_history = runtime.action_history()["count"]
                before_revision = runtime.status()["state_revision"]
                policy_response = client.get("/terminus/policy-actuator")
                living_response = client.get("/terminus/living-loop")
                after_history = runtime.action_history()["count"]
                after_revision = runtime.status()["state_revision"]

        self.assertEqual(action_response.status_code, 200)
        self.assertEqual(policy_response.status_code, 200)
        self.assertEqual(living_response.status_code, 200)
        body = policy_response.json()
        living_body = living_response.json()
        self.assertEqual(body["schema_version"], 1)
        self.assertEqual(body["action"], "continue_current_policy")
        self.assertTrue(body["advisory"])
        self.assertFalse(body["executable"])
        self.assertIsNone(body["target_episode_id"])
        self.assertIsNone(body["target_action_id"])
        self.assertIsNone(body["action_id"])
        self.assertIn("suggested_endpoint", body)
        self.assertIn("suggested_input", body)
        self.assertIn("input", body)
        self.assertEqual(body["subcortical_control_candidates"]["surface"], "subcortical_control_candidates.v1")
        self.assertTrue(body["subcortical_control_candidates"]["advisory"])
        self.assertFalse(body["subcortical_control_candidates"]["executable"])
        self.assertFalse(body["subcortical_control_candidates"]["promotion_summary"]["eligible_for_action"])
        self.assertFalse(body["subcortical_control_candidates"]["promotion_summary"]["eligible_for_fact_promotion"])
        self.assertEqual(
            body["subcortical_self_repair_candidates"]["surface"],
            "subcortical_self_repair_candidates.v1",
        )
        self.assertTrue(body["subcortical_self_repair_candidates"]["advisory"])
        self.assertFalse(body["subcortical_self_repair_candidates"]["executable"])
        self.assertFalse(
            body["subcortical_self_repair_candidates"]["promotion_summary"]["eligible_for_structural_mutation"]
        )
        self.assertFalse(
            body["subcortical_self_repair_candidates"]["promotion_gate"]["eligible_for_structural_mutation"]
        )
        self.assertEqual(
            living_body["living_loop"]["subcortical_control_candidates"]["surface"],
            "subcortical_control_candidates.v1",
        )
        self.assertFalse(
            living_body["living_loop"]["subcortical_control_candidates"]["promotion_summary"]["eligible_for_action"]
        )
        self.assertEqual(
            living_body["living_loop"]["subcortical_self_repair_candidates"]["surface"],
            "subcortical_self_repair_candidates.v1",
        )
        self.assertFalse(
            living_body["living_loop"]["subcortical_self_repair_candidates"]["promotion_summary"][
                "eligible_for_structural_mutation"
            ]
        )
        self.assertFalse(
            living_body["living_loop"]["subcortical_self_repair_candidates"]["promotion_gate"][
                "eligible_for_structural_mutation"
            ]
        )
        self.assertEqual(before_history, after_history)
        self.assertEqual(before_revision, after_revision)

    def test_replay_plan_endpoint_is_advisory_and_does_not_mutate_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            app = create_app(
                _build_checkpoint(root, test_case="service_api_replay_plan"),
                trace_dir=root / "traces",
                env_root=root,
            )
            manager = app.state.marulho_manager
            runtime = app.state.marulho_runtime
            with TestClient(app) as client:
                feed_response = client.post("/feed", json={"text": "Cats chase mice at night."})
                episode_id = feed_response.json()["runtime_episode"]["episode_id"]
                feedback_response = client.post(
                    "/terminus/runtime-feedback",
                    json={
                        "target_type": "runtime_episode",
                        "target_id": episode_id,
                        "verdict": "contradicted",
                        "confidence": 0.91,
                        "summary": "Manual review contradicted this episode.",
                        "corrected_output": {"summary": "Cats chase mice at night."},
                    },
                )
                before_revision = runtime.status()["state_revision"]
                before_history = runtime.action_history()["count"]
                replay_response = client.get("/terminus/replay-plan?limit=5")
                after_revision = runtime.status()["state_revision"]
                after_history = runtime.action_history()["count"]

        self.assertEqual(feed_response.status_code, 200)
        self.assertEqual(feedback_response.status_code, 200)
        self.assertEqual(replay_response.status_code, 200)
        body = replay_response.json()
        self.assertEqual(body["schema_version"], 1)
        self.assertTrue(body["advisory"])
        self.assertFalse(body["executable"])
        self.assertEqual(body["endpoint"], "/terminus/replay-plan")
        self.assertGreaterEqual(body["count"], 1)
        top = body["candidates"][0]
        self.assertNotIn("subcortical_control_candidates", top)
        self.assertEqual(top["target_type"], "runtime_episode")
        self.assertEqual(top["target_id"], episode_id)
        self.assertIn("contradicted_feedback", top["reason_codes"])
        self.assertEqual(top["suggested_consolidation_action"], "review_contradiction")
        self.assertEqual(before_revision, after_revision)
        self.assertEqual(before_history, after_history)

    def test_replay_sample_endpoint_records_audit_only_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            app = create_app(
                _build_checkpoint(root, test_case="service_api_replay_sample"),
                trace_dir=root / "traces",
                env_root=root,
            )
            manager = app.state.marulho_manager
            runtime = app.state.marulho_runtime
            with TestClient(app) as client:
                feed_response = client.post("/feed", json={"text": "Cats chase mice at night."})
                episode_id = feed_response.json()["runtime_episode"]["episode_id"]
                feedback_response = client.post(
                    "/terminus/runtime-feedback",
                    json={
                        "target_type": "runtime_episode",
                        "target_id": episode_id,
                        "verdict": "contradicted",
                        "confidence": 0.91,
                        "summary": "Manual review contradicted this episode.",
                        "corrected_output": {"summary": "Cats chase mice at night."},
                    },
                )
                plan_response = client.get("/terminus/replay-plan?limit=5")
                candidate = plan_response.json()["candidates"][0]
                before_revision = runtime.status()["state_revision"]
                before_history = runtime.action_history()["count"]
                rejected_response = client.post(
                    "/terminus/replay-sample",
                    json={
                        "mode": "sample",
                        "candidate_id": candidate["candidate_id"],
                        "operator_id": "operator-a",
                        "confirmation": False,
                    },
                )
                sample_response = client.post(
                    "/terminus/replay-sample",
                    json={
                        "mode": "sample",
                        "candidate_id": candidate["candidate_id"],
                        "target_type": "runtime_episode",
                        "target_id": episode_id,
                        "operator_id": "operator-a",
                        "operator_note": "Audit contradicted replay candidate only.",
                        "confirmation": True,
                        "seed": 123,
                    },
                )
                history_response = client.get("/terminus/replay-sample/history?limit=5")
                alias_history_response = client.get("/terminus/replay-execute/history?limit=5")
                living_response = client.get("/terminus/living-loop")
                export_response = client.get("/terminus/runtime-traces/export?limit=5")
                replay_dataset_response = client.get("/terminus/replay-dataset/preview?limit=5")
                rejected_bundle_response = client.post(
                    "/terminus/replay-dataset/bundle",
                    json={
                        "operator_id": "operator-a",
                        "confirmation": False,
                        "limit": 5,
                    },
                )
                replay_dataset_bundle_response = client.post(
                    "/terminus/replay-dataset/bundle",
                    json={
                        "operator_id": "operator-a",
                        "operator_note": "Package preview only for offline review.",
                        "confirmation": True,
                        "limit": 5,
                        "holdout_fraction": 0.0,
                        "eval_fraction": 0.0,
                        "seed": 123,
                    },
                )
                replay_dataset_candidates_response = client.get("/terminus/replay-dataset/candidates?limit=5")
                replay_dataset_history_response = client.get("/terminus/replay-dataset/history?limit=5")
                seeded_sample_a = manager._sample_replay_candidates(
                    [
                        {"candidate_id": "a", "priority_score": 100.0, "target_type": "runtime_episode"},
                        {"candidate_id": "b", "priority_score": 95.0, "target_type": "runtime_episode"},
                        {"candidate_id": "c", "priority_score": 90.0, "target_type": "action"},
                    ],
                    count=2,
                    alpha=1.0,
                    seed=1,
                )
                seeded_sample_b = manager._sample_replay_candidates(
                    [
                        {"candidate_id": "a", "priority_score": 100.0, "target_type": "runtime_episode"},
                        {"candidate_id": "b", "priority_score": 95.0, "target_type": "runtime_episode"},
                        {"candidate_id": "c", "priority_score": 90.0, "target_type": "action"},
                    ],
                    count=2,
                    alpha=1.0,
                    seed=1,
                )
                after_revision = runtime.status()["state_revision"]
                after_history = runtime.action_history()["count"]

        self.assertEqual(feed_response.status_code, 200)
        self.assertEqual(feedback_response.status_code, 200)
        self.assertEqual(plan_response.status_code, 200)
        self.assertEqual(rejected_response.status_code, 422)
        self.assertEqual(sample_response.status_code, 200)
        body = sample_response.json()
        self.assertEqual(body["schema_version"], 1)
        self.assertEqual(body["mode"], "sample")
        self.assertEqual(body["status"], "recorded")
        self.assertEqual(body["operator_id"], "operator-a")
        self.assertEqual(body["selected_candidate_ids"], [candidate["candidate_id"]])
        self.assertTrue(body["safety_checks"]["passed"])
        self.assertTrue(body["safety_flags"]["audit_only"])
        self.assertFalse(body["safety_flags"]["training_started"])
        self.assertFalse(body["safety_flags"]["sleep_started"])
        self.assertFalse(body["safety_flags"]["feedback_posted"])
        self.assertFalse(body["safety_flags"]["digital_action_executed"])
        self.assertFalse(body["safety_flags"]["external_calls_made"])
        self.assertTrue(body["selected_candidates"][0]["safety"]["not_promoted"])
        self.assertTrue(body["selected_candidates"][0]["safety"]["non_factual"])
        self.assertEqual(body["before"]["state_revision"], body["after"]["state_revision"])
        self.assertEqual(body["before"]["token_count"], body["after"]["token_count"])
        self.assertEqual(body["before"]["action_history_count"], body["after"]["action_history_count"])
        self.assertEqual(body["before"]["feedback_count"], body["after"]["feedback_count"])
        self.assertEqual(before_revision, after_revision)
        self.assertEqual(before_history, after_history)
        self.assertEqual(history_response.status_code, 200)
        history = history_response.json()
        self.assertEqual(history["count"], 1)
        self.assertEqual(history["history"][0]["replay_sample_id"], body["replay_sample_id"])
        self.assertEqual(alias_history_response.status_code, 200)
        self.assertEqual(alias_history_response.json()["history"][0]["replay_sample_id"], body["replay_sample_id"])
        self.assertEqual(living_response.status_code, 200)
        living_loop = living_response.json()["living_loop"]
        replay_summary = living_loop["replay_sample_summary"]
        self.assertEqual(replay_summary["endpoint"], "/terminus/replay-sample")
        self.assertEqual(replay_summary["execution_endpoint"], "/terminus/replay-execute")
        self.assertEqual(replay_summary["history_endpoint"], "/terminus/replay-sample/history")
        self.assertEqual(replay_summary["count"], 1)
        self.assertEqual(replay_summary["mode_counts"]["sample"], 1)
        self.assertEqual(replay_summary["status_counts"]["recorded"], 1)
        self.assertEqual(replay_summary["latest_selected_count"], 1)
        self.assertTrue(replay_summary["safety_flags"]["audit_only"])
        self.assertFalse(replay_summary["safety_flags"]["external_calls_made"])
        self.assertEqual(living_loop["benchmark_telemetry"]["replay_sample_summary"]["count"], 1)
        self.assertEqual(living_loop["replay_executor_summary"]["count"], 1)
        living_dataset_summary = living_loop["replay_dataset_summary"]
        self.assertEqual(living_dataset_summary["export_kind"], "terminus_replay_dataset_preview")
        self.assertEqual(living_dataset_summary["endpoint"], "/terminus/replay-dataset/preview")
        self.assertGreaterEqual(living_dataset_summary["positive_count"], 1)
        self.assertGreaterEqual(living_dataset_summary["negative_count"], 1)
        self.assertEqual(living_dataset_summary["latest_history_timestamp"], body["created_at"])
        self.assertEqual(
            living_loop["benchmark_telemetry"]["replay_dataset_summary"]["endpoint"],
            "/terminus/replay-dataset/preview",
        )
        self.assertEqual(export_response.status_code, 200)
        export_body = export_response.json()
        self.assertEqual(export_body["replay_sample_summary"]["count"], 1)
        self.assertEqual(export_body["replay_dataset_summary"]["endpoint"], "/terminus/replay-dataset/preview")
        self.assertEqual(export_body["replay_dataset_summary"]["latest_history_timestamp"], body["created_at"])
        self.assertTrue(export_body["replay_sample_summary"]["safety_flags"]["audit_only"])
        if export_body["examples"]:
            self.assertEqual(export_body["examples"][0]["replay_sample_summary"]["count"], 1)
            self.assertTrue(export_body["examples"][0]["replay_sample_summary"]["safety_flags"]["audit_only"])
        self.assertEqual(replay_dataset_response.status_code, 200)
        replay_dataset = replay_dataset_response.json()
        self.assertEqual(replay_dataset["export_kind"], "terminus_replay_dataset_preview")
        self.assertEqual(replay_dataset["training_role"], "replay_dataset_preview_only_not_training_no_mutation")
        self.assertGreaterEqual(replay_dataset["count"], 1)
        self.assertGreaterEqual(replay_dataset["positive_count"], 1)
        self.assertGreaterEqual(replay_dataset["negative_count"], 1)
        self.assertEqual(replay_dataset["endpoint"], "/terminus/replay-dataset/preview")
        self.assertIsNotNone(replay_dataset["latest_export_timestamp"])
        self.assertEqual(replay_dataset["latest_history_timestamp"], body["created_at"])
        self.assertFalse(replay_dataset["safety_flags"]["training_started"])
        self.assertFalse(replay_dataset["safety_flags"]["memory_mutated"])
        self.assertFalse(replay_dataset["safety_flags"]["feedback_posted"])
        self.assertFalse(replay_dataset["safety_flags"]["digital_action_executed"])
        self.assertFalse(replay_dataset["safety_flags"]["external_calls_made"])
        dataset_item = next(
            item
            for item in replay_dataset["items"]
            if item["target_id"] == episode_id
        )
        self.assertEqual(dataset_item["verification_label"], "contradicted")
        self.assertFalse(dataset_item["is_verified_fact"])
        self.assertTrue(dataset_item["has_positive_example"])
        self.assertTrue(dataset_item["has_negative_example"])
        self.assertEqual(dataset_item["sft_example"]["output_source"], "corrected_output")
        self.assertEqual(dataset_item["preference_pair"]["chosen_source"], "corrected_output")
        self.assertIn("contradicted", dataset_item["preference_pair"]["rejected_source"])
        self.assertTrue(dataset_item["replay_sample_linkage"]["selected"])
        self.assertEqual(dataset_item["replay_sample_linkage"]["replay_sample_ids"], [body["replay_sample_id"]])
        self.assertFalse(dataset_item["safety_flags"]["eligible_for_training"])
        self.assertEqual(replay_dataset_candidates_response.status_code, 200)
        self.assertEqual(replay_dataset_candidates_response.json()["export_kind"], "terminus_replay_dataset_candidates_preview")
        self.assertGreaterEqual(replay_dataset_candidates_response.json()["count"], 1)
        self.assertEqual(replay_dataset_history_response.status_code, 200)
        self.assertEqual(replay_dataset_history_response.json()["export_kind"], "terminus_replay_dataset_history_preview")
        self.assertEqual(replay_dataset_history_response.json()["history"][0]["replay_sample_id"], body["replay_sample_id"])
        self.assertEqual(rejected_bundle_response.status_code, 422)
        self.assertEqual(replay_dataset_bundle_response.status_code, 200)
        replay_dataset_bundle = replay_dataset_bundle_response.json()
        self.assertEqual(replay_dataset_bundle["export_kind"], "terminus_replay_dataset_bundle_preview")
        self.assertEqual(
            replay_dataset_bundle["training_role"],
            "replay_dataset_bundle_preview_only_not_training_operator_approved",
        )
        self.assertEqual(replay_dataset_bundle["endpoint"], "/terminus/replay-dataset/bundle")
        self.assertEqual(replay_dataset_bundle["source_endpoint"], "/terminus/replay-dataset/preview")
        self.assertTrue(replay_dataset_bundle["operator_approval"]["approved"])
        self.assertEqual(replay_dataset_bundle["operator_approval"]["operator_id"], "operator-a")
        self.assertGreaterEqual(replay_dataset_bundle["count"], 1)
        self.assertGreaterEqual(replay_dataset_bundle["preference_pair_count"], 1)
        self.assertEqual(replay_dataset_bundle["split_counts"]["holdout"], 0)
        self.assertEqual(replay_dataset_bundle["split_counts"]["eval"], 0)
        self.assertEqual(replay_dataset_bundle["split_counts"]["train"], replay_dataset_bundle["count"])
        self.assertFalse(replay_dataset_bundle["safety_flags"]["training_started"])
        self.assertFalse(replay_dataset_bundle["safety_flags"]["memory_mutated"])
        self.assertFalse(replay_dataset_bundle["safety_flags"]["feedback_posted"])
        self.assertFalse(replay_dataset_bundle["safety_flags"]["digital_action_executed"])
        self.assertFalse(replay_dataset_bundle["safety_flags"]["external_calls_made"])
        self.assertTrue(replay_dataset_bundle["safety_flags"]["requires_separate_training_approval"])
        self.assertEqual(replay_dataset_bundle["training_gate"]["status"], "blocked_preview_only")
        self.assertFalse(replay_dataset_bundle["training_gate"]["eligible_for_training"])
        self.assertFalse(replay_dataset_bundle["training_gate"]["satisfied_conditions"]["offline_regression_benchmark"])
        self.assertFalse(replay_dataset_bundle["training_gate"]["satisfied_conditions"]["explicit_operator_training_approval"])
        bundled_item = next(
            item
            for item in replay_dataset_bundle["splits"]["train"]
            if item["target_id"] == episode_id
        )
        self.assertEqual(bundled_item["verification_label"], "contradicted")
        self.assertEqual(bundled_item["split"], "train")
        self.assertIsNotNone(bundled_item["preference_pair"])
        self.assertIn("bundle_hash", replay_dataset_bundle["manifest"])
        self.assertEqual(
            [candidate["candidate_id"] for candidate in seeded_sample_a],
            [candidate["candidate_id"] for candidate in seeded_sample_b],
        )
        self.assertEqual(
            {candidate["target_type"] for candidate in seeded_sample_a},
            {"runtime_episode", "action"},
        )

    def test_replay_sample_endpoint_marks_clean_runtime_dirty_without_revision(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            app = create_app(
                _build_checkpoint(root, test_case="service_api_replay_sample_clean_state"),
                trace_dir=root / "traces",
                env_root=root,
            )
            manager = app.state.marulho_manager
            runtime = app.state.marulho_runtime
            with TestClient(app) as client:
                feed_response = client.post("/feed", json={"text": "Cats chase mice at night."})
                episode_id = feed_response.json()["runtime_episode"]["episode_id"]
                feedback_response = client.post(
                    "/terminus/runtime-feedback",
                    json={
                        "target_type": "runtime_episode",
                        "target_id": episode_id,
                        "verdict": "contradicted",
                        "confidence": 0.91,
                        "summary": "Manual review contradicted this episode.",
                        "corrected_output": {"summary": "Cats chase mice at night."},
                    },
                )
                plan_response = client.get("/terminus/replay-plan?limit=5")
                candidate = plan_response.json()["candidates"][0]
                candidate_id = candidate["candidate_id"]
                saved_checkpoint = runtime.save_checkpoint()
                before_state = runtime.status()
                sample_response = client.post(
                    "/terminus/replay-sample",
                    json={
                        "mode": "sample",
                        "candidate_id": candidate_id,
                        "target_type": "runtime_episode",
                        "target_id": episode_id,
                        "operator_id": "operator-a",
                        "operator_note": "Audit contradicted replay candidate only.",
                        "confirmation": True,
                        "seed": 123,
                    },
                )
                after_state = runtime.status()
                restore_result = runtime.restore_checkpoint(saved_checkpoint["path"])
                restored_state = runtime.status()

        self.assertEqual(feed_response.status_code, 200)
        self.assertEqual(feedback_response.status_code, 200)
        self.assertEqual(plan_response.status_code, 200)
        self.assertFalse(before_state["dirty_state"])
        self.assertEqual(before_state["state_revision"], after_state["state_revision"])
        self.assertTrue(after_state["dirty_state"])
        self.assertEqual(sample_response.status_code, 200)
        sample_body = sample_response.json()
        self.assertFalse(sample_body["safety_flags"]["state_revision_mutated"])
        self.assertEqual(sample_body["before"]["state_revision"], sample_body["after"]["state_revision"])
        self.assertEqual(sample_body["before"]["state_revision"], before_state["state_revision"])
        self.assertEqual(sample_body["after"]["state_revision"], after_state["state_revision"])
        self.assertEqual(sample_body["mode"], "sample")
        self.assertTrue(sample_body["safety_flags"]["audit_only"])
        before_runtime = before_state["terminus_runtime"]
        after_runtime = after_state["terminus_runtime"]
        restored_runtime = restored_state["terminus_runtime"]
        expected_restored_revision = before_state["state_revision"] + 1
        self.assertEqual(after_runtime["last_event"], before_runtime["last_event"])
        self.assertEqual(after_runtime["recent_events"], before_runtime["recent_events"])
        self.assertFalse(restore_result["dirty_state"])
        self.assertEqual(restore_result["state_revision"], expected_restored_revision)
        self.assertFalse(restored_state["dirty_state"])
        self.assertEqual(restored_state["state_revision"], expected_restored_revision)
        self.assertEqual(restored_runtime["last_event"], before_runtime["last_event"])
        self.assertEqual(restored_runtime["recent_events"], before_runtime["recent_events"])

    def test_terminus_action_endpoint_executes_workspace_read_and_records_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "notes.md").write_text(
                "Cats rest indoors during the day.\nCats chase mice at night.\n",
                encoding="utf-8",
            )
            app = create_app(
                _build_checkpoint(root, test_case="service_api_terminus_action_read"),
                trace_dir=root / "traces",
                env_root=root,
            )
            with TestClient(app) as client:
                action_response = client.post(
                    "/terminus/action",
                    json={
                        "action_type": "workspace_read",
                        "path": "notes.md",
                        "query_text": "cats chase night",
                        "predicted_outcome": "I expect notes.md to say what cats chase at night.",
                    },
                )
                history_response = client.get("/terminus/actions")

            self.assertEqual(action_response.status_code, 200)
            self.assertEqual(history_response.status_code, 200)
            action_body = action_response.json()
            history_body = history_response.json()
            self.assertTrue(action_body["accepted"])
            self.assertEqual(action_body["result"]["action_type"], "workspace_read")
            self.assertEqual(action_body["result"]["verification"]["status"], "verified")
            self.assertEqual(action_body["result"]["inputs"]["path"], "notes.md")
            self.assertEqual(history_body["count"], 1)
            self.assertEqual(history_body["actions"][0]["action_type"], "workspace_read")

    def test_terminus_action_endpoint_executes_web_fetch_and_records_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "page.html").write_text(
                "<html><body><main><p>Cats chase mice at night.</p><p>Cats rest indoors during the day.</p></main></body></html>",
                encoding="utf-8",
            )
            port = _free_port()
            handler = partial(_SilentSimpleHTTPRequestHandler, directory=str(root))
            server = ThreadingHTTPServer(("127.0.0.1", port), handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                app = create_app(
                    _build_checkpoint(root, test_case="service_api_terminus_action_fetch"),
                    trace_dir=root / "traces",
                    env_root=root,
                )
                with TestClient(app) as client:
                    action_response = client.post(
                        "/terminus/action",
                        json={
                            "action_type": "web_fetch",
                            "url": f"http://127.0.0.1:{port}/page.html",
                            "query_text": "cats chase night",
                            "predicted_outcome": "I expect the page to say what cats chase at night.",
                        },
                    )
                    history_response = client.get("/terminus/actions")
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2.0)

            self.assertEqual(action_response.status_code, 200)
            self.assertEqual(history_response.status_code, 200)
            action_body = action_response.json()
            history_body = history_response.json()
            self.assertTrue(action_body["accepted"])
            self.assertEqual(action_body["result"]["action_type"], "web_fetch")
            self.assertEqual(action_body["result"]["verification"]["status"], "verified")
            self.assertIn("http://127.0.0.1", action_body["result"]["inputs"]["url"])
            self.assertEqual(history_body["count"], 1)
            self.assertEqual(history_body["actions"][0]["action_type"], "web_fetch")

    def test_terminus_action_endpoint_executes_api_request_and_records_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "data.json").write_text(
                '{"facts": {"chase": "mice at night", "rest": "indoors during the day"}}',
                encoding="utf-8",
            )
            port = _free_port()
            handler = partial(_SilentSimpleHTTPRequestHandler, directory=str(root))
            server = ThreadingHTTPServer(("127.0.0.1", port), handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                app = create_app(
                    _build_checkpoint(root, test_case="service_api_terminus_action_api"),
                    trace_dir=root / "traces",
                    env_root=root,
                )
                with TestClient(app) as client:
                    action_response = client.post(
                        "/terminus/action",
                        json={
                            "action_type": "api_request",
                            "url": f"http://127.0.0.1:{port}/data.json",
                            "query_text": "cats chase night",
                            "predicted_outcome": "I expect the JSON endpoint to say what cats chase at night.",
                        },
                    )
                    history_response = client.get("/terminus/actions")
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2.0)

            self.assertEqual(action_response.status_code, 200)
            self.assertEqual(history_response.status_code, 200)
            action_body = action_response.json()
            history_body = history_response.json()
            self.assertTrue(action_body["accepted"])
            self.assertEqual(action_body["result"]["action_type"], "api_request")
            self.assertEqual(action_body["result"]["verification"]["status"], "verified")
            self.assertIn("http://127.0.0.1", action_body["result"]["inputs"]["url"])
            self.assertEqual(history_body["count"], 1)
            self.assertEqual(history_body["actions"][0]["action_type"], "api_request")

    def test_terminus_action_endpoint_executes_parameterized_api_request_and_records_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            port = _free_port()
            server = ThreadingHTTPServer(("127.0.0.1", port), _EchoJsonApiHandler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                app = create_app(
                    _build_checkpoint(root, test_case="service_api_terminus_action_parameterized_api"),
                    trace_dir=root / "traces",
                    env_root=root,
                )
                with TestClient(app) as client:
                    action_response = client.post(
                        "/terminus/action",
                        json={
                            "action_type": "api_request",
                            "url": f"http://127.0.0.1:{port}/api/echo",
                            "method": "POST",
                            "params": {"kind": "feline"},
                            "json_body": {"topic": "cats", "fact": "mice at night"},
                            "query_text": "cats mice night feline",
                            "predicted_outcome": "I expect the JSON endpoint to echo a structured request about cats and mice at night.",
                        },
                    )
                    history_response = client.get("/terminus/actions")
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2.0)

            self.assertEqual(action_response.status_code, 200)
            self.assertEqual(history_response.status_code, 200)
            action_body = action_response.json()
            history_body = history_response.json()
            self.assertTrue(action_body["accepted"])
            self.assertEqual(action_body["result"]["action_type"], "api_request")
            self.assertEqual(action_body["result"]["inputs"]["method"], "POST")
            self.assertEqual(action_body["result"]["inputs"]["params"]["kind"], "feline")
            self.assertEqual(action_body["result"]["inputs"]["json_body"]["topic"], "cats")
            self.assertEqual(action_body["result"]["verification"]["status"], "verified")
            self.assertEqual(history_body["count"], 1)
            self.assertEqual(history_body["actions"][0]["action_type"], "api_request")
            self.assertEqual(history_body["actions"][0]["inputs"]["method"], "POST")

    def test_terminus_action_endpoint_preserves_structured_api_verification_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "animals.json").write_text(
                json.dumps(
                    {
                        "animals": [
                            {"name": "cat", "diet": "mice", "active_time": "night"},
                            {"name": "cow", "diet": "grass", "active_time": "day"},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            port = _free_port()
            handler = partial(_SilentSimpleHTTPRequestHandler, directory=str(root))
            server = ThreadingHTTPServer(("127.0.0.1", port), handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                app = create_app(
                    _build_checkpoint(root, test_case="service_api_terminus_action_structured_api_verification"),
                    trace_dir=root / "traces",
                    env_root=root,
                )
                with TestClient(app) as client:
                    action_response = client.post(
                        "/terminus/action",
                        json={
                            "action_type": "api_request",
                            "url": f"http://127.0.0.1:{port}/animals.json",
                            "query_text": "cat mice night",
                            "predicted_outcome": "I expect the JSON endpoint to identify the animal that hunts mice at night.",
                        },
                    )
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2.0)

            self.assertEqual(action_response.status_code, 200)
            action_body = action_response.json()
            self.assertTrue(action_body["accepted"])
            self.assertEqual(action_body["result"]["action_type"], "api_request")
            self.assertEqual(action_body["result"]["verification"]["status"], "verified")
            self.assertEqual(action_body["result"]["verification"]["evidence"][0]["json_path"], "$.animals[0]")
            self.assertEqual(action_body["result"]["verification"]["evidence"][0]["structure_kind"], "object")
            self.assertGreaterEqual(action_body["result"]["verification"]["evidence"][0]["field_count"], 3)

    def test_terminus_action_endpoint_verifies_expected_json_paths_and_response_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "animals.json").write_text(
                json.dumps(
                    {
                        "animals": [
                            {"name": "cat", "diet": "mice", "active_time": "night"},
                            {"name": "cow", "diet": "grass", "active_time": "day"},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            port = _free_port()
            handler = partial(_SilentSimpleHTTPRequestHandler, directory=str(root))
            server = ThreadingHTTPServer(("127.0.0.1", port), handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                app = create_app(
                    _build_checkpoint(root, test_case="service_api_terminus_action_api_assertions"),
                    trace_dir=root / "traces",
                    env_root=root,
                )
                with TestClient(app) as client:
                    action_response = client.post(
                        "/terminus/action",
                        json={
                            "action_type": "api_request",
                            "url": f"http://127.0.0.1:{port}/animals.json",
                            "expected_json_paths": ["$.animals[0]", "$.animals[0].diet"],
                            "expected_response_shape": "object",
                            "query_text": "cat mice night",
                            "predicted_outcome": "I expect the JSON endpoint to expose the first animal entry and its diet.",
                        },
                    )
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2.0)

            self.assertEqual(action_response.status_code, 200)
            action_body = action_response.json()
            self.assertTrue(action_body["accepted"])
            self.assertEqual(action_body["result"]["action_type"], "api_request")
            self.assertEqual(action_body["result"]["inputs"]["expected_json_paths"], ["$.animals[0]", "$.animals[0].diet"])
            self.assertEqual(action_body["result"]["inputs"]["expected_response_shape"], "object")
            self.assertEqual(action_body["result"]["verification"]["status"], "verified")
            self.assertTrue(any(item.get("assertion_kind") == "expected_json_path" for item in action_body["result"]["verification"]["evidence"]))
            self.assertTrue(any(item.get("assertion_kind") == "expected_response_shape" for item in action_body["result"]["verification"]["evidence"]))

    def test_terminus_action_endpoint_verifies_expected_json_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "animals.json").write_text(
                json.dumps(
                    {
                        "animals": [
                            {"name": "cat", "diet": "mice", "active_time": "night"},
                            {"name": "cow", "diet": "grass", "active_time": "day"},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            port = _free_port()
            handler = partial(_SilentSimpleHTTPRequestHandler, directory=str(root))
            server = ThreadingHTTPServer(("127.0.0.1", port), handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                app = create_app(
                    _build_checkpoint(root, test_case="service_api_terminus_action_api_value_assertions"),
                    trace_dir=root / "traces",
                    env_root=root,
                )
                with TestClient(app) as client:
                    action_response = client.post(
                        "/terminus/action",
                        json={
                            "action_type": "api_request",
                            "url": f"http://127.0.0.1:{port}/animals.json",
                            "expected_json_values": {
                                "$.animals[0].diet": "mice",
                                "$.animals[0].active_time": "night",
                            },
                            "query_text": "cat mice night",
                            "predicted_outcome": "I expect the JSON endpoint to confirm the first animal diet and active time.",
                        },
                    )
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2.0)

            self.assertEqual(action_response.status_code, 200)
            action_body = action_response.json()
            self.assertTrue(action_body["accepted"])
            self.assertEqual(action_body["result"]["action_type"], "api_request")
            self.assertEqual(action_body["result"]["inputs"]["expected_json_values"]["$.animals[0].diet"], "mice")
            self.assertEqual(action_body["result"]["verification"]["status"], "verified")
            self.assertTrue(any(item.get("assertion_kind") == "expected_json_value" for item in action_body["result"]["verification"]["evidence"]))

    def test_terminus_action_endpoint_verifies_expected_json_predicates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "metrics.json").write_text(
                json.dumps(
                    {
                        "metrics": {"score": 0.91, "count": 7},
                        "animals": [{"name": "cat", "diet": "mice at night"}],
                    }
                ),
                encoding="utf-8",
            )
            port = _free_port()
            handler = partial(_SilentSimpleHTTPRequestHandler, directory=str(root))
            server = ThreadingHTTPServer(("127.0.0.1", port), handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                app = create_app(
                    _build_checkpoint(root, test_case="service_api_terminus_action_api_predicate_assertions"),
                    trace_dir=root / "traces",
                    env_root=root,
                )
                with TestClient(app) as client:
                    action_response = client.post(
                        "/terminus/action",
                        json={
                            "action_type": "api_request",
                            "url": f"http://127.0.0.1:{port}/metrics.json",
                            "expected_json_predicates": [
                                {"path": "$.animals[0].diet", "op": "contains", "value": "night"},
                                {"path": "$.metrics.score", "op": "gte", "value": 0.9},
                            ],
                            "query_text": "cat metrics night",
                            "predicted_outcome": "I expect the JSON endpoint to satisfy text and score predicates.",
                        },
                    )
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2.0)

            self.assertEqual(action_response.status_code, 200)
            action_body = action_response.json()
            self.assertTrue(action_body["accepted"])
            self.assertEqual(action_body["result"]["action_type"], "api_request")
            self.assertEqual(action_body["result"]["inputs"]["expected_json_predicates"][0]["op"], "contains")
            self.assertEqual(action_body["result"]["verification"]["status"], "verified")
            self.assertTrue(any(item.get("assertion_kind") == "expected_json_predicate" for item in action_body["result"]["verification"]["evidence"]))

    def test_terminus_action_endpoint_verifies_composite_json_predicates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "metrics.json").write_text(
                json.dumps(
                    {
                        "metrics": {"score": 0.91, "count": 7},
                        "animals": [{"name": "cat", "diet": "mice at night"}],
                        "tags": ["night-hunter", "feline-companion"],
                    }
                ),
                encoding="utf-8",
            )
            port = _free_port()
            handler = partial(_SilentSimpleHTTPRequestHandler, directory=str(root))
            server = ThreadingHTTPServer(("127.0.0.1", port), handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                app = create_app(
                    _build_checkpoint(root, test_case="service_api_terminus_action_api_composite_predicates"),
                    trace_dir=root / "traces",
                    env_root=root,
                )
                with TestClient(app) as client:
                    action_response = client.post(
                        "/terminus/action",
                        json={
                            "action_type": "api_request",
                            "url": f"http://127.0.0.1:{port}/metrics.json",
                            "expected_json_predicates": [
                                {"path": "$.metrics.score", "op": "between", "value": {"min": 0.9, "max": 1.0}},
                                {"path": "$.animals[0].diet", "op": "startswith", "value": "mice"},
                                {"path": "$.tags", "op": "any_contains", "value": "hunter"},
                            ],
                            "query_text": "cat metrics night",
                            "predicted_outcome": "I expect the JSON endpoint to satisfy composite predicate checks.",
                        },
                    )
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2.0)

            self.assertEqual(action_response.status_code, 200)
            action_body = action_response.json()
            self.assertTrue(action_body["accepted"])
            self.assertEqual(action_body["result"]["action_type"], "api_request")
            self.assertEqual(action_body["result"]["inputs"]["expected_json_predicates"][0]["op"], "between")
            self.assertEqual(action_body["result"]["verification"]["status"], "verified")
            self.assertTrue(any(item.get("predicate_op") == "between" for item in action_body["result"]["verification"]["evidence"]))
            self.assertTrue(any(item.get("predicate_op") == "any_contains" for item in action_body["result"]["verification"]["evidence"]))

    def test_terminus_action_endpoint_verifies_logical_predicate_groups(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "logic.json").write_text(
                json.dumps(
                    {
                        "metrics": {"score": 0.91},
                        "animals": [{"diet": "mice at night"}],
                        "traits": {"primary": "night-hunter", "secondary": "feline-companion"},
                    }
                ),
                encoding="utf-8",
            )
            port = _free_port()
            handler = partial(_SilentSimpleHTTPRequestHandler, directory=str(root))
            server = ThreadingHTTPServer(("127.0.0.1", port), handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                app = create_app(
                    _build_checkpoint(root, test_case="service_api_terminus_action_api_logical_groups"),
                    trace_dir=root / "traces",
                    env_root=root,
                )
                with TestClient(app) as client:
                    action_response = client.post(
                        "/terminus/action",
                        json={
                            "action_type": "api_request",
                            "url": f"http://127.0.0.1:{port}/logic.json",
                            "expected_json_predicates": [
                                {"path": "$.traits", "op": "all_regex", "value": "^[a-z-]+$"},
                                {"path": "$.traits", "op": "any_contains", "value": "hunter"},
                            ],
                            "expected_json_predicate_groups": [
                                {
                                    "logic": "any",
                                    "predicates": [
                                        {"path": "$.metrics.score", "op": "lt", "value": 0.5},
                                        {"path": "$.animals[0].diet", "op": "contains", "value": "night"},
                                    ],
                                }
                            ],
                            "query_text": "logic night",
                            "predicted_outcome": "I expect the JSON endpoint to satisfy logical predicate groups and object quantifiers.",
                        },
                    )
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2.0)

            self.assertEqual(action_response.status_code, 200)
            action_body = action_response.json()
            self.assertTrue(action_body["accepted"])
            self.assertEqual(action_body["result"]["action_type"], "api_request")
            self.assertEqual(action_body["result"]["inputs"]["expected_json_predicate_groups"][0]["logic"], "any")
            self.assertEqual(action_body["result"]["verification"]["status"], "verified")
            self.assertTrue(any(item.get("assertion_kind") == "expected_json_predicate_group" for item in action_body["result"]["verification"]["evidence"]))
            self.assertTrue(any(item.get("predicate_op") == "all_regex" for item in action_body["result"]["verification"]["evidence"]))

    def test_terminus_action_endpoint_verifies_wildcard_and_nested_group_assertions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "wild.json").write_text(
                json.dumps(
                    {
                        "animals": [
                            {"name": "cat", "diet": "mice at night", "traits": ["hunter", "feline"]},
                            {"name": "owl", "diet": "mice at dawn", "traits": ["bird", "night"]},
                        ],
                        "groups": {"predators": [{"name": "cat"}, {"name": "owl"}]},
                    }
                ),
                encoding="utf-8",
            )
            port = _free_port()
            handler = partial(_SilentSimpleHTTPRequestHandler, directory=str(root))
            server = ThreadingHTTPServer(("127.0.0.1", port), handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                app = create_app(
                    _build_checkpoint(root, test_case="service_api_terminus_action_api_wildcard_nested_groups"),
                    trace_dir=root / "traces",
                    env_root=root,
                )
                with TestClient(app) as client:
                    action_response = client.post(
                        "/terminus/action",
                        json={
                            "action_type": "api_request",
                            "url": f"http://127.0.0.1:{port}/wild.json",
                            "expected_json_paths": ["$.animals[*].diet"],
                            "expected_json_values": {"$.animals[*].name": "owl"},
                            "expected_json_predicate_groups": [
                                {
                                    "logic": "all",
                                    "groups": [
                                        {
                                            "logic": "any",
                                            "predicates": [
                                                {"path": "$.animals[*].diet", "op": "contains", "value": "night"},
                                                {"path": "$.animals[*].diet", "op": "contains", "value": "reptile"},
                                            ],
                                        },
                                        {
                                            "logic": "none",
                                            "predicates": [
                                                {"path": "$.animals[*].traits[*]", "op": "contains", "value": "reptile"},
                                            ],
                                        },
                                    ],
                                }
                            ],
                            "query_text": "wild json nested",
                            "predicted_outcome": "I expect wildcard JSON checks and nested groups to pass on the maintained path.",
                        },
                    )
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2.0)

            self.assertEqual(action_response.status_code, 200)
            action_body = action_response.json()
            self.assertTrue(action_body["accepted"])
            self.assertEqual(action_body["result"]["action_type"], "api_request")
            self.assertTrue(any(item.get("asserted_json_path") == "$.animals[*].diet" for item in action_body["result"]["verification"]["evidence"]))
            self.assertEqual(action_body["result"]["inputs"]["expected_json_predicate_groups"][0]["logic"], "all")
            self.assertEqual(action_body["result"]["verification"]["status"], "verified")
            self.assertTrue(any(item.get("assertion_kind") == "expected_json_predicate_group" for item in action_body["result"]["verification"]["evidence"]))

    def test_respond_endpoint_auto_executes_workspace_action_for_gap_query(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "notes.md").write_text(
                "Cats rest indoors during the day.\nCats chase mice at night.\n",
                encoding="utf-8",
            )
            (root / "page.html").write_text(
                "<html><body><main><p>Cats chase mice at night.</p><p>Cats rest indoors during the day.</p></main></body></html>",
                encoding="utf-8",
            )
            (root / "data.json").write_text(
                '{"facts": {"chase": "mice at night", "rest": "indoors during the day"}}',
                encoding="utf-8",
            )
            port = _free_port()
            handler = partial(_SilentSimpleHTTPRequestHandler, directory=str(root))
            server = ThreadingHTTPServer(("127.0.0.1", port), handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                app = create_app(
                    _build_checkpoint(root, test_case="service_api_respond_auto_action"),
                    trace_dir=root / "traces",
                    env_root=root,
                )
                with TestClient(app) as client:
                    respond_response = client.post(
                        "/respond",
                        json={
                            "query_text": "What do cats chase at night?",
                            "max_evidence_items": 3,
                            "learn_mode": "none",
                        },
                    )
                    read_response = client.post(
                        "/respond",
                        json={
                            "query_text": "What does notes.md say cats chase at night?",
                            "max_evidence_items": 3,
                            "learn_mode": "none",
                        },
                    )
                    fetch_response = client.post(
                        "/respond",
                        json={
                            "query_text": f"What does http://127.0.0.1:{port}/page.html say cats chase at night?",
                            "max_evidence_items": 3,
                            "learn_mode": "none",
                        },
                    )
                    api_response = client.post(
                        "/respond",
                        json={
                            "query_text": f"What does http://127.0.0.1:{port}/data.json say cats chase at night?",
                            "max_evidence_items": 3,
                            "learn_mode": "none",
                        },
                    )
                    actions_response = client.get("/terminus/actions")
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2.0)

            self.assertEqual(respond_response.status_code, 200)
            self.assertEqual(read_response.status_code, 200)
            self.assertEqual(fetch_response.status_code, 200)
            self.assertEqual(api_response.status_code, 200)
            self.assertEqual(actions_response.status_code, 200)
            body = respond_response.json()
            read_body = read_response.json()
            fetch_body = fetch_response.json()
            api_body = api_response.json()
            actions = actions_response.json()
            self.assertEqual(body["query_result"]["action_assist"]["reason"], "query_gap_auto_search")
            self.assertTrue(body["query_result"]["action_assist"]["executed"])
            self.assertIn("cats chase mice at night", body["response"]["response_text"].lower())
            self.assertEqual(read_body["query_result"]["action_assist"]["reason"], "query_gap_auto_read")
            self.assertEqual(read_body["query_result"]["action_assist"]["result"]["action_type"], "workspace_read")
            self.assertIn("cats chase mice at night", read_body["response"]["response_text"].lower())
            self.assertEqual(fetch_body["query_result"]["action_assist"]["reason"], "query_gap_auto_fetch")
            self.assertEqual(fetch_body["query_result"]["action_assist"]["result"]["action_type"], "web_fetch")
            self.assertIn("cats chase mice at night", fetch_body["response"]["response_text"].lower())
            self.assertEqual(api_body["query_result"]["action_assist"]["reason"], "query_gap_auto_api_request")
            self.assertEqual(api_body["query_result"]["action_assist"]["result"]["action_type"], "api_request")
            self.assertIn("mice at night", api_body["response"]["response_text"].lower())
            self.assertEqual(actions["count"], 4)
            self.assertIn(actions["actions"][0]["trigger_reason"], {"query_gap_auto_api_request", "query_gap_auto_fetch", "query_gap_auto_read", "query_gap_auto_search"})

    def test_terminus_tick_endpoint_rejects_when_runtime_running(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_path = root / "terminus_source.txt"
            source_path.write_text("runtime ownership safety signal " * 32, encoding="utf-8")
            app = create_app(_build_checkpoint(root, test_case="service_api_terminus_tick_running"), trace_dir=root / "traces")
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
                        "repeat_sources": True,
                    },
                )
                start_response = client.post("/terminus/start")
                tick_response = client.post("/terminus/tick", json={"steps": 1})
                stop_response = client.post("/terminus/stop")

            self.assertEqual(configure_response.status_code, 200)
            self.assertEqual(start_response.status_code, 200)
            self.assertEqual(tick_response.status_code, 422)
            self.assertIn("background runtime is active", tick_response.json()["detail"])
            self.assertEqual(stop_response.status_code, 200)

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

    def test_feed_query_respond_endpoints_return_runtime_episodes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            app = create_app(_build_checkpoint(root, test_case="service_api_runtime_episodes"), trace_dir=root / "traces")
            with TestClient(app) as client:
                feed_response = client.post(
                    "/feed",
                    json={"text": "Cats chase mice at night. Cats rest indoors during the day. " * 4},
                )
                query_response = client.post("/query", json={"query_text": "cats chase mice", "top_k_memories": 4})
                respond_response = client.post(
                    "/respond",
                    json={"query_text": "cats chase mice", "top_k_memories": 4, "learn_mode": "none"},
                )
                living_response = client.get("/terminus/living-loop")
                replay_response = client.get("/terminus/replay-plan?limit=3")
                export_response = client.get("/terminus/runtime-traces/export?limit=2")
                query_export_response = client.get(
                    "/terminus/runtime-traces/export",
                    params={"endpoint": "query", "limit": 5},
                )
                oversized_export_response = client.get("/terminus/runtime-traces/export?limit=51")

        self.assertEqual(feed_response.status_code, 200)
        self.assertEqual(query_response.status_code, 200)
        self.assertEqual(respond_response.status_code, 200)
        self.assertEqual(living_response.status_code, 200)
        self.assertEqual(replay_response.status_code, 200)
        self.assertEqual(export_response.status_code, 200)
        self.assertEqual(query_export_response.status_code, 200)
        self.assertEqual(oversized_export_response.status_code, 422)
        self.assertEqual(feed_response.json()["runtime_episode"]["operation"], "feed")
        self.assertEqual(query_response.json()["runtime_episode"]["operation"], "query")
        self.assertEqual(respond_response.json()["runtime_episode"]["operation"], "respond")
        living_loop = living_response.json()["living_loop"]
        self.assertLessEqual(
            {"feed", "query", "respond"},
            {episode["operation"] for episode in living_loop["runtime_episodes"]},
        )
        self.assertIn("runtime_episode_trace", living_loop["capabilities"])
        self.assertEqual(living_loop["policy_decision"]["schema_version"], 1)
        self.assertIsInstance(living_loop["policy_decision"]["action"], str)
        replay_body = replay_response.json()
        self.assertEqual(replay_body["schema_version"], 1)
        self.assertTrue(replay_body["advisory"])
        self.assertFalse(replay_body["executable"])
        self.assertEqual(replay_body["endpoint"], "/terminus/replay-plan")
        self.assertLessEqual(replay_body["count"], 3)
        self.assertIn("snapshot_counts", replay_body)
        self.assertIn("candidates", replay_body)
        export_body = export_response.json()
        self.assertEqual(export_body["export_kind"], "terminus_runtime_trace_dataset_preview")
        self.assertIn("not_training", export_body["training_role"])
        self.assertEqual(export_body["count"], 2)
        self.assertEqual(export_body["policy_decision"]["action"], living_loop["policy_decision"]["action"])
        self.assertEqual(export_body["replay_plan_summary"]["endpoint"], "/terminus/replay-plan")
        self.assertEqual(export_body["replay_dataset_summary"]["endpoint"], "/terminus/replay-dataset/preview")
        self.assertEqual(export_body["replay_dataset_summary"]["count"], 2)
        self.assertIn("latest_export_timestamp", export_body["replay_dataset_summary"])
        self.assertEqual([example["endpoint"] for example in export_body["examples"]], ["/respond", "/query"])
        for example in export_body["examples"]:
            self.assertLessEqual(
                {
                    "context",
                    "prediction",
                    "actual_output",
                    "verification",
                    "provenance",
                    "latency_ms",
                    "state_revision",
                    "token_count",
                    "policy_decision",
                    "replay_plan_summary",
                },
                set(example),
            )
            self.assertEqual(example["policy_decision"]["action"], export_body["policy_decision"]["action"])
            self.assertEqual(example["replay_plan_summary"]["endpoint"], "/terminus/replay-plan")
        query_export = query_export_response.json()
        self.assertEqual(query_export["endpoint"], "query")
        self.assertEqual(query_export["count"], 1)
        self.assertEqual(query_export["examples"][0]["type"], "query")

    def test_runtime_feedback_endpoint_updates_runtime_episode_and_action_targets(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "notes.md").write_text(
                "Cats rest indoors during the day.\nCats chase mice at night.\n",
                encoding="utf-8",
            )
            app = create_app(
                _build_checkpoint(root, test_case="service_api_runtime_feedback"),
                trace_dir=root / "traces",
                env_root=root,
            )
            with TestClient(app) as client:
                feed_response = client.post("/feed", json={"text": "Cats chase mice at night."})
                episode_id = feed_response.json()["runtime_episode"]["episode_id"]
                episode_feedback_response = client.post(
                    "/terminus/runtime-feedback",
                    json={
                        "target_type": "runtime_episode",
                        "target_id": episode_id,
                        "verdict": "verified",
                        "confidence": 0.91,
                        "summary": "Runtime trace reviewed.",
                        "evidence": [{"note": "manual verification"}],
                        "tags": ["Reviewed"],
                        "evaluator_id": "api-test",
                    },
                )
                action_response = client.post(
                    "/terminus/action",
                    json={
                        "action_type": "workspace_search",
                        "query_text": "cats chase mice",
                        "predicted_outcome": "I expect to find evidence about cats chasing mice.",
                    },
                )
                action_id = action_response.json()["result"]["action_id"]
                action_feedback_response = client.post(
                    "/terminus/runtime-feedback",
                    json={
                        "target_type": "action",
                        "target_id": action_id,
                        "verdict": "unverified",
                        "confidence": 0.33,
                        "summary": "Needs a second review.",
                        "tags": ["needs-review"],
                    },
                )
                missing_response = client.post(
                    "/terminus/runtime-feedback",
                    json={
                        "target_type": "action",
                        "target_id": "missing-action",
                        "verdict": "verified",
                        "confidence": 0.5,
                    },
                )
                history_response = client.get("/terminus/actions")

        self.assertEqual(episode_feedback_response.status_code, 200)
        self.assertEqual(action_feedback_response.status_code, 200)
        self.assertEqual(missing_response.status_code, 422)
        episode_body = episode_feedback_response.json()
        action_body = action_feedback_response.json()
        self.assertEqual(episode_body["target"]["verification"]["status"], "verified")
        self.assertEqual(episode_body["target"]["provenance"], "verified")
        self.assertEqual(episode_body["feedback"]["tags"], ["reviewed"])
        self.assertTrue(episode_body["dirty_state"])
        self.assertEqual(action_body["target"]["verification"]["status"], "unverified")
        self.assertEqual(action_body["target"]["verification"]["provenance"], "unverified")
        self.assertEqual(history_response.json()["actions"][0]["feedback"][0]["summary"], "Needs a second review.")

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
                with patch("marulho.data.source_catalog._search_remote_provider", side_effect=fake_search):
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
                with patch("marulho.data.source_catalog._search_remote_provider", side_effect=fake_search):
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
                with patch("marulho.data.source_catalog._search_remote_provider", side_effect=fake_search):
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
                with patch("marulho.data.source_catalog._search_remote_provider", side_effect=fake_search):
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
                with patch("marulho.data.source_catalog._search_remote_provider", side_effect=fake_search):
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
                with patch("marulho.data.source_catalog._search_remote_provider", side_effect=fake_search):
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
                with patch("marulho.data.source_catalog._search_remote_provider", side_effect=fake_search):
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
                        with app.state.marulho_manager._lock:
                            app.state.marulho_manager._interaction_pipeline.record_recent_query_gap(
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

    def test_architecture_endpoint_returns_layer_topology(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            app = create_app(_build_checkpoint(root, test_case="service_api_architecture"), trace_dir=root / "traces")
            with TestClient(app) as client:
                resp = client.get("/architecture")

            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertEqual(data["model_name"], "Terminus")
            self.assertEqual(data["core_name"], "GPCSN")
            self.assertEqual(data["version"], "current")
            layers = data["layers"]
            self.assertIsInstance(layers, list)
            self.assertGreater(len(layers), 0)
            layer_ids = [l["id"] for l in layers]
            self.assertIn("input_encoding", layer_ids)
            self.assertIn("competitive_routing", layer_ids)
            self.assertIn("predictive_columns", layer_ids)
            self.assertIn("memory_consolidation", layer_ids)
            self.assertNotIn("cortex_backend", layer_ids)
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
            from marulho.service.manager import MarulhoServiceManager
            mgr = MarulhoServiceManager(
                _build_checkpoint(root, test_case="service_api_animation"),
                trace_dir=root / "traces",
            )
            snapshot = mgr.runtime_facade.telemetry_snapshot()
            mgr.close()

        self.assertIn("animation", snapshot)
        anim = snapshot["animation"]
        self.assertIn("n_columns", anim)
        self.assertIn("activations", anim)
        self.assertIn("spike_counts", anim)
        self.assertIn("memory_fill", anim)
        self.assertEqual(len(anim["activations"]), anim["n_columns"])

    def test_quick_start_presets_endpoint(self) -> None:
        """GET /terminus/presets returns available preset list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ckpt = _build_checkpoint(root, test_case="service_api_presets")
            app = create_app(ckpt, trace_dir=root / "traces")
            with TestClient(app) as client:
                resp = client.get("/terminus/presets")
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertIsInstance(data, list)
            self.assertGreater(len(data), 0)
            ids = [p["id"] for p in data]
            self.assertEqual(ids, ["curriculum"])
            self.assertNotIn("multimodal", ids)
            self.assertNotIn("multimodal_fast", ids)
            self.assertEqual(data[0]["id"], "curriculum")
            self.assertTrue(data[0].get("default"))
            for preset in data:
                self.assertIn("label", preset)
                self.assertIn("description", preset)
                self.assertIn("source_count", preset)
                self.assertIn("default", preset)
                self.assertIn("legacy", preset)

    def test_quick_start_configures_and_starts_terminus(self) -> None:
        """POST /terminus/quick-start uses the recommended curriculum preset by default."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_path = root / "quick_start_source.txt"
            source_path.write_text("The platypus is an egg-laying mammal found in eastern Australia. " * 5)
            ckpt = _build_checkpoint(root, test_case="service_api_quick_start")
            app = create_app(ckpt, trace_dir=root / "traces")

            def _fake_start(self):
                return {
                    "terminus_runtime": self._brain_runtime_snapshot_locked(),
                    **self._runtime_state.mutation_summary(),
                    "token_count": int(self._trainer.token_count),
                }

            with patch("marulho.service.runtime_control.RuntimeControl.start_terminus", autospec=True, side_effect=_fake_start):
                with TestClient(app) as client:
                    resp = client.post("/terminus/quick-start")
                    self.assertEqual(resp.status_code, 200)
                    data = resp.json()
                    self.assertTrue(data["terminus_runtime"]["configured"])
                    self.assertFalse(data.get("already_running", False))
                    self.assertEqual(data.get("preset_applied"), "curriculum")
                    self.assertEqual(data["terminus_runtime"]["source_count"], 3)
                    self.assertEqual(data["terminus_runtime"]["tick_tokens"], 128)
                    self.assertEqual(
                        data["terminus_runtime"]["execution_schedule"]["quantum_tokens"],
                        16,
                    )
                    self.assertEqual(
                        data["terminus_runtime"]["ingestion"]["queue_target_tokens"],
                        4096,
                    )
                    self.assertTrue(
                        data["terminus_runtime"]["ingestion"]["prewarm_on_startup"]
                    )
                    self.assertTrue(data["terminus_runtime"]["autonomy"]["enabled"])
                    self.assertEqual(data["terminus_runtime"]["autonomy"]["candidate_bank"][0]["catalog_mode"], "semantic_registry")
                    manager = app.state.marulho_manager
                    self.assertEqual(manager._trainer.config.memory_capacity, 1000)
                    self.assertEqual(manager._trainer.model.memory_store.capacity, 1000)
                    self.assertFalse(manager._trainer.config.enable_context_layer)
                    self.assertFalse(manager._trainer.config.enable_binding_layer)
                    self.assertNotIn("curriculum", data["terminus_runtime"])
                    stop_resp = client.post("/terminus/stop")
                    self.assertEqual(stop_resp.status_code, 200)

    def test_datasets_endpoint_reports_current_runtime_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ckpt = _build_checkpoint(root, test_case="service_api_datasets")
            app = create_app(ckpt, trace_dir=root / "traces")
            with TestClient(app) as client:
                resp = client.get("/datasets")
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            names = [entry["name"] for entry in data["datasets"]]
            self.assertIn("fineweb_edu", names)
            self.assertIn("open_textbooks", names)
            self.assertNotIn("wikipedia_en", names)
            self.assertIn("s2orc_arxiv_abstracts", names)
            self.assertIn("science_figures", names)
            self.assertIn("environmental_audio", names)
            self.assertNotIn("nim_curriculum", names)
            self.assertNotIn("N-MNIST", names)
            self.assertNotIn("FSDD", names)
            self.assertIn("huggingface", data)
            self.assertIn("token_configured", data["huggingface"])

    def test_sensory_recent_endpoint_returns_media_previews(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ckpt = _build_checkpoint(root, test_case="service_api_sensory_recent")
            app = create_app(ckpt, trace_dir=root / "traces")
            manager = app.state.marulho_manager
            with manager._lock:
                manager._sensory_preview_history.appendleft(
                    {
                        "preview_id": "preview-1",
                        "captured_at": "2026-04-21T00:00:00+00:00",
                        "source_name": "science_figures",
                        "adapter": "s1_mmalign",
                        "text": "example figure",
                        "semantic_match": 0.8,
                        "modality_need": 0.5,
                        "selection_score": 0.9,
                        "window_budget": 6,
                        "topics": ["scientific", "diagram"],
                        "focus_terms": ["scientific", "diagram"],
                        "metadata": {"title": "phase diagram"},
                        "visual": {"mime_type": "image/png", "bytes": b"fakepng", "width": 16, "height": 16},
                        "audio": None,
                    }
                )
            with TestClient(app) as client:
                resp = client.get("/terminus/sensory/recent?limit=1")
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertEqual(data["count"], 1)
            self.assertEqual(data["latest_preview_id"], "preview-1")
            self.assertEqual(len(data["previews"]), 1)
            self.assertTrue(data["previews"][0]["visual"]["data_url"].startswith("data:image/png;base64,"))

    def test_quick_start_rejects_unknown_preset(self) -> None:
        """POST /terminus/quick-start with bad preset returns 422."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ckpt = _build_checkpoint(root, test_case="service_api_qs_bad")
            app = create_app(ckpt, trace_dir=root / "traces")
            with TestClient(app) as client:
                resp = client.post("/terminus/quick-start?preset=nonexistent")
            self.assertEqual(resp.status_code, 422)

    def test_newborn_synapse_pruning_routes_are_registered(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            app = create_app(
                _build_checkpoint(
                    root,
                    test_case="service_api_newborn_synapse_pruning_routes",
                ),
                trace_dir=root / "traces",
            )
            paths = set(app.openapi()["paths"])

        prefix = "/terminus/snn-language-sequence/readout-ledger/"
        self.assertIn(
            prefix + "snn-language-newborn-neuron-maturation-outcome-review",
            paths,
        )
        self.assertIn(
            prefix + "snn-language-newborn-synapse-pruning-design",
            paths,
        )
        self.assertIn(
            prefix + "snn-language-newborn-synapse-pruning-preflight",
            paths,
        )
        self.assertIn(
            prefix + "snn-language-newborn-synapse-pruning-executor",
            paths,
        )


if __name__ == "__main__":
    unittest.main()
