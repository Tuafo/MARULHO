from __future__ import annotations

from copy import deepcopy
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

from hecsn.config.model_config import HECSNConfig
from hecsn.semantics import (
    build_snn_language_transition_memory_prediction_evaluation,
    build_spike_language_decoder_probe,
    predict_spike_language_sequence,
)
from hecsn.service.api import DEFAULT_WEB_DIST_DIR, create_app
from hecsn.service.server import build_arg_parser
from hecsn.training.runner_utils import set_seed
from hecsn.training.checkpointing import load_trainer_checkpoint, save_trainer_checkpoint
from hecsn.training.model import HECSNModel
from hecsn.training.trainer import HECSNTrainer


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
    model = HECSNModel(cfg)
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
    def test_app_creation_health_status_do_not_eagerly_initialize_cortex(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            app = create_app(_build_checkpoint(root, test_case="service_api_cortex_lazy_startup"), trace_dir=root / "traces")
            with TestClient(app) as client:
                health_response = client.get("/health")
                status_response = client.get("/status")
            app.state.hecsn_manager.close()

            self.assertEqual(health_response.status_code, 200)
            self.assertEqual(status_response.status_code, 200)
            terminus_runtime = status_response.json()["terminus_runtime"]
            self.assertNotIn("cortex", terminus_runtime)
            self.assertNotIn("retired_runtime_path", terminus_runtime)
            self.assertNotIn("retired_runtime_dependency", terminus_runtime["living_loop"]["subcortex_sleep_pressure"])

    def test_status_and_terminus_endpoints_expose_runtime_truth_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            app = create_app(_build_checkpoint(root, test_case="service_api_runtime_truth"), trace_dir=root / "traces")
            with TestClient(app) as client:
                status_response = client.get("/status")
                terminus_response = client.get("/terminus")
            app.state.hecsn_manager.close()

        self.assertEqual(status_response.status_code, 200)
        self.assertEqual(terminus_response.status_code, 200)
        status_truth = status_response.json()["runtime_truth"]
        terminus_truth = terminus_response.json()["runtime_truth"]
        self.assertEqual(status_truth["schema_version"], 1)
        self.assertEqual(status_truth["verdict"], "partial")
        self.assertEqual(status_truth["recommended_action"], "configure_terminus_sources")
        self.assertIn("evidence", status_truth)
        self.assertIn("memory_pressure", status_truth)
        self.assertIn("safety_flags", status_truth)
        self.assertNotIn("retired_runtime_path", status_truth)
        self.assertNotIn("retired_runtime_path", status_truth["evidence"])
        self.assertEqual(terminus_truth["verdict"], status_truth["verdict"])
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
        self.assertIn("local_plasticity_report_available", status_structural_gate)
        self.assertIn("local_plasticity_homeostatic_state_available", status_structural_gate)
        self.assertNotIn("structural_cases", status_structural_gate)
        self.assertNotIn("endpoint", status_structural_gate)
        self.assertNotIn("device_evidence", status_structural_gate)
        self.assertNotIn("local_plasticity", status_structural_gate)
        self.assertEqual(terminus_structural_gate["artifact_kind"], status_structural_gate["artifact_kind"])
        self.assertEqual(terminus_structural_gate["surface"], status_structural_gate["surface"])
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
        self.assertTrue(status_language_gate["requires_hecsn_owned_implementation"])
        self.assertTrue(status_language_gate["hecsn_spike_readout_evidence_available"])
        self.assertTrue(status_language_gate["hecsn_spike_readout_grounded"])
        self.assertTrue(status_language_gate["hecsn_spike_readout_non_generative"])
        self.assertIn("hecsn_spike_decoder_probe_available", status_language_gate)
        self.assertIn("hecsn_spike_decoder_probe_owned", status_language_gate)
        self.assertIn("hecsn_spike_decoder_probe_non_generative", status_language_gate)
        self.assertIn("hecsn_spike_decoder_probe_sparse", status_language_gate)
        self.assertIn("hecsn_spike_decoder_probe_device_evidence_available", status_language_gate)
        self.assertIn("hecsn_spike_decoder_probe_grounding_supported", status_language_gate)
        self.assertIn("hecsn_spike_language_neuron_adapter_available", status_language_gate)
        self.assertIn("hecsn_spike_language_neuron_adapter_owned", status_language_gate)
        self.assertIn("hecsn_spike_language_neuron_adapter_sparse", status_language_gate)
        self.assertIn("hecsn_spike_language_neuron_adapter_dynamic", status_language_gate)
        self.assertEqual(
            terminus_language_gate["hecsn_spike_readout_evidence_available"],
            status_language_gate["hecsn_spike_readout_evidence_available"],
        )
        self.assertEqual(
            terminus_language_gate["hecsn_spike_readout_non_generative"],
            status_language_gate["hecsn_spike_readout_non_generative"],
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
                pending_evaluation_response = client.post(
                    "/terminus/snn-language-sequence/readout-draft",
                    json={
                        "prediction_report": prediction_report,
                        "readout_vocabulary_slots": vocabulary,
                        "device_evidence": {"device": "cpu", "source": "service_api_readout_draft"},
                        "max_draft_terms": 4,
                    },
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
            app.state.hecsn_manager.close()

        self.assertEqual(status_response.status_code, 200)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(pending_evaluation_response.status_code, 200)
        self.assertEqual(blocked_record_response.status_code, 200)
        self.assertEqual(record_response.status_code, 200)
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
        self.assertEqual(readout_synapse_audit_response.status_code, 200)
        self.assertEqual(checkpoint_save_response.status_code, 200)
        self.assertEqual(checkpoint_restore_response.status_code, 200)
        self.assertEqual(restored_readout_plasticity_runtime_state_response.status_code, 200)
        self.assertEqual(restored_readout_synapse_audit_response.status_code, 200)
        self.assertEqual(restored_ledger_response.status_code, 200)
        self.assertEqual(restored_replay_priority_response.status_code, 200)
        body = response.json()
        pending_evaluation_body = pending_evaluation_response.json()
        blocked_record = blocked_record_response.json()
        record = record_response.json()
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
        readout_synapse_audit = readout_synapse_audit_response.json()
        restored_readout_plasticity_runtime_state = (
            restored_readout_plasticity_runtime_state_response.json()
        )
        restored_readout_synapse_audit = restored_readout_synapse_audit_response.json()
        restored_ledger = restored_ledger_response.json()
        restored_replay_priority = restored_replay_priority_response.json()
        self.assertEqual(body["surface"], "snn_language_readout_draft.v1")
        self.assertTrue(body["generates_text"])
        self.assertTrue(body["decodes_text"])
        self.assertFalse(body["freeform_language_generation"])
        self.assertFalse(body["mutates_runtime_state"])
        self.assertIn("memory pressure", body["draft"]["text"])
        self.assertTrue(body["transition_memory_evaluation_evidence"]["review_ready"])
        self.assertTrue(body["promotion_gate"]["eligible_for_bounded_readout_generation"])
        self.assertFalse(body["promotion_gate"]["eligible_for_cognition_substrate"])
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
            "hecsn.snn_language.sparse_transition_weights",
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
        self.assertEqual(restored_replay_priority["candidate_count"], 1)

    def test_readout_synapse_audit_blocks_incomplete_checkpoint_restore_halves(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            app = create_app(
                _build_checkpoint(root, test_case="service_api_readout_synapse_restore_halves"),
                trace_dir=root / "traces",
            )
            runtime = app.state.hecsn_runtime
            manager = app.state.hecsn_manager
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

    def test_evaluated_transition_memory_replay_artifact_uses_internal_readout_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            app = create_app(
                _build_checkpoint(root, test_case="service_api_evaluated_snn_replay_artifact"),
                trace_dir=root / "traces",
            )
            manager = app.state.hecsn_manager
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
                            "target_id": "hecsn.snn_language.sparse_transition_weights",
                            "owned_by_hecsn": True,
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
            app.state.hecsn_manager.close()

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
        self.assertTrue(regeneration_replay_artifact["owned_by_hecsn"])
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
            "reference_for_hecsn_owned_reimplementation",
        )
        self.assertIn(
            "hecsn_owned_language_neuron_module",
            readiness["research_candidates"][0]["required_local_evidence"],
        )
        self.assertIn("hecsn_native_snn_decoder", readiness["research_candidates"][0]["required_local_evidence"])
        self.assertEqual(evaluation["evaluation_cases"][0]["target"], "spike_language_neuron_adapter")
        self.assertIn("adapter_activation_sparsity_delta", evaluation["success_evidence"])
        self.assertEqual(heldout["heldout_summary"]["case_count"], 1)
        self.assertGreater(heldout["adapter_delta"]["mean_active_spike_count"], 0.0)

    def test_subcortical_self_repair_endpoint_is_gate_artifact_not_replay_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            app = create_app(_build_checkpoint(root, test_case="service_api_self_repair_gate"), trace_dir=root / "traces")
            runtime = app.state.hecsn_runtime
            with TestClient(app) as client:
                before_revision = runtime.status()["state_revision"]
                before_history = runtime.action_history()["count"]
                response = client.get("/terminus/subcortical-self-repair")
                second_response = client.get("/terminus/subcortical-self-repair")
                after_revision = runtime.status()["state_revision"]
                after_history = runtime.action_history()["count"]
            app.state.hecsn_manager.close()

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
            runtime = app.state.hecsn_runtime
            with TestClient(app) as client:
                before_revision = runtime.status()["state_revision"]
                before_history = runtime.action_history()["count"]
                response = client.get("/terminus/subcortical-self-repair/evaluation")
                after_revision = runtime.status()["state_revision"]
                after_history = runtime.action_history()["count"]
            app.state.hecsn_manager.close()

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
            runtime = app.state.hecsn_runtime
            with TestClient(app) as client:
                before_revision = runtime.status()["state_revision"]
                before_history = runtime.action_history()["count"]
                response = client.get("/terminus/subcortical-structural-plasticity")
                evaluation_response = client.post(
                    "/terminus/subcortical-structural-plasticity/evaluate",
                    json={
                        "pre_snapshot": {
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
                        },
                        "post_snapshot": {
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
                        "rollback_policy": {"available": True, "snapshot_id": "pre-structural-eval"},
                    },
                )
                after_revision = runtime.status()["state_revision"]
                after_history = runtime.action_history()["count"]
            app.state.hecsn_manager.close()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(evaluation_response.status_code, 200)
        body = response.json()
        evaluation = evaluation_response.json()
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
        self.assertEqual(before_revision, after_revision)
        self.assertEqual(before_history, after_history)

    def test_validation_report_endpoints_list_and_read_reports(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
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
            readme_path.write_text("# Phase 15\n", encoding="utf-8")
            app = create_app(
                _build_checkpoint(root, test_case="service_api_validation_reports"),
                trace_dir=root / "traces",
                env_root=root,
            )
            with TestClient(app) as client:
                list_response = client.get("/terminus/validation/reports")
                read_response = client.get("/terminus/validation/report", params={"path": "phase15/README.md"})
            app.state.hecsn_manager.close()

        self.assertEqual(list_response.status_code, 200)
        listed = list_response.json()
        self.assertEqual(listed["phase_status"]["phase15"]["status"], "ready_for_bounded_level_5_experiment")
        self.assertEqual(listed["reports"][0]["readme_path"], "phase15/README.md")
        self.assertEqual(read_response.status_code, 200)
        self.assertEqual(read_response.json()["media_type"], "text/markdown")
        self.assertIn("Phase 15", read_response.json()["content"])

    def test_static_ui_default_points_to_built_frontend_dist(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ckpt = _build_checkpoint(root, test_case="service_api_static_default")
            app = create_app(ckpt, trace_dir=root / "traces")
            self.assertEqual(app.state.web_dist_dir, DEFAULT_WEB_DIST_DIR)
            app.state.hecsn_manager.close()

        self.assertEqual(DEFAULT_WEB_DIST_DIR, Path("HECSN_UI") / "dist")

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
                        "sleep_interval_seconds": 0.01,
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
            manager = app.state.hecsn_manager
            runtime = app.state.hecsn_runtime
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
            manager = app.state.hecsn_manager
            runtime = app.state.hecsn_runtime
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
            manager = app.state.hecsn_manager
            runtime = app.state.hecsn_runtime
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
            manager = app.state.hecsn_manager
            runtime = app.state.hecsn_runtime
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
                            app.state.hecsn_manager._interaction_pipeline.record_recent_query_gap(
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
            from hecsn.service.manager import HECSNServiceManager
            mgr = HECSNServiceManager(
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

            with patch("hecsn.service.runtime_control.RuntimeControl.start_terminus", autospec=True, side_effect=_fake_start):
                with TestClient(app) as client:
                    resp = client.post("/terminus/quick-start")
                    self.assertEqual(resp.status_code, 200)
                    data = resp.json()
                    self.assertTrue(data["terminus_runtime"]["configured"])
                    self.assertFalse(data.get("already_running", False))
                    self.assertEqual(data.get("preset_applied"), "curriculum")
                    self.assertEqual(data["terminus_runtime"]["source_count"], 3)
                    self.assertLessEqual(data["terminus_runtime"]["tick_tokens"], 64)
                    self.assertTrue(data["terminus_runtime"]["autonomy"]["enabled"])
                    self.assertEqual(data["terminus_runtime"]["autonomy"]["candidate_bank"][0]["catalog_mode"], "semantic_registry")
                    manager = app.state.hecsn_manager
                    self.assertTrue(manager._trainer.config.enable_context_layer)
                    self.assertTrue(manager._trainer.config.enable_binding_layer)
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
            self.assertIn("wikipedia_en", names)
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
            manager = app.state.hecsn_manager
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


if __name__ == "__main__":
    unittest.main()
