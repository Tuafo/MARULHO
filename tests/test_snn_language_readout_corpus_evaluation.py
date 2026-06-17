from __future__ import annotations

import io
import json

from marulho.evaluation.snn_language_readout_corpus import (
    build_snn_language_readout_corpus_checkpoint_review,
    evaluate_snn_language_readout_corpus,
    main as readout_corpus_main,
)
from marulho.semantics.spike_language_decoder import build_spike_language_decoder_probe
from marulho.semantics.spike_language_neurons import predict_spike_language_sequence


def _slot(label: str, pressure_band: str = "medium") -> dict[str, object]:
    return {"label": label, "pressure_band": pressure_band, "grounded": True}


def _promotable_fixture() -> dict[str, object]:
    current = [_slot("concept focus")]
    target = [_slot("memory pressure")]
    baseline = predict_spike_language_sequence(
        [[_slot("prediction error", "high")], current],
        current,
        {"device": "cpu", "source": "readout_corpus_fixture"},
        top_k=4,
    )
    current_index = baseline["current_sparse_code"]["active_indices"][0]
    target_probe = build_spike_language_decoder_probe(
        {
            "readout_slots": target,
            "device_evidence": {"device": "cpu", "source": "readout_corpus_fixture"},
        }
    )
    target_index = target_probe["sparse_code_evidence"]["active_indices"][0]
    return {
        "corpus": {
            "name": "bounded-fixture-readout-corpus",
            "source_type": "local_fixture",
            "license": "repo-test-fixture",
            "terms": "local deterministic fixture",
            "split": "eval",
            "sample_size": 2,
            "cache_path": "tests/fixtures/snn-language-readout-corpus.json",
        },
        "training_readout_slot_batches": [[_slot("prediction error", "high")], current],
        "evaluation_readout_slot_batches": [current, target],
        "transition_memory_state": {
            "sparse_transition_weights": {f"{current_index}:{target_index}": 0.9}
        },
        "device_evidence": {"device": "cpu", "source": "readout_corpus_fixture"},
    }


def test_readout_corpus_evaluation_promotes_only_bounded_operator_review() -> None:
    report = evaluate_snn_language_readout_corpus(
        _promotable_fixture(),
        top_k=4,
    )

    assert report["artifact_kind"] == "terminus_snn_language_readout_corpus_evaluation"
    assert report["surface"] == "snn_language_readout_corpus_evaluation.v1"
    assert report["owned_by_marulho"] is True
    assert report["external_dependency"] is False
    assert report["loads_external_checkpoint"] is False
    assert report["generates_text"] is False
    assert report["trains_runtime_model"] is False
    assert report["mutates_runtime_state"] is False
    assert report["passed"] is True
    assert report["status"] == "promote_bounded_readout_review"
    assert report["corpus_provenance"]["dataset_name"] == "bounded-fixture-readout-corpus"
    assert report["corpus_provenance"]["license"] == "repo-test-fixture"
    assert report["corpus_provenance"]["sample_size"] == 2
    assert report["sequence_evaluation_summary"]["evaluation_pair_count"] == 1
    assert report["sequence_evaluation_summary"]["persistent_transition_weight_count"] == 1
    assert report["grounding_evidence"]["supported"] is True
    assert report["device_evidence"]["tensor_device"] == "cpu"
    assert report["metabolism_evidence"]["latency_ms"] >= 0.0
    assert report["metabolism_evidence"]["python_peak_memory_bytes"] > 0
    assert report["runtime_truth_gate"]["available_status"] == "available"
    assert report["runtime_truth_gate"]["trained_status"] == "isolated_evaluation_only_not_runtime_trained"
    assert report["runtime_truth_gate"]["grounded_status"] == "grounded"
    assert report["runtime_truth_gate"]["mutation_gate_status"] == "mutation_absent"
    assert report["runtime_truth_gate"]["promotion_decision"] == "promote_bounded_readout_review"
    assert report["promotion_gate"]["eligible_for_bounded_readout_generation_review"] is True
    assert report["promotion_gate"]["eligible_for_freeform_language_generation"] is False
    assert report["promotion_gate"]["eligible_for_cognition_substrate"] is False
    assert report["provenance_evidence"]["report_hash"]


def test_readout_corpus_evaluation_rejects_missing_transition_memory() -> None:
    corpus = _promotable_fixture()
    corpus["transition_memory_state"] = {"sparse_transition_weights": {}}

    report = evaluate_snn_language_readout_corpus(corpus, top_k=4)

    assert report["passed"] is False
    assert report["status"] == "reject_live_readout_collect_evidence"
    assert report["promotion_gate"]["status"] == "rejected_for_live_readout"
    assert (
        report["promotion_gate"]["required_evidence"][
            "persistent_transition_memory_available"
        ]
        is False
    )
    assert "persistent_transition_memory_available" in report["runtime_truth_gate"]["reason_codes"]
    assert report["mutates_runtime_state"] is False
    assert report["trains_runtime_model"] is False


def test_readout_corpus_checkpoint_review_writes_restorable_sparse_checkpoint(tmp_path) -> None:
    corpus = _promotable_fixture()
    report = evaluate_snn_language_readout_corpus(corpus, top_k=4)
    checkpoint_path = tmp_path / "readout-checkpoint.json"

    review = build_snn_language_readout_corpus_checkpoint_review(
        corpus,
        report,
        checkpoint_path,
    )

    assert review["artifact_kind"] == "terminus_snn_language_readout_corpus_checkpoint_review"
    assert review["surface"] == "snn_language_readout_corpus_checkpoint_review.v1"
    assert review["ready"] is True
    assert review["status"] == "promote_checkpointed_bounded_readout_review"
    assert review["writes_checkpoint"] is True
    assert review["writes_live_checkpoint"] is False
    assert review["mutates_runtime_state"] is False
    assert review["loads_external_checkpoint"] is False
    assert checkpoint_path.exists()
    assert review["rollback_evidence"]["available"] is True
    assert review["rollback_evidence"]["checkpoint_restore_verified"] is True
    assert review["rollback_evidence"]["production_runtime_changed"] is False
    assert review["checkpoint_weight_evidence"]["transition_weight_count"] == 1
    payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert payload["artifact_kind"] == "marulho_snn_language_sparse_readout_checkpoint"
    assert payload["surface"] == "snn_language_sparse_readout_checkpoint.v1"
    assert payload["sparse_transition_weights"]
    assert payload["source_report_hash"] == report["provenance_evidence"]["report_hash"]
    assert review["runtime_truth_gate"]["checkpoint_status"] == "restore_verified"
    assert review["promotion_gate"]["eligible_for_checkpointed_bounded_readout_review"] is True


def test_readout_corpus_checkpoint_review_rejects_failed_evaluation_without_writing(tmp_path) -> None:
    corpus = _promotable_fixture()
    corpus["transition_memory_state"] = {"sparse_transition_weights": {}}
    report = evaluate_snn_language_readout_corpus(corpus, top_k=4)
    checkpoint_path = tmp_path / "blocked-checkpoint.json"

    review = build_snn_language_readout_corpus_checkpoint_review(
        corpus,
        report,
        checkpoint_path,
    )

    assert review["ready"] is False
    assert review["status"] == "reject_checkpointed_readout_collect_evidence"
    assert review["writes_checkpoint"] is False
    assert checkpoint_path.exists() is False
    assert "evaluation_passed" in review["runtime_truth_gate"]["reason_codes"]
    assert "transition_weights_available" in review["runtime_truth_gate"]["reason_codes"]


def test_readout_corpus_cli_writes_report(tmp_path) -> None:
    input_path = tmp_path / "corpus.json"
    output_path = tmp_path / "readout-corpus-evaluation.json"
    input_path.write_text(json.dumps(_promotable_fixture()), encoding="utf-8")
    stdout = io.StringIO()

    exit_code = readout_corpus_main(
        [
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--top-k",
            "4",
        ],
        stdout=stdout,
    )

    assert exit_code == 0
    assert output_path.exists()
    assert (tmp_path / "README.md").exists()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["surface"] == "snn_language_readout_corpus_evaluation.v1"
    assert payload["output_path"] == str(output_path)
    assert json.loads(stdout.getvalue())["passed"] is True


def test_readout_corpus_cli_writes_checkpoint_review(tmp_path) -> None:
    input_path = tmp_path / "corpus.json"
    output_path = tmp_path / "readout-corpus-evaluation.json"
    checkpoint_path = tmp_path / "readout-corpus-checkpoint.json"
    review_path = tmp_path / "readout-corpus-checkpoint-review.json"
    input_path.write_text(json.dumps(_promotable_fixture()), encoding="utf-8")
    stdout = io.StringIO()

    exit_code = readout_corpus_main(
        [
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--checkpoint-output",
            str(checkpoint_path),
            "--checkpoint-review-output",
            str(review_path),
            "--top-k",
            "4",
        ],
        stdout=stdout,
    )

    assert exit_code == 0
    assert output_path.exists()
    assert checkpoint_path.exists()
    assert review_path.exists()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    review = json.loads(review_path.read_text(encoding="utf-8"))
    assert payload["checkpoint_review"]["ready"] is True
    assert review["ready"] is True
    assert review["rollback_evidence"]["checkpoint_restore_verified"] is True
    assert json.loads(stdout.getvalue())["checkpoint_review"]["ready"] is True
