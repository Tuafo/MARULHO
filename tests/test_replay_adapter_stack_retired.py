from __future__ import annotations

import importlib.util

import pytest

from marulho.evaluation.artifact_io import (
    _sha256_json,
    hash_json_file,
    load_json_object,
)
from marulho.service.api import REPORT_SUMMARY_KINDS


RETIRED_REPLAY_ADAPTER_MODULES = (
    "marulho.training.replay_adapter_experiment",
    "marulho.evaluation.replay_adapter_promotion_gate",
    "marulho.evaluation.replay_adaptation_experiment_1",
    "marulho.evaluation.replay_training_plan",
    "marulho.evaluation.replay_training_approval",
)


def test_replay_adapter_stack_modules_are_deleted() -> None:
    for module_name in RETIRED_REPLAY_ADAPTER_MODULES:
        assert importlib.util.find_spec(module_name) is None, module_name


def test_replay_adapter_report_kinds_are_not_service_visible() -> None:
    assert "terminus_replay_adapter_promotion_gate" not in REPORT_SUMMARY_KINDS
    assert "terminus_replay_adaptation_experiment_1" not in REPORT_SUMMARY_KINDS


def test_artifact_io_replaces_replay_training_approval_helpers(tmp_path) -> None:
    path = tmp_path / "artifact.json"
    path.write_text('{"kind": "utility", "value": 3}', encoding="utf-8")

    loaded = load_json_object(path, label="Artifact")
    loaded_again, digest = hash_json_file(path, label="Artifact")

    assert loaded == {"kind": "utility", "value": 3}
    assert loaded_again == loaded
    assert digest == _sha256_json(loaded)

    bad_path = tmp_path / "list.json"
    bad_path.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ValueError, match="Artifact must be a JSON object"):
        load_json_object(bad_path, label="Artifact")
