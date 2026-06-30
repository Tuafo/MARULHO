from __future__ import annotations

import pytest

from marulho.evaluation.artifact_io import (
    _sha256_json,
    hash_json_file,
    load_json_object,
)


def test_artifact_io_loads_and_hashes_json_objects(tmp_path) -> None:
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
