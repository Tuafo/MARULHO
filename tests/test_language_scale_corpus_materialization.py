from __future__ import annotations

from marulho.evaluation.language_scale_corpus_materialization import (
    _generic_filter_reason,
    _hash_list_sha256,
)


def test_v80_generic_filter_rejects_templates_and_broken_encoding() -> None:
    assert _generic_filter_reason("This chapter will" + " text" * 500) == (
        "contains_rejected_template_phrase"
    )
    assert _generic_filter_reason("clean" * 500 + "\ufffd" * 20) == (
        "replacement_character_ratio_above_0_001"
    )
    assert _generic_filter_reason("Natural prose " * 300) is None


def test_v80_hash_list_digest_is_order_sensitive() -> None:
    assert _hash_list_sha256(["a", "b"]) != _hash_list_sha256(["b", "a"])
    assert _hash_list_sha256(["a", "b"]) == _hash_list_sha256(["a", "b"])
