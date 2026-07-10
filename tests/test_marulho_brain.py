from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import time

from fastapi.testclient import TestClient
import torch

from marulho.brain import MarulhoBrain
from marulho.config.model_config import MarulhoConfig
from marulho.data.language_tokenizer import ByteLevelLanguageTokenizer
from marulho.service.api import create_app
from marulho.training.language_model import (
    LanguageModelConfig,
    MarulhoLanguageModel,
    build_language_model_splits,
    evaluate_language_model,
    save_language_model_checkpoint,
)


def _tiny_config() -> MarulhoConfig:
    return MarulhoConfig(
        n_columns=16,
        column_latent_dim=16,
        window_size=4,
        bootstrap_tokens=0,
        k_routing=4,
        memory_capacity=128,
        enable_learned_chunking=False,
        micro_sleep_interval_tokens=10_000,
        deep_sleep_interval_tokens=10_000,
        trainer_telemetry_interval_tokens=10_000,
        device="cpu",
    )


def _trained_brain() -> MarulhoBrain:
    brain = MarulhoBrain.fresh(_tiny_config())
    feed = brain.feed("marulho learns local sparse loops.", source="test")
    assert feed["accepted_tokens"] > 0
    tick = brain.tick(tokens=feed["accepted_tokens"], source="test")
    assert tick["trained_tokens"] == feed["accepted_tokens"]
    return brain


def _language_model_fixture() -> tuple[
    MarulhoLanguageModel,
    ByteLevelLanguageTokenizer,
    dict[str, object],
]:
    torch.manual_seed(20260710)
    tokenizer = ByteLevelLanguageTokenizer()
    split = build_language_model_splits(
        [
            "marulho predicts local language from a checkpoint-owned transformer. " * 4,
            "heldout evaluation remains separate from the training windows. " * 4,
        ],
        tokenizer,
        sequence_length=8,
        eval_fraction=0.25,
    )
    model = MarulhoLanguageModel(
        LanguageModelConfig(
            vocab_size=tokenizer.vocab_size,
            embedding_dim=16,
            state_dim=16,
            state_layers=1,
            attention_heads=2,
            transformer_context_length=32,
        )
    )
    report = evaluate_language_model(model, split.eval)
    return model, tokenizer, report


def test_language_split_reports_text_tokens_without_reencoding() -> None:
    texts = ("first source", "second source")
    tokenizer = ByteLevelLanguageTokenizer()
    split = build_language_model_splits(
        texts,
        tokenizer,
        eval_texts=("held out",),
        sequence_length=4,
    )

    expected_train_tokens = sum(
        len(tokenizer.encode(text, add_bos=False, add_eos=False))
        for text in texts
    )
    expected_eval_tokens = len(
        tokenizer.encode("held out", add_bos=False, add_eos=False)
    )
    assert split.report["train_text_token_count"] == expected_train_tokens
    assert split.report["train_token_stream_count"] == expected_train_tokens + 4
    assert split.report["eval_text_token_count"] == expected_eval_tokens
    assert split.report["eval_token_stream_count"] == expected_eval_tokens + 2
    digest = hashlib.sha256()
    for batch in split.train:
        digest.update(batch.input_ids.contiguous().numpy().tobytes())
        digest.update(batch.target_ids.contiguous().numpy().tobytes())
    assert split.report["train_split_hash"] == digest.hexdigest()
    assert len(split.train) > 1
    assert (
        split.train[0].input_ids.untyped_storage().data_ptr()
        == split.train[1].input_ids.untyped_storage().data_ptr()
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_marulho_brain_feed_tick_generate_and_trace() -> None:
    brain = _trained_brain()

    generation = brain.generate(prompt="mar", max_tokens=12)
    status = brain.status()

    assert generation["owned_by_marulho"] is True
    assert generation["external_llm_used"] is False
    assert generation["emitted_tokens"] > 0
    assert status["surface"] == "marulho_brain_runtime.v1"
    assert status["readout"]["observed_transition_count"] > 0
    assert status["last_trace"]["executor"] == brain.trainer.config.cuda_graph_sequence_executor


def test_marulho_brain_checkpoint_roundtrip_preserves_readout(tmp_path: Path) -> None:
    brain = _trained_brain()
    before = brain.generate(max_tokens=12)
    saved = brain.save(tmp_path / "brain.pt")
    restored = MarulhoBrain.load(saved["path"])
    after = restored.generate(max_tokens=12)

    assert restored.status()["token_count"] == brain.status()["token_count"]
    assert after["text"] == before["text"]


def test_marulho_brain_owns_transformer_checkpoint_lifecycle(tmp_path: Path) -> None:
    brain = MarulhoBrain.fresh(_tiny_config())
    model, tokenizer, report = _language_model_fixture()

    install = brain.install_language_model(model, tokenizer, evaluation_report=report)
    generation = brain.generate(prompt="marulho", max_tokens=4)
    saved = brain.save(tmp_path / "brain-transformer.pt")
    restored = MarulhoBrain.load(saved["path"])
    restored_generation = restored.generate(prompt="marulho", max_tokens=4)
    status = restored.status()

    assert install["active_language_path"] == "marulho_transformer"
    assert generation["surface"] == "marulho_brain_transformer_generation.v2"
    assert generation["state_core"] == "transformer"
    assert generation["external_llm_used"] is False
    assert generation["loads_external_checkpoint"] is False
    assert generation["tokenizer_hash"] == tokenizer.vocabulary_hash()
    assert restored_generation["generated_token_ids"] == generation["generated_token_ids"]
    assert status["active_language_path"] == "marulho_transformer"
    assert status["language_model"]["installed"] is True
    assert status["language_model"]["state_core"] == "transformer"
    assert status["language_model"]["continual_learning_enabled"] is False
    assert status["language_model"]["structural_plasticity_enabled"] is False


def test_direct_reviewed_transformer_checkpoint_install_is_hash_guarded(
    tmp_path: Path,
) -> None:
    brain = MarulhoBrain.fresh(_tiny_config())
    model, tokenizer, _report = _language_model_fixture()
    checkpoint = tmp_path / "transformer.pt"
    save_language_model_checkpoint(checkpoint, model, tokenizer, metadata={"source": "test"})
    checkpoint_hash = _sha256_file(checkpoint)

    blocked = brain.install_language_checkpoint_from_direct_review(
        checkpoint,
        expected_sha256="wrong",
        operator_approved=True,
    )
    installed = brain.install_language_checkpoint_from_direct_review(
        checkpoint,
        expected_sha256=checkpoint_hash,
        operator_approved=True,
        operator_id="pytest",
    )

    assert blocked["installed"] is False
    assert blocked["candidate_checkpoint"]["hash_verified"] is False
    assert installed["installed"] is True
    assert installed["active_language_path"] == "marulho_transformer"
    assert installed["external_llm_used"] is False


def test_marulho_brain_sustained_transformer_generation(tmp_path: Path) -> None:
    brain = MarulhoBrain.fresh(_tiny_config())
    model, tokenizer, report = _language_model_fixture()
    brain.install_language_model(model, tokenizer, evaluation_report=report)

    generation = brain.generate_sustained_language(
        output_path=tmp_path / "sustained.json",
        target_tokens=8,
        prompt="MARULHO",
        timeout_seconds=60.0,
    )

    assert generation["surface"] == "marulho_brain_sustained_language_generation.v1"
    assert generation["runtime_owner"] == "MarulhoBrain"
    assert generation["success"] is True
    assert generation["token_delta"] == 8
    assert generation["runtime"]["state_core"] == "transformer"
    assert generation["runtime"]["spiking_present"] is False
    assert generation["external_llm_used"] is False


def test_service_restores_brain_owned_transformer(tmp_path: Path) -> None:
    brain = MarulhoBrain.fresh(_tiny_config())
    model, tokenizer, report = _language_model_fixture()
    brain.install_language_model(model, tokenizer, evaluation_report=report)
    checkpoint = tmp_path / "service-transformer.pt"
    brain.save(checkpoint)
    app = create_app(
        checkpoint_path=checkpoint,
        trace_dir=tmp_path / "traces",
        env_root=tmp_path,
    )

    with TestClient(app) as client:
        generation = client.post(
            "/brain/generate",
            json={"prompt": "marulho", "max_tokens": 4},
        )
        status = client.get("/brain/status")

    assert generation.status_code == 200
    assert generation.json()["active_language_path"] == "marulho_transformer"
    assert generation.json()["external_llm_used"] is False
    assert status.status_code == 200
    assert status.json()["language_model"]["state_core"] == "transformer"


def test_marulho_brain_replay_growth_and_loop_remain_brain_owned() -> None:
    brain = _trained_brain()

    replay = brain.replay(window="micro", cycles=1)
    growth = brain.grow_prune(budget="small")
    start = brain.start(tick_tokens=4, interval_seconds=0.01, source="loop-test")
    deadline = time.time() + 1.0
    while time.time() < deadline and brain.status()["loop"]["tick_count"] == 0:
        time.sleep(0.01)
    stop = brain.stop(timeout_seconds=1.0)

    assert replay["surface"] == "marulho_brain_replay.v1"
    assert growth["surface"] == "marulho_brain_growth_prune.v1"
    assert start["loop"]["owner"] == "MarulhoBrain"
    assert stop["loop"]["running"] is False


def test_deleted_legacy_service_surfaces_are_not_importable() -> None:
    for module_name in (
        "marulho.service.manager",
        "marulho.service.brain_runtime",
        "marulho.service.runtime_control",
        "marulho.service.runtime_facade",
        "marulho.service.status_read_model",
    ):
        assert importlib.util.find_spec(module_name) is None
