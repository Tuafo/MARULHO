from __future__ import annotations

import json

import pytest

from marulho.evaluation.language_training_experiment import (
    SURFACE,
    LanguageTrainingExperimentConfig,
    run_language_training_experiment,
)
from marulho.training.language_model import load_language_model_checkpoint


def test_transformer_training_experiment_trains_and_restores(tmp_path) -> None:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(
        (
            "MARULHO trains a local causal Transformer. "
            "The checkpoint owns its learned subword vocabulary.\n"
        )
        * 96,
        encoding="utf-8",
    )
    output = tmp_path / "transformer.json"
    report = run_language_training_experiment(
        output_path=output,
        corpus_path=corpus,
        prompts=("MARULHO trains",),
        config=LanguageTrainingExperimentConfig(
            tokenizer_kind="bpe",
            tokenizer_vocab_size=512,
            embedding_dim=16,
            state_dim=16,
            state_layers=1,
            attention_heads=4,
            transformer_context_length=32,
            transformer_mlp_ratio=2.0,
            sequence_length=16,
            stride=8,
            batch_size=2,
            max_train_batches=2,
            max_eval_batches=2,
            train_epochs=1,
            generation_tokens=2,
            sustained_target_tokens=0,
            device="cpu",
        ),
    )
    written = json.loads(output.read_text(encoding="utf-8"))

    assert report["surface"] == SURFACE == written["surface"]
    assert report["state_core"] == "transformer"
    assert report["external_llm_used"] is False
    assert report["training"]["token_count"] == 64
    assert report["training"]["optimizer"] == "AdamW"
    assert report["training"]["loss_record_count"] == 2
    assert report["training"]["per_step_host_metric_readback"] is False
    assert report["training"]["peak_cuda_memory_bytes"] == 0
    assert report["eval_before"]["heldout_loss"] > 0.0
    assert report["eval_after"]["heldout_loss"] > 0.0
    assert report["tokenizer"]["vocabulary_trained_by_marulho"] is True
    assert report["experiment_review"]["recurrent_language_path_present"] is False
    assert report["experiment_review"]["routing_present"] is False
    assert report["sustained_summary"] is None
    checkpoint = tmp_path / "transformer-checkpoint.pt"
    model, tokenizer, metadata = load_language_model_checkpoint(checkpoint)
    assert model.config.state_core == "transformer"
    assert tokenizer.vocab_size == model.config.vocab_size
    assert metadata["experiment_report"] == str(output)


def test_transformer_training_rejects_recurrent_configuration(tmp_path) -> None:
    with pytest.raises(ValueError, match="only Transformer"):
        run_language_training_experiment(
            output_path=tmp_path / "rejected.json",
            config=LanguageTrainingExperimentConfig(
                tokenizer_kind="byte",
                state_core="gru",
                embedding_dim=16,
                state_dim=16,
                state_layers=1,
                attention_heads=4,
                transformer_context_length=32,
                sequence_length=16,
                batch_size=2,
                max_train_batches=1,
                max_eval_batches=1,
                generation_tokens=0,
                device="cpu",
            ),
        )
