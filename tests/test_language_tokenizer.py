from __future__ import annotations

import torch

from marulho.data.language_tokenizer import (
    ByteLevelLanguageTokenizer,
    BytePairLanguageTokenizer,
    LANGUAGE_DOCUMENT_SEPARATOR,
    iter_language_corpus_chunks,
    iter_language_corpus_documents,
    load_language_tokenizer_state,
)
from marulho.training.language_model import (
    LanguageModelConfig,
    MarulhoLanguageModel,
    load_language_model_checkpoint,
    save_language_model_checkpoint,
)


CORPUS = (
    "MARULHO learns language continuously. Café data remains reversible.\n"
    "Replay protects prior knowledge while checkpoints make mutation rollbackable.\n"
) * 64


def test_language_corpus_chunks_reconstruct_text_at_document_boundaries() -> None:
    text = "first document\n\nsecond document is longer\n\nthird"
    chunks = list(iter_language_corpus_chunks((text,), max_characters=24))

    assert "".join(chunks) == text
    assert len(chunks) >= 2
    assert max(len(chunk) for chunk in chunks) <= 24
    assert list(iter_language_corpus_documents((text,))) == [
        "first document",
        "second document is longer",
        "third",
    ]
    explicit = LANGUAGE_DOCUMENT_SEPARATOR.join(
        ("first paragraph\n\nstill first story", "second story")
    )
    assert list(iter_language_corpus_documents((explicit,))) == [
        "first paragraph\n\nstill first story",
        "second story",
    ]


def test_marulho_bpe_tokenizer_is_lossless_compressed_and_checkpoint_owned() -> None:
    tokenizer = BytePairLanguageTokenizer.train([CORPUS], vocab_size=512)
    encoded = tokenizer.encode(CORPUS, add_bos=False, add_eos=False)
    byte_encoded = ByteLevelLanguageTokenizer().encode(
        CORPUS,
        add_bos=False,
        add_eos=False,
    )

    assert tokenizer.decode(encoded) == CORPUS
    assert tokenizer.unk_id not in encoded
    assert len(encoded) < len(byte_encoded) * 0.40
    assert tokenizer.state_dict()["vocabulary_trained_by_marulho"] is True
    assert tokenizer.state_dict()["loads_external_checkpoint"] is False

    restored = load_language_tokenizer_state(tokenizer.state_dict())
    assert isinstance(restored, BytePairLanguageTokenizer)
    assert restored.vocabulary_hash() == tokenizer.vocabulary_hash()
    assert restored.encode(CORPUS) == tokenizer.encode(CORPUS)
    assert restored.decode(restored.encode(CORPUS)) == CORPUS
    assert tokenizer.encode_batch(("MARULHO learns", "Café data")) == [
        tokenizer.encode("MARULHO learns"),
        tokenizer.encode("Café data"),
    ]


def test_language_checkpoint_round_trips_bpe_tokenizer(tmp_path) -> None:
    torch.manual_seed(11)
    tokenizer = BytePairLanguageTokenizer.train([CORPUS], vocab_size=512)
    model = MarulhoLanguageModel(
        LanguageModelConfig(
            vocab_size=tokenizer.vocab_size,
            embedding_dim=16,
            state_dim=16,
            state_core="transformer",
            state_layers=1,
            attention_heads=4,
            transformer_context_length=64,
            transformer_mlp_ratio=2.0,
        )
    ).eval()
    prompt = torch.tensor(
        [tokenizer.encode("MARULHO learns", add_eos=False)],
        dtype=torch.long,
    )
    with torch.no_grad():
        expected = model(prompt, collect_telemetry=False)["logits"]

    path = save_language_model_checkpoint(
        tmp_path / "bpe-checkpoint.pt",
        model,
        tokenizer,
        metadata={"phase": "tokenizer_test"},
    )
    restored_model, restored_tokenizer, metadata = load_language_model_checkpoint(path)
    restored_model.eval()
    with torch.no_grad():
        actual = restored_model(prompt, collect_telemetry=False)["logits"]

    assert isinstance(restored_tokenizer, BytePairLanguageTokenizer)
    assert restored_tokenizer.decode(restored_tokenizer.encode(CORPUS)) == CORPUS
    assert metadata == {"phase": "tokenizer_test"}
    assert torch.equal(actual, expected)
