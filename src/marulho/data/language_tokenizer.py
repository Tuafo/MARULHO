from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import (
    Any,
    Iterable,
    Iterator,
    Mapping,
    Protocol,
    Sequence,
    runtime_checkable,
)


BPE_TRAINING_CHUNK_CHARACTERS = 64 * 1024
LANGUAGE_SOURCE_ENCODING_CHUNK_CHARACTERS = 1024 * 1024


def iter_language_corpus_chunks(
    texts: Iterable[str],
    *,
    max_characters: int,
) -> Iterator[str]:
    maximum = max(1, int(max_characters))
    for raw_text in texts:
        text = str(raw_text)
        cursor = 0
        while cursor < len(text):
            end = min(len(text), cursor + maximum)
            if end < len(text):
                boundary = text.rfind("\n\n", cursor + 1, end)
                if boundary > cursor:
                    end = boundary + 2
            yield text[cursor:end]
            cursor = end


@dataclass(frozen=True)
class ByteLevelLanguageTokenizerState:
    surface: str
    special_tokens: tuple[str, ...]
    byte_offset: int
    normalization_policy: str
    vocab_size: int


class ByteLevelLanguageTokenizer:
    """Deterministic MARULHO-owned UTF-8 byte tokenizer with checkpoint state."""

    SURFACE = "marulho_byte_level_language_tokenizer.v1"
    NORMALIZATION_POLICY = "utf8_byte_identity"
    SPECIAL_TOKENS = (
        "<pad>",
        "<bos>",
        "<eos>",
        "<unk>",
        "<checkpoint>",
        "<replay>",
    )

    def __init__(self) -> None:
        self._token_to_id = {token: index for index, token in enumerate(self.SPECIAL_TOKENS)}
        self._id_to_token = {index: token for token, index in self._token_to_id.items()}
        self._byte_offset = len(self.SPECIAL_TOKENS)

    @property
    def pad_id(self) -> int:
        return self._token_to_id["<pad>"]

    @property
    def bos_id(self) -> int:
        return self._token_to_id["<bos>"]

    @property
    def eos_id(self) -> int:
        return self._token_to_id["<eos>"]

    @property
    def unk_id(self) -> int:
        return self._token_to_id["<unk>"]

    @property
    def checkpoint_id(self) -> int:
        return self._token_to_id["<checkpoint>"]

    @property
    def replay_id(self) -> int:
        return self._token_to_id["<replay>"]

    @property
    def byte_offset(self) -> int:
        return self._byte_offset

    @property
    def vocab_size(self) -> int:
        return self._byte_offset + 256

    def token_for_id(self, token_id: int) -> str:
        if token_id in self._id_to_token:
            return self._id_to_token[token_id]
        byte_value = int(token_id) - self._byte_offset
        if 0 <= byte_value <= 255:
            return f"<byte:{byte_value:02x}>"
        return "<unk>"

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> list[int]:
        token_ids: list[int] = []
        if add_bos:
            token_ids.append(self.bos_id)
        token_ids.extend(self._byte_offset + value for value in text.encode("utf-8"))
        if add_eos:
            token_ids.append(self.eos_id)
        return token_ids

    def decode(
        self,
        token_ids: Iterable[int],
        *,
        skip_special_tokens: bool = True,
    ) -> str:
        byte_values: list[int] = []
        text_parts: list[str] = []
        for raw_id in token_ids:
            token_id = int(raw_id)
            byte_value = token_id - self._byte_offset
            if 0 <= byte_value <= 255:
                byte_values.append(byte_value)
                continue
            if skip_special_tokens:
                continue
            if byte_values:
                text_parts.append(bytes(byte_values).decode("utf-8", errors="replace"))
                byte_values.clear()
            text_parts.append(self.token_for_id(token_id))
        if byte_values:
            text_parts.append(bytes(byte_values).decode("utf-8", errors="replace"))
        return "".join(text_parts)

    def state(self) -> ByteLevelLanguageTokenizerState:
        return ByteLevelLanguageTokenizerState(
            surface=self.SURFACE,
            special_tokens=tuple(self.SPECIAL_TOKENS),
            byte_offset=self._byte_offset,
            normalization_policy=self.NORMALIZATION_POLICY,
            vocab_size=self.vocab_size,
        )

    def state_dict(self) -> dict[str, Any]:
        state = self.state()
        return {
            "surface": state.surface,
            "special_tokens": list(state.special_tokens),
            "byte_offset": state.byte_offset,
            "normalization_policy": state.normalization_policy,
            "vocab_size": state.vocab_size,
            "external_dependency": False,
            "loads_external_checkpoint": False,
            "tokenizer_owner": "marulho.data",
        }

    @classmethod
    def load_state_dict(
        cls,
        state: Mapping[str, Any],
    ) -> "ByteLevelLanguageTokenizer":
        tokenizer = cls()
        expected = tokenizer.state_dict()
        for key in ("surface", "special_tokens", "byte_offset", "vocab_size"):
            if state.get(key) != expected[key]:
                raise ValueError(
                    f"Unsupported language tokenizer checkpoint field {key}: "
                    f"{state.get(key)!r}"
                )
        normalization = state.get("normalization_policy")
        if normalization != expected["normalization_policy"]:
            raise ValueError(
                "Unsupported language tokenizer normalization policy: "
                f"{normalization!r}"
            )
        return tokenizer

    def vocabulary_hash(self) -> str:
        payload = {
            "surface": self.SURFACE,
            "special_tokens": list(self.SPECIAL_TOKENS),
            "byte_offset": self._byte_offset,
            "normalization_policy": self.NORMALIZATION_POLICY,
            "byte_values": list(range(256)),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def sequence_hash(self, token_ids: Sequence[int]) -> str:
        encoded = json.dumps(
            [int(token_id) for token_id in token_ids],
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@runtime_checkable
class LanguageTokenizer(Protocol):
    """Checkpoint-owned tokenizer contract used by MARULHO language models."""

    @property
    def pad_id(self) -> int: ...

    @property
    def bos_id(self) -> int: ...

    @property
    def eos_id(self) -> int: ...

    @property
    def unk_id(self) -> int: ...

    @property
    def checkpoint_id(self) -> int: ...

    @property
    def replay_id(self) -> int: ...

    @property
    def vocab_size(self) -> int: ...

    def token_for_id(self, token_id: int) -> str: ...

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> list[int]: ...

    def decode(
        self,
        token_ids: Iterable[int],
        *,
        skip_special_tokens: bool = True,
    ) -> str: ...

    def state_dict(self) -> dict[str, Any]: ...

    def vocabulary_hash(self) -> str: ...

    def sequence_hash(self, token_ids: Sequence[int]) -> str: ...


class BytePairLanguageTokenizer:
    """MARULHO-trained byte-level BPE vocabulary stored whole in checkpoints."""

    SURFACE = "marulho_byte_pair_language_tokenizer.v1"
    NORMALIZATION_POLICY = "utf8_identity_bytelevel_bpe"
    SPECIAL_TOKENS = ByteLevelLanguageTokenizer.SPECIAL_TOKENS

    @staticmethod
    def _byte_decoder() -> dict[str, int]:
        visible = list(range(ord("!"), ord("~") + 1))
        visible += list(range(ord("¡"), ord("¬") + 1))
        visible += list(range(ord("®"), ord("ÿ") + 1))
        mapped = list(visible)
        extra = 0
        for byte_value in range(256):
            if byte_value not in visible:
                visible.append(byte_value)
                mapped.append(256 + extra)
                extra += 1
        return {
            chr(codepoint): int(byte_value)
            for byte_value, codepoint in zip(visible, mapped)
        }

    def __init__(self, tokenizer_json: str) -> None:
        from tokenizers import Tokenizer

        self._tokenizer_json = str(tokenizer_json)
        self._tokenizer = Tokenizer.from_str(self._tokenizer_json)
        self._token_to_id: dict[str, int] = {}
        for token in self.SPECIAL_TOKENS:
            token_id = self._tokenizer.token_to_id(token)
            if token_id is None:
                raise ValueError(f"BPE tokenizer is missing special token {token!r}")
            self._token_to_id[token] = int(token_id)
        if int(self._tokenizer.get_vocab_size(with_added_tokens=True)) <= len(
            self.SPECIAL_TOKENS
        ):
            raise ValueError("BPE tokenizer vocabulary contains no learned tokens")

    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        *,
        vocab_size: int = 4096,
        min_frequency: int = 2,
        max_token_length: int = 64,
    ) -> "BytePairLanguageTokenizer":
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

        requested_vocab = max(512, int(vocab_size))
        tokenizer = Tokenizer(
            models.BPE(
                unk_token="<unk>",
                byte_fallback=True,
            )
        )
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
            add_prefix_space=False,
            trim_offsets=False,
            use_regex=True,
        )
        tokenizer.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=requested_vocab,
            min_frequency=max(1, int(min_frequency)),
            show_progress=False,
            special_tokens=list(cls.SPECIAL_TOKENS),
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            max_token_length=max(2, int(max_token_length)),
        )
        tokenizer.train_from_iterator(
            iter_language_corpus_chunks(
                texts,
                max_characters=BPE_TRAINING_CHUNK_CHARACTERS,
            ),
            trainer=trainer,
        )
        return cls(tokenizer.to_str(pretty=False))

    @property
    def pad_id(self) -> int:
        return self._token_to_id["<pad>"]

    @property
    def bos_id(self) -> int:
        return self._token_to_id["<bos>"]

    @property
    def eos_id(self) -> int:
        return self._token_to_id["<eos>"]

    @property
    def unk_id(self) -> int:
        return self._token_to_id["<unk>"]

    @property
    def checkpoint_id(self) -> int:
        return self._token_to_id["<checkpoint>"]

    @property
    def replay_id(self) -> int:
        return self._token_to_id["<replay>"]

    @property
    def vocab_size(self) -> int:
        return int(self._tokenizer.get_vocab_size(with_added_tokens=True))

    def token_for_id(self, token_id: int) -> str:
        token = self._tokenizer.id_to_token(int(token_id))
        return str(token) if token is not None else "<unk>"

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> list[int]:
        token_ids: list[int] = []
        if add_bos:
            token_ids.append(self.bos_id)
        token_ids.extend(
            int(token_id)
            for token_id in self._tokenizer.encode(
                str(text),
                add_special_tokens=False,
            ).ids
        )
        if add_eos:
            token_ids.append(self.eos_id)
        return token_ids

    def decode(
        self,
        token_ids: Iterable[int],
        *,
        skip_special_tokens: bool = True,
    ) -> str:
        decoder = self._byte_decoder()
        byte_values = bytearray()
        text_parts: list[str] = []
        special_ids = set(self._token_to_id.values())

        def _flush_bytes() -> None:
            if byte_values:
                text_parts.append(bytes(byte_values).decode("utf-8", errors="replace"))
                byte_values.clear()

        for raw_id in token_ids:
            token_id = int(raw_id)
            if token_id in special_ids:
                if not skip_special_tokens:
                    _flush_bytes()
                    text_parts.append(self.token_for_id(token_id))
                continue
            token = self._tokenizer.id_to_token(token_id)
            if token is None:
                if not skip_special_tokens:
                    _flush_bytes()
                    text_parts.append("<unk>")
                continue
            for character in token:
                byte_value = decoder.get(character)
                if byte_value is None:
                    byte_values.extend(character.encode("utf-8"))
                else:
                    byte_values.append(byte_value)
        _flush_bytes()
        return "".join(text_parts)

    def state_dict(self) -> dict[str, Any]:
        return {
            "surface": self.SURFACE,
            "special_tokens": list(self.SPECIAL_TOKENS),
            "normalization_policy": self.NORMALIZATION_POLICY,
            "vocab_size": self.vocab_size,
            "tokenizer_json": self._tokenizer_json,
            "external_dependency": "tokenizers",
            "loads_external_checkpoint": False,
            "vocabulary_trained_by_marulho": True,
            "tokenizer_owner": "marulho.data",
        }

    @classmethod
    def load_state_dict(
        cls,
        state: Mapping[str, Any],
    ) -> "BytePairLanguageTokenizer":
        if state.get("surface") != cls.SURFACE:
            raise ValueError(
                f"Unsupported BPE tokenizer surface: {state.get('surface')!r}"
            )
        if state.get("special_tokens") != list(cls.SPECIAL_TOKENS):
            raise ValueError("Unsupported BPE tokenizer special-token layout")
        if state.get("normalization_policy") != cls.NORMALIZATION_POLICY:
            raise ValueError("Unsupported BPE tokenizer normalization policy")
        tokenizer_json = state.get("tokenizer_json")
        if not isinstance(tokenizer_json, str) or not tokenizer_json:
            raise ValueError("BPE tokenizer checkpoint is missing tokenizer_json")
        tokenizer = cls(tokenizer_json)
        if int(state.get("vocab_size", -1)) != tokenizer.vocab_size:
            raise ValueError("BPE tokenizer checkpoint vocab_size does not match JSON")
        return tokenizer

    def vocabulary_hash(self) -> str:
        return hashlib.sha256(self._tokenizer_json.encode("utf-8")).hexdigest()

    def sequence_hash(self, token_ids: Sequence[int]) -> str:
        encoded = json.dumps(
            [int(token_id) for token_id in token_ids],
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def load_language_tokenizer_state(state: Mapping[str, Any]) -> LanguageTokenizer:
    """Restore the concrete checkpoint-owned tokenizer by its explicit surface."""

    surface = state.get("surface")
    if surface == ByteLevelLanguageTokenizer.SURFACE:
        return ByteLevelLanguageTokenizer.load_state_dict(state)
    if surface == BytePairLanguageTokenizer.SURFACE:
        return BytePairLanguageTokenizer.load_state_dict(state)
    raise ValueError(f"Unsupported language tokenizer surface: {surface!r}")
