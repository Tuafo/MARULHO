from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Iterable, Mapping, Sequence


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
