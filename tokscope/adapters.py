"""Tokenizer adapters.

A tokenizer spec is a string with an optional backend prefix:

- ``hf:xlm-roberta-base``  — any Hugging Face tokenizer (needs ``transformers``)
- ``tiktoken:gpt-4o``      — OpenAI tokenizers (needs ``tiktoken``)
- ``sp:/path/model.model`` — a SentencePiece model file (needs ``sentencepiece``)
- ``ws:``                  — whitespace splitter (no dependencies; fertility
  baseline of exactly 1.0 by construction)

A spec without a prefix defaults to the ``hf`` backend. Backends are imported
lazily so the core package has zero required dependencies.

Custom adapters implement ``tokenize(text) -> (tokens, unk_count)``. To also
report byte tokens, add ``tokenize_with_bytes(text) -> (tokens, unk_count,
byte_token_count)``, as the built-in adapters do.
"""

from __future__ import annotations

import re
from typing import Callable, Dict, List, Optional, Protocol, Sequence, Tuple, cast

WithBytes = Tuple[List[str], int, int]

# SentencePiece byte-fallback pieces: <0x00> ... <0xFF>
_BYTE_PIECE = re.compile(r"^<0x[0-9A-Fa-f]{2}>$")


class Adapter(Protocol):
    name: str
    vocab_size: Optional[int]

    def tokenize(self, text: str) -> Tuple[List[str], int]:
        """Return (tokens, unk_count) for the text."""
        ...  # pragma: no cover


class _BuiltinAdapter:
    name: str
    vocab_size: Optional[int]

    def tokenize(self, text: str) -> Tuple[List[str], int]:
        tokens, unk_count, _ = self.tokenize_with_bytes(text)
        return tokens, unk_count

    def tokenize_with_bytes(self, text: str) -> WithBytes:
        raise NotImplementedError  # pragma: no cover


def run(adapter: Adapter, text: str) -> Tuple[List[str], int, Optional[int]]:
    """Tokenize with any adapter; byte_token_count is None if not reported."""
    with_bytes = getattr(adapter, "tokenize_with_bytes", None)
    if with_bytes is not None:
        tokens, unk_count, byte_count = with_bytes(text)
        return tokens, unk_count, byte_count
    tokens, unk_count = adapter.tokenize(text)
    return tokens, unk_count, None


def _count_partial(
    ids: Sequence[int], is_partial: Dict[int, bool], check: Callable[[int], bool]
) -> int:
    """Count ids whose token alone is not whole characters, caching per id."""
    count = 0
    for i in ids:
        partial = is_partial.get(i)
        if partial is None:
            partial = is_partial[i] = check(i)
        count += partial
    return count


class WhitespaceAdapter(_BuiltinAdapter):
    """Splits on whitespace. Zero-dependency baseline: fertility == 1.0."""

    def __init__(self, spec: str = "") -> None:
        self.name = "whitespace"
        self.vocab_size = None

    def tokenize_with_bytes(self, text: str) -> WithBytes:
        return text.split(), 0, 0


class HuggingFaceAdapter(_BuiltinAdapter):
    def __init__(self, model_name: str) -> None:
        try:
            from transformers import AutoTokenizer
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "the 'hf' backend needs transformers: pip install 'tokscope[hf]'"
            ) from e
        self._tok = AutoTokenizer.from_pretrained(model_name)
        self.name = model_name
        # len() includes added tokens; .vocab_size counts only the base vocab.
        self.vocab_size = len(self._tok)
        self._partial: Dict[int, bool] = {}

    def _is_partial(self, token_id: int) -> bool:
        token = self._tok.convert_ids_to_tokens(token_id)
        if isinstance(token, str) and _BYTE_PIECE.match(token):
            return True
        # Byte-level BPE: a token holding part of a character decodes to U+FFFD.
        return "\ufffd" in self._tok.decode([token_id])

    def tokenize_with_bytes(self, text: str) -> WithBytes:
        ids = self._tok.encode(text, add_special_tokens=False)
        tokens = cast(List[str], self._tok.convert_ids_to_tokens(ids))
        unk_id = self._tok.unk_token_id
        unk_count = sum(1 for i in ids if i == unk_id) if unk_id is not None else 0
        byte_count = _count_partial(ids, self._partial, self._is_partial)
        return tokens, unk_count, byte_count


class TiktokenAdapter(_BuiltinAdapter):
    def __init__(self, model_name: str) -> None:
        try:
            import tiktoken
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "the 'tiktoken' backend needs tiktoken: "
                "pip install 'tokscope[tiktoken]'"
            ) from e
        try:
            self._enc = tiktoken.encoding_for_model(model_name)
        except KeyError:
            self._enc = tiktoken.get_encoding(model_name)
        self.name = f"tiktoken:{model_name}"
        self.vocab_size = self._enc.n_vocab
        self._partial: Dict[int, bool] = {}

    def _is_partial(self, token_id: int) -> bool:
        try:
            self._enc.decode_single_token_bytes(token_id).decode("utf-8")
        except UnicodeDecodeError:
            return True
        return False

    def tokenize_with_bytes(self, text: str) -> WithBytes:
        # disallowed_special=() treats text like "<|endoftext|>" as plain text
        ids = self._enc.encode(text, disallowed_special=())
        # Byte-level BPE has no UNK; decode per-token for display.
        tokens = [
            self._enc.decode_single_token_bytes(i).decode("utf-8", errors="replace")
            for i in ids
        ]
        byte_count = _count_partial(ids, self._partial, self._is_partial)
        return tokens, 0, byte_count


class SentencePieceAdapter(_BuiltinAdapter):
    def __init__(self, model_path: str) -> None:
        try:
            import sentencepiece as spm
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "the 'sp' backend needs sentencepiece: pip install 'tokscope[sp]'"
            ) from e
        self._sp = spm.SentencePieceProcessor()
        self._sp.load(model_path)
        self.name = f"sp:{model_path.rsplit('/', 1)[-1]}"
        self.vocab_size = self._sp.get_piece_size()

    def tokenize_with_bytes(self, text: str) -> WithBytes:
        ids = self._sp.encode_as_ids(text)
        tokens = [self._sp.id_to_piece(i) for i in ids]
        unk_id = self._sp.unk_id()
        unk_count = sum(1 for i in ids if i == unk_id) if unk_id >= 0 else 0
        byte_count = sum(1 for i in ids if self._sp.is_byte(i))
        return tokens, unk_count, byte_count


_BACKENDS: Dict[str, Callable[[str], Adapter]] = {
    "hf": HuggingFaceAdapter,
    "tiktoken": TiktokenAdapter,
    "sp": SentencePieceAdapter,
    "ws": WhitespaceAdapter,
}


def load(spec: str) -> Adapter:
    """Load a tokenizer from a spec string like 'tiktoken:gpt-4o'."""
    backend, sep, rest = spec.partition(":")
    if not sep or backend not in _BACKENDS:
        backend, rest = "hf", spec
    return _BACKENDS[backend](rest)
