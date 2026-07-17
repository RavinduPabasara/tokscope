"""Tokenizer adapters.

A tokenizer spec is a string with an optional backend prefix:

- ``hf:xlm-roberta-base``  — any Hugging Face tokenizer (needs ``transformers``)
- ``tiktoken:gpt-4o``      — OpenAI tokenizers (needs ``tiktoken``)
- ``sp:/path/model.model`` — a SentencePiece model file (needs ``sentencepiece``)
- ``ws:``                  — whitespace splitter (no dependencies; fertility
  baseline of exactly 1.0 by construction)

A spec without a prefix defaults to the ``hf`` backend. Backends are imported
lazily so the core package has zero required dependencies.
"""

from __future__ import annotations

from typing import List, Optional, Protocol, Tuple


class Adapter(Protocol):
    name: str
    vocab_size: Optional[int]

    def tokenize(self, text: str) -> Tuple[List[str], int]:
        """Return (tokens, unk_count) for the text."""
        ...  # pragma: no cover


class WhitespaceAdapter:
    """Splits on whitespace. Zero-dependency baseline: fertility == 1.0."""

    def __init__(self, spec: str = "") -> None:
        self.name = "whitespace"
        self.vocab_size = None

    def tokenize(self, text: str) -> Tuple[List[str], int]:
        return text.split(), 0


class HuggingFaceAdapter:
    def __init__(self, model_name: str) -> None:
        try:
            from transformers import AutoTokenizer
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "the 'hf' backend needs transformers: pip install 'tokscope[hf]'"
            ) from e
        self._tok = AutoTokenizer.from_pretrained(model_name)
        self.name = model_name
        self.vocab_size = getattr(self._tok, "vocab_size", None)

    def tokenize(self, text: str) -> Tuple[List[str], int]:
        ids = self._tok.encode(text, add_special_tokens=False)
        tokens = self._tok.convert_ids_to_tokens(ids)
        unk_id = self._tok.unk_token_id
        unk_count = sum(1 for i in ids if i == unk_id) if unk_id is not None else 0
        return tokens, unk_count


class TiktokenAdapter:
    def __init__(self, model_name: str) -> None:
        try:
            import tiktoken
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "the 'tiktoken' backend needs tiktoken: pip install 'tokscope[tiktoken]'"
            ) from e
        try:
            self._enc = tiktoken.encoding_for_model(model_name)
        except KeyError:
            self._enc = tiktoken.get_encoding(model_name)
        self.name = f"tiktoken:{model_name}"
        self.vocab_size = self._enc.n_vocab

    def tokenize(self, text: str) -> Tuple[List[str], int]:
        ids = self._enc.encode(text)
        # Byte-level BPE has no UNK; decode per-token for display.
        tokens = [
            self._enc.decode_single_token_bytes(i).decode("utf-8", errors="replace")
            for i in ids
        ]
        return tokens, 0


class SentencePieceAdapter:
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

    def tokenize(self, text: str) -> Tuple[List[str], int]:
        ids = self._sp.encode_as_ids(text)
        tokens = [self._sp.id_to_piece(i) for i in ids]
        unk_id = self._sp.unk_id()
        unk_count = sum(1 for i in ids if i == unk_id) if unk_id >= 0 else 0
        return tokens, unk_count


_BACKENDS = {
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
