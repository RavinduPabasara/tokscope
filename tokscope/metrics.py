"""Tokenizer efficiency metrics.

Definitions:
- fertility: tokens per whitespace-separated word. 1.0 means one token per
  word; higher means the tokenizer fragments the text more. The standard
  measure of how well a tokenizer "fits" a language.
- chars_per_token / bytes_per_token: how much text each token carries.
  Lower fertility / higher chars-per-token generally means cheaper and
  better-conditioned inputs for an LLM.
- unk_rate: fraction of tokens that are the unknown token. Any nonzero
  value means the tokenizer cannot represent parts of the text at all.
- premium: ratio of this tokenizer's token count to a reference token
  count for parallel text (e.g. Sinhala vs English translation). A
  premium of 3.0 means the language costs 3x more tokens for the same
  content.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class Report:
    """Tokenization metrics for one tokenizer over one text."""

    tokenizer: str
    vocab_size: Optional[int]
    token_count: int
    word_count: int
    char_count: int
    byte_count: int
    unk_count: int
    sample_tokens: List[str] = field(default_factory=list)
    premium: Optional[float] = None

    @property
    def fertility(self) -> float:
        return self.token_count / self.word_count if self.word_count else 0.0

    @property
    def chars_per_token(self) -> float:
        return self.char_count / self.token_count if self.token_count else 0.0

    @property
    def bytes_per_token(self) -> float:
        return self.byte_count / self.token_count if self.token_count else 0.0

    @property
    def unk_rate(self) -> float:
        return self.unk_count / self.token_count if self.token_count else 0.0

    def as_dict(self) -> dict:
        d = {
            "tokenizer": self.tokenizer,
            "vocab_size": self.vocab_size,
            "token_count": self.token_count,
            "word_count": self.word_count,
            "fertility": round(self.fertility, 4),
            "chars_per_token": round(self.chars_per_token, 4),
            "bytes_per_token": round(self.bytes_per_token, 4),
            "unk_rate": round(self.unk_rate, 4),
        }
        if self.premium is not None:
            d["premium"] = round(self.premium, 4)
        return d


def build_report(
    name: str,
    text: str,
    tokens: List[str],
    unk_count: int,
    vocab_size: Optional[int],
    reference_token_count: Optional[int] = None,
    sample_size: int = 12,
) -> Report:
    """Assemble a Report from a tokenized text."""
    token_count = len(tokens)
    premium = None
    if reference_token_count:
        premium = token_count / reference_token_count
    return Report(
        tokenizer=name,
        vocab_size=vocab_size,
        token_count=token_count,
        word_count=len(text.split()),
        char_count=len(text),
        byte_count=len(text.encode("utf-8")),
        unk_count=unk_count,
        sample_tokens=tokens[:sample_size],
        premium=premium,
    )
