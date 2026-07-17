"""Compare tokenizers over a text or corpus."""

from __future__ import annotations

from typing import List, Optional, Sequence, Union

from . import adapters
from .metrics import Report, build_report

TokenizerSpec = Union[str, adapters.Adapter]


def _resolve(spec: TokenizerSpec) -> adapters.Adapter:
    return adapters.load(spec) if isinstance(spec, str) else spec


def analyze(
    text: str,
    tokenizer: TokenizerSpec,
    reference_text: Optional[str] = None,
) -> Report:
    """Tokenize `text` with one tokenizer and compute metrics.

    If `reference_text` is given (a parallel version of the same content,
    e.g. its English translation), the report includes the token premium:
    tokens(text) / tokens(reference_text) under the same tokenizer.
    """
    adapter = _resolve(tokenizer)
    tokens, unk_count = adapter.tokenize(text)
    reference_token_count = None
    if reference_text is not None:
        ref_tokens, _ = adapter.tokenize(reference_text)
        reference_token_count = len(ref_tokens)
    return build_report(
        name=adapter.name,
        text=text,
        tokens=tokens,
        unk_count=unk_count,
        vocab_size=adapter.vocab_size,
        reference_token_count=reference_token_count,
    )


def compare(
    text: str,
    tokenizers: Sequence[TokenizerSpec],
    reference_text: Optional[str] = None,
) -> List[Report]:
    """Run `analyze` for several tokenizers, sorted by fertility (best first)."""
    reports = [analyze(text, t, reference_text) for t in tokenizers]
    return sorted(reports, key=lambda r: r.fertility)
