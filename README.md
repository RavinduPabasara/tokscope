# tokscope

**Measure how efficiently LLM tokenizers handle your language.**

[![PyPI version](https://img.shields.io/pypi/v/tokscope.svg)](https://pypi.org/project/tokscope/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Tokenizers are the invisible tax on every LLM: a language that fragments into
3× more tokens costs 3× more money, fills the context window 3× faster, and is
usually understood worse. For low-resource languages the situation can be
catastrophic — multilingual BERT tokenizes Sinhala text into **100% `[UNK]`
tokens** (it literally cannot see the language).

`tokscope` makes this measurable in one command: **fertility** (tokens per
word), **UNK rate**, **byte-token rate**, **chars/bytes per token**, and the
**token premium** your language pays relative to a parallel English text.

## Installation

```bash
pip install tokscope            # zero-dependency core
pip install 'tokscope[all]'     # + transformers, tiktoken, sentencepiece backends
```

## CLI

```bash
tokscope corpus_si.txt \
    -t xlm-roberta-base \
    -t google/mt5-small \
    -t tiktoken:gpt-4o \
    -t sp:sinhala_bpe.model \
    --reference corpus_en.txt \
    --csv results.csv
```

```
Tokenizer                           Vocab     Tokens     Fertility  Chars/tok  UNK rate
---------------------------------------------------------------------------------------
sp:sinhala_bpe_v2.model             32000     41126      1.5801     4.05       0.0007
xlm-roberta-base                    250002    52982      2.0357     3.1437     0.0
tiktoken:gpt-4o                     200019    84752      3.2563     1.9653     0.0
tiktoken:cl100k_base                100277    254974     9.7965     0.6532     0.0
```

(Real measurements on OSCAR Sinhala — full results in [the study](study/README.md).)

Tokenizer specs: `hf:NAME` (default), `tiktoken:MODEL`, `sp:PATH`, and `ws:`
(whitespace baseline — fertility exactly 1.0 by construction).

## Python API

```python
import tokscope

report = tokscope.analyze(text, "tiktoken:gpt-4o", reference_text=english_text)
report.fertility        # tokens per word
report.unk_rate         # fraction of tokens the tokenizer can't represent
report.byte_token_rate  # fraction of tokens that are raw bytes, not characters
report.premium          # token cost multiplier vs the reference text

reports = tokscope.compare(text, ["xlm-roberta-base", "tiktoken:gpt-4o", "ws:"])
```

Custom tokenizers: pass any object with a `name`, a `vocab_size`, and a
`tokenize(text) -> (tokens, unk_count)` method. To report byte tokens too,
add `tokenize_with_bytes(text) -> (tokens, unk_count, byte_token_count)`.

## Metrics

| Metric | Definition | Why it matters |
|---|---|---|
| fertility | tokens ÷ whitespace words | The standard measure of tokenizer fit; higher = more fragmentation |
| chars_per_token | characters ÷ tokens | How much text each token carries |
| bytes_per_token | UTF-8 bytes ÷ tokens | Script-independent capacity measure |
| unk_rate | UNK tokens ÷ tokens | Any nonzero value = the tokenizer cannot represent the text |
| byte_token_rate | byte tokens ÷ tokens | Byte-fallback pieces (`<0xE0>`) or tokens holding part of a character. These tokenizers never emit UNK, so this is how you see that they barely know the script |
| premium | tokens(text) ÷ tokens(reference) | The cost multiplier your language pays for the same content |

The UNK rate alone can be misleading. On the same Sinhala paragraph:

| Tokenizer | Fertility | UNK rate | Byte rate |
|---|---:|---:|---:|
| tiktoken:gpt-4o | 3.55 | 0.00 | 0.00 |
| tiktoken:cl100k_base | 11.75 | 0.00 | **0.99** |
| gpt2-medium | 17.00 | 0.00 | **0.94** |
| bert-base-multilingual-cased | 1.10 | **0.91** | 0.00 |

`cl100k_base` reports no unknown tokens, yet 99% of its tokens are fragments
of Sinhala letters: it has almost no Sinhala vocabulary. mBERT looks
efficient (fertility 1.10) only because it replaces nearly every word with a
single `[UNK]`.

## Why this exists

Built while studying how tokenizers treat **Sinhala** (~20M speakers). The
headline findings from [the full study](study/README.md): Sinhala pays a
**2.26× token premium** vs English for the same content on GPT-4o (7.37× on
the original GPT-4 tokenizer); a 32k-vocab SentencePiece BPE trained on OSCAR
Sinhala by one student beats 200k–250k multilingual vocabularies by 1.3–2×;
and mBERT silently replaces **62%** of Sinhala tokens with `[UNK]`. If you
work on a low-resource language, measuring this is step one of fixing it.

## License

MIT — see [LICENSE](LICENSE).

## Author

**Ravindu Pabasara Karunarathna** — also the author of
[sinhaladate](https://pypi.org/project/sinhaladate/) and
[slnic](https://pypi.org/project/slnic/).
