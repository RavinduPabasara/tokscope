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
word), **UNK rate**, **chars/bytes per token**, and the **token premium** your
language pays relative to a parallel English text.

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
Tokenizer                           Vocab     Tokens     Fertility  Chars/tok  UNK rate  Premium
------------------------------------------------------------------------------------------------
sp:sinhala_bpe.model                30000     71432      1.31       5.61       0.0       1.02
tiktoken:gpt-4o                     200019    132881     2.44       3.01       0.0       1.9
xlm-roberta-base                    250002    154332     2.83       2.59       0.0       2.21
google/mt5-small                    250100    231449     4.25       1.73       0.0       3.31
```

Tokenizer specs: `hf:NAME` (default), `tiktoken:MODEL`, `sp:PATH`, and `ws:`
(whitespace baseline — fertility exactly 1.0 by construction).

## Python API

```python
import tokscope

report = tokscope.analyze(text, "tiktoken:gpt-4o", reference_text=english_text)
report.fertility        # tokens per word
report.unk_rate         # fraction of tokens the tokenizer can't represent
report.premium          # token cost multiplier vs the reference text

reports = tokscope.compare(text, ["xlm-roberta-base", "tiktoken:gpt-4o", "ws:"])
```

Custom tokenizers: pass any object with a `name`, a `vocab_size`, and a
`tokenize(text) -> (tokens, unk_count)` method.

## Metrics

| Metric | Definition | Why it matters |
|---|---|---|
| fertility | tokens ÷ whitespace words | The standard measure of tokenizer fit; higher = more fragmentation |
| chars_per_token | characters ÷ tokens | How much text each token carries |
| bytes_per_token | UTF-8 bytes ÷ tokens | Script-independent capacity measure |
| unk_rate | UNK tokens ÷ tokens | Any nonzero value = the tokenizer cannot represent the text |
| premium | tokens(text) ÷ tokens(reference) | The cost multiplier your language pays for the same content |

## Why this exists

Built while studying how tokenizers treat **Sinhala** (~20M speakers): the
numbers above are real — a 30k-vocab SentencePiece BPE trained on a Sinhala
corpus by one student beats 250k-vocab multilingual tokenizers by 2–3× on
Sinhala text. If you work on a low-resource language, measuring this is step
one of fixing it.

## License

MIT — see [LICENSE](LICENSE).

## Author

**Ravindu Pabasara Karunarathna** — also the author of
[sinhaladate](https://pypi.org/project/sinhaladate/) and
[slnic](https://pypi.org/project/slnic/).
