"""Tests for adapter spec parsing, the SentencePiece backend, and the CLI."""

import json

import pytest

from tokscope import load
from tokscope.adapters import WhitespaceAdapter
from tokscope.cli import main

SINHALA_BPE = (
    "/Users/ravindupabasarakarunarathna/Documents/Self_Studies/Self_Studies_archive"
    "/SinhalaCustomTokenizer/trained_models/sinhala_bpe.model"
)


class TestSpecParsing:
    def test_ws_prefix(self):
        assert isinstance(load("ws:"), WhitespaceAdapter)

    def test_unknown_prefix_falls_back_to_hf_name(self):
        # 'some/model:tag' has no registered backend prefix, treated as hf spec;
        # transformers may not be installed, so only check the error type is sane
        with pytest.raises(Exception):
            load("definitely-not-a-real-backend:nope")


class TestSentencePieceAdapter:
    def test_sinhala_bpe_model(self):
        sp = pytest.importorskip("sentencepiece")  # noqa: F841
        import os

        if not os.path.exists(SINHALA_BPE):
            pytest.skip("local sinhala_bpe.model not available")
        adapter = load(f"sp:{SINHALA_BPE}")
        assert adapter.vocab_size == 30000
        tokens, unk = adapter.tokenize("මම සිංහල භාෂාව ඉගෙන ගනිමි.")
        assert 0 < len(tokens) <= 10
        assert unk == 0


class TestByteFallback:
    def test_sentencepiece_byte_fallback_counted(self, tmp_path):
        spm = pytest.importorskip("sentencepiece")
        corpus = tmp_path / "corpus.txt"
        corpus.write_text("the cat sat on the mat\n" * 200, encoding="utf-8")
        spm.SentencePieceTrainer.train(
            input=str(corpus),
            model_prefix=str(tmp_path / "m"),
            vocab_size=300,
            byte_fallback=True,
            hard_vocab_limit=False,
            character_coverage=1.0,
            minloglevel=2,
        )
        adapter = load(f"sp:{tmp_path / 'm.model'}")
        # Sinhala never appeared in training: each letter becomes 3 byte pieces.
        tokens, unk, byte_count = adapter.tokenize_with_bytes("අම")
        assert adapter.tokenize("අම") == (tokens, unk)
        assert unk == 0
        assert byte_count == 6
        assert all(t.startswith("<0x") for t in tokens if t != "▁")

    def test_tiktoken_partial_characters_counted(self):
        pytest.importorskip("tiktoken")
        try:
            adapter = load("tiktoken:cl100k_base")
        except Exception:  # pragma: no cover - encoding download unavailable
            pytest.skip("cl100k_base encoding not available offline")
        _, _, byte_count = adapter.tokenize_with_bytes("cat සිංහල")
        assert byte_count > 0
        assert adapter.tokenize_with_bytes("cat")[2] == 0


class TestCli:
    def test_table_json_csv(self, tmp_path, capsys):
        corpus = tmp_path / "corpus.txt"
        corpus.write_text("හෙළෝ ලෝකය මෙය පරීක්ෂණයකි", encoding="utf-8")
        json_out = tmp_path / "out.json"
        csv_out = tmp_path / "out.csv"
        rc = main(
            [str(corpus), "-t", "ws:", "--json", str(json_out), "--csv", str(csv_out)]
        )
        assert rc == 0
        table = capsys.readouterr().out
        assert "whitespace" in table and "Fertility" in table
        assert "Byte rate" in table
        data = json.loads(json_out.read_text(encoding="utf-8"))
        assert data[0]["fertility"] == 1.0
        assert csv_out.read_text(encoding="utf-8").startswith("tokenizer,")

    def test_reference_premium_column(self, tmp_path, capsys):
        si = tmp_path / "si.txt"
        en = tmp_path / "en.txt"
        si.write_text("එක දෙක තුන හතර", encoding="utf-8")
        en.write_text("one two", encoding="utf-8")
        main([str(si), "-t", "ws:", "--reference", str(en)])
        assert "Premium" in capsys.readouterr().out
