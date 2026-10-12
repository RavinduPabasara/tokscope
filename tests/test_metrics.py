"""Tests for metric math and the whitespace baseline."""

from tokscope import analyze, compare
from tokscope.metrics import build_report


class TestReportMath:
    def test_fertility(self):
        r = build_report("t", "two words", ["tw", "o", "wor", "ds"], 0, 100)
        assert r.word_count == 2
        assert r.token_count == 4
        assert r.fertility == 2.0

    def test_chars_and_bytes_per_token(self):
        # "අම" is 2 chars, 6 utf-8 bytes
        r = build_report("t", "අම", ["අ", "ම"], 0, None)
        assert r.chars_per_token == 1.0
        assert r.bytes_per_token == 3.0

    def test_unk_rate(self):
        r = build_report("t", "a b c d", ["a", "b", "<unk>", "<unk>"], 2, 10)
        assert r.unk_rate == 0.5

    def test_premium(self):
        r = build_report("t", "text", ["x"] * 30, 0, None, reference_token_count=10)
        assert r.premium == 3.0

    def test_empty_text_no_division_error(self):
        r = build_report("t", "", [], 0, None)
        assert r.fertility == 0.0
        assert r.chars_per_token == 0.0
        assert r.unk_rate == 0.0

    def test_as_dict_omits_premium_when_absent(self):
        r = build_report("t", "a", ["a"], 0, None)
        assert "premium" not in r.as_dict()


class TestWhitespaceBaseline:
    def test_fertility_is_exactly_one(self):
        r = analyze("මම සිංහල භාෂාව ඉගෙන ගනිමි.", "ws:")
        assert r.fertility == 1.0
        assert r.unk_rate == 0.0
        assert r.tokenizer == "whitespace"

    def test_premium_via_reference(self):
        # 6 Sinhala words vs 3 English words -> premium 2.0 under whitespace
        r = analyze("එක දෙක තුන හතර පහ හය", "ws:", reference_text="one two three")
        assert r.premium == 2.0


class TestCompare:
    def test_sorted_by_fertility(self):
        class Doubler:
            name = "doubler"
            vocab_size = None

            def tokenize(self, text):
                return [c for w in text.split() for c in (w[:1], w[1:])], 0

        reports = compare("some words here", ["ws:", Doubler()])
        assert [r.tokenizer for r in reports] == ["whitespace", "doubler"]
        assert reports[1].fertility == 2.0


class TestByteTokens:
    def test_rate(self):
        r = build_report(
            "t", "අ", ["<0xE0>", "<0xB6>", "<0x85>", "x"], 0, None, byte_token_count=3
        )
        assert r.byte_token_rate == 0.75
        assert r.as_dict()["byte_token_rate"] == 0.75

    def test_unknown_when_adapter_does_not_report(self):
        r = build_report("t", "a", ["a"], 0, None)
        assert r.byte_token_rate is None
        assert "byte_token_rate" not in r.as_dict()

    def test_whitespace_reports_zero(self):
        assert analyze("මම යමි", "ws:").byte_token_rate == 0.0

    def test_custom_adapters_with_and_without_byte_count(self):
        class TwoTuple:
            name, vocab_size = "two", None

            def tokenize(self, text):
                return text.split(), 0

        class WithBytes(TwoTuple):
            def tokenize_with_bytes(self, text):
                return list(text), 0, 2

        assert analyze("a b", TwoTuple()).byte_token_rate is None
        assert analyze("abcd", WithBytes()).byte_token_rate == 0.5
