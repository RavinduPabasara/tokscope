"""tokscope command-line interface."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import List

from .metrics import Report
from .runner import compare

_COLUMNS = [
    ("tokenizer", "Tokenizer", 34),
    ("vocab_size", "Vocab", 8),
    ("token_count", "Tokens", 9),
    ("fertility", "Fertility", 9),
    ("chars_per_token", "Chars/tok", 9),
    ("unk_rate", "UNK rate", 8),
    ("premium", "Premium", 8),
]


def _format_table(reports: List[Report]) -> str:
    rows = [r.as_dict() for r in reports]
    has_premium = any("premium" in r for r in rows)
    cols = [c for c in _COLUMNS if has_premium or c[0] != "premium"]
    header = "  ".join(f"{title:<{w}}" for _, title, w in cols)
    lines = [header, "-" * len(header)]
    for row in rows:
        cells = []
        for key, _, w in cols:
            value = row.get(key)
            cells.append(f"{'-' if value is None else value:<{w}}")
        lines.append("  ".join(cells))
    return "\n".join(lines)


def main(argv: List[str] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tokscope",
        description="Compare how efficiently tokenizers handle your text.",
    )
    parser.add_argument("file", help="text file to analyze ('-' for stdin)")
    parser.add_argument(
        "-t",
        "--tokenizer",
        action="append",
        required=True,
        dest="tokenizers",
        help="tokenizer spec (repeatable): hf:NAME, tiktoken:MODEL, "
        "sp:PATH, ws: — bare names default to hf:",
    )
    parser.add_argument(
        "--reference",
        help="parallel text file for token-premium computation "
        "(e.g. the English translation of the corpus)",
    )
    parser.add_argument("--json", dest="json_path", help="also write results to a JSON file")
    parser.add_argument("--csv", dest="csv_path", help="also write results to a CSV file")
    parser.add_argument(
        "--max-chars",
        type=int,
        default=2_000_000,
        help="read at most this many characters (default 2M)",
    )
    args = parser.parse_args(argv)

    if args.file == "-":
        text = sys.stdin.read()[: args.max_chars]
    else:
        text = Path(args.file).read_text(encoding="utf-8")[: args.max_chars]
    reference_text = None
    if args.reference:
        reference_text = Path(args.reference).read_text(encoding="utf-8")[: args.max_chars]

    reports = compare(text, args.tokenizers, reference_text)

    print(_format_table(reports))

    if args.json_path:
        Path(args.json_path).write_text(
            json.dumps([r.as_dict() for r in reports], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    if args.csv_path:
        rows = [r.as_dict() for r in reports]
        with open(args.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
