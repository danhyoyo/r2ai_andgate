from __future__ import annotations

import argparse
import re
from pathlib import Path

from src.common.io import read_text, write_text
from src.common.text import normalize_unicode


def clean_text(text: str) -> str:
    text = normalize_unicode(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\ufeff", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"(?m)^\s+", "", text)
    return text.strip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Clean a raw Vietnamese legal text file.")
    parser.add_argument("--input", required=True, help="Raw .txt file")
    parser.add_argument("--output", required=True, help="Cleaned .txt file")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    write_text(output_path, clean_text(read_text(input_path)))
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

