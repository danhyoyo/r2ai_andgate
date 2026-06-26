from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

from src.common.io import read_text, write_json_records
from src.common.schema import full_text_for_embedding, normalize_article_number
from src.common.text import LEGAL_PHRASES
from src.preprocess.clean_text import clean_text
from src.preprocess.normalize_law import build_doc_title, infer_doc_type, infer_law_id, infer_raw_title

ARTICLE_HEADING_RE = re.compile(r"(?im)^\s*Điều\s+0*([0-9]+[a-zA-Z]?)\s*[.:]?\s*(.*?)\s*$")


def extract_keywords(text: str) -> list[str]:
    lower = text.lower()
    keywords = [phrase for phrase in LEGAL_PHRASES if phrase in lower]
    return keywords[:20]


def _guess_article_title(heading_tail: str, body: str) -> str:
    title = heading_tail.strip(" .:-")
    if title:
        return title
    for line in body.splitlines()[:5]:
        line = line.strip()
        if not line:
            continue
        if re.match(r"^[0-9]+[.)]\s", line):
            continue
        if len(line) <= 180:
            return line.strip(" .:-")
    return ""


def parse_law_text(
    text: str,
    *,
    law_id: str,
    doc_title: str,
    doc_type: str,
    source_url: str = "",
    effective_date: str = "",
    status: str = "chưa rõ",
) -> list[dict[str, Any]]:
    cleaned = clean_text(text)
    matches = list(ARTICLE_HEADING_RE.finditer(cleaned))
    if not matches:
        raise ValueError("No article headings matching 'Điều X' were found")

    articles: list[dict[str, Any]] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(cleaned)
        content = cleaned[start:end].strip()
        article_number = normalize_article_number(f"Điều {match.group(1)}")
        article_title = _guess_article_title(match.group(2), content)
        article = {
            "article_id": f"{law_id}|{article_number}",
            "law_id": law_id,
            "doc_type": doc_type,
            "doc_title": doc_title,
            "article_number": article_number,
            "article_title": article_title,
            "content": content,
            "keywords": extract_keywords(" ".join([doc_title, article_title, content])),
            "effective_date": effective_date,
            "status": status,
            "source_url": source_url,
        }
        article["full_text_for_embedding"] = full_text_for_embedding(article)
        articles.append(article)
    return articles


def parse_file(
    path: Path,
    *,
    law_id: str = "",
    doc_title: str = "",
    doc_type: str = "",
    source_url: str = "",
    effective_date: str = "",
    status: str = "chưa rõ",
) -> list[dict[str, Any]]:
    text = read_text(path)
    resolved_law_id = law_id or infer_law_id(text, path.name)
    resolved_doc_type = doc_type or infer_doc_type(text)
    resolved_doc_title = doc_title or build_doc_title(resolved_doc_type, resolved_law_id, infer_raw_title(text, path.name))
    return parse_law_text(
        text,
        law_id=resolved_law_id,
        doc_title=resolved_doc_title,
        doc_type=resolved_doc_type,
        source_url=source_url,
        effective_date=effective_date,
        status=status,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Parse raw Vietnamese legal text into article-level JSONL.")
    parser.add_argument("--input", required=True, help="Raw .txt file or directory of .txt files")
    parser.add_argument("--output", default="data/processed/articles.jsonl", help="Output articles JSONL")
    parser.add_argument("--law-id", default="", help="Override law id for a single input file")
    parser.add_argument("--doc-title", default="", help="Override document title for a single input file")
    parser.add_argument("--doc-type", default="", help="Override document type for a single input file")
    parser.add_argument("--source-url", default="", help="Source URL metadata")
    parser.add_argument("--effective-date", default="", help="Effective date metadata")
    parser.add_argument("--status", default="chưa rõ", help="Document status metadata")
    args = parser.parse_args()

    input_path = Path(args.input)
    files = sorted(input_path.glob("*.txt")) if input_path.is_dir() else [input_path]
    articles: list[dict[str, Any]] = []
    for file_path in files:
        articles.extend(
            parse_file(
                file_path,
                law_id=args.law_id,
                doc_title=args.doc_title,
                doc_type=args.doc_type,
                source_url=args.source_url,
                effective_date=args.effective_date,
                status=args.status,
            )
        )

    write_json_records(articles, args.output, jsonl=True)
    print(f"Wrote {len(articles)} articles to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

