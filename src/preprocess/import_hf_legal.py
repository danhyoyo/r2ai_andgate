from __future__ import annotations

import argparse
import json
import re
import time
import urllib.parse
import urllib.request
from html import unescape
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError

from src.common.io import read_json_records, write_json_records
from src.common.schema import ensure_article_id
from src.preprocess.normalize_law import build_doc_title
from src.preprocess.parse_articles import parse_law_text


HF_ROWS_ENDPOINT = "https://datasets-server.huggingface.co/rows"
DEFAULT_DATASET = "th1nhng0/vietnamese-legal-documents"
DEFAULT_HF_URL = "https://huggingface.co/datasets/th1nhng0/vietnamese-legal-documents"

METADATA_TEXT_FIELDS = (
    "title",
    "so_ky_hieu",
    "loai_van_ban",
    "nganh",
    "linh_vuc",
    "co_quan_ban_hanh",
    "pham_vi",
    "thong_tin_ap_dung",
    "tinh_trang_hieu_luc",
)


def _mojibake_score(text: str) -> int:
    markers = ("Ã", "Ä", "Â", "Æ", "â€", "á»", "áº")
    return sum(text.count(marker) for marker in markers)


def fix_mojibake(value: Any) -> Any:
    """Repair UTF-8 text that was accidentally decoded as latin-1."""
    if isinstance(value, dict):
        return {key: fix_mojibake(item) for key, item in value.items()}
    if isinstance(value, list):
        return [fix_mojibake(item) for item in value]
    if not isinstance(value, str) or _mojibake_score(value) == 0:
        return value
    try:
        repaired = value.encode("latin1").decode("utf-8")
    except UnicodeError:
        return value
    return repaired if _mojibake_score(repaired) < _mojibake_score(value) else value


def fetch_rows_from_api(
    *,
    dataset: str,
    config: str,
    split: str,
    offset: int,
    length: int,
    timeout: int = 90,
    retries: int = 5,
) -> list[dict[str, Any]]:
    params = urllib.parse.urlencode(
        {
            "dataset": dataset,
            "config": config,
            "split": split,
            "offset": offset,
            "length": length,
        }
    )
    request = urllib.request.Request(
        f"{HF_ROWS_ENDPOINT}?{params}",
        headers={"User-Agent": "r2ai-andgate-legal-importer/1.0"},
    )
    payload: dict[str, Any] = {}
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.load(response)
            break
        except HTTPError as exc:
            if exc.code not in {429, 502, 503, 504} or attempt >= retries:
                raise
            retry_after = exc.headers.get("Retry-After")
            delay = int(retry_after) if retry_after and retry_after.isdigit() else min(60, 5 * (attempt + 1))
            time.sleep(delay)
        except URLError:
            if attempt >= retries:
                raise
            time.sleep(min(30, 3 * (attempt + 1)))
    return [fix_mojibake(item.get("row", {})) for item in payload.get("rows", [])]


def fetch_rows_from_datasets(
    *,
    dataset: str,
    config: str,
    split: str,
    offset: int,
    length: int,
) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("The datasets backend requires: pip install datasets") from exc

    if length == 0:
        return []
    stream = load_dataset(dataset, config, split=split, streaming=True)
    if offset > 0:
        stream = stream.skip(offset)
    if length > 0:
        stream = stream.take(length)
    return [fix_mojibake(dict(row)) for row in stream]


def fetch_rows(
    *,
    dataset: str,
    config: str,
    split: str,
    offset: int,
    length: int,
    backend: str = "auto",
) -> list[dict[str, Any]]:
    errors: list[Exception] = []
    if backend in {"auto", "datasets"}:
        try:
            return fetch_rows_from_datasets(
                dataset=dataset,
                config=config,
                split=split,
                offset=offset,
                length=length,
            )
        except Exception as exc:
            if backend == "datasets":
                raise
            errors.append(exc)

    try:
        return fetch_rows_from_api(
            dataset=dataset,
            config=config,
            split=split,
            offset=offset,
            length=length,
        )
    except Exception as exc:
        if errors:
            raise RuntimeError(f"datasets backend failed: {errors[-1]}; rows API backend failed: {exc}") from exc
        raise

def html_to_text(value: str) -> str:
    text = fix_mojibake(value or "")
    text = re.sub(r"(?is)<(script|style)\b.*?</\1>", " ", str(text))
    text = re.sub(r"(?i)<\s*(br|/p|/div|/h[1-6]|/li|/tr|/table)\b[^>]*>", "\n", text)
    text = re.sub(r"(?i)<\s*(p|div|h[1-6]|li|tr|td|th)\b[^>]*>", "\n", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = unescape(text).replace("\xa0", " ")
    text = re.sub(r"\s+(?=Điều\s+0*[0-9]+[a-zA-Z]?\s*[.:])", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n+ *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_metadata_map(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    offset = 0
    while args.metadata_scan_limit <= 0 or offset < args.metadata_scan_limit:
        length = args.batch_size
        if args.metadata_scan_limit > 0:
            length = min(length, args.metadata_scan_limit - offset)
        if length <= 0:
            break
        rows = fetch_rows(
            dataset=args.dataset,
            config=args.metadata_config,
            split=args.split,
            offset=offset,
            length=length,
            backend=args.backend,
        )
        if not rows:
            break
        for row in rows:
            doc_id = str(row.get("id", "")).strip()
            if doc_id:
                metadata[doc_id] = row
        offset += len(rows)
        if len(rows) < length:
            break
    return metadata


def metadata_matches_keywords(metadata: dict[str, Any], keywords: list[str]) -> bool:
    if not keywords:
        return True
    haystack = " ".join(str(metadata.get(field, "")) for field in METADATA_TEXT_FIELDS).lower()
    return any(keyword.lower() in haystack for keyword in keywords)


def source_url_for(doc_id: str) -> str:
    if doc_id.isdigit():
        return f"https://vbpl.vn/pages/vbpq-toanvan.aspx?ItemID={doc_id}"
    return DEFAULT_HF_URL


def parse_document(row: dict[str, Any], metadata: dict[str, Any]) -> list[dict[str, Any]]:
    doc_id = str(row.get("id", "")).strip()
    html = str(row.get("content_html", "") or "")
    text = html_to_text(html)
    law_id = str(metadata.get("so_ky_hieu") or f"VBPL-{doc_id}").strip()
    doc_type = str(metadata.get("loai_van_ban") or "Văn bản").strip()
    raw_title = str(metadata.get("title") or "").strip()
    doc_title = build_doc_title(doc_type, law_id, raw_title)
    articles = parse_law_text(
        text,
        law_id=law_id,
        doc_title=doc_title,
        doc_type=doc_type,
        source_url=source_url_for(doc_id),
        effective_date=str(metadata.get("ngay_co_hieu_luc") or ""),
        status=str(metadata.get("tinh_trang_hieu_luc") or "chưa rõ"),
    )
    for article in articles:
        article["source_dataset"] = DEFAULT_DATASET
        article["source_document_id"] = doc_id
        article["issuing_agency"] = metadata.get("co_quan_ban_hanh", "")
        article["legal_domain"] = metadata.get("linh_vuc", "")
        ensure_article_id(article)
    return articles


def import_articles(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, int]]:
    metadata_by_id = load_metadata_map(args)
    articles: list[dict[str, Any]] = []
    stats = {
        "metadata_rows": len(metadata_by_id),
        "documents_seen": 0,
        "documents_with_metadata": 0,
        "documents_kept": 0,
        "documents_failed": 0,
    }

    offset = args.offset
    remaining = args.max_docs
    while remaining != 0:
        length = args.batch_size if remaining < 0 else min(args.batch_size, remaining)
        rows = fetch_rows(
            dataset=args.dataset,
            config=args.content_config,
            split=args.split,
            offset=offset,
            length=length,
            backend=args.backend,
        )
        if not rows:
            break
        for row in rows:
            stats["documents_seen"] += 1
            doc_id = str(row.get("id", "")).strip()
            metadata = metadata_by_id.get(doc_id, {})
            if metadata:
                stats["documents_with_metadata"] += 1
            if args.require_metadata and not metadata:
                continue
            if not metadata_matches_keywords(metadata, args.keywords):
                continue
            try:
                parsed = parse_document(row, metadata)
            except Exception as exc:
                stats["documents_failed"] += 1
                if args.strict:
                    raise RuntimeError(f"Failed to parse document id={doc_id}: {exc}") from exc
                continue
            if parsed:
                stats["documents_kept"] += 1
                articles.extend(parsed)
        offset += len(rows)
        if remaining > 0:
            remaining -= len(rows)
        if len(rows) < length:
            break

    return articles, stats


def merge_existing(output: Path, new_articles: list[dict[str, Any]], append: bool) -> list[dict[str, Any]]:
    if not append or not output.exists():
        return new_articles
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for article in read_json_records(output):
        article_id = ensure_article_id(article)
        if article_id and article_id not in seen:
            seen.add(article_id)
            merged.append(article)
    for article in new_articles:
        article_id = ensure_article_id(article)
        if article_id and article_id not in seen:
            seen.add(article_id)
            merged.append(article)
    return merged


def main() -> int:
    parser = argparse.ArgumentParser(description="Import official Vietnamese legal documents from Hugging Face.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--content-config", default="content")
    parser.add_argument("--metadata-config", default="metadata")
    parser.add_argument("--backend", choices=("auto", "datasets", "rows"), default="auto")
    parser.add_argument("--split", default="data")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-docs", type=int, default=1000, help="Number of content rows to scan; use -1 for all rows.")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--metadata-scan-limit", type=int, default=10000, help="Metadata rows to cache; use -1 for all rows.")
    parser.add_argument("--keywords", nargs="*", default=[], help="Optional metadata keywords to keep.")
    parser.add_argument("--require-metadata", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--output", default="data/processed/articles.jsonl")
    args = parser.parse_args()

    output = Path(args.output)
    articles, stats = import_articles(args)
    articles = merge_existing(output, articles, args.append)
    write_json_records(articles, output, jsonl=True)

    print(
        "Imported "
        f"{len(articles)} article rows to {output} "
        f"(seen={stats['documents_seen']}, kept={stats['documents_kept']}, "
        f"metadata={stats['documents_with_metadata']}/{stats['metadata_rows']}, "
        f"failed={stats['documents_failed']})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
