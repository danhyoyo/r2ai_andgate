from __future__ import annotations

import re
from pathlib import Path

from src.common.schema import normalize_article_number
from src.common.text import normalize_unicode

LAW_ID_RE = re.compile(r"\b\d{1,3}/\d{4}/[A-ZĐĐa-zđ\-]+(?:-[A-ZĐa-zđ]+)?\b", re.UNICODE)
DOC_TYPE_PATTERNS = (
    ("Bộ luật", re.compile(r"\bBộ\s+luật\b", re.IGNORECASE)),
    ("Luật", re.compile(r"\bLuật\b", re.IGNORECASE)),
    ("Nghị định", re.compile(r"\bNghị\s+định\b", re.IGNORECASE)),
    ("Thông tư", re.compile(r"\bThông\s+tư\b", re.IGNORECASE)),
    ("Quyết định", re.compile(r"\bQuyết\s+định\b", re.IGNORECASE)),
    ("Nghị quyết", re.compile(r"\bNghị\s+quyết\b", re.IGNORECASE)),
)


def infer_law_id(text: str, filename: str = "") -> str:
    for source in (filename, text[:5000]):
        match = LAW_ID_RE.search(normalize_unicode(source))
        if match:
            return match.group(0).upper()
    return Path(filename).stem if filename else ""


def infer_doc_type(text: str, fallback: str = "Văn bản") -> str:
    head = normalize_unicode(text[:3000])
    for doc_type, pattern in DOC_TYPE_PATTERNS:
        if pattern.search(head):
            return doc_type
    return fallback


def infer_raw_title(text: str, filename: str = "") -> str:
    lines = [line.strip(" -:\t") for line in normalize_unicode(text).splitlines() if line.strip()]
    candidates: list[str] = []
    for line in lines[:80]:
        if len(line) < 8:
            continue
        if any(pattern.search(line) for _, pattern in DOC_TYPE_PATTERNS):
            candidates.append(line)
    if candidates:
        return re.sub(r"\s+", " ", candidates[-1]).strip()
    return Path(filename).stem.replace("_", " ").replace("-", " ").strip()


def build_doc_title(doc_type: str, law_id: str, raw_title: str) -> str:
    raw = re.sub(r"\s+", " ", raw_title or "").strip()
    if not raw:
        raw = doc_type
    raw_lower = raw.lower()
    if law_id and law_id.lower() in raw_lower and raw_lower.startswith(doc_type.lower()):
        return raw
    if raw_lower.startswith(doc_type.lower()):
        title_tail = raw[len(doc_type) :].strip(" -:")
    else:
        title_tail = raw
    parts = [doc_type]
    if law_id:
        parts.append(law_id)
    if title_tail and title_tail.lower() not in {doc_type.lower(), law_id.lower()}:
        parts.append(title_tail)
    return " ".join(parts)


__all__ = [
    "build_doc_title",
    "infer_doc_type",
    "infer_law_id",
    "infer_raw_title",
    "normalize_article_number",
]

