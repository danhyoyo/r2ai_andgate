from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

ARTICLE_NUMBER_RE = re.compile(r"Điều\s*0*([0-9]+[a-zA-Z]?)", re.IGNORECASE)


def normalize_article_number(value: Any) -> str:
    """Normalize variants such as 'Điều 05' to 'Điều 5'."""
    text = str(value or "").strip()
    match = ARTICLE_NUMBER_RE.search(text)
    if not match:
        return text
    raw = match.group(1)
    match_number = re.match(r"0*([0-9]+)([a-zA-Z]?)", raw)
    if not match_number:
        return f"Điều {raw}"
    number = str(int(match_number.group(1)))
    suffix = match_number.group(2)
    return f"Điều {number}{suffix}"


def make_doc_key(article: dict[str, Any]) -> str:
    law_id = str(article.get("law_id", "")).strip()
    doc_title = str(article.get("doc_title", "")).strip()
    return f"{law_id}|{doc_title}"


def make_article_key(article: dict[str, Any]) -> str:
    return f"{make_doc_key(article)}|{normalize_article_number(article.get('article_number', ''))}"


def ensure_article_id(article: dict[str, Any]) -> str:
    article_id = str(article.get("article_id", "")).strip()
    if article_id:
        return article_id
    law_id = str(article.get("law_id", "")).strip()
    number = normalize_article_number(article.get("article_number", ""))
    article_id = f"{law_id}|{number}".strip("|")
    article["article_id"] = article_id
    return article_id


def full_text_for_embedding(article: dict[str, Any]) -> str:
    keywords = article.get("keywords") or []
    if isinstance(keywords, list):
        keyword_text = ", ".join(str(keyword) for keyword in keywords)
    else:
        keyword_text = str(keywords)
    return "\n".join(
        part
        for part in (
            f"Tên văn bản: {article.get('doc_title', '')}",
            f"{normalize_article_number(article.get('article_number', ''))}. {article.get('article_title', '')}".strip(),
            f"Nội dung: {article.get('content', '')}",
            f"Từ khóa: {keyword_text}" if keyword_text else "",
        )
        if part.strip()
    )


def ensure_full_text(article: dict[str, Any]) -> str:
    text = str(article.get("full_text_for_embedding", "")).strip()
    if not text:
        text = full_text_for_embedding(article)
        article["full_text_for_embedding"] = text
    return text


def docs_from_articles(articles: list[dict[str, Any]]) -> list[str]:
    docs: list[str] = []
    seen: set[str] = set()
    for article in articles:
        key = make_doc_key(article)
        if key and key not in seen:
            seen.add(key)
            docs.append(key)
    return docs


def article_keys(articles: list[dict[str, Any]]) -> list[str]:
    keys: list[str] = []
    seen: set[str] = set()
    for article in articles:
        key = make_article_key(article)
        if key and key not in seen:
            seen.add(key)
            keys.append(key)
    return keys


@dataclass
class RetrievalHit:
    article_id: str
    score: float
    rank: int = 0
    source_scores: dict[str, float] = field(default_factory=dict)
    article: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "article_id": self.article_id,
            "score": self.score,
            "rank": self.rank,
            "source_scores": dict(self.source_scores),
        }

