from __future__ import annotations

import math
from typing import Any

from src.common.schema import ensure_full_text
from src.common.text import tokenize
from src.retrieval.query_processing import build_query_profile


def _safe_text(article: dict[str, Any]) -> str:
    return " ".join(
        str(article.get(key, ""))
        for key in ("doc_title", "article_title", "content", "full_text_for_embedding")
    )


class RuleBasedReranker:
    def rerank(
        self,
        question: str,
        hits: list[dict[str, Any]],
        articles_by_id: dict[str, dict[str, Any]],
        *,
        top_k: int = 20,
    ) -> list[dict[str, Any]]:
        profile = build_query_profile(question)
        query_tokens = set(tokenize(profile.normalized))
        reranked: list[dict[str, Any]] = []
        for hit in hits:
            article = articles_by_id.get(hit["article_id"])
            if not article:
                continue
            text = _safe_text(article).lower()
            article_tokens = set(tokenize(ensure_full_text(article)))
            overlap = len(query_tokens & article_tokens) / max(1.0, math.sqrt(len(query_tokens) * max(1, len(article_tokens))))
            boost = 0.0
            if "doanh_nghiep" in profile.domains and "doanh nghiệp" in text:
                boost += 0.08
            if "thue" in profile.domains and "thuế" in text:
                boost += 0.08
            if "lao_dong" in profile.domains and "lao động" in text:
                boost += 0.08
            if "bao_hiem" in profile.domains and "bảo hiểm" in text:
                boost += 0.08
            if "xu_phat" in profile.domains and ("phạt" in text or "xử phạt" in text):
                boost += 0.08
            if "điều kiện" in profile.normalized.lower() and any(term in text for term in ("điều kiện", "tiêu chí", "đáp ứng")):
                boost += 0.08
            if any(concept in text for concept in profile.concepts):
                boost += 0.04
            updated = dict(hit)
            updated["score"] = float(hit.get("score", 0.0)) + overlap + boost
            updated["rule_overlap"] = overlap
            updated["rule_boost"] = boost
            reranked.append(updated)
        reranked.sort(key=lambda item: item["score"], reverse=True)
        for rank, hit in enumerate(reranked, start=1):
            hit["rank"] = rank
        return reranked[:top_k]


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3") -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError("Cross-encoder reranking requires sentence-transformers.") from exc
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        question: str,
        hits: list[dict[str, Any]],
        articles_by_id: dict[str, dict[str, Any]],
        *,
        top_k: int = 20,
    ) -> list[dict[str, Any]]:
        pairs: list[tuple[str, str]] = []
        usable_hits: list[dict[str, Any]] = []
        for hit in hits:
            article = articles_by_id.get(hit["article_id"])
            if not article:
                continue
            pairs.append((question, ensure_full_text(article)))
            usable_hits.append(hit)
        scores = self.model.predict(pairs)
        reranked: list[dict[str, Any]] = []
        for hit, score in zip(usable_hits, scores):
            updated = dict(hit)
            updated["score"] = float(score)
            reranked.append(updated)
        reranked.sort(key=lambda item: item["score"], reverse=True)
        for rank, hit in enumerate(reranked, start=1):
            hit["rank"] = rank
        return reranked[:top_k]

