from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.common.io import read_json_records
from src.common.schema import ensure_article_id, ensure_full_text
from src.common.text import tokenize
from src.retrieval.bm25_retriever import BM25Index
from src.retrieval.dense_retriever import DenseIndex, DenseRetrieverUnavailable
from src.retrieval.query_processing import build_query_profile
from src.retrieval.reranker import CrossEncoderReranker, RuleBasedReranker
from src.retrieval.rrf_fusion import reciprocal_rank_fusion

SELECTION_POLICIES = {
    "recall": {"min_articles": 7, "max_articles": 10, "relative_threshold": 0.72},
    "balanced": {"min_articles": 4, "max_articles": 7, "relative_threshold": 0.85},
    "precision": {"min_articles": 2, "max_articles": 5, "relative_threshold": 0.93},
}


class RetrievalPipeline:
    def __init__(
        self,
        *,
        articles_path: str | Path = "data/processed/articles.jsonl",
        bm25_index_path: str | Path = "indexes/bm25/bm25.pkl",
        dense_index_path: str | Path = "indexes/faiss_bge_m3/dense.pkl",
        use_dense: bool = False,
        use_cross_encoder: bool = False,
    ) -> None:
        self.articles = read_json_records(articles_path)
        for article in self.articles:
            ensure_article_id(article)
            ensure_full_text(article)
        self.articles_by_id = {article["article_id"]: article for article in self.articles}

        bm25_path = Path(bm25_index_path)
        if bm25_path.exists():
            try:
                self.bm25 = BM25Index.load(bm25_path)
            except Exception as exc:
                print(f"Rebuilding BM25 index because existing index is incompatible: {exc}")
                self.bm25 = BM25Index.from_articles(self.articles)
                self.bm25.save(bm25_path)
        else:
            self.bm25 = BM25Index.from_articles(self.articles)
            self.bm25.save(bm25_path)

        self.dense: DenseIndex | None = None
        if use_dense:
            try:
                dense_path = Path(dense_index_path)
                if dense_path.exists():
                    self.dense = DenseIndex.load(dense_path)
                else:
                    self.dense = DenseIndex()
                    self.dense.build(self.articles)
                    self.dense.save(dense_path)
            except DenseRetrieverUnavailable as exc:
                print(f"Dense retrieval disabled: {exc}")
                self.dense = None

        if use_cross_encoder:
            try:
                self.reranker = CrossEncoderReranker()
            except RuntimeError as exc:
                print(f"Cross-encoder reranker disabled: {exc}")
                self.reranker = RuleBasedReranker()
        else:
            self.reranker = RuleBasedReranker()

    def _metadata_search(self, question: str, *, top_k: int = 50) -> list[tuple[str, float]]:
        query_tokens = set(tokenize(question))
        scored: list[tuple[str, float]] = []
        for article in self.articles:
            metadata = " ".join(
                [
                    str(article.get("doc_title", "")),
                    str(article.get("article_title", "")),
                    " ".join(str(keyword) for keyword in article.get("keywords", []) if keyword),
                ]
            )
            tokens = set(tokenize(metadata))
            if not tokens:
                continue
            overlap = len(query_tokens & tokens)
            if overlap:
                scored.append((article["article_id"], overlap / max(1, len(query_tokens))))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]

    def retrieve_candidates(self, question: str, *, bm25_top_k: int = 150, dense_top_k: int = 150) -> list[dict[str, Any]]:
        profile = build_query_profile(question)
        ranked_lists: list[list[str]] = []
        weights: list[float] = []

        for variant in profile.variants:
            bm25_hits = self.bm25.search(variant, top_k=bm25_top_k)
            ranked_lists.append([article_id for article_id, _ in bm25_hits])
            weights.append(1.0)

        metadata_hits = self._metadata_search(profile.normalized, top_k=50)
        ranked_lists.append([article_id for article_id, _ in metadata_hits])
        weights.append(0.6)

        if self.dense is not None:
            dense_hits = self.dense.search(profile.normalized, top_k=dense_top_k)
            ranked_lists.append([article_id for article_id, _ in dense_hits])
            weights.append(1.0)

        fused = reciprocal_rank_fusion(ranked_lists, k=60, weights=weights)[:100]
        return [
            {
                "article_id": article_id,
                "score": score,
                "rank": rank,
                "source_scores": source_scores,
            }
            for rank, (article_id, score, source_scores) in enumerate(fused, start=1)
        ]

    def rerank(self, question: str, candidates: list[dict[str, Any]], *, top_k: int = 20) -> list[dict[str, Any]]:
        return self.reranker.rerank(question, candidates, self.articles_by_id, top_k=top_k)

    def select_articles(self, hits: list[dict[str, Any]], *, variant: str = "balanced") -> list[dict[str, Any]]:
        policy = SELECTION_POLICIES.get(variant, SELECTION_POLICIES["balanced"])
        if not hits:
            return []
        min_articles = policy["min_articles"]
        max_articles = policy["max_articles"]
        threshold = policy["relative_threshold"]
        top_score = max(float(hits[0].get("score", 0.0)), 1e-9)

        selected_ids: list[str] = []
        for hit in hits:
            article_id = hit["article_id"]
            if len(selected_ids) < min_articles or float(hit["score"]) >= threshold * top_score:
                if article_id not in selected_ids:
                    selected_ids.append(article_id)
            if len(selected_ids) >= max_articles:
                break
        return [self.articles_by_id[article_id] for article_id in selected_ids if article_id in self.articles_by_id]

    def retrieve(self, question: str, *, variant: str = "balanced") -> list[dict[str, Any]]:
        candidates = self.retrieve_candidates(question)
        reranked = self.rerank(question, candidates, top_k=20)
        return self.select_articles(reranked, variant=variant)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the hybrid retrieval pipeline for one question.")
    parser.add_argument("--articles", default="data/processed/articles.jsonl")
    parser.add_argument("--question", required=True)
    parser.add_argument("--variant", choices=sorted(SELECTION_POLICIES), default="balanced")
    parser.add_argument("--use-dense", action="store_true")
    parser.add_argument("--use-cross-encoder", action="store_true")
    args = parser.parse_args()

    pipeline = RetrievalPipeline(
        articles_path=args.articles,
        use_dense=args.use_dense,
        use_cross_encoder=args.use_cross_encoder,
    )
    for article in pipeline.retrieve(args.question, variant=args.variant):
        print(f"{article.get('law_id')}|{article.get('doc_title')}|{article.get('article_number')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

