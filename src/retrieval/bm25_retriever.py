from __future__ import annotations

import argparse
import math
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.common.io import read_json_records
from src.common.schema import ensure_article_id, ensure_full_text
from src.common.text import tokenize


class BM25Index:
    def __init__(self, *, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.doc_ids: list[str] = []
        self.doc_lengths: list[int] = []
        self.avgdl: float = 0.0
        self.postings: dict[str, dict[int, int]] = {}
        self.idf: dict[str, float] = {}

    @classmethod
    def from_articles(cls, articles: list[dict[str, Any]], *, k1: float = 1.5, b: float = 0.75) -> "BM25Index":
        index = cls(k1=k1, b=b)
        posting_builder: dict[str, dict[int, int]] = defaultdict(dict)
        for doc_idx, article in enumerate(articles):
            article_id = ensure_article_id(article)
            tokens = tokenize(ensure_full_text(article))
            counts = Counter(tokens)
            index.doc_ids.append(article_id)
            index.doc_lengths.append(sum(counts.values()))
            for token, count in counts.items():
                posting_builder[token][doc_idx] = count
        index.postings = dict(posting_builder)
        index.avgdl = sum(index.doc_lengths) / max(1, len(index.doc_lengths))
        total_docs = max(1, len(index.doc_ids))
        index.idf = {
            token: math.log(1 + (total_docs - len(posting) + 0.5) / (len(posting) + 0.5))
            for token, posting in index.postings.items()
        }
        return index

    def search(self, query: str, *, top_k: int = 100) -> list[tuple[str, float]]:
        query_terms = set(tokenize(query))
        scores: dict[int, float] = defaultdict(float)
        for term in query_terms:
            posting = self.postings.get(term)
            if not posting:
                continue
            idf = self.idf.get(term, 0.0)
            for doc_idx, tf in posting.items():
                doc_len = self.doc_lengths[doc_idx]
                denom = tf + self.k1 * (1 - self.b + self.b * doc_len / max(self.avgdl, 1e-9))
                scores[doc_idx] += idf * (tf * (self.k1 + 1)) / max(denom, 1e-9)
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
        return [(self.doc_ids[doc_idx], score) for doc_idx, score in ranked]

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "r2ai_bm25_index",
            "version": 1,
            "k1": self.k1,
            "b": self.b,
            "doc_ids": self.doc_ids,
            "doc_lengths": self.doc_lengths,
            "avgdl": self.avgdl,
            "postings": self.postings,
            "idf": self.idf,
        }
        with output.open("wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path: str | Path) -> "BM25Index":
        with Path(path).open("rb") as f:
            value = pickle.load(f)
        if isinstance(value, BM25Index):
            return value
        if isinstance(value, dict) and value.get("format") == "r2ai_bm25_index":
            index = BM25Index(k1=float(value["k1"]), b=float(value["b"]))
            index.doc_ids = list(value["doc_ids"])
            index.doc_lengths = list(value["doc_lengths"])
            index.avgdl = float(value["avgdl"])
            index.postings = value["postings"]
            index.idf = value["idf"]
            return index
        raise TypeError(f"{path} is not a supported BM25 index")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build or query a BM25 article index.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--articles", default="data/processed/articles.jsonl")
    build_parser.add_argument("--output", default="indexes/bm25/bm25.pkl")

    search_parser = subparsers.add_parser("search")
    search_parser.add_argument("--index", default="indexes/bm25/bm25.pkl")
    search_parser.add_argument("--query", required=True)
    search_parser.add_argument("--top-k", type=int, default=10)

    args = parser.parse_args()
    if args.command == "build":
        articles = read_json_records(args.articles)
        index = BM25Index.from_articles(articles)
        index.save(args.output)
        print(f"Wrote BM25 index with {len(index.doc_ids)} articles to {args.output}")
        return 0

    index = BM25Index.load(args.index)
    for article_id, score in index.search(args.query, top_k=args.top_k):
        print(f"{score:.6f}\t{article_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

