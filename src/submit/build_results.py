from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.common.io import read_json_records, write_json_records
from src.common.schema import article_keys, docs_from_articles
from src.generation.generate_answer import generate_answer
from src.generation.verifier import repair_result_item
from src.retrieval.pipeline import RetrievalPipeline


def build_results(
    questions: list[dict[str, Any]],
    *,
    articles_path: str | Path = "data/processed/articles.jsonl",
    bm25_index_path: str | Path = "indexes/bm25/bm25.pkl",
    variant: str = "balanced",
    use_dense: bool = False,
    use_cross_encoder: bool = False,
    model_path: str = "",
) -> list[dict[str, Any]]:
    pipeline = RetrievalPipeline(
        articles_path=articles_path,
        bm25_index_path=bm25_index_path,
        use_dense=use_dense,
        use_cross_encoder=use_cross_encoder,
    )
    results: list[dict[str, Any]] = []
    for index, question_item in enumerate(questions, start=1):
        question = str(question_item.get("question", "")).strip()
        articles = pipeline.retrieve(question, variant=variant)
        answer = generate_answer(question, articles, model_path=model_path)
        item = {
            "id": question_item.get("id", index),
            "question": question,
            "answer": answer,
            "relevant_docs": docs_from_articles(articles),
            "relevant_articles": article_keys(articles),
        }
        results.append(repair_result_item(item))
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Build competition results.json from test questions.")
    parser.add_argument("--test", default="data/test/test.json")
    parser.add_argument("--articles", default="data/processed/articles.jsonl")
    parser.add_argument("--output", default="results/results.json")
    parser.add_argument("--bm25-index", default="indexes/bm25/bm25.pkl")
    parser.add_argument("--variant", choices=("recall", "balanced", "precision"), default="balanced")
    parser.add_argument("--use-dense", action="store_true")
    parser.add_argument("--use-cross-encoder", action="store_true")
    parser.add_argument("--model-path", default="", help="Optional local open-source LLM path")
    args = parser.parse_args()

    questions = read_json_records(args.test)
    results = build_results(
        questions,
        articles_path=args.articles,
        bm25_index_path=args.bm25_index,
        variant=args.variant,
        use_dense=args.use_dense,
        use_cross_encoder=args.use_cross_encoder,
        model_path=args.model_path,
    )
    write_json_records(results, args.output)
    print(f"Wrote {len(results)} predictions to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

