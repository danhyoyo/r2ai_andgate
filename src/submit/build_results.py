from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

from src.common.io import read_json_records, write_json_records
from src.common.schema import article_keys, docs_from_articles
from src.generation.generate_answer import generate_answer
from src.generation.verifier import repair_result_item
from src.retrieval.pipeline import RetrievalPipeline


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.1f}m"
    return f"{minutes / 60:.1f}h"


def _partial_output_path(output_path: str | Path) -> Path:
    path = Path(output_path)
    return path.with_name(f"{path.stem}.partial{path.suffix}")


def build_results(
    questions: list[dict[str, Any]],
    *,
    articles_path: str | Path = "data/processed/articles.jsonl",
    bm25_index_path: str | Path = "indexes/bm25/bm25.pkl",
    variant: str = "balanced",
    use_dense: bool = False,
    use_cross_encoder: bool = False,
    model_path: str = "",
    log_every: int = 0,
    checkpoint_every: int = 0,
    checkpoint_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    total = len(questions)
    started_at = time.time()
    print(
        "[build_results] Initializing pipeline "
        f"(questions={total}, variant={variant}, dense={use_dense}, "
        f"cross_encoder={use_cross_encoder}, llm={bool(model_path)})",
        flush=True,
    )
    pipeline = RetrievalPipeline(
        articles_path=articles_path,
        bm25_index_path=bm25_index_path,
        use_dense=use_dense,
        use_cross_encoder=use_cross_encoder,
    )
    print(f"[build_results] Pipeline ready in {_format_duration(time.time() - started_at)}", flush=True)

    results: list[dict[str, Any]] = []
    for index, question_item in enumerate(questions, start=1):
        item_id = question_item.get("id", index)
        if log_every > 0 and (index == 1 or (index - 1) % log_every == 0):
            print(f"[build_results] Start {index}/{total} id={item_id}", flush=True)

        item_started_at = time.time()
        question = str(question_item.get("question", "")).strip()
        articles: list[dict[str, Any]] = []
        try:
            articles = pipeline.retrieve(question, variant=variant)
        except Exception as exc:
            print(f"[build_results] WARNING: retrieval failed for id={item_id}: {exc}", flush=True)

        try:
            answer = generate_answer(question, articles, model_path=model_path)
        except Exception as exc:
            print(f"[build_results] WARNING: generation failed for id={item_id}: {exc}", flush=True)
            answer = generate_answer(question, articles, model_path="")

        item = {
            "id": item_id,
            "question": question,
            "answer": answer,
            "relevant_docs": docs_from_articles(articles),
            "relevant_articles": article_keys(articles),
        }
        try:
            results.append(repair_result_item(item))
        except Exception as exc:
            print(f"[build_results] WARNING: repair failed for id={item_id}: {exc}", flush=True)
            results.append(item)

        should_report = log_every > 0 and (index % log_every == 0 or index == total)
        if should_report:
            elapsed = time.time() - started_at
            avg = elapsed / max(1, index)
            remaining = avg * max(0, total - index)
            print(
                "[build_results] Done "
                f"{index}/{total} id={item_id} "
                f"last={_format_duration(time.time() - item_started_at)} "
                f"elapsed={_format_duration(elapsed)} "
                f"eta={_format_duration(remaining)} "
                f"articles={len(articles)}",
                flush=True,
            )

        if checkpoint_every > 0 and checkpoint_path and (index % checkpoint_every == 0 or index == total):
            write_json_records(results, checkpoint_path)
            print(f"[build_results] Checkpoint wrote {len(results)} records to {checkpoint_path}", flush=True)
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
    parser.add_argument("--log-every", type=int, default=10, help="Print progress every N questions; use 0 to disable.")
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=10,
        help="Write a partial results file every N questions; use 0 to disable.",
    )
    parser.add_argument(
        "--checkpoint-output",
        default="",
        help="Optional partial output path. Defaults to '<output-stem>.partial.json'.",
    )
    args = parser.parse_args()

    questions = read_json_records(args.test)
    checkpoint_path = None
    if args.checkpoint_every > 0:
        checkpoint_path = Path(args.checkpoint_output) if args.checkpoint_output else _partial_output_path(args.output)

    results = build_results(
        questions,
        articles_path=args.articles,
        bm25_index_path=args.bm25_index,
        variant=args.variant,
        use_dense=args.use_dense,
        use_cross_encoder=args.use_cross_encoder,
        model_path=args.model_path,
        log_every=args.log_every,
        checkpoint_every=args.checkpoint_every,
        checkpoint_path=checkpoint_path,
    )
    write_json_records(results, args.output)
    print(f"Wrote {len(results)} predictions to {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
