from __future__ import annotations

import argparse
from typing import Any

from src.common.io import read_json_records
from src.generation.verifier import normalize_relevant_article


def _article_set(item: dict[str, Any]) -> set[str]:
    return {
        normalize_relevant_article(value)
        for value in item.get("relevant_articles", [])
        if len(str(value).split("|")) == 3
    }


def precision_recall_f2(predicted: set[str], gold: set[str]) -> tuple[float, float, float]:
    if not predicted and not gold:
        return 1.0, 1.0, 1.0
    if not predicted:
        return 0.0, 0.0, 0.0
    correct = len(predicted & gold)
    precision = correct / len(predicted)
    recall = correct / len(gold) if gold else 0.0
    denominator = 4 * precision + recall
    f2 = (5 * precision * recall / denominator) if denominator else 0.0
    return precision, recall, f2


def macro_scores(predictions: list[dict[str, Any]], gold: list[dict[str, Any]]) -> dict[str, float]:
    gold_by_id = {item.get("id"): item for item in gold}
    scores = []
    for prediction in predictions:
        gold_item = gold_by_id.get(prediction.get("id"))
        if gold_item is None:
            continue
        scores.append(precision_recall_f2(_article_set(prediction), _article_set(gold_item)))
    if not scores:
        return {"precision": 0.0, "recall": 0.0, "f2": 0.0, "count": 0}
    return {
        "precision": sum(score[0] for score in scores) / len(scores),
        "recall": sum(score[1] for score in scores) / len(scores),
        "f2": sum(score[2] for score in scores) / len(scores),
        "count": len(scores),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute macro Precision/Recall/F2 for a labeled dev set.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--gold", required=True)
    args = parser.parse_args()

    scores = macro_scores(read_json_records(args.predictions), read_json_records(args.gold))
    print(
        "count={count} precision={precision:.4f} recall={recall:.4f} f2={f2:.4f}".format(
            **scores
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

