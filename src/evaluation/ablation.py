from __future__ import annotations

import argparse
from pathlib import Path

from src.common.io import read_json_records, write_json_records
from src.evaluation.compute_f2 import macro_scores
from src.submit.build_results import build_results


def main() -> int:
    parser = argparse.ArgumentParser(description="Run recall/balanced/precision variants on a labeled dev set.")
    parser.add_argument("--dev", default="data/dev/dev_labeled.json")
    parser.add_argument("--articles", default="data/processed/articles.jsonl")
    parser.add_argument("--output-dir", default="results/ablation")
    args = parser.parse_args()

    dev_records = read_json_records(args.dev)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for variant in ("recall", "balanced", "precision"):
        predictions = build_results(dev_records, articles_path=args.articles, variant=variant)
        output_path = output_dir / f"{variant}.json"
        write_json_records(predictions, output_path)
        scores = macro_scores(predictions, dev_records)
        print(
            f"{variant}: count={scores['count']} precision={scores['precision']:.4f} "
            f"recall={scores['recall']:.4f} f2={scores['f2']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

