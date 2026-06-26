from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


def zip_submission(results_path: str | Path = "results/results.json", output_path: str | Path = "results/submission.zip") -> None:
    results = Path(results_path)
    if results.name != "results.json":
        raise ValueError("Submission file inside the zip must be named results.json")
    if not results.exists():
        raise FileNotFoundError(results)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(results, arcname="results.json")


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a flat submission.zip containing only results.json.")
    parser.add_argument("--results", default="results/results.json")
    parser.add_argument("--output", default="results/submission.zip")
    args = parser.parse_args()
    zip_submission(args.results, args.output)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

