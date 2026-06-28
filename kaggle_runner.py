from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def run_cmd(cmd: list[str | Path], *, label: str = "", required: bool = False) -> bool:
    title = label or " ".join(str(part) for part in cmd[:4])
    print("\n$", " ".join(str(part) for part in cmd), flush=True)
    try:
        result = subprocess.run([str(part) for part in cmd], check=False)
    except FileNotFoundError as exc:
        print(f"WARNING: {title} could not start: {exc}", flush=True)
        if required:
            raise
        return False
    if result.returncode != 0:
        print(f"WARNING: {title} exited with code {result.returncode}", flush=True)
        if required:
            raise RuntimeError(f"Required command failed: {title}")
        return False
    return True


def count_lines(path: str | Path) -> int:
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            return sum(1 for _ in handle)
    except FileNotFoundError:
        return 0


def install_dependencies() -> None:
    packages = [
        "numpy",
        "transformers",
        "sentence-transformers",
        "huggingface-hub",
        "datasets",
        "pyarrow",
        "accelerate",
        "safetensors",
    ]
    run_cmd([sys.executable, "-m", "pip", "install", "-q", "-U", *packages], label="install dependencies")


def print_environment() -> None:
    try:
        import torch

        print("torch:", torch.__version__, flush=True)
        print("cuda:", torch.cuda.is_available(), flush=True)
        if torch.cuda.is_available():
            print("gpu:", torch.cuda.get_device_name(0), flush=True)
            print("vram GB:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2), flush=True)
    except Exception as exc:
        print("WARNING: Torch/GPU check failed:", exc, flush=True)
    run_cmd([sys.executable, "-m", "src.models.check_environment"], label="environment check")


def find_uploaded_articles(input_root: Path) -> Path | None:
    if not input_root.exists():
        return None
    matches = sorted(input_root.glob("**/articles.jsonl"))
    return matches[0] if matches else None


def prepare_corpus(args: argparse.Namespace) -> bool:
    articles_path = Path(args.articles)
    articles_path.parent.mkdir(parents=True, exist_ok=True)

    uploaded = find_uploaded_articles(Path(args.kaggle_input)) if args.prefer_uploaded_articles else None
    if uploaded and not args.force_import:
        shutil.copy2(uploaded, articles_path)
        print(f"Copied uploaded corpus: {uploaded} -> {articles_path}", flush=True)
        return True

    if articles_path.exists() and not args.force_import:
        print(f"Using existing corpus: {articles_path.resolve()} ({count_lines(articles_path)} rows)", flush=True)
        return True

    temp_path = articles_path.with_name(f"{articles_path.stem}.importing{articles_path.suffix}")
    if temp_path.exists():
        temp_path.unlink()

    import_ok = True
    if args.import_docs == -1:
        cmd = [
            sys.executable,
            "-m",
            "src.preprocess.import_hf_legal",
            "--backend",
            args.import_backend,
            "--output",
            temp_path,
            "--max-docs",
            "-1",
            "--metadata-scan-limit",
            "-1",
            "--batch-size",
            str(args.import_batch_size),
        ]
        if args.require_metadata:
            cmd.append("--require-metadata")
        import_ok = run_cmd(cmd, label="full corpus import")
    else:
        for offset in range(0, max(0, args.import_docs), max(1, args.import_chunk_size)):
            chunk_size = min(args.import_chunk_size, args.import_docs - offset)
            cmd = [
                sys.executable,
                "-m",
                "src.preprocess.import_hf_legal",
                "--backend",
                args.import_backend,
                "--output",
                temp_path,
                "--offset",
                str(offset),
                "--max-docs",
                str(chunk_size),
                "--metadata-scan-limit",
                str(args.metadata_scan_limit),
                "--batch-size",
                str(args.import_batch_size),
                "--append",
            ]
            if args.require_metadata:
                cmd.append("--require-metadata")
            print(f"Import chunk offset={offset} size={chunk_size}", flush=True)
            if not run_cmd(cmd, label=f"corpus import chunk {offset}"):
                import_ok = False
                break

    if temp_path.exists() and count_lines(temp_path) > 0:
        shutil.move(str(temp_path), str(articles_path))
        print(f"Imported corpus: {articles_path} ({count_lines(articles_path)} rows)", flush=True)
        return True

    if articles_path.exists():
        print(f"WARNING: import failed; keeping existing corpus: {articles_path}", flush=True)
        return True

    print("WARNING: no corpus is available. Upload articles.jsonl or enable Internet.", flush=True)
    return False


def make_test_shard(args: argparse.Namespace) -> tuple[Path | None, Path]:
    test_path = Path(args.test)
    output_path = Path(args.output)
    if not test_path.exists():
        print(f"WARNING: missing test file: {test_path}", flush=True)
        return None, output_path

    num_shards = max(1, args.num_shards)
    shard_id = args.shard_id
    if shard_id < 0 or shard_id >= num_shards:
        print(f"WARNING: shard-id {shard_id} is outside 0..{num_shards - 1}; using 0", flush=True)
        shard_id = 0

    if num_shards == 1:
        return test_path, output_path

    records = json.loads(test_path.read_text(encoding="utf-8"))
    shard_records = [row for index, row in enumerate(records) if index % num_shards == shard_id]
    shard_test_path = test_path.with_name(f"test_shard_{shard_id}_of_{num_shards}.json")
    shard_output_path = output_path.with_name(f"results_shard_{shard_id}_of_{num_shards}.json")
    shard_test_path.write_text(json.dumps(shard_records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Prepared shard {shard_id}/{num_shards}: {len(shard_records)} rows", flush=True)
    return shard_test_path, shard_output_path


def maybe_download_llm(args: argparse.Namespace) -> list[str]:
    if not args.use_llm:
        return []
    model_dir = Path(args.model_dir)
    model_dir.parent.mkdir(parents=True, exist_ok=True)
    ok = run_cmd(
        ["huggingface-cli", "download", args.model_id, "--local-dir", model_dir],
        label="download local LLM",
    )
    if not ok:
        print("WARNING: LLM download failed; continuing without --model-path.", flush=True)
        return []
    return ["--model-path", str(model_dir)]


def build_results(args: argparse.Namespace, test_path: Path, output_path: Path, model_args: list[str]) -> bool:
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "src.submit.build_results",
        "--test",
        test_path,
        "--articles",
        args.articles,
        "--output",
        output_path,
        "--variant",
        args.variant,
        "--log-every",
        str(args.log_every),
        "--checkpoint-every",
        str(args.checkpoint_every),
    ]
    if args.use_dense:
        cmd.append("--use-dense")
    if args.use_cross_encoder:
        cmd.append("--use-cross-encoder")
    cmd.extend(model_args)

    started = time.time()
    ok = run_cmd(cmd, label="build results")
    partial_path = output_path.with_name(f"{output_path.stem}.partial{output_path.suffix}")
    if not output_path.exists() and partial_path.exists():
        shutil.copy2(partial_path, output_path)
        print(f"Copied partial output to final output: {partial_path} -> {output_path}", flush=True)
    print(f"Inference finished in {round((time.time() - started) / 60, 2)} minutes", flush=True)
    return ok and output_path.exists()


def repair_validate_zip(args: argparse.Namespace, output_path: Path) -> None:
    if not output_path.exists():
        print(f"WARNING: output not found, skipping validate/zip: {output_path}", flush=True)
        return

    run_cmd(
        [
            sys.executable,
            "-m",
            "src.submit.validate_results",
            "--input",
            output_path,
            "--output",
            output_path,
            "--repair",
        ],
        label="repair results",
    )
    validate_cmd = [sys.executable, "-m", "src.submit.validate_results", "--input", output_path]
    if args.strict_validate:
        validate_cmd.append("--strict")
    run_cmd(validate_cmd, label="validate results")

    if args.num_shards == 1:
        zip_input = output_path
        if output_path.name != "results.json":
            zip_input = Path("results/results.json")
            zip_input.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(output_path, zip_input)
        run_cmd(
            [sys.executable, "-m", "src.submit.zip_submission", "--results", zip_input, "--output", args.submission],
            label="zip submission",
        )


def merge_shards(args: argparse.Namespace) -> None:
    results_dir = Path(args.output).parent
    shard_paths = sorted(results_dir.glob("results_shard_*_of_*.json"))
    if not shard_paths:
        print(f"No shard files found in {results_dir}", flush=True)
        return

    merged = []
    seen = set()
    for path in shard_paths:
        try:
            rows = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"WARNING: skipping unreadable shard {path}: {exc}", flush=True)
            continue
        for row in rows:
            row_id = row.get("id")
            if row_id not in seen:
                seen.add(row_id)
                merged.append(row)

    merged.sort(key=lambda item: str(item.get("id", "")))
    final_path = Path("results/results.json")
    final_path.parent.mkdir(parents=True, exist_ok=True)
    final_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Merged {len(merged)} records -> {final_path}", flush=True)
    original_num_shards = args.num_shards
    args.num_shards = 1
    repair_validate_zip(args, final_path)
    args.num_shards = original_num_shards


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the R2AI Kaggle pipeline from A to Z.")
    parser.add_argument("--install-deps", action="store_true", help="Install Kaggle runtime dependencies first.")
    parser.add_argument("--skip-env-check", action="store_true")
    parser.add_argument("--kaggle-input", default="/kaggle/input")

    parser.add_argument("--articles", default="data/processed/articles.jsonl")
    parser.add_argument("--test", default="data/test/test.json")
    parser.add_argument("--output", default="results/results.json")
    parser.add_argument("--submission", default="results/submission.zip")

    parser.add_argument("--prefer-uploaded-articles", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force-import", action="store_true")
    parser.add_argument("--import-docs", type=int, default=5000, help="Use -1 for full import.")
    parser.add_argument("--import-chunk-size", type=int, default=1000)
    parser.add_argument("--metadata-scan-limit", type=int, default=50000)
    parser.add_argument("--import-batch-size", type=int, default=100)
    parser.add_argument("--import-backend", choices=("auto", "parquet", "datasets", "rows"), default="parquet")
    parser.add_argument("--require-metadata", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--variant", choices=("recall", "balanced", "precision"), default="balanced")
    parser.add_argument("--use-dense", action="store_true")
    parser.add_argument("--use-cross-encoder", action="store_true")
    parser.add_argument("--use-llm", action="store_true")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--model-dir", default="/kaggle/working/models/Qwen2.5-7B-Instruct")

    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--merge-shards", action="store_true", help="Merge existing results_shard_* files and zip.")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--strict-validate", action="store_true")

    parser.add_argument("--rerank-device", default="auto")
    parser.add_argument("--rerank-batch-size", default="2")
    parser.add_argument("--min-rerank-free-gb", default="6")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ["R2AI_RERANK_DEVICE"] = args.rerank_device
    os.environ["R2AI_RERANK_BATCH_SIZE"] = str(args.rerank_batch_size)
    os.environ["R2AI_MIN_RERANK_FREE_GB"] = str(args.min_rerank_free_gb)

    print("R2AI Kaggle runner args:", vars(args), flush=True)

    if args.install_deps:
        install_dependencies()
    if not args.skip_env_check:
        print_environment()

    if args.merge_shards:
        merge_shards(args)
        return 0

    corpus_ready = prepare_corpus(args)
    test_path, output_path = make_test_shard(args)
    if not corpus_ready or test_path is None:
        print("Stopping gracefully because required input is missing.", flush=True)
        return 0

    run_cmd(
        [
            sys.executable,
            "-m",
            "src.retrieval.bm25_retriever",
            "build",
            "--articles",
            args.articles,
            "--output",
            "indexes/bm25/bm25.pkl",
        ],
        label="BM25 build",
    )

    model_args = maybe_download_llm(args)
    build_results(args, test_path, output_path, model_args)
    repair_validate_zip(args, output_path)

    print("Done. Key files:", flush=True)
    for path in [Path(args.output), Path(args.submission), output_path, Path(args.articles)]:
        if path.exists():
            print(f"- {path}: {round(path.stat().st_size / 1024 / 1024, 2)} MB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
