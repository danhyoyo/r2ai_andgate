from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any


RECOMMENDED_STACK = {
    "generator": "Qwen/Qwen2.5-7B-Instruct",
    "embedding": "BAAI/bge-m3",
    "reranker": "BAAI/bge-reranker-v2-m3",
}

OPTIONAL_PACKAGES = (
    "numpy",
    "torch",
    "transformers",
    "sentence_transformers",
    "huggingface_hub",
    "datasets",
    "pyarrow",
    "faiss",
)


def package_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def load_sources(path: str | Path = "configs/sources.json") -> dict[str, Any]:
    source_path = Path(path)
    if not source_path.exists():
        return {}
    return json.loads(source_path.read_text(encoding="utf-8"))


def main() -> int:
    packages = {name: package_available(name) for name in OPTIONAL_PACKAGES}
    sources = load_sources()
    final_models = [
        item
        for item in sources.get("model_candidates", [])
        if item.get("use_for_final") is True
    ]

    print("Recommended contest-safe stack:")
    for role, model in RECOMMENDED_STACK.items():
        print(f"- {role}: {model}")

    print("\nOptional package availability:")
    for name, available in packages.items():
        print(f"- {name}: {'OK' if available else 'missing'}")

    if final_models:
        print("\nModels marked for final use in configs/sources.json:")
        for item in final_models:
            print(f"- {item.get('role')}: {item.get('name')} ({item.get('url')})")

    missing_for_dense = [name for name in ("numpy", "sentence_transformers") if not packages[name]]
    missing_for_llm = [name for name in ("torch", "transformers") if not packages[name]]
    if missing_for_dense or missing_for_llm:
        print("\nInstall optional dependencies before running the full pretrained stack:")
        print("python -m pip install -r requirements-optional.txt")
    if missing_for_llm:
        print("\nLocal Qwen generation is not runnable in this environment until torch/transformers are installed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
