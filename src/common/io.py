from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def read_json_records(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSON array, JSON object, or JSONL file as a list of records."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if file_path.suffix.lower() == ".jsonl":
        records: list[dict[str, Any]] = []
        with file_path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{file_path}:{line_no} is not a JSON object")
                records.append(value)
        return records

    with file_path.open("r", encoding="utf-8") as f:
        value = json.load(f)

    if isinstance(value, list):
        if not all(isinstance(item, dict) for item in value):
            raise ValueError(f"{file_path} must contain objects only")
        return value
    if isinstance(value, dict):
        for key in ("data", "records", "questions", "items"):
            nested = value.get(key)
            if isinstance(nested, list):
                if not all(isinstance(item, dict) for item in nested):
                    raise ValueError(f"{file_path}.{key} must contain objects only")
                return nested
        return [value]
    raise ValueError(f"{file_path} must be a JSON object, JSON array, or JSONL")


def write_json_records(records: Iterable[dict[str, Any]], path: str | Path, *, jsonl: bool = False) -> None:
    """Write records as JSON array or JSONL."""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(records)

    if jsonl:
        with file_path.open("w", encoding="utf-8", newline="\n") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False, sort_keys=False))
                f.write("\n")
        return

    with file_path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
        f.write("\n")


def read_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def write_text(path: str | Path, text: str) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(text, encoding="utf-8", newline="\n")

