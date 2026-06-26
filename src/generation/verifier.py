from __future__ import annotations

import argparse
import re
from typing import Any

from src.common.io import read_json_records, write_json_records
from src.common.schema import docs_from_articles, normalize_article_number

ANSWER_ARTICLE_RE = re.compile(r"Điều\s*0*([0-9]+[a-zA-Z]?)", re.IGNORECASE)


def extract_article_numbers(text: str) -> list[str]:
    numbers: list[str] = []
    for match in ANSWER_ARTICLE_RE.finditer(text or ""):
        number = normalize_article_number(f"Điều {match.group(1)}")
        if number not in numbers:
            numbers.append(number)
    return numbers


def normalize_relevant_article(value: str) -> str:
    parts = [part.strip() for part in str(value).split("|")]
    if len(parts) != 3:
        return str(value).strip()
    return f"{parts[0]}|{parts[1]}|{normalize_article_number(parts[2])}"


def normalize_relevant_doc(value: str) -> str:
    parts = [part.strip() for part in str(value).split("|")]
    if len(parts) != 2:
        return str(value).strip()
    return f"{parts[0]}|{parts[1]}"


def _dedupe(values: list[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            seen.add(value)
            output.append(value)
    return output


def repair_result_item(item: dict[str, Any]) -> dict[str, Any]:
    repaired = dict(item)
    relevant_articles = _dedupe([normalize_relevant_article(value) for value in repaired.get("relevant_articles", [])])
    repaired["relevant_articles"] = relevant_articles

    docs_from_refs = []
    for article_ref in relevant_articles:
        parts = article_ref.split("|")
        if len(parts) == 3:
            docs_from_refs.append(f"{parts[0]}|{parts[1]}")
    explicit_docs = [normalize_relevant_doc(value) for value in repaired.get("relevant_docs", [])]
    repaired["relevant_docs"] = _dedupe(explicit_docs + docs_from_refs)

    relevant_numbers = _dedupe([normalize_article_number(ref.split("|")[-1]) for ref in relevant_articles if "|" in ref])
    answer = str(repaired.get("answer", "")).strip()
    answer_numbers = set(extract_article_numbers(answer))
    missing = [number for number in relevant_numbers if number not in answer_numbers]
    if missing:
        basis = ", ".join(relevant_numbers[:-1]) + (" và " + relevant_numbers[-1] if len(relevant_numbers) > 1 else relevant_numbers[0])
        prefix = f"Căn cứ {basis}, "
        if answer:
            repaired["answer"] = prefix + answer[0].lower() + answer[1:]
        else:
            repaired["answer"] = prefix + "dựa trên dữ liệu được cung cấp, chưa đủ căn cứ để kết luận đầy đủ."
    return repaired


def validate_result_item(item: dict[str, Any], *, index: int = 0) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    prefix = f"item[{index}]"

    for field in ("id", "question", "answer", "relevant_docs", "relevant_articles"):
        if field not in item:
            errors.append(f"{prefix}: missing field '{field}'")

    if "relevant_docs" in item and not isinstance(item["relevant_docs"], list):
        errors.append(f"{prefix}: relevant_docs must be a list")
    if "relevant_articles" in item and not isinstance(item["relevant_articles"], list):
        errors.append(f"{prefix}: relevant_articles must be a list")

    for value in item.get("relevant_docs", []):
        if len(str(value).split("|")) != 2:
            errors.append(f"{prefix}: invalid relevant_docs format: {value}")
    for value in item.get("relevant_articles", []):
        if len(str(value).split("|")) != 3:
            errors.append(f"{prefix}: invalid relevant_articles format: {value}")

    answer_numbers = set(extract_article_numbers(str(item.get("answer", ""))))
    relevant_numbers = {
        normalize_article_number(str(value).split("|")[-1])
        for value in item.get("relevant_articles", [])
        if len(str(value).split("|")) == 3
    }
    extra = answer_numbers - relevant_numbers
    missing = relevant_numbers - answer_numbers
    if extra:
        warnings.append(f"{prefix}: answer cites articles not in relevant_articles: {sorted(extra)}")
    if missing:
        warnings.append(f"{prefix}: relevant_articles not cited in answer: {sorted(missing)}")
    return errors, warnings


def validate_results(records: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    for index, item in enumerate(records):
        item_errors, item_warnings = validate_result_item(item, index=index)
        errors.extend(item_errors)
        warnings.extend(item_warnings)
    return errors, warnings


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate and optionally repair results.json.")
    parser.add_argument("--input", default="results/results.json")
    parser.add_argument("--output", default="")
    parser.add_argument("--repair", action="store_true")
    parser.add_argument("--strict", action="store_true", help="Treat warnings as failures")
    args = parser.parse_args()

    records = read_json_records(args.input)
    if args.repair:
        records = [repair_result_item(item) for item in records]
        if args.output:
            write_json_records(records, args.output)

    errors, warnings = validate_results(records)
    for warning in warnings:
        print(f"WARNING: {warning}")
    for error in errors:
        print(f"ERROR: {error}")
    if errors or (args.strict and warnings):
        return 1
    print(f"Validated {len(records)} records")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

