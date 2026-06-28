from __future__ import annotations

import argparse
import math
from functools import lru_cache
from typing import Any

from src.common.io import read_json_records
from src.common.schema import normalize_article_number
from src.common.text import split_sentences, tokenize
from src.generation.answer_prompt import build_answer_prompt


def _join_vietnamese_list(values: list[str]) -> str:
    if not values:
        return ""
    if len(values) == 1:
        return values[0]
    return ", ".join(values[:-1]) + " và " + values[-1]


def _best_snippets(question: str, article: dict[str, Any], *, limit: int = 2) -> list[str]:
    query_tokens = set(tokenize(question))
    sentences = split_sentences(str(article.get("content", "")))
    scored: list[tuple[float, str]] = []
    for sentence in sentences:
        sentence_tokens = set(tokenize(sentence))
        if not sentence_tokens:
            continue
        overlap = len(query_tokens & sentence_tokens) / max(1.0, math.sqrt(len(query_tokens) * len(sentence_tokens)))
        if overlap > 0:
            scored.append((overlap, sentence))
    scored.sort(key=lambda item: item[0], reverse=True)
    snippets = [sentence for _, sentence in scored[:limit]]
    if not snippets and sentences:
        snippets = sentences[:limit]
    return snippets


def generate_extractive_answer(question: str, articles: list[dict[str, Any]]) -> str:
    if not articles:
        return (
            "- Căn cứ pháp luật: Dựa trên dữ liệu được cung cấp, chưa đủ căn cứ để kết luận đầy đủ.\n"
            "- Trả lời ngắn gọn: Chưa tìm thấy điều luật phù hợp trong corpus hiện tại.\n"
            "- Lưu ý thực tiễn: Cần bổ sung văn bản pháp luật chính thống trước khi sử dụng kết quả."
        )

    article_numbers = []
    for article in articles:
        number = normalize_article_number(article.get("article_number", ""))
        if number and number not in article_numbers:
            article_numbers.append(number)

    basis = []
    for article in articles:
        number = normalize_article_number(article.get("article_number", ""))
        law_id = article.get("law_id", "")
        title = article.get("doc_title", "")
        basis.append(f"{number} {title}" if title else f"{number} {law_id}".strip())

    points: list[str] = []
    for article in articles[:5]:
        number = normalize_article_number(article.get("article_number", ""))
        snippets = _best_snippets(question, article, limit=1)
        if snippets:
            points.append(f"{number}: {snippets[0]}")
        elif article.get("article_title"):
            points.append(f"{number}: {article['article_title']}")

    answer_body = " ".join(points).strip()
    if not answer_body:
        answer_body = "Các điều luật được truy hồi là căn cứ liên quan trực tiếp đến câu hỏi, nhưng nội dung chi tiết cần được kiểm tra thêm trong văn bản gốc."

    return (
        f"- Căn cứ pháp luật: Căn cứ {_join_vietnamese_list(article_numbers)} trong các văn bản đã truy hồi.\n"
        f"- Trả lời ngắn gọn: {answer_body}\n"
        "- Lưu ý thực tiễn: Đây là tư vấn sơ bộ dựa trên corpus được cung cấp; khi áp dụng thực tế nên kiểm tra hiệu lực văn bản và tình huống cụ thể."
    )



@lru_cache(maxsize=2)
def _load_local_llm(model_path: str) -> tuple[Any, Any]:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
    except ImportError as exc:
        raise RuntimeError("Local LLM generation requires transformers and torch.") from exc

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return tokenizer, model


def generate_answer(question: str, articles: list[dict[str, Any]], *, model_path: str = "") -> str:
    """Generate a grounded answer.

    The default is an extractive fallback so the pipeline can run without closed APIs.
    For final submissions, connect this function to a local open-source LLM such as
    Qwen2.5-7B-Instruct and pass build_answer_prompt(question, articles) to it.
    """
    if not model_path:
        return generate_extractive_answer(question, articles)

    prompt = build_answer_prompt(question, articles)
    try:
        tokenizer, model = _load_local_llm(model_path)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        output = model.generate(
            **inputs,
            max_new_tokens=768,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        text = tokenizer.decode(output[0], skip_special_tokens=True)
        return text[len(prompt) :].strip() if text.startswith(prompt) else text.strip()
    except Exception as exc:
        print(f"Local LLM generation disabled after runtime failure: {exc}")
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        return generate_extractive_answer(question, articles)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a grounded answer from retrieved articles.")
    parser.add_argument("--question", required=True)
    parser.add_argument("--articles-json", required=True, help="JSON/JSONL containing retrieved article objects")
    parser.add_argument("--model-path", default="")
    args = parser.parse_args()

    articles = read_json_records(args.articles_json)
    print(generate_answer(args.question, articles, model_path=args.model_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

