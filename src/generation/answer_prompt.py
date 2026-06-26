from __future__ import annotations

from typing import Any

from src.common.schema import normalize_article_number

ANSWER_SYSTEM_PROMPT = """Bạn là trợ lý pháp lý AI cho doanh nghiệp SME tại Việt Nam.

Nhiệm vụ:
Trả lời câu hỏi pháp lý dựa DUY NHẤT trên các căn cứ pháp luật được cung cấp.

Nguyên tắc bắt buộc:
1. Không sử dụng kiến thức ngoài context.
2. Không bịa điều luật, không bịa văn bản.
3. Phải nêu rõ căn cứ theo dạng "Điều X" trong câu trả lời.
4. Nếu nhiều điều cùng liên quan, hãy tổng hợp ngắn gọn.
5. Nếu căn cứ chưa đủ, nói rõ: "Dựa trên dữ liệu được cung cấp, chưa đủ căn cứ để kết luận đầy đủ."
6. Văn phong rõ ràng, dễ hiểu cho người không chuyên.
7. Không đưa ra cam kết pháp lý tuyệt đối; chỉ tư vấn sơ bộ.
"""


def format_retrieved_context(articles: list[dict[str, Any]], *, max_chars_per_article: int = 2400) -> str:
    blocks: list[str] = []
    for index, article in enumerate(articles, start=1):
        content = str(article.get("content", "")).strip()
        if len(content) > max_chars_per_article:
            content = content[:max_chars_per_article].rstrip() + "..."
        blocks.append(
            "\n".join(
                [
                    f"[{index}] {article.get('law_id', '')}|{article.get('doc_title', '')}|{normalize_article_number(article.get('article_number', ''))}",
                    f"Tiêu đề điều: {article.get('article_title', '')}",
                    f"Nội dung: {content}",
                ]
            )
        )
    return "\n\n".join(blocks)


def build_answer_prompt(question: str, articles: list[dict[str, Any]]) -> str:
    return f"""{ANSWER_SYSTEM_PROMPT}
Câu hỏi:
{question}

Căn cứ pháp luật:
{format_retrieved_context(articles)}

Hãy trả lời theo cấu trúc:
- Căn cứ pháp luật:
- Trả lời ngắn gọn:
- Lưu ý thực tiễn:
"""

