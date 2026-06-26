from __future__ import annotations

from dataclasses import dataclass

from src.common.text import normalize_query, tokenize

DOMAIN_KEYWORDS = {
    "doanh_nghiep": ("doanh nghiệp", "công ty", "vốn điều lệ", "thành lập", "cổ đông", "thành viên", "DNNVV", "SME"),
    "thue": ("thuế", "GTGT", "TNDN", "TNCN", "hóa đơn", "kê khai", "quản lý thuế"),
    "lao_dong": ("lao động", "hợp đồng lao động", "lương", "người lao động", "người sử dụng lao động", "thử việc"),
    "bao_hiem": ("bảo hiểm", "BHXH", "BHYT", "BHTN"),
    "hop_dong": ("hợp đồng", "giao dịch", "vi phạm hợp đồng", "bồi thường"),
    "xu_phat": ("phạt", "xử phạt", "mức phạt", "vi phạm hành chính"),
}


@dataclass(frozen=True)
class QueryProfile:
    original: str
    normalized: str
    domains: tuple[str, ...]
    concepts: tuple[str, ...]
    variants: tuple[str, ...]


def detect_domains(question: str) -> tuple[str, ...]:
    lower = question.lower()
    domains = [domain for domain, terms in DOMAIN_KEYWORDS.items() if any(term.lower() in lower for term in terms)]
    return tuple(domains)


def extract_concepts(question: str, max_terms: int = 12) -> tuple[str, ...]:
    stopwords = {
        "theo",
        "được",
        "phải",
        "cần",
        "nào",
        "gì",
        "khi",
        "cho",
        "của",
        "với",
        "trong",
        "và",
        "hoặc",
        "một",
        "các",
        "những",
        "là",
        "có",
    }
    seen: set[str] = set()
    concepts: list[str] = []
    for token in tokenize(question, add_ngrams=False):
        if token in stopwords or token in seen:
            continue
        seen.add(token)
        concepts.append(token)
        if len(concepts) >= max_terms:
            break
    return tuple(concepts)


def generate_query_variants(question: str) -> tuple[str, ...]:
    normalized = normalize_query(question)
    variants: list[str] = [normalized]
    lower = normalized.lower()

    if "doanh nghiệp nhỏ và vừa" in lower:
        variants.extend(
            [
                "doanh nghiệp nhỏ và vừa điều kiện tiêu chí hỗ trợ",
                "DNNVV số lao động doanh thu nguồn vốn tiêu chí xác định",
                "Luật Hỗ trợ doanh nghiệp nhỏ và vừa điều kiện hỗ trợ",
            ]
        )
    if "thuế" in lower:
        variants.append(f"{normalized} nghĩa vụ kê khai nộp thuế quản lý thuế")
    if "lao động" in lower or "hợp đồng lao động" in lower:
        variants.append(f"{normalized} người lao động người sử dụng lao động hợp đồng lao động")
    if "bảo hiểm" in lower:
        variants.append(f"{normalized} bảo hiểm xã hội bảo hiểm y tế bảo hiểm thất nghiệp")
    if "phạt" in lower or "xử phạt" in lower:
        variants.append(f"{normalized} mức phạt xử phạt vi phạm hành chính")
    if "điều kiện" in lower:
        variants.append(f"{normalized} điều kiện tiêu chí đáp ứng")

    compact = " ".join(extract_concepts(normalized, max_terms=10))
    if compact and compact not in variants:
        variants.append(compact)

    deduped: list[str] = []
    seen: set[str] = set()
    for variant in variants:
        key = variant.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(variant)
    return tuple(deduped[:5])


def build_query_profile(question: str) -> QueryProfile:
    normalized = normalize_query(question)
    return QueryProfile(
        original=question,
        normalized=normalized,
        domains=detect_domains(normalized),
        concepts=extract_concepts(normalized),
        variants=generate_query_variants(normalized),
    )

