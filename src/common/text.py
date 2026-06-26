from __future__ import annotations

import re
import unicodedata

TOKEN_RE = re.compile(r"[0-9A-Za-zÀ-ỹĐđ]+", re.UNICODE)

ABBREVIATIONS = (
    (re.compile(r"\bSME\b", re.IGNORECASE), "doanh nghiệp nhỏ và vừa"),
    (re.compile(r"\bDNNVV\b", re.IGNORECASE), "doanh nghiệp nhỏ và vừa"),
    (re.compile(r"\bBHXH\b", re.IGNORECASE), "bảo hiểm xã hội"),
    (re.compile(r"\bBHYT\b", re.IGNORECASE), "bảo hiểm y tế"),
    (re.compile(r"\bBHTN\b", re.IGNORECASE), "bảo hiểm thất nghiệp"),
    (re.compile(r"\bGTGT\b", re.IGNORECASE), "giá trị gia tăng"),
    (re.compile(r"\bTNDN\b", re.IGNORECASE), "thu nhập doanh nghiệp"),
    (re.compile(r"\bTNCN\b", re.IGNORECASE), "thu nhập cá nhân"),
    (re.compile(r"\bHĐLĐ\b", re.IGNORECASE), "hợp đồng lao động"),
)

LEGAL_PHRASES = (
    "doanh nghiệp nhỏ và vừa",
    "luật doanh nghiệp",
    "hỗ trợ doanh nghiệp nhỏ và vừa",
    "bảo hiểm xã hội",
    "hợp đồng lao động",
    "thuế giá trị gia tăng",
    "thuế thu nhập doanh nghiệp",
    "thuế thu nhập cá nhân",
    "quản lý thuế",
    "xử phạt vi phạm hành chính",
    "mức phạt",
    "điều kiện",
    "nghĩa vụ",
    "trách nhiệm",
    "người lao động",
    "người sử dụng lao động",
    "vốn điều lệ",
    "thành lập doanh nghiệp",
)


def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFC", text or "")


def strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text or "")
    return "".join(char for char in normalized if unicodedata.category(char) != "Mn")


def expand_abbreviations(text: str) -> str:
    expanded = normalize_unicode(text)
    for pattern, replacement in ABBREVIATIONS:
        expanded = pattern.sub(replacement, expanded)
    return expanded


def normalize_query(text: str) -> str:
    normalized = expand_abbreviations(text)
    normalized = normalized.replace("“", '"').replace("”", '"').replace("’", "'")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def tokenize(text: str, *, add_ngrams: bool = True) -> list[str]:
    normalized = normalize_query(text).lower()
    words = TOKEN_RE.findall(normalized)
    tokens = [word for word in words if len(word) > 1 or word.isdigit()]
    if add_ngrams:
        for n in (2, 3):
            for i in range(0, max(0, len(words) - n + 1)):
                gram = "_".join(words[i : i + n])
                if len(gram) > 3:
                    tokens.append(gram)
        for phrase in LEGAL_PHRASES:
            if phrase in normalized:
                tokens.append(phrase.replace(" ", "_"))
    return tokens


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", normalize_unicode(text)).strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?。])\s+|(?<=;)\s+", text)
    return [part.strip() for part in parts if part.strip()]

