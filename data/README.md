# Data layout

Put official legal texts as UTF-8 `.txt` files in `data/raw_laws/`.

Generated files:

- `data/processed/articles.jsonl`: article-level corpus used for retrieval.
- `data/test/test.json`: contest test questions, each with `id` and `question`.
- `data/dev/dev_labeled.json`: optional internal dev set with `relevant_articles`.

Each article record should keep the contest-critical metadata:

```json
{
  "article_id": "04/2017/QH14|Điều 4",
  "law_id": "04/2017/QH14",
  "doc_type": "Luật",
  "doc_title": "Luật 04/2017/QH14 Luật Hỗ trợ doanh nghiệp nhỏ và vừa",
  "article_number": "Điều 4",
  "article_title": "Tiêu chí xác định doanh nghiệp nhỏ và vừa",
  "content": "...",
  "full_text_for_embedding": "...",
  "keywords": ["doanh nghiệp nhỏ và vừa"],
  "effective_date": "",
  "status": "chưa rõ",
  "source_url": ""
}
```

