# Vietnamese Legal RAG Competition

Pipeline này bám theo rule cuộc thi: retrieval là trọng tâm vì F2 phụ thuộc mạnh vào `relevant_docs` và `relevant_articles`. Mặc định repo chạy được bằng standard library với BM25 + answer extractive; khi máy có package/model local thì bật thêm BGE-M3, BGE reranker và Qwen2.5-7B-Instruct.

Luồng chính:

```text
Question
-> query preprocessing / expansion
-> BM25 + optional dense retrieval
-> RRF fusion
-> reranking
-> dynamic article selection
-> grounded answer generation
-> citation / JSON validation
-> results.json
-> flat submission.zip
```

## 1. Dữ liệu hiện tại

Test set của ban tổ chức đặt đúng tại:

```powershell
data/test/test.json
```

Corpus điều luật mà pipeline dùng nằm tại:

```powershell
data/processed/articles.jsonl
```

Nếu file này chưa tồn tại, cần import hoặc parse văn bản luật trước khi build index.

## 2. Nguồn luật khuyến nghị

Nguồn chính nên dùng là `th1nhng0/vietnamese-legal-documents` trên Hugging Face, lấy từ cổng VBPL chính thống. Manifest nguồn nằm ở:

```powershell
configs/sources.json
```

Import toàn bộ corpus chính cho bản final:

```powershell
python -m src.preprocess.import_hf_legal `
  --output data/processed/articles.jsonl `
  --max-docs -1 `
  --metadata-scan-limit -1 `
  --batch-size 200 `
  --require-metadata
```

Smoke test nhanh trước khi chạy dài:

```powershell
python -m src.preprocess.import_hf_legal `
  --output data/processed/sample_articles.jsonl `
  --max-docs 20 `
  --metadata-scan-limit 500 `
  --batch-size 20
```

Nếu muốn bổ sung corpus thủ công, đặt `.txt` UTF-8 vào:

```powershell
data/raw_laws/
```

Rồi parse theo cấp `Điều`:

```powershell
python -m src.preprocess.parse_articles --input data/raw_laws --output data/processed/articles.jsonl
```

## 3. Build BM25 index

```powershell
python -m src.retrieval.bm25_retriever build `
  --articles data/processed/articles.jsonl `
  --output indexes/bm25/bm25.pkl
```

Test retrieval nhanh:

```powershell
python -m src.retrieval.pipeline `
  --articles data/processed/articles.jsonl `
  --question "Doanh nghiệp nhỏ và vừa phải đáp ứng điều kiện nào để được hỗ trợ?"
```

## 4. Model tốt nhất trong rule

Cấu hình model nằm ở:

```powershell
configs/model.yaml
```

Stack final khuyến nghị:

- Retrieval bắt buộc: BM25.
- Dense retrieval: `BAAI/bge-m3`.
- Reranker: `BAAI/bge-reranker-v2-m3`.
- Generator local: `Qwen/Qwen2.5-7B-Instruct`.

Cài dependency tùy chọn khi muốn chạy pretrained stack đầy đủ:

```powershell
python -m pip install -r requirements-optional.txt
```

Kiểm tra máy hiện tại đã sẵn sàng chưa:

```powershell
python -m src.models.check_environment
```

Không dùng API model đóng như GPT-4o hoặc Gemini cho bài nộp.

## 5. Tạo `results.json`

Bản chạy được mọi máy, dùng BM25 + answer extractive:

```powershell
python -m src.submit.build_results `
  --test data/test/test.json `
  --articles data/processed/articles.jsonl `
  --output results/results.json `
  --variant balanced
```

Bản mạnh nhất khi đã có dependency và model local:

```powershell
python -m src.submit.build_results `
  --test data/test/test.json `
  --articles data/processed/articles.jsonl `
  --output results/results.json `
  --variant balanced `
  --use-dense `
  --use-cross-encoder `
  --model-path D:/models/Qwen2.5-7B-Instruct
```

Các biến thể chọn điều luật:

- `--variant recall`: 7-10 điều, ưu tiên recall cho probing.
- `--variant balanced`: 4-7 điều, cân bằng F2 và QA.
- `--variant precision`: 2-5 điều, answer sạch hơn.

## 6. Validate và zip phẳng

```powershell
python -m src.submit.validate_results --input results/results.json --strict
python -m src.submit.zip_submission --results results/results.json --output results/submission.zip
```

Zip hợp lệ chỉ có:

```text
submission.zip
└── results.json
```

## 7. Dev set và F2 nội bộ

Tạo `data/dev/dev_labeled.json` có cùng format output, kèm `relevant_articles` gán thủ công. Sau đó tính macro Precision/Recall/F2:

```powershell
python -m src.evaluation.compute_f2 `
  --predictions results/results.json `
  --gold data/dev/dev_labeled.json
```

Chạy ablation 3 policy:

```powershell
python -m src.evaluation.ablation `
  --dev data/dev/dev_labeled.json `
  --articles data/processed/articles.jsonl `
  --output-dir results/ablation
```

## 8. Files chính

- `src/preprocess/import_hf_legal.py`: tải dataset luật từ Hugging Face, sửa mojibake, strip HTML, split theo `Điều`.
- `src/preprocess/parse_articles.py`: clean và split luật local `.txt` theo cấp `Điều`.
- `src/retrieval/bm25_retriever.py`: BM25 không cần dependency ngoài.
- `src/retrieval/pipeline.py`: query expansion, BM25, optional dense, RRF, rerank, selection.
- `src/generation/answer_prompt.py`: prompt grounded cho local LLM.
- `src/generation/generate_answer.py`: generator fallback extractive, có hook local LLM.
- `src/generation/verifier.py`: sửa/kiểm tra citation và schema submission.
- `src/submit/build_results.py`: tạo `results.json`.
- `src/submit/zip_submission.py`: tạo zip phẳng.
