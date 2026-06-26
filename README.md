# Vietnamese Legal RAG Competition

Pipeline này bám theo rule cuộc thi: tối ưu retrieval trước, sau đó sinh câu trả lời có grounding và kiểm tra citation trước khi nộp.

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

## 1. Chuẩn bị dữ liệu luật

Đặt các văn bản pháp luật chính thống dạng UTF-8 `.txt` vào:

```powershell
data/raw_laws/
```

Parse theo cấp `Điều`:

```powershell
python -m src.preprocess.parse_articles --input data/raw_laws --output data/processed/articles.jsonl
```

Nếu parser không suy ra đúng metadata, truyền thủ công cho từng file:

```powershell
python -m src.preprocess.parse_articles `
  --input data/raw_laws/luat_ho_tro_dnnvv.txt `
  --output data/processed/articles.jsonl `
  --law-id "04/2017/QH14" `
  --doc-type "Luật" `
  --doc-title "Luật 04/2017/QH14 Luật Hỗ trợ doanh nghiệp nhỏ và vừa"
```

## 2. Build BM25 index

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

## 3. Tạo `results.json`

Đặt test set của ban tổ chức tại:

```powershell
data/test/test.json
```

Tạo bản balanced, nên dùng cho final/private:

```powershell
python -m src.submit.build_results `
  --test data/test/test.json `
  --articles data/processed/articles.jsonl `
  --output results/results.json `
  --variant balanced
```

Các biến thể:

- `--variant recall`: 7-10 điều, ưu tiên recall cho public probing.
- `--variant balanced`: 4-7 điều, cân bằng F2 và QA.
- `--variant precision`: 2-5 điều, answer sạch hơn.

Nếu máy có local model hợp lệ và package sẵn:

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

Không dùng API model đóng như GPT-4o hoặc Gemini cho bài nộp.

## 4. Validate và zip phẳng

```powershell
python -m src.submit.validate_results --input results/results.json --strict
python -m src.submit.zip_submission --results results/results.json --output results/submission.zip
```

Zip hợp lệ chỉ có:

```text
submission.zip
└── results.json
```

## 5. Dev set và F2 nội bộ

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

## 6. Những điểm dễ mất điểm

- Answer nhắc `Điều X` nhưng `relevant_articles` không có `Điều X`.
- `relevant_articles` đúng nhưng answer không nhắc rõ `Điều X`.
- Tên văn bản sai format `<Loại văn bản> <mã văn bản> <trích yếu>`.
- Chunk mất metadata `article_number`.
- Chỉ dùng dense, bỏ BM25.
- Chọn quá ít điều làm recall thấp hoặc quá nhiều điều làm precision thấp.
- Nén zip có thư mục con.

## 7. Files chính

- `src/preprocess/parse_articles.py`: clean và split luật theo cấp `Điều`.
- `src/retrieval/bm25_retriever.py`: BM25 không cần dependency ngoài.
- `src/retrieval/pipeline.py`: query expansion, BM25, optional dense, RRF, rerank, selection.
- `src/generation/answer_prompt.py`: prompt grounded cho local LLM.
- `src/generation/generate_answer.py`: generator fallback extractive, có hook local LLM.
- `src/generation/verifier.py`: sửa/kiểm tra citation và schema submission.
- `src/submit/build_results.py`: tạo `results.json`.
- `src/submit/zip_submission.py`: tạo zip phẳng.

