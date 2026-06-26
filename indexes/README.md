# Index layout

Generated indexes are stored here:

- `indexes/bm25/bm25.pkl`: standard-library BM25 index.
- `indexes/faiss_bge_m3/dense.pkl`: optional dense index if `sentence-transformers` and local BGE-M3 weights are available.
- `indexes/chroma/`: optional ChromaDB experiments.

The default pipeline works with BM25 + metadata + RRF even when dense dependencies are not installed.

