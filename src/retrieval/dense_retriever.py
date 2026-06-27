from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

from src.common.schema import ensure_article_id, ensure_full_text
from src.retrieval.device import sentence_transformer_device


class DenseRetrieverUnavailable(RuntimeError):
    pass


class DenseIndex:
    """Optional dense index using sentence-transformers and numpy/faiss when installed."""

    def __init__(self, *, model_name: str = "BAAI/bge-m3") -> None:
        self.model_name = model_name
        self.article_ids: list[str] = []
        self.embeddings: Any = None
        self._model: Any = None

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise DenseRetrieverUnavailable(
                "Dense retrieval requires sentence-transformers. "
                "Install it and make sure model weights are available locally."
            ) from exc
        device = sentence_transformer_device(purpose="dense retrieval")
        self._model = SentenceTransformer(self.model_name, device=device) if device else SentenceTransformer(self.model_name)
        return self._model

    def build(self, articles: list[dict[str, Any]], *, batch_size: int = 16) -> None:
        try:
            import numpy as np
        except ImportError as exc:
            raise DenseRetrieverUnavailable("Dense retrieval requires numpy.") from exc
        model = self._load_model()
        self.article_ids = [ensure_article_id(article) for article in articles]
        texts = [ensure_full_text(article) for article in articles]
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        )
        self.embeddings = np.asarray(embeddings, dtype="float32")

    def search(self, query: str, *, top_k: int = 100) -> list[tuple[str, float]]:
        if self.embeddings is None:
            raise DenseRetrieverUnavailable("Dense index is empty.")
        try:
            import numpy as np
        except ImportError as exc:
            raise DenseRetrieverUnavailable("Dense retrieval requires numpy.") from exc
        model = self._load_model()
        query_embedding = model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0].astype("float32")
        scores = np.dot(self.embeddings, query_embedding)
        top_indices = np.argsort(-scores)[:top_k]
        return [(self.article_ids[int(idx)], float(scores[int(idx)])) for idx in top_indices]

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("wb") as f:
            pickle.dump({"model_name": self.model_name, "article_ids": self.article_ids, "embeddings": self.embeddings}, f)

    @classmethod
    def load(cls, path: str | Path) -> "DenseIndex":
        with Path(path).open("rb") as f:
            payload = pickle.load(f)
        index = cls(model_name=payload["model_name"])
        index.article_ids = payload["article_ids"]
        index.embeddings = payload["embeddings"]
        return index

