from __future__ import annotations

from collections import defaultdict


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    *,
    k: int = 60,
    weights: list[float] | None = None,
) -> list[tuple[str, float, dict[str, float]]]:
    """Fuse ranked article-id lists using Reciprocal Rank Fusion."""
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    scores: dict[str, float] = defaultdict(float)
    source_scores: dict[str, dict[str, float]] = defaultdict(dict)
    for source_idx, ranked in enumerate(ranked_lists):
        weight = weights[source_idx] if source_idx < len(weights) else 1.0
        source_name = f"source_{source_idx}"
        for rank, article_id in enumerate(ranked, start=1):
            contribution = weight / (k + rank)
            scores[article_id] += contribution
            source_scores[article_id][source_name] = contribution
    return sorted(
        ((article_id, score, source_scores[article_id]) for article_id, score in scores.items()),
        key=lambda item: item[1],
        reverse=True,
    )

