from __future__ import annotations

from typing import Any


def recall_at_k(predicted_lemmas: list[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    top = predicted_lemmas[:k]
    return 1.0 if any(lemma in relevant for lemma in top) else 0.0


def first_relevant_rank(predicted_lemmas: list[str], relevant: set[str]) -> int | None:
    for index, lemma in enumerate(predicted_lemmas, start=1):
        if lemma in relevant:
            return index
    return None


def mean_reciprocal_rank(ranks: list[int | None]) -> float:
    if not ranks:
        return 0.0
    return sum((1.0 / rank) if rank else 0.0 for rank in ranks) / len(ranks)


def summarize_retrieval(
    per_query: list[dict[str, Any]], k_values: tuple[int, ...] = (1, 5, 10)
) -> dict[str, Any]:
    ranks = [item["rank"] for item in per_query]
    metrics: dict[str, Any] = {
        "queries": len(per_query),
        "mrr": round(mean_reciprocal_rank(ranks), 4),
    }
    for k in k_values:
        hits = sum(1 for item in per_query if item["rank"] is not None and item["rank"] <= k)
        metrics[f"recall_at_{k}"] = round(hits / len(per_query), 4) if per_query else 0.0
    return metrics
