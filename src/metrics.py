from __future__ import annotations

import math
from collections.abc import Sequence


def reciprocal_rank(ranked_items: Sequence[str], target: str) -> float:
    target = str(target)
    for index, item in enumerate(ranked_items, start=1):
        if str(item) == target:
            return 1.0 / index
    return 0.0


def mrr(ranked_lists: Sequence[Sequence[str]], targets: Sequence[str]) -> float:
    if not ranked_lists:
        return 0.0
    values = [
        reciprocal_rank(ranked_items, target)
        for ranked_items, target in zip(ranked_lists, targets)
    ]
    return float(sum(values) / len(values))


def hit_rate_at_k(
    ranked_lists: Sequence[Sequence[str]], targets: Sequence[str], k: int
) -> float:
    if not ranked_lists:
        return 0.0
    hits = 0
    for ranked_items, target in zip(ranked_lists, targets):
        hits += str(target) in {str(item) for item in ranked_items[:k]}
    return hits / len(ranked_lists)


def recall_at_k(
    ranked_lists: Sequence[Sequence[str]], targets: Sequence[str], k: int
) -> float:
    return hit_rate_at_k(ranked_lists, targets, k)


def ndcg_at_k(
    ranked_lists: Sequence[Sequence[str]], targets: Sequence[str], k: int
) -> float:
    if not ranked_lists:
        return 0.0
    scores = []
    for ranked_items, target in zip(ranked_lists, targets):
        rr = reciprocal_rank(ranked_items[:k], target)
        scores.append(0.0 if rr == 0.0 else 1.0 / math.log2((1.0 / rr) + 1.0))
    return float(sum(scores) / len(scores))


def evaluate_ranking(
    ranked_lists: Sequence[Sequence[str]],
    targets: Sequence[str],
    ks: Sequence[int] = (1, 5, 10),
) -> dict[str, float]:
    results = {"MRR": mrr(ranked_lists, targets)}
    for k in ks:
        results[f"Recall@{k}"] = recall_at_k(ranked_lists, targets, k)
        results[f"HitRate@{k}"] = hit_rate_at_k(ranked_lists, targets, k)
        results[f"NDCG@{k}"] = ndcg_at_k(ranked_lists, targets, k)
    return results
