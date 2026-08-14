"""Recall@K, NDCG@K, Coverage@K for top-K recommendation lists."""

import math

import numpy as np


def recall_at_k(recs_by_user: dict, relevant: dict) -> tuple[float, int]:
    scores = []
    for uid, rel in relevant.items():
        if not rel:
            continue
        rec = recs_by_user.get(uid, [])
        scores.append(len(set(rec) & rel) / len(rel))
    return (float(np.mean(scores)) if scores else 0.0), len(scores)


def ndcg_at_k(recs_by_user: dict, relevant: dict, k: int) -> tuple[float, int]:
    scores = []
    for uid, rel in relevant.items():
        if not rel:
            continue
        rec = recs_by_user.get(uid, [])
        dcg = sum(1.0 / math.log2(i + 2) for i, item in enumerate(rec[:k]) if item in rel)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(rel))))
        scores.append(dcg / idcg if idcg > 0 else 0.0)
    return (float(np.mean(scores)) if scores else 0.0), len(scores)


def coverage_at_k(recs_by_user: dict, catalog_size: int) -> tuple[float, int]:
    unique_recommended = set()
    for rec in recs_by_user.values():
        unique_recommended.update(rec)
    return len(unique_recommended) / catalog_size, len(unique_recommended)
