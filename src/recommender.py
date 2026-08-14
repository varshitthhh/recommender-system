"""ALS + Item-CF candidate generation."""

import numpy as np
import pandas as pd


def generate_candidates(
    als_model, itemcf_model, user_item_csr, user_idx_array,
    purchased_item_idx: dict, candidate_n=50, retrieve_n=80, batch_size=20000,
):
    """Top-N candidates per user from each model, unioned into a long-format pool.

    Retrieval does NOT use implicit's filter_already_liked_items: that would drop
    every train-interacted item, including merely-viewed ones -- which are exactly
    the items most likely to convert on a return visit (browse-then-buy is the norm
    in e-commerce). Instead, only items the user already PURCHASED in train are
    removed, post-hoc, matching the roadmap's literal spec. retrieve_n > candidate_n
    gives buffer room for that post-hoc filtering.

    purchased_item_idx: {user_idx: set(item_idx)} of train purchases to exclude.

    Returns a DataFrame [user_idx, item_idx, als_score, itemcf_score, candidate_source]
    with candidate_source in {"als", "itemcf", "both"}.
    """
    n = len(user_idx_array)
    als_ids = np.full((n, retrieve_n), -1, dtype=np.int64)
    als_scores = np.zeros((n, retrieve_n), dtype=np.float32)
    itemcf_ids = np.full((n, retrieve_n), -1, dtype=np.int64)
    itemcf_scores = np.zeros((n, retrieve_n), dtype=np.float32)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_idx = user_idx_array[start:end]
        batch_matrix = user_item_csr[batch_idx]

        ids, scores = als_model.recommend(batch_idx, batch_matrix, N=retrieve_n, filter_already_liked_items=False)
        als_ids[start:end, : ids.shape[1]] = ids
        als_scores[start:end, : ids.shape[1]] = scores

        ids2, scores2 = itemcf_model.recommend(batch_idx, batch_matrix, N=retrieve_n, filter_already_liked_items=False)
        itemcf_ids[start:end, : ids2.shape[1]] = ids2
        itemcf_scores[start:end, : ids2.shape[1]] = scores2

    def _mask_purchased(ids_arr):
        keep = np.ones_like(ids_arr, dtype=bool)
        for row, uidx in enumerate(user_idx_array):
            purchased = purchased_item_idx.get(uidx)
            if purchased:
                keep[row] &= ~np.isin(ids_arr[row], list(purchased))
        return keep

    als_keep = _mask_purchased(als_ids) & (als_ids >= 0)
    itemcf_keep = _mask_purchased(itemcf_ids) & (itemcf_ids >= 0)

    user_idx_repeated = np.repeat(user_idx_array, retrieve_n)

    als_long = pd.DataFrame({"user_idx": user_idx_repeated, "item_idx": als_ids.flatten(), "als_score": als_scores.flatten()})
    als_long = als_long[als_keep.flatten()]
    als_long = als_long.sort_values(["user_idx", "als_score"], ascending=[True, False]).groupby("user_idx").head(candidate_n)

    itemcf_long = pd.DataFrame({"user_idx": user_idx_repeated, "item_idx": itemcf_ids.flatten(), "itemcf_score": itemcf_scores.flatten()})
    itemcf_long = itemcf_long[itemcf_keep.flatten()]
    itemcf_long = itemcf_long.sort_values(["user_idx", "itemcf_score"], ascending=[True, False]).groupby("user_idx").head(candidate_n)

    pool = pd.merge(als_long, itemcf_long, on=["user_idx", "item_idx"], how="outer", indicator=True)
    pool["als_score"] = pool["als_score"].fillna(0.0)
    pool["itemcf_score"] = pool["itemcf_score"].fillna(0.0)
    pool["candidate_source"] = pool["_merge"].map({"left_only": "als", "right_only": "itemcf", "both": "both"})
    return pool.drop(columns="_merge")
