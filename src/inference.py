"""Serving-time recommendation lookup.

Everything is precomputed offline (candidate generation + XGBoost ranking, see
notebooks/03_model.ipynb, 03b_ranking.ipynb, 04_evaluation.ipynb). At request
time this is a dict lookup only -- no ALS, Item-CF, or XGBoost inference call.
"""

import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def load_serving_artifacts():
    """Load the precomputed lookup table + popularity fallback list once at startup.

    Uses personalized_recs_test.parquet (warm users only), NOT the blended
    final_recs_test.parquet -- that file already has the popularity fallback
    merged in for cold users, which would make every test user look
    "personalized" here even when their rec is really just the fallback.
    """
    personalized_recs = pd.read_parquet(PROCESSED_DIR / "personalized_recs_test.parquet")
    precomputed_final_recs = (
        personalized_recs.sort_values(["visitorid", "rank"])
        .groupby("visitorid")["itemid"].apply(list).to_dict()
    )
    with open(PROCESSED_DIR / "global_popularity_top10.json") as f:
        global_popularity = json.load(f)
    return precomputed_final_recs, global_popularity


def recommend(visitorid: int, precomputed_final_recs: dict, global_popularity: list, k: int = 10):
    """user -> precomputed Top-K -> popularity fallback. Returns (item_ids, source)."""
    if visitorid in precomputed_final_recs:
        return precomputed_final_recs[visitorid][:k], "personalized"
    return global_popularity[:k], "popularity_fallback"
