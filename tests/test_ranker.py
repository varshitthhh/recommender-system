import pickle
from pathlib import Path

import pandas as pd

from src.ranker import FEATURE_COLS, build_training_set, score_candidates, top_k

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


def test_score_and_topk():
    with open(PROCESSED_DIR / "xgboost_ranker.pkl", "rb") as f:
        model = pickle.load(f)
    candidates = pd.read_parquet(PROCESSED_DIR / "candidates.parquet")
    user_features = pd.read_parquet(PROCESSED_DIR / "user_features.parquet")
    item_features = pd.read_parquet(PROCESSED_DIR / "item_features.parquet")
    user_item_features = pd.read_parquet(PROCESSED_DIR / "user_item_features.parquet")

    df = build_training_set(candidates, user_features, item_features, user_item_features)
    assert set(FEATURE_COLS).issubset(df.columns)

    df["xgb_score"] = score_candidates(model, df)
    ranked = top_k(df, "xgb_score", k=10)

    assert (ranked.groupby("visitorid").size() <= 10).all()
    assert ranked.groupby("visitorid")["rank"].max().le(10).all()
