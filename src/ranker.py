"""XGBoost candidate ranking: build the labeled training set, train, score."""

import pandas as pd
import xgboost as xgb

FEATURE_COLS = [
    # USER
    "user_total_views", "user_total_carts", "user_total_purchases",
    "user_unique_items_viewed", "user_top_category",
    "user_days_since_last_event", "user_interaction_span_days",
    # ITEM
    "item_total_views", "item_total_carts", "item_total_purchases",
    "item_cart_rate", "item_purchase_rate", "item_category",
    "item_days_since_first_seen", "item_popularity",
    # USER-ITEM
    "user_item_views", "user_item_carts", "user_item_purchases",
    "user_item_days_since_last_interaction", "category_affinity",
    # RANKING-ONLY
    "als_score", "itemcf_score", "candidate_source_code",
]

SOURCE_MAP = {"als": 0, "itemcf": 1, "both": 2}


def build_training_set(
    candidates: pd.DataFrame,
    user_features: pd.DataFrame,
    item_features: pd.DataFrame,
    user_item_features: pd.DataFrame,
    positive_pairs: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Join candidates with feature tables; label if positive_pairs (visitorid, itemid) is given."""
    df = candidates.merge(user_features, on="visitorid", how="left")
    df = df.merge(item_features, on="itemid", how="left", suffixes=("", "_item"))
    df = df.merge(user_item_features, on=["visitorid", "itemid"], how="left")

    for col in ["user_item_views", "user_item_carts", "user_item_purchases"]:
        df[col] = df[col].fillna(0)

    df["user_top_category"] = pd.to_numeric(df["user_top_category"], errors="coerce")
    df["item_category"] = pd.to_numeric(df["item_category"], errors="coerce")
    df["candidate_source_code"] = df["candidate_source"].map(SOURCE_MAP)

    if positive_pairs is not None:
        pos = positive_pairs[["visitorid", "itemid"]].drop_duplicates()
        pos["label"] = 1
        df = df.merge(pos, on=["visitorid", "itemid"], how="left")
        df["label"] = df["label"].fillna(0).astype(int)
    return df


def train_ranker(df: pd.DataFrame, **xgb_params) -> xgb.XGBClassifier:
    params = dict(objective="binary:logistic", max_depth=5, n_estimators=200, learning_rate=0.1, random_state=42)
    params.update(xgb_params)
    model = xgb.XGBClassifier(**params)
    model.fit(df[FEATURE_COLS], df["label"])
    return model


def score_candidates(model: xgb.XGBClassifier, df: pd.DataFrame) -> pd.Series:
    return pd.Series(model.predict_proba(df[FEATURE_COLS])[:, 1], index=df.index)


def top_k(df: pd.DataFrame, score_col: str, k: int = 10) -> pd.DataFrame:
    ranked = df.sort_values(["visitorid", score_col], ascending=[True, False]).groupby("visitorid").head(k).copy()
    ranked["rank"] = ranked.groupby("visitorid").cumcount() + 1
    return ranked
