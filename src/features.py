"""USER / ITEM / USER-ITEM feature engineering, computed from train-only data.

Mirrors notebooks/03_model.ipynb Cells 2-4. `cutoff_ms` is the split boundary
(T1 for the train->val transition, T2 for train+val->test) that all recency
features are measured relative to.
"""

import numpy as np
import pandas as pd

DAY_MS = 1000 * 60 * 60 * 24
EVENT_WEIGHTS = {"view": 1, "addtocart": 2, "transaction": 4}


def build_user_features(train: pd.DataFrame, item_category: pd.Series, cutoff_ms: float) -> pd.DataFrame:
    event_counts = train.pivot_table(index="visitorid", columns="event", values="itemid", aggfunc="count", fill_value=0)
    for col in ["view", "addtocart", "transaction"]:
        if col not in event_counts.columns:
            event_counts[col] = 0

    user_features = event_counts.rename(columns={
        "view": "user_total_views", "addtocart": "user_total_carts", "transaction": "user_total_purchases",
    })[["user_total_views", "user_total_carts", "user_total_purchases"]]

    user_features["user_unique_items_viewed"] = (
        train[train["event"] == "view"].groupby("visitorid")["itemid"].nunique()
    ).reindex(user_features.index, fill_value=0)

    view_events = train[train["event"] == "view"].copy()
    view_events["categoryid"] = view_events["itemid"].map(item_category)
    view_cat = view_events.dropna(subset=["categoryid"])
    user_features["user_top_category"] = view_cat.groupby("visitorid")["categoryid"].agg(lambda s: s.value_counts().idxmax())

    last_event = train.groupby("visitorid")["timestamp"].max()
    first_event = train.groupby("visitorid")["timestamp"].min()
    user_features["user_days_since_last_event"] = (cutoff_ms - last_event) / DAY_MS
    user_features["user_interaction_span_days"] = (last_event - first_event) / DAY_MS
    return user_features


def build_item_features(train: pd.DataFrame, item_category: pd.Series, popularity_score: pd.Series, cutoff_ms: float) -> pd.DataFrame:
    item_event_counts = train.pivot_table(index="itemid", columns="event", values="visitorid", aggfunc="count", fill_value=0)
    for col in ["view", "addtocart", "transaction"]:
        if col not in item_event_counts.columns:
            item_event_counts[col] = 0

    item_features = item_event_counts.rename(columns={
        "view": "item_total_views", "addtocart": "item_total_carts", "transaction": "item_total_purchases",
    })[["item_total_views", "item_total_carts", "item_total_purchases"]]

    item_features["item_cart_rate"] = np.where(
        item_features["item_total_views"] > 10,
        item_features["item_total_carts"] / item_features["item_total_views"],
        np.nan,
    )
    item_features["item_purchase_rate"] = np.where(
        item_features["item_total_carts"] > 5,
        item_features["item_total_purchases"] / item_features["item_total_carts"],
        np.nan,
    )
    item_features["item_category"] = item_category.reindex(item_features.index)

    first_seen = train.groupby("itemid")["timestamp"].min()
    item_features["item_days_since_first_seen"] = (cutoff_ms - first_seen) / DAY_MS
    item_features["item_popularity"] = popularity_score.reindex(item_features.index, fill_value=0)
    return item_features


def build_user_item_features(train: pd.DataFrame, item_category: pd.Series, user_top_category: pd.Series, cutoff_ms: float) -> pd.DataFrame:
    ui_pivot = (
        train.pivot_table(index=["visitorid", "itemid"], columns="event", values="timestamp", aggfunc="count", fill_value=0)
        .reset_index()
    )
    for col in ["view", "addtocart", "transaction"]:
        if col not in ui_pivot.columns:
            ui_pivot[col] = 0

    user_item_features = ui_pivot.rename(columns={
        "view": "user_item_views", "addtocart": "user_item_carts", "transaction": "user_item_purchases",
    })[["visitorid", "itemid", "user_item_views", "user_item_carts", "user_item_purchases"]]

    ui_last = train.groupby(["visitorid", "itemid"])["timestamp"].max().reset_index(name="last_ts")
    user_item_features = user_item_features.merge(ui_last, on=["visitorid", "itemid"])
    user_item_features["user_item_days_since_last_interaction"] = (cutoff_ms - user_item_features["last_ts"]) / DAY_MS
    user_item_features = user_item_features.drop(columns="last_ts")

    item_cat_for_ui = user_item_features["itemid"].map(item_category)
    user_cat_for_ui = user_item_features["visitorid"].map(user_top_category)
    both_known = item_cat_for_ui.notna() & user_cat_for_ui.notna()
    category_affinity = pd.Series(np.nan, index=user_item_features.index)
    category_affinity[both_known] = (item_cat_for_ui[both_known] == user_cat_for_ui[both_known]).astype(float)
    user_item_features["category_affinity"] = category_affinity
    return user_item_features
