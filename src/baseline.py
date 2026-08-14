"""Category-aware popularity baseline (also used as the cold-start fallback)."""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"

EVENT_WEIGHTS = {"view": 1, "addtocart": 2, "transaction": 4}


def compute_popularity_score(train: pd.DataFrame) -> pd.Series:
    """transactions*4 + addtocarts*2 + views*1, computed on train only."""
    weighted = train.assign(w=train["event"].map(EVENT_WEIGHTS))
    return weighted.groupby("itemid")["w"].sum().sort_values(ascending=False)


def load_item_category_snapshot(t1_ms: float, raw_dir: Path = RAW_DIR) -> pd.Series:
    """itemid -> categoryid, using only property records with timestamp < t1_ms."""
    dtypes = {"itemid": "int32", "property": "category", "value": "object"}
    parts = []
    for fname in ["item_properties_part1.csv", "item_properties_part2.csv"]:
        df = pd.read_csv(raw_dir / fname, dtype=dtypes)
        df = df[(df["property"] == "categoryid") & (df["timestamp"] < t1_ms)]
        parts.append(df[["itemid", "timestamp", "value"]])
    cat_raw = pd.concat(parts, ignore_index=True)
    return (
        cat_raw.sort_values("timestamp")
        .drop_duplicates("itemid", keep="last")
        .set_index("itemid")["value"]
    )


def compute_user_top_category(train: pd.DataFrame, item_category: pd.Series) -> pd.Series:
    train_cat = train.assign(categoryid=train["itemid"].map(item_category)).dropna(subset=["categoryid"])
    return train_cat.groupby("visitorid")["categoryid"].agg(lambda s: s.value_counts().idxmax())


def build_category_topk(popularity_score: pd.Series, item_category: pd.Series) -> dict:
    item_cat_pop = pd.DataFrame({"itemid": popularity_score.index, "score": popularity_score.values})
    item_cat_pop["categoryid"] = item_cat_pop["itemid"].map(item_category)
    item_cat_pop = item_cat_pop.dropna(subset=["categoryid"])
    return (
        item_cat_pop.sort_values("score", ascending=False)
        .groupby("categoryid")["itemid"]
        .apply(list)
        .to_dict()
    )


def recommend(
    uid,
    k: int,
    global_topk: list,
    category_topk: dict,
    user_top_category: pd.Series,
    purchased_in_train: dict,
    buffer: int = 50,
) -> list:
    """Category-aware top-K with global-popularity fallback for cold-start users."""
    excluded = purchased_in_train.get(uid, set())
    cat = user_top_category.get(uid)
    if cat is not None and cat in category_topk:
        candidates = [i for i in category_topk[cat][: k + buffer] if i not in excluded][:k]
        if len(candidates) < k:
            fill = [i for i in global_topk[: k + buffer] if i not in excluded and i not in candidates]
            candidates += fill[: k - len(candidates)]
        return candidates
    return [i for i in global_topk[: k + buffer] if i not in excluded][:k]
