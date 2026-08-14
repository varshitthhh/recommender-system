import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from src.recommender import generate_candidates

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


def _load_artifacts():
    with open(PROCESSED_DIR / "als_model.pkl", "rb") as f:
        als_model = pickle.load(f)
    with open(PROCESSED_DIR / "itemcf_matrix.pkl", "rb") as f:
        itemcf_model = pickle.load(f)
    with open(PROCESSED_DIR / "id_mappings.pkl", "rb") as f:
        id_mappings = pickle.load(f)
    user_item_features = pd.read_parquet(PROCESSED_DIR / "user_item_features.parquet")
    return als_model, itemcf_model, id_mappings, user_item_features


def _build_csr(user_item_features, id_mappings):
    user_to_idx, item_to_idx = id_mappings["user_to_idx"], id_mappings["item_to_idx"]
    weight = (
        user_item_features["user_item_views"] * 1
        + user_item_features["user_item_carts"] * 2
        + user_item_features["user_item_purchases"] * 4
    ).astype(np.float32)
    row = user_item_features["visitorid"].map(user_to_idx).values
    col = user_item_features["itemid"].map(item_to_idx).values
    return csr_matrix((weight.values, (row, col)), shape=(len(user_to_idx), len(item_to_idx)))


def test_output_shape():
    als_model, itemcf_model, id_mappings, user_item_features = _load_artifacts()
    user_item_csr = _build_csr(user_item_features, id_mappings)

    sample_user_idx = np.arange(5)
    candidates = generate_candidates(als_model, itemcf_model, user_item_csr, sample_user_idx, {}, candidate_n=10)

    assert len(candidates) > 0
    assert candidates["item_idx"].dtype.kind in "iu"
    # generate_candidates returns the UNION of each model's own top-N (up to 2N per user);
    # the caller is responsible for capping to a final pool size after alpha-blending scores.
    assert candidates.groupby("user_idx").size().max() <= 20
    assert set(candidates["candidate_source"].unique()).issubset({"als", "itemcf", "both"})
