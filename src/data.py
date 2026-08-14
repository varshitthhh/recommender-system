"""Load the cleaned interaction log produced by notebooks/01_eda.ipynb."""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def load_interactions() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR / "interactions.parquet")


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the train/val/test splits produced by notebooks/02_baseline.ipynb."""
    train = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    val = pd.read_parquet(PROCESSED_DIR / "val.parquet")
    test = pd.read_parquet(PROCESSED_DIR / "test.parquet")
    return train, val, test
