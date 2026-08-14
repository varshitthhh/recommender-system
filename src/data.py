"""Load the cleaned interaction log produced by notebooks/01_eda.ipynb."""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def load_interactions() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR / "interactions.parquet")
