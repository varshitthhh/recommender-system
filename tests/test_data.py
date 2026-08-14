from src.data import load_interactions


def test_schema():
    df = load_interactions()
    assert set(["visitorid", "itemid", "event", "timestamp"]).issubset(df.columns)
    assert df["event"].isin(["view", "addtocart", "transaction"]).all()
