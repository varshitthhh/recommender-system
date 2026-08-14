from src.data import load_splits


def test_no_leakage():
    train, val, test = load_splits()
    assert train["timestamp"].max() < val["timestamp"].min()
    assert val["timestamp"].max() < test["timestamp"].min()
