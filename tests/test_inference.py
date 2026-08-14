from src.inference import load_serving_artifacts, recommend


def test_unknown_user_fallback():
    precomputed, global_popularity = load_serving_artifacts()
    recs, source = recommend(999999999, precomputed, global_popularity, k=10)
    assert len(recs) == 10
    assert source == "popularity_fallback"


def test_known_user_personalized():
    precomputed, global_popularity = load_serving_artifacts()
    known_uid = next(iter(precomputed))
    recs, source = recommend(known_uid, precomputed, global_popularity, k=10)
    assert len(recs) <= 10
    assert source == "personalized"
