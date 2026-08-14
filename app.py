"""Gradio demo: looks up precomputed recommendations. No live model inference."""

import random

import gradio as gr
import pandas as pd

from src.inference import load_serving_artifacts, recommend

precomputed_final_recs, global_popularity = load_serving_artifacts()

_warm_examples = random.sample(list(precomputed_final_recs.keys()), min(3, len(precomputed_final_recs)))
_unknown_example = 999999999


def get_recommendations(user_id: str, k: int):
    try:
        uid = int(user_id)
    except (TypeError, ValueError):
        return pd.DataFrame({"error": ["user_id must be an integer visitorid"]})

    recs, source = recommend(uid, precomputed_final_recs, global_popularity, k=int(k))
    return pd.DataFrame({"rank": range(1, len(recs) + 1), "itemid": recs, "source": source})


description = (
    "Looks up a precomputed recommendation list for the given visitorid (ALS+Item-CF candidates, "
    "ranked by XGBoost) with a popularity fallback for unknown/cold-start users.\n\n"
    f"Try a personalized user: **{_warm_examples}** -- or an unknown one: **{_unknown_example}** "
    "(falls back to popularity)."
)

gr.Interface(
    fn=get_recommendations,
    inputs=[gr.Textbox(label="User ID (visitorid)"), gr.Slider(1, 20, value=10, step=1, label="Top-K")],
    outputs=gr.Dataframe(label="Recommendations"),
    title="RetailRocket Recommender",
    description=description,
).launch()
