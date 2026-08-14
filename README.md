# RetailRocket Recommender

A personalized e-commerce recommendation system built on the RetailRocket implicit-feedback clickstream dataset — popularity baseline, ALS + Item-CF candidate generation, XGBoost ranking, offline evaluation, and a Gradio demo.

## 1. Business problem

> Personalize product discovery by recommending products that are more relevant to a user's observed behavior than a generic popularity-based recommendation, with the downstream objective of improving add-to-cart and purchase outcomes.

**What this project does NOT claim:** measured revenue uplift (no online A/B test), a proprietary company architecture, or an actual conversion-rate improvement.

**What this project DOES claim:** offline behavioral evaluation shows the hybrid model surfaces more relevant items than popularity for warm users; AddToCart-Recall@10 and Purchase-Recall@10 are used as behavioral proxies, not business KPIs. The same ranking abstraction (candidate generation → learned ranking → fallback) transfers to any implicit-feedback, session-based recommendation problem.

## 2. Dataset

[RetailRocket E-Commerce Dataset](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset) — ~2.76M behavioral events (views, add-to-cart, transactions) from ~1.4M visitors over ~4.5 months (2015-05-03 to 2015-09-18), pure implicit feedback, no explicit ratings. Files: `events.csv`, `item_properties_part1/2.csv`, `category_tree.csv`.

## 3. Data engineering decisions

- Exact duplicate events (same visitor/item/event/timestamp) dropped as logging errors (460 rows).
- Events weighted `view=1, addtocart=2, transaction=3` for exploratory ordering; the modeling weight used throughout is `views*1 + carts*2 + purchases*4`.
- `categoryid`/`available` are the only human-readable `item_properties`; every other property is a hashed numeric id (confirmed empirically — there is no usable `price` field, contrary to what a literal reading of "extract price where present" might suggest).
- Sparsity: 99.9994% (2.15M observed interactions / 330.9M possible user-item pairs). 91.7% of users have <3 interactions; 65.2% of items have <5 interactions — cold-start and long-tail dominate this dataset.

See [notebooks/01_eda.ipynb](notebooks/01_eda.ipynb).

## 4. Baseline: category-aware popularity

`popularity_score(item) = transactions*4 + addtocarts*2 + views*1`, computed on train only. For each user, recommend top-K from their train-derived top category (mode of viewed items' categories), backfilled with global top-K; pure global top-K for cold-start users. Also serves as the cold-start fallback in production.

**Val metrics** (Phase 2): Recall@10 0.0143, NDCG@10 0.0083, Coverage@10 0.0360, AddToCart-Recall@10 0.0194, Purchase-Recall@10 0.0251.

See [notebooks/02_baseline.ipynb](notebooks/02_baseline.ipynb), [src/baseline.py](src/baseline.py).

## 5. Final model: ALS + Item-CF candidates → XGBoost ranking

Two single-responsibility stages, never reported as competing standalone models:

- **Candidate generation** (`alpha * ALS + (1-alpha) * Item-CF`, alpha tuned on val, top-50 candidates/user): ALS (`implicit`, factors=50, iterations=20, regularization=0.1) learns global collaborative factors from confidence-weighted implicit interactions. Item-CF (cosine similarity, K=20 neighbors) captures local "viewed X, also viewed Y" behavior ALS can miss and adds catalog coverage. Combining both gives the ranking stage a broader, more diverse candidate pool than either alone.
- **Ranking** (XGBoost, `binary:logistic`, max_depth=5, n_estimators=200, learning_rate=0.1): re-ranks the ~50 candidates using behavioral + collaborative features. Never run against the full catalog — only the pre-filtered candidate set, keeping training/inference cheap.

**A real mid-course fix, documented in [notebooks/03_model.ipynb](notebooks/03_model.ipynb) Cell 8:** candidate retrieval originally filtered out *all* train-interacted items (`filter_already_liked_items=True`), which also removed merely-*viewed* items — exactly the ones most likely to convert on a return visit. That left only 46/762,000 candidate rows positive, too sparse to train on. Filtering now excludes only *purchased* items (matching the roadmap's spec), raising candidate Recall@50 from ~0.03 to 0.21 and positives from 46 to 226.

See [notebooks/03_model.ipynb](notebooks/03_model.ipynb), [notebooks/03b_ranking.ipynb](notebooks/03b_ranking.ipynb), [src/recommender.py](src/recommender.py), [src/ranker.py](src/ranker.py).

## 6. Feature engineering

All features computed from **train only** (timestamp < split boundary), with an explicit leakage assertion in every notebook (recency features must be non-negative relative to the cutoff).

- **User:** total views/carts/purchases, unique items viewed, top category (mode of viewed items), days since last event, interaction span.
- **Item:** total views/carts/purchases, cart rate (views>10 only), purchase rate (carts>5 only), category, days since first seen, popularity score.
- **User-item:** views/carts/purchases on that specific item, days since last interaction, category affinity.
- **Ranking-only:** `als_score`, `itemcf_score`, `candidate_source` (als/itemcf/both).

Null rates >20% are documented, not silently imputed: `item_cart_rate` (80.5%) and `item_purchase_rate` (99.1%) are undefined below their activity thresholds by design; `item_category` (21.0%) reflects items with no `categoryid` property record at all before the cutoff. All left as `NaN` for XGBoost's native missing-value handling rather than an invented sentinel.

See [src/features.py](src/features.py).

## 7. Evaluation

**Protocol:** temporal split (80/10/10 by event-timestamp percentile) → train on train-only artifacts (no retraining on val/test) → val used only for alpha tuning and to build the ranker's train→val supervised labels → **test held out entirely** until Phase 4, used only for final reported metrics.

**Test-set results** ([notebooks/04_evaluation.ipynb](notebooks/04_evaluation.ipynb)):

| Metric | Baseline | Final Pipeline (deployed) | Δ% |
|---|---|---|---|
| Recall@10 | 0.0100 | 0.0144 | +43.7% |
| NDCG@10 | 0.0059 | 0.0103 | +75.8% |
| Coverage@10 | 0.0344 | 0.1505 | +337.9% |
| AddToCart-Recall@10 | 0.0098 | 0.0134 | +36.2% |
| Purchase-Recall@10 | 0.0182 | 0.0247 | +35.2% |

"Final Pipeline (deployed)" = personalized ranking for the 11,587 warm test users (7.3%) blended with the popularity fallback for everyone else — exactly what the serving system in Section 10 produces. Isolating personalization from the fallback-diluted population (warm users only, both systems evaluated on the same subset) shows a larger effect: Recall@10 +138.0%, NDCG@10 +221.5%, AddToCart-Recall@10 +132.3%, Purchase-Recall@10 +112.8%.

## 8. Failure analysis

All measured from real test-set data, not asserted:

1. **Cold-start users:** 92.7% of test users have 0 train interactions → served entirely by the popularity fallback.
2. **Sparse users:** Recall@10 rises monotonically with train activity: 0.0074 (cold) → 0.0795 (1-2 interactions) → 0.1193 (3-5) → 0.1726 (6+).
3. **Popularity bias in ALS:** 27.3% of ALS-sourced candidate *rows* fall in the train top-100 popular items, but only 2.4% of ALS's *unique* recommended items do — ALS pulls from a much wider catalog than raw popularity would suggest, though a meaningful share of volume still skews popular.
4. **Long-tail items:** Recall@10 is 0.0041 for long-tail-relevant items (<5 train interactions) vs. 0.0176 for popular ones — a 4.3x gap.
5. **Repeated recommendations:** 21.5% of personalized top-10 recs are items the user already viewed (not purchased) in train — the deliberate consequence of the Section 5 fix; purchased items are excluded by construction, viewed-not-purchased items are not.
6. **Temporal staleness:** using the item's *most recent* `available` property (unrestricted by the train cutoff — this one deliberately reflects "right now," not train-only history), 56.2% of items the final system recommends show `available=0`. A production system needs a real-time availability filter at serving time; this project does not implement one.
7. **Metric-business gap:** Recall@10/NDCG@10/the proxy recalls are all computed on historical holdout data — they measure whether the model would have surfaced items the user *actually* went on to engage with, in hindsight. That is not the same claim as a conversion-rate or revenue increase, which requires an online A/B test this project does not run.

## 9. MLOps

- **Git:** one repo, `master` branch, one commit per phase.
- **DVC** (`dvc.yaml`, 4 stages — `prepare`, `split`, `train`, `rank`): tracks only the meaningful pipeline artifacts (interactions table, splits, ALS/Item-CF/XGBoost models, candidates), not every intermediate file. `cmd` for each stage executes the actual notebook that produces it (`python -m nbconvert --to notebook --execute --inplace ...`) — the notebooks are the real source of truth in this project, not standalone scripts, so the pipeline reflects that honestly rather than duplicating logic into a parallel script layer.
- **MLflow:** two tracked runs — `popularity_baseline` and `candidate_gen_xgboost_ranking` — logging real params (ALS factors/iterations, tuned alpha, XGBoost hyperparameters) and the actual test-set metrics from Section 7. See [notebooks/05_deployment.ipynb](notebooks/05_deployment.ipynb).

## 10. Deployment

Gradio app ([app.py](app.py)) backed by [src/inference.py](src/inference.py). Everything is precomputed offline — candidate generation, XGBoost ranking, and the popularity fallback list are all built in earlier notebooks. At request time the app does a **dict lookup only** (no ALS/Item-CF/XGBoost call): known warm users get their precomputed personalized top-K; everyone else (cold test users, or any user id typed into the demo) gets the global popularity top-K.

```bash
python app.py
```

## 11. Limitations

- Offline evaluation only — no online A/B test, no measured conversion/revenue impact.
- 92.7% cold-start rate in this dataset — the vast majority of users are served by the non-personalized fallback, not the learned model.
- Long-tail items are recommended far less reliably (4.3x lower recall) than popular ones.
- No real-time inventory/availability filtering — 56.2% of recommended items show stale `available=0` status in their latest record.
- User tastes and the catalog both shift over time; this model is a static snapshot trained on a fixed historical window.

## 12. Business impact interpretation

Offline Recall@K/NDCG@K are **not** online CTR, add-to-cart rate, or revenue. They measure ranking quality on held-out historical data — whether the model would have surfaced items the user went on to interact with. AddToCart-Recall@10 and Purchase-Recall@10 are used here as behavioral *proxies* for business impact, not as business KPIs themselves. Claiming a revenue or conversion uplift would require a live A/B test (control = popularity, treatment = final pipeline, measured on add-to-cart rate / purchase rate / session depth over ~2+ weeks for statistical significance) — not run in this project.

## 13. Reproducibility

```bash
pip install -r requirements.txt
dvc repro          # re-executes notebooks 01 -> 02 -> 03 -> 03b in dependency order
python app.py       # Gradio demo (needs notebooks/04_evaluation.ipynb's outputs too)
```

Raw data (`data/raw/*.csv`) and all generated artifacts (`data/processed/`, `*.pkl`) are gitignored; DVC manages their versioning locally. Each notebook is independently re-runnable given its declared inputs.

## 14. Interview Q&A

**Why implicit feedback?** Users don't rate products. Views, carts, and purchases are naturally captured signals. Explicit ratings are sparse and biased; implicit feedback is abundant and realistic.

**Why temporal split?** Random split leaks future interactions into training, making the model appear better than it is. In production, you always train on past and predict future.

**Why popularity baseline?** It is the actual production fallback at most companies. It is harder to beat than people expect. If the model can't beat popularity, it doesn't justify deployment.

**Why ALS?** Standard, fast, well-understood method for implicit CF. Works on sparse matrices. Scales. Has a clear probabilistic interpretation. `implicit` library is production-grade.

**Why Item-CF?** Captures local behavioral similarity. High catalog coverage. Interpretable. Compensates for ALS's popularity bias.

**Why combine ALS + Item-CF for candidate generation?** Each has complementary strengths. ALS: ranking quality. Item-CF: coverage and discovery. Combining both gives the ranking stage a broader, better candidate pool than either alone — this is candidate generation, not the final decision.

**Why XGBoost?** ALS and Item-CF are strong candidate-generation methods for implicit feedback, but their final scores are primarily collaborative signals. The final recommendation decision also depends on heterogeneous behavioral features such as user-item interaction history, recency, item popularity, and category affinity. XGBoost is used as a lightweight tabular ranking layer to combine these signals and produce the final Top-K ranking.

**Why not use XGBoost directly on the entire catalog?** Candidate generation reduces the ranking problem from the full catalog to a small, relevant candidate set (~50 items), making training and inference practical.

**Why not use deep learning?** No demonstrated need. XGBoost provides a strong, interpretable tabular ranking layer with much lower complexity, and there's no evidence DL would outperform this pipeline at this dataset's scale.

**Why not use XGBoost as the baseline?** The baseline must stay intentionally simple and non-personalized so the value of personalization can be measured against it.

**How do you handle cold start?** Users with 0 train interactions fall back to global popularity top-K (92.7% of test users, measured). Users with 1-2 interactions are handled by the same pipeline but show meaningfully lower recall (0.0795 vs. 0.1726 for 6+ interactions) — the model degrades gracefully rather than failing.

**How do you avoid leakage?** Strict timestamp-based split. All features computed from train-only data, asserted in every notebook. Items purchased in train filtered from recommendations.

**What is the business metric?** Offline: Recall@10, NDCG@10, Coverage@10, AddToCart-Recall@10, Purchase-Recall@10. Online (not measured): CTR, add-to-cart rate, conversion rate.

**Can you claim conversion uplift?** No. Offline Recall@K ≠ online conversion. An A/B test is required. This project makes no revenue claim.

**How would you A/B test in production?** 50/50 split: control (popularity) vs. treatment (final pipeline). Measure add-to-cart rate, purchase rate, session depth over ~2 weeks minimum for statistical significance.

**What if the model/API fails?** Popularity fallback is always available. Inference uses precomputed artifacts — no live model call. System degrades gracefully.

**What is the biggest limitation?** Offline evaluation on historical data, with a 92.7% cold-start rate meaning the learned model only ever gets to act for a small fraction of users in this dataset.

**How would you scale it?** Distributed ALS (Spark), Faiss for Item-CF at scale, batch XGBoost scoring instead of per-request inference, Redis for precomputed final-ranking lookup, daily batch retraining, stateless API behind a load balancer. Same design, larger infrastructure.
