#!/usr/bin/env python3
"""Train the MerRec next-month purchase model scored by 27_propensity_xgboost.sql.

The input is the Parquet result of 26_propensity_training_features.sql: one row per
shopper with the ten log-scaled behavior features of the August 1 cohort and the
label for a purchase in the following 30 days.
"""

import argparse
from pathlib import Path

import pyarrow.parquet as pq
import xgboost

FEATURES = ["views", "likes", "intents", "buys", "gmv",
            "views_30d", "likes_30d", "intents_30d", "buys_30d", "mean_price"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("features", type=Path, help="26_propensity_training_features result Parquet")
    parser.add_argument("model", type=Path, help="output XGBoost JSON model")
    args = parser.parse_args()
    table = pq.read_table(args.features, columns=FEATURES + ["label"])
    x = table.select(FEATURES).to_pandas().astype("float32").to_numpy()
    y = table.column("label").to_numpy()
    model = xgboost.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                                  tree_method="hist", random_state=42)
    model.fit(x, y)
    args.model.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(args.model)
    print(f"trained on {len(y)} rows with positive rate {y.mean():.4f}; wrote {args.model}")


if __name__ == "__main__":
    main()
