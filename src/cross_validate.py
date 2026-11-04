import argparse
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

from features import OUTPUT_DIR
from models import MODEL_NAMES, load_features, make_model, rank_blend

warnings.filterwarnings("ignore")

BLENDS = [
    ("lgb", "xgb"),
    ("lgb", "xgb", "hgb"),
    ("lgb", "xgb", "et"),
    ("lgb", "xgb", "et", "lr"),
    tuple(MODEL_NAMES),
]


def main(use_shifted):
    extra = pd.read_pickle(OUTPUT_DIR / "shifted_pred.pkl") if use_shifted else None
    X, y, _, _ = load_features(extra)

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
    oof = {name: np.zeros(len(y)) for name in MODEL_NAMES}
    counts = np.zeros(len(y))

    for train_idx, valid_idx in cv.split(X, y):
        counts[valid_idx] += 1
        for name in MODEL_NAMES:
            model = make_model(name)
            model.fit(X.iloc[train_idx], y[train_idx])
            oof[name][valid_idx] += model.predict_proba(X.iloc[valid_idx])[:, 1]

    print("Single models")
    for name in MODEL_NAMES:
        oof[name] /= counts
        print(f"  {name:4s} {roc_auc_score(y, oof[name]):.5f}")

    print("Rank-average blends")
    for blend in BLENDS:
        score = roc_auc_score(y, rank_blend([oof[m] for m in blend]))
        print(f"  {' + '.join(blend):28s} {score:.5f}")

    print(f"Baseline (2013-14 seats only): {roc_auc_score(y, X['sub_seats_2013']):.5f}")
    pd.to_pickle(oof, OUTPUT_DIR / "oof_predictions.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-shifted", action="store_true")
    main(parser.parse_args().with_shifted)
