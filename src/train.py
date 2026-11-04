import warnings

import numpy as np
import pandas as pd

from features import DATA_DIR, OUTPUT_DIR
from models import FINAL_BLEND, load_features, make_model, rank_blend

warnings.filterwarnings("ignore")

N_SEEDS = 5


def main():
    X, y, X_test, test_ids = load_features()

    predictions = []
    for name in FINAL_BLEND:
        seed_preds = []
        for seed in range(N_SEEDS):
            model = make_model(name, seed)
            model.fit(X, y)
            seed_preds.append(model.predict_proba(X_test)[:, 1])
        predictions.append(np.mean(seed_preds, axis=0))
        print(f"trained {name}")

    submission = pd.DataFrame({"ID": test_ids, "Predicted": rank_blend(predictions)})

    sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
    assert list(submission["ID"]) == list(sample["ID"])
    assert submission["Predicted"].between(0, 1).all()

    submission.to_csv(OUTPUT_DIR / "submission.csv", index=False)
    print(f"wrote {OUTPUT_DIR / 'submission.csv'} ({len(submission)} rows)")


if __name__ == "__main__":
    main()
