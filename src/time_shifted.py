import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from features import DATA_DIR, OUTPUT_DIR

warnings.filterwarnings("ignore")

FIRST_TARGET, LAST_TARGET = 2001, 2013


def season_tables():
    subs = pd.read_csv(DATA_DIR / "subscriptions.csv").rename(columns={"account.id": "id"})
    accounts = pd.read_csv(DATA_DIR / "account.csv", encoding="latin1")["account.id"]
    subs["yr"] = subs["season"].str[:4].astype(int)
    subs["tier"] = subs["subscription_tier"].fillna(0)
    subs["full"] = (subs["package"] == "Full").astype(int)

    def pivot(values, agg, fill=0):
        table = subs.pivot_table(index="id", columns="yr", values=values, aggfunc=agg).reindex(accounts)
        return table if fill is None else table.fillna(fill)

    return accounts, {
        "seats": pivot("no.seats", "sum"),
        "tier": pivot("tier", "sum"),
        "price": pivot("price.level", "min", fill=None),
        "full": pivot("full", "max"),
        "locations": pivot("location", "nunique"),
    }


def history_features(t, target_year, accounts):
    seats = t["seats"]
    history = [y for y in sorted(seats.columns) if y < target_year]
    subscribed = seats[history].values > 0
    years = np.array(history)

    def col(name, year, default=0):
        return t[name][year].values if year in t[name].columns else default

    f = pd.DataFrame(index=accounts)
    for k in range(1, 6):
        f[f"seats_m{k}"] = col("seats", target_year - k)
        f[f"tier_m{k}"] = col("tier", target_year - k)
    f["price_m1"] = col("price", target_year - 1, np.nan)
    f["full_m1"] = col("full", target_year - 1)
    f["locations_m1"] = col("locations", target_year - 1)

    has_history = subscribed.any(axis=1)
    last = np.where(has_history, years[np.argmax(np.where(subscribed, years, 0), axis=1)], np.nan)
    first = np.where(has_history, years[np.argmax(subscribed, axis=1)], np.nan)
    f["n_seasons"] = subscribed.sum(axis=1)
    f["gap"] = target_year - last
    f["tenure"] = target_year - first

    alive = np.ones(len(f), dtype=bool)
    run = np.zeros(len(f))
    for k in range(1, len(history) + 1):
        alive &= subscribed[:, -k]
        run += alive
    f["streak"] = run
    f["recency_w"] = (subscribed * 0.7 ** (target_year - years)).sum(axis=1)
    f["n_last5"] = subscribed[:, -5:].sum(axis=1)
    f["seat_trend"] = f["seats_m1"] - f["seats_m2"]
    return f


def main():
    accounts, tables = season_tables()

    frames = []
    for year in range(FIRST_TARGET, LAST_TARGET + 1):
        f = history_features(tables, year, accounts)
        f["label"] = (tables["seats"][year].values > 0).astype(int)
        f["target_year"] = year
        frames.append(f[f["n_seasons"] > 0])
    history = pd.concat(frames)
    cols = [c for c in history.columns if c not in ("label", "target_year")]
    print(f"{len(history)} rows, {history['label'].mean():.1%} positive")

    model = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.02, num_leaves=31,
                               min_child_samples=50, verbose=-1)
    past = history[history["target_year"] < LAST_TARGET]
    holdout = history[history["target_year"] == LAST_TARGET]
    model.fit(past[cols], past["label"])
    print(f"holdout {LAST_TARGET} AUC: {roc_auc_score(holdout['label'], model.predict_proba(holdout[cols])[:, 1]):.4f}")

    model.fit(history[cols], history["label"])
    upcoming = history_features(tables, LAST_TARGET + 1, accounts)
    pred = pd.Series(model.predict_proba(upcoming[cols])[:, 1], index=accounts, name="shifted_pred")
    pred[upcoming["n_seasons"] == 0] = 0.0
    pred.to_frame().to_pickle(OUTPUT_DIR / "shifted_pred.pkl")


if __name__ == "__main__":
    main()
