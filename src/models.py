import lightgbm as lgb
import pandas as pd
import xgboost as xgb
from scipy.stats import rankdata
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import OUTPUT_DIR

MODEL_NAMES = ["lgb", "xgb", "hgb", "et", "lr"]
FINAL_BLEND = ["lgb", "xgb", "et", "lr"]


def load_features(extra=None):
    train = pd.read_pickle(OUTPUT_DIR / "train_features.pkl")
    test = pd.read_pickle(OUTPUT_DIR / "test_features.pkl")
    if extra is not None:
        train = train.merge(extra, left_on="id", right_index=True, how="left")
        test = test.merge(extra, left_on="id", right_index=True, how="left")
    cols = [c for c in train.columns if c not in ("id", "label")]
    return train[cols].astype(float), train["label"].values, test[cols].astype(float), test["id"]


def make_model(name, seed=0):
    if name == "lgb":
        return lgb.LGBMClassifier(n_estimators=600, learning_rate=0.01, num_leaves=15, min_child_samples=20,
                                  subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=2,
                                  random_state=seed, verbose=-1)
    if name == "xgb":
        return xgb.XGBClassifier(n_estimators=700, learning_rate=0.01, max_depth=4, subsample=0.8,
                                 colsample_bytree=0.5, min_child_weight=2, reg_lambda=2,
                                 eval_metric="auc", random_state=seed)
    if name == "hgb":
        return HistGradientBoostingClassifier(learning_rate=0.03, max_iter=400, max_leaf_nodes=15,
                                              l2_regularization=1.0, min_samples_leaf=20, random_state=seed)
    if name == "et":
        return make_pipeline(SimpleImputer(strategy="median"),
                             ExtraTreesClassifier(n_estimators=600, min_samples_leaf=3, max_features=0.3,
                                                  n_jobs=-1, random_state=seed))
    if name == "lr":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(C=0.05, max_iter=3000))
    raise ValueError(f"unknown model: {name}")


def rank_blend(predictions):
    total = sum(rankdata(p) for p in predictions)
    return (total - total.min()) / (total.max() - total.min())
