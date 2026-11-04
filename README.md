# Classical Music Meets Classical ML

My code for the Fall 2026 Kaggle competition *Classical Music Meets Classical ML*. The goal is to predict which past patrons of a classical orchestra will buy a subscription for the 2014-15 season. Submissions are scored with AUROC.

## Results

| | AUROC |
|---|---|
| Baseline (just "did they subscribe last season") | 0.920 |
| Final blend, cross-validated | 0.976 |
| Kaggle public leaderboard | 0.972 |

## Project structure

```
KaggleCompetitionSabrina/
├── data/                  competition CSVs (not included)
├── notebooks/
│   └── classical_music_ml.ipynb   the full pipeline with charts and results
├── outputs/               features, predictions and submission.csv end up here
├── src/
│   ├── features.py        turns the raw CSVs into one row of features per account
│   ├── models.py          model definitions and the rank-average blend
│   ├── cross_validate.py  compares models with repeated 5-fold CV
│   ├── train.py           trains the final blend and writes the submission
│   └── time_shifted.py    an extra experiment that didn't make it into the final model
├── requirements.txt
└── README.md
```

## Running it

Download the data from the competition page and put the CSV files in `data/`.

The easiest way to see everything is the notebook in `notebooks/`. It goes through the data, the features, model comparison and the final submission, and it already has the outputs saved.

To run it as scripts instead, from the project folder:

```bash
pip install -r requirements.txt

python src/features.py
python src/cross_validate.py
python src/train.py
```

`outputs/submission.csv` is the file to upload to Kaggle. Cross-validation takes a minute or two.

To try the time-shifted experiment:

```bash
python src/time_shifted.py
python src/cross_validate.py --with-shifted
```

## Approach

The biggest pattern in the data is that people who subscribed in 2013-14 renewed about 82% of the time, compared with under 1% of everyone else. So most of the features describe each patron's subscription history:

- how recently they last subscribed
- how many seasons in a row
- how many seats and what tier
- which package, venue and section they bought most recently

On top of that there are features for single-ticket purchases, donations, and location. For location, I used distance to the San Francisco and Berkeley venues and average income in the patron's zip code. Altogether there are 104 features.

I compared five models with stratified 5-fold cross-validation repeated three times:

| Model | CV AUROC |
|---|---|
| XGBoost | 0.9758 |
| Logistic regression | 0.9750 |
| LightGBM | 0.9750 |
| ExtraTrees | 0.9728 |
| HistGradientBoosting | 0.9718 |

The final submission averages the ranks from LightGBM, XGBoost, ExtraTrees and logistic regression. Each model is trained on all the training data with 5 different seeds. Ranks are averaged instead of probabilities because AUROC only depends on ordering. Mixing in logistic regression and ExtraTrees worked better than using LightGBM and XGBoost alone, since the two boosting models make very similar predictions.

Two things I tried that didn't help:

- training a renewal model on earlier seasons for all accounts (`time_shifted.py`)
- trimming down to the top 25-60 features

## Notes

The competition data isn't included here because of Kaggle's rules. Scores can shift slightly with different library versions.
