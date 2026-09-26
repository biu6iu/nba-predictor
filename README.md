# NBA Win Predictor

Predicts whether the home team wins an NBA game, using team form and prior-season strength. An XGBoost model is tuned with Bayesian optimisation, calibrated, and visualised through a Streamlit dashboard.

Data comes from [Basketball Reference](https://www.basketball-reference.com/): game results plus per-game and advanced team stats.

## Quickstart

Requires [conda](https://docs.conda.io/). From the project root:

```bash
make setup       # create the conda env 
make train       # tune, train and write artefacts/
make dashboard   # open the dashboard at http://localhost:8501
```

`make help` lists all targets. Without `make`:

```bash
conda env create -f environment.yml
conda activate nba-predictor
python train.py
streamlit run dashboard/app.py
```

The dashboard reads from `artefacts/`, so run the training step first.

## Project layout

| Path | What it does |
|---|---|
| `data/` | Input CSVs: game results (`matchData.csv`) and per-season team stats |
| `src/config.py` | Paths, season splits, feature list, tuning settings |
| `src/data_loader.py` | Loads and merges the CSVs |
| `src/preprocessor.py` | Builds the per-game features |
| `src/model.py` | Hyperparameter search, training, calibration, threshold choice |
| `src/evaluate.py` | Metrics and plots |
| `train.py` | Runs the whole pipeline and writes `artefacts/` |
| `dashboard/app.py` | Streamlit app |
| `model.ipynb` | OG notebook which this project started off as |

## Features

The model uses 28 features, mostly the difference between the home and visiting team. Everything is computed from information available before tip-off.

| Group | Examples |
|---|---|
| Recent scoring and results | average points scored/allowed, win % and point differential over the last 5 and 10 games |
| Schedule | days of rest, back-to-back games |
| Venue | home win % at home, visitor win % away |
| Team strength (previous season) | SRS, offensive/defensive/net rating, pace |
| Efficiency (previous season) | TS%, eFG%, turnover and rebound rates, free-throw and 3-point rates, and their defensive counterparts |
| Momentum | recent win-rate trend |

## Data split

Games are split by season, never randomly, so the model is always tested on games that happen after the ones it trained on.

| Season | Used for |
|---|---|
| 2020-21 | Previous-season stats for 2021-22 games only |
| 2021-22, 2022-23, 2023-24 | Training |
| 2024-25 | Validation (threshold selection) |
| 2025-26 | Test |

## Modelling notes

- Hyperparameters are tuned with Bayesian optimisation, scored by log loss over time-ordered cross-validation folds.
- Predicted probabilities are calibrated with Platt scaling.
- Rolling-window features reset each season, so roughly a quarter of rows in every split have an early-season NaN feature. These are kept rather than dropped and handled by XGBoost's native missing-value support, since real predictions happen in early season too.
- The win/loss threshold is chosen to reach at least 65% precision while maximising recall.
- `artefacts/model.pkl` stores the model together with its threshold, tuned parameters, feature list and library versions. The dashboard warns if the installed versions differ.

## Results

Test numbers are from the 2025-26 season, which was never used for tuning, calibration or threshold selection. The baseline always predicts the majority class (home win) using that season's home-win rate.

| Metric | Baseline | Model |
|---|---|---|
| Accuracy | 55.5% | 65.0% |
| Log loss | 0.687 | 0.619 |
| Brier score | 0.247 | 0.215 |
| ROC-AUC | n/a | 0.712 |
| Precision / recall | n/a | 65.6% / 77.5% |

The model beats the baseline on every metric, by about 9.5 points of accuracy. The 65% precision target was met on both the 2024-25 validation season (65.0%, accuracy 65.5%) and the test season (65.6%, accuracy 65.0%), so the threshold chosen on validation carries over to unseen games.

All metrics are written to `artefacts/metrics.json` by `make train`, so rerunning training after changing features or data will change these numbers.

## Development

```bash
conda activate nba-predictor
pip install -r requirements-dev.txt
make lint
```
