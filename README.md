# F1 Predictor

Streamlit race analysis plus an initial historical-data XGBoost predictor. The
existing five analysis tabs remain on **Race Analysis**. Select **Race Predictor**
in the sidebar for collection, training, evaluation, and upcoming classifications.

## Run on Windows

From the project directory in PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m streamlit run app.py
```

Activation is optional if you invoke `.\venv\Scripts\python.exe` directly.

## Collect and train

```powershell
python -m src.data_collection --seasons 2023 2024 2025 --no-laps
python -m src.train_model
```

Omit `--no-laps` to also collect fastest race lap times. These are stored for
analysis and are **not model inputs**. Neither mode downloads telemetry. Use
`--refresh` to refresh processed race records; FastF1 still uses its HTTP cache.
Failed races are logged, skipped, and retried on the next collection run.
Only events whose race start is more than four hours in the past are attempted;
published positions and final statuses must also be available.

Update current-season form before forecasting:

```powershell
python -m src.data_collection --seasons 2026 --no-laps
```

Retraining after that update is optional. If retrained, the newest collected
season (possibly partial) becomes the holdout. To maintain a full 2025 holdout,
train on 2023–2025 first, then update 2026 history without retraining.

The UI provides the same collection and training buttons. First collection
requires internet access and may take several minutes. Successful per-race
records and a combined CSV are stored under `data/processed/`. Caches, generated
CSVs, per-race records, models, and virtual environments are ignored by Git.
Small manually reviewed fixtures elsewhere inside `data/processed/` can be tracked.

## Features and leakage rules

For each driver, features include previous 3/5 race mean finishes, previous 3/5
qualifying means, previous 3/5 race points sums, previous 5 DNF rate, previous 5
grid mean, previous finish, race-only season points before the race, prior race
count, event round, and current entry count. Team features are previous 5
team-race mean finishes and qualifying positions.

All outcome histories are shifted before rolling. Team aggregation happens
**before** shifting, preventing teammate leakage. Current grid, current
qualifying, race lap times, status, points, and finishing results are not direct
inputs. Inference uses the same feature builder with outcome-free target rows
and removes all target/future results first. Missing values remain NaN for
XGBoost's native handling, without filling from future observations.

Rolling windows count previous collected driver appearances/team races and
span seasons; season points reset each year. Missing races produce incomplete
histories. Race-only season points exclude sprints, championship penalties,
and results absent from the collected data; they are not official standings.

## Model and evaluation

The native XGBoost regressor learns numerical finishing position. Lower predicted
scores indicate better expected performance. Sorting scores yields unique P1
through Pn for every supplied driver, with alphabetical tie breaks. Scores are
not probabilities and do not imply calibrated uncertainty.

By default the latest collected season is held out in its entirety; older seasons fit the
evaluation model. Validation features can use actual results from earlier
validation races, matching forecasts made sequentially before each race. No
random split or hyperparameter tuning against the holdout is used. Metrics:
score MAE/RMSE, ranked-position MAE, and a previous-five-races mean baseline MAE.
The production model is then refitted on all collected seasons. Evaluation
metrics describe the separate held-out model, not a second test of that refit.

Outputs: `models/f1_xgb_model.json` and `models/metadata.json` (feature schema,
metrics, gain importance, training settings, and cutoff). Forecasts at or before
the training cutoff are rejected, preventing misleading in-sample backtests.

## Predict

Choose an upcoming scheduled event in **Race Predictor**, confirm/edit the
driver/team list, and click **Predict finishing order**. The initial list is
the latest historical race roster, not an authoritative future entry list.
It can include obsolete drivers if you have not updated history. The table
contains numerical predicted position, driver, team, and score and supports CSV export.

For CLI use, provide an upcoming season and round:

```powershell
python -m src.predict --season 2026 --round ROUND_NUMBER
```

Replace `ROUND_NUMBER` with an upcoming round from the published schedule.
Optionally add `--entrants path\to\entrants.csv` with `driver,team` columns.

## Verify

```powershell
python -m unittest discover -s tests -v
```

Tests use synthetic data to check leakage, feature parity, cache reuse,
failure recovery, XGBoost fitting/serialization, and deterministic ranking.
Synthetic evaluations are not production metrics.

## Recency-aware model (feature schema v2)

The model retains every original feature and adds five exponentially weighted
driver means (finish, qualifying, points, grid, DNF), previous-race finish /
qualifying / points, previous-two and previous-three finish / qualifying means,
and current-season-only driver and team form. Team features also include previous
one/two/three race means, weighted finish/qualifying, and season race points.
Season averages reset to missing at season start; season points reset to zero.
All features use strictly prior results, including teammates' prior results.

`src/features.py:RECENCY_DECAY = 0.30` gives unnormalized newest-to-oldest weights
`1.000, 0.741, 0.549, 0.407, 0.301` over five prior appearances. Means normalize
over available values; missing values retain their age slots. Team windows count
team races, not individual driver rows. Training uses native XGBoost DMatrix sample
weights: `max(0.20, exp(-0.025 * race_age))`, normalized to mean one. Age counts
distinct races in the fitting subset; all drivers in a race receive the same
weight. A race 24 events old retains about 55% of the latest race's weight and
one 48 events old retains about 30%, before normalization.

Reproduce the original 2025 holdout comparison while refitting production on all
available history (including 2026):

```powershell
python -m src.train_model --validation-season 2025 --feature-decay 0.30 --sample-decay 0.025
```

The predictor's training panel exposes these settings. Changing feature decay
requires retraining; inference always uses the decay saved with its model.
Do not choose settings by repeatedly optimizing against the held-out year.

Training saves old/new metrics and improving-form cohort results in
`models/recency_comparison.json`, all holdout classifications in
`models/validation_predictions.csv`, and frozen-model later-season classifications
in `models/recent_race_predictions.csv` when later seasons exist. Both models in
that later backtest fit only through the holdout year. Production uses a separate
refit on all available history, and cannot be used to backtest races it fitted.
Previous production model/metadata are copied to `models/previous_*` on training.
Gain, total gain, split count, feature coverage, and group importance are saved.
Correlated duplicate features may have zero individual split importance.

See [the recency validation report](reports/recency_validation.md) for full recent
race rankings and observed improvements/regressions. No driver-specific bonuses,
standings bonuses, or post-prediction rank adjustments are used.

## Limits and next improvements

This is a form-based baseline: no circuit features, pre-race qualifying/grid,
weather forecast, sprint standings, explicit retirement model, or calibrated
uncertainty. Tyre strategy and telemetry remain analysis tools. New drivers
have sparse personal history; team names that change create new team histories.
Upcoming rosters require confirmation, and missing races/qualifying reduce quality.

Next: collect more complete seasons, normalize team identities, incorporate
verified future entries and sprint points, add walk-forward multi-season
backtests and ranking baselines, then introduce circuit-specific history and
available qualifying/grid with an explicit forecast-time cutoff.
