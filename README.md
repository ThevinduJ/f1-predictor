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


## Interface and driver profiles

The top navigation opens Race Predictor, Race Analysis (including driver
comparison), and Model Insights. Prediction results persist across theme and
page reruns; changing the race, entry list, history size, or model invalidates
the displayed result until a fresh prediction is requested. The ML modules and
saved model are unaffected by UI theme changes.

`src/theme.py` owns light/dark CSS variables, responsive layout and Matplotlib
colors. The floating sun/moon button at the bottom-right saves the theme in Streamlit session state.
`src/ui_components.py` owns reusable podium/grid/comparison cards, headers,
exact-value tables, and model-management UI. Cards expose profile details on
hover or keyboard focus; touch users can tap a card and tap outside to dismiss.

`src/driver_data.py` owns the driver-to-official-profile lookup, team accents,
metadata parsing, age calculation, and source validation. Image URLs, birth
dates and published career wins/podiums are stored centrally in the small
`src/driver_profiles.json` snapshot, with source URLs and retrieval timestamps.
These are official profile facts, not totals extrapolated from our partial
2024–2026 ML dataset. Ages are calculated from birth dates using today's date
in Australia/Sydney. Snapshot reads are cached; rendering never fetches profiles
or loads a FastF1 session. Images load lazily in the browser from F1's CDN.

To explicitly update the bounded list of driver profiles:

```powershell
python -m src.driver_data --refresh
```

The profile list includes current drivers plus a historical lookup where available.
If an official profile disappears or changes layout, refresh preserves the last
successful record. Unknown drivers get a neutral silhouette and unavailable
statistics; metadata is never fabricated. All HTML text is escaped, image URLs
are restricted to official Formula 1 hosts, and no remote scripts are injected.

Official F1 portrait URLs may change or be blocked by a browser/network policy.
A failed image cannot block predictions; card text and neutral backgrounds remain
usable. Career statistics are a dated snapshot and should be refreshed periodically.
Some canvas-based Streamlit table controls retain framework styling; the exact
prediction table uses themed HTML, and chart colors follow the selected theme.
CSS selectors for native widgets may need adjustment after a future Streamlit update.

### Race Results archive

Open **Race Results** in the top navigation to browse published races. The page
chooses the latest season with results and its newest completed race by default.
It includes the podium and full classification cards, a sortable results table,
grid-to-finish movement chart, fastest laps, constructor totals, factual race
insights, actual tyre stints, and an optional selected-driver progression chart.
The existing floating theme button also themes these cards, tables and charts.

`src/race_results.py` obtains the list of completed races through FastF1's
Ergast/Jolpica interface (winner results only). It shares a cached race session
with Race Analysis, with FastF1's persistent cache still under `data/`. Published
event lists are cached for 15 minutes; loaded sessions and processed result
bundles are cached for an hour, with bounded cache sizes. Theme changes and chart
selection reuse the processed bundle instead of loading sessions again.

The archive offers seasons back to 1950, subject to the upstream results service.
Detailed timing, stints and lap positions are generally available from 2018.
Unavailable sections are omitted or marked N/A. Grid 0 is a pit-lane start and
has no calculated positions-changed value. DNFs count retirements, including
classified retirees, but exclude disqualifications and non-starters. The pole
metric reflects grid P1, which may differ from the qualifying winner after
penalties. Race points exclude sprint points. Hover profiles use the existing
current official-site snapshot, not career statistics as of the historical race.

### Season-aware driver portraits

All driver cards require an explicit season. Predictor passes the predicted
season; Race Results passes the selected historical season; Comparison passes
the session season. The shared `get_driver_image(code, season, team=None)`
resolver in `src/driver_data.py` reads `src/driver_images.json`, keyed by season,
then driver abbreviation, with one record per historical team. FastF1's race
team remains authoritative. Multiple team records in a year require team context.
The registry includes full-season rosters and replacement drivers for 2018–2026.

Only official F1 URLs with an explicit matching season in the path are accepted.
There is no cross-season fallback: unavailable or ambiguous imagery gets the
neutral silhouette. Yearless FastF1 headshots and current profile portraits are
never used to fill historical gaps. Image failures do not affect Python app
execution; the existing silhouette/alt text remains available behind the image.
Current profile snapshots still supply birth dates and current career stats,
while season records supply the historical name, team and race number.

Refresh imagery explicitly, outside Streamlit, with:

```powershell
python -m src.refresh_driver_images --seasons 2018 2019 2020 2021 2022 2023 2024 2025 2026
```

This queries paginated FastF1 Ergast/Jolpica rosters and validates official image
URLs using HTTP HEAD; no scraping or image validation happens during reruns.
Validation timestamps are stored with each usable portrait. Local registry reads
are cached, with file modification time invalidating the cache after updates.

At the initial validation, all 2025–2026 driver/team entries had official assets.
2018 and seasons before it have no validated portraits in the registry. Missing
2019–2024 cases include the 2019 Gasly/Albon team moves, Russell's 2020 team move,
Kubica (2021), Hulkenberg/de Vries (2022), Lawson (2023), and Doohan (2024).
Placeholders are deliberate in these cases rather than using the wrong team era.
