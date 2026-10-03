# Recency validation comparison

Original features/unweighted fit versus schema v2/weighted fit, using identical XGBoost settings (150 trees, depth 3, eta 0.05, seed 42). No driver-specific biases. Decays were fixed conservatively before evaluation; no holdout tuning.

## Same 2025 holdout

Both evaluation models fit 2024 only (479 driver rows, 24 races). Both evaluate the same 479 rows across all 24 races of 2025. Production is separately refitted on all 1,288 collected rows through 2026 round 15; it is not used for these historical comparisons.

| Metric | Old | New | New minus old |
| --- | --- | --- | --- |
| mae | 3.7647 | 3.6575 | -0.1072 |
| rmse | 4.7390 | 4.6090 | -0.1300 |
| rank_mae | 4.2171 | 3.9791 | -0.2380 |

## Frozen 2026 backtest

Both models fit through 2025 only, then score the same 330 driver rows from 15 races of 2026. Previous 2026 outcomes may enter later race features, but no 2026 labels enter either fit. These results are additional evidence, not the original holdout.

| Metric | Old | New |
| --- | --- | --- |
| mae | 4.0514 | 4.0695 |
| rmse | 5.2746 | 5.3098 |
| rank_mae | 4.1697 | 4.2970 |

Overall 2026 accuracy worsened slightly. More responsiveness does not guarantee every improving driver will be promoted or every ranking will be more accurate.

## Drivers with improving prior form

This cohort is selected without the current result: previous-three mean finish is more than one position better than previous-five mean finish. It measures recent change, not driver identity.

| Evaluation | Rows | Old rank MAE | New rank MAE |
| --- | --- | --- | --- |
| 2025 | 124 | 4.9597 | 4.6774 |
| 2026 | 71 | 4.7183 | 4.5352 |

Ranking error improved for this cohort in both periods, despite the slight aggregate 2026 regression.

## Importance

New form features account for 65.72% of production total split gain. This includes current-season and team features. This is strong model use, not proof of causation.

| Feature | Total gain | Gain share |
| --- | --- | --- |
| current_season_avg_qualifying_before_race | 83124.3 | 28.80% |
| points_sum_5 | 45350.3 | 15.71% |
| weighted_points_form | 22665.7 | 7.85% |
| current_season_team_avg_finish | 18104.7 | 6.27% |
| team_weighted_recent_qualifying | 12223.2 | 4.24% |
| team_qualifying_mean_5 | 10999.9 | 3.81% |
| prior_driver_races | 8559.8 | 2.97% |
| grid_mean_5 | 8037.6 | 2.78% |
| team_finish_mean_5 | 6883.1 | 2.38% |
| team_previous_3_avg_finish | 6561.2 | 2.27% |

Some explicit requested names duplicate retained legacy features (previous finish, three-race means, season points). Trees may credit one alias and give the other zero gain. New-feature training coverage is 95–100%; no missing-data collapse was observed.

## Full rankings for the three latest completed races

Each row shows a strictly pre-race forecast. Positive positions gained means the new model ranks the driver higher. Actual finish is shown only for evaluation. No post-result rank adjustment is applied.

### 2026 round 13 — Italian Grand Prix

| Driver | Old rank | New rank | Actual finish | Positions gained | Improving prior form |
| --- | --- | --- | --- | --- | --- |
| ANT | 1 | 1 | 1 | +0 | Yes |
| RUS | 4 | 2 | 2 | +2 | No |
| HAM | 3 | 3 | 6 | +0 | No |
| LEC | 2 | 4 | 22 | -2 | No |
| NOR | 6 | 5 | 4 | +1 | No |
| VER | 5 | 6 | 3 | -1 | No |
| PIA | 7 | 7 | 5 | +0 | No |
| HUL | 9 | 8 | 12 | +1 | Yes |
| GAS | 10 | 9 | 7 | +1 | No |
| LAW | 8 | 10 | 14 | -2 | No |
| COL | 13 | 11 | 9 | +2 | No |
| OCO | 17 | 12 | 16 | +5 | No |
| TSU | 14 | 13 | 10 | +1 | Yes |
| BOR | 11 | 14 | 11 | -3 | No |
| LIN | 12 | 15 | 8 | -3 | No |
| SAI | 15 | 16 | 13 | -1 | No |
| BEA | 18 | 17 | 15 | +1 | No |
| STR | 19 | 18 | 20 | +1 | No |
| ALO | 16 | 19 | 21 | -3 | Yes |
| PER | 21 | 20 | 18 | +1 | No |
| BOT | 22 | 21 | 19 | +1 | No |
| ALB | 20 | 22 | 17 | -2 | Yes |

### 2026 round 14 — Spanish Grand Prix

| Driver | Old rank | New rank | Actual finish | Positions gained | Improving prior form |
| --- | --- | --- | --- | --- | --- |
| ANT | 1 | 1 | 1 | +0 | Yes |
| HAM | 5 | 2 | 22 | +3 | No |
| LEC | 7 | 3 | 4 | +4 | No |
| NOR | 4 | 4 | 3 | +0 | Yes |
| VER | 3 | 5 | 2 | -2 | No |
| PIA | 6 | 6 | 8 | +0 | No |
| RUS | 2 | 7 | 5 | -5 | Yes |
| GAS | 10 | 8 | 12 | +2 | No |
| HUL | 11 | 9 | 10 | +2 | Yes |
| LAW | 8 | 10 | 6 | -2 | No |
| COL | 12 | 11 | 7 | +1 | No |
| BEA | 18 | 12 | 16 | +6 | No |
| OCO | 15 | 13 | 11 | +2 | No |
| TSU | 14 | 14 | 14 | +0 | No |
| LIN | 9 | 15 | 9 | -6 | No |
| SAI | 16 | 16 | 19 | +0 | No |
| BOR | 13 | 17 | 13 | -4 | No |
| ALO | 17 | 18 | 17 | -1 | Yes |
| STR | 19 | 19 | 21 | +0 | No |
| BOT | 21 | 20 | 18 | +1 | No |
| PER | 22 | 21 | 20 | +1 | No |
| ALB | 20 | 22 | 15 | -2 | No |

### 2026 round 15 — Azerbaijan Grand Prix

| Driver | Old rank | New rank | Actual finish | Positions gained | Improving prior form |
| --- | --- | --- | --- | --- | --- |
| ANT | 1 | 1 | 5 | +0 | No |
| NOR | 3 | 2 | 19 | +1 | No |
| VER | 4 | 3 | 2 | +1 | No |
| PIA | 5 | 4 | 13 | +1 | Yes |
| HAM | 6 | 5 | 6 | +1 | No |
| RUS | 2 | 6 | 1 | -4 | Yes |
| LEC | 7 | 7 | 4 | +0 | No |
| HAD | 8 | 8 | 3 | +0 | No |
| BOR | 11 | 9 | 15 | +2 | No |
| LAW | 9 | 10 | 12 | -1 | No |
| OCO | 17 | 11 | 8 | +6 | No |
| HUL | 13 | 12 | 11 | +1 | No |
| COL | 10 | 13 | 17 | -3 | No |
| GAS | 14 | 14 | 18 | +0 | No |
| SAI | 15 | 15 | 10 | +0 | No |
| BEA | 19 | 16 | 9 | +3 | No |
| ALO | 16 | 17 | 21 | -1 | No |
| LIN | 12 | 18 | 7 | -6 | No |
| PER | 20 | 19 | 14 | +1 | Yes |
| STR | 21 | 20 | 22 | +1 | No |
| ALB | 18 | 21 | 20 | -3 | No |
| BOT | 22 | 22 | 16 | +0 | No |

## Reproduce

```powershell
python -m src.train_model --validation-season 2025 --feature-decay 0.30 --sample-decay 0.025
python -m unittest discover -s tests -v
```

Feature weights: exp(-0.30 * appearance age), normalized over known prior values. Training weights: exp(-0.025 * training race age), floored at 0.20 and normalized to mean 1. Both decay settings are saved with the model. Current-season points remain Grand Prix-only; sprint points and penalties are excluded.
