"""Regression tests for chronology, leakage, collection recovery, and ranking."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import fastf1
import fastf1.plotting

from src import data_collection as dc
from src.features import FEATURE_COLUMNS, RECENCY_DECAY, build_features, pre_race_features, weighted_prior_form
from src.predict import predict_race
from src.train_model import train_model, sample_weights
from src.predict import load_model


def sample_history():
    rows = []
    for season in [2022, 2023, 2024]:
        for race in range(1, 7):
            for index, driver in enumerate(["AAA", "BBB", "CCC", "DDD"]):
                position = (index + race) % 4 + 1
                rows.append({"season": season, "round": race, "event": "Demo",
                             "race_date": f"{season}-0{race}-01T12:00:00Z",
                             "driver": driver, "team": "A" if index < 2 else "B",
                             "finish_position": position, "qualifying_position": index + 1,
                             "grid_position": index + 1, "points": 5 - position,
                             "finished": 1.0, "status": "Finished", "fastest_lap_seconds": 90.0})
    return pd.DataFrame(rows)


class FeaturesTests(unittest.TestCase):
    def test_weighted_prior_form_age_and_missing_values(self):
        values = pd.Series([20., 16., np.nan, 8., 4., 99.])
        actual = weighted_prior_form(values)
        prior = values.iloc[:5].to_numpy()
        weights = np.exp(-RECENCY_DECAY * np.arange(4, -1, -1))
        known = np.isfinite(prior)
        self.assertAlmostEqual(actual.iloc[5], np.average(prior[known], weights=weights[known]))
        self.assertTrue(np.isnan(actual.iloc[0]))
        self.assertLess(actual.iloc[5], np.nanmean(prior))

    def test_season_features_reset_driver_and_team(self):
        features = build_features(sample_history())
        start = features[(features.season == 2023) & (features['round'] == 1)]
        for column in ['current_season_avg_finish_before_race', 'current_season_dnf_rate',
                       'current_season_team_avg_finish', 'current_season_team_avg_qualifying']:
            self.assertTrue(start[column].isna().all(), column)
        self.assertTrue((start.current_season_points_before_race == 0).all())
        self.assertTrue((start.team_current_season_points_before_race == 0).all())
        self.assertTrue(start.weighted_finish_form.notna().all())

    def test_current_and_future_outcomes_never_change_target_features(self):
        history = sample_history()
        original = build_features(history)
        changed = history.copy()
        mask = (changed.season == 2023) & (changed["round"] >= 3)
        changed.loc[mask, ["finish_position", "qualifying_position", "grid_position", "points"]] = 99
        changed.loc[mask, "finished"] = 0
        updated = build_features(changed)
        target = (original.season == 2023) & (original["round"] == 3)
        pd.testing.assert_frame_equal(original.loc[target, FEATURE_COLUMNS], updated.loc[target, FEATURE_COLUMNS])

    def test_inference_matches_training_and_new_driver_is_safe(self):
        history = sample_history()
        entries = history.loc[(history.season == 2023) & (history["round"] == 3), ["driver", "team"]]
        actual = build_features(history)
        actual = actual.loc[(actual.season == 2023) & (actual["round"] == 3)].sort_values("driver")
        predicted = pre_race_features(history, entries, 2023, 3).sort_values("driver")
        pd.testing.assert_frame_equal(actual[FEATURE_COLUMNS].reset_index(drop=True),
                                      predicted[FEATURE_COLUMNS].reset_index(drop=True))
        rookie = pre_race_features(history, pd.DataFrame({"driver": ["NEW"], "team": ["A"]}), 2025, 1)
        self.assertTrue(pd.isna(rookie.iloc[0].finish_mean_3))
        self.assertFalse(pd.isna(rookie.iloc[0].team_finish_mean_5))
        self.assertEqual(rookie.iloc[0].season_race_points_before, 0)

    def test_team_window_counts_races_not_driver_rows(self):
        frame = build_features(sample_history())
        target = frame.loc[(frame.season == 2022) & (frame["round"] == 2) & (frame.team == "A")]
        self.assertTrue((target.team_finish_mean_5 == 2.5).all())
        first = frame.loc[(frame.season == 2022) & (frame["round"] == 1)]
        self.assertTrue(first.finish_mean_5.isna().all())
        self.assertTrue(first.team_finish_mean_5.isna().all())

    def test_duplicate_records_rejected(self):
        history = sample_history()
        with self.assertRaises(ValueError):
            build_features(pd.concat([history, history.iloc[[0]]]))


class TrainingTests(unittest.TestCase):
    def test_sample_weights_are_race_based_and_keep_old_data(self):
        frame = sample_history()
        weights = pd.Series(sample_weights(frame), index=frame.index)
        self.assertAlmostEqual(weights.mean(), 1)
        groups = frame.assign(weight=weights).groupby(['season', 'round']).weight
        self.assertTrue((groups.nunique() == 1).all())
        self.assertTrue(groups.first().is_monotonic_increasing)
        self.assertLessEqual(weights.max() / weights.min(), 5)
        np.testing.assert_allclose(sample_weights(frame, decay=0), 1)
        with self.assertRaises(ValueError):
            sample_weights(frame, decay=-1)

    def test_pinned_holdout_excludes_later_seasons_from_fit(self):
        with tempfile.TemporaryDirectory() as folder:
            metadata = train_model(sample_history(), folder, num_boost_round=5,
                                   validation_season=2023, feature_decay=0.4)
            self.assertEqual(metadata['train_seasons'], [2022])
            self.assertEqual(metadata['production_seasons'], [2022, 2023, 2024])
            self.assertEqual(metadata['feature_recency_decay'], 0.4)
            self.assertTrue((Path(folder) / 'recent_race_predictions.csv').exists())
            result = predict_race(sample_history(), 2025, 1, model_dir=folder)
            self.assertEqual(len(result), 4)

    def test_real_xgboost_fit_save_load_and_rank(self):
        with tempfile.TemporaryDirectory() as folder:
            history = sample_history()
            metadata = train_model(history, folder, num_boost_round=10)
            self.assertEqual(metadata["train_seasons"], [2022, 2023])
            self.assertEqual(metadata["validation_season"], 2024)
            self.assertTrue(np.isfinite(metadata["metrics"]["mae"]))
            self.assertTrue((Path(folder) / "f1_xgb_model.json").exists())
            result = predict_race(history, 2025, 1, model_dir=folder)
            self.assertEqual(result["Predicted Position"].tolist(), [1, 2, 3, 4])
            self.assertEqual(result.Driver.nunique(), 4)
            self.assertTrue(result["Predicted Score"].is_monotonic_increasing)
            pd.testing.assert_frame_equal(result, predict_race(history, 2025, 1, model_dir=folder))
            with self.assertRaises(ValueError):
                predict_race(history, 2024, 5, model_dir=folder)


class CollectionTests(unittest.TestCase):
    def test_finished_and_missing_status(self):
        self.assertEqual(dc.finished_flag("+12 Laps"), 1)
        self.assertEqual(dc.finished_flag("Engine"), 0)
        self.assertTrue(np.isnan(dc.finished_flag(None)))

    def test_failure_retried_success_cached_future_excluded(self):
        schedule = pd.DataFrame({"RoundNumber": [1, 2, 3], "EventName": ["One", "Two", "Future"],
                                 "Session5": ["Race"] * 3,
                                 "Session5DateUtc": pd.to_datetime(["2023-01-01", "2023-02-01", "2030-01-01"])})
        def fake_collect(year, event, include_laps):
            if event.RoundNumber == 2:
                raise RuntimeError("Unavailable session")
            return sample_history().query('season == 2023 and round == 1')
        with tempfile.TemporaryDirectory() as folder:
            processed = Path(folder) / "processed"
            with patch.object(dc, "PROCESSED_DIR", processed), patch.object(dc, "HISTORY_PATH", processed / "history.csv"), \
                 patch.object(dc, "load_history", side_effect=lambda: pd.read_csv(processed / "history.csv") if (processed / "history.csv").exists() else pd.DataFrame(columns=dc.COLUMNS)), \
                 patch.object(dc, "enable_cache"), patch.object(dc, "get_race_schedule", return_value=schedule), \
                 patch.object(dc, "collect_race", side_effect=fake_collect) as collect:
                first, issues = dc.collect_history([2023], now="2024-01-01")
                self.assertEqual(len(first), 4)
                self.assertEqual(len(issues), 1)
                self.assertEqual(collect.call_count, 2)
                second, _ = dc.collect_history([2023], now="2024-01-01")
                self.assertEqual(len(second), 4)
                self.assertEqual(collect.call_count, 3)  # Only failed round retried.


class StreamlitTests(unittest.TestCase):
    def test_analysis_preserved_and_predictor_renders(self):
        from streamlit.testing.v1 import AppTest

        class Schedule(pd.DataFrame):
            def is_testing(self):
                return pd.Series(False, index=self.index)

        class EmptyLaps(pd.DataFrame):
            @property
            def _constructor(self):
                return EmptyLaps
            def pick_drivers(self, driver):
                return self
            def pick_accurate(self):
                return self
            def pick_wo_box(self):
                return self

        class Session:
            results = pd.DataFrame({"Abbreviation": ["AAA", "BBB"], "FullName": ["Driver A", "Driver B"]})
            laps = EmptyLaps({"LapTime": pd.Series(dtype="timedelta64[ns]"), "TyreLife": pd.Series(dtype=float)})
            weather_data = pd.DataFrame()
            def load(self):
                pass

        history = sample_history()
        schedule = Schedule({"RoundNumber": [1], "EventName": ["Demo GP"], "EventDate": [pd.Timestamp("2023-01-01")]})
        with tempfile.TemporaryDirectory() as folder:
            train_model(history, folder, num_boost_round=5)
            fitted = load_model(folder)
            with patch.object(fastf1, "get_event_schedule", return_value=schedule), \
                 patch.object(fastf1, "get_session", return_value=Session()), \
                 patch.object(fastf1.plotting, "driver_color", return_value="#ff0000", create=True), \
                 patch("src.data_collection.load_history", return_value=history), \
                 patch("src.predict.load_model", return_value=fitted), \
                 patch("src.predict.upcoming_events", return_value=schedule):
                app = AppTest.from_file(str(dc.ROOT / "app.py")).run(timeout=30)
                self.assertFalse(app.exception, [e.message for e in app.exception])
                self.assertEqual([tab.label for tab in app.tabs],
                                 ["Telemetry", "Tire Degradation", "Track Map", "Sector Times", "Weather"])
                app.sidebar.radio[0].set_value("Race Predictor").run(timeout=30)
                self.assertFalse(app.exception, [e.message for e in app.exception])
                self.assertEqual(len(app.metric), 3)
                button = next(button for button in app.button if button.label == "Predict finishing order")
                button.click().run(timeout=30)
                self.assertFalse(app.exception, [e.message for e in app.exception])
                self.assertTrue(any("ML predicted classification" in title.value for title in app.subheader))


if __name__ == "__main__":
    unittest.main()
