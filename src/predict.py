"""Produce a classification by sorting pre-race XGBoost scores."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from .data_collection import get_race_schedule, load_history, race_date
from .features import FEATURE_COLUMNS, FEATURE_VERSION, RECENCY_DECAY, pre_race_features
from .train_model import MODEL_DIR, MODEL_PATH, METADATA_PATH


def load_model(model_dir=MODEL_DIR):
    directory = Path(model_dir)
    if not (directory / MODEL_PATH.name).exists() or not (directory / METADATA_PATH.name).exists():
        raise FileNotFoundError("No trained model found. Collect history and train the model first.")
    metadata = json.loads((directory / METADATA_PATH.name).read_text(encoding="utf-8"))
    if metadata.get("feature_version") != FEATURE_VERSION or metadata.get("feature_columns") != FEATURE_COLUMNS:
        raise ValueError("Feature schema changed; retrain the model")
    model = xgb.Booster()
    model.load_model(directory / MODEL_PATH.name)
    return model, metadata


def history_before(history, season, round_number):
    return history.loc[(history["season"] < season) |
                       ((history["season"] == season) & (history["round"] < round_number))].copy()


def latest_entrants(history, season, round_number):
    """Latest race roster is an explicit editable assumption, not an official entry list."""
    past = history_before(history, season, round_number)
    if past.empty:
        raise ValueError("No prior race roster; supply entrants with driver and team columns")
    last = past.sort_values(["season", "round"]).iloc[-1]
    return past.loc[(past["season"] == last["season"]) & (past["round"] == last["round"]),
                    ["driver", "team"]].sort_values("driver").reset_index(drop=True)


def predict_race(history, season, round_number, entrants=None, model_dir=MODEL_DIR):
    model, metadata = load_model(model_dir)
    boundary = metadata["trained_through"]
    if (int(season), int(round_number)) <= (boundary["season"], boundary["round"]):
        raise ValueError("Target must be after the model's training cutoff. Historical backtests require an earlier model.")
    past = history_before(history, int(season), int(round_number))
    if past.empty:
        raise ValueError("No historical data available before the selected race")
    # Guard accidental imports of results dated in the future.
    if "race_date" in past:
        dates = pd.to_datetime(past["race_date"], utc=True, errors="coerce")
        past = past.loc[dates.notna() & (dates < pd.Timestamp.now(tz="UTC"))]
    if past.empty:
        raise ValueError("No dated, completed history is available for prediction")
    if entrants is None:
        entrants = latest_entrants(past, int(season), int(round_number))
    features = pre_race_features(past, entrants, int(season), int(round_number),
                                metadata.get("feature_recency_decay", RECENCY_DECAY))
    scores = model.predict(xgb.DMatrix(features[metadata["feature_columns"]]))
    if not np.isfinite(scores).all():
        raise ValueError("Model produced invalid scores")
    result = features[["driver", "team"]].rename(columns={"driver": "Driver", "team": "Team"})
    result["Predicted Score"] = scores
    result = result.sort_values(["Predicted Score", "Driver"]).reset_index(drop=True)
    result.insert(0, "Predicted Position", np.arange(1, len(result) + 1))
    return result


def upcoming_events(season, now=None):
    schedule = get_race_schedule(season)
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.to_datetime(now, utc=True)
    return schedule.loc[schedule.apply(race_date, axis=1) > now].copy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", required=True, type=int)
    parser.add_argument("--round", dest="round_number", required=True, type=int)
    parser.add_argument("--entrants", type=Path, help="CSV with driver,team columns")
    args = parser.parse_args()
    events = upcoming_events(args.season)
    if args.round_number not in events["RoundNumber"].values:
        raise SystemExit("Select a future race from the season schedule")
    entries = pd.read_csv(args.entrants) if args.entrants else None
    print(predict_race(load_history(), args.season, args.round_number, entries).to_string(index=False))


if __name__ == "__main__":
    main()
