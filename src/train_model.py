"""Train an XGBoost regressor with an entire latest-season holdout."""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from .data_collection import HISTORY_PATH, ROOT, load_history
from .features import (FEATURE_COLUMNS, FEATURE_VERSION, LEGACY_FEATURE_COLUMNS,
                       RECENT_FEATURE_COLUMNS, RECENCY_DECAY, build_features)

MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "f1_xgb_model.json"
METADATA_PATH = MODEL_DIR / "metadata.json"
PARAMS = {"objective": "reg:squarederror", "eval_metric": "rmse", "max_depth": 3,
          "eta": 0.05, "subsample": 1.0, "colsample_bytree": 1.0,
          "seed": 42, "nthread": 2, "tree_method": "hist", "lambda": 5.0}
SAMPLE_RECENCY_DECAY = 0.025
MIN_SAMPLE_WEIGHT = 0.20


def sample_weights(frame, decay=SAMPLE_RECENCY_DECAY):
    """Age in distinct races within the fitting subset; same weight for every entrant.

    Never use holdout dates to choose the weight reference. Normalize to mean 1
    so changing decay does not inadvertently rescale regularization strength.
    """
    if not np.isfinite(decay) or decay < 0:
        raise ValueError("Sample decay must be finite and nonnegative")
    races = frame[["season", "round"]].drop_duplicates().sort_values(["season", "round"])
    age = {tuple(row): len(races) - 1 - i for i, row in enumerate(races.to_numpy())}
    weights = np.array([max(MIN_SAMPLE_WEIGHT, np.exp(-decay * age[tuple(row)]))
                        for row in frame[["season", "round"]].to_numpy()])
    return weights / weights.mean()


def matrix(frame, labels=False, columns=FEATURE_COLUMNS, weights=None):
    return xgb.DMatrix(frame[columns],
                       label=frame["finish_position"] if labels else None, weight=weights)


def errors(actual, predicted):
    residual = np.asarray(actual) - np.asarray(predicted)
    return {"mae": float(np.abs(residual).mean()), "rmse": float(np.sqrt((residual ** 2).mean()))}


def ranked_positions(frame, scores):
    ranked = frame[["season", "round", "driver"]].copy()
    ranked["score"] = scores
    # Deterministic alphabetical tie break, also used by live predictions.
    ranked = ranked.sort_values(["season", "round", "score", "driver"])
    ranked["rank"] = ranked.groupby(["season", "round"]).cumcount() + 1
    return ranked["rank"].reindex(frame.index).to_numpy()


def evaluate_pair(train, test, num_boost_round, sample_decay):
    """Frozen old/new fits on identical rows, with no holdout parameter tuning."""
    old = xgb.train(PARAMS, matrix(train, True, LEGACY_FEATURE_COLUMNS), num_boost_round=num_boost_round)
    new = xgb.train(PARAMS, matrix(train, True, weights=sample_weights(train, sample_decay)), num_boost_round=num_boost_round)
    comparison = test[["season", "round", "event", "driver", "team", "finish_position",
                       "finish_mean_3", "finish_mean_5"]].copy()
    metrics = {}
    for name, model, columns in [("old", old, LEGACY_FEATURE_COLUMNS), ("new", new, FEATURE_COLUMNS)]:
        scores = model.predict(matrix(test, columns=columns))
        ranks = ranked_positions(test, scores)
        comparison[f"{name}_score"] = scores
        comparison[f"{name}_rank"] = ranks
        metrics[name] = errors(test["finish_position"], scores)
        metrics[name]["rank_mae"] = errors(test["finish_position"], ranks)["mae"]
    comparison["positions_gained"] = comparison["old_rank"] - comparison["new_rank"]
    comparison["improving_prior_form"] = comparison["finish_mean_3"] < comparison["finish_mean_5"] - 1
    cohort = comparison.loc[comparison["improving_prior_form"]]
    metrics["improving_form_cohort"] = {"rows": len(cohort)}
    if not cohort.empty:
        for name in ("old", "new"):
            metrics["improving_form_cohort"][f"{name}_rank_mae"] = errors(cohort["finish_position"], cohort[f"{name}_rank"])["mae"]
    return new, metrics, comparison


def train_model(history, model_dir=MODEL_DIR, num_boost_round=150, *,
                validation_season=None, sample_decay=SAMPLE_RECENCY_DECAY,
                feature_decay=RECENCY_DECAY):
    frame = build_features(history, feature_decay)
    frame = frame.loc[frame["finish_position"].notna() & (frame["finish_position"] > 0)].copy()
    seasons = sorted(frame["season"].unique())
    if len(seasons) < 2:
        raise ValueError("Collect at least two seasons for a chronological season holdout")
    validation_season = int(seasons[-1]) if validation_season is None else int(validation_season)
    if validation_season not in seasons:
        raise ValueError("Validation season is not in the collected data")
    train = frame.loc[frame["season"] < validation_season]
    test = frame.loc[frame["season"] == validation_season]
    if train.groupby(["season", "round"]).ngroups < 2 or test["round"].nunique() < 2:
        raise ValueError("Need at least two training races and two validation races")
    validation_model, comparison_metrics, comparison = evaluate_pair(train, test, num_boost_round, sample_decay)
    scores = validation_model.predict(matrix(test))
    metrics = errors(test["finish_position"], scores)
    metrics["rank_mae"] = errors(test["finish_position"], ranked_positions(test, scores))["mae"]
    baseline = test["finish_mean_5"].fillna(train["finish_position"].mean())
    metrics["baseline_mae"] = errors(test["finish_position"], baseline)["mae"]

    # Evaluate once, then refit the production model on all collected history.
    model = xgb.train(PARAMS, matrix(frame, True, weights=sample_weights(frame, sample_decay)), num_boost_round=num_boost_round)
    importance = model.get_score(importance_type="gain")
    last = frame.sort_values(["season", "round"]).iloc[-1]
    metadata = {
        "feature_version": FEATURE_VERSION, "feature_columns": FEATURE_COLUMNS,
        "target": "finish_position", "lower_score_is_better": True,
        "train_seasons": [int(s) for s in seasons if s < validation_season], "validation_season": validation_season,
        "train_rows": len(train), "validation_rows": len(test),
        "train_races": train.groupby(["season", "round"]).ngroups,
        "validation_races": test["round"].nunique(), "metrics": metrics,
        "feature_importance_gain": {name: float(importance.get(name, 0)) for name in FEATURE_COLUMNS},
        "production_seasons": [int(s) for s in seasons],
        "trained_through": {"season": int(last["season"]), "round": int(last["round"])},
        "trained_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "xgboost_version": xgb.__version__, "parameters": PARAMS,
        "num_boost_round": num_boost_round,
        "validation_protocol": "Seasons strictly before holdout fit; holdout evaluated sequentially using prior results. No parameter tuning on holdout. Production model refit on all history, including later seasons when available.",
        "points_scope": "Grand Prix race points only; sprint points and penalties excluded.",
        "feature_recency_decay": feature_decay, "sample_recency_decay": sample_decay,
        "sample_weight_floor": MIN_SAMPLE_WEIGHT,
        "sample_weight_protocol": "exp(-decay * age_in_distinct_training_races), floor 0.20, normalized to mean 1",
        "comparison": comparison_metrics,
    }
    directory = Path(model_dir)
    directory.mkdir(parents=True, exist_ok=True)
    # Preserve the previous production model for review, before replacing it.
    for filename in [MODEL_PATH.name, METADATA_PATH.name]:
        existing = directory / filename
        if existing.exists():
            shutil.copy2(existing, directory / f"previous_{filename}")
    comparison.to_csv(directory / "validation_predictions.csv", index=False)
    report = {"validation_season": validation_season, "holdout": comparison_metrics}
    later = frame.loc[frame["season"] > validation_season]
    if not later.empty:
        _, later_metrics, later_predictions = evaluate_pair(
            frame.loc[frame["season"] <= validation_season], later, num_boost_round, sample_decay)
        later_predictions.to_csv(directory / "recent_race_predictions.csv", index=False)
        report["later_seasons_frozen_backtest"] = later_metrics
        report["later_backtest_protocol"] = "Both models frozen through holdout season; later race outcomes excluded from fitting, earlier later-season outcomes allowed only in features."
    total_gain = model.get_score(importance_type="total_gain")
    gain_sum = sum(total_gain.values())
    share = sum(total_gain.get(column, 0) for column in RECENT_FEATURE_COLUMNS) / gain_sum if gain_sum else 0
    metadata["recent_feature_total_gain_share"] = share
    metadata["feature_importance_total_gain"] = {name: float(total_gain.get(name, 0)) for name in FEATURE_COLUMNS}
    metadata["feature_importance_split_count"] = {name: float(model.get_score(importance_type="weight").get(name, 0)) for name in FEATURE_COLUMNS}
    report["recent_feature_total_gain_share"] = share
    report["importance_note"] = "Correlated aliases can receive zero individual importance. Assess recent features collectively; importance is not causal or a guarantee of accuracy."
    if share < 0.10:
        report["importance_investigation"] = "Recent features receive less than 10% of split gain; inspect missing coverage and correlated legacy features before changing model capacity."
    coverage = {column: float(train[column].notna().mean()) for column in RECENT_FEATURE_COLUMNS}
    report["recent_feature_training_coverage"] = coverage
    (directory / "recency_comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    model.save_model(directory / MODEL_PATH.name)
    (directory / METADATA_PATH.name).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=HISTORY_PATH)
    parser.add_argument("--validation-season", type=int, help="Pin a holdout for comparable evaluations")
    parser.add_argument("--sample-decay", type=float, default=SAMPLE_RECENCY_DECAY)
    parser.add_argument("--feature-decay", type=float, default=RECENCY_DECAY)
    args = parser.parse_args()
    history = load_history(args.data)
    if history.empty:
        raise SystemExit("No history found. Run python -m src.data_collection --seasons 2023 2024 2025")
    metadata = train_model(history, validation_season=args.validation_season,
                           sample_decay=args.sample_decay, feature_decay=args.feature_decay)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
