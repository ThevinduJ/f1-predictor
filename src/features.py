"""Identical shifted feature construction for training and inference."""
import numpy as np
import pandas as pd

LEGACY_FEATURE_COLUMNS = [
    "finish_mean_3", "finish_mean_5", "qualifying_mean_3", "qualifying_mean_5",
    "points_sum_3", "points_sum_5", "dnf_rate_5", "grid_mean_5",
    "previous_finish", "team_finish_mean_5", "team_qualifying_mean_5",
    "season_race_points_before", "prior_driver_races", "round", "field_size",
]
RECENCY_DECAY = 0.30
RECENT_FEATURE_COLUMNS = [
    "weighted_finish_form", "weighted_qualifying_form", "weighted_points_form",
    "weighted_grid_form", "weighted_dnf_form", "previous_race_finish",
    "previous_race_qualifying", "previous_race_points", "previous_2_race_avg_finish",
    "previous_2_race_avg_qualifying", "previous_3_race_avg_finish",
    "previous_3_race_avg_qualifying", "current_season_avg_finish_before_race",
    "current_season_avg_qualifying_before_race", "current_season_points_before_race",
    "current_season_dnf_rate", "current_season_team_avg_finish",
    "current_season_team_avg_qualifying", "team_previous_race_avg_finish",
    "team_previous_2_avg_finish", "team_previous_3_avg_finish",
    "team_weighted_recent_finish", "team_weighted_recent_qualifying",
    "team_current_season_points_before_race",
]
FEATURE_COLUMNS = LEGACY_FEATURE_COLUMNS + RECENT_FEATURE_COLUMNS
FEATURE_VERSION = 2


def weighted_prior_form(series, decay=RECENCY_DECAY):
    """Decay by appearance age, retaining missing slots and renormalizing known values."""
    def average(values):
        weights = np.exp(-decay * np.arange(len(values) - 1, -1, -1))
        valid = np.isfinite(values)
        return np.average(values[valid], weights=weights[valid]) if valid.any() else np.nan
    return series.shift(1).rolling(5, min_periods=1).apply(average, raw=True)


def build_features(history, recency_decay=RECENCY_DECAY):
    """One row per driver/race. Every outcome feature shifts before rolling.

    Team histories first collapse both teammates into one row per event, so
    one teammate's current result cannot leak into the other's features.
    Rolling windows span seasons; race-only season points reset each season.
    """
    if not np.isfinite(recency_decay) or recency_decay < 0:
        raise ValueError("Feature recency decay must be finite and nonnegative")
    required = {"season", "round", "driver", "team"}
    if not required.issubset(history):
        raise ValueError(f"Missing columns: {sorted(required - set(history.columns))}")
    data = history.copy()
    if data.duplicated(["season", "round", "driver"]).any():
        raise ValueError("Duplicate driver/race records")
    for column in ["finish_position", "qualifying_position", "points", "finished", "grid_position"]:
        data[column] = pd.to_numeric(data.get(column, np.nan), errors="coerce")
    data = data.sort_values(["season", "round", "driver"]).reset_index(drop=True)
    drivers = data.groupby("driver", sort=False)
    for window in (3, 5):
        for source, prefix in [("finish_position", "finish"), ("qualifying_position", "qualifying")]:
            data[f"{prefix}_mean_{window}"] = drivers[source].transform(
                lambda series: series.shift(1).rolling(window, min_periods=1).mean())
        data[f"points_sum_{window}"] = drivers["points"].transform(
            lambda series: series.shift(1).rolling(window, min_periods=1).sum())
    data["dnf"] = 1 - data["finished"]
    data["dnf_rate_5"] = data.groupby("driver")["dnf"].transform(
        lambda series: series.shift(1).rolling(5, min_periods=1).mean())
    data["grid_mean_5"] = drivers["grid_position"].transform(
        lambda series: series.shift(1).rolling(5, min_periods=1).mean())
    data["previous_finish"] = drivers["finish_position"].shift(1)
    data["prior_driver_races"] = drivers.cumcount()
    data["season_race_points_before"] = data.groupby(["season", "driver"])["points"].transform(
        lambda series: series.fillna(0).cumsum().shift(1, fill_value=0))
    data["field_size"] = data.groupby(["season", "round"])["driver"].transform("size")

    for source, target in [("finish_position", "finish"), ("qualifying_position", "qualifying"),
                           ("points", "points"), ("grid_position", "grid"), ("dnf", "dnf")]:
        data[f"weighted_{target}_form"] = data.groupby("driver")[source].transform(
            lambda series: weighted_prior_form(series, recency_decay))
    for source, target in [("finish_position", "finish"), ("qualifying_position", "qualifying"), ("points", "points")]:
        data[f"previous_race_{target}"] = drivers[source].shift(1)
    for window in (2, 3):
        for source, target in [("finish_position", "finish"), ("qualifying_position", "qualifying")]:
            data[f"previous_{window}_race_avg_{target}"] = drivers[source].transform(
                lambda series: series.shift(1).rolling(window, min_periods=1).mean())
    for source, target in [("finish_position", "current_season_avg_finish_before_race"),
                           ("qualifying_position", "current_season_avg_qualifying_before_race"),
                           ("dnf", "current_season_dnf_rate")]:
        data[target] = data.groupby(["season", "driver"])[source].transform(
            lambda series: series.shift(1).expanding(min_periods=1).mean())
    data["current_season_points_before_race"] = data["season_race_points_before"]

    teams = data.groupby(["season", "round", "team"], as_index=False, dropna=False)[
        ["finish_position", "qualifying_position"]].mean()
    teams = teams.sort_values(["season", "round", "team"])
    team_points = data.groupby(["season", "round", "team"], as_index=False, dropna=False)["points"].sum(min_count=1)
    teams = teams.merge(team_points, on=["season", "round", "team"], validate="one_to_one")
    for source, target in [("finish_position", "team_finish_mean_5"),
                           ("qualifying_position", "team_qualifying_mean_5")]:
        teams[target] = teams.groupby("team", dropna=False)[source].transform(
            lambda series: series.shift(1).rolling(5, min_periods=1).mean())
    for window in (1, 2, 3):
        name = "team_previous_race_avg_finish" if window == 1 else f"team_previous_{window}_avg_finish"
        teams[name] = teams.groupby("team", dropna=False)["finish_position"].transform(
            lambda series: series.shift(1).rolling(window, min_periods=1).mean())
    for source, suffix in [("finish_position", "finish"), ("qualifying_position", "qualifying")]:
        teams[f"team_weighted_recent_{suffix}"] = teams.groupby("team", dropna=False)[source].transform(
            lambda series: weighted_prior_form(series, recency_decay))
        teams[f"current_season_team_avg_{suffix}"] = teams.groupby(["season", "team"], dropna=False)[source].transform(
            lambda series: series.shift(1).expanding(min_periods=1).mean())
    teams["team_current_season_points_before_race"] = teams.groupby(["season", "team"], dropna=False)["points"].transform(
        lambda series: series.fillna(0).cumsum().shift(1, fill_value=0))
    team_features = [column for column in FEATURE_COLUMNS if column.startswith("team_") or column.startswith("current_season_team_")]
    data = data.merge(teams[["season", "round", "team", *team_features]],
                      on=["season", "round", "team"], validate="many_to_one")
    data[FEATURE_COLUMNS] = data[FEATURE_COLUMNS].astype(float).replace([np.inf, -np.inf], np.nan)
    return data


def pre_race_features(history, entrants, season, round_number, recency_decay=RECENCY_DECAY):
    """Append outcome-free entry rows after discarding the target and all future races."""
    if not {"driver", "team"}.issubset(entrants):
        raise ValueError("Entrants need driver and team columns")
    past = history.loc[(history["season"] < season) |
                       ((history["season"] == season) & (history["round"] < round_number))].copy()
    entry = entrants[["driver", "team"]].copy()
    entry["driver"] = entry["driver"].astype("string").str.strip().str.upper()
    if entry.empty or entry["driver"].isna().any() or entry["driver"].duplicated().any():
        raise ValueError("Entrants must contain unique, nonempty driver abbreviations")
    if entry["driver"].astype(str).str.strip().eq("").any():
        raise ValueError("Driver abbreviations cannot be blank")
    entry["team"] = entry["team"].fillna("Unknown")
    entry["season"], entry["round"] = int(season), int(round_number)
    features = build_features(pd.concat([past, entry], ignore_index=True), recency_decay)
    return features.loc[(features["season"] == season) & (features["round"] == round_number)].copy()
