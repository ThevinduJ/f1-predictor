"""Collect race results without downloading expensive telemetry.

FastF1's HTTP cache remains in data/. Small, versioned per-race records live
in data/processed/races/; failed/incomplete races are never marked cached.
"""
import argparse
from datetime import timedelta
import logging
import re
from pathlib import Path

import fastf1
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
HISTORY_PATH = PROCESSED_DIR / "historical_races.csv"
SCHEMA_VERSION = 1
LOGGER = logging.getLogger(__name__)
COLUMNS = ["season", "round", "event", "race_date", "driver", "team",
           "grid_position", "finish_position", "points", "status", "finished",
           "qualifying_position", "fastest_lap_seconds"]


def enable_cache():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(DATA_DIR))


def race_date(event):
    """Use the named Race session's UTC start, rather than a guessed session slot."""
    for number in range(1, 6):
        if event.get(f"Session{number}") == "Race":
            value = event.get(f"Session{number}DateUtc")
            if pd.notna(value):
                return pd.to_datetime(value, utc=True)
    # A date-only fallback is conservative: use the end of the event day.
    value = pd.to_datetime(event.get("EventDate"), utc=True)
    if pd.isna(value):
        raise ValueError("Event has no usable race date")
    return value.normalize() + pd.Timedelta(hours=23, minutes=59)


def get_race_schedule(season):
    enable_cache()
    schedule = fastf1.get_event_schedule(int(season))
    return schedule.loc[~schedule.is_testing()].copy()


def numeric(value, positive=False):
    value = pd.to_numeric(value, errors="coerce")
    if pd.isna(value) or not np.isfinite(value) or (positive and value <= 0):
        return np.nan
    return float(value)


def finished_flag(status):
    if pd.isna(status) or not str(status).strip():
        return np.nan
    # Matches FastF1 DriverResult.dnf: lapped finishers still finished.
    text = str(status)
    return float(text == "Finished" or re.fullmatch(r"\+\d+ Laps?", text) is not None)


def collect_race(season, event, include_laps=True):
    round_number = int(event["RoundNumber"])
    race = event.get_session("R") if hasattr(event, "get_session") else fastf1.get_session(int(season), round_number, "R")
    race.load(laps=include_laps, telemetry=False, weather=False, messages=False)
    results = race.results
    if results.empty or "Position" not in results:
        raise ValueError("Race results are not published")
    valid = pd.to_numeric(results["Position"], errors="coerce") > 0
    results = results.loc[valid]
    if len(results) < 2:
        raise ValueError("Insufficient classified race results")
    if "Status" not in results or results["Status"].fillna("").eq("").all():
        raise ValueError("Final race statuses are not published")

    qualifying = {}
    try:
        quali = event.get_session("Q") if hasattr(event, "get_session") else fastf1.get_session(int(season), round_number, "Q")
        quali.load(laps=False, telemetry=False, weather=False, messages=False)
        qualifying = {row.get("Abbreviation"): numeric(row.get("Position"), True)
                      for _, row in quali.results.iterrows()}
    except Exception as exc:
        LOGGER.warning("Qualifying unavailable for %s round %s: %s", season, round_number, exc)

    fastest = {}
    if include_laps:
        try:
            fastest = race.laps.groupby("Driver")["LapTime"].min().dt.total_seconds().to_dict()
        except Exception as exc:
            LOGGER.warning("Lap times unavailable: %s", exc)

    rows = []
    for _, result in results.iterrows():
        driver = result.get("Abbreviation")
        if pd.isna(driver) or not str(driver).strip():
            continue
        rows.append({
            "season": int(season), "round": round_number, "event": event["EventName"],
            "race_date": race_date(event).isoformat(), "driver": str(driver),
            "team": result.get("TeamName", "Unknown"),
            "grid_position": numeric(result.get("GridPosition"), True),
            "finish_position": numeric(result.get("Position"), True),
            "points": numeric(result.get("Points")), "status": result.get("Status"),
            "finished": finished_flag(result.get("Status")),
            "qualifying_position": qualifying.get(driver, np.nan),
            "fastest_lap_seconds": fastest.get(driver, np.nan),
        })
    if len(rows) < 2:
        raise ValueError("No usable driver identities")
    return pd.DataFrame(rows, columns=COLUMNS)


def load_history(path=HISTORY_PATH):
    if not Path(path).exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(path)


def collect_history(seasons, *, include_laps=True, refresh=False, progress=None,
                    now=None, rounds=None):
    """Merge successful races with existing history; retry failures on later runs.

    A four-hour buffer prevents treating an in-progress race as completed.
    rounds is an optional iterable useful for small collection smoke runs.
    """
    enable_cache()
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.to_datetime(now, utc=True)
    cache_dir = PROCESSED_DIR / "races"
    cache_dir.mkdir(parents=True, exist_ok=True)
    chunks, issues = [], []
    for season in sorted(set(map(int, seasons))):
        try:
            schedule = get_race_schedule(season)
        except Exception as exc:
            issues.append(f"{season} schedule: {exc}")
            continue
        for _, event in schedule.iterrows():
            try:
                number = int(event["RoundNumber"])
                start = race_date(event)
            except (ValueError, TypeError, KeyError) as exc:
                issues.append(f"{season} invalid event: {exc}")
                continue
            if (rounds is not None and number not in rounds) or (start.to_pydatetime() + timedelta(hours=4) > now.to_pydatetime()):
                continue
            description = f"{season} round {number}: {event['EventName']}"
            if progress:
                progress(description)
            path = cache_dir / f"v{SCHEMA_VERSION}_{season}_{number}_{'laps' if include_laps else 'results'}.json"
            try:
                records = pd.read_json(path) if path.exists() and not refresh else collect_race(season, event, include_laps)
                if records.empty:
                    raise ValueError("Empty cached race")
                if not path.exists() or refresh:
                    records.to_json(path, orient="records", indent=2)
                chunks.append(records)
            except Exception as exc:
                issues.append(f"{description}: {exc}")
                LOGGER.warning("Skipping %s: %s", description, exc)
    old = load_history()
    available = [chunk for chunk in [old, *chunks] if not chunk.empty]
    combined = pd.concat(available, ignore_index=True) if available else pd.DataFrame(columns=COLUMNS)
    if not combined.empty:
        combined = combined.drop_duplicates(["season", "round", "driver"], keep="last")
        combined = combined.sort_values(["season", "round", "driver"]).reset_index(drop=True)
        combined.to_csv(HISTORY_PATH, index=False)
    return combined, issues


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", type=int, nargs="+", required=True)
    parser.add_argument("--no-laps", action="store_true", help="Results only; fastest lap remains missing")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    history, issues = collect_history(args.seasons, include_laps=not args.no_laps,
                                      refresh=args.refresh, progress=print)
    print(f"Saved {len(history)} driver/race rows to {HISTORY_PATH}")
    for issue in issues:
        print(f"Skipped: {issue}")
    if history.empty:
        raise SystemExit("No usable historical races collected")


if __name__ == "__main__":
    main()
