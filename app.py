import fastf1
import streamlit as st
import matplotlib

# Streamlit renders figures in the browser; no Windows/Tk GUI backend is needed.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
(PROJECT_DIR / 'data').mkdir(exist_ok=True)
fastf1.Cache.enable_cache(str(PROJECT_DIR / 'data'))

from datetime import datetime
from zoneinfo import ZoneInfo

import fastf1.plotting
import numpy as np
import pandas as pd

from src.data_collection import collect_history, load_history
from src.predict import latest_entrants, load_model, predict_race, upcoming_events
from src.train_model import train_model
from src.train_model import SAMPLE_RECENCY_DECAY
from src.features import RECENCY_DECAY


st.set_page_config(page_title="F1 Predictor & Analysis", page_icon="🏎️", layout="wide")
fastf1.plotting.setup_mpl()
plt.style.use("dark_background")

# FastF1 3.8 renamed driver_color; retain the requested API on newer versions.
if not hasattr(fastf1.plotting, "driver_color"):
    fastf1.plotting.driver_color = fastf1.plotting.get_driver_color


@st.cache_data(ttl=3600, show_spinner=False)
def get_schedule(year):
    schedule = fastf1.get_event_schedule(year)
    return schedule.loc[~schedule.is_testing()].copy()


@st.cache_resource(ttl=3600, max_entries=4, show_spinner=False)
def load_session(year, round_number):
    session = fastf1.get_session(year, round_number, "R")
    session.load()
    return session


def show_plot(fig):
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def fastest_lap(session, driver):
    laps = session.laps.pick_drivers(driver).dropna(subset=["LapTime"])
    if laps.empty:
        return None
    return laps.pick_fastest()


def driver_color(driver, session, fallback):
    try:
        return fastf1.plotting.driver_color(driver, session=session)
    except (KeyError, ValueError):
        return fallback


def predictor_training(today):
    with st.expander("Collect historical data and train", expanded=False):
        st.caption("First download can take several minutes. Successful races are cached. "
                   "Training holds out the latest collected season, then refits on all history.")
        seasons = st.multiselect("Historical seasons", list(range(today.year, 2017, -1)),
                                 default=[today.year - 3, today.year - 2, today.year - 1])
        include_laps = st.checkbox("Also collect fastest lap times (slower)", value=False)
        available_history = load_history()
        historical_seasons = sorted(available_history["season"].dropna().astype(int).unique()) if not available_history.empty else []
        validation_season = None
        if len(historical_seasons) >= 2:
            try:
                _, saved = load_model()
                preferred = saved["validation_season"]
            except (FileNotFoundError, ValueError):
                preferred = historical_seasons[-1]
            candidates = historical_seasons[1:]
            validation_season = st.selectbox("Validation season", candidates,
                index=candidates.index(preferred) if preferred in candidates else len(candidates) - 1)
        feature_decay = st.number_input("Feature decay per prior race", min_value=0.0,
                                       value=RECENCY_DECAY, step=0.05)
        sample_decay = st.number_input("Training weight decay per race", min_value=0.0,
                                      value=SAMPLE_RECENCY_DECAY, step=0.005, format="%.3f")
        st.caption("Decay changes take effect after retraining. Later seasons are excluded from holdout fitting, "
                   "but included in the final production refit.")
        if st.button("Collect / update history"):
            if not seasons:
                st.error("Select at least one historical season.")
            else:
                status = st.empty()
                with st.spinner("Collecting completed races…"):
                    history, issues = collect_history(seasons, include_laps=include_laps,
                                                       progress=status.caption)
                status.empty()
                st.success(f"Stored {len(history)} driver/race records.")
                if issues:
                    st.warning(f"Skipped {len(issues)} unavailable races/schedules; they can be retried.")
                    st.text("\n".join(issues))
        if st.button("Train XGBoost model"):
            try:
                with st.spinner("Training and evaluating…"):
                    train_model(load_history(), validation_season=validation_season,
                                sample_decay=sample_decay, feature_decay=feature_decay)
                st.success("Model and evaluation saved.")
            except (ValueError, OSError) as exc:
                st.error(str(exc))


def model_evaluation(metadata):
    metrics = metadata["metrics"]
    st.caption(f"Holdout season: {metadata['validation_season']} · "
               f"{metadata['validation_races']} races · {metadata['validation_rows']} driver results")
    left, middle, right = st.columns(3)
    left.metric("Score MAE", f"{metrics['mae']:.2f} positions")
    middle.metric("Score RMSE", f"{metrics['rmse']:.2f} positions")
    right.metric("Rank MAE", f"{metrics['rank_mae']:.2f} positions")
    with st.expander("Evaluation and feature importance"):
        st.write(metadata["validation_protocol"])
        st.write(f"Previous-five-race mean baseline MAE: {metrics['baseline_mae']:.2f}")
        if "comparison" in metadata:
            comparison = metadata["comparison"]
            st.write("Same-split comparison against the original features and unweighted training:")
            st.dataframe(pd.DataFrame({name: comparison[name] for name in ("old", "new")}).T,
                         width="stretch")
            st.caption(f"New form features receive {metadata['recent_feature_total_gain_share']:.1%} of production split gain. "
                       f"Feature decay: {metadata['feature_recency_decay']:.3f}; training decay: {metadata['sample_recency_decay']:.3f}.")
        importance = pd.Series(metadata["feature_importance_gain"], name="Gain").sort_values(ascending=False)
        st.dataframe(importance.to_frame(), width="stretch")
        st.caption("Gain describes the fitted model's split importance; it is not causal evidence.")


def predictor_page():
    st.title("Race Predictor")
    st.caption("ML prediction: XGBoost scores sorted into a finishing order. Lower score means "
               "a better expected finish; scores are not probabilities.")
    today = datetime.now(ZoneInfo("Australia/Sydney"))
    predictor_training(today)
    history = load_history()
    if history.empty:
        st.info("Collect historical races using the panel above, then train a model.")
        return
    try:
        _, metadata = load_model()
    except (FileNotFoundError, ValueError) as exc:
        st.info(str(exc))
        return
    model_evaluation(metadata)
    boundary = metadata["trained_through"]
    st.caption(f"Production model trained through {boundary['season']} round {boundary['round']}.")
    season = st.selectbox("Prediction season", [today.year, today.year + 1])
    try:
        events = upcoming_events(season)
    except Exception as exc:
        st.error(f"Could not load upcoming events: {exc}")
        return
    if events.empty:
        st.info("No upcoming races have been published for this season.")
        return
    names = dict(zip(events["RoundNumber"].astype(int), events["EventName"]))
    round_number = st.selectbox("Upcoming race", list(names), format_func=lambda number: names[number])
    try:
        roster = latest_entrants(history, season, round_number)
    except ValueError as exc:
        st.error(str(exc))
        return
    latest = history.sort_values(["season", "round"]).iloc[-1]
    st.caption(f"History available through {int(latest['season'])} round {int(latest['round'])}. "
               "Collect the current season to update driver form before predicting.")
    st.warning("The entry list below assumes the latest completed race roster. "
               "Confirm drivers and teams for the selected race; add/remove replacements as needed.")
    entries = st.data_editor(roster, num_rows="dynamic", hide_index=True, width="stretch",
                             key=f"entries_{season}_{round_number}_{len(history)}")
    if st.button("Predict finishing order", type="primary"):
        try:
            result = predict_race(history, season, round_number, entrants=entries)
        except (ValueError, OSError) as exc:
            st.error(str(exc))
            return
        st.subheader(f"ML predicted classification · {season} {names[round_number]}")
        st.dataframe(result.style.format({"Predicted Score": "{:.2f}"}), hide_index=True, width="stretch")
        st.caption("Deterministic ranking for the supplied entry list. No uncertainty estimates are calculated. "
                   "Historical race points exclude sprint points and championship penalties.")
        st.download_button("Download prediction CSV", result.to_csv(index=False),
                           file_name=f"prediction_{season}_{round_number}.csv", mime="text/csv", on_click="ignore")


def analysis_page():
    st.title("F1 Driver Comparison")
    st.caption("Compare race telemetry, tyre life, fastest-lap sectors, and weather.")

    today = datetime.now(ZoneInfo("Australia/Sydney"))
    with st.sidebar:
        st.header("Race selection")
        year = st.selectbox("Year", list(range(today.year, 2017, -1)))
        try:
            with st.spinner("Fetching race schedule…"):
                schedule = get_schedule(year)
        except Exception as exc:
            st.error(f"Could not fetch the schedule: {exc}")
            st.stop()
        if schedule.empty:
            st.info("No races are available for this year.")
            st.stop()

        # Start with the latest past event while still allowing any race selection.
        event_dates = pd.to_datetime(schedule["EventDate"], utc=True)
        past = schedule.loc[event_dates < pd.Timestamp(today)]
        rounds = schedule["RoundNumber"].astype(int).tolist()
        default_round = int(past.iloc[-1]["RoundNumber"]) if not past.empty else rounds[0]
        race_names = dict(zip(rounds, schedule["EventName"]))
        round_number = st.selectbox(
            "Race", rounds, index=rounds.index(default_round),
            format_func=lambda number: race_names[number],
        )
        try:
            with st.spinner("Loading race session and telemetry…"):
                session = load_session(year, round_number)
        except Exception as exc:
            st.error(f"Could not load this race session: {exc}")
            st.caption("A future race or a race without published data cannot be loaded yet.")
            st.stop()

        results = session.results.dropna(subset=["Abbreviation"])
        drivers = results["Abbreviation"].drop_duplicates().tolist()
        if not drivers:
            st.info("This session has no driver results yet.")
            st.stop()
        names = results.set_index("Abbreviation")["FullName"].to_dict()
        label = lambda driver: f"{driver} — {names.get(driver, driver)}"
        driver1 = st.selectbox("Driver 1", drivers, format_func=label)
        driver2 = st.selectbox("Driver 2", drivers, index=min(1, len(drivers) - 1), format_func=label)

    color1 = driver_color(driver1, session, "#00d2be")
    color2 = driver_color(driver2, session, "#ff8700")
    styles = ["-", "--" if color1 == color2 else "-"]
    selections = list(zip([driver1, driver2], [color1, color2], styles))
    if driver1 == driver2:
        st.info("Both selections are the same driver. Select another driver to compare.")
    st.subheader(f"{year} {race_names[round_number]} · Race")
    st.caption(f"{driver1}: solid line · {driver2}: {'dashed' if styles[1] == '--' else 'solid'} line")

    lap_data = []
    for driver, color, style in selections:
        lap = fastest_lap(session, driver)
        telemetry = None
        if lap is not None:
            try:
                telemetry = lap.get_telemetry().add_distance()
            except Exception as exc:
                st.warning(f"Fastest-lap telemetry is unavailable for {driver}: {exc}")
        lap_data.append((driver, color, style, lap, telemetry))

    telemetry_tab, tyres_tab, track_tab, sectors_tab, weather_tab = st.tabs(
        ["Telemetry", "Tire Degradation", "Track Map", "Sector Times", "Weather"]
    )

    with telemetry_tab:
        st.caption("Fastest race laps, aligned by distance from the start line.")
        channels = [("Speed", "Speed (km/h)"), ("Throttle", "Throttle (%)"),
                    ("Brake", "Brake (0/1)"), ("RPM", "RPM"), ("nGear", "Gear")]
        available = [item for item in lap_data if item[4] is not None and not item[4].empty]
        if not available:
            st.info("No fastest-lap telemetry is available for these drivers.")
        else:
            fig, axes = plt.subplots(5, 1, sharex=True, figsize=(12, 12))
            for ax, (channel, ylabel) in zip(axes, channels):
                for driver, color, style, _, telemetry in available:
                    if channel in telemetry and "Distance" in telemetry:
                        ax.plot(telemetry["Distance"], telemetry[channel].astype(float),
                                color=color, linestyle=style, label=driver,
                                drawstyle="steps-post" if channel in ("Brake", "nGear") else "default")
                ax.set_ylabel(ylabel)
                ax.grid(alpha=0.2)
                if ax.lines:
                    ax.legend()
            axes[-1].set_xlabel("Distance (m)")
            show_plot(fig)

    with tyres_tab:
        st.caption("Clean race laps; pit-in, pit-out, and inaccurate laps are excluded. "
                   "Linear trends summarize tyre age and lap time, including fuel and traffic effects.")
        fig, ax = plt.subplots(figsize=(12, 6))
        plotted = False
        for driver, color, style in selections:
            laps = session.laps.pick_drivers(driver).pick_accurate().pick_wo_box().copy()
            laps["LapSeconds"] = laps["LapTime"].dt.total_seconds()
            laps = laps.dropna(subset=["TyreLife", "LapSeconds"])
            laps = laps.loc[np.isfinite(laps["TyreLife"]) & np.isfinite(laps["LapSeconds"])]
            if laps.empty:
                st.info(f"No clean tyre-life laps are available for {driver}.")
                continue
            plotted = True
            ax.scatter(laps["TyreLife"], laps["LapSeconds"], color=color, alpha=0.6,
                       marker="o" if style == "-" else "x", label=f"{driver} laps")
            if len(laps) >= 2 and laps["TyreLife"].nunique() >= 2:
                coefficients = np.polyfit(laps["TyreLife"], laps["LapSeconds"], 1)
                age = np.linspace(laps["TyreLife"].min(), laps["TyreLife"].max(), 100)
                ax.plot(age, np.polyval(coefficients, age), color=color, linestyle=style,
                        label=f"{driver} trend")
            else:
                st.info(f"{driver} needs at least two distinct tyre ages for a trendline.")
        ax.set(xlabel="TyreLife (laps)", ylabel="LapTime (seconds)")
        ax.grid(alpha=0.2)
        if plotted:
            ax.legend()
            show_plot(fig)
        else:
            plt.close(fig)

    with track_tab:
        fig, ax = plt.subplots(figsize=(10, 8))
        for driver, color, style, _, telemetry in lap_data:
            if telemetry is not None and {"X", "Y"}.issubset(telemetry.columns):
                coordinates = telemetry.dropna(subset=["X", "Y"])
                if not coordinates.empty:
                    ax.plot(coordinates["X"], coordinates["Y"], color=color,
                            linestyle=style, label=driver)
        if ax.lines:
            ax.set_aspect("equal", adjustable="datalim")
            ax.set(xlabel="X (telemetry coordinates)", ylabel="Y (telemetry coordinates)")
            ax.legend()
            show_plot(fig)
        else:
            plt.close(fig)
            st.info("No position telemetry is available for these fastest laps.")

    with sectors_tab:
        st.caption("Sector times from each driver's fastest race lap, in seconds.")
        rows = []
        for driver, _, _, lap, _ in lap_data:
            row = {"Driver": driver}
            for sector in range(1, 4):
                value = lap[f"Sector{sector}Time"] if lap is not None else pd.NaT
                row[f"Sector {sector}"] = value.total_seconds() if pd.notna(value) else np.nan
            rows.append(row)
        st.dataframe(pd.DataFrame(rows).set_index("Driver").style.format("{:.3f}", na_rep="Unavailable"),
                     width="stretch")

    with weather_tab:
        weather = session.weather_data
        if weather is None or weather.empty:
            st.info("No weather data is available for this session.")
        else:
            fig, ax = plt.subplots(figsize=(12, 5))
            elapsed = weather["Time"].dt.total_seconds() / 60
            ax.plot(elapsed, weather["TrackTemp"], color="#ff8700", label="TrackTemp")
            ax.plot(elapsed, weather["AirTemp"], color="#56b4e9", label="AirTemp")
            ax.set(xlabel="Time since session start (minutes)", ylabel="Temperature (°C)")
            ax.grid(alpha=0.2)
            ax.legend()
            show_plot(fig)


def main():
    page = st.sidebar.radio("Page", ["Race Analysis", "Race Predictor"])
    if page == "Race Predictor":
        predictor_page()
    else:
        analysis_page()


if __name__ == "__main__":
    main()
