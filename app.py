import fastf1
import streamlit as st
import matplotlib.pyplot as plt

fastf1.Cache.enable_cache('./data')

from datetime import datetime
from zoneinfo import ZoneInfo

import fastf1.plotting
import numpy as np
import pandas as pd


st.set_page_config(page_title="F1 Driver Comparison", page_icon="🏎️", layout="wide")
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


def main():
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


if __name__ == "__main__":
    main()
