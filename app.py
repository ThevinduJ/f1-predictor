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

from src.data_collection import load_history, HISTORY_PATH
from src.predict import latest_entrants, load_model, predict_race, upcoming_events
from src.theme import apply_theme, styled_dataframe
from src.race_results import load_race_session as load_session
from src.race_results_page import race_results_page
from src.ui_components import (render_header, render_floating_theme_toggle, section_header, event_header, render_prediction_cards,
                               render_prediction_table, render_comparison, predictor_training,
                               model_evaluation, footer)


st.set_page_config(page_title="F1 Race Predictor", page_icon="🏎️", layout="wide", initial_sidebar_state="collapsed")
fastf1.plotting.setup_mpl()
plt.style.use("dark_background")

# FastF1 3.8 renamed driver_color; retain the requested API on newer versions.
if not hasattr(fastf1.plotting, "driver_color"):
    fastf1.plotting.driver_color = fastf1.plotting.get_driver_color


@st.cache_data(ttl=3600, show_spinner=False)
def get_schedule(year):
    schedule = fastf1.get_event_schedule(year)
    return schedule.loc[~schedule.is_testing()].copy()


@st.cache_data(ttl=900, show_spinner=False)
def upcoming_schedule(year):
    return upcoming_events(year)


# ---------------------------------------------------------------------------
# Pit-wall telemetry skin: UI-only overlay injected after the base theme.
# ---------------------------------------------------------------------------
PIT_WALL_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;600;700&family=JetBrains+Mono:wght@400;600&family=Inter:wght@400;500;600&display=swap');
:root {--neon:#00F0FF;--neon-red:#FF2D55;--neon-violet:#B14CFF;--glass:rgba(18,24,38,.55);--glass-edge:rgba(120,200,255,.16);--mono:'JetBrains Mono',ui-monospace,monospace;--display:'Rajdhani','Inter',sans-serif}
.stApp, [data-testid="stAppViewContainer"] {
  background:
    radial-gradient(1100px 600px at 12% -10%, rgba(0,240,255,.10), transparent 60%),
    radial-gradient(900px 600px at 95% 0%, rgba(255,45,85,.10), transparent 55%),
    radial-gradient(800px 500px at 50% 120%, rgba(177,76,255,.08), transparent 60%),
    linear-gradient(rgba(255,255,255,.025) 1px, transparent 1px) 0 0/48px 48px,
    linear-gradient(90deg, rgba(255,255,255,.025) 1px, transparent 1px) 0 0/48px 48px,
    #070A12 !important;
  font-family:'Inter',sans-serif}
[data-testid="stHeader"] {background:transparent !important}
[data-testid="stMainBlockContainer"] {padding-top:3.2rem}
.stApp h1,.stApp h2,.stApp h3,.brand-title,.event-card h2 {font-family:var(--display) !important;text-transform:uppercase;letter-spacing:.04em !important}
.stApp h2 {font-size:1.7rem !important}
.brand {padding:16px 22px;border:1px solid var(--glass-edge);border-radius:16px;background:linear-gradient(100deg,rgba(0,240,255,.08),rgba(255,45,85,.07)),var(--glass);backdrop-filter:blur(14px);-webkit-backdrop-filter:blur(14px);box-shadow:0 0 0 1px rgba(0,0,0,.4),0 10px 40px rgba(0,240,255,.07);position:relative;overflow:hidden}
.brand::after {content:"";position:absolute;left:0;right:0;bottom:0;height:2px;background:linear-gradient(90deg,var(--neon-red),var(--neon),var(--neon-violet));box-shadow:0 0 14px var(--neon)}
.brand-mark {background:linear-gradient(135deg,var(--neon-red),#ff6a3d);box-shadow:0 0 22px rgba(255,45,85,.55);font-family:var(--display)}
.brand-title {font-size:1.9rem !important;font-weight:700;background:linear-gradient(90deg,#fff,#9fe9ff);-webkit-background-clip:text;background-clip:text;color:transparent !important}
.brand-subtitle {font-family:var(--mono);font-size:.7rem !important;letter-spacing:.14em;text-transform:uppercase;color:#7f93ad !important}
.status-strip {display:flex;flex-wrap:wrap;gap:10px 22px;margin:12px 4px 4px;font-family:var(--mono);font-size:.68rem;letter-spacing:.12em;text-transform:uppercase;color:#7f93ad}
.status-strip b {color:var(--neon);font-weight:600}
.status-strip .live::before {content:"";display:inline-block;width:7px;height:7px;border-radius:50%;background:#2dff9a;box-shadow:0 0 10px #2dff9a;margin-right:8px;animation:pulse 1.6s infinite}
@keyframes pulse {50% {opacity:.25}}
.stApp [data-testid="stMain"] [data-testid="stRadio"] div[role="radiogroup"] {gap:6px;padding:6px;border:1px solid var(--glass-edge);border-radius:14px;background:var(--glass);backdrop-filter:blur(12px);width:fit-content;max-width:100%;flex-wrap:wrap}
.stApp [data-testid="stMain"] [data-testid="stRadio"] label {border:1px solid transparent;background:transparent;border-radius:10px;padding:8px 18px;font-family:var(--display);font-weight:600;letter-spacing:.1em;text-transform:uppercase;transition:all .18s}
.stApp [data-testid="stMain"] [data-testid="stRadio"] label:hover {background:rgba(0,240,255,.07)}
.stApp [data-testid="stMain"] [data-testid="stRadio"] label:has(input:checked) {background:linear-gradient(135deg,rgba(0,240,255,.18),rgba(177,76,255,.18));border-color:rgba(0,240,255,.55);box-shadow:0 0 18px rgba(0,240,255,.25),inset 0 0 12px rgba(0,240,255,.08)}
.stApp [data-testid="stMain"] [data-testid="stRadio"] label:has(input:checked) p {color:#fff !important;text-shadow:0 0 10px var(--neon)}
.eyebrow {font-family:var(--mono);color:var(--neon) !important;text-shadow:0 0 12px rgba(0,240,255,.6)}
.eyebrow::before {content:"// "}
.section-heading {padding-left:14px;border-left:3px solid var(--neon-red);box-shadow:-8px 0 20px -10px var(--neon-red)}
.stApp [data-testid="stVerticalBlockBorderWrapper"],.stApp [data-testid="stExpander"],.stApp [data-testid="stMetric"],.table-wrap,.event-card {
  background:var(--glass) !important;border:1px solid var(--glass-edge) !important;border-radius:16px !important;backdrop-filter:blur(14px);-webkit-backdrop-filter:blur(14px);box-shadow:0 10px 36px rgba(0,0,0,.45),inset 0 1px 0 rgba(255,255,255,.05) !important}
.stApp [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stVerticalBlockBorderWrapper"] {box-shadow:none !important;background:rgba(255,255,255,.02) !important}
.stApp [data-testid="stVerticalBlockBorderWrapper"] {padding:.4rem .6rem}
.event-card {border-left:3px solid var(--neon-red) !important;background:linear-gradient(100deg,rgba(255,45,85,.10),transparent 55%),var(--glass) !important}
.event-card h2 {font-size:2.2rem !important;text-shadow:0 0 24px rgba(0,240,255,.25)}
.event-details {font-family:var(--mono);font-size:.72rem !important;letter-spacing:.1em;text-transform:uppercase}
.stApp [data-testid="stMetric"] {position:relative;overflow:hidden}
.stApp [data-testid="stMetric"]::before {content:"";position:absolute;left:0;top:0;bottom:0;width:3px;background:linear-gradient(var(--neon),var(--neon-violet));box-shadow:0 0 12px var(--neon)}
.stApp [data-testid="stMetricLabel"] p {font-family:var(--mono);font-size:.68rem !important;letter-spacing:.14em;text-transform:uppercase;color:#7f93ad !important}
.stApp [data-testid="stMetricValue"] {font-family:var(--mono);font-weight:600;color:#fff;text-shadow:0 0 16px rgba(0,240,255,.35)}
.stApp [data-baseweb="select"] > div,.stApp input,.stApp [data-baseweb="input"],.stApp [data-baseweb="base-input"],.stApp [data-testid="stSelectbox"] [role="group"] {background:rgba(8,12,22,.75) !important;border-color:rgba(0,240,255,.22) !important;font-family:var(--mono);font-size:.85rem}
.stApp [data-baseweb="select"] > div:hover {border-color:var(--neon) !important;box-shadow:0 0 14px rgba(0,240,255,.2)}
.stApp [data-testid="stWidgetLabel"] p {font-family:var(--mono);font-size:.66rem !important;letter-spacing:.16em;text-transform:uppercase;color:#7f93ad !important}
.stApp [data-testid="stButton"] button,.stApp [data-testid="stDownloadButton"] button {background:rgba(8,12,22,.7);border:1px solid rgba(0,240,255,.35);font-family:var(--display);letter-spacing:.12em;text-transform:uppercase;transition:all .18s}
.stApp [data-testid="stButton"] button:hover,.stApp [data-testid="stDownloadButton"] button:hover {border-color:var(--neon);box-shadow:0 0 18px rgba(0,240,255,.35);transform:translateY(-1px)}
.stApp [data-testid="stButton"] button[kind="primary"] {background:linear-gradient(135deg,#FF2D55,#ff6a3d);border:0;font-size:1.05rem;min-height:54px;box-shadow:0 0 26px rgba(255,45,85,.5),inset 0 1px 0 rgba(255,255,255,.3);animation:glow 2.6s ease-in-out infinite}
.stApp [data-testid="stButton"] button[kind="primary"]:hover {box-shadow:0 0 38px rgba(255,45,85,.8);transform:translateY(-2px) scale(1.01)}
@keyframes glow {50% {box-shadow:0 0 40px rgba(255,45,85,.75),inset 0 1px 0 rgba(255,255,255,.3)}}
.panel-title {display:flex;align-items:center;gap:10px;font-family:var(--display);font-weight:700;font-size:1.05rem;letter-spacing:.18em;text-transform:uppercase;color:#fff;margin:.2rem 0 .6rem}
.panel-title::before {content:"";width:10px;height:10px;border-radius:2px;background:var(--neon);box-shadow:0 0 12px var(--neon);transform:rotate(45deg)}
.panel-title small {margin-left:auto;font-family:var(--mono);font-size:.62rem;letter-spacing:.14em;color:#7f93ad;font-weight:400}
hr,.stApp [data-testid="stDivider"] hr {border:0 !important;height:1px !important;background:linear-gradient(90deg,transparent,rgba(0,240,255,.5),rgba(255,45,85,.5),transparent) !important;margin:1.4rem 0 !important}
.stApp [data-testid="stTabs"] [role="tablist"] {border-bottom:1px solid var(--glass-edge);gap:.4rem}
.stApp [role="tab"] {font-family:var(--display);font-weight:600;letter-spacing:.12em;text-transform:uppercase;font-size:.95rem;padding:.6rem 1rem}
.stApp [role="tab"][aria-selected="true"] {color:var(--neon) !important;text-shadow:0 0 12px rgba(0,240,255,.7)}
.stApp [data-testid="stTabs"] [data-baseweb="tab-highlight"] {background:var(--neon) !important;box-shadow:0 0 12px var(--neon);height:2px}
.driver-card {background:linear-gradient(160deg,rgba(255,255,255,.06),rgba(255,255,255,.01)),rgba(12,17,30,.8) !important;border:1px solid var(--glass-edge) !important;border-top:3px solid var(--team-accent) !important;backdrop-filter:blur(10px);box-shadow:0 10px 30px rgba(0,0,0,.5),0 -6px 22px -12px var(--team-accent)}
.driver-card:hover,.driver-card:focus-within {box-shadow:0 14px 40px rgba(0,0,0,.6),0 0 26px -4px var(--team-accent)}
.portrait {background:radial-gradient(circle at 50% 100%,color-mix(in srgb,var(--team-accent) 28%,transparent),transparent 70%) !important}
.position {font-family:var(--display);color:#fff;text-shadow:0 0 14px var(--team-accent)}
.driver-code,.card-score,.stats-row,.race-detail,.stats-source {font-family:var(--mono)}
.card-name {font-family:var(--display);font-size:1.15rem;letter-spacing:.04em;text-transform:uppercase}
.card-score strong {color:var(--neon) !important;text-shadow:0 0 10px rgba(0,240,255,.5)}
.driver-stats {background:rgba(8,12,22,.94) !important;backdrop-filter:blur(10px)}
.versus {font-family:var(--display);font-size:1.6rem !important;color:var(--neon-red) !important;text-shadow:0 0 18px var(--neon-red)}
.exact-table th {font-family:var(--mono);font-size:.68rem;letter-spacing:.14em;text-transform:uppercase;color:var(--neon) !important}
.exact-table td {font-family:var(--mono)}
.exact-table tbody tr:hover {background:rgba(0,240,255,.06)}
.stApp [data-testid="stSidebar"] {background:rgba(8,12,22,.82) !important;backdrop-filter:blur(18px);border-right:1px solid var(--glass-edge)}
.stApp [data-testid="stAlert"] {background:var(--glass);border:1px solid var(--glass-edge);backdrop-filter:blur(10px)}
.stApp [data-testid="stImage"] img,.stApp [data-testid="stPyplot"] img {background:transparent !important;border-radius:12px}
.site-foot {font-family:var(--mono);letter-spacing:.08em;text-transform:uppercase;font-size:.62rem;border-top:1px solid var(--glass-edge)}
::-webkit-scrollbar {width:8px;height:8px} ::-webkit-scrollbar-thumb {background:rgba(0,240,255,.3);border-radius:8px} ::-webkit-scrollbar-track {background:transparent}
@media(max-width:800px){.brand-title{font-size:1.4rem !important}.event-card h2{font-size:1.6rem !important}.stApp [data-testid="stMain"] [data-testid="stRadio"] label{padding:6px 10px}}
@media(prefers-reduced-motion:reduce){.status-strip .live::before,.stApp [data-testid="stButton"] button[kind="primary"]{animation:none}}
"""


def dark_mode():
    return st.session_state.get("ui_theme", "Dark Mode") == "Dark Mode"


def apply_pit_wall_skin():
    """Inject the telemetry skin and make Matplotlib figures transparent (dark mode only)."""
    if not dark_mode():
        return
    st.markdown(f"<style>{PIT_WALL_CSS}</style>", unsafe_allow_html=True)
    plt.rcParams.update({"figure.facecolor": "none", "axes.facecolor": (1, 1, 1, 0.03), "savefig.facecolor": "none",
                         "savefig.transparent": True, "text.color": "#E6EEF8", "axes.labelcolor": "#B9C6D8",
                         "xtick.color": "#9AA9BF", "ytick.color": "#9AA9BF", "axes.edgecolor": (0, 0.94, 1, 0.25),
                         "grid.color": (1, 1, 1, 0.10), "legend.facecolor": (0.03, 0.05, 0.09, 0.7),
                         "legend.edgecolor": (0, 0.94, 1, 0.3), "legend.labelcolor": "#E6EEF8"})


def render_status_strip():
    now = datetime.now(ZoneInfo("Australia/Sydney"))
    st.markdown('<div class="status-strip"><span class="live">Telemetry link <b>online</b></span>'
                f'<span>Pit wall <b>{now.strftime("%H:%M")} Sydney</b></span>'
                '<span>Engine <b>XGBoost</b></span><span>Feed <b>FastF1</b></span></div>', unsafe_allow_html=True)


def panel_title(title, tag=""):
    st.markdown(f'<div class="panel-title">{title}<small>{tag}</small></div>', unsafe_allow_html=True)


def show_plot(fig):
    if dark_mode():  # transparent figure so the glass background shows through
        fig.patch.set_alpha(0)
        for ax in fig.axes:
            ax.set_facecolor((1, 1, 1, 0.03))
            for spine in ax.spines.values():
                spine.set_color((0, 0.94, 1, 0.25))
            ax.tick_params(colors="#9AA9BF")
            ax.xaxis.label.set_color("#B9C6D8")
            ax.yaxis.label.set_color("#B9C6D8")
            ax.grid(color=(1, 1, 1, 0.10))
    fig.tight_layout()
    st.pyplot(fig, transparent=True)
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






def predictor_page():
    section_header("Your next race, forecast", "Compare the model's expected finishing order before lights out.", "Race predictor")
    today = datetime.now(ZoneInfo("Australia/Sydney"))
    history = load_history()
    if history.empty:
        st.info("Open Model Insights to collect historical races and train a model.")
        return
    try:
        _, metadata = load_model()
    except (FileNotFoundError, ValueError) as exc:
        st.info(str(exc))
        return
    boundary = metadata["trained_through"]
    with st.container(border=True):
        panel_title("Prediction control panel", f"Model · trained through {boundary['season']} R{boundary['round']}")
        season_column, race_column = st.columns([1, 3])
        with season_column:
            season = st.selectbox("Prediction season", [today.year, today.year + 1])
        try:
            events = upcoming_schedule(season)
        except Exception as exc:
            st.error(f"Could not load upcoming events: {exc}")
            return
        if events.empty:
            st.info("No upcoming races have been published for this season.")
            return
        names = dict(zip(events["RoundNumber"].astype(int), events["EventName"]))
        with race_column:
            round_number = st.selectbox("Upcoming race", list(names), format_func=lambda number: names[number])
    event = events.loc[events["RoundNumber"] == round_number].iloc[0]
    circuit = event.get("Location") or event.get("Country") or "Circuit location unavailable"
    event_header(names[round_number], circuit, season,
                 f"XGBoost · trained through {boundary['season']} / R{boundary['round']}")
    try:
        roster = latest_entrants(history, season, round_number)
    except ValueError as exc:
        st.error(str(exc))
        return
    latest = history.sort_values(["season", "round"]).iloc[-1]
    grid_column, launch_column = st.columns([3, 2], gap="large")
    with grid_column:
        with st.container(border=True):
            panel_title("Entry list", f"{len(roster)} cars")
            st.caption(f"History available through {int(latest['season'])} round {int(latest['round'])}. "
                       "Collect the current season to update driver form before predicting.")
            with st.expander("Review race entry list", expanded=False):
                st.caption("Based on the latest completed race. Confirm teams and add/remove replacement drivers before predicting.")
                entries = st.data_editor(roster, num_rows="dynamic", hide_index=True, width="stretch",
                                         key=f"entries_{season}_{round_number}_{len(history)}")
    history_revision = HISTORY_PATH.stat().st_mtime_ns if HISTORY_PATH.exists() else 0
    signature = f"{season}:{round_number}:{len(history)}:{history_revision}:{metadata['trained_at_utc']}:{entries.to_json()}"
    with launch_column:
        with st.container(border=True):
            panel_title("Launch", "Race simulation")
            st.caption("Run the XGBoost ranking for the confirmed entry list.")
            launch = st.button("Predict finishing order", type="primary", width="stretch")
    st.divider()
    if launch:
        try:
            result = predict_race(history, season, round_number, entrants=entries)
        except (ValueError, OSError) as exc:
            st.error(str(exc))
            return
        st.session_state["prediction_output"] = {"signature": signature, "result": result,
            "timestamp": datetime.now(ZoneInfo("Australia/Sydney")).strftime("%d %b %Y · %H:%M %Z")}
    saved = st.session_state.get("prediction_output")
    if saved and saved["signature"] == signature:
        result = saved["result"]
        st.caption(f"Prediction generated {saved['timestamp']} · ML scores, not probabilities")
        render_prediction_cards(result, season)
        st.caption("Deterministic ranking for the supplied entry list. No uncertainty estimates are calculated. "
                   "Historical race points exclude sprint points and championship penalties.")
        with st.expander("Exact prediction values"):
            render_prediction_table(result)
        st.download_button("Download prediction CSV", result.to_csv(index=False),
                           file_name=f"prediction_{season}_{round_number}.csv", mime="text/csv", on_click="ignore")
    else:
        st.info("Confirm the entry list, then generate a prediction to reveal the podium and full grid.")
    with st.expander("Model evaluation summary"):
        model_evaluation(metadata)


def analysis_page():
    section_header("Two drivers. One circuit.", "Explore fastest-lap telemetry, tyre performance, sectors and weather.", "Race analysis / driver comparison")
    st.caption("Open the sidebar to choose a season, race and two drivers.")

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
    event = schedule.loc[schedule["RoundNumber"] == round_number].iloc[0]
    event_header(race_names[round_number], event.get("Location", "Race session"), year)
    render_comparison(driver1, driver2, results, year)
    st.caption(f"{driver1}: solid line · {driver2}: {'dashed' if styles[1] == '--' else 'solid'} line")
    st.divider()

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
        section_header("Fastest-lap telemetry", "Five channels aligned by distance")
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
        section_header("Tyre performance", "Lap time against tyre age")
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
        section_header("Circuit trace", "Fastest-lap position telemetry")
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
        section_header("Sector breakdown", "Fastest-lap sector times in seconds")
        st.caption("Sector times from each driver's fastest race lap, in seconds.")
        rows = []
        for driver, _, _, lap, _ in lap_data:
            row = {"Driver": driver}
            for sector in range(1, 4):
                value = lap[f"Sector{sector}Time"] if lap is not None else pd.NaT
                row[f"Sector {sector}"] = value.total_seconds() if pd.notna(value) else np.nan
            rows.append(row)
        st.dataframe(styled_dataframe(pd.DataFrame(rows).set_index("Driver")).format("{:.3f}", na_rep="Unavailable"),
                     width="stretch")

    with weather_tab:
        section_header("Track conditions", "Track and air temperatures during the session")
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
    render_header()
    render_floating_theme_toggle()
    apply_theme()
    apply_pit_wall_skin()
    render_status_strip()
    st.write("")
    page = st.radio("Navigation", ["Race Predictor", "Race Results", "Race Analysis", "Model Insights"],
                    horizontal=True, label_visibility="collapsed", key="ui_page")
    with st.sidebar:
        panel_title("Race workspace", "Pit wall")
        st.caption("Forecast upcoming races or explore completed sessions.")
    if page == "Race Predictor":
        predictor_page()
    elif page == "Race Results":
        race_results_page()
    elif page == "Model Insights":
        section_header("Behind the prediction", "Validation, feature importance and model management", "Model insights")
        predictor_training(datetime.now(ZoneInfo("Australia/Sydney")))
        try:
            _, metadata = load_model()
            model_evaluation(metadata)
        except (FileNotFoundError, ValueError) as exc:
            st.info(str(exc))
    else:
        analysis_page()
    st.divider()
    footer()


if __name__ == "__main__":
    main()
