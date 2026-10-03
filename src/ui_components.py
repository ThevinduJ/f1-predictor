"""Reusable, escaped HTML components; images load independently in the browser."""
from html import escape
import math

import streamlit as st

from .driver_data import get_driver, team_accent


def text(value):
    return escape(str(value), quote=True)


def section_header(title, description="", eyebrow=""):
    st.markdown(f'<div class="section-heading"><div class="eyebrow">{text(eyebrow)}</div>'
                f'<h2>{text(title)}</h2><p>{text(description)}</p></div>', unsafe_allow_html=True)


def render_header():
    st.markdown('<div class="brand"><div class="brand-mark">F1</div><div><div class="brand-title">F1 Race Predictor</div>'
                '<div class="brand-subtitle">Machine-learning race predictions powered by FastF1 data</div></div></div>', unsafe_allow_html=True)


def _toggle_theme():
    st.session_state.ui_theme = "Light Mode" if st.session_state.get("ui_theme", "Dark Mode") == "Dark Mode" else "Dark Mode"


def render_floating_theme_toggle():
    """A native Streamlit button: callback runs before the app rerenders."""
    # Reassign to detach this value from the removed selectbox's widget lifecycle.
    st.session_state.ui_theme = st.session_state.get("ui_theme", "Dark Mode")
    dark = st.session_state.ui_theme == "Dark Mode"
    action = "Switch to light mode" if dark else "Switch to dark mode"
    # The keyed container scopes the fixed positioning and icon-only styling.
    # Keep the real label accessible; CSS visually hides it without removing it.
    with st.container(key="floating_theme_control"):
        st.button(action, icon=":material/light_mode:" if dark else ":material/dark_mode:",
                  key="floating_theme_toggle", help=action, on_click=_toggle_theme)


def event_header(race, circuit, season, detail=""):
    st.markdown(f'<div class="event-card"><div class="eyebrow">Race weekend · {text(season)}</div>'
                f'<h2>{text(race)}</h2><div class="event-details"><span>{text(circuit)}</span>'
                f'<span>{text(detail)}</span></div></div>', unsafe_allow_html=True)


def driver_card_html(abbreviation, team=None, full_name=None, position=None, score=None, race_details=None):
    driver = get_driver(abbreviation, full_name, team)
    display = lambda value: "Unavailable" if value is None else text(value)
    image = f'<img src="{text(driver["image_url"])}" alt="{text(driver["full_name"])} portrait" loading="lazy" decoding="async" referrerpolicy="no-referrer">' if driver["image_url"] else ''
    placeholder = '<div class="avatar-fallback" aria-hidden="true"><svg viewBox="0 0 140 160" fill="currentColor"><circle cx="70" cy="44" r="29"/><path d="M12 160v-28c0-34 26-55 58-55s58 21 58 55v28z"/></svg></div>'
    position_label = f"P{int(position)}" if position is not None else "DRIVER"
    score_html = ''
    if score is not None and math.isfinite(float(score)):
        score_html = f'<div class="card-score"><span>Predicted score</span><strong>{float(score):.2f}</strong></div>'
    if race_details:
        score_html += '<div class="race-card-details">' + ''.join(
            f'<div class="race-detail"><span>{text(label)}</span><strong class="{text(tone)}">{text(value)}</strong></div>'
            for label, value, tone in race_details) + '</div>'
    extra_profile = ''
    if race_details:
        extra_profile = (f'<div class="stats-row"><span>Nationality</span><strong>{display(driver.get("nationality"))}</strong></div>'
                         f'<div class="stats-row"><span>Driver number</span><strong>{display(driver.get("driver_number"))}</strong></div>')
    source = "Career statistics unavailable"
    if driver.get("stats_as_of"):
        source = f"Official F1 profile · snapshot {driver['stats_as_of'][:10]}"
    return (f'<article class="driver-card" tabindex="0" style="--team-accent:{team_accent(driver["team"])}" '
            f'aria-label="{text(position_label)} {text(driver["full_name"])}. Focus to view profile statistics.">'
            f'<div class="card-top"><span class="position">{text(position_label)}</span><span class="driver-code">{text(abbreviation)}</span></div>'
            f'<div class="portrait">{placeholder}{image}</div><div class="card-copy"><div class="card-name">{text(driver["full_name"])}</div>'
            f'<div class="card-team">{text(driver["team"])}</div>{score_html}</div>'
            f'<div class="driver-stats"><div class="stats-name">{text(driver["full_name"])}</div>'
            f'<div class="stats-row"><span>Age</span><strong>{display(driver.get("age"))}</strong></div>'
            f'<div class="stats-row"><span>Career wins</span><strong>{display(driver.get("career_wins"))}</strong></div>'
            f'<div class="stats-row"><span>Career podiums</span><strong>{display(driver.get("career_podiums"))}</strong></div>'
            f'{extra_profile}'
            f'<div class="stats-source">{text(source)}<br>Hover, tap or focus a card to view profile details.</div></div></article>')


def render_driver_card(**kwargs):
    st.markdown(driver_card_html(**kwargs), unsafe_allow_html=True)


def render_result_cards(classification):
    """Extend the shared driver card; never substitute prediction scores for results."""
    def value(number):
        return f"{number:g}" if number is not None and math.isfinite(float(number)) else "N/A"

    def card(row):
        change = row['Positions Changed']
        tone = 'gain' if math.isfinite(float(change)) and change > 0 else 'loss' if math.isfinite(float(change)) and change < 0 else ''
        change_label = f'{change:+.0f}' if math.isfinite(float(change)) and change != 0 else value(change)
        grid = 'Pit lane' if row['Grid'] == 0 else value(row['Grid'])
        details = [('Grid', grid, ''), ('Positions changed', change_label, tone),
                   ('Points', value(row['Points']), ''), ('Time / gap', row['Time/Gap'], ''),
                   ('Status', row['Status'], '')]
        position = row['Position'] if math.isfinite(float(row['Position'])) else None
        return driver_card_html(row['Driver'], team=row['Team'], full_name=row['Name'], position=position, race_details=details)

    podium = classification.loc[classification['Position'].isin([1, 2, 3])]
    section_header('Race podium', 'Official finishing order', 'Race results')
    st.markdown('<div class="driver-grid podium race-podium">' + ''.join(card(row) for _, row in podium.iterrows()) + '</div>', unsafe_allow_html=True)
    section_header('Full classification', 'Grid positions, movement and official race status')
    remainder = classification.loc[~classification['Position'].isin([1, 2, 3])]
    st.markdown('<div class="driver-grid race-classification">' + ''.join(card(row) for _, row in remainder.iterrows()) + '</div>', unsafe_allow_html=True)


def render_prediction_cards(result):
    ordered = result.sort_values("Predicted Position")
    def cards(rows):
        return ''.join(driver_card_html(row["Driver"], team=row["Team"],
                     position=row["Predicted Position"], score=row["Predicted Score"]) for _, row in rows.iterrows())
    section_header("Predicted podium", "The model's three highest-ranked drivers", "ML prediction")
    st.caption("Hover a driver card for career details. On mobile, tap the card; keyboard users can Tab to it.")
    st.markdown(f'<div class="driver-grid podium">{cards(ordered.head(3))}</div>', unsafe_allow_html=True)
    if len(ordered) > 3:
        section_header("Full predicted grid", "Remaining entrants, ranked by predicted score")
        st.markdown(f'<div class="driver-grid">{cards(ordered.iloc[3:])}</div>', unsafe_allow_html=True)


def render_comparison(driver1, driver2, results):
    def card(driver):
        rows = results.loc[results["Abbreviation"] == driver]
        row = rows.iloc[0] if not rows.empty else {}
        return driver_card_html(driver, full_name=row.get("FullName"), team=row.get("TeamName"))
    st.markdown(f'<div class="comparison">{card(driver1)}<div class="versus">VS</div>{card(driver2)}</div>', unsafe_allow_html=True)


def render_prediction_table(result):
    headings = ''.join(f'<th>{text(column)}</th>' for column in result.columns)
    rows = ''.join('<tr>' + ''.join(f'<td>{float(value):.2f}</td>' if column == "Predicted Score" else f'<td>{text(value)}</td>'
                                 for column, value in row.items()) + '</tr>' for _, row in result.iterrows())
    st.markdown(f'<div class="table-wrap"><table class="exact-table"><thead><tr>{headings}</tr></thead><tbody>{rows}</tbody></table></div>', unsafe_allow_html=True)


def footer():
    st.markdown('<div class="site-foot">Independent analytics project · FastF1 / XGBoost · Driver imagery and profile facts sourced from Formula1.com. Not affiliated with Formula 1.</div>', unsafe_allow_html=True)


# Model management is UI-only; underlying collection/training functions are unchanged.
import pandas as pd
from .data_collection import collect_history, load_history
from .predict import load_model
from .train_model import train_model, SAMPLE_RECENCY_DECAY
from .features import RECENCY_DECAY

def predictor_training(today):
    with st.expander("Collect historical data and train", expanded=False):
        st.caption("First download can take several minutes. Successful races are cached. "
                   "Training uses the selected holdout season, then refits on all history.")
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


