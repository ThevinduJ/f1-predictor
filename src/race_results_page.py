"""Streamlit flow for published historical results, separate from the ML UI."""
from datetime import datetime
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from .race_results import completed_events, load_race_results
from .ui_components import section_header, event_header, render_result_cards
from .theme import PALETTES, styled_dataframe


def show_table(frame):
    formats = {'Position': '%d', 'Grid': '%d', 'Positions Changed': '%+d',
               'Laps': '%d', 'Lap': '%d', 'Points': '%g', 'Average Finish': '%.1f'}
    config = {name: st.column_config.NumberColumn(format=format_string)
              for name, format_string in formats.items() if name in frame}
    st.dataframe(styled_dataframe(frame), hide_index=True, width='stretch', column_config=config)


def show_plot(fig):
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def render_movement_chart(classification):
    section_header('Grid vs finish', 'Positive values indicate positions gained; pit-lane and unknown starts are excluded.')
    valid = classification.dropna(subset=['Positions Changed']).sort_values('Positions Changed')
    if valid.empty:
        st.info('Grid positions are unavailable for this race.')
        return
    palette = PALETTES[st.session_state.get('ui_theme', 'Dark Mode')]
    fig, ax = plt.subplots(figsize=(10, max(4, len(valid) * .28)))
    colors = [palette['gain'] if change > 0 else palette['loss'] if change < 0 else palette['text-secondary'] for change in valid['Positions Changed']]
    labels = [f"{row['Driver']}  ·  Grid {row['Grid']:.0f} → Finish {row['Position']:.0f}" for _, row in valid.iterrows()]
    ax.barh(labels, valid['Positions Changed'], color=colors, height=.65)
    ax.axvline(0, color=palette['text-secondary'], linewidth=.8)
    ax.set_xlabel('Positions gained / lost')
    ax.grid(axis='x', alpha=.2)
    show_plot(fig)


def render_insights(insights):
    section_header('Race insights', 'Calculated from the published classification')
    columns = st.columns(3)
    for index, (label, value) in enumerate(insights.items()):
        if isinstance(value, float):
            value = f'{value:.1f}' if pd.notna(value) else 'N/A'
        with columns[index % 3]:
            st.metric(label, value)


def render_progression(progression, classification):
    if not {'Driver', 'LapNumber', 'Position'}.issubset(progression.columns):
        return
    with st.expander('Lap-by-lap race progression'):
        st.caption('Race position during each lap; post-race penalties can differ from the final classification.')
        choices = classification['Driver'].tolist()
        selected = st.multiselect('Drivers in progression chart', choices, default=choices[:3], key='results_progression_drivers')
        valid = progression.loc[progression['Driver'].isin(selected)].dropna(subset=['LapNumber', 'Position'])
        if valid.empty:
            st.info('No lap position data is available for these drivers.')
            return
        fig, ax = plt.subplots(figsize=(10, 5))
        for driver, group in valid.groupby('Driver'):
            group = group.sort_values('LapNumber')
            ax.plot(group['LapNumber'], group['Position'], label=driver, linewidth=1.5)
        ax.invert_yaxis()
        ax.set(xlabel='Lap', ylabel='Race position')
        ax.legend(ncol=3)
        ax.grid(alpha=.2)
        show_plot(fig)


def race_results_page():
    section_header('Race Results', 'Explore published classifications, driver movement and actual race strategies.', 'Race archive')
    current_year = datetime.now(ZoneInfo('Australia/Sydney')).year
    # Resolve the latest season with published results once per session.
    if 'results_latest_season' not in st.session_state:
        try:
            for year in [current_year, current_year - 1]:
                if not completed_events(year).empty:
                    st.session_state.results_latest_season = year
                    break
        except Exception as exc:
            st.warning(f'Could not retrieve published race results: {exc}')
            return
    latest = st.session_state.get('results_latest_season', current_year - 1)
    season_column, race_column = st.columns([1, 3])
    with season_column:
        season = st.selectbox('Season', list(range(latest, 1949, -1)), key='results_season')
    try:
        events = completed_events(season)
    except Exception as exc:
        st.warning(f'Could not load this season: {exc}')
        return
    if events.empty:
        st.info('No completed races with published results are available for this season.')
        return
    names = dict(zip(events['round'].astype(int), events['raceName']))
    with race_column:
        race = st.selectbox('Race / Grand Prix', list(names), format_func=names.get, key=f'results_round_{season}')
    event = events.loc[events['round'].eq(race)].iloc[0]
    date = pd.to_datetime(event.get('raceDate'), errors='coerce')
    date_label = date.strftime('%d %b %Y') if pd.notna(date) else 'Date unavailable'
    location = ' · '.join(str(event.get(key)) for key in ['locality', 'country'] if pd.notna(event.get(key)) and event.get(key))
    event_header(names[race], event.get('circuitName') or 'Circuit unavailable', season,
                 f'Round {race} · {date_label}' + (f' · {location}' if location else ''))
    try:
        with st.spinner('Loading race classification and lap data…'):
            bundle = load_race_results(season, race)
    except Exception as exc:
        st.warning(f'This race could not be loaded. Select another race and try again. Details: {exc}')
        return
    classification, fastest = bundle['classification'], bundle['fastest']
    winner = classification.loc[classification['Position'].eq(1), 'Driver']
    pole = classification.loc[classification['Grid'].eq(1), 'Driver']
    metrics = [('Winner', winner.iloc[0] if len(winner) else 'N/A'), ('Pole / grid P1', pole.iloc[0] if len(pole) else 'N/A'),
               ('Fastest lap', f"{fastest.iloc[0]['Driver']} · {fastest.iloc[0]['Fastest Lap']}" if len(fastest) else 'N/A'),
               ('Finishers', int(classification['Finished'].sum())), ('DNFs', int(classification['DNF'].sum()))]
    columns = st.columns(3)
    for index, (label, value) in enumerate(metrics):
        with columns[index % 3]:
            st.metric(label, value)
    st.caption('Pole uses the published starting grid. Retirements count as DNFs even when officially classified; disqualifications and non-starters are shown separately in status. Hover or focus cards for current career profiles.')
    render_result_cards(classification, season)
    with st.expander('Sortable classification table'):
        show_table(classification[['Position', 'Name', 'Driver', 'Team', 'Grid', 'Positions Changed', 'Points', 'Status', 'Classification', 'Time/Gap', 'Laps']])
        st.caption('Grid 0 denotes a pit-lane start. Position changes are unavailable for pit-lane or missing grid positions. Time is the winner’s total race time; other finishers show the leader gap or laps behind.')
    render_movement_chart(classification)
    section_header('Fastest laps', 'Best recorded, non-deleted race lap for each driver')
    if fastest.empty:
        st.info('Fastest-lap timing is unavailable for this race.')
    else:
        st.success(f"Overall fastest lap: {fastest.iloc[0]['Driver']} — {fastest.iloc[0]['Fastest Lap']}")
        show_table(fastest)
    section_header('Team performance', 'Race points only; sprint results are excluded.')
    show_table(bundle['teams'])
    render_insights(bundle['insights'])
    if not bundle['strategies'].empty:
        with st.expander('Actual tyre strategies'):
            show_table(bundle['strategies'])
            st.caption('Recorded compounds in stint order. Incomplete timing data may omit stints.')
    render_progression(bundle['progression'], classification)
