"""Historical race data adapters. No prediction or training logic lives here."""
import fastf1
from fastf1.ergast.interface import Ergast
import pandas as pd
import streamlit as st


@st.cache_data(ttl=900, show_spinner=False)
def completed_events(season):
    """Only published results, including circuit metadata, in newest-first order.

    Query the winner only: one row per race avoids downloading every driver's
    results just to populate a selector. FastF1's HTTP cache also backs this API.
    """
    response = Ergast().get_race_results(season=int(season), results_position=1, limit=100)
    events = pd.DataFrame(response.description).copy()
    if events.empty:
        return events
    events['round'] = pd.to_numeric(events['round'], errors='coerce')
    dates = pd.to_datetime(events['raceDate'], utc=True, errors='coerce')
    events = events.loc[dates <= pd.Timestamp.now(tz='UTC')]
    return events.dropna(subset=['round']).sort_values('round', ascending=False).reset_index(drop=True)


@st.cache_resource(ttl=3600, max_entries=4, show_spinner=False)
def load_race_session(year, round_number):
    """Shared with Analysis so visiting either page reuses the same session."""
    session = fastf1.get_session(int(year), int(round_number), 'R')
    session.load()
    return session


def column(frame, name, default=None):
    return frame[name] if name in frame else pd.Series(default, index=frame.index)


def calculate_positions_changed(grid, finish):
    grid = pd.to_numeric(grid, errors='coerce')
    finish = pd.to_numeric(finish, errors='coerce')
    # Grid 0 is a pit-lane start, not P0. Don't assign a misleading loss.
    return (grid - finish).where((grid > 0) & (finish > 0))


def format_time(value):
    if pd.isna(value):
        return 'N/A'
    seconds = pd.Timedelta(value).total_seconds()
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f'{int(hours)}:{int(minutes):02d}:{seconds:06.3f}' if hours else f'{int(minutes)}:{seconds:06.3f}'


def build_classification(results):
    raw = pd.DataFrame(results).copy().reset_index(drop=True)
    frame = pd.DataFrame(index=raw.index)
    for target, source in [('Position', 'Position'), ('Grid', 'GridPosition'), ('Points', 'Points'), ('Laps', 'Laps')]:
        frame[target] = pd.to_numeric(column(raw, source), errors='coerce')
    frame['Driver'] = column(raw, 'Abbreviation', '').fillna('').replace('', pd.NA)
    frame['Driver'] = frame['Driver'].fillna(column(raw, 'DriverNumber', 'Unknown'))
    frame['Name'] = column(raw, 'FullName', '').fillna('').replace('', pd.NA).fillna(frame['Driver'])
    frame['Team'] = column(raw, 'TeamName', 'Unknown').fillna('Unknown')
    frame['Status'] = column(raw, 'Status', 'N/A').fillna('N/A').replace('', 'N/A')
    frame['Classification'] = column(raw, 'ClassifiedPosition', '').fillna('')
    frame['Positions Changed'] = calculate_positions_changed(frame['Grid'], frame['Position'])
    # Match FastF1's definition of a finisher, including drivers a lap down.
    finished = frame['Status'].eq('Finished') | frame['Status'].str.match(r'^\+\s*\d+\s+Laps?$', case=False)
    excluded = frame['Classification'].isin(['D', 'E', 'W', 'F']) | frame['Status'].str.contains('disqual|excluded|withdraw|did not start', case=False)
    frame['Finished'] = finished
    frame['DNF'] = ~finished & ~excluded & frame['Status'].ne('N/A')
    frame['Classified'] = pd.to_numeric(frame['Classification'], errors='coerce').notna()
    times = pd.to_timedelta(column(raw, 'Time'), errors='coerce')
    def time_gap(index):
        value, status = times.loc[index], frame.loc[index, 'Status']
        if not finished.loc[index]:
            return 'N/A'
        if status != 'Finished':
            return status
        if pd.isna(value):
            return 'N/A'
        if frame.loc[index, 'Position'] == 1:
            return format_time(value)
        # Despite the upstream name totalRaceTime, FastF1's parsed Time for
        # non-winners contains the published leader gap (verified on real data).
        gap = value.total_seconds()
        return f'+{gap:.3f}s' if gap >= 0 else 'N/A'
    frame['Time/Gap'] = [time_gap(index) for index in frame.index]
    return frame.sort_values('Position', na_position='last').reset_index(drop=True)


def get_fastest_laps(laps, classification):
    columns = ['Driver', 'Team', 'Fastest Lap', 'Lap', 'Compound']
    if not {'Driver', 'LapTime'}.issubset(laps.columns):
        return pd.DataFrame(columns=columns)
    valid = pd.DataFrame(laps).dropna(subset=['Driver', 'LapTime']).copy()
    if 'Deleted' in valid:
        valid = valid.loc[~valid['Deleted'].fillna(False).astype(bool)]
    if 'FastF1Generated' in valid:
        valid = valid.loc[~valid['FastF1Generated'].fillna(False).astype(bool)]
    valid = valid.loc[valid['LapTime'] > pd.Timedelta(0)]
    valid = valid.sort_values('LapTime').drop_duplicates('Driver')
    teams = classification.set_index('Driver')['Team'].to_dict()
    return pd.DataFrame({'Driver': valid['Driver'], 'Team': valid['Driver'].map(teams).fillna('Unknown'),
                         'Fastest Lap': valid['LapTime'].map(format_time),
                         'Lap': column(valid, 'LapNumber'), 'Compound': column(valid, 'Compound', 'N/A')}).reset_index(drop=True)


def get_team_summary(classification):
    rows = []
    for team, group in classification.groupby('Team', sort=False):
        rows.append({'Team': team, 'Drivers': ', '.join(group['Driver'].astype(str)),
                     'Finishes': ', '.join(f"P{int(row.Position)}" if pd.notna(row.Position) else 'N/A' for row in group.itertuples()),
                     'Points': group['Points'].sum(min_count=1), 'Average Finish': group['Position'].mean(),
                     'DNFs': int(group['DNF'].sum())})
    return pd.DataFrame(rows).sort_values('Points', ascending=False, na_position='last') if rows else pd.DataFrame()


def get_race_insights(classification):
    changes = classification.dropna(subset=['Positions Changed'])
    def extreme(largest):
        if changes.empty:
            return 'N/A'
        value = changes['Positions Changed'].max() if largest else changes['Positions Changed'].min()
        if (largest and value <= 0) or (not largest and value >= 0):
            return 'None'
        drivers = changes.loc[changes['Positions Changed'].eq(value), 'Driver']
        return f"{', '.join(drivers)} ({value:+.0f})"
    return {'Biggest gainer': extreme(True), 'Biggest loser': extreme(False),
            'DNFs': int(classification['DNF'].sum()), 'Classified finishers': int(classification['Classified'].sum()),
            'Teams in the points': classification.loc[classification['Points'] > 0, 'Team'].nunique(),
            'Top 10 average grid': classification.loc[classification['Position'].between(1, 10) & (classification['Grid'] > 0), 'Grid'].mean()}


def get_strategy_summary(laps):
    if not {'Driver', 'Stint', 'Compound', 'LapNumber'}.issubset(laps.columns):
        return pd.DataFrame()
    valid = pd.DataFrame(laps).dropna(subset=['Driver', 'Stint', 'Compound']).sort_values('LapNumber')
    valid = valid.loc[valid['Compound'].ne('UNKNOWN') & valid['Compound'].ne('')]
    stints = valid.groupby(['Driver', 'Stint'], sort=True)['Compound'].first()
    return stints.groupby(level='Driver').agg(lambda values: ' → '.join(values)).rename('Actual stints').reset_index()


@st.cache_data(ttl=3600, max_entries=12, show_spinner=False)
def load_race_results(year, round_number):
    session = load_race_session(year, round_number)
    classification = build_classification(session.results)
    if classification.empty or classification['Position'].notna().sum() == 0:
        raise ValueError('This race has no published classification yet.')
    try:
        laps = pd.DataFrame(session.laps).copy()
    except (AttributeError, fastf1.core.DataNotLoadedError):
        laps = pd.DataFrame()
    progression_columns = [name for name in ['Driver', 'LapNumber', 'Position'] if name in laps]
    return {'classification': classification, 'fastest': get_fastest_laps(laps, classification),
            'teams': get_team_summary(classification), 'insights': get_race_insights(classification),
            'strategies': get_strategy_summary(laps), 'progression': laps[progression_columns],
            'event': dict(session.event)}
