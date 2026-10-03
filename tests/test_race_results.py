"""Tests for result semantics and the new navigation flow, without network calls."""
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest

from src.race_results import (build_classification, get_fastest_laps, get_team_summary,
                              get_race_insights, get_strategy_summary, completed_events, load_race_results)


def fixture():
    results = pd.DataFrame({
        'Position': [3, 1, 2, 4, 5], 'Abbreviation': ['CCC', 'AAA', 'BBB', 'DDD', 'EEE'],
        'FullName': ['Driver C', 'Driver A', 'Driver B', 'Driver D', 'Driver E'],
        'TeamName': ['Blue', 'Red', 'Red', 'Blue', 'Green'],
        'GridPosition': [0, 8, 3, 4, None], 'Points': [15, 25, 18, 0, 0],
        'Status': ['+1 Lap', 'Finished', 'Finished', 'Engine', 'Disqualified'],
        'ClassifiedPosition': ['3', '1', '2', '4', 'D'],
        'Time': pd.to_timedelta([None, '1:30:00', '0:00:18.4', None, None]), 'Laps': [49, 50, 50, 48, 45]})
    laps = pd.DataFrame({'Driver': ['AAA', 'AAA', 'BBB', 'BBB'],
                         'LapTime': pd.to_timedelta(['0:01:22', '0:01:18', '0:01:21', '0:01:20']),
                         'Deleted': [False, True, False, False], 'LapNumber': [5, 6, 5, 6],
                         'Compound': ['MEDIUM', 'HARD', 'SOFT', 'HARD'], 'Stint': [1, 2, 1, 2],
                         'Position': [1, 1, 2, 2]})
    return results, laps


class ResultTests(unittest.TestCase):
    def test_published_event_list_is_newest_first_and_excludes_future_dates(self):
        from types import SimpleNamespace
        dates = pd.DataFrame({'round': [1, 2, 3], 'raceDate': ['2025-01-01', '2025-02-01', '2099-01-01'],
                              'raceName': ['First', 'Second', 'Future']})
        with patch('src.race_results.Ergast') as ergast:
            ergast.return_value.get_race_results.return_value = SimpleNamespace(description=dates)
            completed_events.clear()
            self.assertEqual(completed_events(2025)['round'].tolist(), [2, 1])
            completed_events(2025)
            ergast.return_value.get_race_results.assert_called_once_with(season=2025, results_position=1, limit=100)
        completed_events.clear()

    def test_classification_movement_status_and_times(self):
        frame = build_classification(fixture()[0]).set_index('Driver')
        self.assertEqual(frame.index.tolist(), ['AAA', 'BBB', 'CCC', 'DDD', 'EEE'])
        self.assertEqual(frame.loc['AAA', 'Positions Changed'], 7)
        self.assertEqual(frame.loc['BBB', 'Positions Changed'], 1)
        self.assertTrue(pd.isna(frame.loc['CCC', 'Positions Changed']))
        self.assertTrue(pd.isna(frame.loc['EEE', 'Positions Changed']))
        self.assertEqual(frame.loc['BBB', 'Time/Gap'], '+18.400s')
        self.assertEqual(frame.loc['AAA', 'Time/Gap'], '1:30:00.000')
        self.assertTrue(frame.loc['CCC', 'Finished'])
        self.assertTrue(frame.loc['DDD', 'DNF'])
        self.assertTrue(frame.loc['DDD', 'Classified'])
        self.assertFalse(frame.loc['EEE', 'DNF'])
        self.assertFalse(frame.loc['EEE', 'Classified'])
        negative = fixture()[0]
        negative.loc[0, 'GridPosition'] = 1
        self.assertEqual(build_classification(negative).iloc[2]['Positions Changed'], -2)

    def test_fastest_laps_points_and_stints(self):
        results, laps = fixture()
        classification = build_classification(results)
        fastest = get_fastest_laps(laps, classification)
        self.assertEqual(fastest.iloc[0]['Driver'], 'BBB')
        self.assertEqual(fastest.iloc[0]['Fastest Lap'], '1:20.000')
        teams = get_team_summary(classification)
        self.assertEqual(teams.iloc[0]['Points'], 43)
        self.assertEqual(teams['Points'].sum(), classification['Points'].sum())
        self.assertEqual(get_race_insights(classification)['DNFs'], 1)
        self.assertEqual(get_strategy_summary(laps).iloc[0]['Actual stints'], 'MEDIUM → HARD')
        self.assertTrue(get_fastest_laps(pd.DataFrame(), classification).empty)
        self.assertTrue(get_strategy_summary(pd.DataFrame()).empty)
        self.assertTrue(build_classification(pd.DataFrame()).empty)

    def test_page_defaults_selection_theme_and_caching(self):
        results, laps = fixture()
        classification = build_classification(results)
        bundle = {'classification': classification, 'fastest': get_fastest_laps(laps, classification),
                  'teams': get_team_summary(classification), 'insights': get_race_insights(classification),
                  'strategies': get_strategy_summary(laps), 'progression': laps[['Driver', 'LapNumber', 'Position']], 'event': {}}
        events = pd.DataFrame({'round': [2, 1], 'raceName': ['Latest GP', 'Earlier GP'],
                               'raceDate': pd.to_datetime(['2025-08-01', '2025-07-01']), 'circuitName': ['Circuit', 'Circuit']})
        from src import race_results_page as page
        with patch.object(page, 'completed_events', side_effect=lambda year: events if year <= 2025 else pd.DataFrame()), \
             patch.object(page, 'load_race_results', return_value=bundle) as loader:
            app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py')).run(timeout=30)
            next(r for r in app.radio if r.label == 'Navigation').set_value('Race Results').run(timeout=30)
            self.assertFalse(app.exception, [e.message for e in app.exception])
            self.assertEqual(app.session_state['results_season'], 2025)
            self.assertEqual(app.session_state['results_round_2025'], 2)
            next(b for b in app.button if b.key == 'floating_theme_toggle').click().run(timeout=30)
            self.assertEqual(app.session_state['ui_theme'], 'Light Mode')
            self.assertFalse(app.exception, [e.message for e in app.exception])
            next(s for s in app.selectbox if s.label == 'Race / Grand Prix').set_value(1).run(timeout=30)
            self.assertEqual(loader.call_args.args, (2025, 1))
            next(s for s in app.selectbox if s.label == 'Season').set_value(2024).run(timeout=30)
            self.assertEqual(app.session_state['results_round_2024'], 2)
            self.assertFalse(app.exception, [e.message for e in app.exception])
            markup = '\n'.join(m.value for m in app.markdown)
            self.assertIn('Grid vs finish', markup)
            self.assertIn('Race podium', markup)
            self.assertIn('P1', markup)
        # Processed results are cached by race, so toggles never load a session.
        from types import SimpleNamespace
        load_race_results.clear()
        with patch('src.race_results.load_race_session', return_value=SimpleNamespace(results=results, laps=laps, event={})) as sessions:
            load_race_results(2025, 1)
            load_race_results(2025, 1)
            self.assertEqual(sessions.call_count, 1)
        load_race_results.clear()


if __name__ == '__main__':
    unittest.main()
