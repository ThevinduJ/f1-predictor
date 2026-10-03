"""UI data handling, escaping and official-source safeguards."""
from datetime import date
import unittest
from unittest.mock import patch

from src.driver_data import age_on, get_driver, get_driver_image, official_image_url, parse_profile
from src.ui_components import driver_card_html


def all_card_views(season):
    import pandas as pd
    from src.race_results import build_classification
    from src.ui_components import render_prediction_cards, render_comparison, render_result_cards
    team = 'Ferrari' if season == 2024 else 'Williams'
    results = pd.DataFrame({'Abbreviation': ['SAI'], 'FullName': ['Carlos Sainz'], 'TeamName': [team],
                            'Position': [1], 'GridPosition': [2], 'Points': [25], 'Status': ['Finished']})
    render_prediction_cards(pd.DataFrame({'Predicted Position': [1], 'Driver': ['SAI'], 'Team': [team], 'Predicted Score': [2.5]}), season)
    render_comparison('SAI', 'SAI', results, season)
    render_result_cards(build_classification(results), season)


class DriverUITests(unittest.TestCase):
    def test_every_shared_card_renderer_uses_passed_season(self):
        from streamlit.testing.v1 import AppTest
        for year, team in [(2024, 'Ferrari'), (2025, 'Williams')]:
            app = AppTest.from_function(all_card_views, args=(year,)).run()
            self.assertFalse(app.exception, [e.message for e in app.exception])
            markup = '\n'.join(item.value for item in app.markdown)
            url = get_driver_image('SAI', year, team)
            # Prediction podium, two comparison cards, and official results podium.
            self.assertEqual(markup.count(f'src="{url}"'), 4)
            self.assertIn(team, markup)

    def test_season_and_team_switches_use_exact_official_assets(self):
        cases = [('HAM', 2021, 'Mercedes', 2026, 'Ferrari'),
                 ('SAI', 2024, 'Ferrari', 2025, 'Williams'),
                 ('ALO', 2022, 'Alpine', 2023, 'Aston Martin')]
        for code, old_year, old_team, new_year, new_team in cases:
            with self.subTest(driver=code):
                old = get_driver_image(code, old_year, old_team)
                new = get_driver_image(code, new_year, new_team)
                self.assertTrue(official_image_url(old))
                self.assertTrue(official_image_url(new))
                self.assertIn(str(old_year), old)
                self.assertIn(str(new_year), new)
                self.assertNotEqual(old, new)
                profile = get_driver(code, old_year, team=old_team)
                self.assertEqual(profile['team'], old_team)
                self.assertEqual(profile['image_url'], old)
        self.assertIn('2021Drivers', get_driver_image('VET', 2021, 'Aston Martin'))
        self.assertEqual(get_driver('VER', 2021, team='Red Bull Racing')['driver_number'], 33)
        self.assertIsNotNone(get_driver_image('HUL', 2025, 'Kick Sauber'))

    def test_wrong_team_missing_year_and_ambiguous_team_use_placeholder(self):
        self.assertIsNone(get_driver_image('HAM', 2021, 'Ferrari'))
        self.assertIsNone(get_driver_image('HAM', 2018, 'Mercedes'))
        self.assertIsNone(get_driver_image('HAM', 2099, 'Ferrari'))
        self.assertIsNone(get_driver_image('UNKNOWN', 2025))
        self.assertIsNone(get_driver_image('LAW', 2025))
        self.assertNotEqual(get_driver_image('LAW', 2025, 'Red Bull Racing'), get_driver_image('LAW', 2025, 'Racing Bulls'))
        with self.assertRaises(TypeError):
            driver_card_html('HAM')
        markup = driver_card_html('HAM', season=2018, team='Mercedes')
        self.assertIn('avatar-fallback', markup)
        self.assertNotIn('<img ', markup)
        misfiled = {'seasons': {'2021': {'HAM': [{'team': 'Mercedes', 'image_url': get_driver_image('HAM', 2026, 'Ferrari')}]}}}
        with patch('src.driver_data._load_images', return_value=misfiled):
            self.assertIsNone(get_driver_image('HAM', 2021, 'Mercedes'))

    def test_age_calculated_at_birthday(self):
        self.assertEqual(age_on('2000-11-13', date(2026, 11, 12)), 25)
        self.assertEqual(age_on('2000-11-13', date(2026, 11, 13)), 26)
        self.assertIsNone(age_on(None))

    def test_nonofficial_images_rejected(self):
        self.assertFalse(official_image_url('https://example.com/driver.webp'))
        self.assertFalse(official_image_url('https://media.formula1.com.evil.test/a.webp'))
        self.assertFalse(official_image_url('javascript:alert(1)'))
        self.assertTrue(official_image_url('https://media.formula1.com/image/upload/driver.webp'))

    def test_unknown_driver_placeholder_and_no_invented_stats(self):
        profile = get_driver('UNKNOWN', 2021, full_name='<script>unsafe</script>', team='New Team')
        self.assertIsNone(profile['image_url'])
        markup = driver_card_html('UNKNOWN', full_name='<script>unsafe</script>', team='New Team', position=4, score=8.432, season=2021)
        self.assertIn('avatar-fallback', markup)
        self.assertIn('Unavailable', markup)
        self.assertIn('&lt;script&gt;', markup)
        self.assertNotIn('<script>', markup)
        self.assertIn('P4', markup)
        self.assertIn('8.43', markup)

    def test_career_fields_are_scoped_to_career_section(self):
        html = '<title>Test Driver - F1 Driver for Team</title><p>Grand Prix Wins</p><p>999</p><h2>CAREER STATS</h2><p>Highest Race Finish</p><p>1 (x2)</p><p>Podiums</p><p>4</p><p>Date of Birth</p><p>01/02/2000</p>'
        parsed = parse_profile(html, 'TST', 'https://www.formula1.com/en/drivers/test-driver')
        self.assertEqual(parsed['career_wins'], 2)
        self.assertEqual(parsed['career_podiums'], 4)
        self.assertEqual(parsed['date_of_birth'], '2000-02-01')


if __name__ == '__main__':
    unittest.main()
