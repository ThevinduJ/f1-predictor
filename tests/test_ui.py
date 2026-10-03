"""UI data handling, escaping and official-source safeguards."""
from datetime import date
import unittest
from unittest.mock import patch

from src.driver_data import age_on, get_driver, official_image_url, parse_profile
from src.ui_components import driver_card_html


class DriverUITests(unittest.TestCase):
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
        profile = get_driver('UNKNOWN', full_name='<script>unsafe</script>', team='New Team')
        self.assertIsNone(profile['image_url'])
        markup = driver_card_html('UNKNOWN', full_name='<script>unsafe</script>', team='New Team', position=4, score=8.432)
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
