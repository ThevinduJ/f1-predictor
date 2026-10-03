"""Explicit maintenance command for official, season-scoped driver portraits.

Run python -m src.refresh_driver_images --seasons 2018 2019 ...
Never invoked by a Streamlit render. Rosters include replacements and team moves.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import unicodedata

import fastf1
from fastf1.ergast.interface import Ergast
import requests

from .driver_data import IMAGE_PATH, PROFILE_PATH, official_image_url, normalize_team


def slug(value):
    value = unicodedata.normalize('NFKD', str(value)).encode('ascii', 'ignore').decode()
    return re.sub('[^a-z0-9]', '', value.lower())


def image_candidates(year, row, profiles):
    team = slug(row['constructorName'])
    team = {'rbf1team': 'racingbulls' if year >= 2025 else 'rb', 'redbull': 'redbullracing',
            'cadillacf1team': 'cadillac', 'haasf1team': 'haas', 'alpinef1team': 'alpine',
            'kicksauber': 'sauber'}.get(team, team)
    code = slug(row['givenName'])[:3] + slug(row['familyName'])[:3] + '01'
    current = profiles.get(row['driverCode'], {}).get('image_url', '')
    # Prefer an observed asset ID for this driver (e.g. Andrea Kimi Antonelli).
    match = re.search(r'/common/f1/\d{4}/[^/]+/([^/]+)/', current)
    if match:
        code = match.group(1)
    candidates = []
    if f'/common/f1/{year}/' in current and normalize_team(profiles.get(row['driverCode'], {}).get('team')) == normalize_team(row['constructorName']):
        candidates.append(current)
    candidates.append(f'https://media.formula1.com/image/upload/common/f1/{year}/{team}/{code}/{year}{team}{code}right.webp')
    # F1's older assets use surname-based, explicitly year-scoped directories.
    # A single generic surname portrait is not safe for a mid-season team move.
    if row['_team_count'] == 1:
        candidates.append(f'https://media.formula1.com/image/upload/content/dam/fom-website/drivers/{year}Drivers/{slug(row["familyName"])}.jpg')
    return candidates


def validate(url):
    if not official_image_url(url):
        return False
    try:
        response = requests.head(url, timeout=10, allow_redirects=True)
        return (response.status_code == 200 and official_image_url(response.url)
                and response.headers.get('Content-Type', '').startswith('image/')
                and not response.headers.get('X-Cld-Error'))
    except requests.RequestException:
        return False


def refresh(seasons):
    fastf1.Cache.enable_cache(str(Path(__file__).resolve().parents[1] / 'data'))
    registry = json.loads(IMAGE_PATH.read_text(encoding='utf-8')) if IMAGE_PATH.exists() else {'seasons': {}}
    profiles = json.loads(PROFILE_PATH.read_text(encoding='utf-8')).get('drivers', {})
    for year in seasons:
        response = Ergast().get_race_results(season=year, limit=2000)
        pages = list(response.content)
        offset = sum(len(frame) for frame in pages)
        # Jolpica can clamp the requested limit to 100; include the whole season
        # so late replacements and mid-season team changes are never missed.
        while offset < response.total_results:
            page = Ergast().get_race_results(season=year, limit=2000, offset=offset)
            size = sum(len(frame) for frame in page.content)
            if not size:
                raise ValueError(f'Incomplete roster response for {year} at offset {offset}')
            pages.extend(page.content)
            offset += size
        import pandas as pd
        races = pd.concat(pages, ignore_index=True) if pages else pd.DataFrame()
        if races.empty:
            print(f'{year}: no published driver entries', flush=True)
            continue
        races = races.dropna(subset=['driverCode', 'constructorName'])
        counts = races.groupby('driverCode')['constructorName'].nunique()
        entries = races.drop_duplicates(['driverCode', 'constructorName'], keep='last').copy()
        entries['_team_count'] = entries['driverCode'].map(counts)
        def build(row):
            candidates = image_candidates(year, row, profiles)
            url = next((url for url in candidates if validate(url)), None)
            return row['driverCode'], {'team': row['constructorName'],
                'full_name': f'{row["givenName"]} {row["familyName"]}',
                'driver_number': int(row['number']) if pd.notna(row.get('number')) else None,
                'image_url': url, 'validated_at_utc': datetime.now(timezone.utc).isoformat() if url else None,
                'source': 'Official F1 season asset; roster from FastF1 Ergast/Jolpica'}
        season = {}
        with ThreadPoolExecutor(max_workers=4) as pool:
            for code, entry in pool.map(build, entries.to_dict('records')):
                season.setdefault(code, []).append(entry)
        registry['seasons'][str(year)] = season
        IMAGE_PATH.write_text(json.dumps(registry, indent=2, ensure_ascii=False), encoding='utf-8')
        found = sum(bool(entry['image_url']) for values in season.values() for entry in values)
        print(f'{year}: {found}/{len(entries)} validated driver/team portraits', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seasons', type=int, nargs='+', required=True)
    refresh(parser.parse_args().seasons)
