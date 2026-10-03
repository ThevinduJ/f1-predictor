"""UI-only driver profiles. No network requests occur while rendering cards.

Refresh the small official-site snapshot explicitly with
python -m src.driver_data --refresh. Missing fields remain unavailable.
"""
import argparse
from datetime import date, datetime
from html.parser import HTMLParser
import json
from pathlib import Path
import re
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

import requests
import streamlit as st

PROFILE_PATH = Path(__file__).with_name("driver_profiles.json")
IMAGE_PATH = Path(__file__).with_name("driver_images.json")
DRIVER_PROFILES = {
    "RUS": "george-russell", "ANT": "kimi-antonelli", "LEC": "charles-leclerc",
    "HAM": "lewis-hamilton", "NOR": "lando-norris", "PIA": "oscar-piastri",
    "VER": "max-verstappen", "HAD": "isack-hadjar", "LAW": "liam-lawson",
    "LIN": "arvid-lindblad", "GAS": "pierre-gasly", "COL": "franco-colapinto",
    "OCO": "esteban-ocon", "BEA": "oliver-bearman", "HUL": "nico-hulkenberg",
    "BOR": "gabriel-bortoleto", "SAI": "carlos-sainz", "ALB": "alexander-albon",
    "ALO": "fernando-alonso", "STR": "lance-stroll", "PER": "sergio-perez",
    "BOT": "valtteri-bottas", "TSU": "yuki-tsunoda",
}
TEAM_ACCENTS = {
    "mercedes": "#27B4A4", "ferrari": "#E54B4B", "mclaren": "#E98A36",
    "red bull racing": "#648BCE", "racing bulls": "#759EE6",
    "alpine": "#D88DB8", "haas f1 team": "#9BA9B5", "haas": "#9BA9B5",
    "audi": "#D45A59", "kick sauber": "#54AD73", "alfa romeo": "#B05B73",
    "williams": "#4E9CDD", "aston martin": "#399F88", "cadillac": "#9A92B6",
}


def official_image_url(url):
    parsed = urlparse(str(url or ""))
    return parsed.scheme == "https" and parsed.hostname in {"media.formula1.com", "www.formula1.com", "formula1.com"}


def age_on(date_of_birth, today=None):
    if not date_of_birth:
        return None
    try:
        born = date.fromisoformat(date_of_birth)
    except (ValueError, TypeError):
        return None
    today = today or datetime.now(ZoneInfo("Australia/Sydney")).date()
    return today.year - born.year - ((today.month, today.day) < (born.month, born.day))


@st.cache_data(show_spinner=False)
def _load_profiles(modified):
    try:
        return json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def normalize_team(team):
    key = re.sub(r'[^a-z0-9]', '', str(team or '').lower())
    return {'rbf1team': 'racingbulls', 'rb': 'racingbulls', 'redbull': 'redbullracing',
            'cadillacf1team': 'cadillac', 'haasf1team': 'haas', 'alphataurif1team': 'alphatauri',
            'alpinef1team': 'alpine', 'astonmartinarmco': 'astonmartin',
            'kicksauber': 'sauber', 'alfaromeoracing': 'alfaromeo', 'racingpointf1team': 'racingpoint'}.get(key, key)


@st.cache_data(show_spinner=False)
def _load_images(modified):
    try:
        return json.loads(IMAGE_PATH.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {'seasons': {}}


def _season_entry(abbreviation, season, team=None):
    try:
        year = str(int(season))
    except (TypeError, ValueError):
        return None
    modified = IMAGE_PATH.stat().st_mtime_ns if IMAGE_PATH.exists() else 0
    entries = _load_images(modified).get('seasons', {}).get(year, {}).get(str(abbreviation).upper(), [])
    if isinstance(team, str) and team.strip():
        entries = [entry for entry in entries if normalize_team(entry.get('team')) == normalize_team(team)]
    # Multiple team eras within one year require explicit race-team context.
    return entries[0] if len(entries) == 1 else None


def get_driver_image(driver_code, season, team=None):
    """Exact season/team only. No mutable headshots or cross-season fallback.

    Conservative placeholders also cover future seasons, retired drivers with
    missing assets, and ambiguous mid-season team changes. This performs no I/O
    beyond reading a cached local metadata file; validation is a maintenance job.
    """
    entry = _season_entry(driver_code, season, team)
    url = entry.get('image_url') if entry else None
    if not official_image_url(url):
        return None
    path = urlparse(url).path
    year = int(season)
    # Reject even an official URL if it is mutable/yearless or misfiled under
    # another season. A maintenance edit must never reintroduce current imagery.
    return url if f'/common/f1/{year}/' in path or f'/drivers/{year}Drivers/' in path else None


def get_driver(abbreviation, season, full_name=None, team=None):
    modified = PROFILE_PATH.stat().st_mtime_ns if PROFILE_PATH.exists() else 0
    snapshot = _load_profiles(modified)
    profile = snapshot.get("drivers", {}).get(str(abbreviation), {}).copy()
    profile["abbreviation"] = str(abbreviation)
    name = full_name if isinstance(full_name, str) and full_name.strip() else None
    team = team if isinstance(team, str) and team.strip() else None
    historical = _season_entry(abbreviation, season, team) or {}
    profile["full_name"] = name or historical.get('full_name') or profile.get("full_name") or str(abbreviation)
    profile["team"] = team or historical.get('team') or "Team unavailable"
    profile['driver_number'] = historical.get('driver_number')
    profile['season'] = season
    profile["age"] = age_on(profile.get("date_of_birth"))
    profile["image_url"] = get_driver_image(abbreviation, season, team)
    profile["stats_as_of"] = profile.get("retrieved_at_utc")
    return profile


def team_accent(team):
    return TEAM_ACCENTS.get(str(team).strip().lower(), "#8894A2")


class ProfileParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tokens, self.images, self.skip = [], [], 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.skip += 1
        if tag == "img":
            src = dict(attrs).get("src", "")
            if official_image_url(src) and "/common/f1/" in src:
                self.images.append(src)

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.skip = max(0, self.skip - 1)

    def handle_data(self, value):
        if not self.skip and value.strip():
            self.tokens.append(value.strip())


def parse_profile(html, abbreviation, source_url):
    parser = ProfileParser()
    parser.feed(html)
    tokens = parser.tokens
    title = next((text for text in tokens if " - F1 Driver for " in text), "")
    if not title:
        raise ValueError("Official driver profile layout not recognized")
    full_name, team = title.split(" - F1 Driver for ", 1)
    career = tokens[tokens.index("CAREER STATS"):] if "CAREER STATS" in tokens else []

    def after(label, scope):
        return scope[scope.index(label) + 1] if label in scope and scope.index(label) + 1 < len(scope) else None

    birth = after("Date of Birth", tokens)
    try:
        birth = datetime.strptime(birth, "%d/%m/%Y").date().isoformat()
    except (TypeError, ValueError):
        birth = None
    finish = after("Highest Race Finish", career)
    wins = None
    if finish:
        match = re.fullmatch(r"1\s*\(x(\d+)\)", finish)
        if match:
            wins = int(match.group(1))
        elif re.match(r"^[2-9]\b|^\d{2}\b", finish):
            wins = 0  # The published career best confirms no wins.
    podiums = after("Podiums", career)
    podiums = int(podiums) if podiums and podiums.isdigit() else None
    number, nationality = None, None
    for i, token in enumerate(tokens):
        if token.startswith("Flag of ") and i + 3 < len(tokens):
            nationality = tokens[i + 1]
            number = int(tokens[i + 3]) if tokens[i + 3].isdigit() else None
            break
    return {"abbreviation": abbreviation, "full_name": full_name, "team": team,
            "date_of_birth": birth, "nationality": nationality, "driver_number": number,
            "image_url": parser.images[0] if parser.images else None,
            "career_wins": wins, "career_podiums": podiums, "source_url": source_url,
            "retrieved_at_utc": datetime.now(ZoneInfo("UTC")).isoformat()}


def refresh_profiles():
    existing = json.loads(PROFILE_PATH.read_text(encoding="utf-8")) if PROFILE_PATH.exists() else {"drivers": {}}
    profiles = existing.get("drivers", {})
    for abbreviation, slug in DRIVER_PROFILES.items():
        url = f"https://www.formula1.com/en/drivers/{slug}"
        try:
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            response.encoding = "utf-8"
            profiles[abbreviation] = parse_profile(response.text, abbreviation, url)
            print(f"Updated {abbreviation}", flush=True)
        except (requests.RequestException, ValueError) as exc:
            print(f"Kept existing profile / fallback for {abbreviation}: {exc}", flush=True)
    PROFILE_PATH.write_text(json.dumps({"drivers": profiles}, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true")
    if parser.parse_args().refresh:
        refresh_profiles()
    else:
        print(f"Snapshot: {PROFILE_PATH}. Use --refresh to fetch the bounded list of official profiles.")
