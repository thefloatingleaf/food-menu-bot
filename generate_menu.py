#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import random
import re
import ssl
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

try:
    from zoneinfo import ZoneInfo
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("Python 3.9+ is required (zoneinfo missing)") from exc

BASE_DIR = Path(__file__).resolve().parent
BREAKFAST_SHISHIR_FILE = BASE_DIR / "breakfast_shishir.json"
MENU_SHISHIR_FILE = BASE_DIR / "menu_shishir.json"
BREAKFAST_VASANT_FILE = BASE_DIR / "breakfast_vasant.json"
MENU_VASANT_FILE = BASE_DIR / "menu_vasant.json"
BREAKFAST_GRISHM_FILE = BASE_DIR / "breakfast_grishm.json"
MENU_GRISHM_FILE = BASE_DIR / "menu_grishm.json"
BREAKFAST_VARSHA_FILE = BASE_DIR / "breakfast_varsha.json"
MENU_VARSHA_FILE = BASE_DIR / "menu_varsha.json"
BREAKFAST_SHARAD_FILE = BASE_DIR / "breakfast_sharad.json"
MENU_SHARAD_FILE = BASE_DIR / "menu_sharad.json"
BREAKFAST_HEMANT_FILE = BASE_DIR / "breakfast_hemant.json"
MENU_HEMANT_FILE = BASE_DIR / "menu_hemant.json"
EKADASHI_FILE = BASE_DIR / "ekadashi_2026_27.json"
PANCHANG_FILE = BASE_DIR / "panchang_2026_27.json"
FESTIVALS_FILE = BASE_DIR / "festivals_2026_27.json"
CONFIG_FILE = BASE_DIR / "config.json"
HISTORY_FILE = BASE_DIR / "history.json"
OUTPUT_FILE = BASE_DIR / "daily_menu.txt"
WEATHER_TAGS_FILE = BASE_DIR / "menu_weather_tags.json"
MANUAL_WEATHER_FILE = BASE_DIR / "manual_weather_override.json"
HEAVY_LIGHT_CLASSIFICATION_FILE = BASE_DIR / "heavy_light_classification_food_items_revised_paratha_rule.csv"


@dataclass
class EkadashiInfo:
    is_ekadashi: bool
    name_hi: str | None
    lunar_month_hi: str | None


@dataclass
class PanchangInfo:
    ritu_hi: str
    maah_hi: str
    tithi_hi: str


@dataclass
class FestivalInfo:
    hindu_hi: list[str]
    sikh_hi: list[str]


@dataclass
class PanchangLookupResult:
    row: dict[str, Any] | None
    status: str
    detail_hi: str
    total_rows: int
    valid_date_rows: int
    invalid_date_rows: int


@dataclass
class WeatherInfo:
    morning_temp_c: float
    max_temp_c: float
    rain_probability_pct: float
    is_rainy: bool
    is_extreme_cold: bool
    is_extreme_hot: bool
    source_hi: str


@dataclass
class WeatherRules:
    preferred_tags: set[str]
    avoid_tags: set[str]


@dataclass
class TransitionPlan:
    active: bool
    current_key: str
    upcoming_key: str
    days_to_next: int
    selected_key: str
    reason: str
    prefer_lighter: bool


@dataclass
class ShringdharaInfo:
    active: bool
    reason_hi: str
    missing_note_hi: str | None


VASANT_REQUIRED_SIDES = [
    "नीम की चटनी",
    "पुदीना की चटनी",
    "लहसुन की चटनी",
    "तीखा अचार (खट्टा नहीं)",
    "मूंग दाल पापड़",
    "मसाला छाछ (जीरा, अजवाइन, कढ़ी पत्ता, हींग, घी का तड़का)",
]

GRISHM_BREAKFAST_REQUIRED_SIDES = [
    "छाछ (काफ़ी पतली)",
    "पुदीना की चटनी",
]

GRISHM_MEAL_REQUIRED_SIDES = [
    "छाछ (काफ़ी पतली)",
    "पुदीना की चटनी",
    "खीरा और ककड़ी",
]

VARSHA_COMMON_REQUIRED_SIDES = [
    "आचार",
    "मिश्री-सौंफ़",
    "छाछ त्रिकटु के साथ",
]

NEW_YEAR_KANJI_NOTE = (
    "कृपया काली गाजर मँगवा लें और कांजी डाल लें। बनाने की विधि इस प्रकार है: "
    "5 लीटर साफ पानी लो। आधा किलो काली गाजर धोकर छील लो और लंबे टुकड़ों में काट लो। "
    "एक साफ बर्तन या काँच की बरनी लो, उसमें गाजर डाल दो। अब उसमें 4 चम्मच राई, 2 चम्मच नमक, "
    "आधा चम्मच काली मिर्च और एक चुटकी लाल मिर्च डालो। फिर 5 लीटर पानी डालकर अच्छी तरह मिला दो। "
    "बर्तन/बरनी को ढककर धूप में रख दो। रोज़ एक बार साफ चम्मच से हिला देना ताकि स्वाद अच्छे से आ जाए। "
    "5 दिन बाद नमक चखकर देखो। अगर नमक कम लगे, तो थोड़ा और नमक डाल दो और फिर से अच्छी तरह मिला दो।"
)

DEFAULT_LIGHT_FALLBACK_ITEMS = [
    "मूंग दाल की हल्की खिचड़ी",
    "नमक अजवाइन की रोटी",
    "दाल की रोटी (मूंग दाल)",
]

VARSHA_BANNED_KEYWORDS = [
    "प्याज",
    "प्याज़",
    "दही",
]

SHARAD_COMMON_REQUIRED_SIDES = [
    "सौंफ-मिश्री की मिश्रण",
    "छाछ त्रिकटु के साथ",
]

SHARAD_BANNED_KEYWORDS = [
    "इमली",
    "लौंग",
    "लहसुन",
    "प्याज",
    "प्याज़",
    "काली मिर्च",
    "गरम मसाला",
    "गर्म मसाला",
]

HEMANT_BANNED_KEYWORDS = [
    "बासमती",
    "मैदा",
    "डिब्बा बंद",
    "मोठ",
    "दोबारा गर्म",
    "जीरा",
    "इमली",
    "सॉस",
    "अचार",
    "कड़वा",
    "कसैला",
    "रिफाइंड",
    "पनीर",
    "एनर्जी ड्रिंक",
    "प्याज",
    "प्याज़",
    "दुबारा गर्म किया पानी",
]


SEASON_ORDER = ["shishir", "vasant", "grishm", "varsha", "sharad", "hemant"]
SEASON_HI = {
    "shishir": "शिशिर",
    "vasant": "वसंत",
    "grishm": "ग्रीष्म",
    "varsha": "वर्षा",
    "sharad": "शरद",
    "hemant": "हेमंत",
}

LUNAR_MAAS_TO_RITU_KEY = {
    "चैत्र": "vasant",
    "वैशाख": "vasant",
    "ज्येष्ठ": "grishm",
    "आषाढ़": "grishm",
    "श्रावण": "varsha",
    "भाद्रपद": "varsha",
    "आश्विन": "sharad",
    "कार्तिक": "sharad",
    "मार्गशीर्ष": "hemant",
    "पौष": "hemant",
    "माघ": "shishir",
    "फाल्गुन": "shishir",
}

TITHI_NUMBER_MAP = {
    "प्रतिपदा": 1,
    "प्रथमा": 1,
    "द्वितीया": 2,
    "तृतीया": 3,
    "चतुर्थी": 4,
    "पंचमी": 5,
    "षष्ठी": 6,
    "षष्ठि": 6,
    "सप्तमी": 7,
    "अष्टमी": 8,
    "नवमी": 9,
    "दशमी": 10,
    "एकादशी": 11,
    "द्वादशी": 12,
    "त्रयोदशी": 13,
    "चतुर्दशी": 14,
    "पूर्णिमा": 15,
    "अमावस्या": 15,
}

SHRINGDHARA_LIGHT_NOTE = "बताशा / खील / खिलौने (हल्का सेवन)"

SHRINGDHARA_DAILY_REMINDER = "शृंगधारा (यमराज की दाड़) अवधि चल रही है। आज भोजन हल्का और कम रखें।"

SEASON_START_MONTH_DAY = {
    "shishir": (1, 15),
    "vasant": (3, 15),
    "grishm": (5, 15),
    "varsha": (7, 15),
    "sharad": (9, 15),
    "hemant": (11, 15),
}


GREGORIAN_MONTH_HI = {
    1: "जनवरी",
    2: "फरवरी",
    3: "मार्च",
    4: "अप्रैल",
    5: "मई",
    6: "जून",
    7: "जुलाई",
    8: "अगस्त",
    9: "सितंबर",
    10: "अक्टूबर",
    11: "नवंबर",
    12: "दिसंबर",
}


def infer_ritu_hi_from_date(target_date: datetime.date) -> str:
    month_day = (target_date.month, target_date.day)
    # Traditional seasonal windows (approx.) used as fallback when panchang date entry is missing.
    if (month_day >= (1, 15) and month_day <= (3, 14)):
        return "शिशिर"
    if (month_day >= (3, 15) and month_day <= (5, 14)):
        return "वसंत"
    if (month_day >= (5, 15) and month_day <= (7, 14)):
        return "ग्रीष्म"
    if (month_day >= (7, 15) and month_day <= (9, 14)):
        return "वर्षा"
    if (month_day >= (9, 15) and month_day <= (11, 14)):
        return "शरद"
    return "हेमंत"


def normalize_lunar_month_name(maah_hi: str) -> str | None:
    if not isinstance(maah_hi, str):
        return None
    text = re.sub(r"\(.*?\)", "", maah_hi).strip()
    text = text.replace(" ", "")
    text = re.sub(r"[^\u0900-\u097F]", "", text)
    text = text.replace("मास", "").replace("माह", "").replace("अधिक", "")
    if not text:
        return None

    if "चैत्र" in text:
        return "चैत्र"
    if "वैशाख" in text:
        return "वैशाख"
    if "ज्येष्ठ" in text or "जेष्ट" in text:
        return "ज्येष्ठ"
    if "आषाढ़" in text or "आषाढ" in text:
        return "आषाढ़"
    if "श्रावण" in text or "सावन" in text:
        return "श्रावण"
    if "भाद्रपद" in text or "भादो" in text:
        return "भाद्रपद"
    if "आश्विन" in text or "क्वार" in text:
        return "आश्विन"
    if "कार्तिक" in text:
        return "कार्तिक"
    if "मार्गशीर्ष" in text or "मार्गशिर्ष" in text or "अगहन" in text:
        return "मार्गशीर्ष"
    if "पौष" in text or "पोष" in text or "पूस" in text:
        return "पौष"
    if "माघ" in text:
        return "माघ"
    if "फाल्गुन" in text or "फागुन" in text:
        return "फाल्गुन"
    return None


def resolve_ritu_key_from_lunar_month(maah_hi: str) -> str | None:
    canonical_maah = normalize_lunar_month_name(maah_hi)
    if canonical_maah is None:
        return None
    return LUNAR_MAAS_TO_RITU_KEY.get(canonical_maah)


def normalize_paksha_name(paksha_hi: str | None) -> str | None:
    if not isinstance(paksha_hi, str):
        return None
    cleaned = paksha_hi.replace(" ", "")
    if "कृष्ण" in cleaned:
        return "कृष्ण"
    if "शुक्ल" in cleaned:
        return "शुक्ल"
    return None


def parse_tithi_and_paksha(tithi_hi: str, paksha_hint: str | None) -> tuple[str | None, int | None]:
    paksha = normalize_paksha_name(paksha_hint)
    if not isinstance(tithi_hi, str):
        return paksha, None

    text = tithi_hi.replace(" ", "")
    if paksha is None:
        if "कृष्ण" in text:
            paksha = "कृष्ण"
        elif "शुक्ल" in text:
            paksha = "शुक्ल"

    for token, number in TITHI_NUMBER_MAP.items():
        if token in text:
            if paksha is None:
                if token == "अमावस्या":
                    paksha = "कृष्ण"
                elif token == "पूर्णिमा":
                    paksha = "शुक्ल"
            return paksha, number
    return paksha, None


def detect_shringdhara_observance(maah_hi: str, tithi_hi: str, paksha_hint: str | None) -> ShringdharaInfo:
    month_name = normalize_lunar_month_name(maah_hi)
    if month_name not in {"कार्तिक", "मार्गशीर्ष"}:
        return ShringdharaInfo(False, "", None)

    paksha, tithi_number = parse_tithi_and_paksha(tithi_hi, paksha_hint)
    if paksha is None or tithi_number is None:
        return ShringdharaInfo(
            False,
            "",
            "[अनुपलब्ध] शृंगधारा जाँच हेतु कार्तिक/मार्गशीर्ष में पक्ष/तिथि डेटा पर्याप्त नहीं",
        )

    if month_name == "कार्तिक" and paksha == "कृष्ण" and tithi_number >= 8:
        return ShringdharaInfo(True, "कार्तिक के अंतिम 8 दिन", None)
    if month_name == "मार्गशीर्ष" and paksha == "शुक्ल" and tithi_number <= 8:
        return ShringdharaInfo(True, "मार्गशीर्ष के प्रथम 8 दिन", None)
    return ShringdharaInfo(False, "", None)


def next_season_key(current_key: str) -> str:
    if current_key not in SEASON_ORDER:
        return "vasant"
    return SEASON_ORDER[(SEASON_ORDER.index(current_key) + 1) % len(SEASON_ORDER)]


def next_season_start_date(current_key: str, target_date: datetime.date) -> datetime.date:
    upcoming_key = next_season_key(current_key)
    month, day = SEASON_START_MONTH_DAY[upcoming_key]
    candidate = datetime(target_date.year, month, day).date()
    if candidate <= target_date:
        candidate = datetime(target_date.year + 1, month, day).date()
    return candidate


def season_weather_fit_score(season_key: str, weather: WeatherInfo) -> float:
    profiles = {
        "shishir": {"temp_center": 16.0, "temp_span": 10.0, "rain_pref": 0.15},
        "vasant": {"temp_center": 27.0, "temp_span": 9.0, "rain_pref": 0.30},
        "grishm": {"temp_center": 38.0, "temp_span": 9.0, "rain_pref": 0.20},
        "varsha": {"temp_center": 30.0, "temp_span": 7.0, "rain_pref": 0.80},
        "sharad": {"temp_center": 29.0, "temp_span": 9.0, "rain_pref": 0.25},
        "hemant": {"temp_center": 22.0, "temp_span": 9.0, "rain_pref": 0.20},
    }
    profile = profiles.get(season_key, profiles["shishir"])
    rain_ratio = max(0.0, min(weather.rain_probability_pct / 100.0, 1.0))
    temp_score = max(0.0, 1.0 - abs(weather.max_temp_c - profile["temp_center"]) / profile["temp_span"])
    rain_score = max(0.0, 1.0 - abs(rain_ratio - profile["rain_pref"]))
    return (0.7 * temp_score) + (0.3 * rain_score)


def resolve_weather_priority_season(
    current_key: str, upcoming_key: str, weather: WeatherInfo, thresholds: dict[str, float]
) -> tuple[str | None, str]:
    pair = {current_key, upcoming_key}
    cold_keys = {"shishir", "hemant"}

    if weather.is_rainy and "varsha" in pair:
        return ("varsha", "मौसम प्राथमिक: वर्षा संकेत")
    if weather.max_temp_c >= thresholds["hot_min_c"] and "grishm" in pair:
        return ("grishm", "मौसम प्राथमिक: गर्मी संकेत")
    if weather.max_temp_c >= thresholds["hot_min_c"]:
        if current_key in cold_keys and upcoming_key not in cold_keys:
            return (upcoming_key, "मौसम प्राथमिक: गर्मी संकेत")
        if upcoming_key in cold_keys and current_key not in cold_keys:
            return (current_key, "मौसम प्राथमिक: गर्मी संकेत")
    if weather.max_temp_c <= thresholds["cold_max_c"] and (pair & cold_keys):
        if current_key in cold_keys and upcoming_key not in cold_keys:
            return (current_key, "मौसम प्राथमिक: ठंड संकेत")
        if upcoming_key in cold_keys and current_key not in cold_keys:
            return (upcoming_key, "मौसम प्राथमिक: ठंड संकेत")

    current_score = season_weather_fit_score(current_key, weather)
    upcoming_score = season_weather_fit_score(upcoming_key, weather)
    if abs(current_score - upcoming_score) >= 0.10:
        if upcoming_score > current_score:
            return (upcoming_key, "मौसम प्राथमिक: आने वाली ऋतु अनुकूल")
        return (current_key, "मौसम प्राथमिक: वर्तमान ऋतु अनुकूल")

    return (None, "मौसम मिश्रित, स्पष्ट झुकाव नहीं")


def resolve_alternating_transition_key(
    target_date: datetime.date, days_to_next: int, current_key: str, upcoming_key: str
) -> tuple[str, str]:
    day_marker = target_date.toordinal()
    if days_to_next >= 11:
        selected = upcoming_key if day_marker % 3 == 0 else current_key
        return (selected, "संक्रमण मिश्रण: वर्तमान ऋतु प्रधान क्रम")
    if days_to_next >= 6:
        selected = upcoming_key if day_marker % 2 == 0 else current_key
        return (selected, "संक्रमण मिश्रण: दैनिक अदला-बदली")
    selected = current_key if day_marker % 3 == 0 else upcoming_key
    return (selected, "संक्रमण मिश्रण: आने वाली ऋतु प्रधान क्रम")


def resolve_transition_plan(
    target_date: datetime.date,
    current_key: str,
    weather: WeatherInfo | None,
    thresholds: dict[str, float],
    transition_window_days: int,
) -> TransitionPlan:
    active_current = current_key if current_key in SEASON_ORDER else "shishir"
    upcoming_key = next_season_key(active_current)
    next_start = next_season_start_date(active_current, target_date)
    days_to_next = (next_start - target_date).days

    if days_to_next > transition_window_days:
        return TransitionPlan(
            active=False,
            current_key=active_current,
            upcoming_key=upcoming_key,
            days_to_next=days_to_next,
            selected_key=active_current,
            reason="संक्रमण विंडो के बाहर",
            prefer_lighter=False,
        )

    if weather is not None:
        weather_selected, reason = resolve_weather_priority_season(active_current, upcoming_key, weather, thresholds)
        if weather_selected is not None:
            return TransitionPlan(
                active=True,
                current_key=active_current,
                upcoming_key=upcoming_key,
                days_to_next=days_to_next,
                selected_key=weather_selected,
                reason=reason,
                prefer_lighter=(weather_selected != active_current),
            )

    selected_key, reason = resolve_alternating_transition_key(target_date, days_to_next, active_current, upcoming_key)
    return TransitionPlan(
        active=True,
        current_key=active_current,
        upcoming_key=upcoming_key,
        days_to_next=days_to_next,
        selected_key=selected_key,
        reason=reason,
        prefer_lighter=True,
    )


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write("\n")


def normalize_item_key(item: str) -> str:
    return re.sub(r"\s+", " ", item).strip()


def load_heavy_light_classification(
    path: Path,
    valid_items: list[str],
) -> tuple[dict[str, str], str | None]:
    valid_keys = {normalize_item_key(item) for item in valid_items}
    mapping: dict[str, str] = {}

    if not path.exists():
        return {}, "[अनुपलब्ध] heavy/light वर्गीकरण फ़ाइल उपलब्ध नहीं"

    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                return {}, "[त्रुटि] heavy/light वर्गीकरण CSV खाली है"
            if "item" not in reader.fieldnames or "classification" not in reader.fieldnames:
                return {}, "[त्रुटि] heavy/light वर्गीकरण CSV में item/classification कॉलम नहीं मिले"

            bad_rows = 0
            for row in reader:
                item = normalize_item_key(str(row.get("item", "")))
                classification = str(row.get("classification", "")).strip().lower()
                if not item or classification not in {"heavy", "light"}:
                    bad_rows += 1
                    continue
                if item not in valid_keys:
                    continue
                mapping[item] = classification

    except OSError as exc:
        return {}, f"[त्रुटि] heavy/light वर्गीकरण CSV पढ़ने में समस्या: {exc}"
    except csv.Error as exc:
        return {}, f"[त्रुटि] heavy/light वर्गीकरण CSV पार्सिंग समस्या: {exc}"

    if not mapping:
        return {}, "[अनुपलब्ध] heavy/light वर्गीकरण डेटा मेनू आइटम से मेल नहीं खा रहा"

    missing_count = len(valid_keys - set(mapping.keys()))
    if missing_count > 0:
        return mapping, f"[अनुपलब्ध] {missing_count} आइटम का heavy/light वर्गीकरण CSV में नहीं मिला"

    return mapping, None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate daily food menu")
    parser.add_argument("--date", help="Date in YYYY-MM-DD format")
    parser.add_argument(
        "--bootstrap-weather-tags",
        action="store_true",
        help="Create or refresh weather tags file from keyword heuristics",
    )
    return parser.parse_args()


def resolve_date(date_arg: str | None, timezone_name: str) -> datetime.date:
    if date_arg:
        try:
            return datetime.strptime(date_arg, "%Y-%m-%d").date()
        except ValueError as exc:
            raise ValueError("--date must be in YYYY-MM-DD format") from exc
    return datetime.now(ZoneInfo(timezone_name)).date()


def get_ekadashi_info(target_date: str, ekadashi_data: dict[str, Any]) -> EkadashiInfo:
    matches = [e for e in ekadashi_data.get("ekadashi_list", []) if e.get("date") == target_date]
    if not matches:
        return EkadashiInfo(False, None, None)
    non_gauna = next((m for m in matches if not m.get("is_gauna", False)), None)
    chosen = non_gauna or matches[0]
    return EkadashiInfo(True, chosen.get("name_hi"), chosen.get("lunar_month_hi"))


def normalize_any_date_to_yyyy_mm_dd(raw_value: Any, timezone_name: str) -> str | None:
    if isinstance(raw_value, datetime):
        dt = raw_value
        if dt.tzinfo is not None:
            try:
                dt = dt.astimezone(ZoneInfo(timezone_name))
            except Exception:
                dt = dt.astimezone(ZoneInfo("Asia/Kolkata"))
        return dt.date().strftime("%Y-%m-%d")

    if not isinstance(raw_value, str):
        return None

    text = raw_value.strip()
    if not text:
        return None

    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        return text
    if re.fullmatch(r"\d{4}/\d{2}/\d{2}", text):
        return text.replace("/", "-")
    if re.fullmatch(r"\d{1,2}-\d{1,2}-\d{4}", text):
        day_str, month_str, year_str = text.split("-")
        try:
            parsed = datetime(int(year_str), int(month_str), int(day_str)).date()
            return parsed.strftime("%Y-%m-%d")
        except ValueError:
            return None
    if re.fullmatch(r"\d{1,2}/\d{1,2}/\d{4}", text):
        day_str, month_str, year_str = text.split("/")
        try:
            parsed = datetime(int(year_str), int(month_str), int(day_str)).date()
            return parsed.strftime("%Y-%m-%d")
        except ValueError:
            return None

    iso_text = text.replace("Z", "+00:00") if text.endswith("Z") else text
    try:
        dt = datetime.fromisoformat(iso_text)
        if dt.tzinfo is not None:
            try:
                dt = dt.astimezone(ZoneInfo(timezone_name))
            except Exception:
                dt = dt.astimezone(ZoneInfo("Asia/Kolkata"))
        return dt.date().strftime("%Y-%m-%d")
    except ValueError:
        pass

    date_token = text.split(" ")[0].split("T")[0]
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", date_token):
        return date_token
    if re.fullmatch(r"\d{4}/\d{2}/\d{2}", date_token):
        return date_token.replace("/", "-")

    return None


def extract_panchang_rows(panchang_data: Any) -> tuple[list[dict[str, Any]], str | None]:
    rows: list[dict[str, Any]] = []

    if isinstance(panchang_data, dict):
        entries = panchang_data.get("entries")
        if "entries" in panchang_data and not isinstance(entries, list):
            return [], "entries फ़ील्ड सूची (list) नहीं है"

        if isinstance(entries, list):
            for row in entries:
                if isinstance(row, dict):
                    rows.append(row)

        for key, val in panchang_data.items():
            if key == "entries":
                continue
            if not isinstance(val, dict):
                continue
            if not re.fullmatch(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}", str(key).strip()):
                continue
            row_copy = dict(val)
            row_copy.setdefault("date", key)
            rows.append(row_copy)

        if rows:
            return rows, None
        return [], "कोई मान्य प्रविष्टि नहीं मिली"

    if isinstance(panchang_data, list):
        for row in panchang_data:
            if isinstance(row, dict):
                rows.append(row)
        if rows:
            return rows, None
        return [], "सूची में मान्य object प्रविष्टि नहीं मिली"

    return [], "रूट JSON dict/list नहीं है"


def lookup_panchang_entry_for_date(
    target_date: datetime.date,
    timezone_name: str,
    panchang_data: Any,
) -> PanchangLookupResult:
    target_key = target_date.strftime("%Y-%m-%d")
    try:
        rows, shape_error = extract_panchang_rows(panchang_data)
        if shape_error:
            return PanchangLookupResult(
                row=None,
                status="source_invalid",
                detail_hi=f"पंचांग डेटा संरचना समस्या: {shape_error}",
                total_rows=0,
                valid_date_rows=0,
                invalid_date_rows=0,
            )

        indexed: dict[str, dict[str, Any]] = {}
        valid_date_rows = 0
        invalid_date_rows = 0

        for row in rows:
            normalized = normalize_any_date_to_yyyy_mm_dd(row.get("date"), timezone_name)
            if normalized is None:
                invalid_date_rows += 1
                continue
            valid_date_rows += 1
            if normalized not in indexed:
                indexed[normalized] = row

        if target_key in indexed:
            detail = "पंचांग प्रविष्टि मिली"
            if invalid_date_rows > 0:
                detail += f" (चेतावनी: {invalid_date_rows} प्रविष्टियों में date फ़ॉर्मैट अमान्य)"
            return PanchangLookupResult(
                row=indexed[target_key],
                status="ok",
                detail_hi=detail,
                total_rows=len(rows),
                valid_date_rows=valid_date_rows,
                invalid_date_rows=invalid_date_rows,
            )

        if valid_date_rows == 0:
            return PanchangLookupResult(
                row=None,
                status="mapping_error",
                detail_hi="पंचांग की तारीख़ें पढ़ी नहीं जा सकीं (date mapping/parsing error)",
                total_rows=len(rows),
                valid_date_rows=0,
                invalid_date_rows=invalid_date_rows,
            )

        return PanchangLookupResult(
            row=None,
            status="date_missing",
            detail_hi="स्रोत पंचांग में इस तिथि की प्रविष्टि उपलब्ध नहीं",
            total_rows=len(rows),
            valid_date_rows=valid_date_rows,
            invalid_date_rows=invalid_date_rows,
        )
    except Exception as exc:
        return PanchangLookupResult(
            row=None,
            status="lookup_error",
            detail_hi=f"पंचांग lookup त्रुटि: {exc}",
            total_rows=0,
            valid_date_rows=0,
            invalid_date_rows=0,
        )


def resolve_panchang_info(
    target_date: datetime.date,
    ekadashi: EkadashiInfo,
    panchang_row: dict[str, Any] | None,
    default_ritu: str,
) -> PanchangInfo:
    if panchang_row:
        ritu_hi = str(panchang_row.get("ritu_hi", default_ritu)).strip() or default_ritu
        maah_hi = str(
            panchang_row.get("maah_hi", ekadashi.lunar_month_hi or GREGORIAN_MONTH_HI[target_date.month])
        ).strip()
        tithi_hi = str(panchang_row.get("tithi_hi", "अज्ञात")).strip() or "अज्ञात"
        return PanchangInfo(ritu_hi=ritu_hi, maah_hi=maah_hi, tithi_hi=tithi_hi)

    inferred_ritu = infer_ritu_hi_from_date(target_date)
    maah_hi = ekadashi.lunar_month_hi or GREGORIAN_MONTH_HI[target_date.month]
    tithi_hi = "एकादशी" if ekadashi.is_ekadashi else "अज्ञात"
    return PanchangInfo(ritu_hi=inferred_ritu or default_ritu, maah_hi=maah_hi, tithi_hi=tithi_hi)


def get_festival_entry_for_date(target_date: str, festivals_data: Any) -> dict[str, Any] | None:
    if isinstance(festivals_data, dict):
        direct = festivals_data.get(target_date)
        if isinstance(direct, dict):
            return direct
        entries = festivals_data.get("entries")
        if isinstance(entries, list):
            for row in entries:
                if isinstance(row, dict) and row.get("date") == target_date:
                    return row
    elif isinstance(festivals_data, list):
        for row in festivals_data:
            if isinstance(row, dict) and row.get("date") == target_date:
                return row
    return None


def resolve_festival_info(target_date: str, festivals_data: Any) -> FestivalInfo:
    row = get_festival_entry_for_date(target_date, festivals_data)
    if not row:
        return FestivalInfo(hindu_hi=[], sikh_hi=[])

    hindu_hi = row.get("hindu_hi", [])
    sikh_hi = row.get("sikh_hi", [])

    hindu = [str(x).strip() for x in hindu_hi if isinstance(x, str) and str(x).strip()] if isinstance(hindu_hi, list) else []
    sikh = [str(x).strip() for x in sikh_hi if isinstance(x, str) and str(x).strip()] if isinstance(sikh_hi, list) else []
    return FestivalInfo(hindu_hi=hindu, sikh_hi=sikh)


def format_festival_line(festival_info: FestivalInfo) -> str | None:
    combined: list[str] = []
    for name in festival_info.hindu_hi + festival_info.sikh_hi:
        if name not in combined:
            combined.append(name)
    if not combined:
        return None
    return "*पर्व/त्योहार:* " + " / ".join(combined)


def is_blocked_item(item: str, keywords: list[str]) -> bool:
    return any(keyword in item for keyword in keywords)


def normalize_history(raw: Any) -> list[dict[str, str]]:
    if not isinstance(raw, list):
        return []

    cleaned: list[dict[str, str]] = []
    for row in raw:
        if not isinstance(row, dict):
            continue
        date_val = row.get("date")
        if not isinstance(date_val, str):
            continue

        breakfast_val = row.get("breakfast")
        meal_val = row.get("meal")

        if isinstance(breakfast_val, str) and isinstance(meal_val, str):
            cleaned.append({"date": date_val, "breakfast": breakfast_val, "meal": meal_val})
            continue

        old_item = row.get("item")
        if isinstance(old_item, str):
            cleaned.append({"date": date_val, "meal": old_item})

    return cleaned


def recent_items(
    history: list[dict[str, str]], target_date: datetime.date, window_days: int, field: str
) -> set[str]:
    earliest = target_date - timedelta(days=window_days)
    blocked: set[str] = set()
    for row in history:
        try:
            row_date = datetime.strptime(row["date"], "%Y-%m-%d").date()
        except (ValueError, KeyError):
            continue
        if earliest <= row_date < target_date:
            value = row.get(field)
            if isinstance(value, str) and value:
                blocked.add(value)
    return blocked


def apply_repeat_rule(pool: list[str], recent_block_set: set[str]) -> list[str]:
    filtered = [item for item in pool if item not in recent_block_set]
    return filtered if filtered else pool[:]


def parse_weather_thresholds(config: dict[str, Any]) -> dict[str, float]:
    raw = config.get("weather_thresholds", {})
    if not isinstance(raw, dict):
        raw = {}
    return {
        "cold_max_c": float(raw.get("cold_max_c", 18)),
        "hot_min_c": float(raw.get("hot_min_c", 30)),
        "extreme_cold_max_c": float(raw.get("extreme_cold_max_c", 10)),
        "extreme_hot_min_c": float(raw.get("extreme_hot_min_c", 35)),
        "rain_probability_high_pct": float(raw.get("rain_probability_high_pct", 50)),
    }


def fetch_url_text(
    request_or_url: Request | str,
    *,
    timeout: int,
    retries: int,
    allow_insecure_ssl: bool,
) -> str | None:
    for attempt in range(retries + 1):
        try:
            with urlopen(request_or_url, timeout=timeout) as response:
                return response.read().decode("utf-8", errors="ignore")
        except URLError as exc:
            if allow_insecure_ssl and "CERTIFICATE_VERIFY_FAILED" in str(exc):
                try:
                    insecure_ctx = ssl._create_unverified_context()
                    with urlopen(request_or_url, timeout=timeout, context=insecure_ctx) as response:
                        return response.read().decode("utf-8", errors="ignore")
                except (URLError, TimeoutError, UnicodeDecodeError):
                    pass
        except (TimeoutError, UnicodeDecodeError):
            pass

        if attempt < retries:
            time.sleep(0.35 * (attempt + 1))

    return None


def parse_imd_base_date(html: str) -> date | None:
    heading_match = re.search(r"<h2>\s*([A-Za-z]+ [A-Za-z]+ \d{1,2}, \d{4})\s*</h2>", html, flags=re.IGNORECASE)
    if heading_match:
        date_text = re.sub(r"\s+", " ", heading_match.group(1).strip())
        try:
            return datetime.strptime(date_text, "%A %B %d, %Y").date()
        except ValueError:
            pass

    observed_match = re.search(r"Maximum\s*:.*?\((\d{4}-\d{2}-\d{2})\)", html, flags=re.IGNORECASE | re.DOTALL)
    if observed_match:
        try:
            return datetime.strptime(observed_match.group(1), "%Y-%m-%d").date()
        except ValueError:
            return None
    return None


def infer_rain_probability_from_imd_description(description: str) -> float:
    text = description.lower()
    if any(word in text for word in ["thunder", "storm", "shower", "drizzle", "rain"]):
        return 75.0
    if any(word in text for word in ["overcast", "cloud"]):
        return 45.0
    if any(word in text for word in ["fog", "mist", "haze"]):
        return 25.0
    return 15.0


def parse_imd_forecast_payload(html: str) -> tuple[list[tuple[float, float]], list[str], float | None]:
    pair_matches = re.findall(
        r'<td\s+id="red">\s*([0-9]+(?:\.[0-9]+)?)\s*</td>\s*<td\s+id="blue">\s*([0-9]+(?:\.[0-9]+)?)\s*</td>',
        html,
        flags=re.IGNORECASE,
    )
    forecast_pairs: list[tuple[float, float]] = []
    for max_text, min_text in pair_matches:
        try:
            forecast_pairs.append((float(max_text), float(min_text)))
        except ValueError:
            continue

    icon_titles = [title.strip() for title in re.findall(r'TITLE="([^"]+)"', html, flags=re.IGNORECASE) if title.strip()]

    past_rain_mm: float | None = None
    rain_match = re.search(
        r"Past24\s*Hrs\s*Rainfall[^:]*:\s*([A-Za-z0-9\.\-]+)",
        html,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if rain_match:
        rain_text = rain_match.group(1).strip()
        if rain_text.upper() == "NIL":
            past_rain_mm = 0.0
        else:
            try:
                past_rain_mm = float(rain_text)
            except ValueError:
                past_rain_mm = None

    return forecast_pairs, icon_titles, past_rain_mm


def fetch_imd_city_weather(target_date: str, config: dict[str, Any], thresholds: dict[str, float]) -> WeatherInfo | None:
    default_url = "https://city.imd.gov.in/citywx/new/new_tour1.php?id=42369"
    imd_url = str(config.get("imd_city_forecast_url", default_url)).strip() or default_url
    timezone_name = str(config.get("weather_timezone", config.get("timezone", "Asia/Kolkata")))

    try:
        target = datetime.strptime(target_date, "%Y-%m-%d").date()
    except ValueError:
        return None

    request = Request(
        imd_url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; FoodMenuBot/1.0)",
            "Accept-Language": "en-US,en;q=0.9",
        },
    )

    html = fetch_url_text(request, timeout=15, retries=2, allow_insecure_ssl=True)
    if not html:
        return None

    base_date = parse_imd_base_date(html)
    if base_date is None:
        try:
            base_date = datetime.now(ZoneInfo(timezone_name)).date()
        except Exception:
            base_date = datetime.now(ZoneInfo("Asia/Kolkata")).date()

    day_offset = (target - base_date).days
    if day_offset < 0:
        return None

    forecast_pairs, icon_titles, past_rain_mm = parse_imd_forecast_payload(html)
    if day_offset >= len(forecast_pairs):
        return None

    max_temp, min_temp = forecast_pairs[day_offset]
    morning_temp = min_temp

    if day_offset < len(icon_titles):
        rain_pct = infer_rain_probability_from_imd_description(icon_titles[day_offset])
    elif day_offset == 0 and past_rain_mm is not None:
        rain_pct = 75.0 if past_rain_mm > 0 else 15.0
    else:
        rain_pct = 20.0

    return WeatherInfo(
        morning_temp_c=morning_temp,
        max_temp_c=max_temp,
        rain_probability_pct=rain_pct,
        is_rainy=rain_pct >= thresholds["rain_probability_high_pct"],
        is_extreme_cold=max_temp <= thresholds["extreme_cold_max_c"],
        is_extreme_hot=max_temp >= thresholds["extreme_hot_min_c"],
        source_hi="IMD (city.imd.gov.in)",
    )


def load_manual_weather(target_date: str, thresholds: dict[str, float]) -> WeatherInfo | None:
    if not MANUAL_WEATHER_FILE.exists():
        return None

    data = load_json(MANUAL_WEATHER_FILE)
    if not isinstance(data, dict):
        return None

    row = data.get(target_date)
    if not isinstance(row, dict):
        return None

    try:
        morning_temp = float(row["morning_temp_c"])
        max_temp = float(row["max_temp_c"])
        rain_pct = float(row.get("rain_probability_pct", 0))
    except (KeyError, TypeError, ValueError):
        return None

    source_hi = str(row.get("source_hi", "मैनुअल अनुमान"))
    return WeatherInfo(
        morning_temp_c=morning_temp,
        max_temp_c=max_temp,
        rain_probability_pct=rain_pct,
        is_rainy=rain_pct >= thresholds["rain_probability_high_pct"],
        is_extreme_cold=max_temp <= thresholds["extreme_cold_max_c"],
        is_extreme_hot=max_temp >= thresholds["extreme_hot_min_c"],
        source_hi=source_hi,
    )


def fetch_open_meteo_weather(target_date: str, config: dict[str, Any], thresholds: dict[str, float]) -> WeatherInfo | None:
    lat = config.get("weather_lat")
    lon = config.get("weather_lon")
    timezone_name = config.get("weather_timezone", config.get("timezone", "Asia/Kolkata"))

    if lat is None or lon is None:
        return None

    params = {
        "latitude": str(lat),
        "longitude": str(lon),
        "hourly": "temperature_2m",
        "daily": "temperature_2m_max,precipitation_probability_max",
        "timezone": timezone_name,
        "start_date": target_date,
        "end_date": target_date,
    }
    url = "https://api.open-meteo.com/v1/forecast?" + urlencode(params)

    request = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; FoodMenuBot/1.0)",
            "Accept-Language": "en-US,en;q=0.9",
        },
    )
    payload_text = fetch_url_text(request, timeout=12, retries=2, allow_insecure_ssl=True)
    if not payload_text:
        return None
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError:
        return None

    daily = payload.get("daily", {}) if isinstance(payload, dict) else {}
    hourly = payload.get("hourly", {}) if isinstance(payload, dict) else {}

    try:
        max_temp = float(daily.get("temperature_2m_max", [None])[0])
    except (TypeError, ValueError, IndexError):
        return None

    try:
        rain_pct = float(daily.get("precipitation_probability_max", [0])[0])
    except (TypeError, ValueError, IndexError):
        rain_pct = 0.0

    hourly_times = hourly.get("time", []) if isinstance(hourly, dict) else []
    hourly_temps = hourly.get("temperature_2m", []) if isinstance(hourly, dict) else []

    morning_temp = None
    target_hour = f"{target_date}T08:00"
    for idx, val in enumerate(hourly_times):
        if isinstance(val, str) and val == target_hour:
            try:
                morning_temp = float(hourly_temps[idx])
            except (TypeError, ValueError, IndexError):
                morning_temp = None
            break

    if morning_temp is None:
        for idx, val in enumerate(hourly_times):
            if isinstance(val, str) and val.startswith(f"{target_date}T"):
                try:
                    morning_temp = float(hourly_temps[idx])
                    break
                except (TypeError, ValueError, IndexError):
                    continue

    if morning_temp is None:
        morning_temp = max_temp

    return WeatherInfo(
        morning_temp_c=morning_temp,
        max_temp_c=max_temp,
        rain_probability_pct=rain_pct,
        is_rainy=rain_pct >= thresholds["rain_probability_high_pct"],
        is_extreme_cold=max_temp <= thresholds["extreme_cold_max_c"],
        is_extreme_hot=max_temp >= thresholds["extreme_hot_min_c"],
        source_hi="Open-Meteo",
    )


def resolve_weather_info(target_date: str, config: dict[str, Any], thresholds: dict[str, float]) -> WeatherInfo | None:
    manual = load_manual_weather(target_date, thresholds)
    if manual:
        return manual

    source_order_raw = config.get("weather_source_order", ["imd", "open_meteo"])
    if isinstance(source_order_raw, list):
        source_order = [str(src).strip().lower() for src in source_order_raw if str(src).strip()]
    else:
        source_order = ["imd", "open_meteo"]

    if not source_order:
        source_order = ["imd", "open_meteo"]

    for source in source_order:
        if source == "imd":
            imd_weather = fetch_imd_city_weather(target_date, config, thresholds)
            if imd_weather is not None:
                return imd_weather
        elif source in {"open_meteo", "open-meteo", "openmeteo"}:
            meteo_weather = fetch_open_meteo_weather(target_date, config, thresholds)
            if meteo_weather is not None:
                return meteo_weather

    return None


def derive_weather_rules(weather: WeatherInfo, thresholds: dict[str, float]) -> WeatherRules:
    preferred: set[str] = set()
    avoid: set[str] = set()

    if weather.is_rainy:
        preferred.update({"rain_friendly", "comfort_hot"})
        avoid.update({"cold_served"})

    if weather.max_temp_c <= thresholds["extreme_cold_max_c"]:
        preferred.update({"comfort_hot", "winter_friendly", "heavy"})
        avoid.update({"cold_served", "cooling"})
    elif weather.max_temp_c <= thresholds["cold_max_c"]:
        preferred.update({"comfort_hot", "winter_friendly"})
        avoid.update({"cold_served"})

    if weather.max_temp_c >= thresholds["hot_min_c"]:
        preferred.update({"light", "hydrating", "summer_friendly"})
        avoid.update({"heavy", "fried", "very_spicy"})

    if weather.max_temp_c >= thresholds["extreme_hot_min_c"]:
        preferred.update({"light", "hydrating", "summer_friendly"})
        avoid.update({"heavy", "fried", "very_spicy", "comfort_hot"})

    return WeatherRules(preferred_tags=preferred, avoid_tags=avoid)


def infer_tags_for_item(item: str) -> list[str]:
    tags: set[str] = set()
    text = item.lower()

    if any(word in text for word in ["खिचड़ी", "पराठा", "बाटी", "कचौड़ी", "पुरी", "कुलचे", "चूरमा", "चावल"]):
        tags.add("heavy")
    if any(word in text for word in ["पकौड़े", "वड़ा", "पुरी", "कचौड़ी", "पराठा"]):
        tags.add("fried")
    if any(word in text for word in ["छाछ", "रायता", "दही", "नारियल", "चटनी"]):
        tags.add("hydrating")
    if any(word in text for word in ["सांभर", "दाल", "साग", "उपमा", "रोटी", "मेथी", "सरसों", "बाजरा", "मक्की"]):
        tags.add("comfort_hot")
    if any(word in text for word in ["मेथी", "सरसों", "बाजरा", "मक्की"]):
        tags.add("winter_friendly")
    if any(word in text for word in ["रायता", "दही", "नारियल", "उपमा"]):
        tags.add("summer_friendly")
    if any(word in text for word in ["उपमा", "खिचड़ी", "सांभर", "पकौड़े"]):
        tags.add("rain_friendly")
    if any(word in text for word in ["उपमा", "दाल", "रोटी"]):
        tags.add("light")

    return sorted(tags)


def bootstrap_weather_tags(items: list[str]) -> dict[str, list[str]]:
    return {item: infer_tags_for_item(item) for item in items}


def load_weather_tags(items: list[str]) -> dict[str, list[str]]:
    if not WEATHER_TAGS_FILE.exists():
        data = bootstrap_weather_tags(items)
        write_json(WEATHER_TAGS_FILE, data)
        return data

    raw = load_json(WEATHER_TAGS_FILE)
    if not isinstance(raw, dict):
        return bootstrap_weather_tags(items)

    normalized: dict[str, list[str]] = {}
    for item in items:
        tags = raw.get(item, [])
        if isinstance(tags, list):
            normalized[item] = sorted({str(t).strip() for t in tags if isinstance(t, str) and t.strip()})
        else:
            normalized[item] = []
    return normalized


def apply_weather_filter(
    pool: list[str],
    weather_rules: WeatherRules,
    weather_tags: dict[str, list[str]],
    warn_bucket: set[str],
) -> list[str]:
    preferred = weather_rules.preferred_tags
    avoid = weather_rules.avoid_tags

    def tags_for(item: str) -> set[str]:
        tags = set(weather_tags.get(item, []))
        if not tags:
            warn_bucket.add(item)
        return tags

    def in_stage_1(item: str) -> bool:
        tags = tags_for(item)
        if not tags:
            return True
        has_avoid = bool(tags & avoid)
        has_pref = bool(tags & preferred)
        if preferred:
            return has_pref and not has_avoid
        return not has_avoid

    stage1 = [item for item in pool if in_stage_1(item)]
    if stage1:
        return stage1

    def in_stage_2(item: str) -> bool:
        tags = tags_for(item)
        if not tags:
            return True
        if preferred:
            return bool(tags & preferred)
        return True

    stage2 = [item for item in pool if in_stage_2(item)]
    if stage2:
        return stage2

    return pool[:]


def lightness_score(
    item: str,
    weather_tags: dict[str, list[str]],
    heavy_light_classification: dict[str, str] | None = None,
) -> int:
    if heavy_light_classification:
        normalized_item = normalize_item_key(item)
        manual = heavy_light_classification.get(normalized_item)
        if manual == "heavy":
            return 1
        if manual == "light":
            return -1

    tags = set(weather_tags.get(item, []))
    score = 0
    if "heavy" in tags:
        score += 3
    if "fried" in tags:
        score += 3
    if "comfort_hot" in tags:
        score += 1
    if "winter_friendly" in tags:
        score += 1
    if "light" in tags:
        score -= 3
    if "hydrating" in tags:
        score -= 1
    if "summer_friendly" in tags:
        score -= 1
    return score


def is_heavy_item(
    item: str,
    weather_tags: dict[str, list[str]],
    heavy_light_classification: dict[str, str] | None = None,
) -> bool:
    return lightness_score(item, weather_tags, heavy_light_classification) > 0


def apply_lighter_preference(
    pool: list[str],
    weather_tags: dict[str, list[str]],
    heavy_light_classification: dict[str, str] | None = None,
) -> list[str]:
    if len(pool) <= 1:
        return pool
    scored = [(lightness_score(item, weather_tags, heavy_light_classification), item) for item in pool]
    min_score = min(score for score, _ in scored)
    lighter_pool = [item for score, item in scored if score == min_score]
    return lighter_pool if lighter_pool else pool


def choose_item(
    items: list[str],
    ekadashi: EkadashiInfo,
    recent_block_set: set[str],
    keywords: list[str],
    disallowed_keywords: list[str],
    fallback_policy: str,
    seed_key: str,
    weather_rules: WeatherRules | None,
    weather_tags: dict[str, list[str]],
    warn_bucket: set[str],
    prefer_lighter: bool,
    light_fallback_items: list[str],
    max_lightness_score: int | None = None,
    heavy_light_classification: dict[str, str] | None = None,
) -> str:
    full_pool = items[:] if items else light_fallback_items[:]

    if ekadashi.is_ekadashi:
        base_pool = [item for item in full_pool if not is_blocked_item(item, keywords)]
    else:
        base_pool = full_pool[:]

    if disallowed_keywords:
        base_pool = [item for item in base_pool if not is_blocked_item(item, disallowed_keywords)]

    if not base_pool and ekadashi.is_ekadashi and fallback_policy == "fallback_full_menu":
        base_pool = full_pool[:]

    if not base_pool:
        base_pool = light_fallback_items[:]

    if not base_pool:
        raise RuntimeError("No menu item available after applying rules")

    pool = apply_repeat_rule(base_pool, recent_block_set)

    if weather_rules is not None:
        weather_pool = apply_weather_filter(pool, weather_rules, weather_tags, warn_bucket)
        if weather_pool:
            pool = weather_pool

    if prefer_lighter:
        pool = apply_lighter_preference(pool, weather_tags, heavy_light_classification)

    if max_lightness_score is not None:
        capped_pool = [
            item
            for item in pool
            if lightness_score(item, weather_tags, heavy_light_classification) <= max_lightness_score
        ]
        if capped_pool:
            pool = capped_pool

    seed_int = int(hashlib.sha256(seed_key.encode("utf-8")).hexdigest(), 16)
    rng = random.Random(seed_int)
    return rng.choice(pool)


def update_history(
    history: list[dict[str, str]],
    target_date: str,
    breakfast_item: str,
    meal_item: str,
    keep_days: int,
) -> list[dict[str, str]]:
    updated = [row for row in history if row.get("date") != target_date]
    updated.append({"date": target_date, "breakfast": breakfast_item, "meal": meal_item})

    cutoff = datetime.strptime(target_date, "%Y-%m-%d").date() - timedelta(days=max(keep_days, 7) + 30)

    retained: list[dict[str, str]] = []
    for row in updated:
        try:
            row_date = datetime.strptime(row["date"], "%Y-%m-%d").date()
        except (ValueError, KeyError):
            continue
        if row_date >= cutoff:
            retained.append(row)

    retained.sort(key=lambda r: r["date"])
    return retained


def validate_menu_list(menu: Any, file_label: str) -> list[str]:
    if not isinstance(menu, list) or not all(isinstance(i, str) and i.strip() for i in menu):
        raise ValueError(f"{file_label} must be a non-empty array of strings")
    return menu


def dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def normalize_ritu_key(ritu_hi: str) -> str:
    cleaned = ritu_hi.replace(" ", "")
    if "हेमंतऋतु" in cleaned or "हेमन्तऋतु" in cleaned or "हेमंत" in cleaned or "हेमन्त" in cleaned:
        return "hemant"
    if "शरदऋतु" in cleaned or "शरद" in cleaned:
        return "sharad"
    if "वर्षाऋतु" in cleaned or "वर्षा" in cleaned:
        return "varsha"
    if "ग्रीष्मऋतु" in cleaned or "ग्रीष्म" in cleaned:
        return "grishm"
    if "वसंत" in cleaned or "बसंत" in cleaned:
        return "vasant"
    return "shishir"


def should_show_weather_line(weather: WeatherInfo, mode: str) -> bool:
    if mode == "always":
        return True
    if mode == "never":
        return False
    return weather.is_rainy or weather.is_extreme_cold or weather.is_extreme_hot


def build_weather_line(weather: WeatherInfo, city_hi: str) -> str:
    return (
        f"*मौसम:* {city_hi} - सुबह {weather.morning_temp_c:.1f}°C, "
        f"अधिकतम {weather.max_temp_c:.1f}°C, वर्षा संभावना {weather.rain_probability_pct:.0f}% "
        f"({weather.source_hi})"
    )


def main() -> int:
    args = parse_args()

    config = load_json(CONFIG_FILE)
    breakfast_shishir_items = validate_menu_list(load_json(BREAKFAST_SHISHIR_FILE), "breakfast_shishir.json")
    meal_shishir_items = validate_menu_list(load_json(MENU_SHISHIR_FILE), "menu_shishir.json")
    breakfast_vasant_items = (
        validate_menu_list(load_json(BREAKFAST_VASANT_FILE), "breakfast_vasant.json")
        if BREAKFAST_VASANT_FILE.exists()
        else []
    )
    meal_vasant_items = (
        validate_menu_list(load_json(MENU_VASANT_FILE), "menu_vasant.json")
        if MENU_VASANT_FILE.exists()
        else []
    )
    breakfast_grishm_items = (
        dedupe_preserve_order(validate_menu_list(load_json(BREAKFAST_GRISHM_FILE), "breakfast_grishm.json"))
        if BREAKFAST_GRISHM_FILE.exists()
        else []
    )
    meal_grishm_items = (
        validate_menu_list(load_json(MENU_GRISHM_FILE), "menu_grishm.json")
        if MENU_GRISHM_FILE.exists()
        else []
    )
    breakfast_varsha_items = (
        validate_menu_list(load_json(BREAKFAST_VARSHA_FILE), "breakfast_varsha.json")
        if BREAKFAST_VARSHA_FILE.exists()
        else []
    )
    meal_varsha_items = (
        validate_menu_list(load_json(MENU_VARSHA_FILE), "menu_varsha.json")
        if MENU_VARSHA_FILE.exists()
        else []
    )
    breakfast_sharad_items = (
        dedupe_preserve_order(validate_menu_list(load_json(BREAKFAST_SHARAD_FILE), "breakfast_sharad.json"))
        if BREAKFAST_SHARAD_FILE.exists()
        else []
    )
    meal_sharad_items = (
        validate_menu_list(load_json(MENU_SHARAD_FILE), "menu_sharad.json")
        if MENU_SHARAD_FILE.exists()
        else []
    )
    breakfast_hemant_items = (
        dedupe_preserve_order(validate_menu_list(load_json(BREAKFAST_HEMANT_FILE), "breakfast_hemant.json"))
        if BREAKFAST_HEMANT_FILE.exists()
        else []
    )
    meal_hemant_items = (
        validate_menu_list(load_json(MENU_HEMANT_FILE), "menu_hemant.json")
        if MENU_HEMANT_FILE.exists()
        else []
    )
    ekadashi_data = load_json(EKADASHI_FILE)
    panchang_data: Any = {}
    panchang_source_error_note: str | None = None
    if PANCHANG_FILE.exists():
        try:
            panchang_data = load_json(PANCHANG_FILE)
        except (OSError, json.JSONDecodeError) as exc:
            panchang_source_error_note = f"[त्रुटि] पंचांग फ़ाइल पढ़ने/पार्सिंग में समस्या: {exc}"
    else:
        panchang_source_error_note = "[अनुपलब्ध] पंचांग फ़ाइल उपलब्ध नहीं"
    festivals_data = load_json(FESTIVALS_FILE) if FESTIVALS_FILE.exists() else {}
    history = normalize_history(load_json(HISTORY_FILE))
    missing_data_notes: list[str] = []

    all_items = (
        breakfast_shishir_items
        + meal_shishir_items
        + breakfast_vasant_items
        + meal_vasant_items
        + breakfast_grishm_items
        + meal_grishm_items
        + breakfast_varsha_items
        + meal_varsha_items
        + breakfast_sharad_items
        + meal_sharad_items
        + breakfast_hemant_items
        + meal_hemant_items
    )

    if args.bootstrap_weather_tags:
        write_json(WEATHER_TAGS_FILE, bootstrap_weather_tags(all_items))
        print(f"Weather tags generated at {WEATHER_TAGS_FILE}")
        return 0

    timezone_name = config.get("timezone", "Asia/Kolkata")
    repeat_window_days = int(config.get("repeat_window_days", 7))
    fallback_policy = config.get("empty_filtered_pool_policy", "fallback_full_menu")
    keywords = config.get("ekadashi_block_keywords", [])
    default_ritu = config.get("ritu_hi", "शिशिर")

    if not isinstance(keywords, list) or not all(isinstance(k, str) for k in keywords):
        raise ValueError("ekadashi_block_keywords must be an array of strings")

    target_date = resolve_date(args.date, timezone_name)
    target_date_str = target_date.strftime("%Y-%m-%d")

    weather_enabled = bool(config.get("weather_enabled", True))
    weather_mode = str(config.get("weather_show_mode", "rain_or_extreme_only"))
    weather_city_hi = str(config.get("weather_city_hi", "लखनऊ"))
    thresholds = parse_weather_thresholds(config)
    light_fallback_items_raw = config.get("light_fallback_items", DEFAULT_LIGHT_FALLBACK_ITEMS)
    if isinstance(light_fallback_items_raw, list):
        light_fallback_items = [str(item).strip() for item in light_fallback_items_raw if str(item).strip()]
    else:
        light_fallback_items = DEFAULT_LIGHT_FALLBACK_ITEMS[:]
    if not light_fallback_items:
        light_fallback_items = DEFAULT_LIGHT_FALLBACK_ITEMS[:]

    weather_info: WeatherInfo | None = None
    weather_rules: WeatherRules | None = None
    if weather_enabled:
        weather_info = resolve_weather_info(target_date_str, config, thresholds)
        if weather_info is not None:
            weather_rules = derive_weather_rules(weather_info, thresholds)
        else:
            missing_data_notes.append("मौसम डेटा उपलब्ध नहीं (मैनुअल/ओपन-मेटियो)")

    weather_tags = load_weather_tags(all_items)
    heavy_light_classification, classification_note = load_heavy_light_classification(
        HEAVY_LIGHT_CLASSIFICATION_FILE, all_items
    )
    if classification_note:
        missing_data_notes.append(classification_note)

    ekadashi = get_ekadashi_info(target_date_str, ekadashi_data)
    if panchang_source_error_note:
        panchang_lookup = PanchangLookupResult(
            row=None,
            status="source_load_error",
            detail_hi=panchang_source_error_note,
            total_rows=0,
            valid_date_rows=0,
            invalid_date_rows=0,
        )
    else:
        panchang_lookup = lookup_panchang_entry_for_date(target_date, timezone_name, panchang_data)

    panchang_info = resolve_panchang_info(target_date, ekadashi, panchang_lookup.row, default_ritu)
    festival_info = resolve_festival_info(target_date_str, festivals_data)

    maah_mapped_ritu_key = resolve_ritu_key_from_lunar_month(panchang_info.maah_hi)
    if maah_mapped_ritu_key is not None:
        base_ritu_key = maah_mapped_ritu_key
    else:
        base_ritu_key = normalize_ritu_key(panchang_info.ritu_hi)
        if panchang_lookup.status == "ok":
            missing_data_notes.append("[त्रुटि] पंचांग माह नाम से ऋतु mapping नहीं हो सकी")
        else:
            missing_data_notes.append("[अनुपलब्ध] माह-आधारित ऋतु निर्धारण हेतु पंचांग माह डेटा उपलब्ध नहीं")

    display_ritu_hi = SEASON_HI.get(base_ritu_key, panchang_info.ritu_hi)
    paksha_hint = (
        str(panchang_lookup.row.get("paksha_hi")).strip()
        if panchang_lookup.row and isinstance(panchang_lookup.row.get("paksha_hi"), str)
        else None
    )
    shringdhara_info = detect_shringdhara_observance(panchang_info.maah_hi, panchang_info.tithi_hi, paksha_hint)
    if shringdhara_info.missing_note_hi:
        missing_data_notes.append(shringdhara_info.missing_note_hi)

    transition_window_days = int(config.get("season_transition_window_days", 15))
    transition_plan = resolve_transition_plan(
        target_date=target_date,
        current_key=base_ritu_key,
        weather=weather_info,
        thresholds=thresholds,
        transition_window_days=transition_window_days,
    )
    ritu_key = base_ritu_key if shringdhara_info.active else transition_plan.selected_key
    if panchang_lookup.status == "date_missing":
        missing_data_notes.append("[अनुपलब्ध] पंचांग स्रोत में इस तिथि की प्रविष्टि नहीं है")
    elif panchang_lookup.status in {"source_load_error", "source_invalid", "mapping_error", "lookup_error"}:
        missing_data_notes.append(panchang_lookup.detail_hi)

    if panchang_lookup.invalid_date_rows > 0:
        missing_data_notes.append(
            f"[त्रुटि] पंचांग में {panchang_lookup.invalid_date_rows} प्रविष्टियों का date फ़ॉर्मैट अमान्य/अपठनीय है"
        )

    if panchang_info.tithi_hi == "अज्ञात" and panchang_lookup.status == "ok":
        missing_data_notes.append("[अनुपलब्ध] पंचांग प्रविष्टि में 'तिथि (पंचांग)' अज्ञात/रिक्त है")

    if ritu_key == "grishm":
        breakfast_items = breakfast_grishm_items[:]
        meal_items = meal_grishm_items[:]
        if not meal_grishm_items:
            missing_data_notes.append("ग्रीष्म भोजन सूची उपलब्ध नहीं (fallback: हल्का डिफ़ॉल्ट)")
        if not breakfast_grishm_items:
            missing_data_notes.append("ग्रीष्म नाश्ता सूची उपलब्ध नहीं (fallback: हल्का डिफ़ॉल्ट)")
    elif ritu_key == "hemant":
        breakfast_items = breakfast_hemant_items[:]
        meal_items = meal_hemant_items[:]
        if not meal_hemant_items:
            missing_data_notes.append("हेमंत भोजन सूची उपलब्ध नहीं (fallback: हल्का डिफ़ॉल्ट)")
        if not breakfast_hemant_items:
            missing_data_notes.append("हेमंत नाश्ता सूची उपलब्ध नहीं (fallback: हल्का डिफ़ॉल्ट)")
    elif ritu_key == "sharad":
        breakfast_items = breakfast_sharad_items[:]
        meal_items = meal_sharad_items[:]
        if not meal_sharad_items:
            missing_data_notes.append("शरद भोजन सूची उपलब्ध नहीं (fallback: हल्का डिफ़ॉल्ट)")
        if not breakfast_sharad_items:
            missing_data_notes.append("शरद नाश्ता सूची उपलब्ध नहीं (fallback: हल्का डिफ़ॉल्ट)")
    elif ritu_key == "varsha":
        breakfast_items = breakfast_varsha_items[:]
        meal_items = meal_varsha_items[:]
        if not meal_varsha_items:
            missing_data_notes.append("वर्षा भोजन सूची उपलब्ध नहीं (fallback: हल्का डिफ़ॉल्ट)")
        if not breakfast_varsha_items:
            missing_data_notes.append("वर्षा नाश्ता सूची उपलब्ध नहीं (fallback: हल्का डिफ़ॉल्ट)")
    elif ritu_key == "vasant":
        breakfast_items = breakfast_vasant_items[:]
        meal_items = meal_vasant_items[:]
        if not meal_vasant_items:
            missing_data_notes.append("वसंत भोजन सूची उपलब्ध नहीं (fallback: हल्का डिफ़ॉल्ट)")
        if not breakfast_vasant_items:
            missing_data_notes.append("वसंत नाश्ता सूची उपलब्ध नहीं (fallback: हल्का डिफ़ॉल्ट)")
    else:
        breakfast_items = breakfast_shishir_items
        meal_items = meal_shishir_items

    if not breakfast_items:
        breakfast_items = light_fallback_items[:]
        if not any("नाश्ता सूची उपलब्ध नहीं" in note for note in missing_data_notes):
            missing_data_notes.append("नाश्ता विकल्प उपलब्ध नहीं (fallback: हल्का डिफ़ॉल्ट)")
    if not meal_items:
        meal_items = light_fallback_items[:]
        if not any("भोजन सूची उपलब्ध नहीं" in note for note in missing_data_notes):
            missing_data_notes.append("भोजन विकल्प उपलब्ध नहीं (fallback: हल्का डिफ़ॉल्ट)")

    if ritu_key == "varsha":
        disallowed_keywords = VARSHA_BANNED_KEYWORDS
    elif ritu_key == "hemant":
        disallowed_keywords = HEMANT_BANNED_KEYWORDS
    elif ritu_key == "sharad":
        disallowed_keywords = SHARAD_BANNED_KEYWORDS
    else:
        disallowed_keywords = []

    breakfast_recent = recent_items(history, target_date, repeat_window_days, "breakfast")
    meal_recent = recent_items(history, target_date, repeat_window_days, "meal")

    warning_items: set[str] = set()

    selected_observance_item: str | None = None
    if shringdhara_info.active:
        observance_items = dedupe_preserve_order(meal_items if meal_items else breakfast_items)
        observance_recent = breakfast_recent | meal_recent
        selected_observance_item = choose_item(
            items=observance_items,
            ekadashi=ekadashi,
            recent_block_set=observance_recent,
            keywords=keywords,
            disallowed_keywords=disallowed_keywords,
            fallback_policy=fallback_policy,
            seed_key=f"{target_date_str}:shringdhara",
            weather_rules=weather_rules,
            weather_tags=weather_tags,
            warn_bucket=warning_items,
            prefer_lighter=True,
            light_fallback_items=light_fallback_items,
            heavy_light_classification=heavy_light_classification,
        )
        selected_breakfast = selected_observance_item
        selected_meal = selected_observance_item
    else:
        selected_breakfast = choose_item(
            items=breakfast_items,
            ekadashi=ekadashi,
            recent_block_set=breakfast_recent,
            keywords=keywords,
            disallowed_keywords=disallowed_keywords,
            fallback_policy=fallback_policy,
            seed_key=f"{target_date_str}:breakfast",
            weather_rules=weather_rules,
            weather_tags=weather_tags,
            warn_bucket=warning_items,
            prefer_lighter=transition_plan.prefer_lighter,
            light_fallback_items=light_fallback_items,
            heavy_light_classification=heavy_light_classification,
        )

        selected_meal = choose_item(
            items=meal_items,
            ekadashi=ekadashi,
            recent_block_set=meal_recent,
            keywords=keywords,
            disallowed_keywords=disallowed_keywords,
            fallback_policy=fallback_policy,
            seed_key=f"{target_date_str}:meal",
            weather_rules=weather_rules,
            weather_tags=weather_tags,
            warn_bucket=warning_items,
            prefer_lighter=transition_plan.prefer_lighter,
            light_fallback_items=light_fallback_items,
            heavy_light_classification=heavy_light_classification,
        )

        if is_heavy_item(selected_breakfast, weather_tags, heavy_light_classification) and is_heavy_item(
            selected_meal, weather_tags, heavy_light_classification
        ):
            rebalanced_breakfast = choose_item(
                items=breakfast_items,
                ekadashi=ekadashi,
                recent_block_set=breakfast_recent,
                keywords=keywords,
                disallowed_keywords=disallowed_keywords,
                fallback_policy=fallback_policy,
                seed_key=f"{target_date_str}:breakfast:light-balance",
                weather_rules=weather_rules,
                weather_tags=weather_tags,
                warn_bucket=warning_items,
                prefer_lighter=True,
                light_fallback_items=light_fallback_items,
                max_lightness_score=0,
                heavy_light_classification=heavy_light_classification,
            )
            if not is_heavy_item(rebalanced_breakfast, weather_tags, heavy_light_classification):
                selected_breakfast = rebalanced_breakfast
            else:
                rebalanced_meal = choose_item(
                    items=meal_items,
                    ekadashi=ekadashi,
                    recent_block_set=meal_recent,
                    keywords=keywords,
                    disallowed_keywords=disallowed_keywords,
                    fallback_policy=fallback_policy,
                    seed_key=f"{target_date_str}:meal:light-balance",
                    weather_rules=weather_rules,
                    weather_tags=weather_tags,
                    warn_bucket=warning_items,
                    prefer_lighter=True,
                    light_fallback_items=light_fallback_items,
                    max_lightness_score=0,
                    heavy_light_classification=heavy_light_classification,
                )
                if not is_heavy_item(rebalanced_meal, weather_tags, heavy_light_classification):
                    selected_meal = rebalanced_meal
                else:
                    forced_light = choose_item(
                        items=light_fallback_items,
                        ekadashi=ekadashi,
                        recent_block_set=set(),
                        keywords=keywords,
                        disallowed_keywords=disallowed_keywords,
                        fallback_policy=fallback_policy,
                        seed_key=f"{target_date_str}:forced-light",
                        weather_rules=None,
                        weather_tags=weather_tags,
                        warn_bucket=warning_items,
                        prefer_lighter=True,
                        light_fallback_items=light_fallback_items,
                        max_lightness_score=0,
                        heavy_light_classification=heavy_light_classification,
                    )
                    if lightness_score(
                        selected_breakfast, weather_tags, heavy_light_classification
                    ) >= lightness_score(selected_meal, weather_tags, heavy_light_classification):
                        selected_breakfast = forced_light
                    else:
                        selected_meal = forced_light

    if warning_items:
        print(
            "WARN: Untagged items were included in weather filtering: " + ", ".join(sorted(warning_items)),
            file=sys.stderr,
        )

    new_history = update_history(history, target_date_str, selected_breakfast, selected_meal, repeat_window_days)
    write_json(HISTORY_FILE, new_history)

    lines = [
        f"*तिथि:* {target_date_str}",
        f"*ऋतु:* {display_ritu_hi}",
        f"*माह:* {panchang_info.maah_hi}",
        f"*तिथि (पंचांग):* {panchang_info.tithi_hi}",
    ]
    if transition_plan.active and not shringdhara_info.active:
        lines.append(
            f"*ऋतु संक्रमण:* {SEASON_HI[transition_plan.current_key]} -> {SEASON_HI[transition_plan.upcoming_key]} ({transition_plan.days_to_next} दिन शेष)"
        )
        lines.append(f"*आज का ऋतु-आधार:* {SEASON_HI[transition_plan.selected_key]} ({transition_plan.reason})")
    festival_line = format_festival_line(festival_info)
    if festival_line:
        lines.append(festival_line)
    if shringdhara_info.active:
        lines.append("*विशेष अवधि:* शृंगधारा (यमराज की दाड़)")
        lines.append(f"*अवधि विवरण:* {shringdhara_info.reason_hi}")
        lines.append(f"*आज का हल्का सेवन:* {selected_observance_item}")
        lines.append("*शृंगधारा स्मरण:* " + SHRINGDHARA_DAILY_REMINDER)
        lines.append("*परंपरागत हल्का विकल्प:* " + SHRINGDHARA_LIGHT_NOTE)
    else:
        lines.append(f"*सुबह का नाश्ता:* {selected_breakfast}")
        lines.append(f"*आज का भोजन:* {selected_meal}")

    if ekadashi.is_ekadashi and ekadashi.name_hi:
        lines.append(f"*एकादशी:* {ekadashi.name_hi}")

    if weather_info is not None and should_show_weather_line(weather_info, weather_mode):
        lines.append(build_weather_line(weather_info, weather_city_hi))

    if not shringdhara_info.active:
        if ritu_key == "vasant":
            lines.append("*वसंत अनिवार्य साथ:* " + " / ".join(VASANT_REQUIRED_SIDES))
        if ritu_key == "grishm":
            lines.append("*ग्रीष्म नाश्ता अनिवार्य साथ:* " + " / ".join(GRISHM_BREAKFAST_REQUIRED_SIDES))
            lines.append("*ग्रीष्म भोजन अनिवार्य साथ:* " + " / ".join(GRISHM_MEAL_REQUIRED_SIDES))
        if ritu_key == "varsha":
            lines.append("*वर्षा नाश्ता अनिवार्य साथ:* " + " / ".join(VARSHA_COMMON_REQUIRED_SIDES))
            lines.append("*वर्षा भोजन अनिवार्य साथ:* " + " / ".join(VARSHA_COMMON_REQUIRED_SIDES))
            lines.append("*वर्षा वर्जित:* प्याज और दही पूर्णतः मना है")
        if ritu_key == "sharad":
            lines.append("*शरद अनिवार्य साथ:* " + " / ".join(SHARAD_COMMON_REQUIRED_SIDES))
            if any(token in (selected_breakfast + " " + selected_meal) for token in ["चावल", "राइस"]):
                lines.append("*शरद चावल नियम:* अगर चावल बन रहे हैं तो जीरा ज़रूर डालें")
            lines.append("*शरद वर्जित:* इमली, लौंग, लहसुन, प्याज़, काली मिर्च और गर्म मसाले नहीं")
            lines.append("*शरद अधिक उपयोग:* नारियल / खीर / पुदीना")
            lines.append("*शरद कम उपयोग:* छोले, टिंडा, करेला, टमाटर, आलू, अरबी, सरसों, पपीता, सौंफ़, हरी मिर्च, लाल मिर्च, अदरक, सौंठ, सरसों का तेल, कढ़ी, दही, लस्सी, शहद")
            lines.append("*शरद जल नियम:* चाँदी के ग्लास या मटके का जल दें")
            lines.append("*शरद रस:* मीठा / कसैला / कड़वा")
        if ritu_key == "hemant":
            lines.append("*हेमंत पूर्णतया निषिद्ध:* बासमती, मैदा, डिब्बा बंद, मोठ, दोबारा गर्म की हुई दाल/सब्ज़ी, जीरा, इमली, सॉस, अचार, कड़वा, कसैला, रिफाइंड, पनीर, एनर्जी ड्रिंक, प्याज़, दुबारा गर्म किया पानी")
            lines.append("*हेमंत जल नियम:* हमेशा गुनगुना, पीतल या तांबे में")

    if target_date.month == 1 and target_date.day == 1:
        lines.append("*वार्षिक स्मरण (1 जनवरी):* " + NEW_YEAR_KANJI_NOTE)

    if missing_data_notes:
        unique_notes: list[str] = []
        for note in missing_data_notes:
            if note not in unique_notes:
                unique_notes.append(note)
        lines.append("*डेटा अलर्ट:* " + " | ".join(unique_notes))

    output_text = "\r\n\r\n".join(lines)

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        f.write(output_text + "\n")

    print(output_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
