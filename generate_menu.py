#!/usr/bin/env python3
import argparse
import hashlib
import json
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import urlopen

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


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write("\n")


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


def get_panchang_entry_for_date(target_date: str, panchang_data: Any) -> dict[str, Any] | None:
    if isinstance(panchang_data, dict):
        direct = panchang_data.get(target_date)
        if isinstance(direct, dict):
            return direct

        entries = panchang_data.get("entries")
        if isinstance(entries, list):
            for row in entries:
                if isinstance(row, dict) and row.get("date") == target_date:
                    return row

    if isinstance(panchang_data, list):
        for row in panchang_data:
            if isinstance(row, dict) and row.get("date") == target_date:
                return row
    return None


def resolve_panchang_info(
    target_date: datetime.date,
    ekadashi: EkadashiInfo,
    panchang_data: Any,
    default_ritu: str,
) -> PanchangInfo:
    target_date_str = target_date.strftime("%Y-%m-%d")
    row = get_panchang_entry_for_date(target_date_str, panchang_data)

    if row:
        ritu_hi = str(row.get("ritu_hi", default_ritu)).strip() or default_ritu
        maah_hi = str(row.get("maah_hi", ekadashi.lunar_month_hi or GREGORIAN_MONTH_HI[target_date.month])).strip()
        tithi_hi = str(row.get("tithi_hi", "अज्ञात")).strip() or "अज्ञात"
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

    try:
        with urlopen(url, timeout=12) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (URLError, TimeoutError, json.JSONDecodeError):
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
    return fetch_open_meteo_weather(target_date, config, thresholds)


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
) -> str:
    full_pool = items[:]

    if ekadashi.is_ekadashi:
        base_pool = [item for item in full_pool if not is_blocked_item(item, keywords)]
    else:
        base_pool = full_pool[:]

    if disallowed_keywords:
        base_pool = [item for item in base_pool if not is_blocked_item(item, disallowed_keywords)]

    if not base_pool and ekadashi.is_ekadashi and fallback_policy == "fallback_full_menu":
        base_pool = full_pool[:]

    if not base_pool:
        raise RuntimeError("No menu item available after applying rules")

    pool = apply_repeat_rule(base_pool, recent_block_set)

    if weather_rules is not None:
        weather_pool = apply_weather_filter(pool, weather_rules, weather_tags, warn_bucket)
        if weather_pool:
            pool = weather_pool

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
    panchang_data = load_json(PANCHANG_FILE) if PANCHANG_FILE.exists() else {}
    festivals_data = load_json(FESTIVALS_FILE) if FESTIVALS_FILE.exists() else {}
    history = normalize_history(load_json(HISTORY_FILE))

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

    weather_info: WeatherInfo | None = None
    weather_rules: WeatherRules | None = None
    if weather_enabled:
        weather_info = resolve_weather_info(target_date_str, config, thresholds)
        if weather_info is not None:
            weather_rules = derive_weather_rules(weather_info, thresholds)

    weather_tags = load_weather_tags(all_items)

    ekadashi = get_ekadashi_info(target_date_str, ekadashi_data)
    panchang_info = resolve_panchang_info(target_date, ekadashi, panchang_data, default_ritu)
    festival_info = resolve_festival_info(target_date_str, festivals_data)
    ritu_key = normalize_ritu_key(panchang_info.ritu_hi)

    if ritu_key == "grishm":
        breakfast_items = breakfast_grishm_items or breakfast_shishir_items
        meal_items = meal_grishm_items or meal_shishir_items
    elif ritu_key == "hemant":
        breakfast_items = breakfast_hemant_items or breakfast_shishir_items
        meal_items = meal_hemant_items or meal_shishir_items
    elif ritu_key == "sharad":
        breakfast_items = breakfast_sharad_items or breakfast_shishir_items
        meal_items = meal_sharad_items or meal_shishir_items
    elif ritu_key == "varsha":
        breakfast_items = breakfast_varsha_items or breakfast_shishir_items
        meal_items = meal_varsha_items or meal_shishir_items
    elif ritu_key == "vasant":
        breakfast_items = breakfast_vasant_items or breakfast_shishir_items
        meal_items = meal_vasant_items or meal_shishir_items
    else:
        breakfast_items = breakfast_shishir_items
        meal_items = meal_shishir_items

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
    )

    if warning_items:
        print(
            "WARN: Untagged items were included in weather filtering: " + ", ".join(sorted(warning_items)),
            file=sys.stderr,
        )

    new_history = update_history(history, target_date_str, selected_breakfast, selected_meal, repeat_window_days)
    write_json(HISTORY_FILE, new_history)

    lines = [
        f"*तिथि:* {target_date_str}",
        f"*ऋतु:* {panchang_info.ritu_hi}",
        f"*माह:* {panchang_info.maah_hi}",
        f"*तिथि (पंचांग):* {panchang_info.tithi_hi}",
        f"*सुबह का नाश्ता:* {selected_breakfast}",
        f"*आज का भोजन:* {selected_meal}",
    ]
    festival_line = format_festival_line(festival_info)
    if festival_line:
        lines.insert(4, festival_line)

    if ekadashi.is_ekadashi and ekadashi.name_hi:
        lines.append(f"*एकादशी:* {ekadashi.name_hi}")

    if weather_info is not None and should_show_weather_line(weather_info, weather_mode):
        lines.append(build_weather_line(weather_info, weather_city_hi))

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

    output_text = "\r\n\r\n".join(lines)

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        f.write(output_text + "\n")

    print(output_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
