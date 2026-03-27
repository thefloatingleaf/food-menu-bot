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
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable
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
    suppress_regular_menu: bool = False
    special_menu_note_hi: str | None = None
    special_menu_lines_hi: list[str] | None = None


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


@dataclass
class DayContext:
    target_date: date
    target_date_str: str
    display_date_str: str
    ekadashi: EkadashiInfo
    panchang_lookup: PanchangLookupResult
    panchang_info: PanchangInfo
    festival_info: FestivalInfo
    display_ritu_hi: str
    shringdhara_info: ShringdharaInfo
    transition_plan: TransitionPlan
    ritu_key: str
    breakfast_items: list[str]
    meal_items: list[str]
    disallowed_keywords: list[str]
    weather_info: WeatherInfo | None
    weather_rules: WeatherRules | None
    breakfast_item_override: str | None


NAVRATRI_DAY_SPECIAL_NOTES = {
    1: "नवरात्रि दिवस 1, माँ शैलपुत्री: आज विशेष रूप से देसी घी ग्रहण करें या भोग में अर्पित करें।",
    2: "नवरात्रि दिवस 2, माँ ब्रह्मचारिणी: आज विशेष रूप से चीनी या मिश्री ग्रहण करें या भोग में अर्पित करें।",
    3: "नवरात्रि दिवस 3, माँ चंद्रघंटा: आज विशेष रूप से खीर ग्रहण करें या भोग में अर्पित करें।",
    4: "नवरात्रि दिवस 4, माँ कूष्मांडा: आज विशेष रूप से मालपुआ ग्रहण करें या भोग में अर्पित करें।",
    5: "नवरात्रि दिवस 5, माँ स्कंदमाता: आज विशेष रूप से केला ग्रहण करें या भोग में अर्पित करें।",
    6: "नवरात्रि दिवस 6, माँ कात्यायनी: आज विशेष रूप से शहद ग्रहण करें या भोग में अर्पित करें।",
    7: "नवरात्रि दिवस 7, माँ कालरात्रि: आज विशेष रूप से गुड़ ग्रहण करें या भोग में अर्पित करें।",
    8: "नवरात्रि दिवस 8, माँ महागौरी: आज विशेष रूप से नारियल ग्रहण करें या भोग में अर्पित करें।",
    9: "नवरात्रि दिवस 9, माँ सिद्धिदात्री: आज विशेष रूप से तिल या सात्त्विक भोग अर्पित करें।",
}

VIJAYADASHAMI_SPECIAL_NOTE = "विजयादशमी: आज विशेष रूप से सात्त्विक भोग अर्पित करें और शुभ कार्य आरंभ करें।"


NAVRATRI_ASHTAMI_SPECIAL_MENU_LINES = [
    "*विशेष अष्टमी मेनू:* अष्टमी के दिन नवरात्रि का भोजन निम्नानुसार बनाया जाए:",
    "1. काले चने — 4 कटोरी।",
    "2. छोले — 3 कटोरी।",
    "3. तरी वाले आलू — कुल 10 से 12 मध्यम आकार के आलू। आलू उचित आकार में काटे जाएँ, ताकि सभी के लिए पर्याप्त रहें।",
    "4. पूरी — 60 पूरी के लिए तैयारी रखें। आटा थोड़ा सख्त गूँथा जाए, ताकि पूरी अच्छी बने।",
    "5. कद्दू — लगभग 2 मध्यम आकार के कद्दू।",
    "6. सूखे आलू की सब्ज़ी — 3-4 आलू की।",
    "*ध्यान रहे:*",
    "1. काले चने और छोले रात में अच्छी तरह भिगो दिए जाएँ।",
    "2. किसी भी वस्तु में प्याज बिल्कुल न डाला जाए।",
    "3. पूरी के लिए आटा पहले से तैयार रखा जाए।",
    "*विशेष निर्देश:* किसी भी वस्तु में प्याज बिल्कुल न डाला जाए।",
    "*सूजी के हलवे के लिए निर्देश:*",
    "1. डेढ़ कटोरी चीनी में 4 कटोरी पानी डालकर अच्छी तरह मिला लें।",
    "2. दूसरी कड़ाही में 1 कटोरी सूजी लें।",
    "3. उसमें इतना घी डालें कि पूरी सूजी अच्छी तरह घी में डूब जाए।",
    "4. सूजी को भूरा होने तक अच्छी तरह भूनें। इतना भुनें कि रंग बादाम के बाहरी रंग के जैसा हो जाए।",
    "5. सूजी भुन जाने पर कटे हुए बादाम डालें।",
    "6. इसके बाद चीनी वाला पानी छान लें।",
    "7. छना हुआ पानी भुनी हुई सूजी वाली कड़ाही में धीरे-धीरे, थोड़ा-थोड़ा करके डालें।",
    "8. पानी डालते समय लगातार चलाते रहें, ताकि हलवा अच्छी तरह तैयार हो।",
]


def build_navratri_fallback_entries(
    start_date_str: str,
    total_days: int,
    *,
    festival_names: list[str],
    day_9_note_hi: str | None = None,
    final_day_extra_names: list[str] | None = None,
    final_day_note_hi: str | None = None,
) -> dict[str, dict[str, Any]]:
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
    entries: dict[str, dict[str, Any]] = {}

    for day_number in range(1, total_days + 1):
        current_date = start_date + timedelta(days=day_number - 1)
        current_names = festival_names[:]
        special_menu_note_hi: str | None
        special_menu_lines_hi: list[str] | None = None

        if day_number == 8:
            special_menu_note_hi = NAVRATRI_DAY_SPECIAL_NOTES[8]
            special_menu_lines_hi = NAVRATRI_ASHTAMI_SPECIAL_MENU_LINES[:]
        elif day_number == 9:
            special_menu_note_hi = day_9_note_hi or NAVRATRI_DAY_SPECIAL_NOTES[9]
        elif day_number == total_days and final_day_note_hi is not None:
            special_menu_note_hi = final_day_note_hi
            if final_day_extra_names:
                for name in final_day_extra_names:
                    if name not in current_names:
                        current_names.append(name)
        else:
            special_menu_note_hi = NAVRATRI_DAY_SPECIAL_NOTES.get(day_number)

        entries[current_date.strftime("%Y-%m-%d")] = {
            "hindu_hi": current_names,
            "special_menu_note_hi": special_menu_note_hi,
            "special_menu_lines_hi": special_menu_lines_hi,
        }

    return entries


NAVRATRI_FALLBACKS: dict[str, dict[str, Any]] = {}
NAVRATRI_FALLBACKS.update(
    build_navratri_fallback_entries(
        "2026-03-19",
        8,
        festival_names=["चैत्र नवरात्रि"],
        day_9_note_hi="नवरात्रि दिवस 9 / राम नवमी, माँ सिद्धिदात्री: आज विशेष रूप से तिल या व्रत-उपयोगी सात्त्विक भोग अर्पित करें।",
        final_day_extra_names=["राम नवमी"],
    )
)
NAVRATRI_FALLBACKS.update(
    build_navratri_fallback_entries(
        "2026-10-11",
        10,
        festival_names=["शारदीय नवरात्रि"],
        final_day_extra_names=["विजयादशमी"],
        final_day_note_hi=VIJAYADASHAMI_SPECIAL_NOTE,
    )
)
NAVRATRI_FALLBACKS.update(
    build_navratri_fallback_entries(
        "2027-04-07",
        9,
        festival_names=["चैत्र नवरात्रि"],
        day_9_note_hi="नवरात्रि दिवस 9 / राम नवमी, माँ सिद्धिदात्री: आज विशेष रूप से तिल या व्रत-उपयोगी सात्त्विक भोग अर्पित करें।",
        final_day_extra_names=["राम नवमी"],
    )
)
NAVRATRI_FALLBACKS.update(
    build_navratri_fallback_entries(
        "2027-09-30",
        10,
        festival_names=["शारदीय नवरात्रि"],
        final_day_extra_names=["विजयादशमी"],
        final_day_note_hi=VIJAYADASHAMI_SPECIAL_NOTE,
    )
)
NAVRATRI_FALLBACKS.update(
    build_navratri_fallback_entries(
        "2028-03-27",
        9,
        festival_names=["चैत्र नवरात्रि"],
        day_9_note_hi="नवरात्रि दिवस 9 / राम नवमी, माँ सिद्धिदात्री: आज विशेष रूप से तिल या व्रत-उपयोगी सात्त्विक भोग अर्पित करें।",
        final_day_extra_names=["राम नवमी"],
    )
)
NAVRATRI_FALLBACKS.update(
    build_navratri_fallback_entries(
        "2028-09-19",
        10,
        festival_names=["शारदीय नवरात्रि"],
        final_day_extra_names=["विजयादशमी"],
        final_day_note_hi=VIJAYADASHAMI_SPECIAL_NOTE,
    )
)


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

OVERNIGHT_BREAKFAST_ITEMS = {
    "पझैया सादम (Pazhaya Sadam): रात में 1 कटोरी कच्चे चावल अच्छी तरह धोकर सादा चावल पकाएँ। चावल पक जाने के बाद उन्हें मिट्टी या स्टील के बर्तन में निकालकर पूरी तरह ठंडा होने दें। ठंडा होने पर उसमें 2–3 कटोरी साफ पानी डालें ताकि चावल पूरी तरह पानी में डूब जाएँ। बर्तन ढककर इसे कमरे के तापमान पर पूरी रात (लगभग 10–12 घंटे) रहने दें। सुबह चावल और उसका पानी हल्का खट्टा हो जाएगा। उसी पानी सहित चावल को हाथ से हल्का मसल दें। इसमें ½ छोटी चम्मच नमक मिलाएँ। 4–5 छोटी कच्ची प्याज छीलकर डालें, 1–2 हरी मिर्च हल्की कुचलकर डालें। अब 2–3 बड़े चम्मच दही या लगभग ½ कटोरी पतली छाछ मिलाकर अच्छी तरह मिला दें। इसे ठंडा ही खाएँ। साथ में साधारण अचार रखें।",
    "पखाला भात (Pakhala Bhata): रात में 1 कटोरी कच्चे चावल धोकर सादा चावल पकाएँ। पकने के बाद चावल को मिट्टी या स्टील के बर्तन में निकालकर ठंडा होने दें। ठंडा होने पर उसमें 2–3 कटोरी पानी और 2 बड़े चम्मच दही डालें। बर्तन ढककर इसे कमरे के तापमान पर पूरी रात (लगभग 10–12 घंटे) रहने दें ताकि हल्का किण्वन हो जाए। सुबह इसमें ½ छोटी चम्मच नमक मिलाएँ। अब कढ़ाही में 1 छोटी चम्मच सरसों का तेल गरम करें। इसमें ½ छोटी चम्मच सरसों के दाने डालें। दाने चटकने पर 1 कटी हरी मिर्च और 4–5 करी पत्ते डालें। यह तड़का चावल पर डाल दें। ऊपर से ½ छोटी चम्मच भुना जीरा पाउडर डालें और हल्का मिला दें। इसे ठंडा परोसें। साथ में आलू भुजा, साग भुजा, उड़द की बड़ी या साधारण अचार रखें।",
}
OVERNIGHT_RICE_PREP_NOTE = (
    "कल सुबह के नाश्ते के लिए आज चावल बनाएं और कम से कम 1 कटोरी कच्चे चावल के बराबर "
    "अतिरिक्त पके हुए सादे चावल अलग रखें। इन्हें रात भर साफ पानी में डुबोकर रखें।"
)
MANGORE_ITEM_TOKENS = ("मंगौड़े", "मंगोड़े", "मंगोड़े", "मंगोरे", "mangore")
MANGORE_PREP_NOTE = (
    "फॉलोवर महोदय, कल के मंगौड़े के लिए आज रात 1 कटोरी धुली मूंग दाल चुनकर अच्छी तरह धो लें, "
    "फिर उसे 3-4 कटोरी पानी में पूरी तरह डुबोकर ढककर 8-10 घंटे भिगोकर रखें, "
    "ताकि सुबह मंगौड़े समय पर तैयार किए जा सकें।"
)
RICE_ITEM_TOKENS = ("चावल", "राइस", "भात")

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

BREAKFAST_REPEAT_ALIASES = {
    "आलू": ("आलू",),
    "उपमा": ("उपमा",),
    "इडली": ("इडली",),
    "उत्तपम": ("उत्तपम",),
    "कचौड़ी": ("कचौड़ी",),
    "गोभी": ("गोभी",),
    "चना": ("चना", "चने", "बेसन", "चना दाल", "सत्तू"),
    "चीला": ("चीला", "चिल्ला"),
    "डोसा": ("डोसा",),
    "दलिया": ("दलिया", "dalia"),
    "पकौड़े": ("पकौड़े",),
    "पझैया सादम": ("पझैया सादम", "pazhaya sadam"),
    "पखाला भात": ("पखाला भात", "pakhala bhata"),
    "पोहा": ("पोहा",),
    "पूरण पोली": ("पूरण पोली",),
    "पूरी": ("पूरी",),
    "फरे": ("फरे",),
    "बथुआ": ("बथुआ",),
    "मटर": ("मटर",),
    "मेथी": ("मेथी",),
    "मूंग": ("मूँग", "मूंग", "मूँगदाल", "मूंगदाल"),
    "मूली": ("मूली",),
    "वड़ा": ("वड़ा", "वड़े", "वडा"),
    "खिचड़ी": ("खिचड़ी",),
    "खीर": ("खीर",),
}
MEAL_REPEAT_ALIASES = {
    "आलू": ("आलू",),
    "कद्दू": ("कद्दू", "पेठा"),
    "करेला": ("करेला", "करेले"),
    "करोंदा": ("करोंदा",),
    "कुंदरु": ("कुंदरु",),
    "केला": ("केला", "केले"),
    "गोभी": ("गोभी",),
    "चकुंदर": ("चकुंदर", "चुकंदर"),
    "चौलाई": ("चौलाई",),
    "टमाटर": ("टमाटर",),
    "तोरई": ("तोरई", "तोरी"),
    "परवल": ("परवल",),
    "पालक": ("पालक",),
    "बथुआ": ("बथुआ",),
    "भिंडी": ("भिंडी",),
    "मटर": ("मटर",),
    "मेथी": ("मेथी",),
    "मूली": ("मूली",),
    "लौकी": ("लौकी", "दूधी", "दुधी"),
    "सरसों": ("सरसों",),
    "सहजन": ("सहजन", "सहजन की फली"),
}
CONSECUTIVE_DAY_REPEAT_NOTE = (
    "[अनुपलब्ध] लगातार दिनों में वही मुख्य नाश्ता या वही मुख्य सब्ज़ी दोहराने से बचने के लिए पर्याप्त विकल्प नहीं मिले"
)
REPEAT_FAMILY_BOUNDARY_CLASS = r"\u0900-\u097Fa-zA-Z"


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

LUNAR_MONTH_SEQUENCE = [
    "चैत्र",
    "वैशाख",
    "ज्येष्ठ",
    "आषाढ़",
    "श्रावण",
    "भाद्रपद",
    "आश्विन",
    "कार्तिक",
    "मार्गशीर्ष",
    "पौष",
    "माघ",
    "फाल्गुन",
]

AMANTA_MONTH_DATE_RANGES = [
    (date(2026, 3, 19), date(2026, 4, 16), "चैत्र"),
    (date(2026, 4, 17), date(2026, 5, 15), "वैशाख"),
    (date(2026, 5, 16), date(2026, 6, 14), "ज्येष्ठ"),
    (date(2026, 6, 15), date(2026, 7, 13), "आषाढ़"),
    (date(2026, 7, 14), date(2026, 8, 12), "श्रावण"),
    (date(2026, 8, 13), date(2026, 9, 10), "भाद्रपद"),
    (date(2026, 9, 11), date(2026, 10, 9), "आश्विन"),
    (date(2026, 10, 10), date(2026, 11, 7), "कार्तिक"),
    (date(2026, 11, 8), date(2026, 12, 6), "मार्गशीर्ष"),
    (date(2026, 12, 7), date(2027, 1, 4), "पौष"),
    (date(2027, 1, 5), date(2027, 2, 3), "माघ"),
    (date(2027, 2, 4), date(2027, 3, 4), "फाल्गुन"),
]


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


def shift_lunar_month_name(maah_hi: str, offset: int) -> str:
    canonical_maah = normalize_lunar_month_name(maah_hi)
    if canonical_maah is None:
        return maah_hi
    try:
        index = LUNAR_MONTH_SEQUENCE.index(canonical_maah)
    except ValueError:
        return canonical_maah
    return LUNAR_MONTH_SEQUENCE[(index + offset) % len(LUNAR_MONTH_SEQUENCE)]


def convert_lunar_month_to_amanta(maah_hi: str, paksha_hint: str | None) -> str:
    paksha = normalize_paksha_name(paksha_hint)
    if paksha == "शुक्ल":
        return shift_lunar_month_name(maah_hi, 1)
    return shift_lunar_month_name(maah_hi, 0)


def resolve_explicit_amanta_month(target_date: date) -> str | None:
    for start_date, end_date, maah_hi in AMANTA_MONTH_DATE_RANGES:
        if start_date <= target_date <= end_date:
            return maah_hi
    return None


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


def previous_season_key(current_key: str) -> str:
    if current_key not in SEASON_ORDER:
        return "hemant"
    return SEASON_ORDER[(SEASON_ORDER.index(current_key) - 1) % len(SEASON_ORDER)]


def next_season_start_date(current_key: str, target_date: datetime.date) -> datetime.date:
    upcoming_key = next_season_key(current_key)
    month, day = SEASON_START_MONTH_DAY[upcoming_key]
    candidate = datetime(target_date.year, month, day).date()
    if candidate <= target_date:
        candidate = datetime(target_date.year + 1, month, day).date()
    return candidate


def current_season_start_date(current_key: str, target_date: datetime.date) -> datetime.date:
    month, day = SEASON_START_MONTH_DAY[current_key]
    candidate = datetime(target_date.year, month, day).date()
    if candidate > target_date:
        candidate = datetime(target_date.year - 1, month, day).date()
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
    pre_transition_days: int,
    post_transition_days: int,
) -> TransitionPlan:
    active_current = current_key if current_key in SEASON_ORDER else "shishir"
    next_start = next_season_start_date(active_current, target_date)
    days_to_next = (next_start - target_date).days
    current_start = current_season_start_date(active_current, target_date)
    days_since_current_start = (target_date - current_start).days

    in_pre_window = 0 <= days_to_next <= max(pre_transition_days, 0)
    in_post_window = 0 <= days_since_current_start < max(post_transition_days, 0)

    if in_pre_window:
        transition_current = active_current
        transition_upcoming = next_season_key(active_current)
        display_days_remaining = days_to_next
    elif in_post_window:
        transition_current = previous_season_key(active_current)
        transition_upcoming = active_current
        display_days_remaining = max(post_transition_days - days_since_current_start, 0)
    else:
        return TransitionPlan(
            active=False,
            current_key=active_current,
            upcoming_key=next_season_key(active_current),
            days_to_next=days_to_next,
            selected_key=active_current,
            reason="संक्रमण विंडो के बाहर",
            prefer_lighter=False,
        )

    if weather is not None:
        weather_selected, reason = resolve_weather_priority_season(
            transition_current, transition_upcoming, weather, thresholds
        )
        if weather_selected is not None:
            return TransitionPlan(
                active=True,
                current_key=transition_current,
                upcoming_key=transition_upcoming,
                days_to_next=display_days_remaining,
                selected_key=weather_selected,
                reason=reason,
                prefer_lighter=(weather_selected != transition_current),
            )

    selected_key, reason = resolve_alternating_transition_key(
        target_date,
        display_days_remaining,
        transition_current,
        transition_upcoming,
    )
    return TransitionPlan(
        active=True,
        current_key=transition_current,
        upcoming_key=transition_upcoming,
        days_to_next=display_days_remaining,
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


def normalize_repeat_family_text(text: str) -> str:
    lowered = text.lower()
    scrubbed = re.sub(r"[()\[\]{}.,;:!?*/+|_=\-–—।]+", " ", lowered)
    return re.sub(r"\s+", " ", scrubbed).strip()


def compile_repeat_family_patterns(
    family_aliases: dict[str, tuple[str, ...]],
) -> dict[str, tuple[re.Pattern[str], ...]]:
    compiled: dict[str, tuple[re.Pattern[str], ...]] = {}
    for family, aliases in family_aliases.items():
        patterns: list[re.Pattern[str]] = []
        seen_aliases: set[str] = set()
        for alias in aliases:
            normalized_alias = normalize_repeat_family_text(alias)
            if not normalized_alias or normalized_alias in seen_aliases:
                continue
            seen_aliases.add(normalized_alias)
            escaped = re.escape(normalized_alias).replace(r"\ ", r"\s+")
            patterns.append(
                re.compile(rf"(?<![{REPEAT_FAMILY_BOUNDARY_CLASS}]){escaped}(?![{REPEAT_FAMILY_BOUNDARY_CLASS}])")
            )
        compiled[family] = tuple(patterns)
    return compiled


BREAKFAST_REPEAT_PATTERNS = compile_repeat_family_patterns(BREAKFAST_REPEAT_ALIASES)
MEAL_REPEAT_PATTERNS = compile_repeat_family_patterns(MEAL_REPEAT_ALIASES)


def extract_repeat_families_from_patterns(
    item: str, pattern_map: dict[str, tuple[re.Pattern[str], ...]]
) -> set[str]:
    normalized_item = normalize_repeat_family_text(item)
    if not normalized_item:
        return set()

    families: set[str] = set()
    for family, patterns in pattern_map.items():
        if any(pattern.search(normalized_item) for pattern in patterns):
            families.add(family)
    return families


def extract_breakfast_repeat_families(item: str) -> set[str]:
    return extract_repeat_families_from_patterns(item, BREAKFAST_REPEAT_PATTERNS)


def extract_meal_repeat_families(item: str) -> set[str]:
    return extract_repeat_families_from_patterns(item, MEAL_REPEAT_PATTERNS)


def extract_any_repeat_families(item: str) -> set[str]:
    return extract_breakfast_repeat_families(item) | extract_meal_repeat_families(item)


def get_item_repeat_family_conflicts(
    item: str, blocked_families: set[str], family_extractor: Callable[[str], set[str]]
) -> list[str]:
    return sorted(family_extractor(item) & blocked_families)


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
            return datetime.strptime(date_arg, "%Y-%m-%d").date() + timedelta(days=1)
        except ValueError as exc:
            raise ValueError("--date must be in YYYY-MM-DD format") from exc
    return datetime.now(ZoneInfo(timezone_name)).date() + timedelta(days=1)


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
    lunar_month_system: str,
) -> PanchangInfo:
    if panchang_row:
        ritu_hi = str(panchang_row.get("ritu_hi", default_ritu)).strip() or default_ritu
        maah_hi = str(
            panchang_row.get("maah_hi", ekadashi.lunar_month_hi or GREGORIAN_MONTH_HI[target_date.month])
        ).strip()
        if lunar_month_system == "amanta":
            explicit_amanta_month = resolve_explicit_amanta_month(target_date)
            if explicit_amanta_month is not None:
                maah_hi = explicit_amanta_month
            else:
                maah_hi = convert_lunar_month_to_amanta(maah_hi, panchang_row.get("paksha_hi"))
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


def resolve_ritu_override(target_date: date, config: dict[str, Any], config_key: str) -> str | None:
    raw_overrides = config.get(config_key, [])
    if not isinstance(raw_overrides, list):
        return None

    for row in raw_overrides:
        if not isinstance(row, dict):
            continue
        start_raw = row.get("start")
        end_raw = row.get("end")
        ritu_hi = str(row.get("ritu_hi", "")).strip()
        if not start_raw or not end_raw or not ritu_hi:
            continue
        try:
            start_date = date.fromisoformat(str(start_raw))
            end_date = date.fromisoformat(str(end_raw))
        except ValueError:
            continue
        if start_date <= target_date <= end_date:
            return normalize_ritu_key(ritu_hi)

    return None


def resolve_item_date_override(target_date: date, config: dict[str, Any], config_key: str) -> str | None:
    raw_overrides = config.get(config_key, [])
    if not isinstance(raw_overrides, list):
        return None

    target_date_str = target_date.isoformat()
    for row in raw_overrides:
        if not isinstance(row, dict):
            continue
        if str(row.get("date", "")).strip() != target_date_str:
            continue
        item = str(row.get("item", "")).strip()
        if item:
            return item
    return None


def parse_boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    if isinstance(value, (int, float)):
        return value != 0
    return False


def normalize_non_empty_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if isinstance(item, str) and str(item).strip()]


def build_festival_info_from_row(row: dict[str, Any]) -> FestivalInfo:
    hindu_hi = row.get("hindu_hi", [])
    sikh_hi = row.get("sikh_hi", [])
    special_menu_note_raw = row.get("special_menu_note_hi")
    special_menu_lines_hi = normalize_non_empty_string_list(row.get("special_menu_lines_hi"))

    hindu = normalize_non_empty_string_list(hindu_hi)
    sikh = normalize_non_empty_string_list(sikh_hi)
    special_menu_note_hi = (
        str(special_menu_note_raw).strip()
        if isinstance(special_menu_note_raw, str) and str(special_menu_note_raw).strip()
        else None
    )
    return FestivalInfo(
        hindu_hi=hindu,
        sikh_hi=sikh,
        suppress_regular_menu=parse_boolish(row.get("suppress_regular_menu", False)),
        special_menu_note_hi=special_menu_note_hi,
        special_menu_lines_hi=special_menu_lines_hi or None,
    )


def get_fallback_festival_info(target_date: str) -> FestivalInfo | None:
    fallback = NAVRATRI_FALLBACKS.get(target_date)
    if not fallback:
        return None
    return FestivalInfo(
        hindu_hi=normalize_non_empty_string_list(fallback.get("hindu_hi", [])),
        sikh_hi=[],
        suppress_regular_menu=True,
        special_menu_note_hi=str(fallback.get("special_menu_note_hi", "")).strip() or None,
        special_menu_lines_hi=normalize_non_empty_string_list(fallback.get("special_menu_lines_hi")) or None,
    )


def merge_festival_info(primary: FestivalInfo, fallback: FestivalInfo | None) -> FestivalInfo:
    if fallback is None:
        return primary

    hindu_hi: list[str] = []
    for name in primary.hindu_hi + fallback.hindu_hi:
        if name and name not in hindu_hi:
            hindu_hi.append(name)

    sikh_hi: list[str] = []
    for name in primary.sikh_hi + fallback.sikh_hi:
        if name and name not in sikh_hi:
            sikh_hi.append(name)

    return FestivalInfo(
        hindu_hi=hindu_hi,
        sikh_hi=sikh_hi,
        suppress_regular_menu=primary.suppress_regular_menu or fallback.suppress_regular_menu,
        special_menu_note_hi=primary.special_menu_note_hi or fallback.special_menu_note_hi,
        special_menu_lines_hi=primary.special_menu_lines_hi or fallback.special_menu_lines_hi,
    )


def resolve_festival_info(target_date: str, festivals_data: Any) -> FestivalInfo:
    row = get_festival_entry_for_date(target_date, festivals_data)
    fallback_info = get_fallback_festival_info(target_date)
    if not row:
        return fallback_info or FestivalInfo(hindu_hi=[], sikh_hi=[])
    return merge_festival_info(build_festival_info_from_row(row), fallback_info)


def format_festival_line(festival_info: FestivalInfo) -> str | None:
    combined: list[str] = []
    for name in festival_info.hindu_hi + festival_info.sikh_hi:
        if name not in combined:
            combined.append(name)
    if not combined:
        return None
    return "*पर्व/त्योहार:* " + " / ".join(combined)


def format_special_menu_note_line(festival_info: FestivalInfo) -> str | None:
    if not festival_info.special_menu_note_hi:
        return None
    return "*विशेष पारंपरिक सेवन/भोग:* " + festival_info.special_menu_note_hi


def is_navratri_festival(festival_info: FestivalInfo) -> bool:
    return any("नवरात्रि" in name for name in festival_info.hindu_hi)


def apply_recurring_festival_menu_overrides(
    festival_info: FestivalInfo,
    panchang_info: PanchangInfo,
) -> FestivalInfo:
    if festival_info.special_menu_lines_hi:
        return festival_info
    if is_navratri_festival(festival_info) and panchang_info.tithi_hi == "अष्टमी":
        return replace(
            festival_info,
            suppress_regular_menu=True,
            special_menu_lines_hi=NAVRATRI_ASHTAMI_SPECIAL_MENU_LINES[:],
        )
    return festival_info


def is_blocked_item(item: str, keywords: list[str]) -> bool:
    return any(keyword in item for keyword in keywords)


def normalize_history(raw: Any) -> list[dict[str, str]]:
    if not isinstance(raw, list):
        return []

    cleaned: list[dict[str, Any]] = []
    for row in raw:
        if not isinstance(row, dict):
            continue
        date_val = row.get("date")
        if not isinstance(date_val, str):
            continue

        breakfast_val = row.get("breakfast")
        meal_val = row.get("meal")
        next_day_breakfast_lock = row.get("next_day_breakfast_lock")
        next_day_requires_rice_prep = row.get("next_day_requires_rice_prep")

        if isinstance(breakfast_val, str) and isinstance(meal_val, str):
            normalized_row: dict[str, Any] = {"date": date_val, "breakfast": breakfast_val, "meal": meal_val}
            if isinstance(next_day_breakfast_lock, str) and next_day_breakfast_lock.strip():
                normalized_row["next_day_breakfast_lock"] = next_day_breakfast_lock.strip()
                normalized_row["next_day_requires_rice_prep"] = True
            elif isinstance(next_day_requires_rice_prep, bool):
                normalized_row["next_day_requires_rice_prep"] = next_day_requires_rice_prep
            cleaned.append(normalized_row)
            continue

        old_item = row.get("item")
        if isinstance(old_item, str):
            normalized_row = {"date": date_val, "meal": old_item}
            if isinstance(next_day_breakfast_lock, str) and next_day_breakfast_lock.strip():
                normalized_row["next_day_breakfast_lock"] = next_day_breakfast_lock.strip()
                normalized_row["next_day_requires_rice_prep"] = True
            elif isinstance(next_day_requires_rice_prep, bool):
                normalized_row["next_day_requires_rice_prep"] = next_day_requires_rice_prep
            cleaned.append(normalized_row)

    return cleaned


def recent_items(
    history: list[dict[str, Any]], target_date: datetime.date, window_days: int, field: str
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


def apply_consecutive_day_repeat_rule(
    pool: list[str], blocked_families: set[str], family_extractor: Callable[[str], set[str]]
) -> tuple[list[str], bool]:
    if not blocked_families:
        return pool[:], False

    filtered = [
        item for item in pool if not get_item_repeat_family_conflicts(item, blocked_families, family_extractor)
    ]
    if filtered:
        return filtered, False
    return pool[:], True


def get_history_row(history: list[dict[str, Any]], target_date_str: str) -> dict[str, Any] | None:
    for row in history:
        if row.get("date") == target_date_str:
            return row
    return None


def get_previous_day_breakfast_lock(history: list[dict[str, Any]], target_date: date) -> str | None:
    previous_date_str = (target_date - timedelta(days=1)).isoformat()
    row = get_history_row(history, previous_date_str)
    if row is None:
        return None
    lock = row.get("next_day_breakfast_lock")
    if isinstance(lock, str) and lock.strip():
        return lock.strip()
    return None


def get_row_repeat_families(row: dict[str, Any]) -> set[str]:
    families: set[str] = set()
    breakfast_value = row.get("breakfast")
    if isinstance(breakfast_value, str) and breakfast_value.strip():
        families.update(extract_breakfast_repeat_families(breakfast_value))

    meal_value = row.get("meal")
    if isinstance(meal_value, str) and meal_value.strip():
        families.update(extract_meal_repeat_families(meal_value))

    legacy_item_value = row.get("item")
    if isinstance(legacy_item_value, str) and legacy_item_value.strip():
        families.update(extract_meal_repeat_families(legacy_item_value))
    return families


def get_previous_day_repeat_families(history: list[dict[str, Any]], target_date: date) -> set[str]:
    previous_date_str = (target_date - timedelta(days=1)).isoformat()
    row = get_history_row(history, previous_date_str)
    if row is None:
        return set()
    return get_row_repeat_families(row)


def is_overnight_breakfast(item: str) -> bool:
    return item in OVERNIGHT_BREAKFAST_ITEMS


def format_overnight_breakfast_label(item: str) -> str:
    primary = item.split(":", 1)[0].strip()
    return primary if primary else item


def exclude_overnight_breakfasts(items: list[str]) -> list[str]:
    filtered = [item for item in items if item not in OVERNIGHT_BREAKFAST_ITEMS]
    return filtered if filtered else items[:]


def can_apply_overnight_breakfast_on_run_date(target_date: date, generation_date: date) -> bool:
    return target_date > generation_date


def is_mangore_item(item: str) -> bool:
    normalized_item = item.casefold()
    return any(token in normalized_item for token in MANGORE_ITEM_TOKENS)


def requires_mangore_prep(*items: str) -> bool:
    return any(is_mangore_item(item) for item in items if item)


def is_rice_item(item: str) -> bool:
    return any(token in item for token in RICE_ITEM_TOKENS)


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
    source_order_raw = config.get("weather_source_order", ["imd", "open_meteo", "manual"])
    if isinstance(source_order_raw, list):
        source_order = [str(src).strip().lower() for src in source_order_raw if str(src).strip()]
    else:
        source_order = ["imd", "open_meteo", "manual"]

    if not source_order:
        source_order = ["imd", "open_meteo", "manual"]

    for source in source_order:
        if source in {"manual", "override"}:
            manual_weather = load_manual_weather(target_date, thresholds)
            if manual_weather is not None:
                return manual_weather
        if source == "imd":
            imd_weather = fetch_imd_city_weather(target_date, config, thresholds)
            if imd_weather is not None:
                return imd_weather
        elif source in {"open_meteo", "open-meteo", "openmeteo"}:
            meteo_weather = fetch_open_meteo_weather(target_date, config, thresholds)
            if meteo_weather is not None:
                return meteo_weather

    return None


def extract_date_strings_from_any(source: Any, timezone_name: str) -> set[str]:
    dates: set[str] = set()

    def collect_from_row(row: Any) -> None:
        if isinstance(row, dict):
            normalized = normalize_any_date_to_yyyy_mm_dd(row.get("date"), timezone_name)
            if normalized:
                dates.add(normalized)

    if isinstance(source, dict):
        for key, val in source.items():
            normalized_key = normalize_any_date_to_yyyy_mm_dd(key, timezone_name)
            if normalized_key:
                dates.add(normalized_key)
            if isinstance(val, dict):
                collect_from_row(val)
            elif isinstance(val, list):
                for row in val:
                    collect_from_row(row)
        entries = source.get("entries")
        if isinstance(entries, list):
            for row in entries:
                collect_from_row(row)
    elif isinstance(source, list):
        for row in source:
            collect_from_row(row)
    return dates


def assess_next_30_day_data_coverage(
    target_date: date,
    timezone_name: str,
    panchang_data: Any,
    festivals_data: Any,
    ekadashi_data: Any,
) -> str | None:
    start = target_date
    end = target_date + timedelta(days=29)
    expected_dates = {(start + timedelta(days=offset)).strftime("%Y-%m-%d") for offset in range(30)}

    panchang_dates = extract_date_strings_from_any(panchang_data, timezone_name)
    missing_panchang = sorted(expected_dates - panchang_dates)

    ekadashi_dates = extract_date_strings_from_any(ekadashi_data, timezone_name)
    ekadashi_in_window = sorted(d for d in ekadashi_dates if start.strftime("%Y-%m-%d") <= d <= end.strftime("%Y-%m-%d"))

    parts: list[str] = []
    if missing_panchang:
        parts.append(f"पंचांग कवरेज कमी: अगले 30 दिनों में {len(missing_panchang)} तिथियां अनुपलब्ध")
    if not ekadashi_dates:
        parts.append("एकादशी डेटा स्रोत अनुपलब्ध/अमान्य")
    elif not ekadashi_in_window:
        parts.append("अगले 30 दिनों में एकादशी प्रविष्टि नहीं (डेटा जाँचें)")

    if not parts:
        return None
    return " | ".join(parts)


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


def apply_hard_filters(
    items: list[str],
    ekadashi: EkadashiInfo,
    keywords: list[str],
    disallowed_keywords: list[str],
) -> list[str]:
    base_pool = items[:]
    if ekadashi.is_ekadashi:
        base_pool = [item for item in base_pool if not is_blocked_item(item, keywords)]
    if disallowed_keywords:
        base_pool = [item for item in base_pool if not is_blocked_item(item, disallowed_keywords)]
    return base_pool


def finalize_choice_pool(
    base_pool: list[str],
    recent_block_set: set[str],
    consecutive_day_block_families: set[str],
    family_extractor: Callable[[str], set[str]],
    weather_rules: WeatherRules | None,
    weather_tags: dict[str, list[str]],
    warn_bucket: set[str],
    constraint_notes: list[str],
    prefer_lighter: bool,
    max_lightness_score: int | None = None,
    heavy_light_classification: dict[str, str] | None = None,
) -> list[str]:
    if not base_pool:
        return []

    pool = apply_repeat_rule(base_pool, recent_block_set)

    pool, repeated_family_fallback = apply_consecutive_day_repeat_rule(
        pool, consecutive_day_block_families, family_extractor
    )
    if repeated_family_fallback and CONSECUTIVE_DAY_REPEAT_NOTE not in constraint_notes:
        constraint_notes.append(CONSECUTIVE_DAY_REPEAT_NOTE)

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

    return pool


def deterministic_choice(pool: list[str], seed_key: str) -> str:
    if not pool:
        raise RuntimeError("No menu item available after applying rules")
    seed_int = int(hashlib.sha256(seed_key.encode("utf-8")).hexdigest(), 16)
    rng = random.Random(seed_int)
    return rng.choice(pool)


def choose_item(
    items: list[str],
    ekadashi: EkadashiInfo,
    recent_block_set: set[str],
    consecutive_day_block_families: set[str],
    family_extractor: Callable[[str], set[str]],
    keywords: list[str],
    disallowed_keywords: list[str],
    fallback_policy: str,
    seed_key: str,
    weather_rules: WeatherRules | None,
    weather_tags: dict[str, list[str]],
    warn_bucket: set[str],
    constraint_notes: list[str],
    prefer_lighter: bool,
    light_fallback_items: list[str],
    max_lightness_score: int | None = None,
    heavy_light_classification: dict[str, str] | None = None,
) -> str:
    full_pool = items[:] if items else light_fallback_items[:]

    base_pool = apply_hard_filters(full_pool, ekadashi, keywords, disallowed_keywords)

    if not base_pool and ekadashi.is_ekadashi and fallback_policy == "fallback_full_menu":
        base_pool = full_pool[:]

    if not base_pool:
        base_pool = light_fallback_items[:]

    if not base_pool:
        raise RuntimeError("No menu item available after applying rules")

    pool = finalize_choice_pool(
        base_pool=base_pool,
        recent_block_set=recent_block_set,
        consecutive_day_block_families=consecutive_day_block_families,
        family_extractor=family_extractor,
        weather_rules=weather_rules,
        weather_tags=weather_tags,
        warn_bucket=warn_bucket,
        constraint_notes=constraint_notes,
        prefer_lighter=prefer_lighter,
        max_lightness_score=max_lightness_score,
        heavy_light_classification=heavy_light_classification,
    )
    return deterministic_choice(pool, seed_key)


def update_history(
    history: list[dict[str, Any]],
    target_date: str,
    breakfast_item: str,
    meal_item: str,
    keep_days: int,
    next_day_breakfast_lock: str | None = None,
    next_day_requires_rice_prep: bool = False,
) -> list[dict[str, Any]]:
    updated = [row for row in history if row.get("date") != target_date]
    new_row: dict[str, Any] = {"date": target_date, "breakfast": breakfast_item, "meal": meal_item}
    if isinstance(next_day_breakfast_lock, str) and next_day_breakfast_lock.strip():
        new_row["next_day_breakfast_lock"] = next_day_breakfast_lock.strip()
        new_row["next_day_requires_rice_prep"] = True
    elif next_day_requires_rice_prep:
        new_row["next_day_requires_rice_prep"] = True
    updated.append(new_row)

    cutoff = datetime.strptime(target_date, "%Y-%m-%d").date() - timedelta(days=max(keep_days, 7) + 30)

    retained: list[dict[str, Any]] = []
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
        f"अधिकतम {weather.max_temp_c:.1f}°C, वर्षा संभावना {weather.rain_probability_pct:.0f}%"
    )


def build_output_text(lines: list[str]) -> str:
    if len(lines) >= 4:
        header_block = "\r\n".join(lines[:4])
        body_block = "\r\n\r\n".join(lines[4:])
        return header_block if not body_block else f"{header_block}\r\n\r\n{body_block}"
    return "\r\n\r\n".join(lines)


def get_disallowed_keywords(ritu_key: str) -> list[str]:
    if ritu_key == "varsha":
        return VARSHA_BANNED_KEYWORDS
    if ritu_key == "hemant":
        return HEMANT_BANNED_KEYWORDS
    if ritu_key == "sharad":
        return SHARAD_BANNED_KEYWORDS
    return []


def get_menu_lists_for_ritu(
    ritu_key: str,
    *,
    breakfast_shishir_items: list[str],
    meal_shishir_items: list[str],
    breakfast_vasant_items: list[str],
    meal_vasant_items: list[str],
    breakfast_grishm_items: list[str],
    meal_grishm_items: list[str],
    breakfast_varsha_items: list[str],
    meal_varsha_items: list[str],
    breakfast_sharad_items: list[str],
    meal_sharad_items: list[str],
    breakfast_hemant_items: list[str],
    meal_hemant_items: list[str],
    light_fallback_items: list[str],
    missing_data_notes: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    if ritu_key == "grishm":
        breakfast_items = breakfast_grishm_items[:]
        meal_items = meal_grishm_items[:]
        if missing_data_notes is not None:
            if not meal_grishm_items:
                missing_data_notes.append("ग्रीष्म भोजन सूची उपलब्ध नहीं (fallback: हल्का डिफ़ॉल्ट)")
            if not breakfast_grishm_items:
                missing_data_notes.append("ग्रीष्म नाश्ता सूची उपलब्ध नहीं (fallback: हल्का डिफ़ॉल्ट)")
    elif ritu_key == "hemant":
        breakfast_items = breakfast_hemant_items[:]
        meal_items = meal_hemant_items[:]
        if missing_data_notes is not None:
            if not meal_hemant_items:
                missing_data_notes.append("हेमंत भोजन सूची उपलब्ध नहीं (fallback: हल्का डिफ़ॉल्ट)")
            if not breakfast_hemant_items:
                missing_data_notes.append("हेमंत नाश्ता सूची उपलब्ध नहीं (fallback: हल्का डिफ़ॉल्ट)")
    elif ritu_key == "sharad":
        breakfast_items = breakfast_sharad_items[:]
        meal_items = meal_sharad_items[:]
        if missing_data_notes is not None:
            if not meal_sharad_items:
                missing_data_notes.append("शरद भोजन सूची उपलब्ध नहीं (fallback: हल्का डिफ़ॉल्ट)")
            if not breakfast_sharad_items:
                missing_data_notes.append("शरद नाश्ता सूची उपलब्ध नहीं (fallback: हल्का डिफ़ॉल्ट)")
    elif ritu_key == "varsha":
        breakfast_items = breakfast_varsha_items[:]
        meal_items = meal_varsha_items[:]
        if missing_data_notes is not None:
            if not meal_varsha_items:
                missing_data_notes.append("वर्षा भोजन सूची उपलब्ध नहीं (fallback: हल्का डिफ़ॉल्ट)")
            if not breakfast_varsha_items:
                missing_data_notes.append("वर्षा नाश्ता सूची उपलब्ध नहीं (fallback: हल्का डिफ़ॉल्ट)")
    elif ritu_key == "vasant":
        breakfast_items = breakfast_vasant_items[:]
        meal_items = meal_vasant_items[:]
        if missing_data_notes is not None:
            if not meal_vasant_items:
                missing_data_notes.append("वसंत भोजन सूची उपलब्ध नहीं (fallback: हल्का डिफ़ॉल्ट)")
            if not breakfast_vasant_items:
                missing_data_notes.append("वसंत नाश्ता सूची उपलब्ध नहीं (fallback: हल्का डिफ़ॉल्ट)")
    else:
        breakfast_items = breakfast_shishir_items[:]
        meal_items = meal_shishir_items[:]

    if not breakfast_items:
        breakfast_items = light_fallback_items[:]
        if missing_data_notes is not None and not any("नाश्ता सूची उपलब्ध नहीं" in note for note in missing_data_notes):
            missing_data_notes.append("नाश्ता विकल्प उपलब्ध नहीं (fallback: हल्का डिफ़ॉल्ट)")
    if not meal_items:
        meal_items = light_fallback_items[:]
        if missing_data_notes is not None and not any("भोजन सूची उपलब्ध नहीं" in note for note in missing_data_notes):
            missing_data_notes.append("भोजन विकल्प उपलब्ध नहीं (fallback: हल्का डिफ़ॉल्ट)")

    return breakfast_items, meal_items


def build_day_context(
    target_date: date,
    *,
    config: dict[str, Any],
    timezone_name: str,
    default_ritu: str,
    lunar_month_system: str,
    thresholds: dict[str, float],
    weather_enabled: bool,
    panchang_data: Any,
    panchang_source_error_note: str | None,
    festivals_data: Any,
    ekadashi_data: dict[str, Any],
    breakfast_shishir_items: list[str],
    meal_shishir_items: list[str],
    breakfast_vasant_items: list[str],
    meal_vasant_items: list[str],
    breakfast_grishm_items: list[str],
    meal_grishm_items: list[str],
    breakfast_varsha_items: list[str],
    meal_varsha_items: list[str],
    breakfast_sharad_items: list[str],
    meal_sharad_items: list[str],
    breakfast_hemant_items: list[str],
    meal_hemant_items: list[str],
    light_fallback_items: list[str],
    missing_data_notes: list[str] | None = None,
) -> DayContext:
    target_date_str = target_date.strftime("%Y-%m-%d")
    target_date_display_str = target_date.strftime("%d-%b-%Y")

    weather_info: WeatherInfo | None = None
    weather_rules: WeatherRules | None = None
    if weather_enabled:
        weather_info = resolve_weather_info(target_date_str, config, thresholds)
        if weather_info is not None:
            weather_rules = derive_weather_rules(weather_info, thresholds)
        elif missing_data_notes is not None:
            missing_data_notes.append("मौसम डेटा उपलब्ध नहीं (मैनुअल/ओपन-मेटियो)")

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

    panchang_info = resolve_panchang_info(
        target_date,
        ekadashi,
        panchang_lookup.row,
        default_ritu,
        lunar_month_system,
    )
    festival_info = resolve_festival_info(target_date_str, festivals_data)
    festival_info = apply_recurring_festival_menu_overrides(festival_info, panchang_info)

    maah_mapped_ritu_key = resolve_ritu_key_from_lunar_month(panchang_info.maah_hi)
    if maah_mapped_ritu_key is not None:
        base_ritu_key = maah_mapped_ritu_key
    else:
        base_ritu_key = normalize_ritu_key(panchang_info.ritu_hi)
        if missing_data_notes is not None:
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
    if missing_data_notes is not None and shringdhara_info.missing_note_hi:
        missing_data_notes.append(shringdhara_info.missing_note_hi)

    legacy_transition_window_days = int(config.get("season_transition_window_days", 7))
    pre_transition_days = int(config.get("season_transition_pre_days", legacy_transition_window_days))
    post_transition_days = int(config.get("season_transition_post_days", 8))
    transition_plan = resolve_transition_plan(
        target_date=target_date,
        current_key=base_ritu_key,
        weather=weather_info,
        thresholds=thresholds,
        pre_transition_days=pre_transition_days,
        post_transition_days=post_transition_days,
    )
    menu_override_ritu_key = resolve_ritu_override(target_date, config, "menu_ritu_date_overrides")
    if menu_override_ritu_key is not None:
        ritu_key = menu_override_ritu_key
    else:
        ritu_key = base_ritu_key if shringdhara_info.active else transition_plan.selected_key

    breakfast_item_override = resolve_item_date_override(target_date, config, "breakfast_item_date_overrides")

    if missing_data_notes is not None:
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

    breakfast_items, meal_items = get_menu_lists_for_ritu(
        ritu_key,
        breakfast_shishir_items=breakfast_shishir_items,
        meal_shishir_items=meal_shishir_items,
        breakfast_vasant_items=breakfast_vasant_items,
        meal_vasant_items=meal_vasant_items,
        breakfast_grishm_items=breakfast_grishm_items,
        meal_grishm_items=meal_grishm_items,
        breakfast_varsha_items=breakfast_varsha_items,
        meal_varsha_items=meal_varsha_items,
        breakfast_sharad_items=breakfast_sharad_items,
        meal_sharad_items=meal_sharad_items,
        breakfast_hemant_items=breakfast_hemant_items,
        meal_hemant_items=meal_hemant_items,
        light_fallback_items=light_fallback_items,
        missing_data_notes=missing_data_notes,
    )

    return DayContext(
        target_date=target_date,
        target_date_str=target_date_str,
        display_date_str=target_date_display_str,
        ekadashi=ekadashi,
        panchang_lookup=panchang_lookup,
        panchang_info=panchang_info,
        festival_info=festival_info,
        display_ritu_hi=display_ritu_hi,
        shringdhara_info=shringdhara_info,
        transition_plan=transition_plan,
        ritu_key=ritu_key,
        breakfast_items=breakfast_items,
        meal_items=meal_items,
        disallowed_keywords=get_disallowed_keywords(ritu_key),
        weather_info=weather_info,
        weather_rules=weather_rules,
        breakfast_item_override=breakfast_item_override,
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
    lunar_month_system = str(config.get("lunar_month_system", "amanta")).strip().lower() or "amanta"

    if not isinstance(keywords, list) or not all(isinstance(k, str) for k in keywords):
        raise ValueError("ekadashi_block_keywords must be an array of strings")

    target_date = resolve_date(args.date, timezone_name)
    target_date_str = target_date.strftime("%Y-%m-%d")
    target_date_display_str = target_date.strftime("%d-%b-%Y")
    generation_date = datetime.now(ZoneInfo(timezone_name)).date()

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

    weather_tags = load_weather_tags(all_items)
    heavy_light_classification, classification_note = load_heavy_light_classification(
        HEAVY_LIGHT_CLASSIFICATION_FILE, all_items
    )
    if classification_note:
        missing_data_notes.append(classification_note)

    current_day = build_day_context(
        target_date,
        config=config,
        timezone_name=timezone_name,
        default_ritu=default_ritu,
        lunar_month_system=lunar_month_system,
        thresholds=thresholds,
        weather_enabled=weather_enabled,
        panchang_data=panchang_data,
        panchang_source_error_note=panchang_source_error_note,
        festivals_data=festivals_data,
        ekadashi_data=ekadashi_data,
        breakfast_shishir_items=breakfast_shishir_items,
        meal_shishir_items=meal_shishir_items,
        breakfast_vasant_items=breakfast_vasant_items,
        meal_vasant_items=meal_vasant_items,
        breakfast_grishm_items=breakfast_grishm_items,
        meal_grishm_items=meal_grishm_items,
        breakfast_varsha_items=breakfast_varsha_items,
        meal_varsha_items=meal_varsha_items,
        breakfast_sharad_items=breakfast_sharad_items,
        meal_sharad_items=meal_sharad_items,
        breakfast_hemant_items=breakfast_hemant_items,
        meal_hemant_items=meal_hemant_items,
        light_fallback_items=light_fallback_items,
        missing_data_notes=missing_data_notes,
    )

    target_date = current_day.target_date
    target_date_str = current_day.target_date_str
    target_date_display_str = current_day.display_date_str
    coverage_note = assess_next_30_day_data_coverage(
        target_date=target_date,
        timezone_name=timezone_name,
        panchang_data=panchang_data,
        festivals_data=festivals_data,
        ekadashi_data=ekadashi_data,
    )
    if coverage_note:
        missing_data_notes.append(coverage_note)

    display_ritu_hi = current_day.display_ritu_hi
    panchang_info = current_day.panchang_info
    festival_info = current_day.festival_info
    shringdhara_info = current_day.shringdhara_info
    transition_plan = current_day.transition_plan
    ritu_key = current_day.ritu_key
    breakfast_items = current_day.breakfast_items
    meal_items = current_day.meal_items
    disallowed_keywords = current_day.disallowed_keywords
    weather_info = current_day.weather_info
    weather_rules = current_day.weather_rules
    ekadashi = current_day.ekadashi
    breakfast_item_override = current_day.breakfast_item_override

    if festival_info.suppress_regular_menu:
        lines = [
            f"*{target_date_display_str} तिथि के लिए भोजन:*",
            f"*ऋतु:* {display_ritu_hi}",
            f"*माह:* {panchang_info.maah_hi}",
            f"*तिथि (पंचांग):* {panchang_info.tithi_hi}",
        ]
        festival_line = format_festival_line(festival_info)
        if festival_line:
            lines.append(festival_line)
        if festival_info.special_menu_lines_hi:
            lines.extend(festival_info.special_menu_lines_hi)
        else:
            lines.append("*नियमित मेनू:* आज पर्व/विशेष पालन के कारण नियमित नाश्ता और भोजन मेनू नहीं दिया जाएगा।")
            special_menu_note_line = format_special_menu_note_line(festival_info)
            if special_menu_note_line:
                lines.append(special_menu_note_line)
        if ekadashi.is_ekadashi and ekadashi.name_hi:
            lines.append(f"*एकादशी:* {ekadashi.name_hi}")
        if weather_info is not None and should_show_weather_line(weather_info, weather_mode):
            lines.append(build_weather_line(weather_info, weather_city_hi))
        if target_date.month == 1 and target_date.day == 1:
            lines.append("*वार्षिक स्मरण (1 जनवरी):* " + NEW_YEAR_KANJI_NOTE)

        output_text = build_output_text(lines)
        with OUTPUT_FILE.open("w", encoding="utf-8") as f:
            f.write(output_text + "\n")

        print(output_text)
        return 0

    breakfast_recent = recent_items(history, target_date, repeat_window_days, "breakfast")
    meal_recent = recent_items(history, target_date, repeat_window_days, "meal")
    previous_day_breakfast_lock = get_previous_day_breakfast_lock(history, target_date)
    previous_day_repeat_families = get_previous_day_repeat_families(history, target_date)

    warning_items: set[str] = set()

    selected_observance_item: str | None = None
    next_day_breakfast_lock: str | None = None
    next_day_requires_rice_prep = False
    rice_support_meal_candidates: list[str] = []
    if shringdhara_info.active:
        observance_items = dedupe_preserve_order(meal_items if meal_items else breakfast_items)
        observance_recent = breakfast_recent | meal_recent
        selected_observance_item = choose_item(
            items=observance_items,
            ekadashi=ekadashi,
            recent_block_set=observance_recent,
            consecutive_day_block_families=previous_day_repeat_families,
            family_extractor=extract_any_repeat_families,
            keywords=keywords,
            disallowed_keywords=disallowed_keywords,
            fallback_policy=fallback_policy,
            seed_key=f"{target_date_str}:shringdhara",
            weather_rules=weather_rules,
            weather_tags=weather_tags,
            warn_bucket=warning_items,
            constraint_notes=missing_data_notes,
            prefer_lighter=True,
            light_fallback_items=light_fallback_items,
            heavy_light_classification=heavy_light_classification,
        )
        selected_breakfast = selected_observance_item
        selected_meal = selected_observance_item
    else:
        breakfast_fixed = False
        breakfast_choice_items = exclude_overnight_breakfasts(breakfast_items)

        if previous_day_breakfast_lock:
            if is_overnight_breakfast(previous_day_breakfast_lock) and not can_apply_overnight_breakfast_on_run_date(
                target_date, generation_date
            ):
                missing_data_notes.append(
                    "[समय नियम] overnight नाश्ता लॉक लागू नहीं किया गया क्योंकि यह मेनू उसी सुबह बनाया जा रहा है"
                )
                previous_day_breakfast_lock = None
            else:
                lock_conflicts = get_item_repeat_family_conflicts(
                    previous_day_breakfast_lock,
                    previous_day_repeat_families,
                    extract_breakfast_repeat_families,
                )
                if previous_day_breakfast_lock in breakfast_items and not lock_conflicts:
                    selected_breakfast = previous_day_breakfast_lock
                    breakfast_fixed = True
                else:
                    if previous_day_breakfast_lock not in breakfast_items:
                        missing_data_notes.append(
                            f"[अनुपलब्ध] पिछली रात से लॉक किया गया नाश्ता आज की सूची में नहीं मिला: {previous_day_breakfast_lock}"
                        )
                    elif lock_conflicts:
                        missing_data_notes.append(
                            "[नियम] पिछली रात से लॉक किया गया नाश्ता लगातार-दिन नियम से टकराया: "
                            + " / ".join(lock_conflicts)
                        )
                    selected_breakfast = choose_item(
                        items=breakfast_choice_items,
                        ekadashi=ekadashi,
                        recent_block_set=breakfast_recent,
                        consecutive_day_block_families=previous_day_repeat_families,
                        family_extractor=extract_breakfast_repeat_families,
                        keywords=keywords,
                        disallowed_keywords=disallowed_keywords,
                        fallback_policy=fallback_policy,
                        seed_key=f"{target_date_str}:breakfast",
                        weather_rules=weather_rules,
                        weather_tags=weather_tags,
                        warn_bucket=warning_items,
                        constraint_notes=missing_data_notes,
                        prefer_lighter=transition_plan.prefer_lighter,
                        light_fallback_items=light_fallback_items,
                        heavy_light_classification=heavy_light_classification,
                    )
        if breakfast_item_override:
            if previous_day_breakfast_lock:
                pass
            elif is_overnight_breakfast(breakfast_item_override) and not can_apply_overnight_breakfast_on_run_date(
                target_date, generation_date
            ):
                missing_data_notes.append(
                    "[समय नियम] आज का overnight नाश्ता override लागू नहीं किया गया क्योंकि तैयारी का समय निकल चुका है"
                )
                selected_breakfast = choose_item(
                    items=breakfast_choice_items,
                    ekadashi=ekadashi,
                    recent_block_set=breakfast_recent,
                    consecutive_day_block_families=previous_day_repeat_families,
                    family_extractor=extract_breakfast_repeat_families,
                    keywords=keywords,
                    disallowed_keywords=disallowed_keywords,
                    fallback_policy=fallback_policy,
                    seed_key=f"{target_date_str}:breakfast",
                    weather_rules=weather_rules,
                    weather_tags=weather_tags,
                    warn_bucket=warning_items,
                    constraint_notes=missing_data_notes,
                    prefer_lighter=transition_plan.prefer_lighter,
                    light_fallback_items=light_fallback_items,
                    heavy_light_classification=heavy_light_classification,
                )
            elif is_overnight_breakfast(breakfast_item_override):
                missing_data_notes.append(
                    "[डेटा चेतावनी] आज का निर्धारित overnight नाश्ता लागू किया गया है, "
                    "लेकिन पिछली रात की चावल तैयारी history में नहीं मिली"
                )
                selected_breakfast = breakfast_item_override
                breakfast_fixed = True
            elif breakfast_item_override in breakfast_items:
                override_conflicts = get_item_repeat_family_conflicts(
                    breakfast_item_override,
                    previous_day_repeat_families,
                    extract_breakfast_repeat_families,
                )
                if override_conflicts:
                    missing_data_notes.append(
                        "[नियम] निर्धारित नाश्ता override लगातार-दिन नियम से टकराया: "
                        + " / ".join(override_conflicts)
                    )
                    selected_breakfast = choose_item(
                        items=breakfast_choice_items,
                        ekadashi=ekadashi,
                        recent_block_set=breakfast_recent,
                        consecutive_day_block_families=previous_day_repeat_families,
                        family_extractor=extract_breakfast_repeat_families,
                        keywords=keywords,
                        disallowed_keywords=disallowed_keywords,
                        fallback_policy=fallback_policy,
                        seed_key=f"{target_date_str}:breakfast",
                        weather_rules=weather_rules,
                        weather_tags=weather_tags,
                        warn_bucket=warning_items,
                        constraint_notes=missing_data_notes,
                        prefer_lighter=transition_plan.prefer_lighter,
                        light_fallback_items=light_fallback_items,
                        heavy_light_classification=heavy_light_classification,
                    )
                else:
                    selected_breakfast = breakfast_item_override
                    breakfast_fixed = True
            else:
                missing_data_notes.append(
                    f"[अनुपलब्ध] निर्धारित नाश्ता override सूची में नहीं मिला: {breakfast_item_override}"
                )
                selected_breakfast = choose_item(
                    items=breakfast_choice_items,
                    ekadashi=ekadashi,
                    recent_block_set=breakfast_recent,
                    consecutive_day_block_families=previous_day_repeat_families,
                    family_extractor=extract_breakfast_repeat_families,
                    keywords=keywords,
                    disallowed_keywords=disallowed_keywords,
                    fallback_policy=fallback_policy,
                    seed_key=f"{target_date_str}:breakfast",
                    weather_rules=weather_rules,
                    weather_tags=weather_tags,
                    warn_bucket=warning_items,
                    constraint_notes=missing_data_notes,
                    prefer_lighter=transition_plan.prefer_lighter,
                    light_fallback_items=light_fallback_items,
                    heavy_light_classification=heavy_light_classification,
                )
        else:
            if not previous_day_breakfast_lock:
                selected_breakfast = choose_item(
                    items=breakfast_choice_items,
                    ekadashi=ekadashi,
                    recent_block_set=breakfast_recent,
                    consecutive_day_block_families=previous_day_repeat_families,
                    family_extractor=extract_breakfast_repeat_families,
                    keywords=keywords,
                    disallowed_keywords=disallowed_keywords,
                    fallback_policy=fallback_policy,
                    seed_key=f"{target_date_str}:breakfast",
                    weather_rules=weather_rules,
                    weather_tags=weather_tags,
                    warn_bucket=warning_items,
                    constraint_notes=missing_data_notes,
                    prefer_lighter=transition_plan.prefer_lighter,
                    light_fallback_items=light_fallback_items,
                    heavy_light_classification=heavy_light_classification,
                )

        next_day = build_day_context(
            target_date + timedelta(days=1),
            config=config,
            timezone_name=timezone_name,
            default_ritu=default_ritu,
            lunar_month_system=lunar_month_system,
            thresholds=thresholds,
            weather_enabled=weather_enabled,
            panchang_data=panchang_data,
            panchang_source_error_note=panchang_source_error_note,
            festivals_data=festivals_data,
            ekadashi_data=ekadashi_data,
            breakfast_shishir_items=breakfast_shishir_items,
            meal_shishir_items=meal_shishir_items,
            breakfast_vasant_items=breakfast_vasant_items,
            meal_vasant_items=meal_vasant_items,
            breakfast_grishm_items=breakfast_grishm_items,
            meal_grishm_items=meal_grishm_items,
            breakfast_varsha_items=breakfast_varsha_items,
            meal_varsha_items=meal_varsha_items,
            breakfast_sharad_items=breakfast_sharad_items,
            meal_sharad_items=meal_sharad_items,
            breakfast_hemant_items=breakfast_hemant_items,
            meal_hemant_items=meal_hemant_items,
            light_fallback_items=light_fallback_items,
        )

        planned_next_day_overnight: str | None = None
        next_day_override = next_day.breakfast_item_override
        selected_breakfast_repeat_families = extract_breakfast_repeat_families(selected_breakfast)
        if not shringdhara_info.active and not next_day.shringdhara_info.active:
            if next_day_override:
                override_conflicts = get_item_repeat_family_conflicts(
                    next_day_override,
                    selected_breakfast_repeat_families,
                    extract_breakfast_repeat_families,
                )
                if is_overnight_breakfast(next_day_override) and not override_conflicts:
                    planned_next_day_overnight = next_day_override
                elif is_overnight_breakfast(next_day_override) and override_conflicts:
                    missing_data_notes.append(
                        "[नियम] अगले दिन का निर्धारित overnight नाश्ता लगातार-दिन नियम से टकराया: "
                        + " / ".join(override_conflicts)
                    )
            else:
                next_day_breakfast_recent = recent_items(history, next_day.target_date, repeat_window_days, "breakfast")
                next_day_breakfast_recent.add(selected_breakfast)
                planned_next_day_breakfast = choose_item(
                    items=next_day.breakfast_items,
                    ekadashi=next_day.ekadashi,
                    recent_block_set=next_day_breakfast_recent,
                    consecutive_day_block_families=selected_breakfast_repeat_families,
                    family_extractor=extract_breakfast_repeat_families,
                    keywords=keywords,
                    disallowed_keywords=next_day.disallowed_keywords,
                    fallback_policy=fallback_policy,
                    seed_key=f"{next_day.target_date_str}:breakfast:planned-by:{target_date_str}",
                    weather_rules=next_day.weather_rules,
                    weather_tags=weather_tags,
                    warn_bucket=warning_items,
                    constraint_notes=missing_data_notes,
                    prefer_lighter=next_day.transition_plan.prefer_lighter,
                    light_fallback_items=light_fallback_items,
                    heavy_light_classification=heavy_light_classification,
                )
                if is_overnight_breakfast(planned_next_day_breakfast):
                    planned_next_day_overnight = planned_next_day_breakfast

        if planned_next_day_overnight:
            rice_support_meal_candidates = apply_hard_filters(
                [item for item in meal_items if is_rice_item(item)],
                ekadashi,
                keywords,
                disallowed_keywords,
            )
            if rice_support_meal_candidates:
                next_day_breakfast_lock = planned_next_day_overnight
                next_day_requires_rice_prep = True
                selected_meal_pool = finalize_choice_pool(
                    base_pool=rice_support_meal_candidates,
                    recent_block_set=meal_recent,
                    consecutive_day_block_families=previous_day_repeat_families,
                    family_extractor=extract_meal_repeat_families,
                    weather_rules=weather_rules,
                    weather_tags=weather_tags,
                    warn_bucket=warning_items,
                    constraint_notes=missing_data_notes,
                    prefer_lighter=transition_plan.prefer_lighter,
                    heavy_light_classification=heavy_light_classification,
                )
                selected_meal = deterministic_choice(selected_meal_pool, f"{target_date_str}:meal:rice-support")
            else:
                if next_day_override and is_overnight_breakfast(next_day_override):
                    missing_data_notes.append(
                        "[अनुपलब्ध] अगले दिन का निर्धारित overnight नाश्ता पिछली रात के चावल-आधारित भोजन से समर्थित नहीं हो सका"
                    )
                selected_meal = choose_item(
                    items=meal_items,
                    ekadashi=ekadashi,
                    recent_block_set=meal_recent,
                    consecutive_day_block_families=previous_day_repeat_families,
                    family_extractor=extract_meal_repeat_families,
                    keywords=keywords,
                    disallowed_keywords=disallowed_keywords,
                    fallback_policy=fallback_policy,
                    seed_key=f"{target_date_str}:meal",
                    weather_rules=weather_rules,
                    weather_tags=weather_tags,
                    warn_bucket=warning_items,
                    constraint_notes=missing_data_notes,
                    prefer_lighter=transition_plan.prefer_lighter,
                    light_fallback_items=light_fallback_items,
                    heavy_light_classification=heavy_light_classification,
                )
        else:
            selected_meal = choose_item(
                items=meal_items,
                ekadashi=ekadashi,
                recent_block_set=meal_recent,
                consecutive_day_block_families=previous_day_repeat_families,
                family_extractor=extract_meal_repeat_families,
                keywords=keywords,
                disallowed_keywords=disallowed_keywords,
                fallback_policy=fallback_policy,
                seed_key=f"{target_date_str}:meal",
                weather_rules=weather_rules,
                weather_tags=weather_tags,
                warn_bucket=warning_items,
                constraint_notes=missing_data_notes,
                prefer_lighter=transition_plan.prefer_lighter,
                light_fallback_items=light_fallback_items,
                heavy_light_classification=heavy_light_classification,
            )

        if is_heavy_item(selected_breakfast, weather_tags, heavy_light_classification) and is_heavy_item(
            selected_meal, weather_tags, heavy_light_classification
        ):
            if not breakfast_fixed:
                rebalanced_breakfast = choose_item(
                    items=breakfast_choice_items,
                    ekadashi=ekadashi,
                    recent_block_set=breakfast_recent,
                    consecutive_day_block_families=previous_day_repeat_families,
                    family_extractor=extract_breakfast_repeat_families,
                    keywords=keywords,
                    disallowed_keywords=disallowed_keywords,
                    fallback_policy=fallback_policy,
                    seed_key=f"{target_date_str}:breakfast:light-balance",
                    weather_rules=weather_rules,
                    weather_tags=weather_tags,
                    warn_bucket=warning_items,
                    constraint_notes=missing_data_notes,
                    prefer_lighter=True,
                    light_fallback_items=light_fallback_items,
                    max_lightness_score=0,
                    heavy_light_classification=heavy_light_classification,
                )
                if not is_heavy_item(rebalanced_breakfast, weather_tags, heavy_light_classification):
                    selected_breakfast = rebalanced_breakfast

            if is_heavy_item(selected_breakfast, weather_tags, heavy_light_classification) and is_heavy_item(
                selected_meal, weather_tags, heavy_light_classification
            ):
                if next_day_requires_rice_prep and rice_support_meal_candidates:
                    rice_light_pool = finalize_choice_pool(
                        base_pool=rice_support_meal_candidates,
                        recent_block_set=meal_recent,
                        consecutive_day_block_families=previous_day_repeat_families,
                        family_extractor=extract_meal_repeat_families,
                        weather_rules=weather_rules,
                        weather_tags=weather_tags,
                        warn_bucket=warning_items,
                        constraint_notes=missing_data_notes,
                        prefer_lighter=True,
                        max_lightness_score=0,
                        heavy_light_classification=heavy_light_classification,
                    )
                    if rice_light_pool:
                        rebalanced_meal = deterministic_choice(
                            rice_light_pool, f"{target_date_str}:meal:rice-support:light-balance"
                        )
                        if not is_heavy_item(rebalanced_meal, weather_tags, heavy_light_classification):
                            selected_meal = rebalanced_meal
                else:
                    rebalanced_meal = choose_item(
                        items=meal_items,
                        ekadashi=ekadashi,
                        recent_block_set=meal_recent,
                        consecutive_day_block_families=previous_day_repeat_families,
                        family_extractor=extract_meal_repeat_families,
                        keywords=keywords,
                        disallowed_keywords=disallowed_keywords,
                        fallback_policy=fallback_policy,
                        seed_key=f"{target_date_str}:meal:light-balance",
                        weather_rules=weather_rules,
                        weather_tags=weather_tags,
                        warn_bucket=warning_items,
                        constraint_notes=missing_data_notes,
                        prefer_lighter=True,
                        light_fallback_items=light_fallback_items,
                        max_lightness_score=0,
                        heavy_light_classification=heavy_light_classification,
                    )
                    if not is_heavy_item(rebalanced_meal, weather_tags, heavy_light_classification):
                        selected_meal = rebalanced_meal

            if (
                not breakfast_fixed
                and is_heavy_item(selected_breakfast, weather_tags, heavy_light_classification)
                and is_heavy_item(selected_meal, weather_tags, heavy_light_classification)
            ):
                forced_light = choose_item(
                    items=light_fallback_items,
                    ekadashi=ekadashi,
                    recent_block_set=set(),
                    consecutive_day_block_families=previous_day_repeat_families,
                    family_extractor=extract_breakfast_repeat_families,
                    keywords=keywords,
                    disallowed_keywords=disallowed_keywords,
                    fallback_policy=fallback_policy,
                    seed_key=f"{target_date_str}:forced-light",
                    weather_rules=None,
                    weather_tags=weather_tags,
                    warn_bucket=warning_items,
                    constraint_notes=missing_data_notes,
                    prefer_lighter=True,
                    light_fallback_items=light_fallback_items,
                    max_lightness_score=0,
                    heavy_light_classification=heavy_light_classification,
                )
                selected_breakfast = forced_light

        today_repeat_families = extract_breakfast_repeat_families(selected_breakfast) | extract_meal_repeat_families(
            selected_meal
        )
        if next_day_breakfast_lock:
            next_day_lock_conflicts = get_item_repeat_family_conflicts(
                next_day_breakfast_lock,
                today_repeat_families,
                extract_breakfast_repeat_families,
            )
            if next_day_lock_conflicts:
                missing_data_notes.append(
                    "[नियम] अगले दिन का overnight नाश्ता आज के चयन से टकराने के कारण हटाया गया: "
                    + " / ".join(next_day_lock_conflicts)
                )
                next_day_breakfast_lock = None
                next_day_requires_rice_prep = False

    if warning_items:
        print(
            "WARN: Untagged items were included in weather filtering: " + ", ".join(sorted(warning_items)),
            file=sys.stderr,
        )

    new_history = update_history(
        history,
        target_date_str,
        selected_breakfast,
        selected_meal,
        repeat_window_days,
        next_day_breakfast_lock=next_day_breakfast_lock,
        next_day_requires_rice_prep=next_day_requires_rice_prep,
    )
    write_json(HISTORY_FILE, new_history)

    lines = [
        f"*{target_date_display_str} तिथि के लिए भोजन:*",
        f"*ऋतु:* {display_ritu_hi}",
        f"*माह:* {panchang_info.maah_hi}",
        f"*तिथि (पंचांग):* {panchang_info.tithi_hi}",
    ]
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
        breakfast_display = (
            format_overnight_breakfast_label(selected_breakfast)
            if selected_breakfast in OVERNIGHT_BREAKFAST_ITEMS
            else selected_breakfast
        )
        lines.append(f"*सुबह का नाश्ता:* {breakfast_display}")
        if selected_breakfast in OVERNIGHT_BREAKFAST_ITEMS:
            prep_date_display_str = (target_date - timedelta(days=1)).strftime("%d-%b-%Y")
            lines.append(f"*नाश्ता विधि:* {selected_breakfast}")
            lines.append(
                "*नाश्ता तैयारी स्मरण:* "
                f"इस नाश्ते की तैयारी आज ({prep_date_display_str}) शाम से शुरू करें ताकि यह सुबह खाने के लिए तैयार रहे।"
            )
            if ritu_key == "vasant":
                lines.append("*नाश्ता स्वाद निर्देश:* वसंत में इसे थोड़ा अधिक तीखा रखें।")
            elif ritu_key == "grishm":
                lines.append("*नाश्ता स्वाद निर्देश:* ग्रीष्म में इसे सामान्य तीखापन रखें।")
        lines.append(f"*आज का भोजन:* {selected_meal}")
        if requires_mangore_prep(selected_breakfast, selected_meal):
            lines.append("*फॉलोवर महोदय हेतु रात की तैयारी:* " + MANGORE_PREP_NOTE)
        if next_day_requires_rice_prep and next_day_breakfast_lock:
            lines.append(
                "*कल सुबह का नाश्ता (रात की तैयारी से):* "
                + format_overnight_breakfast_label(next_day_breakfast_lock)
            )
            lines.append("*रात की तैयारी:* " + OVERNIGHT_RICE_PREP_NOTE)

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

    output_text = build_output_text(lines)

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        f.write(output_text + "\n")

    print(output_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
