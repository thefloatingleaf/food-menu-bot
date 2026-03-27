import unittest
from datetime import date

import generate_menu


class ConsecutiveDayRepeatRuleTests(unittest.TestCase):
    def test_meal_repeat_families_handle_variant_forms(self) -> None:
        item = "जो की रोटी और करेला–भिंडी मिश्रित सब्ज़ी"
        self.assertEqual(generate_menu.extract_meal_repeat_families(item), {"करेला", "भिंडी"})

    def test_previous_day_repeat_families_use_breakfast_main_and_meal_sabzi_only(self) -> None:
        history = [
            {
                "date": "2026-03-11",
                "breakfast": "आलू प्याज़ की रोटी",
                "meal": "ज्वार की रोटी, करेला, मूँग दाल धुली",
            }
        ]
        self.assertEqual(
            generate_menu.get_previous_day_repeat_families(history, date(2026, 3, 12)),
            {"आलू", "करेला"},
        )

    def test_breakfast_repeat_families_ignore_common_bases_and_track_main_breakfast(self) -> None:
        self.assertEqual(
            generate_menu.extract_breakfast_repeat_families("मूँग की दाल का चीला। पुदीने की चटनी।"),
            {"चीला", "मूंग"},
        )
        self.assertEqual(
            generate_menu.extract_meal_repeat_families("दही चावल ज्यादा करी पत्ता व सौंफ के साथ"),
            set(),
        )
        self.assertEqual(
            generate_menu.extract_meal_repeat_families("मूंग दाल वाली पतली खिचड़ी"),
            set(),
        )

    def test_apply_consecutive_day_repeat_rule_filters_conflicting_meal_sabzi(self) -> None:
        pool = [
            "जो की रोटी और भरवां करेला",
            "जो की रोटी और लौकी की सब्ज़ी",
            "भिंडी की सब्ज़ी, गेहूँ की रोटी",
        ]
        filtered, fell_back = generate_menu.apply_consecutive_day_repeat_rule(
            pool,
            {"करेला", "भिंडी"},
            generate_menu.extract_meal_repeat_families,
        )
        self.assertEqual(filtered, ["जो की रोटी और लौकी की सब्ज़ी"])
        self.assertFalse(fell_back)

    def test_apply_consecutive_day_repeat_rule_falls_back_when_pool_is_exhausted(self) -> None:
        pool = [
            "जो की रोटी और भरवां करेला",
            "करेला, गेहूँ की रोटी",
        ]
        filtered, fell_back = generate_menu.apply_consecutive_day_repeat_rule(
            pool,
            {"करेला"},
            generate_menu.extract_meal_repeat_families,
        )
        self.assertEqual(filtered, pool)
        self.assertTrue(fell_back)


class OvernightBreakfastFormattingTests(unittest.TestCase):
    def test_format_overnight_breakfast_label_uses_short_name(self) -> None:
        full_item = (
            "पखाला भात (Pakhala Bhata): रात में 1 कटोरी कच्चे चावल धोकर सादा चावल पकाएँ।"
        )
        self.assertEqual(
            generate_menu.format_overnight_breakfast_label(full_item),
            "पखाला भात (Pakhala Bhata)",
        )

    def test_format_overnight_breakfast_label_keeps_plain_item(self) -> None:
        plain_item = "पझैया सादम (Pazhaya Sadam)"
        self.assertEqual(generate_menu.format_overnight_breakfast_label(plain_item), plain_item)


class MangorePrepInstructionTests(unittest.TestCase):
    def test_requires_mangore_prep_for_mangaunde_spelling(self) -> None:
        self.assertTrue(generate_menu.requires_mangore_prep("उबले हुए मंगौड़े", "सादा भोजन"))

    def test_requires_mangore_prep_for_breakfast_item(self) -> None:
        self.assertTrue(generate_menu.requires_mangore_prep("उबले हुए मंगोड़े", "सादा भोजन"))

    def test_requires_mangore_prep_for_meal_item(self) -> None:
        self.assertTrue(generate_menu.requires_mangore_prep("", "उबले मंगोड़े की रस्सेदार सब्ज़ी और गेहूं रोटी"))

    def test_requires_mangore_prep_ignores_other_items(self) -> None:
        self.assertFalse(generate_menu.requires_mangore_prep("पोहा", "दाल और रोटी"))


class FestivalSpecialMenuTests(unittest.TestCase):
    def test_resolve_festival_info_reads_explicit_special_menu_lines(self) -> None:
        info = generate_menu.resolve_festival_info(
            "2026-03-26",
            {
                "entries": [
                    {
                        "date": "2026-03-26",
                        "hindu_hi": ["चैत्र नवरात्रि"],
                        "sikh_hi": [],
                        "suppress_regular_menu": True,
                        "special_menu_lines_hi": [
                            "*विशेष अष्टमी मेनू:*",
                            "1. काले चने — 4 कटोरी।",
                        ],
                    }
                ]
            },
        )

        self.assertEqual(
            info.special_menu_lines_hi,
            ["*विशेष अष्टमी मेनू:*", "1. काले चने — 4 कटोरी।"],
        )

    def test_resolve_festival_info_reads_no_menu_special_note(self) -> None:
        info = generate_menu.resolve_festival_info(
            "2026-03-19",
            {
                "entries": [
                    {
                        "date": "2026-03-19",
                        "hindu_hi": ["चैत्र नवरात्रि"],
                        "sikh_hi": [],
                        "suppress_regular_menu": True,
                        "special_menu_note_hi": (
                            "नवरात्रि दिवस 1, माँ शैलपुत्री: आज विशेष रूप से देसी घी ग्रहण करें या भोग में अर्पित करें।"
                        ),
                    }
                ]
            },
        )

        self.assertEqual(info.hindu_hi, ["चैत्र नवरात्रि"])
        self.assertTrue(info.suppress_regular_menu)
        self.assertEqual(
            info.special_menu_note_hi,
            "नवरात्रि दिवस 1, माँ शैलपुत्री: आज विशेष रूप से देसी घी ग्रहण करें या भोग में अर्पित करें।",
        )

    def test_resolve_festival_info_suppresses_chaitra_navratri_day_three(self) -> None:
        info = generate_menu.resolve_festival_info(
            "2026-03-21",
            {
                "entries": [
                    {
                        "date": "2026-03-21",
                        "hindu_hi": ["चैत्र नवरात्रि"],
                        "sikh_hi": [],
                        "suppress_regular_menu": True,
                        "special_menu_note_hi": (
                            "नवरात्रि दिवस 3, माँ चंद्रघंटा: आज विशेष रूप से खीर ग्रहण करें या भोग में अर्पित करें।"
                        ),
                    }
                ]
            },
        )

        self.assertEqual(info.hindu_hi, ["चैत्र नवरात्रि"])
        self.assertTrue(info.suppress_regular_menu)
        self.assertEqual(
            info.special_menu_note_hi,
            "नवरात्रि दिवस 3, माँ चंद्रघंटा: आज विशेष रूप से खीर ग्रहण करें या भोग में अर्पित करें।",
        )

    def test_resolve_festival_info_falls_back_when_navratri_row_is_missing(self) -> None:
        info = generate_menu.resolve_festival_info("2026-03-22", {"entries": []})

        self.assertEqual(info.hindu_hi, ["चैत्र नवरात्रि"])
        self.assertTrue(info.suppress_regular_menu)
        self.assertEqual(
            info.special_menu_note_hi,
            "नवरात्रि दिवस 4, माँ कूष्मांडा: आज विशेष रूप से मालपुआ ग्रहण करें या भोग में अर्पित करें।",
        )

    def test_resolve_festival_info_falls_back_for_sharad_navratri_2026(self) -> None:
        info = generate_menu.resolve_festival_info("2026-10-11", {"entries": []})

        self.assertEqual(info.hindu_hi, ["शारदीय नवरात्रि"])
        self.assertTrue(info.suppress_regular_menu)
        self.assertEqual(
            info.special_menu_note_hi,
            "नवरात्रि दिवस 1, माँ शैलपुत्री: आज विशेष रूप से देसी घी ग्रहण करें या भोग में अर्पित करें।",
        )

    def test_resolve_festival_info_falls_back_for_chaitra_navratri_2027_ashtami_menu(self) -> None:
        info = generate_menu.resolve_festival_info("2027-04-14", {"entries": []})

        self.assertEqual(info.hindu_hi, ["चैत्र नवरात्रि"])
        self.assertTrue(info.suppress_regular_menu)
        self.assertEqual(
            info.special_menu_note_hi,
            "नवरात्रि दिवस 8, माँ महागौरी: आज विशेष रूप से नारियल ग्रहण करें या भोग में अर्पित करें।",
        )
        self.assertIsNotNone(info.special_menu_lines_hi)
        self.assertIn("1. काले चने — 4 कटोरी।", info.special_menu_lines_hi)

    def test_resolve_festival_info_marks_vijayadashami_in_2026_range(self) -> None:
        info = generate_menu.resolve_festival_info("2026-10-20", {"entries": []})

        self.assertEqual(info.hindu_hi, ["शारदीय नवरात्रि", "विजयादशमी"])
        self.assertTrue(info.suppress_regular_menu)
        self.assertEqual(
            info.special_menu_note_hi,
            "विजयादशमी: आज विशेष रूप से सात्त्विक भोग अर्पित करें और शुभ कार्य आरंभ करें।",
        )

    def test_resolve_festival_info_falls_back_for_sharad_navratri_2027_ashtami_menu(self) -> None:
        info = generate_menu.resolve_festival_info("2027-10-07", {"entries": []})

        self.assertEqual(info.hindu_hi, ["शारदीय नवरात्रि"])
        self.assertTrue(info.suppress_regular_menu)
        self.assertEqual(
            info.special_menu_note_hi,
            "नवरात्रि दिवस 8, माँ महागौरी: आज विशेष रूप से नारियल ग्रहण करें या भोग में अर्पित करें।",
        )
        self.assertIsNotNone(info.special_menu_lines_hi)
        self.assertIn("1. काले चने — 4 कटोरी।", info.special_menu_lines_hi)

    def test_resolve_festival_info_falls_back_for_chaitra_navratri_2028_ashtami_menu(self) -> None:
        info = generate_menu.resolve_festival_info("2028-04-03", {"entries": []})

        self.assertEqual(info.hindu_hi, ["चैत्र नवरात्रि"])
        self.assertTrue(info.suppress_regular_menu)
        self.assertEqual(
            info.special_menu_note_hi,
            "नवरात्रि दिवस 8, माँ महागौरी: आज विशेष रूप से नारियल ग्रहण करें या भोग में अर्पित करें।",
        )
        self.assertIsNotNone(info.special_menu_lines_hi)
        self.assertIn("1. काले चने — 4 कटोरी।", info.special_menu_lines_hi)

    def test_resolve_festival_info_marks_vijayadashami_in_2028_range(self) -> None:
        info = generate_menu.resolve_festival_info("2028-09-28", {"entries": []})

        self.assertEqual(info.hindu_hi, ["शारदीय नवरात्रि", "विजयादशमी"])
        self.assertTrue(info.suppress_regular_menu)
        self.assertEqual(
            info.special_menu_note_hi,
            "विजयादशमी: आज विशेष रूप से सात्त्विक भोग अर्पित करें और शुभ कार्य आरंभ करें।",
        )

    def test_resolve_festival_info_has_no_2026_chaitra_navratri_fallback_on_march_27(self) -> None:
        info = generate_menu.resolve_festival_info("2026-03-27", {"entries": []})

        self.assertEqual(info.hindu_hi, [])
        self.assertFalse(info.suppress_regular_menu)
        self.assertIsNone(info.special_menu_note_hi)

    def test_resolve_festival_info_merges_in_fallback_when_flag_is_missing(self) -> None:
        info = generate_menu.resolve_festival_info(
            "2027-04-15",
            {
                "entries": [
                    {
                        "date": "2027-04-15",
                        "hindu_hi": ["राम नवमी"],
                        "sikh_hi": [],
                    }
                ]
            },
        )

        self.assertEqual(info.hindu_hi, ["राम नवमी", "चैत्र नवरात्रि"])
        self.assertTrue(info.suppress_regular_menu)
        self.assertEqual(
            info.special_menu_note_hi,
            "नवरात्रि दिवस 9 / राम नवमी, माँ सिद्धिदात्री: आज विशेष रूप से तिल या व्रत-उपयोगी सात्त्विक भोग अर्पित करें।",
        )

    def test_format_special_menu_note_line_uses_special_note(self) -> None:
        info = generate_menu.FestivalInfo(
            hindu_hi=["चैत्र नवरात्रि"],
            sikh_hi=[],
            suppress_regular_menu=True,
            special_menu_note_hi="नवरात्रि दिवस 8, माँ महागौरी: आज विशेष रूप से नारियल ग्रहण करें या भोग में अर्पित करें।",
        )

        self.assertEqual(
            generate_menu.format_special_menu_note_line(info),
            "*विशेष पारंपरिक सेवन/भोग:* नवरात्रि दिवस 8, माँ महागौरी: आज विशेष रूप से नारियल ग्रहण करें या भोग में अर्पित करें।",
        )

    def test_apply_recurring_festival_menu_overrides_adds_navratri_ashtami_menu(self) -> None:
        festival_info = generate_menu.FestivalInfo(
            hindu_hi=["चैत्र नवरात्रि"],
            sikh_hi=[],
            suppress_regular_menu=True,
            special_menu_note_hi="नवरात्रि दिवस 8, माँ महागौरी: आज विशेष रूप से नारियल ग्रहण करें या भोग में अर्पित करें।",
        )
        panchang_info = generate_menu.PanchangInfo(ritu_hi="वसंत", maah_hi="चैत्र", tithi_hi="अष्टमी")

        updated = generate_menu.apply_recurring_festival_menu_overrides(festival_info, panchang_info)

        self.assertIsNotNone(updated.special_menu_lines_hi)
        self.assertIn("1. काले चने — 4 कटोरी।", updated.special_menu_lines_hi)
        self.assertIn("*विशेष निर्देश:* किसी भी वस्तु में प्याज बिल्कुल न डाला जाए।", updated.special_menu_lines_hi)

    def test_apply_recurring_festival_menu_overrides_keeps_existing_explicit_lines(self) -> None:
        festival_info = generate_menu.FestivalInfo(
            hindu_hi=["चैत्र नवरात्रि"],
            sikh_hi=[],
            suppress_regular_menu=True,
            special_menu_lines_hi=["*विशेष अष्टमी मेनू:*", "1. Custom item"],
        )
        panchang_info = generate_menu.PanchangInfo(ritu_hi="वसंत", maah_hi="चैत्र", tithi_hi="अष्टमी")

        updated = generate_menu.apply_recurring_festival_menu_overrides(festival_info, panchang_info)

        self.assertEqual(updated.special_menu_lines_hi, ["*विशेष अष्टमी मेनू:*", "1. Custom item"])


if __name__ == "__main__":
    unittest.main()
