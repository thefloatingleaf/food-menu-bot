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


class VarietyCycleRuleTests(unittest.TestCase):
    def test_normalize_history_preserves_ritu_key(self) -> None:
        normalized = generate_menu.normalize_history(
            [
                {
                    "date": "2026-03-12",
                    "breakfast": "पोहा",
                    "meal": "दाल और रोटी",
                    "ritu_key": "वसंत",
                }
            ]
        )

        self.assertEqual(normalized[0]["ritu_key"], "vasant")

    def test_get_variety_cycle_used_items_reads_same_ritu_only(self) -> None:
        history = [
            {
                "date": "2026-03-11",
                "breakfast": "पोहा",
                "meal": "दाल और रोटी",
                "ritu_key": "vasant",
            },
            {
                "date": "2026-03-12",
                "breakfast": "उपमा",
                "meal": "भिंडी",
                "ritu_key": "grishm",
            },
            {
                "date": "2026-03-13",
                "breakfast": "इडली",
                "meal": "लौकी",
                "ritu_key": "vasant",
            },
        ]

        self.assertEqual(
            generate_menu.get_variety_cycle_used_items(history, date(2026, 3, 14), "breakfast", "vasant"),
            {"पोहा", "इडली"},
        )

    def test_apply_variety_cycle_rule_filters_until_cycle_exhausts(self) -> None:
        filtered, reset = generate_menu.apply_variety_cycle_rule(["पोहा", "उपमा", "इडली"], {"पोहा", "इडली"})
        self.assertEqual(filtered, ["उपमा"])
        self.assertFalse(reset)

    def test_apply_variety_cycle_rule_resets_after_full_cycle(self) -> None:
        filtered, reset = generate_menu.apply_variety_cycle_rule(["पोहा", "उपमा"], {"पोहा", "उपमा"})
        self.assertEqual(filtered, ["पोहा", "उपमा"])
        self.assertTrue(reset)

    def test_choose_item_prefers_unused_item_in_same_ritu_cycle(self) -> None:
        selected = generate_menu.choose_item(
            items=["पोहा", "उपमा"],
            ekadashi=generate_menu.EkadashiInfo(False, None, None),
            cycle_block_set={"पोहा"},
            recent_block_set=set(),
            consecutive_day_block_families=set(),
            family_extractor=generate_menu.extract_breakfast_repeat_families,
            keywords=[],
            disallowed_keywords=[],
            fallback_policy="fallback_full_menu",
            seed_key="2026-03-14:breakfast",
            weather_rules=None,
            weather_tags={},
            warn_bucket=set(),
            constraint_notes=[],
            prefer_lighter=False,
            light_fallback_items=[],
        )

        self.assertEqual(selected, "उपमा")


class VasantDayTenTests(unittest.TestCase):
    def test_is_vasant_day_ten_matches_tenth_day_from_vasant_start(self) -> None:
        self.assertTrue(generate_menu.is_vasant_day_ten(date(2026, 3, 24), "vasant"))

    def test_is_vasant_day_ten_rejects_other_days_or_ritus(self) -> None:
        self.assertFalse(generate_menu.is_vasant_day_ten(date(2026, 3, 23), "vasant"))
        self.assertFalse(generate_menu.is_vasant_day_ten(date(2026, 3, 24), "grishm"))

    def test_append_vasant_neem_ghee_lines_adds_full_recipe(self) -> None:
        lines = []
        generate_menu.append_vasant_neem_ghee_lines(lines)

        self.assertEqual(lines[0], "*वसंत दशम-दिवस स्मरण:* नीम का घी बनाएं।")
        self.assertIn("1. ताज़ी नीम की पत्तियाँ अच्छी तरह साफ कर लें।", lines)
        self.assertIn("6. इसके बाद घी को छान लें और प्रयोग में लाएँ।", lines)


class VasantRotiRotationTests(unittest.TestCase):
    def test_canonicalize_vasant_meal_items_uses_exact_allowed_grain_labels(self) -> None:
        canonical = generate_menu.canonicalize_vasant_meal_items(
            [
                "जो की रोटी और लौकी की सब्ज़ी",
                "गेहू की रोटी और भिंडी की सूखी सब्ज़ी",
                "चने और जो की रोटी (मिस्सी रोटी) और अरहर दाल",
            ]
        )

        self.assertEqual(
            canonical,
            [
                "जौ (केवल पुराना) की रोटी और लौकी की सब्ज़ी",
                "गेहूँ (केवल पुराना) की रोटी और भिंडी की सूखी सब्ज़ी",
                "चने और जौ की रोटी (मिस्सी रोटी) और अरहर दाल",
            ],
        )

    def test_extract_vasant_roti_grain_option_identifies_allowed_option(self) -> None:
        self.assertEqual(
            generate_menu.extract_vasant_roti_grain_option(
                "रागी (केवल पुराना) की रोटी और परवल की सब्ज़ी",
                "vasant",
            ),
            "रागी (केवल पुराना)",
        )
        self.assertIsNone(
            generate_menu.extract_vasant_roti_grain_option("मूंग दाल और चावल", "vasant")
        )

    def test_apply_vasant_roti_grain_rotation_rule_filters_used_grain_options(self) -> None:
        filtered, reset = generate_menu.apply_vasant_roti_grain_rotation_rule(
            [
                "जौ (केवल पुराना) की रोटी और लौकी की सब्ज़ी",
                "ज्वार (केवल पुराना) की रोटी और लौकी की सब्ज़ी",
                "मूंग दाल और चावल",
            ],
            {"जौ (केवल पुराना)"},
            "vasant",
        )

        self.assertEqual(
            filtered,
            [
                "ज्वार (केवल पुराना) की रोटी और लौकी की सब्ज़ी",
                "मूंग दाल और चावल",
            ],
        )
        self.assertFalse(reset)

    def test_apply_vasant_roti_grain_rotation_rule_resets_after_all_present_options_are_used(self) -> None:
        filtered, reset = generate_menu.apply_vasant_roti_grain_rotation_rule(
            [
                "जौ (केवल पुराना) की रोटी और लौकी की सब्ज़ी",
                "ज्वार (केवल पुराना) की रोटी और भिंडी की सूखी सब्ज़ी",
            ],
            {"जौ (केवल पुराना)", "ज्वार (केवल पुराना)", "रागी (केवल पुराना)"},
            "vasant",
        )

        self.assertEqual(
            filtered,
            [
                "जौ (केवल पुराना) की रोटी और लौकी की सब्ज़ी",
                "ज्वार (केवल पुराना) की रोटी और भिंडी की सूखी सब्ज़ी",
            ],
        )
        self.assertTrue(reset)

    def test_get_vasant_roti_grain_cycle_used_options_reads_history_by_option(self) -> None:
        history = [
            {
                "date": "2026-03-15",
                "meal": "जो की रोटी और लौकी की सब्ज़ी",
                "ritu_key": "vasant",
            },
            {
                "date": "2026-03-16",
                "meal": "मूंग दाल और चावल",
                "ritu_key": "vasant",
            },
            {
                "date": "2026-03-17",
                "meal": "ज्वार (केवल पुराना) की रोटी और अरहर दाल",
                "ritu_key": "vasant",
            },
        ]

        self.assertEqual(
            generate_menu.get_vasant_roti_grain_cycle_used_options(history, date(2026, 3, 18), "vasant"),
            {"जौ (केवल पुराना)", "ज्वार (केवल पुराना)"},
        )


class OvernightBreakfastFormattingTests(unittest.TestCase):
    def test_same_day_generation_cannot_apply_overnight_breakfast(self) -> None:
        self.assertFalse(
            generate_menu.can_apply_overnight_breakfast_on_run_date(date(2026, 3, 27), date(2026, 3, 27))
        )

    def test_next_day_generation_can_apply_overnight_breakfast(self) -> None:
        self.assertTrue(
            generate_menu.can_apply_overnight_breakfast_on_run_date(date(2026, 3, 27), date(2026, 3, 26))
        )

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
