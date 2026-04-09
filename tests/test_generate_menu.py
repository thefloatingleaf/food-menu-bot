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

    def test_previous_day_repeat_families_include_second_meal_sabzi(self) -> None:
        history = [
            {
                "date": "2026-03-11",
                "breakfast": "आलू प्याज़ की रोटी",
                "meal": "ज्वार की रोटी, करेला, मूँग दाल धुली",
                "second_meal": "गेहूँ की रोटी और भिंडी की सब्ज़ी",
            }
        ]
        self.assertEqual(
            generate_menu.get_previous_day_repeat_families(history, date(2026, 3, 12)),
            {"आलू", "करेला", "भिंडी"},
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
    def test_normalize_history_preserves_fruit(self) -> None:
        normalized = generate_menu.normalize_history(
            [
                {
                    "date": "2026-04-10",
                    "breakfast": "पोहा",
                    "meal": "दाल और रोटी",
                    "fruit": "बेल",
                    "ritu_key": "वसंत",
                }
            ]
        )

        self.assertEqual(normalized[0]["fruit"], "बेल")

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

    def test_normalize_history_preserves_second_meal(self) -> None:
        normalized = generate_menu.normalize_history(
            [
                {
                    "date": "2026-04-10",
                    "breakfast": "पोहा",
                    "meal": "दाल और रोटी",
                    "second_meal": "भिंडी और रोटी",
                    "ritu_key": "वसंत",
                }
            ]
        )

        self.assertEqual(normalized[0]["second_meal"], "भिंडी और रोटी")

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

    def test_recent_items_includes_second_meal_for_meal_history(self) -> None:
        history = [
            {
                "date": "2026-04-09",
                "breakfast": "उपमा",
                "meal": "चने और जौ (Barley) की रोटी (मिस्सी रोटी) और परवल-मूँगदाल की सूखी सब्ज़ी",
                "second_meal": "गेहूँ (Wheat) (केवल पुराना) की रोटी और भिंडी की सूखी सब्ज़ी",
                "ritu_key": "vasant",
            }
        ]

        self.assertEqual(
            generate_menu.recent_items(history, date(2026, 4, 10), 7, "meal"),
            {
                "चने और जौ (Barley) की रोटी (मिस्सी रोटी) और परवल-मूँगदाल की सूखी सब्ज़ी",
                "गेहूँ (Wheat) (केवल पुराना) की रोटी और भिंडी की सूखी सब्ज़ी",
            },
        )

    def test_get_variety_cycle_used_items_includes_second_meal_for_meal_history(self) -> None:
        history = [
            {
                "date": "2026-04-09",
                "breakfast": "उपमा",
                "meal": "चने और जौ (Barley) की रोटी (मिस्सी रोटी) और परवल-मूँगदाल की सूखी सब्ज़ी",
                "second_meal": "गेहूँ (Wheat) (केवल पुराना) की रोटी और भिंडी की सूखी सब्ज़ी",
                "ritu_key": "vasant",
            }
        ]

        self.assertEqual(
            generate_menu.get_variety_cycle_used_items(history, date(2026, 4, 10), "meal", "vasant"),
            {
                "चने और जौ (Barley) की रोटी (मिस्सी रोटी) और परवल-मूँगदाल की सूखी सब्ज़ी",
                "गेहूँ (Wheat) (केवल पुराना) की रोटी और भिंडी की सूखी सब्ज़ी",
            },
        )

    def test_vasant_meal_rotation_readers_include_second_meal(self) -> None:
        history = [
            {
                "date": "2026-04-09",
                "breakfast": "उपमा",
                "meal": "ज्वार (Sorghum) (केवल पुराना) की रोटी और लौकी की सब्ज़ी",
                "second_meal": "चने और जौ (Barley) की रोटी (मिस्सी रोटी) और अरहर दाल",
                "ritu_key": "vasant",
            }
        ]

        self.assertEqual(
            generate_menu.get_vasant_roti_grain_cycle_used_options(history, date(2026, 4, 10), "vasant"),
            {
                "ज्वार (Sorghum) (केवल पुराना)",
                "चने और जौ (Barley) की रोटी (मिस्सी रोटी)",
            },
        )
        self.assertEqual(
            generate_menu.get_vasant_dal_cycle_used_options(history, date(2026, 4, 10), "vasant"),
            {"अरहर"},
        )

    def test_apply_variety_cycle_rule_filters_until_cycle_exhausts(self) -> None:
        filtered, reset = generate_menu.apply_variety_cycle_rule(["पोहा", "उपमा", "इडली"], {"पोहा", "इडली"})
        self.assertEqual(filtered, ["उपमा"])
        self.assertFalse(reset)

    def test_apply_variety_cycle_rule_resets_after_full_cycle(self) -> None:
        filtered, reset = generate_menu.apply_variety_cycle_rule(["पोहा", "उपमा"], {"पोहा", "उपमा"})
        self.assertEqual(filtered, ["पोहा", "उपमा"])
        self.assertTrue(reset)

    def test_update_history_persists_second_meal(self) -> None:
        updated = generate_menu.update_history(
            history=[],
            target_date="2026-04-10",
            breakfast_item="उपमा",
            meal_item="ज्वार (Sorghum) (केवल पुराना) की रोटी और लौकी की सब्ज़ी",
            second_meal_item="गेहूँ (Wheat) (केवल पुराना) की रोटी और भिंडी की सूखी सब्ज़ी",
            fruit_item="सेब 🍎",
            keep_days=7,
            ritu_key="vasant",
        )

        self.assertEqual(
            updated[0]["second_meal"],
            "गेहूँ (Wheat) (केवल पुराना) की रोटी और भिंडी की सूखी सब्ज़ी",
        )

    def test_choose_item_prefers_unused_item_in_same_ritu_cycle(self) -> None:
        selected = generate_menu.choose_item(
            items=["पोहा", "उपमा"],
            ekadashi=generate_menu.EkadashiInfo(False, None, None),
            cycle_block_set={"पोहा"},
            recent_block_set=set(),
            consecutive_day_block_families=set(),
            recent_family_block_families=set(),
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

    def test_choose_item_blocks_chilla_family_with_weekly_family_rule(self) -> None:
        selected = generate_menu.choose_item(
            items=["मूँग दाल चीला", "उपमा"],
            ekadashi=generate_menu.EkadashiInfo(False, None, None),
            cycle_block_set=set(),
            recent_block_set=set(),
            consecutive_day_block_families=set(),
            recent_family_block_families={"चीला"},
            family_extractor=generate_menu.extract_breakfast_repeat_families,
            keywords=[],
            disallowed_keywords=[],
            fallback_policy="fallback_full_menu",
            seed_key="2026-03-15:breakfast",
            weather_rules=None,
            weather_tags={},
            warn_bucket=set(),
            constraint_notes=[],
            prefer_lighter=False,
            light_fallback_items=[],
        )

        self.assertEqual(selected, "उपमा")

    def test_get_recent_breakfast_family_block_families_tracks_chilla_within_window(self) -> None:
        history = [
            {
                "date": "2026-03-10",
                "breakfast": "मूँग दाल चीला",
                "meal": "दाल और चावल",
                "ritu_key": "vasant",
            },
            {
                "date": "2026-03-02",
                "breakfast": "बेसन चीला",
                "meal": "भिंडी",
                "ritu_key": "vasant",
            },
        ]

        self.assertEqual(
            generate_menu.get_recent_breakfast_family_block_families(history, date(2026, 3, 14), 7),
            {"चीला"},
        )

    def test_select_monthly_fruit_prefers_unused_april_fruit(self) -> None:
        history = [
            {
                "date": "2026-04-08",
                "breakfast": "पोहा",
                "meal": "दाल और चावल",
                "fruit": "शहतूत",
                "ritu_key": "vasant",
            },
            {
                "date": "2026-04-09",
                "breakfast": "उपमा",
                "meal": "लौकी",
                "fruit": "लोकाट",
                "ritu_key": "vasant",
            },
        ]
        monthly_fruit_map = {4: ["शहतूत", "लोकाट", "चीकू", "अंगूर"]}
        selection = generate_menu.select_monthly_fruit(history, date(2026, 4, 10), monthly_fruit_map, {})

        self.assertTrue(selection.available)
        self.assertIn(selection.fruit, {"चीकू", "अंगूर"})

    def test_select_monthly_fruit_allows_mango_priority_in_may(self) -> None:
        history = [
            {
                "date": "2026-05-08",
                "breakfast": "पोहा",
                "meal": "दाल और चावल",
                "fruit": "आम",
                "ritu_key": "grishm",
            }
        ]
        monthly_fruit_map = {5: ["आम", "तरबूज", "खरबूजा"]}
        priority_rules = {"आम": {"months": [5, 6], "weight": 4}}
        selection = generate_menu.select_monthly_fruit(history, date(2026, 5, 10), monthly_fruit_map, priority_rules)

        self.assertTrue(selection.available)
        self.assertIn(selection.fruit, {"आम", "तरबूज", "खरबूजा"})

    def test_select_monthly_fruit_returns_unavailable_when_month_missing(self) -> None:
        selection = generate_menu.select_monthly_fruit([], date(2026, 4, 10), {}, {})
        self.assertFalse(selection.available)
        self.assertIsNone(selection.fruit)

    def test_resolve_item_date_override_supports_fruit_override_entries(self) -> None:
        override = generate_menu.resolve_item_date_override(
            date(2026, 4, 12),
            {
                "fruit_item_date_overrides": [
                    {"date": "2026-04-12", "item": "सेब 🍎"},
                    {"date": "2026-04-13", "item": "नाशपाती"},
                ]
            },
            "fruit_item_date_overrides",
        )

        self.assertEqual(override, "सेब 🍎")

    def test_get_monthly_fruit_usage_counts_resets_on_month_transition(self) -> None:
        history = [
            {
                "date": "2026-04-30",
                "breakfast": "पोहा",
                "meal": "दाल और चावल",
                "fruit": "बेल",
                "ritu_key": "vasant",
            },
            {
                "date": "2026-05-01",
                "breakfast": "उपमा",
                "meal": "लौकी",
                "fruit": "आम",
                "ritu_key": "grishm",
            },
        ]

        self.assertEqual(generate_menu.get_monthly_fruit_usage_counts(history, date(2026, 5, 2)), {"आम": 1})

    def test_format_today_fruit_line_uses_new_label_and_vasant_timing_note(self) -> None:
        line = generate_menu.format_today_fruit_line(generate_menu.FruitSelection("पपीता", True), "vasant")
        self.assertEqual(line, "*आज का फल:* पपीता (फल सुबह 6–10 में न लें)")

    def test_format_today_fruit_line_uses_unavailable_fallback(self) -> None:
        line = generate_menu.format_today_fruit_line(generate_menu.FruitSelection(None, False), "vasant")
        self.assertEqual(line, "*आज का फल:* फल उपलब्ध नहीं है")

    def test_build_next_day_overnight_prep_line_does_not_create_second_breakfast_heading(self) -> None:
        line = generate_menu.build_next_day_overnight_prep_line(
            "पखाला भात (Pakhala Bhata): रात में 1 कटोरी कच्चे चावल धोकर सादा चावल पकाएँ।"
        )
        self.assertTrue(line.startswith("*रात की तैयारी (पखाला भात (Pakhala Bhata) के लिए):*"))
        self.assertNotIn("कल सुबह का नाश्ता", line)

    def test_resolve_available_override_item_matches_canonical_vasant_meal_label(self) -> None:
        resolved = generate_menu.resolve_available_override_item(
            "ज्वार की रोटी और लौकी की सब्ज़ी",
            ["ज्वार (Sorghum) (केवल पुराना) की रोटी और लौकी की सब्ज़ी"],
        )
        self.assertEqual(resolved, "ज्वार (Sorghum) (केवल पुराना) की रोटी और लौकी की सब्ज़ी")

    def test_apply_second_meal_override_rejects_same_meal_as_first(self) -> None:
        second, note = generate_menu.apply_second_meal_override(
            "मूंग दाल और चावल",
            "मूंग दाल और चावल",
            "जौ (Barley) (केवल पुराना) की रोटी और लौकी की सब्ज़ी",
            ["मूंग दाल और चावल", "जौ (Barley) (केवल पुराना) की रोटी और लौकी की सब्ज़ी"],
        )
        self.assertEqual(second, "जौ (Barley) (केवल पुराना) की रोटी और लौकी की सब्ज़ी")
        self.assertEqual(note, "[नियम] निर्धारित दूसरा भोजन override पहले भोजन से अलग होना चाहिए")

    def test_collect_vasant_prohibited_warnings_finds_actual_conflicts_only(self) -> None:
        lines = [
            "*आज का फल:* पपीता (फल सुबह 6–10 में न लें)",
            "*नाश्ता विधि:* पझैया सादम में दही मिलाएँ।",
            "*वसंत भोजन अनिवार्य साथ:* तीखा अचार (खट्टा नहीं) / मसाला छाछ",
        ]

        self.assertEqual(generate_menu.collect_vasant_prohibited_warnings(lines), ["दही"])

    def test_collect_vasant_prohibited_warnings_detects_explicit_fruit_timing_violation(self) -> None:
        lines = ["*आज का फल:* पपीता फल 6–10 में लें"]
        self.assertEqual(generate_menu.collect_vasant_prohibited_warnings(lines), ["फल सुबह 6 से 10 के बीच"])

    def test_exclude_meals_incompatible_with_pazhaya_sadam(self) -> None:
        meals = ["छाछ की सब्ज़ी चावल के साथ", "मूंग दाल और चावल"]
        filtered = generate_menu.exclude_meals_incompatible_with_breakfast(
            "पझैया सादम (Pazhaya Sadam): बचे हुए चावल लें या फिर 1 कटोरी कच्चे चावल अच्छी तरह धोकर सादा चावल पकाएँ।",
            meals,
        )
        self.assertEqual(filtered, ["मूंग दाल और चावल"])


class DateResolutionTests(unittest.TestCase):
    def test_resolve_date_uses_explicit_date_as_target_menu_date(self) -> None:
        self.assertEqual(
            generate_menu.resolve_date("2026-04-06", "Asia/Kolkata"),
            date(2026, 4, 6),
        )

    def test_resolve_date_without_explicit_date_defaults_to_tomorrow(self) -> None:
        self.assertEqual(
            generate_menu.resolve_date(None, "Asia/Kolkata", now_date=date(2026, 4, 6)),
            date(2026, 4, 7),
        )

    def test_is_double_meal_window_matches_requested_april_range(self) -> None:
        self.assertTrue(generate_menu.is_double_meal_window(date(2026, 4, 8)))
        self.assertTrue(generate_menu.is_double_meal_window(date(2026, 4, 14)))
        self.assertFalse(generate_menu.is_double_meal_window(date(2026, 4, 7)))
        self.assertFalse(generate_menu.is_double_meal_window(date(2026, 4, 15)))

    def test_select_second_meal_for_window_avoids_second_rice_meal(self) -> None:
        second = generate_menu.select_second_meal_for_window(
            selected_meal="मूंग दाल और चावल",
            meal_choice_items=["मूंग दाल और चावल", "जौ (Barley) (केवल पुराना) की रोटी और लौकी की सब्ज़ी"],
            ekadashi=generate_menu.EkadashiInfo(False, None, None),
            meal_cycle_block_set=set(),
            meal_recent=set(),
            previous_day_repeat_families=set(),
            keywords=[],
            disallowed_keywords=[],
            fallback_policy="fallback_full_menu",
            target_date_str="2026-04-08",
            weather_rules=None,
            weather_tags={},
            warning_items=set(),
            missing_data_notes=[],
            transition_prefer_lighter=False,
            light_fallback_items=[],
            heavy_light_classification=None,
        )

        self.assertEqual(second, "जौ (Barley) (केवल पुराना) की रोटी और लौकी की सब्ज़ी")


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
                "काला चना और चावल",
                "रागी की रोटी और कटहल की सब्ज़ी",
            ]
        )

        self.assertIn("जौ (Barley) (केवल पुराना) की रोटी और लौकी की सब्ज़ी", canonical)
        self.assertIn("गेहूँ (Wheat) (केवल पुराना) की रोटी और भिंडी की सूखी सब्ज़ी", canonical)
        self.assertIn("चने और जौ (Barley) की रोटी (मिस्सी रोटी) और अरहर दाल", canonical)
        self.assertIn("चने-लौकी की दाल और चावल", canonical)
        self.assertIn("रागी (Finger Millet) (केवल पुराना) की रोटी और कटहल की सब्ज़ी", canonical)
        self.assertIn(
            "चने और जौ (Barley) की रोटी (मिस्सी रोटी) और चने-लौकी की दाल",
            canonical,
        )

    def test_extract_vasant_roti_grain_option_identifies_allowed_option(self) -> None:
        self.assertEqual(
            generate_menu.extract_vasant_roti_grain_option(
                "रागी (Finger Millet) (केवल पुराना) की रोटी और परवल की सब्ज़ी",
                "vasant",
            ),
            "रागी (Finger Millet) (केवल पुराना)",
        )
        self.assertIsNone(
            generate_menu.extract_vasant_roti_grain_option("मूंग दाल और चावल", "vasant")
        )

    def test_apply_vasant_roti_grain_rotation_rule_filters_used_grain_options(self) -> None:
        filtered, reset = generate_menu.apply_vasant_roti_grain_rotation_rule(
            [
                "जौ (Barley) (केवल पुराना) की रोटी और लौकी की सब्ज़ी",
                "ज्वार (Sorghum) (केवल पुराना) की रोटी और लौकी की सब्ज़ी",
                "मूंग दाल और चावल",
            ],
            {"जौ (Barley) (केवल पुराना)"},
            "vasant",
        )

        self.assertEqual(
            filtered,
            [
                "ज्वार (Sorghum) (केवल पुराना) की रोटी और लौकी की सब्ज़ी",
                "मूंग दाल और चावल",
            ],
        )
        self.assertFalse(reset)

    def test_apply_vasant_roti_grain_rotation_rule_resets_after_all_present_options_are_used(self) -> None:
        filtered, reset = generate_menu.apply_vasant_roti_grain_rotation_rule(
            [
                "जौ (Barley) (केवल पुराना) की रोटी और लौकी की सब्ज़ी",
                "ज्वार (Sorghum) (केवल पुराना) की रोटी और भिंडी की सूखी सब्ज़ी",
            ],
            {
                "जौ (Barley) (केवल पुराना)",
                "ज्वार (Sorghum) (केवल पुराना)",
                "रागी (Finger Millet) (केवल पुराना)",
            },
            "vasant",
        )

        self.assertEqual(
            filtered,
            [
                "जौ (Barley) (केवल पुराना) की रोटी और लौकी की सब्ज़ी",
                "ज्वार (Sorghum) (केवल पुराना) की रोटी और भिंडी की सूखी सब्ज़ी",
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
                "meal": "ज्वार (Sorghum) (केवल पुराना) की रोटी और अरहर दाल",
                "ritu_key": "vasant",
            },
        ]

        self.assertEqual(
            generate_menu.get_vasant_roti_grain_cycle_used_options(history, date(2026, 3, 18), "vasant"),
            {"जौ (Barley) (केवल पुराना)", "ज्वार (Sorghum) (केवल पुराना)"},
        )


class VasantDalRotationTests(unittest.TestCase):
    def test_extract_vasant_dal_option_identifies_allowed_dals(self) -> None:
        self.assertEqual(
            generate_menu.extract_vasant_dal_option("मूंग दाल और चावल", "vasant"),
            "मूँग",
        )
        self.assertEqual(
            generate_menu.extract_vasant_dal_option(
                "चने और जौ (Barley) की रोटी (मिस्सी रोटी) और चने-लौकी की दाल",
                "vasant",
            ),
            "चने-लौकी की दाल",
        )
        self.assertIsNone(
            generate_menu.extract_vasant_dal_option(
                "जौ (Barley) (केवल पुराना) की रोटी और लौकी की सब्ज़ी",
                "vasant",
            )
        )

    def test_apply_vasant_dal_rotation_rule_filters_used_strict_dals_but_keeps_moong(self) -> None:
        filtered, reset = generate_menu.apply_vasant_dal_rotation_rule(
            [
                "मूंग दाल और चावल",
                "मसूर दाल और चावल",
                "अरहर दाल और चावल",
                "चने-लौकी की दाल और चावल",
            ],
            {"मसूर", "अरहर"},
            "vasant",
        )

        self.assertEqual(
            filtered,
            [
                "मूंग दाल और चावल",
                "चने-लौकी की दाल और चावल",
            ],
        )
        self.assertFalse(reset)

    def test_apply_vasant_dal_rotation_rule_resets_only_after_strict_dals_are_exhausted(self) -> None:
        filtered, reset = generate_menu.apply_vasant_dal_rotation_rule(
            [
                "मूंग दाल और चावल",
                "मसूर दाल और चावल",
                "अरहर दाल और चावल",
                "चने-लौकी की दाल और चावल",
            ],
            {"मसूर", "अरहर", "चने-लौकी की दाल"},
            "vasant",
        )

        self.assertEqual(
            filtered,
            [
                "मूंग दाल और चावल",
                "मसूर दाल और चावल",
                "अरहर दाल और चावल",
                "चने-लौकी की दाल और चावल",
            ],
        )
        self.assertTrue(reset)

    def test_get_vasant_dal_cycle_used_options_reads_only_strict_dals(self) -> None:
        history = [
            {
                "date": "2026-03-15",
                "meal": "मूंग दाल और चावल",
                "ritu_key": "vasant",
            },
            {
                "date": "2026-03-16",
                "meal": "मसूर दाल और चावल",
                "ritu_key": "vasant",
            },
            {
                "date": "2026-03-17",
                "meal": "काला चना और चावल",
                "ritu_key": "vasant",
            },
        ]

        self.assertEqual(
            generate_menu.get_vasant_dal_cycle_used_options(history, date(2026, 3, 18), "vasant"),
            {"मसूर", "चने-लौकी की दाल"},
        )

    def test_normalize_vasant_dal_meal_text_uses_requested_vasant_labels(self) -> None:
        self.assertEqual(
            generate_menu.normalize_vasant_dal_meal_text("धुली मूंग दाल खिचड़ी"),
            "मूँग दाल खिचड़ी",
        )
        self.assertEqual(
            generate_menu.normalize_vasant_dal_meal_text("काला चना और चावल"),
            "चने-लौकी की दाल और चावल",
        )
        self.assertEqual(
            generate_menu.normalize_vasant_dal_meal_text("जो की रोटी और लौकी का भरता"),
            "जो की रोटी और लौकी का भरता (भुनी लौकी + सरसों का तड़का)",
        )
        self.assertEqual(
            generate_menu.normalize_vasant_dal_meal_text("जो की रोटी और कद्दू की सब्ज़ी"),
            "जो की रोटी और कद्दू की सब्ज़ी (मीठा या खट्टा-मीठा, बिना ज्यादा गुड़/अमचूर)",
        )

    def test_extract_vasant_dal_option_recognizes_new_exact_moong_variants(self) -> None:
        self.assertEqual(generate_menu.extract_vasant_dal_option("मूँग दाल खिचड़ी", "vasant"), "मूँग")
        self.assertEqual(generate_menu.extract_vasant_dal_option("सादी मूँग दाल", "vasant"), "मूँग")


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


class WeeklyPazhayaSadamRuleTests(unittest.TestCase):
    def test_should_force_weekly_pazhaya_sadam_when_missing_in_last_six_days(self) -> None:
        history = [
            {
                "date": "2026-03-18",
                "breakfast": "उपमा",
                "meal": "दाल और चावल",
                "ritu_key": "vasant",
            },
            {
                "date": "2026-03-19",
                "breakfast": "पोहा",
                "meal": "लौकी",
                "ritu_key": "vasant",
            },
            {
                "date": "2026-03-20",
                "breakfast": "इडली",
                "meal": "भिंडी",
                "ritu_key": "vasant",
            },
            {
                "date": "2026-03-21",
                "breakfast": "डोसा",
                "meal": "कद्दू",
                "ritu_key": "grishm",
            },
            {
                "date": "2026-03-22",
                "breakfast": "चीला",
                "meal": "परवल",
                "ritu_key": "grishm",
            },
            {
                "date": "2026-03-23",
                "breakfast": "दलिया",
                "meal": "करेला",
                "ritu_key": "grishm",
            },
        ]

        self.assertTrue(generate_menu.should_force_weekly_pazhaya_sadam(history, date(2026, 3, 24), "vasant"))
        self.assertTrue(generate_menu.should_force_weekly_pazhaya_sadam(history, date(2026, 5, 20), "grishm"))

    def test_should_not_force_weekly_pazhaya_sadam_when_recently_used(self) -> None:
        history = [
            {
                "date": "2026-05-16",
                "breakfast": "पझैया सादम (Pazhaya Sadam): बचे हुए चावल लें या फिर 1 कटोरी कच्चे चावल अच्छी तरह धोकर सादा चावल पकाएँ।",
                "meal": "दाल और चावल",
                "ritu_key": "grishm",
            }
        ]

        self.assertFalse(generate_menu.should_force_weekly_pazhaya_sadam(history, date(2026, 5, 20), "grishm"))

    def test_should_not_force_weekly_pazhaya_sadam_outside_vasant_grishm(self) -> None:
        self.assertFalse(generate_menu.should_force_weekly_pazhaya_sadam([], date(2026, 7, 1), "varsha"))

    def test_should_force_required_window_pazhaya_sadam_until_window_has_one(self) -> None:
        history = [
            {
                "date": "2026-04-07",
                "breakfast": "उपमा",
                "meal": "दाल और चावल",
                "ritu_key": "vasant",
            },
            {
                "date": "2026-04-08",
                "breakfast": "पोहा",
                "meal": "लौकी",
                "ritu_key": "vasant",
            },
        ]

        self.assertTrue(generate_menu.should_force_required_window_pazhaya_sadam(history, date(2026, 4, 9)))

    def test_should_not_force_required_window_pazhaya_sadam_after_window_use(self) -> None:
        history = [
            {
                "date": "2026-04-09",
                "breakfast": "पझैया सादम (Pazhaya Sadam): बचे हुए चावल लें या फिर 1 कटोरी कच्चे चावल अच्छी तरह धोकर सादा चावल पकाएँ।",
                "meal": "दाल और चावल",
                "ritu_key": "vasant",
            }
        ]

        self.assertFalse(generate_menu.should_force_required_window_pazhaya_sadam(history, date(2026, 4, 10)))

    def test_should_not_force_required_window_pazhaya_sadam_outside_window(self) -> None:
        self.assertFalse(generate_menu.should_force_required_window_pazhaya_sadam([], date(2026, 4, 7)))
        self.assertFalse(generate_menu.should_force_required_window_pazhaya_sadam([], date(2026, 4, 13)))


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
