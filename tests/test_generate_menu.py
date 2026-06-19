import unittest
from unittest.mock import patch
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
    def test_guest_menu_file_validates_and_contains_daal_baati(self) -> None:
        guest_menu = generate_menu.validate_guest_menu_entries(
            generate_menu.load_json(generate_menu.GUEST_MENU_FILE),
            "guest_menu.json",
        )
        daal_baati = next(entry for entry in guest_menu if entry["id"] == "daal_baati")

        self.assertEqual(daal_baati["dish_hi"], "दाल बाटी")
        self.assertIn("अनिल जी बाटी और सत्तू मसाला अच्छा बनाते हैं।", daal_baati["responsibilities_hi"])
        self.assertIn("ऊषा इसे बेक करती हैं।", daal_baati["responsibilities_hi"])
        self.assertIn("शोभ्रन आटा गूंथते हैं।", daal_baati["responsibilities_hi"])

    def test_validate_guest_menu_entries_rejects_duplicate_ids(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate id"):
            generate_menu.validate_guest_menu_entries(
                [
                    {
                        "id": "daal_baati",
                        "dish_hi": "दाल बाटी",
                        "responsibilities_hi": ["अनिल जी बाटी बनाते हैं।"],
                        "specific_instructions_hi": [],
                    },
                    {
                        "id": "daal_baati",
                        "dish_hi": "दाल बाटी",
                        "responsibilities_hi": ["ऊषा इसे बेक करती हैं।"],
                        "specific_instructions_hi": [],
                    },
                ]
            )

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

    def test_bootstrap_published_archive_entries_filters_future_rows(self) -> None:
        history = [
            {
                "date": "2026-04-20",
                "breakfast": "उपमा",
                "meal": "दाल और चावल",
                "ritu_key": "vasant",
            },
            {
                "date": "2030-01-01",
                "breakfast": "पोहा",
                "meal": "लौकी",
                "ritu_key": "vasant",
            },
        ]

        entries = generate_menu.bootstrap_published_archive_entries(history, "2026-04-23")

        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["date"], "2026-04-20")
        self.assertEqual(entries[0]["archive_source"], "history_backfill")

    def test_upsert_published_archive_entry_replaces_same_date(self) -> None:
        original = [
            {
                "date": "2026-04-22",
                "archive_source": "history_backfill",
                "meal": "दाल और चावल",
            }
        ]
        replacement = {
            "date": "2026-04-22",
            "archive_source": "publish_run",
            "meal": "छाछ की सब्ज़ी चावल के साथ",
            "output_text": "*22-Apr-2026 तिथि के लिए भोजन:*",
        }

        updated = generate_menu.upsert_published_archive_entry(original, replacement)

        self.assertEqual(len(updated), 1)
        self.assertEqual(updated[0]["archive_source"], "publish_run")
        self.assertEqual(updated[0]["meal"], "छाछ की सब्ज़ी चावल के साथ")

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

    def test_choose_item_does_not_reallow_blocked_items_on_ekadashi_fallback(self) -> None:
        selected = generate_menu.choose_item(
            items=["चावल", "पोहा"],
            ekadashi=generate_menu.EkadashiInfo(True, "अपरा एकादशी", "ज्येष्ठ"),
            cycle_block_set=set(),
            recent_block_set=set(),
            consecutive_day_block_families=set(),
            recent_family_block_families=set(),
            family_extractor=generate_menu.extract_breakfast_repeat_families,
            keywords=["चावल", "पोहा"],
            disallowed_keywords=[],
            fallback_policy="fallback_full_menu",
            seed_key="2026-05-13:breakfast",
            weather_rules=None,
            weather_tags={},
            warn_bucket=set(),
            constraint_notes=[],
            prefer_lighter=False,
            light_fallback_items=["उपमा"],
        )

        self.assertEqual(selected, "उपमा")

    def test_is_blocked_by_ekadashi_rule_only_blocks_on_ekadashi(self) -> None:
        rice_item = "पझैया सादम: चावल"

        self.assertTrue(
            generate_menu.is_blocked_by_ekadashi_rule(
                rice_item,
                generate_menu.EkadashiInfo(True, "अपरा एकादशी", "ज्येष्ठ"),
                ["चावल"],
            )
        )
        self.assertFalse(
            generate_menu.is_blocked_by_ekadashi_rule(
                rice_item,
                generate_menu.EkadashiInfo(False, None, None),
                ["चावल"],
            )
        )

    def test_is_blocked_item_matches_common_poha_variant(self) -> None:
        self.assertTrue(generate_menu.is_blocked_item("पोहे", ["पोहा"]))

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

    def test_is_moong_dal_chilla_item_matches_variant_spellings(self) -> None:
        self.assertTrue(generate_menu.is_moong_dal_chilla_item("मूंग दाल चिल्ला"))
        self.assertTrue(generate_menu.is_moong_dal_chilla_item("मूँग की दाल का चीला"))
        self.assertFalse(generate_menu.is_moong_dal_chilla_item("बेसन का चीला"))
        self.assertFalse(generate_menu.is_moong_dal_chilla_item("मूंग दाल की खिचड़ी"))

    def test_apply_moong_dal_chilla_repeat_rule_blocks_within_fourteen_day_window(self) -> None:
        history = [
            {
                "date": "2026-03-01",
                "breakfast": "मूंग दाल चिल्ला",
                "meal": "दाल और चावल",
                "ritu_key": "vasant",
            }
        ]
        pool = ["मूंग दाल चिल्ला", "बेसन का चीला", "उपमा"]

        filtered, applied = generate_menu.apply_moong_dal_chilla_repeat_rule(
            pool,
            history,
            date(2026, 3, 14),
        )

        self.assertEqual(filtered, ["बेसन का चीला", "उपमा"])
        self.assertTrue(applied)

    def test_apply_moong_dal_chilla_repeat_rule_allows_after_fourteen_days(self) -> None:
        history = [
            {
                "date": "2026-03-01",
                "breakfast": "मूंग दाल चिल्ला",
                "meal": "दाल और चावल",
                "ritu_key": "vasant",
            }
        ]
        pool = ["मूंग दाल चिल्ला", "उपमा"]

        filtered, applied = generate_menu.apply_moong_dal_chilla_repeat_rule(
            pool,
            history,
            date(2026, 3, 15),
        )

        self.assertEqual(filtered, pool)
        self.assertFalse(applied)

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

    def test_select_monthly_fruit_blocks_kharbuja_on_rainy_day(self) -> None:
        rainy_weather = generate_menu.WeatherInfo(
            morning_temp_c=26,
            max_temp_c=33,
            rain_probability_pct=80,
            is_rainy=True,
            is_extreme_cold=False,
            is_extreme_hot=False,
            source_hi="test",
        )
        selection = generate_menu.select_monthly_fruit(
            [],
            date(2026, 6, 10),
            {6: ["खरबूजा", "केला"]},
            {},
            weather_info=rainy_weather,
        )

        self.assertTrue(selection.available)
        self.assertEqual(selection.fruit, "केला")

    def test_select_monthly_fruit_never_falls_back_to_kharbuja_when_rainy(self) -> None:
        rainy_weather = generate_menu.WeatherInfo(
            morning_temp_c=26,
            max_temp_c=33,
            rain_probability_pct=80,
            is_rainy=True,
            is_extreme_cold=False,
            is_extreme_hot=False,
            source_hi="test",
        )
        selection = generate_menu.select_monthly_fruit(
            [],
            date(2026, 6, 10),
            {6: ["खरबूजा"]},
            {},
            weather_info=rainy_weather,
        )

        self.assertFalse(selection.available)
        self.assertIsNone(selection.fruit)

    def test_select_monthly_fruit_deprioritizes_kharbuja_when_alternatives_exist(self) -> None:
        selection = generate_menu.select_monthly_fruit(
            [],
            date(2026, 6, 10),
            {6: ["खरबूजा", "तरबूज", "केला"]},
            {},
        )

        self.assertTrue(selection.available)
        self.assertNotEqual(selection.fruit, "खरबूजा")

    def test_select_monthly_fruit_blocks_recent_kharbuja_repeat_when_possible(self) -> None:
        history = [
            {"date": "2026-06-01", "fruit": "खरबूजा"},
            {"date": "2026-06-02", "fruit": "तरबूज"},
        ]
        selection = generate_menu.select_monthly_fruit(
            history,
            date(2026, 6, 10),
            {6: ["खरबूजा", "तरबूज"]},
            {},
        )

        self.assertTrue(selection.available)
        self.assertEqual(selection.fruit, "तरबूज")

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

    def test_select_drink_of_the_day_is_deterministic_for_date(self) -> None:
        first = generate_menu.select_drink_of_the_day(date(2026, 5, 24))
        second = generate_menu.select_drink_of_the_day(date(2026, 5, 24))
        self.assertEqual(first, second)

    def test_format_drink_of_the_day_line_includes_hindi_heading_and_fallback(self) -> None:
        line = generate_menu.format_drink_of_the_day_line(date(2026, 5, 24))
        self.assertTrue(line.startswith("*आज का पेय:* "))
        self.assertIn("*पेय विकल्प:* यदि आज का पेय उपलब्ध न हो, तो सत्तू का शर्बत दिया जा सकता है।", line)

    def test_format_drink_of_the_day_line_includes_recipe_when_available(self) -> None:
        target_date = next(
            date(2026, 5, day)
            for day in range(1, 32)
            if generate_menu.select_drink_of_the_day(date(2026, 5, day))[1] is not None
        )
        drink_name, recipe = generate_menu.select_drink_of_the_day(target_date)
        line = generate_menu.format_drink_of_the_day_line(target_date)

        self.assertIn(f"*आज का पेय:* {drink_name} — ", line)
        self.assertIn(recipe, line)

    def test_format_meal_display_adds_dal_with_parwal_bhujiya(self) -> None:
        meal = "ज्वार (Sorghum) (केवल पुराना) की रोटी और परवल की भुजिया"
        self.assertEqual(
            generate_menu.format_meal_display(meal),
            "ज्वार (Sorghum) (केवल पुराना) की रोटी और परवल की भुजिया, साथ में सादी मूंग दाल",
        )

    def test_format_meal_display_does_not_duplicate_existing_dal(self) -> None:
        meal = "ज्वार की रोटी और परवल की भुजिया, साथ में मूंग दाल"
        self.assertEqual(generate_menu.format_meal_display(meal), meal)

    def test_format_meal_display_leaves_other_parwal_meals_unchanged(self) -> None:
        meal = "ज्वार की रोटी और परवल-मूँगदाल की सूखी सब्ज़ी"
        self.assertEqual(generate_menu.format_meal_display(meal), meal)

    def test_build_navishti_grishm_plan_line_ignores_household_replacements(self) -> None:
        line = generate_menu.build_navishti_grishm_plan_line(
            date(2026, 5, 24),
            {2: "छाछ की सब्ज़ी और शालि चावल"},
        )

        self.assertEqual(
            line.splitlines(),
            [
                "*नविष्टि भोजन (ग्रीष्म):*",
                "भोजन 1: दूध + गेहूँ दलिया",
                "भोजन 2: लौकी राइस मैश (थोड़े घी के साथ)",
                "भोजन 3: पपीता",
                "भोजन 4: मक्खन + मुलायम चावल",
                "भोजन 5: सूजी का हलवा",
            ],
        )
        self.assertNotIn("ऊपर लिखित अलग कटोरी", line)
        self.assertNotIn("तड़का", line)
        self.assertNotIn("सभी के लिए बन रहे", line)

    def test_resolve_navishti_grishm_plan_items_avoids_previous_day_duplicate(self) -> None:
        previous_items = list(generate_menu.NAVISHTI_GRISHM_WEEKLY_PLAN[6])
        items = generate_menu.resolve_navishti_grishm_plan_items(
            date(2026, 5, 25),
            {},
            previous_items,
        )

        previous_keys = {generate_menu.normalize_navishti_food_key(item) for item in previous_items}
        repeated_key = generate_menu.normalize_navishti_food_key("मक्खन + मुलायम चावल")
        self.assertIn(repeated_key, previous_keys)
        self.assertEqual(items[3], "मक्खन + रोटी")
        self.assertTrue(
            all(generate_menu.normalize_navishti_food_key(item) not in previous_keys for item in items)
        )

    def test_normalize_navishti_food_key_strips_old_shared_note(self) -> None:
        old_shared_slot = (
            "छाछ की सब्ज़ी और शालि चावल "
            "(सभी के लिए बन रहे इसी भोजन से तड़का लगाने से पहले निकालें)"
        )

        self.assertEqual(
            generate_menu.normalize_navishti_food_key(old_shared_slot),
            generate_menu.normalize_navishti_food_key("छाछ की सब्ज़ी और शालि चावल"),
        )

    def test_resolve_navishti_grishm_plan_items_ignores_repeated_shared_replacement(self) -> None:
        old_shared_slot = (
            "छाछ की सब्ज़ी और शालि चावल "
            "(सभी के लिए बन रहे इसी भोजन से तड़का लगाने से पहले निकालें)"
        )
        items = generate_menu.resolve_navishti_grishm_plan_items(
            date(2026, 5, 25),
            {2: "छाछ की सब्ज़ी और शालि चावल"},
            [old_shared_slot],
        )

        self.assertEqual(items[1], "मूंग दाल खिचड़ी + लौकी")
        self.assertTrue(all("सभी के लिए बन रहे" not in item for item in items))

    def test_update_history_persists_navishti_grishm_plan(self) -> None:
        updated = generate_menu.update_history(
            history=[],
            target_date="2026-05-25",
            breakfast_item="ज्वार की रोटी (आलू-प्याज़ भरी)",
            meal_item="छाछ की सब्ज़ी और शालि चावल",
            second_meal_item=None,
            fruit_item="लीची",
            keep_days=7,
            ritu_key="grishm",
            navishti_grishm_plan=["दूध + जौ दलिया", "मक्खन + रोटी"],
        )

        self.assertEqual(updated[0]["navishti_grishm_plan"], ["दूध + जौ दलिया", "मक्खन + रोटी"])

    def test_build_next_day_overnight_prep_line_does_not_create_second_breakfast_heading(self) -> None:
        line = generate_menu.build_next_day_overnight_prep_line(
            "पखाला भात (Pakhala Bhata): रात में 1 कटोरी कच्चे चावल धोकर सादा चावल पकाएँ।"
        )
        self.assertTrue(line.startswith("*रात की चावल तैयारी:*"))
        self.assertNotIn("पखाला भात", line)
        self.assertNotIn("Pakhala Bhata", line)
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
        meals = ["छाछ की सब्ज़ी चावल के साथ", "छाछ की सब्ज़ी और शालि चावल", "मूंग दाल और चावल"]
        filtered = generate_menu.exclude_meals_incompatible_with_breakfast(
            "पझैया सादम (Pazhaya Sadam): बचे हुए चावल लें या फिर 1 कटोरी कच्चे चावल अच्छी तरह धोकर सादा चावल पकाएँ।",
            meals,
        )
        self.assertEqual(filtered, ["मूंग दाल और चावल"])

    def test_exclude_meals_incompatible_with_pakhala_bhata(self) -> None:
        meals = ["छाछ की सब्ज़ी चावल के साथ", "छाछ की सब्ज़ी और साठी चावल", "मूंग दाल और चावल"]
        filtered = generate_menu.exclude_meals_incompatible_with_breakfast(
            "पखाला भात (Pakhala Bhata): रात में 1 कटोरी कच्चे चावल धोकर सादा चावल पकाएँ।",
            meals,
        )
        self.assertEqual(filtered, ["मूंग दाल और चावल"])

    def test_exclude_chaach_sabzi_meals_for_overnight_support(self) -> None:
        meals = ["छाछ की सब्ज़ी चावल के साथ", "छाछ की सब्ज़ी और साठी चावल", "मूंग दाल और चावल"]
        filtered = generate_menu.exclude_chaach_sabzi_meals_for_overnight_support(meals)

        self.assertEqual(filtered, ["मूंग दाल और चावल"])

    def test_exclude_chaach_sabzi_meals_for_overnight_support_can_empty_pool(self) -> None:
        meals = ["छाछ की सब्ज़ी चावल के साथ", "छाछ की सब्ज़ी और शालि चावल"]
        filtered = generate_menu.exclude_chaach_sabzi_meals_for_overnight_support(meals)

        self.assertEqual(filtered, [])

    def test_chaach_sabzi_meal_detection_matches_all_rice_variants(self) -> None:
        self.assertTrue(generate_menu.is_chaach_sabzi_meal("छाछ की सब्ज़ी चावल के साथ"))
        self.assertTrue(generate_menu.is_chaach_sabzi_meal("छाछ की सब्ज़ी और शालि चावल"))
        self.assertTrue(generate_menu.is_chaach_sabzi_meal("छाछ की सब्ज़ी और साठी चावल"))
        self.assertFalse(generate_menu.is_chaach_sabzi_meal("मसाला छाछ"))


class DateResolutionTests(unittest.TestCase):
    def test_resolve_date_rejects_explicit_date_override(self) -> None:
        with self.assertRaisesRegex(ValueError, "always generated for tomorrow's date"):
            generate_menu.resolve_date("2026-04-06", "Asia/Kolkata")

    def test_resolve_date_without_explicit_date_defaults_to_tomorrow(self) -> None:
        self.assertEqual(
            generate_menu.resolve_date(None, "Asia/Kolkata", now_date=date(2026, 4, 6)),
            date(2026, 4, 7),
        )

    def test_resolve_runtime_today_uses_env_override_for_tomorrow_flow(self) -> None:
        with patch.dict("os.environ", {generate_menu.MENU_GENERATOR_NOW_DATE_ENV: "2026-04-11"}, clear=False):
            self.assertEqual(generate_menu.resolve_runtime_today("Asia/Kolkata"), date(2026, 4, 11))


class OutputFreshnessTests(unittest.TestCase):
    def test_parse_output_target_date_reads_header_date(self) -> None:
        output_text = "*22-Apr-2026 तिथि के लिए भोजन:*\r\n*ऋतु:* वसंत"
        self.assertEqual(generate_menu.parse_output_target_date(output_text), date(2026, 4, 22))

    def test_parse_output_target_date_rejects_missing_header(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing or malformed"):
            generate_menu.parse_output_target_date("*ऋतु:* वसंत")

    def test_verify_output_target_date_rejects_stale_output(self) -> None:
        with self.assertRaisesRegex(ValueError, "expected 2026-04-22, found 2026-04-21"):
            generate_menu.verify_output_target_date(
                "*21-Apr-2026 तिथि के लिए भोजन:*\r\n*ऋतु:* वसंत",
                date(2026, 4, 22),
            )

    def test_parse_navishti_output_target_date_reads_header_date(self) -> None:
        output_text = "*22-Apr-2026 तिथि के लिए नविष्टि भोजन:*\r\n*ऋतु:* ग्रीष्म"
        self.assertEqual(generate_menu.parse_navishti_output_target_date(output_text), date(2026, 4, 22))

    def test_format_navishti_daily_menu_text_renders_standalone_grishm_message(self) -> None:
        text = generate_menu.format_navishti_daily_menu_text(
            date(2026, 6, 20),
            "grishm",
            ["दूध + जौ दलिया", "मूंग दाल खिचड़ी + लौकी"],
        )

        self.assertEqual(
            text.splitlines(),
            [
                "*20-Jun-2026 तिथि के लिए नविष्टि भोजन:*",
                "*ऋतु:* ग्रीष्म",
                "*भोजन 1:* दूध + जौ दलिया",
                "*भोजन 2:* मूंग दाल खिचड़ी + लौकी",
            ],
        )
        self.assertNotIn("आज का भोजन", text)
        self.assertNotIn("सभी के लिए बन रहे", text)
        self.assertNotIn("तड़का", text)

    def test_format_navishti_daily_menu_text_handles_non_grishm(self) -> None:
        text = generate_menu.format_navishti_daily_menu_text(date(2026, 8, 1), "varsha", [])

        self.assertEqual(
            text.splitlines(),
            [
                "*01-Aug-2026 तिथि के लिए नविष्टि भोजन:*",
                "*आज का निर्देश:* नविष्टि के लिए अलग ग्रीष्म भोजन आज लागू नहीं है।",
            ],
        )


class PanchangEkadashiDisplayTests(unittest.TestCase):
    def test_resolve_panchang_info_does_not_show_ekadashi_before_observance_date(self) -> None:
        info = generate_menu.resolve_panchang_info(
            date(2026, 5, 12),
            generate_menu.EkadashiInfo(False, None, None),
            {
                "ritu_hi": "वसंत",
                "maah_hi": "वैशाख",
                "tithi_hi": "एकादशी",
                "paksha_hi": "कृष्ण पक्ष",
            },
            "वसंत",
            "amanta",
        )

        self.assertEqual(info.tithi_hi, "दशमी")

    def test_resolve_panchang_info_uses_ekadashi_calendar_for_observance_date(self) -> None:
        info = generate_menu.resolve_panchang_info(
            date(2026, 5, 13),
            generate_menu.EkadashiInfo(True, "अपरा एकादशी", "ज्येष्ठ"),
            {
                "ritu_hi": "वसंत",
                "maah_hi": "वैशाख",
                "tithi_hi": "द्वादशी",
                "paksha_hi": "कृष्ण पक्ष",
            },
            "वसंत",
            "amanta",
        )

        self.assertEqual(info.tithi_hi, "एकादशी")


class WeatherTagWarningTests(unittest.TestCase):
    def test_rainy_weather_still_drives_internal_weather_rules(self) -> None:
        weather = generate_menu.WeatherInfo(
            morning_temp_c=24,
            max_temp_c=36,
            rain_probability_pct=90,
            is_rainy=True,
            is_extreme_cold=False,
            is_extreme_hot=True,
            source_hi="test",
        )
        rules = generate_menu.derive_weather_rules(
            weather,
            {
                "extreme_cold_max_c": 10,
                "cold_max_c": 18,
                "hot_min_c": 30,
                "extreme_hot_min_c": 35,
                "rain_probability_high_pct": 50,
            },
        )

        self.assertIn("rain_friendly", rules.preferred_tags)
        self.assertIn("cold_served", rules.avoid_tags)

    def test_infer_tags_for_common_breakfast_items_avoids_empty_results(self) -> None:
        self.assertEqual(
            set(generate_menu.infer_tags_for_item("पोहे")),
            {"light", "summer_friendly"},
        )
        self.assertEqual(
            set(generate_menu.infer_tags_for_item("बेसन का चीला")),
            {"comfort_hot", "light", "rain_friendly"},
        )
        self.assertEqual(
            set(generate_menu.infer_tags_for_item("उबले हुए मंगोड़े")),
            {"comfort_hot", "light"},
        )

    def test_format_weather_tag_warning_explains_cause_and_fix(self) -> None:
        warning = generate_menu.format_weather_tag_warning({"पोहे", "बेसन का चीला"})

        self.assertIn("missing or empty", warning)
        self.assertIn("neutral fallback", warning)
        self.assertIn("menu_weather_tags.json", warning)
        self.assertIn("पोहे", warning)
        self.assertIn("बेसन का चीला", warning)

    def test_is_double_meal_window_matches_requested_april_range(self) -> None:
        self.assertTrue(generate_menu.is_double_meal_window(date(2026, 4, 8)))
        self.assertTrue(generate_menu.is_double_meal_window(date(2026, 4, 14)))
        self.assertFalse(generate_menu.is_double_meal_window(date(2026, 4, 7)))
        self.assertFalse(generate_menu.is_double_meal_window(date(2026, 4, 15)))

    def test_select_second_meal_for_window_avoids_second_rice_meal(self) -> None:
        second = generate_menu.select_second_meal_for_window(
            selected_meal="मूंग दाल और चावल",
            meal_choice_items=["मूंग दाल और चावल", "जौ (Barley) (केवल पुराना) की रोटी और लौकी की सब्ज़ी"],
            ritu_key="vasant",
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

    def test_extract_grishm_roti_grain_option_identifies_allowed_option(self) -> None:
        self.assertEqual(
            generate_menu.extract_grishm_roti_grain_option(
                "ज्वार की रोटी, लौकी की सब्ज़ी, मसूर दाल",
                "grishm",
            ),
            "ज्वार",
        )
        self.assertEqual(
            generate_menu.extract_grishm_roti_grain_option(
                "रागी की रोटी (मूंग दाल भरवां)",
                "grishm",
            ),
            "रागी",
        )
        self.assertIsNone(
            generate_menu.extract_grishm_roti_grain_option(
                "पुराना गेहूं की रोटी, लौकी और चना दाल",
                "grishm",
            )
        )

    def test_get_ritu_roti_grain_preference_weight_blocks_grishm_wheat(self) -> None:
        self.assertEqual(
            generate_menu.get_ritu_roti_grain_preference_weight(
                "गेहूँ (Wheat) (केवल पुराना) की रोटी और लौकी की सब्ज़ी",
                "vasant",
            ),
            5,
        )
        self.assertEqual(
            generate_menu.get_ritu_roti_grain_preference_weight(
                "जौ (Barley) (केवल पुराना) की रोटी और लौकी की सब्ज़ी",
                "vasant",
            ),
            30,
        )
        self.assertEqual(
            generate_menu.get_ritu_roti_grain_preference_weight(
                "पुराना गेहूं की रोटी, लौकी और चना दाल",
                "grishm",
            ),
            1,
        )
        self.assertTrue(generate_menu.is_grishm_forbidden_wheat_roti("पुराना गेहूं की रोटी, लौकी और चना दाल"))
        self.assertEqual(
            generate_menu.get_ritu_roti_grain_preference_weight(
                "ज्वार की रोटी, लौकी की सब्ज़ी, मसूर दाल",
                "grishm",
            ),
            35,
        )

    def test_choose_weighted_meal_item_prefers_higher_weight_grain_over_many_seeds(self) -> None:
        pool = [
            "ज्वार की रोटी, लौकी की सब्ज़ी, मसूर दाल",
            "झंगोरा आटा की रोटी, लौकी और चना दाल",
        ]
        counts = {item: 0 for item in pool}
        for index in range(200):
            selected = generate_menu.choose_weighted_meal_item(pool, f"seed-{index}", "grishm")
            counts[selected] += 1
        self.assertGreater(counts["ज्वार की रोटी, लौकी की सब्ज़ी, मसूर दाल"], counts["झंगोरा आटा की रोटी, लौकी और चना दाल"])

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

    def test_is_vasant_ragi_roti_only_window_matches_requested_dates(self) -> None:
        self.assertTrue(generate_menu.is_vasant_ragi_roti_only_window(date(2026, 4, 30), "vasant"))
        self.assertTrue(generate_menu.is_vasant_ragi_roti_only_window(date(2026, 5, 5), "vasant"))
        self.assertFalse(generate_menu.is_vasant_ragi_roti_only_window(date(2026, 4, 29), "vasant"))
        self.assertFalse(generate_menu.is_vasant_ragi_roti_only_window(date(2026, 5, 6), "vasant"))
        self.assertFalse(generate_menu.is_vasant_ragi_roti_only_window(date(2026, 5, 1), "grishm"))

    def test_apply_vasant_ragi_roti_only_window_rule_keeps_only_ragi_among_roti_items(self) -> None:
        filtered, applied = generate_menu.apply_vasant_ragi_roti_only_window_rule(
            [
                "जौ (Barley) (केवल पुराना) की रोटी और लौकी की सब्ज़ी",
                "रागी (Finger Millet) (केवल पुराना) की रोटी और परवल की सब्ज़ी",
                "मूंग दाल और चावल",
            ],
            date(2026, 5, 1),
            "vasant",
        )

        self.assertEqual(
            filtered,
            [
                "रागी (Finger Millet) (केवल पुराना) की रोटी और परवल की सब्ज़ी",
                "मूंग दाल और चावल",
            ],
        )
        self.assertTrue(applied)

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


class PakhalaServingNoteTests(unittest.TestCase):
    def test_build_pakhala_serving_note_returns_hindi_onion_note(self) -> None:
        self.assertEqual(
            generate_menu.build_pakhala_serving_note("पखाला भात (Pakhala Bhata): रात में 1 कटोरी कच्चे चावल धोकर सादा चावल पकाएँ।"),
            "*साथ में:* मोटा चौकोर कटा प्याज",
        )

    def test_build_pakhala_serving_note_skips_non_pakhala_breakfast(self) -> None:
        self.assertIsNone(generate_menu.build_pakhala_serving_note("पझैया सादम (Pazhaya Sadam)"))


class CurdRuleTests(unittest.TestCase):
    def test_build_curd_raita_note_returns_short_hindi_note_for_vasant(self) -> None:
        self.assertEqual(
            generate_menu.build_curd_raita_note(
                "vasant",
                "मूंग दाल दहीवाले फरे (भाप में पकाएं)",
                "मूंग दाल और चावल",
                None,
            ),
            "*दही रूप:* केवल लौकी/खीरे का रायता",
        )

    def test_build_curd_raita_note_skips_when_specific_raita_is_already_named(self) -> None:
        self.assertIsNone(
            generate_menu.build_curd_raita_note(
                "grishm",
                "उपमा",
                "ज्वार की रोटी, लौकी की सब्ज़ी, मसूर दाल, लौकी का रायता",
                None,
            )
        )

    def test_build_curd_raita_note_still_applies_when_other_curd_item_is_unspecified(self) -> None:
        self.assertEqual(
            generate_menu.build_curd_raita_note(
                "grishm",
                "मूंग दाल दहीवाले फरे (भाप में पकाएं)",
                "ज्वार की रोटी, लौकी की सब्ज़ी, मसूर दाल, लौकी का रायता",
                None,
            ),
            "*दही रूप:* केवल लौकी/खीरे का रायता",
        )

    def test_build_curd_raita_note_skips_pakhala_breakfast(self) -> None:
        self.assertIsNone(
            generate_menu.build_curd_raita_note(
                "grishm",
                "पखाला भात (Pakhala Bhata): रात में 1 कटोरी कच्चे चावल धोकर सादा चावल पकाएँ। 2 बड़े चम्मच दही डालें।",
                "मूंग दाल और चावल",
                None,
            )
        )

    def test_build_curd_raita_note_skips_pazhaya_sadam_breakfast(self) -> None:
        self.assertIsNone(
            generate_menu.build_curd_raita_note(
                "grishm",
                "पझैया सादम (Pazhaya Sadam): अब 2–3 बड़े चम्मच दही या लगभग ½ कटोरी पतली छाछ मिलाएँ।",
                "मूंग दाल और चावल",
                None,
            )
        )

    def test_build_curd_raita_note_skips_special_dahi_chawal_meal(self) -> None:
        self.assertIsNone(
            generate_menu.build_curd_raita_note(
                "grishm",
                "रागी चीला",
                "दही चावल ज्यादा करी पत्ता व सौंफ के साथ",
                None,
            )
        )

    def test_build_curd_raita_note_skips_non_vasant_grishm_ritu(self) -> None:
        self.assertIsNone(
            generate_menu.build_curd_raita_note(
                "shishir",
                "उपमा",
                "धुली उड़द दाल - गेहूँ रोटी और चकुंदर का रायता",
                None,
            )
        )

    def test_get_yearly_used_curd_items_ignores_hemant_shishir_entries(self) -> None:
        archive = [
            {
                "date": "2026-01-15",
                "breakfast": "उपमा",
                "meal": "धुली उड़द दाल - गेहूँ रोटी और चकुंदर का रायता",
                "ritu_key": "shishir",
            },
            {
                "date": "2026-05-15",
                "breakfast": "मूंग दाल दहीवाले फरे (भाप में पकाएं)",
                "meal": "मूंग दाल और चावल",
                "ritu_key": "vasant",
            },
        ]
        used = generate_menu.get_yearly_used_curd_items(archive, date(2026, 6, 1))
        self.assertIn(
            generate_menu.normalize_item_key("मूंग दाल दहीवाले फरे (भाप में पकाएं)"),
            used,
        )
        self.assertNotIn(
            generate_menu.normalize_item_key("धुली उड़द दाल - गेहूँ रोटी और चकुंदर का रायता"),
            used,
        )

    def test_apply_yearly_curd_repeat_rule_blocks_repeated_curd_item_outside_winter(self) -> None:
        filtered, applied = generate_menu.apply_yearly_curd_repeat_rule(
            [
                "मूंग दाल दहीवाले फरे (भाप में पकाएं)",
                "उपमा",
            ],
            {generate_menu.normalize_item_key("मूंग दाल दहीवाले फरे (भाप में पकाएं)")},
            "vasant",
        )
        self.assertEqual(filtered, ["उपमा"])
        self.assertTrue(applied)

    def test_apply_yearly_curd_repeat_rule_does_not_block_winter(self) -> None:
        filtered, applied = generate_menu.apply_yearly_curd_repeat_rule(
            ["धुली उड़द दाल - गेहूँ रोटी और चकुंदर का रायता"],
            {generate_menu.normalize_item_key("धुली उड़द दाल - गेहूँ रोटी और चकुंदर का रायता")},
            "shishir",
        )
        self.assertEqual(filtered, ["धुली उड़द दाल - गेहूँ रोटी और चकुंदर का रायता"])
        self.assertFalse(applied)


class DateSpecificRotiAttaRuleTests(unittest.TestCase):
    def test_apply_grishm_roti_atta_rule_removes_wheat_roti(self) -> None:
        filtered, applied = generate_menu.apply_grishm_roti_atta_rule(
            [
                "पुराना गेहूं की रोटी, लौकी और चना दाल",
                "कनक की रोटी, तोरई",
                "ज्वार की रोटी, लौकी की सब्ज़ी, मसूर दाल",
            ],
            date(2026, 5, 25),
            "grishm",
        )
        self.assertEqual(filtered, ["ज्वार की रोटी, लौकी की सब्ज़ी, मसूर दाल"])
        self.assertTrue(applied)

    def test_apply_grishm_roti_atta_rule_rewrites_generic_stuffed_roti(self) -> None:
        filtered, applied = generate_menu.apply_grishm_roti_atta_rule(
            [
                "दाल की रोटी (मूंग दाल)",
                "आलू प्याज़ की रोटी",
                "पोहा",
            ],
            date(2026, 5, 24),
            "grishm",
        )
        self.assertEqual(
            filtered,
            [
                "रागी की रोटी (मूंग दाल भरवां)",
                "रागी की रोटी (आलू-प्याज़ भरी)",
                "पोहा",
            ],
        )
        self.assertTrue(applied)
        self.assertEqual(generate_menu.extract_roti_atta_key(filtered[0]), "रागी")

    def test_apply_grishm_roti_atta_rule_uses_sprouted_ragi_display_when_scheduled(self) -> None:
        filtered, applied = generate_menu.apply_grishm_roti_atta_rule(
            ["दाल की रोटी (मूंग दाल)"],
            date(2026, 5, 12),
            "grishm",
        )
        self.assertEqual(filtered, ["स्प्राउटेड रागी की रोटी (मूंग दाल भरवां)"])
        self.assertTrue(applied)
        self.assertEqual(generate_menu.extract_roti_atta_key(filtered[0]), "रागी")

    def test_apply_grishm_roti_atta_rule_ignores_other_ritus(self) -> None:
        filtered, applied = generate_menu.apply_grishm_roti_atta_rule(
            ["गेहूँ (Wheat) (केवल पुराना) की रोटी और लौकी की सब्ज़ी"],
            date(2026, 5, 24),
            "vasant",
        )
        self.assertEqual(filtered, ["गेहूँ (Wheat) (केवल पुराना) की रोटी और लौकी की सब्ज़ी"])
        self.assertFalse(applied)

    def test_apply_date_specific_roti_atta_rule_removes_chana_sattu_in_exclusion_window(self) -> None:
        filtered, applied = generate_menu.apply_date_specific_roti_atta_rule(
            [
                "जो के सत्तू की रोटी",
                "चने के सत्तू की रोटी (बिना खटास के)",
                "उपमा",
            ],
            date(2026, 5, 9),
        )
        self.assertEqual(filtered, ["जो के सत्तू की रोटी", "उपमा"])
        self.assertTrue(applied)

    def test_apply_date_specific_roti_atta_rule_restricts_to_ragi_during_sprouted_window(self) -> None:
        filtered, applied = generate_menu.apply_date_specific_roti_atta_rule(
            [
                "ज्वार (Sorghum) (केवल पुराना) की रोटी और लौकी की सब्ज़ी",
                "रागी (Finger Millet) (केवल पुराना) की रोटी और लौकी की सब्ज़ी",
                "मूंग दाल और चावल",
            ],
            date(2026, 5, 12),
        )
        self.assertEqual(
            filtered,
            [
                "रागी (Finger Millet) (केवल पुराना) की रोटी और लौकी की सब्ज़ी",
                "मूंग दाल और चावल",
            ],
        )
        self.assertTrue(applied)

    def test_apply_date_specific_roti_atta_rule_uses_fallback_only_when_primary_missing(self) -> None:
        filtered, applied = generate_menu.apply_date_specific_roti_atta_rule(
            [
                "ज्वार की रोटी, लौकी की सब्ज़ी, मसूर दाल",
                "मूंग दाल और चावल",
            ],
            date(2026, 6, 19),
        )
        self.assertEqual(filtered, ["ज्वार की रोटी, लौकी की सब्ज़ी, मसूर दाल", "मूंग दाल और चावल"])
        self.assertFalse(applied)

    def test_apply_date_specific_roti_atta_rule_prefers_primary_when_available(self) -> None:
        filtered, applied = generate_menu.apply_date_specific_roti_atta_rule(
            [
                "जौ की रोटी, लौकी की सब्ज़ी, मूँग दाल धुली",
                "ज्वार की रोटी, लौकी की सब्ज़ी, मसूर दाल",
                "मूंग दाल और चावल",
            ],
            date(2026, 6, 19),
        )
        self.assertEqual(filtered, ["जौ की रोटी, लौकी की सब्ज़ी, मूँग दाल धुली", "मूंग दाल और चावल"])
        self.assertTrue(applied)

    def test_build_roti_atta_note_uses_scheduled_display_label(self) -> None:
        self.assertEqual(
            generate_menu.build_roti_atta_note(
                date(2026, 5, 12),
                "उपमा",
                "रागी (Finger Millet) (केवल पुराना) की रोटी और लौकी की सब्ज़ी",
                None,
            ),
            "*आज का आटा:* स्प्राउटेड रागी",
        )


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


class WeeklyChaachSabziRuleTests(unittest.TestCase):
    def test_is_chaach_sabzi_rice_item_requires_both_chaach_sabzi_and_rice(self) -> None:
        self.assertTrue(generate_menu.is_chaach_sabzi_rice_item("छाछ की सब्ज़ी चावल के साथ"))
        self.assertTrue(generate_menu.is_chaach_sabzi_rice_item("छाछ की सब्ज़ी और शालि चावल"))
        self.assertFalse(generate_menu.is_chaach_sabzi_rice_item("छाछ त्रिकटु के साथ"))

    def test_should_force_weekly_chaach_sabzi_when_missing_in_last_six_days(self) -> None:
        history = [
            {
                "date": "2026-04-10",
                "breakfast": "उपमा",
                "meal": "मूंग दाल और चावल",
                "ritu_key": "vasant",
            },
            {
                "date": "2026-04-11",
                "breakfast": "पोहा",
                "meal": "जौ (Barley) (केवल पुराना) की रोटी और लौकी की सब्ज़ी",
                "ritu_key": "vasant",
            },
            {
                "date": "2026-04-12",
                "breakfast": "इडली",
                "meal": "ज्वार (Sorghum) (केवल पुराना) की रोटी और परवल की सब्ज़ी",
                "ritu_key": "vasant",
            },
            {
                "date": "2026-04-13",
                "breakfast": "डोसा",
                "meal": "शालि चावल, मसूर दाल, लौकी की सब्ज़ी",
                "ritu_key": "grishm",
            },
            {
                "date": "2026-04-14",
                "breakfast": "दलिया",
                "meal": "पुराना गेहूं की रोटी, तोरई, अरहर दाल",
                "ritu_key": "grishm",
            },
            {
                "date": "2026-04-15",
                "breakfast": "चीला",
                "meal": "जौ की रोटी, लौकी की सब्ज़ी, मूँग दाल धुली",
                "ritu_key": "grishm",
            },
        ]

        self.assertTrue(generate_menu.should_force_weekly_chaach_sabzi(history, date(2026, 4, 16), "vasant"))
        self.assertTrue(generate_menu.should_force_weekly_chaach_sabzi(history, date(2026, 5, 20), "grishm"))

    def test_should_not_force_weekly_chaach_sabzi_when_recently_used_as_main_or_second_meal(self) -> None:
        history = [
            {
                "date": "2026-04-15",
                "breakfast": "उपमा",
                "meal": "मूंग दाल और चावल",
                "second_meal": "छाछ की सब्ज़ी चावल के साथ",
                "ritu_key": "vasant",
            }
        ]

        self.assertFalse(generate_menu.should_force_weekly_chaach_sabzi(history, date(2026, 4, 20), "vasant"))

    def test_should_not_force_weekly_chaach_sabzi_outside_vasant_grishm(self) -> None:
        self.assertFalse(generate_menu.should_force_weekly_chaach_sabzi([], date(2026, 7, 1), "varsha"))


class FortnightlyKadhiChawalRuleTests(unittest.TestCase):
    def test_is_kadhi_item_matches_named_and_majjida_variants(self) -> None:
        self.assertTrue(generate_menu.is_kadhi_item("कढ़ी और बासमती चावल"))
        self.assertTrue(generate_menu.is_kadhi_item("चावल और मजीदा कढ़ी"))
        self.assertTrue(generate_menu.is_kadhi_item("Majjida Karhi"))
        self.assertFalse(generate_menu.is_kadhi_item("ज्वार की रोटी और लौकी की सब्ज़ी"))

    def test_is_kadhi_chawal_item_requires_kadhi_and_rice(self) -> None:
        self.assertTrue(generate_menu.is_kadhi_chawal_item("कढ़ी और बासमती चावल"))
        self.assertTrue(generate_menu.is_kadhi_chawal_item("कढ़ी और शालि चावल"))
        self.assertFalse(generate_menu.is_kadhi_chawal_item("कढ़ी और ज्वार की रोटी"))

    def test_exclude_kadhi_items_on_rainy_day_filters_all_kadhi_variants(self) -> None:
        rainy_weather = generate_menu.WeatherInfo(
            morning_temp_c=25,
            max_temp_c=31,
            rain_probability_pct=80,
            is_rainy=True,
            is_extreme_cold=False,
            is_extreme_hot=False,
            source_hi="test",
        )
        filtered, applied = generate_menu.exclude_kadhi_items_on_rainy_day(
            [
                "कढ़ी और शालि चावल",
                "चावल और मजीदा कढ़ी",
                "ज्वार की रोटी और लौकी की सब्ज़ी",
            ],
            rainy_weather,
        )

        self.assertEqual(filtered, ["ज्वार की रोटी और लौकी की सब्ज़ी"])
        self.assertTrue(applied)

    def test_exclude_kadhi_items_on_non_rainy_day_keeps_pool(self) -> None:
        clear_weather = generate_menu.WeatherInfo(
            morning_temp_c=24,
            max_temp_c=33,
            rain_probability_pct=15,
            is_rainy=False,
            is_extreme_cold=False,
            is_extreme_hot=False,
            source_hi="test",
        )
        pool = ["कढ़ी और शालि चावल", "ज्वार की रोटी और लौकी की सब्ज़ी"]
        filtered, applied = generate_menu.exclude_kadhi_items_on_rainy_day(pool, clear_weather)

        self.assertEqual(filtered, pool)
        self.assertFalse(applied)

    def test_should_force_fortnightly_kadhi_chawal_when_missing_in_last_fourteen_days(self) -> None:
        history = [
            {"date": "2026-03-31", "breakfast": "उपमा", "meal": "दाल और चावल", "ritu_key": "vasant"},
            {"date": "2026-04-01", "breakfast": "पोहा", "meal": "लौकी", "ritu_key": "vasant"},
            {"date": "2026-04-02", "breakfast": "इडली", "meal": "परवल", "ritu_key": "vasant"},
            {"date": "2026-04-03", "breakfast": "डोसा", "meal": "भिंडी", "ritu_key": "vasant"},
            {"date": "2026-04-04", "breakfast": "दलिया", "meal": "मूंग दाल और चावल", "ritu_key": "vasant"},
            {"date": "2026-04-05", "breakfast": "चीला", "meal": "मसूर दाल और चावल", "ritu_key": "vasant"},
            {"date": "2026-04-06", "breakfast": "उपमा", "meal": "अरहर दाल और चावल", "ritu_key": "vasant"},
            {"date": "2026-04-07", "breakfast": "पोहा", "meal": "काला चना और चावल", "ritu_key": "vasant"},
            {"date": "2026-04-08", "breakfast": "इडली", "meal": "धुली मूंग दाल खिचड़ी", "ritu_key": "vasant"},
            {"date": "2026-04-09", "breakfast": "डोसा", "meal": "सादी मूँग दाल", "ritu_key": "vasant"},
            {"date": "2026-04-10", "breakfast": "दलिया", "meal": "जो की रोटी और लौकी की सब्ज़ी", "ritu_key": "vasant"},
            {"date": "2026-04-11", "breakfast": "चीला", "meal": "जौ की रोटी, लौकी की सब्ज़ी, मूँग दाल धुली", "ritu_key": "grishm"},
            {"date": "2026-04-12", "breakfast": "उपमा", "meal": "भिंडी की सब्ज़ी, गेहूँ की रोटी", "ritu_key": "varsha"},
            {"date": "2026-04-13", "breakfast": "पोहा", "meal": "मूँग की दाल, और साठी चावल", "ritu_key": "sharad"},
        ]

        self.assertTrue(generate_menu.should_force_fortnightly_kadhi_chawal(history, date(2026, 4, 14), "vasant"))

    def test_should_not_force_fortnightly_kadhi_chawal_when_recently_used(self) -> None:
        history = [
            {
                "date": "2026-04-10",
                "breakfast": "उपमा",
                "meal": "दाल और चावल",
                "second_meal": "कढ़ी और शालि चावल",
                "ritu_key": "grishm",
            }
        ]

        self.assertFalse(generate_menu.should_force_fortnightly_kadhi_chawal(history, date(2026, 4, 20), "grishm"))

    def test_should_not_force_fortnightly_kadhi_chawal_in_varsha(self) -> None:
        self.assertFalse(generate_menu.should_force_fortnightly_kadhi_chawal([], date(2026, 7, 20), "varsha"))


class WeeklyMainMealRiceLimitTests(unittest.TestCase):
    def test_count_current_week_main_meal_rice_counts_meal_and_second_meal_only(self) -> None:
        history = [
            {"date": "2026-05-31", "breakfast": "उपमा", "meal": "दाल और चावल", "ritu_key": "grishm"},
            {"date": "2026-06-01", "breakfast": "पखाला भात", "meal": "ज्वार की रोटी", "ritu_key": "grishm"},
            {"date": "2026-06-02", "breakfast": "उपमा", "meal": "दाल और चावल", "ritu_key": "grishm"},
            {
                "date": "2026-06-03",
                "breakfast": "इडली",
                "meal": "जौ की रोटी",
                "second_meal": "कढ़ी और शालि चावल",
                "ritu_key": "grishm",
            },
        ]

        self.assertEqual(generate_menu.count_current_week_main_meal_rice(history, date(2026, 6, 5)), 2)

    def test_apply_weekly_main_meal_rice_limit_filters_rice_after_two_this_week(self) -> None:
        history = [
            {"date": "2026-06-01", "breakfast": "उपमा", "meal": "दाल और चावल", "ritu_key": "grishm"},
            {"date": "2026-06-03", "breakfast": "इडली", "meal": "कढ़ी और शालि चावल", "ritu_key": "grishm"},
        ]

        filtered, applied = generate_menu.apply_weekly_main_meal_rice_limit(
            ["शालि चावल, मसूर दाल, लौकी की सब्ज़ी", "ज्वार की रोटी, लौकी की सब्ज़ी"],
            history,
            date(2026, 6, 5),
        )

        self.assertEqual(filtered, ["ज्वार की रोटी, लौकी की सब्ज़ी"])
        self.assertTrue(applied)

    def test_apply_weekly_main_meal_rice_limit_allows_second_rice_this_week(self) -> None:
        history = [
            {"date": "2026-06-01", "breakfast": "उपमा", "meal": "दाल और चावल", "ritu_key": "grishm"},
        ]

        filtered, applied = generate_menu.apply_weekly_main_meal_rice_limit(
            ["शालि चावल, मसूर दाल, लौकी की सब्ज़ी", "ज्वार की रोटी, लौकी की सब्ज़ी"],
            history,
            date(2026, 6, 5),
        )

        self.assertEqual(filtered, ["शालि चावल, मसूर दाल, लौकी की सब्ज़ी", "ज्वार की रोटी, लौकी की सब्ज़ी"])
        self.assertFalse(applied)


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
