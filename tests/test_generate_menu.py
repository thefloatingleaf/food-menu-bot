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


if __name__ == "__main__":
    unittest.main()
