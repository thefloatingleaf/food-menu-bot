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
                            "Day 1 - Maa Shailputri: Desi ghee विशेष रूप से ग्रहण या अर्पित करें।"
                        ),
                    }
                ]
            },
        )

        self.assertEqual(info.hindu_hi, ["चैत्र नवरात्रि"])
        self.assertTrue(info.suppress_regular_menu)
        self.assertEqual(
            info.special_menu_note_hi,
            "Day 1 - Maa Shailputri: Desi ghee विशेष रूप से ग्रहण या अर्पित करें।",
        )

    def test_format_special_menu_note_line_uses_special_note(self) -> None:
        info = generate_menu.FestivalInfo(
            hindu_hi=["चैत्र नवरात्रि"],
            sikh_hi=[],
            suppress_regular_menu=True,
            special_menu_note_hi="Day 8 - Maa Mahagauri: Coconut विशेष रूप से ग्रहण या अर्पित करें।",
        )

        self.assertEqual(
            generate_menu.format_special_menu_note_line(info),
            "*विशेष पारंपरिक सेवन/भोग:* Day 8 - Maa Mahagauri: Coconut विशेष रूप से ग्रहण या अर्पित करें।",
        )


if __name__ == "__main__":
    unittest.main()
