import unittest
from datetime import date

from scripts import verify_daily_menu_freshness


class PanchangFreshnessGuardTests(unittest.TestCase):
    def test_accepts_target_date_present_in_panchang_source(self) -> None:
        verify_daily_menu_freshness.verify_panchang_target_date(
            date(2027, 3, 15),
            "Asia/Kolkata",
            {"entries": [{"date": "2027-03-15", "maah_hi": "फाल्गुन", "tithi_hi": "सप्तमी"}]},
        )

    def test_rejects_target_date_after_panchang_coverage(self) -> None:
        with self.assertRaisesRegex(ValueError, "2027-03-16"):
            verify_daily_menu_freshness.verify_panchang_target_date(
                date(2027, 3, 16),
                "Asia/Kolkata",
                {"entries": [{"date": "2027-03-15", "maah_hi": "फाल्गुन", "tithi_hi": "सप्तमी"}]},
            )


if __name__ == "__main__":
    unittest.main()
