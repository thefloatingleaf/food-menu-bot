import unittest

import household_purchase_ledger as ledger


class PurchaseLedgerNormalizationTests(unittest.TestCase):
    def test_normalize_purchase_entry_accepts_alias_fields(self) -> None:
        entry = ledger.normalize_purchase_entry(
            {
                "date": "09-05-2026",
                "item": "Mango",
                "category": "fruit",
                "quantity": "3",
                "unit": "kg",
                "price": "450",
                "vendor": "Local mandi",
                "payment_mode": "credit card",
                "expected_consumption_period": 5,
            }
        )

        self.assertEqual(entry["date_of_purchase"], "2026-05-09")
        self.assertEqual(entry["item_name"], "Mango")
        self.assertEqual(entry["quantity_purchased"], 3.0)
        self.assertEqual(entry["unit_of_measurement"], "kg")
        self.assertEqual(entry["vendor_source"], "Local mandi")
        self.assertEqual(entry["mode_of_payment"], "credit card")
        self.assertEqual(entry["expected_consumption_period"], {"value": 5.0, "unit": "days"})

    def test_normalize_purchase_entry_requires_core_fields(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing required fields"):
            ledger.normalize_purchase_entry({"item": "Apple"})


class PurchaseLedgerSummaryTests(unittest.TestCase):
    def test_build_analysis_snapshot_computes_reorder_and_spend_summary(self) -> None:
        entries = [
            ledger.normalize_purchase_entry(
                {
                    "date_of_purchase": "2026-05-01",
                    "item_name": "Banana",
                    "category": "fruit",
                    "quantity_purchased": 12,
                    "unit_of_measurement": "pieces",
                    "price": 60,
                    "vendor_source": "Vendor A",
                    "mode_of_payment": "upi",
                    "actual_consumption_period": {"value": 4, "unit": "days"},
                }
            ),
            ledger.normalize_purchase_entry(
                {
                    "date_of_purchase": "2026-05-05",
                    "item_name": "Banana",
                    "category": "fruit",
                    "quantity_purchased": 12,
                    "unit_of_measurement": "pieces",
                    "price": 65,
                    "vendor_source": "Vendor A",
                    "mode_of_payment": "upi",
                    "actual_consumption_period": {"value": 4, "unit": "days"},
                }
            ),
        ]

        snapshot = ledger.build_analysis_snapshot(entries)
        banana = snapshot["items"]["banana"]

        self.assertEqual(snapshot["totals"]["purchase_count"], 2)
        self.assertEqual(snapshot["totals"]["total_spend"], 125.0)
        self.assertEqual(banana["average_days_between_purchases"], 4.0)
        self.assertEqual(banana["average_actual_consumption_period_days"], 4.0)
        self.assertEqual(banana["suggested_next_reorder_date"], "2026-05-09")

    def test_build_analysis_snapshot_labels_only_possible_anomalies(self) -> None:
        entries = [
            ledger.normalize_purchase_entry(
                {
                    "date_of_purchase": "2026-05-01",
                    "item_name": "Milk",
                    "category": "dairy",
                    "quantity_purchased": 2,
                    "unit_of_measurement": "litre",
                    "price": 120,
                    "vendor_source": "Store A",
                    "mode_of_payment": "card",
                    "actual_consumption_period": {"value": 4, "unit": "days"},
                }
            ),
            ledger.normalize_purchase_entry(
                {
                    "date_of_purchase": "2026-05-05",
                    "item_name": "Milk",
                    "category": "dairy",
                    "quantity_purchased": 2,
                    "unit_of_measurement": "litre",
                    "price": 120,
                    "vendor_source": "Store A",
                    "mode_of_payment": "card",
                    "actual_consumption_period": {"value": 4, "unit": "days"},
                }
            ),
            ledger.normalize_purchase_entry(
                {
                    "date_of_purchase": "2026-05-09",
                    "item_name": "Milk",
                    "category": "dairy",
                    "quantity_purchased": 2,
                    "unit_of_measurement": "litre",
                    "price": 120,
                    "vendor_source": "Store A",
                    "mode_of_payment": "card",
                    "actual_consumption_period": {"value": 4, "unit": "days"},
                }
            ),
            ledger.normalize_purchase_entry(
                {
                    "date_of_purchase": "2026-05-11",
                    "item_name": "Milk",
                    "category": "dairy",
                    "quantity_purchased": 5,
                    "unit_of_measurement": "litre",
                    "price": 300,
                    "vendor_source": "Store A",
                    "mode_of_payment": "card",
                    "actual_consumption_period": {"value": 1, "unit": "days"},
                }
            ),
        ]

        snapshot = ledger.build_analysis_snapshot(entries)
        anomaly_messages = [anomaly["message"] for anomaly in snapshot["possible_anomalies"]]

        self.assertTrue(anomaly_messages)
        self.assertTrue(all("possible" in message.casefold() for message in anomaly_messages))
        self.assertTrue(all("confirmed" not in message.casefold() for message in anomaly_messages))


if __name__ == "__main__":
    unittest.main()
