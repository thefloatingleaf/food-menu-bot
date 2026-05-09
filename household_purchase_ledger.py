#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from statistics import median
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
LEDGER_DIR = BASE_DIR / "data" / "household_purchases"
LEDGER_FILE = LEDGER_DIR / "purchase_ledger.json"
ANALYSIS_FILE = LEDGER_DIR / "analysis_snapshot.json"
DEFAULT_CURRENCY = "INR"
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ConsumptionPeriod:
    value: float
    unit: str


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def normalize_date(value: Any) -> str | None:
    if isinstance(value, date):
        return value.isoformat()
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date().isoformat()
    except ValueError:
        return None


def normalize_number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip().replace(",", "")
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    return None


def normalize_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def normalize_period(value: Any) -> dict[str, Any] | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float, str)):
        numeric = normalize_number(value)
        if numeric is not None:
            return {"value": numeric, "unit": "days"}
        text = normalize_text(value)
        return {"value": None, "unit": text} if text else None
    if not isinstance(value, dict):
        return None
    numeric = normalize_number(value.get("value"))
    unit = normalize_text(value.get("unit")) or "days"
    if numeric is None and not unit:
        return None
    return {"value": numeric, "unit": unit}


def normalize_item_key(value: str) -> str:
    return " ".join(value.strip().casefold().split())


def default_ledger() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now_iso(),
        "updated_at": utc_now_iso(),
        "purchases": [],
    }


def default_analysis() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": None,
        "totals": {
            "purchase_count": 0,
            "tracked_item_count": 0,
            "currency": DEFAULT_CURRENCY,
            "total_spend": 0.0,
        },
        "items": {},
        "possible_anomalies": [],
        "monthly_spend": {},
        "category_spend": {},
    }


def ensure_storage_files() -> None:
    LEDGER_DIR.mkdir(parents=True, exist_ok=True)
    if not LEDGER_FILE.exists():
        write_json(LEDGER_FILE, default_ledger())
    if not ANALYSIS_FILE.exists():
        write_json(ANALYSIS_FILE, default_analysis())


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def load_ledger() -> dict[str, Any]:
    ensure_storage_files()
    data = load_json(LEDGER_FILE)
    if not isinstance(data, dict):
        raise ValueError("purchase ledger root must be a JSON object")
    purchases = data.get("purchases")
    if not isinstance(purchases, list):
        raise ValueError("purchase ledger purchases must be a list")
    data.setdefault("schema_version", SCHEMA_VERSION)
    if not data.get("created_at"):
        data["created_at"] = utc_now_iso()
    if not data.get("updated_at"):
        data["updated_at"] = utc_now_iso()
    return data


def generate_purchase_id(normalized_entry: dict[str, Any]) -> str:
    basis = "|".join(
        [
            normalized_entry.get("date_of_purchase") or "",
            normalized_entry.get("item_name") or "",
            str(normalized_entry.get("quantity_purchased") or ""),
            normalized_entry.get("unit_of_measurement") or "",
            str(normalized_entry.get("price") or ""),
            normalized_entry.get("vendor_source") or "",
        ]
    )
    digest = hashlib.sha1(basis.encode("utf-8")).hexdigest()[:10]
    return f"purchase-{digest}"


def normalize_purchase_entry(entry: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(entry, dict):
        raise ValueError("purchase entry must be a JSON object")

    normalized = {
        "purchase_id": normalize_text(entry.get("purchase_id")),
        "date_of_purchase": normalize_date(entry.get("date_of_purchase") or entry.get("date")),
        "item_name": normalize_text(entry.get("item_name") or entry.get("item")),
        "category": normalize_text(entry.get("category")) or "uncategorized",
        "quantity_purchased": normalize_number(entry.get("quantity_purchased") or entry.get("quantity")),
        "unit_of_measurement": normalize_text(entry.get("unit_of_measurement") or entry.get("unit")),
        "price": normalize_number(entry.get("price")),
        "currency": normalize_text(entry.get("currency")) or DEFAULT_CURRENCY,
        "vendor_source": normalize_text(entry.get("vendor_source") or entry.get("vendor") or entry.get("source")),
        "mode_of_payment": normalize_text(entry.get("mode_of_payment") or entry.get("payment_mode")),
        "payment_reference": normalize_text(entry.get("payment_reference")),
        "bill_invoice_reference": normalize_text(
            entry.get("bill_invoice_reference") or entry.get("bill_reference") or entry.get("invoice_reference")
        ),
        "expected_consumption_period": normalize_period(entry.get("expected_consumption_period")),
        "actual_consumption_period": normalize_period(entry.get("actual_consumption_period")),
        "remarks": normalize_text(entry.get("remarks")),
        "entered_at": normalize_text(entry.get("entered_at")) or utc_now_iso(),
        "last_updated_at": utc_now_iso(),
    }

    if normalized["purchase_id"] is None:
        normalized["purchase_id"] = generate_purchase_id(normalized)

    required_fields = (
        "date_of_purchase",
        "item_name",
        "quantity_purchased",
        "unit_of_measurement",
        "price",
        "vendor_source",
        "mode_of_payment",
    )
    missing = [field for field in required_fields if normalized.get(field) in (None, "")]
    if missing:
        raise ValueError(f"purchase entry missing required fields: {', '.join(missing)}")

    return normalized


def append_purchase_entries(raw_entries: list[dict[str, Any]]) -> dict[str, Any]:
    ledger = load_ledger()
    existing_entries = [normalize_purchase_entry(entry) for entry in ledger.get("purchases", [])]
    existing_ids = {entry["purchase_id"] for entry in existing_entries}
    new_entries = []
    for raw in raw_entries:
        normalized = normalize_purchase_entry(raw)
        if normalized["purchase_id"] in existing_ids:
            raise ValueError(f"duplicate purchase_id: {normalized['purchase_id']}")
        existing_ids.add(normalized["purchase_id"])
        new_entries.append(normalized)

    ledger["purchases"] = sorted(existing_entries + new_entries, key=lambda item: (item["date_of_purchase"], item["purchase_id"]))
    ledger["updated_at"] = utc_now_iso()
    write_json(LEDGER_FILE, ledger)
    analysis = build_analysis_snapshot(ledger["purchases"])
    write_json(ANALYSIS_FILE, analysis)
    return ledger


def period_to_days(period: dict[str, Any] | None) -> float | None:
    if not period or period.get("value") is None:
        return None
    unit = normalize_text(period.get("unit")) or "days"
    value = normalize_number(period.get("value"))
    if value is None:
        return None
    unit_key = unit.casefold()
    if unit_key.startswith("day"):
        return value
    if unit_key.startswith("week"):
        return value * 7.0
    if unit_key.startswith("month"):
        return value * 30.0
    return value


def compute_item_summary(item_name: str, entries: list[dict[str, Any]]) -> dict[str, Any]:
    dates = [datetime.strptime(entry["date_of_purchase"], "%Y-%m-%d").date() for entry in entries]
    dates.sort()
    quantities = [entry["quantity_purchased"] for entry in entries if entry.get("quantity_purchased") is not None]
    prices = [entry["price"] for entry in entries if entry.get("price") is not None]
    units = {entry.get("unit_of_measurement") for entry in entries if entry.get("unit_of_measurement")}
    category_counts: dict[str, int] = defaultdict(int)
    vendor_counts: dict[str, int] = defaultdict(int)
    expected_days = [
        period_to_days(entry.get("expected_consumption_period")) for entry in entries if entry.get("expected_consumption_period")
    ]
    expected_days = [value for value in expected_days if value is not None]
    actual_days = [
        period_to_days(entry.get("actual_consumption_period")) for entry in entries if entry.get("actual_consumption_period")
    ]
    actual_days = [value for value in actual_days if value is not None]

    for entry in entries:
        category_counts[entry.get("category") or "uncategorized"] += 1
        vendor_counts[entry.get("vendor_source") or "unknown"] += 1

    intervals = [(dates[index] - dates[index - 1]).days for index in range(1, len(dates))]
    quantity_unit = next(iter(units)) if len(units) == 1 else None
    total_quantity = sum(quantities) if quantities else None
    total_spend = round(sum(prices), 2) if prices else 0.0
    average_quantity = round(total_quantity / len(quantities), 3) if quantities else None
    average_interval_days = round(sum(intervals) / len(intervals), 2) if intervals else None
    expected_duration_days = round(sum(expected_days) / len(expected_days), 2) if expected_days else None
    actual_duration_days = round(sum(actual_days) / len(actual_days), 2) if actual_days else None

    average_daily_consumption = None
    if quantity_unit and actual_days and quantities:
        usable_pairs = [
            (entry["quantity_purchased"], period_to_days(entry.get("actual_consumption_period")))
            for entry in entries
            if entry.get("quantity_purchased") is not None and entry.get("actual_consumption_period")
        ]
        usable_pairs = [(quantity, days) for quantity, days in usable_pairs if days not in (None, 0)]
        if usable_pairs:
            total_qty = sum(quantity for quantity, _days in usable_pairs)
            total_days = sum(days for _quantity, days in usable_pairs)
            if total_days:
                average_daily_consumption = round(total_qty / total_days, 4)

    expected_stock_duration_days = actual_duration_days or expected_duration_days
    next_reorder_date = None
    if expected_stock_duration_days is not None:
        next_reorder_date = (dates[-1] + timedelta(days=round(expected_stock_duration_days))).isoformat()
    elif average_interval_days is not None:
        next_reorder_date = (dates[-1] + timedelta(days=round(average_interval_days))).isoformat()

    possible_anomalies = compute_possible_anomalies(item_name, entries, intervals, quantities, actual_days)

    return {
        "item_name": item_name,
        "purchase_count": len(entries),
        "last_purchase_date": dates[-1].isoformat(),
        "quantity_unit": quantity_unit,
        "total_quantity_purchased": round(total_quantity, 3) if total_quantity is not None else None,
        "average_quantity_purchased": average_quantity,
        "total_spend": total_spend,
        "average_days_between_purchases": average_interval_days,
        "average_expected_consumption_period_days": expected_duration_days,
        "average_actual_consumption_period_days": actual_duration_days,
        "average_daily_consumption": average_daily_consumption,
        "expected_stock_duration_days": expected_stock_duration_days,
        "suggested_next_reorder_date": next_reorder_date,
        "dominant_category": max(category_counts, key=category_counts.get) if category_counts else None,
        "common_vendor": max(vendor_counts, key=vendor_counts.get) if vendor_counts else None,
        "possible_anomalies": possible_anomalies,
    }


def compute_possible_anomalies(
    item_name: str,
    entries: list[dict[str, Any]],
    intervals: list[int],
    quantities: list[float],
    actual_days: list[float],
) -> list[dict[str, Any]]:
    anomalies: list[dict[str, Any]] = []
    if len(quantities) >= 3:
        current_quantity = quantities[-1]
        baseline_quantity = median(quantities[:-1])
        if baseline_quantity > 0 and current_quantity >= baseline_quantity * 1.75:
            anomalies.append(
                {
                    "kind": "possible_over_purchase",
                    "severity": "low",
                    "message": (
                        f"{item_name} की नवीनतम खरीदी मात्रा पहले के सामान्य स्तर से काफी अधिक दिख रही है. "
                        "इसे केवल possible anomaly मानें; वास्तविक कारण की अलग पुष्टि चाहिए."
                    ),
                }
            )

    if len(intervals) >= 3:
        current_interval = intervals[-1]
        baseline_interval = median(intervals[:-1])
        if baseline_interval > 0 and current_interval <= baseline_interval * 0.5:
            anomalies.append(
                {
                    "kind": "possible_fast_depletion",
                    "severity": "medium",
                    "message": (
                        f"{item_name} को सामान्य से काफी जल्दी दोबारा खरीदा गया है. "
                        "यह तेज खपत, अतिरिक्त उपयोग, wastage, या data gap का possible signal हो सकता है."
                    ),
                }
            )

    if len(actual_days) >= 3:
        current_actual_days = actual_days[-1]
        baseline_actual_days = median(actual_days[:-1])
        if baseline_actual_days > 0 and current_actual_days <= baseline_actual_days * 0.5:
            anomalies.append(
                {
                    "kind": "possible_unusual_depletion",
                    "severity": "medium",
                    "message": (
                        f"{item_name} का actual consumption period सामान्य से बहुत कम दिख रहा है. "
                        "इसे possible anomaly के रूप में देखें; shortage या misuse का निष्कर्ष तभी मानें जब अलग सबूत मिले."
                    ),
                }
            )
    return anomalies


def build_analysis_snapshot(entries: list[dict[str, Any]]) -> dict[str, Any]:
    snapshot = default_analysis()
    snapshot["generated_at"] = utc_now_iso()
    if not entries:
        return snapshot

    by_item: dict[str, list[dict[str, Any]]] = defaultdict(list)
    monthly_spend: dict[str, float] = defaultdict(float)
    category_spend: dict[str, float] = defaultdict(float)

    for entry in sorted(entries, key=lambda row: (row["date_of_purchase"], row["purchase_id"])):
        by_item[normalize_item_key(entry["item_name"])].append(entry)
        purchase_month = entry["date_of_purchase"][:7]
        monthly_spend[purchase_month] += entry.get("price") or 0.0
        category_spend[entry.get("category") or "uncategorized"] += entry.get("price") or 0.0

    item_summaries = {
        item_key: compute_item_summary(item_entries[0]["item_name"], item_entries) for item_key, item_entries in by_item.items()
    }
    anomalies = [
        {"item_key": item_key, **anomaly}
        for item_key, summary in item_summaries.items()
        for anomaly in summary["possible_anomalies"]
    ]

    snapshot["totals"] = {
        "purchase_count": len(entries),
        "tracked_item_count": len(item_summaries),
        "currency": DEFAULT_CURRENCY,
        "total_spend": round(sum((entry.get("price") or 0.0) for entry in entries), 2),
    }
    snapshot["items"] = item_summaries
    snapshot["possible_anomalies"] = anomalies
    snapshot["monthly_spend"] = {month: round(amount, 2) for month, amount in sorted(monthly_spend.items())}
    snapshot["category_spend"] = {category: round(amount, 2) for category, amount in sorted(category_spend.items())}
    return snapshot


def validate_ledger_payload(ledger: dict[str, Any]) -> None:
    if ledger.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version: {ledger.get('schema_version')}")
    purchases = ledger.get("purchases")
    if not isinstance(purchases, list):
        raise ValueError("ledger purchases must be a list")
    seen_ids: set[str] = set()
    for raw_entry in purchases:
        normalized = normalize_purchase_entry(raw_entry)
        if normalized["purchase_id"] in seen_ids:
            raise ValueError(f"duplicate purchase_id found in ledger: {normalized['purchase_id']}")
        seen_ids.add(normalized["purchase_id"])


def load_entries_from_input(path: Path | None, use_stdin: bool) -> list[dict[str, Any]]:
    if path is None and not use_stdin:
        raise ValueError("provide either --input PATH or --stdin")
    if path is not None and use_stdin:
        raise ValueError("use only one of --input PATH or --stdin")

    payload_text = path.read_text(encoding="utf-8") if path is not None else input_stream_text()
    payload = json.loads(payload_text)
    if isinstance(payload, dict):
        return [payload]
    if isinstance(payload, list):
        if not all(isinstance(item, dict) for item in payload):
            raise ValueError("input list must contain only JSON objects")
        return payload
    raise ValueError("input payload must be a JSON object or list of objects")


def input_stream_text() -> str:
    import sys

    return sys.stdin.read()


def command_ensure(_args: argparse.Namespace) -> int:
    ensure_storage_files()
    ledger = load_ledger()
    write_json(LEDGER_FILE, ledger)
    write_json(ANALYSIS_FILE, build_analysis_snapshot(ledger["purchases"]))
    return 0


def command_validate(_args: argparse.Namespace) -> int:
    ledger = load_ledger()
    validate_ledger_payload(ledger)
    return 0


def command_add(args: argparse.Namespace) -> int:
    entries = load_entries_from_input(args.input, args.stdin)
    append_purchase_entries(entries)
    return 0


def command_summarize(_args: argparse.Namespace) -> int:
    ledger = load_ledger()
    validate_ledger_payload(ledger)
    write_json(ANALYSIS_FILE, build_analysis_snapshot(ledger["purchases"]))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Maintain a silent household purchase ledger.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ensure_parser = subparsers.add_parser("ensure", help="Create ledger and analysis files if missing")
    ensure_parser.set_defaults(handler=command_ensure)

    validate_parser = subparsers.add_parser("validate", help="Validate the current ledger structure")
    validate_parser.set_defaults(handler=command_validate)

    add_parser = subparsers.add_parser("add", help="Append one or more purchase entries from JSON")
    add_parser.add_argument("--input", type=Path, help="Path to a JSON object or JSON array")
    add_parser.add_argument("--stdin", action="store_true", help="Read a JSON object or JSON array from stdin")
    add_parser.set_defaults(handler=command_add)

    summarize_parser = subparsers.add_parser("summarize", help="Refresh the derived analysis snapshot")
    summarize_parser.set_defaults(handler=command_summarize)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
