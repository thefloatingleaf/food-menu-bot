#!/usr/bin/env python3
import sys
from datetime import timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import generate_menu


def verify_panchang_target_date(expected_target_date, timezone_name: str, panchang_data) -> None:
    lookup = generate_menu.lookup_panchang_entry_for_date(
        expected_target_date,
        timezone_name,
        panchang_data,
    )
    if lookup.status != "ok" or lookup.row is None:
        raise ValueError(
            "authoritative Panchang data is missing for target date "
            f"{expected_target_date.isoformat()}: {lookup.detail_hi}"
        )


def main() -> int:
    config = generate_menu.load_json(generate_menu.CONFIG_FILE) if generate_menu.CONFIG_FILE.exists() else {}
    timezone_name = str(config.get("timezone", "Asia/Kolkata"))
    runtime_today = generate_menu.resolve_runtime_today(timezone_name)
    expected_target_date = runtime_today + timedelta(days=1)

    if not generate_menu.PANCHANG_FILE.exists():
        raise SystemExit(f"missing Panchang source file: {generate_menu.PANCHANG_FILE}")
    try:
        panchang_data = generate_menu.load_json(generate_menu.PANCHANG_FILE)
        verify_panchang_target_date(expected_target_date, timezone_name, panchang_data)
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    if not generate_menu.OUTPUT_FILE.exists():
        raise SystemExit(f"missing output file: {generate_menu.OUTPUT_FILE}")

    output_text = generate_menu.OUTPUT_FILE.read_text(encoding="utf-8")
    generate_menu.verify_output_target_date(output_text, expected_target_date)
    if "नविष्टि" in output_text:
        raise SystemExit(
            f"{generate_menu.OUTPUT_FILE.name} must remain the standard food message; "
            "Navishti belongs only in navishti_daily_menu.txt"
        )

    history = generate_menu.normalize_history(generate_menu.load_json(generate_menu.HISTORY_FILE))
    history_row = generate_menu.get_history_row(history, expected_target_date.isoformat())
    if history_row is None:
        raise SystemExit(
            "history.json is missing the generated target date entry: "
            f"{expected_target_date.isoformat()}"
        )

    print(
        "Verified daily menu freshness for "
        f"{expected_target_date.isoformat()} in {timezone_name}: "
        f"{generate_menu.OUTPUT_FILE.name} and {generate_menu.HISTORY_FILE.name}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
