#!/usr/bin/env python3
import sys
from datetime import date

import generate_menu

MISSING_STANDARD_ROW_EXIT = 42


def get_stored_navishti_plan(row: dict[str, object]) -> list[str]:
    stored_plan = row.get("navishti_grishm_plan")
    if not isinstance(stored_plan, list):
        return []
    return [str(item).strip() for item in stored_plan if isinstance(item, str) and str(item).strip()]


def resolve_navishti_target_date_from_standard_menu(
    output_text: str,
    timezone_name: str,
    now_date: date | None = None,
) -> date:
    standard_target_date = generate_menu.parse_output_target_date(output_text)
    expected_target_date = generate_menu.resolve_date(None, timezone_name, now_date=now_date)
    if standard_target_date != expected_target_date:
        raise ValueError(
            "standard menu date mismatch: "
            f"expected {expected_target_date.isoformat()}, found {standard_target_date.isoformat()}"
        )
    return standard_target_date


def main() -> int:
    config = generate_menu.load_json(generate_menu.CONFIG_FILE) if generate_menu.CONFIG_FILE.exists() else {}
    timezone_name = str(config.get("timezone", "Asia/Kolkata"))

    if not generate_menu.OUTPUT_FILE.exists():
        print(
            f"{generate_menu.OUTPUT_FILE.name} is missing; generate the standard daily menu before Navishti.",
            file=sys.stderr,
        )
        return MISSING_STANDARD_ROW_EXIT

    try:
        target_date = resolve_navishti_target_date_from_standard_menu(
            generate_menu.OUTPUT_FILE.read_text(encoding="utf-8"),
            timezone_name,
        )
    except ValueError as exc:
        print(
            "standard daily menu is not ready for Navishti generation: " + str(exc),
            file=sys.stderr,
        )
        return MISSING_STANDARD_ROW_EXIT

    target_date_str = target_date.isoformat()

    history = generate_menu.normalize_history(generate_menu.load_json(generate_menu.HISTORY_FILE))
    history_row = generate_menu.get_history_row(history, target_date_str)
    if history_row is None:
        print(
            "history.json is missing the standard menu row required for Navishti generation: "
            f"{target_date_str}. Run the daily menu generator first.",
            file=sys.stderr,
        )
        return MISSING_STANDARD_ROW_EXIT

    ritu_key = generate_menu.normalize_ritu_key(str(history_row.get("ritu_key", "")))
    plan_items = get_stored_navishti_plan(history_row)
    if ritu_key == "grishm" and not plan_items:
        previous_items = generate_menu.get_previous_navishti_grishm_plan_items(history, target_date)
        plan_items = generate_menu.resolve_navishti_grishm_plan_items(target_date, None, previous_items)

    output_text = generate_menu.format_navishti_daily_menu_text(target_date, ritu_key, plan_items)
    generate_menu.write_output_text(generate_menu.NAVISHTI_OUTPUT_FILE, output_text)

    print(output_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
