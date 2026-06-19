#!/usr/bin/env python3
from datetime import timedelta

import generate_menu


def get_stored_navishti_plan(row: dict[str, object]) -> list[str]:
    stored_plan = row.get("navishti_grishm_plan")
    if not isinstance(stored_plan, list):
        return []
    return [str(item).strip() for item in stored_plan if isinstance(item, str) and str(item).strip()]


def main() -> int:
    config = generate_menu.load_json(generate_menu.CONFIG_FILE) if generate_menu.CONFIG_FILE.exists() else {}
    timezone_name = str(config.get("timezone", "Asia/Kolkata"))
    target_date = generate_menu.resolve_runtime_today(timezone_name) + timedelta(days=1)
    target_date_str = target_date.isoformat()

    history = generate_menu.normalize_history(generate_menu.load_json(generate_menu.HISTORY_FILE))
    history_row = generate_menu.get_history_row(history, target_date_str)
    if history_row is None:
        raise SystemExit(
            "history.json is missing the standard menu row required for Navishti generation: "
            f"{target_date_str}. Run the daily menu generator first."
        )

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
