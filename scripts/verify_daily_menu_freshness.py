#!/usr/bin/env python3
import sys
from datetime import timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import generate_menu


def main() -> int:
    config = generate_menu.load_json(generate_menu.CONFIG_FILE) if generate_menu.CONFIG_FILE.exists() else {}
    timezone_name = str(config.get("timezone", "Asia/Kolkata"))
    runtime_today = generate_menu.resolve_runtime_today(timezone_name)
    expected_target_date = runtime_today + timedelta(days=1)

    if not generate_menu.OUTPUT_FILE.exists():
        raise SystemExit(f"missing output file: {generate_menu.OUTPUT_FILE}")

    output_text = generate_menu.OUTPUT_FILE.read_text(encoding="utf-8")
    generate_menu.verify_output_target_date(output_text, expected_target_date)

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
