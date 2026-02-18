#!/usr/bin/env python3
import argparse
import hashlib
import json
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

try:
    from zoneinfo import ZoneInfo
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("Python 3.9+ is required (zoneinfo missing)") from exc

BASE_DIR = Path(__file__).resolve().parent
MENU_FILE = BASE_DIR / "menu_shishir.json"
EKADASHI_FILE = BASE_DIR / "ekadashi_2026_27.json"
CONFIG_FILE = BASE_DIR / "config.json"
HISTORY_FILE = BASE_DIR / "history.json"
OUTPUT_FILE = BASE_DIR / "daily_menu.txt"


@dataclass
class EkadashiInfo:
    is_ekadashi: bool
    name_hi: str | None


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate daily food menu")
    parser.add_argument("--date", help="Date in YYYY-MM-DD format")
    return parser.parse_args()


def resolve_date(date_arg: str | None, timezone_name: str) -> datetime.date:
    if date_arg:
        try:
            return datetime.strptime(date_arg, "%Y-%m-%d").date()
        except ValueError as exc:
            raise ValueError("--date must be in YYYY-MM-DD format") from exc
    return datetime.now(ZoneInfo(timezone_name)).date()


def get_ekadashi_info(target_date: str, ekadashi_data: dict[str, Any]) -> EkadashiInfo:
    matches = [e for e in ekadashi_data.get("ekadashi_list", []) if e.get("date") == target_date]
    if not matches:
        return EkadashiInfo(False, None)
    # Prefer non-gauna label if both are present on same date.
    non_gauna = next((m for m in matches if not m.get("is_gauna", False)), None)
    chosen = non_gauna or matches[0]
    return EkadashiInfo(True, chosen.get("name_hi"))


def is_blocked_item(item: str, keywords: list[str]) -> bool:
    return any(keyword in item for keyword in keywords)


def normalize_history(raw: Any) -> list[dict[str, str]]:
    if not isinstance(raw, list):
        return []
    cleaned: list[dict[str, str]] = []
    for row in raw:
        if not isinstance(row, dict):
            continue
        date_val = row.get("date")
        item_val = row.get("item")
        if isinstance(date_val, str) and isinstance(item_val, str):
            cleaned.append({"date": date_val, "item": item_val})
    return cleaned


def recent_items(history: list[dict[str, str]], target_date: datetime.date, window_days: int) -> set[str]:
    earliest = target_date - timedelta(days=window_days)
    blocked: set[str] = set()
    for row in history:
        try:
            row_date = datetime.strptime(row["date"], "%Y-%m-%d").date()
        except ValueError:
            continue
        if earliest <= row_date < target_date:
            blocked.add(row["item"])
    return blocked


def choose_menu_item(
    menu_items: list[str],
    ekadashi: EkadashiInfo,
    recent_block_set: set[str],
    keywords: list[str],
    fallback_policy: str,
    seed_date: str,
) -> str:
    full_pool = menu_items[:]

    if ekadashi.is_ekadashi:
        base_pool = [item for item in full_pool if not is_blocked_item(item, keywords)]
    else:
        base_pool = full_pool[:]

    pool = [item for item in base_pool if item not in recent_block_set]
    if not pool:
        pool = base_pool[:]

    if not pool and ekadashi.is_ekadashi and fallback_policy == "fallback_full_menu":
        pool = [item for item in full_pool if item not in recent_block_set]
        if not pool:
            pool = full_pool[:]

    if not pool:
        raise RuntimeError("No menu item available after applying rules")

    seed_int = int(hashlib.sha256(seed_date.encode("utf-8")).hexdigest(), 16)
    rng = random.Random(seed_int)
    return rng.choice(pool)


def update_history(history: list[dict[str, str]], target_date: str, item: str, keep_days: int) -> list[dict[str, str]]:
    updated = [row for row in history if row.get("date") != target_date]
    updated.append({"date": target_date, "item": item})

    cutoff = datetime.strptime(target_date, "%Y-%m-%d").date() - timedelta(days=max(keep_days, 7) + 30)

    retained: list[dict[str, str]] = []
    for row in updated:
        try:
            row_date = datetime.strptime(row["date"], "%Y-%m-%d").date()
        except ValueError:
            continue
        if row_date >= cutoff:
            retained.append(row)

    retained.sort(key=lambda r: r["date"])
    return retained


def main() -> int:
    args = parse_args()

    config = load_json(CONFIG_FILE)
    menu_items = load_json(MENU_FILE)
    ekadashi_data = load_json(EKADASHI_FILE)
    history = normalize_history(load_json(HISTORY_FILE))

    if not isinstance(menu_items, list) or not all(isinstance(i, str) and i.strip() for i in menu_items):
        raise ValueError("menu_shishir.json must be a non-empty array of strings")

    timezone_name = config.get("timezone", "Asia/Kolkata")
    repeat_window_days = int(config.get("repeat_window_days", 7))
    fallback_policy = config.get("empty_filtered_pool_policy", "fallback_full_menu")
    keywords = config.get("ekadashi_block_keywords", [])

    if not isinstance(keywords, list) or not all(isinstance(k, str) for k in keywords):
        raise ValueError("ekadashi_block_keywords must be an array of strings")

    target_date = resolve_date(args.date, timezone_name)
    target_date_str = target_date.strftime("%Y-%m-%d")

    ekadashi = get_ekadashi_info(target_date_str, ekadashi_data)
    blocked_by_history = recent_items(history, target_date, repeat_window_days)

    selected_item = choose_menu_item(
        menu_items=menu_items,
        ekadashi=ekadashi,
        recent_block_set=blocked_by_history,
        keywords=keywords,
        fallback_policy=fallback_policy,
        seed_date=target_date_str,
    )

    new_history = update_history(history, target_date_str, selected_item, repeat_window_days)
    with HISTORY_FILE.open("w", encoding="utf-8") as f:
        json.dump(new_history, f, ensure_ascii=False, indent=2)
        f.write("\n")

    lines = [
        f"तिथि: {target_date_str}",
        f"आज का भोजन: {selected_item}",
    ]
    if ekadashi.is_ekadashi and ekadashi.name_hi:
        lines.append(f"एकादशी: {ekadashi.name_hi}")

    output_text = "\n".join(lines)

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        f.write(output_text + "\n")

    print(output_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
