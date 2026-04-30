#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import sys
from contextlib import contextmanager
from datetime import date, timedelta
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import generate_menu


CONFIG_FILE = BASE_DIR / "config.json"
HISTORY_FILE = BASE_DIR / "history.json"
OUTPUT_FILE = BASE_DIR / "daily_menu.txt"
OUTPUT_PAYLOAD_FILE = BASE_DIR / "daily_menu_payload.json"
PUBLISHED_ARCHIVE_FILE = BASE_DIR / "published_menu_archive.json"
TEST_OUTPUT_DIR = BASE_DIR / "test_outputs" / "menu_triggers"

TEST_CASES = [
    {
        "slug": "mangore",
        "target_date": "2026-04-20",
        "item": "उबले हुए मंगोड़े",
    },
    {
        "slug": "pazhaya-sadam",
        "target_date": "2026-07-21",
        "item_label": "पझैया सादम (Pazhaya Sadam)",
    },
    {
        "slug": "pakhala-bhata",
        "target_date": "2026-07-22",
        "item_label": "पखाला भात (Pakhala Bhata)",
    },
]


def read_text_if_exists(path: Path) -> str | None:
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def restore_text(path: Path, content: str | None) -> None:
    if content is None:
        if path.exists():
            path.unlink()
        return
    path.write_text(content, encoding="utf-8")


def resolve_overnight_item(item_label: str) -> str:
    for item in generate_menu.OVERNIGHT_BREAKFAST_ITEMS:
        if item.split(":", 1)[0].strip() == item_label:
            return item
    raise ValueError(f"Overnight breakfast item not found: {item_label}")


def build_test_overrides() -> list[dict[str, str]]:
    overrides: list[dict[str, str]] = []
    for case in TEST_CASES:
        item = case.get("item")
        if not item:
            item = resolve_overnight_item(str(case["item_label"]))
        overrides.append({"date": str(case["target_date"]), "item": str(item)})
    return overrides


@contextmanager
def preserved_runtime_files() -> None:
    original_config = read_text_if_exists(CONFIG_FILE)
    original_history = read_text_if_exists(HISTORY_FILE)
    original_output = read_text_if_exists(OUTPUT_FILE)
    original_payload = read_text_if_exists(OUTPUT_PAYLOAD_FILE)
    original_archive = read_text_if_exists(PUBLISHED_ARCHIVE_FILE)
    try:
        yield
    finally:
        restore_text(CONFIG_FILE, original_config)
        restore_text(HISTORY_FILE, original_history)
        restore_text(OUTPUT_FILE, original_output)
        restore_text(OUTPUT_PAYLOAD_FILE, original_payload)
        restore_text(PUBLISHED_ARCHIVE_FILE, original_archive)


def run_generator_for_target(target_date_str: str) -> str:
    target_date = date.fromisoformat(target_date_str)
    simulated_today = target_date - timedelta(days=1)
    env = os.environ.copy()
    env[generate_menu.MENU_GENERATOR_NOW_DATE_ENV] = simulated_today.isoformat()
    subprocess.run(
        [sys.executable, str(BASE_DIR / "generate_menu.py")],
        cwd=BASE_DIR,
        check=True,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return OUTPUT_FILE.read_text(encoding="utf-8")


def main() -> int:
    config = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    config["breakfast_item_date_overrides"] = build_test_overrides()
    TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest_lines = [
        "Generated menu trigger test outputs:",
        "",
    ]

    with preserved_runtime_files():
        CONFIG_FILE.write_text(
            json.dumps(config, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        for case in TEST_CASES:
            HISTORY_FILE.write_text("[]\n", encoding="utf-8")
            output_text = run_generator_for_target(str(case["target_date"]))
            output_path = TEST_OUTPUT_DIR / f"{case['slug']}.txt"
            output_path.write_text(output_text, encoding="utf-8")
            manifest_lines.append(f"- {case['slug']}: {output_path}")

    manifest_path = TEST_OUTPUT_DIR / "README.txt"
    manifest_path.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")

    print(f"Created trigger test menus in {TEST_OUTPUT_DIR}")
    for case in TEST_CASES:
        print(f"- {case['slug']}: {TEST_OUTPUT_DIR / (case['slug'] + '.txt')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
