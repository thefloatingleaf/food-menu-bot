# Food Menu Daily WhatsApp System

## Run locally

```bash
cd "/Users/gg/Documents/Food Menu"
source .venv/bin/activate
python3 generate_menu.py
```

## Test with specific date

```bash
python3 generate_menu.py --date 2026-02-27
```

## Bootstrap weather tags (one-time)

```bash
python3 generate_menu.py --bootstrap-weather-tags
```

## Output format

- `*तिथि:* YYYY-MM-DD`
- `*ऋतु:* <value>`
- `*माह:* <value>`
- `*तिथि (पंचांग):* <value>`
- `*पर्व/त्योहार:* <हिन्दू/सिख पर्व>`
- `*सुबह का नाश्ता:* <item>`
- `*आज का भोजन:* <item>`
- `*एकादशी:* <name_hi>` (only on Ekadashi/Gauna dates)
- `*मौसम:* <details>` (only rainy/extreme days)

## Data files

- `breakfast_shishir.json`
- `menu_shishir.json`
- `ekadashi_2026_27.json`
- `panchang_2026_27.json`
- `festivals_2026_27.json`
- `menu_weather_tags.json`
- `manual_weather_override.json`
- `config.json`

## Festivals data format (Hindu + Sikh)

```json
{
  "entries": [
    {
      "date": "2026-02-18",
      "hindu_hi": ["महाशिवरात्रि"],
      "sikh_hi": ["गुरु पर्व"]
    }
  ]
}
```

## Weather flow (free)

1. Use manual override if date exists in `manual_weather_override.json`
2. Else fetch Open-Meteo forecast for configured coordinates
3. Else continue without weather filter (menu generation never fails)

## Manual weather override format

```json
{
  "2026-02-19": {
    "morning_temp_c": 14,
    "max_temp_c": 27,
    "rain_probability_pct": 20,
    "source_hi": "मैनुअल अनुमान"
  }
}
```

## Panchang data format

```json
{
  "entries": [
    {
      "date": "2026-02-18",
      "ritu_hi": "शिशिर",
      "maah_hi": "फाल्गुन",
      "tithi_hi": "प्रतिपदा"
    }
  ]
}
```

## GitHub Action schedule

- Workflow file: `.github/workflows/daily-menu.yml`
- Runs daily at `00:00 UTC` (5:30 AM IST)
- Updates `daily_menu.txt` and `history.json`

## iPhone Shortcuts

Use `Get Contents of URL` with raw GitHub URL of `daily_menu.txt`, then send the fetched text through WhatsApp action.
