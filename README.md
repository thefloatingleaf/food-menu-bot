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
- `*पर्व/त्योहार:* <festival names>` (only when present on that date)
- `*सुबह का नाश्ता:* <item>`
- `*आज का भोजन:* <item>`
- `*एकादशी:* <name_hi>` (only on Ekadashi/Gauna dates)
- `*मौसम:* <details>` (only rainy/extreme days)
- `*वसंत अनिवार्य साथ:* ...` (only when ऋतु is वसंत)

## Data files

- `breakfast_shishir.json`
- `menu_shishir.json`
- `breakfast_vasant.json`
- `menu_vasant.json`
- `breakfast_grishm.json`
- `menu_grishm.json`
- `breakfast_varsha.json`
- `menu_varsha.json`
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

## Seasonal menu behavior

1. If पंचांग ऋतु is `वसंत`, the script uses:
   - `breakfast_vasant.json`
   - `menu_vasant.json`
2. For वसंत days, output also includes:
   - `*वसंत अनिवार्य साथ:* नीम की चटनी / पुदीना की चटनी / लहसुन की चटनी / तीखा अचार (खट्टा नहीं) / मूंग दाल पापड़ / मसाला छाछ ...`
3. Otherwise, script uses Shishir files:
   - `breakfast_shishir.json`
   - `menu_shishir.json`
4. If पंचांग ऋतु is `ग्रीष्म` or `ग्रीष्म ऋतु`, the script uses:
   - `breakfast_grishm.json`
   - `menu_grishm.json`
5. For ग्रीष्म days, output also includes:
   - `*ग्रीष्म नाश्ता अनिवार्य साथ:* छाछ (काफ़ी पतली) / पुदीना की चटनी`
   - `*ग्रीष्म भोजन अनिवार्य साथ:* छाछ (काफ़ी पतली) / पुदीना की चटनी / खीरा और ककड़ी`
6. `breakfast_grishm.json` duplicate entries are deduplicated (first occurrence kept) before random selection.
7. If पंचांग ऋतु is `वर्षा` or `वर्षा ऋतु`, the script uses:
   - `breakfast_varsha.json`
   - `menu_varsha.json`
8. For वर्षा days, output also includes:
   - `*वर्षा नाश्ता अनिवार्य साथ:* आचार / मिश्री-सौंफ़ / छाछ त्रिकटु के साथ`
   - `*वर्षा भोजन अनिवार्य साथ:* आचार / मिश्री-सौंफ़ / छाछ त्रिकटु के साथ`
   - `*वर्षा वर्जित:* प्याज और दही पूर्णतः मना है`
9. वर्षा days enforce hard filtering for `प्याज`, `प्याज़`, `दही` in breakfast and भोजन selection.

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
