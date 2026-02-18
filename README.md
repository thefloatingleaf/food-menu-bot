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
- `breakfast_sharad.json`
- `menu_sharad.json`
- `breakfast_hemant.json`
- `menu_hemant.json` (optional; if missing, food falls back to Shishir)
- `ekadashi_2026_27.json`
- `panchang_2026_27.json`
- `festivals_2026_27.json`
- `menu_weather_tags.json`
- `manual_weather_override.json`
- `lunar_calendar_2026_2027.json` (reference calendar: lunar months, sankranti, amavasya, purnima, ekadashi, partial daily tables)
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
10. If पंचांग ऋतु is `शरद` or `शरद ऋतु`, the script uses:
   - `breakfast_sharad.json`
   - `menu_sharad.json`
11. For शरद days, output also includes:
   - `*शरद अनिवार्य साथ:* सौंफ-मिश्री की मिश्रण / छाछ त्रिकटु के साथ`
   - `*शरद चावल नियम:* अगर चावल बन रहे हैं तो जीरा ज़रूर डालें` (only when selected items contain चावल)
   - `*शरद वर्जित:* इमली, लौंग, लहसुन, प्याज़, काली मिर्च और गर्म मसाले नहीं`
   - `*शरद अधिक उपयोग:* नारियल / खीर / पुदीना`
   - `*शरद कम उपयोग:* छोले, टिंडा, करेला, टमाटर, आलू, अरबी, सरसों, पपीता, सौंफ़, हरी मिर्च, लाल मिर्च, अदरक, सौंठ, सरसों का तेल, कढ़ी, दही, लस्सी, शहद`
   - `*शरद जल नियम:* चाँदी के ग्लास या मटके का जल दें`
   - `*शरद रस:* मीठा / कसैला / कड़वा`
12. शरद days enforce hard filtering for `इमली`, `लौंग`, `लहसुन`, `प्याज`, `प्याज़`, `काली मिर्च`, `गरम मसाला`, `गर्म मसाला`.
13. If पंचांग ऋतु is `हेमंत`, `हेमन्त`, `हेमंत ऋतु`, or `हेमन्त ऋतु`, the script uses:
   - `breakfast_hemant.json`
   - `menu_hemant.json` (if available; otherwise fallback meal file is `menu_shishir.json`)
14. For हेमंत days, output includes:
   - `*हेमंत पूर्णतया निषिद्ध:* बासमती, मैदा, डिब्बा बंद, मोठ, दोबारा गर्म की हुई दाल/सब्ज़ी, जीरा, इमली, सॉस, अचार, कड़वा, कसैला, रिफाइंड, पनीर, एनर्जी ड्रिंक, प्याज़, दुबारा गर्म किया पानी`
   - `*हेमंत जल नियम:* हमेशा गुनगुना, पीतल या तांबे में`
15. हेमंत days enforce hard filtering for the same prohibited keywords listed above.

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

## ऋतु auto-detect fallback

If a date is missing in `panchang_2026_27.json`, script now auto-detects ऋतु by date window:

- 15 Jan - 14 Mar: `शिशिर`
- 15 Mar - 14 May: `वसंत`
- 15 May - 14 Jul: `ग्रीष्म`
- 15 Jul - 14 Sep: `वर्षा`
- 15 Sep - 14 Nov: `शरद`
- 15 Nov - 14 Jan: `हेमंत`

## GitHub Action schedule

- Workflow file: `.github/workflows/daily-menu.yml`
- Runs daily at `00:00 UTC` (5:30 AM IST)
- Updates `daily_menu.txt` and `history.json`

## iPhone Shortcuts

Use `Get Contents of URL` with raw GitHub URL of `daily_menu.txt`, then send the fetched text through WhatsApp action.
