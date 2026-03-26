# Food Menu Daily WhatsApp System

## VPK Questionnaire Web App

The repository now also contains a self-contained VPK questionnaire application at `apps/vpk-assessment`.

### One-command setup

```bash
./scripts/setup-vpk.sh
```

Expected result: npm dependencies install, questionnaire validation passes, and the SQLite file is initialized under `apps/vpk-assessment/data/`.

### One-command run

```bash
./scripts/dev-vpk.sh
```

Expected result: the Next development server starts and prints a local URL such as `http://localhost:3000`.

### One-command tests

```bash
./scripts/test-vpk.sh
```

Expected result: lint and Vitest checks pass for the VPK module.

## Run locally

```bash
cd "/Users/gg/Documents/Ayurveda"
source .venv/bin/activate
python3 generate_menu.py
```

## Test with specific date

```bash
python3 generate_menu.py --date 2026-02-27
```

## Menu generator tests

```bash
python3 -m unittest discover -s tests
```

## Generate the 3 trigger menus for manual checking

```bash
python3 scripts/generate_trigger_test_menus.py
```

Expected result: three files are created in `test_outputs/menu_triggers/` for:
- `mangore`
- `pazhaya-sadam`
- `pakhala-bhata`

The script temporarily applies breakfast overrides, generates the next-day menus, saves the outputs, and then restores `config.json`, `history.json`, and `daily_menu.txt`.

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
- `*नियमित मेनू:* आज पर्व/विशेष पालन के कारण नियमित नाश्ता और भोजन मेनू नहीं दिया जाएगा।` (only on festival no-menu dates)
- `*विशेष अष्टमी मेनू:* ...` plus its numbered preparation lines (only when a festival row or recurring rule supplies `special_menu_lines_hi`)
- `*विशेष पारंपरिक सेवन/भोग:* <festival special note>` (only on festival no-menu dates)
- `*सुबह का नाश्ता:* <item>`
- `*आज का भोजन:* <item>`
- `*फॉलोवर महोदय हेतु रात की तैयारी:* <instruction>` (only when the generated next-day menu includes मंगौड़े)
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
      "date": "2026-03-19",
      "hindu_hi": ["चैत्र नवरात्रि"],
      "sikh_hi": [],
      "suppress_regular_menu": true,
      "special_menu_note_hi": "नवरात्रि दिवस 1, माँ शैलपुत्री: आज विशेष रूप से देसी घी ग्रहण करें या भोग में अर्पित करें।",
      "special_menu_lines_hi": [
        "*विशेष अष्टमी मेनू:* अष्टमी के दिन नवरात्रि का भोजन निम्नानुसार बनाया जाए:",
        "1. काले चने — 4 कटोरी।"
      ]
    }
  ]
}
```

For Chaitra Navratri 2026, the generator also has a built-in no-menu fallback for `2026-03-19` through `2026-03-27`, so those dates still suppress the regular menu even if a festival row is missing or incomplete.

If a festival row includes `special_menu_lines_hi`, those lines replace the generic `*नियमित मेनू:*` / `*विशेष पारंपरिक सेवन/भोग:*` block for that date.

The generator also applies a recurring override for any festival day that is both `नवरात्रि` and पंचांग `अष्टमी`: it outputs the fixed Ashtami menu (काले चने, छोले, तरी वाले आलू, पूरी, कद्दू, and सूजी हलवा instructions) and suppresses all regular seasonal menu selection for that day.

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

## Consecutive-day repeat rule

1. The generator looks at yesterday's `breakfast` and `meal` together from `history.json`.
2. It blocks only key repeats:
   - breakfast main items such as `पोहा`, `उपमा`, `चीला`, `डोसा`, `इडली`, or the key breakfast filling/base such as `आलू`, `मूंग`, `मेथी`
   - main sabzi-style meal items such as `करेला`, `लौकी`, `परवल`, `भिंडी`
3. Common bases and support ingredients such as `चावल`, `रोटी`, `दाल`, spices, and everyday cooking ingredients are not used for this rule by themselves.
4. Example: if today contains `करेला`, tomorrow avoids `करेला`, `भरवां करेला`, or mixed items like `करेला-भिंडी`.
5. If the seasonal pool becomes too small after this rule, the script falls back to the best available menu and adds a `*डेटा अलर्ट:*` note instead of failing.

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
