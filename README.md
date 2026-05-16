# Food Menu Daily WhatsApp System

## VPK Questionnaire Web App

The repository contains a self-contained VPK questionnaire application at `apps/vpk-assessment`.

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

## Household Inventory App

The repository also contains a completely separate standalone Household Inventory application at `apps/household-inventory`.

### One-command setup

```bash
./scripts/setup-household-inventory.sh
```

Expected result: npm dependencies install, the inventory app is ready, and the shared household purchase ledger files are initialized if missing.

### One-command run

```bash
./scripts/dev-household-inventory.sh
```

Expected result: the standalone Next development server starts at `http://localhost:3001`.

### One-command tests

```bash
./scripts/test-household-inventory.sh
```

Expected result: lint and Vitest checks pass for the household inventory module.

### Amor Farm invoice import

```bash
./scripts/import-household-amor-farm.sh --milk-only /absolute/path/to/invoice.pdf
```

Expected result: the Amor Farm monthly PDF is parsed and its milk entries are added to the shared household ledger while repeated imports skip already saved rows.

### New purchase screenshots or pasted order data

```bash
./scripts/import-household-purchases.sh --dry-run --stdin
```

Expected result: pasted purchase text is parsed and previewed without saving. This is the safest first pass after OCR from screenshots.

```bash
./scripts/import-household-purchases.sh --stdin
```

Expected result: parsed purchase rows are saved into the shared household ledger, `analysis_snapshot.json` is refreshed, and unclear rows are left in the review queue.

### Recurring supply context

```bash
./scripts/add-household-supply-context.sh --stdin
```

Expected result: daily or standing household supply facts can be recorded separately from purchase entries, so analysis remains honest when some stock comes from a non-invoice source.

## Run locally

```bash
./scripts/generate-daily-menu.sh
```

The generator always identifies tomorrow in the configured timezone and builds the entire menu for that next date. It never generates today's menu.
The script also verifies that `daily_menu.txt` and `history.json` were both updated for that exact next date, so stale output fails fast.
Each publish run also maintains `published_menu_archive.json`, which is the inspectable ledger for what was published by date.

## Menu generator tests

```bash
python3 -m unittest discover -s tests
```

## Household purchase ledger

This repo maintains a shared household purchase ledger used by the standalone Household Inventory app. It is not attached to the VPK app.

### Storage files

- `data/household_purchases/purchase_ledger.json`
- `data/household_purchases/analysis_snapshot.json`

### Internal interface

- Open the standalone Household Inventory app at `http://localhost:3001`.
- The landing page is a decision-oriented dashboard for reorder timing, consumption pace, stock risk, and review signals.
- The complete transaction register remains available separately at `http://localhost:3001/purchase-log`.

### Supported purchase fields

- `date_of_purchase`
- `item_name`
- `category`
- `quantity_purchased`
- `unit_of_measurement`
- `price`
- `vendor_source`
- `expected_consumption_period`
- `actual_consumption_period`
- `remarks`

The internal importer can accept raw pasted purchase text from notes, bills, messages, or order history and will auto-detect category where possible. Uncertain rows are saved with `Needs Review` or `Unclear` labels instead of being discarded.

### One-command initialization

```bash
python3 household_purchase_ledger.py ensure
```

Expected result: the purchase ledger and analysis snapshot files exist and validate as empty structured records.

### One-command validation

```bash
python3 household_purchase_ledger.py validate
```

Expected result: the current purchase ledger passes schema checks with no output.

### One-command summary refresh

```bash
python3 household_purchase_ledger.py summarize
```

Expected result: `analysis_snapshot.json` is refreshed with item-wise consumption, reorder, spend, and possible-anomaly insights derived only from available data.

## Generate the 3 trigger menus for manual checking

```bash
python3 scripts/generate_trigger_test_menus.py
```

Expected result: three files are created in `test_outputs/menu_triggers/` for:
- `mangore`
- `pazhaya-sadam`
- `pakhala-bhata`

The script temporarily applies breakfast overrides, generates the requested target-date menus, saves the outputs, and then restores `config.json`, `history.json`, and `daily_menu.txt`.
It simulates the previous day internally so each generated file is still produced through the same tomorrow-only runtime path as production.

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
- `*आज का भोजन 1:* <item>` and `*आज का भोजन 2:* <item>` (only for the temporary 08-Apr-2026 through 14-Apr-2026 dual-meal window)
- `*आज का फल:* <item>` or `*आज का फल:* फल उपलब्ध नहीं है`
- `*फॉलोवर महोदय हेतु रात की तैयारी:* <instruction>` (only when the generated next-day menu includes मंगौड़े)
- `*साथ में:* मोटा चौकोर कटा प्याज` (only when the selected breakfast is `पखाला भात`)
- `*एकादशी:* <name_hi>` (only on Ekadashi/Gauna dates)
- `*भोजन के साथ अनिवार्य:* ...` (only when ऋतु is वसंत)

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
- `fruit_months.json`
- `lunar_calendar_2026_2027.json` (reference calendar: lunar months, sankranti, amavasya, purnima, ekadashi, partial daily tables)
- `config.json`

## Date-specific menu overrides

`config.json` supports date-specific item pinning when a particular day needs a fixed output:

- `breakfast_item_date_overrides`
- `meal_item_date_overrides`
- `second_meal_item_date_overrides`
- `fruit_item_date_overrides`

Each entry uses:

```json
{
  "date": "2026-04-09",
  "item": "सूजी की इडली"
}
```

Notes:
- `meal_item_date_overrides` sets the main `आज का भोजन` item.
- `second_meal_item_date_overrides` applies only on dates that already use the temporary dual-meal output.
- `fruit_item_date_overrides` pins the exact `आज का फल` text for that date.
- Meal overrides still validate against the active seasonal menu list, and the dual-meal rice guard still prevents both meals from containing rice on the same day.

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

The generator also has built-in Navratri fallback coverage for:

- `2026-03-19` through `2026-03-27` (`चैत्र नवरात्रि 2026`)
- `2026-10-11` through `2026-10-20` (`शारदीय नवरात्रि 2026`, ending with `विजयादशमी`)
- `2027-04-07` through `2027-04-15` (`चैत्र नवरात्रि 2027`)
- `2027-09-30` through `2027-10-09` (`शारदीय नवरात्रि 2027`, ending with `विजयादशमी`)
- `2028-03-27` through `2028-04-04` (`चैत्र नवरात्रि 2028`)
- `2028-09-19` through `2028-09-28` (`शारदीय नवरात्रि 2028`, ending with `विजयादशमी`)

These ranges still suppress the regular menu even if a festival row is missing or incomplete.

If a festival row includes `special_menu_lines_hi`, those lines replace the generic `*नियमित मेनू:*` / `*विशेष पारंपरिक सेवन/भोग:*` block for that date.

The generator also applies a recurring override for any festival day that is both `नवरात्रि` and पंचांग `अष्टमी`: it outputs the fixed Ashtami menu (काले चने, छोले, तरी वाले आलू, पूरी, कद्दू, and सूजी हलवा instructions) and suppresses all regular seasonal menu selection for that day.

## Weather flow (free)

1. Use manual override if date exists in `manual_weather_override.json`
2. Else fetch Open-Meteo forecast for configured coordinates
3. Else continue without weather filter (menu generation never fails)
4. Weather is used only for internal software rules and is never rendered in the final menu message.

## Seasonal menu behavior

1. If पंचांग ऋतु is `वसंत`, the script uses:
   - `breakfast_vasant.json`
   - `menu_vasant.json`
2. For वसंत days, output also includes:
   - `*भोजन के साथ अनिवार्य:* नीम की चटनी / पुदीना की चटनी / लहसुन की चटनी / तीखा अचार (खट्टा नहीं) / मसाला छाछ ... / मूंग दाल पापड़`
   - `*वसंत दशम-दिवस स्मरण:* नीम का घी बनाएं।` plus the 6-step recipe below it (only on the 10th day of वसंत ऋतु)
   - every fruit line is rendered as `*आज का फल:* ... (फल सुबह 6–10 में न लें)`
   - if the generated Vasant output contains prohibited items/behaviours, it also appends `❌ वर्जित (वसंत ऋतु में विशेष रूप से निषिद्ध):` with each actual conflict listed separately
3. For any वसंत भोजन that uses `रोटी` and does not use `चावल`, the grain is restricted to exactly one of:
   - `जौ (Barley) (केवल पुराना)`
   - `ज्वार (Sorghum) (केवल पुराना)`
   - `रागी (Finger Millet) (केवल पुराना)`
   - `गेहूँ (Wheat) (केवल पुराना)`
   - `चने और जौ (Barley) की रोटी (मिस्सी रोटी)`
4. In eligible `वसंत` roti meals, grain preference is weighted so `जौ` is suggested most, then `ज्वार`, then `रागी` and `मिस्सी रोटी`, while `गेहूँ` is the least recommended option.
5. In eligible `ग्रीष्म` roti meals, grain preference is weighted so `ज्वार` is suggested most, then `जौ`, then `झंगोरा`, while `पुराना गेहूँ` remains the least recommended option.
6. For `30-Apr-2026` through `05-May-2026`, if a वसंत meal uses `रोटी`, the grain is restricted temporarily to `रागी (Finger Millet) (केवल पुराना)` only.
7. From `09-May-2026` through `14-May-2026`, `चने के सत्तू की रोटी` is excluded.
8. From `10-May-2026` through `20-Jun-2026`, if a selected breakfast or भोजन contains `रोटी`, it is restricted to the date-specific atta schedule in `generate_menu.py`. On `19-Jun-2026` the rule prefers `जौ` and falls back to `ज्वार` only if no `जौ` roti option is available; on `20-Jun-2026` it prefers `ज्वार` and falls back to `रागी` on the same basis. The rendered output also adds `*आज का आटा:* ...` whenever such a scheduled-date menu includes roti.
9. For वसंत dal-based meals, only these dal options are used:
   - `मूँग`
   - `मसूर`
   - `अरहर`
   - `चने-लौकी की दाल`
10. In eligible वसंत dal meals, `मसूर`, `अरहर`, and `चने-लौकी की दाल` follow strict rotation across meals; `मूँग` is exempt and may still appear before that strict dal cycle is complete.
11. In `वसंत` and `ग्रीष्म`, whenever a selected breakfast or भोजन contains `दही`/`रायता`, output adds the short note `*दही रूप:* केवल लौकी/खीरे का रायता` only as a fallback. If the selected item already names a specific raita such as `लौकी का रायता`, that extra note is skipped for that item path.
12. Outside `हेमंत` and `शिशिर`, any exact breakfast or भोजन item that needs `दही`/`रायता` is blocked from repeating again within the same calendar year, using `published_menu_archive.json` as the annual memory.
13. Across `वसंत` and `ग्रीष्म`, `पझैया सादम` is enforced as an overnight breakfast at least once in every 7-day window when the menu is generated in time for night-before preparation. If the menu is generated on the same morning, the generator records a timing note instead of forcing an impossible overnight prep.
14. A date-specific safeguard also forces `पझैया सादम` at least once in the window `08-Apr-2026` through `12-Apr-2026`, again only when there is enough lead time for overnight preparation.
15. Any breakfast `चीला/चिल्ला` variant is blocked for the next 7 days after it appears, so no kind of चिल्ला is repeated more than once in a week. `मूंग दाल चिल्ला` is stricter and cannot repeat within a 14-day window.
16. Year-round except in `वर्षा`, `कढ़ी` with the active-season rice variant is enforced at least once in every 15-day window. This rule never overrides Ekadashi because rice remains disallowed there, and it is also skipped on rainy target dates because `कढ़ी` is not allowed on any rainy day. `शिशिर` and `हेमंत` use `बासमती चावल`, `ग्रीष्म` uses `शालि चावल`, and `शरद` uses `साठी चावल`.
17. Across `वसंत` and `ग्रीष्म`, a `छाछ की सब्ज़ी` meal paired with a rice variant is enforced at least once in every 7-day window.
16. `पझैया सादम` or `पखाला भात` and any `छाछ की सब्ज़ी` meal are never allowed on the same day; if both would otherwise be selected, the meal side is changed to a different valid option. The generator also does not use `छाछ की सब्ज़ी` as the previous-night rice-support meal for a next-day `पझैया सादम` or `पखाला भात` prep note.
17. For the target menu dates `08-Apr-2026` through `14-Apr-2026`, breakfast selection remains unchanged but the output includes two meal lines: `*आज का भोजन 1:* ...` and `*आज का भोजन 2:* ...`.
18. In that same temporary dual-meal window, rice is allowed in at most one of the two daily meal selections.
19. Otherwise, script uses Shishir files:
   - `breakfast_shishir.json`
   - `menu_shishir.json`
16. If पंचांग ऋतु is `ग्रीष्म` or `ग्रीष्म ऋतु`, the script uses:
   - `breakfast_grishm.json`
   - `menu_grishm.json`
17. For ग्रीष्म days, output also includes:
   - `*ग्रीष्म नाश्ता अनिवार्य साथ:* छाछ (काफ़ी पतली) / पुदीना की चटनी`
   - `*ग्रीष्म भोजन अनिवार्य साथ:* छाछ (काफ़ी पतली) / पुदीना की चटनी / खीरा और ककड़ी`
18. `breakfast_grishm.json` duplicate entries are deduplicated (first occurrence kept) before random selection.
19. If पंचांग ऋतु is `वर्षा` or `वर्षा ऋतु`, the script uses:
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

## Monthly fruit behavior

1. Every generated menu includes a `*आज का फल:*` line in the main message, including regular days, festival-only days, and शृंगधारा days.
2. Fruit choices come only from `fruit_months.json`, keyed by calendar month.
3. Fruit rotation is tracked persistently in `history.json` using the generated date and selected fruit.
4. Within the same calendar month, a fruit does not repeat until the other approved fruits for that month have been used at least once.
5. If the monthly fruit list is exhausted, the fruit cycle resets automatically for that same month and selection starts again.
6. In May and June, `आम` gets higher weight and may reappear before all other fruits are exhausted, but the selector still avoids unnecessary monotony such as immediate back-to-back repetition when other options are available.
7. If a month has no configured fruit list, or no valid fruit can be selected, the menu prints `*आज का फल:* फल उपलब्ध नहीं है`.

## Consecutive-day repeat rule

1. The generator looks at yesterday's `breakfast` and `meal` together from `history.json`.
2. It blocks only key repeats:
   - breakfast main items such as `पोहा`, `उपमा`, `चीला`, `डोसा`, `इडली`, or the key breakfast filling/base such as `आलू`, `मूंग`, `मेथी`
   - main sabzi-style meal items such as `करेला`, `लौकी`, `परवल`, `भिंडी`
3. Common bases and support ingredients such as `चावल`, `रोटी`, `दाल`, spices, and everyday cooking ingredients are not used for this rule by themselves.
4. Example: if today contains `करेला`, tomorrow avoids `करेला`, `भरवां करेला`, or mixed items like `करेला-भिंडी`.
5. If the seasonal pool becomes too small after this rule, the script falls back to the best available menu and adds a `*डेटा अलर्ट:*` note instead of failing.

## Seasonal variety cycle rule

1. Breakfast and भोजन now each maintain a separate variety cycle per active `ऋतु`.
2. Within the same `ऋतु`, an item is not repeated in that category until every other currently suitable option in that seasonal pool has been used once.
3. After the full eligible pool has been exhausted, the cycle resets automatically and selection starts a new round.
4. The existing consecutive-day family rule still applies on top of the variety cycle, so the script continues to avoid back-to-back repeats even right after a cycle reset.

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
- Scheduled backup windows run at `02:00`, `04:00`, `06:00`, `08:00`, and `09:00 UTC` (7:30 AM through 2:30 PM IST)
- The last backup run is intentionally before the `15:10 IST` iPhone Shortcut send time
- Pushes to the generator, workflow, or menu data files also trigger an immediate refresh, so a workflow edit cannot leave the published file stale
- Each run updates `daily_menu.txt` and `history.json`, then verifies that both match tomorrow's date before any commit is allowed

## iPhone Shortcuts

Use `Get Contents of URL` with raw GitHub URL of `daily_menu.txt`, then send the fetched text through WhatsApp action.
If WhatsApp sends the same date twice, first check whether the Shortcut ran before the latest `chore: update daily menu` commit was pushed.
