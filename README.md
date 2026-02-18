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

## Output format

- `*तिथि:* YYYY-MM-DD`
- `*ऋतु:* <value>`
- `*माह:* <value>`
- `*तिथि (पंचांग):* <value>`
- `*सुबह का नाश्ता:* <item>`
- `*आज का भोजन:* <item>`
- `*एकादशी:* <name_hi>` (only on Ekadashi/Gauna dates)

## Data files

- `breakfast_shishir.json`
- `menu_shishir.json`
- `ekadashi_2026_27.json`
- `panchang_2026_27.json` (daily tithi/month/ritu mapping)
- `config.json`

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
