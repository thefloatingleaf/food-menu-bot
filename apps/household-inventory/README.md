# Household Inventory App

## Prerequisites

- Node.js 20+
- npm 10+

## One-command setup

```bash
npm run setup
```

Expected result: the shared household purchase ledger files are initialized if missing.

## One-command run

```bash
npm run dev
```

Expected result: the standalone Next.js app starts at `http://localhost:3001`.

## One-command tests

```bash
npm run lint && npm run test
```

Expected result: lint and Vitest checks pass.

## Production build check

```bash
npm run build
```

Expected result: the app compiles successfully for production.

## Current flow

1. Open the standalone app at `http://localhost:3001`.
2. Use the dashboard landing page to paste raw purchase records and review reorder, stock, confidence, and possible-anomaly signals.
3. Use `View Detailed Purchase Log` when you need the full transaction register at `http://localhost:3001/purchase-log`.
4. The app reads and writes the shared ledger at `../../data/household_purchases/`.

## New purchase screenshots or pasted order data

When new purchase details arrive as screenshots, first extract the visible text through Codex/OCR, then import the cleaned text through the same ledger normalizer used by the app.

```bash
npm run import:raw-purchases -- --dry-run --stdin
```

Expected result: the command previews parsed items, review flags, and projected ledger counts without saving anything.

```bash
npm run import:raw-purchases -- --stdin
```

Expected result: parsed purchase rows are appended to `../../data/household_purchases/purchase_ledger.json`, `analysis_snapshot.json` is regenerated, and uncertain details remain marked for review instead of being guessed.

The importer also accepts a text file:

```bash
npm run import:raw-purchases -- /absolute/path/to/purchases.txt
```

## Amor Farm PDF import

```bash
npm run import:amor-farm -- --milk-only /absolute/path/to/invoice.pdf
```

Expected result: the monthly vendor PDF is parsed and the milk line items are appended to the shared ledger without duplicating already imported rows.

## Recurring supply context

```bash
npm run add:supply-context -- --stdin
```

Expected result: a non-purchase recurring supply fact, such as daily milk coming from another source, is stored separately so the analysis can show it as context without pretending it was a normal paid invoice row.
