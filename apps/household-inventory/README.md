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
2. Use `Purchase Log` to paste raw purchase records.
3. Use `Consumption & Analysis` to review stock-duration, reorder, spend, and possible-anomaly insights.
4. The app reads and writes the shared ledger at `../../data/household_purchases/`.
