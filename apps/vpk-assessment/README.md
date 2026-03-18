# VPK Assessment App

## Prerequisites

- Node.js 20+
- npm 10+
- macOS local filesystem access for the SQLite database under `data/`

## One-command setup

```bash
npm run setup
```

Expected result: questionnaire validation passes, the SQLite database is initialized, and a local admin account is seeded if one does not already exist.

Default initial admin credentials unless overridden by environment variables:

- Username: `admin`
- Password: `admin1234`

Optional overrides:

- `VPK_INITIAL_ADMIN_USERNAME`
- `VPK_INITIAL_ADMIN_PASSWORD`
- `VPK_DB_PATH`

## One-command run

```bash
npm run dev
```

Expected result: the Next.js development server starts and prints a local URL such as `http://localhost:3000`.

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

1. Sign in with the admin account or a user account created by the admin.
2. Admin users can open `Manage Accounts` and create additional logins with passwords.
3. Regular user accounts receive one assessment attempt at a time, with a 6-hour test window starting from an eligible login.
4. Admin accounts can complete the assessment multiple times and can use `Allow Test Again` to reopen access for any regular account.
5. Assessment content is protected with disabled copy/select/context-menu/print behavior and visible watermarking.

## Privacy note

This is a web app, so browser-level hardening can discourage screenshots but cannot guarantee screenshot blocking at the operating-system level on macOS or iPhone browsers.

## Troubleshooting

- If login fails on first run, re-run `npm run setup` to ensure the seeded admin account exists.
- If the app needs a fresh local database, remove `data/vpk-assessment.sqlite*` only if you intentionally want to reset local assessment data.
- If the UI behaves oddly after schema changes, stop the dev server and run `npm run build` once to catch type or route issues.
