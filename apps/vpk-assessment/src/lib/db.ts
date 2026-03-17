import fs from "node:fs";
import path from "node:path";

import Database from "better-sqlite3";

import { questionnaireContent, scoringConfig } from "@/data/vpkQuestionnaire";
import { hashPassword, normalizeUsername, verifyPassword } from "@/lib/auth";
import { registerScoringCategories, scoreResponses } from "@/lib/scoring";
import type {
  AccountRole,
  AttemptSnapshot,
  AttemptStatus,
  AuthenticatedAccount,
  ManagedAccountSummary,
  QuestionPayload,
  ResultPayload,
  WizardStage,
} from "@/lib/types";
import { normalizeEmail, normalizePhone } from "@/lib/validation";

type DbInstance = Database.Database;

type AttemptRow = {
  id: string;
  account_id?: string | null;
  status: AttemptStatus;
  instructions_acknowledged_at: string | null;
  full_name?: string | null;
};

type AccountRow = {
  id: string;
  username: string;
  username_normalized: string;
  display_name: string;
  role: AccountRole;
  password_hash: string;
  created_at: string;
  updated_at: string;
  last_login_at: string | null;
};

type IdentityPayload = {
  firstName: string;
  middleName?: string;
  lastName: string;
  dateOfBirth: string;
  age: number;
  location: string;
  email: string;
  countryCode: string;
  localPhoneNumber: string;
};

const globalForDb = globalThis as unknown as { vpkDb?: DbInstance };
const dbFilePath =
  process.env.VPK_DB_PATH ?? path.join(process.cwd(), "data", "vpk-assessment.sqlite");

registerScoringCategories(questionnaireContent.categories);

function ensureDatabaseDirectory() {
  fs.mkdirSync(path.dirname(dbFilePath), { recursive: true });
}

function createSchema(db: DbInstance) {
  db.exec(`
    PRAGMA journal_mode = WAL;

    CREATE TABLE IF NOT EXISTS auth_accounts (
      id TEXT PRIMARY KEY,
      username TEXT NOT NULL UNIQUE,
      username_normalized TEXT NOT NULL UNIQUE,
      display_name TEXT NOT NULL,
      role TEXT NOT NULL,
      password_hash TEXT NOT NULL,
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL,
      last_login_at TEXT
    );

    CREATE TABLE IF NOT EXISTS auth_sessions (
      id TEXT PRIMARY KEY,
      account_id TEXT NOT NULL,
      created_at TEXT NOT NULL,
      expires_at TEXT NOT NULL,
      FOREIGN KEY(account_id) REFERENCES auth_accounts(id)
    );

    CREATE TABLE IF NOT EXISTS users (
      id TEXT PRIMARY KEY,
      first_name TEXT NOT NULL,
      middle_name TEXT,
      last_name TEXT NOT NULL,
      full_name TEXT NOT NULL,
      date_of_birth TEXT NOT NULL,
      age INTEGER NOT NULL,
      location TEXT NOT NULL,
      email_original TEXT NOT NULL,
      email_normalized TEXT NOT NULL UNIQUE,
      country_code TEXT NOT NULL,
      phone_local_number TEXT NOT NULL,
      phone_original TEXT NOT NULL,
      phone_normalized TEXT NOT NULL UNIQUE,
      created_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS assessment_attempts (
      id TEXT PRIMARY KEY,
      account_id TEXT,
      user_id TEXT NOT NULL,
      status TEXT NOT NULL,
      instructions_acknowledged_at TEXT,
      started_at TEXT,
      completed_at TEXT,
      created_at TEXT NOT NULL,
      FOREIGN KEY(account_id) REFERENCES auth_accounts(id),
      FOREIGN KEY(user_id) REFERENCES users(id)
    );

    CREATE TABLE IF NOT EXISTS responses (
      id TEXT PRIMARY KEY,
      attempt_id TEXT NOT NULL,
      category_id TEXT NOT NULL,
      lifetime_option_id TEXT NOT NULL,
      present_option_id TEXT NOT NULL,
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL,
      UNIQUE(attempt_id, category_id),
      FOREIGN KEY(attempt_id) REFERENCES assessment_attempts(id)
    );

    CREATE TABLE IF NOT EXISTS results (
      id TEXT PRIMARY KEY,
      attempt_id TEXT NOT NULL UNIQUE,
      lifetime_v_total INTEGER NOT NULL,
      lifetime_p_total INTEGER NOT NULL,
      lifetime_k_total INTEGER NOT NULL,
      present_v_total INTEGER NOT NULL,
      present_p_total INTEGER NOT NULL,
      present_k_total INTEGER NOT NULL,
      lifetime_type_label TEXT NOT NULL,
      present_type_label TEXT NOT NULL,
      mixed_type_threshold INTEGER NOT NULL,
      created_at TEXT NOT NULL,
      FOREIGN KEY(attempt_id) REFERENCES assessment_attempts(id)
    );
  `);

  const authAccountColumns = new Set(
    (db.prepare(`PRAGMA table_info(auth_accounts)`).all() as Array<{ name: string }>).map(
      (column) => column.name,
    ),
  );

  const optionalAuthAccountColumns = [
    ["username_normalized", "TEXT"],
    ["display_name", "TEXT"],
    ["role", "TEXT"],
    ["password_hash", "TEXT"],
    ["updated_at", "TEXT"],
    ["last_login_at", "TEXT"],
  ] as const;

  for (const [name, type] of optionalAuthAccountColumns) {
    if (!authAccountColumns.has(name)) {
      db.exec(`ALTER TABLE auth_accounts ADD COLUMN ${name} ${type};`);
    }
  }

  const userColumns = new Set(
    (db.prepare(`PRAGMA table_info(users)`).all() as Array<{ name: string }>).map(
      (column) => column.name,
    ),
  );

  const optionalUserColumns = [
    ["first_name", "TEXT"],
    ["middle_name", "TEXT"],
    ["last_name", "TEXT"],
    ["full_name", "TEXT"],
    ["date_of_birth", "TEXT"],
    ["country_code", "TEXT"],
    ["phone_local_number", "TEXT"],
  ] as const;

  for (const [name, type] of optionalUserColumns) {
    if (!userColumns.has(name)) {
      db.exec(`ALTER TABLE users ADD COLUMN ${name} ${type};`);
    }
  }

  const assessmentAttemptColumns = new Set(
    (db.prepare(`PRAGMA table_info(assessment_attempts)`).all() as Array<{ name: string }>).map(
      (column) => column.name,
    ),
  );

  if (!assessmentAttemptColumns.has("account_id")) {
    db.exec(`ALTER TABLE assessment_attempts ADD COLUMN account_id TEXT;`);
  }

  seedInitialAdmin(db);
}

function seedInitialAdmin(db: DbInstance) {
  const adminCount = db
    .prepare(`SELECT COUNT(*) AS count FROM auth_accounts WHERE role = 'admin'`)
    .get() as { count: number };

  if (adminCount.count > 0) {
    return;
  }

  const timestamp = nowIso();
  const username = process.env.VPK_INITIAL_ADMIN_USERNAME?.trim() || "admin";
  const password = process.env.VPK_INITIAL_ADMIN_PASSWORD?.trim() || "admin1234";

  db.prepare(
    `INSERT INTO auth_accounts (
      id, username, username_normalized, display_name, role, password_hash, created_at, updated_at, last_login_at
    ) VALUES (?, ?, ?, ?, 'admin', ?, ?, ?, NULL)`,
  ).run(
    crypto.randomUUID(),
    username,
    normalizeUsername(username),
    "Administrator",
    hashPassword(password),
    timestamp,
    timestamp,
  );
}

export function getDb() {
  if (!globalForDb.vpkDb) {
    ensureDatabaseDirectory();
    const db = new Database(dbFilePath);
    createSchema(db);
    globalForDb.vpkDb = db;
  }

  return globalForDb.vpkDb;
}

export function initializeDatabase() {
  return getDb();
}

function mapAccount(row: AccountRow): AuthenticatedAccount {
  return {
    id: row.id,
    username: row.username,
    displayName: row.display_name,
    role: row.role,
  };
}

function findAccountBySessionToken(sessionToken: string) {
  const db = getDb();
  const now = nowIso();
  db.prepare(`DELETE FROM auth_sessions WHERE expires_at <= ?`).run(now);

  return db
    .prepare(
      `SELECT
        accounts.id,
        accounts.username,
        accounts.username_normalized,
        accounts.display_name,
        accounts.role,
        accounts.password_hash,
        accounts.created_at,
        accounts.updated_at,
        accounts.last_login_at
       FROM auth_sessions AS sessions
       JOIN auth_accounts AS accounts ON accounts.id = sessions.account_id
       WHERE sessions.id = ?
         AND sessions.expires_at > ?`,
    )
    .get(sessionToken, now) as AccountRow | undefined;
}

export function getAuthenticatedAccount(sessionToken: string): AuthenticatedAccount | null {
  const row = findAccountBySessionToken(sessionToken);
  return row ? mapAccount(row) : null;
}

export function loginAccount(username: string, password: string) {
  const db = getDb();
  const normalizedUsername = normalizeUsername(username);
  const account = db
    .prepare(
      `SELECT
        id, username, username_normalized, display_name, role, password_hash, created_at, updated_at, last_login_at
       FROM auth_accounts
       WHERE username_normalized = ?`,
    )
    .get(normalizedUsername) as AccountRow | undefined;

  if (!account || !verifyPassword(password, account.password_hash)) {
    throw new Error("Invalid username or password.");
  }

  const sessionToken = crypto.randomUUID();
  const createdAt = nowIso();
  const expiresAt = addMonths(new Date(createdAt), 1).toISOString();

  const transaction = db.transaction(() => {
    db.prepare(
      `INSERT INTO auth_sessions (id, account_id, created_at, expires_at)
       VALUES (?, ?, ?, ?)`,
    ).run(sessionToken, account.id, createdAt, expiresAt);

    db.prepare(
      `UPDATE auth_accounts
       SET last_login_at = ?, updated_at = ?
       WHERE id = ?`,
    ).run(createdAt, createdAt, account.id);
  });

  transaction();

  return {
    sessionToken,
    expiresAt,
    account: mapAccount({ ...account, last_login_at: createdAt, updated_at: createdAt }),
    attemptId: getLatestAttemptIdForAccount(account.id),
  };
}

export function logoutAccount(sessionToken: string) {
  const db = getDb();
  db.prepare(`DELETE FROM auth_sessions WHERE id = ?`).run(sessionToken);
}

export function listAccounts(): ManagedAccountSummary[] {
  const db = getDb();
  const rows = db
    .prepare(
      `SELECT
        id, username, username_normalized, display_name, role, password_hash, created_at, updated_at, last_login_at
       FROM auth_accounts
       ORDER BY role DESC, created_at ASC`,
    )
    .all() as AccountRow[];

  return rows.map((row) => ({
    ...mapAccount(row),
    createdAt: row.created_at,
    updatedAt: row.updated_at,
    lastLoginAt: row.last_login_at,
  }));
}

export function createManagedAccount(payload: { displayName: string; username: string; password: string }) {
  const db = getDb();
  const normalizedUsername = normalizeUsername(payload.username);
  const existing = db
    .prepare(`SELECT id FROM auth_accounts WHERE username_normalized = ?`)
    .get(normalizedUsername) as { id: string } | undefined;

  if (existing) {
    throw new Error("That username is already in use.");
  }

  const timestamp = nowIso();
  const id = crypto.randomUUID();

  db.prepare(
    `INSERT INTO auth_accounts (
      id, username, username_normalized, display_name, role, password_hash, created_at, updated_at, last_login_at
    ) VALUES (?, ?, ?, ?, 'user', ?, ?, ?, NULL)`,
  ).run(
    id,
    payload.username.trim(),
    normalizedUsername,
    payload.displayName.trim(),
    hashPassword(payload.password),
    timestamp,
    timestamp,
  );

  return {
    id,
    username: payload.username.trim(),
    displayName: payload.displayName.trim(),
    role: "user" as const,
    createdAt: timestamp,
    updatedAt: timestamp,
    lastLoginAt: null,
  };
}

function nowIso() {
  return new Date().toISOString();
}

function addMonths(input: Date, months: number) {
  const output = new Date(input);
  output.setMonth(output.getMonth() + months);
  return output;
}

function deriveStage(row: AttemptRow, answeredCount: number): WizardStage {
  if (row.status === "completed") {
    return "result";
  }
  if (!row.instructions_acknowledged_at) {
    return "instructions";
  }
  if (answeredCount > 0) {
    return "assessment";
  }
  return "start";
}

function getAnsweredCount(attemptId: string) {
  const db = getDb();
  const row = db
    .prepare(`SELECT COUNT(*) as count FROM responses WHERE attempt_id = ?`)
    .get(attemptId) as { count: number };
  return row.count;
}

function getLatestAttemptIdForAccount(accountId: string) {
  const db = getDb();
  const row = db
    .prepare(
      `SELECT id
       FROM assessment_attempts
       WHERE account_id = ?
       ORDER BY created_at DESC
       LIMIT 1`,
    )
    .get(accountId) as { id: string } | undefined;

  return row?.id ?? null;
}

export function resolveAttemptIdForAccount(accountId: string, requestedAttemptId: string | null) {
  const db = getDb();

  if (requestedAttemptId) {
    const ownedAttempt = db
      .prepare(`SELECT id FROM assessment_attempts WHERE id = ? AND account_id = ?`)
      .get(requestedAttemptId, accountId) as { id: string } | undefined;

    if (ownedAttempt) {
      return ownedAttempt.id;
    }
  }

  return getLatestAttemptIdForAccount(accountId);
}

export function getAttemptSnapshot(accountId: string, attemptId: string): AttemptSnapshot | null {
  const db = getDb();
  const attempt = db
    .prepare(
      `SELECT attempts.id, attempts.account_id, attempts.status, attempts.instructions_acknowledged_at, users.full_name
       FROM assessment_attempts AS attempts
       JOIN users ON users.id = attempts.user_id
       WHERE attempts.id = ?
         AND attempts.account_id = ?`,
    )
    .get(attemptId, accountId) as AttemptRow | undefined;

  if (!attempt) {
    return null;
  }

  const answeredCount = getAnsweredCount(attemptId);

  return {
    attemptId: attempt.id,
    status: attempt.status,
    stage: deriveStage(attempt, answeredCount),
    questionIndex: Math.min(answeredCount + 1, questionnaireContent.categories.length),
    instructionsAcknowledgedAt: attempt.instructions_acknowledged_at,
    registrantName: attempt.full_name ?? null,
  };
}

export function createIdentityAttempt(accountId: string, payload: IdentityPayload) {
  const db = getDb();
  const emailNormalized = normalizeEmail(payload.email);
  const phoneNormalized = normalizePhone(payload.countryCode, payload.localPhoneNumber);
  const fullName = [payload.firstName, payload.middleName?.trim(), payload.lastName]
    .filter(Boolean)
    .join(" ");
  const latestAccountAttempt = db
    .prepare(
      `SELECT status, completed_at
       FROM assessment_attempts
       WHERE account_id = ?
       ORDER BY created_at DESC
       LIMIT 1`,
    )
    .get(accountId) as { status: AttemptStatus; completed_at: string | null } | undefined;

  if (latestAccountAttempt && latestAccountAttempt.status !== "completed") {
    return {
      duplicate: true as const,
      message: "An assessment for this account is already in progress. Continue the existing session.",
    };
  }

  const existingUsers = db
    .prepare(
      `SELECT id
       FROM users
       WHERE email_normalized = ? OR phone_normalized = ?`,
    )
    .all(emailNormalized, phoneNormalized) as Array<{ id: string }>;

  const distinctExistingUserIds = [...new Set(existingUsers.map((item) => item.id))];
  if (distinctExistingUserIds.length > 1) {
    return {
      duplicate: true as const,
      message:
        "This email and phone number are linked to different records. Please contact support to continue safely.",
    };
  }

  const now = new Date();
  const timestamp = now.toISOString();
  const existingUserId = distinctExistingUserIds[0] ?? null;

  if (existingUserId) {
    const latestAttempt = db
      .prepare(
        `SELECT status, completed_at
         FROM assessment_attempts
         WHERE user_id = ?
         ORDER BY created_at DESC
         LIMIT 1`,
      )
      .get(existingUserId) as { status: AttemptStatus; completed_at: string | null } | undefined;

    if (latestAttempt && latestAttempt.status !== "completed") {
      return {
        duplicate: true as const,
        message:
          "An assessment for this identity is already in progress. Continue the existing session.",
      };
    }

    if (latestAttempt?.completed_at) {
      const eligibleAt = addMonths(new Date(latestAttempt.completed_at), 6);
      if (now < eligibleAt) {
        return {
          duplicate: true as const,
          message: `Next VPK retest is available on ${eligibleAt.toLocaleDateString("en-CA")}.`,
        };
      }
    }
  }

  const userId = existingUserId ?? crypto.randomUUID();
  const attemptId = crypto.randomUUID();

  const transaction = db.transaction(() => {
    if (!existingUserId) {
      db.prepare(
        `INSERT INTO users (
          id, first_name, middle_name, last_name, full_name, date_of_birth, age, location,
          email_original, email_normalized, country_code, phone_local_number, phone_original,
          phone_normalized, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      ).run(
        userId,
        payload.firstName.trim(),
        payload.middleName?.trim() || null,
        payload.lastName.trim(),
        fullName,
        payload.dateOfBirth,
        payload.age,
        payload.location.trim(),
        payload.email.trim(),
        emailNormalized,
        payload.countryCode,
        payload.localPhoneNumber.replace(/\D/g, ""),
        `${payload.countryCode} ${payload.localPhoneNumber.trim()}`,
        phoneNormalized,
        timestamp,
      );
    }

    db.prepare(
      `INSERT INTO assessment_attempts (
        id, account_id, user_id, status, instructions_acknowledged_at, started_at, completed_at, created_at
      ) VALUES (?, ?, ?, 'identity_created', NULL, NULL, NULL, ?)`,
    ).run(attemptId, accountId, userId, timestamp);
  });

  transaction();

  return {
    duplicate: false as const,
    attemptId,
    stage: "instructions" as const,
  };
}

export function acknowledgeInstructions(attemptId: string) {
  const db = getDb();
  const acknowledgedAt = nowIso();
  const updated = db
    .prepare(
      `UPDATE assessment_attempts
       SET status = 'instructions_acknowledged',
           instructions_acknowledged_at = ?
       WHERE id = ?
         AND status IN ('identity_created', 'instructions_acknowledged')`,
    )
    .run(acknowledgedAt, attemptId);

  if (updated.changes === 0) {
    throw new Error("The instructions acknowledgement could not be recorded.");
  }

  return acknowledgedAt;
}

function getOwnedAttempt(accountId: string, attemptId: string) {
  const db = getDb();
  return db
    .prepare(
      `SELECT id, account_id, status, instructions_acknowledged_at
       FROM assessment_attempts
       WHERE id = ?
         AND account_id = ?`,
    )
    .get(attemptId, accountId) as AttemptRow | undefined;
}

function assertAssessmentAccess(accountId: string, attemptId: string) {
  const attempt = getOwnedAttempt(accountId, attemptId);

  if (!attempt) {
    throw new Error("Assessment attempt not found.");
  }

  if (!attempt.instructions_acknowledged_at) {
    throw new Error("Instructions must be acknowledged before the assessment can begin.");
  }

  if (attempt.status === "completed") {
    throw new Error("This assessment has already been completed.");
  }

  return attempt;
}

export function getQuestionPayload(accountId: string, attemptId: string, index: number): QuestionPayload {
  const db = getDb();
  assertAssessmentAccess(accountId, attemptId);
  const answeredCount = getAnsweredCount(attemptId);
  const maximumVisibleIndex = Math.min(answeredCount + 1, questionnaireContent.categories.length);

  if (index < 1 || index > maximumVisibleIndex) {
    throw new Error("That question is not available yet.");
  }

  const category = questionnaireContent.categories[index - 1];
  if (!category) {
    throw new Error("Question index out of range.");
  }

  const saved = db
    .prepare(
      `SELECT lifetime_option_id, present_option_id
       FROM responses
       WHERE attempt_id = ? AND category_id = ?`,
    )
    .get(attemptId, category.id) as
    | { lifetime_option_id: string; present_option_id: string }
    | undefined;

  return {
    index,
    total: questionnaireContent.categories.length,
    category: {
      id: category.id,
      title: category.title,
      note: category.note,
    },
    lifetimeLabel: "Lifetime (Prakriti)",
    presentLabel: "Present (Vikriti)",
    options: category.options.map((option) => ({ id: option.id, text: option.text })),
    savedResponse: {
      lifetimeOptionId: saved?.lifetime_option_id ?? null,
      presentOptionId: saved?.present_option_id ?? null,
    },
  };
}

export function saveResponse(
  accountId: string,
  attemptId: string,
  categoryId: string,
  lifetimeOptionId: string,
  presentOptionId: string,
) {
  const db = getDb();
  const attempt = assertAssessmentAccess(accountId, attemptId);
  const category = questionnaireContent.categories.find((item) => item.id === categoryId);

  if (!category) {
    throw new Error("Unknown category.");
  }

  const validOptionIds = new Set(category.options.map((option) => option.id));
  if (!validOptionIds.has(lifetimeOptionId) || !validOptionIds.has(presentOptionId)) {
    throw new Error("Selected options do not belong to this category.");
  }

  const timestamp = nowIso();
  db.prepare(
    `INSERT INTO responses (
      id, attempt_id, category_id, lifetime_option_id, present_option_id, created_at, updated_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(attempt_id, category_id)
    DO UPDATE SET
      lifetime_option_id = excluded.lifetime_option_id,
      present_option_id = excluded.present_option_id,
      updated_at = excluded.updated_at`,
  ).run(
    crypto.randomUUID(),
    attemptId,
    categoryId,
    lifetimeOptionId,
    presentOptionId,
    timestamp,
    timestamp,
  );

  if (attempt.status !== "in_progress") {
    db.prepare(
      `UPDATE assessment_attempts
       SET status = 'in_progress',
           started_at = COALESCE(started_at, ?)
       WHERE id = ?`,
    ).run(timestamp, attemptId);
  }

  const answeredCount = getAnsweredCount(attemptId);

  return {
    completionReady: answeredCount === questionnaireContent.categories.length,
    nextIndex: Math.min(questionnaireContent.categories.length, category.order + 1),
  };
}

export function completeAssessment(accountId: string, attemptId: string): ResultPayload {
  const db = getDb();
  assertAssessmentAccess(accountId, attemptId);

  const responses = db
    .prepare(
      `SELECT category_id, lifetime_option_id, present_option_id
       FROM responses
       WHERE attempt_id = ?
       ORDER BY created_at ASC`,
    )
    .all(attemptId) as Array<{
      category_id: string;
      lifetime_option_id: string;
      present_option_id: string;
    }>;

  if (responses.length !== questionnaireContent.categories.length) {
    throw new Error(`All ${questionnaireContent.categories.length} categories must be answered before results can be shown.`);
  }

  const completedAt = nowIso();
  const result = scoreResponses(attemptId, responses, completedAt);

  const transaction = db.transaction(() => {
    db.prepare(
      `UPDATE assessment_attempts
       SET status = 'completed',
           completed_at = ?
       WHERE id = ?`,
    ).run(completedAt, attemptId);

    db.prepare(
      `INSERT INTO results (
        id, attempt_id, lifetime_v_total, lifetime_p_total, lifetime_k_total,
        present_v_total, present_p_total, present_k_total,
        lifetime_type_label, present_type_label, mixed_type_threshold, created_at
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
      ON CONFLICT(attempt_id) DO UPDATE SET
        lifetime_v_total = excluded.lifetime_v_total,
        lifetime_p_total = excluded.lifetime_p_total,
        lifetime_k_total = excluded.lifetime_k_total,
        present_v_total = excluded.present_v_total,
        present_p_total = excluded.present_p_total,
        present_k_total = excluded.present_k_total,
        lifetime_type_label = excluded.lifetime_type_label,
        present_type_label = excluded.present_type_label,
        mixed_type_threshold = excluded.mixed_type_threshold,
        created_at = excluded.created_at`,
    ).run(
      crypto.randomUUID(),
      attemptId,
      result.lifetime.V,
      result.lifetime.P,
      result.lifetime.K,
      result.present.V,
      result.present.P,
      result.present.K,
      result.lifetime.constitutionLabel,
      result.present.constitutionLabel,
      scoringConfig.mixedTypeThreshold,
      completedAt,
    );
  });

  transaction();
  return result;
}

export function getResultByAttemptId(accountId: string, attemptId: string): ResultPayload | null {
  if (!getOwnedAttempt(accountId, attemptId)) {
    throw new Error("Assessment attempt not found.");
  }
  const db = getDb();
  const row = db
    .prepare(
      `SELECT
        lifetime_v_total, lifetime_p_total, lifetime_k_total,
        present_v_total, present_p_total, present_k_total,
        lifetime_type_label, present_type_label,
        created_at
       FROM results
       WHERE attempt_id = ?`,
    )
    .get(attemptId) as
    | {
        lifetime_v_total: number;
        lifetime_p_total: number;
        lifetime_k_total: number;
        present_v_total: number;
        present_p_total: number;
        present_k_total: number;
        lifetime_type_label: string;
        present_type_label: string;
        created_at: string;
      }
    | undefined;

  if (!row) {
    return null;
  }

  return {
    attemptId,
    lifetime: {
      V: row.lifetime_v_total,
      P: row.lifetime_p_total,
      K: row.lifetime_k_total,
      constitutionLabel: row.lifetime_type_label,
    },
    present: {
      V: row.present_v_total,
      P: row.present_p_total,
      K: row.present_k_total,
      constitutionLabel: row.present_type_label,
    },
    charts: {
      lifetime: [
        { key: "V", value: row.lifetime_v_total },
        { key: "P", value: row.lifetime_p_total },
        { key: "K", value: row.lifetime_k_total },
      ],
      present: [
        { key: "V", value: row.present_v_total },
        { key: "P", value: row.present_p_total },
        { key: "K", value: row.present_k_total },
      ],
    },
    completedAt: row.created_at,
  };
}
