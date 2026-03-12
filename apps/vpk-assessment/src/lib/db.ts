import fs from "node:fs";
import path from "node:path";

import Database from "better-sqlite3";

import { questionnaireContent, scoringConfig } from "@/data/vpkQuestionnaire";
import { registerScoringCategories, scoreResponses } from "@/lib/scoring";
import type {
  AttemptSnapshot,
  AttemptStatus,
  QuestionPayload,
  ResultPayload,
  WizardStage,
} from "@/lib/types";
import { normalizeEmail, normalizePhone } from "@/lib/validation";

type DbInstance = Database.Database;

type AttemptRow = {
  id: string;
  status: AttemptStatus;
  instructions_acknowledged_at: string | null;
};

type IdentityPayload = {
  name: string;
  age: number;
  location: string;
  email: string;
  phone: string;
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

    CREATE TABLE IF NOT EXISTS users (
      id TEXT PRIMARY KEY,
      name TEXT NOT NULL,
      age INTEGER NOT NULL,
      location TEXT NOT NULL,
      email_original TEXT NOT NULL,
      email_normalized TEXT NOT NULL UNIQUE,
      phone_original TEXT NOT NULL,
      phone_normalized TEXT NOT NULL UNIQUE,
      created_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS assessment_attempts (
      id TEXT PRIMARY KEY,
      user_id TEXT NOT NULL,
      status TEXT NOT NULL,
      instructions_acknowledged_at TEXT,
      started_at TEXT,
      completed_at TEXT,
      created_at TEXT NOT NULL,
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

function nowIso() {
  return new Date().toISOString();
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

export function getAttemptSnapshot(attemptId: string): AttemptSnapshot | null {
  const db = getDb();
  const attempt = db
    .prepare(
      `SELECT id, status, instructions_acknowledged_at
       FROM assessment_attempts
       WHERE id = ?`,
    )
    .get(attemptId) as AttemptRow | undefined;

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
  };
}

export function createIdentityAttempt(payload: IdentityPayload) {
  const db = getDb();
  const emailNormalized = normalizeEmail(payload.email);
  const phoneNormalized = normalizePhone(payload.phone);

  const existing = db
    .prepare(
      `SELECT id
       FROM users
       WHERE email_normalized = ? OR phone_normalized = ?
       LIMIT 1`,
    )
    .get(emailNormalized, phoneNormalized) as { id: string } | undefined;

  if (existing) {
    return {
      duplicate: true as const,
      message:
        "This assessment has already been taken with the same email address or phone number.",
    };
  }

  const userId = crypto.randomUUID();
  const attemptId = crypto.randomUUID();
  const timestamp = nowIso();

  const transaction = db.transaction(() => {
    db.prepare(
      `INSERT INTO users (
        id, name, age, location, email_original, email_normalized, phone_original, phone_normalized, created_at
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
    ).run(
      userId,
      payload.name.trim(),
      payload.age,
      payload.location.trim(),
      payload.email.trim(),
      emailNormalized,
      payload.phone.trim(),
      phoneNormalized,
      timestamp,
    );

    db.prepare(
      `INSERT INTO assessment_attempts (
        id, user_id, status, instructions_acknowledged_at, started_at, completed_at, created_at
      ) VALUES (?, ?, 'identity_created', NULL, NULL, NULL, ?)`,
    ).run(attemptId, userId, timestamp);
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

function assertAssessmentAccess(attemptId: string) {
  const db = getDb();
  const attempt = db
    .prepare(
      `SELECT id, status, instructions_acknowledged_at
       FROM assessment_attempts
       WHERE id = ?`,
    )
    .get(attemptId) as AttemptRow | undefined;

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

export function getQuestionPayload(attemptId: string, index: number): QuestionPayload {
  const db = getDb();
  assertAssessmentAccess(attemptId);

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
  attemptId: string,
  categoryId: string,
  lifetimeOptionId: string,
  presentOptionId: string,
) {
  const db = getDb();
  const attempt = assertAssessmentAccess(attemptId);
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

export function completeAssessment(attemptId: string): ResultPayload {
  const db = getDb();
  assertAssessmentAccess(attemptId);

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
    throw new Error("All 40 categories must be answered before results can be shown.");
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

export function getResultByAttemptId(attemptId: string): ResultPayload | null {
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
