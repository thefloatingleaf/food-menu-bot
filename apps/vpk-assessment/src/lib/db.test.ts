import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import { afterEach, describe, expect, it, vi } from "vitest";

type DbModule = typeof import("./db");

const identityPayload = {
  firstName: "Asha",
  middleName: "",
  lastName: "Patel",
  dateOfBirth: "1990-04-12",
  age: 35,
  location: "Pune",
  email: "asha.patel@example.com",
  countryCode: "IN",
  localPhoneNumber: "9876543210",
};

async function loadDbModule() {
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), "vpk-db-"));
  const dbPath = path.join(tempDir, "assessment.sqlite");

  process.env.VPK_DB_PATH = dbPath;
  vi.resetModules();
  delete (globalThis as { vpkDb?: unknown }).vpkDb;

  const dbModule = await import("./db");
  const dataModule = await import("@/data/vpkQuestionnaire");
  dbModule.initializeDatabase();

  return {
    tempDir,
    dbModule,
    questionnaireContent: dataModule.questionnaireContent,
  };
}

async function completeAttempt(
  dbModule: DbModule,
  accountId: string,
  attemptId: string,
  categoryIds: string[],
) {
  dbModule.acknowledgeInstructions(attemptId);

  for (const [index, categoryId] of categoryIds.entries()) {
    const question = dbModule.getQuestionPayload(accountId, attemptId, index + 1);
    expect(question.category.id).toBe(categoryId);
    const optionId = question.options[0].id;
    dbModule.saveResponse(accountId, attemptId, categoryId, optionId, optionId);
  }

  return dbModule.completeAssessment(accountId, attemptId);
}

afterEach(() => {
  vi.useRealTimers();
  delete process.env.VPK_DB_PATH;
  delete (globalThis as { vpkDb?: unknown }).vpkDb;
});

describe("assessment access rules", () => {
  it("blocks a regular account after the 6-hour test window expires", async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-03-18T04:00:00.000Z"));

    const { dbModule } = await loadDbModule();
    const account = dbModule.createManagedAccount({
      displayName: "Asha Patel",
      username: "asha",
      password: "password123",
    });

    dbModule.loginAccount("asha", "password123");
    const created = dbModule.createIdentityAttempt(account.id, identityPayload);
    expect(created.duplicate).toBe(false);

    dbModule.acknowledgeInstructions(created.attemptId);
    expect(dbModule.getQuestionPayload(account.id, created.attemptId, 1).index).toBe(1);

    vi.setSystemTime(new Date("2026-03-18T10:01:00.000Z"));

    expect(() => dbModule.getQuestionPayload(account.id, created.attemptId, 1)).toThrow(
      "The 6-hour test window has expired. Ask the admin to allow another attempt.",
    );
  });

  it("lets admin reopen a fresh attempt after a regular account has used its one chance", async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-03-18T04:00:00.000Z"));

    const { dbModule, questionnaireContent } = await loadDbModule();
    const account = dbModule.createManagedAccount({
      displayName: "Nina Rao",
      username: "nina",
      password: "password123",
    });

    dbModule.loginAccount("nina", "password123");
    const firstAttempt = dbModule.createIdentityAttempt(account.id, {
      ...identityPayload,
      email: "nina.rao@example.com",
      localPhoneNumber: "9123456780",
    });
    expect(firstAttempt.duplicate).toBe(false);

    await completeAttempt(
      dbModule,
      account.id,
      firstAttempt.attemptId,
      questionnaireContent.categories.map((category) => category.id),
    );

    expect(
      dbModule.createIdentityAttempt(account.id, {
        ...identityPayload,
        email: "nina.rao@example.com",
        localPhoneNumber: "9123456780",
      }),
    ).toMatchObject({
      duplicate: true,
      message: "This account has already used its available test. Ask the admin to allow another attempt.",
    });

    const reopenedAccount = dbModule.allowManagedAccountRetest(account.id);
    expect(reopenedAccount?.availableAttempts).toBe(1);
    expect(reopenedAccount?.accessWindowStartedAt).toBeNull();

    dbModule.loginAccount("nina", "password123");
    const secondAttempt = dbModule.createIdentityAttempt(account.id, {
      ...identityPayload,
      email: "nina.rao@example.com",
      localPhoneNumber: "9123456780",
    });
    expect(secondAttempt.duplicate).toBe(false);
    expect(secondAttempt.attemptId).not.toBe(firstAttempt.attemptId);
  });

  it("keeps admin accounts eligible for repeated fresh attempts", async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-03-18T04:00:00.000Z"));

    const { dbModule, questionnaireContent } = await loadDbModule();
    const adminSession = dbModule.loginAccount("admin", "admin1234");

    const firstAttempt = dbModule.createIdentityAttempt(adminSession.account.id, {
      ...identityPayload,
      email: "admin.one@example.com",
      localPhoneNumber: "9000000001",
    });
    expect(firstAttempt.duplicate).toBe(false);

    await completeAttempt(
      dbModule,
      adminSession.account.id,
      firstAttempt.attemptId,
      questionnaireContent.categories.map((category) => category.id),
    );

    const secondSession = dbModule.loginAccount("admin", "admin1234");
    expect(secondSession.attemptId).toBeNull();

    const secondAttempt = dbModule.createIdentityAttempt(secondSession.account.id, {
      ...identityPayload,
      email: "admin.two@example.com",
      localPhoneNumber: "9000000002",
    });
    expect(secondAttempt.duplicate).toBe(false);
    expect(secondAttempt.attemptId).not.toBe(firstAttempt.attemptId);
  });
});
