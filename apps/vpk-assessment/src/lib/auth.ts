import { randomBytes, scryptSync, timingSafeEqual } from "node:crypto";

const PASSWORD_KEY_LENGTH = 64;

export function normalizeUsername(username: string) {
  return username.trim().toLowerCase();
}

export function hashPassword(password: string, salt = randomBytes(16).toString("hex")) {
  const derivedKey = scryptSync(password, salt, PASSWORD_KEY_LENGTH).toString("hex");
  return `${salt}:${derivedKey}`;
}

export function verifyPassword(password: string, storedValue: string) {
  const [salt, expectedHash] = storedValue.split(":");
  if (!salt || !expectedHash) {
    return false;
  }

  const actualHash = scryptSync(password, salt, PASSWORD_KEY_LENGTH).toString("hex");
  const expectedBuffer = Buffer.from(expectedHash, "hex");
  const actualBuffer = Buffer.from(actualHash, "hex");

  if (expectedBuffer.length !== actualBuffer.length) {
    return false;
  }

  return timingSafeEqual(expectedBuffer, actualBuffer);
}
