import { z } from "zod";

const MINIMUM_AGE = 1;
const MAXIMUM_AGE = 120;

export const identitySchema = z.object({
  name: z.string().trim().min(2, "Enter the present name.").max(120),
  age: z.coerce
    .number({
      error: "Enter a valid age.",
    })
    .int("Age must be a whole number.")
    .min(MINIMUM_AGE, "Age must be at least 1.")
    .max(MAXIMUM_AGE, "Age must be 120 or below."),
  location: z.string().trim().min(2, "Enter the present location.").max(160),
  email: z.email("Enter a valid email address."),
  phone: z
    .string()
    .trim()
    .min(7, "Enter a valid phone number.")
    .max(25, "Enter a valid phone number."),
});

export const responseSchema = z.object({
  attemptId: z.string().trim().optional(),
  lifetimeOptionId: z.string().trim().min(1),
  presentOptionId: z.string().trim().min(1),
});

export function normalizeEmail(email: string): string {
  return email.trim().toLowerCase();
}

export function normalizePhone(phone: string): string {
  const compact = phone.trim().replace(/[^\d+]/g, "");
  const normalized = compact.startsWith("+")
    ? `+${compact.slice(1).replace(/\D/g, "")}`
    : compact.replace(/\D/g, "");
  return normalized.replace(/^00/, "+");
}

export function validateIdentityPayload(payload: unknown) {
  return identitySchema.safeParse(payload);
}
