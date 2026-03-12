import { describe, expect, it } from "vitest";

import { questionnaireContent } from "@/data/vpkQuestionnaire";
import { validateQuestionnaireContent } from "@/lib/content-validation";
import { deriveConstitutionLabel, registerScoringCategories, scoreResponses } from "@/lib/scoring";
import {
  deriveAgeFromDateOfBirth,
  normalizeEmail,
  normalizePhone,
  validateIdentityPayload,
} from "@/lib/validation";

registerScoringCategories(questionnaireContent.categories);

describe("content validation", () => {
  it("contains the full 40-category questionnaire", () => {
    expect(validateQuestionnaireContent()).toBe(true);
    expect(questionnaireContent.categories).toHaveLength(40);
  });
});

describe("identity normalization", () => {
  it("normalizes email addresses", () => {
    expect(normalizeEmail("  Example@Email.COM ")).toBe("example@email.com");
  });

  it("normalizes phone numbers", () => {
    expect(normalizePhone("IN", "98765 43210")).toBe("+91|9876543210");
    expect(normalizePhone("US", "(415) 555-1234")).toBe("+1|4155551234");
  });

  it("derives age from date of birth", () => {
    expect(deriveAgeFromDateOfBirth("2000-03-10", new Date("2026-03-12T00:00:00.000Z"))).toBe(26);
  });

  it("rejects invalid Indian numbers and malformed emails", () => {
    const parsed = validateIdentityPayload({
      firstName: "Asha",
      middleName: "",
      lastName: "Patel",
      dateOfBirth: "2001-02-31",
      location: "Mumbai",
      email: "abc@123",
      countryCode: "IN",
      localPhoneNumber: "12345",
    });

    expect(parsed.success).toBe(false);
  });

  it("accepts a valid structured identity payload", () => {
    const parsed = validateIdentityPayload({
      firstName: "Asha",
      middleName: "R",
      lastName: "Patel",
      dateOfBirth: "2001-02-28",
      location: "Mumbai",
      email: "asha.patel@example.com",
      countryCode: "IN",
      localPhoneNumber: "9876543210",
    });

    expect(parsed.success).toBe(true);
  });
});

describe("scoring", () => {
  it("supports dual constitutions when scores are close", () => {
    expect(deriveConstitutionLabel({ V: 15, P: 14, K: 11 }, 1)).toBe("V-P");
  });

  it("scores Lifetime and Present independently", () => {
    const responses = questionnaireContent.categories.map((category, index) => ({
      category_id: category.id,
      lifetime_option_id: index % 2 === 0 ? category.options[0].id : category.options[1].id,
      present_option_id: index % 2 === 0 ? category.options[2].id : category.options[1].id,
    }));

    const result = scoreResponses("attempt-1", responses, "2026-03-12T00:00:00.000Z");

    expect(result.lifetime.V).toBe(20);
    expect(result.lifetime.P).toBe(20);
    expect(result.lifetime.K).toBe(0);
    expect(result.present.V).toBe(0);
    expect(result.present.P).toBe(20);
    expect(result.present.K).toBe(20);
  });
});
