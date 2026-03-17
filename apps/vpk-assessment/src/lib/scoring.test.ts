import { describe, expect, it } from "vitest";

import { questionnaireContent } from "@/data/vpkQuestionnaire";
import { validateQuestionnaireContent } from "@/lib/content-validation";
import { deriveConstitutionLabel, registerScoringCategories, scoreResponses } from "@/lib/scoring";
import {
  validateAccountCreationPayload,
  deriveAgeFromDateOfBirth,
  normalizeEmail,
  normalizePhone,
  validateIdentityPayload,
  validateLoginPayload,
} from "@/lib/validation";

registerScoringCategories(questionnaireContent.categories);

describe("content validation", () => {
  it("contains the expanded questionnaire", () => {
    expect(validateQuestionnaireContent()).toBe(true);
    expect(questionnaireContent.categories).toHaveLength(58);
  });

  it("covers the added constitution guideline areas", () => {
    const titles = questionnaireContent.categories.map((category) => category.title);

    expect(titles).toEqual(
      expect.arrayContaining([
        "Nose",
        "Eyes",
        "Lips",
        "Chin",
        "Cheeks",
        "Neck",
        "Chest",
        "Belly",
        "Belly Button",
        "Hips",
        "Joints",
        "Taste Preference",
        "Thirst",
        "Faith",
        "Intellect",
        "Recollection",
        "Dreams",
        "Financial Capacity",
      ]),
    );
  });

  it("keeps the expanded questionnaire in a body-to-mind flow", () => {
    expect(questionnaireContent.categories.slice(0, 6).map((category) => category.id)).toEqual([
      "body-frame",
      "body-weight",
      "face-shape",
      "nose",
      "eyes",
      "teeth",
    ]);

    expect(questionnaireContent.categories.slice(-4).map((category) => category.id)).toEqual([
      "life-goals",
      "neurotic-tendencies",
      "dreams",
      "sleep",
    ]);
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

describe("account validation", () => {
  it("accepts valid login credentials", () => {
    const parsed = validateLoginPayload({
      username: "asha.user",
      password: "SecurePass123",
    });

    expect(parsed.success).toBe(true);
  });

  it("rejects short account passwords", () => {
    const parsed = validateAccountCreationPayload({
      displayName: "Asha Patel",
      username: "asha.user",
      password: "short",
    });

    expect(parsed.success).toBe(false);
  });
});

describe("scoring", () => {
  it("supports dual constitutions when top two are within 20 percent", () => {
    expect(deriveConstitutionLabel({ V: 20, P: 17, K: 3 }, 0.2)).toBe("V-P");
  });

  it("returns a single constitution when top gap is above 20 percent", () => {
    expect(deriveConstitutionLabel({ V: 20, P: 15, K: 5 }, 0.2)).toBe("V");
  });

  it("scores Lifetime and Present independently", () => {
    const responses = questionnaireContent.categories.map((category, index) => ({
      category_id: category.id,
      lifetime_option_id: index % 2 === 0 ? category.options[0].id : category.options[1].id,
      present_option_id: index % 2 === 0 ? category.options[2].id : category.options[1].id,
    }));

    const result = scoreResponses("attempt-1", responses, "2026-03-12T00:00:00.000Z");

    expect(result.lifetime.V).toBe(29);
    expect(result.lifetime.P).toBe(29);
    expect(result.lifetime.K).toBe(0);
    expect(result.present.V).toBe(0);
    expect(result.present.P).toBe(29);
    expect(result.present.K).toBe(29);
  });
});
