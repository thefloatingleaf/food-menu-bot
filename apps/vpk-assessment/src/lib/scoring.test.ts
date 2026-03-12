import { describe, expect, it } from "vitest";

import { questionnaireContent } from "@/data/vpkQuestionnaire";
import { validateQuestionnaireContent } from "@/lib/content-validation";
import { deriveConstitutionLabel, registerScoringCategories, scoreResponses } from "@/lib/scoring";
import { normalizeEmail, normalizePhone } from "@/lib/validation";

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
    expect(normalizePhone("+91 98765 43210")).toBe("+919876543210");
    expect(normalizePhone("0091-98765-43210")).toBe("+919876543210");
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
