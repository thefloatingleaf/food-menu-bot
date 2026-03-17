import { questionnaireContent } from "@/data/vpkQuestionnaire";

export function validateQuestionnaireContent() {
  if (questionnaireContent.categories.length < 40) {
    throw new Error(`Expected at least 40 categories, found ${questionnaireContent.categories.length}.`);
  }

  const categoryIds = new Set<string>();
  const optionIds = new Set<string>();

  questionnaireContent.categories.forEach((category, index) => {
    if (category.order !== index + 1) {
      throw new Error(`Category order mismatch for ${category.id}.`);
    }

    if (categoryIds.has(category.id)) {
      throw new Error(`Duplicate category id ${category.id}.`);
    }
    categoryIds.add(category.id);

    category.options.forEach((option) => {
      if (optionIds.has(option.id)) {
        throw new Error(`Duplicate option id ${option.id}.`);
      }
      optionIds.add(option.id);
      if (!option.text.trim()) {
        throw new Error(`Option ${option.id} has empty text.`);
      }
    });
  });

  return true;
}
