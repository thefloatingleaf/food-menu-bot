import { scoringConfig } from "@/data/vpkQuestionnaire";
import type { DoshaKey, ResultPayload, TrackTotals } from "@/lib/types";

type PersistedResponse = {
  category_id: string;
  lifetime_option_id: string;
  present_option_id: string;
};

type ScoringCategory = {
  id: string;
  options: Array<{ id: string; doshaKey: DoshaKey }>;
};

const scoringCategories: ScoringCategory[] = [];

export function registerScoringCategories(categories: ScoringCategory[]) {
  scoringCategories.length = 0;
  scoringCategories.push(...categories);
}

function scoreTrack(
  responses: PersistedResponse[],
  optionField: "lifetime_option_id" | "present_option_id",
): TrackTotals {
  const totals: Record<DoshaKey, number> = { V: 0, P: 0, K: 0 };

  for (const response of responses) {
    const category = scoringCategories.find((item) => item.id === response.category_id);
    if (!category) {
      throw new Error(`Unknown category id: ${response.category_id}`);
    }

    const selectedOption = category.options.find((option) => option.id === response[optionField]);
    if (!selectedOption) {
      throw new Error(`Unknown option id: ${response[optionField]}`);
    }

    totals[selectedOption.doshaKey] += 1;
  }

  return {
    ...totals,
    constitutionLabel: deriveConstitutionLabel(totals),
  };
}

export function deriveConstitutionLabel(
  totals: Record<DoshaKey, number>,
  threshold = scoringConfig.mixedTypeThreshold,
): string {
  const ranking = Object.entries(totals)
    .map(([key, value]) => ({ key: key as DoshaKey, value }))
    .sort((left, right) => right.value - left.value);

  const [first, second] = ranking;
  if (!first || !second) {
    throw new Error("Unable to derive constitution label.");
  }

  const denominator = Math.max(first.value, 1);
  const relativeGap = (first.value - second.value) / denominator;
  if (relativeGap <= threshold) {
    return `${first.key}-${second.key}`;
  }

  return first.key;
}

export function scoreResponses(
  attemptId: string,
  responses: PersistedResponse[],
  completedAt: string,
): ResultPayload {
  const lifetime = scoreTrack(responses, "lifetime_option_id");
  const present = scoreTrack(responses, "present_option_id");

  return {
    attemptId,
    lifetime,
    present,
    charts: {
      lifetime: [
        { key: "V", value: lifetime.V },
        { key: "P", value: lifetime.P },
        { key: "K", value: lifetime.K },
      ],
      present: [
        { key: "V", value: present.V },
        { key: "P", value: present.P },
        { key: "K", value: present.K },
      ],
    },
    completedAt,
  };
}
