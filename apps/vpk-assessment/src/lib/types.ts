export type DoshaKey = "V" | "P" | "K";

export type QuestionnaireOption = {
  id: string;
  text: string;
  doshaKey: DoshaKey;
};

export type QuestionnaireCategory = {
  id: string;
  order: number;
  title: string;
  note?: string;
  options: [QuestionnaireOption, QuestionnaireOption, QuestionnaireOption];
};

export type QuestionnaireContent = {
  instructionText: string;
  categories: QuestionnaireCategory[];
};

export type AttemptStatus =
  | "identity_created"
  | "instructions_acknowledged"
  | "in_progress"
  | "completed"
  | "blocked_duplicate";

export type WizardStage =
  | "opening"
  | "identity"
  | "duplicate"
  | "instructions"
  | "start"
  | "assessment"
  | "result";

export type QuestionPayload = {
  index: number;
  total: number;
  category: {
    id: string;
    title: string;
    note?: string;
  };
  lifetimeLabel: string;
  presentLabel: string;
  options: Array<{
    id: string;
    text: string;
  }>;
  savedResponse: {
    lifetimeOptionId: string | null;
    presentOptionId: string | null;
  };
};

export type TrackTotals = {
  V: number;
  P: number;
  K: number;
  constitutionLabel: string;
};

export type ResultPayload = {
  attemptId: string;
  lifetime: TrackTotals;
  present: TrackTotals;
  charts: {
    lifetime: Array<{ key: DoshaKey; value: number }>;
    present: Array<{ key: DoshaKey; value: number }>;
  };
  completedAt: string;
};

export type AttemptSnapshot = {
  attemptId: string;
  status: AttemptStatus;
  stage: WizardStage;
  questionIndex: number;
  instructionsAcknowledgedAt: string | null;
};
