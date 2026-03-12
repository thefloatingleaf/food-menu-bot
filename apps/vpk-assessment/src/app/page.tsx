import { cookies } from "next/headers";

import { AssessmentApp } from "@/components/AssessmentApp";
import { getAttemptSnapshot, getQuestionPayload, getResultByAttemptId } from "@/lib/db";
import { ATTEMPT_COOKIE_NAME } from "@/lib/session";

export default async function Home() {
  const cookieStore = await cookies();
  const attemptId = cookieStore.get(ATTEMPT_COOKIE_NAME)?.value ?? null;
  const snapshot = attemptId ? getAttemptSnapshot(attemptId) : null;
  const question =
    snapshot?.attemptId && snapshot.stage === "assessment"
      ? getQuestionPayload(snapshot.attemptId, snapshot.questionIndex)
      : null;
  const result =
    snapshot?.attemptId && snapshot.stage === "result"
      ? getResultByAttemptId(snapshot.attemptId)
      : null;

  return (
    <AssessmentApp
      initialSnapshot={snapshot}
      initialQuestion={question}
      initialResult={result}
    />
  );
}
