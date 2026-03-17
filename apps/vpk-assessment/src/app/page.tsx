import { cookies } from "next/headers";

import { AssessmentApp } from "@/components/AssessmentApp";
import {
  getAttemptSnapshot,
  getAuthenticatedAccount,
  getQuestionPayload,
  getResultByAttemptId,
  listAccounts,
  resolveAttemptIdForAccount,
} from "@/lib/db";
import { ATTEMPT_COOKIE_NAME, AUTH_SESSION_COOKIE_NAME } from "@/lib/session";

export default async function Home() {
  const cookieStore = await cookies();
  const sessionToken = cookieStore.get(AUTH_SESSION_COOKIE_NAME)?.value ?? null;
  const account = sessionToken ? getAuthenticatedAccount(sessionToken) : null;
  const requestedAttemptId = cookieStore.get(ATTEMPT_COOKIE_NAME)?.value ?? null;
  const attemptId = account ? resolveAttemptIdForAccount(account.id, requestedAttemptId) : null;
  const snapshot =
    account && attemptId ? getAttemptSnapshot(account.id, attemptId) : null;
  const question =
    account && snapshot?.attemptId && snapshot.stage === "assessment"
      ? getQuestionPayload(account.id, snapshot.attemptId, snapshot.questionIndex)
      : null;
  const result =
    account && snapshot?.attemptId && snapshot.stage === "result"
      ? getResultByAttemptId(account.id, snapshot.attemptId)
      : null;
  const accounts = account?.role === "admin" ? listAccounts() : [];

  return (
    <AssessmentApp
      initialAccount={account}
      initialAccounts={accounts}
      initialSnapshot={snapshot}
      initialQuestion={question}
      initialResult={result}
    />
  );
}
