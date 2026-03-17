import { cookies } from "next/headers";
import { NextResponse } from "next/server";

import { completeAssessment, getAuthenticatedAccount, resolveAttemptIdForAccount } from "@/lib/db";
import { ATTEMPT_COOKIE_NAME, AUTH_SESSION_COOKIE_NAME } from "@/lib/session";

export async function POST(request: Request) {
  try {
    const payload = (await request.json()) as { attemptId?: string };
    const cookieStore = await cookies();
    const sessionToken = cookieStore.get(AUTH_SESSION_COOKIE_NAME)?.value;
    const account = sessionToken ? getAuthenticatedAccount(sessionToken) : null;
    const attemptId = account
      ? resolveAttemptIdForAccount(account.id, payload.attemptId ?? cookieStore.get(ATTEMPT_COOKIE_NAME)?.value ?? null)
      : null;

    if (!account) {
      return NextResponse.json({ error: "Login is required before completing an assessment." }, { status: 401 });
    }

    if (!attemptId) {
      return NextResponse.json({ error: "A valid assessment session is required." }, { status: 400 });
    }

    return NextResponse.json(completeAssessment(account.id, attemptId));
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Unable to complete assessment." },
      { status: 400 },
    );
  }
}
