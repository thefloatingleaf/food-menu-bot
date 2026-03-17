import { cookies } from "next/headers";
import { NextResponse } from "next/server";

import { acknowledgeInstructions, getAuthenticatedAccount, resolveAttemptIdForAccount } from "@/lib/db";
import { ATTEMPT_COOKIE_NAME, AUTH_SESSION_COOKIE_NAME } from "@/lib/session";

export async function POST(request: Request) {
  try {
    const payload = (await request.json()) as { attemptId?: string; acknowledged?: boolean };
    const cookieStore = await cookies();
    const sessionToken = cookieStore.get(AUTH_SESSION_COOKIE_NAME)?.value;
    const account = sessionToken ? getAuthenticatedAccount(sessionToken) : null;
    const attemptId = account
      ? resolveAttemptIdForAccount(account.id, payload.attemptId ?? cookieStore.get(ATTEMPT_COOKIE_NAME)?.value ?? null)
      : null;

    if (!account) {
      return NextResponse.json({ error: "Login is required before starting an assessment." }, { status: 401 });
    }

    if (!attemptId || !payload.acknowledged) {
      return NextResponse.json({ error: "A valid acknowledgement is required." }, { status: 400 });
    }

    const acknowledgedAt = acknowledgeInstructions(attemptId);
    return NextResponse.json({ canStart: true, acknowledgedAt });
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Unable to record acknowledgement." },
      { status: 400 },
    );
  }
}
