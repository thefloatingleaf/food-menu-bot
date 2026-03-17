import { cookies } from "next/headers";
import { NextResponse } from "next/server";

import { getAuthenticatedAccount, getQuestionPayload, resolveAttemptIdForAccount, saveResponse } from "@/lib/db";
import { ATTEMPT_COOKIE_NAME, AUTH_SESSION_COOKIE_NAME } from "@/lib/session";
import { responseSchema } from "@/lib/validation";

type Context = {
  params: Promise<{ item: string }>;
};

export async function GET(_: Request, context: Context) {
  try {
    const { item } = await context.params;
    const cookieStore = await cookies();
    const sessionToken = cookieStore.get(AUTH_SESSION_COOKIE_NAME)?.value;
    const account = sessionToken ? getAuthenticatedAccount(sessionToken) : null;
    const attemptId = account
      ? resolveAttemptIdForAccount(account.id, cookieStore.get(ATTEMPT_COOKIE_NAME)?.value ?? null)
      : null;
    const questionIndex = Number(item);

    if (!account) {
      return NextResponse.json({ error: "Login is required before viewing the assessment." }, { status: 401 });
    }

    if (!attemptId || Number.isNaN(questionIndex)) {
      return NextResponse.json({ error: "A valid assessment session is required." }, { status: 400 });
    }

    return NextResponse.json(getQuestionPayload(account.id, attemptId, questionIndex));
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Unable to load question." },
      { status: 400 },
    );
  }
}

export async function POST(request: Request, context: Context) {
  try {
    const payload = await request.json();
    const parsed = responseSchema.safeParse(payload);
    const { item } = await context.params;
    const cookieStore = await cookies();
    const sessionToken = cookieStore.get(AUTH_SESSION_COOKIE_NAME)?.value;
    const account = sessionToken ? getAuthenticatedAccount(sessionToken) : null;
    const attemptId =
      parsed.success && account
        ? resolveAttemptIdForAccount(
            account.id,
            parsed.data.attemptId ?? cookieStore.get(ATTEMPT_COOKIE_NAME)?.value ?? null,
          )
        : undefined;

    if (!account) {
      return NextResponse.json({ error: "Login is required before saving responses." }, { status: 401 });
    }

    if (!parsed.success || !attemptId) {
      return NextResponse.json({ error: "Save requests must include both selections." }, { status: 400 });
    }

    return NextResponse.json(
      saveResponse(account.id, attemptId, item, parsed.data.lifetimeOptionId, parsed.data.presentOptionId),
    );
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Unable to save response." },
      { status: 400 },
    );
  }
}
