import { cookies } from "next/headers";
import { NextResponse } from "next/server";

import { getAuthenticatedAccount, getResultByAttemptId } from "@/lib/db";
import { AUTH_SESSION_COOKIE_NAME } from "@/lib/session";

type Context = {
  params: Promise<{ attemptId: string }>;
};

export async function GET(_: Request, context: Context) {
  const cookieStore = await cookies();
  const sessionToken = cookieStore.get(AUTH_SESSION_COOKIE_NAME)?.value;
  const account = sessionToken ? getAuthenticatedAccount(sessionToken) : null;
  if (!account) {
    return NextResponse.json({ error: "Login is required before viewing results." }, { status: 401 });
  }

  const { attemptId } = await context.params;
  const result = getResultByAttemptId(account.id, attemptId);

  if (!result) {
    return NextResponse.json({ error: "Result not found." }, { status: 404 });
  }

  return NextResponse.json(result);
}
