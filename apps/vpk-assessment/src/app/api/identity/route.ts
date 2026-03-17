import { NextResponse } from "next/server";

import { createIdentityAttempt, getAuthenticatedAccount } from "@/lib/db";
import { ATTEMPT_COOKIE_NAME, AUTH_SESSION_COOKIE_NAME } from "@/lib/session";
import { deriveAgeFromDateOfBirth, validateIdentityPayload } from "@/lib/validation";
import { cookies } from "next/headers";

export async function POST(request: Request) {
  const cookieStore = await cookies();
  const sessionToken = cookieStore.get(AUTH_SESSION_COOKIE_NAME)?.value;
  const account = sessionToken ? getAuthenticatedAccount(sessionToken) : null;

  if (!account) {
    return NextResponse.json({ error: "Login is required before starting an assessment." }, { status: 401 });
  }

  const payload = await request.json();
  const parsed = validateIdentityPayload(payload);

  if (!parsed.success) {
    const fieldErrors = parsed.error.flatten().fieldErrors;
    return NextResponse.json(
      {
        fieldErrors: Object.fromEntries(
          Object.entries(fieldErrors).map(([key, value]) => [key, value?.[0] ?? "Invalid value."]),
        ),
      },
      { status: 400 },
    );
  }

  const age = deriveAgeFromDateOfBirth(parsed.data.dateOfBirth);
  if (age === null) {
    return NextResponse.json(
      { fieldErrors: { dateOfBirth: "Enter a valid date of birth." } },
      { status: 400 },
    );
  }

  const created = createIdentityAttempt(account.id, {
    ...parsed.data,
    age,
  });
  if (created.duplicate) {
    return NextResponse.json(
      { duplicate: true, message: created.message },
      { status: 409 },
    );
  }

  const response = NextResponse.json(created);
  response.cookies.set({
    name: ATTEMPT_COOKIE_NAME,
    value: created.attemptId,
    httpOnly: true,
    sameSite: "lax",
    path: "/",
    secure: process.env.NODE_ENV === "production",
  });

  return response;
}
