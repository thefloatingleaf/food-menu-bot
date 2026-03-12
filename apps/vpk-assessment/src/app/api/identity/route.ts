import { NextResponse } from "next/server";

import { createIdentityAttempt } from "@/lib/db";
import { ATTEMPT_COOKIE_NAME } from "@/lib/session";
import { validateIdentityPayload } from "@/lib/validation";

export async function POST(request: Request) {
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

  const created = createIdentityAttempt(parsed.data);
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
