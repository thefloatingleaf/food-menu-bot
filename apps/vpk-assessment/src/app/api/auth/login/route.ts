import { NextResponse } from "next/server";

import { getAuthenticatedAccount, loginAccount } from "@/lib/db";
import { ATTEMPT_COOKIE_NAME, AUTH_SESSION_COOKIE_NAME } from "@/lib/session";
import { validateLoginPayload } from "@/lib/validation";

export async function POST(request: Request) {
  try {
    const payload = await request.json();
    const parsed = validateLoginPayload(payload);

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

    const session = loginAccount(parsed.data.username, parsed.data.password);
    const response = NextResponse.json({
      account: session.account,
      hasAssessment: Boolean(session.attemptId),
    });

    response.cookies.set({
      name: AUTH_SESSION_COOKIE_NAME,
      value: session.sessionToken,
      httpOnly: true,
      sameSite: "lax",
      path: "/",
      secure: process.env.NODE_ENV === "production",
      expires: new Date(session.expiresAt),
    });

    if (session.attemptId) {
      response.cookies.set({
        name: ATTEMPT_COOKIE_NAME,
        value: session.attemptId,
        httpOnly: true,
        sameSite: "lax",
        path: "/",
        secure: process.env.NODE_ENV === "production",
        expires: new Date(session.expiresAt),
      });
    } else {
      response.cookies.set({
        name: ATTEMPT_COOKIE_NAME,
        value: "",
        httpOnly: true,
        sameSite: "lax",
        path: "/",
        secure: process.env.NODE_ENV === "production",
        expires: new Date(0),
      });
    }

    return response;
  } catch (error) {
    const message =
      error instanceof Error ? error.message : "Unable to log in to the assessment.";
    return NextResponse.json({ error: message }, { status: 400 });
  }
}

export async function GET(request: Request) {
  const cookieHeader = request.headers.get("cookie") ?? "";
  const sessionToken = cookieHeader
    .split(";")
    .map((chunk) => chunk.trim())
    .find((chunk) => chunk.startsWith(`${AUTH_SESSION_COOKIE_NAME}=`))
    ?.split("=")[1];
  const account = sessionToken ? getAuthenticatedAccount(sessionToken) : null;

  return NextResponse.json({ account });
}
