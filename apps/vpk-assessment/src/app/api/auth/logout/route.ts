import { cookies } from "next/headers";
import { NextResponse } from "next/server";

import { logoutAccount } from "@/lib/db";
import { ATTEMPT_COOKIE_NAME, AUTH_SESSION_COOKIE_NAME } from "@/lib/session";

export async function POST() {
  const cookieStore = await cookies();
  const sessionToken = cookieStore.get(AUTH_SESSION_COOKIE_NAME)?.value;

  if (sessionToken) {
    logoutAccount(sessionToken);
  }

  const response = NextResponse.json({ ok: true });
  const expired = new Date(0);

  response.cookies.set({
    name: AUTH_SESSION_COOKIE_NAME,
    value: "",
    httpOnly: true,
    sameSite: "lax",
    path: "/",
    secure: process.env.NODE_ENV === "production",
    expires: expired,
  });
  response.cookies.set({
    name: ATTEMPT_COOKIE_NAME,
    value: "",
    httpOnly: true,
    sameSite: "lax",
    path: "/",
    secure: process.env.NODE_ENV === "production",
    expires: expired,
  });

  return response;
}
