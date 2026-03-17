import { cookies } from "next/headers";
import { NextResponse } from "next/server";

import { createManagedAccount, getAuthenticatedAccount, listAccounts } from "@/lib/db";
import { AUTH_SESSION_COOKIE_NAME } from "@/lib/session";
import { validateAccountCreationPayload } from "@/lib/validation";

async function requireAdmin() {
  const cookieStore = await cookies();
  const sessionToken = cookieStore.get(AUTH_SESSION_COOKIE_NAME)?.value;
  const account = sessionToken ? getAuthenticatedAccount(sessionToken) : null;

  if (!account || account.role !== "admin") {
    return null;
  }

  return account;
}

export async function GET() {
  const admin = await requireAdmin();
  if (!admin) {
    return NextResponse.json({ error: "Admin login is required." }, { status: 403 });
  }

  return NextResponse.json({ accounts: listAccounts() });
}

export async function POST(request: Request) {
  const admin = await requireAdmin();
  if (!admin) {
    return NextResponse.json({ error: "Admin login is required." }, { status: 403 });
  }

  const payload = await request.json();
  const parsed = validateAccountCreationPayload(payload);

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

  try {
    const account = createManagedAccount(parsed.data);
    return NextResponse.json({ account });
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Unable to create the account." },
      { status: 400 },
    );
  }
}
