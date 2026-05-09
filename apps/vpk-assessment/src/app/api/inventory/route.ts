import { cookies } from "next/headers";
import { NextResponse } from "next/server";

import { getAuthenticatedAccount } from "@/lib/db";
import { appendParsedInventoryEntries, getInventorySnapshot } from "@/lib/inventory";
import { AUTH_SESSION_COOKIE_NAME } from "@/lib/session";

async function requireAdminAccount() {
  const cookieStore = await cookies();
  const sessionToken = cookieStore.get(AUTH_SESSION_COOKIE_NAME)?.value ?? null;
  const account = sessionToken ? getAuthenticatedAccount(sessionToken) : null;

  if (!account) {
    return NextResponse.json({ error: "Authentication required." }, { status: 401 });
  }
  if (account.role !== "admin") {
    return NextResponse.json({ error: "Admin access required." }, { status: 403 });
  }
  return account;
}

export async function GET() {
  const account = await requireAdminAccount();
  if (account instanceof NextResponse) {
    return account;
  }
  return NextResponse.json(getInventorySnapshot());
}

export async function POST(request: Request) {
  const account = await requireAdminAccount();
  if (account instanceof NextResponse) {
    return account;
  }

  try {
    const payload = (await request.json()) as { rawText?: string };
    if (!payload.rawText?.trim()) {
      return NextResponse.json({ error: "Raw purchase text is required." }, { status: 400 });
    }

    return NextResponse.json(appendParsedInventoryEntries(payload.rawText));
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Inventory import failed." },
      { status: 400 },
    );
  }
}
