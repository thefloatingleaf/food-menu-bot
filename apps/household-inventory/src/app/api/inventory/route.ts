import { NextResponse } from "next/server";

import { appendParsedInventoryEntries, getInventorySnapshot } from "@/lib/inventory";

export async function GET() {
  return NextResponse.json(getInventorySnapshot());
}

export async function POST(request: Request) {
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
