import { cookies } from "next/headers";
import { NextResponse } from "next/server";

import { completeAssessment } from "@/lib/db";
import { ATTEMPT_COOKIE_NAME } from "@/lib/session";

export async function POST(request: Request) {
  try {
    const payload = (await request.json()) as { attemptId?: string };
    const cookieStore = await cookies();
    const attemptId = payload.attemptId ?? cookieStore.get(ATTEMPT_COOKIE_NAME)?.value;

    if (!attemptId) {
      return NextResponse.json({ error: "A valid assessment session is required." }, { status: 400 });
    }

    return NextResponse.json(completeAssessment(attemptId));
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Unable to complete assessment." },
      { status: 400 },
    );
  }
}
