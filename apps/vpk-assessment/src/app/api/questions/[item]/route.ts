import { cookies } from "next/headers";
import { NextResponse } from "next/server";

import { getQuestionPayload, saveResponse } from "@/lib/db";
import { ATTEMPT_COOKIE_NAME } from "@/lib/session";
import { responseSchema } from "@/lib/validation";

type Context = {
  params: Promise<{ item: string }>;
};

export async function GET(_: Request, context: Context) {
  try {
    const { item } = await context.params;
    const cookieStore = await cookies();
    const attemptId = cookieStore.get(ATTEMPT_COOKIE_NAME)?.value;
    const questionIndex = Number(item);

    if (!attemptId || Number.isNaN(questionIndex)) {
      return NextResponse.json({ error: "A valid assessment session is required." }, { status: 400 });
    }

    return NextResponse.json(getQuestionPayload(attemptId, questionIndex));
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
    const attemptId = parsed.success
      ? parsed.data.attemptId ?? cookieStore.get(ATTEMPT_COOKIE_NAME)?.value
      : undefined;

    if (!parsed.success || !attemptId) {
      return NextResponse.json({ error: "Save requests must include both selections." }, { status: 400 });
    }

    return NextResponse.json(
      saveResponse(attemptId, item, parsed.data.lifetimeOptionId, parsed.data.presentOptionId),
    );
  } catch (error) {
    return NextResponse.json(
      { error: error instanceof Error ? error.message : "Unable to save response." },
      { status: 400 },
    );
  }
}
