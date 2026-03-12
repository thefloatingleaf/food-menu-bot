import { NextResponse } from "next/server";

import { getResultByAttemptId } from "@/lib/db";

type Context = {
  params: Promise<{ attemptId: string }>;
};

export async function GET(_: Request, context: Context) {
  const { attemptId } = await context.params;
  const result = getResultByAttemptId(attemptId);

  if (!result) {
    return NextResponse.json({ error: "Result not found." }, { status: 404 });
  }

  return NextResponse.json(result);
}
