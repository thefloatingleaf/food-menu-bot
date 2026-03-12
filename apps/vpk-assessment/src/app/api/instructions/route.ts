import { NextResponse } from "next/server";

import { questionnaireContent } from "@/data/vpkQuestionnaire";

export function GET() {
  return NextResponse.json({ instructionText: questionnaireContent.instructionText });
}
