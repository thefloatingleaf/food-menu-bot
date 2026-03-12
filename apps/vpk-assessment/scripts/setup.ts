import { validateQuestionnaireContent } from "@/lib/content-validation";
import { initializeDatabase } from "@/lib/db";

validateQuestionnaireContent();
initializeDatabase();

console.log("VPK assessment setup complete.");
