import fs from "node:fs";
import process from "node:process";

import { buildContextNotes, loadSupplyContext, upsertUsageObservationEntries } from "../src/lib/inventory";

function readPayloadText(args: string[]) {
  if (args[0] === "--stdin") {
    return fs.readFileSync(0, "utf-8");
  }
  if (args[0]) {
    return fs.readFileSync(args[0], "utf-8");
  }
  throw new Error("Provide a JSON file path or use --stdin.");
}

function main() {
  const payloadText = readPayloadText(process.argv.slice(2));
  const payload = JSON.parse(payloadText) as unknown;
  const entries = Array.isArray(payload) ? payload : [payload];
  const result = upsertUsageObservationEntries(entries);
  const contextNotes = buildContextNotes(loadSupplyContext().entries, result.ledger.observations);

  console.log(
    JSON.stringify(
      {
        savedEntries: result.savedEntries.length,
        usageObservations: result.ledger.observations.length,
        contextNotes,
      },
      null,
      2,
    ),
  );
}

try {
  main();
} catch (error) {
  console.error(error instanceof Error ? error.message : "Usage observation import failed.");
  process.exitCode = 1;
}
