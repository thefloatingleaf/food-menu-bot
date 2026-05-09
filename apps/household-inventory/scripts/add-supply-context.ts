import fs from "node:fs";
import process from "node:process";

import { buildContextNotes, upsertSupplyContextEntries } from "../src/lib/inventory";

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
  const merged = upsertSupplyContextEntries(entries);
  const contextNotes = buildContextNotes(merged.entries);
  console.log(
    JSON.stringify(
      {
        activeEntries: merged.entries.filter((entry) => entry.active).length,
        savedEntries: entries.length,
        contextNotes,
      },
      null,
      2,
    ),
  );
}

main();
