import fs from "node:fs";
import process from "node:process";

import {
  appendParsedInventoryEntries,
  buildInventoryAnalysis,
  loadInventoryLedger,
  parseRawInventoryText,
  type InventoryEntry,
} from "../src/lib/inventory";

type ParsedArgs = {
  dryRun: boolean;
  source: "file" | "stdin";
  filePath: string | null;
};

function usage() {
  return [
    "Usage:",
    "  npm run import:raw-purchases -- --stdin",
    "  npm run import:raw-purchases -- /absolute/path/to/purchases.txt",
    "  npm run import:raw-purchases -- --dry-run --stdin",
  ].join("\n");
}

function parseArgs(argv: string[]): ParsedArgs {
  let dryRun = false;
  let source: "file" | "stdin" | null = null;
  let filePath: string | null = null;

  for (const arg of argv) {
    if (arg === "--dry-run") {
      dryRun = true;
      continue;
    }
    if (arg === "--stdin") {
      source = "stdin";
      continue;
    }
    if (arg === "--help" || arg === "-h") {
      console.log(usage());
      process.exit(0);
    }
    if (arg.startsWith("-")) {
      throw new Error(`Unknown option: ${arg}\n${usage()}`);
    }
    if (filePath) {
      throw new Error(`Only one purchase text file can be imported at a time.\n${usage()}`);
    }
    source = "file";
    filePath = arg;
  }

  if (!source) {
    throw new Error(`Provide a purchase text file or use --stdin.\n${usage()}`);
  }
  if (source === "file" && !filePath) {
    throw new Error(`Provide a purchase text file.\n${usage()}`);
  }

  return { dryRun, source, filePath };
}

function readRawPurchaseText(args: ParsedArgs) {
  if (args.source === "stdin") {
    return fs.readFileSync(0, "utf-8");
  }
  return fs.readFileSync(args.filePath ?? "", "utf-8");
}

function entryDigest(entry: InventoryEntry) {
  return {
    date: entry.date_of_purchase,
    item: entry.item_name,
    quantity: entry.quantity_purchased,
    unit: entry.unit_of_measurement,
    price: entry.price,
    vendor: entry.vendor_source,
    reviewStatus: entry.review_status,
    reviewNotes: entry.review_notes,
  };
}

function printJson(payload: unknown) {
  console.log(JSON.stringify(payload, null, 2));
}

function main() {
  const args = parseArgs(process.argv.slice(2));
  const rawText = readRawPurchaseText(args);

  if (args.dryRun) {
    const { entries, notes } = parseRawInventoryText(rawText);
    if (!entries.length) {
      throw new Error("No purchase entries could be extracted from the provided text.");
    }
    const ledger = loadInventoryLedger();
    const projectedAnalysis = buildInventoryAnalysis([...ledger.purchases, ...entries]);
    printJson({
      dryRun: true,
      saved: false,
      parsedEntries: entries.length,
      entriesNeedingReview: entries.filter((entry) => entry.review_status === "needs_review").length,
      currentPurchaseCount: ledger.purchases.length,
      projectedPurchaseCount: ledger.purchases.length + entries.length,
      projectedTrackedItemCount: projectedAnalysis.overview.trackedItemCount,
      projectedReviewItemCount: projectedAnalysis.overview.reviewItemCount,
      notes,
      parsedEntryPreview: entries.map(entryDigest),
    });
    return;
  }

  const result = appendParsedInventoryEntries(rawText);
  printJson({
    dryRun: false,
    saved: true,
    savedEntries: result.savedEntries.length,
    entriesNeedingReview: result.savedEntries.filter((entry) => entry.review_status === "needs_review").length,
    purchaseCount: result.analysis.overview.purchaseCount,
    trackedItemCount: result.analysis.overview.trackedItemCount,
    reviewItemCount: result.analysis.overview.reviewItemCount,
    inventoryHealthScore: result.analysis.overview.inventoryHealthScore,
    notes: result.notes,
    savedEntryPreview: result.savedEntries.map(entryDigest),
  });
}

try {
  main();
} catch (error) {
  console.error(error instanceof Error ? error.message : "Raw purchase import failed.");
  process.exitCode = 1;
}
