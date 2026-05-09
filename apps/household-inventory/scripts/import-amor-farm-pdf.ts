import process from "node:process";

import { importAmorFarmPdfFiles } from "../src/lib/amorFarm";

function parseArgs(argv: string[]) {
  const pdfPaths: string[] = [];
  let milkOnly = false;
  for (const arg of argv) {
    if (arg === "--milk-only") {
      milkOnly = true;
      continue;
    }
    pdfPaths.push(arg);
  }
  if (!pdfPaths.length) {
    throw new Error("Provide one or more Amor Farm PDF paths.");
  }
  return { milkOnly, pdfPaths };
}

function main() {
  const { milkOnly, pdfPaths } = parseArgs(process.argv.slice(2));
  const result = importAmorFarmPdfFiles(pdfPaths, { milkOnly });

  const totalLitres = result.added_entries
    .filter((entry) => entry.unit_of_measurement === "litre" && entry.quantity_purchased !== null)
    .reduce((sum, entry) => sum + (entry.quantity_purchased ?? 0), 0);
  const totalSpend = result.added_entries.reduce((sum, entry) => sum + (entry.price ?? 0), 0);

  console.log(
    JSON.stringify(
      {
        addedEntries: result.added_entries.length,
        skippedExistingEntries: result.skipped_existing_count,
        filteredOutEntries: result.filtered_out_count,
        totalParsedEntries: result.total_parsed_entries,
        totalLitres: Number(totalLitres.toFixed(2)),
        totalSpend: Number(totalSpend.toFixed(2)),
        trackedItemCount: result.analysis.overview.trackedItemCount,
        purchaseCount: result.analysis.overview.purchaseCount,
      },
      null,
      2,
    ),
  );
}

main();
