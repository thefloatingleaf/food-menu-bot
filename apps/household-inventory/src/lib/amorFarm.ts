import crypto from "node:crypto";
import { execFileSync } from "node:child_process";
import path from "node:path";

import type { InventoryAnalysis, InventoryEntry, InventoryLedger } from "./inventory";
import {
  buildInventoryAnalysis,
  loadInventoryLedger,
  normalizeInventoryEntry,
  saveInventoryAnalysis,
  saveInventoryLedger,
} from "./inventory";

type AmorFarmLine = {
  amount: number;
  date_of_purchase: string;
  item_name: string;
  quantity_purchased: number;
  rate: number;
  time_slot: string;
  unit_of_measurement: string;
};

export type AmorFarmParseResult = {
  entries: InventoryEntry[];
  filtered_out_count: number;
  invoice_period_end: string;
  invoice_period_start: string;
  invoice_report_date: string | null;
  invoice_total: number | null;
  vendor_source: string;
};

export type AmorFarmImportResult = {
  added_entries: InventoryEntry[];
  analysis: InventoryAnalysis;
  filtered_out_count: number;
  invoice_files: string[];
  ledger: InventoryLedger;
  skipped_existing_count: number;
  total_parsed_entries: number;
};

type ParseOptions = {
  extractText?: (pdfPath: string) => string;
  milkOnly?: boolean;
  sourceFile?: string;
};

const DEFAULT_VENDOR = "Amor Farm";

const MONTH_LOOKUP: Record<string, number> = {
  Jan: 1,
  Feb: 2,
  Mar: 3,
  Apr: 4,
  May: 5,
  Jun: 6,
  Jul: 7,
  Aug: 8,
  Sep: 9,
  Oct: 10,
  Nov: 11,
  Dec: 12,
};

function formatIsoDate(day: number, monthAbbrev: string, year: number) {
  const month = MONTH_LOOKUP[monthAbbrev];
  if (!month) {
    throw new Error(`Unsupported month abbreviation: ${monthAbbrev}`);
  }
  return `${year.toString().padStart(4, "0")}-${month.toString().padStart(2, "0")}-${day.toString().padStart(2, "0")}`;
}

function parseInvoicePeriod(text: string) {
  const match = text.match(/\(\s*(\d{1,2}\s+[A-Za-z]{3},\s+\d{4})\s*-\s*(\d{1,2}\s+[A-Za-z]{3},\s+\d{4})\s*\)/);
  if (!match) {
    throw new Error("Could not find invoice period in Amor Farm PDF text.");
  }
  const start = match[1].match(/(\d{1,2})\s+([A-Za-z]{3}),\s+(\d{4})/);
  const end = match[2].match(/(\d{1,2})\s+([A-Za-z]{3}),\s+(\d{4})/);
  if (!start || !end) {
    throw new Error("Could not parse invoice period dates.");
  }
  return {
    invoice_period_start: formatIsoDate(Number(start[1]), start[2], Number(start[3])),
    invoice_period_end: formatIsoDate(Number(end[1]), end[2], Number(end[3])),
    invoice_year: Number(start[3]),
  };
}

function parseInvoiceReportDate(text: string) {
  const match = text.match(/^\s*(\d{1,2})\s+([A-Za-z]{3}),\s+(\d{4})/m);
  if (!match) {
    return null;
  }
  return formatIsoDate(Number(match[1]), match[2], Number(match[3]));
}

function parseInvoiceTotal(text: string) {
  const match = text.match(/\(\+\)\s*Total Buy \(₹\)\s+([0-9]+(?:\.[0-9]+)?)/);
  return match ? Number(match[1]) : null;
}

function buildPurchaseId(line: AmorFarmLine, invoicePeriodStart: string, invoicePeriodEnd: string) {
  const basis = [
    DEFAULT_VENDOR,
    invoicePeriodStart,
    invoicePeriodEnd,
    line.date_of_purchase,
    line.item_name,
    line.quantity_purchased.toString(),
    line.unit_of_measurement,
    line.rate.toString(),
    line.time_slot,
    line.amount.toString(),
  ].join("|");
  const digest = crypto.createHash("sha1").update(basis).digest("hex").slice(0, 12);
  return `amor-farm-${digest}`;
}

function parseInvoiceLines(text: string, invoiceYear: number) {
  const lines = text.split(/\r?\n/);
  const parsedLines: AmorFarmLine[] = [];
  const linePattern =
    /^\s*(\d{1,2})\s+([A-Za-z]{3})\s+(.+?)\s+([0-9]+(?:\.[0-9]+)?)\s+([A-Za-z]+)\s+([0-9]+(?:\.[0-9]+)?)\s+([A-Za-z-]+)\s+([0-9]+(?:\.[0-9]+)?)\s*$/;

  for (const rawLine of lines) {
    const match = rawLine.match(linePattern);
    if (!match) {
      continue;
    }
    const itemName = match[3].trim();
    if (itemName === "-") {
      continue;
    }
    parsedLines.push({
      date_of_purchase: formatIsoDate(Number(match[1]), match[2], invoiceYear),
      item_name: itemName,
      quantity_purchased: Number(match[4]),
      unit_of_measurement: match[5],
      rate: Number(match[6]),
      time_slot: match[7],
      amount: Number(match[8]),
    });
  }
  return parsedLines;
}

export function parseAmorFarmInvoiceText(text: string, options: ParseOptions = {}): AmorFarmParseResult {
  const { invoice_period_start, invoice_period_end, invoice_year } = parseInvoicePeriod(text);
  const invoice_report_date = parseInvoiceReportDate(text);
  const invoice_total = parseInvoiceTotal(text);
  const parsedLines = parseInvoiceLines(text, invoice_year);
  const filteredLines = options.milkOnly
    ? parsedLines.filter((line) => line.item_name.toLowerCase().includes("milk"))
    : parsedLines;
  const sourceLabel = options.sourceFile ? path.basename(options.sourceFile) : "Amor Farm PDF";
  const entries = filteredLines.map((line) =>
    normalizeInventoryEntry({
      purchase_id: buildPurchaseId(line, invoice_period_start, invoice_period_end),
      date_of_purchase: line.date_of_purchase,
      item_name: line.item_name,
      quantity_purchased: line.quantity_purchased,
      unit_of_measurement: line.unit_of_measurement,
      price: line.amount,
      vendor_source: DEFAULT_VENDOR,
      remarks: `Imported from ${sourceLabel}. Invoice period ${invoice_period_start} to ${invoice_period_end}. Delivery slot ${line.time_slot}.`,
      raw_source_text: `${line.date_of_purchase} | ${line.item_name} | ${line.quantity_purchased} ${line.unit_of_measurement} | rate ${line.rate} | amount ${line.amount}`,
    }),
  );

  return {
    entries,
    filtered_out_count: parsedLines.length - filteredLines.length,
    invoice_period_start,
    invoice_period_end,
    invoice_report_date,
    invoice_total,
    vendor_source: DEFAULT_VENDOR,
  };
}

export function extractAmorFarmPdfText(pdfPath: string) {
  try {
    return execFileSync("pdftotext", ["-layout", pdfPath, "-"], { encoding: "utf-8" });
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(`Failed to extract text from ${pdfPath}. Ensure pdftotext is installed. ${message}`);
  }
}

export function importAmorFarmPdfFiles(pdfPaths: string[], options: ParseOptions = {}): AmorFarmImportResult {
  const ledger = loadInventoryLedger();
  const existingIds = new Set(ledger.purchases.map((entry) => entry.purchase_id));
  const addedEntries: InventoryEntry[] = [];
  let skippedExistingCount = 0;
  let filteredOutCount = 0;
  let totalParsedEntries = 0;

  for (const pdfPath of pdfPaths) {
    const text = options.extractText ? options.extractText(pdfPath) : extractAmorFarmPdfText(pdfPath);
    const parsed = parseAmorFarmInvoiceText(text, { ...options, sourceFile: pdfPath });
    filteredOutCount += parsed.filtered_out_count;
    totalParsedEntries += parsed.entries.length;
    for (const entry of parsed.entries) {
      if (existingIds.has(entry.purchase_id)) {
        skippedExistingCount += 1;
        continue;
      }
      existingIds.add(entry.purchase_id);
      addedEntries.push(entry);
    }
  }

  if (!addedEntries.length) {
    return {
      added_entries: [],
      analysis: buildInventoryAnalysis(ledger.purchases),
      filtered_out_count: filteredOutCount,
      invoice_files: pdfPaths,
      ledger,
      skipped_existing_count: skippedExistingCount,
      total_parsed_entries: totalParsedEntries,
    };
  }

  const mergedLedger: InventoryLedger = {
    ...ledger,
    purchases: [...ledger.purchases, ...addedEntries].sort((left, right) => {
      const leftDate = left.date_of_purchase ?? "9999-99-99";
      const rightDate = right.date_of_purchase ?? "9999-99-99";
      return leftDate.localeCompare(rightDate) || left.item_name.localeCompare(right.item_name);
    }),
  };
  const analysis = buildInventoryAnalysis(mergedLedger.purchases);
  saveInventoryLedger(mergedLedger);
  saveInventoryAnalysis(analysis);

  return {
    added_entries: addedEntries,
    analysis,
    filtered_out_count: filteredOutCount,
    invoice_files: pdfPaths,
    ledger: mergedLedger,
    skipped_existing_count: skippedExistingCount,
    total_parsed_entries: totalParsedEntries,
  };
}
