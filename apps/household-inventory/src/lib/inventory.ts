import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";

export type InventoryCategory =
  | "Fruits"
  | "Vegetables"
  | "Dairy"
  | "Groceries"
  | "Dry Fruits"
  | "Household Consumables"
  | "Cleaning Items"
  | "Baby Items"
  | "Medicines"
  | "Needs Review"
  | "Unclear";

export type PeriodRecord = {
  value: number | null;
  unit: string;
};

export type InventoryEntry = {
  purchase_id: string;
  date_of_purchase: string | null;
  item_name: string;
  category: InventoryCategory;
  category_status: "auto" | "needs_review";
  quantity_purchased: number | null;
  unit_of_measurement: string | null;
  price: number | null;
  vendor_source: string | null;
  remarks: string | null;
  expected_consumption_period: PeriodRecord | null;
  actual_consumption_period: PeriodRecord | null;
  raw_source_text: string | null;
  review_status: "ok" | "needs_review";
  review_notes: string[];
  entered_at: string;
  last_updated_at: string;
};

export type InventoryLedger = {
  schema_version: number;
  created_at: string | null;
  updated_at: string | null;
  purchases: InventoryEntry[];
};

export type ItemInsight = {
  itemName: string;
  category: InventoryCategory;
  purchaseCount: number;
  lastPurchaseDate: string | null;
  averageQuantityPurchased: number | null;
  quantityUnit: string | null;
  averageConsumptionRate: string | null;
  expectedStockDurationDays: number | null;
  actualConsumptionDurationDays: number | null;
  reorderFrequencyDays: number | null;
  monthlyPurchasePattern: Array<{ month: string; purchases: number; spend: number }>;
  flags: string[];
  suggestions: string[];
};

export type InventoryAnalysis = {
  schema_version: number;
  generated_at: string | null;
  overview: {
    purchaseCount: number;
    trackedItemCount: number;
    totalSpend: number;
    currentMonthSpend: number;
  };
  monthly_spend: Array<{ month: string; amount: number }>;
  category_spend: Array<{ category: InventoryCategory; amount: number }>;
  item_insights: ItemInsight[];
  possible_anomalies: Array<{
    itemName: string;
    kind: string;
    message: string;
    severity: "low" | "medium";
  }>;
};

export type ParsedInventoryPayload = {
  savedEntries: InventoryEntry[];
  notes: string[];
  ledger: InventoryLedger;
  analysis: InventoryAnalysis;
};

const SCHEMA_VERSION = 2;
const DEFAULT_LEDGER_FILE = "purchase_ledger.json";
const DEFAULT_ANALYSIS_FILE = "analysis_snapshot.json";
const CATEGORY_RULES: Array<{ category: InventoryCategory; keywords: string[] }> = [
  { category: "Fruits", keywords: ["apple", "banana", "mango", "papaya", "orange", "grape", "kiwi", "pear", "guava", "melon", "watermelon", "muskmelon", "litchi", "lichi", "anar", "amrood", "seb", "kela", "aam", "chikoo", "sapota", "pomegranate", "coconut", "phal", "fruit"] },
  { category: "Vegetables", keywords: ["potato", "onion", "tomato", "lauki", "bhindi", "parwal", "parval", "tori", "torai", "cabbage", "cauliflower", "palak", "spinach", "carrot", "beetroot", "cucumber", "karela", "ginger", "garlic", "methi", "sabzi", "vegetable", "veg"] },
  { category: "Dairy", keywords: ["milk", "curd", "paneer", "butter", "ghee", "cheese", "yogurt", "dahi", "lassi"] },
  { category: "Dry Fruits", keywords: ["almond", "badam", "cashew", "kaju", "pista", "walnut", "akhrot", "raisin", "kishmish", "fig", "anjeer", "makhana"] },
  { category: "Cleaning Items", keywords: ["detergent", "surf", "soap", "phenyl", "toilet cleaner", "floor cleaner", "harpic", "bleach", "dishwash", "rin", "vim", "scrubber", "cleaner"] },
  { category: "Baby Items", keywords: ["diaper", "wipes", "formula", "baby", "feeding bottle", "rash cream", "nappy"] },
  { category: "Medicines", keywords: ["tablet", "capsule", "syrup", "ointment", "medicine", "medicines", "paracetamol", "crocin", "dolo", "vitamin"] },
  { category: "Household Consumables", keywords: ["tissue", "foil", "garbage bag", "dustbin bag", "candle", "matchbox", "battery", "toothpaste", "toothbrush", "napkin"] },
  { category: "Groceries", keywords: ["atta", "rice", "dal", "daliya", "flour", "besan", "oil", "sugar", "salt", "masala", "turmeric", "haldi", "jeera", "tea", "coffee", "poha", "suji", "rava", "grocery", "papad", "sattu", "corn flakes", "peanuts", "sabudana"] },
];
const KNOWN_VENDOR_PATTERNS = [
  "amazon",
  "flipkart",
  "blinkit",
  "zepto",
  "instamart",
  "swiggy",
  "bigbasket",
  "dmart",
  "reliance fresh",
  "apna bazar",
  "local mandi",
  "twf",
];

function repoRoot() {
  if (process.env.HOUSEHOLD_PURCHASE_LEDGER_DIR) {
    return process.env.HOUSEHOLD_PURCHASE_LEDGER_DIR;
  }
  return path.resolve(process.cwd(), "..", "..", "data", "household_purchases");
}

function ledgerFilePath() {
  return path.join(repoRoot(), DEFAULT_LEDGER_FILE);
}

function analysisFilePath() {
  return path.join(repoRoot(), DEFAULT_ANALYSIS_FILE);
}

function nowIso() {
  return new Date().toISOString();
}

function ensureInventoryStorage() {
  fs.mkdirSync(repoRoot(), { recursive: true });
  if (!fs.existsSync(ledgerFilePath())) {
    fs.writeFileSync(
      ledgerFilePath(),
      JSON.stringify(
        {
          schema_version: SCHEMA_VERSION,
          created_at: nowIso(),
          updated_at: nowIso(),
          purchases: [],
        },
        null,
        2,
      ) + "\n",
      "utf-8",
    );
  }
  if (!fs.existsSync(analysisFilePath())) {
    fs.writeFileSync(analysisFilePath(), JSON.stringify(emptyAnalysis(), null, 2) + "\n", "utf-8");
  }
}

function readJsonFile<T>(filePath: string): T {
  return JSON.parse(fs.readFileSync(filePath, "utf-8")) as T;
}

function writeJsonFile(filePath: string, payload: unknown) {
  fs.writeFileSync(filePath, JSON.stringify(payload, null, 2) + "\n", "utf-8");
}

function normalizeDate(value: unknown): string | null {
  if (typeof value !== "string") {
    return null;
  }
  const trimmed = value.trim();
  if (!trimmed) {
    return null;
  }
  const candidates = [trimmed];
  if (/^\d{2}-\d{2}-\d{4}$/.test(trimmed) || /^\d{2}\/\d{2}\/\d{4}$/.test(trimmed)) {
    const [day, month, year] = trimmed.split(/[-/]/);
    candidates.unshift(`${year}-${month}-${day}`);
  }
  for (const candidate of candidates) {
    const parsed = new Date(candidate);
    if (!Number.isNaN(parsed.getTime())) {
      return parsed.toISOString().slice(0, 10);
    }
  }
  return null;
}

function normalizeNumber(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  if (typeof value !== "string") {
    return null;
  }
  const trimmed = value.replace(/,/g, "").trim();
  if (!trimmed) {
    return null;
  }
  const numeric = Number(trimmed);
  return Number.isFinite(numeric) ? numeric : null;
}

function normalizeText(value: unknown): string | null {
  if (typeof value !== "string") {
    return null;
  }
  const trimmed = value.trim();
  return trimmed ? trimmed : null;
}

function normalizePeriod(value: unknown): PeriodRecord | null {
  if (!value) {
    return null;
  }
  if (typeof value === "number") {
    return { value, unit: "days" };
  }
  if (typeof value === "string") {
    const matched = value.match(/(\d+(?:\.\d+)?)\s*(day|days|week|weeks|month|months)/i);
    if (matched) {
      return { value: Number(matched[1]), unit: matched[2].toLowerCase() };
    }
    const numeric = normalizeNumber(value);
    if (numeric !== null) {
      return { value: numeric, unit: "days" };
    }
    return { value: null, unit: value.trim() };
  }
  if (typeof value === "object" && value !== null) {
    const record = value as { value?: unknown; unit?: unknown };
    return {
      value: normalizeNumber(record.value),
      unit: normalizeText(record.unit) ?? "days",
    };
  }
  return null;
}

function periodToDays(period: PeriodRecord | null): number | null {
  if (!period || period.value === null) {
    return null;
  }
  const unit = period.unit.toLowerCase();
  if (unit.startsWith("week")) {
    return period.value * 7;
  }
  if (unit.startsWith("month")) {
    return period.value * 30;
  }
  return period.value;
}

export function detectCategory(itemName: string): { category: InventoryCategory; status: "auto" | "needs_review" } {
  const normalized = itemName.toLowerCase();
  const matches = CATEGORY_RULES.filter(({ keywords }) => keywords.some((keyword) => normalized.includes(keyword)));
  if (matches.length === 1) {
    return { category: matches[0].category, status: "auto" };
  }
  if (matches.length > 1) {
    return { category: "Needs Review", status: "needs_review" };
  }
  return { category: "Unclear", status: "needs_review" };
}

function normalizeUnit(value: string | null): string | null {
  if (!value) {
    return null;
  }
  const normalized = value.trim().toLowerCase();
  const aliases: Record<string, string> = {
    kg: "kg",
    kgs: "kg",
    kilogram: "kg",
    kilograms: "kg",
    g: "g",
    gm: "g",
    grams: "g",
    litre: "litre",
    litres: "litre",
    liter: "litre",
    liters: "litre",
    l: "litre",
    ml: "ml",
    pcs: "pieces",
    pc: "pieces",
    piece: "pieces",
    pieces: "pieces",
    dozen: "dozen",
    packet: "packet",
    packets: "packet",
    pack: "packet",
    packs: "packet",
    bottle: "bottle",
    bottles: "bottle",
    box: "box",
    boxes: "box",
    strip: "strip",
    strips: "strip",
  };
  return aliases[normalized] ?? value.trim();
}

export function normalizeInventoryEntry(
  input: Partial<InventoryEntry> & { item_name?: string | null; raw_source_text?: string | null },
): InventoryEntry {
  const itemName = normalizeText(input.item_name) ?? normalizeText(input.raw_source_text)?.slice(0, 80) ?? "Unclear item";
  const categoryResolution = input.category
    ? { category: input.category, status: input.category_status ?? "auto" }
    : detectCategory(itemName);
  const reviewNotes: string[] = Array.isArray(input.review_notes) ? input.review_notes.filter(Boolean) : [];
  if (!normalizeDate(input.date_of_purchase ?? null)) {
    reviewNotes.push("Date missing or unclear");
  }
  if (categoryResolution.status === "needs_review") {
    reviewNotes.push("Category needs review");
  }
  if (normalizeNumber(input.quantity_purchased ?? null) === null) {
    reviewNotes.push("Quantity missing or unclear");
  }
  return {
    purchase_id: input.purchase_id ?? crypto.randomUUID(),
    date_of_purchase: normalizeDate(input.date_of_purchase ?? null),
    item_name: itemName,
    category: categoryResolution.category,
    category_status: categoryResolution.status,
    quantity_purchased: normalizeNumber(input.quantity_purchased ?? null),
    unit_of_measurement: normalizeUnit(normalizeText(input.unit_of_measurement ?? null)),
    price: normalizeNumber(input.price ?? null),
    vendor_source: normalizeText(input.vendor_source ?? null),
    remarks: normalizeText(input.remarks ?? null),
    expected_consumption_period: normalizePeriod(input.expected_consumption_period ?? null),
    actual_consumption_period: normalizePeriod(input.actual_consumption_period ?? null),
    raw_source_text: normalizeText(input.raw_source_text ?? null),
    review_status: reviewNotes.length > 0 ? "needs_review" : "ok",
    review_notes: Array.from(new Set(reviewNotes)),
    entered_at: normalizeText(input.entered_at) ?? nowIso(),
    last_updated_at: normalizeText(input.last_updated_at) ?? nowIso(),
  };
}

function normalizeStoredLedger(rawLedger: unknown): InventoryLedger {
  const ledger = (typeof rawLedger === "object" && rawLedger !== null ? rawLedger : {}) as Partial<InventoryLedger> & {
    purchases?: unknown[];
  };
  const purchases = Array.isArray(ledger.purchases)
    ? ledger.purchases.map((entry) => normalizeInventoryEntry(entry as Partial<InventoryEntry>))
    : [];
  return {
    schema_version: SCHEMA_VERSION,
    created_at: normalizeText(ledger.created_at) ?? nowIso(),
    updated_at: normalizeText(ledger.updated_at) ?? nowIso(),
    purchases: purchases.sort((left, right) => {
      const leftDate = left.date_of_purchase ?? "9999-99-99";
      const rightDate = right.date_of_purchase ?? "9999-99-99";
      return leftDate.localeCompare(rightDate) || left.item_name.localeCompare(right.item_name);
    }),
  };
}

export function loadInventoryLedger(): InventoryLedger {
  ensureInventoryStorage();
  return normalizeStoredLedger(readJsonFile<InventoryLedger>(ledgerFilePath()));
}

export function saveInventoryLedger(ledger: InventoryLedger) {
  writeJsonFile(ledgerFilePath(), {
    ...ledger,
    schema_version: SCHEMA_VERSION,
    updated_at: nowIso(),
  });
}

function emptyAnalysis(): InventoryAnalysis {
  return {
    schema_version: SCHEMA_VERSION,
    generated_at: nowIso(),
    overview: {
      purchaseCount: 0,
      trackedItemCount: 0,
      totalSpend: 0,
      currentMonthSpend: 0,
    },
    monthly_spend: [],
    category_spend: [],
    item_insights: [],
    possible_anomalies: [],
  };
}

function normalizedItemKey(value: string) {
  return value.trim().toLowerCase().replace(/\s+/g, " ");
}

function buildItemSuggestions(flags: string[], entry: InventoryEntry, reorderFrequencyDays: number | null) {
  const suggestions: string[] = [];
  if (entry.review_status === "needs_review") {
    suggestions.push("Review raw entry details for missing date, quantity, or category.");
  }
  if (flags.some((flag) => flag.includes("fast"))) {
    suggestions.push("Check whether the recent usage spike is genuine demand or a possible recording gap.");
  }
  if (flags.some((flag) => flag.includes("slow"))) {
    suggestions.push("Reduce the next order size slightly if this slower usage pattern continues.");
  }
  if (reorderFrequencyDays !== null) {
    suggestions.push(`A reorder reminder around every ${Math.round(reorderFrequencyDays)} days would match current purchase history.`);
  }
  if (!suggestions.length) {
    suggestions.push("Continue collecting purchase entries to improve stock-duration and reorder estimates.");
  }
  return suggestions;
}

export function buildInventoryAnalysis(entries: InventoryEntry[]): InventoryAnalysis {
  if (!entries.length) {
    return emptyAnalysis();
  }

  const byItem = new Map<string, InventoryEntry[]>();
  const monthlySpend = new Map<string, number>();
  const categorySpend = new Map<InventoryCategory, number>();

  for (const entry of entries) {
    const key = normalizedItemKey(entry.item_name);
    const itemEntries = byItem.get(key) ?? [];
    itemEntries.push(entry);
    byItem.set(key, itemEntries);

    if (entry.date_of_purchase && entry.price !== null) {
      const monthKey = entry.date_of_purchase.slice(0, 7);
      monthlySpend.set(monthKey, (monthlySpend.get(monthKey) ?? 0) + entry.price);
    }
    if (entry.price !== null) {
      categorySpend.set(entry.category, (categorySpend.get(entry.category) ?? 0) + entry.price);
    }
  }

  const itemInsights: ItemInsight[] = [];
  const possibleAnomalies: InventoryAnalysis["possible_anomalies"] = [];

  for (const itemEntries of byItem.values()) {
    const sortedEntries = [...itemEntries].sort((left, right) =>
      (left.date_of_purchase ?? "").localeCompare(right.date_of_purchase ?? ""),
    );
    const dates = sortedEntries
      .map((entry) => (entry.date_of_purchase ? new Date(entry.date_of_purchase) : null))
      .filter(Boolean) as Date[];
    const quantities = sortedEntries
      .map((entry) => entry.quantity_purchased)
      .filter((value): value is number => value !== null);
    const actualDurations = sortedEntries
      .map((entry) => periodToDays(entry.actual_consumption_period))
      .filter((value): value is number => value !== null);
    const expectedDurations = sortedEntries
      .map((entry) => periodToDays(entry.expected_consumption_period))
      .filter((value): value is number => value !== null);
    const reorderIntervals = dates
      .slice(1)
      .map((value, index) => Math.round((value.getTime() - dates[index].getTime()) / 86400000));
    const averageQuantityPurchased = quantities.length
      ? Number((quantities.reduce((sum, value) => sum + value, 0) / quantities.length).toFixed(2))
      : null;
    const reorderFrequencyDays = reorderIntervals.length
      ? Number((reorderIntervals.reduce((sum, value) => sum + value, 0) / reorderIntervals.length).toFixed(1))
      : null;
    const averageActualDurationDays = actualDurations.length
      ? Number((actualDurations.reduce((sum, value) => sum + value, 0) / actualDurations.length).toFixed(1))
      : null;
    const averageExpectedDurationDays = expectedDurations.length
      ? Number((expectedDurations.reduce((sum, value) => sum + value, 0) / expectedDurations.length).toFixed(1))
      : null;
    const expectedStockDurationDays = averageActualDurationDays ?? averageExpectedDurationDays ?? reorderFrequencyDays;

    let averageConsumptionRate: string | null = null;
    if (quantities.length && actualDurations.length && sortedEntries[0].unit_of_measurement) {
      const pairValues = sortedEntries
        .map((entry) => ({
          quantity: entry.quantity_purchased,
          days: periodToDays(entry.actual_consumption_period),
        }))
        .filter(
          (pair): pair is { quantity: number; days: number } => pair.quantity !== null && pair.days !== null && pair.days > 0,
        );
      if (pairValues.length) {
        const totalQuantity = pairValues.reduce((sum, pair) => sum + pair.quantity, 0);
        const totalDays = pairValues.reduce((sum, pair) => sum + pair.days, 0);
        averageConsumptionRate = `${(totalQuantity / totalDays).toFixed(2)} ${sortedEntries[0].unit_of_measurement}/day`;
      }
    }

    const flags: string[] = [];
    if (reorderIntervals.length >= 2) {
      const currentInterval = reorderIntervals[reorderIntervals.length - 1];
      const baselineInterval =
        reorderIntervals.slice(0, -1).reduce((sum, value) => sum + value, 0) / (reorderIntervals.length - 1);
      if (baselineInterval > 0 && currentInterval <= baselineInterval * 0.6) {
        flags.push("possible fast consumption");
        possibleAnomalies.push({
          itemName: sortedEntries[0].item_name,
          kind: "possible_fast_consumption",
          severity: "medium",
          message: `${sortedEntries[0].item_name} was reordered much sooner than usual. Treat this only as a possible anomaly until more context is reviewed.`,
        });
      }
      if (baselineInterval > 0 && currentInterval >= baselineInterval * 1.6) {
        flags.push("possible slow consumption");
      }
    }
    if (averageQuantityPurchased !== null && quantities.length >= 3) {
      const latestQuantity = quantities[quantities.length - 1];
      const baselineQuantity = quantities.slice(0, -1).reduce((sum, value) => sum + value, 0) / (quantities.length - 1);
      if (baselineQuantity > 0 && latestQuantity >= baselineQuantity * 1.7) {
        flags.push("possible over-purchase or wastage");
        possibleAnomalies.push({
          itemName: sortedEntries[0].item_name,
          kind: "possible_wastage",
          severity: "low",
          message: `${sortedEntries[0].item_name} quantity is much higher than its earlier pattern. This is only a possible anomaly and may reflect planned stocking, possible wastage, or a raw-data mismatch.`,
        });
      }
    }
    if (actualDurations.length >= 2) {
      const currentDuration = actualDurations[actualDurations.length - 1];
      const baselineDuration =
        actualDurations.slice(0, -1).reduce((sum, value) => sum + value, 0) / (actualDurations.length - 1);
      if (baselineDuration > 0 && currentDuration <= baselineDuration * 0.6) {
        flags.push("possible unexplained depletion");
        possibleAnomalies.push({
          itemName: sortedEntries[0].item_name,
          kind: "possible_unexplained_depletion",
          severity: "medium",
          message: `${sortedEntries[0].item_name} appears to have finished much faster than before. Mark this only as a possible anomaly; loss or misuse should be considered only if separate supporting evidence appears.`,
        });
      }
    }

    const monthlyPatternMap = new Map<string, { purchases: number; spend: number }>();
    for (const entry of sortedEntries) {
      const month = entry.date_of_purchase?.slice(0, 7);
      if (!month) {
        continue;
      }
      const current = monthlyPatternMap.get(month) ?? { purchases: 0, spend: 0 };
      current.purchases += 1;
      current.spend += entry.price ?? 0;
      monthlyPatternMap.set(month, current);
    }

    itemInsights.push({
      itemName: sortedEntries[0].item_name,
      category: sortedEntries[0].category,
      purchaseCount: sortedEntries.length,
      lastPurchaseDate: sortedEntries[sortedEntries.length - 1].date_of_purchase,
      averageQuantityPurchased,
      quantityUnit: sortedEntries.find((entry) => entry.unit_of_measurement)?.unit_of_measurement ?? null,
      averageConsumptionRate,
      expectedStockDurationDays,
      actualConsumptionDurationDays: averageActualDurationDays,
      reorderFrequencyDays,
      monthlyPurchasePattern: Array.from(monthlyPatternMap.entries())
        .sort(([left], [right]) => left.localeCompare(right))
        .map(([month, values]) => ({ month, purchases: values.purchases, spend: Number(values.spend.toFixed(2)) })),
      flags,
      suggestions: buildItemSuggestions(flags, sortedEntries[sortedEntries.length - 1], reorderFrequencyDays),
    });
  }

  const totalSpend = entries.reduce((sum, entry) => sum + (entry.price ?? 0), 0);
  const currentMonth = nowIso().slice(0, 7);
  return {
    schema_version: SCHEMA_VERSION,
    generated_at: nowIso(),
    overview: {
      purchaseCount: entries.length,
      trackedItemCount: byItem.size,
      totalSpend: Number(totalSpend.toFixed(2)),
      currentMonthSpend: Number((monthlySpend.get(currentMonth) ?? 0).toFixed(2)),
    },
    monthly_spend: Array.from(monthlySpend.entries())
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([month, amount]) => ({ month, amount: Number(amount.toFixed(2)) })),
    category_spend: Array.from(categorySpend.entries())
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([category, amount]) => ({ category, amount: Number(amount.toFixed(2)) })),
    item_insights: itemInsights.sort((left, right) => left.itemName.localeCompare(right.itemName)),
    possible_anomalies: possibleAnomalies,
  };
}

export function loadInventoryAnalysis() {
  ensureInventoryStorage();
  if (!fs.existsSync(analysisFilePath())) {
    const empty = emptyAnalysis();
    writeJsonFile(analysisFilePath(), empty);
    return empty;
  }
  const analysis = readJsonFile<InventoryAnalysis>(analysisFilePath());
  return {
    ...emptyAnalysis(),
    ...analysis,
  };
}

export function saveInventoryAnalysis(analysis: InventoryAnalysis) {
  writeJsonFile(analysisFilePath(), analysis);
}

function removeMatchedSegment(source: string, matchedText: string | null) {
  return matchedText ? source.replace(matchedText, " ") : source;
}

function parseVendor(raw: string): string | null {
  const lowered = raw.toLowerCase();
  for (const pattern of KNOWN_VENDOR_PATTERNS) {
    if (lowered.includes(pattern)) {
      return pattern.replace(/\b\w/g, (char) => char.toUpperCase());
    }
  }
  const explicit = raw.match(/(?:from|vendor|source|shop|store)\s*[:\-]?\s*([^,;]+)/i);
  return explicit?.[1]?.trim() ?? null;
}

function parseQuantityAndUnit(raw: string): { quantity: number | null; unit: string | null; matchedText: string | null } {
  const match = raw.match(/(\d+(?:\.\d+)?)\s*(kg|kgs|kilogram|kilograms|g|gm|grams|litre|litres|liter|liters|l|ml|pcs|pc|pieces|piece|dozen|packet|packets|pack|packs|bottle|bottles|box|boxes|strip|strips)\b/i);
  if (!match) {
    return { quantity: null, unit: null, matchedText: null };
  }
  return {
    quantity: Number(match[1]),
    unit: normalizeUnit(match[2]),
    matchedText: match[0],
  };
}

function parsePrice(raw: string): { price: number | null; matchedText: string | null } {
  const match =
    raw.match(/(?:₹|rs\.?|inr)\s*([0-9]+(?:\.[0-9]+)?)/i) ??
    raw.match(/(?:amount|price|total)\s*[:\-]?\s*([0-9]+(?:\.[0-9]+)?)/i);
  if (!match) {
    return { price: null, matchedText: null };
  }
  return {
    price: Number(match[1]),
    matchedText: match[0],
  };
}

function parseDateFromRaw(raw: string): { date: string | null; matchedText: string | null } {
  const match = raw.match(/\b(\d{4}-\d{2}-\d{2}|\d{2}-\d{2}-\d{4}|\d{2}\/\d{2}\/\d{4})\b/);
  if (!match) {
    return { date: null, matchedText: null };
  }
  return {
    date: normalizeDate(match[1]),
    matchedText: match[0],
  };
}

function parsePeriodFromRaw(raw: string, label: "expected" | "actual"): PeriodRecord | null {
  const regex =
    label === "expected"
      ? /(?:expected(?: consumption)?|for)\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*(day|days|week|weeks|month|months)/i
      : /(?:actual(?: consumption)?|consumed in|finished in)\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*(day|days|week|weeks|month|months)/i;
  const match = raw.match(regex);
  return match ? { value: Number(match[1]), unit: match[2].toLowerCase() } : null;
}

function parseKeyValueBlock(block: string): (Partial<InventoryEntry> & { item_name?: string | null }) | null {
  const lines = block
    .split(/\n+/)
    .map((line) => line.trim())
    .filter(Boolean);
  if (!lines.length || !lines.some((line) => line.includes(":"))) {
    return null;
  }

  const values = new Map<string, string>();
  for (const line of lines) {
    const match = line.match(/^([^:]+):\s*(.+)$/);
    if (!match) {
      continue;
    }
    values.set(match[1].trim().toLowerCase(), match[2].trim());
  }
  if (!values.size) {
    return null;
  }

  return {
    date_of_purchase: values.get("date") ?? values.get("date of purchase") ?? null,
    item_name: values.get("item") ?? values.get("item name") ?? values.get("name") ?? undefined,
    quantity_purchased: normalizeNumber(values.get("quantity") ?? values.get("qty") ?? null),
    unit_of_measurement: values.get("unit") ?? undefined,
    price: normalizeNumber(values.get("price") ?? values.get("amount") ?? null),
    vendor_source: values.get("vendor") ?? values.get("source") ?? undefined,
    remarks: values.get("remarks") ?? values.get("remark") ?? undefined,
    expected_consumption_period: normalizePeriod(
      values.get("expected consumption period") ?? values.get("expected period") ?? null,
    ),
    actual_consumption_period: normalizePeriod(
      values.get("actual consumption period") ?? values.get("actual period") ?? null,
    ),
    raw_source_text: block,
  };
}

export function parseRawInventoryText(rawText: string): { entries: InventoryEntry[]; notes: string[] } {
  const trimmed = rawText.trim();
  if (!trimmed) {
    return { entries: [], notes: ["No raw purchase text was provided."] };
  }

  const blocks = trimmed
    .split(/\n\s*\n+/)
    .flatMap((block) =>
      block.includes("\n")
        ? [block]
        : block
            .split(/\n+/)
            .map((line) => line.trim())
            .filter(Boolean),
    )
    .map((block) => block.replace(/^[\-\*\u2022]\s*/, "").trim())
    .filter(Boolean);

  const entries: InventoryEntry[] = [];
  const notes: string[] = [];

  for (const block of blocks) {
    const keyValueEntry = parseKeyValueBlock(block);
    if (keyValueEntry) {
      const normalized = normalizeInventoryEntry(keyValueEntry);
      entries.push(normalized);
      continue;
    }

    const dateInfo = parseDateFromRaw(block);
    const quantityInfo = parseQuantityAndUnit(block);
    const priceInfo = parsePrice(block);
    const vendor = parseVendor(block);
    const expectedPeriod = parsePeriodFromRaw(block, "expected");
    const actualPeriod = parsePeriodFromRaw(block, "actual");

    let residue = block;
    residue = removeMatchedSegment(residue, dateInfo.matchedText);
    residue = removeMatchedSegment(residue, quantityInfo.matchedText);
    residue = removeMatchedSegment(residue, priceInfo.matchedText);
    if (vendor) {
      residue = residue.replace(new RegExp(vendor, "ig"), " ");
    }
    residue = residue
      .replace(/(?:from|vendor|source|shop|store|amount|price|total|expected|actual|consumed in|finished in)[:\-]?/gi, " ")
      .replace(/\s+/g, " ")
      .replace(/[;,|]+/g, " ")
      .trim();

    const itemName = residue.split(" / ")[0].trim() || block.slice(0, 80);
    const remarks = residue && itemName !== residue ? residue.replace(itemName, "").trim() || block : block;

    const normalized = normalizeInventoryEntry({
      date_of_purchase: dateInfo.date,
      item_name: itemName,
      quantity_purchased: quantityInfo.quantity,
      unit_of_measurement: quantityInfo.unit,
      price: priceInfo.price,
      vendor_source: vendor,
      remarks,
      expected_consumption_period: expectedPeriod,
      actual_consumption_period: actualPeriod,
      raw_source_text: block,
    });

    if (normalized.review_status === "needs_review") {
      notes.push(`Saved "${normalized.item_name}" with review flags: ${normalized.review_notes.join(", ")}.`);
    }
    entries.push(normalized);
  }

  return { entries, notes };
}

export function appendParsedInventoryEntries(rawText: string): ParsedInventoryPayload {
  const { entries, notes } = parseRawInventoryText(rawText);
  if (!entries.length) {
    throw new Error("No purchase entries could be extracted from the pasted text.");
  }

  const ledger = loadInventoryLedger();
  const mergedLedger: InventoryLedger = {
    ...ledger,
    updated_at: nowIso(),
    purchases: [...ledger.purchases, ...entries].sort((left, right) => {
      const leftDate = left.date_of_purchase ?? "9999-99-99";
      const rightDate = right.date_of_purchase ?? "9999-99-99";
      return leftDate.localeCompare(rightDate) || left.item_name.localeCompare(right.item_name);
    }),
  };
  const analysis = buildInventoryAnalysis(mergedLedger.purchases);
  saveInventoryLedger(mergedLedger);
  saveInventoryAnalysis(analysis);

  return {
    savedEntries: entries,
    notes,
    ledger: mergedLedger,
    analysis,
  };
}

export function getInventorySnapshot() {
  const ledger = loadInventoryLedger();
  const analysis = buildInventoryAnalysis(ledger.purchases);
  saveInventoryAnalysis(analysis);
  return {
    ledger,
    analysis,
  };
}
