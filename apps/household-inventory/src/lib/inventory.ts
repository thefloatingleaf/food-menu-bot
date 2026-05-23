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

export type ConsumptionTrend =
  | "Normal"
  | "Higher than usual"
  | "Lower than usual"
  | "Insufficient data";

export type StockBucket =
  | "Items to Order Today"
  | "Items to Order Within 3 Days"
  | "Items Sufficient for More Than 7 Days"
  | "Monitor"
  | "Requires Review";

export type ConfidenceLevel = "High" | "Medium" | "Low" | "Insufficient data";

export type ItemInsight = {
  itemName: string;
  category: InventoryCategory;
  purchaseCount: number;
  lastPurchaseDate: string | null;
  lastPurchasedQuantity: number | null;
  lastPurchaseSpend: number | null;
  averageQuantityPurchased: number | null;
  quantityUnit: string | null;
  averageDailyConsumption: number | null;
  averageWeeklyConsumption: number | null;
  averageConsumptionRate: string | null;
  averageDailyConsumptionLabel: string | null;
  averageWeeklyConsumptionLabel: string | null;
  consumptionBasis: "actual" | "purchase_history" | "insufficient_data";
  expectedStockDurationDays: number | null;
  estimatedDaysOfStockRemaining: number | null;
  actualConsumptionDurationDays: number | null;
  reorderFrequencyDays: number | null;
  normalPurchaseFrequencyDays: number | null;
  recommendedReorderDate: string | null;
  suggestedReorderQuantity: number | null;
  currentConsumptionTrend: ConsumptionTrend;
  reorderPriorityScore: number | null;
  stockBucket: StockBucket;
  dataConfidenceScore: number;
  dataConfidenceLabel: ConfidenceLevel;
  daysSinceLastPurchase: number | null;
  monthlyPurchasePattern: Array<{ month: string; purchases: number; spend: number }>;
  monthlyQuantityPattern: Array<{ month: string; purchases: number; quantity: number | null; spend: number; unit: string | null }>;
  flags: string[];
  suggestions: string[];
  totalSpend: number;
};

export type DashboardAlert = {
  itemName: string;
  label: string;
  detail: string;
  severity: "low" | "medium";
  suggestion: string;
};

export type RankedItem = {
  itemName: string;
  value: number;
  label: string;
};

export type RecentEntryDigest = {
  purchaseId: string;
  dateOfPurchase: string | null;
  itemName: string;
  category: InventoryCategory;
  quantityLabel: string;
  amountLabel: string;
  reviewStatus: "ok" | "needs_review";
  reviewNotes: string[];
  enteredAt: string;
};

export type DuplicateEntryDigest = {
  signature: string;
  itemName: string;
  dateOfPurchase: string | null;
  quantityLabel: string;
  amountLabel: string;
  vendorSource: string | null;
  occurrences: number;
};

export type PairedPurchaseInsight = {
  pairLabel: string;
  frequency: number;
  lastSeenDate: string | null;
};

export type ItemTrendDigest = {
  itemName: string;
  unit: string | null;
  months: Array<{ month: string; purchases: number; quantity: number | null; spend: number }>;
};

export type CategoryTrendDigest = {
  category: InventoryCategory;
  months: Array<{ month: string; purchases: number; spend: number }>;
};

export type InventoryDashboard = {
  items_to_order_today: ItemInsight[];
  items_to_order_within_3_days: ItemInsight[];
  items_sufficient_for_more_than_7_days: ItemInsight[];
  unusual_consumption_alerts: DashboardAlert[];
  possible_wastage_indicators: DashboardAlert[];
  unexplained_depletion_indicators: DashboardAlert[];
  items_consumed_faster_than_expected: ItemInsight[];
  items_consumed_slower_than_expected: ItemInsight[];
  items_not_purchased_for_a_long_time: ItemInsight[];
  items_purchased_too_frequently: ItemInsight[];
  items_purchased_in_excess_quantity: ItemInsight[];
  items_usually_ordered_together: PairedPurchaseInsight[];
  monthly_item_consumption_trend: ItemTrendDigest[];
  monthly_category_consumption_trend: CategoryTrendDigest[];
  monthly_category_spending_trend: CategoryTrendDigest[];
  top_frequently_purchased_items: RankedItem[];
  top_highest_spending_items: RankedItem[];
  items_with_rising_consumption: ItemInsight[];
  items_with_falling_consumption: ItemInsight[];
  seasonal_consumption_pattern: string[];
  guest_event_festival_impact: string[];
  perishable_items_requiring_faster_use: ItemInsight[];
  slow_moving_items: ItemInsight[];
  dead_stock_items: ItemInsight[];
  stock_out_risk_items: ItemInsight[];
  overstock_risk_items: ItemInsight[];
  items_requiring_manual_review: ItemInsight[];
  recently_added_purchase_data: RecentEntryDigest[];
  recently_parsed_items_needing_correction: RecentEntryDigest[];
  duplicate_or_suspicious_entries: DuplicateEntryDigest[];
  items_with_unclear_quantity_or_unit: ItemInsight[];
  items_with_unclear_category: ItemInsight[];
  auto_categorisation: {
    autoEntryCount: number;
    needsReviewEntryCount: number;
    autoEntryRate: number;
    autoItemCount: number;
    needsReviewItemCount: number;
  };
};

export type InventoryAnalysis = {
  schema_version: number;
  generated_at: string | null;
  overview: {
    purchaseCount: number;
    trackedItemCount: number;
    totalSpend: number;
    currentMonthSpend: number;
    itemsToOrderTodayCount: number;
    itemsToOrderSoonCount: number;
    stableItemCount: number;
    reviewItemCount: number;
    totalMonthlyHouseholdConsumptionValue: number;
    monthOnMonthSpendingChange: number | null;
    monthOnMonthConsumptionChange: number | null;
    inventoryHealthScore: number;
    householdConsumptionStabilityScore: number;
    duplicateEntryCount: number;
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
  dashboard: InventoryDashboard;
};

export type SupplyContextEntry = {
  context_id: string;
  item_name: string;
  category: InventoryCategory;
  quantity_per_day: number | null;
  unit_of_measurement: string | null;
  beneficiary: string | null;
  source_name: string | null;
  source_description: string | null;
  effective_start_date: string | null;
  effective_end_date: string | null;
  active: boolean;
  price: number | null;
  remarks: string | null;
  review_status: "ok" | "needs_review";
  review_notes: string[];
  entered_at: string;
  last_updated_at: string;
};

export type ContextNote = {
  context_id: string;
  severity: "info" | "needs_review";
  title: string;
  message: string;
};

export type InventorySnapshot = {
  analysis: InventoryAnalysis;
  contextNotes: ContextNote[];
  ledger: InventoryLedger;
  supplyContext: SupplyContextEntry[];
};

export type ParsedInventoryPayload = {
  savedEntries: InventoryEntry[];
  notes: string[];
  ledger: InventoryLedger;
  analysis: InventoryAnalysis;
  contextNotes: ContextNote[];
  supplyContext: SupplyContextEntry[];
};

const SCHEMA_VERSION = 2;
const DEFAULT_LEDGER_FILE = "purchase_ledger.json";
const DEFAULT_ANALYSIS_FILE = "analysis_snapshot.json";
const DEFAULT_SUPPLY_CONTEXT_FILE = "supply_context.json";
const CATEGORY_RULES: Array<{ category: InventoryCategory; keywords: string[] }> = [
  { category: "Fruits", keywords: ["apple", "banana", "mango", "papaya", "orange", "grape", "kiwi", "pear", "guava", "melon", "watermelon", "muskmelon", "litchi", "lichi", "anar", "amrood", "seb", "kela", "aam", "chikoo", "sapota", "pomegranate", "coconut", "phal", "fruit"] },
  { category: "Vegetables", keywords: ["potato", "onion", "tomato", "lauki", "bhindi", "parwal", "parval", "tori", "torai", "cabbage", "cauliflower", "palak", "spinach", "carrot", "beetroot", "cucumber", "karela", "ginger", "garlic", "methi", "sabzi", "vegetable", "veg"] },
  { category: "Dairy", keywords: ["milk", "curd", "paneer", "butter", "ghee", "cheese", "yogurt", "dahi", "lassi", "cream"] },
  { category: "Dry Fruits", keywords: ["almond", "badam", "cashew", "kaju", "pista", "walnut", "akhrot", "raisin", "kishmish", "fig", "anjeer", "makhana"] },
  { category: "Cleaning Items", keywords: ["detergent", "surf", "soap", "phenyl", "toilet cleaner", "floor cleaner", "harpic", "bleach", "dishwash", "rin", "vim", "scrubber", "cleaner"] },
  { category: "Baby Items", keywords: ["diaper", "wipes", "formula", "baby laundry", "baby", "feeding bottle", "rash cream", "nappy"] },
  { category: "Medicines", keywords: ["tablet", "capsule", "syrup", "ointment", "medicine", "medicines", "paracetamol", "crocin", "dolo", "vitamin"] },
  { category: "Household Consumables", keywords: ["tissue", "foil", "garbage bag", "dustbin bag", "gift bag", "paper bag", "candle", "matchbox", "battery", "toothpaste", "toothbrush", "napkin"] },
  { category: "Groceries", keywords: ["atta", "rice", "dal", "masoor", "daliya", "flour", "maida", "besan", "oil", "sugar", "salt", "masala", "turmeric", "haldi", "jeera", "tea", "coffee", "poha", "suji", "rava", "grocery", "papad", "sattu", "corn flakes", "peanut", "peanuts", "sabudana", "bread", "semolina", "roti", "rotis", "coriander", "fennel", "spices", "chutney", "ketchup", "sauce", "bhujia", "namkeen", "snack", "soft drink", "coke", "cola", "sprite", "mountain dew"] },
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
const GENERATED_ENTRY_REVIEW_NOTES = new Set([
  "Date missing or unclear",
  "Category needs review",
  "Quantity missing or unclear",
  "Unit missing or unclear",
]);
const GENERATED_SUPPLY_REVIEW_NOTES = new Set([
  "Daily quantity missing or unclear",
  "Source name not yet provided",
  "Start date not yet provided",
  "Price not yet provided",
  "Category needs review",
]);

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

function supplyContextFilePath() {
  return path.join(repoRoot(), DEFAULT_SUPPLY_CONTEXT_FILE);
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
  if (!fs.existsSync(supplyContextFilePath())) {
    fs.writeFileSync(
      supplyContextFilePath(),
      JSON.stringify(
        {
          schema_version: SCHEMA_VERSION,
          created_at: nowIso(),
          updated_at: nowIso(),
          entries: [],
        },
        null,
        2,
      ) + "\n",
      "utf-8",
    );
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
  const matches = CATEGORY_RULES
    .map(({ category, keywords }) => ({
      category,
      matchedKeywords: keywords.filter((keyword) => normalized.includes(keyword)),
    }))
    .filter((result) => result.matchedKeywords.length > 0);
  if (!matches.length) {
    return { category: "Unclear", status: "needs_review" };
  }
  const scoredMatches = matches
    .map((result) => ({
      category: result.category,
      score: result.matchedKeywords.reduce((sum, keyword) => sum + keyword.length, 0),
    }))
    .sort((left, right) => right.score - left.score);
  if (scoredMatches.length === 1 || scoredMatches[0].score > scoredMatches[1].score) {
    return { category: scoredMatches[0].category, status: "auto" };
  }
  if (scoredMatches.length > 1) {
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

function normalizeSupplyContextEntry(input: Partial<SupplyContextEntry> & { item_name?: string | null }): SupplyContextEntry {
  const itemName = normalizeText(input.item_name) ?? "Unclear item";
  const categoryResolution = input.category
    ? { category: input.category, status: "auto" as const }
    : detectCategory(itemName);
  const reviewNotes: string[] = Array.isArray(input.review_notes)
    ? input.review_notes.filter((note) => note && !GENERATED_SUPPLY_REVIEW_NOTES.has(note))
    : [];
  if (normalizeNumber(input.quantity_per_day ?? null) === null) {
    reviewNotes.push("Daily quantity missing or unclear");
  }
  if (!normalizeText(input.source_name ?? null)) {
    reviewNotes.push("Source name not yet provided");
  }
  if (!normalizeDate(input.effective_start_date ?? null)) {
    reviewNotes.push("Start date not yet provided");
  }
  if (normalizeNumber(input.price ?? null) === null) {
    reviewNotes.push("Price not yet provided");
  }
  if (categoryResolution.status === "needs_review") {
    reviewNotes.push("Category needs review");
  }
  return {
    context_id: input.context_id ?? crypto.randomUUID(),
    item_name: itemName,
    category: categoryResolution.category,
    quantity_per_day: normalizeNumber(input.quantity_per_day ?? null),
    unit_of_measurement: normalizeUnit(normalizeText(input.unit_of_measurement ?? null)),
    beneficiary: normalizeText(input.beneficiary ?? null),
    source_name: normalizeText(input.source_name ?? null),
    source_description: normalizeText(input.source_description ?? null),
    effective_start_date: normalizeDate(input.effective_start_date ?? null),
    effective_end_date: normalizeDate(input.effective_end_date ?? null),
    active: input.active ?? true,
    price: normalizeNumber(input.price ?? null),
    remarks: normalizeText(input.remarks ?? null),
    review_status: reviewNotes.length > 0 ? "needs_review" : "ok",
    review_notes: Array.from(new Set(reviewNotes)),
    entered_at: normalizeText(input.entered_at) ?? nowIso(),
    last_updated_at: normalizeText(input.last_updated_at) ?? nowIso(),
  };
}

export function normalizeInventoryEntry(
  input: Partial<InventoryEntry> & { item_name?: string | null; raw_source_text?: string | null },
): InventoryEntry {
  const itemName = normalizeText(input.item_name) ?? normalizeText(input.raw_source_text)?.slice(0, 80) ?? "Unclear item";
  const shouldRedetectCategory =
    !input.category || input.category_status === "needs_review" || input.category === "Needs Review" || input.category === "Unclear";
  const categoryResolution = shouldRedetectCategory
    ? detectCategory(itemName)
    : { category: input.category as InventoryCategory, status: input.category_status ?? "auto" };
  const reviewNotes: string[] = Array.isArray(input.review_notes)
    ? input.review_notes.filter((note) => note && !GENERATED_ENTRY_REVIEW_NOTES.has(note))
    : [];
  if (!normalizeDate(input.date_of_purchase ?? null)) {
    reviewNotes.push("Date missing or unclear");
  }
  if (categoryResolution.status === "needs_review") {
    reviewNotes.push("Category needs review");
  }
  if (normalizeNumber(input.quantity_purchased ?? null) === null) {
    reviewNotes.push("Quantity missing or unclear");
  }
  if (normalizeNumber(input.quantity_purchased ?? null) !== null && !normalizeUnit(normalizeText(input.unit_of_measurement ?? null))) {
    reviewNotes.push("Unit missing or unclear");
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

function normalizeStoredSupplyContext(rawContext: unknown): {
  created_at: string;
  entries: SupplyContextEntry[];
  schema_version: number;
  updated_at: string;
} {
  const supplyContext =
    typeof rawContext === "object" && rawContext !== null
      ? (rawContext as { created_at?: unknown; entries?: unknown[]; updated_at?: unknown })
      : {};
  const entries = Array.isArray(supplyContext.entries)
    ? supplyContext.entries.map((entry) => normalizeSupplyContextEntry(entry as Partial<SupplyContextEntry>))
    : [];
  return {
    schema_version: SCHEMA_VERSION,
    created_at: normalizeText(supplyContext.created_at) ?? nowIso(),
    updated_at: normalizeText(supplyContext.updated_at) ?? nowIso(),
    entries,
  };
}

export function loadSupplyContext() {
  ensureInventoryStorage();
  return normalizeStoredSupplyContext(readJsonFile(supplyContextFilePath()));
}

export function saveSupplyContext(context: { created_at: string; entries: SupplyContextEntry[]; schema_version: number; updated_at: string }) {
  writeJsonFile(supplyContextFilePath(), {
    ...context,
    schema_version: SCHEMA_VERSION,
    updated_at: nowIso(),
  });
}

export function upsertSupplyContextEntries(
  rawEntries: Array<Partial<SupplyContextEntry> & { item_name?: string | null }>,
) {
  const context = loadSupplyContext();
  const entryMap = new Map(context.entries.map((entry) => [entry.context_id, entry]));
  const savedEntries: SupplyContextEntry[] = [];

  for (const rawEntry of rawEntries) {
    const contextId = rawEntry.context_id ?? crypto.randomUUID();
    const normalized = normalizeSupplyContextEntry({
      ...entryMap.get(contextId),
      ...rawEntry,
      context_id: contextId,
      entered_at: entryMap.get(contextId)?.entered_at ?? rawEntry.entered_at,
    });
    entryMap.set(contextId, normalized);
    savedEntries.push(normalized);
  }

  const merged = {
    ...context,
    entries: Array.from(entryMap.values()).sort((left, right) => left.item_name.localeCompare(right.item_name)),
  };
  saveSupplyContext(merged);
  return merged;
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
      itemsToOrderTodayCount: 0,
      itemsToOrderSoonCount: 0,
      stableItemCount: 0,
      reviewItemCount: 0,
      totalMonthlyHouseholdConsumptionValue: 0,
      monthOnMonthSpendingChange: null,
      monthOnMonthConsumptionChange: null,
      inventoryHealthScore: 0,
      householdConsumptionStabilityScore: 0,
      duplicateEntryCount: 0,
    },
    monthly_spend: [],
    category_spend: [],
    item_insights: [],
    possible_anomalies: [],
    dashboard: {
      items_to_order_today: [],
      items_to_order_within_3_days: [],
      items_sufficient_for_more_than_7_days: [],
      unusual_consumption_alerts: [],
      possible_wastage_indicators: [],
      unexplained_depletion_indicators: [],
      items_consumed_faster_than_expected: [],
      items_consumed_slower_than_expected: [],
      items_not_purchased_for_a_long_time: [],
      items_purchased_too_frequently: [],
      items_purchased_in_excess_quantity: [],
      items_usually_ordered_together: [],
      monthly_item_consumption_trend: [],
      monthly_category_consumption_trend: [],
      monthly_category_spending_trend: [],
      top_frequently_purchased_items: [],
      top_highest_spending_items: [],
      items_with_rising_consumption: [],
      items_with_falling_consumption: [],
      seasonal_consumption_pattern: ["Insufficient data for seasonal consumption patterns."],
      guest_event_festival_impact: ["Guest or festival impact cannot be isolated until such events are tagged in the records."],
      perishable_items_requiring_faster_use: [],
      slow_moving_items: [],
      dead_stock_items: [],
      stock_out_risk_items: [],
      overstock_risk_items: [],
      items_requiring_manual_review: [],
      recently_added_purchase_data: [],
      recently_parsed_items_needing_correction: [],
      duplicate_or_suspicious_entries: [],
      items_with_unclear_quantity_or_unit: [],
      items_with_unclear_category: [],
      auto_categorisation: {
        autoEntryCount: 0,
        needsReviewEntryCount: 0,
        autoEntryRate: 0,
        autoItemCount: 0,
        needsReviewItemCount: 0,
      },
    },
  };
}

function normalizedItemKey(value: string) {
  return value.trim().toLowerCase().replace(/\s+/g, " ");
}

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

function roundTo(value: number, digits = 1) {
  return Number(value.toFixed(digits));
}

function percentageChange(previous: number | null, current: number | null) {
  if (previous === null || current === null || previous === 0) {
    return current === 0 ? 0 : null;
  }
  return roundTo(((current - previous) / previous) * 100, 1);
}

function toDate(value: string | null) {
  if (!value) {
    return null;
  }
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime()) ? null : parsed;
}

function dayDifference(left: Date, right: Date) {
  return Math.max(0, Math.round((right.getTime() - left.getTime()) / 86400000));
}

function addDaysToIso(dateValue: string | null, days: number | null) {
  const parsed = toDate(dateValue);
  if (!parsed || days === null) {
    return null;
  }
  const next = new Date(parsed);
  next.setUTCDate(next.getUTCDate() + Math.round(days));
  return next.toISOString().slice(0, 10);
}

function buildQuantityLabel(quantity: number | null, unit: string | null) {
  if (quantity === null) {
    return unit ? `Unclear ${unit}` : "Unclear quantity";
  }
  return `${quantity}${unit ? ` ${unit}` : ""}`;
}

function buildAmountLabel(price: number | null) {
  return price !== null ? `₹${price.toFixed(2)}` : "—";
}

function isPerishableCategory(category: InventoryCategory) {
  return category === "Fruits" || category === "Vegetables" || category === "Dairy";
}

function scoreConfidence(item: {
  purchaseCount: number;
  quantityUnit: string | null;
  reviewStatus: "ok" | "needs_review";
  categoryStatus: "auto" | "needs_review";
  hasConsumptionRate: boolean;
  hasReorderFrequency: boolean;
  hasKnownDate: boolean;
}) {
  let score = 0;
  if (item.purchaseCount >= 6) {
    score += 35;
  } else if (item.purchaseCount >= 3) {
    score += 25;
  } else if (item.purchaseCount >= 2) {
    score += 16;
  } else if (item.purchaseCount >= 1) {
    score += 8;
  }
  if (item.quantityUnit) {
    score += 12;
  }
  if (item.hasConsumptionRate) {
    score += 22;
  } else if (item.hasReorderFrequency) {
    score += 12;
  }
  if (item.hasKnownDate) {
    score += 10;
  }
  if (item.reviewStatus === "ok") {
    score += 11;
  }
  if (item.categoryStatus === "auto") {
    score += 10;
  }
  score = clamp(score, 0, 100);
  const label: ConfidenceLevel =
    score >= 75 ? "High" : score >= 52 ? "Medium" : score >= 28 ? "Low" : "Insufficient data";
  return { score, label };
}

function buildRecentEntryDigest(entry: InventoryEntry): RecentEntryDigest {
  return {
    purchaseId: entry.purchase_id,
    dateOfPurchase: entry.date_of_purchase,
    itemName: entry.item_name,
    category: entry.category,
    quantityLabel: buildQuantityLabel(entry.quantity_purchased, entry.unit_of_measurement),
    amountLabel: buildAmountLabel(entry.price),
    reviewStatus: entry.review_status,
    reviewNotes: entry.review_notes,
    enteredAt: entry.entered_at,
  };
}

function buildItemSuggestions({
  flags,
  entry,
  reorderFrequencyDays,
  stockBucket,
  trend,
}: {
  flags: string[];
  entry: InventoryEntry;
  reorderFrequencyDays: number | null;
  stockBucket: StockBucket;
  trend: ConsumptionTrend;
}) {
  const suggestions: string[] = [];
  if (entry.review_status === "needs_review") {
    suggestions.push("Requires review before relying on the estimate fully.");
  }
  if (stockBucket === "Items to Order Today") {
    suggestions.push("Order now.");
  } else if (stockBucket === "Items to Order Within 3 Days") {
    suggestions.push("Order soon.");
  } else if (stockBucket === "Items Sufficient for More Than 7 Days") {
    suggestions.push("Order later.");
  }
  if (trend === "Higher than usual") {
    suggestions.push("Verify consumption and check for an unusual usage spike.");
  }
  if (trend === "Lower than usual") {
    suggestions.push("Reduce purchase quantity slightly if this slower pace continues.");
  }
  if (flags.includes("possible over-purchase or wastage")) {
    suggestions.push("Reduce purchase quantity or check possible wastage.");
  }
  if (flags.includes("possible unexplained depletion")) {
    suggestions.push("Check unexplained depletion.");
  }
  if (flags.includes("quantity or unit unclear")) {
    suggestions.push("Confirm quantity and unit.");
  }
  if (flags.includes("category unclear")) {
    suggestions.push("Confirm the category.");
  }
  if (reorderFrequencyDays !== null && !suggestions.includes("Order now.") && !suggestions.includes("Order soon.")) {
    suggestions.push(`A reminder roughly every ${Math.round(reorderFrequencyDays)} days would match current history.`);
  }
  if (!suggestions.length) {
    suggestions.push("Continue collecting purchase entries to strengthen the estimate.");
  }
  return suggestions;
}

export function buildInventoryAnalysis(entries: InventoryEntry[]): InventoryAnalysis {
  if (!entries.length) {
    return emptyAnalysis();
  }

  const today = toDate(nowIso().slice(0, 10)) ?? new Date();
  const byItem = new Map<string, InventoryEntry[]>();
  const monthlySpend = new Map<string, number>();
  const monthlyPurchaseCount = new Map<string, number>();
  const categorySpend = new Map<InventoryCategory, number>();
  const categoryTrendMaps = new Map<InventoryCategory, Map<string, { purchases: number; spend: number }>>();
  const groupedPurchases = new Map<string, InventoryEntry[]>();

  for (const entry of entries) {
    const key = normalizedItemKey(entry.item_name);
    const itemEntries = byItem.get(key) ?? [];
    itemEntries.push(entry);
    byItem.set(key, itemEntries);

    if (entry.date_of_purchase) {
      const monthKey = entry.date_of_purchase.slice(0, 7);
      monthlyPurchaseCount.set(monthKey, (monthlyPurchaseCount.get(monthKey) ?? 0) + 1);
      if (entry.price !== null) {
        monthlySpend.set(monthKey, (monthlySpend.get(monthKey) ?? 0) + entry.price);
      }

      const categoryMonthMap = categoryTrendMaps.get(entry.category) ?? new Map<string, { purchases: number; spend: number }>();
      const categoryMonth = categoryMonthMap.get(monthKey) ?? { purchases: 0, spend: 0 };
      categoryMonth.purchases += 1;
      categoryMonth.spend += entry.price ?? 0;
      categoryMonthMap.set(monthKey, categoryMonth);
      categoryTrendMaps.set(entry.category, categoryMonthMap);

      const purchaseGroupKey = `${entry.date_of_purchase}|${entry.vendor_source ?? "unknown"}`;
      const purchaseGroup = groupedPurchases.get(purchaseGroupKey) ?? [];
      purchaseGroup.push(entry);
      groupedPurchases.set(purchaseGroupKey, purchaseGroup);
    }
    if (entry.price !== null) {
      categorySpend.set(entry.category, (categorySpend.get(entry.category) ?? 0) + entry.price);
    }
  }

  const itemInsights: ItemInsight[] = [];
  const possibleAnomalies: InventoryAnalysis["possible_anomalies"] = [];
  const unusualConsumptionAlerts: DashboardAlert[] = [];
  const possibleWastageIndicators: DashboardAlert[] = [];
  const unexplainedDepletionIndicators: DashboardAlert[] = [];

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
    const lastEntry = sortedEntries[sortedEntries.length - 1];
    const lastPurchaseDate = lastEntry.date_of_purchase;
    const lastPurchaseDateValue = toDate(lastPurchaseDate);
    const daysSinceLastPurchase = lastPurchaseDateValue ? dayDifference(lastPurchaseDateValue, today) : null;
    const estimatedDaysOfStockRemaining =
      expectedStockDurationDays !== null && daysSinceLastPurchase !== null
        ? roundTo(Math.max(expectedStockDurationDays - daysSinceLastPurchase, 0), 1)
        : null;

    let averageDailyConsumption: number | null = null;
    let averageWeeklyConsumption: number | null = null;
    let averageConsumptionRate: string | null = null;
    let consumptionBasis: ItemInsight["consumptionBasis"] = "insufficient_data";
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
        averageDailyConsumption = roundTo(totalQuantity / totalDays, 2);
        averageWeeklyConsumption = roundTo((totalQuantity / totalDays) * 7, 2);
        averageConsumptionRate = `${averageDailyConsumption.toFixed(2)} ${sortedEntries[0].unit_of_measurement}/day`;
        consumptionBasis = "actual";
      }
    } else if (
      averageQuantityPurchased !== null &&
      expectedStockDurationDays !== null &&
      expectedStockDurationDays > 0 &&
      sortedEntries[0].unit_of_measurement
    ) {
      averageDailyConsumption = roundTo(averageQuantityPurchased / expectedStockDurationDays, 2);
      averageWeeklyConsumption = roundTo((averageQuantityPurchased / expectedStockDurationDays) * 7, 2);
      averageConsumptionRate = `${averageDailyConsumption.toFixed(2)} ${sortedEntries[0].unit_of_measurement}/day`;
      consumptionBasis = "purchase_history";
    }

    let currentConsumptionTrend: ConsumptionTrend = "Insufficient data";
    let higherThanUsual = false;
    let lowerThanUsual = false;
    if (reorderIntervals.length >= 2) {
      const currentInterval = reorderIntervals[reorderIntervals.length - 1];
      const baselineInterval =
        reorderIntervals.slice(0, -1).reduce((sum, value) => sum + value, 0) / (reorderIntervals.length - 1);
      if (baselineInterval > 0 && currentInterval <= baselineInterval * 0.75) {
        currentConsumptionTrend = "Higher than usual";
        higherThanUsual = true;
      } else if (baselineInterval > 0 && currentInterval >= baselineInterval * 1.25) {
        currentConsumptionTrend = "Lower than usual";
        lowerThanUsual = true;
      } else {
        currentConsumptionTrend = "Normal";
      }
    }

    const flags: string[] = [];
    if (higherThanUsual) {
      flags.push("possible fast consumption");
      possibleAnomalies.push({
        itemName: sortedEntries[0].item_name,
        kind: "possible_fast_consumption",
        severity: "medium",
        message: `${sortedEntries[0].item_name} was reordered sooner than its earlier pattern. Treat this only as a possible anomaly until more context is reviewed.`,
      });
      unusualConsumptionAlerts.push({
        itemName: sortedEntries[0].item_name,
        label: "Higher than usual",
        detail: "Recent replenishment came sooner than the earlier pattern.",
        severity: "medium",
        suggestion: "Verify consumption and check for any unusual demand.",
      });
    }
    if (lowerThanUsual) {
      flags.push("possible slow consumption");
      unusualConsumptionAlerts.push({
        itemName: sortedEntries[0].item_name,
        label: "Lower than usual",
        detail: "Recent replenishment came later than the earlier pattern.",
        severity: "low",
        suggestion: "Reduce the next purchase quantity if this pattern continues.",
      });
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
        possibleWastageIndicators.push({
          itemName: sortedEntries[0].item_name,
          label: "Possible wastage indicator",
          detail: "Latest purchased quantity is much higher than the earlier pattern.",
          severity: "low",
          suggestion: "Reduce purchase quantity or confirm that the larger buy was intentional.",
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
        unexplainedDepletionIndicators.push({
          itemName: sortedEntries[0].item_name,
          label: "Possible unexplained depletion",
          detail: "Actual consumption duration dropped sharply versus earlier records.",
          severity: "medium",
          suggestion: "Verify usage and check whether the record is complete.",
        });
      }
    }
    if (estimatedDaysOfStockRemaining !== null && estimatedDaysOfStockRemaining <= 0) {
      flags.push("stock-out risk");
    }
    if (estimatedDaysOfStockRemaining !== null && estimatedDaysOfStockRemaining > 14 && lowerThanUsual) {
      flags.push("possible overstock risk");
    }
    if (lastEntry.review_status === "needs_review") {
      flags.push("requires review");
    }
    if (!lastEntry.unit_of_measurement) {
      flags.push("quantity or unit unclear");
    }
    if (lastEntry.category_status === "needs_review") {
      flags.push("category unclear");
    }
    if (daysSinceLastPurchase !== null) {
      if (reorderFrequencyDays !== null && daysSinceLastPurchase >= reorderFrequencyDays * 1.5) {
        flags.push("not purchased for a long time");
      }
      if ((reorderFrequencyDays === null && daysSinceLastPurchase >= 90) || (reorderFrequencyDays !== null && daysSinceLastPurchase >= reorderFrequencyDays * 2.5)) {
        flags.push("possible dead stock");
      }
    }

    const monthlyPatternMap = new Map<string, { purchases: number; spend: number }>();
    const monthlyQuantityMap = new Map<string, { purchases: number; spend: number; quantity: number | null; unit: string | null }>();
    for (const entry of sortedEntries) {
      const month = entry.date_of_purchase?.slice(0, 7);
      if (!month) {
        continue;
      }
      const current = monthlyPatternMap.get(month) ?? { purchases: 0, spend: 0 };
      current.purchases += 1;
      current.spend += entry.price ?? 0;
      monthlyPatternMap.set(month, current);

      const hadMonthEntry = monthlyQuantityMap.has(month);
      const monthlyQuantity = monthlyQuantityMap.get(month) ?? {
        purchases: 0,
        spend: 0,
        quantity: 0,
        unit: entry.unit_of_measurement ?? null,
      };
      monthlyQuantity.purchases += 1;
      monthlyQuantity.spend += entry.price ?? 0;
      if (entry.quantity_purchased !== null && monthlyQuantity.quantity !== null) {
        monthlyQuantity.quantity = hadMonthEntry ? monthlyQuantity.quantity + entry.quantity_purchased : entry.quantity_purchased;
      } else {
        monthlyQuantity.quantity = null;
      }
      if (monthlyQuantity.unit && entry.unit_of_measurement && monthlyQuantity.unit !== entry.unit_of_measurement) {
        monthlyQuantity.quantity = null;
        monthlyQuantity.unit = null;
      }
      monthlyQuantityMap.set(month, monthlyQuantity);
    }

    let stockBucket: StockBucket = "Requires Review";
    if (estimatedDaysOfStockRemaining !== null) {
      if (estimatedDaysOfStockRemaining <= 0) {
        stockBucket = "Items to Order Today";
      } else if (estimatedDaysOfStockRemaining <= 3) {
        stockBucket = "Items to Order Within 3 Days";
      } else if (estimatedDaysOfStockRemaining > 7) {
        stockBucket = "Items Sufficient for More Than 7 Days";
      } else {
        stockBucket = "Monitor";
      }
    } else if (lastEntry.review_status === "ok" && reorderFrequencyDays !== null) {
      stockBucket = "Monitor";
    }

    const suggestedReorderQuantity =
      averageQuantityPurchased !== null
        ? roundTo(
            averageQuantityPurchased *
              (currentConsumptionTrend === "Higher than usual"
                ? 1.15
                : currentConsumptionTrend === "Lower than usual"
                  ? 0.9
                  : 1),
            2,
          )
        : null;
    const confidence = scoreConfidence({
      purchaseCount: sortedEntries.length,
      quantityUnit: sortedEntries.find((entry) => entry.unit_of_measurement)?.unit_of_measurement ?? null,
      reviewStatus: lastEntry.review_status,
      categoryStatus: lastEntry.category_status,
      hasConsumptionRate: averageDailyConsumption !== null,
      hasReorderFrequency: reorderFrequencyDays !== null,
      hasKnownDate: Boolean(lastPurchaseDate),
    });

    let reorderPriorityScore: number | null = null;
    if (estimatedDaysOfStockRemaining !== null) {
      if (estimatedDaysOfStockRemaining <= 0) {
        reorderPriorityScore = 95;
      } else if (estimatedDaysOfStockRemaining <= 3) {
        reorderPriorityScore = 78 - estimatedDaysOfStockRemaining * 8;
      } else if (estimatedDaysOfStockRemaining <= 7) {
        reorderPriorityScore = 48 - (estimatedDaysOfStockRemaining - 3) * 5;
      } else {
        reorderPriorityScore = 18;
      }
      if (currentConsumptionTrend === "Higher than usual") {
        reorderPriorityScore += 8;
      }
      if (lastEntry.review_status === "needs_review") {
        reorderPriorityScore -= 6;
      }
      reorderPriorityScore = clamp(Math.round(reorderPriorityScore), 0, 100);
    }

    itemInsights.push({
      itemName: sortedEntries[0].item_name,
      category: sortedEntries[0].category,
      purchaseCount: sortedEntries.length,
      lastPurchaseDate,
      lastPurchasedQuantity: lastEntry.quantity_purchased,
      lastPurchaseSpend: lastEntry.price,
      averageQuantityPurchased,
      quantityUnit: sortedEntries.find((entry) => entry.unit_of_measurement)?.unit_of_measurement ?? null,
      averageDailyConsumption,
      averageWeeklyConsumption,
      averageConsumptionRate,
      averageDailyConsumptionLabel:
        averageDailyConsumption !== null && sortedEntries[0].unit_of_measurement
          ? `${averageDailyConsumption.toFixed(2)} ${sortedEntries[0].unit_of_measurement}/day`
          : null,
      averageWeeklyConsumptionLabel:
        averageWeeklyConsumption !== null && sortedEntries[0].unit_of_measurement
          ? `${averageWeeklyConsumption.toFixed(2)} ${sortedEntries[0].unit_of_measurement}/week`
          : null,
      consumptionBasis,
      expectedStockDurationDays,
      estimatedDaysOfStockRemaining,
      actualConsumptionDurationDays: averageActualDurationDays,
      reorderFrequencyDays,
      normalPurchaseFrequencyDays: reorderFrequencyDays,
      recommendedReorderDate: addDaysToIso(lastPurchaseDate, expectedStockDurationDays),
      suggestedReorderQuantity,
      currentConsumptionTrend,
      reorderPriorityScore,
      stockBucket,
      dataConfidenceScore: confidence.score,
      dataConfidenceLabel: confidence.label,
      daysSinceLastPurchase,
      monthlyPurchasePattern: Array.from(monthlyPatternMap.entries())
        .sort(([left], [right]) => left.localeCompare(right))
        .map(([month, values]) => ({ month, purchases: values.purchases, spend: Number(values.spend.toFixed(2)) })),
      monthlyQuantityPattern: Array.from(monthlyQuantityMap.entries())
        .sort(([left], [right]) => left.localeCompare(right))
        .map(([month, values]) => ({
          month,
          purchases: values.purchases,
          quantity: values.quantity !== null ? roundTo(values.quantity, 2) : null,
          spend: roundTo(values.spend, 2),
          unit: values.unit,
        })),
      flags,
      suggestions: buildItemSuggestions({
        flags,
        entry: lastEntry,
        reorderFrequencyDays,
        stockBucket,
        trend: currentConsumptionTrend,
      }),
      totalSpend: roundTo(sortedEntries.reduce((sum, entry) => sum + (entry.price ?? 0), 0), 2),
    });
  }

  const totalSpend = entries.reduce((sum, entry) => sum + (entry.price ?? 0), 0);
  const currentMonth = nowIso().slice(0, 7);
  const sortedMonthlySpend = Array.from(monthlySpend.entries())
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([month, amount]) => ({ month, amount: Number(amount.toFixed(2)) }));
  const sortedMonthlyPurchaseCounts = Array.from(monthlyPurchaseCount.entries()).sort(([left], [right]) => left.localeCompare(right));
  const currentMonthSpend = Number((monthlySpend.get(currentMonth) ?? 0).toFixed(2));
  const previousMonthSpend =
    sortedMonthlySpend.length >= 2 ? sortedMonthlySpend[sortedMonthlySpend.length - 2].amount : null;
  const currentMonthPurchaseCount = sortedMonthlyPurchaseCounts.length
    ? sortedMonthlyPurchaseCounts[sortedMonthlyPurchaseCounts.length - 1][1]
    : null;
  const previousMonthPurchaseCount =
    sortedMonthlyPurchaseCounts.length >= 2 ? sortedMonthlyPurchaseCounts[sortedMonthlyPurchaseCounts.length - 2][1] : null;

  const duplicateMap = new Map<string, { entry: InventoryEntry; count: number }>();
  for (const entry of entries) {
    const signature = [
      normalizedItemKey(entry.item_name),
      entry.date_of_purchase ?? "unknown-date",
      entry.quantity_purchased ?? "unknown-quantity",
      entry.unit_of_measurement ?? "unknown-unit",
      entry.price ?? "unknown-price",
      entry.vendor_source ?? "unknown-vendor",
    ].join("|");
    const current = duplicateMap.get(signature) ?? { entry, count: 0 };
    current.count += 1;
    duplicateMap.set(signature, current);
  }
  const duplicateEntries = Array.from(duplicateMap.entries())
    .filter(([, value]) => value.count > 1)
    .map(([signature, value]) => ({
      signature,
      itemName: value.entry.item_name,
      dateOfPurchase: value.entry.date_of_purchase,
      quantityLabel: buildQuantityLabel(value.entry.quantity_purchased, value.entry.unit_of_measurement),
      amountLabel: buildAmountLabel(value.entry.price),
      vendorSource: value.entry.vendor_source,
      occurrences: value.count,
    }))
    .sort((left, right) => right.occurrences - left.occurrences || left.itemName.localeCompare(right.itemName));

  const pairCounts = new Map<string, { frequency: number; lastSeenDate: string | null }>();
  for (const [groupKey, purchaseGroup] of groupedPurchases.entries()) {
    const [groupDate] = groupKey.split("|");
    const items = Array.from(new Set(purchaseGroup.map((entry) => entry.item_name))).sort((left, right) => left.localeCompare(right));
    for (let index = 0; index < items.length; index += 1) {
      for (let pairIndex = index + 1; pairIndex < items.length; pairIndex += 1) {
        const pairLabel = `${items[index]} + ${items[pairIndex]}`;
        const current = pairCounts.get(pairLabel) ?? { frequency: 0, lastSeenDate: null };
        current.frequency += 1;
        current.lastSeenDate = groupDate;
        pairCounts.set(pairLabel, current);
      }
    }
  }

  const autoEntryCount = entries.filter((entry) => entry.category_status === "auto").length;
  const needsReviewEntryCount = entries.length - autoEntryCount;
  const itemsToOrderToday = itemInsights
    .filter((item) => item.stockBucket === "Items to Order Today")
    .sort((left, right) => (right.reorderPriorityScore ?? 0) - (left.reorderPriorityScore ?? 0));
  const itemsToOrderSoon = itemInsights
    .filter((item) => item.stockBucket === "Items to Order Within 3 Days")
    .sort((left, right) => (right.reorderPriorityScore ?? 0) - (left.reorderPriorityScore ?? 0));
  const stableItems = itemInsights
    .filter((item) => item.stockBucket === "Items Sufficient for More Than 7 Days")
    .sort((left, right) => (right.estimatedDaysOfStockRemaining ?? 0) - (left.estimatedDaysOfStockRemaining ?? 0));
  const reviewItems = itemInsights
    .filter((item) => item.flags.includes("requires review") || item.dataConfidenceLabel === "Insufficient data")
    .sort((left, right) => left.dataConfidenceScore - right.dataConfidenceScore || left.itemName.localeCompare(right.itemName));
  const risingConsumptionItems = itemInsights.filter((item) => item.currentConsumptionTrend === "Higher than usual");
  const fallingConsumptionItems = itemInsights.filter((item) => item.currentConsumptionTrend === "Lower than usual");
  const itemsWithUnclearQuantityOrUnit = itemInsights.filter((item) => item.flags.includes("quantity or unit unclear"));
  const itemsWithUnclearCategory = itemInsights.filter((item) => item.flags.includes("category unclear"));
  const notPurchasedForLongTime = itemInsights.filter((item) => item.flags.includes("not purchased for a long time"));
  const overstockRiskItems = itemInsights.filter(
    (item) => item.flags.includes("possible over-purchase or wastage") || item.flags.includes("possible overstock risk"),
  );
  const deadStockItems = itemInsights.filter((item) => item.flags.includes("possible dead stock"));
  const perishableNeedsAttention = itemInsights.filter(
    (item) =>
      isPerishableCategory(item.category) &&
      (item.currentConsumptionTrend === "Lower than usual" || item.flags.includes("possible over-purchase or wastage")),
  );
  const stockOutRiskItems = itemInsights.filter((item) => item.flags.includes("stock-out risk"));
  const slowMovingItems = itemInsights.filter(
    (item) => item.currentConsumptionTrend === "Lower than usual" || item.flags.includes("not purchased for a long time"),
  );
  const monthlyItemTrend = itemInsights
    .filter((item) => item.monthlyQuantityPattern.length >= 1)
    .sort((left, right) => right.purchaseCount - left.purchaseCount || right.totalSpend - left.totalSpend)
    .slice(0, 10)
    .map((item) => ({
      itemName: item.itemName,
      unit: item.quantityUnit,
      months: item.monthlyQuantityPattern.map((month) => ({
        month: month.month,
        purchases: month.purchases,
        quantity: month.quantity,
        spend: month.spend,
      })),
    }));

  const monthlyCategoryTrends = Array.from(categoryTrendMaps.entries())
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([category, monthMap]) => ({
      category,
      months: Array.from(monthMap.entries())
        .sort(([left], [right]) => left.localeCompare(right))
        .map(([month, values]) => ({
          month,
          purchases: values.purchases,
          spend: roundTo(values.spend, 2),
        })),
    }));

  const topFrequentlyPurchasedItems = [...itemInsights]
    .sort((left, right) => right.purchaseCount - left.purchaseCount || right.totalSpend - left.totalSpend)
    .slice(0, 10)
    .map((item) => ({
      itemName: item.itemName,
      value: item.purchaseCount,
      label: `${item.purchaseCount} purchases`,
    }));
  const topHighestSpendingItems = [...itemInsights]
    .sort((left, right) => right.totalSpend - left.totalSpend || right.purchaseCount - left.purchaseCount)
    .slice(0, 10)
    .map((item) => ({
      itemName: item.itemName,
      value: item.totalSpend,
      label: `₹${item.totalSpend.toFixed(2)}`,
    }));

  const trendScores = itemInsights.map((item) => {
    if (item.currentConsumptionTrend === "Normal") {
      return 100;
    }
    if (item.currentConsumptionTrend === "Higher than usual" || item.currentConsumptionTrend === "Lower than usual") {
      return 58;
    }
    return 42;
  });
  const inventoryHealthPenalty =
    itemsToOrderToday.length * 18 +
    itemsToOrderSoon.length * 10 +
    reviewItems.length * 7 +
    possibleAnomalies.length * 6 +
    duplicateEntries.length * 5;
  const inventoryHealthScore = clamp(
    Math.round(100 - inventoryHealthPenalty / Math.max(itemInsights.length, 1)),
    0,
    100,
  );
  const householdConsumptionStabilityScore = clamp(
    Math.round(trendScores.reduce((sum, value) => sum + value, 0) / Math.max(trendScores.length, 1)),
    0,
    100,
  );

  return {
    schema_version: SCHEMA_VERSION,
    generated_at: nowIso(),
    overview: {
      purchaseCount: entries.length,
      trackedItemCount: byItem.size,
      totalSpend: Number(totalSpend.toFixed(2)),
      currentMonthSpend,
      itemsToOrderTodayCount: itemsToOrderToday.length,
      itemsToOrderSoonCount: itemsToOrderSoon.length,
      stableItemCount: stableItems.length,
      reviewItemCount: reviewItems.length,
      totalMonthlyHouseholdConsumptionValue: currentMonthSpend,
      monthOnMonthSpendingChange: percentageChange(previousMonthSpend, currentMonthSpend),
      monthOnMonthConsumptionChange: percentageChange(previousMonthPurchaseCount, currentMonthPurchaseCount),
      inventoryHealthScore,
      householdConsumptionStabilityScore,
      duplicateEntryCount: duplicateEntries.length,
    },
    monthly_spend: sortedMonthlySpend,
    category_spend: Array.from(categorySpend.entries())
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([category, amount]) => ({ category, amount: Number(amount.toFixed(2)) })),
    item_insights: itemInsights
      .sort(
        (left, right) =>
          (right.reorderPriorityScore ?? -1) - (left.reorderPriorityScore ?? -1) || left.itemName.localeCompare(right.itemName),
      ),
    possible_anomalies: possibleAnomalies,
    dashboard: {
      items_to_order_today: itemsToOrderToday,
      items_to_order_within_3_days: itemsToOrderSoon,
      items_sufficient_for_more_than_7_days: stableItems,
      unusual_consumption_alerts: unusualConsumptionAlerts,
      possible_wastage_indicators: possibleWastageIndicators,
      unexplained_depletion_indicators: unexplainedDepletionIndicators,
      items_consumed_faster_than_expected: risingConsumptionItems,
      items_consumed_slower_than_expected: fallingConsumptionItems,
      items_not_purchased_for_a_long_time: notPurchasedForLongTime,
      items_purchased_too_frequently: risingConsumptionItems,
      items_purchased_in_excess_quantity: overstockRiskItems,
      items_usually_ordered_together: Array.from(pairCounts.entries())
        .map(([pairLabel, value]) => ({
          pairLabel,
          frequency: value.frequency,
          lastSeenDate: value.lastSeenDate,
        }))
        .sort((left, right) => right.frequency - left.frequency || left.pairLabel.localeCompare(right.pairLabel))
        .slice(0, 10),
      monthly_item_consumption_trend: monthlyItemTrend,
      monthly_category_consumption_trend: monthlyCategoryTrends,
      monthly_category_spending_trend: monthlyCategoryTrends,
      top_frequently_purchased_items: topFrequentlyPurchasedItems,
      top_highest_spending_items: topHighestSpendingItems,
      items_with_rising_consumption: risingConsumptionItems,
      items_with_falling_consumption: fallingConsumptionItems,
      seasonal_consumption_pattern:
        sortedMonthlySpend.length >= 6
          ? [
              `Highest recorded household spend month so far: ${[...sortedMonthlySpend]
                .sort((left, right) => right.amount - left.amount)[0].month}.`,
              "Seasonal interpretation is still low-confidence until a longer history is available.",
            ]
          : ["Insufficient data for seasonal consumption patterns."],
      guest_event_festival_impact: ["No guest, event, or festival tags are recorded yet, so impact cannot be separated from normal purchases."],
      perishable_items_requiring_faster_use: perishableNeedsAttention,
      slow_moving_items: slowMovingItems,
      dead_stock_items: deadStockItems,
      stock_out_risk_items: stockOutRiskItems,
      overstock_risk_items: overstockRiskItems,
      items_requiring_manual_review: reviewItems,
      recently_added_purchase_data: [...entries]
        .sort((left, right) => right.entered_at.localeCompare(left.entered_at))
        .slice(0, 8)
        .map(buildRecentEntryDigest),
      recently_parsed_items_needing_correction: [...entries]
        .filter((entry) => entry.review_status === "needs_review")
        .sort((left, right) => right.entered_at.localeCompare(left.entered_at))
        .slice(0, 8)
        .map(buildRecentEntryDigest),
      duplicate_or_suspicious_entries: duplicateEntries,
      items_with_unclear_quantity_or_unit: itemsWithUnclearQuantityOrUnit,
      items_with_unclear_category: itemsWithUnclearCategory,
      auto_categorisation: {
        autoEntryCount,
        needsReviewEntryCount,
        autoEntryRate: roundTo((autoEntryCount / Math.max(entries.length, 1)) * 100, 1),
        autoItemCount: itemInsights.filter((item) => item.category !== "Needs Review" && item.category !== "Unclear").length,
        needsReviewItemCount: itemInsights.filter((item) => item.category === "Needs Review" || item.category === "Unclear").length,
      },
    },
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

export function buildContextNotes(supplyContext: SupplyContextEntry[]): ContextNote[] {
  return supplyContext
    .filter((entry) => entry.active)
    .map((entry) => {
      const sourceLabel = entry.source_name ?? entry.source_description ?? "another source";
      const quantityLabel =
        entry.quantity_per_day !== null && entry.unit_of_measurement
          ? `${entry.quantity_per_day} ${entry.unit_of_measurement}/day`
          : "daily quantity not yet fully recorded";
      const beneficiaryLabel = entry.beneficiary ? ` for ${entry.beneficiary}` : "";
      const historyQualifier = entry.effective_start_date
        ? `This supply is marked active from ${entry.effective_start_date}.`
        : "Historical purchase-based analysis does not backfill this supply because its start date is not yet recorded.";
      const spendQualifier =
        entry.price !== null
          ? "Price has been recorded separately where available."
          : "Purchase totals and spend on this page do not include this supply because no price has been recorded yet.";

      return {
        context_id: entry.context_id,
        severity: entry.review_status === "needs_review" ? "needs_review" : "info",
        title: `${entry.item_name}: recurring supply context`,
        message: `Additional recurring supply recorded: ${quantityLabel} of ${entry.item_name} from ${sourceLabel}${beneficiaryLabel}. ${historyQualifier} ${spendQualifier}`,
      };
    });
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
  const supplyContext = loadSupplyContext().entries;
  const contextNotes = buildContextNotes(supplyContext);
  saveInventoryLedger(mergedLedger);
  saveInventoryAnalysis(analysis);

  return {
    savedEntries: entries,
    notes,
    ledger: mergedLedger,
    analysis,
    supplyContext,
    contextNotes,
  };
}

export function getInventorySnapshot(): InventorySnapshot {
  const ledger = loadInventoryLedger();
  const analysis = buildInventoryAnalysis(ledger.purchases);
  const supplyContext = loadSupplyContext().entries;
  const contextNotes = buildContextNotes(supplyContext);
  saveInventoryAnalysis(analysis);
  return {
    ledger,
    analysis,
    supplyContext,
    contextNotes,
  };
}
