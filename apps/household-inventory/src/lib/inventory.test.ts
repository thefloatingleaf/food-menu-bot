import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import { afterEach, describe, expect, it, vi } from "vitest";

async function loadInventoryModule(tempDir: string) {
  process.env.HOUSEHOLD_PURCHASE_LEDGER_DIR = tempDir;
  vi.resetModules();
  return import("./inventory");
}

afterEach(() => {
  delete process.env.HOUSEHOLD_PURCHASE_LEDGER_DIR;
});

describe("inventory parsing and analysis", () => {
  it("auto-detects categories from item names", async () => {
    const inventory = await loadInventoryModule(fs.mkdtempSync(path.join(os.tmpdir(), "inventory-")));
    expect(inventory.detectCategory("Mango")).toEqual({ category: "Fruits", status: "auto" });
    expect(inventory.detectCategory("Floor cleaner refill")).toEqual({
      category: "Cleaning Items",
      status: "auto",
    });
  });

  it("parses a free-form raw purchase line and saves uncertain details with review flags", async () => {
    const inventory = await loadInventoryModule(fs.mkdtempSync(path.join(os.tmpdir(), "inventory-")));
    const parsed = inventory.parseRawInventoryText("09-05-2026 Mango 3 kg Rs 450 from local mandi");

    expect(parsed.entries).toHaveLength(1);
    expect(parsed.entries[0]).toMatchObject({
      date_of_purchase: "2026-05-09",
      item_name: "Mango",
      category: "Fruits",
      quantity_purchased: 3,
      unit_of_measurement: "kg",
      price: 450,
    });
  });

  it("keeps an entry even when category and date are unclear", async () => {
    const inventory = await loadInventoryModule(fs.mkdtempSync(path.join(os.tmpdir(), "inventory-")));
    const parsed = inventory.parseRawInventoryText("mystery refill pack amount 220");

    expect(parsed.entries).toHaveLength(1);
    expect(parsed.entries[0].review_status).toBe("needs_review");
    expect(parsed.entries[0].category_status).toBe("needs_review");
  });

  it("builds only possible-anomaly wording in analysis output", async () => {
    const inventory = await loadInventoryModule(fs.mkdtempSync(path.join(os.tmpdir(), "inventory-")));
    const entries = [
      inventory.normalizeInventoryEntry({
        date_of_purchase: "2026-05-01",
        item_name: "Milk",
        quantity_purchased: 2,
        unit_of_measurement: "litre",
        price: 120,
        actual_consumption_period: { value: 4, unit: "days" },
      }),
      inventory.normalizeInventoryEntry({
        date_of_purchase: "2026-05-05",
        item_name: "Milk",
        quantity_purchased: 2,
        unit_of_measurement: "litre",
        price: 120,
        actual_consumption_period: { value: 4, unit: "days" },
      }),
      inventory.normalizeInventoryEntry({
        date_of_purchase: "2026-05-09",
        item_name: "Milk",
        quantity_purchased: 2,
        unit_of_measurement: "litre",
        price: 120,
        actual_consumption_period: { value: 4, unit: "days" },
      }),
      inventory.normalizeInventoryEntry({
        date_of_purchase: "2026-05-11",
        item_name: "Milk",
        quantity_purchased: 5,
        unit_of_measurement: "litre",
        price: 300,
        actual_consumption_period: { value: 1, unit: "days" },
      }),
    ];

    const analysis = inventory.buildInventoryAnalysis(entries);
    expect(analysis.possible_anomalies.length).toBeGreaterThan(0);
    expect(
      analysis.possible_anomalies.every(
        (item) => item.message.toLowerCase().includes("possible") && !item.message.toLowerCase().includes("confirmed"),
      ),
    ).toBe(true);
  });
});
