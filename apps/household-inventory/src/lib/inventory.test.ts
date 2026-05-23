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
    expect(inventory.detectCategory("Amul Salted Butter")).toEqual({
      category: "Dairy",
      status: "auto",
    });
    expect(inventory.detectCategory("THF Multigrain Bread")).toEqual({
      category: "Groceries",
      status: "auto",
    });
    expect(inventory.detectCategory("Heinz Tomato Ketchup")).toEqual({
      category: "Groceries",
      status: "auto",
    });
    expect(inventory.detectCategory("Rajdhani Maida")).toEqual({
      category: "Groceries",
      status: "auto",
    });
    expect(inventory.detectCategory("Tata Sampann Meethi Imli Khajur Chutney Dip")).toEqual({
      category: "Groceries",
      status: "auto",
    });
    expect(inventory.detectCategory("Vestta Vintage Arabic Designed Paper Gift Bag With Tags")).toEqual({
      category: "Household Consumables",
      status: "auto",
    });
    expect(inventory.detectCategory("Mee Mee Mild Baby Laundry Detergent Refill")).toEqual({
      category: "Baby Items",
      status: "auto",
    });
    expect(inventory.detectCategory("Mountain Dew Soft Drink")).toEqual({
      category: "Groceries",
      status: "auto",
    });
    expect(inventory.detectCategory("Haldiram's Bhujia")).toEqual({
      category: "Groceries",
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

  it("stores recurring supply context separately from purchase rows and exposes an analysis note", async () => {
    const inventory = await loadInventoryModule(fs.mkdtempSync(path.join(os.tmpdir(), "inventory-")));
    inventory.upsertSupplyContextEntries([
      {
        context_id: "navishti-special-cow-milk",
        item_name: "Milk",
        quantity_per_day: 0.5,
        unit_of_measurement: "litre",
        beneficiary: "Navishti",
        source_description: "special cows from another place",
        active: true,
        remarks: "User reported this as a daily non-purchase milk source.",
      },
    ]);

    const snapshot = inventory.getInventorySnapshot();

    expect(snapshot.ledger.purchases).toHaveLength(0);
    expect(snapshot.supplyContext).toHaveLength(1);
    expect(snapshot.supplyContext[0].review_status).toBe("needs_review");
    expect(snapshot.contextNotes).toHaveLength(1);
    expect(snapshot.contextNotes[0].message).toContain("0.5 litre/day");
    expect(snapshot.contextNotes[0].message).toContain("Navishti");
    expect(snapshot.contextNotes[0].message).toContain("purchase");
  });
});
