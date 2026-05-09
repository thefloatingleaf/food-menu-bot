import fs from "node:fs";
import os from "node:os";
import path from "node:path";

import { afterEach, describe, expect, it, vi } from "vitest";

async function loadModules(tempDir: string) {
  process.env.HOUSEHOLD_PURCHASE_LEDGER_DIR = tempDir;
  vi.resetModules();
  const inventory = await import("./inventory");
  const amorFarm = await import("./amorFarm");
  return { inventory, amorFarm };
}

const JAN_INVOICE_TEXT = `1 Feb, 2026

                                                           Invoice
                                                       Amor Farm
                                                        9838204786
                                                ( 1 Jan, 2026 - 31 Jan, 2026 )

         Date                       Product                    Qty       Unit       Rate(₹)       Time            Amount (₹)
         19 Jan               Amor Farm Fresh Milk             1.0       Litre       70.0          M                 70.0
         20 Jan               Amor Farm Fresh Milk             1.0       Litre       70.0          M                 70.0
         21 Jan               Amor Farm Fresh Milk             1.0       Litre       70.0          M                 70.0
         22 Jan               Amor Farm Fresh Milk             3.0       Litre       70.0          M                210.0
         23 Jan               Amor Farm Fresh Milk             3.0       Litre       70.0          M                210.0
         24 Jan               Amor Farm Fresh Milk             3.0       Litre       70.0          M                210.0
         25 Jan               Amor Farm Fresh Milk             4.0       Litre       70.0          M                280.0
         26 Jan               Amor Farm Fresh Milk             4.0       Litre       70.0          M                280.0
         27 Jan               Amor Farm Fresh Milk             4.0       Litre       70.0          M                280.0
         28 Jan               Amor Farm Fresh Milk             4.0       Litre       70.0          M                280.0
         29 Jan               Amor Farm Fresh Milk             3.0       Litre       70.0          M                210.0
         30 Jan               Amor Farm Fresh Milk             4.0       Litre       70.0          M                280.0
         31 Jan               Amor Farm Fresh Milk             4.0       Litre       70.0          M                280.0

Amor Farm Fresh Milk 39.0 Litre

Payment Mode Details                                                                         (+) Total Buy (₹)              2730.0
`;

const FEB_INVOICE_TEXT = `2 Mar, 2026

                                                             Invoice
                                                         Amor Farm
                                                          9838204786
                                                  ( 1 Feb, 2026 - 28 Feb, 2026 )

         Date                           Product                      Qty       Unit      Rate(₹)       Time          Amount (₹)
         1 Feb                    Amor Farm Fresh Milk                 4.0     Litre       70.0            M           280.0
         2 Feb                    Amor Farm Fresh Milk                 4.0     Litre       70.0            M           280.0
         3 Feb                    Amor Farm Fresh Milk                 4.0     Litre       70.0            M           280.0
         3 Feb                Amor Cultured Cow Ghee                   1.0      Kg       1800.0            M          1800.0
         4 Feb                    Amor Farm Fresh Milk                 4.0     Litre       70.0            M           280.0
         5 Feb                    Amor Farm Fresh Milk                 3.0     Litre       70.0            M           210.0

Amor Farm Fresh Milk 69.0 Litre         Amor Cultured Cow Ghee 1.0 kg

Payment Mode Details                                                                            (+) Total Buy (₹)              6630.0
`;

afterEach(() => {
  delete process.env.HOUSEHOLD_PURCHASE_LEDGER_DIR;
});

describe("Amor Farm PDF import support", () => {
  it("parses milk invoice lines into deterministic inventory entries", async () => {
    const { amorFarm } = await loadModules(fs.mkdtempSync(path.join(os.tmpdir(), "amor-farm-")));
    const parsed = amorFarm.parseAmorFarmInvoiceText(JAN_INVOICE_TEXT, {
      milkOnly: true,
      sourceFile: "/tmp/january.pdf",
    });

    expect(parsed.entries).toHaveLength(13);
    expect(parsed.invoice_period_start).toBe("2026-01-01");
    expect(parsed.invoice_period_end).toBe("2026-01-31");
    expect(parsed.invoice_total).toBe(2730);
    expect(parsed.entries[0]).toMatchObject({
      date_of_purchase: "2026-01-19",
      item_name: "Amor Farm Fresh Milk",
      quantity_purchased: 1,
      unit_of_measurement: "litre",
      price: 70,
      vendor_source: "Amor Farm",
    });
    expect(parsed.entries.at(-1)).toMatchObject({
      date_of_purchase: "2026-01-31",
      quantity_purchased: 4,
      price: 280,
    });
  });

  it("can filter non-milk lines out of mixed vendor invoices", async () => {
    const { amorFarm } = await loadModules(fs.mkdtempSync(path.join(os.tmpdir(), "amor-farm-")));
    const parsed = amorFarm.parseAmorFarmInvoiceText(FEB_INVOICE_TEXT, {
      milkOnly: true,
      sourceFile: "/tmp/february.pdf",
    });

    expect(parsed.entries).toHaveLength(5);
    expect(parsed.filtered_out_count).toBe(1);
    expect(parsed.entries.every((entry) => entry.item_name.includes("Milk"))).toBe(true);
  });

  it("skips already imported invoice lines on repeated imports", async () => {
    const { amorFarm } = await loadModules(fs.mkdtempSync(path.join(os.tmpdir(), "amor-farm-")));
    const pdfPath = path.join(os.tmpdir(), `amor-${Date.now()}.pdf`);

    const first = amorFarm.importAmorFarmPdfFiles([pdfPath], {
      milkOnly: true,
      extractText: () => JAN_INVOICE_TEXT,
    });
    const second = amorFarm.importAmorFarmPdfFiles([pdfPath], {
      milkOnly: true,
      extractText: () => JAN_INVOICE_TEXT,
    });

    expect(first.added_entries).toHaveLength(13);
    expect(second.added_entries).toHaveLength(0);
    expect(second.skipped_existing_count).toBe(13);
  });
});
