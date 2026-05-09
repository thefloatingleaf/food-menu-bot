"use client";

import Link from "next/link";
import { FormEvent, useState } from "react";

import type { InventoryAnalysis, InventoryEntry } from "@/lib/inventory";

type InventoryModuleProps = {
  initialEntries: InventoryEntry[];
  initialAnalysis: InventoryAnalysis;
  initialTab: "log" | "analysis";
  accountDisplayName: string;
};

export function InventoryModule({
  initialEntries,
  initialAnalysis,
  initialTab,
  accountDisplayName,
}: InventoryModuleProps) {
  const [entries, setEntries] = useState(initialEntries);
  const [analysis, setAnalysis] = useState(initialAnalysis);
  const [activeTab, setActiveTab] = useState<"log" | "analysis">(initialTab);
  const [rawText, setRawText] = useState("");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
  const [importNotes, setImportNotes] = useState<string[]>([]);

  async function handleImport(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setSaving(true);
    setError("");
    setSuccess("");
    setImportNotes([]);

    try {
      const response = await fetch("/api/inventory", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ rawText }),
      });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.error ?? "Import failed.");
      }
      setEntries(payload.ledger.purchases);
      setAnalysis(payload.analysis);
      setImportNotes(payload.notes ?? []);
      setSuccess(`${payload.savedEntries?.length ?? 0} purchase entr${payload.savedEntries?.length === 1 ? "y" : "ies"} saved.`);
      setRawText("");
      setActiveTab("log");
    } catch (caughtError) {
      setError(caughtError instanceof Error ? caughtError.message : "Import failed.");
    } finally {
      setSaving(false);
    }
  }

  return (
    <main className="inventory-shell">
      <div className="inventory-shell__inner">
        <header className="inventory-header panel stack">
          <div className="badge-row">
            <span className="badge">Private internal module</span>
            <span className="badge">Signed in as {accountDisplayName}</span>
          </div>
          <div className="inventory-header__topline">
            <div className="stack">
              <span className="eyebrow">Household Inventory</span>
              <h1 className="section-title inventory-title">Track purchases, review stock behaviour, and flag only data-backed possible anomalies.</h1>
              <p className="muted">
                Paste raw purchase records as they come. The system parses what it can, auto-detects categories, saves uncertain rows with review flags, and keeps the analysis private and utilitarian.
              </p>
            </div>
            <div className="inventory-header__actions">
              <Link className="button button--secondary" href="/">
                Assessment Home
              </Link>
            </div>
          </div>
          <div className="inventory-tabs" role="tablist" aria-label="Inventory views">
            <button
              className={`inventory-tab ${activeTab === "log" ? "inventory-tab--active" : ""}`}
              type="button"
              role="tab"
              aria-selected={activeTab === "log"}
              onClick={() => setActiveTab("log")}
            >
              Purchase Log
            </button>
            <button
              className={`inventory-tab ${activeTab === "analysis" ? "inventory-tab--active" : ""}`}
              type="button"
              role="tab"
              aria-selected={activeTab === "analysis"}
              onClick={() => setActiveTab("analysis")}
            >
              Consumption &amp; Analysis
            </button>
          </div>
        </header>

        {activeTab === "log" ? (
          <>
            <section className="panel stack">
              <div className="stack">
                <h2 className="section-title">Import Raw Purchase Data</h2>
                <p className="muted">
                  Paste bills, order history, notes, bank narration snippets, or mixed purchase text. The importer will extract what it can and save uncertain rows with `Needs Review` instead of blocking the save.
                </p>
              </div>
              <form className="stack" onSubmit={handleImport}>
                <label className="field" htmlFor="inventory-raw-text">
                  <span>Raw purchase text</span>
                  <textarea
                    className="input inventory-textarea"
                    id="inventory-raw-text"
                    value={rawText}
                    onChange={(event) => setRawText(event.target.value)}
                    placeholder="Example: 09-05-2026 Mango 3 kg Rs 450 from local mandi"
                    rows={8}
                  />
                </label>
                <div className="button-row">
                  <button className="button button--primary" type="submit" disabled={saving || !rawText.trim()}>
                    {saving ? "Saving..." : "Parse and Save"}
                  </button>
                </div>
              </form>
              {error ? <p className="error-text">{error}</p> : null}
              {success ? <p className="success-text">{success}</p> : null}
              {importNotes.length ? (
                <div className="stack">
                  <h3 className="inventory-subtitle">Import notes</h3>
                  <ul className="inventory-notes">
                    {importNotes.map((note) => (
                      <li key={note}>{note}</li>
                    ))}
                  </ul>
                </div>
              ) : null}
            </section>

            <section className="panel stack">
              <div className="inventory-section-heading">
                <div>
                  <h2 className="section-title">Purchase Log</h2>
                  <p className="muted">All saved entries appear here, including rows that still need review.</p>
                </div>
                <span className="badge">{entries.length} entries</span>
              </div>
              <div className="inventory-table-wrap">
                <table className="inventory-table">
                  <thead>
                    <tr>
                      <th>Date</th>
                      <th>Item</th>
                      <th>Category</th>
                      <th>Quantity</th>
                      <th>Unit</th>
                      <th>Amount</th>
                      <th>Vendor / Source</th>
                      <th>Remarks</th>
                    </tr>
                  </thead>
                  <tbody>
                    {entries.length ? (
                      entries.map((entry) => (
                        <tr key={entry.purchase_id}>
                          <td>{entry.date_of_purchase ?? "Needs Review"}</td>
                          <td>
                            <div className="inventory-item-cell">
                              <strong>{entry.item_name}</strong>
                              {entry.review_status === "needs_review" ? (
                                <span className="inventory-inline-note">Review: {entry.review_notes.join(", ")}</span>
                              ) : null}
                            </div>
                          </td>
                          <td>
                            <span className={`inventory-category inventory-category--${entry.category_status}`}>
                              {entry.category}
                            </span>
                          </td>
                          <td>{entry.quantity_purchased ?? "—"}</td>
                          <td>{entry.unit_of_measurement ?? "—"}</td>
                          <td>{entry.price !== null ? `₹${entry.price.toFixed(2)}` : "—"}</td>
                          <td>{entry.vendor_source ?? "—"}</td>
                          <td>{entry.remarks ?? "—"}</td>
                        </tr>
                      ))
                    ) : (
                      <tr>
                        <td colSpan={8} className="inventory-empty-cell">
                          No purchase data has been saved yet.
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            </section>
          </>
        ) : (
          <>
            <section className="inventory-overview-grid">
              <article className="panel panel--dense stack">
                <span className="eyebrow">Purchases</span>
                <strong className="inventory-metric">{analysis.overview.purchaseCount}</strong>
                <p className="muted">Total saved purchase records.</p>
              </article>
              <article className="panel panel--dense stack">
                <span className="eyebrow">Tracked Items</span>
                <strong className="inventory-metric">{analysis.overview.trackedItemCount}</strong>
                <p className="muted">Distinct item groups with analysis.</p>
              </article>
              <article className="panel panel--dense stack">
                <span className="eyebrow">Total Spend</span>
                <strong className="inventory-metric">₹{analysis.overview.totalSpend.toFixed(2)}</strong>
                <p className="muted">Only where price data was available.</p>
              </article>
              <article className="panel panel--dense stack">
                <span className="eyebrow">Current Month Spend</span>
                <strong className="inventory-metric">₹{analysis.overview.currentMonthSpend.toFixed(2)}</strong>
                <p className="muted">Spend recorded in the current calendar month.</p>
              </article>
            </section>

            <section className="panel stack">
              <div className="inventory-section-heading">
                <div>
                  <h2 className="section-title">Consumption &amp; Analysis</h2>
                  <p className="muted">
                    Suggestions and flags below are intentionally conservative. They indicate only what the recorded data may suggest.
                  </p>
                </div>
              </div>
              <div className="inventory-table-wrap">
                <table className="inventory-table">
                  <thead>
                    <tr>
                      <th>Item</th>
                      <th>Avg Consumption Rate</th>
                      <th>Expected Stock Duration</th>
                      <th>Actual Consumption Duration</th>
                      <th>Reorder Frequency</th>
                      <th>Flags</th>
                      <th>Suggestions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {analysis.item_insights.length ? (
                      analysis.item_insights.map((item) => (
                        <tr key={item.itemName}>
                          <td>
                            <div className="inventory-item-cell">
                              <strong>{item.itemName}</strong>
                              <span className="inventory-inline-note">
                                {item.category} • {item.purchaseCount} purchase{item.purchaseCount === 1 ? "" : "s"}
                              </span>
                            </div>
                          </td>
                          <td>{item.averageConsumptionRate ?? "Not enough data"}</td>
                          <td>{item.expectedStockDurationDays !== null ? `${item.expectedStockDurationDays} days` : "Not enough data"}</td>
                          <td>{item.actualConsumptionDurationDays !== null ? `${item.actualConsumptionDurationDays} days` : "Not provided"}</td>
                          <td>{item.reorderFrequencyDays !== null ? `${item.reorderFrequencyDays} days` : "Not enough data"}</td>
                          <td>{item.flags.length ? item.flags.join(", ") : "None"}</td>
                          <td>{item.suggestions.join(" ")}</td>
                        </tr>
                      ))
                    ) : (
                      <tr>
                        <td colSpan={7} className="inventory-empty-cell">
                          No analysis is available until purchase data is added.
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            </section>

            <section className="split inventory-analysis-split">
              <article className="panel stack">
                <h2 className="section-title">Monthly Purchase Pattern</h2>
                <div className="inventory-table-wrap">
                  <table className="inventory-table">
                    <thead>
                      <tr>
                        <th>Month</th>
                        <th>Spend</th>
                      </tr>
                    </thead>
                    <tbody>
                      {analysis.monthly_spend.length ? (
                        analysis.monthly_spend.map((row) => (
                          <tr key={row.month}>
                            <td>{row.month}</td>
                            <td>₹{row.amount.toFixed(2)}</td>
                          </tr>
                        ))
                      ) : (
                        <tr>
                          <td colSpan={2} className="inventory-empty-cell">
                            No monthly spend data yet.
                          </td>
                        </tr>
                      )}
                    </tbody>
                  </table>
                </div>
              </article>

              <article className="panel stack">
                <h2 className="section-title">Possible Anomalies</h2>
                {analysis.possible_anomalies.length ? (
                  <ul className="inventory-anomaly-list">
                    {analysis.possible_anomalies.map((anomaly) => (
                      <li key={`${anomaly.itemName}-${anomaly.kind}`}>
                        <strong>{anomaly.itemName}</strong>
                        <span>{anomaly.message}</span>
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p className="muted">No possible anomalies are currently indicated by the recorded data.</p>
                )}
              </article>
            </section>
          </>
        )}
      </div>
    </main>
  );
}
