"use client";

import Link from "next/link";
import { FormEvent, Fragment, useState } from "react";

import type {
  ContextNote,
  DashboardAlert,
  InventoryAnalysis,
  InventoryEntry,
  ItemInsight,
  PairedPurchaseInsight,
  RankedItem,
  RecentEntryDigest,
} from "@/lib/inventory";

type InventoryModuleProps = {
  initialEntries: InventoryEntry[];
  initialAnalysis: InventoryAnalysis;
  initialContextNotes: ContextNote[];
};

function formatDate(value: string | null) {
  if (!value) {
    return "Insufficient data";
  }
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) {
    return value;
  }
  return parsed.toLocaleDateString("en-IN", {
    day: "2-digit",
    month: "short",
    year: "numeric",
  });
}

function formatPercent(value: number | null) {
  if (value === null) {
    return "Insufficient data";
  }
  const prefix = value > 0 ? "+" : "";
  return `${prefix}${value.toFixed(1)}%`;
}

function formatQuantity(value: number | null, unit: string | null) {
  if (value === null) {
    return "Insufficient data";
  }
  return `${value} ${unit ?? ""}`.trim();
}

function formatDays(value: number | null) {
  return value === null ? "Insufficient data" : `${value} day${value === 1 ? "" : "s"}`;
}

function ScoreCard({
  label,
  value,
  note,
  tone = "calm",
}: {
  label: string;
  value: string;
  note: string;
  tone?: "calm" | "urgent" | "watch" | "good";
}) {
  return (
    <article className={`score-card score-card--${tone}`}>
      <span className="eyebrow">{label}</span>
      <strong className="score-card__value">{value}</strong>
      <p className="muted">{note}</p>
    </article>
  );
}

function PriorityLane({
  title,
  description,
  items,
  tone,
}: {
  title: string;
  description: string;
  items: ItemInsight[];
  tone: "urgent" | "watch" | "good";
}) {
  return (
    <section className={`panel lane lane--${tone}`}>
      <div className="stack lane__header">
        <div className="lane__title-row">
          <div>
            <span className="eyebrow">{title}</span>
            <h2 className="section-heading">{title}</h2>
          </div>
          <span className="pill">{items.length}</span>
        </div>
        <p className="muted">{description}</p>
      </div>
      {items.length ? (
        <ul className="compact-list">
          {items.slice(0, 8).map((item) => (
            <li key={item.itemName} className="compact-list__item">
              <div>
                <strong>{item.itemName}</strong>
                <span className="compact-list__meta">
                  {item.suggestedReorderQuantity !== null
                    ? `Suggested reorder: ${formatQuantity(item.suggestedReorderQuantity, item.quantityUnit)}`
                    : "Suggested reorder: Insufficient data"}
                </span>
              </div>
              <div className="compact-list__stat">
                <strong>{formatDays(item.estimatedDaysOfStockRemaining)}</strong>
                <span>{item.currentConsumptionTrend}</span>
              </div>
            </li>
          ))}
        </ul>
      ) : (
        <p className="muted">No items are currently in this bucket.</p>
      )}
    </section>
  );
}

function AlertGroup({
  title,
  items,
  emptyText,
}: {
  title: string;
  items: DashboardAlert[];
  emptyText: string;
}) {
  return (
    <section className="panel stack">
      <div className="inventory-section-heading">
        <h2 className="section-heading">{title}</h2>
        <span className="pill">{items.length}</span>
      </div>
      {items.length ? (
        <ul className="signal-list">
          {items.map((item) => (
            <li key={`${item.itemName}-${item.label}`} className={`signal signal--${item.severity}`}>
              <div className="signal__title-row">
                <strong>{item.itemName}</strong>
                <span className="signal__label">{item.label}</span>
              </div>
              <p>{item.detail}</p>
              <span className="signal__suggestion">{item.suggestion}</span>
            </li>
          ))}
        </ul>
      ) : (
        <p className="muted">{emptyText}</p>
      )}
    </section>
  );
}

function ItemChipList({
  title,
  items,
  emptyText,
  note,
}: {
  title: string;
  items: ItemInsight[];
  emptyText: string;
  note?: string;
}) {
  return (
    <section className="panel stack">
      <div className="inventory-section-heading">
        <div>
          <h2 className="section-heading">{title}</h2>
          {note ? <p className="muted">{note}</p> : null}
        </div>
        <span className="pill">{items.length}</span>
      </div>
      {items.length ? (
        <div className="chip-grid">
          {items.slice(0, 10).map((item) => (
            <article key={item.itemName} className="item-chip">
              <div className="item-chip__row">
                <strong>{item.itemName}</strong>
                <span className={`status-dot status-dot--${item.dataConfidenceLabel.toLowerCase().replace(/\s+/g, "-")}`} />
              </div>
              <p>{item.currentConsumptionTrend}</p>
              <span>
                {item.estimatedDaysOfStockRemaining !== null
                  ? `${item.estimatedDaysOfStockRemaining} days left`
                  : item.dataConfidenceLabel}
              </span>
            </article>
          ))}
        </div>
      ) : (
        <p className="muted">{emptyText}</p>
      )}
    </section>
  );
}

function RankedList({
  title,
  items,
  formatter,
}: {
  title: string;
  items: RankedItem[];
  formatter?: (value: number, label: string) => string;
}) {
  return (
    <section className="panel stack">
      <div className="inventory-section-heading">
        <h2 className="section-heading">{title}</h2>
        <span className="pill">{items.length}</span>
      </div>
      {items.length ? (
        <ol className="ranked-list">
          {items.map((item) => (
            <li key={item.itemName} className="ranked-list__item">
              <span className="ranked-list__name">{item.itemName}</span>
              <strong>{formatter ? formatter(item.value, item.label) : item.label}</strong>
            </li>
          ))}
        </ol>
      ) : (
        <p className="muted">Insufficient data.</p>
      )}
    </section>
  );
}

function RecentEntries({
  title,
  items,
  emptyText,
}: {
  title: string;
  items: RecentEntryDigest[];
  emptyText: string;
}) {
  return (
    <section className="panel stack">
      <div className="inventory-section-heading">
        <h2 className="section-heading">{title}</h2>
        <span className="pill">{items.length}</span>
      </div>
      {items.length ? (
        <ul className="recent-list">
          {items.map((item) => (
            <li key={item.purchaseId} className="recent-list__item">
              <div>
                <strong>{item.itemName}</strong>
                <span className="compact-list__meta">
                  {formatDate(item.dateOfPurchase)} • {item.category}
                </span>
              </div>
              <div className="recent-list__stat">
                <strong>{item.quantityLabel}</strong>
                <span>{item.amountLabel}</span>
              </div>
            </li>
          ))}
        </ul>
      ) : (
        <p className="muted">{emptyText}</p>
      )}
    </section>
  );
}

function PairingList({ items }: { items: PairedPurchaseInsight[] }) {
  return (
    <section className="panel stack">
      <div className="inventory-section-heading">
        <h2 className="section-heading">Items Usually Ordered Together</h2>
        <span className="pill">{items.length}</span>
      </div>
      {items.length ? (
        <ul className="ranked-list">
          {items.map((item) => (
            <li key={item.pairLabel} className="ranked-list__item">
              <div>
                <strong>{item.pairLabel}</strong>
                <span className="compact-list__meta">Last seen {formatDate(item.lastSeenDate)}</span>
              </div>
              <strong>{item.frequency} times</strong>
            </li>
          ))}
        </ul>
      ) : (
        <p className="muted">Not enough grouped order history yet.</p>
      )}
    </section>
  );
}

function TrendPanel({
  title,
  rows,
  emptyText,
}: {
  title: string;
  rows: Array<{
    label: string;
    details: string;
    points: Array<{ month: string; value: string }>;
  }>;
  emptyText: string;
}) {
  return (
    <section className="panel stack">
      <div className="inventory-section-heading">
        <h2 className="section-heading">{title}</h2>
        <span className="pill">{rows.length}</span>
      </div>
      {rows.length ? (
        <div className="trend-grid">
          {rows.map((row) => (
            <article key={row.label} className="trend-card">
              <strong>{row.label}</strong>
              <span className="compact-list__meta">{row.details}</span>
              <div className="trend-points">
                {row.points.map((point) => (
                  <Fragment key={`${row.label}-${point.month}`}>
                    <span>{point.month}</span>
                    <strong>{point.value}</strong>
                  </Fragment>
                ))}
              </div>
            </article>
          ))}
        </div>
      ) : (
        <p className="muted">{emptyText}</p>
      )}
    </section>
  );
}

function ContextNotes({ notes }: { notes: ContextNote[] }) {
  if (!notes.length) {
    return null;
  }

  return (
    <section className="panel stack">
      <div className="inventory-section-heading">
        <div>
          <h2 className="section-heading">Known Supply & Usage Context</h2>
          <p className="muted">These notes keep the decision screen from over-reading purchase gaps or missing practical yield facts.</p>
        </div>
        <span className="pill">{notes.length}</span>
      </div>
      <ul className="signal-list">
        {notes.map((note) => (
          <li key={note.context_id} className={`signal signal--${note.severity === "needs_review" ? "medium" : "low"}`}>
            <div className="signal__title-row">
              <strong>{note.title}</strong>
              <span className="signal__label">{note.severity === "needs_review" ? "Requires review" : "Context"}</span>
            </div>
            <p>{note.message}</p>
          </li>
        ))}
      </ul>
    </section>
  );
}

export function InventoryModule({
  initialEntries,
  initialAnalysis,
  initialContextNotes,
}: InventoryModuleProps) {
  const [entries, setEntries] = useState(initialEntries);
  const [analysis, setAnalysis] = useState(initialAnalysis);
  const [contextNotes, setContextNotes] = useState(initialContextNotes);
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
      setContextNotes(payload.contextNotes ?? []);
      setImportNotes(payload.notes ?? []);
      setSuccess(`${payload.savedEntries?.length ?? 0} purchase entr${payload.savedEntries?.length === 1 ? "y" : "ies"} saved.`);
      setRawText("");
    } catch (caughtError) {
      setError(caughtError instanceof Error ? caughtError.message : "Import failed.");
    } finally {
      setSaving(false);
    }
  }

  const reviewQueueTotal =
    analysis.dashboard.items_requiring_manual_review.length +
    analysis.dashboard.items_with_unclear_quantity_or_unit.length +
    analysis.dashboard.items_with_unclear_category.length;

  return (
    <main className="inventory-shell">
      <div className="inventory-shell__inner">
        <header className="inventory-header">
          <div className="inventory-header__copy stack">
            <div className="badge-row">
              <span className="badge">Private internal module</span>
              <span className="badge">Separate from VPK</span>
              <span className="badge">Decision-oriented summary</span>
            </div>
            <div className="stack">
              <span className="eyebrow">Household Inventory</span>
              <h1 className="inventory-title">What is being consumed, what must be ordered, and what requires review.</h1>
              <p className="muted">
                This landing page is intentionally compact. It shows only the signals that help household decisions: consumption pace,
                reorder timing, stock-risk, review queues, and unusual patterns that deserve a closer look.
              </p>
            </div>
            <div className="button-row">
              <Link className="button button--primary" href="/purchase-log">
                View Detailed Purchase Log
              </Link>
              <a className="button button--secondary" href="#item-decision-matrix">
                Jump to Item Decision Matrix
              </a>
            </div>
          </div>

          <aside className="inventory-import panel stack">
            <div className="stack">
              <span className="eyebrow">Add New Purchase Data</span>
              <h2 className="section-heading">Paste raw purchase text</h2>
              <p className="muted">
                Bills, notes, order history, bank narration, or copied app text can all be pasted here. Unclear rows are still saved with a
                review flag.
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
                  rows={6}
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
              <ul className="compact-list compact-list--notes">
                {importNotes.map((note) => (
                  <li key={note}>{note}</li>
                ))}
              </ul>
            ) : null}
          </aside>
        </header>

        <section className="score-grid">
          <ScoreCard
            label="Inventory Health Score"
            value={`${analysis.overview.inventoryHealthScore}/100`}
            note="Higher means fewer immediate reorder risks, fewer review gaps, and fewer anomaly signals."
            tone={analysis.overview.inventoryHealthScore < 50 ? "urgent" : analysis.overview.inventoryHealthScore < 75 ? "watch" : "good"}
          />
          <ScoreCard
            label="Household Consumption Stability"
            value={`${analysis.overview.householdConsumptionStabilityScore}/100`}
            note="A calmer score means recent buying cadence is closer to the normal pattern."
            tone={analysis.overview.householdConsumptionStabilityScore < 55 ? "watch" : "good"}
          />
          <ScoreCard
            label="Monthly Household Consumption Value"
            value={`₹${analysis.overview.totalMonthlyHouseholdConsumptionValue.toFixed(2)}`}
            note={`Month-on-month spending change: ${formatPercent(analysis.overview.monthOnMonthSpendingChange)}`}
          />
          <ScoreCard
            label="Auto-categorisation / Review Queue"
            value={`${analysis.dashboard.auto_categorisation.autoEntryRate}% auto`}
            note={`${reviewQueueTotal} item signals currently need review or correction.`}
            tone={reviewQueueTotal > 0 ? "watch" : "good"}
          />
        </section>

        <section className="priority-grid">
          <PriorityLane
            title="Items to Order Today"
            description="Estimated to be at or beyond the normal replenishment point."
            items={analysis.dashboard.items_to_order_today}
            tone="urgent"
          />
          <PriorityLane
            title="Items to Order Within 3 Days"
            description="Likely to need replenishment soon if the current pattern holds."
            items={analysis.dashboard.items_to_order_within_3_days}
            tone="watch"
          />
          <PriorityLane
            title="Items Sufficient for More Than 7 Days"
            description="Recorded history suggests these items are comfortable for now."
            items={analysis.dashboard.items_sufficient_for_more_than_7_days}
            tone="good"
          />
        </section>

        <ContextNotes notes={contextNotes} />

        <section className="panel stack" id="item-decision-matrix">
          <div className="inventory-section-heading">
            <div>
              <h2 className="section-heading">Item Decision Matrix</h2>
              <p className="muted">
                This is the main working surface. It summarizes average daily and weekly consumption, days of stock remaining, reorder date,
                last purchase details, normal frequency, trend, confidence, and the next practical suggestion for each item.
              </p>
            </div>
            <span className="pill">{analysis.item_insights.length} items</span>
          </div>
          <div className="inventory-table-wrap">
            <table className="inventory-table inventory-table--summary">
              <thead>
                <tr>
                  <th>Item</th>
                  <th>Daily / Weekly Consumption</th>
                  <th>Days Remaining</th>
                  <th>Recommended Reorder</th>
                  <th>Last Purchase</th>
                  <th>Normal Frequency</th>
                  <th>Trend</th>
                  <th>Priority</th>
                  <th>Confidence</th>
                  <th>Suggestion</th>
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
                      <td>
                        <div className="metric-stack">
                          <strong>{item.averageDailyConsumptionLabel ?? "Insufficient data"}</strong>
                          <span>{item.averageWeeklyConsumptionLabel ?? "Insufficient data"}</span>
                        </div>
                      </td>
                      <td>
                        <div className="metric-stack">
                          <strong>{formatDays(item.estimatedDaysOfStockRemaining)}</strong>
                          <span>{item.stockBucket}</span>
                        </div>
                      </td>
                      <td>
                        <div className="metric-stack">
                          <strong>{formatDate(item.recommendedReorderDate)}</strong>
                          <span>
                            {item.suggestedReorderQuantity !== null
                              ? `${item.suggestedReorderQuantity} ${item.quantityUnit ?? ""}`.trim()
                              : "Insufficient data"}
                          </span>
                        </div>
                      </td>
                      <td>
                        <div className="metric-stack">
                          <strong>{formatDate(item.lastPurchaseDate)}</strong>
                          <span>
                            {item.lastPurchasedQuantity !== null
                              ? `${item.lastPurchasedQuantity} ${item.quantityUnit ?? ""}`.trim()
                              : "Insufficient data"}
                          </span>
                        </div>
                      </td>
                      <td>
                        <div className="metric-stack">
                          <strong>{formatDays(item.normalPurchaseFrequencyDays)}</strong>
                          <span>{item.daysSinceLastPurchase !== null ? `${item.daysSinceLastPurchase} days since last purchase` : "—"}</span>
                        </div>
                      </td>
                      <td>
                        <span className={`trend-badge trend-badge--${item.currentConsumptionTrend.toLowerCase().replace(/\s+/g, "-")}`}>
                          {item.currentConsumptionTrend}
                        </span>
                      </td>
                      <td>
                        <div className="metric-stack">
                          <strong>{item.reorderPriorityScore !== null ? `${item.reorderPriorityScore}/100` : "—"}</strong>
                          <span>{item.stockBucket}</span>
                        </div>
                      </td>
                      <td>
                        <div className="metric-stack">
                          <strong>{item.dataConfidenceLabel}</strong>
                          <span>{item.dataConfidenceScore}/100</span>
                        </div>
                      </td>
                      <td>{item.suggestions[0] ?? "Continue recording data."}</td>
                    </tr>
                  ))
                ) : (
                  <tr>
                    <td colSpan={10} className="inventory-empty-cell">
                      No item-level analysis is available yet.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </section>

        <section className="analysis-grid">
          <AlertGroup
            title="Unusual Consumption Alerts"
            items={analysis.dashboard.unusual_consumption_alerts}
            emptyText="No unusual consumption alerts are currently indicated."
          />
          <AlertGroup
            title="Possible Wastage Indicators"
            items={analysis.dashboard.possible_wastage_indicators}
            emptyText="No possible wastage indicators are currently suggested."
          />
          <AlertGroup
            title="Unexplained Depletion Indicators"
            items={analysis.dashboard.unexplained_depletion_indicators}
            emptyText="No unexplained depletion indicators are currently suggested."
          />
        </section>

        <section className="analysis-grid">
          <ItemChipList
            title="Stock-out Risk Items"
            items={analysis.dashboard.stock_out_risk_items}
            emptyText="No stock-out risk is currently indicated."
          />
          <ItemChipList
            title="Overstock Risk Items"
            items={analysis.dashboard.overstock_risk_items}
            emptyText="No overstock risk is currently indicated."
          />
          <ItemChipList
            title="Slow-moving / Possible Dead Stock"
            items={[...analysis.dashboard.slow_moving_items, ...analysis.dashboard.dead_stock_items].filter(
              (item, index, array) => array.findIndex((candidate) => candidate.itemName === item.itemName) === index,
            )}
            emptyText="No slow-moving or possible dead-stock pattern is currently indicated."
          />
        </section>

        <section className="analysis-grid">
          <ItemChipList
            title="Items Requiring Manual Review"
            items={analysis.dashboard.items_requiring_manual_review}
            emptyText="No manual review queue at the moment."
            note="These items still have insufficient detail, weak confidence, or unresolved parsing ambiguity."
          />
          <ItemChipList
            title="Items with Unclear Quantity or Unit"
            items={analysis.dashboard.items_with_unclear_quantity_or_unit}
            emptyText="No unclear quantity or unit issue is currently detected."
          />
          <ItemChipList
            title="Items with Unclear Category"
            items={analysis.dashboard.items_with_unclear_category}
            emptyText="No unclear category issue is currently detected."
          />
        </section>

        <section className="analysis-grid">
          <ItemChipList
            title="Items with Rising Consumption"
            items={analysis.dashboard.items_with_rising_consumption}
            emptyText="No item is currently trending upward versus its earlier pattern."
          />
          <ItemChipList
            title="Items with Falling Consumption"
            items={analysis.dashboard.items_with_falling_consumption}
            emptyText="No item is currently trending downward versus its earlier pattern."
          />
          <ItemChipList
            title="Perishable Items Requiring Faster Use"
            items={analysis.dashboard.perishable_items_requiring_faster_use}
            emptyText="No perishable item currently looks slow enough to require faster use."
          />
        </section>

        <section className="analysis-grid">
          <RecentEntries
            title="Recently Added Purchase Data"
            items={analysis.dashboard.recently_added_purchase_data}
            emptyText="No recent purchase additions yet."
          />
          <RecentEntries
            title="Recently Parsed Items Needing Correction"
            items={analysis.dashboard.recently_parsed_items_needing_correction}
            emptyText="No recent parsed corrections are waiting."
          />
          <section className="panel stack">
            <div className="inventory-section-heading">
              <h2 className="section-heading">Duplicate or Suspicious Entries</h2>
              <span className="pill">{analysis.dashboard.duplicate_or_suspicious_entries.length}</span>
            </div>
            {analysis.dashboard.duplicate_or_suspicious_entries.length ? (
              <ul className="ranked-list">
                {analysis.dashboard.duplicate_or_suspicious_entries.map((entry) => (
                  <li key={entry.signature} className="ranked-list__item">
                    <div>
                      <strong>{entry.itemName}</strong>
                      <span className="compact-list__meta">
                        {formatDate(entry.dateOfPurchase)} • {entry.quantityLabel} • {entry.amountLabel}
                      </span>
                    </div>
                    <strong>{entry.occurrences} rows</strong>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="muted">No duplicate or suspicious entry pattern is currently detected.</p>
            )}
          </section>
        </section>

        <section className="analysis-grid">
          <RankedList title="Top 10 Most Frequently Purchased Items" items={analysis.dashboard.top_frequently_purchased_items} />
          <RankedList
            title="Top 10 Highest Spending Items"
            items={analysis.dashboard.top_highest_spending_items}
            formatter={(value) => `₹${value.toFixed(2)}`}
          />
          <PairingList items={analysis.dashboard.items_usually_ordered_together} />
        </section>

        <section className="analysis-grid analysis-grid--wide">
          <TrendPanel
            title="Monthly Item-wise Consumption Trend"
            emptyText="Not enough item history yet."
            rows={analysis.dashboard.monthly_item_consumption_trend.slice(0, 6).map((item) => ({
              label: item.itemName,
              details: item.unit ? `Recorded in ${item.unit}` : "Mixed or unclear units",
              points: item.months.map((month) => ({
                month: month.month,
                value:
                  month.quantity !== null && item.unit
                    ? `${month.quantity} ${item.unit}`
                    : `${month.purchases} purchase${month.purchases === 1 ? "" : "s"}`,
              })),
            }))}
          />
          <TrendPanel
            title="Monthly Category-wise Consumption Trend"
            emptyText="Not enough category history yet."
            rows={analysis.dashboard.monthly_category_consumption_trend.map((category) => ({
              label: category.category,
              details: "Purchase-activity proxy where mixed units limit exact consumption totals.",
              points: category.months.map((month) => ({
                month: month.month,
                value: `${month.purchases} purchase${month.purchases === 1 ? "" : "s"}`,
              })),
            }))}
          />
          <TrendPanel
            title="Monthly Category-wise Spending Trend"
            emptyText="Not enough category spend history yet."
            rows={analysis.dashboard.monthly_category_spending_trend.map((category) => ({
              label: category.category,
              details: "Recorded spend where price data was available.",
              points: category.months.map((month) => ({
                month: month.month,
                value: `₹${month.spend.toFixed(2)}`,
              })),
            }))}
          />
        </section>

        <section className="analysis-grid">
          <section className="panel stack">
            <div className="inventory-section-heading">
              <h2 className="section-heading">Seasonal Consumption Pattern</h2>
            </div>
            <ul className="compact-list compact-list--notes">
              {analysis.dashboard.seasonal_consumption_pattern.map((note) => (
                <li key={note}>{note}</li>
              ))}
            </ul>
          </section>
          <section className="panel stack">
            <div className="inventory-section-heading">
              <h2 className="section-heading">Guest / Event / Festival Impact</h2>
            </div>
            <ul className="compact-list compact-list--notes">
              {analysis.dashboard.guest_event_festival_impact.map((note) => (
                <li key={note}>{note}</li>
              ))}
            </ul>
          </section>
          <section className="panel stack">
            <div className="inventory-section-heading">
              <h2 className="section-heading">Decision Snapshot</h2>
            </div>
            <ul className="snapshot-list">
              <li>
                <span>Tracked items</span>
                <strong>{analysis.overview.trackedItemCount}</strong>
              </li>
              <li>
                <span>Items to order today</span>
                <strong>{analysis.overview.itemsToOrderTodayCount}</strong>
              </li>
              <li>
                <span>Items to order within 3 days</span>
                <strong>{analysis.overview.itemsToOrderSoonCount}</strong>
              </li>
              <li>
                <span>Items sufficient for more than 7 days</span>
                <strong>{analysis.overview.stableItemCount}</strong>
              </li>
              <li>
                <span>Total saved purchase rows</span>
                <strong>{entries.length}</strong>
              </li>
            </ul>
          </section>
        </section>
      </div>
    </main>
  );
}
