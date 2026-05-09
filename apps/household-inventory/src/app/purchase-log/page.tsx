import Link from "next/link";

import { getInventorySnapshot } from "@/lib/inventory";

function formatDate(value: string | null) {
  if (!value) {
    return "Needs Review";
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

export default function PurchaseLogPage() {
  const snapshot = getInventorySnapshot();
  const entries = snapshot.ledger.purchases;

  return (
    <main className="inventory-shell">
      <div className="inventory-shell__inner inventory-shell__inner--narrow">
        <header className="panel stack">
          <div className="badge-row">
            <span className="badge">Detailed Purchase Log</span>
            <span className="badge">Secondary drill-down page</span>
          </div>
          <div className="stack">
            <span className="eyebrow">Household Inventory</span>
            <h1 className="inventory-title inventory-title--compact">Detailed Purchase Log</h1>
            <p className="muted">
              This page keeps the raw register out of the main landing page. Use the dashboard for decisions, and use this page only when
              you need to inspect the full saved record.
            </p>
          </div>
          <div className="button-row">
            <Link className="button button--primary" href="/">
              Back to Dashboard
            </Link>
          </div>
        </header>

        <section className="panel stack">
          <div className="inventory-section-heading">
            <div>
              <h2 className="section-heading">Saved Purchase Entries</h2>
              <p className="muted">The full purchase log remains here as a secondary inspection view.</p>
            </div>
            <span className="pill">{entries.length} entries</span>
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
                      <td>{formatDate(entry.date_of_purchase)}</td>
                      <td>
                        <div className="inventory-item-cell">
                          <strong>{entry.item_name}</strong>
                          {entry.review_status === "needs_review" ? (
                            <span className="inventory-inline-note">Review: {entry.review_notes.join(", ")}</span>
                          ) : null}
                        </div>
                      </td>
                      <td>
                        <span className={`inventory-category inventory-category--${entry.category_status}`}>{entry.category}</span>
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
      </div>
    </main>
  );
}
