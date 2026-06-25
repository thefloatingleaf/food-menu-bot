import { InventoryModule } from "@/components/InventoryModule";
import { getInventorySnapshot } from "@/lib/inventory";

export default async function Home() {
  const snapshot = getInventorySnapshot();

  return (
    <InventoryModule
      initialEntries={snapshot.ledger.purchases}
      initialAnalysis={snapshot.analysis}
      initialContextNotes={snapshot.contextNotes}
    />
  );
}
