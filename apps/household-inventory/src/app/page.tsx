import { InventoryModule } from "@/components/InventoryModule";
import { getInventorySnapshot } from "@/lib/inventory";

type HomeProps = {
  searchParams?: Promise<{
    tab?: string;
  }>;
};

export default async function Home({ searchParams }: HomeProps) {
  const params = (await searchParams) ?? {};
  const initialTab = params.tab === "analysis" ? "analysis" : "log";
  const snapshot = getInventorySnapshot();

  return (
    <InventoryModule
      initialEntries={snapshot.ledger.purchases}
      initialAnalysis={snapshot.analysis}
      initialContextNotes={snapshot.contextNotes}
      initialSupplyContext={snapshot.supplyContext}
      initialTab={initialTab}
    />
  );
}
