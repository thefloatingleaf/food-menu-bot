import { cookies } from "next/headers";
import { redirect } from "next/navigation";

import { InventoryModule } from "@/components/InventoryModule";
import { getAuthenticatedAccount } from "@/lib/db";
import { getInventorySnapshot } from "@/lib/inventory";
import { AUTH_SESSION_COOKIE_NAME } from "@/lib/session";

type InventoryPageProps = {
  searchParams?: Promise<{
    tab?: string;
  }>;
};

export default async function InventoryPage({ searchParams }: InventoryPageProps) {
  const cookieStore = await cookies();
  const sessionToken = cookieStore.get(AUTH_SESSION_COOKIE_NAME)?.value ?? null;
  const account = sessionToken ? getAuthenticatedAccount(sessionToken) : null;

  if (!account) {
    redirect("/");
  }

  if (account.role !== "admin") {
    redirect("/");
  }

  const params = (await searchParams) ?? {};
  const initialTab = params.tab === "analysis" ? "analysis" : "log";
  const snapshot = getInventorySnapshot();

  return (
    <InventoryModule
      initialEntries={snapshot.ledger.purchases}
      initialAnalysis={snapshot.analysis}
      initialTab={initialTab}
      accountDisplayName={account.displayName}
    />
  );
}
