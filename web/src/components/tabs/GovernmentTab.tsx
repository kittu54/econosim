"use client";

import KpiCard from "@/components/KpiCard";
import MetricChart from "@/components/charts/MetricChart";
import { PeriodData, AggregateData } from "@/lib/types";
import { fmtNumber } from "@/lib/format";
import { Receipt, ArrowUpDown, Banknote, TrendingUp } from "lucide-react";

interface Props {
  data: PeriodData[];
  aggregate?: AggregateData[] | null;
}

export default function GovernmentTab({ data, aggregate }: Props) {
  const final = data[data.length - 1];

  return (
    <div className="space-y-4 animate-fade-in">
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <KpiCard
          label="Tax Revenue"
          value={fmtNumber(final?.govt_tax_revenue)}
          icon={<Receipt className="w-4 h-4" />}
          accentColor="from-emerald-500/20 to-emerald-600/5"
        />
        <KpiCard
          label="Transfers"
          value={fmtNumber(final?.govt_transfers)}
          icon={<ArrowUpDown className="w-4 h-4" />}
          accentColor="from-amber-500/20 to-amber-600/5"
        />
        <KpiCard
          label="Spending"
          value={fmtNumber(final?.govt_spending)}
          icon={<Banknote className="w-4 h-4" />}
          accentColor="from-blue-500/20 to-blue-600/5"
        />
        <KpiCard
          label="Budget Balance"
          value={fmtNumber(final?.govt_budget_balance)}
          icon={<TrendingUp className="w-4 h-4" />}
          accentColor="from-indigo-500/20 to-indigo-600/5"
          deltaDirection={
            (final?.govt_budget_balance ?? 0) >= 0 ? "up" : "down"
          }
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <MetricChart
          data={data}
          aggregate={aggregate}
          metrics={[
            { key: "govt_tax_revenue", label: "Tax Revenue", color: "emerald" },
            { key: "govt_transfers", label: "Transfers", color: "amber" },
            { key: "govt_spending", label: "Spending", color: "blue" },
          ]}
          title="Fiscal Flows"
        />
        <MetricChart
          data={data}
          aggregate={aggregate}
          metrics={[
            {
              key: "govt_budget_balance",
              label: "Budget Balance",
              color: "rose",
            },
          ]}
          title="Budget Balance (T - G - Tr)"
        />
      </div>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <MetricChart
          data={data}
          aggregate={aggregate}
          metrics={[
            { key: "govt_deposits", label: "Deposits", color: "indigo" },
          ]}
          title="Government Deposits"
        />
        <MetricChart
          data={data}
          aggregate={aggregate}
          metrics={[
            {
              key: "govt_money_created",
              label: "Per-period",
              color: "amber",
            },
            {
              key: "govt_cumulative_money_created",
              label: "Cumulative",
              color: "rose",
            },
          ]}
          title="Sovereign Money Creation"
        />
      </div>
    </div>
  );
}
