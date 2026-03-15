"use client";

import MetricChart from "@/components/charts/MetricChart";
import { PeriodData, AggregateData } from "@/lib/types";

interface Props {
  data: PeriodData[];
  aggregate?: AggregateData[] | null;
}

export default function MoneyTab({ data, aggregate }: Props) {
  return (
    <div className="space-y-4 animate-fade-in">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <MetricChart
          data={data}
          metrics={[
            { key: "total_hh_deposits", label: "Households", color: "blue" },
            { key: "total_firm_deposits", label: "Firms", color: "amber" },
            { key: "govt_deposits", label: "Government", color: "emerald" },
          ]}
          title="Deposit Distribution"
          stacked
        />
        <MetricChart
          data={data}
          aggregate={aggregate}
          metrics={[
            { key: "total_hh_deposits", label: "Household", color: "blue" },
            { key: "total_firm_deposits", label: "Firm", color: "amber" },
          ]}
          title="Deposits by Sector"
        />
      </div>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <MetricChart
          data={data}
          aggregate={aggregate}
          metrics={[
            {
              key: "total_loans_outstanding",
              label: "Loans",
              color: "rose",
            },
          ]}
          title="Total Loans Outstanding"
        />
        <MetricChart
          data={data}
          aggregate={aggregate}
          metrics={[
            { key: "bank_equity", label: "Bank Equity", color: "emerald" },
          ]}
          title="Bank Equity"
        />
      </div>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <MetricChart
          data={data}
          aggregate={aggregate}
          metrics={[
            { key: "bank_capital_ratio", label: "CAR", color: "indigo" },
          ]}
          title="Bank Capital Adequacy Ratio"
          yAxisFormat="percent"
        />
        <MetricChart
          data={data}
          aggregate={aggregate}
          metrics={[
            { key: "velocity", label: "Velocity", color: "purple" },
          ]}
          title="Velocity of Money"
          yAxisFormat="decimal"
        />
      </div>
    </div>
  );
}
