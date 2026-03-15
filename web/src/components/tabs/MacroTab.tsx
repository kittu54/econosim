"use client";

import MetricChart from "@/components/charts/MetricChart";
import { PeriodData, AggregateData } from "@/lib/types";

interface Props {
  data: PeriodData[];
  aggregate?: AggregateData[] | null;
}

export default function MacroTab({ data, aggregate }: Props) {
  return (
    <div className="space-y-4 animate-fade-in">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <MetricChart
          data={data}
          aggregate={aggregate}
          metrics={[{ key: "gdp", label: "GDP", color: "blue" }]}
          title="Gross Domestic Product"
        />
        <MetricChart
          data={data}
          aggregate={aggregate}
          metrics={[{ key: "avg_price", label: "Price Level", color: "amber" }]}
          title="Average Price"
          yAxisFormat="decimal"
        />
      </div>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <MetricChart
          data={data}
          aggregate={aggregate}
          metrics={[
            { key: "unemployment_rate", label: "Unemployment", color: "rose" },
          ]}
          title="Unemployment Rate"
          yAxisFormat="percent"
        />
        <MetricChart
          data={data}
          aggregate={aggregate}
          metrics={[
            { key: "inflation_rate", label: "Inflation", color: "emerald" },
          ]}
          title="Inflation Rate"
          yAxisFormat="percent"
        />
      </div>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <MetricChart
          data={data}
          aggregate={aggregate}
          metrics={[
            { key: "gini_deposits", label: "Gini", color: "purple" },
          ]}
          title="Wealth Inequality (Gini)"
          yAxisFormat="decimal"
        />
        <MetricChart
          data={data}
          aggregate={aggregate}
          metrics={[
            { key: "gdp_growth", label: "GDP Growth", color: "teal" },
          ]}
          title="GDP Growth Rate"
          yAxisFormat="percent"
        />
      </div>
    </div>
  );
}
