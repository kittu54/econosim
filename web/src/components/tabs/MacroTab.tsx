"use client";

import MetricChart from "@/components/charts/MetricChart";
import { PeriodData, AggregateData } from "@/lib/types";

interface Props {
  data: PeriodData[];
  aggregate?: AggregateData[] | null;
}

export default function MacroTab({ data, aggregate }: Props) {
  return (
    <div className="space-y-4">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <MetricChart
          data={data}
          aggregate={aggregate}
          metrics={[{ key: "gdp", label: "GDP", color: "blue" }]}
          title="Gross Domestic Product"
          subtitle="Total output of goods and services"
        />
        <MetricChart
          data={data}
          aggregate={aggregate}
          metrics={[{ key: "avg_price", label: "Price Level", color: "amber" }]}
          title="Average Price"
          subtitle="Weighted average goods price"
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
          subtitle="Share of labor force without jobs"
          yAxisFormat="percent"
        />
        <MetricChart
          data={data}
          aggregate={aggregate}
          metrics={[
            { key: "inflation_rate", label: "Inflation", color: "emerald" },
          ]}
          title="Inflation Rate"
          subtitle="Period-over-period price change"
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
          subtitle="0 = perfect equality, 1 = maximum inequality"
          yAxisFormat="decimal"
        />
        <MetricChart
          data={data}
          aggregate={aggregate}
          metrics={[
            { key: "gdp_growth", label: "GDP Growth", color: "teal" },
          ]}
          title="GDP Growth Rate"
          subtitle="Period-over-period GDP change"
          yAxisFormat="percent"
        />
      </div>
    </div>
  );
}
