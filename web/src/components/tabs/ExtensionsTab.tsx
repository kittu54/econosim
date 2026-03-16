"use client";

import MetricChart from "@/components/charts/MetricChart";
import { PeriodData, AggregateData } from "@/lib/types";

interface Props {
  data: PeriodData[];
  aggregate?: AggregateData[] | null;
}

export default function ExtensionsTab({ data, aggregate }: Props) {
  const hasNetworks = data.some((d) => d.trade_network_density !== undefined);
  const hasBonds = data.some((d) => d.bond_outstanding !== undefined);
  const hasExpectations = data.some(
    (d) => d.avg_price_forecast_error !== undefined
  );

  if (!hasNetworks && !hasBonds && !hasExpectations) {
    return (
      <div className="flex items-center justify-center py-20 text-muted text-sm">
        No extensions enabled for this run. Enable them in the sidebar and run again.
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
      {hasNetworks && (
        <>
          <MetricChart
            title="Trade Network Density"
            data={data}
            series={[{ key: "trade_network_density", name: "Density", color: "#8b5cf6" }]}
            aggregate={aggregate}
            domain={[0, "auto"]}
            subtitle="Connectivity of the trade graph over time"
          />
          <MetricChart
            title="Credit Systemic Risk"
            data={data}
            series={[{ key: "credit_systemic_risk", name: "Systemic Risk", color: "#f43f5e" }]}
            aggregate={aggregate}
            domain={[0, "auto"]}
            subtitle="Measure of fragility in lending relationships"
          />
        </>
      )}

      {hasBonds && (
        <>
          <MetricChart
            title="Sovereign Bond Outstanding"
            data={data}
            series={[{ key: "bond_outstanding", name: "Outstanding", color: "#10b981" }]}
            aggregate={aggregate}
            domain={[0, "auto"]}
            subtitle="Total face value of active government bonds"
          />
          <MetricChart
            title="Debt-to-GDP Ratio"
            data={data}
            series={[{ key: "bond_debt_to_gdp", name: "Debt/GDP", color: "#3b82f6" }]}
            aggregate={aggregate}
            domain={[0, "auto"]}
            subtitle="Ratio of outstanding sovereign debt to nominal GDP"
          />
        </>
      )}

      {hasExpectations && (
        <>
          <MetricChart
            title="Price Forecast Error"
            data={data}
            series={[{ key: "avg_price_forecast_error", name: "Error", color: "#f59e0b" }]}
            aggregate={aggregate}
            domain={["auto", "auto"]}
            subtitle="Average absolute error in firm price expectations"
          />
          <MetricChart
            title="Demand Forecast Error"
            data={data}
            series={[{ key: "avg_demand_forecast_error", name: "Error", color: "#0ea5e9" }]}
            aggregate={aggregate}
            domain={["auto", "auto"]}
            subtitle="Average absolute error in firm demand expectations"
          />
        </>
      )}
    </div>
  );
}
