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
        <div className="text-center space-y-2">
          <p className="text-base font-semibold text-foreground">
            No extensions enabled
          </p>
          <p>
            Toggle extensions in the sidebar to see network, bond, and
            expectations data.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Expectations */}
      {hasExpectations && (
        <>
          <h3 className="text-xs font-semibold uppercase tracking-wider text-muted pt-2">
            Adaptive Expectations
          </h3>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <MetricChart
              data={data}
              aggregate={aggregate}
              metrics={[
                {
                  key: "avg_price_forecast_error",
                  label: "Price Forecast Error",
                  color: "amber",
                },
              ]}
              title="Price Forecast Error"
              subtitle="Average absolute error across firms"
            />
            <MetricChart
              data={data}
              aggregate={aggregate}
              metrics={[
                {
                  key: "avg_demand_forecast_error",
                  label: "Demand Forecast Error",
                  color: "rose",
                },
              ]}
              title="Demand Forecast Error"
              subtitle="Average absolute error across firms"
            />
          </div>
        </>
      )}

      {/* Networks */}
      {hasNetworks && (
        <>
          <h3 className="text-xs font-semibold uppercase tracking-wider text-muted pt-2">
            Network Effects
          </h3>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <MetricChart
              data={data}
              aggregate={aggregate}
              metrics={[
                {
                  key: "trade_network_density",
                  label: "Trade Density",
                  color: "blue",
                },
                {
                  key: "credit_network_density",
                  label: "Credit Density",
                  color: "cyan",
                },
              ]}
              title="Network Density"
              subtitle="Ratio of actual to possible connections"
            />
            <MetricChart
              data={data}
              aggregate={aggregate}
              metrics={[
                {
                  key: "trade_seller_concentration",
                  label: "Seller HHI",
                  color: "amber",
                },
                {
                  key: "credit_network_concentration",
                  label: "Credit HHI",
                  color: "rose",
                },
              ]}
              title="Market Concentration (HHI)"
              subtitle="Higher values = more concentrated"
            />
          </div>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <MetricChart
              data={data}
              aggregate={aggregate}
              metrics={[
                {
                  key: "credit_systemic_risk",
                  label: "Systemic Risk",
                  color: "rose",
                },
              ]}
              title="Systemic Risk Score"
              subtitle="Density × concentration × exposure"
            />
          </div>
        </>
      )}

      {/* Bond Market */}
      {hasBonds && (
        <>
          <h3 className="text-xs font-semibold uppercase tracking-wider text-muted pt-2">
            Bond Market
          </h3>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <MetricChart
              data={data}
              aggregate={aggregate}
              metrics={[
                {
                  key: "bond_outstanding",
                  label: "Outstanding",
                  color: "indigo",
                },
              ]}
              title="Bonds Outstanding"
              subtitle="Total face value of government bonds"
            />
            <MetricChart
              data={data}
              aggregate={aggregate}
              metrics={[
                {
                  key: "bond_debt_to_gdp",
                  label: "Debt/GDP",
                  color: "rose",
                },
              ]}
              title="Debt-to-GDP Ratio"
              subtitle="Government bond debt relative to GDP"
            />
          </div>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <MetricChart
              data={data}
              aggregate={aggregate}
              metrics={[
                { key: "bond_issued", label: "Issued", color: "emerald" },
                { key: "bond_redeemed", label: "Redeemed", color: "amber" },
              ]}
              title="Bond Flows"
              subtitle="Per-period issuance and redemptions"
            />
            <MetricChart
              data={data}
              aggregate={aggregate}
              metrics={[
                {
                  key: "bond_interest_expense",
                  label: "Interest",
                  color: "rose",
                },
              ]}
              title="Bond Interest Expense"
              subtitle="Per-period coupon payments"
            />
          </div>
        </>
      )}
    </div>
  );
}
