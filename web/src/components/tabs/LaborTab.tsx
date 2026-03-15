"use client";

import MetricChart from "@/components/charts/MetricChart";
import { PeriodData, AggregateData } from "@/lib/types";

interface Props {
  data: PeriodData[];
  aggregate?: AggregateData[] | null;
}

export default function LaborTab({ data, aggregate }: Props) {
  return (
    <div className="space-y-4 animate-fade-in">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <MetricChart
          data={data}
          aggregate={aggregate}
          metrics={[
            { key: "total_employed", label: "Employed", color: "emerald" },
            { key: "labor_force", label: "Labor Force", color: "slate" },
          ]}
          title="Employment"
        />
        <MetricChart
          data={data}
          aggregate={aggregate}
          metrics={[
            { key: "total_vacancies", label: "Vacancies", color: "orange" },
          ]}
          title="Firm Vacancies"
        />
      </div>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <MetricChart
          data={data}
          aggregate={aggregate}
          metrics={[{ key: "avg_wage", label: "Wage", color: "blue" }]}
          title="Average Wage"
        />
        <MetricChart
          data={data}
          aggregate={aggregate}
          metrics={[
            { key: "total_production", label: "Production", color: "teal" },
          ]}
          title="Total Production (units)"
        />
      </div>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <MetricChart
          data={data}
          aggregate={aggregate}
          metrics={[
            { key: "total_inventory", label: "Inventory", color: "purple" },
          ]}
          title="Total Inventory"
        />
        <MetricChart
          data={data}
          aggregate={aggregate}
          metrics={[
            {
              key: "total_consumption",
              label: "Consumption",
              color: "rose",
            },
          ]}
          title="Total Consumption Spending"
        />
      </div>
    </div>
  );
}
