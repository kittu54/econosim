"use client";

import MetricChart from "@/components/charts/MetricChart";
import { PeriodData, AggregateData } from "@/lib/types";

interface Props {
  data: PeriodData[];
  aggregate?: AggregateData[] | null;
}

export default function LaborTab({ data, aggregate }: Props) {
  return (
    <div className="space-y-4">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <MetricChart
          data={data}
          aggregate={aggregate}
          metrics={[
            { key: "total_employed", label: "Employed", color: "emerald" },
            { key: "labor_force", label: "Labor Force", color: "slate" },
          ]}
          title="Employment"
          subtitle="Workers employed vs total labor force"
        />
        <MetricChart
          data={data}
          aggregate={aggregate}
          metrics={[
            { key: "total_vacancies", label: "Vacancies", color: "orange" },
          ]}
          title="Firm Vacancies"
          subtitle="Open positions across all firms"
        />
      </div>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <MetricChart
          data={data}
          aggregate={aggregate}
          metrics={[{ key: "avg_wage", label: "Wage", color: "blue" }]}
          title="Average Wage"
          subtitle="Mean wage offered by firms"
        />
        <MetricChart
          data={data}
          aggregate={aggregate}
          metrics={[
            { key: "total_production", label: "Production", color: "teal" },
          ]}
          title="Total Production"
          subtitle="Units produced per period"
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
          subtitle="Unsold goods across firms"
        />
        <MetricChart
          data={data}
          aggregate={aggregate}
          metrics={[
            { key: "total_consumption", label: "Consumption", color: "rose" },
          ]}
          title="Total Consumption Spending"
          subtitle="Household expenditure on goods"
        />
      </div>
    </div>
  );
}
