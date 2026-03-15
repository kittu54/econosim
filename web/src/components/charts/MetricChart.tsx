"use client";

import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Area,
  AreaChart,
  ComposedChart,
} from "recharts";
import { PeriodData, AggregateData } from "@/lib/types";

const COLORS: Record<string, string> = {
  blue: "#3b82f6",
  indigo: "#6366f1",
  emerald: "#10b981",
  amber: "#f59e0b",
  rose: "#f43f5e",
  purple: "#a855f7",
  teal: "#14b8a6",
  slate: "#64748b",
  cyan: "#06b6d4",
  orange: "#f97316",
};

interface MetricSpec {
  key: string;
  label: string;
  color: string;
}

interface MetricChartProps {
  data: PeriodData[];
  aggregate?: AggregateData[] | null;
  metrics: MetricSpec[];
  title: string;
  yAxisFormat?: "number" | "percent" | "decimal";
  stacked?: boolean;
  height?: number;
}

function formatTick(value: number, format: string) {
  if (format === "percent") return `${(value * 100).toFixed(0)}%`;
  if (format === "decimal") return value.toFixed(2);
  if (Math.abs(value) >= 1000) return `${(value / 1000).toFixed(1)}k`;
  return value.toFixed(0);
}

function formatTooltip(value: number, format: string) {
  if (format === "percent") return `${(value * 100).toFixed(2)}%`;
  if (format === "decimal") return value.toFixed(4);
  return value.toLocaleString("en-US", { maximumFractionDigits: 2 });
}

export default function MetricChart({
  data,
  aggregate,
  metrics,
  title,
  yAxisFormat = "number",
  stacked = false,
  height = 280,
}: MetricChartProps) {
  const hasCi = aggregate && aggregate.length > 0;
  const chartData = hasCi ? aggregate : data;

  if (stacked) {
    return (
      <div className="rounded-xl border border-border bg-surface p-4">
        <h3 className="text-sm font-semibold text-foreground mb-3">{title}</h3>
        <ResponsiveContainer width="100%" height={height}>
          <AreaChart data={data}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
            <XAxis
              dataKey="period"
              stroke="#475569"
              fontSize={11}
              tickLine={false}
            />
            <YAxis
              stroke="#475569"
              fontSize={11}
              tickLine={false}
              tickFormatter={(v) => formatTick(v, yAxisFormat)}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: "#1e293b",
                border: "1px solid #334155",
                borderRadius: "8px",
                fontSize: "12px",
                color: "#e2e8f0",
              }}
              formatter={(value) => formatTooltip(Number(value), yAxisFormat)}
            />
            {metrics.map((m) => (
              <Area
                key={m.key}
                type="monotone"
                dataKey={m.key}
                stackId="1"
                stroke={COLORS[m.color] || m.color}
                fill={COLORS[m.color] || m.color}
                fillOpacity={0.4}
                strokeWidth={1}
                name={m.label}
              />
            ))}
          </AreaChart>
        </ResponsiveContainer>
      </div>
    );
  }

  if (hasCi) {
    return (
      <div className="rounded-xl border border-border bg-surface p-4">
        <h3 className="text-sm font-semibold text-foreground mb-3">{title}</h3>
        <ResponsiveContainer width="100%" height={height}>
          <ComposedChart data={aggregate!}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
            <XAxis
              dataKey="period"
              stroke="#475569"
              fontSize={11}
              tickLine={false}
            />
            <YAxis
              stroke="#475569"
              fontSize={11}
              tickLine={false}
              tickFormatter={(v) => formatTick(v, yAxisFormat)}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: "#1e293b",
                border: "1px solid #334155",
                borderRadius: "8px",
                fontSize: "12px",
                color: "#e2e8f0",
              }}
              formatter={(value) => formatTooltip(Number(value), yAxisFormat)}
            />
            {metrics.map((m) => (
              <Area
                key={`${m.key}_band`}
                type="monotone"
                dataKey={`${m.key}_hi`}
                stroke="none"
                fill={COLORS[m.color] || m.color}
                fillOpacity={0.1}
                name={`${m.label} CI`}
                legendType="none"
              />
            ))}
            {metrics.map((m) => (
              <Line
                key={m.key}
                type="monotone"
                dataKey={`${m.key}_mean`}
                stroke={COLORS[m.color] || m.color}
                strokeWidth={2}
                dot={false}
                name={m.label}
              />
            ))}
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    );
  }

  return (
    <div className="rounded-xl border border-border bg-surface p-4">
      <h3 className="text-sm font-semibold text-foreground mb-3">{title}</h3>
      <ResponsiveContainer width="100%" height={height}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
          <XAxis
            dataKey="period"
            stroke="#475569"
            fontSize={11}
            tickLine={false}
          />
          <YAxis
            stroke="#475569"
            fontSize={11}
            tickLine={false}
            tickFormatter={(v) => formatTick(v, yAxisFormat)}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: "#1e293b",
              border: "1px solid #334155",
              borderRadius: "8px",
              fontSize: "12px",
              color: "#e2e8f0",
            }}
            formatter={(value) => formatTooltip(Number(value), yAxisFormat)}
          />
          {metrics.map((m) => (
            <Line
              key={m.key}
              type="monotone"
              dataKey={m.key}
              stroke={COLORS[m.color] || m.color}
              strokeWidth={2}
              dot={false}
              name={m.label}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
