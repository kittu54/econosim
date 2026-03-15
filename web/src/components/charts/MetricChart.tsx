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
  Legend,
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
  pink: "#ec4899",
  lime: "#84cc16",
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
  subtitle?: string;
  yAxisFormat?: "number" | "percent" | "decimal";
  stacked?: boolean;
  height?: number;
}

function formatTick(value: number, format: string) {
  if (format === "percent") return `${(value * 100).toFixed(0)}%`;
  if (format === "decimal") return value.toFixed(2);
  if (Math.abs(value) >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`;
  if (Math.abs(value) >= 1000) return `${(value / 1000).toFixed(1)}k`;
  return value.toFixed(0);
}

function formatTooltip(value: number, format: string) {
  if (format === "percent") return `${(value * 100).toFixed(2)}%`;
  if (format === "decimal") return value.toFixed(4);
  return value.toLocaleString("en-US", { maximumFractionDigits: 2 });
}

const tooltipStyle = {
  backgroundColor: "rgba(13, 17, 23, 0.95)",
  border: "1px solid #2d3a4e",
  borderRadius: "10px",
  fontSize: "12px",
  color: "#e8edf5",
  boxShadow: "0 8px 32px rgba(0, 0, 0, 0.4)",
  backdropFilter: "blur(8px)",
  padding: "10px 14px",
};

const gridStroke = "#1b2232";
const axisStroke = "#3d4f6a";

function ChartWrapper({
  title,
  subtitle,
  children,
}: {
  title: string;
  subtitle?: string;
  children: React.ReactNode;
}) {
  return (
    <div className="chart-container rounded-xl border border-border/60 bg-surface/60 backdrop-blur-sm p-4 transition-all hover:border-border-2/60">
      <div className="mb-3">
        <h3 className="text-sm font-semibold text-foreground">{title}</h3>
        {subtitle && (
          <p className="text-[11px] text-muted mt-0.5">{subtitle}</p>
        )}
      </div>
      {children}
    </div>
  );
}

export default function MetricChart({
  data,
  aggregate,
  metrics,
  title,
  subtitle,
  yAxisFormat = "number",
  stacked = false,
  height = 280,
}: MetricChartProps) {
  const hasCi = aggregate && aggregate.length > 0;

  if (stacked) {
    return (
      <ChartWrapper title={title} subtitle={subtitle}>
        <ResponsiveContainer width="100%" height={height}>
          <AreaChart data={data}>
            <defs>
              {metrics.map((m) => (
                <linearGradient
                  key={`grad-${m.key}`}
                  id={`grad-${m.key}`}
                  x1="0"
                  y1="0"
                  x2="0"
                  y2="1"
                >
                  <stop
                    offset="5%"
                    stopColor={COLORS[m.color] || m.color}
                    stopOpacity={0.5}
                  />
                  <stop
                    offset="95%"
                    stopColor={COLORS[m.color] || m.color}
                    stopOpacity={0.05}
                  />
                </linearGradient>
              ))}
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke={gridStroke} vertical={false} />
            <XAxis
              dataKey="period"
              stroke={axisStroke}
              fontSize={10}
              tickLine={false}
              axisLine={false}
            />
            <YAxis
              stroke={axisStroke}
              fontSize={10}
              tickLine={false}
              axisLine={false}
              tickFormatter={(v) => formatTick(v, yAxisFormat)}
            />
            <Tooltip
              contentStyle={tooltipStyle}
              formatter={(value) => formatTooltip(Number(value), yAxisFormat)}
              cursor={{ stroke: "#3b82f6", strokeWidth: 1, strokeDasharray: "4 4" }}
            />
            <Legend
              wrapperStyle={{ fontSize: "11px", paddingTop: "8px" }}
              iconType="circle"
              iconSize={8}
            />
            {metrics.map((m) => (
              <Area
                key={m.key}
                type="monotone"
                dataKey={m.key}
                stackId="1"
                stroke={COLORS[m.color] || m.color}
                fill={`url(#grad-${m.key})`}
                strokeWidth={1.5}
                name={m.label}
              />
            ))}
          </AreaChart>
        </ResponsiveContainer>
      </ChartWrapper>
    );
  }

  if (hasCi) {
    return (
      <ChartWrapper title={title} subtitle={subtitle}>
        <ResponsiveContainer width="100%" height={height}>
          <ComposedChart data={aggregate!}>
            <CartesianGrid strokeDasharray="3 3" stroke={gridStroke} vertical={false} />
            <XAxis
              dataKey="period"
              stroke={axisStroke}
              fontSize={10}
              tickLine={false}
              axisLine={false}
            />
            <YAxis
              stroke={axisStroke}
              fontSize={10}
              tickLine={false}
              axisLine={false}
              tickFormatter={(v) => formatTick(v, yAxisFormat)}
            />
            <Tooltip
              contentStyle={tooltipStyle}
              formatter={(value) => formatTooltip(Number(value), yAxisFormat)}
              cursor={{ stroke: "#3b82f6", strokeWidth: 1, strokeDasharray: "4 4" }}
            />
            <Legend
              wrapperStyle={{ fontSize: "11px", paddingTop: "8px" }}
              iconType="circle"
              iconSize={8}
            />
            {metrics.map((m) => (
              <Area
                key={`${m.key}_band`}
                type="monotone"
                dataKey={`${m.key}_hi`}
                stroke="none"
                fill={COLORS[m.color] || m.color}
                fillOpacity={0.08}
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
                activeDot={{ r: 4, strokeWidth: 0, fill: COLORS[m.color] || m.color }}
              />
            ))}
          </ComposedChart>
        </ResponsiveContainer>
      </ChartWrapper>
    );
  }

  return (
    <ChartWrapper title={title} subtitle={subtitle}>
      <ResponsiveContainer width="100%" height={height}>
        <LineChart data={data}>
          <defs>
            {metrics.map((m) => (
              <linearGradient
                key={`line-grad-${m.key}`}
                id={`line-grad-${m.key}`}
                x1="0"
                y1="0"
                x2="0"
                y2="1"
              >
                <stop
                  offset="5%"
                  stopColor={COLORS[m.color] || m.color}
                  stopOpacity={0.15}
                />
                <stop
                  offset="95%"
                  stopColor={COLORS[m.color] || m.color}
                  stopOpacity={0}
                />
              </linearGradient>
            ))}
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke={gridStroke} vertical={false} />
          <XAxis
            dataKey="period"
            stroke={axisStroke}
            fontSize={10}
            tickLine={false}
            axisLine={false}
          />
          <YAxis
            stroke={axisStroke}
            fontSize={10}
            tickLine={false}
            axisLine={false}
            tickFormatter={(v) => formatTick(v, yAxisFormat)}
          />
          <Tooltip
            contentStyle={tooltipStyle}
            formatter={(value) => formatTooltip(Number(value), yAxisFormat)}
            cursor={{ stroke: "#3b82f6", strokeWidth: 1, strokeDasharray: "4 4" }}
          />
          {metrics.length > 1 && (
            <Legend
              wrapperStyle={{ fontSize: "11px", paddingTop: "8px" }}
              iconType="circle"
              iconSize={8}
            />
          )}
          {metrics.map((m) => (
            <Line
              key={m.key}
              type="monotone"
              dataKey={m.key}
              stroke={COLORS[m.color] || m.color}
              strokeWidth={2}
              dot={false}
              name={m.label}
              activeDot={{
                r: 4,
                strokeWidth: 0,
                fill: COLORS[m.color] || m.color,
              }}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </ChartWrapper>
  );
}
