"use client";

import { useState, useMemo } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import { PeriodData } from "@/lib/types";
import { Trash2, Plus } from "lucide-react";

interface SavedRun {
  name: string;
  data: PeriodData[];
  config: Record<string, unknown>;
}

interface Props {
  currentData: PeriodData[];
  currentName: string;
  savedRuns: SavedRun[];
  onSave: () => void;
  onRemove: (index: number) => void;
}

const COLORS = [
  "#3b82f6",
  "#f59e0b",
  "#10b981",
  "#f43f5e",
  "#a855f7",
  "#06b6d4",
];

const COMPARE_METRICS = [
  { key: "gdp", label: "GDP" },
  { key: "unemployment_rate", label: "Unemployment Rate" },
  { key: "avg_price", label: "Average Price" },
  { key: "avg_wage", label: "Average Wage" },
  { key: "gini_deposits", label: "Gini Coefficient" },
  { key: "total_loans_outstanding", label: "Loans Outstanding" },
  { key: "total_production", label: "Total Production" },
  { key: "total_consumption", label: "Total Consumption" },
  { key: "govt_budget_balance", label: "Budget Balance" },
  { key: "bank_capital_ratio", label: "Bank Capital Ratio" },
];

export default function CompareTab({
  currentData,
  currentName,
  savedRuns,
  onSave,
  onRemove,
}: Props) {
  const [selectedMetric, setSelectedMetric] = useState("gdp");

  const allRuns = useMemo(() => {
    const runs = [
      { name: currentName, data: currentData },
      ...savedRuns,
    ];
    return runs;
  }, [currentData, currentName, savedRuns]);

  // Build merged dataset for overlay chart
  const mergedData = useMemo(() => {
    const maxLen = Math.max(...allRuns.map((r) => r.data.length));
    const merged: Record<string, unknown>[] = [];
    for (let i = 0; i < maxLen; i++) {
      const point: Record<string, unknown> = { period: i };
      allRuns.forEach((run, idx) => {
        if (i < run.data.length) {
          point[`${run.name}`] = run.data[i][selectedMetric];
        }
      });
      merged.push(point);
    }
    return merged;
  }, [allRuns, selectedMetric]);

  const metricLabel =
    COMPARE_METRICS.find((m) => m.key === selectedMetric)?.label ??
    selectedMetric;

  if (savedRuns.length === 0) {
    return (
      <div className="space-y-6">
        <div className="flex items-center justify-center py-16 text-muted text-sm">
          <div className="text-center space-y-3">
            <p className="text-base font-semibold text-foreground">
              Scenario Comparison
            </p>
            <p>
              Save your current run, then run a different scenario to compare
              them side-by-side.
            </p>
            <button
              onClick={onSave}
              className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-accent text-white text-xs font-semibold
                         hover:bg-accent/90 transition-all"
            >
              <Plus className="w-3.5 h-3.5" />
              Save Current Run
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Controls row */}
      <div className="flex items-center justify-between gap-4 flex-wrap">
        <div className="flex items-center gap-2">
          <label className="text-[11px] text-muted font-semibold uppercase tracking-wider">
            Metric
          </label>
          <select
            value={selectedMetric}
            onChange={(e) => setSelectedMetric(e.target.value)}
            className="rounded-lg border border-border-2/60 bg-surface-2/80 px-3 py-1.5 text-xs text-foreground
                       focus:border-accent focus:outline-none"
          >
            {COMPARE_METRICS.map((m) => (
              <option key={m.key} value={m.key}>
                {m.label}
              </option>
            ))}
          </select>
        </div>
        <button
          onClick={onSave}
          className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-accent/40 text-accent text-[11px] font-semibold
                     hover:bg-accent/10 transition-all"
        >
          <Plus className="w-3 h-3" />
          Save Current Run
        </button>
      </div>

      {/* Chart */}
      <div className="rounded-2xl border border-border/40 bg-surface/60 backdrop-blur-sm p-5">
        <h3 className="text-sm font-semibold text-foreground mb-1">
          {metricLabel} — Scenario Overlay
        </h3>
        <p className="text-[11px] text-muted mb-4">
          Comparing {allRuns.length} scenarios
        </p>
        <ResponsiveContainer width="100%" height={350}>
          <LineChart data={mergedData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#333" opacity={0.3} />
            <XAxis
              dataKey="period"
              tick={{ fill: "#888", fontSize: 10 }}
              axisLine={{ stroke: "#444" }}
            />
            <YAxis
              tick={{ fill: "#888", fontSize: 10 }}
              axisLine={{ stroke: "#444" }}
              width={65}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: "#1a1a2e",
                border: "1px solid #333",
                borderRadius: "8px",
                fontSize: "11px",
              }}
            />
            <Legend
              wrapperStyle={{ fontSize: "11px", paddingTop: "8px" }}
            />
            {allRuns.map((run, idx) => (
              <Line
                key={run.name}
                type="monotone"
                dataKey={run.name}
                stroke={COLORS[idx % COLORS.length]}
                strokeWidth={idx === 0 ? 2.5 : 1.5}
                dot={false}
                strokeDasharray={idx === 0 ? undefined : "5 3"}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Saved runs list */}
      <div className="rounded-2xl border border-border/40 bg-surface/60 backdrop-blur-sm p-4">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-muted mb-3">
          Saved Runs
        </h3>
        <div className="space-y-2">
          {savedRuns.map((run, idx) => (
            <div
              key={idx}
              className="flex items-center justify-between px-3 py-2 rounded-lg bg-surface-2/40 border border-border/30"
            >
              <div className="flex items-center gap-2">
                <div
                  className="w-3 h-3 rounded-full"
                  style={{
                    backgroundColor: COLORS[(idx + 1) % COLORS.length],
                  }}
                />
                <span className="text-xs font-medium text-foreground">
                  {run.name}
                </span>
                <span className="text-[10px] text-muted">
                  {run.data.length} periods
                </span>
              </div>
              <button
                onClick={() => onRemove(idx)}
                className="p-1 rounded hover:bg-danger/20 text-muted hover:text-danger transition-all"
                title="Remove run"
              >
                <Trash2 className="w-3.5 h-3.5" />
              </button>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
