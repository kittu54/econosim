"use client";

import { useState, useMemo } from "react";
import { Download, Table2, BarChart3 } from "lucide-react";
import { PeriodData } from "@/lib/types";

interface Props {
  data: PeriodData[];
  summary: Record<string, Record<string, number>>;
}

const DEFAULT_COLUMNS = [
  "period",
  "gdp",
  "unemployment_rate",
  "avg_price",
  "avg_wage",
  "total_hh_deposits",
  "total_firm_deposits",
  "govt_budget_balance",
];

export default function DataTab({ data, summary }: Props) {
  const allColumns = useMemo(
    () => (data.length > 0 ? Object.keys(data[0]) : []),
    [data]
  );
  const [selectedColumns, setSelectedColumns] = useState<string[]>(
    DEFAULT_COLUMNS.filter((c) => allColumns.includes(c))
  );
  const [showSummary, setShowSummary] = useState(false);

  const toggleColumn = (col: string) => {
    setSelectedColumns((prev) =>
      prev.includes(col) ? prev.filter((c) => c !== col) : [...prev, col]
    );
  };

  const downloadCsv = () => {
    const cols = selectedColumns.length > 0 ? selectedColumns : allColumns;
    const header = cols.join(",");
    const rows = data.map((row) =>
      cols.map((c) => row[c] ?? "").join(",")
    );
    const csv = [header, ...rows].join("\n");
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "econosim_metrics.csv";
    a.click();
    URL.revokeObjectURL(url);
  };

  const downloadJson = () => {
    const json = JSON.stringify(summary, null, 2);
    const blob = new Blob([json], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "econosim_summary.json";
    a.click();
    URL.revokeObjectURL(url);
  };

  const displayCols = selectedColumns.length > 0 ? selectedColumns : allColumns;

  return (
    <div className="space-y-4 animate-fade-in">
      {/* Column selector */}
      <div className="rounded-xl border border-border bg-surface p-4">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-semibold flex items-center gap-2">
            <Table2 className="w-4 h-4 text-accent" />
            Select Columns
          </h3>
          <div className="flex gap-2">
            <button
              onClick={() => setSelectedColumns(allColumns)}
              className="text-xs px-2 py-1 rounded-md bg-surface-2 text-muted hover:text-foreground transition-colors"
            >
              All
            </button>
            <button
              onClick={() =>
                setSelectedColumns(
                  DEFAULT_COLUMNS.filter((c) => allColumns.includes(c))
                )
              }
              className="text-xs px-2 py-1 rounded-md bg-surface-2 text-muted hover:text-foreground transition-colors"
            >
              Default
            </button>
          </div>
        </div>
        <div className="flex flex-wrap gap-1.5">
          {allColumns.map((col) => (
            <button
              key={col}
              onClick={() => toggleColumn(col)}
              className={`text-xs px-2.5 py-1 rounded-full border transition-all ${
                selectedColumns.includes(col)
                  ? "border-accent bg-accent/10 text-accent"
                  : "border-border-2 text-muted hover:border-border-2 hover:text-foreground"
              }`}
            >
              {col}
            </button>
          ))}
        </div>
      </div>

      {/* Data table */}
      <div className="rounded-xl border border-border bg-surface overflow-hidden">
        <div className="overflow-x-auto max-h-[420px] overflow-y-auto">
          <table className="w-full text-xs">
            <thead className="sticky top-0 z-10">
              <tr className="bg-surface-2">
                {displayCols.map((col) => (
                  <th
                    key={col}
                    className="px-3 py-2.5 text-left font-semibold text-muted uppercase tracking-wider whitespace-nowrap"
                  >
                    {col}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {data.map((row, i) => (
                <tr
                  key={i}
                  className="border-t border-border/50 hover:bg-surface-2/50 transition-colors"
                >
                  {displayCols.map((col) => {
                    const val = row[col];
                    return (
                      <td
                        key={col}
                        className="px-3 py-2 whitespace-nowrap font-mono text-foreground"
                      >
                        {typeof val === "number"
                          ? val.toLocaleString("en-US", {
                              maximumFractionDigits: 4,
                            })
                          : val ?? "—"}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Summary statistics */}
      <div className="rounded-xl border border-border bg-surface p-4">
        <button
          onClick={() => setShowSummary(!showSummary)}
          className="flex items-center gap-2 text-sm font-semibold text-foreground mb-3"
        >
          <BarChart3 className="w-4 h-4 text-accent" />
          Summary Statistics
          <span className="text-xs text-muted">
            {showSummary ? "(hide)" : "(show)"}
          </span>
        </button>
        {showSummary && (
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="bg-surface-2">
                  <th className="px-3 py-2 text-left font-semibold text-muted uppercase tracking-wider">
                    Metric
                  </th>
                  <th className="px-3 py-2 text-right font-semibold text-muted uppercase tracking-wider">
                    Mean
                  </th>
                  <th className="px-3 py-2 text-right font-semibold text-muted uppercase tracking-wider">
                    Std
                  </th>
                  <th className="px-3 py-2 text-right font-semibold text-muted uppercase tracking-wider">
                    Min
                  </th>
                  <th className="px-3 py-2 text-right font-semibold text-muted uppercase tracking-wider">
                    Max
                  </th>
                  <th className="px-3 py-2 text-right font-semibold text-muted uppercase tracking-wider">
                    Final
                  </th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(summary).map(([key, stats]) => (
                  <tr
                    key={key}
                    className="border-t border-border/50 hover:bg-surface-2/50"
                  >
                    <td className="px-3 py-2 font-medium text-foreground">
                      {key}
                    </td>
                    <td className="px-3 py-2 text-right font-mono text-foreground">
                      {stats.mean?.toFixed(4)}
                    </td>
                    <td className="px-3 py-2 text-right font-mono text-muted">
                      {stats.std?.toFixed(4)}
                    </td>
                    <td className="px-3 py-2 text-right font-mono text-muted">
                      {stats.min?.toFixed(4)}
                    </td>
                    <td className="px-3 py-2 text-right font-mono text-muted">
                      {stats.max?.toFixed(4)}
                    </td>
                    <td className="px-3 py-2 text-right font-mono text-foreground">
                      {stats.final?.toFixed(4)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Download buttons */}
      <div className="flex gap-3">
        <button
          onClick={downloadCsv}
          className="flex items-center gap-2 px-4 py-2.5 rounded-xl border border-border bg-surface text-sm font-medium
                     hover:border-accent hover:text-accent transition-all"
        >
          <Download className="w-4 h-4" />
          Download CSV
        </button>
        <button
          onClick={downloadJson}
          className="flex items-center gap-2 px-4 py-2.5 rounded-xl border border-border bg-surface text-sm font-medium
                     hover:border-accent hover:text-accent transition-all"
        >
          <Download className="w-4 h-4" />
          Download Summary JSON
        </button>
      </div>
    </div>
  );
}
