"use client";

import { useState, useMemo } from "react";
import { Download, Table2, BarChart3, CheckCircle2 } from "lucide-react";
import clsx from "clsx";
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
    const json = JSON.stringify({ summary, periods: data }, null, 2);
    const blob = new Blob([json], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "econosim_data.json";
    a.click();
    URL.revokeObjectURL(url);
  };

  const displayCols =
    selectedColumns.length > 0 ? selectedColumns : allColumns;

  return (
    <div className="space-y-4">
      {/* Column selector */}
      <div className="rounded-xl border border-border/60 bg-surface/60 backdrop-blur-sm p-4">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-xs font-semibold flex items-center gap-2 uppercase tracking-wider text-muted">
            <Table2 className="w-3.5 h-3.5 text-accent" />
            Select Columns
            <span className="text-[10px] font-normal text-muted-2">
              ({selectedColumns.length}/{allColumns.length})
            </span>
          </h3>
          <div className="flex gap-1.5">
            <button
              onClick={() => setSelectedColumns([...allColumns])}
              className="text-[10px] px-2.5 py-1 rounded-md bg-surface-2/60 text-muted hover:text-foreground transition-colors"
            >
              All
            </button>
            <button
              onClick={() =>
                setSelectedColumns(
                  DEFAULT_COLUMNS.filter((c) => allColumns.includes(c))
                )
              }
              className="text-[10px] px-2.5 py-1 rounded-md bg-surface-2/60 text-muted hover:text-foreground transition-colors"
            >
              Default
            </button>
            <button
              onClick={() => setSelectedColumns(["period"])}
              className="text-[10px] px-2.5 py-1 rounded-md bg-surface-2/60 text-muted hover:text-foreground transition-colors"
            >
              Clear
            </button>
          </div>
        </div>
        <div className="flex flex-wrap gap-1.5">
          {allColumns.map((col) => (
            <button
              key={col}
              onClick={() => toggleColumn(col)}
              className={clsx(
                "text-[10px] px-2.5 py-1 rounded-full border transition-all duration-200 flex items-center gap-1",
                selectedColumns.includes(col)
                  ? "border-accent/40 bg-accent/10 text-accent"
                  : "border-border-2/40 text-muted-2 hover:text-muted hover:border-border-2"
              )}
            >
              {selectedColumns.includes(col) && (
                <CheckCircle2 className="w-2.5 h-2.5" />
              )}
              {col}
            </button>
          ))}
        </div>
      </div>

      {/* Data table */}
      <div className="rounded-xl border border-border/60 bg-surface/60 backdrop-blur-sm overflow-hidden">
        <div className="overflow-x-auto max-h-[420px] overflow-y-auto">
          <table className="w-full text-[11px]">
            <thead className="sticky top-0 z-10">
              <tr className="bg-surface-2/90 backdrop-blur-sm">
                {displayCols.map((col) => (
                  <th
                    key={col}
                    className="px-3 py-2.5 text-left font-semibold text-muted uppercase tracking-wider whitespace-nowrap border-b border-border/40"
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
                  className="border-t border-border/20 hover:bg-surface-2/30 transition-colors"
                >
                  {displayCols.map((col) => {
                    const val = row[col];
                    return (
                      <td
                        key={col}
                        className="px-3 py-1.5 whitespace-nowrap font-mono text-foreground/90"
                      >
                        {typeof val === "number"
                          ? val.toLocaleString("en-US", {
                              maximumFractionDigits: 4,
                            })
                          : val ?? "\u2014"}
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
      <div className="rounded-xl border border-border/60 bg-surface/60 backdrop-blur-sm p-4">
        <button
          onClick={() => setShowSummary(!showSummary)}
          className="flex items-center gap-2 text-xs font-semibold text-foreground mb-3 uppercase tracking-wider"
        >
          <BarChart3 className="w-3.5 h-3.5 text-accent" />
          Summary Statistics
          <span className="text-[10px] font-normal text-muted normal-case tracking-normal">
            {showSummary ? "(hide)" : "(show)"}
          </span>
        </button>
        {showSummary && (
          <div className="overflow-x-auto">
            <table className="w-full text-[11px]">
              <thead>
                <tr className="bg-surface-2/60">
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
                    className="border-t border-border/20 hover:bg-surface-2/30"
                  >
                    <td className="px-3 py-1.5 font-medium text-foreground">
                      {key}
                    </td>
                    <td className="px-3 py-1.5 text-right font-mono text-foreground">
                      {stats.mean?.toFixed(4)}
                    </td>
                    <td className="px-3 py-1.5 text-right font-mono text-muted">
                      {stats.std?.toFixed(4)}
                    </td>
                    <td className="px-3 py-1.5 text-right font-mono text-muted">
                      {stats.min?.toFixed(4)}
                    </td>
                    <td className="px-3 py-1.5 text-right font-mono text-muted">
                      {stats.max?.toFixed(4)}
                    </td>
                    <td className="px-3 py-1.5 text-right font-mono text-foreground">
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
          className="flex items-center gap-2 px-4 py-2.5 rounded-xl border border-border/60 bg-surface/60 backdrop-blur-sm text-xs font-medium
                     hover:border-accent/40 hover:text-accent transition-all"
        >
          <Download className="w-3.5 h-3.5" />
          Download CSV
        </button>
        <button
          onClick={downloadJson}
          className="flex items-center gap-2 px-4 py-2.5 rounded-xl border border-border/60 bg-surface/60 backdrop-blur-sm text-xs font-medium
                     hover:border-accent/40 hover:text-accent transition-all"
        >
          <Download className="w-3.5 h-3.5" />
          Download JSON
        </button>
      </div>
    </div>
  );
}
