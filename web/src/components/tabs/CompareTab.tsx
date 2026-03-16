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
import { Trash2, Plus, GitBranch, Share2 } from "lucide-react";
import clsx from "clsx";

import { getRun } from "@/lib/api";

export interface SavedRun {
  id: string; // Add ID to differentiate runs
  name: string;
  data?: PeriodData[]; // Data might not be loaded yet
  config: Record<string, unknown>;
}

interface Props {
  savedRuns: SavedRun[];
  onDeleteRun: (id: string) => void;
  currentRun?: SavedRun | null;
  onSaveCurrentRun?: (name: string) => void;
}

const METRICS = [
  { key: "gdp", label: "GDP" },
  { key: "unemployment_rate", label: "Unemployment Rate (%)" },
  { key: "avg_price", label: "Average Price" },
  { key: "avg_wage", label: "Average Wage" },
  { key: "total_loans_outstanding", label: "Total Loans Outstanding" },
  { key: "total_production", label: "Total Production" },
  { key: "gini_deposits", label: "Gini Coefficient (Deposits)" },
  { key: "govt_budget_balance", label: "Govt Budget Balance" }
];

// Recharts colors
const COLORS = [
  "#3b82f6", // Blue
  "#f43f5e", // Rose
  "#10b981", // Emerald
  "#f59e0b", // Amber
  "#a855f7", // Purple
  "#06b6d4", // Cyan
  "#f97316", // Orange
  "#8b5cf6", // Violet
];

export default function CompareTab({ savedRuns, onDeleteRun, currentRun, onSaveCurrentRun }: Props) {
  const [selectedRuns, setSelectedRuns] = useState<string[]>([]);
  const [saveName, setSaveName] = useState("");
  // We keep the details of fully fetched runs here
  const [fetchedRuns, setFetchedRuns] = useState<Record<string, SavedRun>>({});
  const [loadingIds, setLoadingIds] = useState<Set<string>>(new Set());

  const runsToCompare = useMemo(() => {
    return selectedRuns
      .map((id) => fetchedRuns[id])
      .filter(Boolean) as SavedRun[];
  }, [selectedRuns, fetchedRuns]);

  const toggleRun = async (id: string) => {
    if (selectedRuns.includes(id)) {
      setSelectedRuns((prev) => prev.filter((rId) => rId !== id));
      return;
    }

    // Attempt to fetch the run if we don't have its full data
    if (!fetchedRuns[id]) {
      setLoadingIds((prev) => new Set(prev).add(id));
      try {
        const fullRun = await getRun(id);
        setFetchedRuns((prev) => ({ ...prev, [id]: fullRun }));
      } catch (err) {
        console.error("Failed to fetch run data", err);
        setLoadingIds((prev) => {
          const next = new Set(prev);
          next.delete(id);
          return next;
        });
        return; // Don't select if we couldn't fetch
      }
      setLoadingIds((prev) => {
        const next = new Set(prev);
        next.delete(id);
        return next;
      });
    }

    setSelectedRuns((prev) => [...prev, id]);
  };

  const handleSave = () => {
    if (onSaveCurrentRun && currentRun && saveName.trim() !== "") {
      onSaveCurrentRun(saveName.trim());
      setSaveName("");
    }
  };

  const comparisonData = useMemo(() => {
    if (runsToCompare.length === 0) return [];
    const maxPeriods = Math.max(
      ...runsToCompare.map((r) => (r.data ? r.data.length : 0))
    );
    const mergedData = [];

    for (let i = 0; i < maxPeriods; i++) {
      const mergedPoint: any = { period: i };
      runsToCompare.forEach((run) => {
        if (run.data && run.data[i]) {
          METRICS.forEach((metric) => {
            mergedPoint[`${run.id}_${metric.key}`] = run.data![i][metric.key];
          });
        }
      });
      mergedData.push(mergedPoint);
    }
    return mergedData;
  }, [runsToCompare]);

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        
        {/* Saved Runs Panel */}
        <div className="lg:col-span-1 space-y-4">
          <div className="p-4 rounded-xl border border-border/40 bg-surface/40 backdrop-blur-sm">
            <h3 className="text-sm font-semibold mb-3 flex items-center gap-2">
              <GitBranch className="w-4 h-4 text-accent" />
              Saved Scenarios
            </h3>
            
            {savedRuns.length === 0 ? (
              <p className="text-xs text-muted mb-4">No scenarios saved yet.</p>
            ) : (
              <div className="space-y-2 mb-4">
                {savedRuns.map((run, idx) => {
                  const isSelected = selectedRuns.includes(run.id);
                  const isLoading = loadingIds.has(run.id);
                  const colorIndex = selectedRuns.indexOf(run.id);
                  const color = isSelected
                    ? COLORS[colorIndex % COLORS.length]
                    : "transparent";

                  return (
                    <div
                      key={run.id}
                      className={clsx(
                        "flex items-center justify-between p-2 rounded-lg border transition-all text-sm",
                        isSelected
                          ? "border-accent bg-accent/5"
                          : "border-border/40 hover:border-accent/40 bg-surface-2/30",
                        isLoading && "opacity-50 pointer-events-none"
                      )}
                    >
                      <button
                        className="flex-1 flex items-center gap-2 text-left"
                        onClick={() => toggleRun(run.id)}
                        disabled={isLoading}
                      >
                        <div
                          className="w-3 h-3 rounded-full border border-border/60"
                          style={{ backgroundColor: color }}
                        />
                        <span className="truncate">
                          {isLoading ? "Loading..." : run.name}
                        </span>
                      </button>

                      <button
                        onClick={() => {
                          onDeleteRun(run.id);
                          if (isSelected) toggleRun(run.id); // Deselect on delete
                        }}
                        className="ml-2 p-1.5 text-muted hover:text-danger rounded-md hover:bg-danger/10 transition-colors"
                        title="Delete run"
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  );
                })}
              </div>
            )}
            
            {/* Save Current Run Form */}
            {currentRun && (
              <div className="pt-4 border-t border-border/40">
                <p className="text-xs text-muted mb-2 font-medium">Save Current Run</p>
                <div className="flex gap-2">
                  <input
                    type="text"
                    value={saveName}
                    onChange={(e) => setSaveName(e.target.value)}
                    placeholder="e.g., Baseline High Taxes..."
                    className="flex-1 rounded-lg border border-border-2/60 bg-surface-2/80 px-2.5 py-1.5 text-xs text-foreground
                              focus:border-accent focus:outline-none focus:ring-1 focus:ring-accent/20"
                    onKeyDown={(e) => e.key === 'Enter' && handleSave()}
                  />
                  <button
                    onClick={handleSave}
                    disabled={!saveName.trim()}
                    className="p-1.5 rounded-lg bg-accent text-white hover:bg-accent/90 disabled:opacity-40 transition-all"
                  >
                    <Plus className="w-4 h-4" />
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Charts Panel */}
        <div className="lg:col-span-2">
          {runsToCompare.length === 0 ? (
            <div className="h-[500px] flex flex-col items-center justify-center border border-border/40 bg-surface/20 rounded-xl backdrop-blur-sm p-8 text-center">
              <Share2 className="w-12 h-12 text-muted-2 mb-4" />
              <h3 className="text-lg font-semibold text-foreground mb-2">Compare Scenarios</h3>
              <p className="text-sm text-muted max-w-sm">
                Select one or more saved scenarios from the sidebar to overlay their metrics and compare the outcomes of different economic policies.
              </p>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {METRICS.slice(0, 4).map(metric => (
                <div key={metric.key} className="p-4 rounded-xl border border-border/40 bg-surface/40 backdrop-blur-sm shadow-sm relative group">
                  <h3 className="text-xs font-semibold text-foreground mb-4 uppercase tracking-wider">
                    {metric.label}
                  </h3>
                  <div className="h-[200px] w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={comparisonData} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                        <XAxis 
                          dataKey="period" 
                          stroke="rgba(255,255,255,0.2)" 
                          fontSize={10} 
                          tickMargin={8} 
                        />
                        <YAxis 
                          stroke="rgba(255,255,255,0.2)" 
                          fontSize={10} 
                          tickFormatter={(val) => 
                            val >= 1000 ? `${(val / 1000).toFixed(1)}k` : val.toString()
                          }
                          width={40}
                        />
                        <Tooltip
                          contentStyle={{
                            backgroundColor: "rgba(10, 10, 15, 0.9)",
                            borderColor: "rgba(255,255,255,0.1)",
                            borderRadius: "8px",
                            boxShadow: "0 4px 20px rgba(0,0,0,0.5)",
                            fontSize: "12px",
                          }}
                          itemStyle={{ padding: "2px 0" }}
                          formatter={(value: any, name: string) => {
                            const runId = name.split('_')[0];
                            const runName = runsToCompare.find(r => r.id === runId)?.name || runId;
                            return [typeof value === 'number' && value < 1000 ? value.toFixed(2) : value, runName];
                          }}
                        />
                        <Legend wrapperStyle={{ fontSize: "10px", paddingTop: "10px" }} />
                        
                        {runsToCompare.map((run, i) => (
                          <Line
                            key={`${run.id}_${metric.key}`}
                            type="monotone"
                            dataKey={`${run.id}_${metric.key}`}
                            name={`${run.id}_${metric.key}`} // used for tooltip matching
                            stroke={COLORS[i % COLORS.length]}
                            strokeWidth={2}
                            dot={false}
                            activeDot={{ r: 4, strokeWidth: 0 }}
                          />
                        ))}
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
