"use client";

import { useState, useCallback } from "react";
import {
  BarChart3,
  HardHat,
  Landmark,
  Coins,
  Table2,
  Activity,
  Users,
  DollarSign,
  Scale,
  CreditCard,
  Zap,
} from "lucide-react";
import clsx from "clsx";
import Sidebar from "@/components/controls/Sidebar";
import KpiCard from "@/components/KpiCard";
import MacroTab from "@/components/tabs/MacroTab";
import LaborTab from "@/components/tabs/LaborTab";
import GovernmentTab from "@/components/tabs/GovernmentTab";
import MoneyTab from "@/components/tabs/MoneyTab";
import DataTab from "@/components/tabs/DataTab";
import { runSimulation } from "@/lib/api";
import {
  SimulationRequest,
  SimulationResponse,
  PeriodData,
} from "@/lib/types";
import { fmtNumber, fmtPercent, fmtDelta, fmtDeltaPercent } from "@/lib/format";

type TabId = "macro" | "labor" | "government" | "money" | "data";

const TABS: { id: TabId; label: string; icon: React.ReactNode }[] = [
  { id: "macro", label: "Macro", icon: <BarChart3 className="w-4 h-4" /> },
  { id: "labor", label: "Labor & Production", icon: <HardHat className="w-4 h-4" /> },
  { id: "government", label: "Government", icon: <Landmark className="w-4 h-4" /> },
  { id: "money", label: "Money & Credit", icon: <Coins className="w-4 h-4" /> },
  { id: "data", label: "Data", icon: <Table2 className="w-4 h-4" /> },
];

function WelcomeScreen() {
  return (
    <div className="flex items-center justify-center min-h-[80vh]">
      <div className="max-w-2xl text-center space-y-8 animate-fade-in">
        <div className="flex justify-center">
          <div className="w-20 h-20 rounded-2xl bg-gradient-to-br from-accent to-accent-2 flex items-center justify-center shadow-2xl shadow-accent/20 animate-pulse-glow">
            <Landmark className="w-10 h-10 text-white" />
          </div>
        </div>
        <div>
          <h1 className="text-4xl font-bold tracking-tight mb-3">
            EconoSim Dashboard
          </h1>
          <p className="text-lg text-muted leading-relaxed">
            Multi-agent economic simulation with stock-flow consistent
            accounting. Households, firms, banks, and government interact in a
            closed economy.
          </p>
        </div>
        <div className="grid grid-cols-2 md:grid-cols-3 gap-3 max-w-lg mx-auto">
          {[
            { icon: <DollarSign className="w-5 h-5" />, label: "Double-entry accounting" },
            { icon: <Zap className="w-5 h-5" />, label: "Sovereign money" },
            { icon: <Users className="w-5 h-5" />, label: "Multi-agent RL" },
            { icon: <Activity className="w-5 h-5" />, label: "Real-time charts" },
            { icon: <Scale className="w-5 h-5" />, label: "Gini tracking" },
            { icon: <CreditCard className="w-5 h-5" />, label: "Credit markets" },
          ].map((f) => (
            <div
              key={f.label}
              className="flex items-center gap-2 rounded-lg border border-border bg-surface p-3 text-xs text-muted"
            >
              <span className="text-accent">{f.icon}</span>
              {f.label}
            </div>
          ))}
        </div>
        <div className="text-sm text-muted">
          <p>
            Configure parameters in the sidebar and click{" "}
            <span className="font-semibold text-accent">Run Simulation</span>{" "}
            to begin.
          </p>
        </div>
      </div>
    </div>
  );
}

export default function Dashboard() {
  const [result, setResult] = useState<SimulationResponse | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<TabId>("macro");

  const handleRun = useCallback(async (config: SimulationRequest) => {
    setIsRunning(true);
    setError(null);
    try {
      const res = await runSimulation(config);
      setResult(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to run simulation");
    } finally {
      setIsRunning(false);
    }
  }, []);

  const data = result?.periods ?? [];
  const aggregate = result?.aggregate ?? null;
  const final: PeriodData | null = data.length > 0 ? data[data.length - 1] : null;
  const first: PeriodData | null = data.length > 0 ? data[0] : null;

  function getDeltaDir(key: string): "up" | "down" | "neutral" {
    if (!first || !final) return "neutral";
    const diff = (final[key] ?? 0) - (first[key] ?? 0);
    if (diff > 0.001) return "up";
    if (diff < -0.001) return "down";
    return "neutral";
  }

  return (
    <div className="flex min-h-screen">
      <Sidebar onRun={handleRun} isRunning={isRunning} />

      <main className="flex-1 overflow-y-auto">
        {error && (
          <div className="mx-6 mt-4 px-4 py-3 rounded-xl bg-rose-500/10 border border-rose-500/20 text-rose-400 text-sm">
            <strong>Error:</strong> {error}
          </div>
        )}

        {!result ? (
          <WelcomeScreen />
        ) : (
          <div className="p-6 space-y-6 animate-fade-in">
            {/* KPI Row */}
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
              <KpiCard
                label="GDP"
                value={fmtNumber(final?.gdp)}
                delta={first ? fmtDelta(final!.gdp, first.gdp) : undefined}
                deltaDirection={getDeltaDir("gdp")}
                accentColor="from-blue-500/20 to-blue-600/5"
              />
              <KpiCard
                label="Unemployment"
                value={fmtPercent(final?.unemployment_rate)}
                delta={
                  first
                    ? fmtDeltaPercent(
                        final!.unemployment_rate,
                        first.unemployment_rate
                      )
                    : undefined
                }
                deltaDirection={getDeltaDir("unemployment_rate")}
                invertDelta
                accentColor="from-rose-500/20 to-rose-600/5"
              />
              <KpiCard
                label="Avg Price"
                value={final?.avg_price?.toFixed(2) ?? "\u2014"}
                delta={
                  first
                    ? `${(final!.avg_price - first.avg_price) >= 0 ? "+" : ""}${(final!.avg_price - first.avg_price).toFixed(2)}`
                    : undefined
                }
                deltaDirection={getDeltaDir("avg_price")}
                accentColor="from-amber-500/20 to-amber-600/5"
              />
              <KpiCard
                label="Avg Wage"
                value={fmtNumber(final?.avg_wage)}
                delta={
                  first ? fmtDelta(final!.avg_wage, first.avg_wage) : undefined
                }
                deltaDirection={getDeltaDir("avg_wage")}
                accentColor="from-emerald-500/20 to-emerald-600/5"
              />
              <KpiCard
                label="Gini"
                value={final?.gini_deposits?.toFixed(3) ?? "\u2014"}
                delta={
                  first
                    ? `${(final!.gini_deposits - first.gini_deposits) >= 0 ? "+" : ""}${(final!.gini_deposits - first.gini_deposits).toFixed(3)}`
                    : undefined
                }
                deltaDirection={getDeltaDir("gini_deposits")}
                invertDelta
                accentColor="from-purple-500/20 to-purple-600/5"
              />
              <KpiCard
                label="Loans"
                value={fmtNumber(final?.total_loans_outstanding)}
                delta={
                  first
                    ? fmtDelta(
                        final!.total_loans_outstanding,
                        first.total_loans_outstanding
                      )
                    : undefined
                }
                deltaDirection={getDeltaDir("total_loans_outstanding")}
                accentColor="from-cyan-500/20 to-cyan-600/5"
              />
            </div>

            {/* Tabs */}
            <div className="flex items-center gap-1 border-b border-border pb-px overflow-x-auto">
              {TABS.map((tab) => (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={clsx(
                    "flex items-center gap-2 px-4 py-2.5 text-sm font-medium rounded-t-lg transition-all whitespace-nowrap",
                    activeTab === tab.id
                      ? "text-accent border-b-2 border-accent bg-accent/5"
                      : "text-muted hover:text-foreground hover:bg-surface-2/50"
                  )}
                >
                  {tab.icon}
                  {tab.label}
                </button>
              ))}
            </div>

            {/* Tab content */}
            {activeTab === "macro" && (
              <MacroTab data={data} aggregate={aggregate} />
            )}
            {activeTab === "labor" && (
              <LaborTab data={data} aggregate={aggregate} />
            )}
            {activeTab === "government" && (
              <GovernmentTab data={data} aggregate={aggregate} />
            )}
            {activeTab === "money" && (
              <MoneyTab data={data} aggregate={aggregate} />
            )}
            {activeTab === "data" && (
              <DataTab data={data} summary={result.summary} />
            )}
          </div>
        )}
      </main>
    </div>
  );
}
