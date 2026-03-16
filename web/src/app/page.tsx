"use client";

import { useState, useCallback, useMemo } from "react";
import Link from "next/link";
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
  Cpu,
  GitBranch,
  TrendingUp,
  ArrowRight,
  PlayCircle,
  BookOpen,
  Layers,
  Factory,
  Building2,
  Repeat,
} from "lucide-react";
import clsx from "clsx";
import Sidebar from "@/components/controls/Sidebar";
import KpiCard from "@/components/KpiCard";
import { DashboardSkeleton } from "@/components/LoadingSkeleton";
import MacroTab from "@/components/tabs/MacroTab";
import LaborTab from "@/components/tabs/LaborTab";
import GovernmentTab from "@/components/tabs/GovernmentTab";
import MoneyTab from "@/components/tabs/MoneyTab";
import DataTab from "@/components/tabs/DataTab";
import ExtensionsTab from "@/components/tabs/ExtensionsTab";
import CompareTab from "@/components/tabs/CompareTab";
import { runSimulation } from "@/lib/api";
import {
  SimulationRequest,
  SimulationResponse,
  PeriodData,
} from "@/lib/types";
import { fmtNumber, fmtPercent, fmtDelta, fmtDeltaPercent } from "@/lib/format";

type TabId = "macro" | "labor" | "government" | "money" | "extensions" | "compare" | "data";

const TABS: { id: TabId; label: string; icon: React.ReactNode }[] = [
  { id: "macro", label: "Macro", icon: <BarChart3 className="w-4 h-4" /> },
  { id: "labor", label: "Labor & Production", icon: <HardHat className="w-4 h-4" /> },
  { id: "government", label: "Government", icon: <Landmark className="w-4 h-4" /> },
  { id: "money", label: "Money & Credit", icon: <Coins className="w-4 h-4" /> },
  { id: "extensions", label: "Extensions", icon: <Activity className="w-4 h-4" /> },
  { id: "compare", label: "Compare", icon: <GitBranch className="w-4 h-4" /> },
  { id: "data", label: "Data", icon: <Table2 className="w-4 h-4" /> },
];

/* ──────────────────────────────────────────────
   Feature cards — each links to a docs section
   ────────────────────────────────────────────── */

const FEATURES = [
  {
    icon: <DollarSign className="w-5 h-5" />,
    label: "Double-Entry Accounting",
    desc: "Every flow tracked via A-L=E balance sheets",
    href: "/docs#accounting",
    color: "emerald",
  },
  {
    icon: <Zap className="w-5 h-5" />,
    label: "Endogenous Money",
    desc: "Bank loans create deposits, repayment destroys them",
    href: "/docs#accounting",
    color: "amber",
  },
  {
    icon: <Users className="w-5 h-5" />,
    label: "4 Agent Types",
    desc: "Households, firms, banks, and government",
    href: "/docs#agents",
    color: "accent",
  },
  {
    icon: <Repeat className="w-5 h-5" />,
    label: "10-Step Simulation",
    desc: "Strict ordering: credit → labor → production → goods",
    href: "/docs#simulation",
    color: "indigo",
  },
  {
    icon: <Scale className="w-5 h-5" />,
    label: "Inequality & Metrics",
    desc: "GDP, Gini, inflation, unemployment — all tracked",
    href: "/docs#metrics",
    color: "violet",
  },
  {
    icon: <CreditCard className="w-5 h-5" />,
    label: "Credit & Banking",
    desc: "Loan approval, capital adequacy, defaults",
    href: "/docs#markets",
    color: "cyan",
  },
  {
    icon: <Cpu className="w-5 h-5" />,
    label: "RL-Ready",
    desc: "Gymnasium & PettingZoo envs for all agent types",
    href: "/docs#rl",
    color: "rose",
  },
  {
    icon: <GitBranch className="w-5 h-5" />,
    label: "Experiments",
    desc: "Batch runs, parameter sweeps, CI bands",
    href: "/docs#experiments",
    color: "teal",
  },
];

const FEATURE_COLORS: Record<string, { icon: string; border: string; bg: string }> = {
  emerald: { icon: "text-emerald-400", border: "hover:border-emerald-500/30", bg: "hover:bg-emerald-500/5" },
  amber: { icon: "text-amber-400", border: "hover:border-amber-500/30", bg: "hover:bg-amber-500/5" },
  accent: { icon: "text-accent", border: "hover:border-accent/30", bg: "hover:bg-accent/5" },
  indigo: { icon: "text-indigo-400", border: "hover:border-indigo-500/30", bg: "hover:bg-indigo-500/5" },
  violet: { icon: "text-violet-400", border: "hover:border-violet-500/30", bg: "hover:bg-violet-500/5" },
  cyan: { icon: "text-cyan-400", border: "hover:border-cyan-500/30", bg: "hover:bg-cyan-500/5" },
  rose: { icon: "text-rose-400", border: "hover:border-rose-500/30", bg: "hover:bg-rose-500/5" },
  teal: { icon: "text-teal-400", border: "hover:border-teal-500/30", bg: "hover:bg-teal-500/5" },
};

/* ──────────────────────────────────────────────
   Welcome / Landing Screen
   ────────────────────────────────────────────── */

function WelcomeScreen() {
  return (
    <div className="flex flex-col items-center justify-center min-h-[calc(100vh-3rem)] px-6 py-12">
      <div className="max-w-4xl w-full space-y-12 animate-fade-in">
        {/* Hero */}
        <div className="text-center space-y-5">
          <div className="flex justify-center">
            <div className="relative">
              <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-accent via-accent-2 to-accent-3 flex items-center justify-center shadow-2xl animate-float">
                <Landmark className="w-8 h-8 text-white" />
              </div>
              <div className="absolute inset-0 rounded-2xl bg-gradient-to-br from-accent to-accent-2 blur-xl opacity-20 animate-pulse-glow" />
            </div>
          </div>
          <div>
            <h1 className="text-3xl md:text-4xl font-bold tracking-tight mb-3">
              EconoSim Dashboard
            </h1>
            <p className="text-sm md:text-base text-muted leading-relaxed max-w-lg mx-auto">
              Multi-agent economic simulation with stock-flow consistent accounting.
              Watch macroeconomic dynamics emerge from micro-level agent interactions.
            </p>
          </div>
        </div>

        {/* Quick-start steps */}
        <div className="max-w-lg mx-auto">
          <div className="flex items-center gap-2 mb-4">
            <PlayCircle className="w-4 h-4 text-accent" />
            <h2 className="text-xs font-semibold uppercase tracking-wider text-muted">Quick Start</h2>
          </div>
          <div className="space-y-2">
            {[
              { step: "1", text: "Configure parameters in the sidebar", sub: "or choose a preset: Baseline, High Growth, Recession, Tight Money" },
              { step: "2", text: "Click Run Simulation", sub: "the API runs the engine and returns period-by-period results" },
              { step: "3", text: "Explore results across 7 tabs", sub: "Macro, Labor, Government, Money, Extensions, Compare, Data" },
            ].map((s) => (
              <div key={s.step} className="flex items-start gap-3 p-3 rounded-xl bg-surface/40 border border-border/50 group hover:border-accent/20 transition-colors">
                <span className="shrink-0 w-7 h-7 rounded-lg bg-accent/10 text-accent text-xs font-bold flex items-center justify-center group-hover:bg-accent/15 transition-colors">
                  {s.step}
                </span>
                <div className="min-w-0">
                  <p className="text-sm font-medium">{s.text}</p>
                  <p className="text-xs text-muted mt-0.5">{s.sub}</p>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Feature grid — linked to docs */}
        <div>
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xs font-semibold uppercase tracking-wider text-muted">Core Features</h2>
            <Link
              href="/docs"
              className="flex items-center gap-1 text-xs text-accent hover:text-accent/80 transition-colors"
            >
              Full documentation <ArrowRight className="w-3 h-3" />
            </Link>
          </div>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-2.5 stagger-children">
            {FEATURES.map((f) => {
              const colors = FEATURE_COLORS[f.color] || FEATURE_COLORS.accent;
              return (
                <Link
                  key={f.label}
                  href={f.href}
                  className={clsx(
                    "flex flex-col items-center gap-2 rounded-xl border border-border/60 bg-surface/40 backdrop-blur-sm p-4",
                    "transition-all duration-300 group animate-fade-in-scale",
                    colors.border,
                    colors.bg
                  )}
                  style={{ opacity: 0 }}
                >
                  <span className={clsx("transition-colors", "text-muted", `group-hover:${colors.icon}`)}>
                    <span className={clsx("block", colors.icon)}>
                      {f.icon}
                    </span>
                  </span>
                  <span className="text-[11px] font-semibold text-foreground text-center leading-tight">
                    {f.label}
                  </span>
                  <span className="text-[10px] text-muted leading-snug text-center">
                    {f.desc}
                  </span>
                </Link>
              );
            })}
          </div>
        </div>

        {/* Footer info bar */}
        <div className="flex flex-wrap items-center justify-center gap-x-6 gap-y-2 text-[10px] text-muted-2 uppercase tracking-wider">
          <span className="flex items-center gap-1.5">
            <BookOpen className="w-3 h-3" />
            494 Tests
          </span>
          <span className="hidden sm:block w-1 h-1 rounded-full bg-muted-2" />
          <span>Next.js + FastAPI</span>
          <span className="hidden sm:block w-1 h-1 rounded-full bg-muted-2" />
          <span>Gymnasium RL</span>
          <span className="hidden sm:block w-1 h-1 rounded-full bg-muted-2" />
          <span>SFC Accounting</span>
        </div>
      </div>
    </div>
  );
}

/* ──────────────────────────────────────────────
   Dashboard
   ────────────────────────────────────────────── */

export default function Dashboard() {
  const [result, setResult] = useState<SimulationResponse | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<TabId>("macro");
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [runCount, setRunCount] = useState(0);
  const [savedRuns, setSavedRuns] = useState<
    { name: string; data: PeriodData[]; config: Record<string, unknown> }[]
  >([]);

  const handleRun = useCallback(async (config: SimulationRequest) => {
    setIsRunning(true);
    setError(null);
    try {
      const res = await runSimulation(config);
      setResult(res);
      setRunCount((c) => c + 1);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Failed to run simulation"
      );
    } finally {
      setIsRunning(false);
    }
  }, []);

  const handleSaveRun = useCallback(() => {
    if (!result) return;
    const name = `Run #${runCount}`;
    setSavedRuns((prev) => [
      ...prev.slice(-4), // keep last 5
      { name, data: result.periods, config: result.config },
    ]);
  }, [result, runCount]);

  const data = result?.periods ?? [];
  const aggregate = result?.aggregate ?? null;
  const final: PeriodData | null =
    data.length > 0 ? data[data.length - 1] : null;
  const first: PeriodData | null = data.length > 0 ? data[0] : null;

  // Sparkline data: extract last N values for each KPI
  const sparkline = useMemo(() => {
    if (data.length < 3) return null;
    const sample = data.length > 30 ? data.filter((_, i) => i % Math.ceil(data.length / 30) === 0 || i === data.length - 1) : data;
    return {
      gdp: sample.map((d) => d.gdp),
      unemployment: sample.map((d) => d.unemployment_rate),
      price: sample.map((d) => d.avg_price),
      wage: sample.map((d) => d.avg_wage),
      gini: sample.map((d) => d.gini_deposits),
      loans: sample.map((d) => d.total_loans_outstanding),
    };
  }, [data]);

  function getDeltaDir(key: string): "up" | "down" | "neutral" {
    if (!first || !final) return "neutral";
    const diff = (final[key] ?? 0) - (first[key] ?? 0);
    if (diff > 0.001) return "up";
    if (diff < -0.001) return "down";
    return "neutral";
  }

  return (
    <div className="flex min-h-[calc(100vh-3rem)] relative z-10">
      <Sidebar
        onRun={handleRun}
        isRunning={isRunning}
        collapsed={sidebarCollapsed}
        onToggle={() => setSidebarCollapsed(!sidebarCollapsed)}
      />

      <main className="flex-1 overflow-y-auto">
        {/* Error banner */}
        {error && (
          <div className="mx-6 mt-4 px-4 py-3 rounded-xl bg-danger/10 border border-danger/20 text-danger text-sm animate-fade-in flex items-start gap-2">
            <span className="shrink-0 mt-0.5 w-4 h-4 rounded-full bg-danger/20 flex items-center justify-center text-[10px] font-bold">!</span>
            <div>
              <strong>Error:</strong> {error}
            </div>
          </div>
        )}

        {/* Loading skeleton */}
        {isRunning && !result && <DashboardSkeleton />}

        {/* Welcome screen */}
        {!result && !isRunning && <WelcomeScreen />}

        {/* Results dashboard */}
        {result && (
          <div className="p-6 space-y-5">
            {/* Run indicator */}
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2 text-[11px] text-muted">
                <TrendingUp className="w-3.5 h-3.5 text-accent" />
                <span>
                  Run #{runCount} &middot; {data.length} periods &middot;{" "}
                  {result.has_ci ? "Batch (CI bands)" : "Single run"}
                </span>
              </div>
              {isRunning && (
                <div className="flex items-center gap-2 text-[11px] text-accent">
                  <div className="w-3 h-3 border-2 border-accent/30 border-t-accent rounded-full animate-spin" />
                  <span>Running...</span>
                </div>
              )}
            </div>

            {/* KPI Row */}
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3 stagger-children">
              <KpiCard
                label="GDP"
                value={fmtNumber(final?.gdp)}
                delta={first ? fmtDelta(final!.gdp, first.gdp) : undefined}
                deltaDirection={getDeltaDir("gdp")}
                accentColor="from-blue-500/20 to-blue-600/5"
                sparklineData={sparkline?.gdp}
                sparklineColor="#3b82f6"
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
                sparklineData={sparkline?.unemployment}
                sparklineColor="#f43f5e"
              />
              <KpiCard
                label="Avg Price"
                value={final?.avg_price?.toFixed(2) ?? "\u2014"}
                delta={
                  first
                    ? `${final!.avg_price - first.avg_price >= 0 ? "+" : ""}${(
                        final!.avg_price - first.avg_price
                      ).toFixed(2)}`
                    : undefined
                }
                deltaDirection={getDeltaDir("avg_price")}
                accentColor="from-amber-500/20 to-amber-600/5"
                sparklineData={sparkline?.price}
                sparklineColor="#f59e0b"
              />
              <KpiCard
                label="Avg Wage"
                value={fmtNumber(final?.avg_wage)}
                delta={
                  first
                    ? fmtDelta(final!.avg_wage, first.avg_wage)
                    : undefined
                }
                deltaDirection={getDeltaDir("avg_wage")}
                accentColor="from-emerald-500/20 to-emerald-600/5"
                sparklineData={sparkline?.wage}
                sparklineColor="#10b981"
              />
              <KpiCard
                label="Gini"
                value={final?.gini_deposits?.toFixed(3) ?? "\u2014"}
                delta={
                  first
                    ? `${
                        final!.gini_deposits - first.gini_deposits >= 0
                          ? "+"
                          : ""
                      }${(
                        final!.gini_deposits - first.gini_deposits
                      ).toFixed(3)}`
                    : undefined
                }
                deltaDirection={getDeltaDir("gini_deposits")}
                invertDelta
                accentColor="from-purple-500/20 to-purple-600/5"
                sparklineData={sparkline?.gini}
                sparklineColor="#a855f7"
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
                sparklineData={sparkline?.loans}
                sparklineColor="#06b6d4"
              />
            </div>

            {/* Tabs */}
            <div className="flex items-center gap-1 border-b border-border/40 pb-px overflow-x-auto">
              {TABS.map((tab) => (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={clsx(
                    "flex items-center gap-2 px-4 py-2.5 text-xs font-semibold rounded-t-lg transition-all duration-200 whitespace-nowrap",
                    activeTab === tab.id
                      ? "text-accent border-b-2 border-accent bg-accent/5"
                      : "text-muted hover:text-foreground hover:bg-surface-2/30"
                  )}
                >
                  {tab.icon}
                  {tab.label}
                </button>
              ))}
            </div>

            {/* Tab content */}
            <div key={activeTab} className="animate-fade-in">
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
              {activeTab === "extensions" && (
                <ExtensionsTab data={data} aggregate={aggregate} />
              )}
              {activeTab === "compare" && (
                <CompareTab
                  currentData={data}
                  currentName={`Run #${runCount}`}
                  savedRuns={savedRuns}
                  onSave={handleSaveRun}
                  onRemove={(idx) =>
                    setSavedRuns((prev) => prev.filter((_, i) => i !== idx))
                  }
                />
              )}
              {activeTab === "data" && (
                <DataTab data={data} summary={result.summary} />
              )}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
