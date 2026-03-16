"use client";

import { useState } from "react";
import {
  Play,
  Settings,
  Users,
  Factory,
  Landmark,
  Building2,
  ChevronDown,
  ChevronRight,
  Loader2,
  RotateCcw,
  Sparkles,
  PanelLeftClose,
  PanelLeftOpen,
  Zap,
  TrendingDown,
  Shield,
} from "lucide-react";
import clsx from "clsx";
import { SimulationRequest, DEFAULT_CONFIG } from "@/lib/types";

interface SidebarProps {
  onRun: (config: SimulationRequest) => void;
  isRunning: boolean;
  collapsed?: boolean;
  onToggle?: () => void;
}

interface NumberInputProps {
  label: string;
  value: number;
  onChange: (v: number) => void;
  min?: number;
  max?: number;
  step?: number;
  suffix?: string;
}

function NumberInput({
  label,
  value,
  onChange,
  min,
  max,
  step = 1,
  suffix,
}: NumberInputProps) {
  return (
    <div className="flex items-center justify-between gap-2">
      <label className="text-[11px] text-muted whitespace-nowrap">{label}</label>
      <div className="flex items-center gap-1">
        <input
          type="number"
          value={value}
          onChange={(e) => onChange(Number(e.target.value))}
          min={min}
          max={max}
          step={step}
          className="w-20 rounded-lg border border-border-2/60 bg-surface-2/80 px-2.5 py-1.5 text-xs text-foreground text-right
                     focus:border-accent focus:outline-none focus:ring-1 focus:ring-accent/20 transition-all"
        />
        {suffix && <span className="text-[10px] text-muted-2">{suffix}</span>}
      </div>
    </div>
  );
}

interface SliderInputProps {
  label: string;
  value: number;
  onChange: (v: number) => void;
  min: number;
  max: number;
  step: number;
  format?: (v: number) => string;
}

function SliderInput({
  label,
  value,
  onChange,
  min,
  max,
  step,
  format,
}: SliderInputProps) {
  const display = format ? format(value) : value.toFixed(2);
  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between">
        <label className="text-[11px] text-muted">{label}</label>
        <span className="text-[11px] font-medium text-accent tabular-nums">
          {display}
        </span>
      </div>
      <input
        type="range"
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        min={min}
        max={max}
        step={step}
        className="w-full"
      />
    </div>
  );
}

interface SectionProps {
  title: string;
  icon: React.ReactNode;
  children: React.ReactNode;
  defaultOpen?: boolean;
}

function Section({ title, icon, children, defaultOpen = false }: SectionProps) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="border-b border-border/40">
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center gap-2.5 w-full px-4 py-3 text-xs font-semibold text-foreground
                   hover:bg-surface-2/40 transition-all duration-200 uppercase tracking-wider"
      >
        <span className="opacity-70">{icon}</span>
        <span className="flex-1 text-left">{title}</span>
        <span className="text-muted-2 transition-transform duration-200">
          {open ? (
            <ChevronDown className="w-3.5 h-3.5" />
          ) : (
            <ChevronRight className="w-3.5 h-3.5" />
          )}
        </span>
      </button>
      <div
        className={clsx(
          "overflow-hidden transition-all duration-300",
          open ? "max-h-[600px] opacity-100" : "max-h-0 opacity-0"
        )}
      >
        <div className="px-4 pb-4 space-y-3">{children}</div>
      </div>
    </div>
  );
}

const PRESETS: {
  name: string;
  icon: React.ReactNode;
  description: string;
  config: Partial<SimulationRequest>;
}[] = [
  {
    name: "Baseline",
    icon: <Sparkles className="w-3.5 h-3.5" />,
    description: "Default balanced economy",
    config: { ...DEFAULT_CONFIG },
  },
  {
    name: "High Growth",
    icon: <Zap className="w-3.5 h-3.5" />,
    description: "Expansionary fiscal policy",
    config: {
      government: {
        ...DEFAULT_CONFIG.government,
        spending_per_period: 5000,
        income_tax_rate: 0.1,
      },
    },
  },
  {
    name: "Recession",
    icon: <TrendingDown className="w-3.5 h-3.5" />,
    description: "Contractionary scenario",
    config: {
      government: {
        ...DEFAULT_CONFIG.government,
        spending_per_period: 500,
        income_tax_rate: 0.35,
      },
      household: {
        ...DEFAULT_CONFIG.household,
        consumption_propensity: 0.5,
      },
    },
  },
  {
    name: "Tight Money",
    icon: <Shield className="w-3.5 h-3.5" />,
    description: "High rates, strict capital rules",
    config: {
      bank: {
        base_interest_rate: 0.03,
        capital_adequacy_ratio: 0.15,
      },
    },
  },
];

export default function Sidebar({
  onRun,
  isRunning,
  collapsed = false,
  onToggle,
}: SidebarProps) {
  const [config, setConfig] = useState<SimulationRequest>({
    ...DEFAULT_CONFIG,
  });

  const updateHousehold = (key: string, value: number) =>
    setConfig((c) => ({
      ...c,
      household: { ...c.household, [key]: value },
    }));
  const updateFirm = (key: string, value: number) =>
    setConfig((c) => ({ ...c, firm: { ...c.firm, [key]: value } }));
  const updateGovt = (key: string, value: number) =>
    setConfig((c) => ({
      ...c,
      government: { ...c.government, [key]: value },
    }));
  const updateBank = (key: string, value: number) =>
    setConfig((c) => ({ ...c, bank: { ...c.bank, [key]: value } }));

  const applyPreset = (preset: (typeof PRESETS)[number]) => {
    setConfig((c) => ({
      ...c,
      ...preset.config,
      household: { ...c.household, ...preset.config.household },
      firm: { ...c.firm, ...preset.config.firm },
      government: { ...c.government, ...preset.config.government },
      bank: { ...c.bank, ...preset.config.bank },
    }));
  };

  if (collapsed) {
    return (
      <aside className="w-14 min-h-screen bg-surface/80 backdrop-blur-sm border-r border-border/60 flex flex-col items-center py-4 gap-3">
        <button
          onClick={onToggle}
          className="p-2 rounded-lg hover:bg-surface-2 text-muted hover:text-foreground transition-all"
          title="Expand sidebar"
        >
          <PanelLeftOpen className="w-4 h-4" />
        </button>
        <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-accent to-accent-2 flex items-center justify-center">
          <Landmark className="w-4 h-4 text-white" />
        </div>
        <div className="flex-1" />
        <button
          onClick={() => onRun(config)}
          disabled={isRunning}
          className="p-2.5 rounded-xl bg-accent text-white hover:bg-accent/90 disabled:opacity-40 transition-all"
          title="Run simulation"
        >
          {isRunning ? (
            <Loader2 className="w-4 h-4 animate-spin" />
          ) : (
            <Play className="w-4 h-4" />
          )}
        </button>
      </aside>
    );
  }

  return (
    <aside className="w-72 min-h-screen bg-surface/80 backdrop-blur-sm border-r border-border/60 flex flex-col relative z-10">
      {/* Header */}
      <div className="px-4 py-4 border-b border-border/40">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-accent to-accent-2 flex items-center justify-center shadow-lg shadow-accent/20">
              <Landmark className="w-4 h-4 text-white" />
            </div>
            <div>
              <h1 className="text-sm font-bold tracking-tight">EconoSim</h1>
              <p className="text-[9px] text-muted font-medium uppercase tracking-[0.15em]">
                Economic Simulation
              </p>
            </div>
          </div>
          {onToggle && (
            <button
              onClick={onToggle}
              className="p-1.5 rounded-md hover:bg-surface-2 text-muted hover:text-foreground transition-all"
            >
              <PanelLeftClose className="w-3.5 h-3.5" />
            </button>
          )}
        </div>
      </div>

      {/* Scenario Presets */}
      <div className="px-4 py-3 border-b border-border/40">
        <div className="flex items-center justify-between mb-2">
          <span className="text-[10px] font-semibold uppercase tracking-wider text-muted">
            Presets
          </span>
          <button
            onClick={() => setConfig({ ...DEFAULT_CONFIG })}
            className="flex items-center gap-1 text-[10px] text-muted hover:text-accent transition-colors"
            title="Reset to defaults"
          >
            <RotateCcw className="w-3 h-3" />
            Reset
          </button>
        </div>
        <div className="grid grid-cols-2 gap-1.5">
          {PRESETS.map((p) => (
            <button
              key={p.name}
              onClick={() => applyPreset(p)}
              className="flex items-center gap-1.5 px-2.5 py-2 rounded-lg text-[10px] font-medium
                         border border-border/60 bg-surface-2/40 text-muted
                         hover:border-accent/40 hover:text-accent hover:bg-accent/5
                         transition-all duration-200"
              title={p.description}
            >
              {p.icon}
              {p.name}
            </button>
          ))}
        </div>
      </div>

      {/* Controls */}
      <div className="flex-1 overflow-y-auto">
        <Section
          title="Simulation"
          icon={<Settings className="w-3.5 h-3.5 text-accent" />}
          defaultOpen={true}
        >
          <NumberInput
            label="Periods"
            value={config.num_periods}
            onChange={(v) => setConfig((c) => ({ ...c, num_periods: v }))}
            min={5}
            max={500}
            step={10}
          />
          <NumberInput
            label="Seed"
            value={config.seed}
            onChange={(v) => setConfig((c) => ({ ...c, seed: v }))}
            min={0}
            max={99999}
          />
          <NumberInput
            label="Batch runs"
            value={config.n_seeds}
            onChange={(v) => setConfig((c) => ({ ...c, n_seeds: v }))}
            min={1}
            max={20}
          />
        </Section>

        <Section
          title="Households"
          icon={<Users className="w-3.5 h-3.5 text-emerald-400" />}
        >
          <NumberInput
            label="Count"
            value={config.household.count}
            onChange={(v) => updateHousehold("count", v)}
            min={10}
            max={500}
            step={10}
          />
          <NumberInput
            label="Initial deposits"
            value={config.household.initial_deposits}
            onChange={(v) => updateHousehold("initial_deposits", v)}
            min={100}
            max={50000}
            step={100}
          />
          <SliderInput
            label="Consumption propensity"
            value={config.household.consumption_propensity}
            onChange={(v) => updateHousehold("consumption_propensity", v)}
            min={0.1}
            max={1.0}
            step={0.05}
          />
          <SliderInput
            label="Wealth propensity"
            value={config.household.wealth_propensity}
            onChange={(v) => updateHousehold("wealth_propensity", v)}
            min={0.0}
            max={1.0}
            step={0.05}
          />
          <NumberInput
            label="Reservation wage"
            value={config.household.reservation_wage}
            onChange={(v) => updateHousehold("reservation_wage", v)}
            min={0}
            max={200}
            step={10}
          />
        </Section>

        <Section
          title="Firms"
          icon={<Factory className="w-3.5 h-3.5 text-amber-400" />}
        >
          <NumberInput
            label="Count"
            value={config.firm.count}
            onChange={(v) => updateFirm("count", v)}
            min={1}
            max={50}
          />
          <NumberInput
            label="Initial deposits"
            value={config.firm.initial_deposits}
            onChange={(v) => updateFirm("initial_deposits", v)}
            min={1000}
            max={100000}
            step={1000}
          />
          <NumberInput
            label="Initial price"
            value={config.firm.initial_price}
            onChange={(v) => updateFirm("initial_price", v)}
            min={1}
            max={100}
          />
          <NumberInput
            label="Initial wage"
            value={config.firm.initial_wage}
            onChange={(v) => updateFirm("initial_wage", v)}
            min={10}
            max={500}
            step={10}
          />
          <NumberInput
            label="Productivity"
            value={config.firm.labor_productivity}
            onChange={(v) => updateFirm("labor_productivity", v)}
            min={1}
            max={50}
          />
          <SliderInput
            label="Price adj. speed"
            value={config.firm.price_adjustment_speed}
            onChange={(v) => updateFirm("price_adjustment_speed", v)}
            min={0.01}
            max={0.2}
            step={0.01}
          />
          <SliderInput
            label="Wage adj. speed"
            value={config.firm.wage_adjustment_speed}
            onChange={(v) => updateFirm("wage_adjustment_speed", v)}
            min={0.01}
            max={0.2}
            step={0.01}
          />
        </Section>

        <Section
          title="Government"
          icon={<Landmark className="w-3.5 h-3.5 text-indigo-400" />}
        >
          <SliderInput
            label="Income tax rate"
            value={config.government.income_tax_rate}
            onChange={(v) => updateGovt("income_tax_rate", v)}
            min={0}
            max={0.5}
            step={0.05}
            format={(v) => `${(v * 100).toFixed(0)}%`}
          />
          <NumberInput
            label="Transfer / unemployed"
            value={config.government.transfer_per_unemployed}
            onChange={(v) => updateGovt("transfer_per_unemployed", v)}
            min={0}
            max={500}
            step={10}
          />
          <NumberInput
            label="Spending / period"
            value={config.government.spending_per_period}
            onChange={(v) => updateGovt("spending_per_period", v)}
            min={0}
            max={20000}
            step={200}
          />
          <NumberInput
            label="Initial deposits"
            value={config.government.initial_deposits}
            onChange={(v) => updateGovt("initial_deposits", v)}
            min={10000}
            max={1000000}
            step={10000}
          />
        </Section>

        <Section
          title="Banking"
          icon={<Building2 className="w-3.5 h-3.5 text-cyan-400" />}
        >
          <SliderInput
            label="Base interest rate"
            value={config.bank.base_interest_rate}
            onChange={(v) => updateBank("base_interest_rate", v)}
            min={0}
            max={0.05}
            step={0.001}
            format={(v) => `${(v * 100).toFixed(1)}%`}
          />
          <SliderInput
            label="Capital adequacy"
            value={config.bank.capital_adequacy_ratio}
            onChange={(v) => updateBank("capital_adequacy_ratio", v)}
            min={0.01}
            max={0.2}
            step={0.01}
            format={(v) => `${(v * 100).toFixed(0)}%`}
          />
        </Section>
      </div>

      {/* Run button */}
      <div className="p-4 border-t border-border/40">
        <button
          onClick={() => onRun(config)}
          disabled={isRunning}
          className={clsx(
            "w-full flex items-center justify-center gap-2 px-4 py-3 rounded-xl",
            "text-sm font-semibold transition-all duration-200",
            isRunning
              ? "bg-surface-2 text-muted cursor-not-allowed"
              : "bg-gradient-to-r from-accent to-accent-2 text-white hover:shadow-lg hover:shadow-accent/30 active:scale-[0.98] glow-blue-sm"
          )}
        >
          {isRunning ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              Simulating...
            </>
          ) : (
            <>
              <Play className="w-4 h-4" />
              Run Simulation
            </>
          )}
        </button>
      </div>
    </aside>
  );
}
