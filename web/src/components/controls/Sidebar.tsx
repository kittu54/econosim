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
} from "lucide-react";
import clsx from "clsx";
import { SimulationRequest, DEFAULT_CONFIG } from "@/lib/types";

interface SidebarProps {
  onRun: (config: SimulationRequest) => void;
  isRunning: boolean;
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
      <label className="text-xs text-muted whitespace-nowrap">{label}</label>
      <div className="flex items-center gap-1">
        <input
          type="number"
          value={value}
          onChange={(e) => onChange(Number(e.target.value))}
          min={min}
          max={max}
          step={step}
          className="w-20 rounded-md border border-border-2 bg-surface-2 px-2 py-1 text-xs text-foreground text-right
                     focus:border-accent focus:outline-none focus:ring-1 focus:ring-accent/30"
        />
        {suffix && <span className="text-xs text-muted">{suffix}</span>}
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
    <div className="space-y-1">
      <div className="flex items-center justify-between">
        <label className="text-xs text-muted">{label}</label>
        <span className="text-xs font-medium text-foreground">{display}</span>
      </div>
      <input
        type="range"
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        min={min}
        max={max}
        step={step}
        className="w-full h-1.5 rounded-full appearance-none bg-border-2 cursor-pointer
                   accent-accent [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3
                   [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:rounded-full
                   [&::-webkit-slider-thumb]:bg-accent [&::-webkit-slider-thumb]:shadow-md"
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
    <div className="border-b border-border/50">
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center gap-2 w-full px-4 py-3 text-sm font-medium text-foreground
                   hover:bg-surface-2/50 transition-colors"
      >
        {icon}
        <span className="flex-1 text-left">{title}</span>
        {open ? (
          <ChevronDown className="w-4 h-4 text-muted" />
        ) : (
          <ChevronRight className="w-4 h-4 text-muted" />
        )}
      </button>
      {open && <div className="px-4 pb-4 space-y-3">{children}</div>}
    </div>
  );
}

export default function Sidebar({ onRun, isRunning }: SidebarProps) {
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

  return (
    <aside className="w-72 min-h-screen bg-surface border-r border-border flex flex-col">
      {/* Header */}
      <div className="px-5 py-5 border-b border-border">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-accent to-accent-2 flex items-center justify-center">
            <Landmark className="w-4 h-4 text-white" />
          </div>
          <div>
            <h1 className="text-base font-bold tracking-tight">EconoSim</h1>
            <p className="text-[10px] text-muted font-medium uppercase tracking-widest">
              Economic Simulation
            </p>
          </div>
        </div>
      </div>

      {/* Controls */}
      <div className="flex-1 overflow-y-auto">
        <Section
          title="Simulation"
          icon={<Settings className="w-4 h-4 text-accent" />}
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
          icon={<Users className="w-4 h-4 text-emerald-400" />}
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
          icon={<Factory className="w-4 h-4 text-amber-400" />}
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
          icon={<Landmark className="w-4 h-4 text-indigo-400" />}
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
          icon={<Building2 className="w-4 h-4 text-cyan-400" />}
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
      <div className="p-4 border-t border-border">
        <button
          onClick={() => onRun(config)}
          disabled={isRunning}
          className={clsx(
            "w-full flex items-center justify-center gap-2 px-4 py-3 rounded-xl",
            "text-sm font-semibold transition-all duration-200",
            isRunning
              ? "bg-surface-2 text-muted cursor-not-allowed"
              : "bg-gradient-to-r from-accent to-accent-2 text-white hover:shadow-lg hover:shadow-accent/25 active:scale-[0.98]"
          )}
        >
          {isRunning ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              Running...
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
