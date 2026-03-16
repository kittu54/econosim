"use client";

import { useState } from "react";
import Link from "next/link";
import {
  BookOpen,
  Layers,
  Users,
  Factory,
  Landmark,
  Building2,
  Zap,
  Activity,
  GitBranch,
  Scale,
  CreditCard,
  Cpu,
  ArrowRight,
  ChevronDown,
  ChevronRight,
  DollarSign,
  Network,
  TrendingUp,
  BarChart3,
  Settings,
  FlaskConical,
  Workflow,
  ShieldCheck,
  CircleDot,
  Repeat,
  ArrowLeftRight,
} from "lucide-react";
import clsx from "clsx";

/* ──────────────────────────────────────────────
   Types
   ────────────────────────────────────────────── */

interface SectionProps {
  id: string;
  title: string;
  icon: React.ReactNode;
  children: React.ReactNode;
}

interface AccordionProps {
  title: string;
  children: React.ReactNode;
  defaultOpen?: boolean;
}

/* ──────────────────────────────────────────────
   Sidebar TOC
   ────────────────────────────────────────────── */

const TOC = [
  { id: "overview", label: "Overview", icon: <BookOpen className="w-3.5 h-3.5" /> },
  { id: "architecture", label: "Architecture", icon: <Layers className="w-3.5 h-3.5" /> },
  { id: "agents", label: "Agents", icon: <Users className="w-3.5 h-3.5" /> },
  { id: "markets", label: "Markets", icon: <ArrowLeftRight className="w-3.5 h-3.5" /> },
  { id: "accounting", label: "Accounting", icon: <DollarSign className="w-3.5 h-3.5" /> },
  { id: "simulation", label: "Simulation Loop", icon: <Repeat className="w-3.5 h-3.5" /> },
  { id: "policy", label: "Policy & Shocks", icon: <Zap className="w-3.5 h-3.5" /> },
  { id: "extensions", label: "Extensions", icon: <Activity className="w-3.5 h-3.5" /> },
  { id: "metrics", label: "Metrics", icon: <BarChart3 className="w-3.5 h-3.5" /> },
  { id: "rl", label: "RL Environments", icon: <Cpu className="w-3.5 h-3.5" /> },
  { id: "experiments", label: "Experiments", icon: <FlaskConical className="w-3.5 h-3.5" /> },
  { id: "api", label: "API Reference", icon: <Settings className="w-3.5 h-3.5" /> },
  { id: "parameters", label: "Parameter Guide", icon: <Scale className="w-3.5 h-3.5" /> },
];

/* ──────────────────────────────────────────────
   Reusable components
   ────────────────────────────────────────────── */

function DocSection({ id, title, icon, children }: SectionProps) {
  return (
    <section id={id} className="scroll-mt-20 mb-16">
      <div className="flex items-center gap-3 mb-6">
        <div className="w-9 h-9 rounded-xl bg-accent/10 border border-accent/20 flex items-center justify-center text-accent">
          {icon}
        </div>
        <h2 className="text-2xl font-bold tracking-tight">{title}</h2>
      </div>
      <div className="space-y-4 text-sm leading-relaxed text-foreground/85">
        {children}
      </div>
    </section>
  );
}

function Accordion({ title, children, defaultOpen = false }: AccordionProps) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="border border-border/60 rounded-xl overflow-hidden bg-surface/40">
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center justify-between w-full px-4 py-3 text-sm font-semibold text-foreground hover:bg-surface-2/30 transition-colors"
      >
        {title}
        {open ? (
          <ChevronDown className="w-4 h-4 text-muted" />
        ) : (
          <ChevronRight className="w-4 h-4 text-muted" />
        )}
      </button>
      {open && <div className="px-4 pb-4 text-sm text-foreground/80 space-y-3 border-t border-border/40">{children}</div>}
    </div>
  );
}

function Code({ children }: { children: React.ReactNode }) {
  return (
    <code className="px-1.5 py-0.5 rounded-md bg-surface-2 border border-border/60 text-accent-3 text-xs font-mono">
      {children}
    </code>
  );
}

function CodeBlock({ children }: { children: string }) {
  return (
    <pre className="rounded-xl bg-surface border border-border/60 p-4 overflow-x-auto text-xs font-mono text-foreground/80 leading-relaxed">
      {children}
    </pre>
  );
}

function InfoCard({
  icon,
  title,
  description,
  color = "accent",
}: {
  icon: React.ReactNode;
  title: string;
  description: string;
  color?: string;
}) {
  const colorMap: Record<string, string> = {
    accent: "bg-accent/10 border-accent/20 text-accent",
    emerald: "bg-emerald-500/10 border-emerald-500/20 text-emerald-400",
    amber: "bg-amber-500/10 border-amber-500/20 text-amber-400",
    rose: "bg-rose-500/10 border-rose-500/20 text-rose-400",
    indigo: "bg-indigo-500/10 border-indigo-500/20 text-indigo-400",
    cyan: "bg-cyan-500/10 border-cyan-500/20 text-cyan-400",
    violet: "bg-violet-500/10 border-violet-500/20 text-violet-400",
  };
  return (
    <div className="rounded-xl border border-border/60 bg-surface/40 p-4 space-y-2">
      <div className="flex items-center gap-2">
        <div className={clsx("w-7 h-7 rounded-lg border flex items-center justify-center", colorMap[color])}>
          {icon}
        </div>
        <h4 className="text-sm font-semibold text-foreground">{title}</h4>
      </div>
      <p className="text-xs text-foreground/70 leading-relaxed">{description}</p>
    </div>
  );
}

function ParamTable({
  rows,
}: {
  rows: { name: string; default: string; range: string; desc: string }[];
}) {
  return (
    <div className="overflow-x-auto rounded-xl border border-border/60">
      <table className="w-full text-xs">
        <thead>
          <tr className="bg-surface-2/50 text-left">
            <th className="px-3 py-2 font-semibold text-muted">Parameter</th>
            <th className="px-3 py-2 font-semibold text-muted">Default</th>
            <th className="px-3 py-2 font-semibold text-muted">Range</th>
            <th className="px-3 py-2 font-semibold text-muted">Description</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={r.name} className={i % 2 === 0 ? "bg-surface/30" : "bg-surface/10"}>
              <td className="px-3 py-2 font-mono text-accent-3">{r.name}</td>
              <td className="px-3 py-2 font-mono">{r.default}</td>
              <td className="px-3 py-2 text-muted">{r.range}</td>
              <td className="px-3 py-2 text-foreground/70">{r.desc}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/* ──────────────────────────────────────────────
   Main docs page
   ────────────────────────────────────────────── */

export default function DocsPage() {
  const [activeSection, setActiveSection] = useState("overview");

  const scrollTo = (id: string) => {
    setActiveSection(id);
    document.getElementById(id)?.scrollIntoView({ behavior: "smooth" });
  };

  return (
    <div className="flex min-h-screen">
      {/* Sidebar TOC */}
      <aside className="hidden lg:block w-56 shrink-0 border-r border-border/40 bg-surface/30 sticky top-12 h-[calc(100vh-3rem)] overflow-y-auto">
        <div className="p-4 space-y-1">
          <p className="text-[10px] uppercase tracking-wider text-muted-2 font-semibold mb-3 px-2">
            On this page
          </p>
          {TOC.map((item) => (
            <button
              key={item.id}
              onClick={() => scrollTo(item.id)}
              className={clsx(
                "flex items-center gap-2 w-full px-2 py-1.5 text-xs rounded-lg transition-all duration-150",
                activeSection === item.id
                  ? "text-accent bg-accent/10 font-medium"
                  : "text-muted hover:text-foreground hover:bg-surface-2/30"
              )}
            >
              {item.icon}
              {item.label}
            </button>
          ))}
        </div>
      </aside>

      {/* Main content */}
      <main className="flex-1 max-w-4xl mx-auto px-6 py-10">
        {/* Hero */}
        <div className="mb-16">
          <div className="flex items-center gap-2 text-xs text-muted mb-4">
            <Link href="/" className="hover:text-accent transition-colors">Dashboard</Link>
            <ArrowRight className="w-3 h-3" />
            <span className="text-foreground">Documentation</span>
          </div>
          <h1 className="text-4xl font-bold tracking-tight mb-4">Documentation</h1>
          <p className="text-base text-muted leading-relaxed max-w-2xl">
            Complete reference for the EconoSim multi-agent economic simulation platform.
            Covers architecture, agent behavior, accounting principles, policy mechanics,
            extensions, and the full parameter space.
          </p>
        </div>

        {/* ──── Overview ──── */}
        <DocSection id="overview" title="Overview" icon={<BookOpen className="w-5 h-5" />}>
          <p>
            <strong>EconoSim</strong> is a multi-agent economic simulation where households, firms, banks, and
            a government interact in a closed economy. Macroeconomic dynamics — GDP, unemployment, inflation,
            inequality — emerge from micro-level agent decisions, market interactions, and policy rules.
          </p>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mt-4">
            <InfoCard
              icon={<ShieldCheck className="w-4 h-4" />}
              title="Stock-Flow Consistent"
              description="Every monetary flow is recorded via double-entry accounting. Assets minus Liabilities always equals Equity. No money appears or vanishes unexpectedly."
              color="emerald"
            />
            <InfoCard
              icon={<CircleDot className="w-4 h-4" />}
              title="Endogenous Money"
              description="Money is created by bank lending (loans create deposits) and government spending (sovereign money creation). Money is destroyed when loans are repaid."
              color="accent"
            />
            <InfoCard
              icon={<Cpu className="w-4 h-4" />}
              title="RL-Ready"
              description="Every agent type has a Gymnasium environment. Train RL policies for firms, households, banks, or government. PettingZoo multi-agent env for coordinated training."
              color="violet"
            />
            <InfoCard
              icon={<FlaskConical className="w-4 h-4" />}
              title="Experimentable"
              description="Parameter sweeps, batch runs with confidence intervals, scenario comparison, and configurable shocks. Reproducible seeded simulations."
              color="amber"
            />
          </div>
          <div className="mt-6 p-4 rounded-xl bg-accent/5 border border-accent/20">
            <p className="text-xs font-semibold text-accent mb-2">Quick Start</p>
            <CodeBlock>{`# Install
pip install -e ".[dev,rl]"

# Run the API backend
cd api && uvicorn main:app --reload   # http://localhost:8000

# Run the Next.js dashboard
cd web && npm install && npm run dev  # http://localhost:3000

# Run a simulation from CLI
python -m econosim --scenario scenarios/baseline.yaml --periods 120

# Run tests
pytest tests/   # 494 tests`}</CodeBlock>
          </div>
        </DocSection>

        {/* ──── Architecture ──── */}
        <DocSection id="architecture" title="Architecture" icon={<Layers className="w-5 h-5" />}>
          <p>
            The system is layered: accounting primitives at the bottom, agents above, markets connecting agents,
            and the simulation engine orchestrating the loop. The RL layer wraps the engine for training.
          </p>
          <CodeBlock>{`Core Accounting → Agents → Markets → Simulation Engine → Metrics
     ↓                                        ↓
  Contracts                              Experiments
  (Loans, Goods)                    (Batch, Sweep, Compare)
                                          ↓
                                    FastAPI Backend
                                          ↓
                                   Next.js Dashboard`}</CodeBlock>
          <Accordion title="Directory Structure" defaultOpen>
            <CodeBlock>{`econosim/
├── core/           # Accounting, contracts, goods tracking
├── agents/         # Household, Firm, Bank, Government
├── markets/        # Labor, Goods, Credit clearing
├── engine/         # Simulation loop orchestrator
├── metrics/        # GDP, inflation, Gini, etc.
├── config/         # Pydantic schemas, YAML loading
├── experiments/    # run_experiment, run_batch, sweep
├── rl/             # Gymnasium & PettingZoo environments
└── extensions/     # Expectations, networks, bonds, sectors, skills`}</CodeBlock>
          </Accordion>
          <Accordion title="Key Design Decisions">
            <ul className="list-disc list-inside space-y-2 mt-2">
              <li><strong>No Mesa dependency</strong> — Pure Python domain logic. The simulation loop is a simple function with explicit ordering, not a framework-managed schedule.</li>
              <li><strong>Central Ledger</strong> — All monetary flows go through a single <Code>Ledger</Code> via double-entry transactions. No agent can modify money balances directly.</li>
              <li><strong>Pydantic config</strong> — Typed, validated configuration with sensible defaults. YAML scenario files for reproducible experiments.</li>
              <li><strong>Rule-based agents</strong> — Simple, interpretable decision rules that can be debugged, compared, and eventually replaced by RL policies.</li>
              <li><strong>Extensions as feature flags</strong> — Expectations, networks, and bonds are toggled via config flags. Disabled by default, zero overhead when off.</li>
            </ul>
          </Accordion>
        </DocSection>

        {/* ──── Agents ──── */}
        <DocSection id="agents" title="Agents" icon={<Users className="w-5 h-5" />}>
          <p>
            Four agent types interact each period. Each maintains a balance sheet tracked by the central Ledger.
          </p>

          <Accordion title="Households" defaultOpen>
            <div className="flex items-center gap-2 mb-2 mt-2">
              <Users className="w-4 h-4 text-emerald-400" />
              <span className="font-semibold">100 households (default)</span>
            </div>
            <p>Households supply labor, earn wages, pay taxes, receive transfers, and consume goods.</p>
            <p className="mt-2"><strong>Consumption rule (buffer-stock):</strong></p>
            <CodeBlock>{`C = α₁ × disposable_income + α₂ × wealth
C = min(C, deposits)   # can't spend more than you have`}</CodeBlock>
            <p className="mt-2">
              The <Code>wealth_propensity</Code> (α₂) term is the critical circuit-breaker against deflationary collapse.
              When income drops, households still spend from savings, maintaining aggregate demand.
            </p>
            <p className="mt-2"><strong>Labor supply:</strong> Households participate based on <Code>labor_participation_rate</Code> and
              accept any wage ≥ <Code>reservation_wage</Code>.</p>
          </Accordion>

          <Accordion title="Firms">
            <div className="flex items-center gap-2 mb-2 mt-2">
              <Factory className="w-4 h-4 text-amber-400" />
              <span className="font-semibold">5 firms (default)</span>
            </div>
            <p>Firms hire workers, produce goods, set prices, and manage inventory. They may borrow from the bank.</p>
            <p className="mt-2"><strong>Hiring:</strong> Firms estimate demand from previous sales and revenue, then hire workers they can afford.</p>
            <CodeBlock>{`demand_estimate = max(prev_units_sold, prev_revenue / price)
target_inv = demand_estimate × target_inventory_ratio
production_needed = demand_estimate + target_inv - inventory
workers_needed = production_needed / labor_productivity
vacancies = min(workers_needed, affordable_workers)`}</CodeBlock>
            <p className="mt-2"><strong>Pricing:</strong> Inventory-target rule — if inventory is above target (+20%), lower price by <Code>price_adjustment_speed</Code>. If below target (-20%) and sales were positive, raise price.</p>
            <p className="mt-2"><strong>Wage setting:</strong> If fill rate is below 50%, raise wage. If above 90%, lower wage slightly.</p>
          </Accordion>

          <Accordion title="Bank">
            <div className="flex items-center gap-2 mb-2 mt-2">
              <Building2 className="w-4 h-4 text-cyan-400" />
              <span className="font-semibold">1 bank</span>
            </div>
            <p>The bank evaluates loan applications, collects repayments, and handles defaults. Lending creates money (endogenous money).</p>
            <p className="mt-2"><strong>Lending constraint:</strong></p>
            <CodeBlock>{`can_lend = (equity / (total_loans + new_loan)) >= capital_adequacy_ratio`}</CodeBlock>
            <p className="mt-2"><strong>Credit check:</strong> Rejects borrowers with deposits &lt; 10% of existing debt.</p>
            <p className="mt-2"><strong>Default handling:</strong> After <Code>default_threshold_periods</Code> missed payments, the loan is written off — bank equity absorbs the loss, borrower debt is forgiven.</p>
          </Accordion>

          <Accordion title="Government">
            <div className="flex items-center gap-2 mb-2 mt-2">
              <Landmark className="w-4 h-4 text-indigo-400" />
              <span className="font-semibold">1 government (sovereign money issuer)</span>
            </div>
            <p>The government collects taxes, pays transfers to unemployed, and purchases goods/services from firms.</p>
            <p className="mt-2"><strong>Sovereign money creation (MMT/SFC):</strong> When spending exceeds deposits, the government creates money via <Code>ensure_solvency()</Code>. This is the monopoly currency issuer — it can always fund its spending.</p>
            <CodeBlock>{`if deposits < fiscal_need:
    shortfall = fiscal_need - deposits
    create_money(shortfall)  # debit deposits, credit equity`}</CodeBlock>
            <p className="mt-2"><strong>Fiscal stabilizer:</strong> Government spending is injected as service contracts even when firms have zero inventory. This prevents a doom loop of zero demand.</p>
          </Accordion>
        </DocSection>

        {/* ──── Markets ──── */}
        <DocSection id="markets" title="Markets" icon={<ArrowLeftRight className="w-5 h-5" />}>
          <p>Three markets clear each period in a fixed order: credit → labor → goods.</p>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-3 mt-4">
            <InfoCard
              icon={<CreditCard className="w-4 h-4" />}
              title="Credit Market"
              description="Firms apply for loans. Bank evaluates capital adequacy and borrower risk. Approved loans create deposits (endogenous money creation)."
              color="cyan"
            />
            <InfoCard
              icon={<Users className="w-4 h-4" />}
              title="Labor Market"
              description="Random matching: unemployed households are shuffled and matched to firm vacancies. Wages paid immediately on hiring."
              color="emerald"
            />
            <InfoCard
              icon={<DollarSign className="w-4 h-4" />}
              title="Goods Market"
              description="Households spend based on consumption rule. Firms are visited in random order. Proportional rationing if inventory is scarce."
              color="amber"
            />
          </div>

          <Accordion title="Market Clearing Order (Critical)">
            <p className="mt-2">The order is strict and economically meaningful. Changing the sequence would violate stock-flow consistency:</p>
            <ol className="list-decimal list-inside space-y-1 mt-2">
              <li>Credit market — firms borrow before hiring</li>
              <li>Labor market — wages paid from deposits (including borrowed funds)</li>
              <li>Production — workers produce output</li>
              <li>Price adjustment — inventory-target pricing</li>
              <li>Goods market — households spend income + savings</li>
              <li>Taxes and transfers — government collects and redistributes</li>
              <li>Debt service — loan payments, delinquency, defaults</li>
              <li>Wage adjustment — signals for next period</li>
            </ol>
          </Accordion>
        </DocSection>

        {/* ──── Accounting ──── */}
        <DocSection id="accounting" title="Accounting System" icon={<DollarSign className="w-5 h-5" />}>
          <p>
            The double-entry accounting system is the foundation. Every monetary flow is a <Code>Transaction</Code>
            recorded in the central <Code>Ledger</Code>. The balance sheet identity <strong>A - L = E</strong> is
            verified after every simulation step.
          </p>

          <Accordion title="Account Types" defaultOpen>
            <div className="overflow-x-auto rounded-xl border border-border/60 mt-2">
              <table className="w-full text-xs">
                <thead>
                  <tr className="bg-surface-2/50 text-left">
                    <th className="px-3 py-2 font-semibold text-muted">Type</th>
                    <th className="px-3 py-2 font-semibold text-muted">Debit</th>
                    <th className="px-3 py-2 font-semibold text-muted">Credit</th>
                    <th className="px-3 py-2 font-semibold text-muted">Examples</th>
                  </tr>
                </thead>
                <tbody>
                  <tr className="bg-surface/30"><td className="px-3 py-2 font-mono text-emerald-400">ASSET</td><td className="px-3 py-2">Increase</td><td className="px-3 py-2">Decrease</td><td className="px-3 py-2 text-muted">Deposits, loans receivable, inventory</td></tr>
                  <tr className="bg-surface/10"><td className="px-3 py-2 font-mono text-rose-400">LIABILITY</td><td className="px-3 py-2">Decrease</td><td className="px-3 py-2">Increase</td><td className="px-3 py-2 text-muted">Loans payable, bank deposits (to depositors)</td></tr>
                  <tr className="bg-surface/30"><td className="px-3 py-2 font-mono text-accent">EQUITY</td><td className="px-3 py-2">Decrease</td><td className="px-3 py-2">Increase</td><td className="px-3 py-2 text-muted">Net worth, retained earnings</td></tr>
                </tbody>
              </table>
            </div>
          </Accordion>

          <Accordion title="Money Creation & Destruction">
            <p className="mt-2"><strong>Bank lending creates money:</strong></p>
            <CodeBlock>{`Loan issuance (creates money):
  Bank:     loans (asset) ↑    |  deposits (liability) ↑
  Borrower: deposits (asset) ↑ |  loans_payable (liability) ↑

Loan repayment (destroys money):
  Borrower: deposits ↓  |  loans_payable ↓
  Bank:     loans ↓     |  deposits (liability) ↓

Government spending (creates money):
  Government: deposits ↑  |  equity ↑

Default write-off:
  Bank:     equity ↓  |  loans ↓    (bank absorbs loss)
  Borrower: loans_payable ↓  |  equity ↑  (debt forgiven)`}</CodeBlock>
          </Accordion>

          <Accordion title="Transfer Mechanism">
            <p className="mt-2">
              Every transfer between agents posts two within-entity double-entry transactions.
              This ensures A-L=E holds on both sides after every transfer:
            </p>
            <CodeBlock>{`transfer_deposits(sender, receiver, amount):
  Sender:   equity ↓, deposits ↓   (expense)
  Receiver: deposits ↑, equity ↑   (income)`}</CodeBlock>
          </Accordion>
        </DocSection>

        {/* ──── Simulation Loop ──── */}
        <DocSection id="simulation" title="Simulation Loop" icon={<Repeat className="w-5 h-5" />}>
          <p>
            Each period, the <Code>step()</Code> function executes sub-steps in a strict, explicit order.
            The sequence is economically meaningful — credit before labor, production before pricing, etc.
          </p>
          <div className="mt-4 space-y-2">
            {[
              { step: "0", label: "Reset period state", desc: "Clear per-period accumulators for all agents" },
              { step: "1", label: "Apply shocks", desc: "Modify parameters if any shocks are scheduled" },
              { step: "1b", label: "Extension state reset", desc: "Network edge decay, debt manager reset" },
              { step: "2", label: "Credit market", desc: "Firms apply for loans, bank approves/rejects" },
              { step: "3", label: "Labor market", desc: "Random matching, wage payment on hire" },
              { step: "4", label: "Production", desc: "Workers produce output (units = workers × productivity)" },
              { step: "5", label: "Goods pricing + market", desc: "Inventory-target pricing, then consumption" },
              { step: "6", label: "Taxes & transfers", desc: "Income tax, unemployment transfers, govt spending" },
              { step: "7", label: "Debt service", desc: "Loan payments, delinquency, defaults, bond service" },
              { step: "8", label: "Wage adjustment", desc: "Firms adjust posted wage based on fill rate" },
              { step: "8b", label: "Update expectations", desc: "Adaptive expectations for firm forecasts" },
              { step: "9", label: "Compute metrics", desc: "GDP, unemployment, prices, Gini, credit metrics" },
              { step: "10", label: "Validate invariants", desc: "Check all balance sheets satisfy A-L=E" },
            ].map((s) => (
              <div key={s.step} className="flex items-start gap-3 p-3 rounded-lg bg-surface/40 border border-border/40">
                <span className="shrink-0 w-7 h-7 rounded-lg bg-accent/10 text-accent text-xs font-bold flex items-center justify-center">
                  {s.step}
                </span>
                <div>
                  <p className="text-sm font-semibold">{s.label}</p>
                  <p className="text-xs text-muted">{s.desc}</p>
                </div>
              </div>
            ))}
          </div>
        </DocSection>

        {/* ──── Policy & Shocks ──── */}
        <DocSection id="policy" title="Policy & Shocks" icon={<Zap className="w-5 h-5" />}>
          <p>
            The simulation supports configurable shocks that modify agent parameters at specific periods.
            Four shock types are available:
          </p>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mt-4">
            <InfoCard icon={<Factory className="w-4 h-4" />} title="Supply Shock" description="Modifies firm labor_productivity. Multiplicative (e.g., 0.5 = halve productivity) or additive. Affects production capacity." color="amber" />
            <InfoCard icon={<Users className="w-4 h-4" />} title="Demand Shock" description="Modifies household consumption_propensity. Reduces or increases aggregate demand. Clamped to [0.01, 1.0]." color="emerald" />
            <InfoCard icon={<Building2 className="w-4 h-4" />} title="Credit Shock" description="Modifies bank capital_adequacy_ratio. Tighter requirements reduce lending capacity (credit crunch)." color="cyan" />
            <InfoCard icon={<Landmark className="w-4 h-4" />} title="Fiscal Shock" description="Modifies income_tax_rate or spending_per_period. Tax changes affect disposable income; spending changes affect aggregate demand." color="indigo" />
          </div>
          <Accordion title="Shock Configuration">
            <CodeBlock>{`shocks:
  - period: 20
    shock_type: supply
    parameter: labor_productivity
    magnitude: 0.5     # multiply productivity by 0.5
    additive: false

  - period: 30
    shock_type: fiscal
    parameter: spending_per_period
    magnitude: 3.0     # triple government spending
    additive: false`}</CodeBlock>
          </Accordion>
        </DocSection>

        {/* ──── Extensions ──── */}
        <DocSection id="extensions" title="Extensions" icon={<Activity className="w-5 h-5" />}>
          <p>
            Three extensions can be toggled in the sidebar. They add richer dynamics without changing the core simulation when disabled.
          </p>

          <Accordion title="Adaptive Expectations" defaultOpen>
            <p className="mt-2">Firms form forecasts for prices, wages, demand, and inflation using exponential smoothing:</p>
            <CodeBlock>{`forecast = α × actual + (1 - α) × previous_forecast`}</CodeBlock>
            <p className="mt-2">Three model types available: <Code>AdaptiveExpectations</Code> (exponential smoothing),
              <Code>RollingExpectations</Code> (moving average with trend), and <Code>WeightedExpectations</Code> (multi-signal combination).
              Forecast errors are tracked and displayed in the Extensions tab.</p>
          </Accordion>

          <Accordion title="Network Tracking">
            <p className="mt-2">Two directed weighted graphs track economic relationships:</p>
            <ul className="list-disc list-inside space-y-1 mt-2">
              <li><strong>Trade Network</strong> — Records goods market flows (households → firms). Computes seller concentration (HHI), buyer diversity, density.</li>
              <li><strong>Credit Network</strong> — Records lending flows (bank → firms). Computes exposure, contagion risk (first-order default impact), systemic risk score.</li>
            </ul>
            <p className="mt-2">Edges decay each period at <Code>edge_decay_rate</Code> (default 10%), so inactive relationships fade out.</p>
          </Accordion>

          <Accordion title="Bond Market">
            <p className="mt-2">Government debt management via fixed-income securities:</p>
            <ul className="list-disc list-inside space-y-1 mt-2">
              <li>When government faces a fiscal shortfall, it issues bonds before sovereign money creation</li>
              <li>Bonds have configurable maturity (default 12 periods) and coupon rate (default 2%)</li>
              <li>Debt-to-GDP ratio is tracked and limits issuance (default max 1.5)</li>
              <li>Bond debt service: coupon payments each period, principal at maturity</li>
            </ul>
          </Accordion>

          <Accordion title="Multi-Sector Production (Standalone)">
            <p className="mt-2">Built but not yet integrated into the core loop. Provides:</p>
            <ul className="list-disc list-inside space-y-1 mt-2">
              <li>Typed commodities: consumption, intermediate, capital goods</li>
              <li>Leontief Input-Output matrix for inter-sector dependencies</li>
              <li>Hawkins-Simon productivity condition validation</li>
              <li>3-sector default economy: agriculture, manufacturing, services</li>
            </ul>
          </Accordion>

          <Accordion title="Skilled Labor (Standalone)">
            <p className="mt-2">Built but not yet integrated. Provides:</p>
            <ul className="list-disc list-inside space-y-1 mt-2">
              <li>4-tier skill system: Unskilled (1.0×), Semi-Skilled (1.5×), Skilled (2.5×), Highly Skilled (4.0×)</li>
              <li>Experience accumulation, skill decay when unemployed, training upgrades</li>
              <li>Skill-based matching in labor market with wage premiums</li>
            </ul>
          </Accordion>
        </DocSection>

        {/* ──── Metrics ──── */}
        <DocSection id="metrics" title="Metrics" icon={<BarChart3 className="w-5 h-5" />}>
          <p>Computed each period from agent state and market outcomes:</p>
          <div className="overflow-x-auto rounded-xl border border-border/60 mt-4">
            <table className="w-full text-xs">
              <thead>
                <tr className="bg-surface-2/50 text-left">
                  <th className="px-3 py-2 font-semibold text-muted">Metric</th>
                  <th className="px-3 py-2 font-semibold text-muted">Formula</th>
                  <th className="px-3 py-2 font-semibold text-muted">Dashboard Tab</th>
                </tr>
              </thead>
              <tbody>
                {[
                  ["GDP", "goods_market.total_transacted + govt.goods_spending", "Macro"],
                  ["Unemployment Rate", "(labor_force − employed) / labor_force", "Macro"],
                  ["Average Price", "mean(firm.price for all firms)", "Macro"],
                  ["Inflation", "pct_change(avg_price)", "Macro"],
                  ["Gini Coefficient", "Lorenz curve on household deposits", "Macro"],
                  ["GDP Growth", "pct_change(gdp)", "Macro"],
                  ["Velocity", "gdp / total_deposits", "Money & Credit"],
                  ["Bank Capital Ratio", "equity / total_loans", "Money & Credit"],
                  ["Budget Balance", "tax_revenue − transfers − spending", "Government"],
                  ["Money Created", "ensure_solvency shortfall amount", "Government"],
                ].map(([metric, formula, tab], i) => (
                  <tr key={metric} className={i % 2 === 0 ? "bg-surface/30" : "bg-surface/10"}>
                    <td className="px-3 py-2 font-semibold">{metric}</td>
                    <td className="px-3 py-2 font-mono text-accent-3 text-[11px]">{formula}</td>
                    <td className="px-3 py-2 text-muted">{tab}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </DocSection>

        {/* ──── RL Environments ──── */}
        <DocSection id="rl" title="RL Environments" icon={<Cpu className="w-5 h-5" />}>
          <p>
            Four Gymnasium-compatible single-agent environments and one PettingZoo multi-agent environment.
            Each wraps the core simulation engine, letting an RL agent replace one agent type's decision rules.
          </p>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mt-4">
            {[
              { name: "FirmEnv", obs: "14-dim", action: "3-dim (price × [0.8,1.2], wage × [0.9,1.1], vacancy frac)", rewards: "profit, gdp, balanced", color: "amber" },
              { name: "HouseholdEnv", obs: "12-dim", action: "2-dim (consumption frac, reservation wage mult)", rewards: "utility, consumption, balanced", color: "emerald" },
              { name: "GovernmentEnv", obs: "12-dim", action: "3-dim (tax rate, transfer mult, spending mult)", rewards: "welfare, gdp, employment, balanced", color: "indigo" },
              { name: "BankEnv", obs: "12-dim", action: "2-dim (interest rate, capital adequacy ratio)", rewards: "profit, stability, growth", color: "cyan" },
            ].map((env) => (
              <InfoCard
                key={env.name}
                icon={<Cpu className="w-4 h-4" />}
                title={env.name}
                description={`Obs: ${env.obs} | Action: ${env.action} | Rewards: ${env.rewards}`}
                color={env.color}
              />
            ))}
          </div>
          <Accordion title="Training Pipeline">
            <CodeBlock>{`# Train a single agent (PPO, 50k steps)
python scripts/train_agent.py --agent firm --timesteps 50000 --reward profit

# Train with normalization
python scripts/train_agent.py --agent government --timesteps 50000 --reward welfare --normalize

# Multi-agent training (sequential)
python scripts/train_multiagent.py --timesteps 50000 --mode sequential

# Compare RL vs baseline across scenarios
python scripts/compare_policies.py --agent firm --model outputs/rl/firm/final_model

# Hyperparameter search
python scripts/tune_hyperparams.py --agent firm --timesteps 20000`}</CodeBlock>
          </Accordion>
        </DocSection>

        {/* ──── Experiments ──── */}
        <DocSection id="experiments" title="Experiments" icon={<FlaskConical className="w-5 h-5" />}>
          <p>
            The experiment runner supports single runs, batch runs (multiple seeds with aggregation),
            and parameter sweeps (cartesian product of values).
          </p>
          <Accordion title="Batch Runs" defaultOpen>
            <p className="mt-2">
              Set <Code>n_seeds &gt; 1</Code> in the dashboard sidebar to enable batch mode. Results
              are aggregated with 95% confidence interval bands. The Compare tab lets you save and
              overlay multiple scenarios.
            </p>
          </Accordion>
          <Accordion title="Parameter Sweeps">
            <CodeBlock>{`from econosim.experiments.runner import run_parameter_sweep

results = run_parameter_sweep(
    base_config=config,
    sweep_params={
        'firm.labor_productivity': [2, 8, 20],
        'government.spending_per_period': [500, 2000, 5000],
    },
    num_periods=60,
    seeds=[42, 43, 44],
)`}</CodeBlock>
          </Accordion>
          <Accordion title="Scenario Presets (Dashboard)">
            <div className="grid grid-cols-2 gap-2 mt-2">
              {[
                { name: "Baseline", desc: "Default balanced economy" },
                { name: "High Growth", desc: "Low tax (10%), high spending (5000)" },
                { name: "Recession", desc: "High tax (35%), low spending (500), low consumption (0.5)" },
                { name: "Tight Money", desc: "High interest (3%), strict capital ratio (15%)" },
              ].map((s) => (
                <div key={s.name} className="p-3 rounded-lg bg-surface/40 border border-border/40">
                  <p className="text-xs font-semibold">{s.name}</p>
                  <p className="text-[11px] text-muted">{s.desc}</p>
                </div>
              ))}
            </div>
          </Accordion>
        </DocSection>

        {/* ──── API Reference ──── */}
        <DocSection id="api" title="API Reference" icon={<Settings className="w-5 h-5" />}>
          <p>The FastAPI backend exposes three endpoints:</p>
          <div className="space-y-3 mt-4">
            {[
              { method: "POST", path: "/api/simulate", desc: "Run a simulation with custom config. Returns period-by-period metrics, summary statistics, and optionally aggregated batch data." },
              { method: "GET", path: "/api/defaults", desc: "Returns the default simulation configuration (all parameter defaults)." },
              { method: "GET", path: "/api/health", desc: "Health check. Returns {\"status\": \"ok\"}." },
            ].map((ep) => (
              <div key={ep.path} className="p-3 rounded-xl bg-surface/40 border border-border/40">
                <div className="flex items-center gap-2 mb-1">
                  <span className={clsx("text-[10px] font-bold px-1.5 py-0.5 rounded", ep.method === "POST" ? "bg-emerald-500/20 text-emerald-400" : "bg-accent/20 text-accent")}>
                    {ep.method}
                  </span>
                  <code className="text-xs font-mono text-accent-3">{ep.path}</code>
                </div>
                <p className="text-xs text-muted">{ep.desc}</p>
              </div>
            ))}
          </div>
          <Accordion title="Simulate Request Body">
            <CodeBlock>{`{
  "num_periods": 60,
  "seed": 42,
  "n_seeds": 1,
  "household": { "count": 100, "initial_deposits": 1000, ... },
  "firm": { "count": 5, "initial_deposits": 15000, ... },
  "government": { "income_tax_rate": 0.2, "spending_per_period": 2000, ... },
  "bank": { "base_interest_rate": 0.005, "capital_adequacy_ratio": 0.08 },
  "extensions": {
    "enable_expectations": false,
    "enable_networks": false,
    "enable_bonds": false
  }
}`}</CodeBlock>
          </Accordion>
        </DocSection>

        {/* ──── Parameter Guide ──── */}
        <DocSection id="parameters" title="Parameter Guide" icon={<Scale className="w-5 h-5" />}>
          <p>Complete reference for all configurable parameters and their effects on the simulation.</p>

          <Accordion title="Household Parameters" defaultOpen>
            <ParamTable rows={[
              { name: "count", default: "100", range: "10–500", desc: "Number of household agents" },
              { name: "initial_deposits", default: "1,000", range: "100–50,000", desc: "Starting money per household" },
              { name: "consumption_propensity", default: "0.8", range: "0.1–1.0", desc: "Fraction of disposable income spent. Higher = more demand." },
              { name: "wealth_propensity", default: "0.4", range: "0.0–1.0", desc: "Fraction of wealth (deposits) spent. CRITICAL: prevents deflation death spiral. Keep > 0." },
              { name: "reservation_wage", default: "50", range: "0–200", desc: "Minimum acceptable wage. Higher = more unemployment." },
            ]} />
          </Accordion>

          <Accordion title="Firm Parameters">
            <ParamTable rows={[
              { name: "count", default: "5", range: "1–50", desc: "Number of firms. More firms = more competition." },
              { name: "initial_deposits", default: "15,000", range: "1,000–100,000", desc: "Starting capital per firm" },
              { name: "initial_price", default: "10", range: "1–100", desc: "Starting goods price" },
              { name: "initial_wage", default: "60", range: "10–500", desc: "Starting posted wage" },
              { name: "labor_productivity", default: "8", range: "1–50", desc: "Units of output per worker per period. Higher = more output per worker." },
              { name: "price_adjustment_speed", default: "0.03", range: "0.01–0.20", desc: "Max price change per period. Higher = faster but more volatile." },
              { name: "wage_adjustment_speed", default: "0.02", range: "0.01–0.20", desc: "Max wage change per period." },
            ]} />
          </Accordion>

          <Accordion title="Government Parameters">
            <ParamTable rows={[
              { name: "income_tax_rate", default: "0.20", range: "0–0.50", desc: "Flat income tax rate. Higher = more redistribution but less disposable income." },
              { name: "transfer_per_unemployed", default: "50", range: "0–500", desc: "Cash transfer to each unemployed worker per period." },
              { name: "spending_per_period", default: "2,000", range: "0–20,000", desc: "Government purchases from firms. KEY fiscal stabilizer." },
              { name: "initial_deposits", default: "100,000", range: "10,000–1,000,000", desc: "Starting government treasury" },
            ]} />
          </Accordion>

          <Accordion title="Bank Parameters">
            <ParamTable rows={[
              { name: "base_interest_rate", default: "0.005", range: "0–0.05", desc: "Per-period interest rate on loans (0.5% ≈ 6% annual)." },
              { name: "capital_adequacy_ratio", default: "0.08", range: "0.01–0.20", desc: "Min equity/loans ratio. Higher = less lending (tighter credit)." },
            ]} />
          </Accordion>

          <div className="mt-6 p-4 rounded-xl bg-amber-500/5 border border-amber-500/20">
            <p className="text-xs font-semibold text-amber-400 mb-2">Parameter Sensitivity Tips</p>
            <ul className="list-disc list-inside text-xs text-foreground/70 space-y-1">
              <li><Code>wealth_propensity = 0</Code> can cause deflationary collapse. Keep it above 0.05.</li>
              <li><Code>spending_per_period = 0</Code> with <Code>transfer_per_unemployed = 0</Code> removes all automatic stabilizers.</li>
              <li>High <Code>price_adjustment_speed</Code> (&gt; 0.1) can cause price oscillations.</li>
              <li><Code>capital_adequacy_ratio &gt; 0.3</Code> effectively prevents all lending.</li>
              <li>Very high <Code>reservation_wage</Code> causes mass unemployment even with available jobs.</li>
            </ul>
          </div>
        </DocSection>

        {/* Bottom spacer */}
        <div className="h-24" />
      </main>
    </div>
  );
}
