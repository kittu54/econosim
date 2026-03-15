# MarketWorld Implementation Plan

## Design objectives and research posture

A good implementation starts by being explicit about *what kind* of “economic simulation” you are building: this is not a DSGE-style equilibrium solver; it’s an **agent-based computational economics** (ACE) system where the economy evolves over time from interacting autonomous agents. entity["people","Leigh Tesfatsion","economist ACE"] characterizes ACE as the computational study of economies modeled as evolving systems of autonomous interacting agents, emphasizing bottom‑up dynamics and learning rather than imposed equilibrium conditions. citeturn0search8turn0search20turn0search28

That posture is aligned with how many researchers argue complex economic systems should be explored: using simulation to understand emergent macro behavior from micro rules and interactions. entity["people","J. Doyne Farmer","complexity economist"] and entity["people","Duncan K. Foley","economist"] famously argued that agent-based modeling is an important direction for economics precisely because it can represent heterogeneity, networks, and out-of-equilibrium dynamics more directly than many traditional models. citeturn1search3turn1search11 The economics literature also treats ABMs as increasingly relevant across markets, macro, and finance, while acknowledging the key engineering and validation hurdles. citeturn2search25turn2search29

For positioning and communicating the project, the closest intuitive analogies are a “systems sandbox” like entity["video_game","SimCity","simulation game series"] or entity["video_game","Civilization","strategy game series"]—but with economics-grade accounting, policies, and shocks rather than pure game mechanics. The goal of the plan below is to make your simulator rigorous enough to support experiments (paper/demo) while remaining engineerable as a student project.

Key design requirement (non-negotiable): **every step maintains consistent accounting and well-defined state transitions**. In economic ABMs, a large fraction of failure modes come from silent accounting inconsistencies (money created/destroyed accidentally, firms paying wages without funds, bank assets not matching liabilities, etc.). Address this as a first-class engineering concern from day one. citeturn2search25turn0search8

## Economic model specification

### Accounting primitives and balance sheets

Build the economy around a small set of balance-sheet objects and accounting identities. Don’t treat “money” as an arbitrary score; treat it as *deposits*, *loans*, and *equity* that must reconcile.

A minimal closed-economy monetary system can be implemented with these positions:

**Households (consumers)**  
Assets: bank deposits  
Liabilities: bank loans (optional in MVP)  
Flows: wages received, taxes paid, consumption spending, loan interest, transfers

**Firms**  
Assets: bank deposits, inventories (real good), capital stock (optional), receivables (optional)  
Liabilities: bank loans, wage obligations (short-lived), equity  
Flows: revenue from sales, wage payments, taxes, investment/production costs, interest, dividends (optional)

**Bank**  
Assets: loans to households and firms, reserves (or “liquidity buffer”), government bond holdings (optional)  
Liabilities: deposits (households + firms + government), bank equity, possibly central bank borrowing (optional)  

This is not just pedantry: when you model credit, you must reflect that modern bank lending creates deposits (i.e., expands the balance sheet) rather than just lending out prior deposits. entity["organization","Bank of England","uk central bank"] explicitly explains that commercial banks create most money by making loans and that they are not merely intermediaries lending out deposits. citeturn1search0turn1search4turn1search8

Even if your MVP uses simplified banking, you should already structure the code so that moving to deposit/loan creation is a change in parameters, not a rewrite of architecture.

**Government**  
Assets: deposits  
Liabilities: (optional) government debt  
Flows: taxes collected, transfers and purchases, interest (optional), deficit financing method

If you include bank solvency constraints, ground them in the concept (not full real-world complexity): international standards like **Basel III** are explicitly aimed at strengthening regulation and risk management after the 2007–09 crisis. entity["organization","Bank for International Settlements","international financial org"] summarizes Basel III as internationally agreed measures developed in response to the crisis to strengthen bank regulation and risk management. citeturn2search2turn2search6 This gives you a credible motivation for a “capital ratio” constraint in the bank’s lending rule.

Implementation note: define a formal invariant that must hold after every simulation sub-step:

- *For each agent*: Assets − Liabilities = Equity  
- *For the bank*: Loans + Reserves + … = Deposits + Equity + …  

Write unit tests that fail if any invariant is violated.

### Markets, matching, and timing of actions

Choose a time unit that matches your macro goals: **monthly** is a solid default because it’s short enough to model unemployment spells and price adjustments but long enough for macro indicators to change meaningfully. This is consistent with the way many macro series (inflation, unemployment) are commonly observed and discussed at monthly cadence. citeturn1search1turn2search7

Use a “within-month” sequence that is stable, interpretable, and easy to debug. A robust baseline is:

1) **Policy announcement** (government sets tax/transfer/spending levers, optionally a policy rate)  
2) **Credit pre-approval** (bank sets lending terms and approvals based on current states)  
3) **Labor market** (firms post vacancies/wages; households offer labor; matching happens)  
4) **Production** (firms produce goods based on labor hired and productivity)  
5) **Goods market** (firms post prices; households decide demand; rationing/clearing rule)  
6) **Settlement and accounting update** (wages, taxes, transfers, interest, defaults)  
7) **Metrics & logging** (compute CPI/GDP/unemployment/inequality/etc.)  
8) **Learning update** (if any agents are adaptive/learning)

This ordering preserves causality and reduces ambiguous “simultaneous” updates, which matters once you add reinforcement learning.

For market mechanisms, start with models that are simple but expressive:

- **Posted-price goods market**: firms set prices; households choose quantities; if demand exceeds supply, allocate via proportional rationing or random order.  
- **Wage posting + matching labor market**: firms set wages and vacancies; households choose whether to work and where to apply; matching uses a priority/lottery rule.  
- **Credit market**: bank maps borrower state to approval/interest rate/loan size, subject to risk and policy parameters.  

These mechanisms are commonly used across ABM designs because they’re computationally straightforward and still produce meaningful emergent macro patterns when combined with heterogeneity and shocks. citeturn2search25turn0search8

### Shocks and policy levers

Design shocks as explicit stochastic processes applied to well-defined variables. Treat them as scenario “toggles” so you can run controlled experiments.

Useful shocks (each can be implemented in MVP-friendly form):

- **Supply shock**: temporary drop in productivity \(A_t\), or increase in input cost.  
- **Demand shock**: lower household confidence → lower propensity to consume.  
- **Credit shock**: tighter lending rule or higher risk premium for the bank.  
- **Policy shock**: step-change in tax rate or transfer schedule; fiscal stimulus.  

This aligns with the kinds of policy and credit dynamics explored in major macro ABM families: for example, “Schumpeter meeting Keynes” (K+S) models explicitly combine demand generation mechanisms with innovation/growth and “Minskian” credit dynamics (credit amplification). citeturn2search4turn2search20 Large-scale ABM programs like EURACE also emphasize realistic labor market and interaction structure as central building blocks. citeturn2search9turn2search1

Policy levers to support in the MVP:

- income tax \( \tau_y \)  
- consumption tax \( \tau_c \) (optional)  
- unemployment benefit or universal transfer \(T\)  
- government spending on goods \(G\) (in closed economy, this directly adds demand)  

You can add an interest-rate-like lever later; for MVP it is acceptable to parameterize a base interest rate and treat it as exogenous.

### Macro indicators and measurement

Define indicators in a way that is faithful to real-world definitions but adapted to your simplified economy.

**Inflation / price index**: Use a CPI-like index: a representative basket price level. The CPI is defined as a measure of the average change over time in prices paid by consumers for a market basket of goods and services. entity["organization","Bureau of Labor Statistics","us labor agency"] citeturn1search1turn1search9turn1search25 In a one-good MVP, the “basket” can be that single good’s average transaction price; inflation is the percent change over time.

**GDP / output**: Implement an expenditure-style GDP consistent with the standard definition: GDP measures the value of final goods and services produced, and can be expressed as C + I + G + (X−M). entity["organization","Bureau of Economic Analysis","us commerce statistics"] citeturn1search10turn1search2turn1search6 In a closed, one-good MVP without trade, GDP reduces to consumption + government spending + (optional) investment.

**Unemployment**: Use the official-style definition (U‑3 concept): unemployed as a percent of the labor force. citeturn2search7turn2search23 This requires explicit definitions of “labor force,” “employed,” and “unemployed” in your household state.

**Inequality**: If you want inequality tracking, implement the Gini index (or coefficient) from wealth or income distributions. The World Bank describes the Gini index via the Lorenz curve area, with 0 indicating perfect equality and 100 perfect inequality. citeturn4search0turn4search27 This is straightforward to compute from your per-household deposits (wealth proxy) or income streams.

These metrics are not just “nice graphs”: they become your evaluation surface for shocks, learning, and policy.

## System architecture and engineering choices

### Core simulation engine and ABM scaffolding

For the rule-based MVP, a mature agent-based modeling framework reduces boilerplate and gives you standard scheduling and data collection. entity["organization","Mesa","python ABM framework"] is explicitly described as a modular framework for building, analyzing, and visualizing agent-based models, emphasizing emergence from interacting agents. citeturn0search1turn0search25turn0search5 It also provides a standard DataCollector that can collect model-level and agent-level outputs in a structured way, which is directly useful for macro metrics and dashboards. citeturn0search9turn0search17

However, design your own domain model on top: keep your *economic* types (balance sheets, markets, contracts) independent of the ABM framework so you can later swap orchestration layers (e.g., for MARL training performance). This is a key engineering trick: **separate the economic state machine from the scheduler**. citeturn2search25turn0search8

Recommended internal structure (conceptual):

- `core/` (economic primitives): money, loans, inventories, taxes, contracts  
- `agents/` (decision logic): household, firm, bank, government  
- `markets/` (matching/clearing): labor, goods, credit  
- `sim/` (time loop): step ordering, shock injection, scenario runner  
- `metrics/` (CPI/GDP/unemployment/inequality)  
- `ui/` (dashboard)  
- `experiments/` (policy experiments, batch runs)  

### Interop with reinforcement learning APIs

Plan from day one for RL compatibility: you’ll eventually want a clean “environment step(reset)” boundary.

entity["organization","Gymnasium","reinforcement learning env API"] formalizes the standard RL environment interface via `reset()` and `step()` functions, and explicitly points multi-agent users toward multi-agent standards such as PettingZoo. citeturn0search3turn0search11turn0search31 For multi-agent RL interoperability, entity["organization","PettingZoo","multi-agent RL api library"] provides both a sequential Agent Environment Cycle (AEC) API and a parallel API for simultaneous actions, making it a strong choice for economic simulations where many decisions are naturally “simultaneous within a period.” citeturn0search10turn0search6turn0search2

For training at scale and role-based policy mapping (e.g., one policy shared across all households), an RL library with multi-agent support is useful. entity["organization","Ray","distributed compute framework"] documents a `MultiAgentEnv` for environments hosting multiple independent agents, and provides mechanisms for mapping agents to policies. citeturn3search5turn3search1turn5search19

If you want a lighter-weight training loop (especially early), entity["organization","Stable Baselines3","rl algorithms library"] provides reliable implementations of common RL algorithms in PyTorch and documents which algorithms support which action/observation space types. citeturn3search15turn3search25turn3search8

Under the hood, plan to implement learning using entity["organization","PyTorch","open source ml framework"] (widely used for deep learning and frequently paired with SB3). citeturn5search6turn5search10

### Graphs and dashboards

Economic interactions naturally create networks: who trades with whom, who borrows from whom, how firm concentration evolves. entity["organization","NetworkX","python network analysis library"] is explicitly positioned as a Python package for creating, manipulating, and studying the structure and dynamics of complex networks, which you can use to compute concentration and connectivity metrics on your credit/trade graphs. citeturn3search6turn3search2

For an MVP-quality demo UI, entity["organization","Streamlit","python data app framework"] is explicitly designed to turn Python scripts into shareable interactive data apps quickly, which is ideal for scenario controls, plotting macro indicators, and replaying runs without building a full frontend. citeturn3search3turn3search7turn3search24

If you later want “startup platform” architecture, treat the simulator as a service. entity["organization","FastAPI","python api framework"] describes itself as a modern high-performance web framework for building APIs with Python type hints; it’s a common choice to wrap a simulation engine behind endpoints. citeturn5search0turn5search4 A production-grade UI layer can be built with entity["organization","React","web ui library"], described in its official docs as a library for building user interfaces from components. citeturn5search5turn5search9

## MVP implementation plan

### Milestone definition for the MVP

Your MVP should prove three things:

1) The economic state machine is coherent (accounting is consistent, markets clear by rules).  
2) Macro metrics react plausibly to shocks (even if not perfectly realistic).  
3) The system is experimentable (batch runs, seeding, plotting, scenario toggles).

This matches ABM best practice: build a simple, robust core before adding behavioral complexity, and keep the model interpretable so you can debug emergent outcomes. citeturn2search25turn0search8

Target MVP scope (as you specified): ~100 households, ~5 firms, 1 bank, 1 government, 1 consumption good, 1 labor type. That is large enough for distributions to matter but small enough to run fast on a laptop.

### Step-by-step build sequence

**Step zero: formal type system for the economy**

Implement the economy as explicit state objects:

- `BalanceSheet` with assets/liabilities dictionaries and methods `post()`, `validate()`.  
- `LoanContract` with principal, interest rate, maturity, amortization rule, default state.  
- `Inventory` for goods quantity and possibly vintage/quality later.  
- `Policy` object for tax rates, transfers, spending.

You are building a simulation where the *dominant debugging tool* is inspecting a few agent states over time. Make that easy: implement pretty-print / serialization of balance sheets and transactions.

**Step one: markets as pure functions**

Implement market modules as functions that map “offers” and “demands” into “allocations” and “prices” plus settlement transfers. Keep them stateless except for random seeds. That makes them testable.

- Labor market takes: firm wage offers + vacancies, household participation + application choices.  
- Goods market takes: firm posted prices + supply, household demand.  
- Credit market takes: borrower applications, bank rules, bank constraints.

This style is consistent with building interpretable ABM components and supporting later RL “action injection” cleanly. citeturn0search8turn2search25

**Step two: household decision rules (rule-based)**

Start simple but economically grounded:

- Labor supply rule: work if expected after-tax wage − disutility threshold > 0.  
- Consumption rule: consume a fraction of disposable income plus a wealth effect (e.g., MPC decreases with wealth), with a “confidence” scalar that can be shocked.  
- Savings rule: deposits increase by income − consumption − taxes + transfers.

This is intentionally not a full microfoundation; the point is to create a controllable baseline that reacts to wages, taxes, prices, and shocks.

**Step three: firm rules (rule-based)**

Model each firm with:

- Pricing rule: adapt price based on inventory depletion / unmet demand (e.g., if stockouts occur, raise; if inventories accumulate, lower).  
- Hiring rule: post vacancies if expected demand + target inventory > current capacity.  
- Wage rule: wage adjusts upward if vacancies unfilled, downward if high idle labor.  
- Production function: \( q = A \cdot L \) for MVP; later you can add diminishing returns.

This approach is consistent with ABM macro traditions that use simple behavioral rules yet generate emergent macro dynamics (business cycles, unemployment fluctuations) when combined with credit and demand feedback. citeturn2search20turn2search4turn2search25

**Step four: bank rule set (rule-based)**

In the MVP, you can keep the bank as a single institution with:

- A base interest rate + risk spread rule.  
- A simple credit scoring rule using debt service burden (for households) and leverage / cash flow (for firms).  
- A capital or liquidity constraint proxy (stop lending if leverage exceeds a limit). Motivation for capital constraints can be tied to the idea of post-crisis regulatory strengthening under Basel III, without implementing full Basel mechanics. citeturn2search2turn2search6

If you model bank lending as deposit creation (recommended), use the core insight that loans create deposits as described by the Bank of England. citeturn1search0turn1search4

Add default mechanics early because defaults are where your accounting will break if you’re sloppy:
- If borrower cannot pay interest/required payment, flag delinquency; after N delinquencies, default triggers loss to bank equity and borrower balance sheet reset (or bankruptcy event).  

**Step five: government budget and policy**

Implement:

- Income tax collection on wage income.  
- Transfers: universal (UBI-like) or unemployment benefit.  
- Government spending: purchase goods from firms at posted prices (adds demand).  

Keep the government’s deposit balance explicit so deficits and surpluses are visible. Even if you don’t model bonds initially, you can represent deficit finance as an overdraft/loan with the bank (again using balance-sheet mechanics).

**Step six: shocks and scenario runner**

Create a scenario object that can inject shocks:

- productivity \(A_t\) shocks (supply)  
- confidence shocks (demand)  
- bank risk appetite shocks (credit tightening)  
- tax/transfer shocks (policy)

This is how you get “recession,” “supply shock inflation,” etc. in a controlled setting and is consistent with how ABM macro models are used for counterfactual policy evaluation once the model is stable. citeturn2search20turn2search24turn2search25

**Step seven: metric computation**

Compute and store at each time step:

- CPI-like price level and inflation (per BLS CPI definition, adapted to your basket). citeturn1search1turn1search17  
- GDP via expenditure identity proxy (per BEA definitions). citeturn1search10turn1search6  
- Unemployment rate (U‑3-like share of labor force unemployed). citeturn2search7turn2search15  
- Gini index (wealth or income distribution). citeturn4search0turn4search17  
- Bank default rate, total lending, firm profits, median household deposits, firm concentration.

Output format: store as columnar files (Parquet) per run so you can compare runs quickly.

**Step eight: dashboard and replay**

Build a minimal dashboard with scenario controls:

- sliders for tax rate, transfer, bank strictness, shock magnitude  
- plots over time for CPI/inflation, unemployment, GDP proxy, lending, defaults  
- run comparison view (multiple seeds or different policies)

This is exactly the “demo surface” that makes the project feel like a platform. citeturn3search3turn3search24

### Testing and reproducibility as first-class deliverables

Agent-based simulations are notorious for “it kind of works but changes every run.” You want controlled randomness:

- Every stochastic component draws from a seeded RNG.  
- Each episode/run stores its seed(s) and scenario configuration.

Write tests around:

- balance-sheet invariants (always)  
- no negative inventories  
- transactions conserve goods (if one good, total produced = total consumed + inventory change + government purchases)  
- default mechanics reduce bank equity appropriately

This aligns with ABM best-practice discussions that emphasize software engineering rigor as central to credible ABM macro work. citeturn2search25turn2search29

## Reinforcement learning integration plan

### Strategy: layer RL onto a stable economic core

Multi-agent RL in a non-stationary economy can be unstable if you try to learn everything at once. The safer roadmap:

- Keep the simulation dynamics and accounting fixed.  
- Convert one agent class at a time from rules → learning policy.  
- Maintain interpretability by comparing RL runs to rule-based baselines under the same shocks and seeds.

This is also consistent with the broader ABM literature emphasizing disciplined experimentation and validation rather than unconstrained “AI behavior” narratives. citeturn2search25turn0search8

### Environment interface design

Your RL environment should expose:

- `observation`: local state features for each agent  
- `action`: discrete or bounded continuous decision variables  
- `reward`: per-agent scalar (or team reward for institutions like government)

Use RL environment standards so you can swap trainers. Gymnasium’s `reset()` and `step()` design is the baseline, while PettingZoo provides a multi-agent standard with a parallel API that matches simultaneous actions. citeturn0search3turn0search6turn0search10

Because your economy has many households and a few firms, you should **parameter-share policies** by agent type (one household policy instance, one firm policy instance, etc.) rather than training 100 separate policies. This is a common scaling trick in multi-agent systems and maps naturally to multi-agent frameworks that support agent IDs and policy mapping. citeturn3search5turn3search1

### Observations and actions by role

Keep observation spaces compact and normalized; use a “local + summary” pattern.

**Household observation** (example)
- deposits, debt, last wage, employment status, price level, expected inflation (rolling), tax rate, transfer level, credit availability proxy

**Household action** (phase-in)
- consumption share \([0,1]\)  
- labor participation decision (0/1) or hours \([0,1]\)  
- borrowing request \([0, L_{max}]\) (optional)

**Firm observation**
- inventory, last sales, last price, competitor average price, wage level, current loans, expected demand proxy, bank lending terms

**Firm action**
- posted price  
- wage offer  
- vacancies/hiring target  
- production target

**Bank observation**
- defaults in last periods, equity ratio proxy, liquidity buffer, macro conditions (inflation/unemployment), applicant stats

**Bank action**
- lending threshold / risk premium / max loan-to-income ratio

**Government observation**
- inflation, unemployment, GDP proxy, inequality metric, tax revenue, deficit

**Government action**
- tax rate and/or transfer level, spending level (with constraints)

You do not have to expose everything as an action; keep some constraints hard-coded early (e.g., non-negative wages).

### Reward design and constraint shaping

Reward design is where you encode “economics” into learning; this must be thought through so you don’t create degenerate incentives.

Recommended baseline reward forms:

- **Household reward (utility)**: \(u(c) - \phi \cdot \text{labor} - \lambda \cdot \text{default}\)  
- **Firm reward (profitability & survival)**: profits minus bankruptcy penalty and maybe volatility penalty  
- **Bank reward**: interest income − default losses − regulatory violation penalty  
- **Government reward**: social welfare proxy: \( w_y \cdot \text{GDP} - w_\pi \cdot \pi^2 - w_u \cdot u^2 - w_g \cdot \text{Gini} \)

You have credible measurement anchors for these terms: inflation via CPI logic, GDP via expenditure identity, unemployment via U‑3 definition, inequality via Gini. citeturn1search1turn1search10turn2search7turn4search0

If you include bank “regulatory” constraints, use a penalty when lending would violate a capital/liquidity threshold, motivated by the existence and purpose of Basel III capital reforms. citeturn2search2

### Algorithm choices and training regimen

Start with on-policy, stable algorithms for continuous control decisions like pricing and wage setting.

- SB3 documents PPO and other algorithms, and PPO is commonly used for relatively stable policy-gradient training. citeturn3search0turn3search25turn3search15  
- For multi-agent training at larger scale with policy mapping and distributed sampling, Ray’s multi-agent environment interfaces are specifically designed for multiple agents identified by IDs. citeturn3search5turn3search1turn5search19

A disciplined training plan:

- **Phase A (single learner)**: learn firm pricing policy while households are rule-based and government/bank fixed.  
- **Phase B (two learners)**: learn households + firms with fixed bank and government.  
- **Phase C (institution learning)**: learn bank policy (credit tightening/loosening) under fixed macro policy.  
- **Phase D (policy learning)**: learn government policy under fixed private-sector learners (or vice versa).  

At each phase, compare to your rule-based baseline under identical shock schedules and seeds.

### Non-stationarity, safety rails, and interpretability

Economies are inherently adaptive; multi-agent RL introduces additional non-stationarity. Your mitigations should be explicit:

- parameter sharing by type (reduces variance)  
- slow policy updates / fewer co-learning agents at once  
- reward clipping and constraint penalties  
- action bounds and “can’t break accounting” hard constraints

Interpretability requirements:

- always log action distributions and state summaries  
- store per-step decomposed reward components (utility term, penalty term, etc.)  
- keep counterfactual replays: the same scenario with RL actions swapped to baseline actions

This is what turns “agents learned stuff” into an actually defensible experiment. citeturn2search25turn0search8

## Experimentation and validation plan

### Experiment design: scenarios and counterfactuals

Treat experiments like controlled studies:

- Create a canonical scenario suite: baseline, supply shock, demand shock, credit crunch, tax hike, stimulus.  
- For each suite entry, run multiple seeds and report distributions (mean + quantiles).  
- For each policy change, run a paired counterfactual under identical random draws where possible.

This reflects how ABM macro work is often evaluated: by comparing policy regimes and shock responses once the model is stable and interpretable. citeturn2search20turn2search24turn2search25

### Validation ladder: from internal validity to stylized facts

Use a staged validation approach:

**Internal validity**  
- accounting identities always hold  
- changes in deposits/loans align with the loan-creates-deposit logic if modeled that way citeturn1search0turn1search4  
- metrics match definitions (CPI/GDP/unemployment) citeturn1search1turn1search10turn2search7

**Behavioral sanity checks**  
- supply shock raises prices and reduces output  
- demand shock reduces output and employment  
- credit tightening reduces lending and amplifies downturns

**Stylized macro relationships (optional but impressive)**  
If you want to demonstrate realism-like behavior, test whether your simulated economy qualitatively exhibits:
- inverse inflation–unemployment dynamics (Phillips-curve-like behavior), as described in accessible explanations of the concept citeturn4search19turn4search23  
- negative correlation between output growth and unemployment changes (Okun’s-law-like behavior), noting that it is used as a rule of thumb but not perfectly stable citeturn4search6turn4search14  

You don’t need to “prove macroeconomics,” but demonstrating that your model generates recognizable qualitative patterns is compelling for a demo or paper.

### Reporting artifacts: paper-quality outputs

To make the project read like research:

- a “Model Card” document: agent definitions, markets, timing, parameters, shock processes  
- a “Reproducibility” document: how to rerun baseline and figure generation with pinned configs  
- figure bundle: impulse responses for each shock under each policy regime  
- ablation results: what changes when you remove credit or change price adjustment speed

These are consistent with the ABM literature’s emphasis on transparency, validation, and cumulative research value. citeturn2search25turn2search29turn0search8

## Platform-quality packaging and long-run roadmap

### Turning the simulator into a portfolio-grade system

A portfolio-grade outcome is not just an algorithm; it’s a system that other people can *use*. The simplest packaging path:

- a local scenario runner (CLI)  
- an interactive dashboard for scenario controls and plots (Streamlit) citeturn3search3turn3search24  
- run “replay files” (config + seed + outputs) enabling easy comparison and sharing  

To make the project feel like “infrastructure,” add:

- experiment registry: each run gets a unique ID, metadata, and stored outputs  
- deterministic run mode: no hidden randomness  
- basic profiling: time per step, time per market, time per metric  

### Scaling up complexity without a rewrite

Once MVP + first RL integration is stable, the most natural “Phase 4” improvements (each should be modular) include:

- **multiple goods / sectors** (enables relative prices and sectoral shocks)  
- **supply chains** (intermediate goods)  
- **heterogeneous labor** (skills, wages, unemployment stratification)  
- **innovation / growth** (ties to K+S style macro ABMs that blend innovation with demand and credit) citeturn2search20turn2search4  
- **multi-bank system** (interbank dynamics, systemic risk)  
- **network analysis** of credit and trade networks (NetworkX) citeturn3search6turn3search2

If you choose to productize, you can expose simulation runs as API calls and a persistent experiment database. FastAPI is explicitly positioned as a fast Python framework for building APIs, and a component-based UI layer can be built with React. citeturn5search0turn5search5

### What “done” looks like for a serious demo or paper

A credible “serious project” end state is:

- Rule-based baseline economy that generates coherent macro metrics and plausible shock responses. citeturn2search25turn0search8  
- At least one learning agent class whose learned policy can be compared to baseline rules under identical scenarios, with interpretable logs and no accounting breaks. citeturn0search3turn0search6turn3search25  
- A dashboard showing indicators, policy toggles, and run comparisons. citeturn3search3turn3search24  
- A written model specification + reproducibility scripts so someone else can reproduce your core results. citeturn2search25turn2search29