"""
EconoSim Dashboard — Streamlit interactive visualization.

Run with:  streamlit run dashboard.py
"""

from __future__ import annotations

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

from econosim.config.schema import SimulationConfig
from econosim.experiments.runner import run_experiment, run_batch
from econosim.metrics.collector import enrich_dataframe, aggregate_runs, summary_statistics

# ── Page config ─────────────────────────────────────────────────────

st.set_page_config(page_title="EconoSim", layout="wide", page_icon="🏛️")

# ── Custom CSS ──────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Global font */
    html, body, [class*="css"] { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }

    /* Compact header */
    .block-container { padding-top: 1.5rem; }

    /* KPI card styling */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    div[data-testid="stMetric"] label {
        color: #64748b;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1e293b;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricDelta"] {
        font-size: 0.8rem;
    }

    /* Tab styling */
    button[data-baseweb="tab"] {
        font-weight: 600;
        font-size: 0.9rem;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] > div {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
    }
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] span {
        color: #e2e8f0 !important;
    }
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stNumberInput label,
    section[data-testid="stSidebar"] .stSlider label {
        color: #94a3b8 !important;
        font-size: 0.8rem;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# ── Color palette ───────────────────────────────────────────────────

COLORS = {
    "blue":    "#3b82f6",
    "indigo":  "#6366f1",
    "emerald": "#10b981",
    "amber":   "#f59e0b",
    "rose":    "#f43f5e",
    "purple":  "#a855f7",
    "teal":    "#14b8a6",
    "slate":   "#64748b",
    "cyan":    "#06b6d4",
    "orange":  "#f97316",
}
FILL_ALPHA = "0.10"

# ── Sidebar ─────────────────────────────────────────────────────────

st.sidebar.markdown("## 🏛️ EconoSim")
st.sidebar.caption("Multi-agent economic simulation")
st.sidebar.divider()

with st.sidebar.expander("**Simulation**", expanded=True):
    num_periods = st.number_input("Periods", 10, 500, 60, step=10)
    seed = st.number_input("Seed", 0, 99999, 42)
    n_seeds = st.number_input("Batch runs (CI bands)", 1, 20, 1,
                               help="Run multiple seeds to get confidence intervals")

with st.sidebar.expander("**Households**"):
    hh_count = st.number_input("Count", 10, 500, 100, step=10)
    hh_deposits = st.number_input("Initial deposits", 100.0, 50000.0, 1000.0, step=100.0)
    consumption_propensity = st.slider("Consumption propensity (α₁)", 0.1, 1.0, 0.8, 0.05,
                                        help="Fraction of disposable income consumed")
    wealth_propensity = st.slider("Wealth propensity (α₂)", 0.0, 1.0, 0.4, 0.05,
                                   help="Fraction of wealth consumed per period")
    reservation_wage = st.number_input("Reservation wage", 0.0, 200.0, 50.0, step=10.0)

with st.sidebar.expander("**Firms**"):
    firm_count = st.number_input("Count", 1, 50, 5)
    firm_deposits = st.number_input("Initial deposits (firm)", 1000.0, 100000.0, 15000.0, step=1000.0)
    initial_price = st.number_input("Initial price", 1.0, 100.0, 10.0, step=1.0)
    initial_wage = st.number_input("Initial wage", 10.0, 500.0, 60.0, step=10.0)
    labor_productivity = st.number_input("Productivity", 1.0, 50.0, 8.0, step=1.0,
                                          help="Goods produced per worker per period")
    price_adj = st.slider("Price adjustment speed", 0.01, 0.20, 0.03, 0.01)
    wage_adj = st.slider("Wage adjustment speed", 0.01, 0.20, 0.02, 0.01)

with st.sidebar.expander("**Government**"):
    tax_rate = st.slider("Income tax rate", 0.0, 0.5, 0.2, 0.05)
    transfer = st.number_input("Transfer per unemployed", 0.0, 500.0, 50.0, step=10.0)
    govt_spending = st.number_input("Spending per period", 0.0, 20000.0, 2000.0, step=200.0)
    govt_deposits = st.number_input("Initial deposits (govt)", 10000.0, 1000000.0, 100000.0, step=10000.0)

with st.sidebar.expander("**Banking**"):
    base_rate = st.number_input("Base interest rate", 0.0, 0.05, 0.005, step=0.001, format="%.3f")
    car = st.slider("Capital adequacy ratio", 0.01, 0.20, 0.08, 0.01)

st.sidebar.divider()
run_btn = st.sidebar.button("▶  Run Simulation", type="primary", use_container_width=True)

# ── State management ────────────────────────────────────────────────

if "results" not in st.session_state:
    st.session_state.results = None
    st.session_state.agg = None
    st.session_state.config_used = None

# ── Run simulation ──────────────────────────────────────────────────

if run_btn:
    config = SimulationConfig(
        num_periods=num_periods,
        seed=seed,
        household={
            "count": hh_count,
            "initial_deposits": hh_deposits,
            "consumption_propensity": consumption_propensity,
            "wealth_propensity": wealth_propensity,
            "reservation_wage": reservation_wage,
        },
        firm={
            "count": firm_count,
            "initial_deposits": firm_deposits,
            "initial_price": initial_price,
            "initial_wage": initial_wage,
            "labor_productivity": labor_productivity,
            "price_adjustment_speed": price_adj,
            "wage_adjustment_speed": wage_adj,
        },
        government={
            "income_tax_rate": tax_rate,
            "transfer_per_unemployed": transfer,
            "spending_per_period": govt_spending,
            "initial_deposits": govt_deposits,
        },
        bank={
            "base_interest_rate": base_rate,
            "capital_adequacy_ratio": car,
        },
    )

    with st.spinner(f"Running {n_seeds} simulation(s) for {num_periods} periods..."):
        if n_seeds == 1:
            result = run_experiment(config)
            st.session_state.results = [result]
            st.session_state.agg = None
        else:
            seeds_list = list(range(seed, seed + n_seeds))
            batch = run_batch(config, seeds_list)
            st.session_state.results = batch["runs"]
            st.session_state.agg = batch["aggregate"]

    st.session_state.config_used = config

# ── Dashboard Header ────────────────────────────────────────────────

if st.session_state.results is None:
    st.markdown("# 🏛️ EconoSim Dashboard")
    st.markdown("---")
    col_l, col_r = st.columns([2, 1])
    with col_l:
        st.markdown("""
        ### Welcome to EconoSim
        A multi-agent economic simulation where **households**, **firms**, **banks**, and
        **government** interact in a closed economy with stock-flow consistent accounting.

        **Get started:**
        1. Adjust parameters in the sidebar
        2. Click **▶ Run Simulation**
        3. Explore the results across tabs

        The simulation models consumption, production, labor markets, credit,
        government fiscal policy, and sovereign money creation.
        """)
    with col_r:
        st.markdown("""
        **Key features:**
        - Double-entry accounting
        - Sovereign money creation
        - Buffer-stock consumption
        - Revenue-based hiring
        - Confidence intervals (batch runs)
        - CSV data export
        """)
    st.stop()

results = st.session_state.results
df = results[0]["dataframe"]
agg = st.session_state.agg
config_used = st.session_state.config_used

# ── Header ──────────────────────────────────────────────────────────

st.markdown("# 🏛️ EconoSim Dashboard")

# ── KPI cards ───────────────────────────────────────────────────────

final = df.iloc[-1]
first = df.iloc[0]

def delta_str(col: str, fmt: str = ",.0f", pct: bool = False) -> str | None:
    if col not in df.columns or len(df) < 2:
        return None
    v0, v1 = first[col], final[col]
    if pct:
        return f"{v1 - v0:+.1%}"
    return f"{v1 - v0:+{fmt}}"

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("GDP", f"{final.get('gdp', 0):,.0f}", delta=delta_str("gdp"))
c2.metric("Unemployment", f"{final.get('unemployment_rate', 0):.1%}",
          delta=delta_str("unemployment_rate", pct=True), delta_color="inverse")
c3.metric("Avg Price", f"{final.get('avg_price', 0):.2f}", delta=delta_str("avg_price", ".2f"))
c4.metric("Avg Wage", f"{final.get('avg_wage', 0):.0f}", delta=delta_str("avg_wage", ".0f"))
c5.metric("Gini", f"{final.get('gini_deposits', 0):.3f}", delta=delta_str("gini_deposits", ".3f"),
          delta_color="inverse")
c6.metric("Loans", f"{final.get('total_loans_outstanding', 0):,.0f}",
          delta=delta_str("total_loans_outstanding"))

st.markdown("")

# ── Plotting helpers ────────────────────────────────────────────────

CHART_LAYOUT = dict(
    template="plotly_white",
    height=340,
    margin=dict(l=50, r=20, t=45, b=45),
    font=dict(family="Inter, -apple-system, sans-serif", size=12),
    title_font=dict(size=14, color="#1e293b"),
    xaxis=dict(
        title="Period", gridcolor="#f1f5f9", zeroline=False,
        title_font=dict(size=11, color="#64748b"),
    ),
    yaxis=dict(
        gridcolor="#f1f5f9", zeroline=True, zerolinecolor="#e2e8f0",
        title_font=dict(size=11, color="#64748b"),
    ),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                font=dict(size=10)),
    plot_bgcolor="white",
)


def _hex_to_rgba(hex_color: str, alpha: str) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def make_chart(
    metrics: list[tuple[str, str, str]],
    title: str,
    yaxis_title: str = "",
    yaxis_fmt: str | None = None,
) -> go.Figure:
    """Create a chart with one or more metrics, with optional CI bands.

    metrics: list of (metric_key, display_name, color_key)
    """
    fig = go.Figure()

    for metric, name, color_key in metrics:
        color = COLORS.get(color_key, color_key)
        fill_color = _hex_to_rgba(color, FILL_ALPHA)

        if agg is not None and f"{metric}_mean" in agg.columns:
            fig.add_trace(go.Scatter(
                x=agg.index, y=agg[f"{metric}_hi"],
                mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip",
            ))
            fig.add_trace(go.Scatter(
                x=agg.index, y=agg[f"{metric}_lo"],
                mode="lines", line=dict(width=0),
                fill="tonexty", fillcolor=fill_color,
                showlegend=False, hoverinfo="skip",
            ))
            fig.add_trace(go.Scatter(
                x=agg.index, y=agg[f"{metric}_mean"],
                mode="lines", line=dict(color=color, width=2.5),
                name=name,
            ))
        elif metric in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[metric],
                mode="lines", line=dict(color=color, width=2.5),
                name=name,
            ))

    layout = {**CHART_LAYOUT, "title": title}
    if yaxis_title:
        layout["yaxis"] = {**layout.get("yaxis", {}), "title": yaxis_title}
    if yaxis_fmt:
        layout["yaxis"] = {**layout.get("yaxis", {}), "tickformat": yaxis_fmt}

    fig.update_layout(**layout)
    return fig


def make_stacked_area(
    metrics: list[tuple[str, str, str]],
    title: str,
    yaxis_title: str = "",
) -> go.Figure:
    """Create a stacked area chart."""
    fig = go.Figure()
    for metric, name, color_key in metrics:
        color = COLORS.get(color_key, color_key)
        if metric in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[metric],
                mode="lines", stackgroup="one",
                line=dict(width=0.5, color=color),
                fillcolor=_hex_to_rgba(color, "0.4"),
                name=name,
            ))
    fig.update_layout(**{**CHART_LAYOUT, "title": title})
    if yaxis_title:
        fig.update_layout(yaxis_title=yaxis_title)
    return fig


def chart_row(charts: list[go.Figure]) -> None:
    """Display charts in equal-width columns."""
    cols = st.columns(len(charts))
    for col, fig in zip(cols, charts):
        col.plotly_chart(fig, use_container_width=True)


# ── Tabs ────────────────────────────────────────────────────────────

tab_macro, tab_labor, tab_fiscal, tab_money, tab_data = st.tabs([
    "📈  Macro", "👷  Labor & Production", "🏛️  Government",
    "💰  Money & Credit", "📋  Data",
])

# ── Macro tab ───────────────────────────────────────────────────────

with tab_macro:
    chart_row([
        make_chart([("gdp", "GDP", "blue")], "Gross Domestic Product"),
        make_chart([("avg_price", "Price Level", "amber")], "Average Price"),
    ])
    chart_row([
        make_chart([("unemployment_rate", "Unemployment", "rose")],
                   "Unemployment Rate", yaxis_fmt=".0%"),
        make_chart([("inflation_rate", "Inflation", "emerald")],
                   "Inflation Rate (period-over-period)", yaxis_fmt=".1%"),
    ])
    chart_row([
        make_chart([("gini_deposits", "Gini", "purple")],
                   "Wealth Inequality (Gini Coefficient)"),
        make_chart([("gdp_growth", "GDP Growth", "teal")],
                   "GDP Growth Rate", yaxis_fmt=".1%"),
    ])

# ── Labor & Production tab ─────────────────────────────────────────

with tab_labor:
    chart_row([
        make_chart([
            ("total_employed", "Employed", "emerald"),
            ("labor_force", "Labor Force", "slate"),
        ], "Employment"),
        make_chart([("total_vacancies", "Vacancies", "orange")], "Firm Vacancies"),
    ])
    chart_row([
        make_chart([("avg_wage", "Wage", "blue")], "Average Wage"),
        make_chart([("total_production", "Production", "teal")], "Total Production (units)"),
    ])
    chart_row([
        make_chart([("total_inventory", "Inventory", "purple")], "Total Inventory"),
        make_chart([("total_consumption", "Consumption", "rose")], "Total Consumption Spending"),
    ])

# ── Government tab ──────────────────────────────────────────────────

with tab_fiscal:
    # Fiscal overview KPIs
    g1, g2, g3, g4 = st.columns(4)
    g1.metric("Tax Revenue", f"{final.get('govt_tax_revenue', 0):,.0f}")
    g2.metric("Transfers", f"{final.get('govt_transfers', 0):,.0f}")
    g3.metric("Spending", f"{final.get('govt_spending', 0):,.0f}")
    g4.metric("Budget Balance", f"{final.get('govt_budget_balance', 0):,.0f}",
              delta_color="normal")
    st.markdown("")

    chart_row([
        make_chart([
            ("govt_tax_revenue", "Tax Revenue", "emerald"),
            ("govt_transfers", "Transfers", "amber"),
            ("govt_spending", "Spending", "blue"),
        ], "Fiscal Flows"),
        make_chart([("govt_budget_balance", "Budget Balance", "rose")],
                   "Budget Balance (T - G - Tr)"),
    ])
    chart_row([
        make_chart([("govt_deposits", "Deposits", "indigo")], "Government Deposits"),
        make_chart([
            ("govt_money_created", "Per-period", "amber"),
            ("govt_cumulative_money_created", "Cumulative", "rose"),
        ], "Sovereign Money Creation"),
    ])

# ── Money & Credit tab ─────────────────────────────────────────────

with tab_money:
    chart_row([
        make_stacked_area([
            ("total_hh_deposits", "Households", "blue"),
            ("total_firm_deposits", "Firms", "amber"),
            ("govt_deposits", "Government", "emerald"),
        ], "Deposit Distribution", "Deposits"),
        make_chart([
            ("total_hh_deposits", "Household", "blue"),
            ("total_firm_deposits", "Firm", "amber"),
        ], "Deposits by Sector"),
    ])
    chart_row([
        make_chart([("total_loans_outstanding", "Loans", "rose")],
                   "Total Loans Outstanding"),
        make_chart([
            ("bank_equity", "Bank Equity", "emerald"),
        ], "Bank Equity"),
    ])
    chart_row([
        make_chart([("bank_capital_ratio", "CAR", "indigo")],
                   "Bank Capital Adequacy Ratio", yaxis_fmt=".1%"),
        make_chart([("velocity", "Velocity", "purple")],
                   "Velocity of Money (GDP / Deposits)"),
    ])

# ── Data tab ────────────────────────────────────────────────────────

with tab_data:
    st.markdown("### 📋 Simulation Data")

    # Column selector
    all_cols = df.columns.tolist()
    default_cols = ["gdp", "unemployment_rate", "avg_price", "avg_wage",
                    "total_hh_deposits", "total_firm_deposits", "govt_budget_balance"]
    selected = st.multiselect("Select columns to display",
                               all_cols,
                               default=[c for c in default_cols if c in all_cols])

    if selected:
        st.dataframe(
            df[selected].style.format("{:.2f}"),
            use_container_width=True,
            height=420,
        )
    else:
        st.dataframe(df, use_container_width=True, height=420)

    st.markdown("### 📊 Summary Statistics")
    stats = summary_statistics(df)
    stats_df = pd.DataFrame(stats).T
    st.dataframe(
        stats_df.style.format("{:.4f}"),
        use_container_width=True,
    )

    st.markdown("---")
    col_dl1, col_dl2, _ = st.columns([1, 1, 2])
    with col_dl1:
        st.download_button(
            "📥  Download Metrics CSV",
            df.to_csv(),
            "econosim_metrics.csv",
            "text/csv",
            use_container_width=True,
        )
    with col_dl2:
        import json as _json
        st.download_button(
            "📥  Download Summary JSON",
            _json.dumps(stats, indent=2, default=str),
            "econosim_summary.json",
            "application/json",
            use_container_width=True,
        )
