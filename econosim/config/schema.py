"""
Pydantic configuration schemas for simulation parameters, agent defaults,
market settings, and scenario definitions.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class HouseholdConfig(BaseModel):
    count: int = 100
    initial_deposits: float = 1000.0
    consumption_propensity: float = 0.8  # fraction of disposable income consumed
    wealth_propensity: float = 0.4  # fraction of wealth consumed per period
    labor_participation_rate: float = 0.95
    reservation_wage: float = 50.0  # minimum acceptable wage


class FirmConfig(BaseModel):
    count: int = 5
    initial_deposits: float = 15000.0
    initial_inventory: float = 100.0
    initial_price: float = 10.0
    initial_wage: float = 60.0
    labor_productivity: float = 8.0  # goods per worker per period
    target_inventory_ratio: float = 0.2  # target inventory / expected sales
    price_adjustment_speed: float = 0.03
    wage_adjustment_speed: float = 0.02
    max_leverage: float = 3.0  # max debt / equity


class BankConfig(BaseModel):
    initial_equity: float = 50000.0
    initial_reserves: float = 20000.0
    base_interest_rate: float = 0.005  # per period (monthly)
    risk_premium: float = 0.002
    capital_adequacy_ratio: float = 0.08  # min equity / loans
    max_loan_to_value: float = 0.8
    default_threshold_periods: int = 3  # periods delinquent before default
    loan_term_periods: int = 12


class GovernmentConfig(BaseModel):
    income_tax_rate: float = 0.2
    transfer_per_unemployed: float = 50.0
    spending_per_period: float = 2000.0  # government purchases from firms
    initial_deposits: float = 100000.0


class MarketConfig(BaseModel):
    labor_market_matching_rate: float = 0.8  # fraction of vacancies filled
    goods_market_rationing: str = "proportional"  # "proportional" or "random"


class ShockSpec(BaseModel):
    """A shock applied at a specific period."""

    period: int
    shock_type: str  # "supply", "demand", "credit", "fiscal"
    parameter: str  # which parameter to modify
    magnitude: float  # multiplicative or additive change
    additive: bool = False  # if True, add magnitude; if False, multiply


class BondConfig(BaseModel):
    """Configuration for government bond market."""
    default_maturity: int = 12  # periods until bond matures
    default_coupon_rate: float = 0.005  # per-period coupon rate
    max_debt_to_gdp: float = 2.0  # maximum debt-to-GDP ratio


class ExpectationsConfig(BaseModel):
    """Configuration for adaptive expectations."""
    price_alpha: float = 0.3  # adaptive expectations smoothing for prices
    wage_alpha: float = 0.2  # adaptive expectations smoothing for wages
    demand_window: int = 4   # rolling window for demand expectations
    demand_use_trend: bool = True  # use trend extrapolation for demand


class NetworkConfig(BaseModel):
    """Configuration for economic networks."""
    track_trade: bool = True  # record goods market transactions
    track_credit: bool = True  # record credit market transactions
    edge_decay_rate: float = 0.1  # per-period decay of network edges


class ExtensionsConfig(BaseModel):
    """Feature flags and configuration for Phase 4 extensions."""
    enable_expectations: bool = False  # adaptive expectations for firms
    enable_networks: bool = False  # trade/credit network tracking
    enable_bonds: bool = False  # government bond market

    bonds: BondConfig = Field(default_factory=BondConfig)
    expectations: ExpectationsConfig = Field(default_factory=ExpectationsConfig)
    networks: NetworkConfig = Field(default_factory=NetworkConfig)


class SimulationConfig(BaseModel):
    """Top-level simulation configuration."""

    name: str = "baseline"
    description: str = "Baseline MVP scenario"
    num_periods: int = 120  # 10 years of monthly steps
    seed: int = 42

    household: HouseholdConfig = Field(default_factory=HouseholdConfig)
    firm: FirmConfig = Field(default_factory=FirmConfig)
    bank: BankConfig = Field(default_factory=BankConfig)
    government: GovernmentConfig = Field(default_factory=GovernmentConfig)
    market: MarketConfig = Field(default_factory=MarketConfig)
    extensions: ExtensionsConfig = Field(default_factory=ExtensionsConfig)

    shocks: list[ShockSpec] = Field(default_factory=list)

    # Logging
    log_every: int = 1  # log metrics every N periods
    output_dir: str = "outputs"

    model_config = {"validate_assignment": True}
