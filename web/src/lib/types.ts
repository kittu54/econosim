export interface HouseholdParams {
  count: number;
  initial_deposits: number;
  consumption_propensity: number;
  wealth_propensity: number;
  reservation_wage: number;
}

export interface FirmParams {
  count: number;
  initial_deposits: number;
  initial_price: number;
  initial_wage: number;
  labor_productivity: number;
  price_adjustment_speed: number;
  wage_adjustment_speed: number;
}

export interface GovernmentParams {
  income_tax_rate: number;
  transfer_per_unemployed: number;
  spending_per_period: number;
  initial_deposits: number;
}

export interface BankParams {
  base_interest_rate: number;
  capital_adequacy_ratio: number;
}

export interface SimulationRequest {
  num_periods: number;
  seed: number;
  n_seeds: number;
  household: HouseholdParams;
  firm: FirmParams;
  government: GovernmentParams;
  bank: BankParams;
}

export interface PeriodData {
  period: number;
  gdp: number;
  unemployment_rate: number;
  avg_price: number;
  avg_wage: number;
  gini_deposits: number;
  total_loans_outstanding: number;
  total_employed: number;
  labor_force: number;
  total_production: number;
  total_consumption: number;
  total_inventory: number;
  total_vacancies: number;
  total_hh_deposits: number;
  total_firm_deposits: number;
  bank_equity: number;
  bank_capital_ratio: number;
  govt_deposits: number;
  govt_tax_revenue: number;
  govt_transfers: number;
  govt_spending: number;
  govt_budget_balance: number;
  govt_money_created: number;
  govt_cumulative_money_created: number;
  inflation_rate?: number;
  gdp_growth?: number;
  velocity?: number;
  [key: string]: number | undefined;
}

export interface AggregateData {
  period: number;
  [key: string]: number | undefined;
}

export interface SimulationResponse {
  periods: PeriodData[];
  summary: Record<string, Record<string, number>>;
  config: Record<string, unknown>;
  has_ci: boolean;
  aggregate: AggregateData[] | null;
}

export const DEFAULT_CONFIG: SimulationRequest = {
  num_periods: 60,
  seed: 42,
  n_seeds: 1,
  household: {
    count: 100,
    initial_deposits: 1000,
    consumption_propensity: 0.8,
    wealth_propensity: 0.4,
    reservation_wage: 50,
  },
  firm: {
    count: 5,
    initial_deposits: 15000,
    initial_price: 10,
    initial_wage: 60,
    labor_productivity: 8,
    price_adjustment_speed: 0.03,
    wage_adjustment_speed: 0.02,
  },
  government: {
    income_tax_rate: 0.2,
    transfer_per_unemployed: 50,
    spending_per_period: 2000,
    initial_deposits: 100000,
  },
  bank: {
    base_interest_rate: 0.005,
    capital_adequacy_ratio: 0.08,
  },
};
