"""
Burn & Expense Computation Engine
Implements "Python computes, LLM explains" rule with real financial logic.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
import logging

from langsmith_config import traced, log_metric, log_assumption

logger = logging.getLogger(__name__)


@dataclass
class BurnMetrics:
    """Structured burn rate metrics."""
    gross_burn: float  # Total cash out per month
    net_burn: float  # Gross burn - cash received per month
    gross_burn_3m_avg: float  # 3-month trailing average gross burn
    net_burn_3m_avg: float  # 3-month trailing average net burn
    one_time_expenses: float  # One-time expenses separate bucket
    recurring_expenses: float  # Recurring operating expenses
    fully_loaded_ratio: float  # 1.25-1.4x multiplier for headcount
    cash_runway_months: float  # Months until cash runs out
    cash_balance: float
    monthly_revenue: float
    burn_multiple: float  # net burn / net new ARR


@dataclass
class ScenarioImpact:
    """Impact of a scenario on burn metrics."""
    incremental_burn: float  # Additional monthly burn
    incremental_revenue: float  # Additional monthly revenue
    net_impact: float  # net_burn + incremental_revenue - incremental_burn
    new_cash_runway: float
    assumptions: Dict[str, Any]
    confidence_interval: Tuple[float, float]  # Lower, upper bounds


@traced("compute_burn_metrics", tags=["burn", "expense"])
def compute_burn(
    transactions_df: pd.DataFrame,
    cash_balance: float,
    monthly_revenue: float,
    current_month: Optional[datetime] = None,
    scenario_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Compute comprehensive burn metrics from transaction data.
    
    Args:
        transactions_df: DataFrame with columns ['date', 'amount', 'type', 'category', 'one_time']
        cash_balance: Current cash balance
        monthly_revenue: Current monthly revenue (cash basis)
        current_month: Reference month for calculation
        scenario_overrides: Scenario parameters (headcount, revenue changes, etc.)
    
    Returns:
        Dict with burn metrics and analysis
    """
    if current_month is None:
        current_month = datetime.now().replace(day=1)
    
    # Ensure date column is datetime
    if 'date' in transactions_df.columns:
        transactions_df['date'] = pd.to_datetime(transactions_df['date'])
        transactions_df['month'] = transactions_df['date'].dt.to_period('M')
    else:
        raise ValueError("transactions_df must have 'date' column")
    
    # Filter last 3 months
    three_months_ago = current_month - pd.DateOffset(months=3)
    recent_df = transactions_df[transactions_df['date'] >= three_months_ago].copy()
    
    # Calculate monthly aggregates
    monthly_agg = recent_df.groupby('month').agg({
        'amount': lambda x: -x[x < 0].sum(),  # Total outflows
        'one_time': lambda x: x[x < 0].sum(),  # One-time outflows
    }).reset_index()
    
    # Identify one-time expenses
    one_time_flag = (
        (recent_df['one_time'] == True) | 
        (recent_df['amount'] < 0) & (recent_df['amount'].abs() > recent_df['amount'].abs().quantile(0.90))
    )
    
    # Separate one-time from recurring
    recurring_expenses_df = recent_df[~one_time_flag]
    one_time_expenses_df = recent_df[one_time_flag]
    
    # Monthly recurring expenses
    recurring_agg = recurring_expenses_df.groupby('month')['amount'].sum()
    recurring_expenses = recurring_agg.mean() if len(recurring_agg) > 0 else 0
    
    # Gross burn (total cash out)
    gross_burn = recurring_expenses
    
    # One-time expenses (separate bucket)
    one_time_expenses = one_time_expenses_df['amount'].sum() if len(one_time_expenses_df) > 0 else 0
    
    # Net burn = gross burn - cash collected (simplified using monthly_revenue)
    net_burn = gross_burn - monthly_revenue
    
    # Trailing 3-month average (smoothing)
    if len(recurring_agg) >= 3:
        gross_burn_3m_avg = recurring_agg.tail(3).mean()
        net_burn_3m_avg = gross_burn_3m_avg - monthly_revenue
    else:
        gross_burn_3m_avg = gross_burn
        net_burn_3m_avg = net_burn
    
    # Fully loaded headcount ratio (standard 1.25-1.4x)
    # Detect headcount-related expenses
    headcount_categories = ['salary', 'payroll', 'benefits', 'contractor']
    headcount_mask = recent_df['category'].str.lower().isin(headcount_categories)
    headcount_expenses = recent_df[headcount_mask]['amount'].sum()
    
    # Estimate fully loaded ratio
    if headcount_expenses != 0:
        base_salary = recent_df[headcount_mask & (recent_df['category'].str.lower() == 'salary')]['amount'].sum()
        fully_loaded_ratio = abs(headcount_expenses / base_salary) if base_salary != 0 else 1.3
    else:
        fully_loaded_ratio = 1.3  # Standard assumption
    
    # Apply scenario overrides if provided
    if scenario_overrides:
        gross_burn, net_burn, one_time_expenses = _apply_scenario(
            gross_burn, net_burn, one_time_expenses, scenario_overrides, fully_loaded_ratio
        )
    
    # Cash runway (months until zero)
    if net_burn_3m_avg > 0:  # Positive net burn (spending > revenue)
        cash_runway_months = cash_balance / net_burn_3m_avg
    else:
        cash_runway_months = float('inf')  # Not burning cash
    
    # Burn multiple = net burn / net new ARR
    # Estimate monthly MRR growth
    if len(monthly_agg) >= 2:
        mrr_growth = (monthly_revenue - monthly_agg['amount'].iloc[-2]) / monthly_agg['amount'].iloc[-2]
        burn_multiple = net_burn / (monthly_revenue * mrr_growth) if mrr_growth > 0 else float('inf')
    else:
        burn_multiple = float('inf')
    
    # Log metrics to LangSmith
    log_metric("gross_burn", gross_burn)
    log_metric("net_burn", net_burn)
    log_metric("cash_runway", cash_runway_months)
    log_metric("burn_multiple", burn_multiple)
    
    return {
        "metrics": BurnMetrics(
            gross_burn=abs(gross_burn),
            net_burn=abs(net_burn),
            gross_burn_3m_avg=abs(gross_burn_3m_avg),
            net_burn_3m_avg=abs(net_burn_3m_avg),
            one_time_expenses=abs(one_time_expenses),
            recurring_expenses=abs(recurring_expenses),
            fully_loaded_ratio=fully_loaded_ratio,
            cash_runway_months=cash_runway_months,
            cash_balance=cash_balance,
            monthly_revenue=monthly_revenue,
            burn_multiple=burn_multiple,
        ),
        "monthly_breakdown": monthly_agg.to_dict('records'),
        "one_time_expenses_breakdown": one_time_expenses_df.to_dict('records'),
        "assumptions": {
            "fully_loaded_ratio": fully_loaded_ratio,
            "smoothing_window": 3,
            "scenario_overrides": scenario_overrides,
        }
    }


def _apply_scenario(
    gross_burn: float,
    net_burn: float,
    one_time_expenses: float,
    scenario_overrides: Dict[str, Any],
    fully_loaded_ratio: float,
) -> Tuple[float, float, float]:
    """Apply scenario overrides to burn metrics."""
    new_gross_burn = gross_burn
    new_net_burn = net_burn
    new_one_time = one_time_expenses
    
    # Headcount changes
    if "headcount_change" in scenario_overrides:
        headcount_delta = scenario_overrides["headcount_change"]
        avg_salary = scenario_overrides.get("avg_salary", 120000)  # Annual
        fully_loaded_cost = avg_salary / 12 * fully_loaded_ratio * headcount_delta
        
        # Ramp time: linear onboarding over 3 months
        ramp_months = scenario_overrides.get("ramp_months", 3)
        ramp_factor = 1 / ramp_months
        
        new_gross_burn += fully_loaded_cost * ramp_factor
        new_net_burn += fully_loaded_cost * ramp_factor
    
    # Revenue changes
    if "revenue_change" in scenario_overrides:
        revenue_delta = scenario_overrides["revenue_change"]
        new_net_burn -= revenue_delta
    
    # One-time expenses
    if "one_time_expenses" in scenario_overrides:
        new_one_time += scenario_overrides["one_time_expenses"]
    
    return new_gross_burn, new_net_burn, new_one_time


@traced("compute_scenario_impact", tags=["scenario", "what-if"])
def compute_scenario_impact(
    base_metrics: Dict[str, Any],
    scenario_overrides: Dict[str, Any],
) -> ScenarioImpact:
    """
    Compute the impact of a scenario on burn metrics.
    """
    base_burn = base_metrics["metrics"].gross_burn
    base_revenue = base_metrics["metrics"].monthly_revenue
    base_cash_runway = base_metrics["metrics"].cash_runway_months
    
    # Calculate incremental burn from scenario
    incremental_burn = 0
    incremental_revenue = 0
    
    if "headcount_change" in scenario_overrides:
        headcount_delta = scenario_overrides["headcount_change"]
        avg_salary = scenario_overrides.get("avg_salary", 120000)
        fully_loaded_ratio = base_metrics["metrics"].fully_loaded_ratio
        ramp_months = scenario_overrides.get("ramp_months", 3)
        
        incremental_burn += (avg_salary / 12 * fully_loaded_ratio * headcount_delta) / ramp_months
    
    if "revenue_change" in scenario_overrides:
        incremental_revenue += scenario_overrides["revenue_change"]
    
    # Net impact
    net_impact = incremental_revenue - incremental_burn
    
    # New cash runway
    new_net_burn = base_metrics["metrics"].net_burn + incremental_burn - incremental_revenue
    if new_net_burn > 0:
        new_cash_runway = base_metrics["metrics"].cash_balance / new_net_burn
    else:
        new_cash_runway = float('inf')
    
    # Confidence interval (simplified)
    confidence_lower = new_cash_runway * 0.85
    confidence_upper = new_cash_runway * 1.15
    
    # Log assumption about scenario
    log_assumption(
        source="scenario_impact",
        parameter="scenario_confidence",
        value=0.85,
        rationale="Standard 15% uncertainty for scenario modeling",
        confidence=0.7,
    )
    
    return ScenarioImpact(
        incremental_burn=incremental_burn,
        incremental_revenue=incremental_revenue,
        net_impact=net_impact,
        new_cash_runway=new_cash_runway,
        assumptions=scenario_overrides,
        confidence_interval=(confidence_lower, confidence_upper),
    )