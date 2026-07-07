"""
Burn & Expense Computation Engine
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Tuple
from dataclasses import dataclass, asdict

from config.langsmith import traced, log_metric, log_assumption


@dataclass
class BurnMetrics:
    gross_burn: float
    net_burn: float
    gross_burn_3m_avg: float
    net_burn_3m_avg: float
    one_time_expenses: float
    recurring_expenses: float
    fully_loaded_ratio: float
    cash_runway_months: float
    cash_balance: float
    monthly_revenue: float
    burn_multiple: float


def convert_to_serializable(obj):
    """Convert numpy/pandas types to Python native types."""
    if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Period):
        return str(obj)
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, float) and np.isinf(obj):
        return None  # Replace infinity with None
    elif isinstance(obj, float) and np.isnan(obj):
        return None  # Replace NaN with None
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(item) for item in obj)
    elif hasattr(obj, '__dataclass_fields__'):
        return convert_to_serializable(asdict(obj))
    elif hasattr(obj, 'isoformat'):
        return obj.isoformat()
    return obj

@traced("compute_burn", tags=["burn", "expense"])
def compute_burn(
    transactions_df: pd.DataFrame,
    cash_balance: float,
    monthly_revenue: float,
    current_month: datetime = None,
    scenario_overrides: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """
    Compute comprehensive burn metrics from transaction data.
    """
    if current_month is None:
        current_month = datetime.now().replace(day=1)
    
    # Ensure date column exists and is datetime
    if 'date' in transactions_df.columns:
        transactions_df['date'] = pd.to_datetime(transactions_df['date'])
        transactions_df['month'] = transactions_df['date'].dt.to_period('M')
    else:
        raise ValueError("transactions_df must have 'date' column")
    
    # Ensure is_one_time column exists
    if 'is_one_time' not in transactions_df.columns:
        transactions_df['is_one_time'] = False
    
    # Filter to last 3 months
    three_months_ago = current_month - pd.DateOffset(months=3)
    recent_df = transactions_df[transactions_df['date'] >= three_months_ago].copy()
    
    # If no recent data, use all data
    if len(recent_df) == 0:
        recent_df = transactions_df.copy()
    
    # Separate outflows only (negative amounts = expenses)
    outflows = recent_df[recent_df['amount'] < 0].copy()
    
    # If no outflows, return zero metrics
    if len(outflows) == 0:
        gross_burn = 0.0
        recurring_expenses = 0.0
        one_time_expenses = 0.0
        net_burn = max(0.0 - monthly_revenue, 0)
        gross_burn_3m_avg = 0.0
        net_burn_3m_avg = max(0.0 - monthly_revenue, 0)
    else:
        # Identify one-time expenses (flagged or in top 10% by size)
        threshold = outflows['amount'].abs().quantile(0.90) if len(outflows) > 5 else float('inf')
        outflows['is_one_time_derived'] = (
            (outflows['is_one_time'] == True) | 
            (outflows['amount'].abs() > threshold)
        )
        
        # Split into recurring vs one-time
        recurring_df = outflows[~outflows['is_one_time_derived']]
        one_time_df = outflows[outflows['is_one_time_derived']]
        
        # Calculate monthly recurring expenses
        if len(recurring_df) > 0:
            recurring_by_month = recurring_df.groupby('month')['amount'].sum()
            recurring_expenses = float(abs(recurring_by_month.mean()))
        else:
            recurring_expenses = 0.0
        
        # Gross burn = average monthly recurring expenses
        gross_burn = float(recurring_expenses)
        
        # One-time expenses total
        one_time_expenses = float(abs(one_time_df['amount'].sum())) if len(one_time_df) > 0 else 0.0
        
        # Net burn = gross burn - monthly revenue (can't be negative)
        net_burn = float(max(gross_burn - monthly_revenue, 0))
        
        # 3-month averages
        if len(recurring_by_month) >= 3:
            gross_burn_3m_avg = float(abs(recurring_by_month.tail(3).mean()))
        elif len(recurring_by_month) > 0:
            gross_burn_3m_avg = float(abs(recurring_by_month.mean()))
        else:
            gross_burn_3m_avg = float(gross_burn)
        
        net_burn_3m_avg = float(max(gross_burn_3m_avg - monthly_revenue, 0))
    
    # Build monthly aggregation for breakdown display
    monthly_agg = outflows.groupby('month').agg(
        total_outflow=('amount', 'sum'),
        transaction_count=('amount', 'count'),
    ).reset_index()
    monthly_agg['total_outflow'] = monthly_agg['total_outflow'].abs()
    
    # Fully loaded ratio for headcount costs
    fully_loaded_ratio = float(1.3)
    
    # Apply scenario overrides if any
    if scenario_overrides:
        gross_burn, net_burn, one_time_expenses = _apply_scenario(
            gross_burn, net_burn, one_time_expenses, scenario_overrides, fully_loaded_ratio
        )
    
    # Calculate runway
    if net_burn_3m_avg > 0:
        cash_runway_months = float(cash_balance / net_burn_3m_avg)
    elif net_burn > 0:
        cash_runway_months = float(cash_balance / net_burn)
    else:
        cash_runway_months = 36.0  # Default if not burning cash
    
    # Calculate burn multiple
    burn_multiple = 0.0
    if len(monthly_agg) >= 2:
        # Get last two months
        last_two = monthly_agg.tail(2)
        prev_outflow = float(last_two['total_outflow'].iloc[0])
        curr_outflow = float(last_two['total_outflow'].iloc[1])
        
        if prev_outflow > 0:
            growth_rate = float((curr_outflow - prev_outflow) / prev_outflow)
            if growth_rate > 0 and net_burn > 0:
                burn_multiple = float(net_burn / (monthly_revenue * growth_rate))
    
    # Log metrics to LangSmith
    log_metric("gross_burn", float(abs(gross_burn)))
    log_metric("net_burn", float(abs(net_burn)))
    log_metric("cash_runway", float(cash_runway_months))
    
    # Log assumptions
    log_assumption(
        source="compute_burn",
        parameter="fully_loaded_ratio",
        value=float(fully_loaded_ratio),
        rationale="Standard 1.3x fully loaded ratio for headcount",
        confidence=0.8,
    )
    
    metrics = BurnMetrics(
        gross_burn=float(abs(gross_burn)),
        net_burn=float(abs(net_burn)),
        gross_burn_3m_avg=float(abs(gross_burn_3m_avg)),
        net_burn_3m_avg=float(abs(net_burn_3m_avg)),
        one_time_expenses=float(abs(one_time_expenses)),
        recurring_expenses=float(abs(recurring_expenses)),
        fully_loaded_ratio=float(fully_loaded_ratio),
        cash_runway_months=float(cash_runway_months),
        cash_balance=float(cash_balance),
        monthly_revenue=float(monthly_revenue),
        burn_multiple=float(burn_multiple),
    )
    
    # Convert monthly_breakdown to serializable
    monthly_breakdown = []
    for record in monthly_agg.to_dict('records'):
        serializable_record = {}
        for k, v in record.items():
            if isinstance(v, pd.Period):
                serializable_record[k] = str(v)
            elif isinstance(v, (pd.Timestamp, datetime)):
                serializable_record[k] = v.isoformat()
            elif isinstance(v, (np.float64, np.float32)):
                serializable_record[k] = float(v)
            elif isinstance(v, (np.int64, np.int32)):
                serializable_record[k] = int(v)
            else:
                serializable_record[k] = v
        monthly_breakdown.append(serializable_record)
    
    return {
        "metrics": metrics,
        "monthly_breakdown": monthly_breakdown,
        "assumptions": {
            "fully_loaded_ratio": float(fully_loaded_ratio),
            "smoothing_window": 3,
            "scenario_overrides": scenario_overrides or {},
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
    new_gross_burn = float(gross_burn)
    new_net_burn = float(net_burn)
    new_one_time = float(one_time_expenses)
    
    if "headcount_change" in scenario_overrides and scenario_overrides["headcount_change"] is not None:
        headcount_delta = float(scenario_overrides["headcount_change"])
        avg_salary = float(scenario_overrides.get("avg_salary", 120000) or 120000)
        fully_loaded_cost = float(avg_salary / 12 * fully_loaded_ratio * headcount_delta)
        ramp_months = float(scenario_overrides.get("ramp_months", 3) or 3)
        ramp_factor = float(1 / ramp_months)
        
        new_gross_burn += float(fully_loaded_cost * ramp_factor)
        new_net_burn += float(fully_loaded_cost * ramp_factor)
    
    if "revenue_change" in scenario_overrides and scenario_overrides["revenue_change"] is not None:
        revenue_delta = float(scenario_overrides["revenue_change"])
        new_net_burn -= float(revenue_delta)
    
    if "one_time_expenses" in scenario_overrides and scenario_overrides["one_time_expenses"] is not None:
        new_one_time += float(scenario_overrides["one_time_expenses"])
    
    return new_gross_burn, new_net_burn, new_one_time