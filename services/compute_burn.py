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
    
    if 'date' in transactions_df.columns:
        transactions_df['date'] = pd.to_datetime(transactions_df['date'])
        transactions_df['month'] = transactions_df['date'].dt.to_period('M')
    else:
        raise ValueError("transactions_df must have 'date' column")
    
    three_months_ago = current_month - pd.DateOffset(months=3)
    recent_df = transactions_df[transactions_df['date'] >= three_months_ago].copy()
    
    monthly_agg = recent_df.groupby('month').agg({
        'amount': lambda x: -x[x < 0].sum(),
        'one_time': lambda x: x[x < 0].sum(),
    }).reset_index()
    
    one_time_flag = (
        (recent_df['one_time'] == True) | 
        (recent_df['amount'] < 0) & (recent_df['amount'].abs() > recent_df['amount'].abs().quantile(0.90))
    )
    
    recurring_expenses_df = recent_df[~one_time_flag]
    one_time_expenses_df = recent_df[one_time_flag]
    
    recurring_agg = recurring_expenses_df.groupby('month')['amount'].sum()
    recurring_expenses = float(recurring_agg.mean()) if len(recurring_agg) > 0 else 0.0
    
    gross_burn = float(recurring_expenses)
    one_time_expenses = float(one_time_expenses_df['amount'].sum()) if len(one_time_expenses_df) > 0 else 0.0
    net_burn = float(gross_burn - monthly_revenue)
    
    if len(recurring_agg) >= 3:
        gross_burn_3m_avg = float(recurring_agg.tail(3).mean())
        net_burn_3m_avg = float(gross_burn_3m_avg - monthly_revenue)
    else:
        gross_burn_3m_avg = float(gross_burn)
        net_burn_3m_avg = float(net_burn)
    
    fully_loaded_ratio = float(1.3)
    
    if scenario_overrides:
        gross_burn, net_burn, one_time_expenses = _apply_scenario(
            gross_burn, net_burn, one_time_expenses, scenario_overrides, fully_loaded_ratio
        )
    
    # Handle infinite runway
    if net_burn_3m_avg > 0:
        cash_runway_months = float(cash_balance / net_burn_3m_avg)
    else:
        cash_runway_months = None  # Use None instead of infinity
    
    if len(monthly_agg) >= 2:
        mrr_growth = float((monthly_revenue - monthly_agg['amount'].iloc[-2]) / monthly_agg['amount'].iloc[-2])
        if mrr_growth > 0:
            burn_multiple = float(net_burn / (monthly_revenue * mrr_growth))
        else:
            burn_multiple = None
    else:
        burn_multiple = None
    
    # Log metrics to LangSmith
    log_metric("gross_burn", float(abs(gross_burn)))
    log_metric("net_burn", float(abs(net_burn)))
    if cash_runway_months is not None:
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
        cash_runway_months=cash_runway_months if cash_runway_months is not None else 0.0,
        cash_balance=float(cash_balance),
        monthly_revenue=float(monthly_revenue),
        burn_multiple=burn_multiple if burn_multiple is not None else 0.0,
    )
    
    # Convert monthly_breakdown to serializable - handle Period objects
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



