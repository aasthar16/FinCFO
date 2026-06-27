"""
Helper utilities for the AI CFO platform.
"""

from typing import Dict, Any, Optional, List
import json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np


def format_currency(amount: float, currency: str = "USD") -> str:
    """Format amount as currency."""
    return f"{currency} ${amount:,.0f}"


def format_percent(value: float) -> str:
    """Format as percentage."""
    return f"{value * 100:.1f}%"


def format_date(date, fmt: str = "%B %d, %Y") -> str:
    """Format date."""
    if isinstance(date, str):
        date = datetime.fromisoformat(date)
    return date.strftime(fmt)


def safe_json_parse(data: str) -> Dict[str, Any]:
    """Safely parse JSON."""
    try:
        return json.loads(data)
    except json.JSONDecodeError:
        return {}


def serialize_dataframe(df: pd.DataFrame) -> str:
    """Serialize DataFrame to JSON."""
    return df.to_json(orient='records', date_format='iso')


def deserialize_dataframe(data: str) -> pd.DataFrame:
    """Deserialize JSON to DataFrame."""
    return pd.read_json(data, orient='records')


def calculate_confidence_interval(values: List[float], confidence: float = 0.95) -> Dict[str, float]:
    """
    Calculate confidence interval for a list of values.
    """
    if not values:
        return {"lower": 0, "upper": 0, "mean": 0}
    
    import scipy.stats as stats
    
    mean = np.mean(values)
    std = np.std(values)
    n = len(values)
    
    # Z-score for confidence level
    z_score = stats.norm.ppf(1 - (1 - confidence) / 2)
    margin = z_score * (std / np.sqrt(n))
    
    return {
        "lower": mean - margin,
        "upper": mean + margin,
        "mean": mean,
        "std": std,
        "n": n,
        "confidence": confidence,
    }


def calculate_breakeven(
    fixed_costs: float,
    variable_cost_per_unit: float,
    price_per_unit: float,
) -> Dict[str, Any]:
    """
    Calculate breakeven point.
    """
    if price_per_unit <= variable_cost_per_unit:
        return {
            "units": None,
            "revenue": None,
            "note": "Cannot breakeven: price <= variable cost",
        }
    
    units = fixed_costs / (price_per_unit - variable_cost_per_unit)
    revenue = units * price_per_unit
    
    return {
        "units": units,
        "revenue": revenue,
        "margin_per_unit": price_per_unit - variable_cost_per_unit,
        "margin_ratio": (price_per_unit - variable_cost_per_unit) / price_per_unit,
    }


def generate_scenario_summary(
    scenario_overrides: Dict[str, Any],
    base_metrics: Dict[str, Any],
    impact_metrics: Dict[str, Any],
) -> str:
    """
    Generate a human-readable summary of a scenario's impact.
    """
    summary = []
    
    # Headcount changes
    if "headcount_change" in scenario_overrides:
        delta = scenario_overrides["headcount_change"]
        direction = "adding" if delta > 0 else "reducing"
        summary.append(f"• {direction} {abs(delta)} headcount positions")
    
    # Revenue changes
    if "revenue_change" in scenario_overrides:
        delta = scenario_overrides["revenue_change"]
        direction = "increase" if delta > 0 else "decrease"
        summary.append(f"• {direction} revenue by ${abs(delta):,.0f}/month")
    
    # Impact summary
    if impact_metrics:
        new_runway = impact_metrics.get("new_cash_runway", 0)
        if new_runway != float('inf'):
            base_runway = base_metrics.get("cash_runway_months", 0)
            change = new_runway - base_runway
            direction = "extend" if change > 0 else "shorten"
            summary.append(f"• {direction} runway by {abs(change):.1f} months")
    
    return "\n".join(summary)