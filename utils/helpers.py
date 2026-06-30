"""
Helper utilities.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional


def generate_mock_transactions(
    start_date: datetime = None,
    months: int = 6,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate mock transaction data for testing.
    """
    if start_date is None:
        start_date = datetime.now() - timedelta(days=months * 30)
    
    np.random.seed(seed)
    
    dates = pd.date_range(start=start_date, periods=months, freq='M')
    categories = ['salary', 'rent', 'software', 'marketing', 'other']
    weights = [0.4, 0.2, 0.15, 0.15, 0.1]
    
    data = []
    for date in dates:
        for _ in range(np.random.randint(10, 20)):
            category = np.random.choice(categories, p=weights)
            amount = -abs(np.random.normal(
                loc={'salary': 5000, 'rent': 3000, 'software': 1000, 'marketing': 2000, 'other': 500}.get(category, 1000),
                scale={'salary': 1000, 'rent': 500, 'software': 300, 'marketing': 800, 'other': 200}.get(category, 300)
            ))
            one_time = np.random.random() < 0.05
            
            data.append({
                'date': date + timedelta(days=np.random.randint(0, 28)),
                'amount': amount,
                'type': 'expense',
                'category': category,
                'one_time': one_time,
            })
    
    df = pd.DataFrame(data)
    df = df.sort_values('date').reset_index(drop=True)
    return df


def transactions_to_serializable(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Convert DataFrame to serializable list of dicts.
    """
    if df is None:
        return []
    
    # Convert date columns to string
    df_copy = df.copy()
    for col in df_copy.columns:
        if pd.api.types.is_datetime64_any_dtype(df_copy[col]):
            df_copy[col] = df_copy[col].dt.isoformat()
        elif pd.api.types.is_timedelta64_dtype(df_copy[col]):
            df_copy[col] = df_copy[col].dt.total_seconds()
    
    return df_copy.to_dict('records')


def format_currency(amount: float, currency: str = "USD") -> str:
    """Format amount as currency."""
    symbols = {
        "USD": "$",
        "EUR": "€",
        "GBP": "£",
        "JPY": "¥",
        "CNY": "¥",
        "CAD": "C$",
        "AUD": "A$",
        "INR": "₹",
        "BRL": "R$",
    }
    symbol = symbols.get(currency, "$")
    return f"{symbol}{amount:,.0f}"


def format_percent(value: float) -> str:
    """Format as percentage."""
    return f"{value * 100:.1f}%"


def format_date(date, fmt: str = "%B %d, %Y") -> str:
    """Format date."""
    if isinstance(date, str):
        date = datetime.fromisoformat(date)
    return date.strftime(fmt)


def calculate_confidence_interval(values: List[float], confidence: float = 0.95) -> Dict[str, float]:
    """Calculate confidence interval for a list of values."""
    if not values:
        return {"lower": 0, "upper": 0, "mean": 0}
    
    import scipy.stats as stats
    
    mean = np.mean(values)
    std = np.std(values)
    n = len(values)
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