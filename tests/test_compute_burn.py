"""
Unit tests for burn computation.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from compute_burn import compute_burn, compute_scenario_impact


@pytest.fixture
def sample_transactions():
    """Create sample transaction data."""
    dates = pd.date_range(start='2024-01-01', periods=6, freq='M')
    
    data = []
    for date in dates:
        # Regular expenses
        data.append({
            'date': date,
            'amount': -150000,
            'type': 'expense',
            'category': 'salary',
            'one_time': False,
        })
        data.append({
            'date': date,
            'amount': -30000,
            'type': 'expense',
            'category': 'rent',
            'one_time': False,
        })
        data.append({
            'date': date,
            'amount': -15000,
            'type': 'expense',
            'category': 'software',
            'one_time': False,
        })
        
        # One-time expense every 3 months
        if (date.month % 3) == 0:
            data.append({
                'date': date,
                'amount': -50000,
                'type': 'expense',
                'category': 'consulting',
                'one_time': True,
            })
    
    return pd.DataFrame(data)


def test_compute_burn_basic(sample_transactions):
    """Test basic burn computation."""
    result = compute_burn(
        transactions_df=sample_transactions,
        cash_balance=1000000,
        monthly_revenue=85000,
    )
    
    metrics = result["metrics"]
    
    assert metrics.gross_burn > 0
    assert metrics.net_burn > 0
    assert metrics.cash_runway_months > 0
    assert metrics.gross_burn_3m_avg > 0
    assert metrics.one_time_expenses > 0
    assert metrics.fully_loaded_ratio >= 1.0


def test_compute_burn_with_scenario(sample_transactions):
    """Test burn computation with scenario overrides."""
    scenario = {
        "headcount_change": 2,
        "avg_salary": 140000,
        "ramp_months": 3,
    }
    
    result = compute_burn(
        transactions_df=sample_transactions,
        cash_balance=1000000,
        monthly_revenue=85000,
        scenario_overrides=scenario,
    )
    
    metrics = result["metrics"]
    
    # Gross burn should increase with headcount
    assert metrics.gross_burn > 0


def test_compute_scenario_impact(sample_transactions):
    """Test scenario impact computation."""
    base_result = compute_burn(
        transactions_df=sample_transactions,
        cash_balance=1000000,
        monthly_revenue=85000,
    )
    
    scenario = {
        "headcount_change": 3,
        "avg_salary": 150000,
        "ramp_months": 3,
        "revenue_change": 50000,
    }
    
    impact = compute_scenario_impact(base_result, scenario)
    
    assert impact.incremental_burn > 0
    assert impact.incremental_revenue > 0
    assert impact.new_cash_runway > 0
    assert impact.confidence_interval[0] < impact.confidence_interval[1]


def test_burn_edge_cases():
    """Test edge cases in burn computation."""
    # Empty DataFrame
    empty_df = pd.DataFrame(columns=['date', 'amount', 'type', 'category'])
    result = compute_burn(
        transactions_df=empty_df,
        cash_balance=500000,
        monthly_revenue=50000,
    )
    assert result["metrics"].gross_burn == 0
    
    # Zero cash balance
    result = compute_burn(
        transactions_df=empty_df,
        cash_balance=0,
        monthly_revenue=50000,
    )
    assert result["metrics"].cash_runway_months == float('inf')


def test_trailing_average(sample_transactions):
    """Test trailing average calculation."""
    result = compute_burn(
        transactions_df=sample_transactions,
        cash_balance=1000000,
        monthly_revenue=85000,
    )
    
    metrics = result["metrics"]
    
    # 3-month average should be reasonable
    assert metrics.gross_burn_3m_avg > 0
    assert abs(metrics.gross_burn_3m_avg - metrics.gross_burn) < metrics.gross_burn * 0.5