"""
Unit tests for forecasting module.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from forecasting import (
    forecast_with_prophet,
    forecast_cash_runway,
    generate_recommendations,
)


@pytest.fixture
def sample_time_series():
    """Create sample time series data."""
    dates = pd.date_range(start='2024-01-01', periods=12, freq='M')
    values = 100000 * (1 + np.linspace(0.01, 0.10, 12)) + np.random.normal(0, 1000, 12)
    
    return pd.DataFrame({
        'ds': dates,
        'revenue': values,
    })


def test_forecast_with_prophet(sample_time_series):
    """Test Prophet forecasting with adequate data."""
    result = forecast_with_prophet(
        df=sample_time_series,
        target_column='revenue',
        forecast_periods=6,
    )
    
    assert "results" in result
    assert len(result["results"]) == 6
    assert result["seasonality_enabled"] == True
    assert result["n_data_points"] == 12


def test_forecast_fallback():
    """Test fallback forecasting with insufficient data."""
    dates = pd.date_range(start='2024-01-01', periods=3, freq='M')
    values = [100000, 105000, 110000]
    
    df = pd.DataFrame({'ds': dates, 'revenue': values})
    
    result = forecast_with_prophet(
        df=df,
        target_column='revenue',
        forecast_periods=6,
    )
    
    assert "note" in result
    assert result["note"] == "Using linear trend fallback"


def test_cash_runway_forecast():
    """Test cash runway forecasting."""
    result = forecast_cash_runway(
        cash_balance=1000000,
        net_burn=150000,
        burn_volatility=0.15,
        forecast_months=24,
    )
    
    assert result.p10_days <= result.p50_days <= result.p90_days
    assert result.model_accuracy > 0


def test_recommendation_generation():
    """Test recommendation generation."""
    burn_metrics = {
        "metrics": {
            "gross_burn": 200000,
            "net_burn": 150000,
            "cash_runway_months": 5,
            "burn_multiple": 2.5,
            "one_time_expenses": 50000,
            "recurring_expenses": 150000,
            "monthly_revenue": 50000,
            "cash_balance": 750000,
        }
    }
    
    forecast_results = {"results": [{"trend": -5000}]}
    
    runway_forecast = forecast_cash_runway(1000000, 150000)
    
    recommendations = generate_recommendations(
        burn_metrics=burn_metrics,
        forecast_results=forecast_results,
        runway_forecast=runway_forecast,
    )
    
    # Should generate recommendations for short runway
    assert len(recommendations) > 0
    assert any(rec["priority"] == "HIGH" for rec in recommendations)