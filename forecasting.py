"""
Prophet-based Forecasting Engine
Handles revenue, expense, and cash runway forecasting with confidence intervals.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, asdict
import logging
import warnings

from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics

from langsmith_config import traced, log_metric, log_assumption

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)


@dataclass
class ForecastResult:
    """Forecast results with confidence intervals."""
    date: datetime
    yhat: float  # Point forecast
    yhat_lower: float  # Lower bound
    yhat_upper: float  # Upper bound
    trend: float
    seasonality: float
    uncertainty: float


@dataclass
class CashRunwayForecast:
    """Cash runway forecast with probability distribution."""
    p10_date: datetime  # 10% probability of running out by this date
    p50_date: datetime  # 50% probability (median)
    p90_date: datetime  # 90% probability of running out by this date
    p10_days: int
    p50_days: int
    p90_days: int
    model_accuracy: float  # MAPE or similar
    assumptions: Dict[str, Any]


@traced("forecast_prophet", tags=["prophet", "forecasting"])
def forecast_with_prophet(
    df: pd.DataFrame,
    target_column: str,
    forecast_periods: int = 12,
    seasonality_mode: str = "additive",
    seasonality_prior_scale: float = 10.0,
    changepoint_prior_scale: float = 0.05,
) -> Dict[str, Any]:
    """
    Forecast time series using Prophet with confidence intervals.
    
    Args:
        df: DataFrame with 'ds' (date) and target_column
        target_column: Column name for the target variable
        forecast_periods: Number of periods to forecast
        seasonality_mode: 'additive' or 'multiplicative'
        seasonality_prior_scale: Strength of seasonality
        changepoint_prior_scale: Flexibility of trend changes
    
    Returns:
        Dict with forecast results and model diagnostics
    """
    # Validate data
    if len(df) < 4:
        logger.warning(f"Insufficient data for Prophet: {len(df)} points. Using fallback.")
        return _fallback_forecast(df, target_column, forecast_periods)
    
    # Prepare data for Prophet
    prophet_df = df[['ds', target_column]].copy()
    prophet_df.columns = ['ds', 'y']
    
    # Check for seasonality viability
    n_months = len(prophet_df)
    has_seasonality = n_months >= 12
    
    # Configure Prophet
    model = Prophet(
        seasonality_mode=seasonality_mode,
        seasonality_prior_scale=seasonality_prior_scale if has_seasonality else 0.01,
        changepoint_prior_scale=changepoint_prior_scale,
        changepoint_range=0.8 if n_months > 6 else 0.6,
        yearly_seasonality=has_seasonality,
        weekly_seasonality=False,
        daily_seasonality=False,
    )
    
    # Add custom seasonalities if data allows
    if n_months >= 6:
        model.add_seasonality(name='monthly', period=30.5, fourier_order=3)
    
    # Fit model
    model.fit(prophet_df)
    
    # Create future dataframe
    future = model.make_future_dataframe(periods=forecast_periods, freq='M')
    forecast = model.predict(future)
    
    # Extract forecast results
    results = []
    for _, row in forecast.iterrows():
        results.append(ForecastResult(
            date=row['ds'],
            yhat=row['yhat'],
            yhat_lower=row['yhat_lower'],
            yhat_upper=row['yhat_upper'],
            trend=row['trend'],
            seasonality=row.get('seasonality', 0),
            uncertainty=row['yhat_upper'] - row['yhat_lower'],
        ))
    
    # Calculate model accuracy if enough data
    model_accuracy = None
    if n_months >= 6:
        try:
            cv_results = cross_validation(model, initial='180 days', period='30 days', horizon='90 days')
            metrics = performance_metrics(cv_results)
            model_accuracy = {
                'mape': metrics['mape'].mean(),
                'rmse': metrics['rmse'].mean(),
                'mae': metrics['mae'].mean(),
            }
        except Exception as e:
            logger.debug(f"Cross-validation failed: {e}")
    
    # Log assumptions
    log_assumption(
        source="prophet_forecaster",
        parameter="seasonality_prior_scale",
        value=seasonality_prior_scale,
        rationale=f"Seasonality {'enabled' if has_seasonality else 'disabled'} due to {n_months} months of data",
        confidence=0.8 if n_months >= 12 else 0.5,
    )
    
    return {
        "results": results,
        "model": model,
        "forecast_df": forecast,
        "model_accuracy": model_accuracy,
        "seasonality_enabled": has_seasonality,
        "n_data_points": n_months,
        "n_forecast_periods": forecast_periods,
    }


def _fallback_forecast(
    df: pd.DataFrame,
    target_column: str,
    forecast_periods: int,
) -> Dict[str, Any]:
    """
    Fallback linear trend forecast when Prophet cannot be used.
    """
    from scipy import stats
    
    # Prepare data
    df_clean = df.dropna()
    if len(df_clean) < 2:
        return {
            "results": [],
            "model": None,
            "forecast_df": None,
            "model_accuracy": None,
            "seasonality_enabled": False,
            "n_data_points": len(df_clean),
            "n_forecast_periods": forecast_periods,
            "note": "Insufficient data for forecasting",
        }
    
    # Linear regression
    x = np.arange(len(df_clean))
    y = df_clean[target_column].values
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    
    # Forecast
    last_date = df_clean['ds'].max()
    results = []
    
    for i in range(1, forecast_periods + 1):
        pred_date = last_date + pd.DateOffset(months=i)
        y_pred = intercept + slope * (len(df_clean) + i - 1)
        
        # Confidence interval (simplified)
        std_dev = np.std(y - (intercept + slope * x))
        ci_lower = y_pred - 1.96 * std_dev
        ci_upper = y_pred + 1.96 * std_dev
        
        results.append(ForecastResult(
            date=pred_date,
            yhat=y_pred,
            yhat_lower=ci_lower,
            yhat_upper=ci_upper,
            trend=slope,
            seasonality=0,
            uncertainty=ci_upper - ci_lower,
        ))
    
    log_assumption(
        source="fallback_forecaster",
        parameter="forecast_method",
        value="linear_trend",
        rationale=f"Insufficient data ({len(df_clean)} points) for Prophet",
        confidence=0.6,
    )
    
    return {
        "results": results,
        "model": None,
        "forecast_df": None,
        "model_accuracy": {"r_squared": r_value**2},
        "seasonality_enabled": False,
        "n_data_points": len(df_clean),
        "n_forecast_periods": forecast_periods,
        "note": "Using linear trend fallback",
    }


@traced("cash_runway_forecast", tags=["runway", "forecast"])
def forecast_cash_runway(
    cash_balance: float,
    net_burn: float,
    burn_volatility: float = 0.15,
    forecast_months: int = 24,
    monte_carlo_runs: int = 1000,
) -> CashRunwayForecast:
    """
    Forecast cash runway using Monte Carlo simulation with uncertainty.
    
    Args:
        cash_balance: Current cash balance
        net_burn: Monthly net burn (positive = spending > revenue)
        burn_volatility: Standard deviation of monthly burn
        forecast_months: Number of months to forecast
        monte_carlo_runs: Number of simulation runs
    
    Returns:
        CashRunwayForecast with P10/P50/P90 dates
    """
    if net_burn <= 0:
        return CashRunwayForecast(
            p10_date=datetime.now() + timedelta(days=365*10),
            p50_date=datetime.now() + timedelta(days=365*10),
            p90_date=datetime.now() + timedelta(days=365*10),
            p10_days=3650,
            p50_days=3650,
            p90_days=3650,
            model_accuracy=1.0,
            assumptions={"note": "Not burning cash"},
        )
    
    # Monte Carlo simulation
    np.random.seed(42)
    all_runways = []
    
    for _ in range(monte_carlo_runs):
        cash = cash_balance
        month = 0
        while cash > 0 and month < forecast_months:
            # Random burn with volatility
            monthly_burn = net_burn * (1 + np.random.normal(0, burn_volatility))
            cash -= monthly_burn
            month += 1
        all_runways.append(month)
    
    # Calculate percentiles
    sorted_runways = sorted(all_runways)
    p10_idx = int(0.1 * len(sorted_runways))
    p50_idx = int(0.5 * len(sorted_runways))
    p90_idx = int(0.9 * len(sorted_runways))
    
    p10_months = sorted_runways[p10_idx]
    p50_months = sorted_runways[p50_idx]
    p90_months = sorted_runways[p90_idx]
    
    # Convert to dates
    today = datetime.now()
    p10_date = today + timedelta(days=30 * p10_months)
    p50_date = today + timedelta(days=30 * p50_months)
    p90_date = today + timedelta(days=30 * p90_months)
    
    # Model accuracy (based on volatility)
    model_accuracy = 1 - (burn_volatility * 2)  # Simple heuristic
    model_accuracy = max(0.5, min(0.95, model_accuracy))
    
    # Log assumptions
    log_assumption(
        source="runway_forecaster",
        parameter="burn_volatility",
        value=burn_volatility,
        rationale="Historical volatility of burn rate",
        confidence=0.7,
    )
    
    return CashRunwayForecast(
        p10_date=p10_date,
        p50_date=p50_date,
        p90_date=p90_date,
        p10_days=p10_months * 30,
        p50_days=p50_months * 30,
        p90_days=p90_months * 30,
        model_accuracy=model_accuracy,
        assumptions={
            "burn_volatility": burn_volatility,
            "monte_carlo_runs": monte_carlo_runs,
            "forecast_months": forecast_months,
        },
    )


@traced("recommendation_engine", tags=["recommendation", "analysis"])
def generate_recommendations(
    burn_metrics: Dict[str, Any],
    forecast_results: Dict[str, Any],
    runway_forecast: CashRunwayForecast,
) -> List[Dict[str, Any]]:
    """
    Generate actionable recommendations based on forecasts.
    """
    recommendations = []
    
    metrics = burn_metrics.get("metrics")
    if not metrics:
        return recommendations
    
    # Check cash runway
    if runway_forecast.p50_days < 180:  # Less than 6 months
        recommendations.append({
            "priority": "HIGH",
            "category": "cash_management",
            "title": "Critical Cash Conservation Needed",
            "description": f"Runway of {runway_forecast.p50_days//30} months (P50). Immediate action required.",
            "suggested_actions": [
                "Review non-essential spending",
                "Accelerate revenue collection",
                "Consider bridge financing",
            ],
            "impact_estimate": f"Could extend runway by 2-4 months with aggressive measures",
        })
    
    # Check burn multiple
    burn_multiple = metrics.burn_multiple
    if burn_multiple > 2.0:
        recommendations.append({
            "priority": "MEDIUM",
            "category": "efficiency",
            "title": "Burn Multiple is High",
            "description": f"Current burn multiple of {burn_multiple:.1f}x is above 2.0x benchmark.",
            "suggested_actions": [
                "Review marketing spend efficiency",
                "Optimize sales processes",
                "Consider headcount adjustments",
            ],
            "impact_estimate": f"Reducing to 1.5x could save ${metrics.net_burn * 0.25:.0f}/month",
        })
    
    # Check revenue growth
    if forecast_results.get("results"):
        trend = forecast_results["results"][-1].trend
        if trend < 0:
            recommendations.append({
                "priority": "HIGH",
                "category": "revenue",
                "title": "Revenue Trend is Negative",
                "description": f"Revenue is declining at ${abs(trend):.0f}/month.",
                "suggested_actions": [
                    "Investigate customer churn",
                    "Review pricing strategy",
                    "Increase sales activity",
                ],
                "impact_estimate": "Stabilizing revenue could add 3-6 months of runway",
            })
    
    # Check one-time expenses
    if metrics.one_time_expenses > metrics.recurring_expenses * 0.5:
        recommendations.append({
            "priority": "LOW",
            "category": "expense_management",
            "title": "High One-Time Expenses",
            "description": f"One-time expenses of ${metrics.one_time_expenses:,.0f} represent {metrics.one_time_expenses/metrics.recurring_expenses*100:.0f}% of recurring.",
            "suggested_actions": [
                "Review vendor contracts",
                "Capitalize eligible expenses",
                "Plan for seasonal spikes",
            ],
            "impact_estimate": "Could reduce burn by 10-15% if managed better",
        })
    
    return recommendations