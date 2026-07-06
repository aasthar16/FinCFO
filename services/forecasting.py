"""
Prophet-based Forecasting Engine
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict
import logging
import warnings

from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics

from config.langsmith import traced, log_metric, log_assumption

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)


def convert_to_serializable(obj):
    """Convert numpy/pandas types to Python native types."""
    if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, (np.ndarray, pd.Series)):
        return obj.tolist()
    elif isinstance(obj, pd.Period):
        return str(obj)
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif isinstance(obj, datetime):
        return obj.isoformat()
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


@dataclass
class ForecastResult:
    date: datetime
    yhat: float
    yhat_lower: float
    yhat_upper: float
    trend: float
    seasonality: float
    uncertainty: float


@dataclass
class CashRunwayForecast:
    p10_date: datetime
    p50_date: datetime
    p90_date: datetime
    p10_days: int
    p50_days: int
    p90_days: int
    # model_accuracy: float
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
    """
    if len(df) < 4:
        logger.warning(f"Insufficient data for Prophet: {len(df)} points. Using fallback.")
        return _fallback_forecast(df, target_column, forecast_periods)
    
    prophet_df = df[['ds', target_column]].copy()
    prophet_df.columns = ['ds', 'y']
    prophet_df['y'] = prophet_df['y'].astype(float)
    
    n_months = len(prophet_df)
    has_seasonality = n_months >= 12
    
    model = Prophet(
        seasonality_mode=seasonality_mode,
        seasonality_prior_scale=seasonality_prior_scale if has_seasonality else 0.01,
        changepoint_prior_scale=changepoint_prior_scale,
        changepoint_range=0.8 if n_months > 6 else 0.6,
        yearly_seasonality=has_seasonality,
        weekly_seasonality=False,
        daily_seasonality=False,
    )
    
    if n_months >= 6:
        model.add_seasonality(name='monthly', period=30.5, fourier_order=3)
    
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=forecast_periods, freq='M')
    forecast = model.predict(future)
    
    results = []
    for _, row in forecast.iterrows():
        results.append(ForecastResult(
            date=row['ds'],
            yhat=float(row['yhat']),
            yhat_lower=float(row['yhat_lower']),
            yhat_upper=float(row['yhat_upper']),
            trend=float(row['trend']),
            seasonality=float(row.get('seasonality', 0)),
            uncertainty=float(row['yhat_upper'] - row['yhat_lower']),
        ))
    
    model_accuracy = None
    if n_months >= 6:
        try:
            cv_results = cross_validation(model, initial='180 days', period='30 days', horizon='90 days')
            metrics = performance_metrics(cv_results)
            model_accuracy = {
                'mape': float(metrics['mape'].mean()),
                'rmse': float(metrics['rmse'].mean()),
                'mae': float(metrics['mae'].mean()),
            }
        except Exception as e:
            logger.debug(f"Cross-validation failed: {e}")
    
    log_assumption(
        source="prophet_forecaster",
        parameter="seasonality_prior_scale",
        value=float(seasonality_prior_scale),
        rationale=f"Seasonality {'enabled' if has_seasonality else 'disabled'} due to {n_months} months of data",
        confidence=0.8 if n_months >= 12 else 0.5,
    )
    
    return {
        "results": [convert_to_serializable(r) for r in results],
        "model_accuracy": convert_to_serializable(model_accuracy),
        "seasonality_enabled": bool(has_seasonality),
        "n_data_points": int(n_months),
        "n_forecast_periods": int(forecast_periods),
    }


def _fallback_forecast(
    df: pd.DataFrame,
    target_column: str,
    forecast_periods: int,
) -> Dict[str, Any]:
    """Fallback linear trend forecast."""
    from scipy import stats
    
    df_clean = df.dropna()
    if len(df_clean) < 2:
        return {
            "results": [],
            "model_accuracy": None,
            "seasonality_enabled": False,
            "n_data_points": int(len(df_clean)),
            "n_forecast_periods": int(forecast_periods),
            "note": "Insufficient data for forecasting",
        }
    
    x = np.arange(len(df_clean))
    y = df_clean[target_column].values.astype(float)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    
    last_date = df_clean['ds'].max()
    results = []
    
    for i in range(1, forecast_periods + 1):
        pred_date = last_date + pd.DateOffset(months=i)
        y_pred = float(intercept + slope * (len(df_clean) + i - 1))
        std_dev = float(np.std(y - (intercept + slope * x)))
        ci_lower = float(y_pred - 1.96 * std_dev)
        ci_upper = float(y_pred + 1.96 * std_dev)
        
        results.append(ForecastResult(
            date=pred_date,
            yhat=y_pred,
            yhat_lower=ci_lower,
            yhat_upper=ci_upper,
            trend=float(slope),
            seasonality=0.0,
            uncertainty=ci_upper - ci_lower,
        ))
    
    return {
        "results": [convert_to_serializable(r) for r in results],
        "model_accuracy": {"r_squared": float(r_value**2)},
        "seasonality_enabled": False,
        "n_data_points": int(len(df_clean)),
        "n_forecast_periods": int(forecast_periods),
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
    Forecast cash runway using Monte Carlo simulation.
    """
    cash_balance = float(cash_balance)
    net_burn = float(net_burn)
    burn_volatility = float(burn_volatility)
    
    if net_burn <= 0:
        return CashRunwayForecast(
            p10_date=datetime.now() + timedelta(days=365*10),
            p50_date=datetime.now() + timedelta(days=365*10),
            p90_date=datetime.now() + timedelta(days=365*10),
            p10_days=3650,
            p50_days=3650,
            p90_days=3650,
            # model_accuracy=1.0,
            assumptions={"note": "Not burning cash"},
        )
    
    np.random.seed(42)
    all_runways = []
    
    for _ in range(monte_carlo_runs):
        cash = cash_balance
        month = 0
        while cash > 0 and month < forecast_months:
            monthly_burn = net_burn * (1 + np.random.normal(0, burn_volatility))
            cash -= monthly_burn
            month += 1
        all_runways.append(month)
    
    sorted_runways = sorted(all_runways)
    p10_idx = int(0.1 * len(sorted_runways))
    p50_idx = int(0.5 * len(sorted_runways))
    p90_idx = int(0.9 * len(sorted_runways))
    
    p10_months = int(sorted_runways[p10_idx])
    p50_months = int(sorted_runways[p50_idx])
    p90_months = int(sorted_runways[p90_idx])
    
    today = datetime.now()
    p10_date = today + timedelta(days=30 * p10_months)
    p50_date = today + timedelta(days=30 * p50_months)
    p90_date = today + timedelta(days=30 * p90_months)
    
    # model_accuracy = float(max(0.5, min(0.95, 1 - (burn_volatility * 2))))
    
    return CashRunwayForecast(
        p10_date=p10_date,
        p50_date=p50_date,
        p90_date=p90_date,
        p10_days=p10_months * 30,
        p50_days=p50_months * 30,
        p90_days=p90_months * 30,
        # model_accuracy=model_accuracy,
        assumptions=convert_to_serializable({
            "burn_volatility": burn_volatility,
            "monte_carlo_runs": monte_carlo_runs,
            "forecast_months": forecast_months,
        }),
    )


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
    if runway_forecast.p50_days < 180:
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
    if hasattr(metrics, 'burn_multiple') and metrics.burn_multiple > 2.0:
        recommendations.append({
            "priority": "MEDIUM",
            "category": "efficiency",
            "title": "Burn Multiple is High",
            "description": f"Current burn multiple of {metrics.burn_multiple:.1f}x is above 2.0x benchmark.",
            "suggested_actions": [
                "Review marketing spend efficiency",
                "Optimize sales processes",
                "Consider headcount adjustments",
            ],
            "impact_estimate": f"Reducing to 1.5x could save ${metrics.net_burn * 0.25:.0f}/month",
        })
    
    # Check one-time expenses
    if hasattr(metrics, 'one_time_expenses') and hasattr(metrics, 'recurring_expenses'):
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