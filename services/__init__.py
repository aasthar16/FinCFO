"""
Services module - business logic.
"""

from services.compute_burn import compute_burn, BurnMetrics, convert_to_serializable
from services.forecasting import (
    forecast_with_prophet, 
    forecast_cash_runway, 
    CashRunwayForecast,
    generate_recommendations
)

__all__ = [
    'compute_burn',
    'BurnMetrics',
    'convert_to_serializable',
    'forecast_with_prophet',
    'forecast_cash_runway',
    'CashRunwayForecast',
    'generate_recommendations',
]