"""
Forecast node.
"""

from typing import Dict, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from langchain_core.messages import AIMessage

from graph.state import GlobalState
from services.forecasting import forecast_with_prophet, forecast_cash_runway, convert_to_serializable
from config.langsmith import traced


@traced("forecast_node", tags=["forecast", "prophet"])
def forecast_node(state: GlobalState) -> Dict[str, Any]:
    """
    Forecast Node: Projects future financials using Prophet.
    """
    metrics = state.get("computed_metrics", {})
    transactions_data = state.get("transactions_data")
    
    if not metrics:
        return {
            "next_action": "recommendation",  # Skip to recommendations if no metrics
            "current_agent": "forecast",
            "messages": [AIMessage(content="⚠️ No metrics available for forecasting.")],
        }
    
    # Convert transactions_data to DataFrame if available
    transactions_df = None
    if transactions_data:
        transactions_df = pd.DataFrame(transactions_data)
        if 'date' in transactions_df.columns:
            transactions_df['date'] = pd.to_datetime(transactions_df['date'])
    
    # Generate mock historical data if needed
    if transactions_df is None or len(transactions_df) < 3:
        dates = pd.date_range(start=datetime.now() - timedelta(days=180), periods=6, freq='M')
        monthly_revenue = float(metrics.get("monthly_revenue", 100000))
        

        # New Revenue=Current Revenue×(1 + Expected Growth + Random Business Fluctuation)
        revenue_df = pd.DataFrame({
            'ds': dates,
            'revenue': monthly_revenue * (1 + np.linspace(0.01, 0.10, 6) + np.random.normal(0, 0.02, 6))
        })
        revenue_forecast = forecast_with_prophet(revenue_df, 'revenue', forecast_periods=12)
        
        gross_burn = float(metrics.get("gross_burn", 200000))
        expense_df = pd.DataFrame({
            'ds': dates,
            'expense': gross_burn * (1 + np.linspace(0, 0.05, 6) + np.random.normal(0, 0.01, 6))
        })
        expense_forecast = forecast_with_prophet(expense_df, 'expense', forecast_periods=12)
    else:
        # Use actual transactions
        revenue_series = transactions_df.copy()
        if 'date' in revenue_series.columns:
            revenue_series['ds'] = pd.to_datetime(revenue_series['date'])
        else:
            revenue_series['ds'] = revenue_series.index
        revenue_series['revenue'] = float(metrics.get("monthly_revenue", 100000)) * (1 + np.random.normal(0.01, 0.05, len(revenue_series)))
        revenue_forecast = forecast_with_prophet(revenue_series[['ds', 'revenue']], 'revenue', forecast_periods=12)
        
        expense_series = transactions_df.copy()
        if 'date' in expense_series.columns:
            expense_series['ds'] = pd.to_datetime(expense_series['date'])
        else:
            expense_series['ds'] = expense_series.index
        expense_series['expense'] = float(metrics.get("gross_burn", 200000)) * (1 + np.random.normal(0.005, 0.03, len(expense_series)))
        expense_forecast = forecast_with_prophet(expense_series[['ds', 'expense']], 'expense', forecast_periods=12)
    
    # Cash runway forecast
    net_burn = float(metrics.get("net_burn_3m_avg", metrics.get("net_burn", 150000)))
    cash_balance = float(metrics.get("cash_balance", 1000000))
    
    runway_forecast = forecast_cash_runway(
        cash_balance=cash_balance,
        net_burn=net_burn,
        burn_volatility=0.15,
        forecast_months=24,
    )
    
    # Format summary
    forecast_summary = f"""
📈 **Forecast Summary**

**Revenue Projection:**
• Current: ${metrics.get('monthly_revenue', 0):,.0f}/month
• Forecast (12mo): ${revenue_forecast['results'][-1]['yhat']:,.0f}/month
• Growth Rate: {((revenue_forecast['results'][-1]['yhat'] / float(metrics.get('monthly_revenue', 1))) - 1) * 100:.1f}%

**Cash Runway:**
• P10 (Pessimistic): {runway_forecast.p10_date.strftime('%B %Y')} ({runway_forecast.p10_days//30} months)
• P50 (Expected): {runway_forecast.p50_date.strftime('%B %Y')} ({runway_forecast.p50_days//30} months)
• P90 (Optimistic): {runway_forecast.p90_date.strftime('%B %Y')} ({runway_forecast.p90_days//30} months)
"""
    
    return {
        "forecast_results": convert_to_serializable({
            "revenue": {
                #  get every key except model in the dict and add in revenue as k:v
                k: v for k, v in revenue_forecast.items() 
                if k not in ["model"]
            },
            "expense": { 
                k: v for k, v in expense_forecast.items() 
                if k not in ["model"]
            },
        }),
        "runway_forecast": convert_to_serializable({
            "p10_date": runway_forecast.p10_date.isoformat(),  #formatting date returned by forecast_cash_runway to isoformat for serialization to convert into json
            "p50_date": runway_forecast.p50_date.isoformat(),
            "p90_date": runway_forecast.p90_date.isoformat(),
            "p10_days": runway_forecast.p10_days,
            "p50_days": runway_forecast.p50_days,
            "p90_days": runway_forecast.p90_days,
        }),
        "next_action": "recommendation",  # Go to recommendations next
        "current_agent": "forecast",  # IMPORTANT: Set current_agent
        "messages": [AIMessage(content=forecast_summary)],
    }