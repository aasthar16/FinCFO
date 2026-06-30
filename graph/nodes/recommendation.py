"""
Recommendation node.
"""

from typing import Dict, Any
from datetime import datetime
from langchain_core.messages import AIMessage

from graph.state import GlobalState
from services.recommendations import generate_recommendations
from services.forecasting import forecast_cash_runway, CashRunwayForecast
from services.compute_burn import BurnMetrics
from config.langsmith import traced


@traced("recommendation_node", tags=["recommendation", "advice"])
def recommendation_node(state: GlobalState) -> Dict[str, Any]:
    """
    Recommendation Node: Generates actionable financial advice.
    """
    metrics = state.get("computed_metrics", {})
    forecast_results = state.get("forecast_results", {})
    runway_forecast_data = state.get("runway_forecast", {})
    
    # Reconstruct runway forecast
    try:
        runway_forecast = CashRunwayForecast(
            p10_date=datetime.fromisoformat(runway_forecast_data.get("p10_date", datetime.now().isoformat())),
            p50_date=datetime.fromisoformat(runway_forecast_data.get("p50_date", datetime.now().isoformat())),
            p90_date=datetime.fromisoformat(runway_forecast_data.get("p90_date", datetime.now().isoformat())),
            p10_days=runway_forecast_data.get("p10_days", 180),
            p50_days=runway_forecast_data.get("p50_days", 365),
            p90_days=runway_forecast_data.get("p90_days", 540),
            model_accuracy=0.8,
            assumptions={},
        )
    except Exception as e:
        cash_balance = metrics.get("cash_balance", 1000000)
        net_burn = metrics.get("net_burn_3m_avg", metrics.get("net_burn", 150000))
        runway_forecast = forecast_cash_runway(
            cash_balance=cash_balance,
            net_burn=net_burn,
        )
    
    # Generate recommendations
    try:
        burn_metrics = BurnMetrics(
            gross_burn=metrics.get("gross_burn", 0),
            net_burn=metrics.get("net_burn", 0),
            gross_burn_3m_avg=metrics.get("gross_burn_3m_avg", 0),
            net_burn_3m_avg=metrics.get("net_burn_3m_avg", 0),
            one_time_expenses=metrics.get("one_time_expenses", 0),
            recurring_expenses=metrics.get("recurring_expenses", 0),
            fully_loaded_ratio=metrics.get("fully_loaded_ratio", 1.3),
            cash_runway_months=metrics.get("cash_runway_months", 0),
            cash_balance=metrics.get("cash_balance", 0),
            monthly_revenue=metrics.get("monthly_revenue", 0),
            burn_multiple=metrics.get("burn_multiple", 0),
        )
        
        recommendations = generate_recommendations(
            burn_metrics={"metrics": burn_metrics},
            forecast_results=forecast_results,
            runway_forecast=runway_forecast,
        )
    except Exception as e:
        recommendations = []
    
    # Format recommendations
    rec_summary = "💡 **Recommendations**\n\n"
    
    if not recommendations:
        rec_summary += "✅ No critical issues detected. Continue monitoring.\n"
    else:
        for rec in recommendations:
            priority_emoji = {
                "HIGH": "🔴",
                "MEDIUM": "🟠",
                "LOW": "🟢"
            }.get(rec.get("priority", "LOW"), "⚪")
            
            rec_summary += f"**{priority_emoji} {rec['priority']} - {rec['title']}**\n"
            rec_summary += f"• {rec['description']}\n"
            rec_summary += f"• **Actions:** {', '.join(rec['suggested_actions'][:2])}\n"
            if rec.get('impact_estimate'):
                rec_summary += f"• **Impact:** {rec['impact_estimate']}\n"
            rec_summary += "\n"
    
    return {
        "recommendations": recommendations,
        "next_action": "end",
        "current_agent": "recommendation",  # Important: set current_agent
        "messages": [AIMessage(content=rec_summary)],
    }