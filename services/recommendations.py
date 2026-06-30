"""
Recommendation engine.
"""

from typing import Dict, Any, List
from services.compute_burn import BurnMetrics
from services.forecasting import CashRunwayForecast
from config.langsmith import traced


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
            "impact_estimate": "Could extend runway by 2-4 months with aggressive measures",
        })
    elif runway_forecast.p50_days < 365:  # Less than 12 months
        recommendations.append({
            "priority": "MEDIUM",
            "category": "cash_management",
            "title": "Monitor Cash Runway Closely",
            "description": f"Runway of {runway_forecast.p50_days//30} months (P50). Consider proactive measures.",
            "suggested_actions": [
                "Review monthly spending",
                "Optimize cash conversion cycle",
                "Prepare for next funding round",
            ],
            "impact_estimate": "Could extend runway by 3-6 months with proactive management",
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
    
    # Check revenue growth (if forecast results available)
    if forecast_results.get("revenue", {}).get("results"):
        results = forecast_results["revenue"]["results"]
        if len(results) > 0:
            trend = results[-1].get("trend", 0) if isinstance(results[-1], dict) else getattr(results[-1], "trend", 0)
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
    
    # Check if burn rate is high relative to revenue
    if hasattr(metrics, 'net_burn') and hasattr(metrics, 'monthly_revenue'):
        if metrics.net_burn > metrics.monthly_revenue:
            recommendations.append({
                "priority": "MEDIUM",
                "category": "financial_health",
                "title": "Burn Exceeds Revenue",
                "description": f"Net burn of ${metrics.net_burn:,.0f}/month exceeds monthly revenue of ${metrics.monthly_revenue:,.0f}.",
                "suggested_actions": [
                    "Focus on revenue growth",
                    "Reduce discretionary spending",
                    "Improve gross margins",
                ],
                "impact_estimate": "Reducing burn to revenue levels could add 6+ months of runway",
            })
    
    return recommendations