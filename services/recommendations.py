"""
Recommendation engine - HYBRID: Deterministic rules + LLM Enrichment.
Uses Pydantic structured output for robust parsing.
"""

import json
import logging
from typing import Dict, Any, List, Optional
from services.compute_burn import BurnMetrics
from services.forecasting import CashRunwayForecast
from config.langsmith import traced
from services.schemas import EnrichedRecommendation, EnrichedRecommendationsResponse
from services.llm_service import call_llm
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
logger = logging.getLogger(__name__)


# Initialize LLM with structured output
from langchain_groq import ChatGroq
from settings import settings

llm = ChatGroq(
    model=settings.groq_model,
    api_key=settings.groq_api_key,
    temperature=0.0,
    max_tokens=500,
)

# Create structured LLM for enrichment
enrichment_llm = llm.with_structured_output(EnrichedRecommendationsResponse)


@traced("recommendation_engine", tags=["recommendation", "analysis"])
def generate_recommendations(
    burn_metrics: Dict[str, Any],
    forecast_results: Dict[str, Any],
    runway_forecast: CashRunwayForecast,
    state: Optional[Dict[str, Any]] = None,
    enable_llm_enrichment: bool = True,
) -> List[Dict[str, Any]]:
    """
    Generate actionable recommendations - HYBRID approach.
    
    Phase 1: Deterministic rules (always runs)
    Phase 2: LLM enrichment with Pydantic structured output
    """
    recommendations = []
    
    metrics = burn_metrics.get("metrics")
    if not metrics:
        return recommendations
    
    # ============================================================
    # PHASE 1: DETERMINISTIC RULES
    # ============================================================
    
    # Check cash runway
    if runway_forecast and runway_forecast.p50_days < 180:
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
            "source": "deterministic"
        })
    elif runway_forecast and runway_forecast.p50_days < 365:
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
            "source": "deterministic"
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
            "source": "deterministic"
        })
    
    # Check revenue growth
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
                    "source": "deterministic"
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
                "source": "deterministic"
            })
    
    # Check burn vs revenue
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
                "source": "deterministic"
            })
    
    # If no deterministic recommendations, add default healthy one
    if not recommendations:
        recommendations.append({
            "priority": "LOW",
            "category": "general",
            "title": "Financial Health Looks Stable",
            "description": "No critical issues detected. Continue monitoring key metrics.",
            "suggested_actions": [
                "Continue tracking burn rate monthly",
                "Maintain cash reserves",
                "Monitor industry benchmarks",
            ],
            "impact_estimate": "Maintaining current course keeps runway stable",
            "source": "deterministic"
        })
    
    # ============================================================
    # PHASE 2: LLM ENRICHMENT with Pydantic Structured Output
    # ============================================================
    
    if enable_llm_enrichment and state:
        enriched = _enrich_recommendations_with_llm_structured(
            recommendations=recommendations,
            state=state,
            metrics=metrics,
        )
        return enriched if enriched else recommendations
    
    return recommendations


def _enrich_recommendations_with_llm_structured(
    recommendations: List[Dict[str, Any]],
    state: Dict[str, Any],
    metrics: Any,
) -> List[Dict[str, Any]]:
    """
    Enrich recommendations using LLM with Pydantic structured output.
    """
    if not recommendations:
        return []
    
    # Build context for LLM
    startup_profile = state.get("startup_profile", {})
    context = {
        "startup_stage": startup_profile.get("stage", "Seed"),
        "currency": startup_profile.get("currency", "USD"),
        "industry": startup_profile.get("industry", "Not specified"),
        "cash_balance": metrics.cash_balance if hasattr(metrics, 'cash_balance') else 0,
        "runway_months": metrics.cash_runway_months if hasattr(metrics, 'cash_runway_months') else 0,
    }
    
    # Build the prompt
    enrichment_prompt = f"""
You are a senior financial advisor for startups. Enrich these deterministic recommendations.

**DETERMINISTIC RECOMMENDATIONS (FOUNDATION):**
{json.dumps(recommendations, indent=2)}

**STARTUP CONTEXT:**
- Stage: {context['startup_stage']}
- Industry: {context['industry']}
- Cash Balance: {context['currency']} {context['cash_balance']:,.0f}
- Runway: {context['runway_months']:.1f} months

**ENRICHMENT GUIDELINES:**
For each recommendation, add/enhance:
1. **contextual_insight**: Specific to their startup stage
2. **priority_justification**: Why this matters NOW
3. **industry_best_practice**: Relevant benchmark
4. **expected_roi**: Estimated ROI or timeline

Also provide:
- **overall_assessment**: Brief financial health summary
- **key_priority**: The single most important action

Return ONLY the enriched recommendations with the exact structure matching the schema.
"""
    
    try:
        # Call LLM with structured output - RETURNS Pydantic object directly!
        result: EnrichedRecommendationsResponse = enrichment_llm.invoke([
            SystemMessage(content=enrichment_prompt),
            HumanMessage(content="Enrich these recommendations with context"),
        ])
        
        # Convert Pydantic objects to dicts
        enriched_recs = [rec.model_dump() for rec in result.recommendations]
        
        # Add overall assessment and key priority to first recommendation's metadata
        if enriched_recs and result.overall_assessment:
            enriched_recs[0]["overall_assessment"] = result.overall_assessment
        if enriched_recs and result.key_priority:
            enriched_recs[0]["key_priority"] = result.key_priority
        
        logger.info(f"✅ LLM enriched {len(enriched_recs)} recommendations with Pydantic")
        return enriched_recs
        
    except Exception as e:
        logger.warning(f"LLM enrichment failed: {e}, using deterministic")
        return recommendations