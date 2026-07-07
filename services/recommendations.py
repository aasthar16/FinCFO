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
    PRESERVES original deterministic titles/descriptions, only adds enrichment fields.
    """
    if not recommendations:
        return []
   
    startup_profile = state.get("startup_profile", {})
    
    # Safely extract metrics
    cash_balance = getattr(metrics, 'cash_balance', 0) if hasattr(metrics, 'cash_balance') else 0
    runway_months = getattr(metrics, 'cash_runway_months', 0) if hasattr(metrics, 'cash_runway_months') else 0
    net_burn = getattr(metrics, 'net_burn', 0) if hasattr(metrics, 'net_burn') else 0
    monthly_revenue = getattr(metrics, 'monthly_revenue', 0) if hasattr(metrics, 'monthly_revenue') else 0
    
    context = {
        "startup_stage": startup_profile.get("stage", "Seed"),
        "currency": startup_profile.get("currency", "USD"),
        "industry": startup_profile.get("industry", "Not specified"),
        "cash_balance": cash_balance,
        "runway_months": runway_months,
        "net_burn": net_burn,
        "monthly_revenue": monthly_revenue,
    }
    
    # Build the prompt - EXPLICITLY tell LLM to keep original content
    enrichment_prompt = f"""
You are a senior financial advisor for startups. Add enrichment context to these EXISTING recommendations. DO NOT change the titles or descriptions.

**EXISTING RECOMMENDATIONS (DO NOT MODIFY THESE FIELDS - title, description, suggested_actions, impact_estimate):**
{json.dumps(recommendations, indent=2)}

**ACTUAL FINANCIAL DATA (USE THESE EXACT NUMBERS):**
- Cash Balance: {context['currency']} {context['cash_balance']:,.0f}
- Net Burn: {context['currency']} {context['net_burn']:,.0f}/month
- Monthly Revenue: {context['currency']} {context['monthly_revenue']:,.0f}/month
- Runway: {context['runway_months']:.1f} months
- Stage: {context['startup_stage']}
- Industry: {context['industry']}

**YOUR TASK:**
For EACH recommendation, add ONLY these enrichment fields (do NOT modify title/description/actions/impact):
1. **contextual_insight**: Why this matters for a {context['startup_stage']}-stage {context['industry']} company with ${context['cash_balance']:,.0f} cash
2. **priority_justification**: Why this is the right priority given {context['runway_months']:.1f} months runway
3. **industry_best_practice**: What successful {context['startup_stage']}-stage startups do
4. **expected_roi**: Estimated impact with actual numbers

Also provide:
- **overall_assessment**: 1-2 sentence health summary using the ACTUAL numbers above
- **key_priority**: Which single action matters most right now

CRITICAL: Keep ALL original fields (title, description, suggested_actions, impact_estimate, priority, category) EXACTLY as provided. Only ADD the enrichment fields.
"""
    
    try:
        # Call LLM with structured output
        result: EnrichedRecommendationsResponse = enrichment_llm.invoke([
            SystemMessage(content=enrichment_prompt),
            HumanMessage(content="Add enrichment context to these recommendations. Keep all original content."),
        ])
        
        enriched_recs = [rec.model_dump() for rec in result.recommendations]
        
        # MERGE enrichment fields into original recommendations (preserve originals!)
        merged_recs = []
        for i, orig_rec in enumerate(recommendations):
            merged = dict(orig_rec)  # Start with original deterministic content
            
            # If LLM returned a corresponding enriched version, add its enrichment fields
            if i < len(enriched_recs):
                enriched = enriched_recs[i]
                for field in ['contextual_insight', 'priority_justification', 'industry_best_practice', 'expected_roi']:
                    if enriched.get(field):
                        merged[field] = enriched[field]
            
            # Add overall context to first recommendation
            if i == 0:
                if result.overall_assessment:
                    merged["overall_assessment"] = result.overall_assessment
                if result.key_priority:
                    merged["key_priority"] = result.key_priority
            
            merged_recs.append(merged)
        
        logger.info(f"✅ LLM enriched {len(merged_recs)} recommendations (original content preserved)")
        return merged_recs
        
    except Exception as e:
        logger.warning(f"LLM enrichment failed: {e}, using deterministic")
        return recommendations  