"""
Tool definitions for the AI CFO Agent.
Lightweight tools — burn calculation moved to burn_calculator node.
"""

import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from services.forecasting import forecast_cash_runway
from services.schemas import ScenarioExtractionResult
from settings import settings

logger = logging.getLogger(__name__)

# Groq for scenario extraction
llm = ChatGroq(
    model=settings.groq_model,
    api_key=settings.groq_api_key,
    temperature=0.0,
    max_tokens=500,
)
scenario_llm = llm.with_structured_output(ScenarioExtractionResult)


def _get_state() -> dict:
    """Get state from Streamlit session."""
    import streamlit as st
    return st.session_state.get("state", {})


# ================================================================
# TOOL 1: Model Scenario
# ================================================================

@tool
def model_scenario(user_query: str) -> Dict[str, Any]:
    """
    Extract hiring/firing/revenue-change parameters from user query.
    Call FIRST when user describes a scenario change.
    """
    state = _get_state()
    
    previous_context = {
        "total_headcount": state.get("scenario_overrides", {}).get("headcount_change", 0),
        "avg_salary": state.get("scenario_overrides", {}).get("avg_salary"),
        "active_scenario": state.get("active_scenario"),
    }
    
    context_str = ""
    if previous_context.get("total_headcount"):
        context_str = f"""
Previous scenario: headcount_change={previous_context['total_headcount']}, 
avg_salary=${previous_context.get('avg_salary', 'N/A')}
"""
    
    system_prompt = f"""Extract scenario parameters from user query.

RULES:
- "hire 2 engineers at $8000/month" → action:hire, count:2, role:engineer, salary:8000
- "fire 2 salespeople" → action:fire, count:2, role:salesperson
- "what if revenue grows 20%" → action:revenue_change
- "add 3 more at same salary" → action:hire, count:3, is_addition:true, salary:null
- If no action found → action:"none"
{context_str}
"""
    
    try:
        result = scenario_llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_query),
        ])
        scenario_dict = result.model_dump()
        
        if scenario_dict.get("action") not in ["none", None]:
            state["scenario_overrides"] = {
                "headcount_change": scenario_dict.get("count", 0),
                "avg_salary": (scenario_dict.get("salary") or 8000) * 12,
                "revenue_change": scenario_dict.get("revenue_change"),
                "one_time_expenses": scenario_dict.get("one_time_expenses"),
                "ramp_months": 3,
            }
            logger.info(f"📋 Scenario: {state['scenario_overrides']}")
        
        return scenario_dict
    except Exception as e:
        logger.error(f"Scenario failed: {e}")
        return {"action": "none", "count": 0, "explanation": str(e)}


# ================================================================
# TOOL 2: Forecast Runway
# ================================================================

@tool
def forecast_runway(forecast_months: int = 24) -> Dict[str, Any]:
    """
    Project cash runway using Monte Carlo simulation.
    Call after burn metrics are calculated.
    """
    state = _get_state()
    metrics = state.get("computed_metrics") or {}
    
    cash_balance = float(
        metrics.cash_balance if hasattr(metrics, 'cash_balance') 
        else metrics.get("cash_balance", state.get("cash_balance", 1200000))
    )
    net_burn = float(
        metrics.net_burn_3m_avg if hasattr(metrics, 'net_burn_3m_avg')
        else metrics.get("net_burn_3m_avg", metrics.get("net_burn", 100000))
    )
    
    logger.info(f"📈 Forecast: cash=${cash_balance:,.0f}, burn=${net_burn:,.0f}")
    
    result = forecast_cash_runway(
        cash_balance=cash_balance,
        net_burn=net_burn,
        burn_volatility=0.15,
        forecast_months=forecast_months,
    )
    
    output = {
        "p50_months": (result.p50_days or 0) // 30,
        "p10_months": (result.p10_days or 0) // 30,
        "p90_months": (result.p90_days or 0) // 30,
        "p50_days": result.p50_days or 0,
    }
    
    state["runway_forecast"] = output
    return output


# ================================================================
# TOOL 3: Generate Recommendations
# ================================================================

@tool
def generate_recommendations() -> List[Dict[str, Any]]:
    """
    Generate prioritized financial recommendations.
    Call when user asks for advice or after full analysis.
    """
    state = _get_state()
    metrics = state.get("computed_metrics") or {}
    runway_data = state.get("runway_forecast") or {}
    
    from services.forecasting import CashRunwayForecast
    
    runway_forecast = None
    if runway_data:
        try:
            runway_forecast = CashRunwayForecast(
                p10_date=datetime.now(),
                p50_date=datetime.now(),
                p90_date=datetime.now(),
                p10_days=runway_data.get("p10_days", 180),
                p50_days=runway_data.get("p50_days", 365),
                p90_days=runway_data.get("p90_days", 540),
                assumptions={},
            )
        except:
            pass
    
    from services.recommendations import generate_recommendations as generate_hybrid_recs
    
    recs = generate_hybrid_recs(
        burn_metrics={"metrics": metrics},
        forecast_results={},
        runway_forecast=runway_forecast,
        state=state,
        enable_llm_enrichment=True,
    )
    
    state["recommendations"] = recs
    logger.info(f"💡 {len(recs)} recommendations generated")
    return recs


