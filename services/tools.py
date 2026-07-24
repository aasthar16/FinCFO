"""
Tool definitions for the AI CFO Agent.
Lightweight tools — burn calculation moved to burn_calculator node.
"""
from langgraph.types import Command
import json
from langchain_core.tools import InjectedToolCallId
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from typing import Dict, Any, List, Optional, Annotated, Literal, Union
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langgraph.prebuilt import InjectedState
from services.forecasting import forecast_financials
from services.schemas import ScenarioExtractionResult
from settings import settings
from services.recommendations import (
    generate_agentic_recommendations,
)
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
def model_scenario(
    user_query: str,
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId], # 🚨 INJECT THE ID HERE
) -> Command:
    """
    Extract scenario parameters from a user's what-if query.
    """
    previous_context = {
        "total_headcount": state.get("scenario_overrides", {}).get("headcount_change", 0),
        "avg_salary": state.get("scenario_overrides", {}).get("avg_salary"),
        "active_scenario": state.get("active_scenario"),
    }

    context_str = ""
    if previous_context["total_headcount"]:
        context_str = f"""
Previous scenario:
- headcount_change = {previous_context["total_headcount"]}
- avg_salary = ${previous_context.get("avg_salary", "N/A")}
"""

    system_prompt = f"""
Extract scenario parameters from the user query.
RULES:
- "hire 2 engineers at $8000/month" -> action=hire, count=2, salary=8000
- "what if revenue grows 20%" -> action=revenue_change
If no scenario exists: action="none"
{context_str}
"""

    try:
        result = scenario_llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_query),
        ])

        scenario = result.model_dump()
        overrides = None

        if scenario.get("action") not in ("none", None):
            overrides = {
                "headcount_change": scenario.get("count", 0),
                "avg_salary": (scenario.get("salary") or 8000) * 12,
                "revenue_change": scenario.get("revenue_change"),
                "one_time_expenses": scenario.get("one_time_expenses"),
                "ramp_months": 3,
            }
            logger.info("📋 Scenario extracted: %s", overrides)

        
        return Command(
            update={
                "scenario_overrides": overrides,
                "messages": [
                    ToolMessage(
                        content=f"Scenario extracted successfully: {overrides}",
                        tool_call_id=tool_call_id
                    )
                ]
            }
        )

    except Exception as e:
        logger.exception("Scenario extraction failed")
        return Command(
            update={"scenario_overrides": None},
            messages=[
                ToolMessage(
                    content=f"Failed to extract scenario: {e}", 
                    tool_call_id=tool_call_id
                )
            ]
        )


# def model_scenario(
#     user_query: str,
#     state: Annotated[dict, InjectedState],
# ) -> Dict[str, Any]:
#     """
#     Extract scenario parameters from a user's what-if query.

#     Examples:
#     - Hire/fire employees
#     - Revenue increase/decrease
#     - One-time expenses
#     - Salary assumptions

#     Uses any existing scenario in the graph state as context, but does
#     not modify the graph state directly. The caller is responsible for
#     persisting the returned scenario into the graph state.
#     """

#     previous_context = {
#         "total_headcount": state.get("scenario_overrides", {}).get(
#             "headcount_change", 0
#         ),
#         "avg_salary": state.get("scenario_overrides", {}).get(
#             "avg_salary"
#         ),
#         "active_scenario": state.get("active_scenario"),
#     }

#     context_str = ""

#     if previous_context["total_headcount"]:
#         context_str = f"""
# Previous scenario:
# - headcount_change = {previous_context["total_headcount"]}
# - avg_salary = ${previous_context.get("avg_salary", "N/A")}
# """

#     system_prompt = f"""
# Extract scenario parameters from the user query.

# RULES:
# - "hire 2 engineers at $8000/month"
#     -> action=hire
#        count=2
#        role=engineer
#        salary=8000

# - "fire 2 salespeople"
#     -> action=fire
#        count=2
#        role=salesperson

# - "what if revenue grows 20%"
#     -> action=revenue_change

# - "add 3 more at same salary"
#     -> action=hire
#        count=3
#        is_addition=true
#        salary=null

# If no scenario exists:
#     action="none"

# {context_str}
# """

#     try:
#         result = scenario_llm.invoke(
#             [
#                 SystemMessage(content=system_prompt),
#                 HumanMessage(content=user_query),
#             ]
#         )

#         scenario = result.model_dump()

#         overrides = None

#         if scenario.get("action") not in ("none", None):
#             overrides = {
#                 "headcount_change": scenario.get("count", 0),
#                 "avg_salary": (scenario.get("salary") or 8000) * 12,
#                 "revenue_change": scenario.get("revenue_change"),
#                 "one_time_expenses": scenario.get("one_time_expenses"),
#                 "ramp_months": 3,
#             }

#             logger.info("📋 Scenario extracted: %s", overrides)

#         return Command(
#             update={
#                 "scenario_overrides": overrides,
#                 "scenario": scenario,
#             }
#         )

#     except Exception as e:
#         logger.exception("Scenario extraction failed")

#         return {
#             "scenario": {
#                 "action": "none",
#                 "count": 0,
#                 "explanation": str(e),
#             },
#             "scenario_overrides": None,
#         }



@tool
def calculate_burn_metrics(state: Annotated[dict, InjectedState]) -> Dict[str, str]:
    """
    Compute the startup's current financial snapshot.

    This tool acts as a trigger for the LangGraph burn calculator node.
    The actual burn computation is performed by the dedicated
    `burn_calculator_node`, which parses the available transaction data,
    computes burn metrics, and stores the results in the graph state.

    Produces:
        state["financial_snapshot"]
        state["financial_timeseries"]

    Required before:
        - forecast_runway
        - generate_recommendations
        - scenario analysis

    Returns:
        A placeholder response indicating that the burn calculation
        should be executed by the graph.
    """
    return {"status": "invoke burn node"}


# ================================================================
# TOOL 2: Forecast Runway
# ================================================================

@tool
def forecast_runway(forecast_months: int = 12, state: Annotated[dict, InjectedState] = None) -> Dict[str, Any]:
    """
    Forecast burn, revenue, cash projection and runway.

    Requires:
        financial_snapshot (produced by calculate_burn_metrics)

    Updates:
        state["forecast_results"]
        state["runway_forecast"]
    """

    # state = _get_state()

    snapshot = state.get("financial_snapshot")

    # 1. AGENT-PROOF ERROR HANDLING (Returns a dict instead of crashing)
    if snapshot is None:
        logger.warning("Agent called forecast_runway without financial_snapshot!")
        return {
            "forecast_results": {"ERROR": "CRITICAL: financial_snapshot not found. You MUST call calculate_burn_metrics tool first before forecasting."},
            "runway_forecast": None
        }

    logger.info(
        "Running Theta forecast for %d months...",
        forecast_months,
    )

    forecast_results = forecast_financials(
        financial_snapshot=snapshot,
        horizon=forecast_months,
    )

    # Update graph state
   

    logger.info("Forecast completed successfully.")

    return {
    "forecast_results": forecast_results,
    "runway_forecast": forecast_results.get("runway"),
}
# ================================================================
# TOOL 3: Generate Recommendations
# ================================================================



@tool
def generate_recommendations(state: Annotated[dict, InjectedState] = None) -> List[Dict[str, Any]]:
    """
    Generate AI-powered CFO recommendations.

    Requires:
        financial_snapshot
        forecast_results
    """

    # state = _get_state()

    snapshot = state["financial_snapshot"]

    if snapshot is None:
        raise ValueError(
            "Run calculate_burn_metrics before requesting recommendations."
        )

    recommendations = generate_agentic_recommendations(
        financial_snapshot=snapshot,
        forecast_results=state.get("forecast_results"),
        startup_profile=state.get("startup_profile", {}),
        scenario_overrides=state.get("scenario_overrides"),
    )

    return {
    "recommendations": recommendations
}