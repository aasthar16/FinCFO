"""
Tool definitions for the AI CFO Agent.
Each tool wraps existing deterministic math functions.
The LLM Agent calls these tools - NEVER does math itself.
"""

import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


# ================================================================
# TOOL DEFINITIONS (for LLM function calling)
# ================================================================

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "calculate_burn_metrics",
            "description": "Calculate burn rate, runway, and financial metrics from transaction data",
            "parameters": {
                "type": "object",
                "properties": {
                    "cash_balance": {
                        "type": "number",
                        "description": "Current cash balance in dollars"
                    },
                    "monthly_revenue": {
                        "type": "number",
                        "description": "Current monthly revenue in dollars"
                    },
                    "scenario_overrides": {
                        "type": "object",
                        "description": "Optional scenario parameters (headcount_change, avg_salary, etc.)",
                        "properties": {
                            "headcount_change": {"type": "integer"},
                            "avg_salary": {"type": "number"},
                            "revenue_change": {"type": "number"},
                            "one_time_expenses": {"type": "number"},
                            "ramp_months": {"type": "integer"}
                        }
                    }
                },
                "required": ["cash_balance", "monthly_revenue"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "forecast_runway",
            "description": "Project future cash runway using Monte Carlo simulation and Prophet forecasting",
            "parameters": {
                "type": "object",
                "properties": {
                    "cash_balance": {"type": "number"},
                    "net_burn": {"type": "number"},
                    "monthly_revenue": {"type": "number"},
                    "forecast_months": {"type": "integer", "default": 24},
                    "burn_volatility": {"type": "number", "default": 0.15}
                },
                "required": ["cash_balance", "net_burn", "monthly_revenue"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "model_scenario",
            "description": "Extract scenario parameters from user's natural language query about hiring, firing, or business changes",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_query": {
                        "type": "string",
                        "description": "The user's original query about scenario changes"
                    },
                    "previous_context": {
                        "type": "object",
                        "description": "Previous scenario context if any",
                        "properties": {
                            "total_headcount": {"type": "integer"},
                            "avg_salary": {"type": "number"},
                            "active_scenario": {"type": "string"}
                        }
                    }
                },
                "required": ["user_query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_recommendations",
            "description": "Generate prioritized financial recommendations based on metrics and forecasts",
            "parameters": {
                "type": "object",
                "properties": {
                    "priority_areas": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Areas to focus recommendations on: cash_management, efficiency, expense_management, revenue_growth"
                    }
                },
                "required": []
            }
        }
    }
]


# ================================================================
# TOOL IMPLEMENTATIONS (wrapping existing math)
# ================================================================

def tool_calculate_burn(
    state: Dict[str, Any],
    scenario_overrides: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Tool: Calculate burn metrics."""
    from services.compute_burn import compute_burn, convert_to_serializable
    from utils.helpers import generate_mock_transactions
    import pandas as pd
    
    transactions_data = state.get("transactions_data", [])
    if transactions_data:
        transactions_df = pd.DataFrame(transactions_data)
        if 'date' in transactions_df.columns:
            transactions_df['date'] = pd.to_datetime(transactions_df['date'])
    else:
        transactions_df = generate_mock_transactions(months=6)
    
    cash_balance = float(state.get("cash_balance", 1200000))
    monthly_revenue = float(state.get("monthly_revenue", 85000))
    
    # Get scenario overrides - from state OR from agent plan
    overrides = scenario_overrides or state.get("scenario_overrides", {})
    
    # If avg_salary looks like monthly (e.g., 8000), convert to annual for compute_burn
    if overrides.get("avg_salary") and overrides["avg_salary"] < 50000:
        overrides["avg_salary"] = overrides["avg_salary"] * 12  # Monthly → Annual
    
    logger.info(f"💰 Calculating burn with overrides: {overrides}")
    
    result = compute_burn(
        transactions_df=transactions_df,
        cash_balance=cash_balance,
        monthly_revenue=monthly_revenue,
        scenario_overrides=overrides,
    )
    
    return convert_to_serializable(result)

def tool_forecast_runway(
    state: Dict[str, Any],
    forecast_months: int = 24
) -> Dict[str, Any]:
    """Tool: Forecast cash runway."""
    from services.forecasting import forecast_cash_runway, convert_to_serializable
    
    metrics = state.get("computed_metrics") or {}
    
    cash_balance = float(metrics.get("cash_balance") or state.get("cash_balance") or 1200000)
    net_burn = float(metrics.get("net_burn_3m_avg") or metrics.get("net_burn") or 100000)
    
    runway_result = forecast_cash_runway(
        cash_balance=cash_balance,
        net_burn=net_burn,
        burn_volatility=0.15,
        forecast_months=forecast_months,
    )
    
    return convert_to_serializable({
        "p10_date": runway_result.p10_date.isoformat() if runway_result.p10_date else "",
        "p50_date": runway_result.p50_date.isoformat() if runway_result.p50_date else "",
        "p90_date": runway_result.p90_date.isoformat() if runway_result.p90_date else "",
        "p10_days": runway_result.p10_days or 0,
        "p50_days": runway_result.p50_days or 0,
        "p90_days": runway_result.p90_days or 0,
        "p10_months": (runway_result.p10_days or 0) // 30,
        "p50_months": (runway_result.p50_days or 0) // 30,
        "p90_months": (runway_result.p90_days or 0) // 30,
    })

def tool_model_scenario(
    user_query: str,
    previous_context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Tool: Extract scenario parameters from natural language.
    Uses LLM for structured extraction.
    """
    from services.llm_service import extract_json_from_llm
    
    context_str = ""
    if previous_context:
        context_str = f"\nPrevious context: {json.dumps(previous_context)}"
    
    system_prompt = f"""You are a financial scenario parser. Extract hiring/firing/business change parameters from user queries.

Return a JSON object with these fields:
{{
    "action": "hire|fire|replace|revenue_change|expense_change|none",
    "count": <number of people to hire/fire>,
    "role": "<job role if mentioned, else 'employee'>",
    "salary": <monthly salary per person, null if not mentioned>,
    "headcount_change": <net change in headcount (+ for hire, - for fire)>,
    "revenue_change": <change in monthly revenue, null if N/A>,
    "one_time_expenses": <one-time expense amount, null if N/A>,
    "is_addition": <true if adding to previous scenario, false if new scenario>,
    "explanation": "<brief explanation of what was parsed>"
}}

Rules:
- "hire 2 engineers at $8000/month" → action:hire, count:2, role:engineer, salary:8000
- "add 3 more at same salary" → action:hire, count:3, salary:null, is_addition:true
- "replace 2 designers with 3 engineers at $9000" → action:replace, count:3, role:engineer, salary:9000
- "what if revenue grows 20%" → action:revenue_change, revenue_change: <calculated>
- "fire 2 salespeople" → action:fire, count:2, role:salesperson
- For "same salary/pay/rate", set salary:null and is_addition:true
- If no specific action found, return action:"none"
- Handle typos gracefully (ppl=people, aat=at, etc.){context_str}"""

    result = extract_json_from_llm(
        system_prompt=system_prompt,
        user_message=user_query,
        temperature=0.0,
    )
    
    return result

def tool_generate_recommendations(
    state: Dict[str, Any],
    priority_areas: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """Tool: Generate recommendations."""
    from services.forecasting import generate_recommendations
    from services.llm_service import call_llm
    
    metrics = state.get("computed_metrics") or {}
    forecast = state.get("forecast_results") or {}
    runway = state.get("runway_forecast") or {}
    
    # Get base recommendations
    base_recs = state.get("recommendations") or []
    
    # If no recommendations exist, generate them
    if not base_recs and metrics:
        try:
            base_recs = generate_recommendations(
                burn_metrics={"metrics": metrics},
                forecast_results=forecast,
                runway_forecast=None,
            )
        except Exception as e:
            logger.warning(f"Failed to generate recommendations: {e}")
            base_recs = []
    
    # Enhance with LLM
    if priority_areas and base_recs:
        try:
            system_prompt = f"""Enhance financial recommendations. Focus: {', '.join(priority_areas)}.
Current: Cash=${metrics.get('cash_balance', 0):,.0f}, Burn=${metrics.get('net_burn', 0):,.0f}/month.
Existing: {json.dumps(base_recs[:3], indent=2)}
Return enhanced as JSON array."""
            
            enhanced = call_llm(system_prompt=system_prompt, user_message="Enhance recommendations", temperature=0.3)
            import re
            json_match = re.search(r'\[.*\]', enhanced, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
    
    return base_recs if base_recs else [
        {"priority": "MEDIUM", "title": "Review Expenses", "description": "Analyze spending patterns", "suggested_actions": ["Review vendor contracts", "Reduce discretionary spending"]}
    ]

# ================================================================
# TOOL DISPATCHER
# ================================================================

TOOL_MAP = {
    "calculate_burn_metrics": tool_calculate_burn,
    "forecast_runway": tool_forecast_runway,
    "model_scenario": tool_model_scenario,
    "generate_recommendations": tool_generate_recommendations,
}


def execute_tool(tool_name: str, tool_args: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a tool by name with given arguments."""
    if tool_name not in TOOL_MAP:
        return {"error": f"Unknown tool: {tool_name}"}
    
    tool_fn = TOOL_MAP[tool_name]
    
    try:
        if tool_name == "model_scenario":
            return tool_fn(
                user_query=tool_args.get("user_query", ""),
                previous_context=tool_args.get("previous_context"),
            )
        elif tool_name == "generate_recommendations":
            return tool_fn(
                state=state,
                priority_areas=tool_args.get("priority_areas"),
            )
        elif tool_name == "forecast_runway":
            return tool_fn(
                state=state,
                forecast_months=tool_args.get("forecast_months", 24),
            )
        elif tool_name == "calculate_burn_metrics":
            return tool_fn(
                state=state,
                scenario_overrides=tool_args.get("scenario_overrides"),
            )
    except Exception as e:
        logger.error(f"Tool execution error: {e}", exc_info=True)
        return {"error": str(e)}