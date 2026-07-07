"""
Tool definitions for the AI CFO Agent.
Each tool wraps existing deterministic math functions.
The LLM Agent calls these tools - NEVER does math itself.
"""

import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from services.compute_burn import compute_burn, convert_to_serializable
from utils.helpers import generate_mock_transactions
import pandas as pd
from services.forecasting import forecast_cash_runway, convert_to_serializable
from services.llm_service import extract_json_from_llm
from services.forecasting import generate_recommendations
from services.llm_service import call_llm
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from services.schemas import ScenarioExtractionResult
from settings import settings

# Initialize Groq
llm = ChatGroq(
    model=settings.groq_model,
    api_key=settings.groq_api_key,
    temperature=0.0,
    max_tokens=500,
)

# Create structured LLM for scenario extraction
scenario_llm = llm.with_structured_output(ScenarioExtractionResult)
    

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
    
    # Use parsed transactions from state
    transactions_data = state.get("transactions_data", [])
    
    if not transactions_data:
        # If no transactions, generate mock data for demo
        logger.warning("No transactions found, using mock data")
        transactions_df = generate_mock_transactions(months=6)
    else:
        transactions_df = pd.DataFrame(transactions_data)
        if 'date' in transactions_df.columns:
            transactions_df['date'] = pd.to_datetime(transactions_df['date'])
    
    
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
    Uses Pydantic structured output - NO regex parsing!
    """
    
    # Build previous context if available
    context_str = ""
    if previous_context:
        context_str = f"""
            Previous scenario context:
            - Total headcount change so far: {previous_context.get('total_headcount', 0)}
            - Average salary: ${previous_context.get('avg_salary', 'N/A')}
            - Active scenario: {previous_context.get('active_scenario', 'None')}
            """
    
    system_prompt = f"""You are a financial scenario parser. Extract hiring/firing/business change parameters from user queries.

RULES:
1. "hire 2 engineers at $8000/month" → action:hire, count:2, role:engineer, salary:8000
2. "add 3 more at same salary" → action:hire, count:3, salary:null, is_addition:true
3. "replace 2 designers with 3 engineers" → action:replace, count:3, role:engineer
4. "what if revenue grows 20%" → action:revenue_change, calculate revenue_change from context
5. "fire 2 salespeople" → action:fire, count:2, role:salesperson
6. For "same salary/pay/rate", set salary:null and is_addition:true
7. If no specific action found, return action:"none"
8. Handle typos gracefully (ppl=people, aat=at, etc.)

{context_str}

Return the extracted scenario parameters in the structured format."""
    
    try:
        # Call LLM with structured output - RETURNS Pydantic object directly!
        result: ScenarioExtractionResult = scenario_llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_query),
        ])
        
        # Convert to dict for storage
        return result.model_dump()
        
    except Exception as e:
        logger.error(f"Scenario extraction failed: {e}")
        # Return default empty scenario
        return {
            "action": "none",
            "count": 0,
            "role": "employee",
            "salary": None,
            "headcount_change": 0,
            "revenue_change": None,
            "one_time_expenses": None,
            "is_addition": False,
            "explanation": f"Failed to parse: {e}",
        }

def tool_generate_recommendations(
    state: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Tool: Generate recommendations with LLM enrichment."""
    
    metrics = state.get("computed_metrics") or {}
    forecast = state.get("forecast_results") or {}
    runway_data = state.get("runway_forecast") or {}
    
    # Reconstruct runway forecast if needed
    runway_forecast = None
    if runway_data:
        try:
            from services.forecasting import CashRunwayForecast
            from datetime import datetime
            
            def parse_date(date_str):
                if not date_str:
                    return datetime.now()
                return datetime.fromisoformat(date_str)
            
            runway_forecast = CashRunwayForecast(
                p10_date=parse_date(runway_data.get("p10_date")),
                p50_date=parse_date(runway_data.get("p50_date")),
                p90_date=parse_date(runway_data.get("p90_date")),
                p10_days=runway_data.get("p10_days", 180),
                p50_days=runway_data.get("p50_days", 365),
                p90_days=runway_data.get("p90_days", 540),
                assumptions=runway_data.get("assumptions", {}),
            )
        except Exception as e:
            logger.warning(f"Failed to reconstruct runway forecast: {e}")
    # Generate hybrid recommendations (deterministic + LLM enrichment)
    from services.recommendations import generate_recommendations as generate_hybrid_recs
    
    recs = generate_hybrid_recs(
        burn_metrics={"metrics": metrics},
        forecast_results=forecast,
        runway_forecast=runway_forecast,
        state=state,
        enable_llm_enrichment=True,  # Always enable for Phase 2
    )
    
    return recs

# ===============================================================
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
                state=state
                # priority_areas=tool_args.get("priority_areas"),
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