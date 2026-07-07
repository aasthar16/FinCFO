"""
Agentic Supervisor - LLM-powered orchestration with structured output.
Uses LangChain + Pydantic to enforce schema on LLM responses.
"""

import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field
from typing_extensions import Literal

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from services.tools import TOOL_MAP, execute_tool
from settings import settings
logger = logging.getLogger(__name__)


# ================================================================
# PYDANTIC SCHEMAS — Enforce LLM output structure
# ================================================================

class PlanResult(BaseModel):
    """Schema for the planning agent's output."""
    plan: List[Literal[
        "calculate_burn_metrics",
        "forecast_runway",
        "model_scenario",
        "generate_recommendations"
    ]] = Field(
        default_factory=list,
        description="Ordered list of tool names to execute. Empty list if no tools needed."
    )
    reasoning: str = Field(
        default="",
        description="Brief explanation of why these specific tools were chosen in this order"
    )
    response_hint: str = Field(
        default="",
        description="What aspect to focus on when generating the final response (e.g., 'Show runway impact', 'Highlight burn reduction')"
    )


class ScenarioResult(BaseModel):
    """Schema for scenario extraction output."""
    action: Literal["hire", "fire", "replace", "revenue_change", "expense_change", "none"] = Field(
        default="none",
        description="Type of scenario action detected"
    )
    count: int = Field(
        default=0,
        description="Number of people to hire or fire"
    )
    role: str = Field(
        default="employee",
        description="Job role mentioned (e.g., engineer, designer, manager)"
    )
    salary: Optional[float] = Field(
        default=None,
        description="Monthly salary per person, if mentioned"
    )
    headcount_change: int = Field(
        default=0,
        description="Net change in headcount (positive for hire, negative for fire)"
    )
    revenue_change: Optional[float] = Field(
        default=None,
        description="Change in monthly revenue, if mentioned"
    )
    one_time_expenses: Optional[float] = Field(
        default=None,
        description="One-time expense amount, if mentioned"
    )
    is_addition: bool = Field(
        default=False,
        description="True if this is adding to a previous scenario"
    )
    explanation: str = Field(
        default="",
        description="Human-readable explanation of what was parsed from the query"
    )




class ResponseResult(BaseModel):
    """Schema for final response generation."""
    section_title: str = Field(
        default="### 📊 Financial Summary",
        description="Section header with emoji (e.g., '### 🔥 Burn Rate Analysis')"
    )
    bullet_points: List[str] = Field(
        default_factory=list,
        description="Key metrics as bullet points (e.g., '- **Net Burn:** $105,000/month')"
    )
    insight: str = Field(
        default="",
        description="One-line actionable insight or warning"
    )
    severity: Literal["critical", "warning", "healthy", "neutral"] = Field(
        default="neutral",
        description="Overall severity assessment"
    )


# ================================================================
# SYSTEM PROMPTS
# ================================================================

PLANNING_SYSTEM_PROMPT = """You are a financial planning agent for startups. Your sole job is to decide which tools to call and in what order.

AVAILABLE TOOLS:
1. calculate_burn_metrics — Computes burn rate, runway, expenses, revenue metrics from loaded transaction data. Always call this FIRST if no metrics exist.
2. forecast_runway — Projects runway using Monte Carlo simulation (P10/P50/P90 confidence intervals). Requires metrics to exist first.
3. model_scenario — Extracts hiring/firing/revenue-change parameters from the user's natural language query. Call this BEFORE calculate_burn_metrics when the user describes a scenario change.
4. generate_recommendations — Creates prioritized financial recommendations. Requires metrics to exist first.

DECISION LOGIC:
- User asks about burn/expenses/runway AND metrics don't exist → [calculate_burn_metrics]
- User asks about runway AND metrics exist AND no forecast → [calculate_burn_metrics, forecast_runway]
- User asks about runway AND forecast already exists → [generate_recommendations]
- User mentions hiring/firing/"what if"/scenario changes → [model_scenario, calculate_burn_metrics, forecast_runway]
- User asks for advice/recommendations → [calculate_burn_metrics, generate_recommendations]
- User asks about financial health or "how are we doing" → [calculate_burn_metrics, generate_recommendations]
- User asks a follow-up about previous scenario → reuse existing data, only run what's needed
- User greets or says thanks → [] (no tools needed)
- Default (any other financial question) → [calculate_burn_metrics, forecast_runway, generate_recommendations]

IMPORTANT:
- NEVER skip calculate_burn_metrics if metrics don't exist
- NEVER call forecast_runway before calculate_burn_metrics
- ONLY call model_scenario when user describes a change/hiring/firing scenario
- Return empty plan [] if all needed data already exists"""


RESPONSE_SYSTEM_PROMPT = """You are FinCFO, a startup financial analyst.

FORMAT RULES - FOLLOW EXACTLY:
1. Use ### for section headers with ONE emoji
2. Use - for bullet points  
3. Write numbers like this: $85,000/month (NO ** around numbers)
4. Only use **bold** for labels like **Gross Burn:** or **Net Burn:**
5. Put a space after ** and before **
6. Each bullet on its own line
7. Max 6 lines total
8. End with one actionable insight line starting with 💡

CURRENT DATA:
{metrics_summary}

RECOMMENDATIONS (if available):
{recommendations_summary}

CONTEXT:
{conversation_summary}

INSTRUCTIONS:
- If recommendations exist, reference them naturally in your response
- Use the enriched insights if provided (contextual_insight, priority_justification)
- Keep response concise and actionable
- Never make up numbers - use ONLY the data provided above"""

# ================================================================
# LLM INITIALIZATION WITH STRUCTURED OUTPUT
# ================================================================

# Initialize Groq via LangChain
llm = ChatGroq(
    model=settings.groq_model,
    api_key=settings.groq_api_key,
    temperature=0.0,
    max_tokens=500,
)

# Create structured LLMs for each output type
planning_llm = llm.with_structured_output(PlanResult)
scenario_llm = llm.with_structured_output(ScenarioResult)


# ================================================================
# AGENT CORE FUNCTIONS
# ================================================================

def plan_actions(
    user_query: str,
    state: Dict[str, Any],
    conversation_history: List[Dict[str, str]],
) -> Dict[str, Any]:
    """
    LLM-driven planning with Pydantic structured output.
    ZERO keyword matching. Pure agentic planning.
    
    The LLM receives:
    - Available tools with descriptions
    - Current state (what data exists)
    - Conversation context (last 6 messages)
    - User query
    
    Returns a validated PlanResult with:
    - plan: List of tool names to execute
    - reasoning: Why these tools were chosen
    - response_hint: What to focus on in response
    """
    
    # Build state context
    has_metrics = state.get("computed_metrics") is not None
    has_forecast = state.get("forecast_results") is not None
    has_runway = state.get("runway_forecast") is not None
    active_scenario = state.get("active_scenario", "none")
    rec_count = len(state.get("recommendations", []))
    
    # Build conversation context (last 3 exchanges = 6 messages)
    context_lines = []
    for msg in conversation_history[-4:]:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")[:150]
        context_lines.append(f"{role}: {content}")
    context_str = "\n".join(context_lines) if context_lines else "No previous messages"
    
    # Build the state context string
    state_context = f"""CURRENT STATE:
        - Metrics computed: {"✅ yes" if has_metrics else "❌ no (needs calculation)"}
        - Forecast available: {"✅ yes" if has_forecast else "❌ no (needs projection)"}
        - Runway forecast available: {"✅ yes" if has_runway else "❌ no"}
        - Active scenario: {active_scenario}
        - Recommendations available: {rec_count}

        CONVERSATION HISTORY (last 3 exchanges):
        {context_str}

        USER QUERY:
        {user_query}

        Based on the above, which tools should I call? Return your plan."""

    try:
        # Call LLM with structured output — returns PlanResult, not raw text
        result: PlanResult = planning_llm.invoke([
            SystemMessage(content=PLANNING_SYSTEM_PROMPT),
            HumanMessage(content=state_context),
        ])
        
        logger.info(f"🧠 Plan: {result.plan} — {result.reasoning[:80]}")
        
        return {
            "plan": result.plan,
            "reasoning": result.reasoning,
            "response_hint": result.response_hint,
        }
        
    except Exception as e:
        logger.error(f"Planning failed: {e}", exc_info=True)
        # Fallback: compute baseline metrics with recommendations
        return {
            "plan": ["calculate_burn_metrics", "forecast_runway", "generate_recommendations"],
            "reasoning": "Planning error, recomputing comprehensive baseline with recommendations",
            "response_hint": "Show current financial status with runway projections and recommendations"
        }


def execute_plan(
    plan: List[str],
    state: Dict[str, Any],
    user_query: str,
) -> Dict[str, Any]:
    """
    Execute a sequence of tool calls deterministically.
    Each tool updates the state with its results.
    """
    results = {}
    
    for tool_name in plan:
        if tool_name not in TOOL_MAP:
            logger.warning(f"Unknown tool in plan: {tool_name}")
            continue
        
        logger.info(f"🔧 Executing: {tool_name}")
        
        # Prepare tool arguments based on tool type
        tool_args = {"state": state}
        
        if tool_name == "model_scenario":
            tool_args["user_query"] = user_query
            tool_args["previous_context"] = {
                "total_headcount": state.get("scenario_overrides", {}).get("headcount_change", 0),
                "avg_salary": state.get("scenario_overrides", {}).get("avg_salary"),
                "active_scenario": state.get("active_scenario"),
            }
        
        # Execute the tool
        result = execute_tool(tool_name, tool_args, state)
        results[tool_name] = result
        
        # Update state with tool results
        _update_state_from_tool(tool_name, result, state)
    
    return results


def _update_state_from_tool(tool_name: str, result: Any, state: Dict[str, Any]) -> None:
    """Update global state based on tool execution results."""
    
    if "error" in str(result):
        logger.warning(f"Tool {tool_name} returned error: {result}")
        return
    
    if tool_name == "model_scenario":
        if result.get("action") not in ["none", None]:
            state["scenario_overrides"] = {
                "headcount_change": result.get("count", 0),
                "avg_salary": (result.get("salary") or 8000) * 12,
                "revenue_change": result.get("revenue_change"),
                "one_time_expenses": result.get("one_time_expenses"),
                "ramp_months": 3,
            }
            state["active_scenario"] = f"hire_{result.get('count', 0)}_{result.get('role', 'employees')}"
            state["requires_recompute"] = True
            logger.info(f"📋 Scenario stored: {state['scenario_overrides']}")
    
    elif tool_name == "calculate_burn_metrics":
        if isinstance(result, dict):
            state["computed_metrics"] = result.get("metrics", result)
        state["requires_recompute"] = False
        logger.info(f"📊 Metrics stored")
    
    elif tool_name == "forecast_runway":
        state["runway_forecast"] = result
        logger.info(f"✈️ Forecast stored")
    
    elif tool_name == "generate_recommendations":
        if isinstance(result, list) and len(result) > 0:
            state["recommendations"] = result
            logger.info(f"💡 Recommendations stored: {len(result)} items")
        else:
            logger.warning(f"⚠️ No recommendations generated or invalid format")


# In generate_final_response, use the structured data

def generate_final_response(
    user_query: str,
    state: Dict[str, Any],
    conversation_history: List[Dict[str, str]],
    plan_results: Dict[str, Any],
    response_hint: str = "",
) -> str:
    """Generate final response with enriched recommendations."""
    metrics = state.get("computed_metrics") or {}
    runway = state.get("runway_forecast") or {}
    recommendations = state.get("recommendations") or []
    
    # Convert metrics to dict if it's a dataclass
    if hasattr(metrics, '__dataclass_fields__'):
        from dataclasses import asdict
        metrics = asdict(metrics)
    
    # Build metrics summary - NO indentation in f-string
    cash = metrics.get('cash_balance', 0) if isinstance(metrics, dict) else getattr(metrics, 'cash_balance', 0)
    gross = metrics.get('gross_burn', 0) if isinstance(metrics, dict) else getattr(metrics, 'gross_burn', 0)
    net = metrics.get('net_burn', 0) if isinstance(metrics, dict) else getattr(metrics, 'net_burn', 0)
    revenue = metrics.get('monthly_revenue', 0) if isinstance(metrics, dict) else getattr(metrics, 'monthly_revenue', 0)
    runway_months = metrics.get('cash_runway_months', 0) if isinstance(metrics, dict) else getattr(metrics, 'cash_runway_months', 0)
    
    metrics_summary = (
        f"Cash Balance: ${cash:,.0f}\n"
        f"Gross Burn: ${gross:,.0f}/month\n"
        f"Net Burn: ${net:,.0f}/month\n"
        f"Monthly Revenue: ${revenue:,.0f}/month\n"
        f"Runway: {runway_months:.1f} months"
    )
    
    # Build recommendations summary
    rec_summary = "No recommendations available."
    if recommendations:
        rec_lines = []
        for rec in recommendations[:3]:
            priority = rec.get('priority', 'MEDIUM')
            title = rec.get('title', '')
            description = rec.get('description', '')
            rec_lines.append(f"- {priority}: {title} — {description[:100]}")
        rec_summary = "\n".join(rec_lines)
    
    # Build clean system prompt
    system_prompt = (
        "You are FinCFO, a startup financial analyst.\n\n"
        f"METRICS:\n{metrics_summary}\n\n"
        f"RECOMMENDATIONS:\n{rec_summary}\n\n"
        "INSTRUCTIONS:\n"
        "1. Use ONLY the numbers provided above\n"
        "2. Reference recommendations naturally in your response\n"
        "3. Keep response concise and actionable\n"
        "4. Use ### for section headers with ONE emoji\n"
        "5. Use - for bullet points\n"
        "6. End with 💡 actionable insight"
    )
    
    if response_hint:
        system_prompt += f"\n\nFocus area: {response_hint}"
    
    try:
        from services.llm_service import call_llm_with_history
        logger.info(f"System prompt being sent:\n{system_prompt[:300]}")
        response = call_llm_with_history(
            system_prompt=system_prompt,
            conversation_history=conversation_history,
            user_message=user_query,
            temperature=0.4,
            max_tokens=500,
        )
        return response
    except Exception as e:
        logger.error(f"Response generation failed: {e}")
        return generate_fallback_response(state)
    
def generate_fallback_response(state: Dict[str, Any]) -> str:
    """Deterministic fallback if LLM fails."""
    metrics = state.get("computed_metrics", {}) or {}
    
    # Convert to dict if dataclass
    if hasattr(metrics, '__dataclass_fields__'):
        from dataclasses import asdict
        metrics = asdict(metrics)
    
    runway = state.get("runway_forecast", {}) or {}
    
    parts = ["### 📊 Financial Summary", ""]
    
    cash = metrics.get('cash_balance', 0) if isinstance(metrics, dict) else 0
    net_burn = metrics.get('net_burn', 0) if isinstance(metrics, dict) else 0
    revenue = metrics.get('monthly_revenue', 0) if isinstance(metrics, dict) else 0
    runway_months = metrics.get('cash_runway_months', 0) if isinstance(metrics, dict) else 0
    
    if cash:
        parts.append(f"- **Cash:** ${cash:,.0f}")
    if net_burn:
        parts.append(f"- **Net Burn:** ${net_burn:,.0f}/month")
    if revenue:
        parts.append(f"- **Revenue:** ${revenue:,.0f}/month")
    if runway_months:
        parts.append(f"- **Runway:** {runway_months:.1f} months")
    
    parts.append("")
    parts.append("💬 Ask me about burn rate, runway, hiring scenarios, or recommendations!")
    
    return "\n".join(parts)

def run_agent(
    user_query: str,
    state: Dict[str, Any],
    conversation_history: List[Dict[str, str]],
) -> str:
    """
    Main agent loop: Plan → Execute → Respond
    
    1. Plan: LLM decides which tools to call (structured output via Pydantic)
    2. Execute: Tools run deterministically, updating state
    3. Respond: LLM generates contextual response using computed data
    """
    logger.info(f"🤖 Agent processing: {user_query[:80]}...")
    
    # Step 1: Plan — LLM-driven with Pydantic schema enforcement
    plan = plan_actions(user_query, state, conversation_history)
    actions = plan.get("plan", [])
    response_hint = plan.get("response_hint", "")
    
    # Step 2: Execute — Deterministic tool calls
    plan_results = {}
    if actions:
        plan_results = execute_plan(actions, state, user_query)
    
    # Step 3: Respond — LLM generates final response with real data
    response = generate_final_response(
        user_query=user_query,
        state=state,
        conversation_history=conversation_history,
        plan_results=plan_results,
        response_hint=response_hint,
    )
    
    return response