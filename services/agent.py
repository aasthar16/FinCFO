"""
Agentic Supervisor - LLM-powered orchestration.
Decides which tools to call and in what order.
"""

import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from services.llm_service import call_llm, call_llm_with_history, extract_json_from_llm
from services.tools import TOOLS_SCHEMA, execute_tool, TOOL_MAP

logger = logging.getLogger(__name__)


# ================================================================
# AGENT SYSTEM PROMPTS
# ================================================================

ROUTING_SYSTEM_PROMPT = """You are FinCFO router. Decide which tools to call.

AVAILABLE TOOLS:
- calculate_burn_metrics: Computes burn rate, expenses, runway from loaded data
- forecast_runway: Projects runway with Monte Carlo simulation  
- model_scenario: Extracts hiring/firing scenario from user query
- generate_recommendations: Creates actionable advice

RULES:
1. If user asks about burn/runway/expenses → MUST call calculate_burn_metrics FIRST
2. If user asks "what if" or mentions hiring → MUST call model_scenario, then calculate_burn_metrics, then forecast_runway
3. NEVER suggest asking user for data - it's already loaded
4. Return JSON: {"plan": ["tool1"], "reasoning": "...", "response_hint": "..."}"""


def plan_actions(user_query, state, conversation_history):
    user_lower = user_query.lower()
    has_metrics = state.get("computed_metrics") is not None
    has_forecast = state.get("forecast_results") is not None
    
    # FORCE tool calls - skip LLM for financial queries
    if any(w in user_lower for w in ["burn", "runway", "expense", "spending", "cash", "financial", "finance", "metric", "health"]):
        if not has_metrics:
            return {"plan": ["calculate_burn_metrics"], "reasoning": "Need metrics", "response_hint": "Show burn and runway"}
        elif not has_forecast and "runway" in user_lower:
            return {"plan": ["calculate_burn_metrics", "forecast_runway"], "reasoning": "Need forecast", "response_hint": "Show runway projections"}
    
    if any(w in user_lower for w in ["hire", "hiring", "what if", "scenario", "add", "employee", "salary"]):
        return {"plan": ["model_scenario", "calculate_burn_metrics", "forecast_runway"], "reasoning": "Scenario analysis", "response_hint": "Show before/after impact"}
    
    if any(w in user_lower for w in ["recommend", "advice", "suggest", "what should"]):
        return {"plan": ["calculate_burn_metrics", "forecast_runway", "generate_recommendations"], "reasoning": "Full analysis + advice", "response_hint": "Provide prioritized recommendations"}
    
    # Fallback to LLM
    try:
        return extract_json_from_llm(ROUTING_SYSTEM_PROMPT, f"Query: {user_query}", temperature=0.0)
    except:
        return {"plan": [], "reasoning": "Fallback", "response_hint": "General response"}
    

RESPONSE_SYSTEM_PROMPT = """You are FinCFO, an experienced startup CFO. Provide financial analysis with actionable insights.

FORMAT RULES:
- Use ### for section headers (### 🔥 Burn Rate)
- Use - for bullet points
- Write numbers as: $85,000/month (no bold markers needed)
- 4-6 lines of metrics + 1-2 lines of insight
- Be concise but helpful

FOR EACH QUERY TYPE, INCLUDE:

**Burn Rate Query:**
- Show gross burn, net burn, revenue, runway
- Insight: Is the burn sustainable? What's the burn multiple?

**Runway Query:**
- Show cash, burn, months remaining
- Insight: Is this comfortable? When should they fundraise?

**Hiring Query:**
- Show cost, new burn, runway impact
- Insight: Can they afford this? What's the risk? Alternatives?

**Recommendations:**
- Top 3 prioritized actions
- Insight: Which one has the biggest impact?

EXAMPLE RESPONSES:

### 🔥 Burn Rate
- Gross Burn: $31,283/month
- Net Burn: $116,283/month  
- Revenue: $85,000/month
- Runway: 10.3 months

⚠️ Net burn exceeds revenue by $31K/month. At this rate, you have 10 months before cash runs out. Consider reducing recurring expenses by 15-20% or accelerating revenue collection.

### 👥 Hiring Impact
- 3 engineers at $5,000/month each
- Additional cost: $15,000/month
- New net burn: $131,283/month
- Runway: 10.3 → 9.1 months

⚠️ This reduces your runway by 1.2 months. You can afford this if revenue grows 10%+ in the next quarter. Alternative: hire 2 now and the 3rd in 6 months.

### 💡 Recommendations
- 🔴 Cut discretionary spending by 20% - saves $6,000/month
- 🟠 Negotiate vendor contracts - potential 10% savings
- 🟢 Accelerate invoice collection - improves cash flow

Start with the first item to add 2 months to your runway immediately.

Current Data:
{metrics_summary}

History:
{conversation_summary}"""
# ================================================================
# AGENT CORE FUNCTIONS
# ================================================================

def plan_actions(
    user_query: str,
    state: Dict[str, Any],
    conversation_history: List[Dict[str, str]],
) -> Dict[str, Any]:
    """Agent plans which tools to call based on user query."""
    
    has_metrics = state.get("computed_metrics") is not None
    has_forecast = state.get("forecast_results") is not None
    
    # FORCE tool calls for financial queries
    user_lower = user_query.lower()
    
    # If user asks about finances and we DON'T have metrics, FORCE burn calculation
    if not has_metrics and any(w in user_lower for w in ["burn", "expense", "spending", "cash", "runway", "financial", "finance"]):
        logger.info("⚡ FORCING calculate_burn_metrics (user asked about finances, no metrics yet)")
        return {
            "plan": ["calculate_burn_metrics"],
            "reasoning": "User asked about financials, need to compute metrics first",
            "response_hint": "Show burn rate, expenses, and runway from computed metrics"
        }
    
    # If user asks about runway and we have metrics but no forecast
    if has_metrics and not has_forecast and any(w in user_lower for w in ["runway", "how long", "forecast", "survive"]):
        logger.info("⚡ FORCING forecast_runway (user asked about runway, no forecast yet)")
        return {
            "plan": ["forecast_runway"],
            "reasoning": "User asked about runway, need forecast",
            "response_hint": "Show runway projections with P10/P50/P90"
        }
    
    # If user asks about hiring/scenario
    if any(w in user_lower for w in ["hire", "hiring", "what if", "scenario", "add", "employee"]):
        logger.info("⚡ FORCING scenario → burn → forecast (hiring query)")
        return {
            "plan": ["model_scenario", "calculate_burn_metrics", "forecast_runway"],
            "reasoning": "User wants scenario analysis",
            "response_hint": "Show before/after impact of scenario on burn and runway"
        }
    
    # If user asks for recommendations and we have metrics
    if has_metrics and any(w in user_lower for w in ["recommend", "advice", "suggest", "what should"]):
        logger.info("⚡ FORCING generate_recommendations")
        return {
            "plan": ["generate_recommendations"],
            "reasoning": "User wants recommendations",
            "response_hint": "Provide actionable recommendations with priorities"
        }
    
    # For everything else, use LLM to plan
    try:
        state_summary = f"metrics={'✅' if has_metrics else '❌'}, forecast={'✅' if has_forecast else '❌'}"
        full_prompt = f"{state_summary}\n\nUser query: {user_query}"
        
        plan = extract_json_from_llm(
            system_prompt=ROUTING_SYSTEM_PROMPT,
            user_message=full_prompt,
            temperature=0.1,
        )
        logger.info(f"Agent plan: {plan.get('plan', [])} - {plan.get('reasoning', '')}")
        return plan
    except Exception as e:
        logger.error(f"Planning failed: {e}")
        return {"plan": [], "reasoning": "Fallback", "response_hint": "General response"}


def execute_plan(
    plan: List[str],
    state: Dict[str, Any],
    user_query: str,
) -> Dict[str, Any]:
    """Execute a sequence of tool calls. Updates state with results from each tool."""
    results = {}
    
    for tool_name in plan:
        if tool_name not in TOOL_MAP:
            logger.warning(f"Unknown tool in plan: {tool_name}")
            continue
        
        logger.info(f"Executing tool: {tool_name}")
        
        # Prepare tool arguments
        tool_args = {"state": state}
        
        if tool_name == "model_scenario":
            tool_args["user_query"] = user_query
            tool_args["previous_context"] = {
                "total_headcount": state.get("scenario_overrides", {}).get("headcount_change", 0),
                "avg_salary": state.get("scenario_overrides", {}).get("avg_salary"),
                "active_scenario": state.get("active_scenario"),
            }
        
        result = execute_tool(tool_name, tool_args, state)
        results[tool_name] = result
        
        # ================================================================
        # KEY FIX: Update state with tool results
        # ================================================================
        
        if tool_name == "model_scenario" and "error" not in str(result):
            if result.get("action") not in ["none", None]:
                # Store extracted scenario data
                state["scenario_overrides"] = {
                    "headcount_change": result.get("count", 0),
                    "avg_salary": (result.get("salary") or 8000) * 12,  # Convert monthly to annual
                    "revenue_change": result.get("revenue_change"),
                    "one_time_expenses": result.get("one_time_expenses"),
                    "ramp_months": 3,
                }
                state["active_scenario"] = f"hire_{result.get('count', 0)}_{result.get('role', 'employees')}"
                state["requires_recompute"] = True
                logger.info(f"📋 Scenario stored: {state['scenario_overrides']}")
        
        if tool_name == "calculate_burn_metrics" and "error" not in str(result):
            # Store computed metrics
            if isinstance(result, dict):
                state["computed_metrics"] = result.get("metrics", result)
            state["requires_recompute"] = False
            logger.info(f"📊 Metrics stored: net_burn=${state.get('computed_metrics', {}).get('net_burn', 'N/A')}")
        
        if tool_name == "forecast_runway" and "error" not in str(result):
            state["runway_forecast"] = result
            logger.info(f"✈️ Forecast stored: p50={result.get('p50_months', 'N/A')} months")
        
        if tool_name == "generate_recommendations":
            if isinstance(result, list):
                state["recommendations"] = result
            logger.info(f"💡 Recommendations stored: {len(state.get('recommendations', []))} items")
    
    return results

def generate_final_response(
    user_query: str,
    state: Dict[str, Any],
    conversation_history: List[Dict[str, str]],
    plan_results: Dict[str, Any],
    response_hint: str = "",
) -> str:
    """
    Generate the final user-facing response using all computed data.
    """
    # Build metrics summary from state
    metrics = state.get("computed_metrics", {}) or {}
    runway = state.get("runway_forecast", {}) or {}
    recommendations = state.get("recommendations", []) or []
    scenario = state.get("active_scenario")
    
    
    metrics_summary = f"""
        Cash: ${metrics.get('cash_balance', 0):,.0f}
        Gross Burn: ${metrics.get('gross_burn', 0):,.0f}/month
        Net Burn: ${metrics.get('net_burn', 0):,.0f}/month
        Revenue: ${metrics.get('monthly_revenue', 0):,.0f}/month
        One-Time Expenses: ${metrics.get('one_time_expenses', 0):,.0f}
        Runway: {metrics.get('cash_runway_months', 0):.1f} months
        P50 Runway: {runway.get('p50_months', runway.get('p50_days', 0)//30)} months
        P10 Runway: {runway.get('p10_months', runway.get('p10_days', 0)//30)} months"""
            
            # ... rest of the function
    if metrics.get('net_burn'):
        metrics_summary += f"""
- Net Burn: ${metrics.get('net_burn'):,.0f}/month
- Gross Burn: ${metrics.get('gross_burn', 0):,.0f}/month
- Monthly Revenue: ${metrics.get('monthly_revenue', 0):,.0f}/month
- One-Time Expenses: ${metrics.get('one_time_expenses', 0):,.0f}"""
    
    if runway.get('p50_months'):
        metrics_summary += f"""
- Runway: {runway['p50_months']} months (P50)
- Pessimistic: {runway.get('p10_months', 'N/A')} months
- Optimistic: {runway.get('p90_months', 'N/A')} months"""
    
    if scenario:
        metrics_summary += f"\n- Active Scenario: {scenario}"
    
    if recommendations:
        metrics_summary += f"\n- Recommendations available: {len(recommendations)}"
    
    # Build conversation summary
    conversation_summary = ""
    for msg in conversation_history[-4:]:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")[:100]
        conversation_summary += f"{role}: {content}\n"
    
    # Build system prompt with real data
    system_prompt = RESPONSE_SYSTEM_PROMPT.format(
        metrics_summary=metrics_summary,
        conversation_summary=conversation_summary or "No previous context",
    )
    
    if response_hint:
        system_prompt += f"\n\n**Focus area for this response:** {response_hint}"
    
    try:
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
    """Fallback response if LLM fails."""
    metrics = state.get("computed_metrics", {}) or {}
    runway = state.get("runway_forecast", {}) or {}
    
    parts = ["### 📊 Financial Summary\n"]
    
    if metrics:
        parts.append(f"- **Cash:** ${metrics.get('cash_balance', 0):,.0f}")
        parts.append(f"- **Net Burn:** ${metrics.get('net_burn', 0):,.0f}/month")
        parts.append(f"- **Revenue:** ${metrics.get('monthly_revenue', 0):,.0f}/month")
    
    if runway:
        parts.append(f"- **Runway:** {runway.get('p50_months', runway.get('p50_days', 0)//30)} months")
    
    parts.append(f"\n💬 Ask me about burn rate, runway, hiring scenarios, or recommendations!")
    
    return "\n".join(parts)


def run_agent(
    user_query: str,
    state: Dict[str, Any],
    conversation_history: List[Dict[str, str]],
) -> str:
    """
    Main agent loop: Plan → Execute → Respond
    """
    logger.info(f"🤖 Agent processing: {user_query[:80]}...")
    
    # Step 1: Plan
    plan = plan_actions(user_query, state, conversation_history)
    actions = plan.get("plan", [])
    response_hint = plan.get("response_hint", "")
    
    # Step 2: Execute
    plan_results = {}
    if actions:
        plan_results = execute_plan(actions, state, user_query)
    
    # Step 3: Generate response
    response = generate_final_response(
        user_query=user_query,
        state=state,
        conversation_history=conversation_history,
        plan_results=plan_results,
        response_hint=response_hint,
    )
    
    return response