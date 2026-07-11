"""
Agentic Supervisor Node - True LLM Agent with Tool Calling.
LLM autonomously decides: call burn calculator, call tools, or respond.
"""

import logging
from typing import Literal
from langchain_core.messages import SystemMessage, AIMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt

from graph.state import GlobalState
from config.langsmith import traced
from settings import settings

logger = logging.getLogger(__name__)


# ================================================================
# SYSTEM PROMPT
# ================================================================

SYSTEM_PROMPT = """You are FinCFO, a startup financial analyst AI agent.

**YOUR TOOLS:**
- `calculate_burn_metrics` — Calculate burn rate, runway, expenses from transaction data. Call FIRST for any financial question.
- `forecast_runway` — Project cash runway. Call after burn metrics exist.
- `model_scenario` — Extract hiring/firing parameters from user query. Call FIRST for scenario changes.
- `generate_recommendations` — Generate financial recommendations. Call after metrics and forecast exist.
- `request_user_confirmation` — Ask user to confirm high-risk decisions.

**HOW TO WORK:**
- "What's our burn rate?" → calculate_burn_metrics → respond
- "What's our runway?" → calculate_burn_metrics → forecast_runway → respond
- "Hire 2 engineers" → model_scenario → calculate_burn_metrics → forecast_runway → respond
- "Recommendations?" → generate_recommendations → respond
- Greetings → respond directly
- If runway < 6 months or burn >> revenue → call request_user_confirmation before responding

**RESPONSE RULES:**
- Use ### for headers with ONE emoji
- Use - for bullet points
- Write numbers like: $85,000/month
- End with 💡 actionable insight
- NEVER make up numbers"""


# ================================================================
# TOOLS (only lightweight ones — burn calc is a separate node)
# ================================================================

from services.tools import (
    forecast_runway,
    model_scenario,
    generate_recommendations,
)


@tool
def request_user_confirmation(reason: str, context: str = "") -> str:
    """
    Pause and ask the user to confirm before proceeding with a critical recommendation.
    
    Args:
        reason: Why confirmation is needed
        context: Additional context
    
    Returns:
        "proceed" or "cancel"
    """
    decision = interrupt({
        "kind": "confirmation",
        "reason": reason,
        "context": context,
    })
    return decision if decision else "cancel"


tools_list = [
    model_scenario,
    forecast_runway,
    generate_recommendations,
    request_user_confirmation,
]

tool_node = ToolNode(tools_list)


# ================================================================
# SUPERVISOR NODE
# ================================================================

@traced("supervisor_agent", tags=["supervisor", "agent", "tool_calling"])
def supervisor_agent_node(state: GlobalState) -> dict:
    """Agentic Supervisor Node."""
    from langchain_groq import ChatGroq
    
    messages = state.get("messages", [])
    
    full_messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(messages[-12:])
    
    llm = ChatGroq(
        model=settings.groq_model,
        api_key=settings.groq_api_key,
        temperature=0.0,
        max_tokens=1000,
    )
    llm_with_tools = llm.bind_tools(tools_list)
    
    logger.info(f"🧠 Agent processing: {len(messages)} messages in history")
    
    try:
        response = llm_with_tools.invoke(full_messages)
        logger.info(f"LLM: tool_calls={bool(response.tool_calls)}, content={str(response.content)[:100] if response.content else 'None'}")
        return {"messages": [response]}
    except Exception as e:
        logger.error(f"❌ Agent error: {e}", exc_info=True)
        return {"messages": [AIMessage(content=f"❌ Something went wrong. Please try again.")]}


# ================================================================
# ROUTER
# ================================================================

def route_after_agent(state: GlobalState) -> Literal["burn", "tools", "end"]:
    """Route based on what the agent decided."""
    messages = state.get("messages", [])
    if not messages:
        return "end"
    
    last_message = messages[-1]
    
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        for tc in last_message.tool_calls:
            name = tc.get('name') if isinstance(tc, dict) else tc.name
            if name == "calculate_burn_metrics":
                logger.info("🔧 Routing to burn calculator")
                return "burn"
        logger.info("🔧 Routing to tools")
        return "tools"
    
    logger.info("✅ Agent finished")
    return "end"