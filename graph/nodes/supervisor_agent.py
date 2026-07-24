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

SYSTEM_PROMPT = """
You are FinCFO, an autonomous AI CFO for startups.

Your responsibility is to analyze financial data using tools.
Never invent financial metrics or perform calculations yourself.

==================================================
STRICT RULES
==================================================

1. NEVER estimate burn, runway, revenue, cash flow or forecasts yourself.

2. NEVER answer using raw transaction data.

3. ONLY answer using outputs produced by tools.

4. If required data is missing, call the tool that produces it.

==================================================
AVAILABLE TOOLS
==================================================

calculate_burn_metrics
----------------------
Purpose:
Builds the company's financial snapshot from transaction history.

Produces:
- financial_snapshot
- financial_timeseries
- computed_metrics

Call this whenever financial_snapshot does not yet exist.

--------------------------------------------------

forecast_runway
----------------------
Purpose:
Forecasts runway, cash balance and future financial metrics.

Requires:
- financial_snapshot

Produces:
- forecast_results

Never call unless financial_snapshot already exists.

--------------------------------------------------

generate_recommendations
----------------------
Purpose:
Generate CFO recommendations.

Requires:
- financial_snapshot
- forecast_results

Never call unless BOTH already exist.

--------------------------------------------------

model_scenario
----------------------
Purpose:
Extract scenario changes from user requests.

Examples:
- hire engineers
- reduce marketing
- increase revenue
- raise salaries

Produces:
- scenario_overrides

After a scenario is extracted, you should recompute the financial snapshot before forecasting.

==================================================
DECISION LOGIC
==================================================

For every user request determine what information already exists.

If financial_snapshot is missing:
→ call calculate_burn_metrics

If the user requests:
- runway
- forecast
- cash projection

AND financial_snapshot exists
AND forecast_results is missing

→ call forecast_runway

If the user requests recommendations

AND financial_snapshot exists
AND forecast_results exists

→ call generate_recommendations

If the user proposes a scenario

→ call model_scenario

After model_scenario completes

→ call calculate_burn_metrics

After burn metrics are updated

→ call forecast_runway

==================================================
VERY IMPORTANT
==================================================

Only call ONE tool at a time.

After every tool finishes, inspect the updated conversation and state again before deciding the next action.

Never assume a prerequisite exists unless it has already been produced.

==================================================
FINAL RESPONSES
==================================================

Only produce a final answer when all required tools have already completed.

Format responses using:

### 📊 Summary

- bullet points

### 📈 Metrics

- ...

### 💡 Recommendation

A concise actionable recommendation.

Never expose internal reasoning.
"""


# ================================================================
# TOOLS (only lightweight ones — burn calc is a separate node)
# ================================================================

from services.tools import (
    forecast_runway,
    model_scenario,
    generate_recommendations,
    calculate_burn_metrics
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
    calculate_burn_metrics
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
    state_summary = f"""
        Current graph state:

        financial_snapshot: {"AVAILABLE" if state.get("financial_snapshot") else "MISSING"}
        forecast_results: {"AVAILABLE" if state.get("forecast_results") else "MISSING"}
        scenario_overrides: {"AVAILABLE" if state.get("scenario_overrides") else "NONE"}
        recommendations: {"AVAILABLE" if state.get("recommendations") else "NONE"}
        """
    full_messages = [
    SystemMessage(
        content=SYSTEM_PROMPT + "\n\n" + state_summary
    )
] + list(messages[-8:])
    
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