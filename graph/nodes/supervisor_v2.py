"""
Supervisor Node V2 - LLM-powered agentic supervisor.
Replaces regex-based routing with Groq LLM agent.
"""

import logging
from typing import Dict, Any, List
from langchain_core.messages import AIMessage, HumanMessage
from graph.state import GlobalState
from config.langsmith import traced
from services.agent import run_agent

logger = logging.getLogger(__name__)


@traced("supervisor_v2_node", tags=["supervisor", "agent", "llm"])
def supervisor_v2_node(state: GlobalState) -> Dict[str, Any]:
    """
    Agentic Supervisor Node V2.
    
    Instead of regex keyword matching, this node:
    1. Extracts conversation history from state
    2. Calls the LLM Agent to plan and execute tools
    3. Returns the LLM-generated response
    
    The Agent handles:
    - Understanding user intent
    - Deciding which tools (math functions) to call
    - Calling tools in correct order
    - Generating contextual response
    """
    
    # ================================================================
    # EXTRACT CONVERSATION DATA
    # ================================================================
    
    messages = state.get("messages", [])
    
    # Get the last user message
    last_user_message = None
    for msg in reversed(messages):
        if hasattr(msg, 'type') and msg.type == "human":
            last_user_message = msg.content
            break
        elif isinstance(msg, dict) and msg.get("role") == "user":
            last_user_message = msg.get("content", "")
            break
    
    # If no user message, return greeting
    if not last_user_message:
        return {
            "next_action": "end",
            "current_agent": "supervisor_v2",
            "messages": [AIMessage(content="👋 Hello! I'm FinCFO, your AI financial analyst. Ask me about your burn rate, runway, or try a scenario like 'What if we hire 2 engineers?'")],
        }
    
    # Build conversation history for context
    conversation_history = []
    for msg in messages:
        if hasattr(msg, 'type') and msg.type == "human":
            conversation_history.append({"role": "user", "content": msg.content})
        elif hasattr(msg, 'type') and msg.type == "ai":
            conversation_history.append({"role": "assistant", "content": msg.content})
        elif isinstance(msg, dict):
            role = msg.get("role", msg.get("type", ""))
            if role in ["user", "assistant", "human", "ai"]:
                conversation_history.append({
                    "role": "assistant" if role in ["assistant", "ai"] else "user",
                    "content": msg.get("content", "")
                })
    
    logger.info(f"🧠 Supervisor V2 processing: '{last_user_message[:80]}...'")
    logger.info(f"📜 Conversation history: {len(conversation_history)} messages")
    
    # ================================================================
    # RUN THE AGENT
    # ================================================================
    
    try:
        response = run_agent(
            user_query=last_user_message,
            state=state,
            conversation_history=conversation_history,
        )
        
        logger.info(f"✅ Agent response generated ({len(response)} chars)")
        
        return {
            "next_action": "end",  # Agent handles everything internally
            "current_agent": "supervisor_v2",
            "messages": [AIMessage(content=response)],
        }
    
    except Exception as e:
        logger.error(f"❌ Agent error: {e}", exc_info=True)
        
        # Fallback: return error message
        return {
            "next_action": "end",
            "current_agent": "supervisor_v2",
            "messages": [AIMessage(content=f"❌ I encountered an error while analyzing your request. Please try again.\n\nError: {str(e)}")],
        }


@traced("supervisor_v2_router", tags=["supervisor", "router"])
def route_from_supervisor_v2(state: GlobalState) -> str:
    """
    Simple router for V2 supervisor.
    Since the agent handles everything internally, always route to end.
    """
    next_action = state.get("next_action", "end")
    
    # The V2 supervisor handles all computation internally
    # Only return "end" to stop the graph
    return "end"